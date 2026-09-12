"""Small, checked, atomic artifact I/O. No unpickling of downloaded datasets."""

import hashlib
import json
import os
import platform
import sys
import urllib.request
from importlib.metadata import version
from pathlib import Path


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def checked_download(path, spec):
    path = Path(path)
    algorithm = "sha256" if "sha256" in spec else "md5"
    expected = spec[algorithm]
    if not spec["url"].startswith("https://"):
        raise ValueError("Downloads must use HTTPS")
    if path.exists():
        if digest(path, algorithm) != expected:
            raise ValueError(f"Checksum mismatch: {path}. Remove the corrupt file and retry.")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    print(f"Downloading {path.name} …", flush=True)
    # ponytail: restart partial transfers; range resume needs pinned ETag handling.
    try:
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "FLYcasso/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as out:
            if not response.url.startswith("https://"):
                raise ValueError("Refusing a download redirected to plain HTTP")
            while chunk := response.read(8 * 1024 * 1024):
                out.write(chunk)
        if "bytes" in spec and temp.stat().st_size != spec["bytes"]:
            raise ValueError(f"Incorrect length for {path.name}")
        if digest(temp, algorithm) != expected:
            raise ValueError(f"Checksum mismatch for {path.name}")
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return path


def load_torch(path):
    """Hash the same opened checkpoint even if training replaces its filename."""
    import torch
    with Path(path).open("rb") as stream:
        state = torch.load(stream, map_location="cpu", weights_only=True)
        stream.seek(0)
        return state, hashlib.file_digest(stream, "sha256").hexdigest()


def save_torch(path, value):
    import torch
    path = Path(path)
    temp = path.with_name(path.name + ".partial")
    torch.save(value, temp)
    temp.replace(path)


def environment():
    import torch
    return {
        "python": sys.version, "platform": platform.platform(),
        "packages": {p: version(p) for p in ("torch", "numpy", "scipy", "pyarrow", "pandas", "pillow")},
        "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "threads": torch.get_num_threads(),
    }


def device_for(name):
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        device = torch.device(name)
    except (RuntimeError, TypeError) as error:
        raise ValueError("Use auto, cpu, cuda or mps.") from error
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("Use auto, cpu, cuda or mps.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple Metal GPU unavailable to this process. Use CPU or allow GPU access.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available; use --device cpu or install CUDA PyTorch.")
    return device


def seed_all(seed, threads=4):
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Fail rather than silently substituting nondeterministic kernels.
    torch.use_deterministic_algorithms(True)


class Plateau:
    """Fixed validation draws; reduce LR at 2/4/6 stale checks, stop at 8."""
    def __init__(self, state=None, min_steps=5000, min_delta=0.0005):
        self.state = state or dict(best=None, stale=0, reductions=0,
                                   min_steps=min_steps, min_delta=min_delta)

    def update(self, loss, step, optimizer):
        import math
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite validation loss")
        s = self.state
        if s["best"] is None or loss < s["best"] - s["min_delta"]:
            s.update(best=loss, stale=0)
        else:
            s["stale"] += 1
        if s["stale"] >= 2 * (s["reductions"] + 1) and s["reductions"] < 3:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
            s["reductions"] += 1
        return step >= s["min_steps"] and s["stale"] >= 8 and s["reductions"] == 3
