"""Download pinned Quick, Draw! strokes; make disjoint drawings and IK demonstrations.

python -m flycasso.prepare_strokes
Human strokes specify foot targets. These are engineered motor demonstrations,
not recorded fly behavior. Live inference never calls the IK teacher.
"""

import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from flycasso.common import ROOT, checked_download, digest, read_json, write_json
from flycasso.paint import PaintingFly, drawing_targets


def split_drawings():
    specs = read_json(ROOT / "configs/stroke-sources.json")
    rng = np.random.default_rng(1729)
    drawings = {split: [] for split in ("train", "val", "test")}
    for category, spec in specs.items():
        path = checked_download(Path("data/raw/quickdraw") / (category + ".ndjson"), spec)
        candidates = []
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                strokes = [s for s in row["drawing"] if len(s[0]) >= 2]
                if row["recognized"] and 1 <= len(strokes) <= 16:
                    candidates.append(dict(id=str(row["key_id"]), category=category, drawing=strokes))
                if len(candidates) == 512:
                    break
        rng.shuffle(candidates)
        for split, subset in (("train", candidates[:96]), ("val", candidates[96:112]), ("test", candidates[112:128])):
            drawings[split].extend(subset)
    ids = [{d["id"] for d in drawings[s]} for s in drawings]
    assert all(not ids[a] & ids[b] for a in range(3) for b in range(a))
    return drawings


def prepare(root="data/processed/strokes"):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    if (root/"manifest.json").exists():
        manifest=read_json(root/"manifest.json")
        if manifest.get("observations") != 40 or manifest.get("drawing_leg") != "lf":
            raise ValueError("Existing motor data uses an older format; choose a new --root")
        for name,expected in manifest["files"].items():
            if digest(root/name)!=expected: raise ValueError(f"Corrupt prepared motor data: {name}")
        print(f"Verified existing motor data at {root}",flush=True);return
    specs=read_json(ROOT / "configs/stroke-sources.json")
    drawings=split_drawings()
    write_json(root / "drawings.json", drawings)
    fly = PaintingFly()
    files = {"drawings.json": digest(root / "drawings.json")}
    for split in ("train", "val", "test"):
        # ponytail: short trajectories cover basic control; DAgger adds difficult visited states.
        subset = [drawing for category in specs for drawing in
                  sorted((d for d in drawings[split] if d["category"] == category),
                         key=lambda d: sum(len(s[0]) for s in d["drawing"]))[:8 if split == "train" else 4]]
        arrays = collect_rollouts(fly, subset)
        path = root / f"{split}.npz"
        np.savez_compressed(path, **arrays)
        files[path.name] = digest(path)
    (ROOT / "web/assets").mkdir(exist_ok=True)
    fly.reset()
    (ROOT / "web/assets/fly-scene.json").write_text(json.dumps(fly.scene(), separators=(",", ":")))
    for name, spec in read_json(ROOT / "configs/web-sources.json").items():
        checked_download((ROOT / "web/vendor") / name, spec)
    write_json(root / "manifest.json", dict(kind="quickdraw_motor_imitation_v1", drawing_leg="lf", sources=specs,
        drawings={s: len(v) for s, v in drawings.items()}, files=files, seed=1729,
        simulation="FlyGym 2.1.0 / MuJoCo 3.9.0", observations=40, actions=14,
        target="Front joint angle setpoints, normalized around the neutral pose",
        scope="Executed one-leg teacher trajectories, joint velocities, physical pen contact; IK is offline only",
        code={n: digest(Path(__file__).with_name(n)) for n in ("paint.py", "prepare_strokes.py")}))
    print(f"Prepared motor demonstrations at {root}", flush=True)


def collect_rollouts(fly, drawings, policy=None):
    """Label states encountered during actual movement; never use IK in the app."""
    observations, actions, goals_out = [], [], []
    for drawing_index, drawing in enumerate(drawings):
        if drawing_index % 24 == 0:
            print(f"Executed demonstrations: {drawing_index}/{len(drawings)} drawings", flush=True)
        fly.reset()
        for goal in drawing_targets(drawing["drawing"], fly.home):
            observation = fly.observe(goal)
            qpos = fly.data.qpos.copy()
            action = fly.expert(goal)
            fly.data.qpos[:] = qpos
            mj.mj_forward(fly.model, fly.data)
            observations.append(observation); actions.append(action); goals_out.append(goal)
            if policy is None:
                command = action
            else:
                import torch
                with torch.no_grad():
                    command = policy(torch.from_numpy(observation)[None].to(next(policy.parameters()).device))[0].cpu().numpy()
            fly.step(command, record=False)
    return dict(observations=np.asarray(observations, dtype=np.float32), actions=np.asarray(actions, dtype=np.float32),
                goals=np.asarray(goals_out, dtype=np.float32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/processed/strokes")
    args = parser.parse_args()
    prepare(args.root)
