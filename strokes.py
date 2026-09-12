"""Category-conditioned stroke diffusion. References are training targets only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import checked_download, digest, read_json, write_json
from diffusion import Diffusion


def vectorize(drawing, points=256):
    strokes = [np.asarray(s, dtype=float).T for s in drawing if len(s[0]) > 1]
    lengths = [np.linalg.norm(np.diff(s, axis=0), axis=1).sum() for s in strokes]
    counts = np.full(len(strokes), 2)
    for _ in range(points-counts.sum()):
        counts[np.argmax(np.asarray(lengths)/counts)] += 1
    result = []
    for stroke, count in zip(strokes, counts):
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(stroke, axis=0), axis=1))]
        at = np.linspace(0, distance[-1], count)
        xy = np.stack([np.interp(at, distance, stroke[:, j]) for j in range(2)]) / 127.5 - 1
        down = np.ones(count); down[-1] = -1
        result.append(np.vstack([xy, down]))
    return np.concatenate(result, axis=1).astype(np.float32)


def decode(sequence):
    """A negative third coordinate lifts the pen after the current point."""
    sequence = np.asarray(sequence)
    if sequence.shape != (3,256) or not np.isfinite(sequence).all():
        raise ValueError("Expected a finite 256-point stroke sequence")
    drawing, current = [], []
    for point in sequence.T:
        current.append(((point[:2].clip(-1,1)+1)*127.5).tolist())
        if point[2] < 0:
            if len(current) > 1: drawing.append(np.asarray(current).T.tolist())
            current = []
    if len(current) > 1: drawing.append(np.asarray(current).T.tolist())
    if not drawing:
        raise ValueError("The stroke model produced no drawable strokes")
    return drawing


class StrokeDenoiser(nn.Module):
    """Use the painting model's full circuit with its learned stroke adapters."""
    def __init__(self, motor):
        super().__init__(); self.motor = motor; self.size = 16
        self.null_label=getattr(motor,"null_label",None);self.guidance_scale=getattr(motor,"guidance_scale",1.)
    def forward(self, image, timestep, label, ablate_edges=False):
        phase = timestep.float()[:, None] * self.motor.stroke_frequency[None]
        time = self.motor.stroke_time(torch.cat([phase.sin(), phase.cos()], dim=1))
        if hasattr(self.motor,"stroke_encoder"):
            x=image.flatten(2);features=[]
            for layer in self.motor.stroke_encoder:
                x=torch.nn.functional.silu(layer(x));features.append(x)
            code=self.motor.stroke_input(x.flatten(1))+time+self.motor.stroke_class(label)
            context=self.motor.neural_readout(code,ablate_edges)
            x=torch.nn.functional.silu(self.motor.stroke_decoder[0](x)+self.motor.stroke_context[0](context)[:,:,None])
            for i,skip in enumerate(reversed(features[:-1]),1):
                x=torch.nn.functional.interpolate(x,scale_factor=2,mode="nearest")
                x=torch.nn.functional.silu(self.motor.stroke_decoder[i](torch.cat([x,skip],1))+self.motor.stroke_context[i](context)[:,:,None])
            return self.motor.stroke_output(x).reshape_as(image)
        code = self.motor.stroke_input(image.flatten(2)) + time + self.motor.stroke_class(label)
        return self.motor.stroke_output(self.motor.neural_readout(code, ablate_edges)).reshape_as(image)


@torch.no_grad()
def generate(motor, category, seed=42, steps=50):
    if not hasattr(motor, "stroke_input") or category not in motor.classes:
        raise ValueError("Load a category-trained painting checkpoint and choose an available category")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Seed must be a nonnegative 63-bit integer")
    labels = torch.tensor([motor.classes.index(category)], device=next(motor.parameters()).device)
    samples, _ = Diffusion(1000, labels.device).sample(StrokeDenoiser(motor), labels, seed, steps)
    return decode(samples[0].reshape(3,256).numpy())


def prepare(root="data/processed/quickdraw-strokes-10", images="data/processed/quickdraw-10"):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    ids_path=Path(images)/"ids.json"
    if (root/"manifest.json").exists():
        manifest=read_json(root/"manifest.json")
        if manifest["image_ids_sha256"]!=digest(ids_path):
            raise ValueError("Image splits changed; choose a new stroke data directory")
        for name,expected in manifest["files"].items():
            if digest(root/name)!=expected: raise ValueError(f"Corrupt prepared stroke data: {name}")
        print(f"Verified existing stroke targets at {root}");return
    ids = read_json(ids_path)
    wanted = {key for values in ids.values() for key in values}
    rows = {}; sources = read_json("stroke-sources.json"); classes = list(sources)
    for label, category in enumerate(classes):
        path = checked_download(Path("data/raw/quickdraw") / f"{category}.ndjson", sources[category])
        with path.open() as stream:
            for line in stream:
                row = json.loads(line); key = str(row["key_id"])
                if key in wanted: rows[key] = (vectorize(row["drawing"]), label)
    files = {}
    for split, keys in ids.items():
        path = root / f"{split}.npz"
        np.savez_compressed(path, strokes=np.asarray([rows[k][0] for k in keys]), labels=np.asarray([rows[k][1] for k in keys], dtype=np.int64))
        files[path.name] = digest(path)
    write_json(root / "manifest.json", dict(classes=classes, files=files, sources=sources,
        image_ids_sha256=digest(ids_path), code_sha256=digest(__file__)))
    print(f"Prepared category stroke targets at {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--root",default="data/processed/quickdraw-strokes-10")
    parser.add_argument("--images",default="data/processed/quickdraw-10")
    args = parser.parse_args()
    if args.prepare: prepare(args.root,args.images)
