"""Rasterize pinned Quick, Draw! vectors into disjoint image and stroke datasets."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from flycasso.common import ROOT, checked_download, digest, read_json, write_json


def rasterize(drawing):
    image = Image.new("L", (128,128), 255); pen = ImageDraw.Draw(image)
    for xs,ys in drawing:
        if len(xs)>1: pen.line([(16+x/255*96,16+y/255*96) for x,y in zip(xs,ys)],fill=0,width=4)
    pixels=np.asarray(image.resize((32,32),Image.Resampling.LANCZOS))
    return np.repeat(pixels[None],3,axis=0)


def prepare(root="data/processed/quickdraw-10", per_class=10000):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    sources = read_json(ROOT / "configs/stroke-sources.json")
    classes = list(sources)
    if (root/"manifest.json").exists():
        manifest=read_json(root/"manifest.json")
        if manifest['sources']!=sources or manifest['classes']!=classes or manifest['drawings']['train']!=per_class*len(classes):
            raise ValueError("Existing dataset has different categories or size; choose a new --root")
        for name,expected in manifest['files'].items():
            if digest(root/name)!=expected: raise ValueError(f"Corrupt prepared image data: {name}")
        print(f"Verified existing images at {root}");return
    split_images = {s: [] for s in ("train", "val", "test")}
    split_labels = {s: [] for s in split_images}
    ids = {s: [] for s in split_images}
    # Reserve the existing motor validation/test drawings from image training as well.
    from flycasso.prepare_strokes import split_drawings
    reserved = split_drawings()
    heldout = {str(d["id"]): split for split in ("val", "test") for d in reserved.get(split, [])}
    rng = np.random.default_rng(1729)
    for label, category in enumerate(classes):
        path = checked_download(Path("data/raw/quickdraw") / f"{category}.ndjson", sources[category])
        rows = []
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["recognized"] and 1 <= len(row["drawing"]) <= 16 and str(row["key_id"]) not in heldout:
                    rows.append(row)
                if len(rows) >= per_class + 1000:
                    break
        if len(rows) < per_class+1000: raise ValueError(f"Insufficient recognized drawings for {category}")
        rng.shuffle(rows)
        for i, row in enumerate(rows):
            split = "train" if i < per_class else "val" if i < per_class + 500 else "test"
            split_images[split].append(rasterize(row["drawing"]))
            split_labels[split].append(label); ids[split].append(str(row["key_id"]))
    files = {}
    for split in split_images:
        for kind, arrays in (("images", split_images), ("labels", split_labels)):
            path = root / f"{split}_{kind}.npy"
            np.save(path, np.asarray(arrays[split], dtype=np.uint8 if kind == "images" else np.int64))
            files[path.name] = digest(path)
    write_json(root / "ids.json", ids); files["ids.json"] = digest(root / "ids.json")
    assert not (set(ids["train"]) & (set(ids["val"]) | set(ids["test"])))
    write_json(root / "manifest.json", dict(kind="quickdraw", image_size=32, classes=classes, files=files,
        drawings={s: len(v) for s,v in ids.items()}, sources=sources, code_sha256=digest(__file__)))
    print(f"Prepared {len(ids['train'])} training images at {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/processed/quickdraw-10")
    parser.add_argument("--per-class", type=int, default=10000)
    args = parser.parse_args()
    if args.per_class < 64: parser.error("Use at least 64 drawings per category")
    prepare(args.root, args.per_class)
