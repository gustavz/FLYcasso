"""Generate images and a denoising animation from a local checkpoint."""

import argparse
import math
from pathlib import Path

import torch
from PIL import Image

from flycasso.common import device_for, digest, load_torch, seed_all, write_json
from flycasso.diffusion import Diffusion
from flycasso.model import FlyDenoiser
from flycasso.train import validate_config


def load_checkpoint(checkpoint, graph=None, device="auto", raw=False):
    checkpoint = Path(checkpoint)
    state, checksum = load_torch(checkpoint)
    if state.get("format_version") != 2:
        raise ValueError("Unsupported checkpoint format")
    config = state["config"]
    validate_config(config)
    graph = Path(graph) if graph else (checkpoint.parent / "graph.npz" if (checkpoint.parent / "graph.npz").exists() else Path(config["graph"]))
    if digest(graph) != state["graph_sha256"]:
        raise ValueError("Graph does not match checkpoint; supply the original --graph")
    seed_all(config["seed"], config["threads"])
    device = device_for(device)
    model = FlyDenoiser(graph, config).to(device)
    model.load_state_dict(state["model"] if raw else state["ema"])
    model.eval()
    info = {"checkpoint_sha256": checksum, "graph_sha256": state["graph_sha256"],
            "data_manifest_sha256": state["data_manifest_sha256"], "training_step": state["step"],
            "weights": "raw" if raw and not state.get("inference_only") else "ema", "config": config,
            "neurons": model.n_neurons, "edges": model.n_edges,
            "graph_kind": model.graph_metadata["kind"], "device": str(device)}
    return model, Diffusion(config["diffusion_steps"], device), info


def to_images(batch):
    pixels = ((batch.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(p) for p in pixels]


def grid(images, scale=8):
    cols = math.ceil(math.sqrt(len(images)))
    size = images[0].width
    canvas = Image.new("RGB", (cols * size, math.ceil(len(images) / cols) * size), "#eeeae1")
    for i, im in enumerate(images):
        canvas.paste(im, (i % cols * size, i // cols * size))
    return canvas.resize((canvas.width * scale, canvas.height * scale), Image.Resampling.NEAREST)


def generate(model, diffusion, category="cat", count=4, seed=42, steps=50, ablate=False, on_step=None):
    if category not in model.classes + ["all"] or type(count) is not int or not 1 <= count <= 16:
        raise ValueError("Choose an available category and 1–16 images")
    if type(seed) is not int or not 0 <= seed < 2**63 or type(steps) is not int:
        raise ValueError("Seed must be a nonnegative 63-bit integer; steps must be an integer")
    labels = list(range(len(model.classes))) if category == "all" else [model.classes.index(category)] * count
    tensor = torch.tensor(labels, device=next(model.parameters()).device)
    samples, frames = diffusion.sample(model, tensor, seed, steps, ablate_edges=ablate, on_step=on_step)
    metadata = {"classes": [model.classes[i] for i in labels], "seed": seed,
                "sampling_steps": steps, "guidance_scale": getattr(model,"guidance_scale",1.), "sampler": "DDIM eta=0, clipped x0", "prediction_type": "x0",
                "edges_ablated": ablate, "resolution": model.size}
    return to_images(samples), frames, metadata


def save_samples(output, images, frames, metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for i, im in enumerate(images):
        im.save(output / f"{i:02d}-{metadata['classes'][i]}.png")
    grid(images).save(output / "grid.png")
    animation = [grid(to_images(frame), scale=4) for frame in frames]
    animation[0].save(output / "denoising.gif", save_all=True, append_images=animation[1:], duration=90, loop=0)
    write_json(output / "metadata.json", metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/diffusion/last.pt")
    parser.add_argument("--graph")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--raw", action="store_true", help="Use current weights instead of EMA")
    parser.add_argument("--class", dest="category", default="cat")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--ablate", action="store_true", help="Disable recurrent edges as a control")
    parser.add_argument("--out", default="samples/cat-42")
    args = parser.parse_args()
    model, diffusion, info = load_checkpoint(args.checkpoint, args.graph, args.device, args.raw)
    images, frames, metadata = generate(model, diffusion, args.category, args.count, args.seed, args.steps, args.ablate)
    save_samples(args.out, images, frames, dict(info, **metadata))
    print(f"Saved images, animation and provenance to {args.out}")
