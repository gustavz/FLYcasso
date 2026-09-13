"""Measure full training steps (including backward + AdamW) before a long run."""

import argparse
import statistics
import time

import torch

from flycasso.common import device_for, digest, environment, read_json, seed_all, write_json
from flycasso.model import FlyDenoiser
from flycasso.train import validate_config


def benchmark(config, device="auto", repeats=5):
    validate_config(config)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    seed_all(config["seed"], config["threads"])
    device = device_for(device)
    model = FlyDenoiser(config["graph"], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    x = torch.randn(config["batch_size"], 3, config["image_size"], config["image_size"], device=device)
    y = torch.arange(len(x), device=device) % len(model.classes)
    t = torch.full_like(y, config["diffusion_steps"] // 2)
    measurements = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(repeats + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        if device.type == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        model(x, t, y).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        if i:
            measurements.append(elapsed)
        print(f"{'warmup' if i == 0 else i}: {elapsed:.3f}s", flush=True)
    median = statistics.median(measurements)
    return {"environment": environment(), "config": config, "graph_sha256": digest(config["graph"]),
            "nodes": model.n_neurons, "edges": model.n_edges, "device": str(device),
            "seconds_per_step_median": median, "images_per_second": len(x) / median,
            "measurements_seconds": measurements,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "scope": "Synthetic tensors through the actual full graph; one warmup; backward, clipping and AdamW included. EMA, validation and checkpoint I/O excluded."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/full.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default="benchmark.json")
    args = parser.parse_args()
    write_json(args.out, benchmark(read_json(args.config), args.device, args.repeats))
