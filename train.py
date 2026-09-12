"""A single-device training loop: data -> noise -> fly -> loss -> checkpoint.

Run: python train.py --config configs/full.json --out runs/diffusion
Resume: python train.py --resume runs/diffusion/last.pt
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import Plateau, device_for, digest, environment, load_torch, read_json, save_torch, seed_all, write_json
from diffusion import Diffusion
from model import FlyDenoiser


def validate_config(c):
    required = {
        "graph", "dataset", "seed", "threads", "image_size", "width", "recurrent_steps", "leak",
        "diffusion_steps", "batch_size", "train_steps", "learning_rate", "weight_decay",
        "gradient_clip", "ema_decay", "checkpoint_every", "validate_every", "validation_batches",
        "log_every", "allow_synthetic", "graph_control",
    }
    optional = {"architecture", "classes", "min_snr_gamma", "residual_blocks", "readout_lr_scale", "condition_dropout", "guidance_scale", "synaptic_output"}
    if not required <= set(c) or set(c) - required - optional:
        raise ValueError(f"Config keys differ: missing={required-set(c)}, unknown={set(c)-required-optional}")
    if c.get("architecture", "linear") not in ("linear", "spatial", "unet"):
        raise ValueError("Unknown image architecture")
    if c.get("architecture") in ("spatial", "unet") and c["image_size"] != 32:
        raise ValueError("Spatial image models require 32 × 32 data")
    if "residual_blocks" in c and (type(c["residual_blocks"]) is not bool or c.get("architecture")!="unet"):
        raise ValueError("Residual blocks require the U-Net architecture")
    if not 0 < c.get("readout_lr_scale",1) <= 1: raise ValueError("Invalid readout learning-rate scale")
    if not 0 <= c.get("condition_dropout",0) < 1: raise ValueError("Invalid condition dropout")
    if not math.isfinite(c.get("guidance_scale",1)) or c.get("guidance_scale",1)<1:
        raise ValueError("Guidance scale must be finite and at least one")
    if c.get("guidance_scale",1)!=1 and not c.get("condition_dropout",0):
        raise ValueError("Guidance requires training an unconditional label")
    if "synaptic_output" in c and type(c["synaptic_output"]) is not bool: raise ValueError("Invalid synaptic readout flag")
    if "classes" in c and (not c["classes"] or len(set(c["classes"])) != len(c["classes"]) or not all(isinstance(x, str) for x in c["classes"])):
        raise ValueError("Classes must be unique names")
    if not math.isfinite(c.get("min_snr_gamma", 0)) or c.get("min_snr_gamma", 0) < 0:
        raise ValueError("min_snr_gamma must be nonnegative")
    for key in ("threads", "image_size", "width", "recurrent_steps", "diffusion_steps", "batch_size",
                "train_steps", "checkpoint_every", "validate_every", "validation_batches", "log_every"):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "gradient_clip"):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if c["width"] % 2 or c["diffusion_steps"] < 2 or c["recurrent_steps"] < 2 or not 0 < c["leak"] <= 1:
        raise ValueError("Need even width, at least 2 diffusion/recurrent steps and 0 < leak <= 1")
    if not 0 <= c["ema_decay"] < 1 or c["weight_decay"] < 0 or not math.isfinite(c["weight_decay"]):
        raise ValueError("Invalid EMA decay or weight decay")
    if type(c["allow_synthetic"]) is not bool or type(c["seed"]) is not int or not 0 <= c["seed"] < 2**63:
        raise ValueError("Invalid synthetic flag or seed")
    if c["graph_control"] not in ("real", "shuffled"):
        raise ValueError("graph_control must be real or shuffled")


def image_data(root, split, size, verify=True):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if manifest["image_size"] != size:
        raise ValueError("Dataset size differs from model config")
    result = []
    for kind in ("images", "labels"):
        path = root / f"{split}_{kind}.npy"
        if verify and digest(path) != manifest["files"][path.name]:
            raise ValueError(f"Dataset checksum mismatch: {path}")
        result.append(np.load(path, mmap_mode="r", allow_pickle=False))
    images, labels = result
    if (images.dtype != np.uint8 or images.shape[1:] != (3, size, size)
            or len(images) != len(labels) or not len(labels) or labels.dtype.kind not in "iu"
            or np.any(labels < 0) or np.any(labels >= 10)):
        raise ValueError("Invalid image dataset")
    return images, labels


def batch(images, labels, indices, device):
    x = torch.from_numpy(np.array(images[indices], copy=True)).float().to(device) / 127.5 - 1
    y = torch.from_numpy(np.array(labels[indices], dtype=np.int64, copy=True)).to(device)
    return x, y


@torch.no_grad()
def evaluate_loss(model, diffusion, images, labels, batch_size, batches, seed=2026, ablate=False, wrong_labels=False, weights=None):
    device = next(model.parameters()).device
    rng = torch.Generator().manual_seed(seed)
    total, baseline, count = 0.0, 0.0, 0
    model.eval()
    for _ in range(batches):
        idx = torch.randint(len(images), (batch_size,), generator=rng).numpy()
        x, y = batch(images, labels, idx, device)
        t = torch.randint(diffusion.steps, (batch_size,), generator=rng).to(device)
        noise = torch.randn(x.shape, generator=rng).to(device)
        xt = diffusion.add_noise(x, t, noise)
        args = (xt, t, (y + 1) % len(model.classes) if wrong_labels else y)
        pred = model(*args, ablate_edges=ablate) if weights is None else torch.func.functional_call(model, weights, args, dict(ablate_edges=ablate))
        total += F.mse_loss(pred, x).item() * x.numel()
        baseline += x.square().mean().item() * x.numel()
        count += x.numel()
    return {"clean_image_mse": total / count, "zero_predictor_mse": baseline / count,
            "examples_drawn_with_replacement": batch_size * batches, "seed": seed}


class AveragedModel(torch.nn.Module):
    def __init__(self, model, weights):
        super().__init__(); self.model=model; self.weights=weights; self.size=model.size
        self.null_label=getattr(model,"null_label",None);self.guidance_scale=getattr(model,"guidance_scale",1.)
    def forward(self, *args, **kwargs):
        return torch.func.functional_call(self.model,self.weights,args,kwargs)


def run(config_path=None, out=None, resume=None, device_name="auto", stop_after=None, graph=None, dataset=None,
        until_convergence=False, threads=None, validation_batches=None, init_from=None):
    if resume and init_from: raise ValueError("Use either --resume or --init-from")
    state = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if state and state.get("format_version") != 2:
        raise ValueError("Unsupported checkpoint format")
    if state and state.get("inference_only"):
        raise ValueError("Inference export has no optimizer; resume the original training checkpoint")
    config = state["config"] if state else read_json(config_path)
    validate_config(config)
    config = dict(config)
    if threads is not None:
        config["threads"] = threads
    if validation_batches is not None:
        config["validation_batches"] = validation_batches
    validate_config(config)
    config["graph"] = graph or config["graph"]
    config["dataset"] = dataset or config["dataset"]
    config["graph"], config["dataset"] = str(Path(config["graph"]).resolve()), str(Path(config["dataset"]).resolve())
    output = Path(out or (Path(resume).parent if resume else "runs/diffusion"))
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError(f"Run directory is not empty: {output}. Use --resume or a fresh --out.")
    output.mkdir(parents=True, exist_ok=True)
    seed_all(config["seed"], config["threads"])
    device = device_for(device_name)
    graph_hash = digest(config["graph"])
    data_hash = digest(Path(config["dataset"]) / "manifest.json")
    if state and (state["graph_sha256"] != graph_hash or state["data_manifest_sha256"] != data_hash):
        raise ValueError("Graph or data manifest changed since checkpoint")
    model = FlyDenoiser(config["graph"], config).to(device)
    initial_checksum=None
    if init_from:
        initial,initial_checksum=load_torch(init_from)
        if initial["graph_sha256"] != graph_hash or initial["config"]["graph_control"] != config["graph_control"]:
            raise ValueError("Initial weights require the same graph")
        current=model.state_dict()
        compatible={k:v for k,v in initial["ema"].items() if k in current and v.shape==current[k].shape}
        old=initial["ema"]["class_input.weight"]
        if current["class_input.weight"].shape[0]==old.shape[0]+1:
            compatible["class_input.weight"]=torch.cat([old,old.mean(0,keepdim=True)])
        model.load_state_dict(compatible,strict=False)
        del initial
    metadata = read_json(Path(config["dataset"]) / "manifest.json")
    if not config["allow_synthetic"]:
        if (model.graph_metadata["kind"] != "malecns_v1" or not model.graph_metadata["source_verified"]
                or model.n_neurons != 166700 or model.n_edges != 25582938 or metadata["kind"] not in ("cifar10", "quickdraw")):
            raise ValueError("Full run requires verified full MaleCNS and a supported image dataset. Synthetic fixtures need allow_synthetic=true.")
    if metadata.get("classes", model.classes) != model.classes:
        raise ValueError("Dataset categories differ from model")
    train_x, train_y = image_data(config["dataset"], "train", config["image_size"])
    val_x, val_y = image_data(config["dataset"], "val", config["image_size"])
    parameters = model.parameters()
    if "readout_lr_scale" in config:
        parameters = [dict(params=[p for n,p in model.named_parameters() if n!="neurons_to_output.weight"]),
            dict(params=[model.neurons_to_output.weight],lr=config["learning_rate"]*config["readout_lr_scale"])]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    diffusion = Diffusion(config["diffusion_steps"], device)
    rng = torch.Generator().manual_seed(config["seed"] + 1)
    step = 0
    best_val = float("inf")
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.set_state(state["batch_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        step, best_val = state["step"], state["best_val"]
    ema = {k: v.clone().to(device) for k, v in (state["ema"] if state else model.state_dict()).items()}
    plateau = Plateau(state.get("plateau") if state else None, min_steps=config["train_steps"]) if until_convergence else None
    del state
    total_parameters = sum(p.numel() for p in model.parameters())
    manifest = {
        "initial_checkpoint_sha256": initial_checksum,
        "config": config, "graph_sha256": graph_hash, "data_manifest_sha256": data_hash,
        "environment": environment(), "nodes": model.n_neurons, "edges": model.n_edges,
        "trainable_parameters": total_parameters, "graph_control": config["graph_control"],
        "prediction_type": "x0", "objective": "Min-SNR clean-image MSE" if config.get("min_snr_gamma") else "Uniform clean-image MSE",
        "synthetic_fixture": config["allow_synthetic"], "device": str(device),
        "code_hashes": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
    }
    write_json(output / ("resume-environment.json" if resume else "run.json"), manifest)
    write_json(output / "config.json", config)
    print(json.dumps({k: manifest[k] for k in ("nodes", "edges", "trainable_parameters", "device", "synthetic_fixture")}), flush=True)
    started, start_step = time.monotonic(), step
    end = (math.inf if until_convergence else config["train_steps"]) if stop_after is None else step + stop_after
    if not until_convergence:
        end = min(end, config["train_steps"])
    if end <= step:
        raise ValueError("No steps requested: training already complete or invalid --stop-after")

    def checkpoint(name="last.pt"):
        save_torch(output / name, {
            "format_version": 2, "inference_only": False, "config": config, "step": step,
            "model": model.state_dict(), "ema": ema, "optimizer": optimizer.state_dict(),
            "batch_rng": rng.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "best_val": best_val, "graph_sha256": graph_hash, "data_manifest_sha256": data_hash,
            "plateau": plateau.state if plateau else None,
        })

    with (output / "metrics.jsonl").open("a") as log:
        converged = False
        try:
            while step < end:
                model.train()
                idx = torch.randint(len(train_x), (config["batch_size"],), generator=rng).numpy()
                x, labels = batch(train_x, train_y, idx, device)
                t = torch.randint(diffusion.steps, (len(x),), generator=rng).to(device)
                noise = torch.randn(x.shape, generator=rng).to(device)
                optimizer.zero_grad(set_to_none=True)
                condition=labels.clone()
                if config.get("condition_dropout"):
                    drop=torch.rand(len(labels),generator=rng).to(device)<config["condition_dropout"]
                    condition[drop]=len(model.classes)
                pred = model(diffusion.add_noise(x, t, noise), t, condition)
                per_image = (pred-x).square().flatten(1).mean(1)
                unweighted_mse = per_image.mean().item()
                if config.get("min_snr_gamma"):
                    alpha = diffusion.alpha_bar[t]
                    per_image = per_image * (alpha / (1-alpha)).clamp(max=config["min_snr_gamma"])
                loss = per_image.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss; last periodic checkpoint remains available")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                with torch.no_grad():
                    decay = min(config["ema_decay"], (step + 1) / (step + 10))
                    for k, value in model.state_dict().items():
                        ema[k].lerp_(value, 1 - decay)
                step += 1
                row = {"step": step, "train_loss": loss.item(), "unweighted_mse": unweighted_mse, "gradient_norm": grad_norm.item(), "device": str(device)}
                if step % config["validate_every"] == 0 or step == end:
                    # Select the EMA weights used for inference; keep the metric unweighted and comparable.
                    result = evaluate_loss(model, diffusion, val_x, val_y, config["batch_size"], config["validation_batches"], weights=ema)
                    row["validation"] = result
                    if not config["allow_synthetic"]:
                        from sample import grid, to_images
                        labels=torch.arange(len(model.classes),device=device).repeat_interleave(3)
                        samples,_=diffusion.sample(AveragedModel(model,ema),labels,2026,50)
                        preview=grid(to_images(samples))
                        (output/"samples").mkdir(exist_ok=True)
                        preview.save(output/"samples"/f"{step:06d}.png");preview.save(output/"preview.png")
                    if plateau:
                        converged = plateau.update(result["clean_image_mse"], step, optimizer)
                        row["plateau"] = dict(plateau.state)
                    if result["clean_image_mse"] < best_val:
                        best_val = result["clean_image_mse"]
                        checkpoint("best.pt")
                row["seconds_per_step"] = (time.monotonic() - started) / (step - start_step)
                row["learning_rate"] = optimizer.param_groups[0]["lr"]
                if step % config["log_every"] == 0 or "validation" in row:
                    line = json.dumps(row, allow_nan=False)
                    print(line, flush=True)
                    log.write(line + "\n")
                    log.flush()
                    write_json(output / "status.json", dict(row, status="validation_plateau" if converged else "training",
                               updated_at=time.time(), until_convergence=until_convergence))
                if step % config["checkpoint_every"] == 0 or converged:
                    checkpoint()
                if converged:
                    break
        except KeyboardInterrupt:
            # Only periodically saved checkpoints promise exact resume; a mid-update interrupt may not.
            print("Interrupted. Resume the last completed periodic checkpoint.", flush=True)
            raise
    checkpoint()
    write_json(output / "status.json", dict(row, status="validation_plateau" if converged else "step_limit",
               updated_at=time.time(), until_convergence=until_convergence))
    print(f"Saved {output / 'last.pt'}", flush=True)
    return output / "last.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/full.json")
    parser.add_argument("--out")
    parser.add_argument("--resume")
    parser.add_argument("--init-from", help="Initialize compatible weights; start a fresh optimizer and log")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--stop-after", type=int, help="Checkpoint after this many additional steps; config stays unchanged for resume")
    parser.add_argument("--graph", help="Relocate the identical graph when resuming on another machine")
    parser.add_argument("--dataset", help="Relocate the identical prepared data when resuming")
    parser.add_argument("--until-convergence", action="store_true", help="Continue beyond the configured step budget until validation plateaus")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--validation-batches", type=int)
    args = parser.parse_args()
    run(args.config, args.out, args.resume, args.device, args.stop_after, args.graph, args.dataset,
        args.until_convergence, args.threads, args.validation_batches, args.init_from)
