"""Compare warmed full-checkpoint inference on CPU and Apple Metal."""

import argparse
import statistics
import time

import torch

from common import write_json
from sample import load_checkpoint
from train_motor import load_motor


def benchmark(diffusion_checkpoint, motor_checkpoint, repeats=3):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple GPU unavailable to this process")
    torch.set_num_threads(3)
    torch.manual_seed(42)
    report = {"scope": "Warmed full-model forward pass; CPU and GPU contend with other running work; excludes HTTP, body physics and checkpoint loading."}
    for task, path in (("diffusion", diffusion_checkpoint), ("motor", motor_checkpoint)):
        if task == "diffusion":
            model, _, info = load_checkpoint(path, device="cpu")
        else:
            model, info = load_motor(path, device="cpu")
        values = {}
        for batch in (1, 4) if task == "diffusion" else (1,):
            inputs = ((torch.randn(batch, 3, model.size, model.size), torch.full((batch,), 500),
                       torch.zeros(batch, dtype=torch.long)) if task == "diffusion" else (torch.randn(batch, model.motor_input.in_features),))
            result = {}
            for device in ("cpu", "mps"):
                model.to(device)
                args = tuple(x.to(device) for x in inputs)
                times = []
                with torch.inference_mode():
                    for i in range(repeats + 1):
                        torch.mps.synchronize()
                        start = time.perf_counter()
                        output = model(*args)
                        torch.mps.synchronize()
                        if i:
                            times.append(time.perf_counter() - start)
                result[device] = statistics.median(times)
                if device == "cpu":
                    reference = output.cpu()
                else:
                    torch.testing.assert_close(output.cpu(), reference, rtol=3e-4, atol=3e-5)
            result["speedup"] = result["cpu"] / result["mps"]
            values[str(batch)] = result
            print(task, batch, result, flush=True)
        report[task] = {"checkpoint_step": info["training_step"], "checkpoint_sha256": info["checkpoint_sha256"], "inference": values}
        del model
        torch.mps.empty_cache()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion-checkpoint", default="runs/diffusion/best.pt")
    parser.add_argument("--motor-checkpoint", default="runs/motor/best.pt")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="benchmark.json")
    args = parser.parse_args()
    write_json(args.out, benchmark(args.diffusion_checkpoint, args.motor_checkpoint, args.repeats))
