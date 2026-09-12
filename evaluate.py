"""Paired held-out clean-image prediction tests. These are NOT image-quality scores."""

import argparse
import json
from pathlib import Path

from common import digest, write_json
from sample import load_checkpoint
from train import evaluate_loss, image_data


def evaluate(checkpoint, graph=None, dataset=None, device="auto", raw=False, split="val", batches=8, batch_size=4):
    if batches < 1 or batch_size < 1 or split not in ("val", "test"):
        raise ValueError("Use val/test and positive batch counts")
    model, diffusion, info = load_checkpoint(checkpoint, graph, device, raw)
    dataset = Path(dataset or info["config"]["dataset"])
    if digest(dataset / "manifest.json") != info["data_manifest_sha256"]:
        raise ValueError("Evaluation dataset differs from training provenance")
    x, y = image_data(dataset, split, model.size)
    results = {}
    for name, options in (("intact", {}), ("edges_disabled", {"ablate": True}),
                          ("wrong_labels", {"wrong_labels": True})):
        results[name] = evaluate_loss(model, diffusion, x, y, batch_size, batches, **options)
    return dict(info, split=split, results=results,
                interpretation="Lower clean-image MSE is better; zero prediction is uniform gray. Paired interventions use identical images/timesteps/noise. "
                "Ablation is not a retrained control and does not prove biological advantage or image quality.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/diffusion/last.pt")
    parser.add_argument("--graph")
    parser.add_argument("--dataset")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", default="evaluation.json")
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.graph, args.dataset, args.device, args.raw,
                      args.split, args.batches, args.batch_size)
    write_json(args.out, result)
    print(json.dumps(result["results"], indent=2))
