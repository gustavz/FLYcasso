"""Make a portable inference folder: trained EMA weights + exact graph + hashes.

Publish the folder as a release asset after reviewing its samples and model card.
Training/optimizer checkpoints stay in runs/ for exact resume.
"""

import argparse
import shutil
from pathlib import Path

from common import digest, load_torch, read_json, save_torch, write_json


def export(checkpoint, output, graph=None):
    state, checksum = load_torch(checkpoint)
    local_graph = Path(checkpoint).parent / "graph.npz"
    graph = Path(graph) if graph else local_graph if local_graph.exists() else Path(state["config"]["graph"])
    if digest(graph) != state["graph_sha256"]:
        raise ValueError("Graph checksum differs from checkpoint")
    output = Path(output)
    if (output / "manifest.json").exists():
        manifest = read_json(output / "manifest.json")
        if manifest["source_checkpoint_sha256"] == checksum and all(
                digest(output / name) == expected for name, expected in manifest["files"].items()):
            print(f"Verified existing inference files at {output}")
            return
        raise ValueError("Export differs from this checkpoint or is corrupt; choose a new --out")
    output.mkdir(parents=True, exist_ok=False)
    config = dict(state["config"], graph="graph.npz", dataset="data/processed/" + Path(state["config"]["dataset"]).name)
    motor = state.get("task") == "front_leg_motor_v1"
    if motor:
        config["dataset"] = "data/processed/strokes"
        portable = {k: state[k] for k in ("task", "step", "graph_sha256", "data_sha256", "model", "best_val")}
        portable.update(inference_only=True, config=config)
        if "stroke_ema" in state:
            portable["model"] = dict(state["model"], **state["stroke_ema"])
        write_json(output / "config.json", config)
    else:
        portable = {k: state[k] for k in ("format_version", "step", "graph_sha256", "data_manifest_sha256")}
        portable.update(inference_only=True, config=config, model=state["ema"], ema=state["ema"])
    save_torch(output / "model.pt", portable)
    shutil.copyfile(graph, output / "graph.npz")
    for name in ("docs/MODEL_CARD.md", "docs/THIRD_PARTY.md", "LICENSE"):
        shutil.copyfile(Path(__file__).parent / name, output / Path(name).name)
    write_json(output / "manifest.json", {
        "training_steps": state["step"], "graph_kind": "synthetic_fixture" if config.get("allow_synthetic") else "malecns_v1",
        "weights": "motor" if motor else "EMA", "source_checkpoint_sha256": checksum,
        "files": {p.name: digest(p) for p in sorted(output.iterdir())},
        "usage": f"python app.py --checkpoint runs/diffusion/best.pt --motor-checkpoint {output}/model.pt" if motor else f"python app.py --checkpoint {output}/model.pt",
        "quality": "Consult the model card and paired evaluation. Export is not quality certification.",
    })
    print(f"Portable inference files saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/diffusion/last.pt")
    parser.add_argument("--out", default="bundles/flycasso")
    parser.add_argument("--graph")
    args = parser.parse_args()
    export(args.checkpoint, args.out, args.graph)
