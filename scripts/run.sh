#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
config="${1:-configs/full.json}"
run_dir="${2:-runs/diffusion}"
python -m flycasso.prepare --target all
if ! python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1])/"status.json"; sys.exit(not (p.exists() and json.loads(p.read_text())["status"]=="validation_plateau"))' "$run_dir"; then
  if [[ -f "$run_dir/last.pt" ]]; then
    python -m flycasso.train --resume "$run_dir/last.pt" --until-convergence
  else
    python -m flycasso.train --config "$config" --out "$run_dir" --until-convergence
  fi
fi
result="$run_dir/inference/$(python -c 'from flycasso.common import digest; import sys; print(digest(sys.argv[1])[:16])' "$run_dir/best.pt")"
if [[ ! -f "$result/samples/metadata.json" ]]; then
  python -m flycasso.sample --checkpoint "$run_dir/best.pt" --class all --out "$result/samples"
fi
python -m flycasso.evaluate --checkpoint "$run_dir/best.pt" --out "$result/evaluation.json"
python -m flycasso.export --checkpoint "$run_dir/best.pt" --out "$result/bundle"
