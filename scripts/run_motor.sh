#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_dir="${1:-runs/motor}"
python -m flycasso.prepare --target malecns
python -m flycasso.prepare_strokes
python -m flycasso.prepare_images
python -m flycasso.strokes --prepare
if [[ ! -f "$run_dir/last.pt" ]]; then
  if ! python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1])/"status.json"; sys.exit(not (p.exists() and json.loads(p.read_text())["status"]=="control_quality_reached"))' "$run_dir-control"; then
    if [[ -f "$run_dir-control/last.pt" ]]; then
      python -m flycasso.train_motor --resume "$run_dir-control/last.pt"
    else
      python -m flycasso.train_motor --out "$run_dir-control"
    fi
  fi
  python -m flycasso.train_motor --phase strokes --init-from "$run_dir-control/best.pt" --out "$run_dir" --until-convergence
else
  if ! python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1])/"status.json"; sys.exit(not (p.exists() and json.loads(p.read_text())["status"]=="validation_plateau"))' "$run_dir"; then
    python -m flycasso.train_motor --phase strokes --resume "$run_dir/last.pt" --until-convergence
  fi
fi
result="$run_dir/inference/$(python -c 'from flycasso.common import digest; import sys; print(digest(sys.argv[1])[:16])' "$run_dir/best.pt")"
python -m flycasso.evaluate_motor --checkpoint "$run_dir/best.pt" --category all --out "$result/evaluation"
python -m flycasso.export --checkpoint "$run_dir/best.pt" --out "$result/bundle"
