#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
run_dir="${1:-runs/motor}"
python prepare.py --target malecns
python prepare_strokes.py
python prepare_images.py
python strokes.py --prepare
if [[ ! -f "$run_dir/last.pt" ]]; then
  if ! python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1])/"status.json"; sys.exit(not (p.exists() and json.loads(p.read_text())["status"]=="control_quality_reached"))' "$run_dir-control"; then
    if [[ -f "$run_dir-control/last.pt" ]]; then
      python train_motor.py --resume "$run_dir-control/last.pt"
    else
      python train_motor.py --out "$run_dir-control"
    fi
  fi
  python train_motor.py --phase strokes --init-from "$run_dir-control/best.pt" --out "$run_dir" --until-convergence
else
  if ! python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1])/"status.json"; sys.exit(not (p.exists() and json.loads(p.read_text())["status"]=="validation_plateau"))' "$run_dir"; then
    python train_motor.py --phase strokes --resume "$run_dir/last.pt" --until-convergence
  fi
fi
result="$run_dir/inference/$(python -c 'from common import digest; import sys; print(digest(sys.argv[1])[:16])' "$run_dir/best.pt")"
python evaluate_motor.py --checkpoint "$run_dir/best.pt" --category all --out "$result/evaluation"
python export.py --checkpoint "$run_dir/best.pt" --out "$result/bundle"
