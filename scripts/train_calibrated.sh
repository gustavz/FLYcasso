#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/flycasso-matplotlib}"
python="${PYTHON:-.venv/bin/python}"
"$python" -m flycasso.prepare --target all
"$python" -m flycasso.prepare_images
"$python" -m flycasso.strokes --prepare
if [[ ! -f data/processed/brain-ports-v3/manifest.json ]]; then
  "$python" -m flycasso.anatomy --out data/processed/brain-ports-v3 --calibrated
fi
[[ -f runs/quality/best.pt ]] || "$python" -m flycasso.quality
[[ -f runs/quality-cifar/best.pt ]] || "$python" -m flycasso.quality --data data/processed/cifar10 --out runs/quality-cifar --steps 3000
# Each process releases its circuit and optimizer before the other uses the GPU.
while true; do
  active=0
  for task in motor image; do
    run_dir="runs/calibrated-$task"
    if "$python" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1])/"status.json"; sys.exit(0 if p.exists() and json.loads(p.read_text()).get("status") in ("needs_review","curriculum_complete") else 1)' "$run_dir"; then
      continue
    fi
    active=1
    "$python" -m flycasso.train_calibrated "$task" --device "${FLYCASSO_DEVICE:-auto}" --updates "$([[ "$task" == motor ]] && echo 32 || echo 8)"
  done
  [[ "$active" == 1 ]] || break
done
