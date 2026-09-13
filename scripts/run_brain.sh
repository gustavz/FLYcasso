#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
task="${1:-all}"
max_steps="${2:-20000}"
case "$task" in image|motor|all) ;; *) echo 'Use image, motor or all' >&2; exit 2;; esac
python -m flycasso.prepare --target all
python -m flycasso.anatomy
for current in image motor; do
  [[ "$task" == "$current" || "$task" == all ]] || continue
  run_dir="runs/brain-$current"
  trainer=flycasso.train_brain
  if [[ "$current" == motor ]]; then
    python -m flycasso.prepare_images
    python -m flycasso.strokes --prepare
    python -m flycasso.train_muscle --prepare
    trainer=flycasso.train_muscle
  fi
  if ! python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1])/"status.json"; sys.exit(not (p.exists() and json.loads(p.read_text())["status"]=="validation_plateau"))' "$run_dir"; then
    args=(--out "$run_dir"); [[ ! -f "$run_dir/last.pt" ]] || args=(--resume "$run_dir/last.pt")
    python -m "$trainer" "${args[@]}" --max-steps "$max_steps"
  fi
  result="$run_dir/inference/$(python -c 'from flycasso.common import digest; import sys; print(digest(sys.argv[1])[:16])' "$run_dir/best.pt")"
  if [[ ! -f "$result/evaluation/evaluation.json" ]]; then
    python -m flycasso.evaluate_brain --checkpoint "$run_dir/best.pt" --out "$result/evaluation"
  fi
  python -m flycasso.export --checkpoint "$run_dir/best.pt" --out "$result/bundle"
done
