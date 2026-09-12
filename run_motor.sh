#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
run_dir="${1:-runs/motor}"
python prepare.py --target malecns
python prepare_strokes.py
python prepare_images.py
python strokes.py --prepare
python train_motor.py --out "$run_dir-control"
python train_motor.py --phase strokes --init-from "$run_dir-control/best.pt" --out "$run_dir" --steps 30000
python evaluate_motor.py --checkpoint "$run_dir/best.pt" --category all --out "$run_dir/evaluation"
python export.py --checkpoint "$run_dir/best.pt" --out "$run_dir/bundle"
