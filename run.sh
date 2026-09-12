#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
config="${1:-configs/full.json}"
run_dir="${2:-runs/diffusion}"
python prepare.py --target all
python train.py --config "$config" --out "$run_dir" --until-convergence
python sample.py --checkpoint "$run_dir/best.pt" --class all --out "$run_dir/final-samples"
python evaluate.py --checkpoint "$run_dir/best.pt" --out "$run_dir/evaluation.json"
python export.py --checkpoint "$run_dir/best.pt" --out "$run_dir/bundle"
python app.py --checkpoint "$run_dir/bundle/model.pt"
