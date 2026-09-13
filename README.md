<p align="center">
  <img src="web/assets/flycasso-flies.png" alt="Two FLYcasso flies" height="100">
  <img src="web/assets/flycasso-wordmark.png" alt="FLYcasso" height="100">
</p>

Stopped experiment: recurrent image diffusion and muscle-driven leg painting using the full MaleCNS connectome. Synaptic gains, neuron dynamics and small local interfaces are learned. Neither run passed its first curriculum gate. The working studio remains on `main`.

| Brain Diffusion | Leg Painting |
| --- | --- |
| ![Circuit image architecture](docs/assets/calibrated-image-architecture.svg) | ![Circuit muscle architecture](docs/assets/calibrated-motor-architecture.svg) |

The screenshots below show the previous models. Training on this feature branch was stopped on 2026-09-13; code and local checkpoints are retained for future work.

| Brain Diffusion | Leg Painting |
| --- | --- |
| ![Completed dog samples](docs/assets/brain-diffusion.png) | ![Completed house drawing](docs/assets/leg-painting.png) |

## Setup

Python 3.12. Run from the repository root.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements/paint.txt
```

## Train

```sh
bash scripts/train_calibrated.sh # prepare both datasets, resume both curricula
```

The launcher alternates GPU slices and advances stages only after measured quality gates pass. Local checkpoints are retained in `runs/calibrated-image/` and `runs/calibrated-motor/`. No pretrained release is provided for this experiment.

## App

```sh
python -m flycasso.app --checkpoint runs/calibrated-image/best.pt \
  --motor-checkpoint runs/calibrated-motor/best.pt --follow-training
```

Open http://localhost:7860 after both tasks have saved a checkpoint.

Architecture and assumptions: [docs/BRAIN_FIRST.md](docs/BRAIN_FIRST.md). The previous models remain available through `scripts/run.sh`, `scripts/run_motor.sh` and their original checkpoint paths.

## Development

```text
flycasso/     Python models, data preparation, training and inference
scripts/      Training launchers, benchmarks and asset preparation
configs/      Training settings and pinned data sources
requirements/ Python dependencies
web/          Studio UI, assets and vendored Three.js
tests/        Python and JavaScript checks
docs/         Model card, references and screenshots
```

```sh
python -m unittest discover -s tests
node --test tests/*.cjs
python -m scripts.benchmark --help
python -m scripts.benchmark_apple --help
```

The Python checks exercise a small synthetic training run, exact resume, export, HTTP inference, pen contact and Apple GPU gradients. They do not train the full fly to convergence. JavaScript checks require Node.js; the app itself has no Node.js dependency.

Model details: [docs/MODEL_CARD.md](docs/MODEL_CARD.md). Code: [MIT](LICENSE). Data and asset credits: [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md).
