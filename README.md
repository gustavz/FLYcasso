<p align="center">
  <img src="web/assets/flycasso-flies.png" alt="Two FLYcasso flies" height="100">
  <img src="web/assets/flycasso-wordmark.png" alt="FLYcasso" height="100">
</p>

Two models built around the full MaleCNS fly connectome: image diffusion and one-leg drawing. Both take a category as input. Diffusion learns CIFAR-10 color images; drawing learns Quick, Draw! strokes.

| Brain Diffusion | Leg Painting |
| --- | --- |
| ![Completed dog samples](docs/assets/brain-diffusion.png) | ![Completed house drawing](docs/assets/leg-painting.png) |

## Architecture

| Brain Diffusion | Leg Painting |
| --- | --- |
| ![Image diffusion architecture](docs/assets/brain-diffusion-architecture.svg) | ![Stroke diffusion and leg control architecture](docs/assets/leg-painting-architecture.svg) |

## Setup

Python 3.12. Run from the repository root.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-paint.txt
```

## Train

```sh
bash run.sh        # diffusion: download, train, evaluate, export
bash run_motor.sh  # category drawing: download, train, evaluate, export
```

Training uses the GPU when available. Painting first learns pen control, then stroke generation. Weights are generated locally; no pretrained release is available yet.

Rerun the same command to resume interrupted training. Inference bundles and evaluation results go in each run's `inference/` directory. Stroke generation uses 30,000 steps; this is a training budget, not a convergence guarantee.

## App

After training has saved a checkpoint:

```sh
python app.py --checkpoint runs/diffusion/best.pt --follow-training
```

Open http://localhost:7860. The app has diffusion, pen painting and training tabs.

## Development

Python entry points stay at the root. `configs/` holds training settings and source pins; `scripts/` holds benchmarks and asset preparation; `web/` contains the app, assets and vendored Three.js; `tests/` contains the checks.

```sh
python -m unittest discover -s tests
node --test tests/*.cjs
python -m scripts.benchmark --help
python -m scripts.benchmark_apple --help
```

The Python checks exercise a small synthetic training run, exact resume, export, HTTP inference, pen contact and Apple GPU gradients. They do not train the full fly to convergence. JavaScript checks require Node.js; the app itself has no Node.js dependency.

Model details: [docs/MODEL_CARD.md](docs/MODEL_CARD.md). Code: [MIT](LICENSE). Data and asset credits: [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md).
