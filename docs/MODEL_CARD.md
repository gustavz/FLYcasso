# Models

Both models use the full MaleCNS v1.0 graph: 166,700 neurons and 25,582,938 directed connections. No additional neuron or edge pruning is applied. Source URLs and checksums are in `sources.json`.

Connections are normalized by incoming contact count. GABA/glutamate sources receive negative weights; others receive positive weights. These are engineered rate dynamics, not receptor-level physiology. Graph weights stay fixed; adapters, neuron gains and biases are learned.

Intermediate updates use `h = (1-leak)*h + leak*tanh(drive + gain*(W @ h))`.

The final update is `h = tanh(bias + gain*(W @ h))`. It removes the direct input/identity route around the synapses. Category and timestep information must cross the fly connections before reaching the image or stroke decoder. Earlier checkpoints without `synaptic_output` are retained only as baselines: removing their connections barely changed denoising error.

## Image diffusion

- All 10 CIFAR-10 categories. Output: 32×32 color images.
- U-Net residual blocks, image skips and bottleneck attention model local detail and whole shapes. Category and timestep conditioning pass through the entire fly circuit. This is a hybrid architecture, not a claim that the connectome alone produces every pixel.
- Width 128; three circuit updates per denoising step. No pretrained image generator.
- Min-SNR clean-image prediction, cosine noise, DDIM sampling and classifier-free guidance. Ten percent of training examples omit the category; inference combines both predictions from the same model. Checkpoints are selected using EMA validation loss.
- CIFAR-10: 45,000 training images, 5,000 validation images, and the untouched official 10,000-image test set.

## Category drawing

- A separate painting model shares one full circuit between stroke generation and physical control. Width 64; three circuit updates.
- First, the controller learns from executed teacher movements and corrections collected at states visited by its own policy. The loss uses exact pen forward kinematics and its MuJoCo Jacobian, alongside joint imitation.
- Inputs include joint positions, velocities and target displacement. The circuit predicts small corrections to the current pose, rather than relearning absolute joint positions. Only the seven left front-leg outputs control the pen; the other front leg remains at rest.
- Initial demonstrations use 24 short training drawings. Recovery trajectories expand this curriculum. Complete held-out drawings measure pen tracking and actual ink coverage.
- Next, category-conditioned diffusion learns 256-point stroke sequences across five spatial scales. The trained control circuit is held fixed while the stroke adapters learn. Quick, Draw! supplies 100,000 training sketches, 5,000 validation sketches, and 5,000 test sketches across cat, flower, butterfly, fish, bird, tree, house, star, apple and umbrella. Motor validation/test IDs are excluded from stroke training. Rasterized copies serve stroke evaluation only; diffusion uses CIFAR-10.
- Inference receives only a category and seed. It generates new strokes and executes them through the learned controller. It does not retrieve references or run inverse kinematics.
- Ink comes only from simulated pen–paper contact. The thorax is tethered. The MaleCNS graph and female-derived NeuroMechFly body are different specimens.

## Evaluation

Low loss is not evidence of recognizable output. Inspect generated samples, class consistency and diversity. Controller checks report actual-ink precision/recall within 0.01 mm and pen-position error. Accuracy against a generated plan measures execution, not whether that plan depicts the requested category.

Reserve the test split for final evaluation. Compare intact and disabled-edge inference, and separately trained shuffled-graph controls, before claiming a benefit from biological wiring.

Image validation plateaus are operational stopping conditions. The controller's quality gate requires stroke F1 ≥0.90 and mean tip error <0.01 mm on three held-out drawings. The default category-training script runs 30,000 steps; this budget is not quality certification.

Exports contain inference weights, the graph, hashes and notices. No reference files are needed for inference. Resume from original training checkpoints, not exports.

Fresh motor runs start with zero joint corrections and scale the wide readout’s learning rate.

`quality.py` can train a separate recognition evaluator with `--data` and `--out`. Evaluate fresh samples with `--checkpoint`, `--evaluator` and `--out`; add `--motor` for strokes. Evaluators must match the generator’s categories and never guide generation.

CPU and Apple Metal are tested; CUDA is untested here. Metal uses float32 CSR kernels and a blocked projection to avoid a PyTorch 2.8 backward reduction error. Physics stays at 10 kHz; animation and ink are recorded at 100 Hz. Cross-device bit identity is not guaranteed.

The brain cap and grooming animation illustrate generation; they are not neural telemetry or learned grooming. Credits: [THIRD_PARTY.md](THIRD_PARTY.md).

Architecture references: [DDPM](https://github.com/hojonathanho/diffusion), [SketchKnitter](https://github.com/wangqiang9/SketchKnitter), [classifier-free guidance](https://arxiv.org/abs/2207.12598).
