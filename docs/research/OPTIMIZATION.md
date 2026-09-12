# Quality and performance experiments

Historical research snapshot: 12 September 2026. These are proposed experiments, not implemented improvements or promised speedups. Keep the full 166,700-neuron, 25,582,938-edge circuit. Diffusion stays on CIFAR-10; drawing stays on Quick, Draw!.

## Observed runs at the time

| | CIFAR diffusion | Category drawing |
|---|---|---|
| Training data | 45,000 images, 10 classes | 100,000 drawings, 10 classes |
| Snapshot step | 2,750 | 3,000 |
| Examples processed at batch 32 | 88,000 (~1.96 dataset passes) | 96,000 (~0.96 passes on the expanded data) |
| EMA validation MSE | 0.05564 | 0.05623 |
| Generated category recognition | Not measured for this run | 12/30, using a classifier with 94.02% held-out accuracy |

Sources: local `runs/{diffusion,motor}/metrics.jsonl`, data manifests and previews. Drawing adapters were initialized from the earlier three-category model; CIFAR training started fresh. Both runs were still training at this snapshot. Their MSE values measure different tasks and are not comparable quality scores. Thirty generated drawings are too few for a reliable model ranking.

The inspected CIFAR preview is blurry, with weak object structure. The drawing preview has fragmented contours and misplaced strokes. Accurate leg control cannot repair a poor generated plan: evaluate the plan and the executed ink separately.

Already present: EMA, cosine noise, Min-SNR weighting, classifier-free guidance, an image U-Net with residual blocks and bottleneck attention, full-graph Metal kernels, and CPU/GPU gradient checks. [Min-SNR](https://arxiv.org/abs/2303.09556) is therefore not a new optimization to add.

## Recommended order

### 1. Improve checkpoint selection and training duration

`common.Plateau` can halve the learning rate after two stale validation checks. Its 20,000-step minimum only delays stopping, not reductions. The CIFAR learning rate is already down from 0.0003 to 0.00015. Validation uses just 256 fixed draws with replacement, and checkpoint selection uses denoising MSE alone.

First compare a continuation with the current schedule against one that delays reductions until sufficient data exposure. Use stratified validation across categories and noise levels, plus generated-sample quality. Review at 20,000 and 50,000 steps; neither milestone means convergence. Keep dataset size fixed while establishing this baseline.

For scale, EDM's reference training defaults to 200 million image presentations with batch 512. That is context for how early our run is, not a sensible budget to copy onto this Mac. Larger batches or gradient accumulation are experiments, not free speed: compare quality per wall-clock hour and per example seen. [EDM training configuration](https://raw.githubusercontent.com/NVlabs/edm/main/train.py)

### 2. Teach the drawing model when to lift its pen

In the current 5,000-drawing validation set, only **1.614%** of the 256-point pen states are lifts, averaging 4.13 strokes per drawing. `strokes.py` diffuses absolute X, Y and a continuous ±1 pen flag; the loss averages all three channels. A small aggregate error can hide incorrect stroke boundaries.

Start with separate reporting of coordinate error and pen-lift precision/recall. Then compare a dedicated pen-state classification head, trained from the same noisy sequence, against the current continuous flag. Calibrate its loss and threshold on validation data; blindly weighting rare lifts too heavily could create more fragmentation. This requires adapting training and decoding together, rather than plugging binary logits into the existing Gaussian sampler.

[SketchRNN](https://arxiv.org/pdf/1704.03477) uses relative movement and categorical pen-down, pen-up and end states, explicitly discussing their imbalance. Borrow the representation lesson while retaining our fly circuit and parallel diffusion generator.

### 3. Improve stroke geometry and global shape

Test relative displacements against absolute coordinates, with training-set normalization and reconstructed-position loss to limit accumulated drift. Preserve pen-up movements between strokes. Relative displacements cannot reuse the current absolute-coordinate clipping and decoder unchanged. Version the prepared data and retrain the stroke adapters; keep the verified motor controller frozen.

[SketchKnitter](https://github.com/wangqiang9/SketchKnitter) is the closest published vector-diffusion reference. Its reported absolute-coordinate ablation is substantially worse than its main model. It also uses a stronger U-Net and recognizability-aware sampling. Those results motivate experiments; they do not establish a gain for FLYcasso.

After testing representation alone, add small residual blocks and one bottleneck attention layer to the current shallow 1D U-Net. If geometric losses still miss recognizable shape, test a low-weight raster reconstruction loss on predicted clean strokes. Retain coordinate and boundary losses. Any recognizability model used for training must be separate from the final evaluator.

### 4. Strengthen the image model's conditioning

Currently the full circuit produces one 128-value context, added at three decoder levels. Test scale-and-shift conditioning inside normalized residual blocks so brain output can modulate features more directly. Keep category and timestep information routed through the circuit. This adapts the normalization approach used in [guided-diffusion](https://github.com/openai/guided-diffusion), not its pretrained image generator.

Before widening the model, inspect neuron saturation, activation variance and gradient norms through the input, recurrence and readout. A large connectome is not evidence that its interface is using the available capacity well. Compare intact edges, disabled edges and shuffled connectivity on identical seeds, and measure category fidelity as well as denoising loss.

If this still plateaus, try EDM-style input/output preconditioning and noise-level sampling as a separate training run. [EDM](https://github.com/NVlabs/edm) demonstrates strong CIFAR results with these choices. This changes the training formulation; it is not a checkpoint-compatible switch. Its published implementation has a noncommercial share-alike license, so do not copy it into this MIT project without handling that distinction.

### 5. Reduce sampling work without sacrificing quality

Benchmark **DPM-Solver++ 2M** at 20, 30 and 50 evaluations against current 50-step DDIM, using identical checkpoint, categories and initial noise. Start with its discrete VP schedule wrapper and `x_start` prediction; test log-SNR spacing for CIFAR. Count model evaluations and effective batch size, not just displayed steps. Our guidance evaluates conditional and unconditional branches together, doubling the batch.

[DPM-Solver's official implementation](https://github.com/LuChengTHU/dpm-solver) supports existing data-prediction models without retraining. Its authors warn that a faster solver cannot fix poor converged sample quality. Establish a high-step DDIM reference on a small fixed subset before choosing fewer steps. Going from 50 to 20 evaluations removes 60% of denoiser calls; the actual latency gain must be measured.

Then compare guidance strength 1, 1.5 and 2, and guidance restricted to intermediate noise levels. [Limited-interval guidance](https://arxiv.org/abs/2404.07724) reports both quality and speed benefits on other diffusion models. Outside the chosen interval, skip the unconditional branch. Tune on our validation seeds rather than assuming the paper's interval transfers.

### 6. Remove measured runtime overhead

`Diffusion.sample` currently copies every intermediate frame to CPU, even when the caller discards the frames. It also synchronizes for a finite-value check every step. Training reads several GPU scalars each iteration. Profile these alongside graph multiplication, convolution, checkpoint loading, image encoding and physics.

Make frame collection optional and decouple preview frequency from denoising steps. Keep finite-output validation before returning or streaming results. Compare warm single-image latency, throughput and memory with training paused versus concurrent training; current timings include GPU contention.

The app already uses Apple's GPU through [PyTorch MPS](https://developer.apple.com/metal/pytorch/) and custom Metal graph multiplication. Use the [MPS profiler](https://docs.pytorch.org/docs/stable/generated/torch.mps.profiler.profile.html) before changing frameworks. Benchmark a newer PyTorch release in an isolated environment, preserving the existing full-graph gradient checks and large-linear workaround until parity passes.

[MLX](https://ml-explore.github.io/mlx/build/html/index.html) offers lazy execution, compilation and shared CPU/GPU memory. It is an option for a small inference benchmark, not a guaranteed acceleration or a reason to maintain two complete training stacks. Try mixed precision only after profiling, retaining float32 graph accumulation and checking sample quality and gradient parity.

## Data and acceptance checks

- Keep the current ten classes while diagnosing learning. The Quick Draw loader takes the first eligible records before shuffling; a later dataset revision should use deterministic reservoir sampling across each full file. Save IDs and hashes, exclude existing evaluation IDs, remove exact duplicates, and split before augmentation. Apply any stroke transform to coordinates and raster targets consistently. The [official dataset format](https://github.com/googlecreativelab/quickdraw-dataset) supplies drawing IDs and recognition flags, but recognition flags are not a guarantee of clean geometry.
- Screen checkpoints with at least 100 generated examples per category across fixed validation seeds. Report per-class recognition, diversity, nearest training examples, and unselected image grids. Keep final test seeds and data out of checkpoint selection. Category-only generation has many valid outputs: pixel distance to one arbitrary reference is not a valid success criterion.
- For CIFAR, add distribution metrics with explicit preprocessing and sample counts. Use smaller evaluations only for internal ranking, with uncertainty; reserve a 50,000-sample FID run for a serious final comparison, following the [EDM evaluation protocol](https://github.com/NVlabs/edm#calculating-fid). Do not compare a tiny-sample score to published FID numbers.
- For drawings, evaluate generated raster quality and stroke-boundary statistics, then run the learned controller on those same plans. Require the existing execution gate (stroke F1 at least 0.9 at 0.01 mm tolerance) to continue passing. Recognizable plans and faithful physical execution are separate requirements.
- Promote a change only when improvements hold across seeds and classes, diversity remains healthy, and fly-edge ablations still degrade conditioning. For speed-only changes, require equivalent quality and record median and tail latency on the same machine.

Run one change at a time, keep the previous checkpoint, and compare at matched compute budgets. First priorities: better quality evaluation, adequate training before decay, and pen-state modelling. Sampler and frame-transfer improvements can then make a good model faster.
