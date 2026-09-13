# Circuit-first models

Experimental feature branch. The previous models and their checkpoints remain available.

```sh
bash scripts/run_brain.sh             # prepare and train both tasks
python -m flycasso.app --checkpoint runs/brain-image/best.pt \
  --motor-checkpoint runs/brain-motor/best.pt --follow-training
```

Each model retains 166,700 neurons and 25,582,938 directed edges. Trainable parameters are one gain per existing edge, neuron biases and leak rates. Input/output ports have **zero trainable parameters**. The two tasks have independent weights. The launcher trains them sequentially; batch 16 is the default for the 16 GB development Mac.

## Image

![Circuit-first image architecture](assets/circuit-image-architecture.svg)

Noisy RGB image → fixed optic-lobe stimulation → recurrent circuit → fixed local neural readout → DDIM update.

L1/L2/L5 column annotations locate the artificial RGB input electrodes; Tm1/Tm2/Tm9 populations supply the readout. This assignment is not natural color physiology. Category, time and random cues stimulate annotated visual-centrifugal neurons, which feed back into the optic lobes. These are artificial task signals. There is no U-Net, learned image decoder, or input-to-output skip around the circuit.

The neural state persists through 16 denoising steps, with four neural updates per step. Training uses correlated forward-noise trajectories and truncated backpropagation through four denoising steps. Sampling uses generated trajectories; this exposure difference needs evaluation. State resets between images, not between denoising steps.

## Muscle painting

![Circuit-first muscle architecture](assets/circuit-motor-architecture.svg)

Category, clock, random cue, own canvas and proprioception → recurrent circuit → named front-left motor populations → 15 muscle activations → FlyMimic/MuJoCo → contact ink and sensory feedback.

Category and clock cues stimulate annotated descending neurons; visual and proprioceptive feedback enter their declared sensory ports. There is no separate stroke planner. Inference does not receive a target path or call inverse kinematics. The circuit advances continuously through a 256-action episode. Training uses Quick Draw targets and inverse-dynamics supervision on the circuit’s own executed states. A teacher-action mixture fades to zero over 512 optimizer steps. Truncated backpropagation spans 16 control actions. Held-out imitation error, on-policy control error and free-running execution are evaluated separately.

Preparation uses 32 training and eight validation/test sketches per category, ten categories. The motion curriculum is deliberately smaller than the source Quick Draw dataset because it requires executed muscle trajectories. The category generator's test references are not training inputs.

The motor interface groups neurons by annotated muscle target. Split muscle heads share their named motor pool. Individual muscle-head innervation and sensory tuning are unresolved; these are declared approximations, not a validated biological reconstruction. The graph and body are different specimens. The body is tethered and only pen–paper collisions are enabled. Exact selected neuron IDs and port checksums are stored in `data/processed/brain-ports-v2/manifest.json`.

## Training and evaluation

Both trainers use AdamW, explicit penalties on departures from initial circuit parameters, gradient clipping, EMA checkpoints, validation-based learning-rate reduction and stopping. They save weights, optimizer, random state, configuration and artifact hashes. Unchanged synaptic coefficients are reused within each backpropagation window and throughout inference. Checkpoints are separate from the original runs.

Stopping on denoising or muscle-imitation error does not certify recognizable drawings. Compare generated samples, free-running pen contact, disabled edges, reset neural state, and separately trained rewired controls. Biological benefit and improved quality remain experimental questions.

```sh
python -m flycasso.evaluate_brain --checkpoint runs/brain-image/best.pt --out reports/image
python -m flycasso.evaluate_brain --checkpoint runs/brain-motor/best.pt --out reports/motor
```

`--evaluator` optionally supplies an independent sketch classifier. It is used only for evaluation.

## Sources

- [MaleCNS](https://male-cns.janelia.org/): measured connectivity and cell annotations.
- [FlyMimic](https://github.com/gizemozd/FlyMimic): muscle-actuated body and motion recordings, distributed through FlyGym 2.1.0.
- [FlyGym muscle documentation](https://neuromechfly.org/tutorials/6_muscle_imitation/): supported muscles, sensors and simulation limits.

Prepared muscle assets retain their upstream licenses. The initial mesh acquisition uses FlyGym's public asset loader; downloaded assets are local and are not committed here.

## Peripheral movement in the leg-painting studio

The studio also reads 24 joint commands from the same recurrent neural state:
three joints in each of the other five legs, neck yaw, abdomen pitch, proboscis
pitch, and both wings, antenna bases and halteres. Distal segments follow their
parent joints. This is coarse articulation, not an individually actuated model
of every anatomical joint or muscle.

The original 15 painting muscles retain their dynamics. Additional joints use
fixed population pooling and bounded position actuators with passive stiffness,
damping and force limits. Named leg-muscle antagonists determine the leg pools;
other axes and gains are engineered body-region mappings. The exact neuron IDs,
weights, graph checksum and annotation checksum are recorded in `body.npz`, built
by `flycasso.anatomy` and included in motor inference exports.

The thorax remains anchored, and only pen–paper collisions are enabled. Other
joint motion has no sensory feedback into the circuit in this experiment. Only
the painting task is trained; the extra readouts have no learned parameters and
are attached during studio execution. They can change as the shared circuit is
trained, but receive no movement targets. This does not claim natural grooming,
flight or locomotion. Paired physics checks verify that the added motion leaves
the painting leg's dynamics unchanged within numerical tolerance.
