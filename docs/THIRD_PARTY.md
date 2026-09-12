# Sources, attribution and licenses

## MaleCNS v1.0 connectome

- Publisher/download: https://male-cns.janelia.org/download/
- Data contributors: FlyEM at HHMI Janelia Research Campus, University of Cambridge, MRC Laboratory of Molecular Biology, Google Research, and the collaborators credited by the release.
- Dataset license: **Creative Commons Attribution 4.0 International** (https://creativecommons.org/licenses/by/4.0/).
- Exact source URLs, release names and hashes: `sources.json`.
- Modifications: select annotated neuronal objects, exclude explicit Glia/unassigned objects, remap exact IDs to matrix positions, aggregate any duplicate edges, assign a simplified transmitter sign, normalize by incoming contact counts, serialize CSR arrays. Original retained contact counts and body IDs remain available in the processed graph.
- Keep this attribution with redistributed processed graphs; follow the release page's current requested publication citations. This project is not endorsed by the data producers or Google.

## CIFAR-10

- Alex Krizhevsky, Vinod Nair and Geoffrey Hinton; https://cave.cs.toronto.edu/kriz/cifar.html
- Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
- Downloads use the official binary archive. FLYcasso does not relicense the dataset or bundle the original images in the source repository. The dataset page does not provide a blanket MIT/CC-BY grant that this project can extend to you. Consult the original source for image/data rights before redistributing datasets or downstream artifacts.
- Preprocessing preserves pixel content; the training split gets a deterministic held-out validation partition. Labels become conditioning IDs.

## Related projects and methods

- DoomFly, https://github.com/nftechie/doomfly (MIT): reference for full MaleCNS node coverage, source hashes and published import policy. FLYcasso does not load DoomFly's policy weights or run its spiking simulator. Its denoiser and data pipeline are separate implementations.
- ngxson's fly language model, https://huggingface.co/ngxson/fly-llm-hf: related fly-network language generation. No weights from it are used here.
- Andrej Karpathy's nanoGPT and nanochat: https://github.com/karpathy/nanoGPT and https://github.com/karpathy/nanochat (MIT). Inspiration for a direct, readable end-to-end project layout; no claim of a fork or shared model architecture.
- Ho et al., *Denoising Diffusion Probabilistic Models* (2020): https://arxiv.org/abs/2006.11239
- Nichol and Dhariwal, *Improved Denoising Diffusion Probabilistic Models* (2021), cosine schedule: https://arxiv.org/abs/2102.09672
- Song, Meng and Ermon, *Denoising Diffusion Implicit Models* (2020): https://arxiv.org/abs/2010.02502

Dependencies retain their own licenses: PyTorch (BSD-style), NumPy/SciPy/pandas (BSD), Apache Arrow (Apache-2.0), Pillow (HPND). See each installed distribution's license notices. FLYcasso's MIT license applies to its original code, not third-party data.

The sampler uses clean-image (`x0`) prediction, also supported as `prediction_type="sample"` in the reference [Diffusers DDIM scheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py). The compact readout motivates this engineering choice; image quality remains an experimental question.
# Front-leg painting additions

- **Quick, Draw! dataset**: made available by Google, Inc. under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source: [googlecreativelab/quickdraw-dataset](https://github.com/googlecreativelab/quickdraw-dataset). The simplified cat, flower and butterfly vector files are pinned in `stroke-sources.json`. FLYcasso selects disjoint drawing IDs, rescales/reorients paths into a fly-sized canvas, adds lift motions and derives simulated joint demonstrations. These are modifications of the human drawing data, not recorded fly motions. Google does not endorse this project.
- **NeuroMechFly / FlyGym 2.1.0**: [NeLy-EPFL/flygym](https://github.com/NeLy-EPFL/flygym), Apache-2.0. Derived body assets retain the license in `web/assets/FLYGYM-LICENSE.txt`. FLYcasso composes a tethered model, adds two front-tarsus brushes and a canvas, and exports simplified body meshes and transforms for viewing. The male connectome and NeuroMechFly's female-derived body come from different specimens; this is an engineered coupling, not a matched biological digital twin.
- **MuJoCo 3.9.0**: [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco), Apache-2.0, installed as a dependency for physical dynamics.
- **Three.js 0.180.0**: [mrdoob/three.js](https://github.com/mrdoob/three.js), MIT. Unmodified module/core/OrbitControls files are vendored with `web/assets/THREE-LICENSE.txt`; exact hashes are in `web/assets/sources.json`.

## Brand illustration

`web/assets/flycasso-flies.png` was generated with the built-in OpenAI image tool on 12 September 2026, using the user's Picasso portrait as a style reference. Brief: two original cubist flies, heavy black contours, flat red/yellow/green/periwinkle planes; one with a black beret and pencil, the other with an electrode cap. Background cleanup removed the generated checkerboard and produced a transparent PNG. The reference portrait is not bundled. `web/assets/flycasso-wordmark.png` is generated FLYcasso lettering inspired by the user’s Picasso signature reference; it is not a font file. The signature reference is not bundled.
