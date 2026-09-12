"""Full-graph rate dynamics inside a class-conditioned pixel-space denoiser.

There is no pretrained image generator or graph cropping. The U-Net variant
uses noisy-image skips; its category and timestep conditioning pass through the fly.
Input/output projections are engineered; this is not a physiological emulation.
"""

import json
import math

import numpy as np
import torch
from torch import nn


class FixedSparseMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, wt):
        ctx.wt = wt
        return torch.sparse.mm(w, x)

    @staticmethod
    def backward(ctx, grad):
        return torch.sparse.mm(ctx.wt, grad), None, None


def load_graph(path, control="real", seed=42):
    with np.load(path, allow_pickle=False) as f:
        ids, ptr, idx, values = (f[k].copy() for k in ("ids", "indptr", "indices", "values"))
        signs = f["signs"].copy()
        metadata = json.loads(str(f["metadata"]))
    n = len(ids)
    if any(a.ndim != 1 for a in (ids, ptr, idx, values, signs)) or any(a.dtype.kind not in "iu" for a in (ids, ptr, idx)):
        raise ValueError("Graph needs one-dimensional arrays and integer IDs/indices")
    if (n < 1 or np.any(ids[1:] <= ids[:-1]) or len(ptr) != n + 1 or ptr[0] != 0
            or ptr[-1] != len(idx) or len(values) != len(idx) or np.any(np.diff(ptr) < 0)
            or np.any(idx < 0) or np.any(idx >= n) or len(signs) != n
            or not np.isfinite(values).all() or np.any(values == 0) or not np.isin(signs, [-1, 1]).all()):
        raise ValueError("Malformed or edge-dropping graph artifact")
    if control == "shuffled":
        # Preserve incoming edge slots and total outgoing edge multiplicities.
        # Signs follow the NEW presynaptic neuron; duplicate pairs may result.
        idx = np.random.default_rng(seed).permutation(idx)
        values = np.abs(values) * signs[idx]
    elif control != "real":
        raise ValueError("graph_control must be real or shuffled")
    # int32 indices halve index traffic for this 166,700-node, 25.6M-edge graph.
    index_type = np.int32 if max(n, len(idx)) < 2**31 else np.int64
    w = torch.sparse_csr_tensor(torch.from_numpy(ptr.astype(index_type)), torch.from_numpy(idx.astype(index_type)),
                                torch.from_numpy(values), size=(n, n))
    return w, metadata


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net=nn.Sequential(nn.GroupNorm(8,channels),nn.SiLU(),nn.Conv2d(channels,channels,3,padding=1),
            nn.GroupNorm(8,channels),nn.SiLU(),nn.Conv2d(channels,channels,3,padding=1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x): return x+self.net(x)


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm=nn.GroupNorm(8,channels)
        self.attention=nn.MultiheadAttention(channels,4,batch_first=True)
        nn.init.zeros_(self.attention.out_proj.weight);nn.init.zeros_(self.attention.out_proj.bias)
    def forward(self,x):
        tokens=self.norm(x).flatten(2).transpose(1,2)
        update=self.attention(tokens,tokens,tokens,need_weights=False)[0]
        return x+update.transpose(1,2).reshape_as(x)


class FlyDenoiser(nn.Module):
    def __init__(self, graph, config):
        super().__init__()
        w, self.graph_metadata = load_graph(graph, config["graph_control"], config["seed"])
        n, width, size = w.size(0), config["width"], config["image_size"]
        self.n_neurons, self.n_edges = n, w.values().numel()
        self.recurrent_steps, self.leak = config["recurrent_steps"], config["leak"]
        self.synaptic_output = config.get("synaptic_output",False)
        self.size, self.width = size, width
        self.register_buffer("w", w, persistent=False)
        self.register_buffer("wt", w.transpose(0, 1).to_sparse_csr(), persistent=False)
        self.classes = config.get("classes", ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
        self.image_input = nn.Linear(3 * size * size, width)
        self.null_label = len(self.classes) if config.get("condition_dropout",0) else None
        self.guidance_scale = config.get("guidance_scale",1.)
        self.class_input = nn.Embedding(len(self.classes)+int(self.null_label is not None), width)
        self.time_input = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.input_to_neurons = nn.Linear(width, n)
        self.recurrent_gain = nn.Parameter(torch.full((n,), math.log(0.45 / 0.55)))
        self.neuron_bias = nn.Parameter(torch.zeros(n))
        self.neurons_to_output = nn.Linear(n, width)
        # Dense 166k-neuron readouts can drift far negative and kill SiLU gradients.
        self.readout_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 3 * size * size)
        if config.get("architecture") == "spatial":
            if size != 32:
                raise ValueError("Spatial image adapters require 32x32 images")
            self.image_input = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.SiLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(), nn.Flatten(), nn.Linear(64*8*8, width))
            self.output = nn.Sequential(nn.Linear(width, 64*4*4), nn.Unflatten(1, (64,4,4)),
                nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(64,64,3,padding=1), nn.SiLU(),
                nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(64,32,3,padding=1), nn.SiLU(),
                nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(32,32,3,padding=1), nn.SiLU(), nn.Conv2d(32,3,3,padding=1))
        if config.get("architecture") == "unet":
            self.encoder = nn.ModuleList([nn.Conv2d(3,32,3,padding=1), nn.Conv2d(32,64,3,2,1), nn.Conv2d(64,128,3,2,1)])
            self.image_input = nn.Linear(128*8*8,width)
            self.context = nn.ModuleList([nn.Linear(width,128),nn.Linear(width,64),nn.Linear(width,32)])
            self.decoder = nn.ModuleList([nn.Sequential(nn.Conv2d(128,128,3,padding=1),nn.GroupNorm(8,128)),
                nn.Sequential(nn.Conv2d(128+64,64,3,padding=1),nn.GroupNorm(8,64)),
                nn.Sequential(nn.Conv2d(64+32,32,3,padding=1),nn.GroupNorm(8,32))])
            self.output = nn.Sequential(nn.Conv2d(32,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,3,3,padding=1))
        if config.get("residual_blocks"):
            self.encoder_refine=nn.ModuleList([ResidualBlock(c) for c in (32,64,128)])
            self.decoder_refine=nn.ModuleList([ResidualBlock(c) for c in (128,64,32)])
            self.spatial_attention=SpatialAttention(128)
        self.spatial = config.get("architecture") == "spatial"
        self.register_buffer("frequency", torch.exp(-math.log(10000) * torch.arange(width // 2)
                             / max(width // 2 - 1, 1)), persistent=False)

    def _apply(self, fn, recurse=True):
        # MPS has no CSR tensor storage: retain CPU topology and upload its plain arrays.
        graphs = {name: self._buffers.pop(name) for name in ("w", "wt")}
        try:
            super()._apply(fn, recurse)
        finally:
            self._buffers.update(graphs)
        self._metal_graphs = None
        if next(self.parameters()).device.type == "mps":
            from metal import graph_arrays
            self._metal_graphs = tuple(graph_arrays(graph.cpu()) for graph in graphs.values())
        else:
            for name, graph in graphs.items():
                self._buffers[name] = fn(graph)
        return self

    def forward(self, image, timestep, label, ablate_edges=False):
        if image.shape[1:] != (3, self.size, self.size):
            raise ValueError("Image size differs from checkpoint architecture")
        phase = timestep.float()[:, None] * self.frequency[None]
        time = torch.cat([phase.sin(), phase.cos()], dim=1)
        if hasattr(self, "encoder"):
            features=[]; x=image
            for i,layer in enumerate(self.encoder):
                x=torch.nn.functional.silu(layer(x))
                if hasattr(self,"encoder_refine"): x=self.encoder_refine[i](x)
                features.append(x)
            code=self.image_input(x.flatten(1))+self.class_input(label)+self.time_input(time)
            context=self.neural_readout(code,ablate_edges)
            x=torch.nn.functional.silu(self.decoder[0](x)+self.context[0](context)[:,:,None,None])
            if hasattr(self,"decoder_refine"): x=self.spatial_attention(self.decoder_refine[0](x))
            for i,skip in enumerate(reversed(features[:-1]),1):
                x=torch.nn.functional.interpolate(x,scale_factor=2,mode="nearest")
                x=torch.nn.functional.silu(self.decoder[i](torch.cat([x,skip],1))+self.context[i](context)[:,:,None,None])
                if hasattr(self,"decoder_refine"): x=self.decoder_refine[i](x)
            return self.output(x)
        code = self.image_input(image if self.spatial else image.flatten(1)) + self.class_input(label) + self.time_input(time)
        return self.output(self.neural_readout(code, ablate_edges)).reshape_as(image)

    def neural_readout(self, code, ablate_edges=False):
        """Shared full-connectome computation for image and motor adapters."""
        code = torch.nn.functional.silu(code)
        projection = self.input_to_neurons
        if code.device.type == "mps" and torch.is_grad_enabled():
            # PyTorch 2.8 MPS gives wrong input gradients for the 166,700-wide GEMM.
            # Bound that backward reduction; keep every neuron and all work on the GPU.
            drive = torch.cat([torch.nn.functional.linear(code, w, b) for w, b in
                zip(projection.weight.split(8192), projection.bias.split(8192))], dim=1)
        else:
            drive = projection(code)
        drive = drive + self.neuron_bias
        state = torch.zeros_like(drive).t().contiguous()
        gain = 2 * self.recurrent_gain.sigmoid()[:, None]
        drive = drive.t().contiguous()
        for step in range(self.recurrent_steps):
            # W @ zero is zero: skip the first sparse multiply without changing dynamics.
            if ablate_edges or step == 0:
                recurrent = 0
            elif state.device.type == "mps":
                from metal import MetalSparseMultiply
                recurrent = MetalSparseMultiply.apply(state, *self._metal_graphs)
            else:
                recurrent = FixedSparseMultiply.apply(state, self.w, self.wt)
            if self.synaptic_output and step==self.recurrent_steps-1:
                # Final activity has no direct input/identity path around the synapses.
                state=torch.tanh(self.neuron_bias[:,None].expand_as(state)+gain*recurrent)
            else:
                state = (1 - self.leak) * state + self.leak * torch.tanh(drive + gain * recurrent)
        return torch.nn.functional.silu(self.readout_norm(self.neurons_to_output(state.t())))
