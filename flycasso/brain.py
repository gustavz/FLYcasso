"""Two independent full-circuit tasks. No learned image encoder or decoder."""
import numpy as np
import torch
from torch import nn
from flycasso.circuit import Circuit
from flycasso.diffusion import Diffusion


def cues(labels, clock, seed):
    category=torch.nn.functional.one_hot(labels,10).float()
    return torch.cat([category,clock[:,None],torch.sin(clock[:,None]*torch.pi),
                      torch.cos(clock[:,None]*torch.pi),seed],1)


class Brain(nn.Module):
    def __init__(self, graph, ports, task="image", ticks=8, control="real"):
        super().__init__()
        if task not in ("image","motor") or type(ticks) is not int or ticks<1:
            raise ValueError("Use image/motor and positive integer neural updates")
        with np.load(ports,allow_pickle=False) as f: ports={k:f[k].copy() for k in f.files}
        if ports['output_index'].shape[0]!=(3072 if task=='image' else 15) or ports['input_channel'].max()>=(3088 if task=='image' else 3102):
            raise ValueError('Ports do not match the task input/output dimensions')
        self.core=Circuit(graph,ports,control);self.task=task;self.ticks=ticks;self.size=32
        self.n_neurons,self.n_edges=self.core.n_neurons,self.core.n_edges
        self.classes=(['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'] if task=='image'
                      else ['cat','flower','butterfly','fish','bird','tree','house','star','apple','umbrella'])
        self.guidance_scale=1.

    def forward(self, image, labels, clock, seed, state=None, proprio=None, ablate=False, coefficients=None):
        signals=torch.cat([image.flatten(1),cues(labels,clock,seed)],1)
        if self.task=="motor":
            if proprio is None or proprio.shape!=(len(image),14): raise ValueError("Motor task needs 14 proprioceptive channels")
            signals=torch.cat([signals,proprio],1)
        state=self.core(signals,state,self.ticks,ablate,coefficients)
        out=self.core.read(state)
        # Fixed amplifier, comparable to a calibrated recording interface.
        out=out.reshape(-1,3,32,32) if self.task=="image" else torch.sigmoid(8*out-2)
        return out,state

    @torch.no_grad()
    def sample(self, labels, seed=42, steps=16, ablate=False, reset_state=False, on_step=None):
        if self.task!="image" or not 2<=steps<=1000: raise ValueError("Image task needs 2–1000 sampling steps")
        device=next(self.parameters()).device;rng=torch.Generator().manual_seed(seed)
        x=torch.randn((len(labels),3,32,32),generator=rng).to(device)
        cue=torch.randn((len(labels),3),generator=rng).to(device).tanh()
        diffusion=Diffusion(1000,device);state=None;frames=[x.clamp(-1,1).cpu()]
        if on_step:on_step(0,frames[-1])
        coefficients=self.core.coefficients()
        times=torch.linspace(999,0,steps).round().long().tolist()
        for i,t in enumerate(times):
            ts=torch.full((len(labels),),t/999,device=device)
            clean,state=self(x,labels,ts,cue,None if reset_state else state,ablate=ablate,coefficients=coefficients)
            clean=clean.clamp(-1,1);a=diffusion.alpha_bar[t]
            eps=(x-a.sqrt()*clean)/(1-a).sqrt()
            prev=diffusion.alpha_bar[times[i+1]] if i+1<len(times) else torch.ones((),device=device)
            x=prev.sqrt()*clean+(1-prev).sqrt()*eps
            if not torch.isfinite(x).all():raise FloatingPointError("Nonfinite recurrent sample")
            frames.append(x.cpu())
            if on_step:on_step(i+1,frames[-1])
        return x.cpu(),frames
