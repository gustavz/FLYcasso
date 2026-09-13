"""Two independent full-circuit tasks. Optional local electrode calibration; no generator bypass."""
import numpy as np
import torch
from torch import nn
from flycasso.circuit import Circuit
from flycasso.diffusion import Diffusion


def cues(labels, clock, seed):
    category=torch.nn.functional.one_hot(labels,11)[:,:10].float()
    return torch.cat([category,clock[:,None],torch.sin(clock[:,None]*torch.pi),
                      torch.cos(clock[:,None]*torch.pi),seed],1)


class Brain(nn.Module):
    def __init__(self, graph, ports, task="image", ticks=8, control="real", calibration=None, size=32):
        super().__init__()
        if task not in ("image","motor") or type(ticks) is not int or ticks<1:
            raise ValueError("Use image/motor and positive integer neural updates")
        with np.load(ports,allow_pickle=False) as f: ports={k:f[k].copy() for k in f.files}
        if ports['output_index'].shape[0]!=((9216 if calibration else 3072) if task=='image' else 15) or ports['input_channel'].max()>=(3088 if task=='image' else 3102):
            raise ValueError('Ports do not match the task input/output dimensions')
        self.core=Circuit(graph,ports,control,calibration=calibration);self.task=task;self.ticks=ticks;self.size=size
        self.n_neurons,self.n_edges=self.core.n_neurons,self.core.n_edges
        self.classes=(['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'] if task=='image'
                      else ['cat','flower','butterfly','fish','bird','tree','house','star','apple','umbrella'])
        self.guidance_scale=1.
        self.calibrated=calibration is not None;self.stage=0
        if size not in (16,32):raise ValueError('Use image size 16 or 32')
        if self.calibrated:
            self.cue_adapter=nn.Linear(16,16)
            nn.init.eye_(self.cue_adapter.weight);nn.init.zeros_(self.cue_adapter.bias)
            self.rgb_adapter=nn.Conv2d(3,3,1)
            with torch.no_grad():
                self.rgb_adapter.weight.copy_(torch.eye(3).reshape(3,3,1,1));self.rgb_adapter.bias.zero_()
            if task=='image':
                self.readout=nn.Conv2d(9,3,1, bias=False)
                nn.init.normal_(self.readout.weight,std=.3)
            else:
                self.proprio_gain=nn.Parameter(torch.ones(14))
                self.motor_gain=nn.Parameter(torch.full((15,),2.07944154))
                self.motor_bias=nn.Parameter(torch.full((15,),-2.))

    def forward(self, image, labels, clock, seed, state=None, proprio=None, ablate=False, coefficients=None):
        if self.calibrated and self.task=='motor' and self.stage<5:seed=torch.zeros_like(seed)
        cue=cues(labels,clock,seed)
        if self.calibrated:
            image=self.rgb_adapter(torch.nn.functional.interpolate(image,(32,32),mode='bilinear',align_corners=False))
            cue=self.cue_adapter(cue)
        signals=torch.cat([image.flatten(1),cue],1)
        if self.task=="motor":
            if proprio is None or proprio.shape!=(len(image),14): raise ValueError("Motor task needs 14 proprioceptive channels")
            signals=torch.cat([signals,proprio*self.proprio_gain if self.calibrated else proprio],1)
        state=self.core(signals,state,self.ticks,ablate,coefficients)
        out=self.core.read(state)
        # Fixed amplifier, comparable to a calibrated recording interface.
        if self.calibrated:
            out=(torch.nn.functional.interpolate(self.readout(out.reshape(-1,9,32,32)),(self.size,self.size),mode='area')
                 if self.task=='image' else torch.sigmoid(self.motor_gain.clamp(-2,4).exp()*out+self.motor_bias))
        else:out=out.reshape(-1,3,32,32) if self.task=="image" else torch.sigmoid(8*out-2)
        return out,state

    @torch.no_grad()
    def sample(self, labels, seed=42, steps=16, ablate=False, reset_state=False, on_step=None):
        if self.task!="image" or not 2<=steps<=1000: raise ValueError("Image task needs 2–1000 sampling steps")
        device=next(self.parameters()).device;rng=torch.Generator().manual_seed(seed)
        x=torch.randn((len(labels),3,self.size,self.size),generator=rng).to(device)
        cue=torch.randn((len(labels),3),generator=rng).to(device).tanh()
        diffusion=Diffusion(1000,device);state=None;unconditional=None;frames=[x.clamp(-1,1).cpu()]
        if on_step:on_step(0,frames[-1])
        coefficients=self.core.coefficients()
        times=torch.linspace(999,0,steps).round().long().tolist()
        for i,t in enumerate(times):
            ts=torch.full((len(labels),),t/999,device=device)
            clean,state=self(x,labels,ts,cue,None if reset_state else state,ablate=ablate,coefficients=coefficients)
            if self.calibrated and self.guidance_scale!=1:
                null,unconditional=self(x,torch.full_like(labels,10),ts,cue,None if reset_state else unconditional,
                    ablate=ablate,coefficients=coefficients)
                clean=null+self.guidance_scale*(clean-null)
            clean=clean.clamp(-1,1);a=diffusion.alpha_bar[t]
            eps=(x-a.sqrt()*clean)/(1-a).sqrt()
            prev=diffusion.alpha_bar[times[i+1]] if i+1<len(times) else torch.ones((),device=device)
            x=prev.sqrt()*clean+(1-prev).sqrt()*eps
            if not torch.isfinite(x).all():raise FloatingPointError("Nonfinite recurrent sample")
            frames.append(x.cpu())
            if on_step:on_step(i+1,frames[-1])
        return x.cpu(),frames
