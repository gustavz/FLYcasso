"""Cosine forward-noise schedule and DDIM sampling with clean-image (x0) prediction."""

import math

import torch


class Diffusion:
    def __init__(self, steps, device="cpu"):
        if steps < 2:
            raise ValueError("diffusion_steps must be at least two")
        t = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
        curve = torch.cos(((t / steps + 0.008) / 1.008) * math.pi / 2).square()
        curve = curve / curve[0]
        beta = (1 - curve[1:] / curve[:-1]).clamp(1e-8, 0.999)
        self.alpha_bar = (1 - beta).cumprod(0).float().to(device)
        self.steps = steps

    def add_noise(self, image, timestep, noise):
        alpha = self.alpha_bar[timestep].reshape(-1, 1, 1, 1)
        return alpha.sqrt() * image + (1 - alpha).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, labels, seed, sample_steps=50, ablate_edges=False, on_step=None):
        if not 2 <= sample_steps <= self.steps:
            raise ValueError(f"sample_steps must be between 2 and {self.steps}")
        device = next(model.parameters()).device
        rng = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn((len(labels), 3, model.size, model.size), generator=rng).to(device)
        times = torch.linspace(self.steps - 1, 0, sample_steps).round().long().tolist()
        frames = [x.clamp(-1, 1).cpu()]
        if on_step:
            on_step(0, frames[-1])
        for i, t in enumerate(times):
            ts = torch.full((len(labels),), t, device=device, dtype=torch.long)
            scale=getattr(model,"guidance_scale",1.)
            if getattr(model,"null_label",None) is not None and scale!=1:
                condition=torch.cat([labels,torch.full_like(labels,model.null_label)])
                conditional,unconditional=model(torch.cat([x,x]),torch.cat([ts,ts]),condition,ablate_edges=ablate_edges).chunk(2)
                x0=(unconditional+scale*(conditional-unconditional)).clamp(-1,1)
            else:
                x0 = model(x, ts, labels, ablate_edges=ablate_edges).clamp(-1, 1)
            alpha = self.alpha_bar[t]
            # Recompute epsilon after clipping for a consistent deterministic update.
            eps = (x - alpha.sqrt() * x0) / (1 - alpha).sqrt()
            prev = self.alpha_bar[times[i + 1]] if i + 1 < len(times) else torch.ones((), device=device)
            x = prev.sqrt() * x0 + (1 - prev).sqrt() * eps
            if not torch.isfinite(x).all():
                raise FloatingPointError("Nonfinite diffusion sample")
            frames.append(x.cpu())
            if on_step:
                on_step(i + 1, frames[-1])
        return x.cpu(), frames
