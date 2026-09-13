"""An independent sketch classifier for evaluation, never used to generate images."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from flycasso.common import digest, read_json, save_torch, seed_all, write_json
from flycasso.train import batch, image_data


class Classifier(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.SiLU(),nn.Conv2d(32,64,3,2,1),nn.SiLU(),
            nn.Flatten(),nn.Linear(64*8*8,128),nn.SiLU(),nn.Linear(128,classes))
    def forward(self,x): return self.net(x)


def train(root="data/processed/quickdraw-10", out="runs/quality", steps=2000):
    seed_all(2026,2);root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=read_json(root/'manifest.json');model=Classifier(len(manifest['classes']))
    x,y=image_data(root,'train',32);vx,vy=image_data(root,'val',32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001);best=0.
    for step in range(1,steps+1):
        model.train();idx=torch.randint(len(x),(64,)).numpy();images,labels=batch(x,y,idx,'cpu')
        images=torch.roll(images,tuple(torch.randint(-2,3,(2,)).tolist()),dims=(2,3))
        flip=torch.rand(len(images))<.5;images[flip]=images[flip].flip(3)
        loss=F.cross_entropy(model(images),labels);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step%250==0 or step==steps:
            model.eval();correct=0
            with torch.no_grad():
                for start in range(0,len(vx),128):
                    images,labels=batch(vx,vy,np.arange(start,min(start+128,len(vx))),'cpu')
                    correct+=(model(images).argmax(1)==labels).sum().item()
            accuracy=correct/len(vx);print(f'Classifier step {step}: validation accuracy {accuracy:.3f}',flush=True)
            if accuracy>best:
                best=accuracy;save_torch(out/'best.pt',dict(model=model.state_dict(),classes=manifest['classes'],
                    data_sha256=digest(root/'manifest.json'),validation_accuracy=accuracy,step=step))
    write_json(out/'evaluation.json',dict(validation_accuracy=best,examples=len(vx),
        interpretation="Independent recognition proxy, not proof of perceptual quality or novelty."))


@torch.no_grad()
def measure(images, labels, checkpoint="runs/quality/best.pt", classes=None):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if classes is not None and state['classes']!=classes: raise ValueError('Evaluator categories differ from generator')
    with torch.random.fork_rng(devices=[]):
        model=Classifier(len(state['classes']))
    model.load_state_dict(state['model']);model.eval()
    images=images.cpu();labels=labels.cpu();prediction=model(images).softmax(1)
    diversity={}
    for label,name in enumerate(state['classes']):
        rows=images[labels==label].flatten(1)
        diversity[name]=float(torch.pdist(rows).square().mean()/rows.shape[1]) if len(rows)>1 else None
    return dict(category_accuracy=float((prediction.argmax(1)==labels).float().mean()),
        mean_requested_category_probability=float(prediction[torch.arange(len(labels)),labels].mean()),
        within_category_pixel_diversity=diversity,evaluator_validation_accuracy=state['validation_accuracy'],
        evaluator_sha256=digest(checkpoint),examples=len(labels),
        interpretation="A recognition proxy; inspect samples and diversity as well.")


@torch.no_grad()
def evaluate(checkpoint, out, motor=False, count=30, seed=7919, steps=50, device="auto", evaluator="runs/quality/best.pt"):
    """Inspect fresh samples and paired edge ablations from the exact same weights."""
    from flycasso.sample import load_checkpoint, grid, to_images
    from flycasso.diffusion import Diffusion
    from flycasso.strokes import StrokeDenoiser, decode
    from flycasso.train_motor import load_motor
    from flycasso.prepare_images import rasterize
    if count < 1 or not 0 <= seed < 2**63-count:
        raise ValueError("Use a positive count and a nonnegative 63-bit seed")
    if motor:
        model, info=load_motor(checkpoint,device=device)
        if info['phase']!='strokes': raise ValueError("Select a category-trained stroke checkpoint")
        net=StrokeDenoiser(model);diffusion=Diffusion(1000,next(model.parameters()).device)
    else:
        model,diffusion,info=load_checkpoint(checkpoint,device=device);net=model
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    results={}
    for ablated in (False,True):
        mode="edges_disabled" if ablated else "intact"
        images=[];labels=[];drawings=[];invalid=0;started=time.monotonic()
        for start in range(0,count,3):
            y=torch.arange(len(model.classes),device=next(model.parameters()).device).repeat_interleave(min(3,count-start))
            samples,_=diffusion.sample(net,y,seed+start,steps,ablate_edges=ablated)
            if motor:
                pixels=[]
                for sequence in samples.reshape(-1,3,256).numpy():
                    try: drawing=decode(sequence)
                    except ValueError: drawing=[];invalid+=1
                    drawings.append(drawing);pixels.append(rasterize(drawing))
                samples=torch.from_numpy(np.stack(pixels)).float()/127.5-1
            images.append(samples);labels.append(y.cpu())
        elapsed=time.monotonic()-started
        images=torch.cat(images);labels=torch.cat(labels)
        results[mode]=dict(measure(images,labels,evaluator,model.classes),seconds=elapsed,invalid_drawings=invalid,
            per_category_accuracy={name:measure(images[labels==i],labels[labels==i],evaluator,model.classes)["category_accuracy"]
                                   for i,name in enumerate(model.classes)})
        order=labels.argsort(stable=True)
        grid(to_images(images[order]),scale=4).save(out/f"{mode}.png")
        if motor: write_json(out/f"{mode}-strokes.json",dict(labels=labels.tolist(),drawings=drawings))
        print(f"{mode}: {results[mode]['category_accuracy']:.1%} recognition on {len(images)} samples",flush=True)
    report=dict(**info,seed=seed,sampling_steps=steps,guidance_scale=net.guidance_scale,
                synaptic_output=model.synaptic_output,results=results)
    write_json(out/'evaluation.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--checkpoint',help="Evaluate generation instead of training the classifier")
    parser.add_argument('--motor',action='store_true')
    parser.add_argument('--out')
    parser.add_argument('--data',default='data/processed/quickdraw-10')
    parser.add_argument('--evaluator',default='runs/quality/best.pt')
    parser.add_argument('--count',type=int,default=30,help="Generated samples per category")
    parser.add_argument('--seed',type=int,default=7919)
    parser.add_argument('--sampling-steps',type=int,default=50)
    parser.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda'])
    args=parser.parse_args()
    if args.steps<1:parser.error('Steps must be positive')
    if args.checkpoint: evaluate(args.checkpoint,args.out or "reports/generation",args.motor,args.count,args.seed,args.sampling_steps,args.device,args.evaluator)
    else: train(root=args.data,out=args.out or "runs/quality",steps=args.steps)
