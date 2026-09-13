"""Paired circuit interventions and free-running outputs from a saved EMA checkpoint."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from flycasso.common import digest, write_json
from flycasso.sample import grid, to_images
from flycasso.train import image_data, batch
from flycasso.train_brain import load, image_sequence


@torch.no_grad()
def evaluate(checkpoint, out, device='auto', count=1, batches=8, split='val', evaluator=None):
    if count<1 or batches<1 or split not in ('val','test'):raise ValueError('Positive counts and val/test split required')
    model,state=load(checkpoint,device);c=state['config'];device=next(model.parameters()).device
    out=Path(out)
    if (out/'evaluation.json').exists():raise ValueError('Choose a new directory for another completed evaluation')
    out.mkdir(parents=True,exist_ok=True);results={};baseline=None
    if c['task']=='image':
        if digest(Path(c['dataset'])/'manifest.json')!=state['hashes']['data']:raise ValueError('Changed evaluation data')
        x,y=image_data(c['dataset'],split,32)
    for name,ablate,reset in [('intact',False,False),('edges_disabled',True,False),('state_reset',False,True)]:
        started=time.monotonic();labels=torch.arange(10,device=device).repeat_interleave(count)
        if c['task']=='image':
            scores=[];rng=torch.Generator().manual_seed(20260914)
            for _ in range(batches):
                idx=torch.randint(len(x),(4,),generator=rng).numpy();pixels,categories=batch(x,y,idx,device)
                scores.append(image_sequence(model,pixels,categories,rng,c['steps'],ablate=ablate,reset_state=reset))
            pixels,_=model.sample(labels,seed=7919,steps=c['steps'],ablate=ablate,reset_state=reset)
            result=dict(held_out_mse=float(np.mean(scores)),validation_examples=4*batches)
        else:
            from flycasso.train_muscle import rollout
            samples=[];contacts=[];paths=[]
            for i,label in enumerate(labels.tolist()):
                seed=np.random.default_rng(7919+i%count).normal(size=3)
                fly,tips,physical=rollout(model,label,seed,ablate=ablate,reset_state=reset)
                samples.append(torch.from_numpy(fly.observe()[0]));contacts.append(physical['contact_samples']);paths.append(tips)
            pixels=torch.stack(samples)
            result=dict(contact_fraction=float(np.mean(contacts)/512),contact_samples=contacts)
            np.savez_compressed(out/f'{name}-trajectories.npz',tips=np.array(paths),labels=labels.cpu().numpy())
            if evaluator is not None:
                from flycasso.quality import measure
                result['recognition_proxy']=measure(pixels,labels,evaluator,model.classes)
        if baseline is None:baseline=pixels
        result.update(seconds=time.monotonic()-started,output_mse_from_intact=float((pixels-baseline).square().mean()),
            pixel_standard_deviation=float(pixels.std()),examples=len(pixels),
            category_separation_mse=float(pixels.reshape(10,count,3,32,32).mean(1).var(0,unbiased=False).mean()))
        grid(to_images(pixels),scale=4).save(out/f'{name}.png');results[name]=result
        print(name,result,flush=True)
    # Same seeds, different category cues: conditional sensitivity, not correctness.
    if c['task']=='image':
        wrong,_=model.sample((labels+1)%10,seed=7919,steps=c['steps'])
        results['wrong_labels']=dict(output_mse_from_intact=float((wrong-baseline).square().mean()))
    report=dict(checkpoint_sha256=state['checkpoint_sha256'],training_step=state['step'],config=c,results=results,
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        fraction_edge_gains_changed=float((state['ema']['core.edge_gain'].abs()>1e-5).float().mean()),
        interpretation='Paired interventions test dependence on connections, state and category. They do not establish biological superiority. Inspect samples; MSE and pen contact alone are not image quality.')
    write_json(out/'evaluation.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='auto');p.add_argument('--count',type=int,default=1);p.add_argument('--batches',type=int,default=8)
    p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--evaluator',help='Optional independent sketch-classifier checkpoint');a=p.parse_args()
    evaluate(a.checkpoint,a.out,a.device,a.count,a.batches,a.split,a.evaluator)
