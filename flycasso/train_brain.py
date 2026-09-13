"""Train, resume, evaluate and sample the circuit-first image model.

python -m flycasso.train_brain --out runs/brain-image
python -m flycasso.train_brain --resume runs/brain-image/last.pt
python -m flycasso.train_brain --sample runs/brain-image/best.pt
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from flycasso.brain import Brain
from flycasso.common import digest, device_for, environment, load_torch, read_json, save_torch, seed_all, write_json
from flycasso.diffusion import Diffusion
from flycasso.train import image_data, batch
from flycasso.sample import grid, to_images


def load(path, device="auto", raw=False, graph=None):
    torch.set_num_threads(2)
    state,sha=load_torch(path);c=state['config']
    if state.get('format')!='brain-first-v1':raise ValueError('Expected a circuit-first checkpoint')
    c=dict(c)
    if c['task']=='motor' and 'body' in c:
        from flycasso.muscle import body_sources
        if c['body']!=body_sources():raise ValueError('Changed muscle body or initial pose')
    if graph is not None:c['graph']=str(Path(graph).resolve())
    for name in ['graph','ports']:
        local=Path(path).parent/Path(c[name]).name
        if not Path(c[name]).is_absolute() and local.exists():c[name]=str(local.resolve())
        if digest(c[name])!=state['hashes'][name]:raise ValueError(f'Changed {name}')
    model=Brain(c['graph'],c['ports'],c['task'],c['ticks'],c['control']).to(device_for(device))
    if c.get('classes',model.classes)!=model.classes:raise ValueError('Checkpoint categories do not match the task')
    model.load_state_dict(state['model'] if raw and not state.get('inference_only') else state['ema']);model.eval()
    state['checkpoint_sha256']=sha
    return model,state


def image_sequence(model,x,y,rng,steps,backward=False,ablate=False,reset_state=False):
    device=x.device;diffusion=Diffusion(1000,device)
    noise=torch.randn(x.shape,generator=rng).to(device)
    cue=torch.randn((len(x),3),generator=rng).to(device).tanh();state=None;losses=[];chunk=0
    times=torch.linspace(999,0,steps).round().long().tolist()
    for i,t in enumerate(times):
        if i%4==0:coefficients=model.core.coefficients()
        ts=torch.full((len(x),),t,device=device,dtype=torch.long)
        noisy=diffusion.add_noise(x,ts,noise)
        pred,state=model(noisy,y,ts.float()/999,cue,None if reset_state else state,ablate=ablate,coefficients=coefficients)
        mse=(pred-x).square().mean()
        alpha=diffusion.alpha_bar[t];weight=(alpha/(1-alpha)).clamp(max=5)
        chunk=chunk+mse*weight/steps;losses.append(mse.detach().item())
        # Truncated BPTT retains the state but bounds its differentiation history.
        if (i+1)%4==0 or i==steps-1:
            if backward:chunk.backward()
            state=state.detach();chunk=0
    return float(np.mean(losses))


def run(args):
    out=Path(args.out or (Path(args.resume).parent if args.resume else 'runs/brain-image'));out.mkdir(parents=True,exist_ok=True)
    saved=load_torch(args.resume)[0] if args.resume else None
    if saved and (saved.get('format')!='brain-first-v1' or saved.get('inference_only')):
        raise ValueError('Resume requires a circuit-first training checkpoint')
    if not saved and (out/'last.pt').exists():raise ValueError('Use --resume for an existing run')
    c=saved['config'] if saved else dict(task='image',graph=str(Path(args.graph).resolve()),
        ports=str(Path(args.ports).resolve()),dataset=str(Path(args.data).resolve()),ticks=args.ticks,
        steps=args.sequence,batch=args.batch,lr=args.lr,seed=42,control=args.control,
        classes=read_json(Path(args.data)/'manifest.json').get('classes',[
            'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']))
    seed_all(c['seed'],2);device=device_for(args.device)
    hashes={k:digest(c[k]) for k in ['graph','ports']};hashes['data']=digest(Path(c['dataset'])/'manifest.json')
    if saved and hashes!=saved['hashes']:raise ValueError('Training artifacts changed')
    model=Brain(c['graph'],c['ports'],ticks=c['ticks'],control=c['control']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=0)
    rng=torch.Generator().manual_seed(c['seed']+1);step=0;best=float('inf');stale=0;reductions=0
    if saved:
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);rng.set_state(saved['rng'])
        step,best,stale,reductions=(saved[k] for k in ['step','best','stale','reductions'])
    ema={k:v.to(device).clone() for k,v in (saved['ema'] if saved else model.state_dict()).items()}
    del saved
    train_x,train_y=image_data(c['dataset'],'train',32);val_x,val_y=image_data(c['dataset'],'val',32)
    write_json(out/'config.json',c)
    write_json(out/'provenance.json',dict(config=c,hashes=hashes,training_options=vars(args),environment=environment(),
        parameters=sum(p.numel() for p in model.parameters()),adapter_parameters=0,
        code={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        history='Training uses correlated forward-noise trajectories; inference uses generated trajectories.'))
    def save(name):
        save_torch(out/name,dict(format='brain-first-v1',config=c,hashes=hashes,step=step,best=best,
            stale=stale,reductions=reductions,model=model.state_dict(),ema=ema,optimizer=opt.state_dict(),rng=rng.get_state()))
    start=time.monotonic();first=step
    while step<args.max_steps:
        model.train();idx=torch.randint(len(train_x),(c['batch'],),generator=rng).numpy()
        x,y=batch(train_x,train_y,idx,device);opt.zero_grad(set_to_none=True)
        score=image_sequence(model,x,y,rng,c['steps'],True)
        (1e-4*model.core.regularization()).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();step+=1
        with torch.no_grad():
            decay=min(.999,step/(step+9))
            for k,v in model.state_dict().items():ema[k].lerp_(v,1-decay)
        row=dict(step=step,device=str(device),train_mse=score,seconds_per_step=(time.monotonic()-start)/(step-first),learning_rate=opt.param_groups[0]['lr'])
        done=False
        if step%args.validate_every==0 or step==args.max_steps:
            raw={k:v.detach().clone() for k,v in model.state_dict().items()};model.load_state_dict(ema);model.eval()
            vrng=torch.Generator().manual_seed(20260913);scores=[]
            with torch.no_grad():
                for _ in range(args.validation_batches):
                    idx=torch.randint(len(val_x),(c['batch'],),generator=vrng).numpy();x,y=batch(val_x,val_y,idx,device)
                    scores.append(image_sequence(model,x,y,vrng,c['steps']))
                y=torch.arange(10,device=device);samples,_=model.sample(y,steps=c['steps'])
                grid(to_images(samples)).save(out/'preview.png')
            val=float(np.mean(scores));row['validation_mse']=val
            model.load_state_dict(raw)
            if val<best-1e-4:best=val;stale=0;save('best.pt')
            else:stale+=1
            if stale>=4 and reductions<3:
                for g in opt.param_groups:g['lr']*=.5
                reductions+=1;stale=0
            done=step>=args.min_steps and reductions==3 and stale>=8
            save('last.pt')
        row.update(status='validation_plateau' if done else 'step_limit' if step==args.max_steps else 'training',updated_at=time.time())
        with (out/'metrics.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
        write_json(out/'status.json',row)
        if step%10==0 or 'validation_mse' in row:print(json.dumps(row),flush=True)
        if done:break
    save('last.pt')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out');p.add_argument('--resume');p.add_argument('--sample')
    p.add_argument('--graph',default='data/processed/malecns/graph.npz');p.add_argument('--ports',default='data/processed/brain-ports-v2/image.npz')
    p.add_argument('--data',default='data/processed/cifar10');p.add_argument('--device',default='auto')
    p.add_argument('--ticks',type=int,default=4);p.add_argument('--sequence',type=int,default=16);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--lr',type=float,default=.003);p.add_argument('--max-steps',type=int,default=20000);p.add_argument('--min-steps',type=int,default=1000)
    p.add_argument('--validate-every',type=int,default=100);p.add_argument('--validation-batches',type=int,default=16)
    p.add_argument('--control',choices=['real','shuffled'],default='real')
    a=p.parse_args()
    if a.sample:
        m,s=load(a.sample,a.device);pixels,_=m.sample(torch.arange(10,device=next(m.parameters()).device),steps=s['config']['steps'])
        out=Path(a.out or Path(a.sample).parent);out.mkdir(parents=True,exist_ok=True);grid(to_images(pixels)).save(out/'samples.png')
    else:
        if min(a.ticks,a.batch,a.lr,a.max_steps,a.validate_every,a.validation_batches)<=0 or not 2<=a.sequence<=1000 or a.min_steps<0:
            p.error('Positive training settings and 2–1000 sequence steps are required')
        run(a)
