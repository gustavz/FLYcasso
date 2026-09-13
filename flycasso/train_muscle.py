"""Prepare muscle demonstrations and train a category-to-muscle recurrent circuit.

The offline teacher has targets. The policy receives only category, episode clock,
random cue, its own canvas and proprioception. No reference is used at inference.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from flycasso.brain import Brain
from flycasso.common import device_for, digest, load_torch, save_torch, seed_all, write_json, read_json, environment
from flycasso.muscle import MuscleFly, body_sources


# Bump when demonstration targets or muscle physics change, not for trainer edits.
PREPARATION_VERSION=2


def prepare(out='data/processed/muscle-strokes-v2',per_class=32):
    if per_class<2:raise ValueError('Use at least two demonstrations per category')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    source=Path('data/processed/quickdraw-strokes-10');manifest=read_json(source/'manifest.json')
    if (out/'manifest.json').exists():
        old=read_json(out/'manifest.json')
        if old['per_class']!=per_class or old.get('version',1)!=PREPARATION_VERSION or old['source_sha256']!=digest(source/'manifest.json'):raise ValueError('Use a new data directory for changed preparation')
        for name,sha in old['files'].items():
            if digest(out/name)!=sha:raise ValueError('Changed demonstrations')
        return
    fly=MuscleFly();files={}
    for split in ['train','val','test']:
        path=source/f'{split}.npz'
        if digest(path)!=manifest['files'][path.name]:raise ValueError('Corrupt source strokes')
        with np.load(path) as f:x,y=f['strokes'].copy(),f['labels'].copy()
        images=[];proprio=[];actions=[];labels=[];targets=[];seeds=[]
        take=per_class if split=='train' else max(2,per_class//4)
        for label in range(10):
            available=np.flatnonzero(y==label)
            if len(available)<take:raise ValueError('Not enough source drawings')
            for idx in available[:take]:
                fly.reset();iv=[];pv=[];av=[];tv=[]
                for point in x[idx].T:
                    goal=fly.center.copy();goal[:2]+=point[:2]*np.array([1,-1])*.09
                    goal[2]=fly.z+.012 if point[2]>0 else fly.z+.08
                    im,pr=fly.observe();a=fly.expert(goal)
                    iv.append(((im+1)*127.5).astype(np.uint8));pv.append(pr);av.append(a);tv.append(goal)
                    fly.step(a)
                images.append(iv);proprio.append(pv);actions.append(av);labels.append(label);targets.append(tv)
                seeds.append(np.random.default_rng(int(idx)+{'train':0,'val':100000,'test':200000}[split]).normal(size=3).astype(np.float32))
            print(f'{split}: {manifest["classes"][label]} ({len(labels)} drawings)',flush=True)
        path=out/f'{split}.npz'
        np.savez_compressed(path,images=np.array(images,np.uint8),proprio=np.array(proprio,np.float32),
            actions=np.array(actions,np.float32),labels=np.array(labels,np.int64),targets=np.array(targets,np.float32),seeds=np.array(seeds,np.float32))
        files[path.name]=digest(path)
    write_json(out/'manifest.json',dict(version=PREPARATION_VERSION,classes=manifest['classes'],files=files,per_class=per_class,
        source_sha256=digest(source/'manifest.json'),code_sha256=digest(__file__),
        teacher='Offline IK + grouped muscle inverse dynamics; no teacher at inference',
        observations='own canvas, joint position and velocity, class, clock, seed',
        limitation='Proprioceptive feature tuning and muscle-head pooling are engineered group mappings'))


@torch.no_grad()
def rollout(model,label,seed,steps=256,ablate=False,reset_state=False):
    fly=MuscleFly();device=next(model.parameters()).device;state=None;marks=0;tips=[];activations=[]
    coefficients=model.core.coefficients()
    label=torch.tensor([label],device=device);cue=torch.tensor(seed,dtype=torch.float32,device=device).reshape(1,3).tanh()
    for i in range(steps):
        im,pr=fly.observe()
        action,state=model(torch.from_numpy(im)[None].to(device),label,torch.tensor([i/(steps-1)],device=device),cue,
            None if reset_state else state,torch.from_numpy(pr)[None].to(device),ablate,coefficients)
        a=action[0].cpu().numpy();marks+=fly.step(a);activations.append(a);tips.append(fly.data.site_xpos[fly.site].copy())
    return fly,np.array(tips),dict(contact_samples=marks,steps=steps,activations=np.array(activations))


def run(args):
    out=Path(args.out or (Path(args.resume).parent if args.resume else 'runs/brain-motor'));out.mkdir(parents=True,exist_ok=True)
    saved=load_torch(args.resume)[0] if args.resume else None
    if saved and (saved.get('format')!='brain-first-v1' or saved.get('inference_only')):
        raise ValueError('Resume requires a circuit-first training checkpoint')
    if not saved and (out/'last.pt').exists():raise ValueError('Use --resume')
    c=saved['config'] if saved else dict(task='motor',graph=str(Path(args.graph).resolve()),ports=str(Path(args.ports).resolve()),
        dataset=str(Path(args.data).resolve()),ticks=args.ticks,batch=args.batch,lr=args.lr,control=args.control,seed=43,closed_loop=True,
        classes=read_json(Path(args.data)/'manifest.json')['classes'])
    body=body_sources()
    if c.get('body',body)!=body:raise ValueError('Changed muscle body or initial pose')
    c['body']=body
    seed_all(c['seed'],2);device=device_for(args.device)
    hashes={k:digest(c[k]) for k in ['graph','ports']};hashes['data']=digest(Path(c['dataset'])/'manifest.json')
    if saved and hashes!=saved['hashes']:raise ValueError('Changed training artifacts')
    arrays={}
    for split in ['train','val']:
        p=Path(c['dataset'])/f'{split}.npz'
        if digest(p)!=read_json(Path(c['dataset'])/'manifest.json')['files'][p.name]:raise ValueError('Corrupt muscle demonstrations')
        with np.load(p) as f:arrays[split]={k:f[k].copy() for k in f.files}
    model=Brain(c['graph'],c['ports'],'motor',c['ticks'],c['control']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=0);rng=torch.Generator().manual_seed(44)
    step=0;best=float('inf');stale=0;reductions=0
    if saved:
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);rng.set_state(saved['rng'])
        step,best,stale,reductions=(saved[k] for k in ['step','best','stale','reductions'])
    ema={k:v.to(device).clone() for k,v in (saved['ema'] if saved else model.state_dict()).items()};del saved
    write_json(out/'config.json',c)
    write_json(out/'provenance.json',dict(config=c,hashes=hashes,training_options=vars(args),environment=environment(),adapter_parameters=0,
        code={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        training='On-policy muscle demonstrations with a fading teacher mixture; truncated BPTT 16 frames; no target inputs to the circuit'))
    def save(name):
        save_torch(out/name,dict(format='brain-first-v1',config=c,hashes=hashes,step=step,best=best,stale=stale,
            reductions=reductions,model=model.state_dict(),ema=ema,optimizer=opt.state_dict(),rng=rng.get_state()))
    def episode(data,indices,training,on_policy=False):
        nonlocal step
        flies=[MuscleFly() for _ in indices] if on_policy or (training and c.get('closed_loop')) else None
        labels=torch.tensor(data['labels'][indices],device=device)
        cue=torch.tensor(data['seeds'][indices],device=device).tanh();state=None;scores=[];chunk=0
        for t in range(data['images'].shape[1]):
            if t%16==0:coefficients=model.core.coefficients()
            if flies:
                observations=[fly.observe() for fly in flies]
                image=torch.tensor(np.stack([v[0] for v in observations]),device=device)
                proprio=torch.tensor(np.stack([v[1] for v in observations]),device=device)
            else:
                image=torch.tensor(data['images'][indices,t],device=device).float()/127.5-1
                proprio=torch.tensor(data['proprio'][indices,t],device=device)
            pred,state=model(image,labels,torch.full((len(indices),),t/255,device=device),cue,state,proprio,coefficients=coefficients)
            target=torch.tensor(np.stack([fly.expert(data['targets'][idx,t]) for fly,idx in zip(flies,indices)]) if flies else data['actions'][indices,t],device=device)
            if flies:
                # Targets supervise actions on the policy's own states, never its inputs.
                teacher=max(0.,1-step/512) if training else 0.
                controls=(teacher*target+(1-teacher)*pred).detach().cpu().numpy()
                for fly,control in zip(flies,controls):fly.step(control)
            loss=(pred-target).square().mean();scores.append(loss.detach().item());chunk=chunk+loss/16
            if (t+1)%16==0:
                if training:
                    (chunk+1e-4*model.core.regularization()).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
                    step+=1
                    with torch.no_grad():
                        for k,v in model.state_dict().items():ema[k].lerp_(v,1-min(.999,step/(step+9)))
                state=state.detach();chunk=0
        return float(np.mean(scores))
    start=time.monotonic();first=step;lastval=step;done=False
    while step<args.max_steps and not done:
        model.train();idx=torch.randint(len(arrays['train']['labels']),(c['batch'],),generator=rng).numpy()
        opt.zero_grad(set_to_none=True);loss=episode(arrays['train'],idx,True)
        row=dict(step=step,device=str(device),teacher_fraction=max(0.,1-step/512) if c.get('closed_loop') else 1.,train_mse=loss,seconds_per_step=(time.monotonic()-start)/(step-first))
        if step-lastval>=args.validate_every or step>=args.max_steps:
            lastval=step;raw={k:v.detach().clone() for k,v in model.state_dict().items()};model.load_state_dict(ema);model.eval()
            with torch.no_grad():
                indices=np.array([np.flatnonzero(arrays['val']['labels']==label)[0] for label in range(10)])
                score=episode(arrays['val'],indices,False)
                policy_score=episode(arrays['val'],indices,False,on_policy=True)
            fly,tips,physical=rollout(model,6,[.1,.2,.3]);fly.canvas.resize((512,512)).save(out/'preview.png')
            row.update(validation_mse=score,policy_control_mse=policy_score,contact_samples=physical['contact_samples'])
            model.load_state_dict(raw)
            if score<best-1e-4:best=score;stale=0;save('best.pt')
            else:stale+=1
            if stale>=4 and reductions<3:
                for g in opt.param_groups:g['lr']*=.5
                reductions+=1;stale=0
            done=step>=args.min_steps and reductions==3 and stale>=8
        row.update(status='validation_plateau' if done else 'step_limit' if step>=args.max_steps else 'training',updated_at=time.time())
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        save('last.pt')
        write_json(out/'status.json',row);print(json.dumps(row),flush=True)
    save('last.pt')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--per-class',type=int,default=32)
    p.add_argument('--data',default='data/processed/muscle-strokes-v2');p.add_argument('--out');p.add_argument('--resume')
    p.add_argument('--graph',default='data/processed/malecns/graph.npz');p.add_argument('--ports',default='data/processed/brain-ports-v2/motor.npz')
    p.add_argument('--control',choices=['real','shuffled'],default='real');p.add_argument('--device',default='auto');p.add_argument('--ticks',type=int,default=4);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--lr',type=float,default=.003);p.add_argument('--max-steps',type=int,default=20000);p.add_argument('--min-steps',type=int,default=1000)
    p.add_argument('--validate-every',type=int,default=128);a=p.parse_args()
    if a.prepare:prepare(a.data,a.per_class)
    else:
        if min(a.ticks,a.batch,a.lr,a.max_steps,a.validate_every)<=0 or a.min_steps<0:
            p.error('Positive training settings are required')
        run(a)
