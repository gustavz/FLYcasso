"""Calibrated full-circuit curricula. One worker, one task, resumable GPU slices.

python -m flycasso.train_calibrated motor --updates 16
python -m flycasso.train_calibrated image --updates 8
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import binary_dilation
from flycasso.brain import Brain
from flycasso.common import digest, device_for, environment, load_torch, save_torch, seed_all, write_json, read_json
from flycasso.diffusion import Diffusion
from flycasso.train import image_data, batch
from flycasso.sample import grid, to_images

MOTOR_STAGES=['hold','touch and lift','line','circle','one sketch','cat variations','three categories','ten categories']
IMAGE_STAGES=['three-image overfit','three categories at 16px','ten categories at 16px','ten categories at 32px']


class SketchEncoder(nn.Module):
    """Training-only posterior; inference samples its three-dimensional prior."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(3,16,5,2,2),nn.SiLU(),nn.Conv1d(16,32,5,2,2),
            nn.SiLU(),nn.AdaptiveAvgPool1d(8),nn.Flatten(),nn.Linear(256,6))
    def forward(self, strokes, epsilon):
        mean,logvar=self.net(strokes).chunk(2,1);logvar=logvar.clamp(-6,2)
        return (mean+(logvar*.5).exp()*epsilon).tanh(), .5*(mean.square()+logvar.exp()-1-logvar).mean()


def goals(fly, stage, stroke=None, steps=256):
    t=np.linspace(0,1,steps);out=np.tile(fly.center,(steps,1));out[:,2]=fly.z+.012
    if stage==1:out[:,2]+=np.where((t>.25)&(t<.75),.015,0)
    if stage==2:out[:,0]+=(2*t-1)*.04
    if stage==3:
        out[:,0]+=.04*np.cos(t*2*np.pi);out[:,1]+=.04*np.sin(t*2*np.pi)
    if stage>=4:
        out[:,:2]+=stroke[:2].T*np.array([1,-1])*[.045,.06,.075,.09][min(stage-4,3)]
        out[:,2]=fly.z+np.where(stroke[2]>0,.012,.027)
    return out.astype(np.float32)


def physical_score(fly, targets, tips, contact):
    target=Image.new('1',(32,32));draw=ImageDraw.Draw(target);last=None
    for point in targets:
        if point[2]<fly.z+.016:
            p=fly.pixel(point);draw.line([last or p,p],fill=1,width=1);last=p
        else:last=None
    a=np.asarray(fly.canvas).min(2)<128;b=np.asarray(target,dtype=bool)
    precision=(a & binary_dilation(b)).sum()/max(1,a.sum())
    recall=(b & binary_dilation(a)).sum()/max(1,b.sum())
    return dict(mean_tip_error_mm=float(np.linalg.norm(tips-targets,axis=1).mean()),
        contact_accuracy=float(np.mean((np.asarray(contact)>0)==(targets[:,2]<fly.z+.016))),
        stroke_f1=float(2*precision*recall/max(1e-8,precision+recall)))


def image_sequence(model, x, labels, rng, steps, exposure=0., backward=False):
    """Uniform x0 loss includes the high-noise steps that determine category."""
    diffusion=Diffusion(1000,x.device);times=torch.linspace(999,0,steps).round().long().tolist()
    noise=torch.randn(x.shape,generator=rng).to(x.device);cue=torch.randn(len(x),3,generator=rng).to(x.device).tanh()
    y=labels.clone();y[torch.rand(len(y),generator=rng).to(y.device)<.1]=10
    state=None;generated=None;scores=[];loss=0
    for i,t in enumerate(times):
        if i%2==0:coefficients=model.core.coefficients()
        alpha=diffusion.alpha_bar[t];noisy=alpha.sqrt()*x+(1-alpha).sqrt()*noise
        if generated is not None:noisy=noisy.lerp(generated,exposure)
        pred,state=model(noisy,y,torch.full((len(x),),t/999,device=x.device),cue,state,coefficients=coefficients)
        mse=(pred-x).square().mean();loss=loss+mse/steps;scores.append(mse.detach().item())
        if i+1<len(times):
            prev=diffusion.alpha_bar[times[i+1]];clean=pred.detach().clamp(-1,1)
            generated=prev.sqrt()*clean+(1-prev).sqrt()*(noisy-alpha.sqrt()*clean)/(1-alpha).sqrt()
        if (i+1)%2==0 or i==steps-1:
            if backward:loss.backward()
            state=state.detach();loss=0
    return float(np.mean(scores))


class Training:
    def __init__(self,args):
        self.args=args;self.out=Path(args.out or f'runs/calibrated-{args.task}');self.out.mkdir(parents=True,exist_ok=True)
        path=self.out/'last.pt';saved=load_torch(path)[0] if path.exists() else None
        c=saved['config'] if saved else dict(task=args.task,recipe='calibrated-v1',
            graph=str(Path(args.graph).resolve()),ports=str(Path(args.ports or f'data/processed/brain-ports-v3/{args.task}.npz').resolve()),
            calibration=str(Path(args.calibration).resolve()),control='real',ticks=8,size=16 if args.task=='image' else 32,
            dataset=str(Path(args.data or ('data/processed/cifar10' if args.task=='image' else 'data/processed/quickdraw-strokes-10')).resolve()),
            batch=3 if args.task=='image' else 1,steps=8,lr=.001,seed=42 if args.task=='image' else 43)
        if c.get('recipe')!='calibrated-v1' or c['task']!=args.task:raise ValueError('Choose a calibrated run of the same task')
        if args.task=='motor':
            from flycasso.muscle import body_sources
            if c.get('body',body_sources())!=body_sources():raise ValueError('Body assets changed')
            c['body']=body_sources();c['body_ports']=str(Path(c['ports']).parent/'body.npz') if 'body_ports' not in c else c['body_ports']
        self.code={name:digest(Path(__file__).parent/name) for name in ['brain.py','circuit.py','train_calibrated.py','muscle.py']}
        self.c=c;seed_all(c['seed'],2);self.device=device_for(args.device)
        self.hashes={k:digest(c[k]) for k in ['graph','ports','calibration']+(['body_ports'] if args.task=='motor' else [])}
        port_manifest=Path(c['ports']).parent/'manifest.json'
        if port_manifest.exists():
            anatomy=read_json(port_manifest)
            if not anatomy.get('calibrated') or anatomy['graph_sha256']!=self.hashes['graph']:
                raise ValueError('Prepare calibrated ports for this graph')
            for name in ['ports','calibration']:
                if anatomy['files'][Path(c[name]).name]!=self.hashes[name]:raise ValueError('Prepared port artifact changed')
        self.hashes['data']=digest(Path(c['dataset'])/'manifest.json')
        if saved and saved['hashes']!=self.hashes:raise ValueError('Training artifacts changed')
        self.model=Brain(c['graph'],c['ports'],args.task,c['ticks'],c['control'],c['calibration'],c['size']).to(self.device)
        self.c['classes']=self.model.classes
        self.model.stage=c.get('stage',0)
        if args.task=='motor' and saved is None:
            from flycasso.muscle import MuscleFly
            fly=MuscleFly();actions=[]
            for target in goals(fly,0):
                action=fly.expert(target);fly.step(action);actions.append(action)
            neutral=np.mean(actions[128:],axis=0);c['neutral_action']=neutral.tolist()
            with torch.no_grad():
                self.model.motor_bias.copy_(torch.logit(torch.tensor(neutral,device=self.device).clamp(.001,.999)))
                self.model.motor_gain.zero_()
        self.aux=nn.ModuleDict()
        if args.task=='motor':
            self.aux['encoder']=SketchEncoder()
            self.aux['critic']=nn.Sequential(nn.Linear(18,64),nn.Tanh(),nn.Linear(64,1))
        self.aux.to(self.device)
        self.opt=torch.optim.AdamW(list(self.model.parameters())+list(self.aux.parameters()),lr=c['lr'],weight_decay=0)
        self.rng=torch.Generator().manual_seed(c['seed']+1)
        self.step=0;self.stage=0;self.stage_step=0;self.best=1e9;self.stale=0;self.passes=0;self.teacher=.95
        if saved:
            self.model.load_state_dict(saved['model']);self.aux.load_state_dict(saved['aux']);self.opt.load_state_dict(saved['optimizer'])
            self.rng.set_state(saved['rng'])
            for key in ['step','stage','stage_step','best','stale','passes','teacher']:setattr(self,key,saved[key])
        self.ema={k:v.to(self.device).clone() for k,v in (saved['ema'] if saved else self.model.state_dict()).items()}
        del saved
        self.start=time.monotonic();self.first=self.step;self.data={};self.gradients={}
        if args.task=='image':
            for split in ['train','val']:self.data[split]=image_data(c['dataset'],split,32)
        else:
            manifest=read_json(Path(c['dataset'])/'manifest.json')
            for split in ['train','val']:
                p=Path(c['dataset'])/f'{split}.npz'
                if digest(p)!=manifest['files'][p.name]:raise ValueError('Corrupt stroke data')
                with np.load(p) as f:self.data[split]=(f['strokes'].copy(),f['labels'].copy())
        self.write_config()
        if self.first==0:
            write_json(self.out/'provenance.json',dict(config=c,hashes=self.hashes,environment=environment(),
                parameters=sum(p.numel() for p in self.model.parameters()),
                adapter_parameters=sum(p.numel() for n,p in self.model.named_parameters() if not n.startswith('core.')),
                training_only_parameters=sum(p.numel() for p in self.aux.parameters()),
                code={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
                stopping='Three passing free-running checks per stage. Stop for review after 48 checks without improvement.',
                limitations='Engineered rate dynamics and ports; curriculum gates are engineering thresholds, not validated quality certification.'))

    def write_config(self):
        self.c.update(stage=self.stage,stage_name=(IMAGE_STAGES if self.c['task']=='image' else MOTOR_STAGES)[self.stage])
        write_json(self.out/'config.json',self.c)

    def save(self,name):
        save_torch(self.out/name,dict(format='brain-first-v1',config=self.c,hashes=self.hashes,code=self.code,
            **{k:getattr(self,k) for k in ['step','stage','stage_step','best','stale','passes','teacher']},
            model=self.model.state_dict(),ema=self.ema,aux=self.aux.state_dict(),optimizer=self.opt.state_dict(),rng=self.rng.get_state()))

    def update(self):
        if self.step%8==0 or not self.gradients:
            self.gradients={n:dict(rms=float(p.grad.square().mean().sqrt()),nonzero=float((p.grad!=0).float().mean()))
                for n,p in self.model.named_parameters() if p.grad is not None and n in ('core.edge_gain','cue_adapter.weight','readout.weight','motor_gain')}
        (1e-5*self.model.core.regularization()).backward()
        torch.nn.utils.clip_grad_norm_(list(self.model.parameters())+list(self.aux.parameters()),1.,error_if_nonfinite=True)
        self.opt.step();self.opt.zero_grad(set_to_none=True);self.step+=1;self.stage_step+=1
        with torch.no_grad():
            for k,v in self.model.state_dict().items():self.ema[k].lerp_(v,1-min(.99,self.step/(self.step+9)))

    def subset(self,split):
        x,y=self.data[split]
        classes=[0,1,6] if self.c['task']=='image' else [0,6,7]
        if self.c['task']=='image':
            idx=np.concatenate([np.flatnonzero(y==k)[:1 if self.stage==0 else 512] for k in classes]) if self.stage<2 else np.arange(len(y))
        else:
            classes=[0] if self.stage<6 else classes if self.stage==6 else list(range(10))
            idx=np.concatenate([np.flatnonzero(y==k)[:1 if self.stage<5 else 1000] for k in classes])
        return x,y,idx

    def image_train(self):
        x,y,indices=self.subset('train');idx=indices[torch.randint(len(indices),(self.c['batch'],),generator=self.rng).numpy()]
        images,labels=batch(x,y,idx,self.device);images=F.interpolate(images,(self.model.size,)*2,mode='area')
        self.model.train();self.opt.zero_grad(set_to_none=True)
        score=image_sequence(self.model,images,labels,self.rng,self.c['steps'],min(.5,self.stage_step/1000),True)
        self.update();return score

    @torch.no_grad()
    def image_validate(self):
        x,y,idx=self.subset('train' if self.stage==0 else 'val');idx=idx[:1] if len(idx)==1 else idx[np.linspace(0,len(idx)-1,min(12,len(idx))).astype(int)]
        images,labels=batch(x,y,idx,self.device);images=F.interpolate(images,(self.model.size,)*2,mode='area')
        loss=image_sequence(self.model,images,labels,torch.Generator().manual_seed(2026),self.c['steps'])
        classes=[0,1,6] if self.stage<2 else list(range(10));samples=[];disabled=[];targets=[]
        for label in classes:
            # Same noise and latent seed across classes, not different batched noise.
            label_t=torch.tensor([label],device=self.device)
            samples.append(self.model.sample(label_t,42,self.c['steps'])[0])
            disabled.append(self.model.sample(label_t,42,self.c['steps'],ablate=True)[0])
            tx,ty=self.data['train'];target,_=batch(tx,ty,[np.flatnonzero(ty==label)[0]],'cpu')
            targets.append(F.interpolate(target,(self.model.size,)*2,mode='area'))
        samples=torch.cat(samples);disabled=torch.cat(disabled);targets=torch.cat(targets)
        grid(to_images(samples)).save(self.out/'preview.png');grid(to_images(targets)).save(self.out/'targets.png')
        sensitivity=float(torch.pdist(samples.flatten(1)).square().mean()/samples[0].numel())
        mse=float((samples-targets).square().mean());ablated=float((disabled-targets).square().mean())
        row=dict(validation_mse=loss,generated_target_mse=mse if self.stage==0 else None,
            same_seed_class_difference=sensitivity,edge_ablation_difference=float((samples-disabled).abs().mean()))
        if self.stage==0:
            row['ablated_target_mse']=ablated;score=mse;passed=mse<.035 and sensitivity>.02 and ablated>mse*1.1
        else:
            # Independent held-out CIFAR classifier, never in the generator or its gradients.
            if not Path(self.args.evaluator).exists():raise ValueError('Train the CIFAR quality classifier before advancing the image curriculum')
            from flycasso.quality import measure
            evaluation=[];evaluation_labels=[]
            for seed in [101,202,303]:
                ys=torch.tensor(classes,device=self.device);evaluation.append(self.model.sample(ys,seed,self.c['steps'])[0]);evaluation_labels.append(ys.cpu())
            result=measure(F.interpolate(torch.cat(evaluation),(32,32),mode='bilinear',align_corners=False),torch.cat(evaluation_labels),self.args.evaluator,self.model.classes)
            row['generation']=result;score=1-result['category_accuracy'];passed=score<.4 and sensitivity>.02 and row['edge_ablation_difference']>.01
        return row,score,passed

    def motor_episode(self, training=True, ablate=False, posterior=True, split='train', variant=0):
        from flycasso.muscle import MuscleFly
        fly=MuscleFly();x,y,idx=self.subset(split)
        which=idx[int(torch.randint(len(idx),(1,),generator=self.rng))] if training else idx[(variant*len(idx)//max(1,self.validation_count()))%len(idx)]
        stroke=x[which];label=torch.tensor([int(y[which])],device=self.device)
        target=goals(fly,self.stage,stroke);epsilon=torch.randn(1,3,generator=self.rng).to(self.device) if training else torch.tensor(np.random.default_rng(101+variant).normal(size=(1,3)),dtype=torch.float32,device=self.device)
        state=None;losses=[];tips=[];contact=[];logps=[];values=[];rewards=[];imitation=0;kl=0
        def privileged(t):
            _,pr=fly.observe();goal=(target[min(t,255)]-fly.center)/.1
            return torch.tensor(np.r_[pr,goal,min(t,255)/255],dtype=torch.float32,device=self.device)[None]
        for t in range(256):
            if t%8==0:
                coefficients=self.model.core.coefficients()
                if self.stage>=5 and posterior:
                    cue,kl=self.aux['encoder'](torch.tensor(stroke,device=self.device)[None],epsilon)
                else:cue=epsilon.tanh() if self.stage>=5 else torch.zeros_like(epsilon);kl=0
            im,pr=fly.observe()
            pred,state=self.model(torch.tensor(im,device=self.device)[None],label,torch.full((1,),t/255,device=self.device),cue,state,
                torch.tensor(pr,device=self.device)[None],ablate=ablate,coefficients=coefficients)
            expert=torch.tensor(fly.expert(target[t]),device=self.device)[None]
            mse=(pred-expert).square().mean();losses.append(mse.detach().item());imitation=imitation+mse/8
            if training:
                # Score-function gradient through actual MuJoCo outcomes; no fictitious differentiable physics.
                distribution=torch.distributions.Normal(torch.logit(pred.clamp(.0001,.9999)),.15)
                action_logits=distribution.loc.detach()+.15*torch.randn(pred.shape,generator=self.rng).to(self.device)
                logps.append(distribution.log_prob(action_logits).sum(1).mean())
                values.append(self.aux['critic'](privileged(t)).squeeze())
                action=self.teacher*expert+(1-self.teacher)*action_logits.sigmoid()
            else:action=pred
            marks=fly.step(action.detach()[0].cpu().numpy());contact.append(marks);tip=fly.data.site_xpos[fly.site].copy();tips.append(tip)
            error=np.linalg.norm(tip-target[t]);reward=float(np.exp(-(error/(.015 if self.stage<4 else .03))**2)+.2*((marks>0)==(target[t,2]<fly.z+.016)))
            if t==255 and self.stage>=4:reward+=.5*physical_score(fly,target,np.asarray(tips),contact)['stroke_f1']
            rewards.append(reward)
            if (t+1)%8==0:
                if training:
                    with torch.no_grad():future=torch.zeros((),device=self.device) if t==255 else self.aux['critic'](privileged(t+1)).squeeze()
                    returns=[]
                    for reward in rewards[::-1]:future=reward+.97*future;returns.append(future)
                    returns=torch.stack(returns[::-1]);value=torch.stack(values);advantage=(returns-value.detach()).clamp(-5,5)
                    actor=-(torch.stack(logps)*advantage).mean()
                    beta=.001*min(1,self.stage_step/1000)
                    (imitation+.02*(1-self.teacher)*actor+.05*F.mse_loss(value,returns)+beta*kl).backward();self.update()
                state=state.detach();imitation=0;logps=[];values=[];rewards=[]
        result=physical_score(fly,target,np.asarray(tips),contact)
        return float(np.mean(losses)),result,fly

    def validation_count(self):
        return 1 if self.stage<5 else 3 if self.stage==5 else 9 if self.stage==6 else 20

    @torch.no_grad()
    def motor_validate(self):
        scores=[];physical=[]
        for v in range(self.validation_count()):
            score,metrics,fly=self.motor_episode(False,split='val' if self.stage>=5 else 'train',variant=v)
            scores.append(score);physical.append(metrics)
        fly.canvas.resize((512,512)).save(self.out/'preview.png')
        error=float(np.mean([p['mean_tip_error_mm'] for p in physical]));f1=float(np.mean([p['stroke_f1'] for p in physical]));contact=float(np.mean([p['contact_accuracy'] for p in physical]))
        threshold=[.004,.013,.009,.012,.02,.03,.035,.04][self.stage]
        passed=error<threshold and contact>.9 and f1>(.5 if self.stage<4 else .8)
        _,ablated,_=self.motor_episode(False,ablate=True,posterior=self.stage>=5)
        row=dict(validation_mse=float(np.mean(scores)),policy_control_mse=float(np.mean(scores)),physical=physical,
            ablated_tip_error_mm=ablated['mean_tip_error_mm'],teacher_fraction=self.teacher)
        if self.stage>=5:
            from flycasso.quality import measure
            images=[];labels=[]
            for variant in range(self.validation_count()):
                _,_,prior=self.motor_episode(False,posterior=False,split='val',variant=variant)
                # Match the evaluator's crop/padding, without changing the simulation canvas.
                bounds=ImageOps.invert(prior.canvas.convert('L')).getbbox()
                canvas=Image.new('RGB',(32,32),'white')
                if bounds:canvas.paste(ImageOps.pad(prior.canvas.crop(bounds),(28,28),color='white'),(2,2))
                images.append(torch.from_numpy(np.asarray(canvas).copy()).permute(2,0,1).float()/127.5-1)
                _,ys,indices=self.subset('val');labels.append(int(ys[indices[(variant*len(indices)//self.validation_count())%len(indices)]]))
            row['generation']=measure(torch.stack(images),torch.tensor(labels),classes=self.model.classes)
            passed=passed and row['generation']['category_accuracy']>=.6
        # A good hold can be a constant command; require circuit contribution for trajectories.
        if self.stage>=2:passed=passed and ablated['mean_tip_error_mm']>error*1.05
        return row,error,passed

    def run(self):
        previous=read_json(self.out/'status.json') if (self.out/'status.json').exists() else {}
        write_json(self.out/'status.json',dict(previous,status='training',stage_name=self.c['stage_name'],updated_at=time.time()))
        losses=[];self.model.train();self.opt.zero_grad(set_to_none=True)
        while self.step-self.first<self.args.updates:
            if self.c['task']=='image':losses.append(self.image_train())
            else:losses.append(self.motor_episode()[0])
        raw={k:v.detach().clone() for k,v in self.model.state_dict().items()}
        self.model.load_state_dict(self.ema);self.model.eval()
        row,score,passed=self.image_validate() if self.c['task']=='image' else self.motor_validate()
        self.model.load_state_dict(raw)
        row.update(step=self.step,code=self.code,stage=self.stage,stage_name=self.c['stage_name'],stage_step=self.stage_step,
            train_mse=float(np.mean(losses)),device=str(self.device),seconds_per_step=(time.monotonic()-self.start)/(self.step-self.first),
            gradient_diagnostics=self.gradients,learning_rate=self.opt.param_groups[0]['lr'],gate_passed=bool(passed),updated_at=time.time())
        improved=score<self.best-1e-5
        if improved:self.best=score;self.stale=0
        else:self.stale+=1
        self.passes=self.passes+1 if passed else 0
        if self.c['task']=='motor' and passed:self.teacher=max(0,self.teacher-.2)
        if self.stale in (16,32):
            for group in self.opt.param_groups:group['lr']*=.5
        status='queued'
        if improved or not (self.out/'best.pt').exists():self.save('best.pt')
        if self.passes>=3:
            last=3 if self.c['task']=='image' else 7
            if self.stage==last:status='curriculum_complete'
            else:
                # Preserve the stage's best before changing its target distribution.
                self.save(f'stage-{self.stage}.pt');self.stage+=1;self.stage_step=0;self.best=1e9;self.stale=0;self.passes=0
                self.teacher=.95;self.model.stage=self.stage;self.c['size']=32 if self.c['task']=='motor' or self.stage==3 else 16;self.model.size=self.c['size']
                for group in self.opt.param_groups:group['lr']=self.c['lr']
                self.write_config()
        elif self.stale>=48:status='needs_review'
        row['status']=status
        self.save('last.pt')
        with (self.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        write_json(self.out/'status.json',row);print(json.dumps(row),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('task',choices=['image','motor'])
    p.add_argument('--out');p.add_argument('--data');p.add_argument('--ports');p.add_argument('--device',default='auto')
    p.add_argument('--graph',default='data/processed/malecns/graph.npz');p.add_argument('--calibration',default='data/processed/brain-ports-v3/calibration.npz')
    p.add_argument('--updates',type=int,default=32);p.add_argument('--evaluator',default='runs/quality-cifar/best.pt')
    a=p.parse_args()
    if a.updates<1:p.error('Use positive update count')
    import fcntl
    Path('runs').mkdir(exist_ok=True)
    with open('runs/.calibrated-training.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        Training(a).run()
