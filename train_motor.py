"""A separate full-fly motor network; imitation learning from front-leg demonstrations.

python train_motor.py                  # train until held-out pen control passes
python train_motor.py --resume runs/motor/last.pt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import Plateau, device_for, digest, load_torch, read_json, save_torch, seed_all, write_json
from model import FlyDenoiser


class FlyMotor(FlyDenoiser):
    """Same full graph, independent weights, proprioception/target inputs and 14 joint outputs."""
    def __init__(self, graph, config):
        super().__init__(graph, dict(config, image_size=1))
        # Retain the common neural circuit, replace every image/time/class adapter.
        del self.image_input, self.class_input, self.time_input, self.output, self.frequency
        self.motor_input = nn.Linear(config.get("observations", 26), self.width)
        self.motor_output = nn.Linear(self.width, 14)
        self.motor_residual = config.get("motor_residual", False)
        if self.motor_residual:
            nn.init.zeros_(self.motor_output.weight);nn.init.zeros_(self.motor_output.bias)

        if config.get("stroke_diffusion"):
            self.stroke_input = nn.Sequential(nn.Conv1d(3,32,5,2,2), nn.SiLU(), nn.Conv1d(32,64,5,2,2), nn.SiLU(), nn.Flatten(), nn.Linear(64*64,self.width))
            self.stroke_output = nn.Sequential(nn.Linear(self.width,64*32), nn.Unflatten(1,(64,32)),
                nn.Upsample(scale_factor=2), nn.Conv1d(64,64,5,padding=2), nn.SiLU(),
                nn.Upsample(scale_factor=2), nn.Conv1d(64,32,5,padding=2), nn.SiLU(),
                nn.Upsample(scale_factor=2), nn.Conv1d(32,3,5,padding=2))
            if config.get("stroke_architecture") == "unet":
                scales=config.get("stroke_scales",3)
                widths=[32]+[64]*(scales-1)
                outputs=[64,32,32] if scales==3 else [64]*(scales-1)+[32]
                self.stroke_encoder=nn.ModuleList([nn.Conv1d(3 if i==0 else widths[i-1],w,5,1 if i==0 else 2,2) for i,w in enumerate(widths)])
                self.stroke_input=nn.Linear(64*(256//2**(scales-1)),self.width)
                self.stroke_context=nn.ModuleList([nn.Linear(self.width,w) for w in outputs])
                self.stroke_decoder=nn.ModuleList([nn.Sequential(nn.Conv1d(64 if i==0 else outputs[i-1]+widths[-i-1],w,5,padding=2),nn.GroupNorm(8,w)) for i,w in enumerate(outputs)])
                self.stroke_output=nn.Conv1d(outputs[-1],3,5,padding=2)
            self.stroke_time = nn.Sequential(nn.Linear(self.width,self.width), nn.SiLU(), nn.Linear(self.width,self.width))
            self.stroke_class = nn.Embedding(len(self.classes)+int(getattr(self,"null_label",None) is not None), self.width)
            self.register_buffer("stroke_frequency", torch.exp(-np.log(10000)*torch.arange(self.width//2)/max(self.width//2-1,1)), persistent=False)

    def forward(self, observation, ablate_edges=False):
        inputs = observation[:, :self.motor_input.in_features]
        if self.motor_residual:
            inputs = inputs.clone()
            inputs[:,20:26] = (observation[:,20:26]-observation[:,14:20])/.05
        command = self.motor_output(self.neural_readout(self.motor_input(inputs), ablate_edges)).tanh()
        return (observation[:,:14] + .05*command).clamp(-1,1) if self.motor_residual else command


class FootPosition(torch.autograd.Function):
    """Exact leg forward kinematics and its MuJoCo Jacobian for the training loss."""
    @staticmethod
    def forward(ctx, actions, fly):
        import mujoco as mj
        tips, jacobians = [], []
        for action in actions.detach().cpu().numpy():
            fly.reset()
            fly.data.qpos[fly.qadr] = fly.rest[fly.qadr] + 1.6 * np.where(fly.drawing_joints, action, 0)
            mj.mj_forward(fly.model, fly.data)
            tips.append(fly.data.site_xpos[fly.sites[0]].copy())
            jac = np.zeros((3, fly.model.nv))
            mj.mj_jacSite(fly.model, fly.data, jac, None, fly.sites[0])
            jacobians.append(jac[:, fly.dadr] * fly.drawing_joints * 1.6)
        ctx.save_for_backward(torch.as_tensor(np.asarray(jacobians), dtype=actions.dtype, device=actions.device))
        return torch.as_tensor(np.asarray(tips), dtype=actions.dtype, device=actions.device)

    @staticmethod
    def backward(ctx, gradient):
        (jac,) = ctx.saved_tensors
        return torch.einsum("bi,bij->bj", gradient, jac), None


@torch.no_grad()
def validation(model, inputs, targets, batch_size=32):
    model.eval()
    device = next(model.parameters()).device
    total = sum(F.mse_loss(model(x.to(device))[:, :7], y.to(device)[:, :7], reduction="sum").item()
                for x, y in zip(inputs.split(batch_size), targets.split(batch_size)))
    return total / targets[:, :7].numel()


def load_motor(path="runs/motor/best.pt", graph=None, device="auto"):
    path = Path(path)
    state, checksum = load_torch(path)
    if state.get("task") != "front_leg_motor_v1":
        raise ValueError("This is not a front-leg motor checkpoint")
    graph = graph or (path.parent / "graph.npz" if (path.parent / "graph.npz").exists() else state["config"]["graph"])
    if digest(graph) != state["graph_sha256"]:
        raise ValueError("Motor graph checksum mismatch")
    model = FlyMotor(graph, state["config"]).to(device_for(device))
    model.load_state_dict(state["model"])
    if "stroke_ema" in state: model.load_state_dict(state["stroke_ema"], strict=False)
    model.eval()
    return model, dict(training_step=state["step"], task=state["task"], neurons=model.n_neurons,
                        edges=model.n_edges, device=next(model.parameters()).device.type, checkpoint_sha256=checksum, phase=state["config"].get("phase", "control"), classes=model.classes, best_validation_metric=state["best_val"])


def train(out="runs/motor", resume=None, steps=None, threads=2, graph="data/processed/malecns/graph.npz", dataset="data/processed/strokes", device_name="auto", init_from=None):
    if resume and init_from:
        raise ValueError("Use either --resume or --init-from")
    output = Path(out); output.mkdir(parents=True, exist_ok=True)
    state = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if state and (state.get("task") != "front_leg_motor_v1" or state.get("inference_only")):
        raise ValueError("Resume needs an original motor training checkpoint, not an inference export")
    if not resume and (output / "last.pt").exists():
        raise ValueError("Existing motor run: use --resume")
    if state and state["config"].get("phase") == "strokes":
        raise ValueError("Use --phase strokes to resume category training")
    config = state["config"] if state else dict(graph=str(Path(graph).resolve()), dataset=str(Path(dataset).resolve()),
            width=64, recurrent_steps=3, motor_residual=True, synaptic_output=True, readout_lr_scale=64/166700, stroke_diffusion=True, classes=["cat","flower","butterfly"], leak=0.5, graph_control="real", seed=43, batch_size=32, learning_rate=0.0003)
    seed_all(config["seed"], threads)
    device = device_for(device_name)
    root = Path(config["dataset"])
    manifest = read_json(root / "manifest.json")
    if manifest.get("drawing_leg") != "lf":
        raise ValueError("Prepare the one-leg stroke dataset before training")
    config["drawing_leg"] = "lf"
    config["observations"] = manifest["observations"]
    if config["observations"] != 40:
        raise ValueError("Rebuild the executed movement dataset with prepare_strokes.py")
    hashes = dict(graph_sha256=digest(config["graph"]), data_sha256=digest(root / "manifest.json"))
    if state and any(state[k] != value for k, value in hashes.items()):
        raise ValueError("Motor graph or training data changed")
    arrays = []
    for split in ("train", "val"):
        path = root / f"{split}.npz"
        if digest(path) != manifest["files"][path.name]:
            raise ValueError("Motor dataset checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            arrays.extend([torch.from_numpy(data[k].copy()) for k in ("observations", "actions", "goals")])
    train_x, train_y, train_goals, val_x, val_y, val_goals = arrays
    val_x, val_y, val_goals = val_x[::max(1,len(val_x)//512)], val_y[::max(1,len(val_y)//512)], val_goals[::max(1,len(val_goals)//512)]
    model = FlyMotor(config["graph"], config).to(device)
    if model.n_neurons != 166700 or model.n_edges != 25582938 or not model.graph_metadata.get("source_verified"):
        raise ValueError("Motor training requires the full verified fly graph")
    if init_from:
        initial,initial_checksum = load_torch(init_from)
        if initial.get("task") != "front_leg_motor_v1" or initial["graph_sha256"] != hashes["graph_sha256"]:
            raise ValueError("Initial motor weights must use the same full graph")
        weights = initial["model"]
        old = weights["motor_input.weight"]
        if old.shape[1] != config["observations"]:
            weights["motor_input.weight"] = F.pad(old, (0, config["observations"] - old.shape[1]))
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if unexpected or any(not name.startswith("stroke_") for name in missing):
            raise ValueError("Incompatible initial motor weights")
        if model.motor_residual and not initial["config"].get("motor_residual"):
            nn.init.zeros_(model.motor_output.weight);nn.init.zeros_(model.motor_output.bias)
        config["initialized_from"] = dict(sha256=initial_checksum, step=initial["step"])
        del initial
    parameters = model.parameters()
    if "readout_lr_scale" in config:
        # Bound a coherent Adam update across the 166k-input readout.
        parameters = [dict(params=[p for n,p in model.named_parameters() if n!="neurons_to_output.weight"]),
            dict(params=[model.neurons_to_output.weight],lr=config["learning_rate"]*config["readout_lr_scale"])]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.0001)
    plateau = Plateau(state.get("plateau") if state else None, min_steps=5000, min_delta=0.00001)
    recovery = state.get("recovery") if state else None
    step, best = 0, float("inf")
    if state:
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"]); step, best = state["step"], state["best_val"]
    del state
    end = step + steps if steps is not None else float("inf")
    write_json(output / "config.json", config)
    write_json(output / "run.json", dict(config=config, **hashes, neurons=model.n_neurons, edges=model.n_edges,
                task="front_leg_motor_v1", device=str(device), code={p: digest(p) for p in ("train_motor.py", "model.py", "paint.py")}))

    def checkpoint(name):
        save_torch(output / name, dict(task="front_leg_motor_v1", config=config, step=step,
            model=model.state_dict(), optimizer=optimizer.state_dict(), plateau=plateau.state,
            best_val=best, recovery=recovery, rng=torch.get_rng_state(), **hashes))

    from paint import PaintingFly
    fly = PaintingFly()
    from prepare_strokes import collect_rollouts
    drawings = read_json(root / "drawings.json")["train"]
    started = time.monotonic(); start_step = step
    print(f"Training independent motor weights through {model.n_neurons} neurons and {model.n_edges} edges", flush=True)
    with (output / "metrics.jsonl").open("a") as log:
        while step < end:
            model.train()
            idx = torch.randint(len(train_x), (config["batch_size"],))
            bx, by, bg = train_x[idx], train_y[idx], train_goals[idx]
            if recovery is not None:
                count=len(idx)//2; pick=torch.randint(len(recovery["observations"]),(count,))
                bx[:count]=recovery["observations"][pick]; by[:count]=recovery["actions"][pick]; bg[:count]=recovery["goals"][pick]
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx.to(device))
            joint_loss = F.mse_loss(pred[:, :7], by.to(device)[:, :7])
            tip_loss = F.mse_loss(FootPosition.apply(pred, fly), bg[:, 0].to(device)) / 0.5**2
            loss = tip_loss + 0.05 * joint_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite motor loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); step += 1
            row = dict(step=step, train_loss=loss.item(), joint_loss=joint_loss.item(), tip_loss=tip_loss.item(), device=str(device), learning_rate=optimizer.param_groups[0]["lr"],
                       seconds_per_step=(time.monotonic()-started)/(step-start_step))
            done = False
            if step % 250 == 0 or step == end or step == 25:
                score = validation(model, val_x, val_y)
                with torch.no_grad():
                    tip_errors = torch.cat([torch.linalg.vector_norm(FootPosition.apply(model(x.to(device)), fly) - g[:, 0].to(device), dim=1)
                        for x, g in zip(val_x.split(32), val_goals.split(32))])
                    tip_error = tip_errors.mean().item()
                row["validation"] = dict(joint_mse=score, tip_error_mm=tip_error, neutral_baseline_mse=val_y[:, :7].square().mean().item(), examples=len(val_x))
                # The initial checkpoint is available for integration; stopping decisions use regular checks.
                if step % 250 == 0:
                    plateau.update(tip_error, step, optimizer)
                    from evaluate_motor import trace_quality
                    heldout = read_json(root / "drawings.json")["val"]
                    checks = [trace_quality(model, heldout[i], output / f"checks/{step}/{i}") for i in (0,16,32)]
                    row["physical"] = checks
                    done = all(q["stroke_f1"] >= .90 and q["mean_tip_error_mm"] < .01 for q in checks)
                row["plateau"] = dict(plateau.state)
                if tip_error < best or done:
                    best = tip_error; checkpoint("best.pt")
                checkpoint("last.pt")
            if step % 10 == 0 or "validation" in row:
                log.write(json.dumps(row) + "\n"); log.flush()
                status = "control_quality_reached" if done else "step_limit" if step == end else "training"
                write_json(output / "status.json", dict(row, status=status, updated_at=time.time()))
                print(json.dumps(row), flush=True)
            if done:
                break
            if step % 250 == 0:
                # DAgger: append teacher corrections at states the learned policy visits.
                model.eval()
                start = (step // 250 * 2) % len(drawings)
                recovered = collect_rollouts(fly, drawings[start:start+2], model)
                incoming = {key: torch.from_numpy(value) for key,value in recovered.items()}
                recovery = {key: torch.cat([recovery[key], value])[-8192:] if recovery is not None else value
                            for key,value in incoming.items()}
                checkpoint("last.pt")
    checkpoint("last.pt")

def remap_stroke_classes(current, old, names, old_names):
    result=current.clone()
    for i,name in enumerate(names):
        if name in old_names: result[i]=old[old_names.index(name)]
    if len(result)>len(names):
        result[-1]=old[-1] if len(old)>len(old_names) else old.mean(0)
    return result


def train_strokes(out, initial, resume=None, steps=None, device_name="auto", threads=2, init_strokes=None, dataset="data/processed/quickdraw-strokes-10"):
    """Learn category-to-strokes while keeping the trained leg controller fixed."""
    from strokes import StrokeDenoiser, decode
    from diffusion import Diffusion
    from PIL import Image, ImageDraw
    if resume and init_strokes: raise ValueError("Use --init-strokes only for a new run")
    source = resume or initial
    if not source: raise ValueError("Stroke training requires --init-from with a trained controller")
    state, source_checksum = load_torch(source)
    if state.get("task") != "front_leg_motor_v1" or state.get("inference_only"):
        raise ValueError("Stroke training requires an original motor training checkpoint")
    if digest(state["config"]["graph"]) != state["graph_sha256"]:
        raise ValueError("Motor graph checksum mismatch")
    root=Path(state["config"].get("stroke_dataset","data/processed/quickdraw-strokes") if resume else dataset)
    manifest=read_json(root/"manifest.json")
    config = dict(state["config"], stroke_architecture="unet")
    if not resume: config.update(stroke_scales=5,condition_dropout=.1,guidance_scale=2.,classes=manifest["classes"],stroke_dataset=str(root.resolve()))
    if not config.get("stroke_diffusion"): raise ValueError("Train the new controller first")
    if resume and config.get("phase") != "strokes": raise ValueError("Resume a stroke-phase checkpoint")
    seed_all(config["seed"], threads); device = device_for(device_name)
    model = FlyMotor(config["graph"], config).to(device)
    current=model.state_dict()
    compatible={k:v for k,v in state["model"].items() if k in current and v.shape==current[k].shape}
    compatible["stroke_class.weight"]=remap_stroke_classes(current["stroke_class.weight"],state["model"]["stroke_class.weight"],model.classes,state["config"]["classes"])
    missing,_=model.load_state_dict(compatible,strict=False)
    if any(not k.startswith("stroke_") for k in missing): raise ValueError("Incompatible control weights")
    if init_strokes:
        previous,planner_checksum=load_torch(init_strokes)
        if previous.get('task')!='front_leg_motor_v1' or previous['graph_sha256']!=state['graph_sha256']:
            raise ValueError("Initial stroke weights require the same fly graph")
        weights=dict(previous['model'],**previous.get('stroke_ema',{}))
        compatible={k:v for k,v in weights.items() if k.startswith('stroke_') and k in current and v.shape==current[k].shape}
        compatible['stroke_class.weight']=remap_stroke_classes(current['stroke_class.weight'],weights['stroke_class.weight'],model.classes,previous['config']['classes'])
        model.load_state_dict(compatible,strict=False)
        config['stroke_initialized_from']=dict(sha256=planner_checksum,step=previous['step'])
        del previous
    config.update(phase="strokes", control_training_steps=config.get("control_training_steps", state["step"]))
    if manifest["classes"] != model.classes: raise ValueError("Stroke categories differ from controller")
    data_hash = digest(root / "manifest.json")
    if resume and state["stroke_data_sha256"] != data_hash: raise ValueError("Stroke training data changed")
    arrays=[]
    for split in ("train","val"):
        path=root/f"{split}.npz"
        if digest(path)!=manifest["files"][path.name]: raise ValueError("Stroke dataset checksum mismatch")
        with np.load(path,allow_pickle=False) as data:
            arrays.extend([torch.from_numpy(data[k].copy()) for k in ("strokes","labels")])
    train_x,train_y,val_x,val_y=arrays
    for name,p in model.named_parameters(): p.requires_grad_(name.startswith("stroke_"))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.0001)
    net=StrokeDenoiser(model); diffusion=Diffusion(1000,device)
    ema={k:v.clone() for k,v in model.state_dict().items() if k.startswith("stroke_")}
    step=0; best=float('inf')
    if resume:
        optimizer.load_state_dict(state["optimizer"]);step=state["step"];best=state["best_val"]
        ema={k:v.to(device) for k,v in state["stroke_ema"].items()};torch.set_rng_state(state["rng"])
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    if not resume and (output/'last.pt').exists(): raise ValueError("Existing stroke run: use --resume")
    hashes={k:state[k] for k in ("graph_sha256","data_sha256")};del state
    write_json(output/'config.json',config)
    write_json(output/('resume-environment.json' if resume else 'run.json'),dict(config=config,**hashes,
        stroke_data_sha256=data_hash,initial_checkpoint_sha256=source_checksum,device=str(device),
        code={name:digest(name) for name in ('train_motor.py','model.py','metal.py','strokes.py','diffusion.py')}))
    def checkpoint(name):
        save_torch(output/name,dict(task="front_leg_motor_v1",config=config,step=step,model=model.state_dict(),
            optimizer=optimizer.state_dict(),stroke_ema=ema,best_val=best,rng=torch.get_rng_state(),stroke_data_sha256=data_hash,**hashes))
    start=time.monotonic(); first=step; end=step+steps if steps else float('inf')
    with (output/'metrics.jsonl').open('a') as log:
        while step<end:
            model.train(); idx=torch.randint(len(train_x),(32,));x=train_x[idx].to(device).reshape(-1,3,16,16);labels=train_y[idx].to(device)
            t=torch.randint(1000,(len(x),)).to(device);noise=torch.randn(x.shape).to(device)
            condition=labels.clone()
            if config.get("condition_dropout"):
                condition[(torch.rand(len(labels))<config["condition_dropout"]).to(device)]=len(model.classes)
            pred=net(diffusion.add_noise(x,t,noise),t,condition)
            alpha=diffusion.alpha_bar[t];weight=(alpha/(1-alpha)).clamp(max=5)
            error=(pred-x).square().flatten(1).mean(1)
            loss=(error*weight).mean()
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite stroke loss")
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True);optimizer.step();step+=1
            with torch.no_grad():
                decay=min(.999,(step+1)/(step+10))
                for k,v in model.state_dict().items():
                    if k in ema: ema[k].lerp_(v,1-decay)
            row=dict(step=step,train_loss=loss.item(),unweighted_mse=error.mean().item(),phase="strokes",device=str(device),seconds_per_step=(time.monotonic()-start)/(step-first))
            if step%250==0 or step==end:
                model.eval();raw={k:v.detach().clone() for k,v in model.state_dict().items() if k in ema};model.load_state_dict(ema,strict=False)
                rng=torch.Generator().manual_seed(2026);scores=[]
                with torch.no_grad():
                    for _ in range(8):
                        idx=torch.randint(len(val_x),(32,),generator=rng);x=val_x[idx].to(device).reshape(-1,3,16,16);labels=val_y[idx].to(device)
                        t=torch.randint(1000,(len(x),),generator=rng).to(device);noise=torch.randn(x.shape,generator=rng).to(device)
                        scores.append(F.mse_loss(net(diffusion.add_noise(x,t,noise),t,labels),x).item())
                    labels=torch.arange(len(model.classes),device=device).repeat_interleave(3)
                    samples,_=diffusion.sample(net,labels,2026,50)
                columns=int(np.ceil(np.sqrt(len(samples))))
                canvas=Image.new("RGB",(columns*256,int(np.ceil(len(samples)/columns))*256),"white");pen=ImageDraw.Draw(canvas)
                evaluation_images=[]
                for i,sequence in enumerate(samples.reshape(-1,3,256).numpy()):
                    try: drawing=decode(sequence)
                    except ValueError: drawing=[]
                    ox=i%columns*256;oy=i//columns*256
                    from prepare_images import rasterize
                    evaluation_images.append(rasterize(drawing))
                    for xs,ys in drawing:pen.line([(ox+16+x/255*224,oy+16+y/255*224) for x,y in zip(xs,ys)],fill="black",width=2)
                if Path('runs/quality-strokes/best.pt').exists():
                    from quality import measure
                    row['generation']=measure(torch.from_numpy(np.stack(evaluation_images)).float()/127.5-1,labels,'runs/quality-strokes/best.pt',model.classes)
                canvas.save(output/'preview.png')
                (output/'samples').mkdir(exist_ok=True);canvas.save(output/'samples'/f'{step:06d}.png')
                model.load_state_dict(raw,strict=False)
                score=sum(scores)/len(scores);row['validation']=dict(stroke_mse=score,weights="ema")
                if score<best:best=score;checkpoint('best.pt')
                checkpoint('last.pt')
            if step%10==0 or 'validation' in row:
                print(json.dumps(row),flush=True);log.write(json.dumps(row)+'\n');log.flush()
                write_json(output/'status.json',dict(row,status="step_limit" if step==end else "training",updated_at=time.time()))
    checkpoint('last.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    parser.add_argument("--phase", choices=["control","strokes"], default="control")
    parser.add_argument("--resume")
    parser.add_argument("--stroke-dataset",default="data/processed/quickdraw-strokes-10")
    parser.add_argument("--init-strokes", help="Initialize stroke adapters while retaining the selected controller")
    parser.add_argument("--init-from", help="Start a new run from motor weights, with a fresh optimizer and training log")
    parser.add_argument("--steps", type=int, help="Optional additional step limit for debugging; normal control training uses held-out drawing accuracy")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--graph", default="data/processed/malecns/graph.npz")
    parser.add_argument("--dataset", default="data/processed/strokes")
    args = parser.parse_args()
    if args.threads < 1 or (args.steps is not None and args.steps < 1):
        parser.error("Threads and step limits must be positive")
    args.out = args.out or (str(Path(args.resume).parent) if args.resume else "runs/motor")
    if args.init_strokes and args.phase!="strokes": parser.error("--init-strokes requires --phase strokes")
    if args.phase == "strokes":
        train_strokes(args.out, args.init_from, args.resume, args.steps, args.device, args.threads, args.init_strokes,args.stroke_dataset)
    else:
        train(args.out, args.resume, args.steps, args.threads, args.graph, args.dataset, args.device, args.init_from)
