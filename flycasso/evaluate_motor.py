"""Evaluate held-out motor commands and one complete, physically executed drawing.

The PNG contains only actual brush contacts. No target paths are substituted.
"""
import argparse
import time
from pathlib import Path

import mujoco as mj
import numpy as np
import torch
from PIL import Image, ImageDraw

from flycasso.common import digest, read_json, write_json
from flycasso.paint import PaintingFly, drawing_targets
from flycasso.train_motor import load_motor


def evaluate(checkpoint="runs/motor/best.pt", out="reports/motor-evaluation", split="val"):
    torch.set_num_threads(2)
    net, info = load_motor(checkpoint)
    device = next(net.parameters()).device
    root = Path("data/processed/strokes")
    manifest = read_json(root / "manifest.json")
    path = root / f"{split}.npz"
    if digest(path) != manifest["files"][path.name]:
        raise ValueError("Motor evaluation dataset checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        observations, actions, goals = (data[k].copy() for k in ("observations", "actions", "goals"))
    with torch.inference_mode():
        predictions = torch.cat([net(x.to(device)) for x in torch.from_numpy(observations).split(32)]).cpu().numpy()
    fly = PaintingFly(); errors, baseline = [], []
    for action, goal in zip(predictions, goals):
        fly.reset();baseline.append(np.linalg.norm(fly.data.site_xpos[fly.sites[0]]-goal[0]))
        fly.data.qpos[fly.qadr] = fly.rest[fly.qadr] + 1.6*np.where(fly.drawing_joints, action, 0)
        mj.mj_forward(fly.model, fly.data)
        errors.append(np.linalg.norm(fly.data.site_xpos[fly.sites[0]]-goal[0]))
    drawings = read_json(root / "drawings.json")[split]
    drawing = min(drawings, key=lambda d: sum(len(s[0]) for s in d["drawing"]))
    fly.reset(); targets = list(drawing_targets(drawing["drawing"], fly.home))
    image = Image.new("RGB", (768, 768), "#fffdf5"); pen = ImageDraw.Draw(image)
    reference = Image.new("RGB", (768, 768), "#fffdf5"); ref = ImageDraw.Draw(reference)
    for xs, ys in drawing["drawing"]:
        ref.line([(144+x/255*480, 144+y/255*480) for x,y in zip(xs,ys)], fill="#5b6553", width=3)
    trace, marks = [], 0; started = time.monotonic()
    with torch.inference_mode():
        for i, goal in enumerate(targets):
            action = net(torch.from_numpy(fly.observe(goal))[None].to(device))[0].cpu().numpy()
            frames = fly.step(action)
            for frame in frames:
                for side, a, b in frame["ink"]:
                    xy = [((.4-p[1])/.8*768, (1.5-p[0])/.8*768) for p in (a,b)]
                    pen.line(xy, fill="#86508d", width=3); marks += 1
            trace.append(float(np.linalg.norm(fly.data.site_xpos[fly.sites[0]]-goal[0])))
            if i % 100 == 0:
                print(f"Physical drawing: {i}/{len(targets)} actions", flush=True)
    output = Path(out); output.mkdir(parents=True, exist_ok=True)
    image.save(output / "painting.png");reference.save(output / "reference.png")
    report = dict(**info, split=split, drawing_leg="lf", normalized_joint_mse=float(np.mean((predictions[:, :7]-actions[:, :7])**2)),
        pose_foot_error_mm=float(np.mean(errors)), neutral_pose_foot_error_mm=float(np.mean(baseline)),
        physical_drawing=dict(id=drawing["id"], category=drawing["category"], actions=len(targets),
            contact_marks=marks, mean_foot_error_mm=float(np.mean(trace)), foot_error_trace_mm=trace,
            simulated_seconds=fly.data.time, wall_seconds=time.monotonic()-started),
        interpretation="Joint and foot errors measure control. Contact marks alone do not establish drawing quality; inspect painting against reference.")
    write_json(output / "evaluation.json", report)
    print(f"Saved motor evaluation to {output}", flush=True)
    return report

def trace_quality(net, drawing, out=None):
    """Score actual ink against a held-out target at 0.01 mm tolerance."""
    from scipy.ndimage import distance_transform_edt
    fly = PaintingFly(); targets = list(drawing_targets(drawing["drawing"], fly.home))
    painting = Image.new("L", (768,768), 255); pen = ImageDraw.Draw(painting)
    reference = Image.new("L", (768,768), 255); ref = ImageDraw.Draw(reference)
    for xs, ys in drawing["drawing"]:
        ref.line([(144+x/255*480,144+y/255*480) for x,y in zip(xs,ys)],fill=0,width=3)
    errors=[]; neural_seconds=physics_seconds=0.; device=next(net.parameters()).device
    with torch.no_grad():
        for goal in targets:
            started=time.monotonic()
            action=net(torch.from_numpy(fly.observe(goal))[None].to(device))[0].cpu().numpy()
            neural_seconds+=time.monotonic()-started; started=time.monotonic()
            frames=fly.step(action); physics_seconds+=time.monotonic()-started
            errors.append(float(np.linalg.norm(fly.data.site_xpos[fly.sites[0]]-goal[0])))
            for frame in frames:
                for _,a,b in frame["ink"]:
                    pen.line([((.4-p[1])/.8*768,(1.5-p[0])/.8*768) for p in (a,b)],fill=0,width=3)
    actual=np.asarray(painting)<128; expected=np.asarray(reference)<128
    tolerance=.01/.8*768
    precision=float((distance_transform_edt(~expected)[actual]<=tolerance).mean()) if actual.any() else 0.
    recall=float((distance_transform_edt(~actual)[expected]<=tolerance).mean()) if actual.any() else 0.
    result=dict(stroke_f1=2*precision*recall/max(precision+recall,1e-10),precision=precision,recall=recall,
        mean_tip_error_mm=float(np.mean(errors)),tolerance_mm=.01,actions=len(targets),
        neural_seconds=neural_seconds,physics_seconds=physics_seconds,simulation_seconds=fly.data.time)
    if out:
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        painting.save(out/'painting.png');reference.save(out/'reference.png');write_json(out/'quality.json',result)
    return result


def evaluate_generated(checkpoint, out, category="all", seed=42):
    from flycasso.strokes import generate
    net,info=load_motor(checkpoint);results={}
    for name in net.classes if category=="all" else [category]:
        started=time.monotonic();drawing=generate(net,name,seed)
        planning=time.monotonic()-started
        results[name]=dict(trace_quality(net,dict(drawing=drawing),Path(out)/name),planning_seconds=planning)
    report=dict(**info,seed=seed,results=results,
        interpretation="Physical execution fidelity to independently generated strokes; not a score of category recognition.")
    write_json(Path(out)/"evaluation.json",report);return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",default="runs/motor/best.pt")
    parser.add_argument("--out",default="reports/motor-evaluation")
    parser.add_argument("--split",choices=["val","test"],default="val")
    parser.add_argument("--category", help="Evaluate category-only generation; use all for every category")
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    if args.category: evaluate_generated(args.checkpoint,args.out,args.category,args.seed)
    else: evaluate(args.checkpoint,args.out,args.split)
