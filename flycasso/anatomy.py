"""Build auditable, fixed electrode and motor-population interfaces from MaleCNS.

Optic-lobe column assignments are measured annotations. RGB/cue assignments and
proprioceptive tuning are artificial interfaces, not recovered receptor responses.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from flycasso.common import digest, write_json

# Group-level correspondence only: split muscle heads share the annotated pool.
MUSCLE_TYPES = [
    "Tergopleural/Pleural promotor MN", "Tergopleural/Pleural promotor MN",
    "Pleural remotor/abductor MN", "Tergopleural/Pleural promotor MN",
    "Sternal anterior rotator MN", "Sternal posterior rotator MN", "Sternal adductor MN",
    "Tr flexor MN", "Sternotrochanter MN", "Tergotr. MN", "Acc. tr flexor MN",
    "Tr extensor MN", "Tr flexor MN", "Ti flexor MN", "Ti extensor MN",
]


def prepare(graph="data/processed/malecns/graph.npz", annotations="data/raw/annotations.feather", out="data/processed/brain-ports-v2"):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    with np.load(graph) as data: ids = data["ids"]
    table = pd.read_feather(annotations).set_index("bodyId").reindex(ids)
    channels = np.full(len(ids), -1, np.int32); scale = np.zeros(len(ids), np.float32)
    located = table.assignedOlHex1.notna() & table.assignedOlHex2.notna() & (table.somaSide == "L")
    coords = table[["assignedOlHex1", "assignedOlHex2"]].to_numpy(float)
    lo, hi = np.nanmin(coords[located], axis=0), np.nanmax(coords[located], axis=0)
    # An affine electrode grid, not a calibrated optical projection.
    xy = (coords-lo)/(hi-lo)*31
    for color, cell in enumerate(["L1", "L2", "L5"]):
        rows = np.flatnonzero(located & (table.type == cell))
        pix = np.rint(xy[rows]).clip(0,31).astype(int)
        channels[rows] = color*1024 + pix[:,1]*32 + pix[:,0]
        scale[rows] = 1
    # Artificial cues enter anatomical visual feedback neurons, not arbitrary IDs.
    # This preserves a measurable route from category electrodes to optic readout.
    cue = np.flatnonzero(table.superclass == "visual_centrifugal")
    np.random.default_rng(42).shuffle(cue)
    channels[cue] = 3072 + np.arange(len(cue))%16; scale[cue] = .5
    xx, yy = np.meshgrid(np.arange(32), np.arange(32)); grid = np.c_[xx.ravel(), yy.ravel()]
    outputs=[]
    for cell in ["Tm1", "Tm2", "Tm9"]:
        rows=np.flatnonzero(located & (table.type==cell))
        if not len(rows): raise ValueError(f"Missing column population {cell}")
        outputs.extend(rows[cKDTree(xy[rows]).query(grid,k=2)[1]])
    common=dict(input_channel=channels,input_scale=scale)
    image=dict(common,output_index=np.asarray(outputs,np.int32),output_scale=np.full((3072,2),4.,np.float32))
    np.savez_compressed(out/"image.npz",**image)
    # Front-left proprioceptive afferents. Channel tuning is a declared engineering assumption.
    sensory=np.flatnonzero((table['class']=="mechanosensory_proprioceptive") & (table.rootSide=="L") & table.entryNerve.fillna('').str.contains('ProN|ProLN'))
    if len(sensory)<14: raise ValueError("Insufficient annotated front-leg proprioceptive inputs")
    channels=channels.copy();scale=scale.copy()
    channels[cue]=-1;scale[cue]=0
    motor_cue=np.flatnonzero(table.superclass=="descending_neuron")
    np.random.default_rng(42).shuffle(motor_cue)
    channels[motor_cue]=3072+np.arange(len(motor_cue))%16;scale[motor_cue]=.5
    channels[sensory]=3088+np.arange(len(sensory))%14;scale[sensory]=.5
    pools=[];mapping=[]
    for name in MUSCLE_TYPES:
        rows=np.flatnonzero((table.superclass=="vnc_motor")&(table.subclass=="fl")&(table.somaSide=="L")&(table.type==name))
        if not len(rows): raise ValueError(f"Missing motor pool: {name}")
        pools.append(rows);mapping.append(dict(type=name,body_ids=ids[rows].tolist()))
    width=max(map(len,pools));index=np.zeros((15,width),np.int32);weights=np.zeros((15,width),np.float32)
    for i,rows in enumerate(pools):index[i,:len(rows)]=rows;weights[i,:len(rows)]=1/len(rows)
    np.savez_compressed(out/"motor.npz",input_channel=channels,input_scale=scale,output_index=index,output_scale=weights)
    from flycasso.body import prepare as prepare_body
    prepare_body(graph,annotations,out/'body.npz')
    write_json(out/"manifest.json",dict(graph_sha256=digest(graph),annotations_sha256=digest(annotations),
        files={p.name:digest(p) for p in out.glob('*.npz')},motor_pools=mapping,
        image_cue_body_ids=ids[cue].tolist(),motor_cue_body_ids=ids[motor_cue].tolist(),
        sensory_body_ids=ids[sensory].tolist(),
        limitations=["RGB assigned to L1/L2/L5 electrodes; not photoreceptor physiology",
            "Cue electrodes are artificial CB inputs", "Front-leg afferent feature tuning is engineered",
            "Motor mapping is by named muscle group; individual head innervation is unresolved",
            "MaleCNS neural anatomy and FlyMimic biomechanics come from different specimens"]))
    print(f"Prepared {len(ids)}-neuron interfaces; {len(sensory)} proprioceptive inputs",flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='data/processed/brain-ports-v2')
    prepare(out=p.parse_args().out)
