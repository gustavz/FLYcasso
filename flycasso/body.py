"""Fixed motor-population readouts for a tethered peripheral-motion experiment.

Leg antagonists use named muscle groups. Other readouts pool a body-region class;
their axes, gains and limits are engineering choices, not recovered innervation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from flycasso.common import digest


def joints():
    result=[]
    def add(body,axis,extent,subclass,side=None,positive=None,negative=None):
        result.append(dict(body=body,axis=axis,extent=extent,subclass=subclass,side=side,
                           positive=positive,negative=negative,name=f'peripheral_{body}_{axis}'))
    for leg,group in [('RF','fl'),('LM','ml'),('RM','ml'),('LH','hl'),('RH','hl')]:
        for part,extent,pos,neg in [('Coxa',.18,'Sternal anterior rotator MN','Sternal posterior rotator MN'),
            ('Trochanter' if leg=='RF' else 'Femur',.22,'Tr flexor MN','Tr extensor MN'),
            ('Tibia',.25,'Ti flexor MN','Ti extensor MN')]:
            add(leg+part,'pitch',extent,group,leg[0],pos,neg)
    add('Head','yaw',.15,'nm')
    add('A1A2','pitch',.12,'ad')
    add('Rostrum','pitch',.15,'pm',positive='MN1',negative='MN2V')
    for side in ['L','R']:
        add(side+'Wing','roll',.12,'wm',side)
        add(side+'Pedicel','pitch',.12,'am',side)
        add(side+'Haltere','roll',.10,'hm',side)
    return result


def add_joints(spec):
    from flygym.utils.mjcf import add_actuator
    axes={'pitch':(0,1,0),'yaw':(0,0,1),'roll':(1,0,0)}
    for item in joints():
        body=spec.body(item['body']);name=f"joint_{item['body']}_{item['axis']}"
        joint=next((j for j in body.joints if j.name==name),None)
        if joint is None:joint=body.add_joint(name=name,axis=axes[item['axis']])
        else:
            for constraint in list(spec.equalities):
                if constraint.name==name+'_locked':spec.delete(constraint)
        limit=item['extent'];joint.limited=True;joint.range=(-limit,limit)
        joint.stiffness[:]=(.4,0,0);joint.damping[:]=(.02,0,0);joint.armature=.0005;joint.springref=0
        add_actuator(spec,'position',name=item['name'],joint=name,kp=40.,kv=.2,
            ctrllimited=True,ctrlrange=(-.6*limit,.6*limit),forcelimited=True,forcerange=(-20.,20.))


def prepare(graph,annotations,out):
    with np.load(graph) as f:ids=f['ids']
    table=pd.read_feather(annotations).set_index('bodyId').reindex(ids)
    pools=[];weights=[];mapping=[]
    for item in joints():
        mask=table.superclass.isin(['vnc_motor','cb_motor']) & (table.subclass==item['subclass'])
        if item['side']:mask &= table.somaSide==item['side']
        groups=[]
        if item['positive']:
            groups=[(mask & (table.type==item['positive']),1.),(mask & (table.type==item['negative']),-1.)]
        elif item['body']=='Head':groups=[(mask & (table.somaSide=='L'),1.),(mask & (table.somaSide=='R'),-1.)]
        else:groups=[(mask,1.)]
        indices=[];values=[]
        for selected,sign in groups:
            rows=np.flatnonzero(selected)
            if not len(rows):raise ValueError(f'Missing motor population for {item["name"]}')
            indices.extend(rows);values.extend([sign/len(rows)]*len(rows))
        pools.append(indices);weights.append(values)
        mapping.append(dict(item,body_ids=ids[indices].tolist(),weights=values))
    width=max(map(len,pools));index=np.zeros((len(pools),width),np.int32);scale=np.zeros_like(index,dtype=np.float32)
    for i,(p,w) in enumerate(zip(pools,weights)):index[i,:len(p)]=p;scale[i,:len(w)]=w
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,index=index,scale=scale,graph_sha256=digest(graph),annotations_sha256=digest(annotations),
        mapping=json.dumps(mapping),version=1)
    print(f'Prepared {len(pools)} peripheral joint readouts',flush=True)


class BodyReadout:
    def __init__(self,path,graph_sha256,n_neurons,device):
        with np.load(path,allow_pickle=False) as f:
            if int(f['version'])!=1 or str(f['graph_sha256'])!=graph_sha256:raise ValueError('Body ports do not match this circuit')
            index=f['index'];scale=f['scale'];self.mapping=json.loads(str(f['mapping']))
        if len(self.mapping)!=len(joints()) or [r['name'] for r in self.mapping]!=[r['name'] for r in joints()]:raise ValueError('Changed body joint mapping')
        if index.ndim!=2 or index.shape!=scale.shape or len(index)!=len(joints()) or index.dtype.kind not in 'iu' or np.any(index<0) or np.any(index>=n_neurons) or not np.isfinite(scale).all():raise ValueError('Invalid body motor ports')
        self.index=torch.tensor(index,dtype=torch.long,device=device);self.scale=torch.tensor(scale,device=device)
        self.extent=torch.tensor([.6*r['extent'] for r in joints()],device=device)

    @torch.no_grad()
    def __call__(self,state):
        activity=(state[self.index]*self.scale[...,None]).sum(1).t()
        return torch.tanh(8*activity)*self.extent
