"""Muscle-actuated painting with an offline inverse-dynamics demonstrator.

The trained circuit, not the demonstrator, controls inference. Named motor pools
can drive several muscle heads together; the mapping is explicitly group-level.
"""
from pathlib import Path
from functools import lru_cache
import numpy as np
import mujoco as mj
from PIL import Image, ImageDraw
from scipy.optimize import lsq_linear
from flygym.compose import MusculoskeletalFly
from flycasso.anatomy import MUSCLE_TYPES
from flycasso.common import digest


def body_sources():
    from importlib.metadata import version
    from flygym import assets_dir
    import flygym_demo as demo
    return dict(flygym=version('flygym'),mujoco=mj.__version__,
        model_sha256=digest(assets_dir/'model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml'),
        initial_pose_sha256=digest(Path(demo.__file__).parent/'muscle_imitation/assets/mocap/qpos/0002.npy'))


@lru_cache(maxsize=1)
def body_model():
    # The compiled body is read-only; every fly has independent dynamic state.
    fly=MusculoskeletalFly();spec=fly.mjcf_root
    for g in spec.geoms:g.contype=g.conaffinity=0
    spec.body('LFTarsus5').add_site(name='pen_tip',pos=(0,0,0),size=(.01,)*3)
    spec.body('LFTarsus5').add_geom(name='pen',type=mj.mjtGeom.mjGEOM_SPHERE,size=(.015,)*3,
                      mass=1e-9,contype=2,conaffinity=4)
    import flygym_demo as demo
    neutral=np.load(Path(demo.__file__).parent/'muscle_imitation/assets/mocap/qpos/0002.npy')[0]
    m=spec.compile();d=mj.MjData(m);d.qpos[:7]=neutral;mj.mj_forward(m,d)
    center=d.site_xpos[m.site('pen_tip').id].copy();z=center[2]-.013
    spec.worldbody.add_geom(name='paper',type=mj.mjtGeom.mjGEOM_BOX,
        pos=(*center[:2],z-.02),size=(.2,.2,.02),contype=4,conaffinity=2,
        friction=(.05,.001,.0001),solref=(.002,1))
    return spec.compile(),neutral,center,z


class MuscleFly:
    dt=.02
    def __init__(self):
        self.model,self.neutral,self.center,self.z=body_model()
        self.data=mj.MjData(self.model);self.teacher_data=mj.MjData(self.model)
        self.site=self.model.site('pen_tip').id;self.brush=self.model.geom('pen').id;self.board=self.model.geom('paper').id
        groups=list(dict.fromkeys(MUSCLE_TYPES))
        self.group=np.array([[float(t==g) for g in groups] for t in MUSCLE_TYPES])
        self.reset()

    def reset(self):
        mj.mj_resetData(self.model,self.data);self.data.qpos[:7]=self.neutral
        self.data.ctrl[:]=.1;self.data.act[:]=.1;mj.mj_forward(self.model,self.data)
        self.canvas=Image.new('RGB',(32,32),'white');self.ink=ImageDraw.Draw(self.canvas);self.last=None
        self.last_world=None;self.frames=[]

    def observe(self):
        image=np.asarray(self.canvas,dtype=np.float32).transpose(2,0,1)/127.5-1
        proprio=np.r_[(self.data.qpos[:7]-self.neutral),self.data.qvel[:7]/20].clip(-3,3).astype(np.float32)
        return image,proprio

    def pixel(self, point):return tuple(((point[:2]-self.center[:2])*np.array([1,-1])/.4*31+15.5).tolist())

    def step(self, action, record=False):
        action=np.asarray(action)
        if action.shape!=(15,) or not np.isfinite(action).all():raise ValueError('Expected 15 finite muscle activations')
        self.data.ctrl[:]=np.clip(action,.0001,1)
        ink=0
        self.frames=[]
        for _ in range(2):
            mj.mj_step(self.model,self.data,nstep=round(self.dt/2/self.model.opt.timestep))
            if not np.isfinite(self.data.qpos).all() or np.abs(self.data.qvel).max()>1e5:raise FloatingPointError('Unstable muscle dynamics')
            contact=next((c for c in self.data.contact if {int(c.geom1),int(c.geom2)}=={self.brush,self.board} and c.dist<=.001),None)
            if contact is not None:
                point=self.pixel(contact.pos)
                self.ink.line([self.last or point,point],fill='black',width=1);self.last=point;ink+=1
                at=contact.pos.copy();at[2]=self.z+.002
                marks=[[0,(self.last_world if self.last_world is not None else at).tolist(),at.tolist()]]
                self.last_world=at
            else:self.last=None;self.last_world=None;marks=[]
            tip=self.data.site_xpos[self.site].copy();tip[2]-=.015
            if contact is not None:tip=at.copy()
            if record:self.frames.append(dict(time=float(self.data.time),positions=self.data.geom_xpos.tolist(),
                rotations=self.data.geom_xmat.tolist(),ink=marks,pen_down=[contact is not None],pen_tip=tip.tolist()))
        return ink

    def scene(self):
        from flycasso.paint import PaintingFly
        scene=PaintingFly.scene(self)
        names={'Head_geom_Head':'nmf/c_head','Thorax_geom_Thorax':'nmf/c_thorax',
            'LEye_geom_LEye':'nmf/l_eye','REye_geom_REye':'nmf/r_eye','LWing_geom_LWing':'nmf/l_wing',
            'RWing_geom_RWing':'nmf/r_wing','pen':'nmf/lf_brush','paper':'canvas'}
        names.update({f'LFTarsus{i}_geom_LFTarsus{i}':f'nmf/lf_tarsus{i}' for i in range(1,6)})
        body_names={'Head':'nmf/c_head','Thorax':'nmf/c_thorax','LEye':'nmf/l_eye','REye':'nmf/r_eye','LWing':'nmf/l_wing','RWing':'nmf/r_wing'}
        body_names.update({f'LFTarsus{i}':f'nmf/lf_tarsus{i}' for i in range(1,6)})
        for i,g in enumerate(scene['geoms']):
            g['name']=names.get(g['name'],body_names.get(self.model.body(self.model.geom_bodyid[i]).name,g['name']))
            if g['name']=='floor':g['color'][3]=0;g['size']=[0,0,0]
        scene.update(canvas=[self.center[0]-.2,self.center[0]+.2,self.center[1]-.2,self.center[1]+.2],
            canvas_z=self.z,pen_tip=(self.data.site_xpos[self.site]-np.array([0,0,.015])).tolist(),source='FlyMimic muscle-driven LF leg / FlyGym 2.1.0',muscle_driven=True)
        return scene

    def expert(self, target):
        """IK plus nonnegative muscle-force fit; called only during data preparation."""
        m,d=self.model,self.teacher_data
        mj.mj_copyData(d,m,self.data)
        q=d.qpos.copy();velocity=d.qvel.copy()
        jac=np.zeros((3,m.nv))
        for _ in range(15):
            mj.mj_forward(m,d);mj.mj_jacSite(m,d,jac,None,self.site)
            delta=jac[:,:7].T@np.linalg.solve(jac[:,:7]@jac[:,:7].T+.001*np.eye(3),target-d.site_xpos[self.site])
            d.qpos[:7]=np.clip(d.qpos[:7]+np.clip(delta,-.05,.05),m.jnt_range[:7,0],m.jnt_range[:7,1])
        goal=d.qpos[:7].copy();d.qpos[:]=q;d.qvel[:]=velocity
        mj.mj_forward(m,d);d.qacc[:]=0;d.qacc[:7]=np.clip(400*(goal-q[:7])-40*velocity[:7],-300,300)
        mj.mj_inverse(m,d);required=d.qfrc_inverse[:7].copy()
        d.act[:]=0;mj.mj_forward(m,d);base=d.qfrc_actuator[:7].copy()
        columns=[]
        for i in range(15):
            d.act[:]=0;d.act[i]=1;mj.mj_forward(m,d);columns.append(d.qfrc_actuator[:7]-base)
        matrix=np.asarray(columns).T@self.group
        # A tiny activity penalty resolves redundant muscles without a learned helper.
        lhs=np.r_[matrix,.001*np.eye(len(self.group.T))];rhs=np.r_[required-base,np.zeros(len(self.group.T))]
        action=self.group@lsq_linear(lhs,rhs,bounds=(.0001,1),tol=1e-5).x
        return action.astype(np.float32)
