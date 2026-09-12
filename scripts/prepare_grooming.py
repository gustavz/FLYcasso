"""Bake collision-checked, display-only grooming. This is not learned behavior."""
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares
from paint import PaintingFly


def prepare(output="web/assets/grooming.json"):
    fly = PaintingFly()
    m, d = fly.model, fly.data
    legs = [[i for i in range(m.ngeom) if m.geom_type[i] == mj.mjtGeom.mjGEOM_MESH
             and m.geom(i).name.startswith("nmf/"+side+"_")] for side in ("lf", "rf")]
    geoms = legs[0]+legs[1]
    body = [m.geom("nmf/"+name).id for name in ("c_head", "c_thorax")]
    pairs = [(a,b) for a in legs[0] for b in legs[1]] + [
        (a,b) for leg in legs for a in leg for b in body
        if not (m.geom(a).name.endswith("coxa") and b == body[1])]
    # The coxa/thorax socket is an anatomical attachment, not a collision pair.
    def distances():
        return np.array([mj.mj_geomDistance(m,d,a,b,1,None) for a,b in pairs])
    rest = fly.rest[fly.qadr].copy()
    clearance = np.minimum(.006, distances())
    assert clearance.min() >= 0
    def residual(q, goals, settle):
        d.qpos[fly.qadr] = q; mj.mj_forward(m,d)
        return np.r_[(d.site_xpos[fly.sites]-goals).flatten(),
                     np.minimum(distances()-clearance,0)*8, (q-rest)*settle]
    duration = 3.2
    times = np.linspace(0,duration,97)
    poses = [rest.copy()]
    for t in times[1:-1]:
        ramp = min(t/.55,(duration-t)/.55,1)
        blend = ramp*ramp*(3-2*ramp)
        slide = .05*np.sin(2*np.pi*2.5*(t-.55))
        goals = np.array([[1.3+slide,.07,.70],[1.3-slide,-.07,.70]])
        goals = fly.home*(1-blend)+goals*blend
        result = least_squares(residual,poses[-1],args=(goals,.0002+(1-blend)*.02),
            bounds=(rest-1.6,rest+1.6),diff_step=1e-4,max_nfev=100,ftol=1e-6)
        poses.append(result.x)
    poses.append(rest.copy())
    frames=[]; minimum=1.;tips=[]
    # Interpolate JOINT angles offline, then bake exact FK at 120 Hz.
    # The browser never independently blends disconnected segment transforms.
    for t in np.linspace(0,duration,385):
        d.qpos[fly.qadr] = [np.interp(t,times,np.asarray(poses)[:,j]) for j in range(len(rest))]
        mj.mj_forward(m,d)
        separation=float(distances().min());minimum=min(minimum,separation)
        assert separation >= -1e-5, f"Grooming collision at {t:.3f}s: {separation:.6f} mm"
        tips.append(d.site_xpos[fly.sites].copy())
        frames.append(dict(positions=d.geom_xpos[geoms].round(6).tolist(),
                           rotations=d.geom_xmat[geoms].round(7).reshape(-1,9).tolist()))
    assert frames[0] == frames[-1]
    assert np.min(np.linalg.norm(np.asarray(tips)[:,0]-np.asarray(tips)[:,1],axis=1)) < .19
    result=dict(description="Illustrative collision-checked joint motion; not a model prediction",
                duration=duration,names=[m.geom(i).name for i in geoms],frames=frames,
                minimum_clearance_mm=minimum,collision_pairs=len(pairs))
    Path(output).write_text(json.dumps(result,separators=(",",":")))
    print(f"Saved {len(frames)} poses; minimum clearance {minimum:.6f} mm across {len(pairs)} collision pairs")


if __name__ == "__main__":
    prepare()
