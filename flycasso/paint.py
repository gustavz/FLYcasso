"""NeuroMechFly front-leg painting: actual MuJoCo joints and brush contacts.

The thorax is tethered. Five supporting legs hold their neutral pose; the left
front leg is driven by the learned motor model. IK runs only during offline preparation.
Coordinates are millimetres; brushes are small engineered spheres on the tarsi.
"""

import json

from flycasso.common import ROOT

import mujoco as mj
import numpy as np
from flygym.anatomy import ActuatedDOFPreset, AxisOrder, BodySegment, JointPreset, Skeleton
from flygym.compose import ActuatorType, KinematicPosePreset, NeuroMechFly, TetheredWorld
from flygym.utils.math import Rotation3D

CANVAS_Z = 0.15
BRUSH_RADIUS = 0.025
CANVAS = [0.85, 1.35, -0.25, 0.25]
CONTROL_DT = 0.04


class PaintingFly:
    def __init__(self):
        fly = NeuroMechFly()
        fly.add_joints(Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL,
                      joint_preset=JointPreset.LEGS_ACTIVE_ONLY),
                      neutral_pose=KinematicPosePreset.NEUTRAL, stiffness=0, damping=0.5)
        fly.add_actuators(fly.skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY),
                          ActuatorType.POSITION, kp=50, neutral_input=KinematicPosePreset.NEUTRAL)
        fly.colorize()
        # Only the engineered brush spheres collide with the canvas; body is tethered.
        for geom in fly.mjcf_root.geoms:
            geom.contype = geom.conaffinity = 0
        for side in ("lf", "rf"):
            body = fly.bodyseg_to_mjcfbody[BodySegment(side + "_tarsus5")]
            body.add_site(name=side + "_tip", pos=(0, 0, -0.03), size=(0.01,)*3)
            body.add_geom(name=side + "_brush", type=mj.mjtGeom.mjGEOM_SPHERE,
                          pos=(0, 0, -0.03), size=(BRUSH_RADIUS,)*3, mass=1e-9,
                          contype=2 if side == "lf" else 0, conaffinity=4 if side == "lf" else 0,
                          rgba=(0.25, 0.1, 0.45, 1 if side == "lf" else 0))
        world = TetheredWorld()
        world.add_fly(fly, (0, 0, 0), Rotation3D("quat", (1, 0, 0, 0)))
        world.mjcf_root.worldbody.add_geom(name="canvas", type=mj.mjtGeom.mjGEOM_BOX,
            pos=(1.1, 0, CANVAS_Z - 0.025), size=(0.4, 0.4, 0.025),
            contype=4, conaffinity=2, friction=(0.1, 0.001, 0.0001),
            solref=(0.002, 1), rgba=(0.98, 0.96, 0.91, 1))
        self.model, self.data = world.compile()
        m, d = self.model, self.data
        mj.mj_resetDataKeyframe(m, d, 0)
        self.rest = d.qpos.copy()
        self.front = np.array([i for i in range(m.njnt) if any(s in m.joint(i).name for s in ("lf_", "rf_"))])
        self.qadr, self.dadr = m.jnt_qposadr[self.front], m.jnt_dofadr[self.front]
        self.actuators = np.array([i for i in range(m.nu) if m.actuator_trnid[i, 0] in self.front])
        self.drawing_joints = np.array(["lf_" in m.joint(i).name for i in self.front])
        self.sites = [m.site("nmf/" + s + "_tip").id for s in ("lf", "rf")]
        self.brushes = [m.geom("nmf/lf_brush").id]
        self.board = m.geom("canvas").id
        mj.mj_forward(m, d)
        self.home = d.site_xpos[self.sites].copy()
        self.last_ink = [None]

    def reset(self):
        mj.mj_resetDataKeyframe(self.model, self.data, 0)
        mj.mj_forward(self.model, self.data)
        self.last_ink = [None]

    def observe(self, goals):
        return np.concatenate([(self.data.qpos[self.qadr] - self.rest[self.qadr]) / 1.6,
                               self.data.site_xpos[self.sites].flatten(), np.asarray(goals).flatten(),
                               np.clip(self.data.qvel[self.dadr] / 10, -5, 5)]).astype(np.float32)

    def expert(self, goals, iterations=45):
        """Damped Jacobian inverse kinematics. Never called by the live policy."""
        m, d = self.model, self.data
        jac = np.zeros((3, m.nv))
        for _ in range(iterations):
            mj.mj_forward(m, d)
            error = (goals - d.site_xpos[self.sites]).flatten()
            if np.linalg.norm(error) < 0.001:
                break
            rows = []
            for site in self.sites:
                mj.mj_jacSite(m, d, jac, None, site)
                rows.append(jac[:, self.dadr].copy())
            j = np.concatenate(rows)
            delta = j.T @ np.linalg.solve(j @ j.T + 0.001 * np.eye(6), error)
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + np.clip(delta, -0.15, 0.15),
                                      self.rest[self.qadr] - 1.6, self.rest[self.qadr] + 1.6)
        mj.mj_forward(m, d)
        return (d.qpos[self.qadr] - self.rest[self.qadr]).astype(np.float32) / 1.6

    def step(self, action, record=True):
        action = np.asarray(action)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("Expected 14 finite front-leg motor outputs")
        m, d = self.model, self.data
        # Keep the existing checkpoint layout; only left-leg outputs drive the pen.
        commands = np.where(self.drawing_joints, np.clip(action, -1, 1), 0)
        d.ctrl[self.actuators] = self.rest[self.qadr] + 1.6 * commands
        if not record:
            mj.mj_step(m, d, nstep=round(CONTROL_DT / m.opt.timestep))
            if not np.isfinite(d.qpos).all() or np.max(np.abs(d.qvel)) > 1e5:
                raise FloatingPointError("Unstable motor simulation")
            return []
        frames = []
        # Actual dynamics, 10 kHz physics; return four 100 Hz body samples per action.
        for _ in range(4):
            ink = []
            # Physics still runs at 10 kHz in MuJoCo's C loop; record ink at 100 Hz.
            mj.mj_step(m, d, nstep=round(CONTROL_DT / 4 / m.opt.timestep))
            for side, brush in enumerate(self.brushes):
                contact = next((c for c in d.contact if {int(c.geom1), int(c.geom2)} == {brush, self.board}
                                and c.dist <= 0.001), None)
                point = None if contact is None else contact.pos.copy()
                if point is not None:
                    point[2] = CANVAS_Z + 0.002
                    if self.last_ink[side] is not None and np.linalg.norm(point - self.last_ink[side]) > 0.00001:
                        ink.append([side, self.last_ink[side].round(5).tolist(), point.round(5).tolist()])
                        self.last_ink[side] = point
                    elif self.last_ink[side] is None:
                        ink.append([side, point.round(5).tolist(), point.round(5).tolist()])
                        self.last_ink[side] = point
                else:
                    self.last_ink[side] = None
            if not np.isfinite(d.qpos).all() or np.max(np.abs(d.qvel)) > 1e5:
                raise FloatingPointError("Unstable motor simulation")
            frames.append(dict(time=round(d.time, 5), positions=d.geom_xpos.round(5).tolist(),
                               rotations=d.geom_xmat.round(6).tolist(), ink=ink,
                               pen_down=[point is not None for point in self.last_ink],
                               pen_tip=(point if point is not None else d.geom_xpos[self.brushes[0]] - np.array([0,0,BRUSH_RADIUS])).round(5).tolist()))
        return frames

    def scene(self):
        m = self.model
        meshes = []
        for i in range(m.nmesh):
            v, n = m.mesh_vertadr[i], m.mesh_vertnum[i]
            f, k = m.mesh_faceadr[i], m.mesh_facenum[i]
            meshes.append(dict(vertices=m.mesh_vert[v:v+n].round(6).flatten().tolist(),
                               faces=m.mesh_face[f:f+k].flatten().tolist()))
        geoms = []
        for i in range(m.ngeom):
            color = m.geom_rgba[i].copy()
            if m.geom_matid[i] >= 0:
                color = m.mat_rgba[m.geom_matid[i]].copy()
            geoms.append(dict(name=m.geom(i).name, type=int(m.geom_type[i]), mesh=int(m.geom_dataid[i]),
                              size=m.geom_size[i].tolist(), color=color.tolist()))
        return dict(meshes=meshes, geoms=geoms, positions=self.data.geom_xpos.tolist(),
                    rotations=self.data.geom_xmat.tolist(), canvas=CANVAS, canvas_z=CANVAS_Z,
                    source="NeuroMechFly / FlyGym 2.1.0", tethered=True)


def drawing_targets(drawing, home):
    """Trace every stroke with the left foreleg; lift between strokes."""
    goals = np.array(home, copy=True)
    for stroke in drawing:
        side = 0
        points = np.array(stroke, dtype=float).T
        points = np.column_stack([0.85 + (255 - points[:, 1]) / 255 * 0.5,
                                  0.25 - points[:, 0] / 255 * 0.5,
                                  np.full(len(points), CANVAS_Z + BRUSH_RADIUS - 0.012)])
        # Rest the inactive foot away from the paper, then position and lower the brush.
        goals[1-side] = home[1-side]
        waypoints = [np.array([*points[0, :2], CANVAS_Z + 0.20]), *points,
                     np.array([*points[-1, :2], CANVAS_Z + 0.20])]
        for target in waypoints:
            start = goals[side].copy()
            n = max(2, int(np.ceil(np.linalg.norm(target - start) / 0.018)))
            for alpha in np.linspace(0, 1, n + 1)[1:]:
                goals[side] = start + alpha * (target - start)
                yield goals.copy()


if __name__ == "__main__":
    fly = PaintingFly()
    (ROOT / "web/assets").mkdir(exist_ok=True)
    (ROOT / "web/assets/fly-scene.json").write_text(json.dumps(fly.scene(), separators=(",", ":")))
    print("Exported the actual MuJoCo visual meshes to web/assets/fly-scene.json")
