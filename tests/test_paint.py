"""Offline checks for stopping, full-circuit motor gradients and physical brush marks."""
import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np
import torch

from common import Plateau
if importlib.util.find_spec("flygym"):
    from paint import PaintingFly, CANVAS_Z, drawing_targets
from prepare import fixture
from strokes import StrokeDenoiser, vectorize, decode
from train_motor import FlyMotor, FootPosition, validation, remap_stroke_classes


class PaintingTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("flygym"), "Install requirements-paint.txt for motor physics checks")
    def test_stopping_and_motor_physics(self):
        old=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
        current=torch.zeros(4,2)
        mapped=remap_stroke_classes(current,old,['flower','new','cat'],['cat','flower'])
        torch.testing.assert_close(mapped,torch.tensor([[3.,4.],[0.,0.],[1.,2.],[5.,6.]]))
        class Optimizer:
            param_groups = [{"lr": 0.008}]
        optimizer = Optimizer(); plateau = Plateau(min_steps=10)
        for step in range(1, 10):
            self.assertFalse(plateau.update(1.0, step, optimizer))
        self.assertTrue(Plateau(plateau.state).update(1.0, 10, optimizer))
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.001)
        self.assertFalse(plateau.update(0.5, 11, optimizer))
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory) / "fixture")
            model = FlyMotor(config["graph"], config)
            x = torch.randn(2, 26, requires_grad=True)
            y = model(x)
            self.assertEqual(y.shape, (2, 14))
            targets = y.detach().clone(); targets[:, 7:] += 100
            self.assertEqual(validation(model, x, targets), 0)
            y.square().mean().backward()
            self.assertGreater(x.grad.abs().sum().item(), 0)
            self.assertFalse(torch.equal(y, model(x, ablate_edges=True)))
            self.assertEqual(model.n_neurons, 64)
            residual=FlyMotor(config["graph"],dict(config,observations=40,motor_residual=True))
            obs=torch.randn(2,40)*.1
            command=residual(obs)
            self.assertLessEqual(float((command-obs[:,:14]).abs().max().detach()),.050001)
            self.assertTrue(torch.equal(command,obs[:,:14]))
            with torch.no_grad(): residual.motor_output.weight.fill_(.01)
            command=residual(obs)
            command.square().mean().backward()
            self.assertGreater(residual.motor_input.weight.grad.abs().sum().item(),0)
            generator=FlyMotor(config["graph"],dict(config,stroke_diffusion=True,stroke_architecture="unet",stroke_scales=5,classes=["cat","flower","butterfly"]))
            noisy=torch.randn(2,3,16,16,requires_grad=True)
            strokes=StrokeDenoiser(generator)(noisy,torch.tensor([2,5]),torch.tensor([0,1]))
            strokes.square().mean().backward()
            self.assertGreater(generator.stroke_class.weight.grad.abs().sum().item(),0)
            self.assertEqual(strokes.shape,noisy.shape)
            reference=[[[0,255],[0,255]],[[255,0],[0,255]]]
            from prepare_images import rasterize
            raster=rasterize(reference)
            self.assertEqual(raster.shape,(3,32,32))
            self.assertLess(raster.min(),64)
            self.assertEqual(int(raster[:,0,0].min()),255)
            restored=decode(vectorize(reference))
            self.assertEqual(len(restored),2)
            np.testing.assert_allclose([s[0][0] for s in restored],[0,255])
        fly = PaintingFly()
        self.assertEqual(len(fly.qadr), 14)
        np.testing.assert_array_equal(fly.drawing_joints, [True]*7 + [False]*7)
        self.assertEqual(len(fly.brushes), 1)
        drawing = [[[0, 255], [0, 255]], [[255, 0], [0, 255]]]
        for goal in drawing_targets(drawing, fly.home):
            np.testing.assert_array_equal(goal[1], fly.home[1])
        # Even arbitrary right-leg network outputs cannot command the resting leg.
        fly.step(np.ones(14))
        np.testing.assert_array_equal(fly.data.ctrl[fly.actuators[~fly.drawing_joints]], fly.rest[fly.qadr[~fly.drawing_joints]])
        fly.reset()
        action = torch.zeros(1,14,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a: FootPosition.apply(a, fly), (action,), atol=1e-5))
        fly.reset(); fly.step(np.zeros(14)); recorded = fly.data.qpos.copy()
        fly.reset(); fly.step(np.zeros(14), record=False)
        np.testing.assert_allclose(fly.data.qpos, recorded, atol=1e-12)
        fly.reset()
        goals = fly.home.copy(); goals[0] = [1.1, 0, CANVAS_Z + .013]
        action = fly.expert(goals)
        self.assertLess(np.max(np.linalg.norm(goals - fly.data.site_xpos[fly.sites], axis=1)), .002)
        fly.reset()
        # An actual actuator-driven transition should move the foot and produce contact ink.
        ink = []
        for _ in range(15):
            frames=fly.step(action)
            for frame in frames:
                for mark in frame["ink"]:
                    np.testing.assert_array_equal(frame["pen_tip"],mark[2])
                    self.assertAlmostEqual(frame["pen_tip"][2],CANVAS_Z+.002)
            ink.extend(mark for frame in frames for mark in frame["ink"])
        self.assertGreater(len(ink), 0)
        self.assertTrue(any(a == b for _, a, b in ink), "First contact must leave a dot")
        self.assertLess(np.linalg.norm(fly.data.site_xpos[fly.sites[0]][:2] - goals[0, :2]), .02)
        self.assertTrue(all(mark[0] == 0 for mark in ink))
        self.assertTrue(all(abs(mark[2][2] - (CANVAS_Z+.002)) < 1e-5 for mark in ink))
        # A lifted foot cannot paint by merely having an associated target path.
        goals[0, 2] = CANVAS_Z + .25
        lift = fly.expert(goals)
        for _ in range(15):
            frames = fly.step(lift)
        self.assertEqual(sum(len(frame["ink"]) for frame in frames), 0)
        self.assertFalse(frames[-1]["pen_down"][0])


if __name__ == "__main__":
    unittest.main()
