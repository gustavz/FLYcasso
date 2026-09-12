"""Optional real Apple GPU parity test: python -m unittest discover -s tests -p test_metal.py."""
import json
import numpy as np
import tempfile
import unittest
from pathlib import Path
import torch
from model import FlyDenoiser
from common import device_for
from prepare import fixture
from metal import MetalSparseMultiply, graph_arrays


@unittest.skipUnless(torch.backends.mps.is_available(), "Apple GPU unavailable")
class MetalTest(unittest.TestCase):
    def test_graph_model_and_backward(self):
        self.assertEqual(device_for("mps:0").type,"mps")
        torch.manual_seed(42)
        w=torch.tensor([[0.,0.,0.],[.2,0.,-.3],[0.,.7,0.]]).to_sparse_csr()
        graphs=graph_arrays(w),graph_arrays(w.t().to_sparse_csr())
        for batch in (1,4,13,32,33,64):
            x=torch.randn(3,batch,requires_grad=True);gpu=x.detach().to('mps').requires_grad_()
            expected=torch.sparse.mm(w,x);actual=MetalSparseMultiply.apply(gpu,*graphs)
            torch.testing.assert_close(actual.cpu(),expected,rtol=1e-5,atol=1e-6)
            expected.square().sum().backward();actual.square().sum().backward()
            torch.testing.assert_close(gpu.grad.cpu(),x.grad,rtol=1e-5,atol=1e-6)
        with tempfile.TemporaryDirectory() as folder:
            config=fixture(Path(folder)/'fixture');model=FlyDenoiser(config['graph'],config)
            image=torch.randn(3,3,8,8);t=torch.tensor([1,2,3]);labels=torch.tensor([0,1,2])
            expected=model(image,t,labels);expected.square().mean().backward()
            gradients={k:p.grad.clone() for k,p in model.named_parameters()}
            keys=set(model.state_dict());model.zero_grad(set_to_none=True);model.to('mps')
            actual=model(image.to('mps'),t.to('mps'),labels.to('mps'));actual.square().mean().backward()
            torch.testing.assert_close(actual.cpu(),expected,rtol=1e-4,atol=2e-5)
            for k,p in model.named_parameters():torch.testing.assert_close(p.grad.cpu(),gradients[k],rtol=5e-4,atol=2e-6)
            assert set(model.state_dict())==keys
            model.cpu();torch.testing.assert_close(model(image,t,labels),expected,rtol=0,atol=0)
            refined=FlyDenoiser(config['graph'],dict(config,image_size=32,architecture='unet',residual_blocks=True,synaptic_output=True,condition_dropout=.1,guidance_scale=2.))
            pixels=torch.randn(3,3,32,32)
            output=refined(pixels,t,labels);output.square().mean().backward()
            gradients={k:p.grad.clone() for k,p in refined.named_parameters()}
            refined.zero_grad(set_to_none=True);refined.to('mps')
            actual=refined(pixels.to('mps'),t.to('mps'),labels.to('mps'));actual.square().mean().backward()
            torch.testing.assert_close(actual.cpu(),output,rtol=1e-4,atol=2e-5)
            for k,p in refined.named_parameters():torch.testing.assert_close(p.grad.cpu(),gradients[k],rtol=1e-3,atol=3e-6)

    def test_large_projection_gradient(self):
        # The tiny graph cannot expose the MPS 166,700-term input-gradient failure.
        with tempfile.TemporaryDirectory() as folder:
            config=fixture(Path(folder)/'fixture');n=166700;graph=Path(folder)/'large.npz'
            np.savez(graph,ids=np.arange(n,dtype=np.int64),indptr=np.arange(n+1,dtype=np.int32),
                     indices=np.arange(n,dtype=np.int32),values=np.full(n,.3,dtype=np.float32),
                     signs=np.ones(n,dtype=np.int8),metadata=np.array(json.dumps({'kind':'synthetic_fixture'})))
            config.update(graph=str(graph),width=64,recurrent_steps=2,synaptic_output=True)
            model=FlyDenoiser(graph,config);image=torch.randn(3,3,8,8);t=torch.tensor([1,2,3]);labels=torch.tensor([0,1,2])
            model(image,t,labels).square().mean().backward()
            expected={k:p.grad.clone() for k,p in model.named_parameters()}
            model.zero_grad(set_to_none=True);model.to('mps')
            model(image.to('mps'),t.to('mps'),labels.to('mps')).square().mean().backward()
            for k,p in model.named_parameters():
                error=(p.grad.cpu()-expected[k]).norm()/expected[k].norm().clamp_min(1e-12)
                self.assertLess(error.item(),.001,k)


if __name__=='__main__':unittest.main()
