"""Calibrated ports, shared gradients, physical gates and exact image resume."""
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
from flycasso.brain import Brain
from flycasso.common import digest, load_torch, write_json
from flycasso.train_calibrated import Training, SketchEncoder, goals, physical_score


def fixture(root,task='image'):
    graph=root/'graph.npz';ports=root/f'{task}.npz';groups=root/'calibration.npz';n=12
    np.savez(graph,ids=np.arange(n),indptr=np.arange(n+1),indices=(np.arange(n)-1)%n,
        values=np.full(n,.9,np.float32),signs=np.ones(n,np.int8),metadata=np.array(json.dumps({'kind':'test'})))
    inputs=np.full(n,-1,np.int32);inputs[0]=3072;scale=np.zeros(n,np.float32);scale[0]=1
    count=9216 if task=='image' else 15
    np.savez(ports,input_channel=inputs,input_scale=scale,output_index=np.full((count,1),2,np.int32),output_scale=np.ones((count,1),np.float32))
    np.savez(groups,edge_group=np.arange(n,dtype=np.int32)%3,neuron_group=np.arange(n,dtype=np.int32)%3)
    return graph,ports,groups


class CalibratedTest(unittest.TestCase):
    def test_gradients_and_resume(self):
        from flycasso.train_brain import load
        from flycasso.export import export
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);graph,ports,groups=fixture(root)
            model=Brain(graph,ports,ticks=8,calibration=groups,size=16)
            self.assertEqual(model.core.edge_gain.numel(),3)
            x=torch.zeros(1,3,16,16);label=torch.zeros(1,dtype=torch.long);clock=torch.zeros(1);cue=torch.zeros(1,3)
            out,state=model(x,label,clock,cue);out.square().mean().backward()
            for param in [model.core.edge_gain,model.cue_adapter.weight,model.readout.weight]:
                self.assertTrue(torch.isfinite(param.grad).all());self.assertGreater(param.grad.abs().sum().item(),0)
            disabled,_=model(x,label,clock,cue,ablate=True)
            self.assertEqual(disabled.abs().sum().item(),0)
            encoder=SketchEncoder();strokes=torch.randn(2,3,256);encoded,kl=encoder(strokes,torch.zeros(2,3))
            (encoded.square().mean()+kl).backward();self.assertEqual(encoded.shape,(2,3))
            data=root/'data';data.mkdir();files={}
            for split in ['train','val']:
                for name,array in [('images',np.random.default_rng(7).integers(0,256,(10,3,32,32),dtype=np.uint8)),('labels',np.arange(10,dtype=np.int64))]:
                    p=data/f'{split}_{name}.npy';np.save(p,array);files[p.name]=digest(p)
            write_json(data/'manifest.json',dict(image_size=32,files=files))
            args=Namespace(task='image',out=str(root/'full'),graph=str(graph),ports=str(ports),calibration=str(groups),data=str(data),device='cpu',updates=2,evaluator='unused')
            Training(args).run();full=load_torch(root/'full/last.pt')[0]
            args.out=str(root/'resume');args.updates=1;Training(args).run();Training(args).run()
            resumed=load_torch(root/'resume/last.pt')[0]
            for key in full['model']:torch.testing.assert_close(full['model'][key],resumed['model'][key],rtol=0,atol=0)
            trained,_=load(root/'full/last.pt','cpu');trained.guidance_scale=2
            pixels,frames=trained.sample(label,steps=4);self.assertEqual(pixels.shape,(1,3,16,16));self.assertEqual(len(frames),5)
            export(root/'full/last.pt',root/'bundle');portable,_=load(root/'bundle/model.pt','cpu');portable.guidance_scale=2
            actual,_=portable.sample(label,steps=4);torch.testing.assert_close(pixels,actual)
            (root/'bundle/calibration.npz').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'Changed calibration'):load(root/'bundle/model.pt','cpu')

    def test_physical_teacher_and_motor_training(self):
        from flycasso.muscle import MuscleFly
        for stage in [0,1,2,3]:
            fly=MuscleFly();target=goals(fly,stage);tips=[];contacts=[]
            for point in target:
                contacts.append(fly.step(fly.expert(point)));tips.append(fly.data.site_xpos[fly.site].copy())
            score=physical_score(fly,target,np.asarray(tips),contacts)
            self.assertLess(score['mean_tip_error_mm'],.013)
            self.assertGreater(score['contact_accuracy'],.9)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);graph,ports,groups=fixture(root,'motor');np.savez(root/'body.npz',fixture=[0])
            data=root/'data';data.mkdir();files={}
            for split in ['train','val']:
                p=data/f'{split}.npz';np.savez(p,strokes=np.zeros((10,3,256),np.float32),labels=np.arange(10,dtype=np.int64));files[p.name]=digest(p)
            write_json(data/'manifest.json',dict(files=files))
            args=Namespace(task='motor',out=str(root/'run'),graph=str(graph),ports=str(ports),calibration=str(groups),data=str(data),device='cpu',updates=1,evaluator='unused')
            Training(args).run();state=load_torch(root/'run/last.pt')[0]
            self.assertEqual(state['step'],32);self.assertEqual(state['teacher'],.95)
            self.assertEqual(state['stage'],0);self.assertIn('critic.0.weight',state['aux'])
            self.assertTrue(all(torch.isfinite(v).all() for v in state['model'].values()))
            # Exercise the posterior-conditioned actor/critic graph over repeated TBPTT chunks.
            trainer=Training(args);trainer.stage=5;trainer.model.stage=5
            before=trainer.aux['encoder'].net[-1].weight.detach().clone()
            loss,_,_=trainer.motor_episode()
            self.assertTrue(np.isfinite(loss))
            self.assertFalse(torch.equal(before,trainer.aux['encoder'].net[-1].weight))


if __name__=='__main__':unittest.main()
