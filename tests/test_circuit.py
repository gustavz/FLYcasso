"""Compare learned sparse edge derivatives against dense reference computation."""
import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
import torch
from flycasso.circuit import EdgeMultiply


class CircuitTest(unittest.TestCase):
    def test_ports_state_and_checkpoint(self):
        from flycasso.brain import Brain
        from flycasso.common import digest,save_torch
        from flycasso.train_brain import load
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);graph=folder/'graph.npz';ports=folder/'ports.npz';n=12
            np.savez(graph,ids=np.arange(n),indptr=np.arange(n+1),indices=(np.arange(n)-1)%n,
                values=np.full(n,.9,np.float32),signs=np.ones(n,np.int8),metadata=np.array(json.dumps({'kind':'test'})))
            inputs=np.full(n,-1,np.int32);inputs[0]=0;scale=np.zeros(n,np.float32);scale[0]=1
            np.savez(ports,input_channel=inputs,input_scale=scale,output_index=np.full((3072,1),2,np.int32),output_scale=np.ones((3072,1),np.float32))
            model=Brain(graph,ports,ticks=2)
            x=torch.ones(1,3,32,32);y=torch.zeros(1,dtype=torch.long);clock=torch.zeros(1);seed=torch.zeros(1,3)
            first,state=model(x,y,clock,seed);second,_=model(x,y,clock,seed,state)
            self.assertEqual(first.abs().max().item(),0)
            self.assertGreater(second.abs().max().item(),0)
            disabled,_=model(x,y,clock,seed,ablate=True);self.assertEqual(disabled.abs().max().item(),0)
            self.assertEqual(set(dict(model.named_parameters())),{'core.edge_gain','core.bias','core.leak'})
            cfg=dict(graph=str(graph),ports=str(ports),task='image',ticks=2,control='real',dataset='data/processed/test')
            checkpoint=folder/'model.pt';save_torch(checkpoint,dict(format='brain-first-v1',config=cfg,
                hashes=dict(graph=digest(graph),ports=digest(ports)),ema=model.state_dict(),step=1))
            restored,_=load(checkpoint,'cpu');actual,_=restored(x,y,clock,seed,state)
            torch.testing.assert_close(actual,second)
            from flycasso.export import export
            export(checkpoint,folder/'bundle');export(checkpoint,folder/'bundle')
            portable,_=load(folder/'bundle/model.pt','cpu');actual,_=portable(x,y,clock,seed,state)
            torch.testing.assert_close(actual,second)
            # Exact CPU resume through the real recurrent image trainer, including EMA.
            from argparse import Namespace
            from flycasso.train_brain import run
            from flycasso.common import load_torch,write_json
            data=folder/'data';data.mkdir();files={}
            for split in ['train','val']:
                for name,array in [('images',np.random.default_rng(7).integers(0,256,(4,3,32,32),dtype=np.uint8)),('labels',np.arange(4,dtype=np.int64))]:
                    path=data/f'{split}_{name}.npy';np.save(path,array);files[path.name]=digest(path)
            write_json(data/'manifest.json',dict(image_size=32,files=files))
            args=Namespace(out=str(folder/'full'),resume=None,graph=str(graph),ports=str(ports),data=str(data),
                ticks=2,sequence=4,batch=2,lr=.003,control='real',device='cpu',max_steps=2,min_steps=1000,
                validate_every=1,validation_batches=1)
            run(args);full=load_torch(folder/'full/last.pt')[0]
            from flycasso.sample import load_checkpoint,generate
            sampled,sampler,info=load_checkpoint(folder/'full/last.pt',device='cpu')
            images,frames,recipe=generate(sampled,sampler,'frog',1,42,4)
            self.assertEqual(images[0].size,(32,32));self.assertEqual(len(frames),5)
            args.out=str(folder/'resumed');args.max_steps=1;run(args)
            args.resume=str(folder/'resumed/last.pt');args.out=None;args.max_steps=2;run(args)
            resumed=load_torch(folder/'resumed/last.pt')[0]
            for name in ['model','ema']:
                for key,value in full[name].items():torch.testing.assert_close(value,resumed[name][key],rtol=0,atol=0)
            # The real muscle-training loop: independent physics states, teacher
            # supervision, optimizer updates, free-running validation and checkpoint.
            from flycasso.train_muscle import run as run_muscle
            from flycasso.muscle import MuscleFly
            ports_motor=folder/'motor-ports.npz'
            np.savez(ports_motor,input_channel=inputs,input_scale=scale,output_index=np.full((15,1),2,np.int32),output_scale=np.ones((15,1),np.float32))
            motor_data=folder/'motor-data';motor_data.mkdir();fly=MuscleFly();target=fly.center.copy();target[2]=fly.z+.012
            arrays=dict(images=np.full((10,256,3,32,32),255,np.uint8),proprio=np.zeros((10,256,14),np.float32),
                actions=np.full((10,256,15),.1,np.float32),labels=np.arange(10,dtype=np.int64),
                targets=np.tile(target,(10,256,1)),seeds=np.zeros((10,3),np.float32))
            files={}
            for split in ['train','val']:
                path=motor_data/f'{split}.npz';np.savez_compressed(path,**arrays);files[path.name]=digest(path)
            classes=['cat','flower','butterfly','fish','bird','tree','house','star','apple','umbrella']
            write_json(motor_data/'manifest.json',dict(classes=classes,files=files))
            args.out=str(folder/'motor-run');args.resume=None;args.data=str(motor_data);args.ports=str(ports_motor)
            args.batch=1;args.max_steps=16;run_muscle(args)
            motor=load_torch(folder/'motor-run/last.pt')[0]
            self.assertEqual(motor['step'],16);self.assertTrue(motor['config']['closed_loop'])
            self.assertTrue((folder/'motor-run/preview.png').exists())
            self.assertTrue(all(torch.isfinite(v).all() for v in motor['ema'].values()))
            (folder/'bundle/ports.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Changed ports'):load(folder/'bundle/model.pt','cpu')

    def test_muscle_contact_and_render_coordinates(self):
        from flycasso.muscle import MuscleFly
        fly=MuscleFly();marks=0
        for i in range(20):
            goal=fly.center.copy();goal[2]=fly.z+.012
            before={k:getattr(fly.data,k).copy() for k in ['qpos','qvel','act','ctrl','qacc_warmstart']}
            action=fly.expert(goal)
            for k,value in before.items():np.testing.assert_array_equal(getattr(fly.data,k),value)
            marks+=fly.step(action,record=True)
        self.assertGreater(marks,0)
        other=MuscleFly();self.assertIs(other.model,fly.model)
        np.testing.assert_array_equal(other.data.qpos[:7],other.neutral)
        self.assertIsNot(other.data,fly.data)
        scene=fly.scene();names={g['name'] for g in scene['geoms']}
        self.assertTrue({'nmf/c_head','nmf/lf_tarsus3','nmf/lf_tarsus4','nmf/lf_tarsus5','canvas','nmf/lf_brush'}<=names)
        for geom in scene['geoms']:
            if geom['type']==7:
                self.assertEqual(geom['color'][3],.25 if geom['name'].endswith('_wing') else 1.,geom['name'])
        for leg in ['LH','LM','RF','RH','RM']:
            self.assertIn(f'{leg}Tibia_geom',names)
        self.assertTrue(any('abdomen' in name for name in names))
        for frame in fly.frames:
            self.assertEqual(len(frame['positions']),len(scene['geoms']))
            for _,a,b in frame['ink']:
                self.assertAlmostEqual(b[2],scene['canvas_z']+.002)
                self.assertEqual(frame['pen_tip'],b)

    def test_learned_edges(self):
        for device in ['cpu']+(['mps'] if torch.backends.mps.is_available() else []):
            torch.manual_seed(12)
            ptr=torch.tensor([0,2,3,5],dtype=torch.int32,device=device)
            idx=torch.tensor([1,2,0,0,2],dtype=torch.int32,device=device)
            rows=torch.tensor([0,0,1,2,2],dtype=torch.int32,device=device)
            order=torch.tensor([2,3,0,1,4],dtype=torch.int32,device=device)
            tptr=torch.tensor([0,2,3,5],dtype=torch.int32,device=device)
            for batch in [1,7,32]:
                x=torch.randn(3,batch,requires_grad=True);v=torch.randn(5,requires_grad=True)
                w=torch.zeros(3,3).index_put((rows.cpu().long(),idx.cpu().long()),v)
                target=w@x;target.square().sum().backward()
                a=x.detach().to(device).requires_grad_();b=v.detach().to(device).requires_grad_()
                actual=EdgeMultiply.apply(a,b,ptr,idx,rows,tptr,rows[order.long()],b.detach()[order.long()])
                actual.square().sum().backward()
                for result,expected in [(actual.cpu(),target),(a.grad.cpu(),x.grad),(b.grad.cpu(),v.grad)]:
                    torch.testing.assert_close(result,expected,rtol=2e-5,atol=2e-5)


if __name__=='__main__': unittest.main()
