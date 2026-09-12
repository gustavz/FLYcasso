"""One offline integration check: preparation, gradients, resume, sampling and HTTP."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch

from app import encode_images, make_server
from common import digest, load_torch, save_torch, write_json
from diffusion import Diffusion
from export import export
from evaluate import evaluate
from model import FixedSparseMultiply, FlyDenoiser
from prepare import exact_ids, fixture, index_edges, prepare_connectome
from sample import generate, load_checkpoint, save_samples, to_images
from train import run, validate_config


class PipelineTest(unittest.TestCase):
    def test_run_scripts(self):
        # Check orchestration separately from the real training/export integration below.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher = root / "python"
            launcher.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ["CALL_LOG"]).open("a") as log:
    log.write(json.dumps(args) + "\\n")
if args[0] == "-c":
    if "status.json" in args[1]: sys.exit(int(os.environ["INCOMPLETE"]))
    print(os.environ["REMAINING"] if "max(0," in args[1] else "checkpoint-hash")
''')
            launcher.chmod(0o755)
            for scenario in ("fresh", "resume", "complete"):
                folder = root / scenario; folder.mkdir()
                for name in ("run.sh", "run_motor.sh"):
                    shutil.copyfile(Path(__file__).resolve().parents[1] / name, folder / name)
                if scenario != "fresh":
                    for name in ("image run", "motor run", "motor run-control"):
                        run = folder / name; run.mkdir(); (run / "last.pt").touch()
                log = folder / "calls.jsonl"
                env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"], CALL_LOG=str(log),
                           INCOMPLETE="0" if scenario == "complete" else "1", REMAINING="0" if scenario == "complete" else "25")
                subprocess.run(["bash", str(folder / "run.sh"), "configs/full.json", "image run"], env=env, check=True)
                subprocess.run(["bash", str(folder / "run_motor.sh"), "motor run"], env=env, check=True)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                training = [args for args in calls if args[0] in ("train.py", "train_motor.py")]
                self.assertEqual(len(training), {"fresh": 3, "resume": 2, "complete": 0}[scenario])
                self.assertTrue(all(("--resume" in args) == (scenario == "resume") for args in training))
                self.assertEqual(sum(args[0] == "export.py" for args in calls), 2)
                self.assertFalse(any(args[0] == "app.py" for args in calls))

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from unittest.mock import patch
            checkpoint=root/'changing.pt';save_torch(checkpoint,{'step':1});expected_hash=digest(checkpoint)
            original_load=torch.load
            def replace_after_load(*args,**kwargs):
                state=original_load(*args,**kwargs);save_torch(checkpoint,{'step':2});return state
            with patch('torch.load',side_effect=replace_after_load):
                state,checksum=load_torch(checkpoint)
            self.assertEqual((state['step'],checksum),(1,expected_hash))
            config = fixture(root / "fixture")
            config_path = root / "fixture/config.json"
            with self.assertRaises(ValueError):
                validate_config(dict(config, recurrent_steps=1))

            # Large biological IDs must not pass through a float; missing endpoints drop only their edges.
            ids = np.array([2**53 + 1, 2**53 + 3], dtype=np.int64)
            a, b, counts, keep = index_edges(ids, ids, [ids[1], 8], [2, 3])
            np.testing.assert_array_equal(keep, [True, False])
            np.testing.assert_array_equal((a, b, counts), ([0], [1], [2]))
            with self.assertRaises(ValueError):
                exact_ids(ids.astype(float))

            # Exercise actual upstream Feather schemas and the Glia/unassigned filter.
            import pandas as pd
            import pyarrow.feather as feather
            raw = root / "raw"
            raw.mkdir()
            for name, frame in {
                "annotations": pd.DataFrame({"bodyId": [1, 2, 3, 4], "superclass": ["cb_intrinsic", "vnc_motor", "cb_intrinsic", None], "status": ["Traced", "Untraced", "Glia", "Traced"]}),
                "neurotransmitters": pd.DataFrame({"body": [1, 2], "consensus_nt": ["gaba", "acetylcholine"]}),
                "edges": pd.DataFrame({"body_pre": [1, 2, 1, 3, 4], "body_post": [2, 1, 1, 1, 1], "weight": [1, 2, 3, 4, 5]}),
            }.items():
                feather.write_feather(frame, raw / f"{name}.feather")
            report = prepare_connectome(raw, root / "prepared", verify=False)
            self.assertEqual((report["nodes"], report["edges"], report["retained_contact_count"]), (2, 3, 6))
            self.assertEqual(report["audit"]["excluded_contact_count"], 9)

            model = FlyDenoiser(config["graph"], config)
            x = torch.randn(model.n_neurons, 3, requires_grad=True)
            sparse = FixedSparseMultiply.apply(x, model.w, model.wt)
            dense = model.w.to_dense() @ x
            torch.testing.assert_close(sparse, dense)
            gs = torch.autograd.grad(sparse.square().sum(), x)[0]
            gd = torch.autograd.grad(dense.square().sum(), x)[0]
            torch.testing.assert_close(gs, gd)
            # Large negative readout offsets must not turn the entire hidden layer off.
            image = torch.randn(2, 3, 8, 8, requires_grad=True)
            hook = model.neurons_to_output.register_forward_hook(lambda module, args, out: out - 50)
            model(image, torch.tensor([1, 2]), torch.tensor([0, 1])).square().mean().backward()
            hook.remove()
            self.assertGreater(image.grad.abs().sum().item(), 0.001)

            spatial=FlyDenoiser(config["graph"],dict(config,image_size=32,architecture="unet",classes=["cat","flower","butterfly"]))
            pixels=torch.randn(2,3,32,32,requires_grad=True)
            spatial(pixels,torch.tensor([2,5]),torch.tensor([0,1])).square().mean().backward()
            self.assertGreater(spatial.class_input.weight.grad.abs().sum().item(),0)
            self.assertGreater(pixels.grad.abs().sum().item(),0)
            refined=FlyDenoiser(config["graph"],dict(config,image_size=32,architecture="unet",residual_blocks=True,classes=spatial.classes))
            refined.load_state_dict(spatial.state_dict(),strict=False)
            args=(pixels,torch.tensor([2,5]),torch.tensor([0,1]))
            self.assertTrue(torch.equal(spatial(*args),refined(*args)))
            refined(*args).square().mean().backward()
            self.assertGreater(refined.decoder_refine[0].net[-1].weight.grad.abs().sum().item(),0)
            self.assertGreater(refined.spatial_attention.attention.out_proj.weight.grad.abs().sum().item(),0)
            connected=FlyDenoiser(config['graph'],dict(config,image_size=32,architecture='unet',synaptic_output=True))
            shared_pixels=torch.randn(1,3,32,32).expand(2,-1,-1,-1);same_time=torch.ones(2,dtype=torch.long);different_labels=torch.tensor([0,1])
            intact=connected(shared_pixels,same_time,different_labels)
            disconnected=connected(shared_pixels,same_time,different_labels,ablate_edges=True)
            self.assertFalse(torch.equal(intact[0],intact[1]))
            self.assertTrue(torch.equal(disconnected[0],disconnected[1]))



            class KnownCleanImage(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.anchor = torch.nn.Parameter(torch.zeros(()))
                    self.size = 8
                def forward(self, image, time, label, ablate_edges=False):
                    return torch.full_like(image, 0.25) + self.anchor
            known, _ = Diffusion(32).sample(KnownCleanImage(), torch.tensor([0]), 42, 4)
            torch.testing.assert_close(known, torch.full_like(known, 0.25))
            class Guided(KnownCleanImage):
                null_label=3;guidance_scale=2.
                def forward(self,image,time,label,ablate_edges=False):
                    return torch.ones_like(image)*(label!=self.null_label)[:,None,None,None]*.2+self.anchor
            guided,_=Diffusion(32).sample(Guided(),torch.tensor([0,1]),42,4)
            torch.testing.assert_close(guided,torch.full_like(guided,.4))


            whole = run(config_path, root / "whole", device_name="cpu")
            partial = run(config_path, root / "resumed", device_name="cpu", stop_after=4)
            resumed = run(resume=partial, device_name="cpu")
            first = torch.load(whole, weights_only=True)
            second = torch.load(resumed, weights_only=True)
            for group in ("model", "ema"):
                for key in first[group]:
                    torch.testing.assert_close(first[group][key], second[group][key], rtol=0, atol=0)
            model, diffusion, info = load_checkpoint(whole, device="cpu")
            images, frames, metadata = generate(model, diffusion, "cat", 1, 42, 4)
            again, _, _ = generate(model, diffusion, "cat", 1, 42, 4)
            self.assertEqual(images[0].tobytes(), again[0].tobytes())
            self.assertEqual(images[0].size, (8, 8))
            save_samples(root / "samples", images, frames, dict(info, **metadata))
            self.assertTrue((root / "samples/denoising.gif").is_file())
            export(whole, root / "bundle")
            export(whole, root / "bundle")  # Restarting the pipeline preserves a verified export.
            portable, schedule, _ = load_checkpoint(root / "bundle/model.pt", device="cpu")
            exported, _, _ = generate(portable, schedule, "cat", 1, 42, 4)
            self.assertEqual(images[0].tobytes(), exported[0].tobytes())
            export(root / "bundle/model.pt", root / "bundle-copy")
            (root / "bundle/LICENSE").write_text("corrupt")
            with self.assertRaises(ValueError):
                export(whole, root / "bundle")
            result = evaluate(whole, device="cpu", batches=2, batch_size=2)
            self.assertTrue(np.isfinite(result["results"]["intact"]["clean_image_mse"]))
            self.assertNotEqual(result["results"]["intact"]["clean_image_mse"], result["results"]["edges_disabled"]["clean_image_mse"])
            with self.assertRaises(ValueError):
                generate(model, diffusion, "cat", 100, 42, 4)
            config["allow_synthetic"] = False
            write_json(config_path, config)
            with self.assertRaises(ValueError):
                run(config_path, root / "must-fail", device_name="cpu")

            followed = root / "custom-best.pt"
            save_torch(followed, dict(first, step=first["step"] + 1))
            server = make_server(model, diffusion, info, port=0, allowed_host="fly.example.ts.net:8443",
                                 follow_training=True, checkpoint=followed)
            # Keep the completed first handler alive while the next request starts.
            # Readiness must precede its terminal reply, not depend on thread timing.
            finish_reply = threading.Event()
            original_reply = server.RequestHandlerClass.reply
            def delayed_return(handler, status, *args):
                original_reply(handler, status, *args)
                if handler.path == "/api/generate" and status == 200:
                    finish_reply.wait(5)
            server.RequestHandlerClass.reply = delayed_return
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                with urllib.request.urlopen(url) as response:
                    page = response.read().decode()
                token = re.search(r"'X-Flycasso-Token':'([^']+)'", page).group(1)
                proxy = urllib.request.Request(url + "/api/info", headers={"Host": "fly.example.ts.net:8443"})
                with urllib.request.urlopen(proxy) as response:
                    self.assertEqual(json.load(response)["neurons"], model.n_neurons)
                for asset in ("flycasso-flies.png", "flycasso-wordmark.png"):
                    with urllib.request.urlopen(url + "/assets/" + asset) as response:
                        self.assertEqual(response.headers["Content-Type"], "image/png")
                        self.assertEqual(response.read(8), b"\x89PNG\r\n\x1a\n")
                for asset in ("three.module.js", "three.core.js", "OrbitControls.js"):
                    with urllib.request.urlopen(url + "/assets/" + asset) as response:
                        self.assertEqual(response.headers["Content-Type"], "text/javascript")
                        self.assertEqual(response.read(), (Path(__file__).resolve().parents[1] / "web/vendor" / asset).read_bytes())
                for host in ("evil.example", "fly.example.ts.net:8443.evil.example"):
                    spoofed = urllib.request.Request(url, headers={"Host": host})
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        urllib.request.urlopen(spoofed)
                    self.assertEqual(error.exception.code, 403)
                body = json.dumps(dict(category="cat", count=1, seed=42, steps=4)).encode()
                request = urllib.request.Request(url + "/api/generate", data=body,
                    headers={"Content-Type": "application/json", "X-Flycasso-Token": token, "Host": "fly.example.ts.net:8443"})
                with urllib.request.urlopen(request) as response:
                    generated = json.load(response)
                    self.assertTrue(generated["images"][0].startswith("data:image/png;base64,"))
                self.assertEqual(info["training_step"], first["step"] + 1)
                stream = urllib.request.Request(url + "/api/diffuse", data=body, headers=request.headers)
                with urllib.request.urlopen(stream) as response:
                    self.assertEqual(response.headers["Content-Type"], "application/x-ndjson")
                    events = [json.loads(line) for line in response]
                self.assertEqual([event["step"] for event in events[:-1]], list(range(5)))
                for event, frame in zip(events[:-1], frames):
                    self.assertEqual(event["images"], encode_images(to_images(frame)))
                self.assertEqual(events[-1]["type"], "done")
                self.assertEqual(events[-1]["images"], generated["images"])
                finish_reply.set()
                invalid = urllib.request.Request(url + "/api/diffuse",
                    data=json.dumps(dict(category="cat", count=1, seed=42, steps=1)).encode(), headers=request.headers)
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(invalid)
                self.assertEqual(error.exception.code, 400)
                bad = urllib.request.Request(url + "/api/generate", data=body, headers={"Content-Type": "application/json"})
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(bad)
                self.assertEqual(error.exception.code, 403)
            finally:
                finish_reply.set()
                server.shutdown()
                server.server_close()
                worker.join()


if __name__ == "__main__":
    unittest.main()
