"""Local image studio. Python stdlib server + one HTML file; no frontend build.

Run: python -m flycasso.app --checkpoint runs/diffusion/last.pt
Optionally share privately through Tailscale Serve with --allowed-host.
This is a research app, not a public internet service.
"""

import argparse
import base64
import gzip
import io
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from flycasso.sample import generate, load_checkpoint, to_images
from flycasso.common import ROOT, load_torch, read_json


def encode_images(images):
    encoded = []
    for im in images:
        buffer = io.BytesIO()
        im.save(buffer, format="PNG")
        encoded.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
    return encoded


def make_server(model, diffusion, info, port=7860, allowed_host=None, follow_training=False, motor_checkpoint=None, checkpoint=None):
    token = secrets.token_urlsafe(32)
    page = (ROOT / "web/studio.html").read_text().replace("__TOKEN__", token).encode()
    # ponytail: one inference at a time; a GPU worker queue only if multi-user serving is needed.
    busy = threading.Lock()
    root = ROOT
    diffusion_path = Path(checkpoint) if checkpoint else root / "runs/diffusion/best.pt"
    motor_path = Path(motor_checkpoint) if motor_checkpoint else root / "runs/motor/best.pt"
    if motor_checkpoint and not (motor_path.parent/'config.json').exists():
        raise ValueError('The motor checkpoint needs its adjacent config.json; start training or use an exported bundle')
    motor_cache = {}
    diffusion_stamp = None

    def refresh_diffusion():
        nonlocal model, diffusion, diffusion_stamp
        path = diffusion_path
        if follow_training and path.exists() and path.stat().st_mtime_ns != diffusion_stamp:
            stamp = path.stat().st_mtime_ns
            state, checksum = load_torch(path)
            if state.get("graph_sha256",state.get('hashes',{}).get('graph')) != info["graph_sha256"]:
                raise ValueError("New image checkpoint uses a different graph")
            if state["config"] != info["config"]:
                model, diffusion, refreshed = load_checkpoint(path, device=str(next(model.parameters()).device))
                info.clear(); info.update(refreshed)
            else:
                model.load_state_dict(state["ema"])
                info.update(training_step=state["step"], checkpoint_sha256=checksum, weights="ema")
            diffusion_stamp = stamp

    class Handler(BaseHTTPRequestHandler):
        streaming = False
        worker_owned = False

        def release_worker(self):
            if self.worker_owned:
                self.worker_owned = False
                busy.release()

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def reply(self, status, data, content_type="application/json"):
            body = data if isinstance(data, bytes) else json.dumps(data, allow_nan=False).encode()
            compressed = len(body) > 8192 and "gzip" in self.headers.get("Accept-Encoding", "")
            if compressed:
                body = gzip.compress(body)
            # A client may send its next request as soon as it receives this reply.
            self.release_worker()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            if compressed:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def local_request(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if allowed_host:
                allowed.add(allowed_host)
            return self.headers.get("Host") in allowed

        def event(self, value):
            # Start only after validation; subsequent failures must stay in the stream.
            if not self.streaming:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.end_headers()
                self.streaming = True
            body = json.dumps(value, allow_nan=False, separators=(",", ":")).encode() + b"\n"
            if value.get("type") in ("done", "error"):
                self.release_worker()
            self.wfile.write(body)
            self.wfile.flush()

        def fail(self, code, message):
            if self.streaming:
                try:
                    self.event(dict(type="error", error=message))
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass
            else:
                self.reply(code, {"error": message})

        def do_GET(self):
            if not self.local_request():
                return self.reply(403, {"error": "Local requests only"})
            if self.path == "/":
                return self.reply(200, page, "text/html; charset=utf-8")
            if self.path == "/api/info":
                classes=[model.classes[i] for i in [0,1,6]] if getattr(model,'calibrated',False) and model.stage<2 else model.classes
                return self.reply(200, dict(info, classes=classes, diffusion_steps=diffusion.steps))
            if self.path == "/api/training":
                result = {}
                for task in ("diffusion", "motor"):
                    folder = diffusion_path.parent if task=='diffusion' else motor_path.parent
                    if task == "motor" and motor_checkpoint is None and not folder.exists(): folder = root / "runs/motor-control"
                    rows = []
                    metrics = folder / "metrics.jsonl"
                    if metrics.exists():
                        for line in metrics.read_text().splitlines():
                            try:
                                rows.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass  # A writer may be halfway through the latest line.
                    status = read_json(folder / "status.json") if (folder / "status.json").exists() else {}
                    rows = sorted({r["step"]: r for r in rows}.values(), key=lambda r: r["step"])
                    config = read_json(folder / 'config.json') if (folder / 'config.json').exists() else {}
                    provenance = read_json(folder / 'provenance.json') if (folder / 'provenance.json').exists() else {}
                    for row in rows:
                        row.setdefault('device', info['device'])
                    result[task] = dict(status=status, metrics=rows[-3000:], config=config, provenance=provenance)
                    control = root / "runs/motor-control/status.json"
                    if task == "motor" and not config.get("task") and control.exists(): result[task]["control"] = read_json(control)
                return self.reply(200, result)
            if self.path == "/api/paint-info":
                config_path = motor_path.parent / "config.json"
                config = read_json(config_path) if config_path.exists() else {}
                classes=config.get('classes',['cat','flower','butterfly'])
                if config.get('recipe')=='calibrated-v1' and config.get('stage',0)<7:
                    classes=[classes[i] for i in ([0] if config.get('stage',0)<6 else [0,6,7])]
                return self.reply(200, dict(classes=classes,
                    checkpoint_ready=motor_path.exists() and (config.get("phase")=="strokes" or config.get('task')=='motor'),
                    stage_name=config.get("stage_name"),muscle_driven=config.get('task')=='motor',description="Category-conditioned one-leg control"))
            if self.path=='/assets/motor-scene.json':
                config_path=motor_path.parent/'config.json'
                if config_path.exists() and read_json(config_path).get('task')=='motor':
                    from flycasso.muscle import MuscleFly
                    return self.reply(200,MuscleFly(full_body=True).scene())
                return self.reply(200,(root/'web/assets/fly-scene.json').read_bytes(),'application/json')
            if self.path.split("?",1)[0] in ("/diffusion-preview.png", "/motor-preview.png"):
                task=self.path.split("-",1)[0][1:]
                path=(diffusion_path.parent if task=='diffusion' else motor_path.parent)/'preview.png'
                if path.exists(): return self.reply(200,path.read_bytes(),"image/png")
                return self.reply(404,{"error":"No evaluated samples yet"})
            assets = {"/assets/flycasso-wordmark.png": (root / "web/assets/flycasso-wordmark.png", "image/png"),
                      "/assets/flycasso-flies.png": (root / "web/assets/flycasso-flies.png", "image/png"),
                      "/sound.js": (root / "web/sound.js", "text/javascript"),
                      "/paint.js": (root / "web/paint.js", "text/javascript"),
                      "/brain.js": (root / "web/brain.js", "text/javascript"),
                      "/assets/calibrated-image-architecture.svg": (root / "docs/assets/calibrated-image-architecture.svg", "image/svg+xml"),
                      "/assets/calibrated-motor-architecture.svg": (root / "docs/assets/calibrated-motor-architecture.svg", "image/svg+xml"),
                      "/fly-body.js": (root / "web/fly-body.js", "text/javascript"),
                      "/training.js": (root / "web/training.js", "text/javascript")}
            for task in ("image", "motor"):
                name=f"circuit-{task}-architecture.svg"
                assets["/assets/"+name]=(root/"docs/assets"/name,"image/svg+xml")
            for name in ("three.module.js", "three.core.js", "OrbitControls.js"):
                assets["/assets/" + name] = (root / "web/vendor" / name, "text/javascript")
            for name in ("fly-scene.json", "grooming.json"):
                assets["/assets/" + name] = (root / "web/assets" / name, "application/json")
            if self.path in assets:
                path, mime = assets[self.path]
                if path.exists():
                    return self.reply(200, path.read_bytes(), mime)
            self.reply(404, {"error": "Not found"})

        def paint_stream(self, request):
            import numpy as np
            import torch
            from flycasso.paint import PaintingFly, drawing_targets
            from flycasso.train_motor import load_motor
            if not isinstance(request, dict) or set(request) != {"category", "seed"} or not isinstance(request["category"], str):
                raise ValueError("Expected category and seed")
            if type(request["seed"]) is not int or not 0 <= request["seed"] < 2**63:
                raise ValueError("Invalid drawing seed")
            path = motor_path
            if not path.exists():
                return self.reply(503, {"error": "The motor model is training its first checkpoint. Try again shortly."})
            config_path=path.parent/'config.json'
            if config_path.exists() and read_json(config_path).get('task')=='motor':
                return self.muscle_stream(request)
            stamp = path.stat().st_mtime_ns
            if motor_cache.get("stamp") != stamp:
                net, metadata = load_motor(path, device=str(next(model.parameters()).device))
                motor_cache.update(model=net, metadata=metadata, stamp=stamp)
            if motor_cache["metadata"].get("phase") != "strokes":
                return self.reply(503, {"error": "The painting model is still learning category generation."})
            if request["category"] not in motor_cache["model"].classes:
                raise ValueError("Unknown category")
            fly = PaintingFly()
            try:
                self.event(dict(type="planning", category=request["category"]))
                from flycasso.strokes import generate as generate_strokes
                planning_started = time.monotonic()
                drawing = generate_strokes(motor_cache["model"], request["category"], request["seed"])
                targets = list(drawing_targets(drawing, fly.home))
                self.event(dict(type="start", **motor_cache["metadata"], category=request["category"], seed=request["seed"],
                                planning_seconds=time.monotonic()-planning_started, actions=len(targets)))
                started = time.monotonic()
                with torch.inference_mode():
                    for i, goal in enumerate(targets):
                        net = motor_cache["model"]
                        action = net(torch.from_numpy(fly.observe(goal))[None].to(next(net.parameters()).device))[0].cpu().numpy()
                        frames = fly.step(action)
                        error = float(np.linalg.norm(goal[0] - fly.data.site_xpos[fly.sites[0]]))
                        self.event(dict(type="frames", frames=frames, progress=(i+1)/len(targets),
                                   foot_error_mm=error, wall_seconds=time.monotonic()-started))
                self.event(dict(type="done", simulation_seconds=fly.data.time, wall_seconds=time.monotonic()-started))
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass  # Closing the phone tab stops its simulation and releases the worker.
            except Exception as error:
                self.log_error("Painting failed: %s", error)
                self.fail(500, "Painting stopped because the simulation failed.")

        def muscle_stream(self, request):
            import numpy as np
            import torch
            from flycasso.train_brain import load
            from flycasso.muscle import MuscleFly
            from flycasso.body import BodyReadout
            stamp=motor_path.stat().st_mtime_ns
            if motor_cache.get('stamp')!=stamp:
                net,state=load(motor_path,str(next(model.parameters()).device))
                if net.task!='motor':raise ValueError('Select a muscle-control checkpoint')
                ports=Path(state['config'].get('body_ports',str(Path(state['config']['ports']).parent/'body.npz')))
                if not ports.is_absolute() and (motor_path.parent/ports).exists():ports=motor_path.parent/ports
                readout=BodyReadout(ports,state['hashes']['graph'],net.n_neurons,next(net.parameters()).device)
                motor_cache.update(model=net,body=readout,state={'step':state['step']},stamp=stamp)
            net=motor_cache['model'];state=motor_cache['state'];device=next(net.parameters()).device
            if request['category'] not in net.classes:raise ValueError('Unknown category')
            fly=MuscleFly(full_body=True);hidden=None;started=time.monotonic();steps=256
            label=torch.tensor([net.classes.index(request['category'])],device=device)
            cue=torch.tensor(np.random.default_rng(request['seed']).normal(size=(1,3)),dtype=torch.float32,device=device).tanh()
            self.event(dict(type='start',training_step=state['step'],neurons=net.n_neurons,device=str(device),muscle_driven=True,
                            peripheral_motion=True,peripheral_joints=motor_cache['body'].mapping))
            with torch.inference_mode():
                coefficients=net.core.coefficients()
                for i in range(steps):
                    im,pr=fly.observe()
                    action,hidden=net(torch.from_numpy(im)[None].to(device),label,torch.tensor([i/(steps-1)],device=device),
                        cue,hidden,torch.from_numpy(pr)[None].to(device),coefficients=coefficients)
                    fly.step(action[0].cpu().numpy(),record=True,body_action=motor_cache['body'](hidden)[0].cpu().numpy())
                    self.event(dict(type='frames',frames=fly.frames,progress=(i+1)/steps,foot_error_mm=None,
                        wall_seconds=time.monotonic()-started))
            self.event(dict(type='done',simulation_seconds=float(fly.data.time),wall_seconds=time.monotonic()-started))

        def do_POST(self):
            if not self.local_request() or not secrets.compare_digest(self.headers.get("X-Flycasso-Token", ""), token):
                return self.reply(403, {"error": "Reload the studio and retry"})
            if self.path not in ("/api/generate", "/api/diffuse", "/api/paint"):
                return self.reply(404, {"error": "Not found"})
            if self.headers.get("Content-Type") != "application/json":
                return self.reply(415, {"error": "Expected JSON"})
            if not busy.acquire(blocking=False):
                return self.reply(429, {"error": "The fly is already painting. Try again when it finishes."})
            self.worker_owned = True
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 2048:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                if self.path == "/api/paint":
                    return self.paint_stream(request)
                if not isinstance(request, dict) or set(request) != {"category", "count", "seed", "steps"}:
                    raise ValueError("Expected category, count, seed and steps")
                started = time.monotonic()
                refresh_diffusion()
                def progress(step, frame):
                    self.event(dict(type="frame", step=step, total=request["steps"],
                                    images=encode_images(to_images(frame))))

                streamed = self.path == "/api/diffuse"
                images, _, metadata = generate(model, diffusion, **request, on_step=progress if streamed else None)
                result = dict(images=encode_images(images), metadata=dict(info, **metadata),
                              seconds=round(time.monotonic() - started, 2))
                if streamed:
                    self.event(dict(type="done", **result))
                else:
                    self.reply(200, result)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass  # Disconnecting cancels at the next denoising step or body frame.
            except (ValueError, TypeError, KeyError) as e:
                self.fail(400, str(e))
            except Exception as e:
                self.log_error("Generation failed: %s", e)
                self.fail(500, "Generation failed. See the server terminal for details.")
            finally:
                self.release_worker()

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/diffusion/last.pt")
    parser.add_argument("--graph")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--allowed-host", help="Exact private proxy hostname, including port if nonstandard")
    parser.add_argument("--follow-training", action="store_true", help="Load improved diffusion checkpoints when generating images")
    parser.add_argument("--motor-checkpoint", help="Use an exported motor checkpoint instead of runs/motor/best.pt")
    args = parser.parse_args()
    model, diffusion, info = load_checkpoint(args.checkpoint, args.graph, args.device, args.raw)
    server = make_server(model, diffusion, info, args.port, args.allowed_host, args.follow_training, args.motor_checkpoint, args.checkpoint)
    print(f"FLYcasso is ready at http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
