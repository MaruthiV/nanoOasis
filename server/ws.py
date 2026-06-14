# nanoOasis WebSocket fallback (Phase 7, W6): runs the model + referee SERVER-SIDE (PyTorch on GPU) for
# browsers without WebGPU. Same loop as the client (infer.py sampler + the JS referee, mirrored in Python);
# streams base64-PNG frames + status over a WebSocket. Deploy: modal deploy server/ws.py. Local test:
# python server/ws.py  (uvicorn on :8000), or the in-process FastAPI TestClient.

import asyncio
import base64
import io
import json

import modal

app = modal.App("nano-oasis-demo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch>=2.5", "numpy<2", "omegaconf>=2.3", "imageio[ffmpeg]",
                 "pyarrow", "zstandard", "fastapi", "uvicorn")
    .add_local_python_source("game", "data", "vae", "model", "diffusion", "infer")
    .add_local_dir("configs", "/root/configs")
)
ckpt_vol = modal.Volume.from_name("nano-oasis-ckpts", create_if_missing=True)

# the SAME checkpoint the client ONNX was exported from -> client and server play identically
CKPT, VAE, CONFIG = "checkpoints/dit_small_gate2_run4_155k.pt", "checkpoints/vae_small.pt", "small"


def build_app(ckpt: str = CKPT, vae: str = VAE, config: str = CONFIG):
    import numpy as np
    import torch
    import imageio.v3 as iio
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    from infer import (load_models, initial_context, make_schedule, sample_next_frame,
                       decode_latent, read_grid)
    from game import DIRS, _REVERSE, GRID_COLS, GRID_ROWS

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    cfg, vae_m, diff = load_models(ckpt, vae, config, device)
    T = int(cfg.dit.context_frames)
    sigmas = make_schedule(cfg, int(cfg.diffusion.num_sample_steps), 0.0, device)
    SS, BAD_HEAD_LIMIT = 0.3, 5
    rng = np.random.default_rng()                                  # per-process; non-deterministic seed variety

    def png_b64(frame) -> str:
        buf = io.BytesIO()
        iio.imwrite(buf, frame, extension=".png")                 # flat blocks -> ~1-2 KB
        return base64.b64encode(buf.getvalue()).decode()

    class Session:
        # mirrors demo/inference.js: rule referee adjudicates wall/self, model renders, head/apple resync.
        def __init__(self, seed: int):
            z, actions, g = initial_context(vae_m, T, seed, device)
            self.z, self.actions = z, list(actions)
            self.head = [int(g.body[0][0]), int(g.body[0][1])]
            self.heading = int(g.heading)
            self.body = [[int(c), int(r)] for c, r in g.body]
            self.apple = [int(g.apple[0]), int(g.apple[1])]
            self.eaten = self.badhead = 0

        def initial(self) -> dict:
            return {"dead": False, "apples": 0, "length": len(self.body), "heading": self.heading,
                    "frame": png_b64(decode_latent(vae_m, self.z[:, -1:]))}

        def step(self, a: int) -> dict:
            if a != _REVERSE[self.heading]:
                self.heading = a
            dc, dr = DIRS[self.heading]
            nh = [self.head[0] + dc, self.head[1] + dr]
            if not (0 <= nh[0] < GRID_COLS and 0 <= nh[1] < GRID_ROWS):
                return {"dead": True, "reason": "wall"}
            will_eat = nh == self.apple
            trunk = self.body if will_eat else self.body[:-1]
            if nh in trunk:
                return {"dead": True, "reason": "self"}

            fa = torch.tensor(self.actions[1:] + [a], dtype=torch.long, device=device)
            new = sample_next_frame(diff, self.z[:, 1:], fa, sigmas, SS, "euler")
            self.z = torch.cat([self.z[:, 1:], new], dim=1)
            self.actions = self.actions[1:] + [a]
            frame = decode_latent(vae_m, new)
            grid, _ = read_grid(frame)

            hs = np.argwhere(grid == 2)                            # HEAD; resync to the rendered head
            if len(hs) == 1:
                new_head = [int(hs[0][1]), int(hs[0][0])]
                self.badhead = 0
            else:
                new_head = nh
                self.badhead += 1
            self.body = [new_head] + self.body
            if not will_eat:
                self.body.pop()
            self.head = new_head
            if will_eat:
                self.eaten += 1
            ap = np.argwhere(grid == 3)                            # apple tracked from the rendered frame
            if len(ap) == 1:
                self.apple = [int(ap[0][1]), int(ap[0][0])]
            if self.badhead >= BAD_HEAD_LIMIT:
                return {"dead": True, "reason": "breakdown"}
            return {"dead": False, "apples": self.eaten, "length": len(self.body), "frame": png_b64(frame)}

    web = FastAPI()

    @web.get("/health")
    def health():
        return {"ok": True, "device": device}

    @web.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        sess: Session | None = None
        try:
            while True:
                msg = json.loads(await sock.receive_text())
                t = msg.get("type")
                if t in ("start", "retry"):
                    sess = Session(int(rng.integers(0, 1_000_000)))
                    await sock.send_text(json.dumps(sess.initial()))
                elif t == "action" and sess is not None:
                    out = await asyncio.to_thread(sess.step, int(msg["a"]))   # don't block the event loop
                    await sock.send_text(json.dumps(out))
        except WebSocketDisconnect:
            pass

    return web


@app.function(image=image, gpu="A10G", volumes={"/checkpoints": ckpt_vol}, timeout=12 * 3600)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def serve():
    import os
    import pathlib
    os.chdir("/root")
    if not pathlib.Path("/root/checkpoints").exists():
        pathlib.Path("/root/checkpoints").symlink_to("/checkpoints")
    return build_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(build_app(), host="0.0.0.0", port=8000)
