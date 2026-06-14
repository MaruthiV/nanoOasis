# nanoOasis ONNX export (Phase 7, W1): DiT denoiser + VAE decoder -> demo/assets for in-browser ORT-Web.
# The browser runs the AR loop; only these two nets ship -- the VAE ENCODER is offline-only (the retry
# re-seed uses precomputed seed contexts, emitted here). model.py is einsum-free so the graphs stay
# WebGPU-friendly (BUGS H007). Run: python export.py --ckpt checkpoints/dit_small_gate2_run4_155k.pt \
#   --vae checkpoints/vae_small.pt --config small

import argparse
import json
import pathlib

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from infer import load_models, initial_context, make_schedule
from game import (BG_COLOR, BODY_COLOR, HEAD_COLOR, APPLE_COLOR, CELL, GAP, GRID_COLS, GRID_ROWS,
                  DIRS, _REVERSE)

OPSET = 18   # dynamo exporter targets >=18 (Split.num_outputs etc.); ORT-Web WebGPU supports it

# ORT-Web WebGPU EP coverage (the common ops; verify the live matrix at W2). Anything outside this set
# is printed as a flag, not a hard fail -- the definitive check is loading in ORT-Web (W2-proper).
WEBGPU_OK = {
    "Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Exp", "Log", "Sin", "Cos", "Erf", "Tanh", "Sigmoid",
    "Neg", "Reciprocal", "MatMul", "Gemm", "Softmax", "ReduceMean", "LayerNormalization", "Gelu",
    "Gather", "Concat", "Reshape", "Transpose", "Slice", "Unsqueeze", "Squeeze", "Cast", "Expand",
    "Where", "Range", "ConstantOfShape", "Constant", "Shape", "Equal", "Less", "Greater", "Trilu",
    "Split", "Clip", "Identity", "Flatten", "ScatterND", "Min", "Max",
}


class DenoiserONNX(nn.Module):
    # EDM precond wrapper: (window x_in, per-frame sigma, per-frame action) -> denoised LAST-frame x0.
    # JS builds the 8-frame window (context held at sigma_stab + the noisy target) and runs euler-4.
    def __init__(self, diff):
        super().__init__()
        self.diff = diff

    def forward(self, x_in, sigma, action):
        # action comes in as int32 (WebGPU-friendlier than int64); Embedding wants long
        return self.diff.denoise(x_in, sigma, action.long())[:, -1]     # (1, C, Hp, Wp)


class DecoderONNX(nn.Module):
    # VAE decoder: latent tokens (1, N, C) -> image (1, Hgame, Wgame, 3) in [-1, 1].
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z)


def _inline(path: pathlib.Path) -> None:
    # the dynamo exporter externalizes weights to <name>.data; inline them into one self-contained .onnx
    # so the browser fetches a single file per model (ORT-Web external-data handling is fiddly).
    m = onnx.load(str(path), load_external_data=True)
    onnx.save_model(m, str(path), save_as_external_data=False)
    data = path.parent / (path.name + ".data")
    if data.exists():
        data.unlink()


def _to_fp16(path: pathlib.Path) -> None:
    # half-precision weights -> ~half the download. keep_io_types keeps float32 I/O so inference.js is
    # unchanged. near-zero quality cost on a noisy diffusion sampler; the real check is playing it (CPU
    # can't fully exercise fp16 kernels).
    from onnxconverter_common import float16
    m16 = float16.convert_float_to_float16(onnx.load(str(path)), keep_io_types=True)
    onnx.save_model(m16, str(path), save_as_external_data=False)


def _census(path: pathlib.Path) -> set[str]:
    ops = {n.op_type for n in onnx.load(str(path)).graph.node}
    flagged = sorted(ops - WEBGPU_OK)
    print(f"  {path.name}: {len(ops)} distinct ops")
    print(f"    ops: {', '.join(sorted(ops))}")
    print(f"    not-in-allowlist: {flagged if flagged else 'none (all common WebGPU ops)'}")
    return ops


def _parity(path: pathlib.Path, torch_out: torch.Tensor, feeds: dict, tol: float = 1e-3) -> None:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, feeds)[0]
    diff = float(np.abs(onnx_out - torch_out.detach().cpu().numpy()).max())
    rel = diff / (float(torch_out.abs().max()) + 1e-9)
    print(f"  {path.name}: max|onnx - torch| = {diff:.3e}  (rel {rel:.2e})  {'OK' if rel < tol else 'CHECK'}")


def main(ckpt: str, vae_path: str, config_name: str, out_dir: str, n_seeds: int, fp16: bool = True) -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cpu"                                                       # export on CPU -> portable graphs
    cfg, vae, diff = load_models(ckpt, vae_path, config_name, device)
    diff.eval(); vae.eval()

    T = cfg.dit.context_frames
    C = diff.model.latent_C
    Hp, Wp, N = vae.Hp, vae.Wp, vae.N

    # ---- DiT denoiser ----
    x_in = torch.randn(1, T, C, Hp, Wp)
    sigma = torch.rand(1, T) * 3.0 + 0.05
    action = torch.zeros(1, T, dtype=torch.int32)
    den = DenoiserONNX(diff).eval()
    with torch.no_grad():
        den_out = den(x_in, sigma, action)
    # dynamo exporter (torch 2.12): decomposes nn.MultiheadAttention + SDPA into WebGPU-friendly matmul/softmax
    torch.onnx.export(den, (x_in, sigma, action), str(out / "dit.onnx"), opset_version=OPSET, dynamo=True,
                      input_names=["x_in", "sigma", "action"], output_names=["x0"], optimize=True)
    _inline(out / "dit.onnx")

    # ---- VAE decoder ----
    z = torch.randn(1, N, C)
    dec = DecoderONNX(vae).eval()
    with torch.no_grad():
        dec_out = dec(z)
    torch.onnx.export(dec, (z,), str(out / "vae_dec.onnx"), opset_version=OPSET, dynamo=True,
                      input_names=["z"], output_names=["img"], optimize=True)
    _inline(out / "vae_dec.onnx")

    if fp16:
        _to_fp16(out / "dit.onnx")
        _to_fp16(out / "vae_dec.onnx")
        print("\nconverted both to FP16 (float32 I/O preserved -> inference.js unchanged; ~half the size)")
    tol = 2e-2 if fp16 else 1e-3

    print("\n=== W3 parity (ONNX CPU vs PyTorch) ===")
    _parity(out / "dit.onnx", den_out,
            {"x_in": x_in.numpy(), "sigma": sigma.numpy(), "action": action.numpy()}, tol)
    _parity(out / "vae_dec.onnx", dec_out, {"z": z.numpy()}, tol)

    print("\n=== W2 op census (WebGPU coverage first pass) ===")
    print("  ORT providers in this build:", ort.get_available_providers())
    _census(out / "dit.onnx")
    _census(out / "vae_dec.onnx")

    # ---- precomputed seed contexts for the retry re-seed (8 latents + 8 actions per seed) ----
    seeds_z, seeds_a, seeds_state = [], [], []
    for s in range(n_seeds):
        zc, actions, g = initial_context(vae, T, s, device)             # g = the real game after the seed steps
        seeds_z.append(zc.squeeze(0).numpy())
        seeds_a.append([int(a) for a in actions])
        # exact ground-truth state for the JS referee (head, ORDERED body head->tail, heading, apple)
        seeds_state.append({"head": [int(g.body[0][0]), int(g.body[0][1])],
                            "body": [[int(c), int(r)] for c, r in g.body],
                            "heading": int(g.heading),
                            "apple": [int(g.apple[0]), int(g.apple[1])]})
    seed_lat = np.stack(seeds_z).astype(np.float32)                     # (n, T, C, Hp, Wp)
    np.savez(out / "seed_contexts.npz", latents=seed_lat, actions=np.array(seeds_a, np.int64))
    seed_lat.tofile(out / "seed_latents.bin")                           # raw f32, row-major -> the browser fetches this

    # the JS sampler reads the precomputed schedule (avoids a JS/Python schedule mismatch)
    sched = [float(s) for s in make_schedule(cfg, int(cfg.diffusion.num_sample_steps), 0.0, device).tolist()]
    img_h, img_w = Hp * vae.P, Wp * vae.P
    manifest = {
        "dit": {"inputs": {"x_in": [1, T, C, Hp, Wp], "sigma": [1, T], "action": [1, T]},
                "output": {"x0": [1, C, Hp, Wp]}, "opset": OPSET, "action_dtype": "int32"},
        "vae_dec": {"inputs": {"z": [1, N, C]}, "output": {"img": [1, img_h, img_w, 3]}},
        "latent": {"C": C, "Hp": Hp, "Wp": Wp, "N": N, "token_order": "row-major Hp-then-Wp"},
        "sampler": {"steps": int(cfg.diffusion.num_sample_steps), "sigma_stab": 0.3, "sigma_schedule": sched},
        "seeds": {"file": "seed_latents.bin", "dtype": "float32",
                  "shape": [n_seeds, T, C, Hp, Wp], "actions": seeds_a, "states": seeds_state},
        "game": {"grid_cols": GRID_COLS, "grid_rows": GRID_ROWS, "cell": CELL, "gap": GAP,
                 "img_h": img_h, "img_w": img_w, "sample_inset": 12, "sample_size": 8,
                 "dirs": [list(d) for d in DIRS], "reverse": list(_REVERSE), "start_len": 3,
                 "colors": {"empty": list(BG_COLOR), "body": list(BODY_COLOR),
                            "head": list(HEAD_COLOR), "apple": list(APPLE_COLOR)}},
        "context_frames": T, "num_actions": int(cfg.dit.num_actions), "n_seeds": n_seeds,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n=== artifacts ===")
    for f in ("dit.onnx", "vae_dec.onnx", "seed_contexts.npz", "manifest.json"):
        p = out / f
        print(f"  {p}  ({p.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/dit_small_gate2_run4_155k.pt")
    p.add_argument("--vae", type=str, default="checkpoints/vae_small.pt")
    p.add_argument("--config", type=str, default="small")
    p.add_argument("--out", type=str, default="demo/assets")
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--fp32", action="store_true", help="keep FP32 weights (default: FP16 -> smaller download)")
    args = p.parse_args()
    main(args.ckpt, args.vae, args.config, args.out, args.n_seeds, fp16=not args.fp32)
