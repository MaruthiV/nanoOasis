# Pre-encode frame shards to frozen-VAE latents. Removes the per-step VAE forward at train time and
# lets a multi-worker loader feed the GPUs -- the dominant launch-run speedup (see the M7 GPU analysis).
# Latent shards mirror the frame shards (same index, same paths) but store `latents` instead of `frames`.

import io
import pathlib
import shutil

import numpy as np
import torch
import zstandard as zstd
from omegaconf import OmegaConf

from vae import VAE


def encode_dir(index_path: str, vae_path: str, config_name: str, out_dir: str,
               device: str | None = None, batch: int = 256) -> None:
    import pyarrow.parquet as pq
    device = device or ("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")
    cfg = OmegaConf.load(f"configs/{config_name}.yaml")
    vae = VAE(cfg.vae).to(device).eval()
    vae.load_state_dict(torch.load(vae_path, weights_only=False, map_location=device)["model"])
    for p in vae.parameters():
        p.requires_grad = False

    root = pathlib.Path(index_path).parent
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(index_path).to_pylist()

    for r in rows:
        raw = zstd.ZstdDecompressor().decompress((root / r["path"]).read_bytes())
        d = dict(np.load(io.BytesIO(raw)))
        frames = d["frames"]                                  # (N, H, W, 3) uint8
        chunks = []
        with torch.no_grad():
            for i in range(0, len(frames), batch):
                fb = torch.from_numpy(frames[i:i + batch]).to(device)
                mu, _ = vae.encode(fb)                         # (b, Hp*Wp, C) -- encoder mean, no reparam
                z = mu.view(-1, vae.Hp, vae.Wp, vae.latent_channels).permute(0, 3, 1, 2).contiguous()
                chunks.append(z.half().cpu().numpy())          # fp16 latents
        latents = np.concatenate(chunks, 0)                   # (N, C, Hp, Wp)
        buf = io.BytesIO()
        np.savez(buf, latents=latents, actions=d["actions"], dones=d["dones"], level_seeds=d["level_seeds"])
        (out / r["path"]).write_bytes(zstd.ZstdCompressor(level=3).compress(buf.getvalue()))
    shutil.copy(index_path, out / "index.parquet")            # same index; paths unchanged
    print(f"encoded {len(rows)} shards ({sum(r['length'] for r in rows)} frames) -> {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, help="frame-shard index, e.g. data/smoke/index.parquet")
    p.add_argument("--vae", required=True, help="frozen VAE checkpoint, e.g. checkpoints/vae_small.pt")
    p.add_argument("--config", default="small", help="config providing the VAE architecture")
    p.add_argument("--out", required=True, help="output dir for latent shards, e.g. data/smoke_latents")
    args = p.parse_args()
    encode_dir(args.index, args.vae, args.config, args.out)
