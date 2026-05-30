# nanoOasis DiT training loop. Frozen VAE + EDM Diffusion Forcing. Karpathy-clean single file.
# Tiny tier: MacBook MPS, ~30 min, target visible loss drop on smoke.

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import math
import pathlib
import time

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from vae import VAE
from model import DiT
from diffusion import EDMDiffusion
from data import EpisodeWindowDataset


def pick_device(spec: str) -> str:
    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cosine_lr(step: int, warmup: int, total: int, base: float) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base * (1.0 + math.cos(math.pi * min(progress, 1.0)))


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)


def encode_window(vae: VAE, frames: torch.Tensor) -> torch.Tensor:
    # frames: (B, T, H, W, 3) uint8 -> latent (B, T, C, Hp, Wp) via VAE encoder mean
    B, T = frames.shape[:2]
    flat = frames.view(B * T, *frames.shape[2:])
    mu, _ = vae.encode(flat)                                          # (B*T, N=Hp*Wp, C)
    z = mu.view(B * T, vae.Hp, vae.Wp, vae.latent_channels)            # row-major matches _patchify
    z = z.permute(0, 3, 1, 2).contiguous()                             # (B*T, C, Hp, Wp)
    return z.view(B, T, vae.latent_channels, vae.Hp, vae.Wp)


def main(stage: str = "dit", config_name: str = "tiny", total_steps: int | None = None) -> None:
    cfg = OmegaConf.load(f"configs/{config_name}.yaml")
    if total_steps is not None:
        cfg.training.total_steps = total_steps
    device = pick_device(cfg.training.device)
    print(f"stage: {stage}  device: {device}  config: {config_name}  steps: {cfg.training.total_steps}")

    torch.manual_seed(cfg.seed)
    T_use = cfg.dit.context_frames                                      # model's RoPE expects exactly this T

    # data
    ds = EpisodeWindowDataset(cfg.data.index_path, split="train",
                              cache_size=cfg.data.cache_size, seed=cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.training.batch_size, num_workers=0)

    # frozen VAE
    vae_ckpt = torch.load(f"checkpoints/vae_{config_name}.pt", weights_only=False, map_location=device)
    vae = VAE(cfg.vae).to(device).eval()
    vae.load_state_dict(vae_ckpt["model"])
    for p in vae.parameters():
        p.requires_grad = False

    # DiT + diffusion
    model = DiT(cfg.dit).to(device)
    diff = EDMDiffusion(model, cfg.diffusion).to(device)
    print(f"DiT params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                            betas=tuple(cfg.training.betas), weight_decay=cfg.training.weight_decay)
    ema = EMA(model, decay=cfg.training.ema_decay)

    ckpt_dir = pathlib.Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"dit_{config_name}.pt"

    step = 0
    t0 = time.time()
    recent: list[float] = []
    for frames, actions in loader:
        if step >= cfg.training.total_steps:
            break
        # use the last T_use frames + their actions; dataset yields 17, model wants context_frames (4 for tiny)
        frames = frames[:, -T_use:].to(device)
        actions = actions[:, -T_use:].long().to(device)

        with torch.no_grad():
            z = encode_window(vae, frames)                              # (B, T, C, Hp, Wp)

        lr = cosine_lr(step, cfg.training.warmup_steps, cfg.training.total_steps, cfg.training.lr)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad()
        loss, info = diff.loss(z, actions)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        opt.step()
        ema.update(model)

        recent.append(loss.item())
        if len(recent) > 100:
            recent.pop(0)

        if step % cfg.training.log_every == 0:
            mean = sum(recent) / len(recent)
            elapsed = time.time() - t0
            sps = (step + 1) / max(elapsed, 1e-6)
            print(f"step {step:5d}  loss {mean:.4f}  σ̄ {info['sigma_mean']:.3f}  "
                  f"lr {lr:.2e}  {sps:.1f} steps/s  {elapsed:.0f}s")

        if step > 0 and step % cfg.training.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "ema": ema.shadow, "step": step,
                        "config": OmegaConf.to_container(cfg)}, ckpt_path)
        step += 1

    torch.save({"model": model.state_dict(), "ema": ema.shadow, "step": step,
                "config": OmegaConf.to_container(cfg)}, ckpt_path)
    elapsed = time.time() - t0
    final = sum(recent) / max(1, len(recent))
    print(f"done. {step} steps, {elapsed:.0f}s, {step/elapsed:.1f} steps/s. "
          f"final loss {final:.4f}. saved {ckpt_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=str, default="dit", choices=["dit", "lcm"])
    p.add_argument("--config", type=str, default="tiny")
    p.add_argument("--steps", type=int, default=None)
    args = p.parse_args()
    main(args.stage, args.config, total_steps=args.steps)
