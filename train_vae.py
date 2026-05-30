# nanoOasis VAE training loop. Tiny tier: MacBook MPS, ~30 min, target L1 < 0.02 on smoke.
# Karpathy-clean: one file, for-loop training, hyperparams from configs/<tier>.yaml.

import os
# MPS quirks fall back to CPU instead of crashing (research_notes.md §4)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import math
import pathlib
import time

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from vae import VAE
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


def main(config_name: str = "tiny", total_steps: int | None = None) -> None:
    cfg = OmegaConf.load(f"configs/{config_name}.yaml")
    if total_steps is not None:
        cfg.training.total_steps = total_steps
    device = pick_device(cfg.training.device)
    print(f"device: {device}  |  config: {config_name}  |  steps: {cfg.training.total_steps}")

    torch.manual_seed(cfg.seed)

    ds = EpisodeWindowDataset(
        cfg.data.index_path, split="train",
        cache_size=cfg.data.cache_size, seed=cfg.seed,
    )
    loader = DataLoader(ds, batch_size=cfg.training.batch_size, num_workers=0)

    model = VAE(cfg.vae).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"VAE params: {n_params:,}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        betas=tuple(cfg.training.betas),
        weight_decay=cfg.training.weight_decay,
    )

    ckpt_dir = pathlib.Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"vae_{config_name}.pt"

    step = 0
    t0 = time.time()
    recent_l1: list[float] = []
    for frames, _ in loader:
        if step >= cfg.training.total_steps:
            break
        # frames: (B, 17, 96, 128, 3) uint8 -> flatten time into batch for per-frame VAE
        B, T = frames.shape[:2]
        x = frames.view(B * T, *frames.shape[2:]).to(device)

        lr = cosine_lr(step, cfg.training.warmup_steps, cfg.training.total_steps, cfg.training.lr)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad()
        recon, mu, logvar = model(x)
        loss, info = model.loss(x, recon, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        opt.step()

        recent_l1.append(info["l1"])
        if len(recent_l1) > 100:
            recent_l1.pop(0)

        if step % cfg.training.log_every == 0:
            mean_l1 = sum(recent_l1) / len(recent_l1)
            elapsed = time.time() - t0
            sps = (step + 1) / max(elapsed, 1e-6)
            print(f"step {step:5d}  L1 {mean_l1:.4f}  KL {info['kl']:.4f}  "
                  f"lr {lr:.2e}  {sps:.1f} steps/s  {elapsed:.0f}s")

        if step > 0 and step % cfg.training.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "step": step,
                        "config": OmegaConf.to_container(cfg)}, ckpt_path)

        step += 1

    torch.save({"model": model.state_dict(), "step": step,
                "config": OmegaConf.to_container(cfg)}, ckpt_path)
    elapsed = time.time() - t0
    final_l1 = sum(recent_l1) / max(1, len(recent_l1))
    print(f"done. {step} steps, {elapsed:.0f}s, {step/elapsed:.1f} steps/s. "
          f"final L1 {final_l1:.4f}. saved {ckpt_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="tiny")
    p.add_argument("--steps", type=int, default=None, help="override total_steps")
    args = p.parse_args()
    main(args.config, total_steps=args.steps)
