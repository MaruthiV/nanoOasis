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


def _maybe_init_wandb(config_name: str, stage: str, cfg) -> object | None:
    # Only init if WANDB_API_KEY is in the env (set via Modal Secret in cloud runs).
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb
    except ImportError:
        print("WANDB_API_KEY set but `wandb` not installed; skipping W&B logging.")
        return None
    return wandb.init(
        project="nano-oasis",
        name=f"{stage}-{config_name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        save_code=False,
    )


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


def main(stage: str = "dit", config_name: str = "tiny", total_steps: int | None = None,
         on_checkpoint=None) -> None:
    cfg = OmegaConf.load(f"configs/{config_name}.yaml")
    if total_steps is not None:
        cfg.training.total_steps = total_steps
    device = pick_device(cfg.training.device)
    print(f"stage: {stage}  device: {device}  config: {config_name}  steps: {cfg.training.total_steps}")

    torch.manual_seed(cfg.seed)
    T_use = cfg.dit.context_frames                                      # model's RoPE expects exactly this T
    pre_encoded = bool(cfg.data.get("pre_encoded", False))             # shards hold VAE latents -> skip the encode

    # data
    ds = EpisodeWindowDataset(cfg.data.index_path, split="train",
                              cache_size=cfg.data.cache_size, seed=cfg.seed,
                              event_frac=float(cfg.data.get("event_frac", 0.0)))
    loader = DataLoader(ds, batch_size=cfg.training.batch_size, num_workers=0)

    # frozen VAE -- only needed for raw frames; pre-encoded latents skip it entirely
    vae = None
    if not pre_encoded:
        vae_ckpt_path = cfg.get("vae_ckpt", f"checkpoints/vae_{config_name}.pt")   # ablation configs share one VAE
        vae_ckpt = torch.load(vae_ckpt_path, weights_only=False, map_location=device)
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
    wandb_run = _maybe_init_wandb(config_name, stage, cfg)

    ckpt_dir = pathlib.Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"dit_{config_name}.pt"

    def save_ckpt(s: int) -> None:
        torch.save({"model": model.state_dict(), "ema": ema.shadow, "opt": opt.state_dict(),
                    "step": s, "config": OmegaConf.to_container(cfg)}, ckpt_path)
        if on_checkpoint is not None:
            on_checkpoint()                # commit the Modal volume so the ckpt survives preemption

    # resume from a committed checkpoint if the run is incomplete -- preemption recovery (needed for M7)
    start_step = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, weights_only=False, map_location=device)
        if int(ck.get("step", 0)) < cfg.training.total_steps:
            model.load_state_dict(ck["model"])
            if "opt" in ck:
                opt.load_state_dict(ck["opt"])
            if "ema" in ck:
                ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
            start_step = int(ck.get("step", 0))
            print(f"resuming from {ckpt_path} at step {start_step}")

    # sigma-bucket loss curve -- mandatory standing diagnostic (EXPERIMENTS.md). 10 log-spaced bins.
    n_buckets = 10
    bucket_edges = torch.logspace(math.log10(cfg.diffusion.sigma_min),
                                  math.log10(cfg.diffusion.sigma_max), n_buckets + 1, device=device)
    bucket_loss = torch.zeros(n_buckets, device=device)
    bucket_cnt = torch.zeros(n_buckets, device=device)

    step = start_step
    t0 = time.time()
    recent: list[float] = []
    for frames, actions in loader:
        if step >= cfg.training.total_steps:
            break
        # dataset yields the last T_use of WINDOW frames/latents + actions (model wants context_frames)
        actions = actions[:, -T_use:].long().to(device)
        if pre_encoded:
            z = frames[:, -T_use:].to(device).float()                  # `frames` is actually latents (B, T, C, Hp, Wp)
        else:
            with torch.no_grad():
                z = encode_window(vae, frames[:, -T_use:].to(device))  # (B, T, C, Hp, Wp)

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

        # bin this step's per-frame losses by sigma; reset each log window
        b_idx = (torch.bucketize(info["sigma_flat"], bucket_edges) - 1).clamp(0, n_buckets - 1)
        bucket_loss.scatter_add_(0, b_idx, info["loss_flat"])
        bucket_cnt.scatter_add_(0, b_idx, torch.ones_like(info["loss_flat"]))

        if step % cfg.training.log_every == 0:
            mean = sum(recent) / len(recent)
            elapsed = time.time() - t0
            sps = (step - start_step + 1) / max(elapsed, 1e-6)
            print(f"step {step:5d}  loss {mean:.4f}  σ̄ {info['sigma_mean']:.3f}  "
                  f"lr {lr:.2e}  {sps:.1f} steps/s  {elapsed:.0f}s")
            if wandb_run is not None:
                wandb_run.log({
                    "train/loss":        loss.item(),
                    "train/loss_smooth": mean,
                    "train/sigma_mean":  info["sigma_mean"],
                    "train/lr":          lr,
                    "throughput/steps_per_s": sps,
                    "time/elapsed_s":    elapsed,
                }, step=step)

            # sigma-bucket loss curve over the last log window, then reset
            occ = bucket_cnt > 0
            bucket_mean = bucket_loss / bucket_cnt.clamp(min=1)            # 0 where empty
            curve = " ".join(f"{m:.2f}" if o else "·"
                             for m, o in zip(bucket_mean.tolist(), occ.tolist()))
            print(f"  sigma-buckets [{cfg.diffusion.sigma_min:.3f}..{cfg.diffusion.sigma_max:.0f}]: {curve}")
            if wandb_run is not None:
                wandb_run.log({f"sigma_bucket/b{i}": bucket_mean[i].item()
                               for i in range(n_buckets) if occ[i]}, step=step)
            bucket_loss.zero_()
            bucket_cnt.zero_()

        if step > 0 and step % cfg.training.ckpt_every == 0:
            save_ckpt(step)
        step += 1

    save_ckpt(step)
    elapsed = time.time() - t0
    final = sum(recent) / max(1, len(recent))
    print(f"done. {step} steps ({step - start_step} this run), {elapsed:.0f}s, "
          f"{(step - start_step) / max(elapsed, 1e-6):.1f} steps/s. final loss {final:.4f}. saved {ckpt_path}")
    if wandb_run is not None:
        wandb_run.summary["final_loss"] = final
        wandb_run.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=str, default="dit", choices=["dit", "lcm"])
    p.add_argument("--config", type=str, default="tiny")
    p.add_argument("--steps", type=int, default=None)
    args = p.parse_args()
    main(args.stage, args.config, total_steps=args.steps)
