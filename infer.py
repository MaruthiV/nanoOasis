# nanoOasis interactive inference. Pygame window (or headless) driven by the trained DiT.
# Diffusion Forcing rollout: context frames at small stab noise; new frame iterates schedule.

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import pathlib

import numpy as np
import torch
from omegaconf import OmegaConf

from vae import VAE
from model import DiT
from diffusion import EDMDiffusion
from game import Game, _keys_to_action, W, H


def pick_device(spec: str) -> str:
    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def edm_sigma_schedule(num_steps: int, sigma_min: float, sigma_max: float,
                       rho: float = 7.0, device: str = "cpu") -> torch.Tensor:
    # Karras EDM Eq. 5 -- exponentially-ramped schedule from sigma_max down to sigma_min
    ramp = torch.linspace(0, 1, num_steps + 1, device=device)
    return (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho


@torch.no_grad()
def sample_next_frame(diff: EDMDiffusion, z_ctx: torch.Tensor,
                      full_actions: torch.Tensor, sigmas: torch.Tensor,
                      sigma_stab: float = 0.1) -> torch.Tensor:
    # z_ctx: (B=1, T-1, C, Hp, Wp) clean context; full_actions: (T,) ints incl. the new action
    B = z_ctx.shape[0]
    T = z_ctx.shape[1] + 1
    C, Hp, Wp = z_ctx.shape[2:]
    device = z_ctx.device

    if full_actions.dim() == 1:
        full_actions = full_actions.unsqueeze(0)

    noisy_ctx = z_ctx + sigma_stab * torch.randn_like(z_ctx)
    x_new = torch.randn(B, 1, C, Hp, Wp, device=device) * sigmas[0]

    for i in range(len(sigmas) - 1):
        sigma_cur = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        sig = torch.cat([
            torch.full((B, T - 1), sigma_stab, device=device),
            torch.full((B, 1),     sigma_cur,  device=device),
        ], dim=1)
        D = diff.denoise(torch.cat([noisy_ctx, x_new], dim=1), sig, full_actions)
        d = (x_new - D[:, -1:]) / sigma_cur                    # score-like estimate at the new frame
        x_new = x_new + (sigma_next - sigma_cur) * d           # Euler step
    return x_new


@torch.no_grad()
def decode_latent(vae: VAE, z: torch.Tensor) -> np.ndarray:
    # z: (1, 1, C, Hp, Wp) -> (H, W, 3) uint8
    z_tok = z.squeeze(0).squeeze(0).permute(1, 2, 0).reshape(-1, vae.latent_channels).unsqueeze(0)
    recon = vae.decode(z_tok)
    return ((recon.squeeze(0).clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()


@torch.no_grad()
def initial_context(vae: VAE, T_ctx: int, seed: int, device: str):
    g = Game(seed=seed, biome="grass")
    frames = [g.step(0)[0].copy() for _ in range(T_ctx)]
    flat = torch.from_numpy(np.stack(frames)).to(device)
    mu, _ = vae.encode(flat)
    z = mu.view(T_ctx, vae.Hp, vae.Wp, vae.latent_channels).permute(0, 3, 1, 2).contiguous()
    return z.unsqueeze(0), [0] * T_ctx                          # (1, T_ctx, C, Hp, Wp), actions list


def load_models(ckpt_path: str, vae_path: str, config_name: str, device: str):
    cfg = OmegaConf.load(f"configs/{config_name}.yaml")
    vae = VAE(cfg.vae).to(device).eval()
    vae.load_state_dict(torch.load(vae_path, weights_only=False, map_location=device)["model"])
    for p in vae.parameters():
        p.requires_grad = False

    model = DiT(cfg.dit).to(device).eval()
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    for p in model.parameters():
        p.requires_grad = False

    diff = EDMDiffusion(model, cfg.diffusion).to(device).eval()
    return cfg, vae, diff


def headless(ckpt_path, vae_path, config_name, n_frames, num_steps, sigma_stab, seed, out_dir):
    import imageio.v3 as iio
    device = pick_device("auto")
    cfg, vae, diff = load_models(ckpt_path, vae_path, config_name, device)
    T_ctx = cfg.dit.context_frames

    z_history, actions = initial_context(vae, T_ctx, seed, device)
    sigmas = edm_sigma_schedule(num_steps, cfg.diffusion.sigma_min, cfg.diffusion.sigma_max, device=device)

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cycle = [0, 2, 2, 2, 5, 5, 1, 1, 0, 3, 3, 3]
    frames_out: list[np.ndarray] = []
    for i in range(n_frames):
        action = cycle[i % len(cycle)]
        z_ctx = z_history[:, 1:]
        full_actions = torch.tensor(actions[1:] + [action], dtype=torch.long, device=device)
        new_lat = sample_next_frame(diff, z_ctx, full_actions, sigmas, sigma_stab)
        z_history = torch.cat([z_ctx, new_lat], dim=1)
        actions = actions[1:] + [action]
        frame = decode_latent(vae, new_lat)
        iio.imwrite(out_dir / f"frame_{i:04d}_a{action}.png", frame)
        frames_out.append(frame)
    # also write a stitched strip for quick eyeball
    strip = np.concatenate(frames_out, axis=1)
    iio.imwrite(out_dir / "strip.png", strip)
    # crude action-response check
    diffs = [float(np.abs(frames_out[i].astype(int) - frames_out[i - 1].astype(int)).mean())
             for i in range(1, len(frames_out))]
    print(f"wrote {n_frames} frames + strip.png to {out_dir}")
    print(f"mean abs frame-to-frame delta: {sum(diffs)/len(diffs):.2f}/255 (>0 => model is producing change)")


def play(ckpt_path, vae_path, config_name, num_steps, sigma_stab, seed):
    import pygame
    device = pick_device("auto")
    cfg, vae, diff = load_models(ckpt_path, vae_path, config_name, device)
    T_ctx = cfg.dit.context_frames

    z_history, actions = initial_context(vae, T_ctx, seed, device)
    sigmas = edm_sigma_schedule(num_steps, cfg.diffusion.sigma_min, cfg.diffusion.sigma_max, device=device)

    pygame.init()
    screen = pygame.display.set_mode((W * 4, H * 4))
    pygame.display.set_caption("nanoOasis (model)")
    clock = pygame.time.Clock()

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        action = _keys_to_action(keys[pygame.K_LEFT], keys[pygame.K_RIGHT], keys[pygame.K_SPACE])

        z_ctx = z_history[:, 1:]
        full_actions = torch.tensor(actions[1:] + [action], dtype=torch.long, device=device)
        new_lat = sample_next_frame(diff, z_ctx, full_actions, sigmas, sigma_stab)
        z_history = torch.cat([z_ctx, new_lat], dim=1)
        actions = actions[1:] + [action]

        frame = decode_latent(vae, new_lat)
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))   # H005: surfarray is (W,H,3)
        screen.blit(pygame.transform.scale(surf, (W * 4, H * 4)), (0, 0))
        pygame.display.flip()
        clock.tick(15)

    pygame.quit()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--vae", type=str, required=True)
    p.add_argument("--config", type=str, default="tiny")
    p.add_argument("--steps", type=int, default=8, help="Euler sampler steps per frame")
    p.add_argument("--sigma-stab", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", type=int, default=0, help="write N frames + strip.png, no window")
    p.add_argument("--out", type=str, default="assets/infer_smoke")
    args = p.parse_args()

    if args.headless > 0:
        headless(args.ckpt, args.vae, args.config, args.headless,
                 args.steps, args.sigma_stab, args.seed, args.out)
    else:
        play(args.ckpt, args.vae, args.config, args.steps, args.sigma_stab, args.seed)
