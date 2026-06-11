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
from game import (Game, _keys_to_action, SCALE, W, H, PALETTE, DB16,
                  BRICK_TOP, BRICK_H, BRICK_W, BRICK_ROWS, BRICK_COLS, BALL_SPEED)


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
                      sigma_stab: float = 0.1, sampler: str = "heun") -> torch.Tensor:
    # z_ctx: (B=1, T-1, C, Hp, Wp) clean context; full_actions: (T,) ints incl. the new action.
    # sampler: "heun" = Karras EDM 2nd-order (deterministic, fewer mode-covering / ghosting artifacts; DIAMOND),
    #          "euler" = 1st-order baseline.
    B = z_ctx.shape[0]
    T = z_ctx.shape[1] + 1
    C, Hp, Wp = z_ctx.shape[2:]
    device = z_ctx.device

    if full_actions.dim() == 1:
        full_actions = full_actions.unsqueeze(0)

    noisy_ctx = z_ctx + sigma_stab * torch.randn_like(z_ctx)
    x_new = torch.randn(B, 1, C, Hp, Wp, device=device) * sigmas[0]

    def deriv(x, sigma):                                       # score-like estimate at the new frame
        sig = torch.cat([torch.full((B, T - 1), sigma_stab, device=device),
                         torch.full((B, 1),     sigma,      device=device)], dim=1)
        D = diff.denoise(torch.cat([noisy_ctx, x], dim=1), sig, full_actions)
        return (x - D[:, -1:]) / sigma

    for i in range(len(sigmas) - 1):
        sigma_cur = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        d = deriv(x_new, sigma_cur)
        x_euler = x_new + (sigma_next - sigma_cur) * d
        if sampler == "heun" and sigma_next > 0:               # Karras EDM Alg. 1 -- 2nd-order correction
            d2 = deriv(x_euler, sigma_next)
            x_new = x_new + (sigma_next - sigma_cur) * 0.5 * (d + d2)
        else:
            x_new = x_euler
    if device.type == "mps":
        torch.mps.empty_cache()                                # free per-frame so long Heun rollouts don't OOM
    return x_new


@torch.no_grad()
def decode_latent(vae: VAE, z: torch.Tensor) -> np.ndarray:
    # z: (1, 1, C, Hp, Wp) -> (H, W, 3) uint8
    z_tok = z.squeeze(0).squeeze(0).permute(1, 2, 0).reshape(-1, vae.latent_channels).unsqueeze(0)
    recon = vae.decode(z_tok)
    return ((recon.squeeze(0).clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()


@torch.no_grad()
def initial_context(vae: VAE, T_ctx: int, seed: int, device: str):
    g = Game(seed=seed, palette="grey")
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
    cycle = [2, 2, 2, 2, 1, 1, 1, 1]                # sweep the paddle right then left (Breakout)
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


@torch.no_grad()
def measure_horizon(ckpt_path, vae_path, config_name, n_frames, num_steps, sigma_stab, seed, out_dir):
    # autoregressive horizon-2x: roll the model and the real game in lockstep on the same actions,
    # report the t where pixel MSE vs ground truth first exceeds 2x the single-step floor.
    import imageio.v3 as iio
    device = pick_device("auto")
    cfg, vae, diff = load_models(ckpt_path, vae_path, config_name, device)
    T_ctx = cfg.dit.context_frames
    torch.manual_seed(seed)

    # ground truth: T_ctx context frames (action 0, matching initial_context), then a fixed action sequence
    g = Game(seed=seed, palette="grey")
    gt = [g.step(0)[0].copy() for _ in range(T_ctx)]
    cycle = [2, 2, 2, 2, 1, 1, 1, 1]                # sweep the paddle right then left (Breakout)
    action_seq = [cycle[i % len(cycle)] for i in range(n_frames)]
    for a in action_seq:
        gt.append(g.step(a)[0].copy())

    flat = torch.from_numpy(np.stack(gt[:T_ctx])).to(device)
    mu, _ = vae.encode(flat)
    z_history = mu.view(T_ctx, vae.Hp, vae.Wp, vae.latent_channels).permute(0, 3, 1, 2).contiguous().unsqueeze(0)
    actions = [0] * T_ctx
    sigmas = edm_sigma_schedule(num_steps, cfg.diffusion.sigma_min, cfg.diffusion.sigma_max, device=device)

    mses, pairs = [], []
    for t, a in enumerate(action_seq):
        z_ctx = z_history[:, 1:]
        full_actions = torch.tensor(actions[1:] + [a], dtype=torch.long, device=device)
        new_lat = sample_next_frame(diff, z_ctx, full_actions, sigmas, sigma_stab)
        z_history = torch.cat([z_ctx, new_lat], dim=1)
        actions = actions[1:] + [a]
        m = decode_latent(vae, new_lat)
        truth = gt[T_ctx + t]
        mses.append(float(np.mean((m.astype(np.float32) - truth.astype(np.float32)) ** 2)))
        pairs.append(np.concatenate([truth, m], axis=0))                # truth on top, model below

    floor = mses[0]
    horizon = next((t for t, mse in enumerate(mses) if mse > 2 * floor), n_frames)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_dir / "horizon_strip.png", np.concatenate(pairs[:16], axis=1))
    print(f"single-step floor (pixel MSE): {floor:.1f}")
    print(f"horizon-2x: {horizon}/{n_frames} frames (first t where pixel MSE vs ground truth > 2x floor)")
    print("MSE trace: " + " ".join(f"{m:.0f}" for m in mses[:24]))


@torch.no_grad()
def action_test(ckpt_path, vae_path, config_name, num_steps, sigma_stab, seed, out_dir):
    # same context + same sampler noise, vary only the action -> isolates how much the model reacts to input
    import imageio.v3 as iio
    device = pick_device("auto")
    cfg, vae, diff = load_models(ckpt_path, vae_path, config_name, device)
    T_ctx = cfg.dit.context_frames
    z_history, actions = initial_context(vae, T_ctx, seed, device)
    z_ctx = z_history[:, 1:]
    sigmas = edm_sigma_schedule(num_steps, cfg.diffusion.sigma_min, cfg.diffusion.sigma_max, device=device)

    names = ["NONE", "LEFT", "RIGHT"]
    frames = []
    for a in range(3):
        torch.manual_seed(seed)                                          # identical noise across actions
        full_actions = torch.tensor(actions[1:] + [a], dtype=torch.long, device=device)
        new_lat = sample_next_frame(diff, z_ctx, full_actions, sigmas, sigma_stab)
        frames.append(decode_latent(vae, new_lat))
    none = frames[0].astype(np.float32)
    print("action response (same context + same noise, vary action):")
    for a in range(3):
        d = float(np.abs(frames[a].astype(np.float32) - none).mean())
        print(f"  {a} {names[a]:11s} mean |delta| vs NONE = {d:.2f}/255")
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_dir / "action_test.png", np.concatenate(frames, axis=1))


@torch.no_grad()
def plausibility(ckpt_path, vae_path, config_name, n_frames, num_steps, sigma_stab, seed, out_dir):
    # reference-free Breakout plausibility -- the right metric for a chaotic ball (no ground-truth needed).
    # reads ball + brick state out of the GENERATED pixels and checks the model obeys the game's rules.
    import imageio.v3 as iio
    device = pick_device("auto")
    cfg, vae, diff = load_models(ckpt_path, vae_path, config_name, device)
    T_ctx = cfg.dit.context_frames
    torch.manual_seed(seed)

    pal = PALETTE["grey"]
    bg = np.array(DB16[pal["bg"]], dtype=int)
    row_colors = [np.array(DB16[i], dtype=int) for i in pal["rows"]]

    def detect_ball(f):
        m = (f[:, :, 0] > 200) & (f[:, :, 1] > 200) & (f[:, :, 2] > 200)
        m[:8 * SCALE, :] = False                            # drop the top HUD + lives strip (scales with the game)
        ys, xs = np.where(m)
        return (float(xs.mean()), float(ys.mean())) if len(xs) else None

    def brick_grid(f):
        g = np.zeros((BRICK_ROWS, BRICK_COLS), dtype=bool)
        for r in range(BRICK_ROWS):
            cy = BRICK_TOP + r * BRICK_H + BRICK_H // 2
            for c in range(BRICK_COLS):
                px = f[cy, c * BRICK_W + BRICK_W // 2].astype(int)
                g[r, c] = np.abs(px - row_colors[r]).sum() < np.abs(px - bg).sum()
        return g

    z_history, actions = initial_context(vae, T_ctx, seed, device)
    sigmas = edm_sigma_schedule(num_steps, cfg.diffusion.sigma_min, cfg.diffusion.sigma_max, device=device)
    cycle = [2, 2, 2, 2, 1, 1, 1, 1]
    balls, grids, frames = [], [], []
    for t in range(n_frames):
        a = cycle[t % len(cycle)]
        z_ctx = z_history[:, 1:]
        full_actions = torch.tensor(actions[1:] + [a], dtype=torch.long, device=device)
        new_lat = sample_next_frame(diff, z_ctx, full_actions, sigmas, sigma_stab)
        z_history = torch.cat([z_ctx, new_lat], dim=1)
        actions = actions[1:] + [a]
        f = decode_latent(vae, new_lat)
        frames.append(f)
        balls.append(detect_ball(f))
        grids.append(brick_grid(f))

    ball_rate = sum(b is not None for b in balls) / n_frames
    speeds = [((balls[i][0] - balls[i - 1][0]) ** 2 + (balls[i][1] - balls[i - 1][1]) ** 2) ** 0.5
              for i in range(1, n_frames) if balls[i] and balls[i - 1]]
    play_speeds = [s for s in speeds if s < 4 * BALL_SPEED]      # drop relaunch teleports
    mean_speed = float(np.mean(play_speeds)) if play_speeds else 0.0
    speed_cv = float(np.std(play_speeds) / (mean_speed + 1e-6)) if play_speeds else 0.0
    counts = [int(g.sum()) for g in grids]
    resurrections = sum(int((grids[i] & ~grids[i - 1]).sum())
                        for i in range(1, n_frames) if counts[i] - counts[i - 1] <= BRICK_COLS)

    print(f"ball detected:        {ball_rate * 100:.0f}% of frames")
    print(f"mean ball speed:      {mean_speed:.2f} px/frame  (real game = {BALL_SPEED})")
    print(f"ball speed variation: {speed_cv:.2f}  (low = constant speed / plausible physics)")
    print(f"bricks:               {counts[0]} -> {counts[-1]}  (broken over the rollout)")
    print(f"brick resurrections:  {resurrections}  (cells refilling without a reset; 0 = good world-state memory)")
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_dir / "plausibility_strip.png", np.concatenate(frames[:16], axis=1))


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
    p.add_argument("--measure-horizon", type=int, default=0, help="short-horizon exact-MSE drift vs the real game")
    p.add_argument("--action-test", action="store_true", help="action-response delta from a fixed context")
    p.add_argument("--plausibility", type=int, default=0, help="reference-free Breakout plausibility over N frames")
    p.add_argument("--out", type=str, default="assets/infer_smoke")
    args = p.parse_args()

    if args.plausibility > 0:
        plausibility(args.ckpt, args.vae, args.config, args.plausibility,
                     args.steps, args.sigma_stab, args.seed, args.out)
    elif args.action_test:
        action_test(args.ckpt, args.vae, args.config, args.steps, args.sigma_stab, args.seed, args.out)
    elif args.measure_horizon > 0:
        measure_horizon(args.ckpt, args.vae, args.config, args.measure_horizon,
                        args.steps, args.sigma_stab, args.seed, args.out)
    elif args.headless > 0:
        headless(args.ckpt, args.vae, args.config, args.headless,
                 args.steps, args.sigma_stab, args.seed, args.out)
    else:
        play(args.ckpt, args.vae, args.config, args.steps, args.sigma_stab, args.seed)
