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
from game import (Game, _poll_action, safe_actions, W, H, CELL, GRID_COLS, GRID_ROWS,
                  DIRS, UP, DOWN, LEFT, RIGHT, BG_COLOR, BODY_COLOR, HEAD_COLOR, APPLE_COLOR)

# fixed exploratory direction cycle for headless rollouts (a box sweep)
EVAL_CYCLE = [RIGHT] * 3 + [DOWN] * 3 + [LEFT] * 3 + [UP] * 3
ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT")


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


def make_schedule(cfg, num_steps: int, sigma_max: float, device: str) -> torch.Tensor:
    # sampling sigma_max decoupled from the training sigma_max 80 (D029): DIAMOND samples from
    # 10x sigma_data (5.0 with sigma_data 0.5); 0 = use that default.
    smax = sigma_max if sigma_max > 0 else 10.0 * float(cfg.diffusion.sigma_data)
    return edm_sigma_schedule(num_steps, cfg.diffusion.sigma_min, smax, device=device)


@torch.no_grad()
def sample_next_frame(diff: EDMDiffusion, z_ctx: torch.Tensor,
                      full_actions: torch.Tensor, sigmas: torch.Tensor,
                      sigma_stab: float = 0.1, sampler: str = "euler",
                      cfg_scale: float | None = None) -> torch.Tensor:
    # z_ctx: (B=1, T-1, C, Hp, Wp) clean context; full_actions: (T,) ints incl. the new action.
    # sampler: "euler" = 1st-order deterministic, the DIAMOND-proven few-step regime (D029);
    #          "heun" = Karras 2nd-order, quality fallback at >= 8 steps.
    # cfg_scale: classifier-free guidance on ACTION (the trained null row, D029); >1 amplifies the
    # action's pull when drift drowns control. Default off; set via NANO_CFG=1.5 etc.
    if cfg_scale is None:
        cfg_scale = float(os.environ.get("NANO_CFG", "0"))
    B = z_ctx.shape[0]
    T = z_ctx.shape[1] + 1
    C, Hp, Wp = z_ctx.shape[2:]
    device = z_ctx.device

    if full_actions.dim() == 1:
        full_actions = full_actions.unsqueeze(0)

    noisy_ctx = z_ctx + sigma_stab * torch.randn_like(z_ctx)
    x_new = torch.randn(B, 1, C, Hp, Wp, device=device) * sigmas[0]
    null_actions = torch.full_like(full_actions, diff.model.num_actions)

    def deriv(x, sigma):                                       # score-like estimate at the new frame
        sig = torch.cat([torch.full((B, T - 1), sigma_stab, device=device),
                         torch.full((B, 1),     sigma,      device=device)], dim=1)
        x_in = torch.cat([noisy_ctx, x], dim=1)
        D = diff.denoise(x_in, sig, full_actions)
        if cfg_scale > 0:
            D_null = diff.denoise(x_in, sig, null_actions)     # whole-window null matches the dropout regime
            D = D_null + cfg_scale * (D - D_null)
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
        torch.mps.empty_cache()                                # free per-frame so long rollouts don't OOM
    return x_new


@torch.no_grad()
def decode_latent(vae: VAE, z: torch.Tensor) -> np.ndarray:
    # z: (1, 1, C, Hp, Wp) -> (H, W, 3) uint8
    z_tok = z.squeeze(0).squeeze(0).permute(1, 2, 0).reshape(-1, vae.latent_channels).unsqueeze(0)
    recon = vae.decode(z_tok)
    return ((recon.squeeze(0).clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()


@torch.no_grad()
def initial_context(vae: VAE, T_ctx: int, seed: int, device: str):
    # drive the real game with safe-random moves so the context contains live play, not a reset loop
    g = Game(seed=seed)
    rng = np.random.default_rng(seed ^ 0xB07)
    frames, actions = [], []
    for _ in range(T_ctx):
        safe = safe_actions(g)
        a = int(rng.choice(safe)) if safe else int(rng.integers(0, 4))
        frames.append(g.step(a)[0].copy())
        actions.append(a)
    flat = torch.from_numpy(np.stack(frames)).to(device)
    mu, _ = vae.encode(flat)
    z = mu.view(T_ctx, vae.Hp, vae.Wp, vae.latent_channels).permute(0, 3, 1, 2).contiguous()
    return z.unsqueeze(0), actions, g                          # g returned for ground-truth continuation


def load_models(ckpt_path: str, vae_path: str, config_name: str, device: str):
    cfg = OmegaConf.load(f"configs/{config_name}.yaml")
    vae = VAE(cfg.vae).to(device).eval()
    vae.load_state_dict(torch.load(vae_path, weights_only=False, map_location=device)["model"])
    for p in vae.parameters():
        p.requires_grad = False

    model = DiT(cfg.dit).to(device).eval()
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    # EMA(0.9999) needs ~30k+ steps to forget the init; short runs must eval RAW weights (overfit-gate lesson)
    use_ema = not os.environ.get("NANO_RAW_WEIGHTS")
    model.load_state_dict(ckpt.get("ema", ckpt["model"]) if use_ema else ckpt["model"])
    for p in model.parameters():
        p.requires_grad = False

    diff = EDMDiffusion(model, cfg.diffusion).to(device).eval()
    return cfg, vae, diff


# ---- Snake cell readout (the eval harness, D029) ----

_STATE_COLORS = np.array([BG_COLOR, BODY_COLOR, HEAD_COLOR, APPLE_COLOR], dtype=np.float32)
EMPTY, BODY, HEAD, APPLE = 0, 1, 2, 3


def read_grid(f: np.ndarray) -> tuple[np.ndarray, float]:
    # classify each cell's center patch by nearest canonical color -> (GRID_ROWS, GRID_COLS) state ids,
    # plus the mean color distance (high = blended/uncommitted cells, the hedging failure signature)
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    dist = 0.0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            patch = f[r * CELL + 12:r * CELL + 20, c * CELL + 12:c * CELL + 20]
            mean = patch.astype(np.float32).mean(axis=(0, 1))
            d = np.abs(_STATE_COLORS - mean).sum(axis=1)
            grid[r, c] = int(d.argmin())
            dist += float(d.min())
    return grid, dist / (GRID_ROWS * GRID_COLS)


def _head(grid: np.ndarray):
    ys, xs = np.where(grid == HEAD)
    return (int(xs[0]), int(ys[0])) if len(xs) == 1 else None


def _connected(grid: np.ndarray):
    # body+head must form one 4-connected component (None if the head isn't unique)
    cells = {(c, r) for r in range(GRID_ROWS) for c in range(GRID_COLS) if grid[r, c] in (BODY, HEAD)}
    h = _head(grid)
    if h is None:
        return None
    seen, stack = {h}, [h]
    while stack:
        c0, r0 = stack.pop()
        for dc, dr in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            n = (c0 + dc, r0 + dr)
            if n in cells and n not in seen:
                seen.add(n)
                stack.append(n)
    return len(seen) == len(cells)


@torch.no_grad()
def snake_eval(ckpt_path, vae_path, config_name, n_frames, num_steps, sigma_stab, seed, out_dir,
               sampler="euler", sigma_max=0.0):
    # reference-free Snake rules check: read the grid out of the GENERATED pixels and verify the model
    # obeys the game -- one head, one apple, connected body, legal length changes, action obedience.
    # The numbers are hints; the GATE is playing it (memory: judge-by-playing).
    import imageio.v3 as iio
    device = pick_device("auto")
    cfg, vae, diff = load_models(ckpt_path, vae_path, config_name, device)
    T_ctx = cfg.dit.context_frames
    torch.manual_seed(seed)

    z_history, actions, _ = initial_context(vae, T_ctx, seed, device)
    sigmas = make_schedule(cfg, num_steps, sigma_max, device)
    grids, blends, frames = [], [], []
    for t in range(n_frames):
        a = EVAL_CYCLE[t % len(EVAL_CYCLE)]
        z_ctx = z_history[:, 1:]
        full_actions = torch.tensor(actions[1:] + [a], dtype=torch.long, device=device)
        new_lat = sample_next_frame(diff, z_ctx, full_actions, sigmas, sigma_stab, sampler)
        z_history = torch.cat([z_ctx, new_lat], dim=1)
        actions = actions[1:] + [a]
        f = decode_latent(vae, new_lat)
        frames.append(f)
        g, b = read_grid(f)
        grids.append(g)
        blends.append(b)

    heads = [_head(g) for g in grids]
    one_head = sum(int((g == HEAD).sum()) == 1 for g in grids) / n_frames
    one_apple = sum(int((g == APPLE).sum()) == 1 for g in grids) / n_frames
    conn = [c for c in (_connected(g) for g in grids) if c is not None]
    connected = (sum(conn) / len(conn)) if conn else 0.0

    lengths = [int(((g == BODY) | (g == HEAD)).sum()) for g in grids]
    deltas = [lengths[i] - lengths[i - 1] for i in range(1, n_frames)]
    # legal deltas: 0 (move), +1 (eat); a reset lands back at the start length 3
    illegal = sum(1 for i, d in enumerate(deltas, 1) if d not in (0, 1) and lengths[i] != 3)

    # action obedience: the head must move one cell in the commanded direction (reversals keep heading)
    obeyed = tested = 0
    for t in range(1, n_frames):
        h0, h1 = heads[t - 1], heads[t]
        if h0 is None or h1 is None:
            continue
        move = (h1[0] - h0[0], h1[1] - h0[1])
        if abs(move[0]) + abs(move[1]) != 1:
            continue                                            # reset/teleport -- not an obedience sample
        cmd = EVAL_CYCLE[t % len(EVAL_CYCLE)]
        heading = None
        if t >= 2 and heads[t - 2] is not None:
            prev = (h0[0] - heads[t - 2][0], h0[1] - heads[t - 2][1])
            heading = DIRS.index(prev) if prev in DIRS else None
        eff = heading if (heading is not None and DIRS[cmd] == (-DIRS[heading][0], -DIRS[heading][1])) else cmd
        tested += 1
        obeyed += int(move == DIRS[eff])

    print(f"one head:        {one_head * 100:.0f}% of frames")
    print(f"one apple:       {one_apple * 100:.0f}% of frames")
    print(f"body connected:  {connected * 100:.0f}% (of frames with a unique head)")
    print(f"illegal length:  {illegal} frames (len change not 0/+1 and not a reset)")
    print(f"length range:    {min(lengths)}..{max(lengths)}")
    print(f"action obeyed:   {obeyed}/{tested} clean moves")
    print(f"cell blend dist: {np.mean(blends):.1f} (low = committed cells; high = hedged mush)")
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_dir / "eval_strip.png", np.concatenate(frames[:16], axis=1))
    for i, f in enumerate(frames):
        iio.imwrite(out_dir / f"frame_{i:04d}.png", f)


def headless(ckpt_path, vae_path, config_name, n_frames, num_steps, sigma_stab, seed, out_dir,
             sampler="euler", sigma_max=0.0):
    import imageio.v3 as iio
    device = pick_device("auto")
    cfg, vae, diff = load_models(ckpt_path, vae_path, config_name, device)
    T_ctx = cfg.dit.context_frames

    z_history, actions, _ = initial_context(vae, T_ctx, seed, device)
    sigmas = make_schedule(cfg, num_steps, sigma_max, device)

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_out: list[np.ndarray] = []
    for i in range(n_frames):
        action = EVAL_CYCLE[i % len(EVAL_CYCLE)]
        z_ctx = z_history[:, 1:]
        full_actions = torch.tensor(actions[1:] + [action], dtype=torch.long, device=device)
        new_lat = sample_next_frame(diff, z_ctx, full_actions, sigmas, sigma_stab, sampler)
        z_history = torch.cat([z_ctx, new_lat], dim=1)
        actions = actions[1:] + [action]
        frame = decode_latent(vae, new_lat)
        iio.imwrite(out_dir / f"frame_{i:04d}_a{action}.png", frame)
        frames_out.append(frame)
    # also write a stitched strip for quick eyeball
    strip = np.concatenate(frames_out, axis=1)
    iio.imwrite(out_dir / "strip.png", strip)
    diffs = [float(np.abs(frames_out[i].astype(int) - frames_out[i - 1].astype(int)).mean())
             for i in range(1, len(frames_out))]
    print(f"wrote {n_frames} frames + strip.png to {out_dir}")
    print(f"mean abs frame-to-frame delta: {sum(diffs)/len(diffs):.2f}/255 (>0 => model is producing change)")


@torch.no_grad()
def measure_horizon(ckpt_path, vae_path, config_name, n_frames, num_steps, sigma_stab, seed, out_dir,
                    sampler="euler", sigma_max=0.0):
    # autoregressive horizon-2x: roll the model and the real game in lockstep on the same actions.
    # NOTE: apple respawns are sampled, so model and game legitimately diverge at the first eat/death --
    # this is a sanity tool, not the gate.
    import imageio.v3 as iio
    device = pick_device("auto")
    cfg, vae, diff = load_models(ckpt_path, vae_path, config_name, device)
    T_ctx = cfg.dit.context_frames
    torch.manual_seed(seed)

    z_history, actions, g = initial_context(vae, T_ctx, seed, device)
    action_seq = [EVAL_CYCLE[i % len(EVAL_CYCLE)] for i in range(n_frames)]
    gt = [g.step(a)[0].copy() for a in action_seq]
    sigmas = make_schedule(cfg, num_steps, sigma_max, device)

    mses, pairs = [], []
    for t, a in enumerate(action_seq):
        z_ctx = z_history[:, 1:]
        full_actions = torch.tensor(actions[1:] + [a], dtype=torch.long, device=device)
        new_lat = sample_next_frame(diff, z_ctx, full_actions, sigmas, sigma_stab, sampler)
        z_history = torch.cat([z_ctx, new_lat], dim=1)
        actions = actions[1:] + [a]
        m = decode_latent(vae, new_lat)
        truth = gt[t]
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
def action_test(ckpt_path, vae_path, config_name, num_steps, sigma_stab, seed, out_dir,
                sampler="euler", sigma_max=0.0):
    # same context + same sampler noise, vary only the action -> isolates how much the model reacts to input
    import imageio.v3 as iio
    device = pick_device("auto")
    cfg, vae, diff = load_models(ckpt_path, vae_path, config_name, device)
    T_ctx = cfg.dit.context_frames
    z_history, actions, _ = initial_context(vae, T_ctx, seed, device)
    z_ctx = z_history[:, 1:]
    sigmas = make_schedule(cfg, num_steps, sigma_max, device)

    frames = []
    for a in range(4):
        torch.manual_seed(seed)                                          # identical noise across actions
        full_actions = torch.tensor(actions[1:] + [a], dtype=torch.long, device=device)
        new_lat = sample_next_frame(diff, z_ctx, full_actions, sigmas, sigma_stab, sampler)
        frames.append(decode_latent(vae, new_lat))
    base = frames[0].astype(np.float32)
    print("action response (same context + same noise, vary action; deltas vs UP):")
    for a in range(4):
        d = float(np.abs(frames[a].astype(np.float32) - base).mean())
        print(f"  {a} {ACTION_NAMES[a]:6s} mean |delta| vs UP = {d:.2f}/255")
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_dir / "action_test.png", np.concatenate(frames, axis=1))


def play(ckpt_path, vae_path, config_name, num_steps, sigma_stab, seed, sampler="euler", sigma_max=0.0,
         fps: int = 4):
    import pygame
    device = pick_device("auto")
    cfg, vae, diff = load_models(ckpt_path, vae_path, config_name, device)
    T_ctx = cfg.dit.context_frames

    z_history, actions, _ = initial_context(vae, T_ctx, seed, device)
    sigmas = make_schedule(cfg, num_steps, sigma_max, device)

    pygame.init()
    screen = pygame.display.set_mode((W * 4, H * 4))
    pygame.display.set_caption("nanoOasis (model)")
    clock = pygame.time.Clock()

    action = actions[-1]
    running = True
    while running:
        action, quit_req = _poll_action(pygame, pygame.event.get(), action)
        if quit_req:
            break

        z_ctx = z_history[:, 1:]
        full_actions = torch.tensor(actions[1:] + [action], dtype=torch.long, device=device)
        new_lat = sample_next_frame(diff, z_ctx, full_actions, sigmas, sigma_stab, sampler)
        z_history = torch.cat([z_ctx, new_lat], dim=1)
        actions = actions[1:] + [action]

        frame = decode_latent(vae, new_lat)
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))   # H005: surfarray is (W,H,3)
        screen.blit(pygame.transform.scale(surf, (W * 4, H * 4)), (0, 0))
        pygame.display.flip()
        clock.tick(fps)                                # tick = one cell move; 4/s per user play-test (D029)

    pygame.quit()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--vae", type=str, required=True)
    p.add_argument("--config", type=str, default="tiny")
    p.add_argument("--steps", type=int, default=4, help="sampler steps per frame (D029: Euler 3-4)")
    p.add_argument("--sampler", type=str, default="euler", choices=["euler", "heun"])
    p.add_argument("--sigma-max", type=float, default=0.0, help="sampling sigma_max; 0 = 10x sigma_data (D029)")
    p.add_argument("--sigma-stab", type=float, default=0.1)
    p.add_argument("--fps", type=int, default=4, help="play-mode tick rate (cells/s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", type=int, default=0, help="write N frames + strip.png, no window")
    p.add_argument("--measure-horizon", type=int, default=0, help="short-horizon exact-MSE drift vs the real game")
    p.add_argument("--action-test", action="store_true", help="action-response delta from a fixed context")
    p.add_argument("--eval", type=int, default=0, help="reference-free Snake rules check over N frames")
    p.add_argument("--out", type=str, default="assets/infer_smoke")
    args = p.parse_args()

    if args.eval > 0:
        snake_eval(args.ckpt, args.vae, args.config, args.eval,
                   args.steps, args.sigma_stab, args.seed, args.out, args.sampler, args.sigma_max)
    elif args.action_test:
        action_test(args.ckpt, args.vae, args.config, args.steps, args.sigma_stab, args.seed,
                    args.out, args.sampler, args.sigma_max)
    elif args.measure_horizon > 0:
        measure_horizon(args.ckpt, args.vae, args.config, args.measure_horizon,
                        args.steps, args.sigma_stab, args.seed, args.out, args.sampler, args.sigma_max)
    elif args.headless > 0:
        headless(args.ckpt, args.vae, args.config, args.headless,
                 args.steps, args.sigma_stab, args.seed, args.out, args.sampler, args.sigma_max)
    else:
        play(args.ckpt, args.vae, args.config, args.steps, args.sigma_stab, args.seed,
             args.sampler, args.sigma_max, args.fps)
