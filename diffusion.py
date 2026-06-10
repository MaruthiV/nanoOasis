# nanoOasis EDM preconditioning + Diffusion Forcing per-frame sigma.
# DECISIONS D001 (EDM), D002 (Diffusion Forcing). Karras EDM Table 1.

import math

import torch
import torch.nn as nn


def _bcast(v: torch.Tensor) -> torch.Tensor:
    # (B, T) -> (B, T, 1, 1, 1) for broadcasting over a latent (C, H, W)
    return v[:, :, None, None, None]


class EDMDiffusion(nn.Module):
    """EDM-preconditioned denoiser. The inner model returns a raw residual F_θ;
    this class wraps it as D(x; σ) = c_skip·x + c_out·F_θ(c_in·x, c_noise, a)
    and exposes loss + per-frame σ sampling for Diffusion Forcing."""

    def __init__(self, model: nn.Module, cfg):
        super().__init__()
        self.model = model
        self.sigma_data = float(cfg.sigma_data)
        self.sigma_min = float(cfg.sigma_min)
        self.sigma_max = float(cfg.sigma_max)
        self.p_mean = float(cfg.p_mean)
        self.p_std = float(cfg.p_std)
        self.forcing = bool(cfg.get("forcing", True))   # D002; False = one shared sigma per window (M2 ablation)
        self.precond_mode = str(cfg.get("precond", "edm"))   # M5; "eps" = naive epsilon-prediction baseline
        self.motion_weight = float(cfg.get("motion_weight", 0.0))   # R5/D018; >0 up-weights moving regions (the ball)
        # context-noise augmentation (D024): >0 trains the context frames (all but the last/target) at LOW noise,
        # matching the autoregressive inference regime, so the model learns to predict the next frame from
        # IMPERFECT history instead of assuming a clean past -> fixes the ball-fade/drift (DIAMOND/GameNGen).
        self.context_noise_min = float(cfg.get("context_noise_min", 0.01))
        self.context_noise_max = float(cfg.get("context_noise_max", 0.0))   # 0 = off (old full-DF behavior)
        # static-consistency (D025): >0 strongly nails the target-frame regions that DON'T move vs the previous
        # frame (idle bricks/background) so they stay constant instead of flickering every frame. static_thresh
        # = the per-frame-normalized motion below which a region counts as static.
        self.static_weight = float(cfg.get("static_weight", 0.0))
        self.static_thresh = float(cfg.get("static_thresh", 0.1))
        # confine the static loss to the TOP fraction of rows (the brick region) so it can't freeze the
        # ball's lower play area (D025·1: the launch run over-stabilized -> ball died; bricks-only fixes it).
        self.static_region = float(cfg.get("static_region", 1.0))   # 1.0 = whole frame; <1 = top rows only

    def sample_sigma(self, B: int, T: int, device) -> torch.Tensor:
        # log-σ ~ N(P_mean, P_std). Diffusion Forcing (D002): per-frame (B, T).
        # forcing=False -> sample one σ per window and share it across frames (flat-σ, M2 ablation).
        log_sigma = self.p_mean + self.p_std * torch.randn(B, T if self.forcing else 1, device=device)
        sigma = log_sigma.exp().clamp(self.sigma_min, self.sigma_max)
        return sigma if self.forcing else sigma.expand(B, T)

    def precond(self, sigma: torch.Tensor):
        # Karras EDM Table 1. sigma: (B, T) -> 4x (B, T)
        sd = self.sigma_data
        denom_sq = sigma * sigma + sd * sd
        denom = denom_sq.sqrt()
        c_skip = (sd * sd) / denom_sq
        c_out = sigma * sd / denom
        c_in = 1.0 / denom
        c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in, c_noise

    def denoise(self, x_noisy: torch.Tensor, sigma: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # x_noisy: (B, T, C, H, W); sigma: (B, T); action: (B, T) ints. Returns the denoised x0 estimate.
        c_skip, c_out, c_in, c_noise = self.precond(sigma)
        if self.precond_mode == "eps":
            # M5 baseline: naive epsilon-prediction (no EDM in/out scaling); x0 = x_noisy - sigma*eps.
            eps = self.model(x_noisy, c_noise, action)
            return x_noisy - _bcast(sigma) * eps
        f_out = self.model(_bcast(c_in) * x_noisy, c_noise, action)
        return _bcast(c_skip) * x_noisy + _bcast(c_out) * f_out

    def loss(self, x_clean: torch.Tensor, action: torch.Tensor, sigma: torch.Tensor | None = None):
        # x_clean: (B, T, C, H, W); action: (B, T)
        B, T = x_clean.shape[:2]
        if sigma is None:
            sigma = self.sample_sigma(B, T, x_clean.device)
        if self.context_noise_max > 0 and T > 1:
            # the last frame is the prediction target (full DF noise); the context frames get a low
            # log-uniform noise in [context_noise_min, context_noise_max] so training matches inference
            # (context held at small sigma_stab) and the model learns to correct imperfect history (D024).
            lo, hi = math.log(self.context_noise_min), math.log(self.context_noise_max)
            u = torch.rand(B, T - 1, device=x_clean.device)
            sigma = sigma.clone()
            sigma[:, :-1] = (lo + u * (hi - lo)).exp()
        noise = torch.randn_like(x_clean)
        x_noisy = x_clean + _bcast(sigma) * noise
        D = self.denoise(x_noisy, sigma, action)
        # EDM loss weighting -- equal-norm contribution across noise levels
        w = (sigma * sigma + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        per_elem = (D - x_clean) ** 2
        weighted = _bcast(w) * per_elem
        if self.motion_weight > 0:
            # up-weight regions that move frame-to-frame (the ball, paddle, brick-breaks) so the model
            # can't minimize loss by hedging the tiny dynamic part away into a static scene (R5, D018).
            mot = (x_clean[:, 1:] - x_clean[:, :-1]).abs().mean(dim=2, keepdim=True)   # (B, T-1, 1, H, W)
            mot = torch.cat([torch.zeros_like(mot[:, :1]), mot], dim=1)               # t=0 has no reference
            mot_norm = mot / (mot.amax(dim=(3, 4), keepdim=True) + 1e-6)              # per-frame spatial max -> [0,1]
            weighted = weighted * (1.0 + self.motion_weight * mot_norm)
        per_frame = weighted.mean(dim=(2, 3, 4))                          # (B, T) -- for the sigma-bucket diagnostic
        if self.context_noise_max > 0 and T > 1:
            # predict-next objective (GameNGen/DIAMOND): loss on the TARGET frame only -- the context frames
            # are noised CONDITIONING, not denoising targets. Required because the EDM weight ~1/sigma^2 makes
            # the low-noise context frames otherwise dominate the loss and starve the prediction (D024).
            loss = weighted[:, -1].mean()
        else:
            loss = weighted.mean()
        if self.static_weight > 0 and T > 1:
            # static-consistency (D025): up-weight the target frame's STATIC regions (no motion vs the previous
            # frame) so idle bricks/background are nailed and don't flicker -- only moving/touched regions change.
            mot = (x_clean[:, -1] - x_clean[:, -2]).abs().mean(dim=1, keepdim=True)   # (B, 1, H, W)
            mot_norm = mot / (mot.amax(dim=(2, 3), keepdim=True) + 1e-6)
            static = (mot_norm < self.static_thresh).float()
            if self.static_region < 1.0:                            # bricks-only: zero the mask below the brick rows
                cut = int(x_clean.shape[3] * self.static_region)
                static[:, :, cut:, :] = 0.0
            loss = loss + self.static_weight * (static * (D[:, -1] - x_clean[:, -1]) ** 2).mean()
        return loss, {
            "sigma_mean": float(sigma.mean().detach()),
            "sigma_std":  float(sigma.std().detach()),
            "sigma_flat": sigma.detach().flatten(),                       # (B*T,)
            "loss_flat":  per_frame.detach().flatten(),                   # (B*T,) weighted per-frame loss
        }
