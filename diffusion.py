# nanoOasis EDM preconditioning + Diffusion Forcing per-frame sigma.
# DECISIONS D001 (EDM), D002 (Diffusion Forcing). Karras EDM Table 1.

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
        loss = weighted.mean()
        per_frame = weighted.mean(dim=(2, 3, 4))                          # (B, T) -- for the sigma-bucket diagnostic
        return loss, {
            "sigma_mean": float(sigma.mean().detach()),
            "sigma_std":  float(sigma.std().detach()),
            "sigma_flat": sigma.detach().flatten(),                       # (B*T,)
            "loss_flat":  per_frame.detach().flatten(),                   # (B*T,) weighted per-frame loss
        }
