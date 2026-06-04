import numpy as np
import torch
from omegaconf import OmegaConf

from game import W, H
from vae import VAE


def _tiny_cfg():
    return OmegaConf.load("configs/tiny.yaml").vae


def test_vae_tiny_param_count_around_1M():
    m = VAE(_tiny_cfg())
    n = sum(p.numel() for p in m.parameters())
    assert 800_000 < n < 1_500_000, f"VAE tiny has {n:,} params"


def test_vae_forward_shapes_round_trip():
    m = VAE(_tiny_cfg())
    x = torch.randint(0, 255, (2, H, W, 3), dtype=torch.uint8)
    recon, mu, logvar = m(x)
    assert recon.shape == (2, H, W, 3)
    n = m.Hp * m.Wp                         # Hp*Wp tokens (768 at 256x192, patch 8)
    assert mu.shape == (2, n, 16)          # 16 latent channels
    assert logvar.shape == (2, n, 16)


def test_vae_loss_runs_and_backprops():
    m = VAE(_tiny_cfg())
    x = torch.randint(0, 255, (2, H, W, 3), dtype=torch.uint8)
    recon, mu, logvar = m(x)
    loss, info = m.loss(x, recon, mu, logvar)
    loss.backward()
    assert torch.isfinite(loss)
    assert "l1" in info and "kl" in info
