import torch
from omegaconf import OmegaConf

from model import DiT, patchify, unpatchify
from diffusion import EDMDiffusion


def _tiny_cfg():
    return OmegaConf.load("configs/tiny.yaml").dit


def _tiny_diff():
    cfg = OmegaConf.load("configs/tiny.yaml")
    return EDMDiffusion(DiT(cfg.dit), cfg.diffusion)


def test_dit_shapes_and_init():
    m = DiT(_tiny_cfg())
    # AdaLN-Zero gate: BOTH weight and bias must be zero (DECISIONS D004 / BUGS H001)
    assert m.adaln_mod.weight.abs().sum() == 0
    assert m.adaln_mod.bias.abs().sum() == 0
    assert m.adaln_block_bias.abs().sum() == 0
    # final projection also zero-init -- model predicts all-zeros at step 0
    assert m.out_proj.weight.abs().sum() == 0
    assert m.out_proj.bias.abs().sum() == 0

    # forward pass on the tiny tier dummy
    B, T = 2, 4
    x = torch.randn(B, T, 16, 12, 16)
    sigma = torch.randn(B, T)
    action = torch.randint(0, 6, (B, T))
    out = m(x, sigma, action)
    assert out.shape == x.shape

    # identity-at-init: with all modulations and out_proj zeroed, model output is exactly zero
    assert out.abs().sum() == 0, "identity-at-init broken"


def test_dit_tiny_param_count_under_2M():
    m = DiT(_tiny_cfg())
    n = sum(p.numel() for p in m.parameters())
    assert 800_000 < n < 2_000_000, f"DiT tiny has {n:,} params"


def test_patchify_unpatchify_round_trip():
    x = torch.randn(2, 4, 16, 12, 16)
    tokens = patchify(x, 2)
    assert tokens.shape == (2, 4, 48, 64)
    back = unpatchify(tokens, 6, 8, 2, 16)
    assert torch.allclose(back, x)


# ---- T4: EDM preconditioning + Diffusion Forcing ----


def test_diffusion_forcing_sigma_sampling_returns_per_frame_shape():
    diff = _tiny_diff()
    sigma = diff.sample_sigma(2, 4, device="cpu")
    assert sigma.shape == (2, 4)
    assert (sigma > 0).all()
    assert (sigma >= diff.sigma_min).all() and (sigma <= diff.sigma_max).all()


def test_diffusion_forcing_denoise_output_shape_matches_input():
    diff = _tiny_diff()
    B, T = 2, 4
    x = torch.randn(B, T, 16, 12, 16)
    sigma = torch.full((B, T), 1.0)
    action = torch.randint(0, 6, (B, T))
    D = diff.denoise(x, sigma, action)
    assert D.shape == x.shape


def test_diffusion_forcing_c_shape_is_per_frame_not_per_window():
    # gate against BUGS H002 -- per-frame c is the *whole point* of Diffusion Forcing
    m = DiT(_tiny_cfg())
    B, T = 2, 4
    sigma = torch.randn(B, T)
    action = torch.randint(0, 6, (B, T))
    c = m.time_emb(sigma) + m.action_emb(action)
    assert c.shape == (B, T, m.d), f"c.shape={c.shape}; expected (B={B}, T={T}, d={m.d})"


def test_diffusion_forcing_loss_runs_and_backprops():
    diff = _tiny_diff()
    x = torch.randn(2, 4, 16, 12, 16)
    action = torch.randint(0, 6, (2, 4))
    loss, info = diff.loss(x, action)
    loss.backward()
    assert torch.isfinite(loss)
    assert "sigma_mean" in info and "sigma_std" in info
