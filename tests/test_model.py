import torch
from omegaconf import OmegaConf

from model import DiT, patchify, unpatchify


def _tiny_cfg():
    return OmegaConf.load("configs/tiny.yaml").dit


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
