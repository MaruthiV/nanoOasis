from omegaconf import OmegaConf


def test_tiny_config_loads_with_required_sections_and_keys():
    cfg = OmegaConf.load("configs/tiny.yaml")
    for sec in ("vae", "dit", "diffusion", "training", "data"):
        assert sec in cfg, f"missing section: {sec}"
    # tier-defining knobs (docs/IMPLEMENTATION_PLAN.md tiny tier)
    assert cfg.vae.hidden_dim == 128
    assert (cfg.vae.enc_layers, cfg.vae.dec_layers) == (2, 3)
    assert cfg.vae.patch_size == 8
    assert cfg.vae.latent_channels == 16
    assert cfg.dit.hidden_dim == 128
    assert cfg.dit.depth == 4
    assert cfg.dit.context_frames == 4
    assert cfg.dit.num_actions == 3
    assert cfg.diffusion.forcing is True
    assert cfg.data.index_path == "data/smoke/index.parquet"
