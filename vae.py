# nanoOasis ViT-VAE. KL-Gaussian bottleneck. PROJECT.md §2.2 Stage 1.

import torch
import torch.nn as nn
import torch.nn.functional as F

from game import H, W


class ViTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        y = self.ln1(x)
        a, _ = self.attn(y, y, y, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, P = cfg.hidden_dim, cfg.patch_size
        self.P, self.Hp, self.Wp = P, H // P, W // P
        self.N = self.Hp * self.Wp                              # 192 tokens
        self.latent_channels = cfg.latent_channels
        self.kl_beta = cfg.kl_beta
        pix = P * P * 3                                          # 192 for P=8

        self.patch_proj = nn.Linear(pix, d)
        self.pos_enc = nn.Parameter(torch.zeros(1, self.N, d))
        self.enc_blocks = nn.ModuleList(
            ViTBlock(d, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.enc_layers)
        )
        self.enc_norm = nn.LayerNorm(d)
        self.to_latent = nn.Linear(d, 2 * cfg.latent_channels)   # mu + logvar

        self.from_latent = nn.Linear(cfg.latent_channels, d)
        self.pos_dec = nn.Parameter(torch.zeros(1, self.N, d))
        self.dec_blocks = nn.ModuleList(
            ViTBlock(d, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.dec_layers)
        )
        self.dec_norm = nn.LayerNorm(d)
        self.unpatch_proj = nn.Linear(d, pix)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, W, 3) uint8 or float -> (B, N, P*P*3) float in [-1, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 127.5 - 1.0
        B = x.shape[0]
        x = x.view(B, self.Hp, self.P, self.Wp, self.P, 3)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B, self.N, self.P * self.P * 3)

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        x = tokens.view(B, self.Hp, self.Wp, self.P, self.P, 3)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B, self.Hp * self.P, self.Wp * self.P, 3)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.patch_proj(self._patchify(x)) + self.pos_enc
        for blk in self.enc_blocks:
            h = blk(h)
        mu, logvar = self.to_latent(self.enc_norm(h)).chunk(2, dim=-1)
        return mu, logvar                                        # each (B, N, latent_channels)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + (0.5 * logvar).exp() * torch.randn_like(mu)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z) + self.pos_dec
        for blk in self.dec_blocks:
            h = blk(h)
        return self._unpatchify(self.unpatch_proj(self.dec_norm(h)))

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        target = x.float() / 127.5 - 1.0 if x.dtype == torch.uint8 else x
        l1 = F.l1_loss(recon, target)
        # KL to N(0, I), mean over batch + tokens
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return l1 + self.kl_beta * kl, {"l1": float(l1.detach()), "kl": float(kl.detach())}
