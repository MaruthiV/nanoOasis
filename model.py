# nanoOasis spatiotemporal DiT. PROJECT.md §2.2 Stage 2 / DECISIONS D001-D007.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- RoPE ----

def precompute_rope_1d(seq_len: int, dim: int, base: float = 10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    angles = torch.arange(seq_len).float()[:, None] * freqs[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def precompute_rope_2d(h: int, w: int, head_dim: int, base: float = 10000.0):
    # split head_dim half-half: first half rotates by row, second half by column (D006)
    half = head_dim // 2
    rc1, rs1 = precompute_rope_1d(h, half, base)
    cc1, cs1 = precompute_rope_1d(w, half, base)
    return (rc1.repeat_interleave(w, 0), rs1.repeat_interleave(w, 0),
            cc1.repeat(h, 1),           cs1.repeat(h, 1))


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_1d(x, cos, sin):
    return x * cos + _rotate_half(x) * sin


def apply_rope_2d(x, row_cos, row_sin, col_cos, col_sin):
    half = x.shape[-1] // 2
    x_row = apply_rope_1d(x[..., :half], row_cos, row_sin)
    x_col = apply_rope_1d(x[..., half:], col_cos, col_sin)
    return torch.cat([x_row, x_col], dim=-1)


# ---- patchify ----

def patchify(x, p: int):
    B, T, C, H, W = x.shape
    h, w = H // p, W // p
    x = x.view(B, T, C, h, p, w, p)
    x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    return x.view(B, T, h * w, C * p * p)


def unpatchify(tokens, h: int, w: int, p: int, C: int):
    B, T = tokens.shape[:2]
    x = tokens.view(B, T, h, w, C, p, p)
    x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    return x.view(B, T, C, h * p, w * p)


# ---- timestep embedder ----

class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_dim: int | None = None):
        super().__init__()
        self.freq_dim = freq_dim or dim
        self.mlp = nn.Sequential(
            nn.Linear(self.freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        args = t.float()[..., None] * freqs
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(emb)


# ---- transformer block: spatial attn + temporal attn + MLP, all AdaLN-Zero ----

class Block(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, mlp_ratio: int):
        super().__init__()
        assert heads * head_dim == dim, (heads, head_dim, dim)
        self.heads, self.head_dim = heads, head_dim

        self.qkv_s = nn.Linear(dim, 3 * dim)
        self.proj_s = nn.Linear(dim, dim)
        self.ln_s = nn.LayerNorm(dim, elementwise_affine=False)

        self.qkv_t = nn.Linear(dim, 3 * dim)
        self.proj_t = nn.Linear(dim, dim)
        self.ln_t = nn.LayerNorm(dim, elementwise_affine=False)

        self.ln_m = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def _attn(self, x, qkv_proj, out_proj, rope, causal: bool):
        B, n, d = x.shape
        H, D = self.heads, self.head_dim
        q, k, v = qkv_proj(x).view(B, n, 3, H, D).permute(2, 0, 3, 1, 4).unbind(0)
        if len(rope) == 2:
            q = apply_rope_1d(q, *rope)
            k = apply_rope_1d(k, *rope)
        else:
            q = apply_rope_2d(q, *rope)
            k = apply_rope_2d(k, *rope)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return out_proj(out.transpose(1, 2).reshape(B, n, d))

    def forward(self, x, mods, sp_rope, tp_rope):
        g1, b1, a1, g2, b2, a2, g3, b3, a3 = mods
        B, T, n, d = x.shape

        # spatial attention -- per-frame, full self-attn
        h = self.ln_s(x) * (1 + g1.unsqueeze(2)) + b1.unsqueeze(2)
        h_attn = self._attn(h.view(B * T, n, d), self.qkv_s, self.proj_s, sp_rope, causal=False)
        x = x + a1.unsqueeze(2) * h_attn.view(B, T, n, d)

        # temporal attention -- per-spatial-position, causal across frames
        h = self.ln_t(x) * (1 + g2.unsqueeze(2)) + b2.unsqueeze(2)
        h_t = h.permute(0, 2, 1, 3).contiguous().view(B * n, T, d)
        h_attn = self._attn(h_t, self.qkv_t, self.proj_t, tp_rope, causal=True)
        x = x + a2.unsqueeze(2) * h_attn.view(B, n, T, d).permute(0, 2, 1, 3).contiguous()

        # MLP
        h = self.ln_m(x) * (1 + g3.unsqueeze(2)) + b3.unsqueeze(2)
        return x + a3.unsqueeze(2) * self.mlp(h)


# ---- DiT ----

# latent grid produced by the VAE -- C channels, H rows, W cols. PROJECT.md §2.2.
LATENT_C, LATENT_H, LATENT_W = 16, 12, 16


class DiT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.hidden_dim
        self.d = d
        self.depth = cfg.depth
        self.heads = cfg.heads
        self.head_dim = cfg.head_dim
        self.patch = cfg.patch_size
        self.context_frames = cfg.context_frames
        self.num_actions = cfg.num_actions

        self.latent_C = LATENT_C
        self.h = LATENT_H // self.patch
        self.w = LATENT_W // self.patch
        self.n_spat = self.h * self.w
        pix = self.patch * self.patch * self.latent_C

        self.in_proj = nn.Linear(pix, d)
        self.time_emb = TimestepEmbedder(d)
        self.action_emb = nn.Embedding(self.num_actions, d)

        # AdaLN-Zero -- DECISIONS D004 (zero-init weight AND bias) + D007 (shared modulator + per-block bias)
        self.adaln_mod = nn.Linear(d, 9 * d)
        nn.init.zeros_(self.adaln_mod.weight)
        nn.init.zeros_(self.adaln_mod.bias)
        self.adaln_block_bias = nn.Parameter(torch.zeros(self.depth, 9, d))

        self.blocks = nn.ModuleList([
            Block(d, self.heads, self.head_dim, cfg.mlp_ratio)
            for _ in range(self.depth)
        ])

        self.final_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, pix)
        # zero-init the final projection so the model predicts 0 at step 0 (DiT paper)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        rc, rs, cc, cs = precompute_rope_2d(self.h, self.w, self.head_dim)
        tc, ts = precompute_rope_1d(self.context_frames, self.head_dim)
        for name, val in dict(row_cos=rc, row_sin=rs, col_cos=cc, col_sin=cs,
                              time_cos=tc, time_sin=ts).items():
            self.register_buffer(name, val, persistent=False)

    def forward(self, x, t, action):
        # x: (B, T, C, H, W) noisy latent; t: (B, T) per-frame conditioning scalar
        # (EDM passes c_noise = 0.25·log σ here, not raw σ); action: (B, T) ints
        B, T = x.shape[:2]
        h = self.in_proj(patchify(x, self.patch))                    # (B, T, n_spat, d)
        c = self.time_emb(t) + self.action_emb(action)               # (B, T, d) -- D005
        shared = self.adaln_mod(F.silu(c)).view(B, T, 9, self.d)     # (B, T, 9, d)

        sp_rope = (self.row_cos, self.row_sin, self.col_cos, self.col_sin)
        tp_rope = (self.time_cos, self.time_sin)
        for i, blk in enumerate(self.blocks):
            mods = (shared + self.adaln_block_bias[i][None, None]).unbind(dim=2)
            h = blk(h, mods, sp_rope, tp_rope)

        out = self.out_proj(self.final_norm(h))                      # (B, T, n_spat, pix)
        return unpatchify(out, self.h, self.w, self.patch, self.latent_C)
