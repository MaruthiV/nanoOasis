# nanoOasis

**It's Snake — but there is no game engine.** Every frame you see is generated, one at a time, by a
13.5M-parameter diffusion model reacting to your arrow keys. The game logic, the physics, the apple, the
growing body — none of it is coded. A neural net learned the whole game from pixels and now *is* the game.

nanoOasis is the **"nanoGPT" of diffusion world models**: a small, readable, from-scratch reference
implementation of the Oasis / GameNGen / DIAMOND paradigm — the entire pipeline in ~2,400 lines of Python,
trainable end-to-end for **under $50**, and playable in your browser.

> ▶ **[Code on GitHub](https://github.com/MaruthiV/nanoOasis)** · play it locally (see [Quickstart](#quickstart)) · live demo + blog coming soon

---

## How it works

A diffusion **world model** predicts the next frame of a game given the recent frames and your action. Roll
that forward and you can *play* a game that has no engine behind it — the model improvises each frame.

```
 game.py (Snake)          ViT-VAE                 spatiotemporal DiT              browser
 ───────────────   →   compress 256×192   →   predict next latent given   →   decode + display
 generates data        frame → 48 latent      past 8 latents + action         (WebGPU, ~4 fps)
                       tokens (1 cell =        (EDM + Diffusion Forcing,
                        1 token)                Euler, 4 steps)
```

1. **`game.py`** is a real grid Snake. It only exists to generate training data — random + apple-seeking
   bots play millions of frames.
2. A small **ViT-VAE** compresses each 256×192 frame to a 16×24×32 latent. The grid is designed so **one
   game cell maps to exactly one DiT token** (an 8×6 board → 48 tokens) — nothing the model has to render is
   ever smaller than a token.
3. A **13.5M-parameter spatiotemporal DiT** predicts the next latent from the past 8 latents + your action,
   trained with **EDM preconditioning + Diffusion Forcing**. Four Euler sampling steps is fast enough to be
   real-time, so there's no distillation step.
4. The VAE decoder + DiT are exported to **ONNX** and run in the browser via **ONNX Runtime Web + WebGPU**.
   A **WebSocket server** (`server/ws.py`, on Modal) is the fallback for browsers without WebGPU.

A diffusion model is a great *renderer* but unreliable at discrete, rare events (like dying). So death is
handled by a tiny deterministic **referee**: the model dreams every pixel, and ~30 lines of rules adjudicate
wall/self collisions. The model dreams the world; the referee calls the game.

## The honest version

This started as Breakout and failed — a small, fast, *continuous* ball is exactly the regime that fights a
latent diffusion model, and the one loss knob that kept the ball alive also stopped the bricks from breaking.
The fix wasn't a better loss; it was a **better-posed game**. Snake is grid-native: discrete one-cell motion,
no sub-token objects, clean eat/grow events. The whole recipe transferred unchanged. Full write-up in the blog.

## Quickstart

```bash
pip install -e .

# play the trained model locally (pygame window, arrow keys)
python infer.py --ckpt checkpoints/dit_small_gate2_run4_155k.pt --vae checkpoints/vae_small.pt --config small

# or play it in the browser (the real demo: WebGPU in-browser inference)
python export.py                       # DiT + VAE decoder -> demo/assets/*.onnx (FP16)
cd demo && python -m http.server 8080  # open http://localhost:8080 in Chrome
```

Train it yourself end-to-end (on [Modal](https://modal.com)):

```bash
modal run modal_data_gen.py --tier baseline               # generate 500k Snake frames
modal run modal_train.py --stage vae --config small       # train the ViT-VAE (~$9)
# then pre-encode frames → latents and train the DiT on them (exact flags in docs/TASKS.md)
modal run modal_train.py --stage dit --config small       # train the 13.5M DiT (~$33)
```

## Architecture notes

Choices that matter, with the reasoning in `docs/` (decision records) and the papers cited inline in the code:

- **EDM preconditioning** (Karras 2022), not DDPM — cleaner SNR coverage, deterministic few-step sampling.
- **Diffusion Forcing** (Chen 2024) — independent per-frame noise + a causal temporal mask, for stable
  autoregressive rollout.
- **Context-noise augmentation** (GameNGen) — train on *corrupted* history so the model corrects its own
  drift at inference. This is the single fix that made long rollouts hold together.
- **Factorized spatial + temporal attention**, **AdaLN-Zero**, a **shared AdaLN MLP** (PixArt-Σ), and **2D
  axial + 1D RoPE** — a clean, small DiT.
- **Dual-path action conditioning** (token + AdaLN) with 10% dropout, freeing classifier-free guidance.
- **Euler, 4 steps** — DIAMOND's regime; real-time everywhere, so no LCM distillation.

## Repo layout

```
game.py         grid Snake (data source + the thing the model imitates)
data_gen.py     parallel bot rollouts → zstd shards
data.py         windowed loader with event-oversampling
vae.py          ViT-VAE (256×192 frame ↔ latent)
model.py        spatiotemporal DiT
diffusion.py    EDM preconditioning + Diffusion Forcing + context-noise
train_vae.py    VAE training loop          train.py   DiT training loop
infer.py        local pygame inference + the eval harness
export.py       ONNX export (DiT + VAE decoder) + seed contexts
modal_*.py      Modal cloud entrypoints
demo/           in-browser WebGPU demo (inference.js sampler + referee, main.js, index.html)
server/ws.py    WebSocket fallback for non-WebGPU browsers
configs/        tiny / small / launch YAMLs
```

## Limitations

It's a 13.5M-parameter model trained for ~$50, and it plays like one. Expect crisp Snake for the first
several apples, then the long-body coherence frays — diffusion models fumble long thin structures, and error
accumulates over a rollout. The retry-on-death re-seed keeps every life starting from a clean context. This
is a *reference implementation*, not a product; it's meant to be read and forked.

## Prior art + credits

Built in the lineage of **DIAMOND** (Alonso et al.), **GameNGen** (Valevski et al.), and **Oasis**
(Decart / Etched). A pixel-space diffusion Snake also exists ([juraam/snake-diffusion](https://github.com/juraam/snake-diffusion));
nanoOasis differs in being a full **latent** stack (ViT-VAE + spatiotemporal DiT + Diffusion Forcing), real-time
few-step play, in-browser WebGPU, and the 1-cell-1-token game design — kept deliberately small and readable,
in the spirit of [nanoGPT](https://github.com/karpathy/nanoGPT).

## License

MIT.
