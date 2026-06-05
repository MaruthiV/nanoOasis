#!/usr/bin/env bash
# Play the trained launch model interactively. Arrow keys = paddle, Esc = quit.
# Heads up: this is the un-distilled teacher, so it's slow (~0.5 fps) on a MacBook -- a test drive, not the smooth game.
# Optional arg = sampler steps (lower = faster/rougher). Default 6.
.venv/bin/python infer.py --ckpt checkpoints/dit_launch.pt --vae checkpoints/vae_launch.pt --config launch --steps "${1:-6}"
