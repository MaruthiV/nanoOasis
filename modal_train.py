# nanoOasis Modal training entrypoints. Three stages: vae / dit / lcm.
#
# usage:
#   modal run modal_train.py --stage vae --config tiny --smoke     # ~5 min on A10G, smoke test
#   modal run modal_train.py --stage vae --config small            # full tiny run on A10G
#   modal run modal_train.py --stage dit --config tiny             # tiny DiT on A10G
#   modal run modal_train.py --stage dit --config launch           # launch DiT on 4xH100 (Phase 5)

import modal


app = modal.App("nano-oasis-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch>=2.5",
        "torchvision",                     # required by lpips
        "numpy<2",
        "omegaconf>=2.3",
        "imageio[ffmpeg]",
        "zstandard>=0.22",
        "pyarrow",
        "tqdm",
        "lpips",
        "wandb",
    )
    # ship every project module the training functions transitively touch
    .add_local_python_source(
        "game", "data_gen", "data", "vae", "model", "diffusion", "train_vae", "train", "encode_latents",
    )
    .add_local_dir("configs", "/root/configs")
)

data_vol = modal.Volume.from_name("nano-oasis-data",  create_if_missing=True)
ckpt_vol = modal.Volume.from_name("nano-oasis-ckpts", create_if_missing=True)
VOLS = {"/data": data_vol, "/checkpoints": ckpt_vol}


def _wire_paths_and_data():
    # bind /data and /checkpoints (Modal volumes) into the cwd so relative paths in
    # train.py / train_vae.py / data.py just work, then ensure the smoke dataset exists
    import os, pathlib
    os.chdir("/root")
    for cloud_dir in ("data", "checkpoints"):
        link = pathlib.Path(f"/root/{cloud_dir}")
        if not link.exists():
            link.symlink_to(f"/{cloud_dir}")
    if not pathlib.Path("/data/smoke/index.parquet").exists():
        print("smoke dataset not on volume; generating 10K frames across 4 workers...")
        import multiprocessing as mp
        from data_gen import _worker, write_index
        out = pathlib.Path("/data/smoke")
        out.mkdir(parents=True, exist_ok=True)
        jobs = [(i, 2500, str(out), i * 100_000) for i in range(4)]
        with mp.Pool(4) as pool:
            rows = pool.map(_worker, jobs)
        write_index(rows, out / "index.parquet")
        data_vol.commit()
        print(f"  wrote {len(rows)} shards + index.parquet to /data/smoke")


# ---- VAE: A100-80GB for tiny/small tiers (fastest at small VAE compute profile) ----

@app.function(image=image, gpu="A100-80GB", volumes=VOLS, timeout=12 * 3600,
              memory=128 * 1024)                  # hold the 256x192 baseline episode cache in RAM (~74 GB; D021)
def train_vae_remote(config_name: str = "tiny", smoke: bool = False, steps: int | None = None) -> None:
    _wire_paths_and_data()
    from train_vae import main
    if smoke and steps is None:
        steps = 500
    main(config_name=config_name, total_steps=steps, on_checkpoint=ckpt_vol.commit)
    ckpt_vol.commit()
    print("checkpoint committed to nano-oasis-ckpts volume")


@app.function(image=image, gpu="B200", volumes=VOLS, timeout=24 * 3600,
              memory=300 * 1024,                           # holds the 256x192 baseline episode cache (~74 GB) in RAM
              secrets=[modal.Secret.from_name("wandb")])
def train_vae_launch(steps: int | None = None) -> None:
    _wire_paths_and_data()
    from train_vae import main
    main(config_name="launch", total_steps=steps, on_checkpoint=ckpt_vol.commit)
    ckpt_vol.commit()


# ---- DiT: A100-80GB for tiny/small tiers, 4xH100 for the launch run ----

@app.function(image=image, gpu="A100-80GB", volumes=VOLS, timeout=12 * 3600,
              memory=128 * 1024,                  # 256x192 baseline episode cache ~74 GB in RAM (D021)
              secrets=[modal.Secret.from_name("wandb")])
def train_dit_remote(config_name: str = "tiny", smoke: bool = False, steps: int | None = None) -> None:
    _wire_paths_and_data()
    from train import main
    if smoke and steps is None:
        steps = 500
    main(stage="dit", config_name=config_name, total_steps=steps, on_checkpoint=ckpt_vol.commit)
    ckpt_vol.commit()


@app.function(image=image, gpu="B200", volumes=VOLS, timeout=24 * 3600,
              memory=256 * 1024,                  # 8M latents (D026): 195 train episodes ~192 GB in RAM + headroom
              secrets=[modal.Secret.from_name("wandb")])
def train_dit_launch() -> None:
    # single B200 (fastest single card; DDP skipped to avoid silent data-sharding bugs on the expensive run).
    # train.py checkpoints+commits every ckpt_every and resumes from the committed ckpt on restart,
    # so preemptions are survivable. The 24h/call cap is just a relaunch (it resumes where it left off).
    _wire_paths_and_data()
    from train import main
    main(stage="dit", config_name="launch", on_checkpoint=ckpt_vol.commit)
    ckpt_vol.commit()


# ---- LCM distillation: implemented in Phase 5 ----

@app.function(image=image, gpu="H100:4", volumes=VOLS, timeout=12 * 3600)
def distill_lcm_remote(config_name: str = "launch") -> None:
    _wire_paths_and_data()
    # distill.py lands in Phase 6 (task I2); this function is the entrypoint that will call it.
    raise NotImplementedError("LCM distillation lands in Phase 6 / task I2")


# ---- latent pre-encoding: VAE-encode a frame dataset to latent shards (the launch-run data path) ----

@app.function(image=image, gpu="A100-80GB", volumes=VOLS, timeout=6 * 3600)
def encode_latents_remote(tier: str = "baseline", vae: str = "vae_launch.pt", config: str = "launch") -> None:
    # run after the launch VAE is trained: writes /data/<tier>_latents for pre-encoded DiT training (M7)
    _wire_paths_and_data()
    from encode_latents import encode_dir
    encode_dir(f"data/{tier}/index.parquet", f"checkpoints/{vae}", config, f"data/{tier}_latents")
    data_vol.commit()


# ---- local entrypoint -- this is what `modal run modal_train.py --stage ... --config ...` invokes ----

@app.local_entrypoint()
def main(stage: str = "vae", config: str = "tiny", smoke: bool = False, steps: int | None = None) -> None:
    if stage == "vae":
        if config == "launch":
            train_vae_launch.remote(steps=steps)
        else:
            train_vae_remote.remote(config_name=config, smoke=smoke, steps=steps)
    elif stage == "dit":
        if config == "launch":
            train_dit_launch.remote()
        else:
            train_dit_remote.remote(config_name=config, smoke=smoke, steps=steps)
    elif stage == "lcm":
        distill_lcm_remote.remote(config_name=config)
    else:
        raise ValueError(f"unknown stage: {stage}")
