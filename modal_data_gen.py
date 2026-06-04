# nanoOasis parallel data generation on Modal. Three tiers: smoke / baseline / full.
#
# usage:
#   modal run modal_data_gen.py --tier smoke      # 10K frames,  ~30s on 4 CPU
#   modal run modal_data_gen.py --tier baseline   # 500K frames, ~5min on 16 CPU
#   modal run modal_data_gen.py --tier full       # 5M frames,   ~60min on 16 CPU
#
# Self-contained: defines its own image + volume so the container's import path doesn't
# need modal_train.py. The volume name is shared, so writes still land on the same persistent
# storage as modal_train.py reads from.

import modal


# app name must NOT collide with the volume name "nano-oasis-data" -- Modal tangles the two
# permission ACLs (an app "does not have write access to itself"), which crash-loops the run.
app = modal.App("nano-oasis-datagen")

# minimal image -- data generation only needs game.py + data_gen.py and their deps
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",
        "zstandard>=0.22",
        "pyarrow",
    )
    .add_local_python_source("game", "data_gen")
)

data_vol = modal.Volume.from_name("nano-oasis-data", create_if_missing=True)


TIERS = {
    # episode_size: max frames per shard -- caps per-worker peak memory at episode_size * 36 KB
    "smoke":    {"frames":    10_000, "workers":  4, "episode_size":      0},   # 0 -> one shard / worker
    "baseline": {"frames":   500_000, "workers": 16, "episode_size":      0},
    "full":     {"frames": 5_000_000, "workers": 16, "episode_size": 25_000},   # 13 shards / worker, ~900 MB buffer each
}


@app.function(image=image, cpu=16, memory=64 * 1024,
              volumes={"/data": data_vol}, timeout=3 * 3600, retries=0)
def gen_remote(tier: str = "smoke") -> None:
    import multiprocessing as mp
    import pathlib

    if tier not in TIERS:
        raise ValueError(f"unknown tier: {tier!r}; pick from {list(TIERS)}")
    spec = TIERS[tier]

    from data_gen import _worker, write_index
    out = pathlib.Path(f"/data/{tier}")
    out.mkdir(parents=True, exist_ok=True)

    per = spec["frames"] // spec["workers"]
    ep_size = spec["episode_size"]
    jobs = [(i, per, str(out), i * 100_000, ep_size) for i in range(spec["workers"])]
    print(f"collecting {spec['frames']} frames across {spec['workers']} workers -> /data/{tier} "
          f"(episode_size={ep_size or 'one'} per shard)")
    with mp.Pool(spec["workers"]) as pool:
        worker_rows = pool.map(_worker, jobs)
    rows = [r for ws in worker_rows for r in ws]
    write_index(rows, out / "index.parquet")
    data_vol.commit()

    disk = sum((out / r["path"]).stat().st_size for r in rows)
    print(f"wrote {len(rows)} shards + index.parquet ({disk / 1024 / 1024:.1f} MiB) -- committed")


@app.local_entrypoint()
def main(tier: str = "smoke") -> None:
    gen_remote.remote(tier=tier)
