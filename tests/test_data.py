import io
import pathlib
import time

import numpy as np
import zstandard as zstd

from data_gen import collect_episode, write_shard, RandomBot, _worker, write_index
from data import EpisodeWindowDataset, WINDOW

from game import Game, W, H

SMOKE_INDEX = pathlib.Path(__file__).resolve().parent.parent / "data" / "smoke" / "index.parquet"


def _read_shard(path):
    raw = zstd.ZstdDecompressor().decompress(path.read_bytes())
    return dict(np.load(io.BytesIO(raw)))


def test_writer_round_trip_shapes_and_dtypes(tmp_path):
    frames, actions, dones, level_seeds = collect_episode(seed=42, n_frames=50)
    shard = tmp_path / "ep_test.npz.zst"
    write_shard(shard, frames, actions, dones, level_seeds)

    data = _read_shard(shard)
    assert data["frames"].shape       == (50, H, W, 3)
    assert data["frames"].dtype       == np.uint8
    assert data["actions"].shape      == (50,)
    assert data["actions"].dtype      == np.uint8
    assert data["dones"].shape        == (50,)
    assert data["dones"].dtype        == np.bool_
    assert data["level_seeds"].shape  == (50,)
    assert data["level_seeds"].dtype  == np.uint32

    np.testing.assert_array_equal(data["frames"], frames)
    np.testing.assert_array_equal(data["actions"], actions)
    np.testing.assert_array_equal(data["dones"], dones)
    np.testing.assert_array_equal(data["level_seeds"], level_seeds)


def test_writer_compresses_pixel_redundancy(tmp_path):
    frames, actions, dones, level_seeds = collect_episode(seed=0, n_frames=200)
    shard = tmp_path / "ep_compress.npz.zst"
    write_shard(shard, frames, actions, dones, level_seeds)

    raw_bytes = frames.nbytes + actions.nbytes + dones.nbytes + level_seeds.nbytes
    compressed_bytes = shard.stat().st_size
    # pixel art is highly redundant; demand at least 10x compression
    assert compressed_bytes * 10 < raw_bytes, (compressed_bytes, raw_bytes)


def test_writer_deterministic_for_same_seed(tmp_path):
    a, b = tmp_path / "a.npz.zst", tmp_path / "b.npz.zst"
    for path in (a, b):
        frames, actions, dones, level_seeds = collect_episode(seed=7, n_frames=100)
        write_shard(path, frames, actions, dones, level_seeds)
    assert a.read_bytes() == b.read_bytes()


# ---- D3: random-walk bot ----


def test_bot_random_histogram_within_5pct():
    bot = RandomBot(np.random.default_rng(0))
    actions = np.fromiter((bot.act(None) for _ in range(5000)), dtype=np.uint8, count=5000)
    hist = np.bincount(actions, minlength=3) / 5000
    diff = np.abs(hist - RandomBot.PROBS).max()
    assert diff < 0.05, (hist.tolist(), RandomBot.PROBS.tolist(), float(diff))


def test_bot_random_stickiness_at_least_3():
    bot = RandomBot(np.random.default_rng(0))
    actions = [bot.act(None) for _ in range(2000)]
    runs = []
    run_len, cur = 1, actions[0]
    for a in actions[1:]:
        if a == cur:
            run_len += 1
        else:
            runs.append(run_len)
            cur, run_len = a, 1
    runs.append(run_len)
    # drop the last run -- it can be truncated by the sample window; first run is full
    interior = runs[:-1]
    assert interior and min(interior) >= 3, runs[:20]


# ---- D5: parallel collection + parquet index ----


def test_parallel_writes_n_shards_all_random(tmp_path):
    import multiprocessing as mp
    jobs = [(i, 200, str(tmp_path), i * 100_000) for i in range(4)]
    with mp.Pool(4) as pool:
        worker_rows = pool.map(_worker, jobs)
    rows = [r for ws in worker_rows for r in ws]                # flatten worker -> shard rows
    write_index(rows, tmp_path / "index.parquet")

    shards = sorted(tmp_path.glob("ep_*.npz.zst"))
    assert len(shards) == 4

    import pyarrow.parquet as pq
    table = pq.read_table(tmp_path / "index.parquet")
    assert table.num_rows == 4
    by_worker = {r["worker_id"]: r for r in table.to_pylist()}
    assert set(by_worker.keys()) == {0, 1, 2, 3}
    # every worker runs RandomBot now -- 100% random play (the control fix, see data_gen.py)
    assert all(r["bot_type"] == "RandomBot" for r in by_worker.values())
    # worker_id 0 -> "val" (0 % 20 == 0); others -> "train"
    assert by_worker[0]["split"] == "val"
    assert by_worker[1]["split"] == "train"


# ---- D7: IterableDataset + LRU cache + random window sampler ----


def test_loader_pulls_100_windows_with_correct_shapes():
    ds = EpisodeWindowDataset(SMOKE_INDEX)
    it = iter(ds)
    for _ in range(100):
        frames, actions = next(it)
        assert frames.shape == (WINDOW, H, W, 3)
        assert frames.dtype == np.uint8
        assert actions.shape == (WINDOW,)
        assert actions.dtype == np.uint8


def test_loader_steady_state_throughput_at_least_10k_fps():
    ds = EpisodeWindowDataset(SMOKE_INDEX)
    it = iter(ds)
    for _ in range(10):                    # warm the LRU cache
        next(it)
    t0 = time.time()
    n = 200
    for _ in range(n):
        next(it)
    fps = n * WINDOW / (time.time() - t0)
    assert fps >= 10_000, f"throughput {fps:.0f} fps/worker"


def test_loader_split_filter_partitions_episodes():
    all_ds   = EpisodeWindowDataset(SMOKE_INDEX)
    train_ds = EpisodeWindowDataset(SMOKE_INDEX, split="train")
    val_ds   = EpisodeWindowDataset(SMOKE_INDEX, split="val")
    assert len(all_ds.episodes) == 4
    assert len(train_ds.episodes) == 3
    assert len(val_ds.episodes) == 1
    assert {e["split"] for e in train_ds.episodes} == {"train"}
    assert {e["split"] for e in val_ds.episodes} == {"val"}
