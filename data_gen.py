# nanoOasis data generation. Headless, deterministic, single-process for now (D5 adds workers).

# SDL env vars MUST be set before pygame is imported (BUGS.md H004).
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import io
import multiprocessing as mp
import pathlib

import numpy as np
import zstandard as zstd

from game import Game, H, W


class RandomBot:
    # action prior -- {NONE, LEFT, RIGHT}; paddle moves 80% of frames in decisive 3-8 frame holds.
    # 100% random (no ball-tracking) so the paddle's motion is decorrelated from the ball: the model
    # can only explain paddle motion via the action, not "follow the ball" (the M7 control bug -- see
    # docs/TASKS.md CURRENT STATUS). This is the entire data half of the control fix.
    PROBS = np.array([0.2, 0.4, 0.4])

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.held_remaining = 0
        self.current = 0

    def act(self, game: Game) -> int:                # game unused; signature kept for collect_episode
        if self.held_remaining == 0:
            self.current = int(self.rng.choice(3, p=self.PROBS))
            self.held_remaining = int(self.rng.integers(3, 9))   # 3..8 inclusive
        self.held_remaining -= 1
        return self.current


def collect_episode(seed: int, n_frames: int, bot_cls=RandomBot
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = Game(seed=seed)
    bot = bot_cls(np.random.default_rng(seed ^ 0xB07))
    frames = np.empty((n_frames, H, W, 3), dtype=np.uint8)
    actions = np.empty(n_frames, dtype=np.uint8)
    dones = np.zeros(n_frames, dtype=bool)
    level_seeds = np.empty(n_frames, dtype=np.uint32)

    for t in range(n_frames):
        action = bot.act(g)
        prev_misses, prev_board = g.misses, g.board
        frame, _, _ = g.step(action)
        frames[t] = frame
        actions[t] = action
        level_seeds[t] = g.board                       # board id in the level_seeds slot (scene reset marker)
        dones[t] = g.misses > prev_misses or g.board != prev_board

    return frames, actions, dones, level_seeds


def write_shard(path: pathlib.Path, frames, actions, dones, level_seeds) -> None:
    buf = io.BytesIO()
    np.savez(buf, frames=frames, actions=actions, dones=dones, level_seeds=level_seeds)
    path.write_bytes(zstd.ZstdCompressor(level=3).compress(buf.getvalue()))


def _worker(args: tuple) -> list[dict]:
    # args: (worker_id, n_frames, out_dir, seed, episode_size=None)
    # Splits n_frames into chunks of episode_size to keep per-worker RAM bounded.
    # Returns a list of one index row per shard written.
    worker_id, n_frames, out_dir, seed = args[:4]
    episode_size = args[4] if len(args) > 4 and args[4] else n_frames
    out_dir = pathlib.Path(out_dir)
    rows: list[dict] = []
    written = 0
    ep_idx = 0
    while written < n_frames:
        n = min(episode_size, n_frames - written)
        frames, actions, dones, level_seeds = collect_episode(seed + ep_idx, n, bot_cls=RandomBot)
        if episode_size >= n_frames:
            shard_name = f"ep_{worker_id:03d}.npz.zst"
        else:
            shard_name = f"ep_{worker_id:03d}_{ep_idx:04d}.npz.zst"
        shard = out_dir / shard_name
        write_shard(shard, frames, actions, dones, level_seeds)
        n_level_changes = int((np.diff(level_seeds.astype(np.int64)) != 0).sum())
        episode_id = worker_id * 10_000 + ep_idx
        rows.append({
            "episode_id":      int(episode_id),
            "path":            shard.name,
            "length":          int(n),
            "n_dones":         int(dones.sum()),
            "n_level_changes": n_level_changes,
            "bot_type":        RandomBot.__name__,
            "worker_id":       int(worker_id),
            "split":           "val" if worker_id % 20 == 0 else "train",
        })
        written += n
        ep_idx += 1
        del frames, actions, dones, level_seeds            # free memory before next chunk
    return rows


def write_index(rows: list[dict], path: pathlib.Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows), path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-frames", type=int, default=1000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", type=str, default="data/scratch")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per = args.n_frames // args.workers
    jobs = [(i, per, str(out), args.seed + i * 100_000) for i in range(args.workers)]

    print(f"collecting {args.n_frames} frames across {args.workers} workers -> {out}")
    if args.workers == 1:
        worker_rows = [_worker(jobs[0])]
    else:
        with mp.Pool(args.workers) as pool:
            worker_rows = pool.map(_worker, jobs)
    rows = [r for ws in worker_rows for r in ws]                # flatten worker -> shard rows

    write_index(rows, out / "index.parquet")
    total = sum(r["length"] for r in rows)
    transitions = sum(r["n_level_changes"] for r in rows)
    deaths = sum(r["n_dones"] - r["n_level_changes"] for r in rows)
    disk = sum((out / r["path"]).stat().st_size for r in rows)
    print(f"wrote {len(rows)} shards + index.parquet "
          f"({total} frames, deaths={deaths}, transitions={transitions}, "
          f"disk={disk / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
