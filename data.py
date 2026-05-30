# nanoOasis training-data loader. Streams (context + target) windows from per-episode shards.

import io
import pathlib
from collections import OrderedDict

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset
import zstandard as zstd


CONTEXT = 16                       # past frames the model conditions on
WINDOW = CONTEXT + 1               # +1 target frame -> 17 total


class EpisodeWindowDataset(IterableDataset):
    def __init__(self, index_path, root=None, split: str | None = None,
                 cache_size: int = 64, seed: int = 0):
        self.index_path = pathlib.Path(index_path)
        self.root = pathlib.Path(root) if root else self.index_path.parent
        rows = pq.read_table(self.index_path).to_pylist()
        if split is not None:
            rows = [r for r in rows if r["split"] == split]
        # keep only shards long enough for one window
        self.episodes = [r for r in rows if r["length"] >= WINDOW]
        if not self.episodes:
            raise ValueError(f"no episodes of length >= {WINDOW} in {self.index_path}")
        self.cache_size = cache_size
        self.seed = seed

    def _load(self, name: str) -> dict:
        raw = zstd.ZstdDecompressor().decompress((self.root / name).read_bytes())
        return dict(np.load(io.BytesIO(raw)))

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = np.random.default_rng(self.seed ^ (worker_id * 0xC0FFEE))
        cache: "OrderedDict[str, dict]" = OrderedDict()

        while True:
            ep = self.episodes[int(rng.integers(0, len(self.episodes)))]
            T = ep["length"]
            start = int(rng.integers(0, T - WINDOW + 1))

            data = cache.get(ep["path"])
            if data is None:
                if len(cache) >= self.cache_size:
                    cache.popitem(last=False)              # LRU evict
                data = self._load(ep["path"])
                cache[ep["path"]] = data
            else:
                cache.move_to_end(ep["path"])              # mark recently used

            yield data["frames"][start:start + WINDOW], data["actions"][start:start + WINDOW]
