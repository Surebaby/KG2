"""NumPy-based dense retriever that reads fp16 vectors via memmap.

Replaces faiss.IndexFlatIP when the index is too large for available RAM.
faiss's IndexFlatIP eagerly loads ALL vectors on read_index(); IO_FLAG_MMAP
does NOT prevent this for Flat indices. This class memmaps the raw fp16
payload and searches in chunks, keeping RSS near zero.

Interface: mimics faiss.IndexFlat enough to drop into DenseRetriever.
"""

from __future__ import annotations

import numpy as np
import time
from typing import Tuple


class MemmapSearch:
    """Brute-force inner-product search over fp16 vectors stored in a numpy memmap."""

    def __init__(self, data_path: str, ntotal: int, d: int, db_chunk: int = 2_000_000):
        self.data_path = data_path
        self.ntotal = ntotal
        self.d = d
        self._db_chunk = db_chunk
        # Open the memmap read-only — RSS stays near zero.
        self._xb = np.memmap(data_path, dtype="float16", mode="r", shape=(ntotal, d))

    @staticmethod
    def load(data_path: str, meta: dict | None = None) -> "MemmapSearch":
        """Factory: auto-detect ntotal, d from file size if meta not given."""
        import os
        sz = os.path.getsize(data_path)
        d = (meta or {}).get("d", 768)
        elem_size = d * 2  # fp16 = 2 bytes per element
        if sz % elem_size != 0:
            raise ValueError(f"File size {sz} not divisible by d*2={elem_size}")
        ntotal = sz // elem_size
        return MemmapSearch(data_path, ntotal=ntotal, d=d)

    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Return (distances, indices) like faiss.Index.search.

        queries: (n_queries, d) float32
        Returns: distances (n_queries, k) float32, indices (n_queries, k) int64
        """
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        queries = queries.astype(np.float32)
        n_q = queries.shape[0]

        all_D = np.full((n_q, k), -np.inf, dtype=np.float32)
        all_I = np.full((n_q, k), -1, dtype=np.int64)

        t0 = time.time()
        for chunk_start in range(0, self.ntotal, self._db_chunk):
            chunk_end = min(chunk_start + self._db_chunk, self.ntotal)
            # Load one chunk of the DB into RAM (cast fp16 → fp32)
            db_chunk = np.array(self._xb[chunk_start:chunk_end], dtype=np.float32)
            # scores: (n_q, chunk_size)
            scores = queries @ db_chunk.T
            # Merge with running top-k
            if chunk_start == 0:
                # First chunk: just take top-k
                if scores.shape[1] <= k:
                    all_D[:, : scores.shape[1]] = scores
                    all_I[:, : scores.shape[1]] = np.arange(scores.shape[1], dtype=np.int64)
                else:
                    idx = np.argpartition(-scores, k, axis=1)[:, :k]
                    all_D = np.take_along_axis(scores, idx, axis=1)
                    all_I = idx.astype(np.int64)
            else:
                # Concatenate running top-k with new scores, re-rank
                combined_D = np.concatenate([all_D, scores], axis=1)
                combined_I = np.concatenate([all_I, np.full_like(scores, chunk_start, dtype=np.int64) + np.arange(scores.shape[1], dtype=np.int64)], axis=1)
                idx = np.argpartition(-combined_D, k, axis=1)[:, :k]
                all_D = np.take_along_axis(combined_D, idx, axis=1)
                all_I = np.take_along_axis(combined_I, idx, axis=1)

        elapsed = time.time() - t0
        if n_q > 1 or chunk_start > 0:
            pass  # Silent in production; DenseRetriever logs its own timings.

        return all_D, all_I
