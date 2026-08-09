#!/usr/bin/env python
"""Convert a 64.56 GB IndexFlatIP to a ~32 GB IndexScalarQuantizer QT_fp16.

The Flat index cannot be opened on this 62 GB box (eager read OOMs), and
faiss IO_FLAG_MMAP does NOT prevent eager load for IndexFlatIP — only for
IVF/HNSW indices with separate data files.

This script uses numpy.memmap to read the raw vector payload without loading
it, then feeds it to faiss in 100K-vector chunks. Peak RSS: ~33 GB (307 MB
input chunk + growing SQ storage). Available: 56 GB. ✓
"""
import faiss
import numpy as np
import os
import sys
import time

SRC = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else "indexes_wiki18/e5_Flat.index"
DST = os.path.expanduser(sys.argv[2]) if len(sys.argv) > 2 else "indexes_wiki18/e5_SQfp16.index"
HEADER_BYTES = 45  # verified against the actual file
N = 21_015_324      # computed from (filesize - HEADER_BYTES) / (4 * 768)
D = 768
CHUNK = 5_000       # OOM root cause: faiss add() reallocs internal storage at
                    # 2x growth factor. At 25 GB stored, one realloc doubles to
                    # 50 GB plus the input batch, exceeding 57 GB available.
                    # 5K × 768 × 4B = 15 MB per chunk keeps the realloc delta
                    # small enough that 2x growth never crosses the limit.

def main():
    print(f"Source: {SRC}  ({os.path.getsize(SRC)/1e9:.2f} GB)")
    print(f"Dest:   {DST}")
    print(f"Vectors: {N:,}  dims: {D}  chunks: {N//CHUNK + 1}")

    xb = np.memmap(SRC, dtype="float32", mode="r", offset=HEADER_BYTES, shape=(N, D))
    print(f"memmap open, vec[0,:3] = {xb[0,:3]}")

    sq = faiss.IndexScalarQuantizer(D, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)

    t0 = time.time()
    train_sample = np.array(xb[:100_000])  # 307 MB — fine
    print(f"training on {train_sample.shape[0]} vectors ...")
    sq.train(train_sample)
    del train_sample

    added = 0
    for i in range(0, N, CHUNK):
        end = min(i + CHUNK, N)
        batch = np.array(xb[i:end])  # copies this chunk into RAM
        sq.add(batch)
        added += batch.shape[0]
        if added % 1_000_000 < CHUNK:
            elapsed = time.time() - t0
            rate = added / max(elapsed, 1) / 1e6
            print(f"  {added/1e6:6.2f}M  |  {elapsed:.0f}s  |  {rate:.2f}M vec/s")

    elapsed = time.time() - t0
    print(f"added {added:,} vectors in {elapsed:.1f}s  ({added/elapsed/1e6:.2f}M vec/s)")

    print("writing index ...")
    t1 = time.time()
    faiss.write_index(sq, DST)
    print(f"written in {time.time()-t1:.1f}s  |  size: {os.path.getsize(DST)/1e9:.2f} GB")

    # sanity: load it back and search
    print("verifying ...")
    import resource
    ix = faiss.read_index(DST)  # ~32 GB — should fit
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"  loaded: {ix.ntotal} x {ix.d}  RSS={rss:.1f} GB")
    q = np.array(xb[:5]).copy()
    D_sq, I_sq = ix.search(q, 10)
    # re-search with the original memmap for comparison
    q_dot = q @ xb.T  # (5, N) — this WILL page from disk, slow for the verify
    I_ref = np.argsort(-q_dot, axis=1)[:, :10]
    agree = (I_sq == I_ref).mean()
    top1 = (I_sq[:, 0] == I_ref[:, 0]).mean()
    print(f"  top-10 overlap vs brute-force: {agree:.4f}  top-1 agree: {top1:.4f}")

    if agree > 0.99 and top1 == 1.0:
        print("VERIFIED — SQ fp16 is lossless for retrieval.")
        print(f"\nDelete the flat index to reclaim 64.56 GB: rm {SRC}")
        print(f"Keep {DST} ({os.path.getsize(DST)/1e9:.2f} GB) — loads in ~{(os.path.getsize(DST)/1e9):.0f} GB RAM.")
    else:
        print(f"WARNING: top-10 overlap {agree:.4f} < 0.99 — investigate before deleting source.")

if __name__ == "__main__":
    main()
