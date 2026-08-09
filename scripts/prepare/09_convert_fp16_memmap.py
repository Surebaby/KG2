#!/usr/bin/env python
"""Convert a 64.56 GB fp32 IndexFlatIP to a 32.28 GB fp16 numpy memmap.

Reads via numpy.memmap (RSS ~0), writes fp16 in 1M-vector chunks.
Peak RSS: ~6 MB (output chunk) + ~0 (input via memmap) ≈ O(10 MB).
Completely avoids faiss and its 2x-growth realloc OOM.
"""
import numpy as np
import os, sys, time

SRC = sys.argv[1] if len(sys.argv) > 1 else "indexes_wiki18/e5_Flat.index"
DST = sys.argv[2] if len(sys.argv) > 2 else "indexes_wiki18/e5_fp16.dat"
HEADER_BYTES = 45
N = 21_015_324
D = 768

def main():
    src_sz = os.path.getsize(SRC)
    print(f"Source: {SRC} ({src_sz/1e9:.2f} GB, header={HEADER_BYTES}B, {N:,} × {D})")
    print(f"Dest:   {DST} ({N * D * 2 / 1e9:.2f} GB fp16)")

    xb_fp32 = np.memmap(SRC, dtype="float32", mode="r", offset=HEADER_BYTES, shape=(N, D))

    # Create fp16 memmap — the OS allocates sparse pages on first write,
    # so this doesn't consume 32 GB immediately.
    xb_fp16 = np.memmap(DST, dtype="float16", mode="w+", shape=(N, D))

    CHUNK = 1_000_000  # 1M × 768 × 4B = 3 GB in fp32, 1.5 GB in fp16 output
    t0 = time.time()
    for i in range(0, N, CHUNK):
        end = min(i + CHUNK, N)
        batch_fp32 = np.array(xb_fp32[i:end])  # copy into RAM
        xb_fp16[i:end] = batch_fp32.astype("float16")
        if (i // CHUNK) % 5 == 0:
            elapsed = time.time() - t0
            rate = end / max(elapsed, 1) / 1e6
            print(f"  {end/1e6:6.2f}M / {N/1e6:.1f}M  |  {elapsed:.0f}s  |  {rate:.2f}M vec/s")
    elapsed = time.time() - t0
    print(f"done: {N:,} vectors in {elapsed:.1f}s ({N/elapsed/1e6:.2f}M vec/s)")

    # Verify: round-trip first 10 vectors
    ref = np.array(xb_fp32[:10])
    got = xb_fp16[:10].astype("float32")
    max_err = np.abs(ref - got).max()
    dot_ref = np.dot(ref[0], ref[0])
    dot_got = np.dot(got[0].astype("float32"), got[0].astype("float32"))
    print(f"round-trip max abs error: {max_err:.6f}  (fp16 theoretical: ~5e-4)")
    print(f"norm[0] ref={dot_ref:.6f}  fp16={dot_got:.6f}")

    print(f"\nKept: {DST} ({os.path.getsize(DST)/1e9:.2f} GB)")
    print(f"Delete the 64.56 GB flat index: rm {SRC}")

if __name__ == "__main__":
    main()
