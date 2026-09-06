"""Fail-fast consistency check for the full 21M-passage wiki18 assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bm25s.utils.corpus import JsonlCorpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dense", required=True)
    parser.add_argument("--bm25", required=True)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--expected_docs", type=int, default=21_015_324)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    corpus_path = Path(args.corpus).resolve()
    dense_path = Path(args.dense).resolve()
    bm25_dir = Path(args.bm25).resolve()
    params = json.loads((bm25_dir / "params.index.json").read_text())
    dense_bytes = dense_path.stat().st_size
    row_bytes = args.dim * 2  # fp16
    if dense_bytes % row_bytes:
        raise SystemExit(f"dense size {dense_bytes} is not divisible by fp16 dim {args.dim}")
    dense_rows = dense_bytes // row_bytes
    corpus = JsonlCorpus(corpus_path, show_progress=False, save_index=False, verbosity=0)
    corpus_rows = len(corpus)
    first = corpus[0]
    last = corpus[-1]
    bm25_rows = int(params.get("num_docs", -1))
    counts = {"corpus": corpus_rows, "dense": dense_rows, "bm25": bm25_rows}
    if set(counts.values()) != {args.expected_docs}:
        raise SystemExit(f"wiki18 count mismatch: {counts}, expected {args.expected_docs}")
    if str(first.get("id")) != "0" or str(last.get("id")) != str(args.expected_docs - 1):
        raise SystemExit(
            f"corpus id boundary mismatch: first={first.get('id')!r} last={last.get('id')!r}"
        )

    report = {
        "status": "PASS",
        "expected_docs": args.expected_docs,
        "counts": counts,
        "embedding_dim": args.dim,
        "embedding_dtype": "float16",
        "paths": {
            "corpus": str(corpus_path),
            "dense": str(dense_path),
            "bm25": str(bm25_dir),
        },
        "sizes_bytes": {
            "corpus": corpus_path.stat().st_size,
            "dense": dense_bytes,
        },
        "corpus_boundary_ids": [str(first.get("id")), str(last.get("id"))],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)


if __name__ == "__main__":
    main()
