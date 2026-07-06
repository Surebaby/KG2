#!/usr/bin/env python
"""Step 0 — Convert a Wikipedia dump to FlashRAG corpus JSONL.

Input format (per line): ``{"id": int, "title": "...", "text": "..."}``
Output format (per line): ``{"id": "str", "contents": "title\\ntext"}``

The conversion is streamed line-by-line; even a 15M-line dump fits easily
in memory because we never buffer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from kgproweight.retrieval.bootstrap import resolve_corpus_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Source JSONL file (raw Wikipedia dump).")
    p.add_argument("--output", default=None, help="Destination JSONL. Defaults to $KGPW_INDEX_DIR/corpus_flashrag.jsonl.")
    p.add_argument("--max_lines", type=int, default=None, help="Stop after N input lines (testing).")
    p.add_argument("--log_interval", type=int, default=500_000)
    return p.parse_args()


def convert_line(raw: str) -> str | None:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None

    doc_id = str(obj.get("id", ""))
    title = (obj.get("title") or "").strip().strip('"')
    text = (obj.get("text") or "").strip()
    if not doc_id:
        return None
    contents = f"{title}\n{text}" if title else text
    return json.dumps({"id": doc_id, "contents": contents}, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Input file not found: {in_path}")

    out_path = Path(args.output) if args.output else Path(resolve_corpus_path())
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input  : {in_path}")
    print(f"Output : {out_path}")
    print(f"Max lines: {args.max_lines or 'all'}")
    print("Starting conversion...")

    t0 = time.time()
    written = 0
    skipped = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for i, raw in enumerate(fin):
            if args.max_lines is not None and i >= args.max_lines:
                break
            cv = convert_line(raw)
            if cv is None:
                skipped += 1
                continue
            fout.write(cv + "\n")
            written += 1
            if written % args.log_interval == 0:
                rate = written / max(time.time() - t0, 1e-6)
                print(f"  {written:>10,} written | {skipped:>6,} skipped | {rate:>9,.0f}/s")
    print(f"Done. written={written:,} skipped={skipped:,} elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
