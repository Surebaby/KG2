#!/usr/bin/env python
"""Step 3 — Download HotpotQA / 2WikiMultiHopQA / MuSiQue from HF.

Files are saved to ``$KGPW_DATA_DIR/<dataset>/<split>.jsonl`` in FlashRAG
format ``{id, question, golden_answers, metadata}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kgproweight.utils.paths import data_dir

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("datasets library required: `pip install datasets`")

DATASETS = {
    "hotpotqa": {"hf_name": "RUC-NLPIR/FlashRAG_datasets", "subset": "hotpotqa"},
    "2wikimultihopqa": {"hf_name": "RUC-NLPIR/FlashRAG_datasets", "subset": "2wikimultihopqa"},
    "musique": {"hf_name": "RUC-NLPIR/FlashRAG_datasets", "subset": "musique"},
}


def convert(item, idx: int) -> dict:
    question = item.get("question", "")
    answers = item.get("golden_answers") or item.get("answer") or []
    if isinstance(answers, str):
        answers = [answers]
    metadata = {k: v for k, v in item.items() if k not in {"question", "golden_answers", "answer", "id"}}
    return {
        "id": str(item.get("id", idx)),
        "question": question,
        "golden_answers": [str(a) for a in answers],
        "metadata": metadata,
    }


def download(name: str, splits, out_dir: Path, force: bool) -> None:
    info = DATASETS[name]
    target = out_dir / name
    target.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===")
    for split in splits:
        outfile = target / f"{split}.jsonl"
        if outfile.exists() and not force:
            print(f"  [{split}] cached: {outfile}")
            continue
        print(f"  [{split}] downloading…")
        try:
            ds = load_dataset(info["hf_name"], name=info["subset"], split=split, trust_remote_code=True)
        except Exception as exc:
            print(f"  ERROR ({split}): {exc}")
            continue
        with open(outfile, "w", encoding="utf-8") as fh:
            for i, item in enumerate(ds):
                fh.write(json.dumps(convert(item, i), ensure_ascii=False) + "\n")
        print(f"  [{split}] {len(ds):,} items → {outfile}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()), choices=list(DATASETS.keys()))
    p.add_argument("--splits", nargs="+", default=["train", "dev"])
    p.add_argument("--output_dir", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else Path(data_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    for ds in args.datasets:
        download(ds, args.splits, out_dir, args.force)


if __name__ == "__main__":
    main()
