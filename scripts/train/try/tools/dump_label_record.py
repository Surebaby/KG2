#!/usr/bin/env python
"""Dump a human-readable per-trajectory label record from a silver JSONL.

Reads a silver-trajectory file (e.g. the output of
``phase1_generate_silver_try.py``) and writes, for every trajectory, the full
per-step label sequence together with acceptance / bucket / answer-score
metadata. Also prints the aggregate +1 / 0 / -1 distribution so you can verify
at a glance that the PRM annotator is producing a genuine three-class signal
(not the binary collapse the original annotator exhibited).

Usage::

    python scripts/train/try/dump_label_record.py [silver.jsonl] [-o out.txt]

Defaults:
    input  : data/silver_data/silver_trajectories_try.jsonl
    output : <input_dir>/label_record_<input_stem>.txt

No API / network / GPU required — it only reads the already-written JSONL.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

LABEL_NAMES = {1: "+1", 0: " 0", -1: "-1"}

# Default input lives next to the package data dir; fall back to a relative
# path when KGPW_DATA_DIR is not set.
_DEFAULT_INPUT = "data/silver_data/silver_trajectories_try.jsonl"


def _load(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _label_distribution(rows: List[Dict]) -> Counter:
    counter: Counter = Counter()
    for r in rows:
        for s in r.get("steps", []):
            counter[s.get("label")] += 1
    return counter


def _fmt_dist(counter: Counter) -> str:
    total = sum(counter.values()) or 1
    body = " / ".join(
        f"{LABEL_NAMES[k].strip()}:{counter.get(k, 0)}({100 * counter.get(k, 0) / total:.1f}%)"
        for k in (1, 0, -1)
    )
    return f"{body}   [n_steps={total}]"


def build_record(rows: List[Dict], source: Path, full: bool = False) -> str:
    lines: List[str] = []
    lines.append("=" * 100)
    lines.append("PER-TRAJECTORY LABEL RECORD")
    lines.append(f"source: {source}")
    lines.append(f"trajectories: {len(rows)}   steps: {sum(len(r.get('steps', [])) for r in rows)}")
    lines.append("=" * 100)

    # aggregate distributions
    all_dist = _label_distribution(rows)
    no_triple = Counter()
    for r in rows:
        for s in r.get("steps", []):
            if not s.get("cited_triples"):
                no_triple[s.get("label")] += 1
    accepted = sum(1 for r in rows if r.get("accepted"))
    buckets = Counter(r.get("metadata", {}).get("bucket") for r in rows)
    acc_buckets = Counter(r.get("metadata", {}).get("bucket") for r in rows if r.get("accepted"))

    lines.append("")
    lines.append("AGGREGATE")
    lines.append(f"  label distribution (all steps)   : {_fmt_dist(all_dist)}")
    lines.append(f"  label distribution (no-triple)   : {_fmt_dist(no_triple)}")
    lines.append(f"  accepted                          : {accepted}/{len(rows)}")
    lines.append(f"  written buckets                   : {dict(buckets)}")
    lines.append(f"  accepted buckets                  : {dict(acc_buckets)}")
    lines.append("")
    lines.append("-" * 100)

    for r in rows:
        m = r.get("metadata", {})
        steps = r.get("steps", [])
        seq = "".join(f"[{LABEL_NAMES[s.get('label')].strip()}]" for s in steps)
        lines.append("")
        lines.append(
            f"qid={r.get('qid')}  accepted={r.get('accepted')}  bucket={m.get('bucket')}  "
            f"triple_rate={m.get('triple_rate', 0):.2f}  answer_score={m.get('answer_score', 0):.2f}  "
            f"coverage={m.get('coverage', 0):.2f}  kg_empty={m.get('kg_empty')}  "
            f"|subgraph|={len(r.get('kg_subgraph', []))}"
        )
        question = r.get("question", "")
        lines.append(f"  Q: {question if full else question[:110]}")
        lines.append(f"  gold={m.get('gold_answer')!r}  pred={r.get('answer')!r}")
        lines.append(f"  label_seq: {seq}   reject_reason={m.get('reject_reason')!r}")
        for s in steps:
            cited = s.get("cited_triples") or []
            raw = str(s.get("text", ""))
            if full:
                lines.append(
                    f"     step{s.get('index')} label={LABEL_NAMES[s.get('label')]}  cited={len(cited)}"
                )
                for tl in raw.splitlines():
                    lines.append(f"        {tl}")
                for t in cited:
                    lines.append(f"        cited_triple: {tuple(t)}")
            else:
                text = " ".join(raw.split())[:90]
                lines.append(
                    f"     step{s.get('index')} label={LABEL_NAMES[s.get('label')]}  "
                    f"cited={len(cited)}  {text}"
                )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", default=_DEFAULT_INPUT,
                   help=f"silver trajectory JSONL (default: {_DEFAULT_INPUT})")
    p.add_argument("-o", "--output", default=None,
                   help="output text path (default: <input_dir>/label_record_<stem>.txt)")
    p.add_argument("--full", action="store_true",
                   help="emit untruncated question + full per-step text + cited triples "
                        "(default: short single-line summaries).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"input not found: {in_path}")

    rows = _load(in_path)
    out_path = Path(args.output) if args.output else in_path.parent / f"label_record_{in_path.stem}.txt"

    record = build_record(rows, in_path, full=args.full)
    out_path.write_text(record, encoding="utf-8")

    # console summary
    print(_fmt_dist(_label_distribution(rows)).join(["LABEL DISTRIBUTION: ", ""]))
    print(f"label record written -> {out_path}")
    print(f"  ({len(rows)} trajectories, {sum(len(r.get('steps', [])) for r in rows)} steps)")


if __name__ == "__main__":
    main()
