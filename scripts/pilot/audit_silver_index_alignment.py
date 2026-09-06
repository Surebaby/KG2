#!/usr/bin/env python
"""Audit exact silver/index KG alignment for one deterministic data fold."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from kgproweight.data.silver_dataset import SilverTrajectory
from kgproweight.data.silver_split import SplitSpec, assign_split


Triple = Tuple[str, str, str]


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_triples(entry: Dict[str, Any]) -> List[Triple]:
    triples = entry.get("triples") or entry.get("t") or []
    out: List[Triple] = []
    for triple in triples:
        if isinstance(triple, dict):
            out.append((str(triple["h"]), str(triple["r"]), str(triple["t"])))
        elif isinstance(triple, (list, tuple)) and len(triple) == 3:
            out.append(tuple(str(value) for value in triple))  # type: ignore[arg-type]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--split_seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    silver_path = Path(args.silver).resolve()
    index_path = Path(args.index).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite {output_path}")

    index_rows = json.loads(index_path.read_text(encoding="utf-8"))
    index = {
        str(row.get("question") or row.get("q") or ""): _index_triples(row)
        for row in index_rows
    }
    spec = SplitSpec(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
    )
    counts: Counter[str] = Counter()
    mismatch_examples: List[Dict[str, Any]] = []
    seen_qids = set()

    with silver_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            counts["records"] += 1
            if not row.get("accepted"):
                continue
            counts["accepted"] += 1
            traj = SilverTrajectory.from_dict(row)
            if assign_split(traj, spec) != args.split:
                continue
            counts["fold_accepted"] += 1
            seen_qids.add(traj.qid)
            if traj.question not in index:
                counts["absent"] += 1
                continue
            indexed = index[traj.question]
            stored = list(traj.kg_subgraph)
            if not indexed:
                counts["covered_empty"] += 1
            else:
                counts["covered_nonempty"] += 1
            if indexed != stored:
                counts["kg_mismatch"] += 1
                if len(mismatch_examples) < 10:
                    mismatch_examples.append({
                        "qid": traj.qid,
                        "line": lineno,
                        "stored_count": len(stored),
                        "indexed_count": len(indexed),
                    })

    fold_n = counts["fold_accepted"]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if counts["absent"] == 0 and counts["kg_mismatch"] == 0 else "FAIL",
        "source": {"path": str(silver_path), "md5": _md5(silver_path)},
        "index": {"path": str(index_path), "md5": _md5(index_path), "entries": len(index)},
        "split": {
            "name": args.split,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.split_seed,
        },
        "counts": dict(counts),
        "unique_fold_qids": len(seen_qids),
        "rates": {
            "absent_pct": 100.0 * counts["absent"] / max(1, fold_n),
            "covered_empty_pct": 100.0 * counts["covered_empty"] / max(1, fold_n),
            "covered_nonempty_pct": 100.0 * counts["covered_nonempty"] / max(1, fold_n),
            "kg_mismatch_pct": 100.0 * counts["kg_mismatch"] / max(1, fold_n),
        },
        "mismatch_examples": mismatch_examples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
