#!/usr/bin/env python
"""Augment an offline entity index from root labels in the local KG cache.

The subgraph cache key stores the root QID while cached triples store surface
labels.  This permits a fully offline, gold-free ``root label -> QID`` recovery
for entities that have a cached graph but are absent from ``entity_cache``.
The base index is never modified and outputs are create-only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Tuple

from scripts.prepare.build_entity_desc_index import _build_description


BUILDER_VERSION = "entity-desc-local-subgraph-roots-1"
_BAD_ROOT_LABELS = {
    "which", "what", "where", "when", "who", "whom", "whose", "why", "how",
    "the", "this", "that", "these", "those", "yes", "no",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def infer_root_label(triples: Iterable[object]) -> Tuple[str, float]:
    """Return the dominant head label and its share among valid triples."""
    heads: Counter[str] = Counter()
    display: Dict[str, str] = {}
    for value in triples:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            continue
        head = str(value[0]).strip()
        key = _clean(head)
        if not key:
            continue
        heads[key] += 1
        display.setdefault(key, head)
    if not heads:
        return "", 0.0
    key, count = heads.most_common(1)[0]
    return display[key], count / sum(heads.values())


def _valid_root_label(label: str) -> bool:
    clean = _clean(label)
    if clean in _BAD_ROOT_LABELS or len(clean) < 2:
        return False
    if re.fullmatch(r"\d+", clean):
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_index", required=True)
    parser.add_argument("--subgraph_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--min_root_share", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_index).resolve()
    cache_path = Path(args.subgraph_cache).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    for source in (base_path, cache_path):
        if not source.is_file():
            raise SystemExit(f"missing input file: {source}")
    for target in (output_path, report_path):
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing path: {target}")
    if not 0.0 <= args.min_root_share <= 1.0:
        raise SystemExit("min_root_share must be in [0, 1]")

    index: Dict[str, List[Dict[str, Any]]] = json.loads(
        base_path.read_text(encoding="utf-8")
    )
    existing_pairs = {
        (label, str(candidate.get("qid") or ""))
        for label, candidates in index.items()
        for candidate in candidates
    }

    counters: Counter[str] = Counter()
    added_by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with cache_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            counters["cache_rows"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counters["invalid_json"] += 1
                continue
            qid = str(row.get("key") or "").split("_", 1)[0]
            if not re.fullmatch(r"Q\d+", qid):
                counters["invalid_qid"] += 1
                continue
            triples = row.get("triples") or []
            label, share = infer_root_label(triples)
            label_key = _clean(label)
            if not _valid_root_label(label):
                counters["invalid_root_label"] += 1
                continue
            if share < args.min_root_share:
                counters["low_root_share"] += 1
                continue
            if (label_key, qid) in existing_pairs:
                counters["already_present"] += 1
                continue
            description = _build_description(triples, label)
            added_by_label[label_key].append(
                {
                    "qid": qid,
                    "label": label,
                    "description": description,
                    "source": "local_subgraph_root",
                    "root_head_share": round(share, 4),
                }
            )
            existing_pairs.add((label_key, qid))
            counters["added_candidates"] += 1

    for label_key in sorted(added_by_label):
        index.setdefault(label_key, []).extend(
            sorted(added_by_label[label_key], key=lambda item: item["qid"])
        )
    counters["added_labels"] = len(added_by_label)
    counters["output_labels"] = len(index)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder_version": BUILDER_VERSION,
        "status": "COMPLETE_NOT_MODEL_EVALUATED",
        "protocol": {
            "offline_only": True,
            "gold_used": False,
            "base_index_read_only": True,
            "subgraph_cache_read_only": True,
            "root_label_rule": "dominant triple head label",
            "min_root_share": args.min_root_share,
        },
        "inputs": {
            "base_index": str(base_path),
            "base_index_sha256": _sha256(base_path),
            "subgraph_cache": str(cache_path),
            "subgraph_cache_sha256": _sha256(cache_path),
        },
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
        "counts": dict(counters),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"counts": dict(counters), "output": report["output"]}, indent=2))


if __name__ == "__main__":
    main()
