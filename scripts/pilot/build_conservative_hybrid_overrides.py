#!/usr/bin/env python
"""Build fixed-cohort conservative old+bridge passage overrides.

The builder never reads gold answers.  It preserves the leading stored passages
verbatim and fills only the remaining prompt slots with deduplicated bridge
retrieval results.  The source silver file is read-only and KG is not overridden.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from kgproweight.utils.logging import dump_manifest


BUILDER_VERSION = "conservative-hybrid-passages-3"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _passage_text(passage: Mapping[str, Any] | str) -> str:
    if isinstance(passage, Mapping):
        return str(passage.get("contents") or passage.get("text") or "").strip()
    return str(passage).strip()


def _normalise(value: str) -> str:
    value = value.strip().strip('"').strip("'").casefold()
    return re.sub(r"\s+", " ", value)


def _passage_keys(passage: Mapping[str, Any] | str) -> Tuple[str, ...]:
    """Return stable keys without assuming IDs match across corpus versions."""
    text = _passage_text(passage)
    keys: List[str] = []
    if isinstance(passage, Mapping) and passage.get("id") is not None:
        keys.append(f"id:{_normalise(str(passage['id']))}")
    if text:
        title = _normalise(text.splitlines()[0])
        if title:
            keys.append(f"title:{title}")
        keys.append(f"text:{hashlib.sha256(_normalise(text).encode()).hexdigest()}")
    return tuple(keys)


def build_hybrid_passages(
    old_passages: Sequence[Mapping[str, Any] | str],
    bridge_passages: Sequence[Mapping[str, Any] | str],
    *,
    old_keep: int,
    bridge_keep: int,
    total: int = 15,
) -> Tuple[List[Mapping[str, Any] | str], Dict[str, int]]:
    """Preserve old prefix, add unique bridge pages, then backfill old tail."""
    if old_keep < 0 or bridge_keep < 0 or total <= 0:
        raise ValueError("passage quotas must be non-negative and total must be positive")
    if old_keep + bridge_keep != total:
        raise ValueError("old_keep + bridge_keep must equal total")

    selected = list(old_passages[:old_keep])
    seen = {key for passage in selected for key in _passage_keys(passage)}
    bridge_added = 0
    # ``old_keep`` is a maximum: a few legacy records contain fewer stored
    # passages.  Keep the prompt protocol at ``total`` by allowing bridge
    # passages to fill those otherwise empty old-prefix slots.  For normal
    # records with >= old_keep passages this remains exactly bridge_keep.
    bridge_target = total - len(selected)
    bridge_skipped_duplicate = 0
    bridge_skipped_empty = 0
    for passage in bridge_passages:
        if bridge_added >= bridge_target:
            break
        keys = _passage_keys(passage)
        if not _passage_text(passage):
            bridge_skipped_empty += 1
            continue
        if any(key in seen for key in keys):
            bridge_skipped_duplicate += 1
            continue
        selected.append(passage)
        seen.update(keys)
        bridge_added += 1

    old_backfilled = 0
    for passage in old_passages[old_keep:]:
        if len(selected) >= total:
            break
        if not _passage_text(passage):
            continue
        # Backfill restores the source retriever's original tail.  The
        # same-page cap is for newly injected bridge evidence only; applying it
        # here can shorten a legacy prompt whose stored tail contains chunks
        # from a page already present in the prefix.
        keys = _passage_keys(passage)
        selected.append(passage)
        seen.update(keys)
        old_backfilled += 1

    return selected[:total], {
        "old_prefix": min(old_keep, len(old_passages)),
        "bridge_target": bridge_target,
        "bridge_added": bridge_added,
        "old_backfilled": old_backfilled,
        "bridge_skipped_duplicate": bridge_skipped_duplicate,
        "bridge_skipped_empty": bridge_skipped_empty,
        "final_passages": min(total, len(selected)),
    }


def _index_unique(rows: Iterable[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("qid") or row.get("id") or "")
        if not qid:
            raise SystemExit(f"{label} contains an empty qid")
        if qid in indexed:
            raise SystemExit(f"{label} contains duplicate qid: {qid}")
        indexed[qid] = row
    return indexed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--bridge_passages", required=True)
    parser.add_argument("--old_keep", type=int, required=True)
    parser.add_argument("--bridge_keep", type=int, required=True)
    parser.add_argument("--total", type=int, default=15)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort_path = Path(args.cohort).resolve()
    silver_path = Path(args.silver).resolve()
    bridge_path = Path(args.bridge_passages).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    run_dir = Path(args.run_dir).resolve()
    for path in (output_path, report_path, run_dir):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing path: {path}")
    if args.total != 15:
        raise SystemExit("this evaluation protocol is frozen at total=15")
    if args.old_keep + args.bridge_keep != args.total:
        raise SystemExit("old_keep + bridge_keep must equal total")

    cohort_rows = _read_jsonl(cohort_path)
    qids = [str(row.get("qid") or row.get("id") or "") for row in cohort_rows]
    if not qids or any(not qid for qid in qids) or len(qids) != len(set(qids)):
        raise SystemExit("cohort requires unique non-empty qids")
    wanted = set(qids)

    silver_rows: List[Dict[str, Any]] = []
    with silver_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("qid") or row.get("id") or "") in wanted:
                silver_rows.append(row)
    silver_by_qid = _index_unique(silver_rows, "silver")
    bridge_by_qid = _index_unique(_read_jsonl(bridge_path), "bridge passages")
    for label, indexed in (("silver", silver_by_qid), ("bridge passages", bridge_by_qid)):
        missing = [qid for qid in qids if qid not in indexed]
        if missing:
            raise SystemExit(f"{label} missing cohort qids: {missing}")

    output_rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for qid in qids:
        source = silver_by_qid[qid]
        bridge = bridge_by_qid[qid]
        question = str(source.get("question") or "").strip()
        if question != str(bridge.get("question") or "").strip():
            raise SystemExit(f"question mismatch for {qid}")
        passages, composition = build_hybrid_passages(
            list(source.get("retrieved_passages") or []),
            list(bridge.get("retrieved_passages") or []),
            old_keep=args.old_keep,
            bridge_keep=args.bridge_keep,
            total=args.total,
        )
        output_rows.append(
            {
                "qid": qid,
                "question": question,
                "retrieval_view": f"old{args.old_keep}_bridge{args.bridge_keep}",
                "retrieved_passages": passages,
            }
        )
        details.append({"qid": qid, **composition})
        counts["questions"] += 1
        for key, value in composition.items():
            counts[key] += value
        counts["short_prompt"] += int(len(passages) < args.total)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as fh:
        for row in output_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "status": "BUILT_NOT_EVALUATED",
        "builder_version": BUILDER_VERSION,
        "protocol": {
            "gold_used_for_build": False,
            "kg_changed": False,
            "old_keep": args.old_keep,
            "bridge_keep": args.bridge_keep,
            "total": args.total,
            "old_prefix_order_preserved": True,
            "bridge_dedup_keys": ["document id", "normalised title", "normalised text"],
            "bridge_same_page_limit": 1,
            "shortfall_policy": "backfill unique stored passages after old prefix",
        },
        "inputs": {
            "cohort": str(cohort_path),
            "cohort_sha256": _sha256(cohort_path),
            "silver": str(silver_path),
            "silver_sha256": _sha256(silver_path),
            "silver_read_only": True,
            "bridge_passages": str(bridge_path),
            "bridge_passages_sha256": _sha256(bridge_path),
        },
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
        "counts": dict(counts),
        "details": details,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        run_dir,
        extra={
            "experiment_id": args.experiment_id,
            "phase": "zero_training_conservative_hybrid_input_build",
            "builder_version": BUILDER_VERSION,
            "protocol": report["protocol"],
            "inputs": report["inputs"],
            "output": report["output"],
            "counts": dict(counts),
        },
    )
    print(json.dumps({"counts": dict(counts), "output": report["output"]}, indent=2))


if __name__ == "__main__":
    main()
