#!/usr/bin/env python
"""Build a question→KG index by copying a repaired silver file's stored KG.

Unlike the general index builder, this tool does not relink entities or refilter
triples.  It preserves the exact KG view used to annotate the derived silver
labels, so PPO prompt KG and online reward verification cannot silently diverge.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from kgproweight.kg.kg_filter import _pid_for_triple
from kgproweight.utils.logging import dump_manifest


BUILDER_VERSION = "stored-silver-exact-1"
Triple = Tuple[str, str, str]


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _triple(value: Sequence[Any]) -> Triple:
    return tuple(str(part).strip() for part in value)  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--max_kg_triples", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    silver_path = Path(args.silver).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    run_dir = output_path.with_name(f"{output_path.stem}_run")
    for path in (output_path, report_path, run_dir):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing path: {path}")
    if args.max_kg_triples != 12:
        raise SystemExit("stored legacy-repair index protocol requires max=12")

    source_md5 = _md5(silver_path)
    by_question: Dict[str, Dict[str, Any]] = {}
    question_kg: Dict[str, List[Triple]] = {}
    datasets: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    accepted_empty = 0
    accepted_total = 0

    with silver_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            counts["records"] += 1
            question = str(row.get("question") or "").strip()
            if not question:
                raise SystemExit(f"{silver_path}:{lineno}: empty question")
            dataset = str(row.get("dataset") or "UNKNOWN")
            datasets[dataset] += 1
            triples = [
                _triple(value)
                for value in row.get("kg_subgraph") or []
                if isinstance(value, (list, tuple)) and len(value) == 3
            ]
            if len(triples) > args.max_kg_triples:
                raise SystemExit(
                    f"{silver_path}:{lineno}: KG has {len(triples)} > {args.max_kg_triples}"
                )
            if not triples:
                counts["empty_kg_records"] += 1
            if row.get("accepted"):
                accepted_total += 1
                accepted_empty += int(not triples)

            previous = question_kg.get(question)
            if previous is not None:
                counts["duplicate_question_records"] += 1
                if previous != triples:
                    raise SystemExit(
                        f"{silver_path}:{lineno}: duplicate question has conflicting stored KG"
                    )
                continue
            question_kg[question] = triples
            by_question[question] = {
                "question_id": str(row.get("qid") or row.get("id") or ""),
                "question": question,
                "dataset": dataset,
                "linked_entities": [],
                "triples": [
                    {
                        "h": head,
                        "pid": _pid_for_triple((head, relation, tail)),
                        "r": relation,
                        "t": tail,
                        "provenance": "stored_silver_exact",
                    }
                    for head, relation, tail in triples
                ],
                "n_before": len(triples),
                "n_after": len(triples),
                "builder_version": BUILDER_VERSION,
                "relation_policy_version": "inherited_from_repaired_silver",
                "source_silver_md5": source_md5,
            }

    entries = list(by_question.values())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "status": "BUILT_NOT_TRAINED",
        "builder_version": BUILDER_VERSION,
        "source": {
            "path": str(silver_path),
            "md5": source_md5,
            "read_only": True,
        },
        "output": {
            "path": str(output_path),
            "md5": _md5(output_path),
            "entries": len(entries),
        },
        "counts": dict(counts),
        "datasets": dict(datasets),
        "accepted": {
            "total": accepted_total,
            "empty_kg": accepted_empty,
            "nonempty_kg": accepted_total - accepted_empty,
            "nonempty_rate_pct": 100.0 * (accepted_total - accepted_empty) / max(1, accepted_total),
        },
        "integrity": {
            "max_kg_triples": args.max_kg_triples,
            "duplicate_conflicts": 0,
            "triples_copied_without_relink_or_refilter": True,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(run_dir, extra={
        "experiment_id": args.experiment_id,
        "phase": "build_stored_silver_kg_index",
        "builder_version": BUILDER_VERSION,
        "silver": str(silver_path),
        "silver_md5": source_md5,
        "output": str(output_path),
        "output_md5": report["output"]["md5"],
        "entries": len(entries),
        "accepted": accepted_total,
        "accepted_empty": accepted_empty,
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
