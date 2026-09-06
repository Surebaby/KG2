#!/usr/bin/env python
"""Measure untouched family capacity for a future query-planner confirmation."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from kgproweight.training.query_planner import balanced_sample
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _raw_annotation_coverage(path: Path, dataset: str) -> dict[str, Any]:
    rows = list(_read_jsonl(path))
    if dataset == "2wikimultihopqa":
        annotated = sum(bool(((row.get("metadata") or {}).get("evidences"))) for row in rows)
        annotation = "metadata.evidences"
    else:
        annotated = sum(bool((((row.get("metadata") or {}).get("metadata") or {})
                              .get("question_decomposition"))) for row in rows)
        annotation = "metadata.metadata.question_decomposition"
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "n": len(rows),
        "required_annotation": annotation,
        "annotated_n": annotated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--evaluated_per_dataset", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    split_root = Path(args.split_root)
    dev_path = split_root / "dev.jsonl"
    assignment_path = split_root / "assignments.jsonl"
    output_dir, experiment_id = prepare_new_run_dir(
        args.output_dir,
        extra={
            "phase": "planner_v2_holdout_capacity_audit",
            "dev": artifact_identity(dev_path),
            "assignments": artifact_identity(assignment_path),
        },
    )
    evaluated = balanced_sample(
        dev_path, per_dataset=args.evaluated_per_dataset, seed=args.seed
    )
    evaluated_keys = {row["question_key"] for row in evaluated}
    assignments = {row["question_key"]: row for row in _read_jsonl(assignment_path)}
    evaluated_families: dict[str, set[str]] = defaultdict(set)
    for key in evaluated_keys:
        assignment = assignments[key]
        evaluated_families[assignment["dataset"]].add(assignment["family_sha256"])

    dev_rows = list(_read_jsonl(dev_path))
    capacity: dict[str, Any] = {}
    for dataset in ("2wikimultihopqa", "musique"):
        rows = [row for row in dev_rows if row["dataset"] == dataset]
        family = lambda row: assignments[row["question_key"]]["family_sha256"]
        untouched = [row for row in rows if family(row) not in evaluated_families[dataset]]
        remaining_seen_family = [
            row for row in rows
            if row["question_key"] not in evaluated_keys
            and family(row) in evaluated_families[dataset]
        ]
        capacity[dataset] = {
            "dev_rows": len(rows),
            "dev_families": len({family(row) for row in rows}),
            "evaluated_rows": sum(row["question_key"] in evaluated_keys for row in rows),
            "evaluated_families": len(evaluated_families[dataset]),
            "untouched_family_rows": len(untouched),
            "untouched_families": len({family(row) for row in untouched}),
            "remaining_rows_in_seen_families": len(remaining_seen_family),
        }

    raw_splits: dict[str, Any] = {}
    data_root = Path(args.data_root)
    for dataset in ("2wikimultihopqa", "musique"):
        dev = _raw_annotation_coverage(data_root / dataset / "dev.jsonl", dataset)
        test = _raw_annotation_coverage(data_root / dataset / "test.jsonl", dataset)
        raw_splits[dataset] = {
            "dev": dev,
            "test": test,
            "dev_test_identical": dev["sha256"] == test["sha256"],
        }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "status": "COMPLETE",
        "evaluated_dev_sample": {
            "per_dataset": args.evaluated_per_dataset,
            "seed": args.seed,
            "question_key_sha256": hashlib.sha256(
                "\n".join(row["question_key"] for row in evaluated).encode()
            ).hexdigest(),
        },
        "untouched_family_capacity": capacity,
        "raw_dev_test_annotation_audit": raw_splits,
        "interpretation": {
            "existing_structural_gold_max_rows_by_dataset": {
                dataset: values["untouched_family_rows"] for dataset, values in capacity.items()
            },
            "raw_dev_test_can_directly_supply_structural_gold": False,
            "reason": "local raw dev/test are identical per dataset and contain zero required decomposition/evidence annotations",
            "minimum_sample_adequacy": "UNKNOWN; must be preregistered before creating a v2 split",
        },
        "inputs": {
            "dev": artifact_identity(dev_path),
            "assignments": artifact_identity(assignment_path),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(output_dir, status="COMPLETE", extra=report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
