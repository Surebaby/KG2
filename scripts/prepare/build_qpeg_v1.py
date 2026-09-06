#!/usr/bin/env python
"""Build versioned QPEG-v1 records from frozen retrieval contexts."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.kg.qpeg import QPEG_EXTRACTOR_VERSION, build_qpeg_record
from kgproweight.utils.logging import dump_manifest


DEFAULT_EXPERIMENT_ID = "QPEG-V1-N1350-SEED42-MATERIALIZATION"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max_edges", type=int, default=12)
    parser.add_argument("--experiment_id", default=DEFAULT_EXPERIMENT_ID)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite QPEG output: {args.out}")
    args.out.mkdir(parents=True)

    contexts: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    seen: set[str] = set()
    for path in args.contexts:
        input_hashes[str(path)] = _sha_file(path)
        for row in _read_jsonl(path):
            key = str(row["question_key"])
            if key in seen:
                raise ValueError(f"duplicate context question_key: {key}")
            seen.add(key)
            contexts.append(row)

    records: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for context in contexts:
        record = build_qpeg_record(
            dataset=str(context["dataset"]),
            qid=str(context["qid"]),
            question=str(context["question"]),
            passages=context["passages"],
            passages_sha256=str(context["passages_sha256"]),
            max_edges=args.max_edges,
        )
        role = str(context.get("role") or "unknown")
        record["role"] = role
        record["family_sha256"] = context.get("family_sha256")
        records.append(record)
        counts[f"dataset::{record['dataset']}"] += 1
        counts[f"role::{role}"] += 1
        counts[f"nonempty::{record['dataset']}"] += int(record["build_status"] == "nonempty")
        counts[f"edges::{record['dataset']}"] += len(record["edges"])
        detail_rows.append({
            "question_key": record["question_key"],
            "dataset": record["dataset"],
            "qid": record["qid"],
            "role": role,
            "edge_count": len(record["edges"]),
            "candidate_count": record["candidate_count"],
            "build_status": record["build_status"],
            "qpeg_sha256": record["qpeg_sha256"],
            "rules": dict(Counter(edge["extraction_rule"] for edge in record["edges"])),
        })

    _write_jsonl(args.out / "question_graph_records.jsonl", records)
    _write_jsonl(args.out / "build_details.jsonl", detail_rows)
    datasets: dict[str, Any] = {}
    all_structure_pass = True
    for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
        n = counts[f"dataset::{dataset}"]
        nonempty = counts[f"nonempty::{dataset}"]
        rate = nonempty / max(1, n)
        datasets[dataset] = {
            "n": n,
            "nonempty": nonempty,
            "nonempty_rate": rate,
            "total_edges": counts[f"edges::{dataset}"],
            "mean_edges": counts[f"edges::{dataset}"] / max(1, n),
            "structure_gate_nonempty_ge_0_80": rate >= 0.80,
        }
        all_structure_pass = all_structure_pass and rate >= 0.80
    report = {
        "schema_version": "qpeg-materialization-report-v1",
        "experiment_id": args.experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_STRUCTURE" if all_structure_pass else "FAIL_STOP_STRUCTURE",
        "extractor_version": QPEG_EXTRACTOR_VERSION,
        "n": len(records),
        "role_counts": {key.removeprefix("role::"): value for key, value in counts.items() if key.startswith("role::")},
        "datasets": datasets,
        "gates": {
            "identity_unique": len(seen) == len(records),
            "gold_access_false": all(record["gold_access"] is False for record in records),
            "provenance_complete_for_nonempty": all(
                record["build_status"] == "empty" or record["provenance_complete"]
                for record in records
            ),
            "max_edges_le_12": all(len(record["edges"]) <= 12 for record in records),
            "three_dataset_nonempty_ge_0_80": all_structure_pass,
        },
        "inputs": input_hashes,
    }
    if not all(value for key, value in report["gates"].items() if key != "three_dataset_nonempty_ge_0_80"):
        raise SystemExit(f"hard QPEG integrity gate failed: {report['gates']}")
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(args.out, extra={"phase": "qpeg_materialization", **report}, status=report["status"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all_structure_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
