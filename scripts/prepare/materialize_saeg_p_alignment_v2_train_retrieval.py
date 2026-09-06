#!/usr/bin/env python
"""Materialise canonical answer-free retrieval for the frozen P-alignment train cohort."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from kgproweight.utils.logging import dump_manifest
from scripts.prepare.materialize_qpeg_v1_retrieval import (
    FORBIDDEN_FIELDS,
    _read_jsonl,
    _sha_file,
    _write_jsonl,
    materialize_dataset,
)


EXPECTED_PROTOCOL_STATUS = "FROZEN_BEFORE_TRAIN_RETRIEVAL_DATA_BUILD_OR_MODEL_UPDATE"
EXPERIMENT_ID = "SAEG-P-ALIGNMENT-V2-TRAIN1800-RETRIEVAL-SEED42"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("outputs/audits/saeg_p_hard_negative_alignment_v2_protocol/protocol.json"),
    )
    parser.add_argument(
        "--requests",
        type=Path,
        default=Path("outputs/audits/qpeg_v4_schema_adaptation_protocol_v1/train.question_only.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/audits/saeg_p_alignment_v2_train1800_retrieval"),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite retrieval output: {args.out}")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != EXPECTED_PROTOCOL_STATUS:
        raise ValueError("unexpected hard-negative protocol status")
    expected_requests = protocol["train_cohort"]
    if _sha_file(args.requests) != expected_requests["sha256"]:
        raise ValueError("retrieval request hash differs from frozen protocol")

    rows = _read_jsonl(args.requests)
    if len(rows) != 1800:
        raise ValueError(f"expected 1800 retrieval requests, got {len(rows)}")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        present = FORBIDDEN_FIELDS & set(row)
        if present or row.get("gold_access") is not False or row.get("role") != "train":
            raise ValueError(f"{row.get('question_key')}: invalid answer-free train retrieval request")
        grouped[str(row["dataset"])].append(row)
    expected_datasets = {"hotpotqa", "2wikimultihopqa", "musique"}
    if set(grouped) != expected_datasets or any(len(grouped[name]) != 600 for name in grouped):
        raise ValueError(f"unexpected train dataset counts: {Counter(row['dataset'] for row in rows)}")

    args.out.mkdir(parents=True, exist_ok=False)
    all_contexts: list[dict] = []
    for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
        contexts = materialize_dataset(dataset, grouped[dataset], args.batch_size)
        for row in contexts:
            row["role"] = "train_alignment_v2"
        _write_jsonl(args.out / f"{dataset}.retrieval_contexts.jsonl", contexts)
        all_contexts.extend(contexts)
    _write_jsonl(args.out / "retrieval_contexts.jsonl", all_contexts)

    forbidden_output = sum(bool(FORBIDDEN_FIELDS & set(row)) for row in all_contexts)
    report = {
        "schema_version": "saeg-p-alignment-v2-train-retrieval-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_ANSWER_FREE_NOT_CLASSIFIED_NOT_TRAINED",
        "counts": {
            "total": len(all_contexts),
            "by_dataset": dict(Counter(row["dataset"] for row in all_contexts)),
        },
        "integrity": {
            "identity_unique": len({row["question_key"] for row in all_contexts}) == 1800,
            "all_have_10_passages": all(len(row.get("passages") or []) == 10 for row in all_contexts),
            "all_gold_access_false": all(row.get("gold_access") is False for row in all_contexts),
            "forbidden_top_level_fields": forbidden_output,
        },
        "retrieval": protocol["automatic_input_path"]["retrieval"],
        "inputs": {
            "protocol": {"path": str(args.protocol), "sha256": _sha_file(args.protocol)},
            "requests": {"path": str(args.requests), "sha256": _sha_file(args.requests)},
        },
        "output_sha256": _sha_file(args.out / "retrieval_contexts.jsonl"),
        "scientific_boundary": "Answer-free train retrieval only; no Gold fields, labels, targets, or model updates.",
    }
    integrity_pass = (
        report["integrity"]["identity_unique"]
        and report["integrity"]["all_have_10_passages"]
        and report["integrity"]["all_gold_access_false"]
        and report["integrity"]["forbidden_top_level_fields"] == 0
    )
    if not integrity_pass:
        raise RuntimeError(f"retrieval integrity gates failed: {report['integrity']}")
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra={"phase": "saeg_p_alignment_v2_train_retrieval", **report}, status=report["status"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
