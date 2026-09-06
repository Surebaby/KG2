#!/usr/bin/env python
"""Materialize frozen QPEG-v4 development/confirmation retrieval contexts."""

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


EXPERIMENT_ID = "QPEG-V4-SCHEMA-ADAPT-EVAL-RETRIEVAL-SEED42"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests", type=Path,
        default=Path("outputs/audits/qpeg_v4_schema_adaptation_protocol_v1/retrieval_requests.jsonl"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/audits/qpeg_v4_schema_adaptation_eval_retrieval_v1"),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite retrieval output: {args.out}")
    rows = _read_jsonl(args.requests)
    if len(rows) != 450:
        raise ValueError(f"expected 450 requests, got {len(rows)}")
    for row in rows:
        present = FORBIDDEN_FIELDS & set(row)
        if present or row.get("gold_access") is not False:
            raise ValueError(f"{row.get('question_key')}: forbidden/gold-bearing request")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    args.out.mkdir(parents=True)
    all_contexts: list[dict] = []
    counts: dict[str, dict[str, int]] = {}
    for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
        dataset_rows = grouped[dataset]
        role_counts = Counter(str(row["role"]) for row in dataset_rows)
        if role_counts != {"development": 50, "confirmation": 100}:
            raise ValueError(f"{dataset}: unexpected role counts {role_counts}")
        contexts = materialize_dataset(dataset, dataset_rows, args.batch_size)
        _write_jsonl(args.out / f"{dataset}.retrieval_contexts.jsonl", contexts)
        all_contexts.extend(contexts)
        counts[dataset] = dict(role_counts)
    _write_jsonl(args.out / "retrieval_contexts.jsonl", all_contexts)
    report = {
        "schema_version": "qpeg-v4-eval-retrieval-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_NOT_EVALUATED",
        "counts": counts,
        "total": len(all_contexts),
        "requests_sha256": _sha_file(args.requests),
        "contexts_sha256": _sha_file(args.out / "retrieval_contexts.jsonl"),
        "retrieval": "E5@100 + BM25@100 -> RRF(k=60)@50 -> bge-reranker-v2-m3@10 -> pack3860",
        "gold_access": False,
        "all_have_10_passages": all(len(row["passages"]) == 10 for row in all_contexts),
        "identity_unique": len({row["question_key"] for row in all_contexts}) == 450,
        "confirmation_predictions_opened": False,
    }
    if not report["all_have_10_passages"] or not report["identity_unique"]:
        raise RuntimeError(f"retrieval gates failed: {report}")
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra={"phase": "qpeg_v4_eval_retrieval", **report}, status=report["status"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
