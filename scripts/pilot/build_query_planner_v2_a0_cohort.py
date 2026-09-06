#!/usr/bin/env python
"""Freeze seen-diagnostic anchor-alias cases before A0 retrieval is run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

from kgproweight.eval.query_planner import _dependency_edges, _norm, _pid_sequence
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _anchors(target: dict[str, Any]) -> list[str]:
    return [str(value) for value in target.get("anchors") or []]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir, experiment_id = prepare_new_run_dir(
        args.output_dir,
        extra={
            "phase": "planner_v2_a0_cohort_freeze",
            "predictions": artifact_identity(args.predictions),
            "records": artifact_identity(args.records),
            "n": args.n,
            "seed": args.seed,
        },
    )
    records = {row["question_key"]: row for row in _read_jsonl(args.records)}
    eligible: list[dict[str, Any]] = []
    for row in _read_jsonl(args.predictions):
        if row.get("dataset") != "2wikimultihopqa" or not row.get("schema_valid"):
            continue
        gold = row.get("gold_target") or {}
        predicted = row.get("predicted_target") or {}
        if _pid_sequence(gold) != _pid_sequence(predicted):
            continue
        if _dependency_edges(gold) != _dependency_edges(predicted):
            continue
        gold_anchors, predicted_anchors = _anchors(gold), _anchors(predicted)
        if not gold_anchors or len(gold_anchors) != len(predicted_anchors):
            continue
        if sorted(_norm(value) for value in gold_anchors) == sorted(
            _norm(value) for value in predicted_anchors
        ):
            continue
        source = records.get(row["question_key"]) or {}
        normalized_question = _norm(source.get("question") or "")
        predicted_in_question = all(
            _norm(value) in normalized_question for value in predicted_anchors
        )
        gold_in_question = all(_norm(value) in normalized_question for value in gold_anchors)
        if not predicted_in_question or gold_in_question:
            continue
        eligible.append({
            "question_key": row["question_key"],
            "qid": row["qid"],
            "question": source.get("question"),
            "predicted_anchors": predicted_anchors,
            "gold_alias_anchors": gold_anchors,
            "pid_sequence": _pid_sequence(gold),
            "scope": "seen_confirmation_diagnostics_only",
        })
    if len(eligible) < args.n:
        raise SystemExit(f"eligible={len(eligible)} is smaller than requested n={args.n}")
    eligible.sort(key=lambda row: row["question_key"])
    selected = random.Random(args.seed).sample(eligible, args.n)
    selected.sort(key=lambda row: row["question_key"])
    cohort_path = output_dir / "cohort.jsonl"
    with cohort_path.open("x", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "status": "FROZEN_BEFORE_RETRIEVAL",
        "eligibility": {
            "dataset": "2wikimultihopqa",
            "schema_valid": True,
            "pid_sequence_exact": True,
            "dependency_exact": True,
            "anchor_exact": False,
            "predicted_anchors_all_question_substrings": True,
            "gold_anchors_not_all_question_substrings": True,
            "anchor_count_equal": True,
        },
        "eligible_n": len(eligible),
        "selected_n": len(selected),
        "seed": args.seed,
        "selected_question_key_sha256": hashlib.sha256(
            "\n".join(row["question_key"] for row in selected).encode()
        ).hexdigest(),
        "cohort": artifact_identity(cohort_path),
        "inputs": {
            "predictions": artifact_identity(args.predictions),
            "records": artifact_identity(args.records),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(output_dir, status="COMPLETE", extra=report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
