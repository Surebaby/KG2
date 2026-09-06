#!/usr/bin/env python
"""Materialize rollout-only silver from complete automatic ProofKG records."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.reward.proofkg_process import is_automatic_proofkg
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - artifact identity, not security
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_silver", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--cohort_report", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--runtime_question_kg", required=True)
    parser.add_argument("--evidence_store_manifest", required=True)
    parser.add_argument("--min_complete", type=int, default=600)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    args = parser.parse_args()

    source_path = Path(args.source_silver).resolve()
    cohort_path = Path(args.cohort).resolve()
    cohort_report_path = Path(args.cohort_report).resolve()
    protocol_path = Path(args.protocol).resolve()
    runtime_path = Path(args.runtime_question_kg).resolve()
    store_manifest_path = Path(args.evidence_store_manifest).resolve()
    cohort_report = json.loads(cohort_report_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cohort_rows = list(_read_jsonl(cohort_path))
    cohort = {str(row["question_key"]): row for row in cohort_rows}
    if len(cohort) != len(cohort_rows):
        raise SystemExit("duplicate question_key in cohort")
    expected_cohort = cohort_report.get("outputs", {}).get("cohort", {})
    if _file_md5(cohort_path) != str(expected_cohort.get("md5") or ""):
        raise SystemExit("cohort MD5 does not match the frozen cohort report")
    if _file_sha256(cohort_path) != str(protocol.get("cohort", {}).get("sha256") or ""):
        raise SystemExit("cohort SHA256 does not match the frozen protocol")

    confirmation_qids: set[str] = set()
    confirmation_families: set[str] = set()
    confirmation_inputs = cohort_report.get("inputs", {}).get("excluded_cohorts", [])
    confirmation_md5s: set[str] = set()
    for identity in confirmation_inputs:
        path = Path(str(identity.get("path") or ""))
        expected_md5 = str(identity.get("md5") or "")
        if not path.is_file() or _file_md5(path) != expected_md5:
            raise SystemExit(f"confirmation cohort missing or changed: {path}")
        confirmation_md5s.add(expected_md5)
        for row in _read_jsonl(path):
            confirmation_qids.add(str(row.get("qid") or ""))
            family = str(row.get("family_sha256") or "")
            if family:
                confirmation_families.add(family)
    train_qids = {str(row.get("qid") or "") for row in cohort_rows}
    train_families = {
        str(row.get("family_sha256") or "") for row in cohort_rows
        if row.get("family_sha256")
    }
    confirmation_qid_overlap = len(train_qids & confirmation_qids)
    confirmation_family_overlap = len(train_families & confirmation_families)

    store_manifest = json.loads(store_manifest_path.read_text(encoding="utf-8"))
    cohort_md5 = _file_md5(cohort_path)
    excluded = {
        str(row.get("md5") or "")
        for row in store_manifest.get("inputs", {}).get("excluded_cohorts", [])
    }
    if cohort_md5 not in excluded:
        raise SystemExit(
            "evidence store manifest does not prove exclusion of the selected training cohort"
        )
    if not confirmation_md5s.issubset(excluded):
        raise SystemExit(
            "evidence store manifest does not prove exclusion of every confirmation cohort"
        )

    source = {}
    for row in _read_jsonl(source_path):
        if str(row.get("dataset")) != "2wikimultihopqa":
            continue
        key = question_key("2wikimultihopqa", str(row.get("qid") or ""))
        if key in cohort:
            source[key] = row
    if set(source) != set(cohort):
        raise SystemExit(
            f"source silver join mismatch: source={len(source)} cohort={len(cohort)}"
        )

    complete_records = []
    seen = set()
    identity_matches = 0
    for record in _read_jsonl(runtime_path):
        key = str(record.get("question_key") or "")
        if key not in cohort:
            raise SystemExit(f"runtime record outside frozen cohort: {key}")
        if key in seen:
            raise SystemExit(f"duplicate runtime question_key: {key}")
        seen.add(key)
        expected = cohort[key]
        if (
            str(record.get("dataset") or "") != str(expected.get("dataset") or "")
            or str(record.get("qid") or "") != str(expected.get("qid") or "")
            or str(record.get("question_sha256") or "")
            != str(expected.get("question_sha256") or "")
        ):
            raise SystemExit(f"runtime identity mismatch: {key}")
        identity_matches += 1
        kg = record.get("kg_subgraph") or []
        runtime = {
            "query_plan": record.get("query_plan") or {},
            "provenance": record.get("provenance") or {},
        }
        if is_automatic_proofkg(runtime, kg):
            complete_records.append(record)
    if seen != set(cohort):
        raise SystemExit(
            f"runtime/cohort identity join mismatch: runtime={len(seen)} cohort={len(cohort)}"
        )
    if len(complete_records) < args.min_complete:
        raise SystemExit(
            f"only {len(complete_records)} complete automatic proofs; need {args.min_complete}"
        )

    # Preserve the frozen cohort order while dropping incomplete executions.
    complete_by_key = {str(row["question_key"]): row for row in complete_records}
    ordered_records = [
        complete_by_key[key] for key in cohort if key in complete_by_key
    ]
    silver_rows = []
    for record in ordered_records:
        key = str(record["question_key"])
        original = source[key]
        if question_sha256(str(original["question"])) != str(record["question_sha256"]):
            raise SystemExit(f"question hash mismatch: {key}")
        gold_answer = str((original.get("metadata") or {}).get("gold_answer") or "").strip()
        if not gold_answer:
            raise SystemExit(f"missing gold outcome answer: {key}")
        source_meta = original.get("metadata") or {}
        silver_rows.append({
            "qid": str(original["qid"]),
            "question": str(original["question"]),
            "answer": gold_answer,
            "dataset": "2wikimultihopqa",
            # PPO never consumes a supervised trace from this file.  Keeping the
            # list empty makes accidental same-file CE replay auditable.
            "steps": [],
            "kg_subgraph": record["kg_subgraph"],
            "retrieved_passages": list(original.get("retrieved_passages") or []),
            "accepted": True,
            "metadata": {
                "gold_answer": gold_answer,
                "question_type": source_meta.get("question_type"),
                "source_split": "train",
                "automatic_proofkg_rollout_only": True,
                "source_gold_trace_removed": True,
                "source_curriculum_gold_derived": bool(source_meta.get("gold_derived")),
                "evaluation_eligible": False,
            },
            "teacher_output": "",
            "teacher_model": "none_ppo_rollout_only",
        })

    question_kg_identity_join_rate = identity_matches / len(cohort) if cohort else 0.0
    proofkg_reward_eligible_rate = (
        sum(
            is_automatic_proofkg(
                {
                    "query_plan": row.get("query_plan") or {},
                    "provenance": row.get("provenance") or {},
                },
                row.get("kg_subgraph") or [],
            )
            for row in ordered_records
        ) / len(ordered_records)
        if ordered_records else 0.0
    )
    source_gold_steps_copied = sum(len(row["steps"]) for row in silver_rows)
    actual_gates = {
        "question_kg_identity_join_rate": question_kg_identity_join_rate,
        "proofkg_reward_eligible_rate": proofkg_reward_eligible_rate,
        "confirmation_qid_overlap": confirmation_qid_overlap,
        "confirmation_family_overlap": confirmation_family_overlap,
        "source_gold_steps_copied": source_gold_steps_copied,
    }
    expected_gates = protocol.get("materialization_gates", {})
    failed_gates = {
        key: {"expected": expected_gates.get(key), "actual": value}
        for key, value in actual_gates.items()
        if value != expected_gates.get(key)
    }
    if args.min_complete != int(expected_gates.get("complete_automatic_proof_target_min", -1)):
        failed_gates["complete_automatic_proof_target_min"] = {
            "expected": expected_gates.get("complete_automatic_proof_target_min"),
            "actual": args.min_complete,
        }
    if failed_gates:
        raise SystemExit(f"materialization gates failed: {failed_gates}")

    out_dir, experiment_id = prepare_new_run_dir(
        args.output_dir,
        experiment_id=args.experiment_id,
        extra={"phase": "materialize_automatic_proofkg_ppo_rollout_data"},
    )
    silver_out = out_dir / "silver_train.jsonl"
    records_out = out_dir / "question_kg_records.jsonl"
    for path, rows in ((silver_out, silver_rows), (records_out, ordered_records)):
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "status": "COMPLETE_NOT_TRAINED",
        "counts": {
            "cohort": len(cohort),
            "runtime_records": len(seen),
            "complete_automatic_proofs": len(ordered_records),
            "by_question_type": dict(Counter(
                row["metadata"]["question_type"] for row in silver_rows
            )),
            "source_gold_steps_copied": source_gold_steps_copied,
            "nonempty_kg": sum(bool(row["kg_subgraph"]) for row in silver_rows),
        },
        "materialization_gates": {
            key: {"expected": expected_gates[key], "actual": value, "passed": True}
            for key, value in actual_gates.items()
        },
        "scientific_boundary": {
            "train_only": True,
            "selected_question_gold_used_for_planner_or_kg": False,
            "gold_final_answer_used_only_by_ppo_outcome_reward": True,
            "gold_process_trace_available_to_ppo": False,
            "evidence_store_excluded_selected_families": True,
            "evaluation_eligible": False,
            "ppo_started": False,
        },
        "inputs": {
            "source_silver": artifact_identity(source_path),
            "cohort": artifact_identity(cohort_path),
            "cohort_report": artifact_identity(cohort_report_path),
            "protocol": artifact_identity(protocol_path),
            "confirmation_cohorts": [
                artifact_identity(identity["path"]) for identity in confirmation_inputs
            ],
            "runtime_question_kg": artifact_identity(runtime_path),
            "evidence_store_manifest": artifact_identity(store_manifest_path),
        },
        "outputs": {
            "silver_train": artifact_identity(silver_out),
            "question_kg_records": artifact_identity(records_out),
        },
    }
    report_path = out_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(out_dir, status=report["status"], extra=report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
