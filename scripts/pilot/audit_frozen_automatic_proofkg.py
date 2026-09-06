#!/usr/bin/env python
"""Post-freeze Gold audit for question-only automatic Proof-KG runtime artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir
from scripts.pilot.audit_query_aware_kg_coverage import _chain_summary, _reference_hops


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _planned_pid_recall(query_plan: Mapping[str, Any], references: list[Mapping[str, Any]]) -> dict[str, Any]:
    remaining = Counter(
        str(pid)
        for hop in query_plan.get("hops") or []
        for pid in hop.get("pids") or []
    )
    evaluable = [row for row in references if row.get("target", {}).get("pids")]
    hits = 0
    for row in evaluable:
        matched = next(
            (str(pid) for pid in row["target"]["pids"] if remaining[str(pid)] > 0),
            None,
        )
        if matched:
            remaining[matched] -= 1
            hits += 1
    return {
        "evaluable_reference_hops": len(evaluable),
        "hit_reference_hops": hits,
        "recall": hits / len(evaluable) if evaluable else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime_details", required=True)
    parser.add_argument("--runtime_report", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    args = parser.parse_args()

    details_path = Path(args.runtime_details).resolve()
    runtime_report_path = Path(args.runtime_report).resolve()
    protocol_path = Path(args.protocol).resolve()
    dataset_path = Path(args.dataset).resolve()
    details = list(_read_jsonl(details_path))
    runtime_report = json.loads(runtime_report_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if runtime_report.get("gold_access") is not False:
        raise SystemExit("runtime artifact does not assert gold_access=false")
    if int(runtime_report.get("counts", {}).get("n") or 0) != len(details):
        raise SystemExit("runtime report/detail count mismatch")

    selected = {str(row["qid"]) for row in details}
    source_by_qid = {
        str(row["id"]): row
        for row in _read_jsonl(dataset_path)
        if str(row.get("id")) in selected
    }
    if set(source_by_qid) != selected:
        raise SystemExit("runtime qids do not exactly match dataset source qids")

    audited: list[dict[str, Any]] = []
    plan_hops = plan_hits = 0
    for row in details:
        references = _reference_hops(str(row["dataset"]), source_by_qid[str(row["qid"])])
        chain = _chain_summary(references, row.get("kg_subgraph") or [])
        plan = _planned_pid_recall(row.get("query_plan") or {}, references)
        plan_hops += plan["evaluable_reference_hops"]
        plan_hits += plan["hit_reference_hops"]
        audited.append(
            {
                "row_id": row.get("row_id"),
                "dataset": row["dataset"],
                "qid": row["qid"],
                "question_sha256": row["question_sha256"],
                "reference_hops": references,
                "planned_pid_audit": plan,
                "proof_chain_audit": chain,
            }
        )

    evaluable = [row for row in audited if row["proof_chain_audit"]["evaluable"]]
    full = sum(
        row["proof_chain_audit"]["all_relation_value_hit"] is True for row in evaluable
    )
    chain_rate = full / len(evaluable) if evaluable else 0.0
    runtime_counts = runtime_report["counts"]
    n = len(details)
    values = {
        "planner_schema_valid_rate": runtime_counts["planner_schema_valid"] / n,
        "anchor_qid_resolved_rate": runtime_counts["anchor_qid_resolved"] / n,
        "proof_kg_nonempty_rate": runtime_counts["proof_kg_nonempty"] / n,
        "complete_plan_execution_rate": runtime_counts["complete_plan_execution"] / n,
        "full_relation_value_chain_rate_evaluable": chain_rate,
        "runtime_exception_count": runtime_counts["runtime_errors"],
    }
    thresholds = protocol["engineering_gates"]
    checks = {
        "planner_schema_valid_rate": values["planner_schema_valid_rate"] >= thresholds["planner_schema_valid_rate_min"],
        "anchor_qid_resolved_rate": values["anchor_qid_resolved_rate"] >= thresholds["anchor_qid_resolved_rate_min"],
        "proof_kg_nonempty_rate": values["proof_kg_nonempty_rate"] >= thresholds["proof_kg_nonempty_rate_min"],
        "complete_plan_execution_rate": values["complete_plan_execution_rate"] >= thresholds["complete_plan_execution_rate_min"],
        "full_relation_value_chain_rate_evaluable": values["full_relation_value_chain_rate_evaluable"] >= thresholds["full_relation_value_chain_rate_evaluable_min"],
        "runtime_exception_count": values["runtime_exception_count"] <= thresholds["runtime_exception_count_max"],
    }
    output_dir, experiment_id = prepare_new_run_dir(
        args.output_dir,
        experiment_id=args.experiment_id,
        extra={
            "phase": "postfreeze_gold_audit_automatic_proofkg",
            "runtime_details": artifact_identity(details_path),
            "runtime_report": artifact_identity(runtime_report_path),
            "protocol": artifact_identity(protocol_path),
            "dataset": artifact_identity(dataset_path),
        },
    )
    audit_path = output_dir / "postbuild_audit_details.jsonl"
    with audit_path.open("x", encoding="utf-8") as handle:
        for row in audited:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "status": "PASS_ENGINEERING_GATES" if all(checks.values()) else "FAIL_STOP_ENGINEERING_GATES",
        "scope": "post-freeze Gold audit; zero training; no model inference",
        "counts": {
            "n": n,
            "proof_evaluable": len(evaluable),
            "full_relation_value_chain": full,
            "reference_hops": plan_hops,
            "planned_pid_hits": plan_hits,
        },
        "rates": {
            **values,
            "planned_pid_recall": plan_hits / plan_hops if plan_hops else None,
        },
        "gates": {"thresholds": thresholds, "values": values, "checks": checks, "all_pass": all(checks.values())},
        "inputs": {
            "runtime_details": artifact_identity(details_path),
            "runtime_report": artifact_identity(runtime_report_path),
            "protocol": artifact_identity(protocol_path),
            "dataset_postfreeze_only": artifact_identity(dataset_path),
        },
        "outputs": {"postbuild_audit_details": artifact_identity(audit_path)},
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(output_dir, status=report["status"], extra=report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
