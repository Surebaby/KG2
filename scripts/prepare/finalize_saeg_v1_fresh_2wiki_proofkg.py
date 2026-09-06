#!/usr/bin/env python
"""Freeze the final closure round as fresh SAEG 2Wiki ProofKG records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.utils.logging import dump_manifest


EXPERIMENT_ID = "SAEG-V1-2WIKI-DEV-CONFIRMATION-PROOFKG-V1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def structural_gates(details: list[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(details)
    values = {
        "plan_recognized": sum(bool((row.get("query_plan") or {}).get("recognized")) for row in details) / n,
        "nonempty": sum(bool(row.get("kg_subgraph")) for row in details) / n,
        "complete_execution": sum(bool((row.get("execution") or {}).get("complete_plan_execution")) for row in details) / n,
        "runtime_errors": sum(bool(row.get("runtime_error")) for row in details),
        "identity_join": len({str(row.get("question_key")) for row in details}) / n,
        "gold_access_false": all((row.get("provenance") or {}).get("gold_access") is False for row in details),
    }
    checks = {
        "plan_recognized_ge_0.80": values["plan_recognized"] >= 0.80,
        "nonempty_ge_0.80": values["nonempty"] >= 0.80,
        "complete_ge_0.70": values["complete_execution"] >= 0.70,
        "runtime_errors_eq_0": values["runtime_errors"] == 0,
        "identity_join_eq_1.0": values["identity_join"] == 1.0,
        "gold_access_false": values["gold_access_false"],
    }
    return {"values": values, "checks": checks, "all_pass": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure", type=Path, default=Path(
        "data/derived/saeg_v1_2wiki_dev_confirmation_closure_v1"))
    parser.add_argument("--cohort", type=Path, default=Path(
        "outputs/audits/saeg_v1_evaluation_protocol_v1/2wiki_dev_confirmation_planner.question_only.jsonl"))
    parser.add_argument("--out", type=Path, default=Path(
        "data/derived/saeg_v1_2wiki_dev_confirmation_proofkg_v1"))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite final ProofKG: {args.out}")
    closure_report_path = args.closure / "closure_report.json"
    closure_report = json.loads(closure_report_path.read_text(encoding="utf-8"))
    final_round = int(closure_report["last_materialized_round"])
    runtime_dir = args.closure / f"round_{final_round}" / "runtime"
    records_path = runtime_dir / "runtime_question_kg.jsonl"
    details_path = runtime_dir / "runtime_details.jsonl"
    for path in (args.cohort, records_path, details_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    cohort = read_jsonl(args.cohort)
    records = read_jsonl(records_path)
    details = read_jsonl(details_path)
    if len(cohort) != 150 or len(records) != 150 or len(details) != 150:
        raise ValueError("expected 150 cohort/record/detail rows")
    cohort_keys = {str(row["question_key"]): row for row in cohort}
    record_keys = {str(row["question_key"]): row for row in records}
    detail_keys = {str(row["question_key"]): row for row in details}
    if set(cohort_keys) != set(record_keys) or set(cohort_keys) != set(detail_keys):
        raise ValueError("cohort/runtime identity join is not 1.0")
    for key, record in record_keys.items():
        if record["question_sha256"] != cohort_keys[key]["question_sha256"]:
            raise ValueError(f"{key}: question hash mismatch")
        if (record.get("provenance") or {}).get("gold_access") is not False:
            raise ValueError(f"{key}: gold_access is not false")
    gates = structural_gates(details)
    # Failed structure remains a valid frozen negative result, but it must not
    # be silently materialized as an evaluation-eligible W branch.
    status = "PASS_STRUCTURAL_NOT_MODEL_EVALUATED" if gates["all_pass"] else "FAIL_STRUCTURAL_NOT_ELIGIBLE"
    ordered = [record_keys[str(row["question_key"])] for row in cohort]
    args.out.mkdir(parents=True, exist_ok=False)
    output_path = args.out / "question_kg_records.jsonl"
    write_jsonl(output_path, ordered)
    report = {
        "schema_version": "saeg-fresh-2wiki-proofkg-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "closure": closure_report,
        "structural_gates": gates,
        "inputs": {
            "cohort": {"path": str(args.cohort), "sha256": sha256_file(args.cohort)},
            "closure_report": {"path": str(closure_report_path), "sha256": sha256_file(closure_report_path)},
            "runtime_records": {"path": str(records_path), "sha256": sha256_file(records_path)},
            "runtime_details": {"path": str(details_path), "sha256": sha256_file(details_path)},
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "scientific_boundary": "Structural eligibility only; semantic correctness and model utility are not inferred.",
    }
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(args.out, extra=report, status=status)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
