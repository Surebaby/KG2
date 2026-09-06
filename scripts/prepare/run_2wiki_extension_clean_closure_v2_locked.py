#!/usr/bin/env python3
"""Execute the frozen 2Wiki n300 clean closure-v2 lock, then attest gates.

The launcher fails before creating the run directory if any locked input or
code file drifted.  The underlying closure remains append-only.  A completed
run is assessed in a separate append-only attestation directory; structural
failure is retained rather than deleted or rewritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_2wiki_extension_clean_closure_v2 import (
    DATASET,
    ROOT,
    SCHEMA_VERSION,
    STATUS,
    build_closure_command,
    file_lock,
    index_plan_rows,
    read_json,
    read_jsonl,
    validate_clean_store,
    validate_historical_cache,
    validate_resolver_outputs,
)


def _assert_same_lock(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    if dict(actual) != dict(expected):
        raise ValueError(f"locked input drift: {label}")


def validate_lock(protocol_path: Path) -> tuple[dict[str, Any], list[str]]:
    protocol = read_json(protocol_path)
    if protocol.get("schema_version") != SCHEMA_VERSION or protocol.get("status") != STATUS:
        raise ValueError("not a frozen clean closure-v2 lock")
    checks = protocol.get("checks") or {}
    if not checks or not all(value is True for value in checks.values()):
        raise ValueError("frozen preflight checks are not all true")
    policy = protocol.get("closure_policy") or {}
    if (
        policy.get("dataset") != DATASET
        or policy.get("exact_entity_cache_only") is not True
        or policy.get("store_first_historical_fallback") is not True
        or policy.get("overwrite") is not False
    ):
        raise ValueError("unsafe closure policy")
    boundary = protocol.get("scientific_boundary") or {}
    if (
        boundary.get("v5_not_complete_ledger_attested") is not True
        or boundary.get("diagnostic_candidate_build_only") is not True
        or boundary.get("final_training_eligibility") is not False
    ):
        raise ValueError("diagnostic-v5 scientific boundary is missing")

    inputs = protocol["inputs"]
    plans_path = Path(inputs["plans"]["path"])
    planner_protocol_path = Path(inputs["planner_protocol"]["path"])
    resolver_protocol_path = Path(inputs["resolver_protocol"]["path"])
    resolver_report_path = Path(inputs["root_resolution"]["report"]["path"])
    resolver_dir = resolver_report_path.parent
    clean_store_dir = Path(inputs["clean_v5_store"]["path"])
    historical_path = Path(inputs["historical_property_cache"]["path"])
    closure_v1_report_path = Path(inputs["closure_v1_report"]["path"])

    plans = index_plan_rows(read_jsonl(plans_path))
    _assert_same_lock(file_lock(plans_path), inputs["plans"], "plans")
    _assert_same_lock(file_lock(planner_protocol_path), inputs["planner_protocol"], "planner protocol")
    _assert_same_lock(file_lock(resolver_protocol_path), inputs["resolver_protocol"], "resolver protocol")

    _, resolver_locks = validate_resolver_outputs(
        resolver_dir=resolver_dir,
        resolver_protocol_path=resolver_protocol_path,
    )
    _assert_same_lock(resolver_locks, inputs["root_resolution"], "root resolver outputs")
    store_locks = validate_clean_store(clean_store_dir, plans)
    expected_store = dict(inputs["clean_v5_store"])
    expected_store.pop("path", None)
    _assert_same_lock(store_locks, expected_store, "clean v5 store")
    _, historical_locks = validate_historical_cache(historical_path, closure_v1_report_path)
    _assert_same_lock(historical_locks["cache"], inputs["historical_property_cache"], "historical cache")
    _assert_same_lock(historical_locks["closure_v1_report"], inputs["closure_v1_report"], "closure-v1 report")

    absent_index = Path(inputs["no_local_entity_index"]["path"])
    if inputs["no_local_entity_index"].get("must_be_absent") is not True or absent_index.exists():
        raise ValueError("local entity index is present; clean v2 requires exact-cache-only linking")
    for label, expected in inputs["code"].items():
        _assert_same_lock(file_lock(Path(expected["path"])), expected, f"code:{label}")

    run_dir = Path(protocol["outputs"]["run_dir"])
    attestation_dir = Path(protocol["outputs"]["attestation_dir"])
    if run_dir.exists() or attestation_dir.exists():
        raise FileExistsError("refusing to overwrite clean closure-v2 run/attestation")
    expected_command = build_closure_command(protocol)
    if protocol.get("closure_command") != expected_command:
        raise ValueError("stored closure command differs from locked inputs/policy")
    return protocol, expected_command


def postflight(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    run_dir = Path(protocol["outputs"]["run_dir"])
    closure_report_path = run_dir / "closure_report.json"
    closure = read_json(closure_report_path)
    if (
        closure.get("schema_version") != "inference-proofkg-closure-v3b-1"
        or closure.get("experiment_id") != protocol.get("experiment_id")
        or closure.get("dataset") != DATASET
        or closure.get("exact_entity_cache_only") is not True
    ):
        raise ValueError("closure report identity/policy mismatch")
    last_round = int(closure.get("last_materialized_round", -1))
    if last_round < 0:
        raise ValueError("invalid last materialized round")
    runtime_dir = run_dir / f"round_{last_round}" / "runtime"
    runtime_report = read_json(runtime_dir / "report.json")
    runtime_rows = read_jsonl(runtime_dir / "runtime_details.jsonl")
    plan_rows = index_plan_rows(read_jsonl(Path(protocol["inputs"]["plans"]["path"])))
    runtime_index: dict[str, Mapping[str, Any]] = {}
    max_triples = 0
    for row in runtime_rows:
        dataset = str(row.get("dataset") or "")
        qid = str(row.get("qid") or "")
        question = str(row.get("question") or "").strip()
        key = question_key(dataset, qid)
        if (
            dataset != DATASET
            or key not in plan_rows
            or question_sha256(question) != str(row.get("question_sha256") or "")
            or question != str(plan_rows[key].get("question") or "").strip()
            or key in runtime_index
        ):
            raise ValueError(f"runtime identity drift: {key}")
        runtime_index[key] = row
        max_triples = max(max_triples, len(row.get("kg_subgraph") or []))
    identity_rate = len(set(plan_rows) & set(runtime_index)) / len(plan_rows)
    counts = runtime_report.get("counts") or {}
    n = len(runtime_rows)
    runtime_errors = sum(bool(row.get("runtime_error")) for row in runtime_rows)
    gold_false = all((row.get("provenance") or {}).get("gold_access") is False for row in runtime_rows)
    rates = {
        "plan_recognized": int(counts.get("plan_recognized", -1)) / n if n else 0.0,
        "anchor_qid_resolved": int(counts.get("anchor_qid_resolved", -1)) / n if n else 0.0,
        "proof_kg_nonempty": int(counts.get("proof_kg_nonempty", -1)) / n if n else 0.0,
        "complete_plan_execution": int(counts.get("complete_plan_execution", -1)) / n if n else 0.0,
    }
    gates = {
        "identity_join_rate_eq_1": identity_rate == 1.0 and set(plan_rows) == set(runtime_index),
        "n_eq_300": n == 300 and int(counts.get("n", -1)) == 300,
        "runtime_errors_zero": runtime_errors == 0 and int(counts.get("runtime_errors", -1)) == 0,
        "gold_access_false": gold_false,
        "plan_recognized_rate_ge_0_80": rates["plan_recognized"] >= 0.80,
        "anchor_qid_resolved_rate_ge_0_80": rates["anchor_qid_resolved"] >= 0.80,
        "proof_kg_nonempty_rate_ge_0_80": rates["proof_kg_nonempty"] >= 0.80,
        "complete_plan_execution_rate_ge_0_70": rates["complete_plan_execution"] >= 0.70,
        "max_triples_le_12": max_triples <= 12,
        "closure_converged": closure.get("stop_reason") == "no_new_requests",
    }
    passed = all(gates.values())
    report = {
        "schema_version": "2wiki-extension-clean-closure-v2-result-1",
        "experiment_id": protocol["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "PASS_DIAGNOSTIC_STRUCTURE_NOT_TRAINING_ELIGIBLE"
            if passed
            else "FAIL_DIAGNOSTIC_STRUCTURE_RETAINED"
        ),
        "counts": counts,
        "rates": rates,
        "identity_join_rate": identity_rate,
        "max_triples_per_question": max_triples,
        "gates": gates,
        "all_pass": passed,
        "decision": (
            "CANDIDATE_FOR_V6_LEDGER_ATTESTED_REEXECUTION"
            if passed
            else "STOP_DO_NOT_USE_IN_PROOF800"
        ),
        "v5_not_complete_ledger_attested": True,
        "final_training_eligibility": False,
        "inputs": {
            "lock_protocol": file_lock(Path(protocol["self_path"])),
            "closure_report": file_lock(closure_report_path),
            "runtime_report": file_lock(runtime_dir / "report.json"),
            "runtime_details": file_lock(runtime_dir / "runtime_details.jsonl"),
        },
        "training_started": False,
    }
    return report, passed


def run_locked(protocol_path: Path) -> dict[str, Any]:
    protocol, command = validate_lock(protocol_path)
    # ``self_path`` is runtime-only metadata used by postflight; it cannot
    # affect the frozen subprocess command.
    protocol["self_path"] = str(protocol_path.resolve())
    completed = subprocess.run(command, cwd=str(ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"clean closure-v2 command failed with exit {completed.returncode}")
    report, passed = postflight(protocol)
    attestation_dir = Path(protocol["outputs"]["attestation_dir"])
    attestation_dir.mkdir(parents=True, exist_ok=False)
    report_path = attestation_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        attestation_dir,
        status=report["status"],
        extra={
            "phase": "attest_2wiki_extension_clean_closure_v2",
            "experiment_id": protocol["experiment_id"],
            "report": file_lock(report_path),
            "training_started": False,
        },
    )
    if not passed:
        raise RuntimeError("clean closure-v2 structural gates failed; result retained")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    report = run_locked(args.protocol)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
