#!/usr/bin/env python3
"""Run the frozen official-raw n=1500 clean closure and attest it.

The launcher validates every SHA256-bound input before network access.  The
postflight then checks that root resolution seen by the final executor is
identical to the preregistered projection and root-resolver dry-run.  It emits
Gold-free strict-eligibility telemetry but deliberately does not choose the
final Proof800 or read passages/answers.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.kg.wikipedia_title_resolver import complete_question_surface_title
from kgproweight.reward.trajectory_source_gate import evaluate_graph_gate
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_2wiki_official_raw_n1500_clean_closure_v1 import (
    CUTOFF,
    DATASET,
    EXPECTED_N,
    EXPECTED_QTYPE_COUNTS,
    LOCK_SCHEMA,
    LOCK_STATUS,
    ROOT,
    build_closure_command,
    file_identity,
    index_question_rows,
    read_json,
    read_jsonl,
    validate_candidate_and_plans,
    validate_policy,
    validate_root_resolution,
    validate_v6_store,
    write_jsonl,
)


REPORT_SCHEMA = "2wiki-official-raw-clean-closure-v3"
PASS_STATUS = "COMPLETE_DIAGNOSTIC_CLEAN_CLOSURE_NOT_SELECTED_NOT_TRAINED"
FAIL_STATUS = "FAIL_DIAGNOSTIC_CLEAN_CLOSURE_RETAINED_NOT_SELECTED_NOT_TRAINED"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_lock(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    if dict(actual) != dict(expected):
        raise ValueError(f"locked input drift: {label}")


def validate_execution_lock(lock_path: Path) -> tuple[dict[str, Any], list[str]]:
    lock = read_json(lock_path)
    if lock.get("schema_version") != LOCK_SCHEMA or lock.get("status") != LOCK_STATUS:
        raise ValueError("not a frozen n1500 clean-closure execution lock")
    checks = lock.get("checks") or {}
    if not checks or not all(value is True for value in checks.values()):
        raise ValueError("execution-lock checks are not all true")
    if lock.get("gold_access") is not False or lock.get("training_started") is not False:
        raise ValueError("unsafe execution-lock scientific boundary")
    policy_path = Path((lock.get("policy_protocol") or {}).get("path") or "")
    _assert_lock(file_identity(policy_path), lock["policy_protocol"], "method policy")
    policy = validate_policy(policy_path)
    if policy.get("experiment_id") != lock.get("experiment_id"):
        raise ValueError("policy/execution experiment identity mismatch")

    inputs = lock["inputs"]
    validate_candidate_and_plans(
        cohort_path=Path(inputs["candidate_cohort"]["path"]),
        candidate_protocol_path=Path(inputs["candidate_protocol"]["path"]),
        plans_path=Path(inputs["plans"]["path"]),
        planner_protocol_path=Path(inputs["planner_protocol"]["path"]),
        planner_postflight_path=Path(inputs["planner_postflight"]["path"]),
    )
    for name in (
        "candidate_cohort",
        "candidate_protocol",
        "plans",
        "planner_protocol",
        "planner_postflight",
        "historical_seed_cache",
    ):
        expected = inputs[name]
        _assert_lock(file_identity(Path(expected["path"])), expected, name)
    for name, expected in inputs["protected_ledger"].items():
        _assert_lock(file_identity(Path(expected["path"])), expected, f"ledger:{name}")
    store = validate_v6_store(
        Path(inputs["v6_store"]["path"]), Path(inputs["candidate_cohort"]["path"])
    )
    _assert_lock(store, inputs["v6_store"], "v6 store")
    root_protocol_path = Path(inputs["root_resolution"]["protocol"]["path"])
    root_dir = Path(inputs["root_resolution"]["report"]["path"]).parent
    root_report, root_locks, _ = validate_root_resolution(
        root_protocol_path=root_protocol_path,
        root_dir=root_dir,
        v6_store=inputs["v6_store"],
    )
    _assert_lock(root_locks, inputs["root_resolution"], "root resolution")
    if lock.get("root_gate_snapshot") != {
        "counts": root_report["counts"],
        "rates": root_report["rates"],
        "gates": root_report["gates"],
    }:
        raise ValueError("root gate snapshot drift")
    absent = Path(inputs["no_local_entity_index"]["path"])
    if inputs["no_local_entity_index"].get("must_be_absent") is not True or absent.exists():
        raise ValueError("local entity index is present")
    for name, expected in inputs["code"].items():
        _assert_lock(file_identity(Path(expected["path"])), expected, f"code:{name}")
    policy_values = lock.get("closure_policy") or {}
    if (
        policy_values.get("dataset") != DATASET
        or policy_values.get("max_rounds") != 4
        or policy_values.get("cutoff") != CUTOFF
        or policy_values.get("exact_entity_cache_only") is not True
        or policy_values.get("store_first_historical_fallback") is not True
        or policy_values.get("overwrite") is not False
    ):
        raise ValueError("unsafe or drifted closure policy")
    if Path(inputs["historical_seed_cache"]["path"]).stat().st_size != 0:
        raise ValueError("fresh historical seed cache is no longer empty")
    for path in (Path(lock["outputs"]["run_dir"]), Path(lock["outputs"]["result_dir"])):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite closure artifact: {path}")
    command = build_closure_command(lock)
    if command != lock.get("closure_command"):
        raise ValueError("stored closure command differs from locked command")
    return lock, command


def _dry_root_index(
    lock: Mapping[str, Any]
) -> dict[tuple[str, str], dict[str, Any]]:
    path = Path(lock["inputs"]["root_resolution"]["consumer_dry_run"]["path"])
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = str(row.get("question_key") or "")
        surface = str(row.get("root_anchor_surface") or "").strip()
        if not key or not surface or (key, surface) in output:
            raise ValueError("invalid/duplicate root dry-run identity")
        output[(key, surface)] = row
    return output


def _resolved_qid(value: Mapping[str, Any]) -> str | None:
    qid = str(value.get("qid") or "").strip()
    return qid if qid and not bool(value.get("abstained")) else None


def _dry_qid(value: Mapping[str, Any]) -> str | None:
    qid = str(value.get("dry_run_qid") or value.get("resolved_qid") or "").strip()
    return qid or None


def compare_runtime_roots_to_dry_run(
    *,
    question_key_value: str,
    anchors: list[str],
    anchor_entities: Mapping[str, Mapping[str, Any]],
    dry_roots: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare one executor root state to the frozen exact-consumer dry-run."""

    if set(anchors) != set(anchor_entities):
        raise ValueError(f"runtime root diagnostics drift: {question_key_value}")
    seen: set[tuple[str, str]] = set()
    resolved = matches = mismatches = 0
    for surface in anchors:
        identity = (question_key_value, str(surface))
        expected = dry_roots.get(identity)
        if expected is None:
            raise ValueError(f"runtime root missing from frozen dry-run: {identity}")
        seen.add(identity)
        runtime_qid = _resolved_qid(anchor_entities[str(surface)])
        expected_qid = _dry_qid(expected)
        resolved += int(runtime_qid is not None)
        matches += int(runtime_qid == expected_qid)
        mismatches += int(runtime_qid != expected_qid)
    return {
        "seen": seen,
        "total": len(anchors),
        "resolved": resolved,
        "all_resolved": bool(anchors) and resolved == len(anchors),
        "matches": matches,
        "mismatches": mismatches,
    }


def postflight(lock: Mapping[str, Any], lock_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    run_dir = Path(lock["outputs"]["run_dir"])
    closure_report_path = run_dir / "closure_report.json"
    closure = read_json(closure_report_path)
    if (
        closure.get("schema_version") != "inference-proofkg-closure-v3b-1"
        or closure.get("experiment_id") != lock.get("experiment_id")
        or closure.get("dataset") != DATASET
        or closure.get("exact_entity_cache_only") is not True
        or int(closure.get("max_rounds", -1)) != 4
        or closure.get("cutoff") != CUTOFF
    ):
        raise ValueError("closure report identity/policy mismatch")
    last_round = int(closure.get("last_materialized_round", -1))
    if last_round < 0 or last_round > 4:
        raise ValueError("invalid final closure round")
    runtime_dir = run_dir / f"round_{last_round}" / "runtime"
    runtime_report_path = runtime_dir / "report.json"
    runtime_path = runtime_dir / "runtime_details.jsonl"
    runtime_report = read_json(runtime_report_path)
    runtime_rows = read_jsonl(runtime_path)
    cohort = index_question_rows(
        read_jsonl(Path(lock["inputs"]["candidate_cohort"]["path"])),
        label="candidate",
        require_qtype=True,
    )
    plans = index_question_rows(
        read_jsonl(Path(lock["inputs"]["plans"]["path"])),
        label="plans",
        require_qtype=False,
    )
    runtime: dict[str, dict[str, Any]] = {}
    for row in runtime_rows:
        dataset = str(row.get("dataset") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        question = str(row.get("question") or "").strip()
        key = question_key(dataset, qid)
        if (
            dataset != DATASET
            or key not in plans
            or key in runtime
            or str(row.get("question_key") or "") != key
            or question != str(plans[key]["question"])
            or str(row.get("question_sha256") or "") != question_sha256(question)
        ):
            raise ValueError(f"runtime identity drift: {key}")
        runtime[key] = row
    identity_join = set(runtime) == set(plans) == set(cohort)
    dry_roots = _dry_root_index(lock)
    seen_dry: set[tuple[str, str]] = set()
    runtime_root_occurrences = 0
    runtime_resolved_occurrences = 0
    runtime_all_roots_questions = 0
    runtime_dry_match = 0
    runtime_dry_mismatch = 0
    telemetry: list[dict[str, Any]] = []
    max_triples = 0
    for key in sorted(runtime):
        row = runtime[key]
        candidate = cohort[key]
        plan = row.get("query_plan") or {}
        execution = row.get("execution") or {}
        anchors = list(plan.get("anchors") or [])
        entities = execution.get("anchor_entities") or {}
        comparison = compare_runtime_roots_to_dry_run(
            question_key_value=key,
            anchors=[str(value) for value in anchors],
            anchor_entities=entities,
            dry_roots=dry_roots,
        )
        seen_dry.update(comparison["seen"])
        runtime_root_occurrences += int(comparison["total"])
        runtime_resolved_occurrences += int(comparison["resolved"])
        runtime_dry_match += int(comparison["matches"])
        runtime_dry_mismatch += int(comparison["mismatches"])
        resolved_here = int(comparison["resolved"])
        all_roots = bool(comparison["all_resolved"])
        runtime_all_roots_questions += int(all_roots)
        triples = list(row.get("kg_subgraph") or [])
        max_triples = max(max_triples, len(triples))
        decision = evaluate_graph_gate(
            row,
            dataset=DATASET,
            qid=str(row["qid"]),
            question=str(row["question"]),
            historical_cutoff=CUTOFF,
        )
        planned_hops = list(plan.get("hops") or [])
        executed_hops = list(execution.get("hops") or [])
        provenance = row.get("provenance") or {}
        telemetry.append(
            {
                "schema_version": "2wiki-official-raw-strict-eligibility-telemetry-v1",
                "question_key": key,
                "dataset": DATASET,
                "qid": str(row["qid"]),
                "question_sha256": str(row["question_sha256"]),
                "family_sha256": str(candidate["family_sha256"]),
                "question_type": str(candidate["question_type"]),
                "planner_schema_valid": bool(row.get("planner_schema_valid")),
                "root_anchors_total": len(anchors),
                "root_anchors_resolved": resolved_here,
                "all_root_anchors_resolved": all_roots,
                "planned_hops": len(planned_hops),
                "executed_hops": len(executed_hops),
                "all_hops_complete": bool(execution.get("complete_plan_execution")),
                "graph_nonempty": bool(triples),
                "gold_access_false": provenance.get("gold_access") is False,
                "runtime_error_zero": row.get("runtime_error") in (None, ""),
                "provenance_complete": bool(provenance.get("builder_version"))
                and provenance.get("gold_access") is False
                and bool(provenance.get("planner_predictions_sha256")),
                "retained_edges_traceable": bool(
                    decision.checks.get("retained_edges_traceable")
                ),
                "no_duplicate_edges": bool(decision.checks.get("no_duplicate_edges")),
                "m_graph": decision.m_graph,
                "routing_reason": decision.routing_reason,
                "eligibility_checks": decision.checks,
                "kg_sha256": decision.kg_sha256,
                "execution_sha256": decision.execution_sha256,
                "runtime_record_sha256": canonical_sha256(row),
            }
        )
    if seen_dry != set(dry_roots):
        raise ValueError("final runtime did not consume every frozen root dry-run occurrence")
    telemetry.sort(key=lambda row: (str(row["question_type"]), str(row["qid"])))
    counts = runtime_report.get("counts") or {}
    n = len(runtime)
    strict_by_type = Counter(
        str(row["question_type"]) for row in telemetry if int(row["m_graph"]) == 1
    )
    strict_total = sum(strict_by_type.values())
    runtime_errors = sum(not bool(row["runtime_error_zero"]) for row in telemetry)
    gold_false = all(bool(row["gold_access_false"]) for row in telemetry)
    rates = {
        "planner_schema_valid": sum(bool(row["planner_schema_valid"]) for row in telemetry) / n,
        "plan_recognized": int(counts.get("plan_recognized", -1)) / n,
        "anchor_qid_resolved": runtime_all_roots_questions / n,
        "root_anchor_occurrence_resolved": (
            runtime_resolved_occurrences / runtime_root_occurrences
            if runtime_root_occurrences
            else 0.0
        ),
        "proof_kg_nonempty": int(counts.get("proof_kg_nonempty", -1)) / n,
        "complete_plan_execution": int(counts.get("complete_plan_execution", -1)) / n,
        "strict_graph_eligible": strict_total / n,
        "root_dry_run_runtime_match": (
            runtime_dry_match / runtime_root_occurrences if runtime_root_occurrences else 0.0
        ),
    }
    cache_policy = runtime_report.get("cache_policy") or {}
    runtime_store_ref = (runtime_report.get("inputs") or {}).get("versioned_store_manifest") or {}
    v6_manifest = Path(lock["inputs"]["v6_store"]["store_manifest"]["path"])
    gates = {
        "identity_join_rate_eq_1": identity_join and n == EXPECTED_N,
        "runtime_report_n_eq_1500": int(counts.get("n", -1)) == EXPECTED_N,
        "runtime_errors_zero": runtime_errors == 0 and int(counts.get("runtime_errors", -1)) == 0,
        "gold_access_false": gold_false,
        "planner_schema_valid_rate_ge_0_97": rates["planner_schema_valid"] >= 0.97,
        "plan_recognized_rate_ge_0_97": rates["plan_recognized"] >= 0.97,
        "anchor_qid_resolved_rate_ge_0_80": rates["anchor_qid_resolved"] >= 0.80,
        "proof_kg_nonempty_rate_ge_0_80": rates["proof_kg_nonempty"] >= 0.80,
        "complete_plan_execution_rate_ge_0_70": rates["complete_plan_execution"] >= 0.70,
        "strict_graph_eligible_each_qtype_ge_200": all(
            strict_by_type[qtype] >= 200 for qtype in EXPECTED_QTYPE_COUNTS
        ),
        "max_triples_per_question_le_12": max_triples <= 12,
        "root_projection_dry_run_runtime_join_eq_1": (
            runtime_dry_mismatch == 0
            and runtime_dry_match == runtime_root_occurrences == len(dry_roots)
            and rates["root_dry_run_runtime_match"] == 1.0
        ),
        "closure_converged": closure.get("stop_reason") == "no_new_requests",
        "closure_policy_exact": (
            closure.get("policy")
            == {"workers": 2, "delay": 0.4, "timeout": 12.0, "retries": 3}
        ),
        "store_first_v6_historical_fallback_exact": (
            cache_policy.get("supply_backend") == "store_first_combined_retriever"
            and cache_policy.get("exact_entity_cache_only") is True
            and cache_policy.get("historical_cutoff") == CUTOFF
            and str(runtime_store_ref.get("md5") or "") == md5_file(v6_manifest)
        ),
        "question_type_population_exact": Counter(
            str(row["question_type"]) for row in telemetry
        )
        == Counter(EXPECTED_QTYPE_COUNTS),
    }
    passed = all(gates.values())
    report = {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": lock["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "counts": {
            "n": n,
            "runtime_report": counts,
            "root_anchor_occurrences": runtime_root_occurrences,
            "root_anchor_occurrences_resolved": runtime_resolved_occurrences,
            "all_roots_resolved_questions": runtime_all_roots_questions,
            "root_dry_run_runtime_matches": runtime_dry_match,
            "root_dry_run_runtime_mismatches": runtime_dry_mismatch,
            "strict_graph_eligible": strict_total,
            "strict_graph_eligible_by_question_type": {
                qtype: strict_by_type[qtype] for qtype in EXPECTED_QTYPE_COUNTS
            },
        },
        "rates": rates,
        "max_triples_per_question": max_triples,
        "gates": gates,
        "all_pass": passed,
        "decision": (
            "CONTINUE_TO_PROOF800_SELECTION"
            if passed
            else "STOP_RETAIN_RESULT_DO_NOT_SELECT_PROOF800"
        ),
        "inputs": {
            "execution_lock": file_identity(lock_path),
            "method_policy": lock["policy_protocol"],
            "root_resolution_report": lock["inputs"]["root_resolution"]["report"],
            "root_consumer_dry_run": lock["inputs"]["root_resolution"]["consumer_dry_run"],
            "v6_store_manifest": lock["inputs"]["v6_store"]["store_manifest"],
            "closure_report": file_identity(closure_report_path),
        },
        "outputs": {
            "runtime_report": file_identity(runtime_report_path),
            "runtime_details": file_identity(runtime_path),
            # Filled after the append-only telemetry is written.
            "strict_eligibility_telemetry": None,
        },
        "scientific_boundary": {
            "structural_and_source_eligibility_only": True,
            "semantic_correctness": "UNKNOWN_NOT_EVALUATED",
            "proof800_selected": False,
            "passages_or_answers_read": False,
            "training_started": False,
        },
        "network_access": True,
        "gold_access": False,
        "training_started": False,
    }
    return report, telemetry, passed


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_locked(lock_path: Path) -> dict[str, Any]:
    lock, command = validate_execution_lock(lock_path)
    completed = subprocess.run(command, check=False, cwd=str(ROOT))
    if completed.returncode != 0:
        raise RuntimeError(f"n1500 clean closure failed with exit {completed.returncode}")
    report, telemetry, passed = postflight(lock, lock_path)
    result_dir = Path(lock["outputs"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=False)
    telemetry_path = result_dir / "strict_eligibility_telemetry.jsonl"
    write_jsonl(telemetry_path, telemetry)
    report["outputs"]["strict_eligibility_telemetry"] = {
        **file_identity(telemetry_path),
        "rows": len(telemetry),
    }
    report_path = result_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        result_dir,
        status=report["status"],
        extra={
            "phase": "attest_2wiki_official_raw_n1500_clean_closure_v1",
            "experiment_id": lock["experiment_id"],
            "report": file_identity(report_path),
            "runtime_details": report["outputs"]["runtime_details"],
            "strict_eligibility_telemetry": report["outputs"]["strict_eligibility_telemetry"],
            "proof800_selected": False,
            "gold_access": False,
            "training_started": False,
        },
    )
    if not passed:
        raise RuntimeError("n1500 closure gates failed; append-only result retained")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_locked(args.protocol), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
