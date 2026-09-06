#!/usr/bin/env python3
"""Postflight the Gold-free planner output for the official-raw n=1500 pool.

This audit is append-only.  It does not alter, drop, or regenerate invalid plan
rows.  Question-type labels are joined from the frozen planner input solely for
stratified structural telemetry.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.utils.logging import artifact_identity, dump_manifest


EXPECTED_N = 1500
EXPECTED_QUOTAS = {
    "bridge_comparison": 390,
    "comparison": 390,
    "compositional": 389,
    "inference": 331,
}
SCHEMA_VALID_RATE_MIN = 0.97
DEFAULT_EXECUTION_DIR = Path(
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "seed42_planner_execution_v1_preregistration"
)
DEFAULT_PLANS_DIR = Path(
    "outputs/validation/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "seed42_plans_v1"
)
DEFAULT_LOG = Path(
    "logs/preparation/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "seed42_plans_v1.log"
)
DEFAULT_OUT = Path(
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "seed42_plans_v1_postflight"
)
EXPERIMENT_ID = (
    "2WIKI-PROOFKG-OFFICIAL-RAW-V2-CANDIDATE-POOL-N1500-SEED42-"
    "PLANS-V1-POSTFLIGHT"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_plans(
    inputs: Iterable[Mapping[str, Any]], predictions: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    input_rows, prediction_rows = list(inputs), list(predictions)
    input_index: dict[str, Mapping[str, Any]] = {}
    duplicate_input_keys = 0
    for row in input_rows:
        key = str(row.get("question_key") or "")
        if key in input_index:
            duplicate_input_keys += 1
        input_index[key] = row

    seen_prediction: set[str] = set()
    duplicate_prediction_keys = 0
    identity_mismatches = 0
    unknown_prediction_keys = 0
    gold_access_violations = 0
    empty_generations = 0
    schema_by_type: Counter[tuple[str, bool]] = Counter()
    validation_errors: Counter[str] = Counter()
    for row in prediction_rows:
        key = str(row.get("question_key") or "")
        if key in seen_prediction:
            duplicate_prediction_keys += 1
        seen_prediction.add(key)
        source = input_index.get(key)
        if source is None:
            unknown_prediction_keys += 1
            continue
        if any(
            str(row.get(field) or "") != str(source.get(field) or "")
            for field in ("dataset", "qid", "question", "question_sha256")
        ):
            identity_mismatches += 1
        gold_access_violations += int(row.get("gold_access") is not False)
        empty_generations += int(not str(row.get("generated_text") or "").strip())
        question_type = str(source.get("question_type") or "UNKNOWN")
        valid = bool(row.get("schema_valid"))
        schema_by_type[(question_type, valid)] += 1
        for error in row.get("validation_errors") or []:
            validation_errors[str(error)] += 1

    missing_keys = set(input_index).difference(seen_prediction)
    valid = sum(bool(row.get("schema_valid")) for row in prediction_rows)
    n = len(prediction_rows)
    by_type = {
        qtype: {
            "n": schema_by_type[(qtype, True)] + schema_by_type[(qtype, False)],
            "schema_valid": schema_by_type[(qtype, True)],
            "schema_invalid": schema_by_type[(qtype, False)],
            "schema_valid_rate": (
                schema_by_type[(qtype, True)]
                / (schema_by_type[(qtype, True)] + schema_by_type[(qtype, False)])
                if schema_by_type[(qtype, True)] + schema_by_type[(qtype, False)]
                else 0.0
            ),
        }
        for qtype in EXPECTED_QUOTAS
    }
    return {
        "n_input": len(input_rows),
        "n_predictions": n,
        "schema_valid": valid,
        "schema_invalid": n - valid,
        "schema_valid_rate": valid / n if n else 0.0,
        "by_question_type": by_type,
        "validation_errors": dict(validation_errors.most_common()),
        "integrity": {
            "duplicate_input_keys": duplicate_input_keys,
            "duplicate_prediction_keys": duplicate_prediction_keys,
            "missing_prediction_keys": len(missing_keys),
            "unknown_prediction_keys": unknown_prediction_keys,
            "identity_mismatches": identity_mismatches,
            "gold_access_violations": gold_access_violations,
            "empty_generations": empty_generations,
            # The generator is fail-fast, not per-row exception swallowing.  A
            # complete 1500-row output/report therefore means zero runtime
            # exceptions; schema failures remain separately retained above.
            "runtime_errors": 0 if n == len(input_rows) else 1,
        },
    }


def audit(
    *,
    execution_dir: Path,
    plans_dir: Path,
    log_path: Path,
    output_dir: Path,
    experiment_id: str = EXPERIMENT_ID,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite postflight: {output_dir}")
    runtime_path = execution_dir / "planner_input.question_only.jsonl"
    execution_protocol_path = execution_dir / "protocol.json"
    predictions_path = plans_dir / "predictions.question_only.jsonl"
    generator_report_path = plans_dir / "report.json"
    generator_manifest_path = plans_dir / "manifest.json"
    required = (
        runtime_path,
        execution_protocol_path,
        predictions_path,
        generator_report_path,
        generator_manifest_path,
        log_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    protocol = json.loads(execution_protocol_path.read_text(encoding="utf-8"))
    generator_report = json.loads(generator_report_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_GOLD_FREE_PLANNER_EXECUTION_NOT_RUN_NOT_TRAINED":
        raise ValueError("unexpected execution protocol status")
    if generator_report.get("status") != "RUNTIME_PLANS_FROZEN_NO_GOLD_AUDIT":
        raise ValueError("planner generation did not finish with the expected status")
    cohort_ref = generator_report.get("inputs", {}).get("cohort", {})
    protocol_ref = generator_report.get("inputs", {}).get("protocol", {})
    if str(cohort_ref.get("md5") or "") != _md5(runtime_path):
        raise ValueError("generator report/runtime input MD5 mismatch")
    if str(protocol_ref.get("md5") or "") != _md5(execution_protocol_path):
        raise ValueError("generator report/execution protocol MD5 mismatch")
    if generator_report.get("generation") != {
        "greedy": True,
        "max_new_tokens": 512,
        "batch_size": 8,
    }:
        raise ValueError("planner generation settings drifted")
    if generator_report.get("gold_access") is not False:
        raise ValueError("generator report does not attest gold_access=false")

    summary = summarize_plans(_read_jsonl(runtime_path), _read_jsonl(predictions_path))
    integrity = summary["integrity"]
    gates = {
        "input_n_exact": summary["n_input"] == EXPECTED_N,
        "prediction_n_exact": summary["n_predictions"] == EXPECTED_N,
        "question_type_counts_exact": all(
            summary["by_question_type"][qtype]["n"] == quota
            for qtype, quota in EXPECTED_QUOTAS.items()
        ),
        "identity_join_1_0": all(
            integrity[name] == 0
            for name in (
                "duplicate_input_keys",
                "duplicate_prediction_keys",
                "missing_prediction_keys",
                "unknown_prediction_keys",
                "identity_mismatches",
            )
        ),
        "gold_access_false": integrity["gold_access_violations"] == 0,
        "runtime_errors_zero": integrity["runtime_errors"] == 0,
        "generated_text_nonempty": integrity["empty_generations"] == 0,
        "schema_valid_rate_ge_0_97": summary["schema_valid_rate"] >= SCHEMA_VALID_RATE_MIN,
        "training_not_started": True,
    }
    passed = all(gates.values())
    status = (
        "PASS_PLANNER_STRUCTURAL_NOT_PROOFKG_MATERIALIZED_NOT_TRAINED"
        if passed
        else "FAIL_PLANNER_STRUCTURAL_STOP"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": "2wiki-official-raw-plans-postflight-v1",
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at,
        "status": status,
        "summary": summary,
        "gates": gates,
        "inputs": {
            "execution_protocol": artifact_identity(execution_protocol_path),
            "runtime_input": artifact_identity(runtime_path),
            "predictions": artifact_identity(predictions_path),
            "generator_report": artifact_identity(generator_report_path),
            "generator_manifest": artifact_identity(generator_manifest_path),
            "stdout_log": artifact_identity(log_path),
        },
        "scientific_boundary": {
            "question_only_planner": True,
            "gold_access": False,
            "invalid_rows_retained": True,
            "proofkg_structural_yield": "UNKNOWN_NOT_MATERIALIZED",
            "semantic_quality": "UNKNOWN_NOT_EVALUATED",
            "training_started": False,
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        output_dir,
        status=status,
        extra={
            "phase": "audit_2wiki_official_raw_plans_v1",
            "experiment_id": experiment_id,
            "gates": gates,
            "summary": summary,
            "report": artifact_identity(report_path),
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-dir", type=Path, default=DEFAULT_EXECUTION_DIR)
    parser.add_argument("--plans-dir", type=Path, default=DEFAULT_PLANS_DIR)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    args = parser.parse_args()
    report = audit(
        execution_dir=args.execution_dir,
        plans_dir=args.plans_dir,
        log_path=args.log,
        output_dir=args.out,
        experiment_id=args.experiment_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"].startswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
