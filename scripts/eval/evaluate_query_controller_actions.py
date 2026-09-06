#!/usr/bin/env python
"""CPU-only schema/mechanism scorer for Query Controller v1 actions.

This evaluator never loads a QA Gold field and never computes EM/F1/IHR.  It
can audit a materialized action release, or join frozen greedy response text to
reference states and measure exact-JSON/schema/query/dependency/state-use
properties.  Model generation is intentionally owned by a later runner.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgproweight.eval.query_controller_v1 import (  # noqa: E402
    audit_action_record,
    evaluate_action_records,
    parse_target_response,
)
from kgproweight.training.query_controller import _canonical_target  # noqa: E402
from kgproweight.utils.logging import artifact_identity  # noqa: E402
from kgproweight.eval.query_controller_runner import (  # noqa: E402
    EVAL_GENERATION_EXPERIMENT_ID,
    EVAL_GENERATION_OUTPUT_DIR,
    evaluate_teacher_forced_mechanism_gate,
    _verify_dev_release_and_pairs,
    _verify_eval_successor_bundle,
    _verify_protocol_bundle,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"row must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _verify_generation_manifest(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    predictions_path: Path,
    expected_predictions_sha256: str,
    protocol_bundle: Mapping[str, Any],
    eval_protocol: Mapping[str, Any],
    eval_protocol_bundle: Mapping[str, Any],
    release_lock: Mapping[str, Any],
) -> dict[str, Any]:
    expected_run_dir = (PROJECT_ROOT / EVAL_GENERATION_OUTPUT_DIR).resolve()
    if manifest_path.parent.resolve() != expected_run_dir:
        raise ValueError("generation manifest is outside the eval-e1 authorized output_dir")
    if _sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("generation manifest differs from external expected SHA256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("generation manifest must be a JSON object")
    run = manifest.get("run")
    if not isinstance(run, dict):
        raise ValueError("generation manifest lacks run provenance")
    report = run.get("report") or {}
    preflight = run.get("evaluation_preflight") or {}
    parent_lineage = eval_protocol.get("parent_training_lineage") or {}
    parent_release = parent_lineage.get("release") or {}
    parent_probe = parent_lineage.get("probe") or {}
    preflight_parent = preflight.get("parent_asset_lineage") or {}
    if (
        manifest.get("status") != "COMPLETE_GENERATION_NOT_MECHANISM_PASS"
        or run.get("phase") != "query_controller_greedy_mechanism_eval"
        or run.get("experiment_id") != EVAL_GENERATION_EXPERIMENT_ID
        or run.get("scope") != "no_retrieval_no_qa_gold_no_em"
        or report.get("generation_status")
        != "COMPLETE_GENERATION_NOT_MECHANISM_PASS"
        or (report.get("mechanism_gate") or {}).get("status")
        != "NOT_EVALUATED_REQUIRES_SEPARATE_SCORER"
        or preflight.get("status") != "PASS"
        or preflight.get("exact_actions") != 240
        or preflight.get("cohort_role") != "dev"
        or (preflight.get("protocol") or {}).get("protocol_sha256")
        != protocol_bundle.get("protocol_sha256")
        or (preflight.get("eval_protocol") or {}).get("eval_protocol_sha256")
        != eval_protocol_bundle.get("eval_protocol_sha256")
        or (preflight.get("release") or {}).get("input_sha256")
        != release_lock.get("input_sha256")
        or (preflight.get("release") or {}).get("action_pair_hash_match_rate") != 1.0
        or (preflight.get("probe") or {}).get("asset_lock_lineage_match") is not True
        or (preflight.get("parent_asset_lineage") or {}).get("status") != "PASS"
        or (preflight.get("parent_asset_lineage") or {}).get("checkpoint_reused")
        is not True
        or (preflight.get("parent_asset_lineage") or {}).get("retraining") is not False
        or preflight_parent.get("training_manifest_sha256")
        != parent_probe.get("manifest_sha256")
        or preflight_parent.get("adapter_sha256") != parent_probe.get("adapter_sha256")
        or preflight_parent.get("dev_sha256") != parent_release.get("dev_sha256")
    ):
        raise ValueError("generation manifest provenance/gate lineage mismatch")
    output_identity = (run.get("output_artifacts") or {}).get("predictions") or {}
    actual_identity = artifact_identity(predictions_path)
    if output_identity != actual_identity:
        raise ValueError("generation manifest does not bind the exact prediction artifact")
    if _sha256_file(predictions_path) != expected_predictions_sha256:
        raise ValueError("prediction file differs from external expected SHA256")
    return {
        "status": "PASS",
        "generation_manifest_sha256": expected_manifest_sha256,
        "predictions_sha256": expected_predictions_sha256,
        "probe_training_manifest_sha256": (preflight.get("probe") or {}).get(
            "training_manifest_sha256"
        ),
        "adapter_sha256": (preflight.get("probe") or {}).get("adapter_sha256"),
    }


def score_prediction_rows(
    examples: list[Mapping[str, Any]],
    predictions: list[Mapping[str, Any]],
    *,
    require_full_runner_rows: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Join ``example_id,response_text`` predictions and score Gold-free gates."""

    references: dict[str, Mapping[str, Any]] = {}
    for row in examples:
        example_id = str(row.get("example_id") or "")
        if not example_id or example_id in references:
            raise ValueError(f"missing or duplicate example_id in examples: {example_id!r}")
        references[example_id] = row
    responses: dict[str, str] = {}
    full_prediction_rows: dict[str, Mapping[str, Any]] = {}
    prediction_schema_kinds: set[str] = set()
    full_fields = {
        "schema_version",
        "example_id",
        "dataset",
        "qid",
        "question_key",
        "question_sha256",
        "family_sha256",
        "split",
        "slot",
        "input_record_sha256",
        "response_text",
        "response_sha256",
        "prompt_tokens",
        "generated_tokens",
        "valid",
        "checks",
        "error_codes",
        "parsed_target",
        "structured_target_exact",
        "canonical_text_exact",
    }
    for row in predictions:
        is_legacy_pair = set(row) == {"example_id", "response_text"}
        is_full_runner_row = (
            set(row) == full_fields
            and row.get("schema_version") == "query-controller-greedy-prediction-v1"
        )
        if not is_legacy_pair and not is_full_runner_row:
            raise ValueError(
                "prediction row must be the legacy exact 2-key schema or the exact "
                "query-controller-greedy-prediction-v1 runner schema"
            )
        prediction_schema_kinds.add("legacy_pair" if is_legacy_pair else "full_runner")
        example_id = str(row.get("example_id") or "")
        response = row.get("response_text")
        if not example_id or example_id in responses or not isinstance(response, str):
            raise ValueError(f"invalid or duplicate prediction: {example_id!r}")
        if is_full_runner_row:
            reference = references.get(example_id)
            if reference is None:
                raise ValueError(f"runner prediction has unknown example_id: {example_id}")
            for field in (
                "dataset", "qid", "question_key", "question_sha256",
                "family_sha256", "split", "slot",
            ):
                if row.get(field) != reference.get(field):
                    raise ValueError(
                        f"runner prediction identity mismatch for {example_id}: field={field}"
                    )
            expected_input_hash = hashlib.sha256(
                _canonical(reference).encode("utf-8")
            ).hexdigest()
            if row.get("input_record_sha256") != expected_input_hash:
                raise ValueError(f"runner prediction input hash mismatch: {example_id}")
            if row.get("response_sha256") != hashlib.sha256(response.encode("utf-8")).hexdigest():
                raise ValueError(f"runner prediction response hash mismatch: {example_id}")
            if (
                type(row.get("prompt_tokens")) is not int
                or row["prompt_tokens"] <= 0
                or type(row.get("generated_tokens")) is not int
                or row["generated_tokens"] < 0
            ):
                raise ValueError(f"runner prediction token telemetry invalid: {example_id}")
            full_prediction_rows[example_id] = row
        responses[example_id] = response
    if len(prediction_schema_kinds) > 1:
        raise ValueError(
            "mixing legacy 2-key and full runner prediction rows is forbidden"
        )
    if require_full_runner_rows and prediction_schema_kinds != {"full_runner"}:
        raise ValueError("formal mechanism scoring requires strict full runner prediction rows")
    missing = sorted(set(references) - set(responses))
    extra = sorted(set(responses) - set(references))
    if missing or extra:
        raise ValueError(f"prediction identity join mismatch: missing={missing[:3]}, extra={extra[:3]}")

    details: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    total = Counter()
    for example_id, reference in references.items():
        response = responses[example_id]
        candidate = dict(reference)
        parse_error: str | None = None
        target: Mapping[str, Any] | None = None
        try:
            target = parse_target_response(response, reference_record=reference)
            candidate["target"] = target
        except Exception as exc:  # stable telemetry, no permissive fallback
            parse_error = getattr(exc, "code", type(exc).__name__)
            # Do not accidentally report the frozen reference target's
            # mechanics for an unparseable prediction.
            candidate["target"] = {}
        audit = audit_action_record(candidate)
        exact = target is not None and _canonical(target) == _canonical(reference["target"])
        full_row = full_prediction_rows.get(example_id)
        if full_row is not None:
            expected_canonical_text_exact = response == _canonical_target(reference["target"])
            if (
                bool(full_row.get("valid")) != bool(audit["valid"])
                or dict(full_row.get("checks") or {}) != dict(audit["checks"])
                or list(full_row.get("error_codes") or []) != list(audit["errors"])
                or full_row.get("parsed_target") != target
                or bool(full_row.get("structured_target_exact")) != bool(exact)
                or bool(full_row.get("canonical_text_exact"))
                != expected_canonical_text_exact
            ):
                raise ValueError(
                    f"runner/scorer mechanical telemetry mismatch for {example_id}"
                )
        dataset, slot = str(reference["dataset"]), str(reference["slot"])
        group = groups[(dataset, slot)]
        total["n"] += 1
        group["n"] += 1
        total["parsed"] += int(target is not None)
        group["parsed"] += int(target is not None)
        total["exact"] += int(exact)
        group["exact"] += int(exact)
        for check, passed in audit["checks"].items():
            total[check] += int(passed)
            group[check] += int(passed)
        details.append(
            {
                "example_id": example_id,
                "dataset": dataset,
                "slot": slot,
                "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                "parsed": target is not None,
                "parse_error": parse_error,
                "checks": audit["checks"],
                "errors": audit["errors"],
                "target_exact": exact,
            }
        )

    def rates(values: Counter[str]) -> dict[str, Any]:
        n = values["n"]
        names = [
            "parsed",
            "exact",
            "schema_valid",
            "query_contract_valid",
            "query_nonrepeat",
            "placeholder_free",
            "dependency_closed",
            "source_action_valid",
            "state_use_valid",
            "gold_boundary_valid",
        ]
        return {"n": n, **{f"{name}_rate": values[name] / n if n else 0.0 for name in names}}

    report = {
        "schema_version": "query-controller-greedy-action-score-v1",
        "n": total["n"],
        "identity_join_rate": 1.0,
        "gold_access": False,
        "answer_scoring_performed": False,
        "metrics": rates(total),
        "by_dataset_slot": {
            f"{dataset}::{slot}": rates(values)
            for (dataset, slot), values in sorted(groups.items())
        },
    }
    return report, details


def run(
    *,
    records_path: Path,
    output_dir: Path,
    predictions_path: Path | None = None,
    protocol_path: Path | None = None,
    eval_protocol_path: Path | None = None,
    cohort_role: str | None = None,
    expected_protocol_sha256: str | None = None,
    expected_eval_protocol_sha256: str | None = None,
    expected_predictions_sha256: str | None = None,
    generation_manifest_path: Path | None = None,
    expected_generation_manifest_sha256: str | None = None,
    expected_split: str | None = None,
    experiment_id: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"append-only output already exists: {output_dir}")
    formal_values = (protocol_path, eval_protocol_path, cohort_role)
    if any(value is not None for value in formal_values) and not all(
        value is not None for value in formal_values
    ):
        raise ValueError(
            "protocol_path, eval_protocol_path, and cohort_role must be supplied together"
        )
    if protocol_path is not None:
        if cohort_role != "dev":
            raise ValueError("v4.4 formal mechanism scoring authorizes dev only")
        if expected_split is not None and expected_split != cohort_role:
            raise ValueError("cohort_role must equal expected_split")
        expected_split = cohort_role
        if (
            not expected_protocol_sha256
            or not expected_eval_protocol_sha256
            or not expected_predictions_sha256
            or generation_manifest_path is None
            or not expected_generation_manifest_sha256
        ):
            raise ValueError(
                "formal mechanism scoring requires external protocol, prediction, and "
                "generation-manifest SHA256 locks"
            )
        if predictions_path is None:
            raise ValueError("formal mechanism scoring requires runner predictions")
    records = _load_jsonl(records_path)
    release_report = evaluate_action_records(records, expected_split=expected_split)
    if predictions_path is None:
        report = {
            "schema_version": "query-controller-action-release-score-v1",
            "experiment_id": experiment_id,
            "status": "PASS" if release_report["all_valid"] else "FAIL_STOP",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "gold_access": False,
            "answer_scoring_performed": False,
            "release": release_report,
            "scientific_boundary": "Schema/mechanism audit only; no EM/F1/IHR claim.",
        }
        details: list[dict[str, Any]] = []
    else:
        if protocol_path is not None:
            protocol, protocol_bundle = _verify_protocol_bundle(
                protocol_path,
                expected_protocol_sha256=str(expected_protocol_sha256),
                verify_current_implementation_locks=False,
            )
            eval_protocol, eval_protocol_bundle = _verify_eval_successor_bundle(
                eval_protocol_path,
                expected_eval_protocol_sha256=str(expected_eval_protocol_sha256),
                parent_protocol_path=protocol_path,
                parent_protocol=protocol,
                parent_bundle=protocol_bundle,
                generation_experiment_id=EVAL_GENERATION_EXPERIMENT_ID,
                generation_output_dir=EVAL_GENERATION_OUTPUT_DIR,
            )
            release_lock = _verify_dev_release_and_pairs(
                records_path,
                records,
                protocol_path=protocol_path,
                protocol=protocol,
                protocol_sha256=protocol_bundle["protocol_sha256"],
                protocol_report_sha256=protocol_bundle["protocol_report_sha256"],
                protocol_manifest_sha256=protocol_bundle["protocol_manifest_sha256"],
            )
            generation_lock = _verify_generation_manifest(
                generation_manifest_path,
                expected_manifest_sha256=str(expected_generation_manifest_sha256),
                predictions_path=predictions_path,
                expected_predictions_sha256=str(expected_predictions_sha256),
                protocol_bundle=protocol_bundle,
                eval_protocol=eval_protocol,
                eval_protocol_bundle=eval_protocol_bundle,
                release_lock=release_lock,
            )
        else:
            protocol = None
            protocol_bundle = None
            eval_protocol_bundle = None
            eval_protocol = None
            release_lock = None
            generation_lock = None
        predictions = _load_jsonl(predictions_path)
        prediction_report, details = score_prediction_rows(
            records,
            predictions,
            require_full_runner_rows=protocol_path is not None,
        )
        all_parsed = prediction_report["metrics"]["parsed_rate"] == 1.0
        mechanism_gate = None
        if protocol_path is not None:
            mechanism_gate = evaluate_teacher_forced_mechanism_gate(
                {
                    "identity_join_rate": prediction_report["identity_join_rate"],
                    "overall": prediction_report["metrics"],
                    "by_dataset_slot": prediction_report["by_dataset_slot"],
                },
                protocol,
                cohort_role=str(cohort_role),
            )
        report = {
            "schema_version": "query-controller-greedy-score-report-v1",
            "experiment_id": experiment_id,
            # Completing the scoring job is not the same as passing the
            # versioned mechanism gate.  Keep these states orthogonal.
            "status": "COMPLETE_SCORED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "gold_access": False,
            "answer_scoring_performed": False,
            "reference_release": release_report,
            "predictions": prediction_report,
            "prediction_file_status": "COMPLETE",
            "all_predictions_parsed": all_parsed,
            "mechanism_gate": mechanism_gate,
            "formal_asset_lock": {
                "protocol": protocol_bundle,
                "eval_protocol": eval_protocol_bundle,
                "release": release_lock,
                "predictions_sha256": expected_predictions_sha256,
                "generation": generation_lock,
            } if protocol_path is not None else None,
            "scientific_boundary": "Greedy action mechanics only; no retrieval utility, EM/F1, or IHR claim.",
        }

    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "report.json"
    _write_json(report_path, report)
    if details:
        _write_jsonl(output_dir / "details.jsonl", details)
    manifest = {
        "schema_version": "query-controller-action-score-manifest-v1",
        "experiment_id": experiment_id,
        "status": report["status"],
        "inputs": {
            "records": {"path": str(records_path), "sha256": _sha256_file(records_path)},
            "predictions": (
                {"path": str(predictions_path), "sha256": _sha256_file(predictions_path)}
                if predictions_path is not None
                else None
            ),
            "protocol": (
                {"path": str(protocol_path), "sha256": _sha256_file(protocol_path)}
                if protocol_path is not None else None
            ),
            "eval_protocol": (
                {
                    "path": str(eval_protocol_path),
                    "sha256": _sha256_file(eval_protocol_path),
                }
                if eval_protocol_path is not None else None
            ),
            "generation_manifest": (
                {
                    "path": str(generation_manifest_path),
                    "sha256": _sha256_file(generation_manifest_path),
                }
                if generation_manifest_path is not None else None
            ),
        },
        "outputs": {
            "report.json": _sha256_file(report_path),
            "details.jsonl": (
                _sha256_file(output_dir / "details.jsonl") if details else None
            ),
        },
        "gold_access": False,
        "answer_scoring_performed": False,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--eval_protocol", type=Path)
    parser.add_argument("--cohort_role", choices=("dev",))
    parser.add_argument("--expected_protocol_sha256")
    parser.add_argument("--expected_eval_protocol_sha256")
    parser.add_argument("--expected_predictions_sha256")
    parser.add_argument("--generation_manifest", type=Path)
    parser.add_argument("--expected_generation_manifest_sha256")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--expected_split", choices=("train", "dev", "confirmation"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run(
        records_path=args.records,
        predictions_path=args.predictions,
        protocol_path=args.protocol,
        eval_protocol_path=args.eval_protocol,
        cohort_role=args.cohort_role,
        expected_protocol_sha256=args.expected_protocol_sha256,
        expected_eval_protocol_sha256=args.expected_eval_protocol_sha256,
        expected_predictions_sha256=args.expected_predictions_sha256,
        generation_manifest_path=args.generation_manifest,
        expected_generation_manifest_sha256=args.expected_generation_manifest_sha256,
        output_dir=args.output_dir,
        expected_split=args.expected_split,
        experiment_id=args.experiment_id,
    )
    print(json.dumps({"status": report["status"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
