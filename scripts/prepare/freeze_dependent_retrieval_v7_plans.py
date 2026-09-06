#!/usr/bin/env python
"""Validate and freeze v7 planner outputs before Gold-free materialisation.

This second CPU-only execution lock joins all 40 planner rows one-to-one to the
frozen question-only cohort, re-parses the raw model response, recomputes both
the supervision-schema and the actual runner execution verdicts, rejects
recursive Gold/decomposition fields, re-hashes the implementation's model and
Wiki18 commitments, and verifies the planner report/manifest.  Only a passing
output authorizes the Gold-free staged retrieval/subanswer materialisation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from kgproweight.eval.query_planner import parse_plan, plan_validation_errors
from kgproweight.kg.question_kg import question_sha256
from kgproweight.retrieval.dependent import validate_plan_for_dependent_retrieval
from kgproweight.utils.logging import artifact_identity
from scripts.pilot import materialize_paired_dependent_retrieval_v7 as retrieval_runner
from scripts.prepare import freeze_dependent_retrieval_v7 as v7_freeze
from scripts.prepare import freeze_dependent_retrieval_v7_implementation as implementation


SCHEMA_VERSION = "subquestion-dependent-retrieval-v7-plan-lock-1"
STATUS = "AUTHORIZED_GOLD_FREE_MATERIALIZATION"
SCOPE = "DEVELOPMENT_ONLY_GLOBALLY_CONSUMED_HOTPOT20_MUSIQUE20"
EXPERIMENT_ID = "SUBQUESTION-DEPENDENT-RETRIEVAL-V7-DEV20X2-SEED20260904-PLAN-LOCK-V1"
EXPECTED_N = 40
EXPECTED_COUNTS = {"hotpotqa": 20, "musique": 20}
MIN_SCHEMA_VALID_RATE_EACH_DATASET = 0.8
MIN_PLAN_EXECUTABLE_RATE_EACH_DATASET = 0.8
MODEL_CONTENT_ROLES = frozenset(
    {"query_planner", "retrieval_encoder", "cross_encoder", "strong_sft", "base_model"}
)
WIKI18_CONTENT_ROLES = frozenset({"corpus", "dense_index", "bm25_index"})
PREDICTION_KEYS = frozenset(
    {
        "row_id",
        "question_key",
        "dataset",
        "qid",
        "question",
        "question_sha256",
        "generated_text",
        "predicted_target",
        "schema_valid",
        "validation_errors",
        "gold_access",
    }
)

DEFAULT_IMPLEMENTATION = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_implementation_lock_v1/protocol.json"
)
DEFAULT_IMPLEMENTATION_MANIFEST = DEFAULT_IMPLEMENTATION.with_name("manifest.json")
DEFAULT_PLANNER_DIR = Path(
    "outputs/validation/subquestion_dependent_retrieval_v7_development_plans_v1"
)
DEFAULT_OUT = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_plans_lock_v1"
)


def canonical_json_bytes(value: Any) -> bytes:
    return implementation.canonical_json_bytes(value)


def file_lock(path: Path) -> dict[str, Any]:
    return implementation.file_lock(path)


def read_json(path: Path) -> dict[str, Any]:
    return implementation.read_json(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return implementation.read_jsonl(path)


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _recorded_file_identity_matches(
    recorded: Mapping[str, Any], path: Path, *, label: str
) -> None:
    resolved = path.expanduser().resolve()
    if Path(str(recorded.get("path") or "")).expanduser().resolve() != resolved:
        raise ValueError(f"{label} path drift")
    if recorded.get("exists") is not True or recorded.get("kind") != "file":
        raise ValueError(f"{label} is not a recorded file")
    if int(recorded.get("size_bytes", -1)) != resolved.stat().st_size:
        raise ValueError(f"{label} size drift")
    if str(recorded.get("md5") or "") != _md5_file(resolved):
        raise ValueError(f"{label} MD5 drift")


def _recorded_adapter_identity_matches(
    recorded: Mapping[str, Any], adapter_lock: Mapping[str, Any]
) -> None:
    """Bind the planner report to the adapter config and weight bytes.

    The planner generator records ``artifact_identity`` (MD5 for the critical
    adapter files), while the implementation lock carries a recursive SHA256
    tree commitment.  Requiring equality to a freshly recomputed report
    identity, and separately revalidating the SHA256 tree below, joins those
    two independently produced records without trusting a directory path.
    """

    adapter_path = implementation._path_from_lock(adapter_lock, "planner adapter")
    current = artifact_identity(adapter_path)
    if dict(recorded) != current:
        raise ValueError("planner report adapter content identity drift")
    files = {
        str(row.get("name") or ""): row
        for row in current.get("files") or []
        if isinstance(row, Mapping)
    }
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        row = files.get(name)
        if not isinstance(row, Mapping) or not str(row.get("md5") or ""):
            raise ValueError(f"planner report adapter lacks hashed {name}")


def _expected_content_locks_from_preregistration(
    preregistration: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    models = preregistration.get("models") or {}
    inherited = models.get("inherited_content_locks") or {}
    model_locks = {
        "query_planner": (models.get("query_planner") or {}).get("content_lock"),
        **{
            name: inherited.get(name)
            for name in ("retrieval_encoder", "cross_encoder", "strong_sft", "base_model")
        },
    }
    retrieval_locks = preregistration.get("retrieval_asset_content_locks") or {}
    wiki18_locks = {
        "corpus": retrieval_locks.get("corpus"),
        "dense_index": retrieval_locks.get("dense_index"),
        "bm25_index": retrieval_locks.get("bm25_index"),
    }
    if any(not isinstance(lock, Mapping) for lock in model_locks.values()):
        raise ValueError("preregistration model content locks are incomplete")
    if any(not isinstance(lock, Mapping) for lock in wiki18_locks.values()):
        raise ValueError("preregistration Wiki18 content locks are incomplete")
    return model_locks, wiki18_locks


def _reverify_implementation_content(
    protocol: Mapping[str, Any], preregistration: Mapping[str, Any]
) -> dict[str, Any]:
    """Close the planner-window TOCTOU gap with fresh full content hashes."""

    recorded = (protocol.get("content_reverification") or {}).get("verified") or {}
    recorded_models = recorded.get("models") if isinstance(recorded, Mapping) else None
    recorded_wiki18 = recorded.get("wiki18") if isinstance(recorded, Mapping) else None
    if not isinstance(recorded_models, Mapping) or set(recorded_models) != MODEL_CONTENT_ROLES:
        raise ValueError("implementation model content-lock role set drift")
    if not isinstance(recorded_wiki18, Mapping) or set(recorded_wiki18) != WIKI18_CONTENT_ROLES:
        raise ValueError("implementation Wiki18 content-lock role set drift")

    expected_models, expected_wiki18 = _expected_content_locks_from_preregistration(
        preregistration
    )
    if dict(recorded_models) != expected_models:
        raise ValueError("implementation model locks differ from preregistration")
    if dict(recorded_wiki18) != expected_wiki18:
        raise ValueError("implementation Wiki18 locks differ from preregistration")

    planner_contract = protocol.get("planner_contract") or {}
    if planner_contract.get("adapter") != recorded_models["query_planner"]:
        raise ValueError("implementation planner adapter lock is not authoritative")
    if planner_contract.get("base_model") != recorded_models["base_model"]:
        raise ValueError("implementation planner base-model lock is not authoritative")

    current_models = {
        name: implementation.verify_tree_lock(
            recorded_models[name], f"implementation.content.models.{name}"
        )
        for name in sorted(MODEL_CONTENT_ROLES)
    }
    current_wiki18 = {
        "corpus": implementation.verify_file_lock(
            recorded_wiki18["corpus"], "implementation.content.wiki18.corpus"
        ),
        "dense_index": implementation.verify_file_lock(
            recorded_wiki18["dense_index"], "implementation.content.wiki18.dense_index"
        ),
        "bm25_index": implementation.verify_tree_lock(
            recorded_wiki18["bm25_index"], "implementation.content.wiki18.bm25_index"
        ),
    }
    return {"models": current_models, "wiki18": current_wiki18}


def _dependent_execution_errors(
    plan: Mapping[str, Any], target_type: str
) -> list[str]:
    """Run the same structural validator and scheduler used by the v7 runner."""

    errors = validate_plan_for_dependent_retrieval(
        plan, target_type, max_steps=retrieval_runner.MAX_PLAN_STEPS
    )
    if not errors:
        try:
            retrieval_runner._step_schedule(list(plan.get("steps") or []))
        except Exception as exc:
            errors = [f"schedule_error:{type(exc).__name__}:{exc}"]
    return list(errors)


def _verify_implementation(
    protocol_path: Path, manifest_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol_lock = file_lock(protocol_path)
    protocol = read_json(protocol_path)
    if protocol.get("schema_version") != implementation.SCHEMA_VERSION:
        raise ValueError("v7 implementation lock schema drift")
    if protocol.get("status") != implementation.STATUS:
        raise ValueError("v7 implementation lock status drift")
    authorization = protocol.get("authorization") or {}
    expected_authorization = {
        "planner": True,
        "gold_free_materialization": False,
        "gold_attachment": False,
        "answer_evaluation": False,
        "training": False,
    }
    if authorization != expected_authorization:
        raise ValueError("v7 implementation authorization drift")
    if protocol.get("gold_access") is not False:
        raise ValueError("v7 implementation Gold boundary drift")
    if (protocol.get("content_reverification") or {}).get(
        "full_hash_verification_performed"
    ) is not True:
        raise ValueError("implementation lock did not fully re-hash model/Wiki18 content")
    runtime_code = protocol.get("runtime_code") or {}
    if set(runtime_code) != set(implementation.DEFAULT_RUNTIME_PATHS):
        raise ValueError("implementation runtime role set drift")
    for name, lock in runtime_code.items():
        if not isinstance(lock, Mapping):
            raise ValueError(f"invalid implementation runtime lock: {name}")
        implementation.verify_file_lock(lock, f"implementation.runtime_code.{name}")
    if Path(str(runtime_code["retrieval_runner"].get("path") or "")).resolve() != Path(
        retrieval_runner.__file__
    ).resolve():
        raise ValueError("implementation retrieval runner is not the validator module in use")
    if retrieval_runner.MAX_PLAN_STEPS != 4 or retrieval_runner.TARGET_TYPES != (
        implementation.EXPECTED_TARGET_TYPES
    ):
        raise ValueError("retrieval runner dependent-plan contract drift")
    for name, lock in (protocol.get("actual_local_import_closure") or {}).items():
        if not isinstance(lock, Mapping):
            raise ValueError(f"invalid implementation import lock: {name}")
        implementation.verify_file_lock(lock, f"implementation.imports.{name}")
    issuer = protocol.get("lock_issuer")
    if not isinstance(issuer, Mapping):
        raise ValueError("implementation lock issuer is missing")
    implementation.verify_file_lock(issuer, "implementation.lock_issuer")
    if Path(str(issuer.get("path") or "")).resolve() != Path(implementation.__file__).resolve():
        raise ValueError("implementation lock issuer path drift")

    parents = protocol.get("parents") or {}
    expected_parent_roles = {
        "preregistration",
        "preregistration_manifest",
        "truncation_addendum",
        "truncation_addendum_manifest",
        "trajectory_semantics_addendum",
        "trajectory_semantics_addendum_manifest",
    }
    if not isinstance(parents, Mapping) or set(parents) != expected_parent_roles:
        raise ValueError("implementation parent-lock role set drift")
    for name, lock in parents.items():
        if not isinstance(lock, Mapping):
            raise ValueError(f"invalid implementation parent lock: {name}")
        implementation.verify_file_lock(lock, f"implementation.parents.{name}")

    preregistration = read_json(Path(str(parents["preregistration"]["path"])))
    truncation_addendum = read_json(Path(str(parents["truncation_addendum"]["path"])))
    trajectory_addendum = read_json(
        Path(str(parents["trajectory_semantics_addendum"]["path"]))
    )
    implementation._validate_addendum_semantics(
        truncation_addendum, parents["preregistration"]
    )
    implementation._validate_trajectory_addendum_semantics(trajectory_addendum)
    implementation._verify_trajectory_addendum_references(
        trajectory_addendum,
        preregistration_lock=parents["preregistration"],
        preregistration_manifest_lock=parents["preregistration_manifest"],
        truncation_addendum_lock=parents["truncation_addendum"],
        truncation_addendum_manifest_lock=parents["truncation_addendum_manifest"],
    )
    implementation._validate_manifest(
        read_json(Path(str(parents["trajectory_semantics_addendum_manifest"]["path"]))),
        protocol_lock=parents["trajectory_semantics_addendum"],
        expected_status=implementation.TRAJECTORY_ADDENDUM_STATUS,
        label="v7 recursive trajectory addendum manifest",
    )
    reverified_content = _reverify_implementation_content(protocol, preregistration)

    manifest_lock = file_lock(manifest_path)
    manifest = read_json(manifest_path)
    implementation._validate_manifest(
        manifest,
        protocol_lock=protocol_lock,
        expected_status=implementation.STATUS,
        label="v7 implementation manifest",
    )
    return protocol, protocol_lock, manifest_lock, reverified_content


def _cohort_index(
    implementation_protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    lock = (implementation_protocol.get("inputs") or {}).get("planner_cohort")
    if not isinstance(lock, Mapping):
        raise ValueError("implementation lock has no planner cohort")
    implementation.verify_file_lock(lock, "implementation.inputs.planner_cohort")
    rows = read_jsonl(Path(str(lock["path"])))
    if len(rows) != EXPECTED_N:
        raise ValueError("planner cohort is not n=40")
    index: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        v7_freeze.assert_answer_free(row, location=f"planner_cohort[{row_number}]")
        key = str(row.get("question_key") or "")
        if not key or key in index:
            raise ValueError(f"duplicate/empty planner cohort identity: {key!r}")
        index[key] = row
    return rows, index


def validate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    cohort_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(predictions) != EXPECTED_N:
        raise ValueError(f"planner predictions must contain {EXPECTED_N} rows")
    if Counter(str(row.get("dataset") or "") for row in predictions) != Counter(
        EXPECTED_COUNTS
    ):
        raise ValueError("planner prediction dataset counts drift")
    cohort = {str(row["question_key"]): dict(row) for row in cohort_rows}
    if len(cohort) != EXPECTED_N:
        raise ValueError("planner cohort identity is not unique")
    seen: set[str] = set()
    schema_valid_counts: Counter[str] = Counter()
    executable_counts: Counter[str] = Counter()
    dependent_invalid_counts: Counter[str] = Counter()
    keys_in_order: list[str] = []
    for index, raw in enumerate(predictions):
        if not isinstance(raw, Mapping):
            raise ValueError(f"prediction[{index}] is not an object")
        row = dict(raw)
        if frozenset(row) != PREDICTION_KEYS:
            raise ValueError(
                f"prediction[{index}] exact fields drift; "
                f"missing={sorted(PREDICTION_KEYS - frozenset(row))}, "
                f"extra={sorted(frozenset(row) - PREDICTION_KEYS)}"
            )
        v7_freeze.assert_answer_free(row, location=f"prediction[{index}]")
        if row.get("gold_access") is not False:
            raise ValueError(f"prediction[{index}] Gold boundary drift")
        key = str(row.get("question_key") or "")
        if key not in cohort or key in seen:
            raise ValueError(f"prediction[{index}] one-to-one identity failure: {key!r}")
        seen.add(key)
        keys_in_order.append(key)
        source = cohort[key]
        for field in (
            "row_id",
            "question_key",
            "dataset",
            "qid",
            "question",
            "question_sha256",
        ):
            if row.get(field) != source.get(field):
                raise ValueError(f"prediction[{index}] cohort join drift: {field}")
        if question_sha256(str(row["question"])) != str(row["question_sha256"]):
            raise ValueError(f"prediction[{index}] question SHA256 mismatch")
        generated_text = row.get("generated_text")
        if not isinstance(generated_text, str):
            raise ValueError(f"prediction[{index}].generated_text is not a string")
        reparsed, parse_error = parse_plan(generated_text)
        if reparsed != row.get("predicted_target"):
            raise ValueError(f"prediction[{index}] raw-text parsed plan mismatch")
        validation_record = {
            "schema_version": "query-planner-supervision-1",
            "question_key": source["question_key"],
            "dataset": source["dataset"],
            "qid": source["qid"],
            "question": source["question"],
            "question_sha256": source["question_sha256"],
            "target_type": source["target_type"],
        }
        errors = (
            plan_validation_errors(validation_record, reparsed)
            if reparsed is not None
            else [str(parse_error)]
        )
        if row.get("validation_errors") != errors:
            raise ValueError(f"prediction[{index}] validation_errors drift")
        if row.get("schema_valid") is not (not errors):
            raise ValueError(f"prediction[{index}] schema_valid drift")
        if reparsed is not None:
            steps = reparsed.get("steps")
            if not isinstance(steps, list) or len(steps) > 4:
                raise ValueError(f"prediction[{index}] plan-step budget drift")
        dataset = str(row["dataset"])
        if not errors:
            schema_valid_counts[dataset] += 1
        dependent_errors = (
            _dependent_execution_errors(reparsed, str(source["target_type"]))
            if reparsed is not None
            else [str(parse_error)]
        )
        if not dependent_errors:
            executable_counts[dataset] += 1
        else:
            dependent_invalid_counts[dataset] += 1
    if seen != set(cohort):
        raise ValueError("planner predictions do not cover the exact cohort")
    if keys_in_order != [str(row["question_key"]) for row in cohort_rows]:
        raise ValueError("planner prediction order differs from frozen cohort")
    schema_rates = {
        dataset: schema_valid_counts[dataset] / EXPECTED_COUNTS[dataset]
        for dataset in EXPECTED_COUNTS
    }
    executable_rates = {
        dataset: executable_counts[dataset] / EXPECTED_COUNTS[dataset]
        for dataset in EXPECTED_COUNTS
    }
    failing_executable = {
        dataset: value
        for dataset, value in executable_rates.items()
        if value < MIN_PLAN_EXECUTABLE_RATE_EACH_DATASET
    }
    if failing_executable:
        raise ValueError(
            f"plan executable preregistered gate failed: {failing_executable}"
        )
    failing_schema = {
        dataset: value
        for dataset, value in schema_rates.items()
        if value < MIN_SCHEMA_VALID_RATE_EACH_DATASET
    }
    if failing_schema:
        raise ValueError(f"plan schema-valid integrity gate failed: {failing_schema}")
    return {
        "n": len(predictions),
        "by_dataset": dict(Counter(str(row["dataset"]) for row in predictions)),
        "question_key_order_sha256": implementation.sha256_text("\n".join(keys_in_order)),
        "schema_valid": {
            dataset: schema_valid_counts[dataset] for dataset in EXPECTED_COUNTS
        },
        "schema_valid_rate": schema_rates,
        "schema_valid_rate_min_each_dataset": MIN_SCHEMA_VALID_RATE_EACH_DATASET,
        "plan_executable": {
            dataset: executable_counts[dataset] for dataset in EXPECTED_COUNTS
        },
        "plan_executable_rate": executable_rates,
        "plan_executable_rate_min_each_dataset": (
            MIN_PLAN_EXECUTABLE_RATE_EACH_DATASET
        ),
        "dependent_execution_invalid": {
            dataset: dependent_invalid_counts[dataset] for dataset in EXPECTED_COUNTS
        },
        "dependent_execution_validator": {
            "validator": (
                "kgproweight.retrieval.dependent."
                "validate_plan_for_dependent_retrieval"
            ),
            "scheduler": (
                "scripts.pilot.materialize_paired_dependent_retrieval_v7."
                "_step_schedule"
            ),
            "max_plan_steps": retrieval_runner.MAX_PLAN_STEPS,
        },
        "plan_executable_gate_pass": True,
    }


def _validate_planner_report(
    *,
    report: Mapping[str, Any],
    report_path: Path,
    manifest: Mapping[str, Any],
    prediction_path: Path,
    prediction_lock: Mapping[str, Any],
    implementation_path: Path,
    implementation_protocol: Mapping[str, Any],
    population: Mapping[str, Any],
) -> None:
    expected_experiment = str(
        (implementation_protocol.get("planner_contract") or {}).get("experiment_id") or ""
    )
    if report.get("status") != "RUNTIME_PLANS_FROZEN_NO_GOLD_AUDIT":
        raise ValueError("planner report status drift")
    if report.get("experiment_id") != expected_experiment:
        raise ValueError("planner report Experiment ID drift")
    if report.get("gold_access") is not False:
        raise ValueError("planner report Gold boundary drift")
    generation = report.get("generation") or {}
    if (
        generation.get("greedy") is not True
        or int(generation.get("max_new_tokens", -1)) != 512
        or int(generation.get("batch_size", 0)) <= 0
    ):
        raise ValueError("planner decoding contract drift")
    counts = report.get("counts") or {}
    if int(counts.get("n", -1)) != EXPECTED_N or counts.get("by_dataset") != EXPECTED_COUNTS:
        raise ValueError("planner report population drift")
    if int(counts.get("schema_valid", -1)) != sum(
        int(value) for value in (population.get("schema_valid") or {}).values()
    ):
        raise ValueError("planner report schema-valid count drift")
    if float((report.get("rates") or {}).get("schema_valid", -1.0)) != (
        sum(int(value) for value in (population.get("schema_valid") or {}).values())
        / EXPECTED_N
    ):
        raise ValueError("planner report schema-valid rate drift")

    recorded_inputs = report.get("inputs") or {}
    planner_cohort_path = Path(
        str((implementation_protocol["inputs"]["planner_cohort"])["path"])
    )
    _recorded_file_identity_matches(
        recorded_inputs.get("cohort") or {}, planner_cohort_path, label="planner report cohort"
    )
    _recorded_file_identity_matches(
        recorded_inputs.get("protocol") or {}, implementation_path, label="planner report protocol"
    )
    _recorded_file_identity_matches(
        recorded_inputs.get("config") or {},
        Path(str((implementation_protocol["planner_contract"]["config"])["path"])),
        label="planner report config",
    )
    _recorded_adapter_identity_matches(
        recorded_inputs.get("adapter") or {},
        implementation_protocol["planner_contract"]["adapter"],
    )
    _recorded_file_identity_matches(
        (report.get("outputs") or {}).get("predictions") or {},
        prediction_path,
        label="planner report predictions",
    )
    if str(prediction_lock["sha256"]) != implementation.sha256_file(prediction_path):
        raise ValueError("planner prediction lock changed during report validation")

    if manifest.get("status") != report.get("status"):
        raise ValueError("planner manifest status drift")
    # ``dump_manifest`` stores caller metadata under ``run``; lightweight
    # synthetic/legacy manifests may expose the same identifier at the root.
    # Accept either documented layout, but when the nested run record exists
    # require it to be the exact report payload so the manifest cannot attest
    # to a different planner invocation.
    manifest_run = manifest.get("run")
    if isinstance(manifest_run, Mapping):
        if dict(manifest_run) != dict(report):
            raise ValueError("planner manifest run/report payload drift")
        manifest_experiment = manifest_run.get("experiment_id")
    else:
        manifest_experiment = manifest.get("experiment_id")
    if manifest_experiment != expected_experiment:
        raise ValueError("planner manifest Experiment ID drift")
    manifest_gold_access = (
        manifest_run.get("gold_access")
        if isinstance(manifest_run, Mapping)
        else manifest.get("gold_access")
    )
    if manifest_gold_access is not False:
        raise ValueError("planner manifest Gold boundary drift")
    # The upstream generator currently carries a historical fixed scope string.
    # It is non-authoritative; n/dataset/order are re-derived above and frozen
    # here rather than silently calling this globally unseen n100.
    if report_path.resolve() == prediction_path.resolve():
        raise ValueError("report and predictions cannot be the same artifact")


def build_plan_lock_protocol(
    *,
    implementation_path: Path,
    implementation_manifest_path: Path,
    predictions_path: Path,
    planner_report_path: Path,
    planner_manifest_path: Path,
) -> dict[str, Any]:
    impl, impl_lock, impl_manifest_lock, reverified_content = _verify_implementation(
        implementation_path, implementation_manifest_path
    )
    cohort_rows, _ = _cohort_index(impl)
    predictions = read_jsonl(predictions_path)
    population = validate_predictions(predictions, cohort_rows)
    predictions_lock = file_lock(predictions_path)
    report_lock = file_lock(planner_report_path)
    manifest_lock = file_lock(planner_manifest_path)
    report = read_json(planner_report_path)
    manifest = read_json(planner_manifest_path)
    _validate_planner_report(
        report=report,
        report_path=planner_report_path,
        manifest=manifest,
        prediction_path=predictions_path,
        prediction_lock=predictions_lock,
        implementation_path=implementation_path,
        implementation_protocol=impl,
        population=population,
    )

    prereg = (impl.get("parents") or {}).get("preregistration")
    addendum = (impl.get("parents") or {}).get("truncation_addendum")
    trajectory_addendum = (impl.get("parents") or {}).get(
        "trajectory_semantics_addendum"
    )
    trajectory_addendum_manifest = (impl.get("parents") or {}).get(
        "trajectory_semantics_addendum_manifest"
    )
    if (
        not isinstance(prereg, Mapping)
        or not isinstance(addendum, Mapping)
        or not isinstance(trajectory_addendum, Mapping)
        or not isinstance(trajectory_addendum_manifest, Mapping)
    ):
        raise ValueError("implementation lock parent locks missing")
    implementation.verify_file_lock(prereg, "implementation parent preregistration")
    implementation.verify_file_lock(addendum, "implementation parent addendum")
    implementation.verify_file_lock(
        trajectory_addendum, "implementation parent trajectory addendum"
    )
    implementation.verify_file_lock(
        trajectory_addendum_manifest,
        "implementation parent trajectory addendum manifest",
    )
    runtime_code = dict(impl.get("runtime_code") or {})
    if set(runtime_code) != set(implementation.DEFAULT_RUNTIME_PATHS):
        raise ValueError("implementation runtime role set drift")
    runner_version = implementation._literal_constant(
        Path(str(runtime_code["retrieval_runner"]["path"])), "RUNNER_VERSION"
    )
    prereg_json = read_json(Path(str(prereg["path"])))
    materialization_experiment_id = str(
        (prereg_json.get("future_experiment_ids") or {}).get("materialization") or ""
    )
    if not materialization_experiment_id:
        raise ValueError("materialization Experiment ID missing")

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "scope": SCOPE,
        "parents": {
            "preregistration": dict(prereg),
            "truncation_addendum": dict(addendum),
            "trajectory_semantics_addendum": dict(trajectory_addendum),
            "trajectory_semantics_addendum_manifest": dict(
                trajectory_addendum_manifest
            ),
            "implementation_lock": impl_lock,
            "implementation_manifest": impl_manifest_lock,
        },
        "inputs": {
            "development": dict(impl["inputs"]["development"]),
            "planner_cohort": dict(impl["inputs"]["planner_cohort"]),
            "canonical_A_contexts": dict(impl["inputs"]["canonical_A_contexts"]),
            "planner_predictions": predictions_lock,
            "planner_report": report_lock,
            "planner_manifest": manifest_lock,
        },
        "runtime_code": runtime_code,
        "lock_issuer": file_lock(Path(__file__)),
        "content_reverification": {
            "performed_after_planner_generation": True,
            "implementation_commitments_equal": True,
            "verified": reverified_content,
        },
        "population": population,
        "planner_contract": {
            "experiment_id": impl["planner_contract"]["experiment_id"],
            "greedy": True,
            "do_sample": False,
            "max_new_tokens": 512,
            "raw_response_reparsed": True,
            "schema_verdict_recomputed": True,
            "dependent_execution_verdict_recomputed": True,
            "dependent_execution_validator": population[
                "dependent_execution_validator"
            ],
            "exact_cohort_identity_join": True,
            "recursive_forbidden_fields": 0,
            "upstream_report_scope_is_non_authoritative_historical_string": str(
                report.get("scope") or ""
            ),
        },
        "materialization_contract": {
            "experiment_id": materialization_experiment_id,
            "runner_version": runner_version,
            "n": EXPECTED_N,
            "by_dataset": EXPECTED_COUNTS,
            "gold_access": False,
            "network_access": False,
            "max_plan_steps": 4,
        },
        "authorization": {
            "planner_complete": True,
            "gold_free_materialization": True,
            "gold_attachment": False,
            "answer_evaluation": False,
            "training": False,
        },
        "gold_access": False,
        "gpu_calls_by_this_command": 0,
        "retrieval_calls_by_this_command": 0,
        "scientific_boundary": (
            "This lock authorizes only the frozen Gold-free v7 materialization. "
            "It is not a retrieval result, subanswer result, Gold score, independent "
            "confirmation result, training run, or utility claim."
        ),
    }


def write_protocol(protocol: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    resolved = output_dir.expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite v7 plan lock: {resolved}")
    resolved.mkdir(parents=True, exist_ok=False)
    protocol_path = resolved / "protocol.json"
    protocol_path.write_bytes(canonical_json_bytes(dict(protocol)))
    protocol_lock = file_lock(protocol_path)
    manifest = {
        "schema_version": "subquestion-dependent-retrieval-v7-plan-lock-manifest-1",
        "experiment_id": protocol["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "scope": SCOPE,
        "protocol": protocol_lock,
        "authorization": dict(protocol["authorization"]),
        "gold_access": False,
        "gpu_calls": 0,
        "retrieval_calls": 0,
    }
    manifest_path = resolved / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return {"protocol": protocol_lock, "manifest": file_lock(manifest_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", type=Path, default=DEFAULT_IMPLEMENTATION)
    parser.add_argument(
        "--implementation_manifest", type=Path, default=DEFAULT_IMPLEMENTATION_MANIFEST
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PLANNER_DIR / "predictions.question_only.jsonl",
    )
    parser.add_argument(
        "--planner_report", type=Path, default=DEFAULT_PLANNER_DIR / "report.json"
    )
    parser.add_argument(
        "--planner_manifest", type=Path, default=DEFAULT_PLANNER_DIR / "manifest.json"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = build_plan_lock_protocol(
        implementation_path=args.implementation,
        implementation_manifest_path=args.implementation_manifest,
        predictions_path=args.predictions,
        planner_report_path=args.planner_report,
        planner_manifest_path=args.planner_manifest,
    )
    result = write_protocol(protocol, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
