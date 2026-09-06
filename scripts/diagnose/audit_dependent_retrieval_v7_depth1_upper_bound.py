#!/usr/bin/env python
"""Gold-free monotonic upper-bound audit for the stopped v7 depth-1 run.

This command never opens dataset Gold and never calls a model, retriever, or
GPU.  It authenticates the append-only retry-1 root materialisation and
subanswer artifacts, reconstructs the currently verified C root slots, and
then propagates an *optimistic* success assumption through the frozen plan
dependency DAG:

* every future paired dependent retrieval that is reachable succeeds;
* every such future B entity extraction succeeds; and
* every future C reader attempt is mechanically verified.

Because every added future reader attempt is counted as verified, the resulting
ratio is a monotonic upper bound on the final mechanically-verified-subanswer
rate under the already-frozen no-retry trajectory.  If either dataset cannot
reach the preregistered 0.40 gate even under that assumption, continuing the
expensive recursive retrieval cannot change the gate decision.

The historical failure notes included in the report are diagnostic only.  The
two preflight notes without persisted logs are explicitly labelled
operator-supplied and unverified; they are never used in the upper-bound
calculation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ROOT_STATE = Path(
    "outputs/validation/subquestion_dependent_retrieval_v7_development_"
    "materialization_v1_retry1/root_state.jsonl"
)
DEFAULT_C_TASKS = Path(
    "outputs/validation/subquestion_dependent_retrieval_v7_development_"
    "materialization_v1_retry1/c_tasks.depth_1.jsonl"
)
DEFAULT_ROOTS_STAGE = Path(
    "outputs/validation/subquestion_dependent_retrieval_v7_development_"
    "materialization_v1_retry1/roots_stage.json"
)
DEFAULT_SUBANSWERS = Path(
    "outputs/validation/subquestion_dependent_retrieval_v7_subanswers_"
    "depth1_v1_retry1/subanswers.jsonl"
)
DEFAULT_GENERATOR_REPORT = Path(
    "outputs/validation/subquestion_dependent_retrieval_v7_subanswers_"
    "depth1_v1_retry1/report.json"
)
DEFAULT_GENERATOR_MANIFEST = Path(
    "outputs/validation/subquestion_dependent_retrieval_v7_subanswers_"
    "depth1_v1_retry1/manifest.json"
)
DEFAULT_IMPLEMENTATION_LOCK = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_"
    "implementation_lock_v1_retry1/protocol.json"
)
DEFAULT_PLAN_LOCK = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_"
    "plans_lock_v1_retry1/protocol.json"
)
DEFAULT_TRAJECTORY_LOCK = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_"
    "preregistration_addendum_recursive_trajectory_v1/protocol.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_depth1_"
    "monotonic_upper_bound_retry1"
)
DEFAULT_EXPERIMENT_ID = (
    "SUBQUESTION-DEPENDENT-RETRIEVAL-V7-DEV20X2-SEED20260904-"
    "DEPTH1-MONOTONIC-UPPER-BOUND-RETRY1"
)

DATASETS = ("hotpotqa", "musique")
EXPECTED_BY_DATASET = {"hotpotqa": 20, "musique": 20}
FORBIDDEN_GOLD_KEYS = frozenset(
    {
        "answer",
        "answers",
        "decomposition",
        "evidence",
        "evidences",
        "gold_answer",
        "gold_answers",
        "gold_target",
        "golden_answers",
        "paragraph_text",
        "question_decomposition",
        "reasoning",
        "sp",
        "supporting_facts",
        "target",
    }
)
TASK_KEYS = frozenset(
    {
        "task_id",
        "question_key",
        "dataset",
        "qid",
        "question",
        "question_sha256",
        "target_type",
        "producer_slot",
        "step",
        "step_sha256",
        "producer_passages",
        "producer_passages_sha256",
        "gold_access",
    }
)
ANSWER_KEYS = frozenset(
    {
        "task_id",
        "question_key",
        "dataset",
        "qid",
        "question_sha256",
        "target_type",
        "producer_slot",
        "step_sha256",
        "producer_passages_sha256",
        "verified",
        "verified_answer",
        "telemetry",
        "gold_access",
    }
)
IDENTITY_FIELDS = (
    "task_id",
    "question_key",
    "dataset",
    "qid",
    "question_sha256",
    "target_type",
    "producer_slot",
    "step_sha256",
    "producer_passages_sha256",
)

IMPLEMENTATION_SCHEMA = "subquestion-dependent-retrieval-v7-implementation-lock-1"
IMPLEMENTATION_STATUS = "AUTHORIZED_PLANNER_ONLY"
PLAN_SCHEMA = "subquestion-dependent-retrieval-v7-plan-lock-1"
PLAN_STATUS = "AUTHORIZED_GOLD_FREE_MATERIALIZATION"
TRAJECTORY_SCHEMA = "subquestion-dependent-retrieval-v7-recursive-trajectory-addendum-1"
TRAJECTORY_STATUS = "FROZEN_APPEND_ONLY_BEFORE_V7_EXECUTION"
ROOT_STAGE_SCHEMA = "paired-dependent-retrieval-v7-stage-descriptor-2"
ROOT_STATE_SCHEMA = "paired-dependent-retrieval-v7-root-state-1"
GENERATOR_REPORT_SCHEMA = "dependent-retrieval-v7-subanswer-report-1"
GENERATOR_OUTPUT_SCHEMA = "dependent-retrieval-v7-subanswer-output-1"
GENERATOR_MODEL_SCHEMA = "dependent-retrieval-v7-strong-sft-model-content-lock-1"

RUNNER_VERSION = "paired-dependent-retrieval-v7-staged-gold-free-1"
GENERATOR_VERSION = "grounded-subanswers-v7-1"
SCOPE = "DEVELOPMENT_ONLY_GLOBALLY_CONSUMED_HOTPOT20_MUSIQUE20"
FAIL_STATUS = "FAIL_STOP_BEFORE_GOLD_MONOTONIC_UPPER_BOUND"
CONTINUE_STATUS = "CONTINUE_GOLD_FREE_RECURSIVE_MATERIALIZATION"

ROOT_FAILURE_LOG = Path(
    "logs/eval/subquestion_dependent_retrieval_v7_development_"
    "materialization_v1.log"
)
DEPTH1_FAILURE_LOG = Path(
    "logs/eval/subquestion_dependent_retrieval_v7_development_"
    "materialization_depth1_v1_retry1.log"
)
_SLOT_RE = re.compile(
    r"^(?:\$(?:hop|step)_([1-9]\d*)|#([1-9]\d*)|(?:hop|step|slot)_([1-9]\d*))$",
    flags=re.IGNORECASE,
)


class AuditIntegrityError(RuntimeError):
    """An input differs from its frozen or append-only commitment."""


class _DuplicateJSONKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _assert_safe_input_path(path, label=label)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise AuditIntegrityError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditIntegrityError(f"{label} is not a JSON object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _assert_safe_input_path(path, label=label)
    result: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(
                    line,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
                if not isinstance(value, dict):
                    raise AuditIntegrityError(
                        f"{label} row {line_number} is not an object"
                    )
                result.append(value)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, AuditIntegrityError):
            raise
        raise AuditIntegrityError(f"cannot read {label}: {path}: {exc}") from exc
    return result


def _assert_safe_input_path(path: Path, *, label: str) -> None:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise AuditIntegrityError(f"{label} is outside the project root: {resolved}") from exc
    if relative.parts[:2] == ("data", "raw"):
        raise AuditIntegrityError(f"{label} attempts to read raw Gold: {resolved}")
    if not resolved.is_file():
        raise AuditIntegrityError(f"{label} is missing: {resolved}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_lock(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise AuditIntegrityError(f"locked file is missing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _assert_file_lock(
    expected: Mapping[str, Any], observed_path: Path, *, label: str
) -> dict[str, Any]:
    _assert_safe_input_path(observed_path, label=label)
    observed = file_lock(observed_path)
    if dict(expected) != observed:
        raise AuditIntegrityError(
            f"{label} lock mismatch: expected={dict(expected)!r}, observed={observed!r}"
        )
    return observed


def _assert_gold_free(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).strip().casefold()
            if key in FORBIDDEN_GOLD_KEYS:
                raise AuditIntegrityError(f"forbidden Gold key at {location}: {raw_key}")
            _assert_gold_free(nested, location=f"{location}.{raw_key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_gold_free(nested, location=f"{location}[{index}]")


def _normalise_slot(value: Any) -> str:
    match = _SLOT_RE.fullmatch(str(value or "").strip())
    if match is None:
        raise AuditIntegrityError(f"invalid dependency slot: {value!r}")
    number = next(group for group in match.groups() if group is not None)
    return f"slot_{int(number)}"


def _question_key(row: Mapping[str, Any]) -> str:
    dataset = str(row.get("dataset") or "").strip()
    qid = str(row.get("qid") or "").strip()
    key = str(row.get("question_key") or "").strip()
    expected = f"{dataset}::{qid}"
    if dataset not in DATASETS or not qid or key != expected:
        raise AuditIntegrityError(f"invalid question identity: {key!r}")
    return key


def _index_unique(
    rows: Iterable[Mapping[str, Any]], *, field: str, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows, start=1):
        value = str(raw.get(field) or "")
        if not value or value in result:
            raise AuditIntegrityError(f"{label} duplicate/empty {field} at row {index}")
        result[value] = dict(raw)
    return result


def _extract_plan(prediction: Mapping[str, Any]) -> dict[str, Any]:
    predicted = prediction.get("predicted_target")
    if isinstance(predicted, Mapping):
        return dict(predicted)
    if predicted is None:
        return {"steps": []}
    raise AuditIntegrityError("planner prediction has non-object predicted_target")


def _recompute_schedule(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    steps = plan.get("steps")
    if not isinstance(steps, list):
        raise AuditIntegrityError("plan steps are not a list")
    produced_depth: dict[str, int] = {}
    result: list[dict[str, Any]] = []
    for index, raw_step in enumerate(steps, start=1):
        if not isinstance(raw_step, Mapping):
            raise AuditIntegrityError(f"plan step {index} is not an object")
        step = dict(raw_step)
        slot = _normalise_slot(step.get("output_slot"))
        if slot in produced_depth:
            raise AuditIntegrityError(f"duplicate output slot: {slot}")
        raw_dependencies = step.get("dependencies") or []
        if not isinstance(raw_dependencies, list):
            raise AuditIntegrityError(f"step {index} dependencies are not a list")
        dependencies: list[str] = []
        for raw_dependency in raw_dependencies:
            dependency = _normalise_slot(raw_dependency)
            if dependency not in dependencies:
                dependencies.append(dependency)
        missing = [value for value in dependencies if value not in produced_depth]
        if missing:
            raise AuditIntegrityError(f"step {index} has unresolved dependencies: {missing}")
        depth = 1 + max((produced_depth[value] for value in dependencies), default=0)
        produced_depth[slot] = depth
        result.append(
            {
                "step_index": index,
                "step": step,
                "step_sha256": sha256_json(step),
                "slot": slot,
                "dependencies": dependencies,
                "dependency_depth": depth,
                "consumers": [],
            }
        )
    for producer in result:
        producer["consumers"] = [
            candidate["slot"]
            for candidate in result
            if producer["slot"] in candidate["dependencies"]
        ]
    return result


def _validate_lock_chain(
    *,
    implementation_path: Path,
    plan_path: Path,
    trajectory_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    implementation = _read_json(implementation_path, label="implementation lock")
    plan_lock = _read_json(plan_path, label="plan lock")
    trajectory = _read_json(trajectory_path, label="trajectory lock")
    for value, schema, status, label in (
        (implementation, IMPLEMENTATION_SCHEMA, IMPLEMENTATION_STATUS, "implementation"),
        (plan_lock, PLAN_SCHEMA, PLAN_STATUS, "plan"),
        (trajectory, TRAJECTORY_SCHEMA, TRAJECTORY_STATUS, "trajectory"),
    ):
        _assert_gold_free(value, location=f"{label}_lock")
        if value.get("schema_version") != schema or value.get("status") != status:
            raise AuditIntegrityError(f"{label} lock schema/status mismatch")
        if value.get("gold_access") is not False:
            raise AuditIntegrityError(f"{label} lock is not Gold-free")

    implementation_file_lock = file_lock(implementation_path)
    plan_file_lock = file_lock(plan_path)
    trajectory_file_lock = file_lock(trajectory_path)
    if (plan_lock.get("parents") or {}).get("implementation_lock") != implementation_file_lock:
        raise AuditIntegrityError("plan lock does not bind the retry-1 implementation lock")
    if (plan_lock.get("parents") or {}).get("trajectory_semantics_addendum") != trajectory_file_lock:
        raise AuditIntegrityError("plan lock does not bind the trajectory lock")
    if (implementation.get("parents") or {}).get("trajectory_semantics_addendum") != trajectory_file_lock:
        raise AuditIntegrityError("implementation lock does not bind the trajectory lock")
    if plan_lock.get("runtime_code") != implementation.get("runtime_code"):
        raise AuditIntegrityError("implementation/plan runtime-code locks differ")

    prereg_lock = (plan_lock.get("parents") or {}).get("preregistration")
    if not isinstance(prereg_lock, Mapping):
        raise AuditIntegrityError("plan lock lacks preregistration commitment")
    prereg_path = Path(str(prereg_lock.get("path") or ""))
    _assert_file_lock(prereg_lock, prereg_path, label="preregistration")
    prereg = _read_json(prereg_path, label="preregistration")
    _assert_gold_free(prereg, location="preregistration")
    threshold = (
        (prereg.get("decision_gates") or {})
        .get("gold_free_mechanism", {})
        .get("mechanically_verified_subanswer_rate_min_each_dataset")
    )
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise AuditIntegrityError("preregistration lacks the mechanical verification gate")
    if float(threshold) != 0.4:
        raise AuditIntegrityError(f"unexpected frozen mechanical gate: {threshold}")
    return implementation, plan_lock, trajectory, prereg


def _validate_root_artifacts(
    *,
    root_state_path: Path,
    tasks_path: Path,
    roots_stage_path: Path,
    implementation_file_lock: Mapping[str, Any],
    plan_file_lock: Mapping[str, Any],
    trajectory_file_lock: Mapping[str, Any],
    plan_lock: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    descriptor = _read_json(roots_stage_path, label="root stage descriptor")
    _assert_gold_free(descriptor, location="roots_stage")
    if (
        descriptor.get("schema_version") != ROOT_STAGE_SCHEMA
        or descriptor.get("runner_version") != RUNNER_VERSION
        or descriptor.get("stage") != "roots"
        or descriptor.get("state_depth") != 1
        or descriptor.get("gold_access") is not False
    ):
        raise AuditIntegrityError("root stage descriptor contract mismatch")
    runtime_locks = descriptor.get("runtime_locks") or {}
    if runtime_locks.get("implementation_lock") != dict(implementation_file_lock):
        raise AuditIntegrityError("root stage implementation lock mismatch")
    if runtime_locks.get("post_plan_execution_lock") != dict(plan_file_lock):
        raise AuditIntegrityError("root stage plan lock mismatch")
    if runtime_locks.get("trajectory_semantics_addendum") != dict(trajectory_file_lock):
        raise AuditIntegrityError("root stage trajectory lock mismatch")
    outputs = descriptor.get("outputs") or {}
    _assert_file_lock(outputs.get("root_state") or {}, root_state_path, label="root_state")
    _assert_file_lock(outputs.get("c_tasks") or {}, tasks_path, label="c_tasks")

    states = _read_jsonl(root_state_path, label="root_state")
    tasks = _read_jsonl(tasks_path, label="c_tasks")
    if len(states) != 40:
        raise AuditIntegrityError(f"expected 40 root states, observed {len(states)}")
    state_by_key: dict[str, dict[str, Any]] = {}
    for state in states:
        _assert_gold_free(state, location="root_state")
        key = _question_key(state)
        if key in state_by_key:
            raise AuditIntegrityError(f"duplicate root state: {key}")
        state_by_key[key] = state
        if (
            state.get("schema_version") != ROOT_STATE_SCHEMA
            or state.get("runner_version") != RUNNER_VERSION
            or state.get("gold_access") is not False
            or state.get("completed_depth") != 1
        ):
            raise AuditIntegrityError(f"root state contract mismatch: {key}")
        question = state.get("question")
        if not isinstance(question, str) or sha256_text(question) != state.get(
            "question_sha256"
        ):
            raise AuditIntegrityError(f"root question/hash mismatch: {key}")
        plan = state.get("plan")
        if not isinstance(plan, Mapping) or sha256_json(plan) != state.get("plan_sha256"):
            raise AuditIntegrityError(f"root plan/hash mismatch: {key}")
        schedule = _recompute_schedule(plan)
        if schedule != state.get("schedule"):
            raise AuditIntegrityError(f"root schedule differs from frozen plan: {key}")
        maximum = max((int(row["dependency_depth"]) for row in schedule), default=1)
        if state.get("max_dependency_depth") != maximum:
            raise AuditIntegrityError(f"root maximum depth mismatch: {key}")
        expected_dependent = any(row["dependencies"] for row in schedule)
        if state.get("has_dependent_step") is not expected_dependent:
            raise AuditIntegrityError(f"root dependent-step flag mismatch: {key}")
        if state.get("plan_executable") is not True or state.get("plan_validation_errors") != []:
            raise AuditIntegrityError(f"retry-1 root plan unexpectedly non-executable: {key}")
        expected_status = "roots_complete" if expected_dependent else "fallback_no_dependent_step"
        if state.get("execution_status") != expected_status:
            raise AuditIntegrityError(f"root execution status mismatch: {key}")
        slots_b = state.get("slot_values_B")
        slots_c = state.get("slot_values_C")
        if not isinstance(slots_b, Mapping) or slots_c != {}:
            raise AuditIntegrityError(f"root slot state mismatch: {key}")
        if state.get("subanswer_telemetry") != []:
            raise AuditIntegrityError(f"root state already contains reader attempts: {key}")
        if state.get("run_locks") != runtime_locks:
            raise AuditIntegrityError(f"root embedded locks mismatch: {key}")

    by_dataset = Counter(str(state["dataset"]) for state in states)
    if dict(by_dataset) != EXPECTED_BY_DATASET:
        raise AuditIntegrityError(f"root dataset population mismatch: {dict(by_dataset)}")

    pending_tasks: list[dict[str, Any]] = []
    for state in states:
        pending = state.get("pending_c_tasks")
        if not isinstance(pending, list):
            raise AuditIntegrityError("root pending_c_tasks is not a list")
        pending_tasks.extend(dict(task) for task in pending)
    pending_tasks.sort(key=lambda row: (str(row["question_key"]), str(row["producer_slot"])))
    if tasks != pending_tasks:
        raise AuditIntegrityError("c_tasks file differs from root pending task union/order")

    task_ids: set[str] = set()
    for task in tasks:
        _assert_gold_free(task, location="c_task")
        if frozenset(task) != TASK_KEYS or task.get("gold_access") is not False:
            raise AuditIntegrityError("C task schema/Gold contract mismatch")
        task_id = str(task.get("task_id") or "")
        if not task_id or task_id in task_ids:
            raise AuditIntegrityError(f"duplicate/empty C task id: {task_id!r}")
        task_ids.add(task_id)
        key = _question_key(task)
        state = state_by_key.get(key)
        if state is None:
            raise AuditIntegrityError(f"C task has no root state: {key}")
        if task.get("question") != state.get("question"):
            raise AuditIntegrityError(f"C task question mismatch: {task_id}")
        if sha256_json(task.get("step")) != task.get("step_sha256"):
            raise AuditIntegrityError(f"C task step hash mismatch: {task_id}")
        if sha256_json(task.get("producer_passages")) != task.get(
            "producer_passages_sha256"
        ):
            raise AuditIntegrityError(f"C task producer passage hash mismatch: {task_id}")

    predictions_lock = (plan_lock.get("inputs") or {}).get("planner_predictions")
    if not isinstance(predictions_lock, Mapping):
        raise AuditIntegrityError("plan lock lacks planner predictions")
    predictions_path = Path(str(predictions_lock.get("path") or ""))
    _assert_file_lock(predictions_lock, predictions_path, label="planner predictions")
    predictions = _read_jsonl(predictions_path, label="planner predictions")
    prediction_by_key = _index_unique(
        predictions, field="question_key", label="planner predictions"
    )
    if set(prediction_by_key) != set(state_by_key):
        raise AuditIntegrityError("root/planner identity sets differ")
    for key, state in state_by_key.items():
        prediction = prediction_by_key[key]
        _assert_gold_free(prediction, location=f"planner_prediction.{key}")
        if sha256_json(prediction) != state.get("plan_row_sha256"):
            raise AuditIntegrityError(f"planner-row hash mismatch: {key}")
        if _extract_plan(prediction) != state.get("plan"):
            raise AuditIntegrityError(f"root plan differs from planner prediction: {key}")
        for field in ("dataset", "qid", "question", "question_sha256"):
            if prediction.get(field) != state.get(field):
                raise AuditIntegrityError(f"root/planner {field} mismatch: {key}")
    return states, tasks, descriptor


def _recompute_generator_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        selected = [row for row in rows if row["dataset"] == dataset]
        strict = sum(bool(row["telemetry"]["strict_parse"]["valid"]) for row in selected)
        verified = sum(bool(row["verified"]) for row in selected)
        reasons = Counter(str(row["telemetry"]["verification"]["reason"]) for row in selected)
        by_dataset[dataset] = {
            "tasks": len(selected),
            "strict_parse_valid": strict,
            "strict_parse_rate": strict / max(1, len(selected)),
            "mechanically_verified": verified,
            "mechanically_verified_rate": verified / max(1, len(selected)),
            "verification_reasons": dict(sorted(reasons.items())),
        }
    return {
        "tasks": len(rows),
        "strict_parse_valid": sum(
            bool(row["telemetry"]["strict_parse"]["valid"]) for row in rows
        ),
        "mechanically_verified": sum(bool(row["verified"]) for row in rows),
        "by_dataset": by_dataset,
    }


def _validate_subanswer_artifacts(
    *,
    tasks: Sequence[Mapping[str, Any]],
    tasks_path: Path,
    subanswers_path: Path,
    report_path: Path,
    manifest_path: Path,
    roots_stage_path: Path,
    implementation: Mapping[str, Any],
    implementation_file_lock: Mapping[str, Any],
    plan_file_lock: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows = _read_jsonl(subanswers_path, label="subanswers")
    report = _read_json(report_path, label="generator report")
    manifest = _read_json(manifest_path, label="generator manifest")
    for label, value in (("subanswers", rows), ("generator_report", report), ("generator_manifest", manifest)):
        _assert_gold_free(value, location=label)
    if (
        report.get("schema_version") != GENERATOR_REPORT_SCHEMA
        or report.get("runner_version") != GENERATOR_VERSION
        or report.get("status") != "COMPLETE_GOLD_FREE_SUBANSWERS"
        or report.get("gold_access") is not False
        or report.get("network_access") is not False
    ):
        raise AuditIntegrityError("generator report contract mismatch")
    if report.get("generation") != {
        "decode": "greedy",
        "do_sample": False,
        "max_new_tokens": 96,
        "retry_count": 0,
        "seed": 42,
        "torch_dtype": "bfloat16",
    }:
        raise AuditIntegrityError("generator decoding contract mismatch")
    tasks_lock = file_lock(tasks_path)
    output_lock = file_lock(subanswers_path)
    roots_stage_lock = file_lock(roots_stage_path)
    report_lock = file_lock(report_path)
    if report.get("input") != {**tasks_lock, "rows": len(tasks)}:
        # Generator report intentionally omitted size_bytes from this compact
        # input commitment.  Compare its exact historical schema explicitly.
        expected_input = {
            "path": tasks_lock["path"],
            "sha256": tasks_lock["sha256"],
            "rows": len(tasks),
        }
        if report.get("input") != expected_input:
            raise AuditIntegrityError("generator report input lock/count mismatch")
    expected_output = {
        "path": output_lock["path"],
        "sha256": output_lock["sha256"],
        "rows": len(rows),
    }
    if report.get("output") != expected_output:
        raise AuditIntegrityError("generator report output lock/count mismatch")
    if len(rows) != len(tasks):
        raise AuditIntegrityError("subanswer/task cardinality mismatch")
    expected_authorization = {
        "implementation_lock": dict(implementation_file_lock),
        "plan_lock": dict(plan_file_lock),
        "producer_stage_descriptor": roots_stage_lock,
    }
    if report.get("authorization_locks") != expected_authorization:
        raise AuditIntegrityError("generator report authorization locks mismatch")

    verified_models = (
        (implementation.get("content_reverification") or {})
        .get("verified", {})
        .get("models", {})
    )
    expected_model_artifact = {
        "schema_version": GENERATOR_MODEL_SCHEMA,
        "base_model": verified_models.get("base_model"),
        "strong_sft_adapter": verified_models.get("strong_sft"),
        "load_contract": {
            "torch_dtype": "bfloat16",
            "device": "cuda:0",
            "local_files_only": True,
            "separate_process_required": True,
        },
        "authorization_locks": {
            "implementation_lock": dict(implementation_file_lock),
            "plan_lock": dict(plan_file_lock),
        },
    }
    if not isinstance(verified_models.get("base_model"), Mapping) or not isinstance(
        verified_models.get("strong_sft"), Mapping
    ):
        raise AuditIntegrityError("implementation lock lacks base/strong-SFT locks")
    if report.get("model_artifact") != expected_model_artifact:
        raise AuditIntegrityError("generator model artifact differs from implementation lock")

    manifest_run = manifest.get("run")
    if (
        manifest.get("status") != "COMPLETE_GOLD_FREE_SUBANSWERS"
        or not isinstance(manifest_run, Mapping)
        or manifest_run.get("experiment_id") != report.get("experiment_id")
        or manifest_run.get("phase") != "dependent_retrieval_v7_grounded_subanswer_generation"
        or manifest_run.get("runner_version") != GENERATOR_VERSION
        or manifest_run.get("input_tasks_sha256") != tasks_lock["sha256"]
        or manifest_run.get("output_sha256") != output_lock["sha256"]
        or manifest_run.get("report_sha256") != report_lock["sha256"]
        or manifest_run.get("authorization_locks") != expected_authorization
        or manifest_run.get("gold_access") is not False
        or manifest_run.get("network_access") is not False
    ):
        raise AuditIntegrityError("generator manifest nested run metadata mismatch")

    task_by_id = _index_unique(tasks, field="task_id", label="C tasks")
    answer_by_id = _index_unique(rows, field="task_id", label="subanswers")
    if list(answer_by_id) != list(task_by_id) or set(answer_by_id) != set(task_by_id):
        raise AuditIntegrityError("subanswer task identities/order differ")

    # Re-run the frozen deterministic parser/verifier after authenticating its
    # implementation-lock hash.  This does not load a model or any Gold.
    import_closure = implementation.get("actual_local_import_closure") or {}
    if isinstance(import_closure, Mapping):
        import_lock = import_closure.get("kgproweight/retrieval/subanswer_v7.py")
    elif isinstance(import_closure, list):
        # Backwards-compatible reader for early development lock drafts.  The
        # retry-1 production lock uses the mapping form above.
        import_lock = next(
            (
                row
                for row in import_closure
                if isinstance(row, Mapping)
                and str(row.get("path") or "").endswith(
                    "kgproweight/retrieval/subanswer_v7.py"
                )
            ),
            None,
        )
    else:
        import_lock = None
    verifier_path = PROJECT_ROOT / "kgproweight/retrieval/subanswer_v7.py"
    if not isinstance(import_lock, Mapping):
        raise AuditIntegrityError("implementation lock lacks subanswer verifier source lock")
    _assert_file_lock(import_lock, verifier_path, label="subanswer verifier source")
    from kgproweight.retrieval.subanswer_v7 import parse_and_verify_subanswer

    for task, row in zip(tasks, rows):
        if frozenset(row) != ANSWER_KEYS or row.get("gold_access") is not False:
            raise AuditIntegrityError("subanswer outer schema/Gold contract mismatch")
        for field in IDENTITY_FIELDS:
            if row.get(field) != task.get(field):
                raise AuditIntegrityError(
                    f"subanswer/task identity mismatch {task['task_id']}::{field}"
                )
        telemetry = row.get("telemetry")
        if not isinstance(telemetry, Mapping):
            raise AuditIntegrityError("subanswer telemetry is not an object")
        raw_response = telemetry.get("raw_response")
        if not isinstance(raw_response, str) or telemetry.get(
            "raw_response_sha256"
        ) != sha256_text(raw_response):
            raise AuditIntegrityError("subanswer raw-response hash mismatch")
        if telemetry.get("input_file_sha256") != tasks_lock["sha256"]:
            raise AuditIntegrityError("subanswer input-file hash mismatch")
        if telemetry.get("input_task_sha256") != sha256_json(task):
            raise AuditIntegrityError("subanswer input-task hash mismatch")
        if telemetry.get("producer_passage_count") != len(task["producer_passages"]):
            raise AuditIntegrityError("subanswer producer passage count mismatch")
        if any(
            telemetry.get(field) != task["producer_passages_sha256"]
            for field in (
                "prompt_passages_sha256",
                "verifier_passages_sha256",
            )
        ) or telemetry.get("same_passage_bytes_for_prompt_and_verifier") is not True:
            raise AuditIntegrityError("prompt/verifier producer passage commitment mismatch")
        if telemetry.get("model_artifact") != expected_model_artifact:
            raise AuditIntegrityError("row-level generator model lock mismatch")
        recomputed_verification = parse_and_verify_subanswer(
            raw_response,
            task["question"],
            task["step"],
            task["producer_passages"],
            target_type=task["target_type"],
        )
        if recomputed_verification != telemetry.get("verification"):
            raise AuditIntegrityError("mechanical subanswer verification drift")
        if bool(row.get("verified")) != bool(recomputed_verification.get("verified")):
            raise AuditIntegrityError("subanswer verified flag drift")
        if row.get("verified_answer") != recomputed_verification.get("verified_answer"):
            raise AuditIntegrityError("subanswer promoted value drift")

    counts = _recompute_generator_counts(rows)
    if counts != report.get("counts"):
        raise AuditIntegrityError("generator report counts differ from subanswers")
    return rows, report, manifest


def question_upper_bound(
    state: Mapping[str, Any],
    *,
    task_by_slot: Mapping[str, Mapping[str, Any]],
    answer_by_slot: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute one question's no-retry optimistic future reader bound."""

    schedule = list(state["schedule"])
    root_producer_slots = {
        str(record["slot"])
        for record in schedule
        if int(record["dependency_depth"]) == 1 and record["consumers"]
    }
    if set(task_by_slot) != root_producer_slots:
        raise AuditIntegrityError(
            f"root reader-task slots differ from producer roots: {state['question_key']}"
        )
    if set(answer_by_slot) != set(task_by_slot):
        raise AuditIntegrityError(
            f"root answer/task slots differ: {state['question_key']}"
        )

    current_attempts = len(task_by_slot)
    current_verified_slots = {
        slot for slot, row in answer_by_slot.items() if bool(row.get("verified"))
    }
    current_verified = len(current_verified_slots)
    reachable_b = {
        _normalise_slot(slot)
        for slot, value in (state.get("slot_values_B") or {}).items()
        if isinstance(value, str) and value.strip()
    }
    reachable_c = set(current_verified_slots)

    future_attempt_slots: list[str] = []
    future_reachable_dependent_hops: list[str] = []
    blocked: list[dict[str, Any]] = []
    for record in schedule:
        if int(record["dependency_depth"]) <= 1:
            continue
        dependencies = set(record["dependencies"])
        missing_b = sorted(dependencies - reachable_b)
        missing_c = sorted(dependencies - reachable_c)
        if missing_b or missing_c:
            blocked.append(
                {
                    "slot": record["slot"],
                    "dependency_depth": record["dependency_depth"],
                    "missing_B": missing_b,
                    "missing_C": missing_c,
                }
            )
            continue
        future_reachable_dependent_hops.append(str(record["slot"]))
        # The frozen runner asks the reader only for producer steps.  Terminal
        # steps are retrieval attempts but do not create another slot/task.
        if record["consumers"]:
            future_attempt_slots.append(str(record["slot"]))
            reachable_b.add(str(record["slot"]))
            reachable_c.add(str(record["slot"]))

    future_attempts_max = len(future_attempt_slots)
    future_verified_max = future_attempts_max
    final_attempts_max = current_attempts + future_attempts_max
    final_verified_max = current_verified + future_verified_max
    rate = final_verified_max / final_attempts_max if final_attempts_max else None
    return {
        "question_key": state["question_key"],
        "dataset": state["dataset"],
        "qid": state["qid"],
        "current_root_reader_attempts": current_attempts,
        "current_verified_root_slots": current_verified,
        "current_verified_root_slot_ids": sorted(current_verified_slots),
        "current_B_root_slot_ids": sorted(reachable_b & root_producer_slots),
        "future_reachable_dependent_hops_max": len(future_reachable_dependent_hops),
        "future_reachable_dependent_hop_ids": future_reachable_dependent_hops,
        "future_reader_attempts_max": future_attempts_max,
        "future_reader_attempt_slot_ids": future_attempt_slots,
        "future_mechanically_verified_max": future_verified_max,
        "final_reader_attempts_max": final_attempts_max,
        "final_mechanically_verified_max": final_verified_max,
        "mechanically_verified_rate_upper_bound": rate,
        "blocked_dependent_steps_even_under_future_success": blocked,
        "gold_access": False,
    }


def compute_upper_bounds(
    states: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    answers: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    state_by_key = {_question_key(row): row for row in states}
    tasks_by_question: dict[str, dict[str, Mapping[str, Any]]] = {
        key: {} for key in state_by_key
    }
    answers_by_question: dict[str, dict[str, Mapping[str, Any]]] = {
        key: {} for key in state_by_key
    }
    for task in tasks:
        key = _question_key(task)
        slot = _normalise_slot(task["producer_slot"])
        if slot in tasks_by_question[key]:
            raise AuditIntegrityError(f"duplicate task producer slot: {key}::{slot}")
        tasks_by_question[key][slot] = task
    for answer in answers:
        key = _question_key(answer)
        slot = _normalise_slot(answer["producer_slot"])
        if slot in answers_by_question[key]:
            raise AuditIntegrityError(f"duplicate answer producer slot: {key}::{slot}")
        answers_by_question[key][slot] = answer

    details = [
        question_upper_bound(
            state,
            task_by_slot=tasks_by_question[_question_key(state)],
            answer_by_slot=answers_by_question[_question_key(state)],
        )
        for state in states
    ]
    by_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        selected = [row for row in details if row["dataset"] == dataset]
        current_attempts = sum(int(row["current_root_reader_attempts"]) for row in selected)
        current_verified = sum(int(row["current_verified_root_slots"]) for row in selected)
        future_attempts = sum(int(row["future_reader_attempts_max"]) for row in selected)
        future_verified = sum(int(row["future_mechanically_verified_max"]) for row in selected)
        final_attempts = current_attempts + future_attempts
        final_verified = current_verified + future_verified
        upper_bound = final_verified / final_attempts if final_attempts else None
        by_dataset[dataset] = {
            "questions": len(selected),
            "current_root_reader_attempts": current_attempts,
            "current_mechanically_verified": current_verified,
            "current_mechanically_verified_rate": (
                current_verified / current_attempts if current_attempts else None
            ),
            "future_reachable_dependent_hops_max": sum(
                int(row["future_reachable_dependent_hops_max"]) for row in selected
            ),
            "future_reader_attempts_max": future_attempts,
            "future_mechanically_verified_max": future_verified,
            "final_reader_attempts_max": final_attempts,
            "final_mechanically_verified_max": final_verified,
            "mechanically_verified_rate_upper_bound": upper_bound,
            "frozen_minimum": threshold,
            "upper_bound_reaches_gate": (
                upper_bound is not None and upper_bound >= threshold
            ),
        }
    status = (
        CONTINUE_STATUS
        if all(row["upper_bound_reaches_gate"] for row in by_dataset.values())
        else FAIL_STATUS
    )
    return details, by_dataset, status


def _diagnostic_history() -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = [
        {
            "sequence": 1,
            "phase": "plan_lock_preflight",
            "exception": "ValueError: planner manifest Experiment ID drift",
            "code_location": (
                "scripts/prepare/freeze_dependent_retrieval_v7_plans.py::"
                "_validate_planner_report"
            ),
            "evidence_basis": "operator_supplied_unverified_no_persisted_log",
            "independently_verified": False,
            "stage_output_created": False,
            "used_in_upper_bound": False,
        },
        {
            "sequence": 2,
            "phase": "plan_lock_preflight",
            "exception": "ValueError: planner manifest Gold boundary drift",
            "code_location": (
                "scripts/prepare/freeze_dependent_retrieval_v7_plans.py::"
                "_validate_planner_report"
            ),
            "evidence_basis": "operator_supplied_unverified_no_persisted_log",
            "independently_verified": False,
            "stage_output_created": False,
            "used_in_upper_bound": False,
        },
    ]
    persisted = (
        (
            3,
            "root_materialization_pre_output",
            ROOT_FAILURE_LOG,
            "V7IntegrityError: v7 query-variant cap differs from the frozen design",
            PROJECT_ROOT
            / "outputs/validation/subquestion_dependent_retrieval_v7_development_materialization_v1",
        ),
        (
            4,
            "depth1_dependent_materialization_post_compute_pre_output",
            DEPTH1_FAILURE_LOG,
            (
                "V7IntegrityError: non-dependent plan has inconsistent status for "
                "hotpotqa::dev_5473"
            ),
            PROJECT_ROOT
            / "outputs/validation/subquestion_dependent_retrieval_v7_development_"
            "materialization_depth1_v1_retry1",
        ),
    )
    for sequence, phase, relative_log, exception, absent_output in persisted:
        log_path = (PROJECT_ROOT / relative_log).resolve()
        _assert_safe_input_path(log_path, label=f"diagnostic log {sequence}")
        text = log_path.read_text(encoding="utf-8")
        if exception not in text:
            raise AuditIntegrityError(
                f"diagnostic log {sequence} lacks its recorded exception"
            )
        if absent_output.exists():
            raise AuditIntegrityError(
                f"failed diagnostic stage unexpectedly has an output: {absent_output}"
            )
        history.append(
            {
                "sequence": sequence,
                "phase": phase,
                "exception": exception,
                "evidence_basis": "verified_from_persisted_log",
                "independently_verified": True,
                "log": file_lock(log_path),
                "expected_stage_output_path": str(absent_output.resolve()),
                "stage_output_created": False,
                "used_in_upper_bound": False,
            }
        )
    return history


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        handle.write("\n")


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def run_audit(
    *,
    root_state_path: Path,
    tasks_path: Path,
    roots_stage_path: Path,
    subanswers_path: Path,
    generator_report_path: Path,
    generator_manifest_path: Path,
    implementation_lock_path: Path,
    plan_lock_path: Path,
    trajectory_lock_path: Path,
    output_dir: Path,
    experiment_id: str,
) -> dict[str, Any]:
    paths = [
        root_state_path,
        tasks_path,
        roots_stage_path,
        subanswers_path,
        generator_report_path,
        generator_manifest_path,
        implementation_lock_path,
        plan_lock_path,
        trajectory_lock_path,
    ]
    paths = [(PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve() for path in paths]
    (
        root_state_path,
        tasks_path,
        roots_stage_path,
        subanswers_path,
        generator_report_path,
        generator_manifest_path,
        implementation_lock_path,
        plan_lock_path,
        trajectory_lock_path,
    ) = paths

    implementation, plan_lock, trajectory, prereg = _validate_lock_chain(
        implementation_path=implementation_lock_path,
        plan_path=plan_lock_path,
        trajectory_path=trajectory_lock_path,
    )
    implementation_file_lock = file_lock(implementation_lock_path)
    plan_file_lock = file_lock(plan_lock_path)
    trajectory_file_lock = file_lock(trajectory_lock_path)
    states, tasks, descriptor = _validate_root_artifacts(
        root_state_path=root_state_path,
        tasks_path=tasks_path,
        roots_stage_path=roots_stage_path,
        implementation_file_lock=implementation_file_lock,
        plan_file_lock=plan_file_lock,
        trajectory_file_lock=trajectory_file_lock,
        plan_lock=plan_lock,
    )
    answers, generator_report, generator_manifest = _validate_subanswer_artifacts(
        tasks=tasks,
        tasks_path=tasks_path,
        subanswers_path=subanswers_path,
        report_path=generator_report_path,
        manifest_path=generator_manifest_path,
        roots_stage_path=roots_stage_path,
        implementation=implementation,
        implementation_file_lock=implementation_file_lock,
        plan_file_lock=plan_file_lock,
    )
    threshold = float(
        prereg["decision_gates"]["gold_free_mechanism"]
        ["mechanically_verified_subanswer_rate_min_each_dataset"]
    )
    details, by_dataset, status = compute_upper_bounds(
        states, tasks, answers, threshold=threshold
    )
    diagnostics = _diagnostic_history()

    output_dir = (
        (PROJECT_ROOT / output_dir).resolve()
        if not output_dir.is_absolute()
        else output_dir.resolve()
    )
    try:
        output_dir.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise AuditIntegrityError("output directory is outside the project root") from exc
    output_dir.mkdir(parents=True, exist_ok=False)
    details_path = output_dir / "per_question_upper_bound.jsonl"
    report_path = output_dir / "report.json"
    manifest_path = output_dir / "manifest.json"
    _write_jsonl_exclusive(details_path, details)

    prereg_path = Path(str((plan_lock["parents"])["preregistration"]["path"]))
    report = {
        "schema_version": "subquestion-dependent-retrieval-v7-depth1-upper-bound-1",
        "experiment_id": experiment_id,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": SCOPE,
        "development_only": True,
        "gold_access": False,
        "gpu_calls": 0,
        "retrieval_calls": 0,
        "model_calls": 0,
        "input_locks": {
            "root_state": file_lock(root_state_path),
            "c_tasks_depth1": file_lock(tasks_path),
            "roots_stage": file_lock(roots_stage_path),
            "subanswers_depth1": file_lock(subanswers_path),
            "generator_report": file_lock(generator_report_path),
            "generator_manifest": file_lock(generator_manifest_path),
            "implementation_lock": implementation_file_lock,
            "plan_lock": plan_file_lock,
            "trajectory_lock": trajectory_file_lock,
            "preregistration": file_lock(prereg_path),
        },
        "authenticated_contracts": {
            "root_runner_version": descriptor["runner_version"],
            "generator_version": generator_report["runner_version"],
            "generator_experiment_id": generator_report["experiment_id"],
            "generator_manifest_nested_run_metadata_verified": True,
            "generator_report_counts_recomputed": True,
            "mechanical_verifier_reexecuted": True,
            "generator_model_lock_exact_match": True,
            "model_bytes_rehashed_by_this_audit": False,
            "model_identity_basis": (
                "exact equality to the implementation lock's prior full content "
                "reverification plus exact row/report/manifest propagation"
            ),
            "plan_dependency_graph_recomputed": True,
            "root_task_answer_identity_join_rate": 1.0,
        },
        "upper_bound_definition": {
            "frozen_gate": threshold,
            "current_fixed_failures": (
                "depth-1 root reader attempts that were not mechanically verified are "
                "permanent under the frozen retry_count=0 trajectory"
            ),
            "optimistic_future_assumptions": [
                "every dependency-reachable future paired retrieval succeeds",
                "every future B entity extraction succeeds",
                "every future C subanswer is mechanically verified",
                "every future producer step with a consumer emits exactly one reader task",
            ],
            "denominator": (
                "all observed depth-1 reader attempts plus every maximally reachable "
                "future producer reader attempt"
            ),
            "numerator": (
                "observed mechanically verified depth-1 attempts plus all maximally "
                "reachable future reader attempts"
            ),
            "why_monotonic": (
                "future attempts are assigned success=1; adding such attempts cannot "
                "decrease a rate whose current value is at most 1"
            ),
            "terminal_retrieval_steps": (
                "reachable terminal steps count as dependent retrieval hops but not as "
                "reader attempts because the frozen runner creates C tasks only for steps "
                "with downstream consumers"
            ),
        },
        "by_dataset": by_dataset,
        "gate_decision": {
            "rule": "stop before Gold if any dataset upper bound is below 0.40",
            "all_dataset_upper_bounds_reach_gate": all(
                row["upper_bound_reaches_gate"] for row in by_dataset.values()
            ),
            "status": status,
            "gold_attachment_authorized": False,
            "answer_evaluation_authorized": False,
            "continue_retrieval_authorized": status != FAIL_STATUS,
        },
        "diagnostic_history": diagnostics,
        "diagnostic_history_boundary": (
            "Diagnostics are provenance only and are excluded from every upper-bound "
            "count. The first two entries have no persisted evidence and remain explicitly "
            "operator-supplied/unverified."
        ),
        "outputs": {"per_question_upper_bound": file_lock(details_path)},
        "scientific_boundary": (
            "Gold-free development-only early-stop audit. It reports no EM, F1, answer "
            "utility, confirmation result, training result, or paper-level claim."
        ),
    }
    _assert_gold_free(report, location="output_report")
    _write_json_exclusive(report_path, report)
    manifest = {
        "schema_version": "subquestion-dependent-retrieval-v7-depth1-upper-bound-manifest-1",
        "experiment_id": experiment_id,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_access": False,
        "gpu_calls": 0,
        "retrieval_calls": 0,
        "model_calls": 0,
        "report": file_lock(report_path),
        "per_question_upper_bound": file_lock(details_path),
        "audit_code": file_lock(Path(__file__).resolve()),
    }
    _assert_gold_free(manifest, location="output_manifest")
    _write_json_exclusive(manifest_path, manifest)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_state", type=Path, default=DEFAULT_ROOT_STATE)
    parser.add_argument("--c_tasks", type=Path, default=DEFAULT_C_TASKS)
    parser.add_argument("--roots_stage", type=Path, default=DEFAULT_ROOTS_STAGE)
    parser.add_argument("--subanswers", type=Path, default=DEFAULT_SUBANSWERS)
    parser.add_argument("--generator_report", type=Path, default=DEFAULT_GENERATOR_REPORT)
    parser.add_argument("--generator_manifest", type=Path, default=DEFAULT_GENERATOR_MANIFEST)
    parser.add_argument("--implementation_lock", type=Path, default=DEFAULT_IMPLEMENTATION_LOCK)
    parser.add_argument("--plan_lock", type=Path, default=DEFAULT_PLAN_LOCK)
    parser.add_argument("--trajectory_lock", type=Path, default=DEFAULT_TRAJECTORY_LOCK)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment_id", default=DEFAULT_EXPERIMENT_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_audit(
        root_state_path=args.root_state,
        tasks_path=args.c_tasks,
        roots_stage_path=args.roots_stage,
        subanswers_path=args.subanswers,
        generator_report_path=args.generator_report,
        generator_manifest_path=args.generator_manifest,
        implementation_lock_path=args.implementation_lock,
        plan_lock_path=args.plan_lock,
        trajectory_lock_path=args.trajectory_lock,
        output_dir=args.output_dir,
        experiment_id=str(args.experiment_id),
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
