#!/usr/bin/env python
"""Run the frozen v9.1 evaluation-time decomposition smoke or fresh pilot.

The v9.1 binder changes one scientific variable relative to the v9 phase-0
interface: a canonical subanswer surface occurring in several q1 documents is
bound to the smallest retrieval rank instead of being rejected.  The original
v8 q1/q2 controller, retrieval, merge, and final-reader algorithms are reused
without modifying their frozen source files.  Because phase 0 was a posthoc
diagnostic on consumed rows, this fresh full-chain run validates the integrated
v9.1 mechanism and is not a binder-only outcome experiment.

The public CLI exposes only two fixed scopes.  It cannot accept a cohort path,
Gold path, model path, output path, or prospective unlock switch.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgproweight.retrieval.canonical_subqa_v9_1 import (  # noqa: E402
    PROVENANCE_BINDER_VERSION as V91_BINDER_VERSION,
    SELECTION_POLICY as V91_SELECTION_POLICY,
    VERIFICATION_SCOPE as V91_VERIFICATION_SCOPE,
    build_canonical_subqa_messages,
    parse_and_bind_canonical_subanswer,
    project_rank_first_binding_for_runtime,
)
from kgproweight.retrieval.dynamic_decomposition_v8 import (  # noqa: E402
    PROVENANCE_BINDER_VERSION as V8_BINDER_VERSION,
)
from scripts.pilot import materialize_dynamic_decomposition_v8 as v8_engine  # noqa: E402
from scripts.pilot import run_dynamic_decomposition_v8 as v8_driver  # noqa: E402


RUNNER_VERSION = "dynamic-decomposition-v9.1-rank-first-canonical-subqa-runner-1"
PROTOCOL_EXPERIMENT_ID = (
    "SUBQUESTION-DECOMPOSITION-V9.1-RANK-FIRST-PILOT30X3-SEED42-PROTOCOL-V1"
)
PROTOCOL_DIR = Path(
    "outputs/audits/"
    "subquestion_decomposition_v9_1_rank_first_pilot30x3_protocol_v1"
)
PROTOCOL_PATH = PROTOCOL_DIR / "protocol.json"
PROTOCOL_MANIFEST_PATH = PROTOCOL_DIR / "manifest.json"
SMOKE_RUN_DIR = Path(
    "outputs/audits/"
    "subquestion_decomposition_v9_1_rank_first_engineering_smoke4x3_seed42_attempt001"
)
PILOT_RUN_DIR = Path(
    "outputs/audits/"
    "subquestion_decomposition_v9_1_rank_first_fresh_pilot30x3_seed42_attempt001"
)
SMOKE_EXPERIMENT_ID = (
    "SUBQUESTION-DECOMPOSITION-V9.1-RANK-FIRST-ENGINEERING-SMOKE4X3-SEED42-ATTEMPT001"
)
PILOT_EXPERIMENT_ID = (
    "SUBQUESTION-DECOMPOSITION-V9.1-RANK-FIRST-FRESH-PILOT30X3-SEED42-ATTEMPT001"
)
SCOPES = ("smoke", "pilot")
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
ARM_A = v8_engine.ARM_A
ARM_B = v8_engine.ARM_B
ARM_C = v8_engine.ARM_C
ALL_ARMS = v8_engine.ALL_ARMS
ARMS = v8_engine.ARMS
Q1_VALID_RATE_MIN = 0.95
CANONICAL_PARSE_RATE_MIN = 0.95
CANONICAL_TRACE_RATE_MIN = 0.95
A1_ADMISSIBLE_RATE_MIN = 0.40
B_Q2_VALID_RATE_MIN = 0.90
C_DYNAMIC_ITT_RATE_MIN = 0.32
INVALID_QUERY_RATE_MAX = 0.05

_ORIGINAL_BUILD_SUBANSWER_MESSAGES = v8_engine.build_subanswer_reader_messages
_ORIGINAL_PARSE_AND_BIND = v8_engine.parse_and_bind_subanswer
_ORIGINAL_BUILD_DYNAMIC_STATE = v8_engine.build_dynamic_q2_state
_ORIGINAL_BUILD_DYNAMIC_ACTION = v8_engine.build_dynamic_q2_action
_ORIGINAL_MERGE = v8_engine.merge_fixed_budget_passages


class V91RunnerError(RuntimeError):
    """A frozen v9.1 input, runtime, or output violates its contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_lock(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V91RunnerError(f"expected JSON object: {path}")
    return value


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write(_canonical_json_bytes(dict(value)))


def _write_rows_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"append-only output already exists: {path}")
    with path.open("xb") as handle:
        for row in rows:
            handle.write(_canonical_json_bytes(dict(row)))


def _assert_file_lock(lock: Mapping[str, Any], *, label: str) -> None:
    path = Path(str(lock.get("path") or "")).resolve()
    current = _file_lock(path)
    if any(current[field] != lock.get(field) for field in ("size_bytes", "sha256")):
        raise V91RunnerError(f"{label} content drift")


def runtime_contract_v91() -> dict[str, Any]:
    """Return v8's frozen runtime contract plus the held-fixed v9 interface."""

    contract = deepcopy(v8_engine.runtime_contract())
    contract["runtime_version"] = RUNNER_VERSION
    contract["canonical_subanswer_override"] = {
        "prompt": "canonical_sft_qa_prompt_with_empty_kg",
        "backend_generation_role": "final_reader",
        "max_new_tokens": 512,
        "held_fixed_from_v9_phase0": True,
    }
    return contract


def expected_gates_v91() -> dict[str, Any]:
    """Freeze parent v8 gates and the two canonical-interface additions."""

    return {
        "smoke": {
            "all_three_arms_present_rate": 1.0,
            "cache_accounting_conservation_rate": 1.0,
            "final_10_unique_rate": 1.0,
            "full_index_passes_max": 3,
            "gold_or_prospective_access_count": 0,
            "logical_budget_exact_rate": 1.0,
            "logical_retrieval_requests": 84,
            "production_staged": True,
            "retrieval_batch_stage_order": ["root_all", "q1_all", "q2_BC_all"],
            "row_count": 12,
            "runtime_error_count": 0,
            "binder_version_rate": 1.0,
            "canonical_trace_integrity_rate": 1.0,
        },
        "development_gold_free": {
            "B_q2_static_valid_rate_min": B_Q2_VALID_RATE_MIN,
            "B_static_allowlist_rate": 1.0,
            "C_dynamic_state_binding_integrity_rate": 1.0,
            "C_dynamic_transition_rate_all_itt_min_each_dataset": C_DYNAMIC_ITT_RATE_MIN,
            "a1_admissible_rate_min_each_dataset": A1_ADMISSIBLE_RATE_MIN,
            "a1_ineligible_full_content_identity_rate": 1.0,
            "cache_accounting_conservation_rate": 1.0,
            "eligible_dynamic_invalid_original_Q_no_third_call_rate": 1.0,
            "empty_repeat_padding_query_rate_max": INVALID_QUERY_RATE_MAX,
            "final_10_unique_rate": 1.0,
            "full_index_passes_max": 3,
            "gold_or_forbidden_recursive_field_access_count": 0,
            "itt_cardinality": 90,
            "logical_B_C_budget_identity_rate": 1.0,
            "logical_ledger_exact_rate": 1.0,
            "logical_retrieval_requests": 630,
            "per_dataset": 30,
            "production_staged": True,
            "q1_schema_valid_rate_min_each_dataset": Q1_VALID_RATE_MIN,
            "retrieval_batch_stage_order": ["root_all", "q1_all", "q2_BC_all"],
            "root_and_q1_shared_byte_identity_rate": 1.0,
            "runtime_error_count": 0,
            "canonical_answer_parse_rate_min_each_dataset": CANONICAL_PARSE_RATE_MIN,
            "canonical_step_trace_rate_min_each_dataset": CANONICAL_TRACE_RATE_MIN,
            "binder_version_rate": 1.0,
            "canonical_trace_integrity_rate": 1.0,
        },
    }


def expected_run_registry_v91() -> dict[str, Any]:
    return {
        "smoke": {
            "experiment_id": SMOKE_EXPERIMENT_ID,
            "output_dir": str((PROJECT_ROOT / SMOKE_RUN_DIR).resolve()),
            "row_count": 12,
            "per_dataset": 4,
            "scientific_role": "consumed_engineering_only",
        },
        "pilot": {
            "experiment_id": PILOT_EXPERIMENT_ID,
            "output_dir": str((PROJECT_ROOT / PILOT_RUN_DIR).resolve()),
            "row_count": 90,
            "per_dataset": 30,
            "scientific_role": "fresh_train_side_family_disjoint_gold_free_pilot",
            "condition": "same_protocol_smoke_all_pass",
        },
    }


def _load_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = (PROJECT_ROOT / PROTOCOL_PATH).resolve()
    manifest_path = (PROJECT_ROOT / PROTOCOL_MANIFEST_PATH).resolve()
    protocol = _read_json(protocol_path)
    manifest = _read_json(manifest_path)
    if (
        protocol.get("schema_version") != "dynamic-decomposition-v9.1-protocol-1"
        or protocol.get("experiment_id") != PROTOCOL_EXPERIMENT_ID
        or protocol.get("status") != "AUTHORIZED_SMOKE_THEN_CONDITIONAL_FRESH_PILOT90_GOLD_FREE"
        or protocol.get("gold_access") is not False
        or protocol.get("answer_scoring") is not False
        or protocol.get("prospective_opened_or_hashed") is not False
        or not v8_driver.implementation.verify_self_commitment(
            protocol, field="protocol_body_canonical_sha256"
        )
        or protocol.get("gates") != expected_gates_v91()
        or protocol.get("runtime_contract") != runtime_contract_v91()
        or protocol.get("run_registry") != expected_run_registry_v91()
    ):
        raise V91RunnerError("v9.1 protocol identity/boundary mismatch")
    if (
        manifest.get("schema_version") != "dynamic-decomposition-v9.1-protocol-manifest-1"
        or manifest.get("experiment_id") != PROTOCOL_EXPERIMENT_ID
        or manifest.get("status") != protocol.get("status")
        or manifest.get("gold_access") is not False
        or manifest.get("prospective_opened_or_hashed") is not False
        or manifest.get("protocol") != _file_lock(protocol_path)
    ):
        raise V91RunnerError("v9.1 protocol manifest mismatch")
    expected_lock_names = {
        "code_locks": {
            "runner",
            "rank_first_binder",
            "canonical_subqa_v9",
            "v8_binder_and_policy",
            "v8_materializer",
            "v8_driver",
            "canonical_prompts",
            "canonical_parsers",
        },
        "cohort_locks": {"consumed_smoke12", "fresh_pilot90"},
        "parent_locks": {
            "v8_implementation_protocol",
            "v8_implementation_manifest",
            "v9_phase0_protocol",
            "v9_phase0_manifest",
            "v9_phase0_report",
            "fresh_pilot_freeze_protocol",
            "fresh_pilot_freeze_report",
            "fresh_pilot_freeze_manifest",
        },
    }
    for group_name, expected_names in expected_lock_names.items():
        group = protocol.get(group_name)
        if not isinstance(group, Mapping) or set(group) != expected_names:
            raise V91RunnerError(f"missing protocol lock group: {group_name}")
        for label, lock in group.items():
            _assert_file_lock(lock, label=f"{group_name}.{label}")
    cohorts = protocol.get("cohorts")
    if not isinstance(cohorts, Mapping) or set(cohorts) != set(SCOPES):
        raise V91RunnerError("protocol cohort registry mismatch")
    for scope, lock_name in (("smoke", "consumed_smoke12"), ("pilot", "fresh_pilot90")):
        spec = cohorts[scope]
        lock = protocol["cohort_locks"][lock_name]
        if (
            not isinstance(spec, Mapping)
            or spec.get("path") != lock.get("path")
            or spec.get("sha256") != lock.get("sha256")
            or spec.get("gold_access") is not False
            or spec.get("prospective_unlocked") is not False
            or spec.get("row_count") != expected_run_registry_v91()[scope]["row_count"]
        ):
            raise V91RunnerError(f"protocol {scope} cohort binding mismatch")
    return protocol, _file_lock(protocol_path)


def _load_fixed_cohort(protocol: Mapping[str, Any], *, scope: str) -> dict[str, Any]:
    if scope not in SCOPES:
        raise V91RunnerError(f"unsupported scope: {scope}")
    cohort_spec = (protocol.get("cohorts") or {}).get(scope)
    if not isinstance(cohort_spec, Mapping):
        raise V91RunnerError(f"protocol lacks {scope} cohort")
    if cohort_spec.get("gold_access") is not False or cohort_spec.get(
        "prospective_unlocked"
    ) is not False:
        raise V91RunnerError("cohort boundary mismatch")
    path = Path(str(cohort_spec.get("path") or "")).resolve()
    expected_sha = str(cohort_spec.get("sha256") or "")
    if _sha256_file(path) != expected_sha:
        raise V91RunnerError(f"{scope} cohort SHA mismatch")
    expected_n = 12 if scope == "smoke" else 90
    expected_per_dataset = 4 if scope == "smoke" else 30
    rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or not line.strip():
                raise V91RunnerError(f"cohort framing error at line {line_number}")
            raw = json.loads(line)
            if not isinstance(raw, dict) or tuple(raw) != ("dataset", "qid", "question"):
                raise V91RunnerError(f"cohort schema error at line {line_number}")
            if raw["dataset"] not in DATASETS or any(
                not isinstance(raw[field], str) or not raw[field].strip()
                for field in ("qid", "question")
            ):
                raise V91RunnerError(f"cohort identity error at line {line_number}")
            key = f"{raw['dataset']}::{raw['qid']}"
            if key in seen:
                raise V91RunnerError(f"duplicate cohort identity: {key}")
            seen.add(key)
            counts[raw["dataset"]] += 1
            rows.append({key: str(value) for key, value in raw.items()})
    if len(rows) != expected_n or dict(counts) != {
        dataset: expected_per_dataset for dataset in DATASETS
    }:
        raise V91RunnerError(f"{scope} cohort cardinality mismatch")
    return {
        "role": scope,
        "row_count": len(rows),
        "per_dataset_counts": dict(counts),
        "path": str(path),
        "sha256": expected_sha,
        "gold_access": False,
        "prospective_unlocked": False,
        "rows": rows,
    }


def _build_subanswer_messages_v91(
    *,
    original_question: str,
    q1_query: str,
    q1_passages: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    # The original question is intentionally not exposed to the single-hop
    # canonical reader.  It is accepted only to match the frozen engine hook.
    if not isinstance(original_question, str) or not original_question.strip():
        raise V91RunnerError("original question is malformed")
    return build_canonical_subqa_messages(
        subquestion=q1_query,
        retrieved_passages=q1_passages,
    )


def _parse_and_bind_v91(
    response_text: str,
    *,
    q1_query: str,
    q1_passages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    parsed = parse_and_bind_canonical_subanswer(
        response_text,
        subquestion=q1_query,
        retrieved_passages=q1_passages,
    )
    binding = deepcopy(parsed["binding"])
    binding["canonical_trace"] = {
        "schema_version": parsed["schema_version"],
        "parser_version": parsed["parser_version"],
        "gold_access": False,
        "generation": response_text,
        "generation_sha256": parsed["generation_sha256"],
        "final_answer_parsed": parsed["final_answer_parsed"],
        "final_answer": parsed["final_answer"],
        "final_answer_sha256": parsed["final_answer_sha256"],
        "parsed_step_count": parsed["parsed_step_count"],
        "has_step_trace": parsed["has_step_trace"],
    }
    return binding


def _build_dynamic_state_v91(
    *,
    original_question: str,
    q1_query: str,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    projection = project_rank_first_binding_for_runtime(binding)
    compatible = deepcopy(dict(projection))
    compatible["binder_version"] = V8_BINDER_VERSION
    return _ORIGINAL_BUILD_DYNAMIC_STATE(
        original_question=original_question,
        q1_query=q1_query,
        binding=compatible,
    )


def _build_dynamic_action_v91(
    response_text: str | None,
    *,
    original_question: str,
    q1_query: str,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    projection = project_rank_first_binding_for_runtime(binding)
    compatible = deepcopy(dict(projection))
    compatible["binder_version"] = V8_BINDER_VERSION
    return _ORIGINAL_BUILD_DYNAMIC_ACTION(
        response_text,
        original_question=original_question,
        q1_query=q1_query,
        binding=compatible,
    )


def _merge_v91(
    root_passages: Sequence[Mapping[str, Any]],
    q1_passages: Sequence[Mapping[str, Any]],
    q2_passages: Sequence[Mapping[str, Any]],
    *,
    root_query: str,
    q1_query: str,
    q2_query: str,
    q1_binding: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reuse the frozen merge after validating and privately adapting version.

    v8's merge reads only the selected-document provenance and guards the
    binder-version string.  v9.1 validates its new binding first, then changes
    that string only in a private compatibility copy.  The materialized binding
    remains v9.1 and the merge algorithm itself is byte-identical to v8.
    """

    compatible: Mapping[str, Any] | None = None
    if q1_binding is not None:
        if q1_binding.get("binder_version") != V91_BINDER_VERSION:
            raise V91RunnerError("merge received a non-v9.1 binding")
        compatible = deepcopy(dict(q1_binding))
        if q1_binding.get("verified") is True:
            project_rank_first_binding_for_runtime(q1_binding)
        compatible["binder_version"] = V8_BINDER_VERSION
    return _ORIGINAL_MERGE(
        root_passages,
        q1_passages,
        q2_passages,
        root_query=root_query,
        q1_query=q1_query,
        q2_query=q2_query,
        q1_binding=compatible,
    )


@contextmanager
def _patched_v91_engine():
    replacements = {
        "build_subanswer_reader_messages": _build_subanswer_messages_v91,
        "parse_and_bind_subanswer": _parse_and_bind_v91,
        "build_dynamic_q2_state": _build_dynamic_state_v91,
        "build_dynamic_q2_action": _build_dynamic_action_v91,
        "merge_fixed_budget_passages": _merge_v91,
    }
    saved = {name: getattr(v8_engine, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(v8_engine, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(v8_engine, name, value)


def materialize(
    rows: Sequence[Mapping[str, Any]],
    *,
    hf_runtime: Any,
    retriever_runtime: Any,
    scope: str,
    experiment_id: str,
    intended_output_dir: str,
    cohort_lock: Mapping[str, Any],
) -> dict[str, Any]:
    controller = v8_engine._CachedTextInvoker(
        hf_runtime.bind_role("controller"), label="controller"
    )
    # Canonical subanswers need the same 512-token role as ordinary final QA,
    # not v8's 96-token bespoke one-line reader role.
    reader = v8_engine._CachedTextInvoker(
        hf_runtime.bind_role("final_reader"), label="canonical_subanswer_reader"
    )
    final_reader = v8_engine._CachedTextInvoker(
        hf_runtime.bind_role("final_reader"), label="final_reader"
    )
    retriever = v8_engine._CachedRetriever(retriever_runtime)
    with _patched_v91_engine():
        outputs = v8_engine._run_complete_rows_staged(
            rows,
            controller=controller,
            subanswer_reader=reader,
            final_reader=final_reader,
            retriever=retriever,
        )
    if len(outputs) != len(rows):
        raise V91RunnerError("materialized row count mismatch")
    for row in outputs:
        row["schema_version"] = "dynamic-decomposition-v9.1-gold-free-row-1"
        row["runner_version"] = RUNNER_VERSION
        row["production_runtime_version"] = RUNNER_VERSION
    invokers = {
        "controller": controller,
        "subanswer_reader": reader,
        "final_reader": final_reader,
        "retrieval": retriever,
    }
    n = len(outputs)
    runtime_contract = runtime_contract_v91()
    expected_aggregate = {
        arm: {
            name: int(count) * n
            for name, count in runtime_contract["logical_budget_by_arm"][arm].items()
        }
        for arm in ALL_ARMS
    }
    observed_aggregate = {
        arm: {
            "retrieval": retriever.logical_calls[arm],
            "controller": controller.logical_calls[arm],
            "subanswer_reader": reader.logical_calls[arm],
            "final_reader": final_reader.logical_calls[arm],
        }
        for arm in ALL_ARMS
    }
    if observed_aggregate != expected_aggregate:
        raise V91RunnerError("aggregate A/B/C logical budget mismatch")
    aggregate_cache = {
        name: {
            "logical_requests": sum(invoker.logical_calls.values()),
            "cache_hits": invoker.logical_cache_hits,
            "cache_misses": invoker.logical_cache_misses,
            "physical_executions": invoker.physical_calls,
        }
        for name, invoker in invokers.items()
    }
    if any(
        values["logical_requests"]
        != values["cache_hits"] + values["cache_misses"]
        or values["physical_executions"] != values["cache_misses"]
        for values in aggregate_cache.values()
    ):
        raise V91RunnerError("aggregate A/B/C cache accounting mismatch")
    return {
        "schema_version": "dynamic-decomposition-v9.1-gold-free-run-1",
        "runner_version": RUNNER_VERSION,
        "production_runtime_version": RUNNER_VERSION,
        "scope": scope,
        "experiment_id": experiment_id,
        "intended_output_dir": intended_output_dir,
        "gold_access": False,
        "prospective_unlocked": False,
        "cohort_lock": deepcopy(dict(cohort_lock)),
        "runtime_contract": runtime_contract,
        "row_count": n,
        "logical_calls_by_arm": observed_aggregate,
        "rows": outputs,
        "joint_cache_accounting": aggregate_cache,
        "retrieval_batch_telemetry": {
            "backend_batch_invocations": retriever.backend_batch_invocations,
            "full_index_passes": retriever.full_index_passes,
            "unique_query_count_by_batch": list(retriever.batch_unique_query_counts),
            "stage_batches": deepcopy(retriever.stage_batch_telemetry),
        },
    }


def _document_identity(passage: Mapping[str, Any]) -> str:
    values = [
        str(passage[key])
        for key in ("document_key", "doc_id", "id", "document_id")
        if passage.get(key) is not None
    ]
    if not values:
        raise V91RunnerError("final passage has no identity")
    return values[0]


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def build_report(
    result: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    scope: str,
    experiment_id: str,
    rows_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Extend the frozen v8 Gold-free report with only v9/v9.1 gates.

    Reusing the frozen reporter is deliberate: logical budgets, staged
    retrieval, cache accounting, query-quality accounting, fail-closed
    counterfactual identity, and forbidden-field checks remain identical to
    v8.  v9.1 adds canonical parsing/trace and binder-version checks; it does
    not silently replace any parent gate.
    """

    rows = result.get("rows")
    if not isinstance(rows, list):
        raise V91RunnerError("result rows missing")
    base_scope = "smoke" if scope == "smoke" else "development"
    base = v8_driver.build_gold_free_mechanism_report(
        result=result,
        protocol=protocol,
        scope=base_scope,
        rows_lock=rows_lock,
        created_at_utc=_utc_now(),
    )

    counts: Counter[str] = Counter()
    answer_parsed: Counter[str] = Counter()
    trace_valid: Counter[str] = Counter()
    reasons: dict[str, Counter[str]] = {dataset: Counter() for dataset in DATASETS}
    binder_valid = 0
    trace_integrity = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("gold_access") is not False:
            raise V91RunnerError(f"row {index} violates Gold-free contract")
        identity = row.get("identity") or {}
        dataset = identity.get("dataset")
        if dataset not in DATASETS:
            raise V91RunnerError(f"row {index} dataset mismatch")
        counts[dataset] += 1
        shared = row.get("shared") or {}
        binding = shared.get("subanswer_binding") or {}
        trace = binding.get("canonical_trace") or {}
        if binding.get("binder_version") != V91_BINDER_VERSION:
            raise V91RunnerError(f"row {index} binding version mismatch")
        binder_valid += 1
        generation = trace.get("generation")
        final_answer = trace.get("final_answer")
        parsed = trace.get("final_answer_parsed") is True
        trace_ok = (
            isinstance(generation, str)
            and trace.get("generation_sha256")
            == hashlib.sha256(generation.encode("utf-8")).hexdigest()
            and parsed is (isinstance(final_answer, str) and bool(final_answer))
            and trace.get("final_answer_sha256")
            == (
                hashlib.sha256(final_answer.encode("utf-8")).hexdigest()
                if parsed
                else None
            )
            and type(trace.get("parsed_step_count")) is int
            and trace.get("parsed_step_count") >= 0
            and trace.get("has_step_trace")
            is (trace.get("parsed_step_count") > 0)
            and binding.get("verification_scope") == V91_VERIFICATION_SCOPE
            and binding.get("selection_policy") == V91_SELECTION_POLICY
        )
        if binding.get("verified") is True:
            trace_ok = (
                trace_ok
                and parsed
                and binding.get("verified_answer") == final_answer
                and type(binding.get("supporting_doc_rank")) is int
                and 1 <= binding.get("supporting_doc_rank") <= 10
            )
        trace_integrity += int(trace_ok)
        if trace.get("final_answer_parsed") is True:
            answer_parsed[dataset] += 1
        if trace.get("has_step_trace") is True:
            trace_valid[dataset] += 1
        reasons[dataset][str(binding.get("reason"))] += 1

    canonical = {
        dataset: {
            "n": counts[dataset],
            "canonical_answer_parse_rate": _rate(answer_parsed[dataset], counts[dataset]),
            "canonical_step_trace_rate": _rate(trace_valid[dataset], counts[dataset]),
            "binding_reason_counts": dict(sorted(reasons[dataset].items())),
        }
        for dataset in DATASETS
    }
    gates = dict(base["gate_results"])
    gates["binder_version_rate_1"] = binder_valid == len(rows)
    gates["canonical_trace_integrity_rate_1"] = trace_integrity == len(rows)
    if scope == "pilot":
        gates["canonical_answer_parse_rate_min_each_dataset"] = all(
            canonical[d]["canonical_answer_parse_rate"] >= CANONICAL_PARSE_RATE_MIN
            for d in DATASETS
        )
        gates["canonical_step_trace_rate_min_each_dataset"] = all(
            canonical[d]["canonical_step_trace_rate"] >= CANONICAL_TRACE_RATE_MIN
            for d in DATASETS
        )

    base.update(
        {
            "schema_version": "dynamic-decomposition-v9.1-gold-free-report-1",
            "runner_version": RUNNER_VERSION,
            "experiment_id": experiment_id,
            "scope": scope,
            "status": "PASS" if all(gates.values()) else "FAIL_STOP_GOLD_FREE_GATES",
            "answer_scoring_performed": False,
            "canonical_subanswer_metrics": canonical,
            "binder_version": V91_BINDER_VERSION,
            "binder_verification_scope": (
                "lexical_surface_locality_only_not_semantic_support"
            ),
            "gate_results": gates,
            "all_pass": all(gates.values()),
            "scientific_boundary": (
                "Gold-free fresh-pilot mechanism result only; no EM/F1/IHR claim. "
                "Rank-first binding proves lexical locality, not semantic support."
            ),
        }
    )
    return base


def _require_smoke_pass(protocol_lock: Mapping[str, Any]) -> None:
    run_dir = (PROJECT_ROOT / SMOKE_RUN_DIR).resolve()
    path = run_dir / "manifest.complete.json"
    running_path = run_dir / "manifest.running.json"
    if not path.is_file() or not running_path.is_file():
        raise V91RunnerError("fresh pilot requires a completed v9.1 smoke")
    manifest = _read_json(path)
    report_lock = manifest.get("report")
    rows_lock = manifest.get("rows")
    running_lock = manifest.get("running")
    if (
        manifest.get("schema_version")
        != "dynamic-decomposition-v9.1-terminal-manifest-1"
        or manifest.get("experiment_id") != SMOKE_EXPERIMENT_ID
        or manifest.get("status") != "PASS"
        or manifest.get("gold_access") is not False
        or manifest.get("answer_scoring_performed") is not False
        or manifest.get("prospective_opened_or_hashed") is not False
        or not isinstance(report_lock, Mapping)
        or not isinstance(rows_lock, Mapping)
        or not isinstance(running_lock, Mapping)
    ):
        raise V91RunnerError("v9.1 smoke terminal status is not PASS")
    _assert_file_lock(report_lock, label="smoke report")
    _assert_file_lock(rows_lock, label="smoke rows")
    _assert_file_lock(running_lock, label="smoke running manifest")
    if dict(running_lock) != _file_lock(running_path):
        raise V91RunnerError("v9.1 smoke running-manifest lock mismatch")
    running = _read_json(running_path)
    if (
        running.get("schema_version")
        != "dynamic-decomposition-v9.1-running-manifest-1"
        or running.get("experiment_id") != SMOKE_EXPERIMENT_ID
        or running.get("status") != "RUNNING_NEW_APPEND_ONLY_ATTEMPT"
        or running.get("protocol") != dict(protocol_lock)
        or running.get("gold_access") is not False
        or running.get("prospective_opened_or_hashed") is not False
    ):
        raise V91RunnerError("v9.1 smoke running manifest is invalid")
    report = _read_json(Path(str(report_lock["path"])))
    if (
        report.get("schema_version")
        != "dynamic-decomposition-v9.1-gold-free-report-1"
        or report.get("experiment_id") != SMOKE_EXPERIMENT_ID
        or report.get("status") != "PASS"
        or report.get("all_pass") is not True
        or report.get("scope") != "smoke"
        or report.get("gold_access") is not False
        or report.get("prospective_opened_or_hashed") is not False
        or report.get("rows") != dict(rows_lock)
        or (report.get("preflight") or {}).get("new_protocol")
        != dict(protocol_lock)
    ):
        raise V91RunnerError("v9.1 smoke report did not pass")
    if manifest.get("protocol_sha256") != protocol_lock.get("sha256"):
        raise V91RunnerError("v9.1 smoke used a different protocol")


def execute(scope: str) -> dict[str, Any]:
    if scope not in SCOPES:
        raise V91RunnerError(f"unsupported scope: {scope}")
    if Path.cwd().resolve() != PROJECT_ROOT.resolve():
        raise V91RunnerError(f"run from project root: {PROJECT_ROOT}")
    protocol, protocol_lock = _load_protocol()
    if scope == "pilot":
        _require_smoke_pass(protocol_lock)
    cohort = _load_fixed_cohort(protocol, scope=scope)
    output_dir = (PROJECT_ROOT / (SMOKE_RUN_DIR if scope == "smoke" else PILOT_RUN_DIR)).resolve()
    experiment_id = SMOKE_EXPERIMENT_ID if scope == "smoke" else PILOT_EXPERIMENT_ID
    if output_dir.exists():
        raise FileExistsError(f"append-only run directory exists: {output_dir}")
    output_dir.mkdir(parents=True)
    _write_json_new(
        output_dir / "manifest.running.json",
        {
            "schema_version": "dynamic-decomposition-v9.1-running-manifest-1",
            "experiment_id": experiment_id,
            "scope": scope,
            "status": "RUNNING_NEW_APPEND_ONLY_ATTEMPT",
            "created_at_utc": _utc_now(),
            "protocol": protocol_lock,
            "cohort": {key: value for key, value in cohort.items() if key != "rows"},
            "gold_access": False,
            "prospective_opened_or_hashed": False,
        },
    )
    terminal_written = False
    rows_path = output_dir / "rows.jsonl"
    try:
        # Parent v8 preflight rehashes the exact shared model and Wiki18 assets
        # before CUDA.  The v9.1 protocol separately locks all new code/cohorts.
        verified = v8_driver.verify_implementation_before_cuda(project_root=PROJECT_ROOT)
        hf_runtime = v8_engine.SharedHuggingFaceRuntime(
            model_asset_identity=verified["model_asset_identity"]
        )
        retriever_runtime = v8_engine.CanonicalRetrieverRuntime.from_local_assets(
            retrieval_asset_identity=verified["retrieval_asset_identity"]
        )
        result = materialize(
            cohort["rows"],
            hf_runtime=hf_runtime,
            retriever_runtime=retriever_runtime,
            scope=scope,
            experiment_id=experiment_id,
            intended_output_dir=str(output_dir),
            cohort_lock={key: value for key, value in cohort.items() if key != "rows"},
        )
        _write_rows_new(rows_path, result["rows"])
        rows_lock = _file_lock(rows_path)
        report = build_report(
            result,
            protocol=protocol,
            scope=scope,
            experiment_id=experiment_id,
            rows_lock=rows_lock,
        )
        report["preflight"] = {
            "new_protocol": protocol_lock,
            "parent_v8_assets_rehashed_before_cuda": True,
            "cohort": {key: value for key, value in cohort.items() if key != "rows"},
        }
        report_path = output_dir / "report.json"
        _write_json_new(report_path, report)
        terminal = {
            "schema_version": "dynamic-decomposition-v9.1-terminal-manifest-1",
            "experiment_id": experiment_id,
            "scope": scope,
            "status": report["status"],
            "gold_access": False,
            "answer_scoring_performed": False,
            "prospective_opened_or_hashed": False,
            "protocol_sha256": protocol_lock["sha256"],
            "running": _file_lock(output_dir / "manifest.running.json"),
            "rows": rows_lock,
            "report": _file_lock(report_path),
        }
        if report["all_pass"] is not True:
            terminal["reason"] = (
                "frozen Gold-free runtime/mechanism gates failed; outputs retained"
            )
            _write_json_new(output_dir / "manifest.failed.json", terminal)
            terminal_written = True
            raise v8_driver.V8MechanismGateError(
                f"{scope} failed Gold-free mechanism gates; see {report_path}"
            )
        _write_json_new(output_dir / "manifest.complete.json", terminal)
        terminal_written = True
        return report
    except BaseException as exc:
        if not terminal_written:
            _write_json_new(
                output_dir / "manifest.failed.json",
                {
                    "schema_version": "dynamic-decomposition-v9.1-terminal-manifest-1",
                    "experiment_id": experiment_id,
                    "scope": scope,
                    "status": "FAILED_RETAINED_APPEND_ONLY",
                    "gold_access": False,
                    "prospective_opened_or_hashed": False,
                    "protocol_sha256": protocol_lock["sha256"],
                    "running": _file_lock(output_dir / "manifest.running.json"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "partial_rows": _file_lock(rows_path) if rows_path.is_file() else None,
                },
            )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, choices=SCOPES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(execute(args.scope), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
