#!/usr/bin/env python
"""Run the frozen v8 Gold-free smoke or development materialization.

The public CLI exposes only ``smoke`` and ``development``.  It has no cohort,
Gold, prospective, model, retriever, output-directory, or resume override.
Before CUDA is touched it verifies the implementation lock, current code, all
model trees, and all Wiki18 assets.  Every attempt is append-only; partial
output from an interrupted attempt is never resumed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare import (  # noqa: E402
    freeze_dynamic_decomposition_v8_implementation as implementation,
)


DRIVER_VERSION = "dynamic-decomposition-v8-production-driver-2"
IMPLEMENTATION_FREEZE_RELATIVE = implementation.DEFAULT_OUTPUT_DIR
IMPLEMENTATION_PROTOCOL_RELATIVE = IMPLEMENTATION_FREEZE_RELATIVE / "protocol.json"
IMPLEMENTATION_MANIFEST_RELATIVE = IMPLEMENTATION_FREEZE_RELATIVE / "manifest.json"
SCOPES = ("smoke", "development")
ATTEMPT_RE = re.compile(r"attempt([0-9]{3})$")
FORBIDDEN_RECURSIVE_KEYS = frozenset(
    {
        "gold",
        "gold_answer",
        "answers",
        "supporting_facts",
        "supporting_titles",
        "decomposition",
        "answerable",
    }
)


class V8ProductionDriverError(RuntimeError):
    """A formal preflight, mechanism gate, or append-only invariant failed."""


class V8MechanismGateError(V8ProductionDriverError):
    """Gold-free output was retained but failed a frozen mechanism gate."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _assert_file_lock(lock: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(lock, Mapping):
        raise V8ProductionDriverError(f"{label} lock is missing")
    path = Path(str(lock.get("path") or "")).expanduser().resolve()
    current = implementation.file_lock(
        path, allow_empty=int(lock.get("size_bytes", -1)) == 0
    )
    if (
        current["size_bytes"] != lock.get("size_bytes")
        or current["sha256"] != lock.get("sha256")
    ):
        raise V8ProductionDriverError(f"{label} content drift")
    return current


def _assert_tree_summary(lock: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(lock, Mapping):
        raise V8ProductionDriverError(f"{label} tree lock is missing")
    current = implementation.tree_lock(Path(str(lock.get("path") or "")))
    for field in ("file_count", "size_bytes", "tree_sha256"):
        if current[field] != lock.get(field):
            raise V8ProductionDriverError(f"{label} tree drift: {field}")
    return implementation.concise_tree_lock(current)


def _assert_parent_lock_tree(value: Any, *, label: str) -> None:
    """Recursively revalidate parent file-lock groups without special cases."""

    if not isinstance(value, Mapping) or not value:
        raise V8ProductionDriverError(f"{label} parent lock group is empty")
    if {"path", "size_bytes", "sha256"}.issubset(value):
        _assert_file_lock(value, label=label)
        return
    for child_name, child in value.items():
        _assert_parent_lock_tree(child, label=f"{label}.{child_name}")


def _validate_protocol_manifest(
    *, project_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol_path = (project_root / IMPLEMENTATION_PROTOCOL_RELATIVE).resolve()
    manifest_path = (project_root / IMPLEMENTATION_MANIFEST_RELATIVE).resolve()
    protocol = implementation.read_json(protocol_path)
    manifest = implementation.read_json(manifest_path)
    if (
        protocol.get("schema_version") != implementation.SCHEMA_VERSION
        or protocol.get("status") != implementation.STATUS
        or protocol.get("experiment_id") != implementation.EXPERIMENT_ID
        or protocol.get("gold_access") is not False
        or protocol.get("prospective_opened_or_hashed_by_this_command") is not False
        or not implementation.verify_self_commitment(
            protocol, field="protocol_body_canonical_sha256"
        )
    ):
        raise V8ProductionDriverError("implementation protocol identity/boundary drift")
    if (
        manifest.get("schema_version") != implementation.MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != implementation.STATUS
        or manifest.get("experiment_id") != implementation.EXPERIMENT_ID
        or manifest.get("gold_access") is not False
        or manifest.get("prospective_opened_or_hashed") is not False
        or not implementation.verify_self_commitment(
            manifest, field="manifest_body_canonical_sha256"
        )
    ):
        raise V8ProductionDriverError("implementation manifest identity/boundary drift")
    protocol_lock = implementation.file_lock(protocol_path)
    recorded = manifest.get("protocol") or {}
    if (
        Path(str(recorded.get("path") or "")).resolve() != protocol_path
        or recorded.get("size_bytes") != protocol_lock["size_bytes"]
        or recorded.get("sha256") != protocol_lock["sha256"]
    ):
        raise V8ProductionDriverError("implementation manifest does not bind protocol")
    return protocol, manifest, protocol_lock


def verify_implementation_before_cuda(
    *, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Revalidate the entire formal lock; this function must run before CUDA."""

    root = project_root.expanduser().resolve()
    if Path.cwd().resolve() != root:
        raise V8ProductionDriverError(
            f"formal v8 driver must run from project root {root}"
        )
    protocol, manifest, protocol_lock = _validate_protocol_manifest(project_root=root)
    authorization = protocol.get("authorization") or {}
    if (
        authorization.get("engineering_smoke_gold_free_materialization") is not True
        or authorization.get(
            "development90_gold_free_materialization_after_smoke_pass"
        )
        is not True
        or authorization.get("gold_attachment") is not False
        or authorization.get("answer_scoring") is not False
        or authorization.get("prospective_open_or_hash") is not False
        or authorization.get("training") is not False
    ):
        raise V8ProductionDriverError("implementation authorization drift")

    for group_name in ("runtime_code", "local_import_closure"):
        group = protocol.get(group_name)
        if not isinstance(group, Mapping) or not group:
            raise V8ProductionDriverError(f"implementation {group_name} is empty")
        for name, lock in group.items():
            _assert_file_lock(lock, label=f"{group_name}.{name}")

    content_section = protocol.get("content_reverification") or {}
    if content_section.get("full_hash_verification_performed_by_this_command") is not True:
        raise V8ProductionDriverError("formal lock skipped full-content verification")
    content = content_section.get("content") or {}
    models = content.get("models") or {}
    wiki18 = content.get("wiki18") or {}
    current_models = {
        name: _assert_tree_summary(models.get(name) or {}, label=f"models.{name}")
        for name in ("base_model", "strong_sft", "retrieval_encoder", "cross_encoder")
    }
    current_wiki = {
        "corpus": _assert_file_lock(wiki18.get("corpus") or {}, label="wiki18.corpus"),
        "dense_index": _assert_file_lock(
            wiki18.get("dense_index") or {}, label="wiki18.dense_index"
        ),
        "bm25_index": _assert_tree_summary(
            wiki18.get("bm25_index") or {}, label="wiki18.bm25_index"
        ),
    }
    tokenizer = protocol.get("tokenizer_and_chat_template") or {}
    current_tokenizer_config = _assert_file_lock(
        tokenizer.get("tokenizer_config") or {}, label="tokenizer_config"
    )
    current_tokenizer_json = _assert_file_lock(
        tokenizer.get("tokenizer_json") or {}, label="tokenizer_json"
    )
    tokenizer_config = implementation.read_json(
        Path(current_tokenizer_config["path"])
    )
    chat_template = tokenizer_config.get("chat_template")
    if (
        not isinstance(chat_template, str)
        or implementation.sha256_text(chat_template)
        != tokenizer.get("chat_template_utf8_sha256")
    ):
        raise V8ProductionDriverError("chat template content drift")

    for name, lock in (protocol.get("parents") or {}).items():
        _assert_parent_lock_tree(lock, label=f"parents.{name}")

    smoke_registry = (protocol.get("run_registry") or {}).get("smoke") or {}
    if (
        smoke_registry.get("next_authorized_attempt_id") != "attempt002"
        or smoke_registry.get("next_authorized_experiment_id")
        != implementation.SMOKE_EXPERIMENT_ID_ATTEMPT002
        or Path(str(smoke_registry.get("next_authorized_attempt_dir") or "")).resolve()
        != (root / implementation.SMOKE_ATTEMPT002).resolve()
        or smoke_registry.get("attempt002_retry_parent")
        != ((protocol.get("parents") or {}).get("v1_failed_smoke_retry_parent") or {}).get(
            "attempt001_failed"
        )
    ):
        raise V8ProductionDriverError("v2 smoke attempt002 retry registry drift")

    runner_path = Path(
        str((protocol.get("runtime_code") or {}).get("runner", {}).get("path") or "")
    )
    runtime_contract = implementation.validate_runtime_contract(
        implementation.literal_constant(
            runner_path, implementation.RUNTIME_CONTRACT_CONSTANT
        )
    )
    if runtime_contract != protocol.get("runtime_contract"):
        raise V8ProductionDriverError("runner runtime contract differs from formal lock")
    role_identity = protocol.get("generation_role_identity") or {}
    if (
        role_identity.get("same_base_tree_sha256")
        != current_models["base_model"]["tree_sha256"]
        or role_identity.get("same_adapter_tree_sha256")
        != current_models["strong_sft"]["tree_sha256"]
        or role_identity.get("same_tokenizer_json_sha256")
        != current_tokenizer_json["sha256"]
        or role_identity.get("same_chat_template_sha256")
        != implementation.sha256_text(chat_template)
        or role_identity.get("runtime_same_python_object_identity_required") is not True
    ):
        raise V8ProductionDriverError("generation role/model identity drift")
    return {
        "protocol": protocol,
        "manifest": manifest,
        "protocol_lock": protocol_lock,
        "model_asset_identity": {
            "base_model_tree_sha256": current_models["base_model"]["tree_sha256"],
            "adapter_tree_sha256": current_models["strong_sft"]["tree_sha256"],
            "tokenizer_tree_sha256": current_models["base_model"]["tree_sha256"],
        },
        "retrieval_asset_identity": {
            "corpus_sha256": current_wiki["corpus"]["sha256"],
            "dense_index_sha256": current_wiki["dense_index"]["sha256"],
            "bm25_tree_sha256": current_wiki["bm25_index"]["tree_sha256"],
            "e5_tree_sha256": current_models["retrieval_encoder"]["tree_sha256"],
            "bge_tree_sha256": current_models["cross_encoder"]["tree_sha256"],
        },
    }


def _attempt_identity(
    protocol: Mapping[str, Any], *, scope: str, attempt_number: int
) -> dict[str, Any]:
    if scope not in SCOPES or not 1 <= attempt_number <= 999:
        raise V8ProductionDriverError("invalid scope or attempt number")
    registry_name = "smoke" if scope == "smoke" else "development"
    first = (protocol.get("run_registry") or {}).get(registry_name) or {}
    first_dir = Path(str(first.get("first_attempt_dir") or ""))
    first_id = str(first.get("experiment_id") or "")
    if not first_dir.name.endswith("_attempt001") or not first_id.endswith(
        "ATTEMPT001"
    ):
        raise V8ProductionDriverError("first-attempt registry is malformed")
    suffix = f"{attempt_number:03d}"
    destination = first_dir.with_name(
        first_dir.name[: -len("001")] + suffix
    ).resolve()
    experiment_id = first_id[: -len("001")] + suffix
    return {
        "scope": scope,
        "attempt_number": attempt_number,
        "attempt_id": f"attempt{suffix}",
        "experiment_id": experiment_id,
        "attempt_dir": destination,
        "cohort_n": int(first.get("cohort_n", -1)),
        "cohort_sha256": str(first.get("cohort_sha256") or ""),
    }


def _retry_parent_lock(
    identity: Mapping[str, Any], *, protocol: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    attempt_number = int(identity["attempt_number"])
    if attempt_number == 1:
        return None
    destination = Path(str(identity["attempt_dir"]))
    previous_suffix = f"{attempt_number - 1:03d}"
    previous = destination.with_name(
        destination.name[: -len(f"{attempt_number:03d}")] + previous_suffix
    )
    failed = previous / implementation.FAILED_MANIFEST
    if not failed.is_file():
        raise V8ProductionDriverError(
            "retry requires the immediately preceding append-only FAILED manifest"
        )
    current = implementation.file_lock(failed)
    if identity.get("scope") == "smoke" and attempt_number == 2:
        expected = (
            ((protocol.get("run_registry") or {}).get("smoke") or {}).get(
                "attempt002_retry_parent"
            )
        )
        if current != expected:
            raise V8ProductionDriverError(
                "smoke attempt002 does not bind the frozen attempt001 FAILED manifest"
            )
    return current


def _document_id(document: Mapping[str, Any]) -> str:
    values = [
        str(document[key])
        for key in ("doc_id", "id", "document_id")
        if key in document and document[key] is not None
    ]
    if not values or len(set(values)) != 1 or not values[0]:
        raise V8ProductionDriverError("final passage lacks one stable document id")
    return values[0]


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        if FORBIDDEN_RECURSIVE_KEYS.intersection(str(key) for key in value):
            return True
        return any(_contains_forbidden_key(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 1.0


def build_gold_free_mechanism_report(
    *,
    result: Mapping[str, Any],
    protocol: Mapping[str, Any],
    scope: str,
    rows_lock: Mapping[str, Any],
    created_at_utc: str,
) -> dict[str, Any]:
    """Compute only Gold-free runtime/mechanism gates from frozen predictions."""

    if scope not in SCOPES:
        raise V8ProductionDriverError("unsupported report scope")
    rows = result.get("rows")
    expected_n = 12 if scope == "smoke" else 90
    expected_per_dataset = 4 if scope == "smoke" else 30
    if not isinstance(rows, list) or len(rows) != expected_n:
        raise V8ProductionDriverError("materialization result cardinality mismatch")
    expected_budget = (protocol["runtime_contract"])["logical_budget_by_arm"]
    counts: Counter[str] = Counter()
    q1_valid: Counter[str] = Counter()
    a1_valid: Counter[str] = Counter()
    dynamic_valid: Counter[str] = Counter()
    b_valid = 0
    repeated_padding = 0
    logical_exact = 0
    cache_exact = 0
    b_allowlist = 0
    c_binding = 0
    ineligible_count = 0
    ineligible_identity = 0
    dynamic_invalid_count = 0
    dynamic_invalid_fallback = 0
    root_shared = 0
    final_unique = 0
    asset_bound_cache = 0
    asset_bound_cache_denominator = 0
    model_object_ids: set[int] = set()
    forbidden_rows = 0

    runtime_contract = protocol.get("runtime_contract") or {}
    staged_contract = runtime_contract.get("staged_retrieval_contract") or {}
    batch_telemetry = result.get("retrieval_batch_telemetry") or {}
    run_cache = result.get("joint_cache_accounting") or {}
    retrieval_run_cache = run_cache.get("retrieval") or {}
    expected_logical_retrieval = expected_n * sum(
        int(values.get("retrieval", 0))
        for values in (runtime_contract.get("logical_budget_by_arm") or {}).values()
        if isinstance(values, Mapping)
    )
    stage_batches = batch_telemetry.get("stage_batches")
    expected_stage_order = list(staged_contract.get("backend_batch_stages") or [])
    observed_stage_order = (
        [str(item.get("stage") or "") for item in stage_batches]
        if isinstance(stage_batches, list)
        and all(isinstance(item, Mapping) for item in stage_batches)
        else []
    )
    full_index_passes = batch_telemetry.get("full_index_passes")
    backend_batch_invocations = batch_telemetry.get("backend_batch_invocations")
    unique_query_count_by_batch = batch_telemetry.get("unique_query_count_by_batch")
    invoked_stage_count = (
        sum(item.get("backend_invoked") is True for item in stage_batches)
        if isinstance(stage_batches, list)
        and all(isinstance(item, Mapping) for item in stage_batches)
        else -1
    )
    batch_telemetry_integrity = (
        runtime_contract.get("production_staged") is True
        and observed_stage_order == expected_stage_order
        and type(full_index_passes) is int
        and type(backend_batch_invocations) is int
        and isinstance(unique_query_count_by_batch, list)
        and all(type(value) is int and value > 0 for value in unique_query_count_by_batch)
        and full_index_passes == backend_batch_invocations == invoked_stage_count
        and len(unique_query_count_by_batch) == full_index_passes
        and sum(unique_query_count_by_batch)
        == retrieval_run_cache.get("physical_executions")
        and all(
            type(item.get("logical_request_groups")) is int
            and item.get("logical_request_groups") >= 0
            and type(item.get("unique_miss_query_count")) is int
            and item.get("unique_miss_query_count") >= 0
            and item.get("backend_invoked")
            is (item.get("unique_miss_query_count") > 0)
            for item in (stage_batches or [])
            if isinstance(item, Mapping)
        )
    )

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("gold_access") is not False:
            raise V8ProductionDriverError(f"row {index} violates Gold-free schema")
        identity = row.get("identity") or {}
        dataset = str(identity.get("dataset") or "")
        if dataset not in implementation.DATASETS:
            raise V8ProductionDriverError(f"row {index} has invalid dataset")
        counts[dataset] += 1
        arms = row.get("arms") or {}
        if set(arms) != {
            "A_canonical_one_shot",
            "B_observation_blind",
            "C_answer_conditioned",
        }:
            raise V8ProductionDriverError(f"row {index} lacks exact A/B/C arms")
        shared = row.get("shared") or {}
        q1_action = shared.get("q1_action") or {}
        binding = shared.get("subanswer_binding") or {}
        b = arms["B_observation_blind"]
        c = arms["C_answer_conditioned"]
        if q1_action.get("proposal_valid") is True:
            q1_valid[dataset] += 1
        if binding.get("verified") is True:
            a1_valid[dataset] += 1
        if b.get("q2_action", {}).get("proposal_valid") is True:
            b_valid += 1
        dynamic_eligible = c.get("dynamic_eligible") is True
        c_action = c.get("q2_action") or {}
        if (
            dynamic_eligible
            and c_action.get("proposal_valid") is True
            and c_action.get("selection_source") == "q2_dynamic"
        ):
            dynamic_valid[dataset] += 1
        if b.get("q2_state_mode") == "q2_no_verified_subanswer":
            b_allowlist += 1
        if (
            (dynamic_eligible and c.get("q2_state_mode") == "q2_dynamic" and binding.get("verified") is True)
            or (not dynamic_eligible and c.get("q2_state_mode") == "q2_no_verified_subanswer")
        ):
            c_binding += 1
        if not dynamic_eligible:
            ineligible_count += 1
            identity_checks = row.get("counterfactual_identity") or {}
            required = (
                "b_c_q2_prompt_byte_identical",
                "b_c_q2_response_byte_identical",
                "b_c_q2_query_byte_identical",
                "b_c_final_passages_byte_identical",
                "b_c_final_prompt_byte_identical",
                "b_c_prediction_byte_identical",
            )
            if all(identity_checks.get(field) is True for field in required):
                ineligible_identity += 1
        elif c_action.get("proposal_valid") is False:
            dynamic_invalid_count += 1
            if (
                c_action.get("selected_query") == identity.get("question")
                and c_action.get("selection_source") == "original_question"
                and c_action.get("used_fallback") is True
                and (row.get("budget") or {})
                .get("logical_by_arm", {})
                .get("C_answer_conditioned", {})
                .get("controller")
                == 2
            ):
                dynamic_invalid_fallback += 1

        for action, previous in (
            (q1_action, {identity.get("question")}),
            (b.get("q2_action") or {}, {identity.get("question"), q1_action.get("selected_query")}),
            (c_action, {identity.get("question"), q1_action.get("selected_query")}),
        ):
            if action.get("proposal_valid") is True and (
                not action.get("selected_query")
                or action.get("selected_query") in previous
            ):
                repeated_padding += 1

        budget = row.get("budget") or {}
        if budget.get("logical_by_arm") == expected_budget:
            logical_exact += 1
        accounting = budget.get("joint_cache_accounting") or {}
        if accounting and all(
            values.get("logical_requests")
            == values.get("cache_hits", -1) + values.get("cache_misses", -1)
            and values.get("physical_executions") == values.get("cache_misses")
            for values in accounting.values()
        ):
            cache_exact += 1
        a_root = arms["A_canonical_one_shot"].get("retrieval") or {}
        bc_root = shared.get("root_retrieval") or {}
        if (
            a_root.get("content_cache_key_sha256")
            == bc_root.get("content_cache_key_sha256")
            and bc_root.get("cache_hit") is True
        ):
            root_shared += 1

        row_all_unique = True
        for arm in arms.values():
            passages = arm.get("final_passages")
            if not isinstance(passages, list) or len(passages) != 10:
                row_all_unique = False
                continue
            ids = [_document_id(passage) for passage in passages]
            row_all_unique = row_all_unique and len(set(ids)) == 10
        if row_all_unique:
            final_unique += 1

        for event in (
            a_root,
            bc_root,
            shared.get("q1_retrieval") or {},
            b.get("q2_retrieval") or {},
            c.get("q2_retrieval") or {},
            shared.get("q1_controller") or {},
            shared.get("subanswer_reader") or {},
            b.get("q2_controller") or {},
            c.get("q2_controller") or {},
            *(arm.get("final", {}).get("reader_event") or {} for arm in arms.values()),
        ):
            if not isinstance(event, Mapping):
                continue
            mode = event.get("content_cache_key_mode")
            if mode is not None:
                asset_bound_cache_denominator += 1
                if mode in {
                    "asset_bound_model_visible_token_ids",
                    "asset_bound_query_and_retrieval_stack",
                }:
                    asset_bound_cache += 1
            telemetry = event.get("runtime_telemetry") or {}
            object_id = telemetry.get("shared_model_object_id")
            if type(object_id) is int:
                model_object_ids.add(object_id)
        if _contains_forbidden_key(row):
            forbidden_rows += 1

    expected_counts = {dataset: expected_per_dataset for dataset in implementation.DATASETS}
    metrics = {
        "row_count": len(rows),
        "per_dataset_counts": dict(counts),
        "q1_schema_valid_rate_each_dataset": {
            dataset: _rate(q1_valid[dataset], counts[dataset])
            for dataset in implementation.DATASETS
        },
        "B_q2_static_valid_rate": _rate(b_valid, len(rows)),
        "a1_admissible_rate_each_dataset": {
            dataset: _rate(a1_valid[dataset], counts[dataset])
            for dataset in implementation.DATASETS
        },
        "C_dynamic_transition_rate_all_itt_each_dataset": {
            dataset: _rate(dynamic_valid[dataset], counts[dataset])
            for dataset in implementation.DATASETS
        },
        "empty_repeat_padding_query_rate": _rate(repeated_padding, len(rows) * 3),
        "logical_ledger_exact_rate": _rate(logical_exact, len(rows)),
        "cache_accounting_conservation_rate": _rate(cache_exact, len(rows)),
        "B_static_allowlist_rate": _rate(b_allowlist, len(rows)),
        "C_dynamic_state_binding_integrity_rate": _rate(c_binding, len(rows)),
        "a1_ineligible_count": ineligible_count,
        "a1_ineligible_full_content_identity_rate": _rate(
            ineligible_identity, ineligible_count
        ),
        "eligible_dynamic_invalid_count": dynamic_invalid_count,
        "eligible_dynamic_invalid_original_Q_no_third_call_rate": _rate(
            dynamic_invalid_fallback, dynamic_invalid_count
        ),
        "root_and_q1_shared_byte_identity_rate": _rate(root_shared, len(rows)),
        "final_10_unique_rate": _rate(final_unique, len(rows)),
        "asset_bound_content_cache_event_rate": _rate(
            asset_bound_cache, asset_bound_cache_denominator
        ),
        "shared_model_object_id_count": len(model_object_ids),
        "production_staged": runtime_contract.get("production_staged") is True,
        "logical_retrieval_requests": retrieval_run_cache.get("logical_requests"),
        "expected_logical_retrieval_requests": expected_logical_retrieval,
        "retrieval_full_index_passes": full_index_passes,
        "retrieval_full_index_passes_max": staged_contract.get(
            "maximum_full_index_passes_per_attempt"
        ),
        "retrieval_backend_batch_invocations": backend_batch_invocations,
        "retrieval_unique_query_count_by_batch": unique_query_count_by_batch,
        "retrieval_stage_batches": stage_batches,
        "retrieval_batch_telemetry_integrity": batch_telemetry_integrity,
        "runtime_error_count": 0,
        "gold_or_forbidden_recursive_field_access_count": forbidden_rows,
    }
    smoke_gates = (protocol.get("gates") or {}).get("smoke") or {}
    if scope == "smoke":
        gate_results = {
            "row_count": metrics["row_count"] == smoke_gates.get("row_count"),
            "per_dataset_counts": metrics["per_dataset_counts"] == expected_counts,
            "runtime_error_count": metrics["runtime_error_count"] == 0,
            "gold_or_prospective_access_count": forbidden_rows == 0,
            "all_three_arms_present_rate": True,
            "logical_budget_exact_rate": metrics["logical_ledger_exact_rate"] == 1.0,
            "staged_logical_retrieval_budget": metrics[
                "logical_retrieval_requests"
            ]
            == smoke_gates.get("logical_retrieval_requests")
            == staged_contract.get("engineering_smoke_logical_retrieval_requests"),
            "staged_retrieval_batch_telemetry": metrics[
                "retrieval_batch_telemetry_integrity"
            ],
            "full_index_passes_within_frozen_max": (
                type(metrics["retrieval_full_index_passes"]) is int
                and metrics["retrieval_full_index_passes"]
                <= int(smoke_gates.get("full_index_passes_max", -1))
            ),
            "cache_accounting_conservation_rate": metrics[
                "cache_accounting_conservation_rate"
            ]
            == 1.0,
            "final_10_unique_rate": metrics["final_10_unique_rate"] == 1.0,
            "asset_bound_content_cache_event_rate": metrics[
                "asset_bound_content_cache_event_rate"
            ]
            == 1.0,
            "one_shared_model_object": len(model_object_ids) == 1,
            "B_observation_blind_allowlist": metrics["B_static_allowlist_rate"]
            == 1.0,
            "ineligible_counterfactual_identity": metrics[
                "a1_ineligible_full_content_identity_rate"
            ]
            == 1.0,
            "dynamic_invalid_no_third_call": metrics[
                "eligible_dynamic_invalid_original_Q_no_third_call_rate"
            ]
            == 1.0,
        }
    else:
        frozen = (protocol.get("gates") or {}).get("development_gold_free") or {}
        gate_results = {
            "itt_cardinality": metrics["row_count"] == frozen.get("itt_cardinality"),
            "per_dataset_counts": metrics["per_dataset_counts"] == expected_counts,
            "q1_schema_valid_rate_min_each_dataset": all(
                value >= float(frozen["q1_schema_valid_rate_min_each_dataset"])
                for value in metrics["q1_schema_valid_rate_each_dataset"].values()
            ),
            "B_q2_static_valid_rate_min": metrics["B_q2_static_valid_rate"]
            >= float(frozen["B_q2_static_valid_rate_min"]),
            "a1_admissible_rate_min_each_dataset": all(
                value >= float(frozen["a1_admissible_rate_min_each_dataset"])
                for value in metrics["a1_admissible_rate_each_dataset"].values()
            ),
            "C_dynamic_transition_rate_all_itt_min_each_dataset": all(
                value
                >= float(frozen["C_dynamic_transition_rate_all_itt_min_each_dataset"])
                for value in metrics[
                    "C_dynamic_transition_rate_all_itt_each_dataset"
                ].values()
            ),
            "empty_repeat_padding_query_rate_max": metrics[
                "empty_repeat_padding_query_rate"
            ]
            <= float(frozen["empty_repeat_padding_query_rate_max"]),
            "logical_ledger_exact_rate": metrics["logical_ledger_exact_rate"] == 1.0,
            "staged_logical_retrieval_budget": metrics[
                "logical_retrieval_requests"
            ]
            == frozen.get("logical_retrieval_requests")
            == expected_logical_retrieval,
            "staged_retrieval_batch_telemetry": metrics[
                "retrieval_batch_telemetry_integrity"
            ],
            "full_index_passes_within_frozen_max": (
                type(metrics["retrieval_full_index_passes"]) is int
                and metrics["retrieval_full_index_passes"]
                <= int(frozen.get("full_index_passes_max", -1))
            ),
            "cache_accounting_conservation_rate": metrics[
                "cache_accounting_conservation_rate"
            ]
            == 1.0,
            "B_static_allowlist_rate": metrics["B_static_allowlist_rate"] == 1.0,
            "C_dynamic_state_binding_integrity_rate": metrics[
                "C_dynamic_state_binding_integrity_rate"
            ]
            == 1.0,
            "a1_ineligible_full_content_identity_rate": metrics[
                "a1_ineligible_full_content_identity_rate"
            ]
            == 1.0,
            "eligible_dynamic_invalid_original_Q_no_third_call_rate": metrics[
                "eligible_dynamic_invalid_original_Q_no_third_call_rate"
            ]
            == 1.0,
            "root_and_q1_shared_byte_identity_rate": metrics[
                "root_and_q1_shared_byte_identity_rate"
            ]
            == 1.0,
            "final_10_unique_rate": metrics["final_10_unique_rate"] == 1.0,
            "asset_bound_content_cache_event_rate": metrics[
                "asset_bound_content_cache_event_rate"
            ]
            == 1.0,
            "one_shared_model_object": len(model_object_ids) == 1,
            "runtime_error_count": metrics["runtime_error_count"] == 0,
            "gold_or_forbidden_recursive_field_access_count": forbidden_rows == 0,
        }
    return {
        "schema_version": "dynamic-decomposition-v8-gold-free-mechanism-report-1",
        "driver_version": DRIVER_VERSION,
        "experiment_id": result.get("experiment_id"),
        "intended_output_dir": result.get("intended_output_dir"),
        "retry_of_attempt001_contract": result.get(
            "retry_of_attempt001_contract", False
        ),
        "original_first_attempt_contract": result.get(
            "original_first_attempt_contract"
        ),
        "scope": scope,
        "created_at_utc": created_at_utc,
        "status": "PASS" if all(gate_results.values()) else "FAIL_STOP_GOLD_FREE_GATES",
        "gold_access": False,
        "prospective_opened_or_hashed": False,
        "rows": dict(rows_lock),
        "metrics": metrics,
        "gate_results": gate_results,
        "all_pass": all(gate_results.values()),
        "scientific_boundary": (
            "Gold-free runtime/mechanism report only; no EM/F1/IHR or utility claim."
        ),
    }


def _write_rows_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
        handle.flush()


def _validate_prior_smoke_pass(
    protocol: Mapping[str, Any],
    *,
    implementation_protocol_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    first = _attempt_identity(protocol, scope="smoke", attempt_number=1)
    first_path = Path(str(first["attempt_dir"]))
    parent = first_path.parent
    prefix = first_path.name[: -len("001")]
    complete_attempts: list[tuple[int, Path]] = []
    for path in parent.glob(f"{prefix}[0-9][0-9][0-9]"):
        match = ATTEMPT_RE.search(path.name)
        if match is not None and (path / implementation.COMPLETE_MANIFEST).is_file():
            complete_attempts.append((int(match.group(1)), path))
    if not complete_attempts:
        raise V8ProductionDriverError(
            "development requires a hash-validated completed smoke attempt"
        )
    _, attempt = max(complete_attempts, key=lambda item: item[0])
    match = ATTEMPT_RE.search(attempt.name)
    if match is None:
        raise V8ProductionDriverError("completed smoke attempt name is malformed")
    expected_identity = _attempt_identity(
        protocol, scope="smoke", attempt_number=int(match.group(1))
    )
    if Path(str(expected_identity["attempt_dir"])) != attempt.resolve():
        raise V8ProductionDriverError("completed smoke attempt path is not canonical")

    terminal_path = attempt / implementation.COMPLETE_MANIFEST
    terminal = implementation.read_json(terminal_path)
    if (
        terminal.get("schema_version")
        != "dynamic-decomposition-v8-run-attempt-terminal-1"
        or terminal.get("status") != "COMPLETE"
        or terminal.get("experiment_id") != expected_identity["experiment_id"]
        or terminal.get("attempt_id") != expected_identity["attempt_id"]
        or terminal.get("gold_access") is not False
        or terminal.get("prospective_opened_or_hashed") is not False
        or not implementation.verify_self_commitment(
            terminal, field="manifest_body_canonical_sha256"
        )
    ):
        raise V8ProductionDriverError("completed smoke terminal manifest is invalid")
    running_path = attempt / implementation.RUNNING_MANIFEST
    current_running = implementation.file_lock(running_path)
    running = implementation.read_json(running_path)
    expected_cohort = {
        "role": "consumed_smoke",
        "row_count": expected_identity["cohort_n"],
        "sha256": expected_identity["cohort_sha256"],
        "prospective_unlocked": False,
        "gold_access": False,
    }
    if (
        running.get("status") != "RUNNING_NEW_ATTEMPT_NO_IN_PLACE_RESUME"
        or running.get("experiment_id") != expected_identity["experiment_id"]
        or running.get("attempt_id") != expected_identity["attempt_id"]
        or running.get("cohort") != expected_cohort
        or running.get("gold_access") is not False
        or running.get("prospective_opened_or_hashed") is not False
        or not implementation.verify_self_commitment(
            running, field="manifest_body_canonical_sha256"
        )
        or (
            implementation_protocol_lock is not None
            and running.get("implementation_protocol")
            != dict(implementation_protocol_lock)
        )
    ):
        raise V8ProductionDriverError("completed smoke RUNNING manifest is invalid")
    if int(expected_identity["attempt_number"]) > 1:
        retry_parent = _retry_parent_lock(expected_identity, protocol=protocol)
        if running.get("retry_of") != retry_parent:
            raise V8ProductionDriverError(
                "completed smoke retry does not bind its preceding FAILED manifest"
            )
    elif running.get("retry_of") is not None:
        raise V8ProductionDriverError("attempt001 unexpectedly records retry_of")
    if terminal.get("running_manifest") != current_running:
        raise V8ProductionDriverError("completed smoke terminal does not bind RUNNING")

    validated_stages = {
        stage: implementation.validate_reusable_stage(attempt, stage)
        for stage in ("gold_free_rows", "gold_free_report")
    }
    recorded_stages = terminal.get("complete_stage_descriptors")
    if not isinstance(recorded_stages, Mapping) or set(recorded_stages) != set(
        validated_stages
    ):
        raise V8ProductionDriverError("completed smoke terminal stage set is invalid")
    for stage, validated in validated_stages.items():
        if recorded_stages.get(stage) != validated["descriptor"]:
            raise V8ProductionDriverError(
                f"completed smoke terminal does not bind {stage} descriptor"
            )

    report_lock = validated_stages["gold_free_report"]["stage"]["artifacts"]
    report_items = [item for item in report_lock if item.get("path") == "report.json"]
    if len(report_items) != 1:
        raise V8ProductionDriverError("completed smoke has no unique report lock")
    report_path = attempt / "report.json"
    current = implementation.file_lock(report_path)
    if (
        current["sha256"] != report_items[0].get("sha256")
        or current["size_bytes"] != report_items[0].get("size_bytes")
    ):
        raise V8ProductionDriverError("completed smoke report drift")
    report = implementation.read_json(report_path)
    if (
        report.get("status") != "PASS"
        or report.get("all_pass") is not True
        or report.get("experiment_id") != expected_identity["experiment_id"]
        or report.get("scope") != "smoke"
        or Path(str(report.get("intended_output_dir") or "")).resolve()
        != attempt.resolve()
        or report.get("gold_access") is not False
        or report.get("prospective_opened_or_hashed") is not False
        or (
            implementation_protocol_lock is not None
            and (report.get("preflight") or {}).get("implementation_protocol")
            != dict(implementation_protocol_lock)
        )
    ):
        raise V8ProductionDriverError("completed smoke did not pass Gold-free gates")
    return {
        "attempt": str(attempt),
        "attempt_number": int(expected_identity["attempt_number"]),
        "implementation_protocol": dict(implementation_protocol_lock)
        if implementation_protocol_lock is not None
        else running.get("implementation_protocol"),
        "terminal": implementation.file_lock(terminal_path),
        "report": current,
    }


def _load_runner() -> ModuleType:
    return importlib.import_module(
        "scripts.pilot.materialize_dynamic_decomposition_v8"
    )


def execute_scope(
    *,
    scope: str,
    attempt_number: int = 1,
    project_root: Path = PROJECT_ROOT,
    verified: Mapping[str, Any] | None = None,
    runner_module: ModuleType | Any | None = None,
    hf_factory: Callable[..., Any] | None = None,
    retriever_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Execute one new append-only attempt; injectable factories are test-only."""

    if scope not in SCOPES:
        raise V8ProductionDriverError("scope must be smoke or development")
    root = project_root.expanduser().resolve()
    verified_bundle = (
        dict(verified)
        if verified is not None
        else verify_implementation_before_cuda(project_root=root)
    )
    protocol = verified_bundle["protocol"]
    identity = _attempt_identity(
        protocol, scope=scope, attempt_number=attempt_number
    )
    smoke_parent = (
        _validate_prior_smoke_pass(
            protocol,
            implementation_protocol_lock=verified_bundle["protocol_lock"],
        )
        if scope == "development"
        else None
    )
    retry_of = _retry_parent_lock(identity, protocol=protocol)
    attempt_dir = Path(str(identity["attempt_dir"]))
    cohort_lock = {
        "role": "consumed_smoke" if scope == "smoke" else "development",
        "row_count": identity["cohort_n"],
        "sha256": identity["cohort_sha256"],
        "prospective_unlocked": False,
        "gold_access": False,
    }
    implementation.reserve_attempt_directory(
        attempt_dir=attempt_dir,
        experiment_id=str(identity["experiment_id"]),
        attempt_id=str(identity["attempt_id"]),
        implementation_protocol=verified_bundle["protocol_lock"],
        cohort_lock=cohort_lock,
        created_at_utc=_utc_now(),
        retry_of=retry_of,
    )
    completed_stages: list[str] = []
    terminal_written = False
    try:
        runner = runner_module or _load_runner()
        if runner.runtime_contract() != protocol["runtime_contract"]:
            raise V8ProductionDriverError("loaded runner contract drifted after preflight")
        hf_builder = hf_factory or runner.SharedHuggingFaceRuntime
        retrieval_builder = retriever_factory or runner.CanonicalRetrieverRuntime.from_local_assets
        # This is the first point at which the formal path may touch CUDA.
        hf_runtime = hf_builder(
            model_asset_identity=verified_bundle["model_asset_identity"]
        )
        retriever_runtime = retrieval_builder(
            retrieval_asset_identity=verified_bundle["retrieval_asset_identity"]
        )
        if scope == "smoke":
            result = runner.materialize_locked_consumed_smoke4x3_production(
                hf_runtime=hf_runtime, retriever_runtime=retriever_runtime
            )
        else:
            result = runner.materialize_frozen_development_production(
                hf_runtime=hf_runtime, retriever_runtime=retriever_runtime
            )
        if result.get("experiment_id") != identity["experiment_id"]:
            # The locked runner owns ATTEMPT001.  A mechanical retry gets a new
            # append-only ID at the outer writer without changing scientific inputs.
            if attempt_number == 1:
                raise V8ProductionDriverError("runner Experiment ID drift")
            result = dict(result)
            result["original_first_attempt_contract"] = {
                "experiment_id": result.get("experiment_id"),
                "intended_output_dir": result.get("intended_output_dir"),
            }
            result["experiment_id"] = identity["experiment_id"]
            result["intended_output_dir"] = str(attempt_dir)
            result["retry_of_attempt001_contract"] = True
        elif result.get("intended_output_dir") is None:
            result = dict(result)
            result["intended_output_dir"] = str(attempt_dir)
        elif result.get("intended_output_dir") is not None and Path(
            str(result["intended_output_dir"])
        ).resolve() != attempt_dir:
            raise V8ProductionDriverError("runner intended output directory drift")
        rows = result.get("rows")
        if not isinstance(rows, list):
            raise V8ProductionDriverError("runner returned no row list")
        rows_path = attempt_dir / "rows.jsonl"
        _write_rows_exclusive(rows_path, rows)
        rows_lock = implementation.file_lock(rows_path)
        stage_config_sha = implementation.sha256_bytes(
            implementation.canonical_json_bytes(
                {
                    "implementation_protocol_sha256": verified_bundle[
                        "protocol_lock"
                    ]["sha256"],
                    "runtime_contract": protocol["runtime_contract"],
                    "scope": scope,
                    "experiment_id": identity["experiment_id"],
                }
            )
        )
        implementation.commit_stage_boundary(
            attempt_dir=attempt_dir,
            stage_name="gold_free_rows",
            artifact_paths=["rows.jsonl"],
            row_count=len(rows),
            stage_config_sha256=stage_config_sha,
            completed_at_utc=_utc_now(),
        )
        completed_stages.append("gold_free_rows")
        report = build_gold_free_mechanism_report(
            result=result,
            protocol=protocol,
            scope=scope,
            rows_lock=rows_lock,
            created_at_utc=_utc_now(),
        )
        report["preflight"] = {
            "implementation_protocol": dict(verified_bundle["protocol_lock"]),
            "all_code_model_and_retrieval_assets_rehashed_before_cuda": True,
            "development_smoke_prerequisite": smoke_parent,
        }
        report_path = attempt_dir / "report.json"
        implementation._write_bytes_exclusive(
            report_path, implementation.canonical_json_bytes(report)
        )
        implementation.commit_stage_boundary(
            attempt_dir=attempt_dir,
            stage_name="gold_free_report",
            artifact_paths=["report.json"],
            row_count=len(rows),
            stage_config_sha256=stage_config_sha,
            completed_at_utc=_utc_now(),
        )
        completed_stages.append("gold_free_report")
        if report["all_pass"] is not True:
            implementation.finalize_attempt(
                attempt_dir=attempt_dir,
                success=False,
                reason="frozen Gold-free mechanism gates failed; outputs retained",
                completed_at_utc=_utc_now(),
                required_complete_stages=completed_stages,
            )
            terminal_written = True
            raise V8MechanismGateError(
                f"{scope} failed Gold-free mechanism gates; see {report_path}"
            )
        terminal = implementation.finalize_attempt(
            attempt_dir=attempt_dir,
            success=True,
            reason="all frozen Gold-free runtime/mechanism gates passed",
            completed_at_utc=_utc_now(),
            required_complete_stages=completed_stages,
        )
        terminal_written = True
        return {
            "scope": scope,
            "experiment_id": identity["experiment_id"],
            "attempt_dir": str(attempt_dir),
            "status": terminal["status"],
            "report": report,
        }
    except Exception as exc:
        if not terminal_written:
            try:
                implementation.finalize_attempt(
                    attempt_dir=attempt_dir,
                    success=False,
                    reason=f"{type(exc).__name__}: {exc}",
                    completed_at_utc=_utc_now(),
                    required_complete_stages=completed_stages,
                )
                terminal_written = True
            except Exception as terminal_exc:
                raise V8ProductionDriverError(
                    f"run failed ({exc}); FAILED manifest also failed ({terminal_exc})"
                ) from exc
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=SCOPES, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    # No cohort/output/model/retriever/Gold/prospective/resume override exists.
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = execute_scope(scope=args.scope, attempt_number=args.attempt)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
