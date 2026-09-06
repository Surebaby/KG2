#!/usr/bin/env python
"""Materialise the conservative v5 plan-once dependent-retrieval pilot.

This runner is intentionally separate from the frozen v4 development runner.
It reuses only v4's input identity/Gold guards and immutable Wiki18 asset
helpers.  Its scientific variable is a precision-first retrieval policy:

* questions without a dependent plan step return frozen Arm A byte-for-byte;
* plan steps are searched in dependency-depth batches;
* every output which is required by a later step passes the Gold-free v5
  bridge admission selector, otherwise the whole row returns Arm A exactly;
* root-hop documents can resolve a bridge but can never enter the final prompt;
* the two replaceable Arm-A tail documents and the top two results of every
  dependent hop are scored together by the same cached BGE cross encoder
  against the original full question; and
* only a strictly better dependent document may replace ranks 9--10.  Ranks
  1--8 are protected exactly and ties retain Arm A.

This program never loads answers, supporting facts, or dataset decompositions.
Gold is joined only by a separate versioned finalizer after these passages have
been written and hashed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.retrieval.dependent import (
    instantiate_dependent_queries,
    normalize_dependency_ref,
    render_root_query,
    validate_plan_for_dependent_retrieval,
)
from kgproweight.retrieval.dependent_merge_v5 import (
    POLICY_VERSION,
    merge_dependent_passages_v5,
    passage_score_key,
)
from kgproweight.retrieval.dependent_v5 import (
    SELECTOR_VERSION,
    select_bridge_candidates_v5,
)
from kgproweight.retrieval.reranker import get_cross_encoder
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir
from scripts.pilot import audit_plan_once_dependent_retrieval as v4
from scripts.pilot.audit_iterative_bridge_retrieval import (
    _build_retriever,
    _validate_full_wiki18_assets,
)
from scripts.prepare import freeze_dependent_retrieval_v5 as v5_freeze


RUNNER_VERSION = "plan-once-dependent-retrieval-v5-precision-first-1"
REPORT_SCHEMA_VERSION = "plan-once-dependent-retrieval-v5-report-1"
DATASETS = v4.DATASETS
TARGET_TYPES = v4.TARGET_TYPES
PROTECTED_ORIGINALS = 8
CANDIDATES_PER_DEPENDENT_HOP = 2
TOTAL_PASSAGES = 10
BRIDGE_MAX_DOCS = 10
BRIDGE_MAX_CANDIDATES = 2
BRIDGE_MAX_BODY_CHARS = 1200
CE_MAX_CHARS = 1200


def _file_lock(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _runtime_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "network_access": False,
        "datasets_in_order": list(args.datasets),
        "n_per_dataset": int(args.n_per_dataset),
        "rrf_candidate_k": int(args.rrf_candidate_k),
        "step_rerank_topk": int(args.step_rerank_topk),
        "cross_encoder_model": str(Path(args.cross_encoder_model).expanduser().resolve()),
        "max_hops": int(args.max_hops),
        "max_query_variants": int(args.max_query_variants),
        "bridge_max_docs": BRIDGE_MAX_DOCS,
        "bridge_max_candidates": BRIDGE_MAX_CANDIDATES,
        "bridge_max_body_chars": BRIDGE_MAX_BODY_CHARS,
        "protected_originals": PROTECTED_ORIGINALS,
        "candidates_per_dependent_hop": CANDIDATES_PER_DEPENDENT_HOP,
        "total_passages": TOTAL_PASSAGES,
        "ce_max_chars": CE_MAX_CHARS,
        "root_hop_injection": False,
        "full_question_union_scoring": "same frozen cross encoder; ties retain Arm A",
        "generation": {
            "seed": 42,
            "decode": "greedy",
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "max_new_tokens": 512,
            "top_k_passages": 10,
        },
    }


def _validate_preregistration_runtime(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the pre-retrieval lock against the exact runtime bytes."""

    path = args.preregistration.expanduser().resolve()
    lock = _file_lock(path)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != v5_freeze.SCHEMA_VERSION:
        raise ValueError("unexpected v5 preregistration schema")
    if protocol.get("status") != v5_freeze.STATUS:
        raise ValueError("v5 preregistration is not frozen before retrieval")
    if protocol.get("scope") != v5_freeze.SCOPE:
        raise ValueError("v5 preregistration scope differs")
    experiment_ids = protocol.get("experiment_ids")
    if experiment_ids != v5_freeze.EXPERIMENT_IDS:
        raise ValueError("v5 preregistration Experiment IDs differ")
    if str(args.experiment_id or "") != str(experiment_ids["materialization"]):
        raise ValueError("materialization Experiment ID differs from preregistration")

    arg_inputs = {
        "cohort": args.cohort,
        "retrieval_contexts": args.retrieval_contexts,
        "musique_plans": args.musique_plans,
        "hotpot_plans": args.hotpot_plans,
    }
    locked_inputs = protocol.get("inputs")
    if not isinstance(locked_inputs, Mapping) or set(locked_inputs) != set(arg_inputs):
        raise ValueError("v5 preregistration input lock set differs")
    current_inputs: dict[str, dict[str, Any]] = {}
    for name, raw_path in arg_inputs.items():
        current = _file_lock(Path(raw_path))
        if current != dict(locked_inputs[name]):
            raise ValueError(f"v5 preregistered input drifted: {name}")
        current_inputs[name] = current

    locked_code = protocol.get("code")
    expected_code_names = set(v5_freeze.DEFAULT_CODE) | {"preregistration_freezer"}
    if not isinstance(locked_code, Mapping) or set(locked_code) != expected_code_names:
        raise ValueError("v5 preregistration code lock set differs")
    current_code: dict[str, dict[str, Any]] = {}
    for name, raw_lock in locked_code.items():
        if not isinstance(raw_lock, Mapping):
            raise ValueError(f"invalid v5 code lock: {name}")
        current = _file_lock(Path(str(raw_lock.get("path") or "")))
        if current != dict(raw_lock):
            raise ValueError(f"v5 preregistered code drifted: {name}")
        current_code[name] = current

    settings = _runtime_settings(args)
    if protocol.get("settings") != settings:
        raise ValueError("v5 runtime settings differ from preregistration")

    locked_models = protocol.get("models")
    if not isinstance(locked_models, Mapping):
        raise ValueError("v5 preregistration model locks are missing")
    current_models: dict[str, Any] = {}
    for name in ("cross_encoder", "strong_sft", "base_model"):
        model_lock = locked_models.get(name)
        if not isinstance(model_lock, Mapping):
            raise ValueError(f"v5 preregistration model lock is missing: {name}")
        current = artifact_identity(Path(str(model_lock.get("path") or "")))
        if current != dict(model_lock):
            raise ValueError(f"v5 preregistered model drifted: {name}")
        current_models[name] = current

    runtime_locks = {
        "preregistration": lock,
        "inputs": current_inputs,
        "code": current_code,
        "models": current_models,
        "settings": settings,
    }
    return protocol, runtime_locks


def _validate_preregistered_retrieval_assets(
    protocol: Mapping[str, Any],
    assets: Mapping[str, Any],
) -> None:
    locked = protocol.get("retrieval_assets")
    if not isinstance(locked, Mapping):
        raise ValueError("v5 preregistration retrieval assets are missing")
    if int(assets.get("expected_docs", -1)) != int(locked.get("expected_documents", -2)):
        raise ValueError("Wiki18 expected document count differs from preregistration")
    if assets.get("counts") != locked.get("counts"):
        raise ValueError("Wiki18 counts differ from preregistration")
    paths = assets.get("paths") or {}
    for asset_name, report_name in (
        ("corpus", "corpus"),
        ("dense_index", "dense"),
        ("bm25_index", "bm25"),
    ):
        item = locked.get(asset_name)
        if not isinstance(item, Mapping):
            raise ValueError(f"v5 preregistration lacks Wiki18 {asset_name}")
        if Path(str(paths.get(report_name) or "")).resolve() != Path(str(item.get("path") or "")).resolve():
            raise ValueError(f"Wiki18 {asset_name} path differs from preregistration")
    for asset_name in ("corpus", "dense_index"):
        item = locked[asset_name]
        path = Path(str(item["path"]))
        if path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"Wiki18 {asset_name} size differs from preregistration")
    if artifact_identity(Path(str(locked["bm25_index"]["path"]))) != dict(locked["bm25_index"]):
        raise ValueError("Wiki18 BM25 artifact differs from preregistration")


# These aliases make the reused boundary explicit and leave the v4 source
# untouched.  They also keep the v5 arm hashes byte-compatible with the old
# finalizer/evaluator contract.
_load_inputs = v4._load_inputs
_sha256_file = v4._sha256_file
_sha256_json = v4._sha256_json
_write_jsonl = v4._write_jsonl
_step_id = v4._step_id
_dependencies = v4._dependencies
_record_slot_values = v4._record_slot_values


def _normalised_dependencies(step: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for raw in _dependencies(step):
        value = normalize_dependency_ref(raw)
        if value is None:
            raise ValueError(f"invalid dependency reference {raw!r}")
        if value not in result:
            result.append(value)
    return result


def _step_schedule(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build a deterministic topological schedule from a validated plan."""

    produced_depth: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        slot = normalize_dependency_ref(step.get("output_slot"))
        if slot is None:
            raise ValueError(f"step_{index} has no canonical output slot")
        dependencies = _normalised_dependencies(step)
        missing = [value for value in dependencies if value not in produced_depth]
        if missing:
            raise ValueError(f"step_{index} has unresolved dependencies {missing}")
        depth = 1 + max((produced_depth[value] for value in dependencies), default=0)
        produced_depth[slot] = depth
        records.append({
            "step_index": index,
            "step": dict(step),
            "slot": slot,
            "dependencies": dependencies,
            "dependency_depth": depth,
            "consumers": [],
        })

    for producer in records:
        producer["consumers"] = [
            dict(candidate["step"])
            for candidate in records
            if candidate["step_index"] > producer["step_index"]
            and producer["slot"] in candidate["dependencies"]
        ]
    return records


def _passage_text(passage: Mapping[str, Any]) -> str:
    return str(passage.get("contents") or passage.get("text") or "")


def _predict_scores(
    cross_encoder: Any,
    pairs: Sequence[tuple[str, str]],
) -> list[float]:
    """Call the configured CE without the library's BM25 fallback."""

    if not pairs:
        return []
    raw = cross_encoder.predict(list(pairs), show_progress_bar=False)
    try:
        values = [float(value) for value in raw]
    except TypeError:
        values = [float(raw)]
    if len(values) != len(pairs):
        raise RuntimeError(
            f"cross encoder returned {len(values)}/{len(pairs)} scores"
        )
    return values


def _rerank_one_query(
    query: str,
    passages: Sequence[Mapping[str, Any]],
    *,
    cross_encoder: Any,
    topk: int,
) -> list[dict[str, Any]]:
    candidates = [dict(value) for value in passages]
    if not candidates:
        return []
    scores = _predict_scores(
        cross_encoder,
        [(query, _passage_text(value)[:CE_MAX_CHARS]) for value in candidates],
    )
    order = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))
    return [candidates[index] for index in order[:topk]]


def _base_detail(
    row: Mapping[str, Any],
    *,
    target_type: str,
    plan: Mapping[str, Any],
    validation_errors: Sequence[str],
    has_dependent_step: bool,
) -> dict[str, Any]:
    return {
        "dataset": str(row["dataset"]),
        "qid": str(row["qid"]),
        "question_sha256": str(row["question_sha256"]),
        "target_type": target_type,
        "plan_sha256": _sha256_json(plan),
        "plan_validation_errors": list(validation_errors),
        "has_dependent_step": has_dependent_step,
        "gold_access": False,
        "execution_status": "pending",
        "fallback_exact": False,
        "fallback_reason": None,
        "plan_executable": not validation_errors,
        "dependent_query_count": 0,
        "second_hop_query_count": 0,
        "new_dependent_candidate_count": 0,
        "hops": [],
        "selector_summary": {
            "required_producers": 0,
            "accepted_producers": 0,
            "raw_candidates": 0,
            "accepted_candidates": 0,
            "rejected_candidates": 0,
        },
        "merge": None,
        "safety": None,
    }


def _exact_safety(arm_a: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "output_count": len(arm_a),
        "prefix8_exact": True,
        "unauthorized_original_displacements": 0,
        "root_passages_injected": 0,
        "fallback_exact": True,
    }


def _fallback(
    arm_a: Sequence[Mapping[str, Any]],
    detail: dict[str, Any],
    *,
    status: str,
    reason: str,
    plan_executable: bool,
    error: Exception | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    passages = [deepcopy(dict(value)) for value in arm_a]
    detail.update({
        "execution_status": status,
        "fallback_exact": True,
        "fallback_reason": reason,
        "plan_executable": plan_executable,
        "new_dependent_candidate_count": 0,
        "safety": _exact_safety(passages),
        "arm_b_passages_sha256": _sha256_json(passages),
    })
    if error is not None:
        detail["error"] = {"type": type(error).__name__, "message": str(error)}
    return passages, detail


def _merge_safety(
    arm_a: Sequence[Mapping[str, Any]],
    arm_b: Sequence[Mapping[str, Any]],
    hop_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    original = [dict(value) for value in arm_a]
    output = [dict(value) for value in arm_b]
    original_keys = [passage_score_key(value) for value in original]
    output_keys = [passage_score_key(value) for value in output]
    root_keys = {
        passage_score_key(value)
        for hop in hop_results
        if not bool(hop.get("dependencies"))
        for value in list(hop.get("passages") or [])
    }
    dependent_keys = {
        passage_score_key(value)
        for hop in hop_results
        if bool(hop.get("dependencies"))
        for value in list(hop.get("passages") or [])[:CANDIDATES_PER_DEPENDENT_HOP]
    }
    new_output_keys = set(output_keys) - set(original_keys)
    prefix_mismatches = sum(
        index >= len(output) or output[index] != original[index]
        for index in range(min(PROTECTED_ORIGINALS, len(original)))
    )
    root_only_injected = new_output_keys & (root_keys - dependent_keys)
    return {
        "output_count": len(output),
        "prefix8_exact": prefix_mismatches == 0,
        "prefix8_mismatch_count": prefix_mismatches,
        "unauthorized_original_displacements": prefix_mismatches,
        "root_passages_injected": len(root_only_injected),
        "root_only_injected_document_keys": sorted(root_only_injected),
        "fallback_exact": output == original,
    }


def _score_document_unions_once(
    states: Sequence[dict[str, Any]],
    *,
    cross_encoder: Any,
) -> dict[int, dict[str, float]]:
    """Score all row-local A/dependent unions in one full-question CE call."""

    pairs: list[tuple[str, str]] = []
    owners: list[tuple[int, str]] = []
    for state in states:
        unique: dict[str, dict[str, Any]] = {}
        for passage in state["arm_a"][PROTECTED_ORIGINALS:]:
            unique.setdefault(passage_score_key(passage), dict(passage))
        for hop in state["hop_results"]:
            if not bool(hop.get("dependencies")):
                continue
            for passage in list(hop.get("passages") or [])[:CANDIDATES_PER_DEPENDENT_HOP]:
                unique.setdefault(passage_score_key(passage), dict(passage))
        for key, passage in unique.items():
            pairs.append((state["question"], _passage_text(passage)[:CE_MAX_CHARS]))
            owners.append((int(state["row_index"]), key))

    scores = _predict_scores(cross_encoder, pairs)
    by_row: dict[int, dict[str, float]] = {}
    for (row_index, key), score in zip(owners, scores):
        by_row.setdefault(row_index, {})[key] = score
    return by_row


def _execute_rows_batched_v5(
    rows: Sequence[Mapping[str, Any]],
    retriever: Any,
    args: argparse.Namespace,
    *,
    cross_encoder: Any,
) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
    """Execute all plans in dependency-depth batches under v5 safeguards."""

    if not hasattr(retriever, "batch_search"):
        raise TypeError("v5 dependent retrieval requires retriever.batch_search")

    outcomes: list[tuple[list[dict[str, Any]], dict[str, Any]] | None] = [
        None for _ in rows
    ]
    states: list[dict[str, Any]] = []
    for row_index, source in enumerate(rows):
        row = dict(source)
        dataset = str(row["dataset"])
        target_type = TARGET_TYPES[dataset]
        plan = dict(row["plan"])
        errors = validate_plan_for_dependent_retrieval(plan, target_type)
        raw_steps = plan.get("steps")
        has_dependent_step = bool(
            not errors
            and isinstance(raw_steps, list)
            and any(
                isinstance(step, Mapping) and bool(_dependencies(step))
                for step in raw_steps
            )
        )
        detail = _base_detail(
            row,
            target_type=target_type,
            plan=plan,
            validation_errors=errors,
            has_dependent_step=has_dependent_step,
        )
        arm_a = [deepcopy(dict(value)) for value in row["arm_a_passages"]]
        if len(arm_a) != TOTAL_PASSAGES:
            raise ValueError(
                f"{dataset}::{row['qid']} expected {TOTAL_PASSAGES} Arm-A passages, "
                f"got {len(arm_a)}"
            )
        if errors:
            outcomes[row_index] = _fallback(
                arm_a,
                detail,
                status="fallback_plan_invalid",
                reason="plan_invalid",
                plan_executable=False,
            )
            continue
        steps = list(plan.get("steps") or [])
        if not steps or len(steps) > int(args.max_hops):
            outcomes[row_index] = _fallback(
                arm_a,
                detail,
                status="fallback_invalid_hop_count",
                reason="invalid_hop_count",
                plan_executable=False,
            )
            continue
        if not has_dependent_step:
            outcomes[row_index] = _fallback(
                arm_a,
                detail,
                status="fallback_no_dependent_step",
                reason="no_dependent_step",
                plan_executable=True,
            )
            continue
        try:
            schedule = _step_schedule(steps)
        except Exception as exc:  # row-local malformed plan despite validation
            outcomes[row_index] = _fallback(
                arm_a,
                detail,
                status="fallback_execution_error",
                reason="execution_error",
                plan_executable=True,
                error=exc,
            )
            continue
        states.append({
            "row_index": row_index,
            "dataset": dataset,
            "qid": str(row["qid"]),
            "question": str(row["question"]),
            "target_type": target_type,
            "arm_a": arm_a,
            "detail": detail,
            "schedule": schedule,
            "slot_values": {},
            "hop_results": [],
            "row_error": None,
            "bridge_abstain": False,
        })

    max_depth = max(
        (
            int(record["dependency_depth"])
            for state in states
            for record in state["schedule"]
        ),
        default=0,
    )
    for depth in range(1, max_depth + 1):
        flat_queries: list[str] = []
        owners: list[tuple[dict[str, Any], dict[str, Any], str]] = []
        queries_by_task: dict[tuple[int, int], list[str]] = {}

        for state in states:
            if state["row_error"] is not None or state["bridge_abstain"]:
                continue
            for record in state["schedule"]:
                if int(record["dependency_depth"]) != depth:
                    continue
                step = record["step"]
                try:
                    queries = (
                        instantiate_dependent_queries(
                            step,
                            state["target_type"],
                            state["slot_values"],
                            max_variants=int(args.max_query_variants),
                        )
                        if record["dependencies"]
                        else [render_root_query(step, state["target_type"])]
                    )
                    queries = list(dict.fromkeys(
                        str(value).strip() for value in queries if str(value).strip()
                    ))
                    if not queries:
                        raise ValueError(f"no executable query for {record['slot']}")
                    task_key = (int(state["row_index"]), int(record["step_index"]))
                    queries_by_task[task_key] = queries
                    for query in queries:
                        flat_queries.append(query)
                        owners.append((state, record, query))
                except Exception as exc:
                    state["row_error"] = exc

        if not flat_queries:
            continue

        # This call is deliberately outside every row-local exception handler.
        # A failed full-index batch invalidates the run instead of creating a
        # deceptively successful all-fallback arm.
        raw_batches = [list(result) for result in retriever.batch_search(flat_queries)]
        if len(raw_batches) != len(flat_queries):
            raise RuntimeError(
                f"batch retriever returned {len(raw_batches)}/{len(flat_queries)} rows"
            )
        ranked_batches = [
            _rerank_one_query(
                query,
                raw,
                cross_encoder=cross_encoder,
                topk=int(args.step_rerank_topk),
            )
            for query, raw in zip(flat_queries, raw_batches)
        ]

        grouped: dict[
            tuple[int, int],
            list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]],
        ] = {}
        for (state, record, query), raw, ranked in zip(
            owners, raw_batches, ranked_batches
        ):
            key = (int(state["row_index"]), int(record["step_index"]))
            grouped.setdefault(key, []).append((query, raw, ranked))

        for state in states:
            if state["row_error"] is not None or state["bridge_abstain"]:
                continue
            for record in state["schedule"]:
                if int(record["dependency_depth"]) != depth:
                    continue
                key = (int(state["row_index"]), int(record["step_index"]))
                if key not in grouped:
                    continue
                try:
                    queries = queries_by_task[key]
                    current_results: list[dict[str, Any]] = []
                    query_details: list[dict[str, Any]] = []
                    seen: set[str] = set()
                    for query, raw, ranked in grouped[key]:
                        for passage in ranked:
                            document_key = passage_score_key(passage)
                            if document_key not in seen:
                                seen.add(document_key)
                                current_results.append(dict(passage))
                        query_details.append({
                            "query": query,
                            "raw_count": len(raw),
                            "reranked_count": len(ranked),
                            "retrieved_ids": [str(value.get("id") or "") for value in ranked],
                        })
                    hop_result = {
                        "hop_id": _step_id(record["step"], int(record["step_index"])),
                        "step_index": int(record["step_index"]),
                        "query": " || ".join(queries),
                        "dependencies": list(record["dependencies"]),
                        "dependency_depth": int(record["dependency_depth"]),
                        "is_dependent": bool(record["dependencies"]),
                        "passages": current_results,
                    }
                    state["hop_results"].append(hop_result)
                    selector_telemetry: dict[str, Any] | None = None
                    candidates: list[dict[str, Any]] = []
                    if record["consumers"]:
                        candidates, selector_telemetry = select_bridge_candidates_v5(
                            step=record["step"],
                            consumers=record["consumers"],
                            target_type=state["target_type"],
                            query=" ; ".join(queries),
                            question=state["question"],
                            passages=current_results,
                            max_docs=BRIDGE_MAX_DOCS,
                            max_candidates=BRIDGE_MAX_CANDIDATES,
                            max_body_chars=BRIDGE_MAX_BODY_CHARS,
                        )
                        summary = state["detail"]["selector_summary"]
                        summary["required_producers"] += 1
                        summary["raw_candidates"] += int(
                            selector_telemetry.get("raw_candidate_count") or 0
                        )
                        summary["accepted_candidates"] += int(
                            selector_telemetry.get("accepted_count") or 0
                        )
                        summary["rejected_candidates"] += sum(
                            value.get("decision") == "reject"
                            for value in selector_telemetry.get("candidate_decisions") or []
                        )
                        if not candidates:
                            state["bridge_abstain"] = True
                        else:
                            summary["accepted_producers"] += 1
                            _record_slot_values(
                                state["slot_values"],
                                step=record["step"],
                                index=int(record["step_index"]),
                                candidates=candidates,
                            )
                    state["detail"]["hops"].append({
                        "hop_id": hop_result["hop_id"],
                        "step_index": hop_result["step_index"],
                        "dependencies": list(record["dependencies"]),
                        "dependency_depth": int(record["dependency_depth"]),
                        "is_dependent": bool(record["dependencies"]),
                        "queries": query_details,
                        "bridge_required": bool(record["consumers"]),
                        "bridge_candidates": candidates,
                        "bridge_selector": selector_telemetry,
                    })
                except Exception as exc:
                    state["row_error"] = exc

    ready_to_score: list[dict[str, Any]] = []
    for state in states:
        state["hop_results"].sort(key=lambda value: int(value["step_index"]))
        state["detail"]["hops"].sort(key=lambda value: int(value["step_index"]))
        state["detail"]["dependent_query_count"] = sum(
            len(hop["queries"])
            for hop in state["detail"]["hops"]
            if hop["dependencies"]
        )
        state["detail"]["second_hop_query_count"] = state["detail"][
            "dependent_query_count"
        ]
        if state["row_error"] is not None:
            outcomes[state["row_index"]] = _fallback(
                state["arm_a"],
                state["detail"],
                status="fallback_execution_error",
                reason="execution_error",
                plan_executable=True,
                error=state["row_error"],
            )
        elif state["bridge_abstain"]:
            outcomes[state["row_index"]] = _fallback(
                state["arm_a"],
                state["detail"],
                status="fallback_bridge_abstain",
                reason="bridge_abstain",
                plan_executable=True,
            )
        else:
            ready_to_score.append(state)

    # One and only one full-question scoring call is made after all retrieval
    # finishes.  Any model/predict failure propagates and marks the entire run
    # FAILED_RUNTIME in main().
    scores_by_row = _score_document_unions_once(
        ready_to_score,
        cross_encoder=cross_encoder,
    )
    for state in ready_to_score:
        merged, merge_telemetry = merge_dependent_passages_v5(
            state["arm_a"],
            state["hop_results"],
            scores_by_row.get(int(state["row_index"]), {}),
            protected_originals=PROTECTED_ORIGINALS,
            candidates_per_hop=CANDIDATES_PER_DEPENDENT_HOP,
            total_passages=TOTAL_PASSAGES,
        )
        safety = _merge_safety(state["arm_a"], merged, state["hop_results"])
        if (
            len(merged) != TOTAL_PASSAGES
            or not safety["prefix8_exact"]
            or int(safety["unauthorized_original_displacements"]) != 0
            or int(safety["root_passages_injected"]) != 0
        ):
            raise RuntimeError(
                f"v5 safety invariant failed for {state['dataset']}::{state['qid']}: "
                f"{safety}"
            )
        changed = bool(merge_telemetry.get("changed"))
        state["detail"].update({
            "execution_status": (
                "executed_changed" if changed else "fallback_no_candidate_strictly_better"
            ),
            "fallback_exact": not changed,
            "fallback_reason": None if changed else str(merge_telemetry.get("fallback_reason")),
            "merge": merge_telemetry,
            "safety": safety,
            "new_dependent_candidate_count": len(
                merge_telemetry.get("selected_new") or []
            ),
            "arm_b_passages_sha256": _sha256_json(merged),
        })
        outcomes[state["row_index"]] = ([dict(value) for value in merged], state["detail"])

    if any(value is None for value in outcomes):
        raise RuntimeError("v5 execution did not produce every row outcome")
    return [value for value in outcomes if value is not None]


def _dry_run(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    report = dict(v4._dry_run(rows, args))
    report.update({
        "runner_version": RUNNER_VERSION,
        "selector_version": SELECTOR_VERSION,
        "merge_policy_version": POLICY_VERSION,
        "v5_policy_not_executed": True,
    })
    return report


def _aggregate_dataset(
    dataset: str,
    execution_rows: Sequence[Mapping[str, Any]],
    arm_a_rows: Sequence[Mapping[str, Any]],
    arm_b_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    current = [row for row in execution_rows if row["dataset"] == dataset]
    arm_a = {str(row["qid"]): row for row in arm_a_rows if row["dataset"] == dataset}
    arm_b = {str(row["qid"]): row for row in arm_b_rows if row["dataset"] == dataset}
    fallbacks = [row for row in current if bool(row.get("fallback_exact"))]
    changed_count = sum(
        arm_a[str(row["qid"])]["passages_sha256"]
        != arm_b[str(row["qid"])]["passages_sha256"]
        for row in current
    )
    plan_executable_count = sum(bool(row["plan_executable"]) for row in current)
    dependent_eligible = sum(bool(row["has_dependent_step"]) for row in current)
    dependent_nonempty = sum(
        bool(row["has_dependent_step"])
        and int(row["second_hop_query_count"]) > 0
        for row in current
    )
    return {
        "n": len(current),
        "executed": sum(row["execution_status"] == "executed_changed" for row in current),
        "fallback": len(fallbacks),
        "fallback_plan_invalid": sum(
            row["execution_status"] == "fallback_plan_invalid" for row in current
        ),
        "fallback_invalid_hop_count": sum(
            row["execution_status"] == "fallback_invalid_hop_count" for row in current
        ),
        "fallback_no_dependent_step": sum(
            row["execution_status"] == "fallback_no_dependent_step" for row in current
        ),
        "fallback_bridge_abstain": sum(
            row["execution_status"] == "fallback_bridge_abstain" for row in current
        ),
        "fallback_no_candidate_strictly_better": sum(
            row["execution_status"] == "fallback_no_candidate_strictly_better"
            for row in current
        ),
        "fallback_execution_error": sum(
            row["execution_status"] == "fallback_execution_error" for row in current
        ),
        "runtime_errors": sum("error" in row for row in current),
        "arm_b_changed": changed_count,
        "retained_new_dependent_document_question_rate": (
            changed_count / len(current) if current else 0.0
        ),
        "plan_executable": plan_executable_count,
        "plan_executable_rate": (
            plan_executable_count / len(current) if current else 0.0
        ),
        "dependent_step_eligible": dependent_eligible,
        "dependent_step_query_nonempty": dependent_nonempty,
        "dependent_hop_query_nonempty_rate": (
            dependent_nonempty / dependent_eligible if dependent_eligible else None
        ),
        "bridge_selector_required_producers": sum(
            int(row["selector_summary"]["required_producers"]) for row in current
        ),
        "bridge_selector_accepted_producers": sum(
            int(row["selector_summary"]["accepted_producers"]) for row in current
        ),
        "bridge_selector_accepted_rows": sum(
            int(row["selector_summary"]["accepted_producers"]) > 0 for row in current
        ),
        "all_top10": all(int(row["safety"]["output_count"]) == 10 for row in current),
        "prefix8_exact": all(bool(row["safety"]["prefix8_exact"]) for row in current),
        "unauthorized_original_displacements": sum(
            int(row["safety"]["unauthorized_original_displacements"]) for row in current
        ),
        "root_passages_injected": sum(
            int(row["safety"]["root_passages_injected"]) for row in current
        ),
        "fallback_exact": all(
            arm_a[str(row["qid"])]["passages_sha256"]
            == arm_b[str(row["qid"])]["passages_sha256"]
            for row in fallbacks
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path(
            "outputs/audits/inference_proofkg_v1_n900_seed42_preregistration/"
            "pilot.question_only.jsonl"
        ),
    )
    parser.add_argument(
        "--retrieval_contexts",
        type=Path,
        default=Path(
            "outputs/audits/inference_proofkg_v1_n900_seed42_preregistration/"
            "retrieval_contexts.jsonl"
        ),
    )
    parser.add_argument(
        "--musique_plans",
        type=Path,
        default=Path(
            "outputs/audits/inference_proofkg_v1_pilot30x3_plans_v1/"
            "predictions.question_only.jsonl"
        ),
    )
    parser.add_argument(
        "--hotpot_plans",
        type=Path,
        default=Path(
            "outputs/audits/inference_proofkg_hotpot_relation_graph_pilot30_v2_plans/"
            "predictions.question_only.jsonl"
        ),
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--n_per_dataset", type=int, default=30)
    parser.add_argument("--corpus_path", default="indexes_wiki18/corpus_flashrag.jsonl")
    parser.add_argument("--dense_index_path", default="indexes_wiki18/e5_fp16.dat")
    parser.add_argument("--bm25_index_path", default="indexes_wiki18/bm25")
    parser.add_argument("--expected_docs", type=int, default=21_015_324)
    parser.add_argument("--rrf_candidate_k", type=int, default=100)
    parser.add_argument("--step_rerank_topk", type=int, default=10)
    parser.add_argument("--cross_encoder_model", default="models/bge-reranker-v2-m3")
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path(
            "outputs/audits/"
            "subquestion_dependent_retrieval_pilot30x2_seed42_v5_preregistration/"
            "protocol.json"
        ),
    )
    parser.add_argument("--max_hops", type=int, default=4)
    parser.add_argument("--max_query_variants", type=int, default=2)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--experiment_id")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate joins/query rendering only; no retrieval, CE, or output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        raise SystemExit("--datasets contains duplicates")
    if args.n_per_dataset <= 0 or args.rrf_candidate_k <= 0:
        raise SystemExit("sample and retrieval sizes must be positive")
    if args.max_hops <= 0 or args.max_query_variants <= 0 or args.step_rerank_topk <= 0:
        raise SystemExit("hop/query/rerank sizes must be positive")
    rows = _load_inputs(args)
    if args.dry_run:
        print(json.dumps(_dry_run(rows, args), ensure_ascii=False, indent=2))
        return
    if tuple(args.datasets) != DATASETS or int(args.n_per_dataset) != 30 or len(rows) != 60:
        raise SystemExit("formal v5 materialisation is locked to HotpotQA30 + MuSiQue30")
    if args.output_dir is None or not str(args.experiment_id or "").strip():
        raise SystemExit("formal materialisation requires --output_dir and --experiment_id")

    preregistration, runtime_locks = _validate_preregistration_runtime(args)

    assets = _validate_full_wiki18_assets(
        args.corpus_path,
        args.dense_index_path,
        args.bm25_index_path,
        expected_docs=int(args.expected_docs),
    )
    _validate_preregistered_retrieval_assets(preregistration, assets)
    run_dir, experiment_id = prepare_new_run_dir(
        args.output_dir,
        experiment_id=args.experiment_id,
        extra={
            "phase": "plan_once_dependent_retrieval_v5_materialisation",
            "gold_access": False,
        },
    )
    try:
        retriever = _build_retriever(
            args.datasets[0],
            int(args.rrf_candidate_k),
            corpus_path=args.corpus_path,
            dense_index_path=args.dense_index_path,
            bm25_index_path=args.bm25_index_path,
        )
        # Loading is not wrapped in a BM25 fallback.  The same cached instance
        # serves per-hop reranking and the one final comparable-logit call.
        cross_encoder = get_cross_encoder(str(args.cross_encoder_model))
        outcomes = _execute_rows_batched_v5(
            rows,
            retriever,
            args,
            cross_encoder=cross_encoder,
        )

        arm_a_rows: list[dict[str, Any]] = []
        arm_b_rows: list[dict[str, Any]] = []
        execution_rows: list[dict[str, Any]] = []
        for index, (row, outcome) in enumerate(zip(rows, outcomes), start=1):
            arm_b, detail = outcome
            common = {
                "row_id": f"dependent-retrieval-pilot::{row['dataset']}::{row['qid']}",
                "question_key": f"{row['dataset']}::{row['qid']}",
                "dataset": row["dataset"],
                "qid": row["qid"],
                "question": row["question"],
                "question_sha256": row["question_sha256"],
                "split": row["split"],
                "gold_access": False,
                "kg_subgraph": row["legacy_kg"],
                "legacy_kg_sha256": row["legacy_kg_sha256"],
            }
            arm_a = [dict(value) for value in row["arm_a_passages"]]
            arm_a_rows.append({
                **common,
                "arm": "A_question_only",
                "retrieved_passages": arm_a,
                "passages_sha256": _sha256_json(arm_a),
            })
            arm_b_rows.append({
                **common,
                "arm": "B_dependent",
                "retrieved_passages": arm_b,
                "passages_sha256": _sha256_json(arm_b),
                "fallback_to_a": bool(detail["fallback_exact"]),
                "retrieval_trace": {
                    "plan_executable": bool(detail["plan_executable"]),
                    "has_dependent_step": bool(detail["has_dependent_step"]),
                    "dependent_query_count": int(detail["dependent_query_count"]),
                    "second_hop_query_count": int(detail["second_hop_query_count"]),
                    "new_dependent_candidate_count": int(
                        detail["new_dependent_candidate_count"]
                    ),
                    "fallback_reason": detail["fallback_reason"],
                    "bridge_selector_required_producers": int(
                        detail["selector_summary"]["required_producers"]
                    ),
                    "bridge_selector_accepted_producers": int(
                        detail["selector_summary"]["accepted_producers"]
                    ),
                    "prefix8_exact": bool(detail["safety"]["prefix8_exact"]),
                    "unauthorized_original_displacements": int(
                        detail["safety"]["unauthorized_original_displacements"]
                    ),
                    "root_passages_injected": int(
                        detail["safety"]["root_passages_injected"]
                    ),
                },
            })
            execution_rows.append(detail)
            print(
                f"v5 dependent retrieval {index}/{len(rows)} "
                f"{row['dataset']}::{row['qid']} status={detail['execution_status']}",
                flush=True,
            )

        arm_a_path = run_dir / "arm_a.jsonl"
        arm_b_path = run_dir / "arm_b.jsonl"
        execution_path = run_dir / "execution_details.jsonl"
        _write_jsonl(arm_a_path, arm_a_rows)
        _write_jsonl(arm_b_path, arm_b_rows)
        _write_jsonl(execution_path, execution_rows)
        by_dataset = {
            dataset: _aggregate_dataset(
                dataset, execution_rows, arm_a_rows, arm_b_rows
            )
            for dataset in args.datasets
        }
        safety_summary = {
            "all_top10": all(value["all_top10"] for value in by_dataset.values()),
            "prefix8_exact": all(value["prefix8_exact"] for value in by_dataset.values()),
            "unauthorized_original_displacements": sum(
                int(value["unauthorized_original_displacements"])
                for value in by_dataset.values()
            ),
            "root_passages_injected": sum(
                int(value["root_passages_injected"]) for value in by_dataset.values()
            ),
            "fallback_exact": all(value["fallback_exact"] for value in by_dataset.values()),
            "runtime_errors": sum(
                int(value["runtime_errors"]) for value in by_dataset.values()
            ),
        }
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "runner_version": RUNNER_VERSION,
            "selector_version": SELECTOR_VERSION,
            "merge_policy_version": POLICY_VERSION,
            "experiment_id": experiment_id,
            "status": "COMPLETE_INPUTS_NOT_ANSWER_EVALUATED",
            "development_only": True,
            "gold_access": False,
            "canonical_pipeline_modified": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "preregistration": runtime_locks["preregistration"],
            "runtime_locks": runtime_locks,
            "inputs": {
                name: {"path": str(path.resolve()), "sha256": _sha256_file(path.resolve())}
                for name, path in {
                    "cohort": args.cohort,
                    "retrieval_contexts": args.retrieval_contexts,
                    "musique_plans": args.musique_plans,
                    "hotpot_plans": args.hotpot_plans,
                }.items()
            },
            "retrieval_assets": assets,
            "settings": runtime_locks["settings"],
            "by_dataset": by_dataset,
            "safety_summary": safety_summary,
            "outputs": {
                "arm_a": {"path": str(arm_a_path), "sha256": _sha256_file(arm_a_path)},
                "arm_b": {"path": str(arm_b_path), "sha256": _sha256_file(arm_b_path)},
                "execution_details": {
                    "path": str(execution_path),
                    "sha256": _sha256_file(execution_path),
                },
            },
            "scientific_boundary": (
                "Development retrieval inputs only. No Gold was loaded and no answer "
                "generation or effectiveness scoring has run. v4 artifacts remain unchanged."
            ),
        }
        report_path = run_dir / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        dump_manifest(
            run_dir,
            status=report["status"],
            extra={
                "experiment_id": experiment_id,
                "phase": "plan_once_dependent_retrieval_v5_materialisation",
                "report_sha256": _sha256_file(report_path),
            },
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "by_dataset": by_dataset,
                    "safety_summary": safety_summary,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception as exc:
        dump_manifest(
            run_dir,
            status="FAILED_RUNTIME",
            extra={
                "experiment_id": experiment_id,
                "phase": "plan_once_dependent_retrieval_v5_materialisation",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise


if __name__ == "__main__":
    main()
