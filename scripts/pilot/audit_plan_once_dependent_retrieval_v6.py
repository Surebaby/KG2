#!/usr/bin/env python
"""Materialise the question-anchored dependent-retrieval v6 pilot.

This append-only development runner keeps the frozen v5 safety envelope but
treats retrieval-derived entity strings as bounded query hints, never as
asserted evidence.  Every dependent query retains the exact original question
as its prefix.  Query variants keep separate top-k candidate slots before one
full-question cross-encoder comparison against Arm-A ranks 9--10.

The runner never loads answers, supporting facts, decompositions, or targets.
Gold may be joined only by the separately versioned v6 finalizer after the
Gold-free outputs and mechanism gates have been frozen.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from kgproweight.retrieval.dependent import (
    render_root_query,
    validate_plan_for_dependent_retrieval,
)
from kgproweight.retrieval.dependent_merge_v6 import (
    POLICY_VERSION,
    merge_dependent_passages_v6,
    passage_score_key,
)
from kgproweight.retrieval.dependent_v6 import (
    QUERY_HINT_POLICY_VERSION,
    QUERY_RENDERER_VERSION,
    render_question_anchored_queries_v6,
    select_bridge_query_hints_v6,
)
from kgproweight.retrieval.reranker import get_cross_encoder
from kgproweight.retrieval.hybrid import EVAL_RETRIEVAL_QUERY_MAX_LENGTH
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir
from kgproweight.utils.paths import model_path
from scripts.pilot import audit_plan_once_dependent_retrieval_v5 as v5
from scripts.pilot.audit_iterative_bridge_retrieval import (
    _build_retriever,
    _validate_full_wiki18_assets,
)
from scripts.prepare import freeze_dependent_retrieval_v6 as v6_freeze


RUNNER_VERSION = "plan-once-dependent-retrieval-v6-question-anchored-1"
REPORT_SCHEMA_VERSION = "plan-once-dependent-retrieval-v6-report-1"
DATASETS = v5.DATASETS
TARGET_TYPES = v5.TARGET_TYPES
PROTECTED_ORIGINALS = 8
CANDIDATES_PER_QUERY_VARIANT = 2
TOTAL_PASSAGES = 10
BRIDGE_MAX_DOCS = 10
BRIDGE_MAX_HINTS = 2
BRIDGE_MAX_BODY_CHARS = 1200
CE_MAX_CHARS = 1200
QUESTION_ANCHOR_TEMPLATE = "{original_question}\n{subquery}"
QUERY_POLICY_VERSION = (
    f"{QUERY_HINT_POLICY_VERSION}+{QUERY_RENDERER_VERSION}"
)

_load_inputs = v5._load_inputs
_sha256_file = v5._sha256_file
_sha256_json = v5._sha256_json
_write_jsonl = v5._write_jsonl
_step_id = v5._step_id
_dependencies = v5._dependencies
_record_slot_values = v5._record_slot_values
_step_schedule = v5._step_schedule
_passage_text = v5._passage_text
_predict_scores = v5._predict_scores
_rerank_one_query = v5._rerank_one_query


def _file_lock(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _runtime_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "network_access": False,
        "datasets_in_order": list(args.datasets),
        "n_per_dataset": int(args.n_per_dataset),
        "rrf_candidate_k": int(args.rrf_candidate_k),
        "step_rerank_topk": int(args.step_rerank_topk),
        "cross_encoder_model": str(Path(args.cross_encoder_model).expanduser().resolve()),
        "retrieval_encoder_model": str(
            Path(args.retrieval_encoder_path).expanduser().resolve()
        ),
        "retrieval_query_max_length": EVAL_RETRIEVAL_QUERY_MAX_LENGTH,
        "max_hops": int(args.max_hops),
        "max_query_variants": int(args.max_query_variants),
        "bridge_max_docs": BRIDGE_MAX_DOCS,
        "bridge_max_hints": BRIDGE_MAX_HINTS,
        "bridge_max_body_chars": BRIDGE_MAX_BODY_CHARS,
        "protected_originals": PROTECTED_ORIGINALS,
        "candidates_per_query_variant": CANDIDATES_PER_QUERY_VARIANT,
        "total_passages": TOTAL_PASSAGES,
        "ce_max_chars": CE_MAX_CHARS,
        "root_hop_injection": False,
        "question_anchor_template": QUESTION_ANCHOR_TEMPLATE,
        "no_hint_relation_fallback": True,
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
    path = args.preregistration.expanduser().resolve()
    lock = _file_lock(path)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != v6_freeze.SCHEMA_VERSION:
        raise ValueError("unexpected v6 preregistration schema")
    if protocol.get("status") != v6_freeze.STATUS:
        raise ValueError("v6 preregistration is not frozen before retrieval")
    if protocol.get("scope") != v6_freeze.SCOPE:
        raise ValueError("v6 preregistration scope differs")
    experiment_ids = protocol.get("experiment_ids")
    if experiment_ids != v6_freeze.EXPERIMENT_IDS:
        raise ValueError("v6 preregistration Experiment IDs differ")
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
        raise ValueError("v6 preregistration input lock set differs")
    current_inputs: dict[str, dict[str, Any]] = {}
    for name, raw_path in arg_inputs.items():
        current = _file_lock(Path(raw_path))
        if current != dict(locked_inputs[name]):
            raise ValueError(f"v6 preregistered input drifted: {name}")
        current_inputs[name] = current

    locked_code = protocol.get("code")
    expected_code_names = set(v6_freeze.DEFAULT_CODE) | {"preregistration_freezer"}
    if not isinstance(locked_code, Mapping) or set(locked_code) != expected_code_names:
        raise ValueError("v6 preregistration code lock set differs")
    current_code: dict[str, dict[str, Any]] = {}
    for name, raw_lock in locked_code.items():
        if not isinstance(raw_lock, Mapping):
            raise ValueError(f"invalid v6 code lock: {name}")
        current = _file_lock(Path(str(raw_lock.get("path") or "")))
        if current != dict(raw_lock):
            raise ValueError(f"v6 preregistered code drifted: {name}")
        current_code[name] = current

    settings = _runtime_settings(args)
    if protocol.get("settings") != settings:
        raise ValueError("v6 runtime settings differ from preregistration")

    locked_models = protocol.get("models")
    if not isinstance(locked_models, Mapping):
        raise ValueError("v6 preregistration model locks are missing")
    current_models: dict[str, Any] = {}
    for name in ("retrieval_encoder", "cross_encoder", "strong_sft", "base_model"):
        model_lock = locked_models.get(name)
        if not isinstance(model_lock, Mapping):
            raise ValueError(f"v6 preregistration model lock is missing: {name}")
        current = artifact_identity(Path(str(model_lock.get("path") or "")))
        if current != dict(model_lock):
            raise ValueError(f"v6 preregistered model drifted: {name}")
        current_models[name] = current
    return protocol, {
        "preregistration": lock,
        "inputs": current_inputs,
        "code": current_code,
        "models": current_models,
        "settings": settings,
    }


def _validate_preregistered_retrieval_assets(
    protocol: Mapping[str, Any], assets: Mapping[str, Any]
) -> None:
    locked = protocol.get("retrieval_assets")
    if not isinstance(locked, Mapping):
        raise ValueError("v6 preregistration retrieval assets are missing")
    if int(assets.get("expected_docs", -1)) != int(locked.get("expected_documents", -2)):
        raise ValueError("Wiki18 expected document count differs from preregistration")
    if assets.get("counts") != locked.get("counts"):
        raise ValueError("Wiki18 counts differ from preregistration")
    paths = assets.get("paths") or {}
    for asset_name, report_name in (
        ("corpus", "corpus"), ("dense_index", "dense"), ("bm25_index", "bm25")
    ):
        item = locked.get(asset_name)
        if not isinstance(item, Mapping):
            raise ValueError(f"v6 preregistration lacks Wiki18 {asset_name}")
        if Path(str(paths.get(report_name) or "")).resolve() != Path(str(item.get("path") or "")).resolve():
            raise ValueError(f"Wiki18 {asset_name} path differs from preregistration")
    for asset_name in ("corpus", "dense_index"):
        item = locked[asset_name]
        path = Path(str(item["path"]))
        if path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"Wiki18 {asset_name} size differs from preregistration")
    if artifact_identity(Path(str(locked["bm25_index"]["path"]))) != dict(locked["bm25_index"]):
        raise ValueError("Wiki18 BM25 artifact differs from preregistration")


def _base_detail(
    row: Mapping[str, Any], *, target_type: str, plan: Mapping[str, Any],
    validation_errors: Sequence[str], has_dependent_step: bool,
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
        "duplicate_dependent_queries": 0,
        "new_dependent_candidate_count": 0,
        "hops": [],
        "query_hint_summary": {
            "required_producers": 0,
            "producers_with_hints": 0,
            "raw_candidates": 0,
            "v5_admitted_hints": 0,
            "exploratory_hints": 0,
            "hard_rejected_candidates": 0,
        },
        "final_ce_pair_count": 0,
        "all_final_ce_pairs_use_exact_original_question": True,
        "merge": None,
        "safety": None,
    }


def _exact_safety(arm_a: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "output_count": len(arm_a),
        "prefix8_exact": True,
        "unauthorized_original_displacements": 0,
        "root_passages_injected": 0,
        "duplicate_output_documents": 0,
        "fallback_exact": True,
    }


def _fallback(
    arm_a: Sequence[Mapping[str, Any]], detail: dict[str, Any], *, status: str,
    reason: str, plan_executable: bool, error: Exception | None = None,
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


def _query_variants(hop: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(hop.get("query_variants") or [])


def _merge_safety(
    arm_a: Sequence[Mapping[str, Any]], arm_b: Sequence[Mapping[str, Any]],
    hop_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    original = [dict(value) for value in arm_a]
    output = [dict(value) for value in arm_b]
    original_keys = [passage_score_key(value) for value in original]
    output_keys = [passage_score_key(value) for value in output]
    root_keys = {
        passage_score_key(passage)
        for hop in hop_results if not bool(hop.get("dependencies"))
        for variant in _query_variants(hop)
        for passage in variant.get("passages") or []
    }
    dependent_keys = {
        passage_score_key(passage)
        for hop in hop_results if bool(hop.get("dependencies"))
        for variant in _query_variants(hop)
        for passage in list(variant.get("passages") or [])[:CANDIDATES_PER_QUERY_VARIANT]
    }
    new_output_keys = set(output_keys) - set(original_keys)
    prefix_mismatches = sum(
        index >= len(output) or output[index] != original[index]
        for index in range(min(PROTECTED_ORIGINALS, len(original)))
    )
    root_only = new_output_keys & (root_keys - dependent_keys)
    return {
        "output_count": len(output),
        "prefix8_exact": prefix_mismatches == 0,
        "prefix8_mismatch_count": prefix_mismatches,
        "unauthorized_original_displacements": prefix_mismatches,
        "root_passages_injected": len(root_only),
        "root_only_injected_document_keys": sorted(root_only),
        "duplicate_output_documents": len(output_keys) - len(set(output_keys)),
        "fallback_exact": output == original,
    }


def _score_document_unions_once(
    states: Sequence[dict[str, Any]], *, cross_encoder: Any,
) -> tuple[dict[int, dict[str, float]], dict[int, int]]:
    pairs: list[tuple[str, str]] = []
    owners: list[tuple[int, str]] = []
    pair_counts: dict[int, int] = {}
    for state in states:
        unique: dict[str, dict[str, Any]] = {}
        for passage in state["arm_a"][PROTECTED_ORIGINALS:]:
            unique.setdefault(passage_score_key(passage), dict(passage))
        for hop in state["hop_results"]:
            if not bool(hop.get("dependencies")):
                continue
            for variant in _query_variants(hop):
                for passage in list(variant.get("passages") or [])[:CANDIDATES_PER_QUERY_VARIANT]:
                    unique.setdefault(passage_score_key(passage), dict(passage))
        pair_counts[int(state["row_index"])] = len(unique)
        for key, passage in unique.items():
            pairs.append((state["question"], _passage_text(passage)[:CE_MAX_CHARS]))
            owners.append((int(state["row_index"]), key))
    scores = _predict_scores(cross_encoder, pairs)
    by_row: dict[int, dict[str, float]] = {}
    for (row_index, key), score in zip(owners, scores):
        by_row.setdefault(row_index, {})[key] = score
    return by_row, pair_counts


def _interleave_ranked(
    variants: Sequence[Mapping[str, Any]], *, cap: int = BRIDGE_MAX_DOCS,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_rank = max((len(variant.get("passages") or []) for variant in variants), default=0)
    for rank in range(max_rank):
        for variant in variants:
            passages = list(variant.get("passages") or [])
            if rank >= len(passages):
                continue
            passage = dict(passages[rank])
            key = passage_score_key(passage)
            if key in seen:
                continue
            seen.add(key)
            result.append(passage)
            if len(result) >= cap:
                return result
    return result


def _execute_rows_batched_v6(
    rows: Sequence[Mapping[str, Any]], retriever: Any, args: argparse.Namespace,
    *, cross_encoder: Any,
) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
    if not hasattr(retriever, "batch_search"):
        raise TypeError("v6 dependent retrieval requires retriever.batch_search")
    outcomes: list[tuple[list[dict[str, Any]], dict[str, Any]] | None] = [None] * len(rows)
    states: list[dict[str, Any]] = []
    for row_index, source in enumerate(rows):
        row = dict(source)
        dataset = str(row["dataset"])
        target_type = TARGET_TYPES[dataset]
        plan = dict(row["plan"])
        errors = validate_plan_for_dependent_retrieval(plan, target_type)
        raw_steps = plan.get("steps")
        has_dependent_step = bool(
            not errors and isinstance(raw_steps, list)
            and any(isinstance(step, Mapping) and bool(_dependencies(step)) for step in raw_steps)
        )
        detail = _base_detail(
            row, target_type=target_type, plan=plan,
            validation_errors=errors, has_dependent_step=has_dependent_step,
        )
        arm_a = [deepcopy(dict(value)) for value in row["arm_a_passages"]]
        if len(arm_a) != TOTAL_PASSAGES:
            raise ValueError(f"{dataset}::{row['qid']} expected 10 Arm-A passages, got {len(arm_a)}")
        if errors:
            outcomes[row_index] = _fallback(
                arm_a, detail, status="fallback_plan_invalid", reason="plan_invalid",
                plan_executable=False,
            )
            continue
        steps = list(plan.get("steps") or [])
        if not steps or len(steps) > int(args.max_hops):
            outcomes[row_index] = _fallback(
                arm_a, detail, status="fallback_invalid_hop_count",
                reason="invalid_hop_count", plan_executable=False,
            )
            continue
        if not has_dependent_step:
            outcomes[row_index] = _fallback(
                arm_a, detail, status="fallback_no_dependent_step",
                reason="no_dependent_step", plan_executable=True,
            )
            continue
        try:
            schedule = _step_schedule(steps)
        except Exception as exc:
            outcomes[row_index] = _fallback(
                arm_a, detail, status="fallback_execution_error",
                reason="execution_error", plan_executable=True, error=exc,
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
        })

    max_depth = max(
        (int(record["dependency_depth"]) for state in states for record in state["schedule"]),
        default=0,
    )
    for depth in range(1, max_depth + 1):
        flat_queries: list[str] = []
        owners: list[tuple[dict[str, Any], dict[str, Any], int, str]] = []
        render_by_task: dict[tuple[int, int], dict[str, Any]] = {}
        for state in states:
            if state["row_error"] is not None:
                continue
            for record in state["schedule"]:
                if int(record["dependency_depth"]) != depth:
                    continue
                step = record["step"]
                try:
                    if record["dependencies"]:
                        queries, render_telemetry = render_question_anchored_queries_v6(
                            question=state["question"], step=step,
                            target_type=state["target_type"], slot_values=state["slot_values"],
                            max_variants=int(args.max_query_variants),
                        )
                    else:
                        queries = [render_root_query(step, state["target_type"])]
                        render_telemetry = {
                            "mode": "root_query_unchanged",
                            "query_count": 1,
                            "question_prefix_exact": None,
                        }
                    queries = list(
                        dict.fromkeys(
                            str(value) for value in queries if str(value).strip()
                        )
                    )
                    if not queries or len(queries) > int(args.max_query_variants):
                        raise ValueError(f"invalid v6 query count for {record['slot']}: {len(queries)}")
                    key = (int(state["row_index"]), int(record["step_index"]))
                    render_by_task[key] = dict(render_telemetry)
                    for variant_index, query in enumerate(queries, start=1):
                        if record["dependencies"] and not query.startswith(state["question"] + "\n"):
                            raise ValueError("dependent query lost its exact original-question prefix")
                        flat_queries.append(query)
                        owners.append((state, record, variant_index, query))
                except Exception as exc:
                    state["row_error"] = exc

        if not flat_queries:
            continue
        raw_batches = [list(result) for result in retriever.batch_search(flat_queries)]
        if len(raw_batches) != len(flat_queries):
            raise RuntimeError(f"batch retriever returned {len(raw_batches)}/{len(flat_queries)} rows")
        ranked_batches = [
            _rerank_one_query(query, raw, cross_encoder=cross_encoder, topk=int(args.step_rerank_topk))
            for query, raw in zip(flat_queries, raw_batches)
        ]
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for (state, record, variant_index, query), raw, ranked in zip(owners, raw_batches, ranked_batches):
            key = (int(state["row_index"]), int(record["step_index"]))
            render_telemetry = render_by_task[key]
            clause = query[len(state["question"]):].lstrip("\n") if record["dependencies"] else query
            dependency_values = [
                str(value)
                for dependency in record["dependencies"]
                for value in (
                    state["slot_values"].get(dependency, [])
                    if isinstance(state["slot_values"].get(dependency, []), Sequence)
                    and not isinstance(state["slot_values"].get(dependency, []), (str, bytes))
                    else [state["slot_values"].get(dependency)]
                )
                if value is not None and str(value).strip() and str(value) in clause
            ]
            grouped.setdefault(key, []).append({
                "query_variant_id": f"{_step_id(record['step'], int(record['step_index']))}::q{variant_index}",
                "query_variant_index": variant_index,
                "query": query,
                "query_sha256": _sha256_text(query),
                "hint": {
                    "mode": render_telemetry.get("mode"),
                    "matched_dependency_values": dependency_values,
                } if record["dependencies"] else None,
                "raw_count": len(raw),
                "reranked_count": len(ranked),
                "retrieved_ids": [str(value.get("id") or "") for value in ranked],
                "passages": [dict(value) for value in ranked],
            })

        for state in states:
            if state["row_error"] is not None:
                continue
            for record in state["schedule"]:
                if int(record["dependency_depth"]) != depth:
                    continue
                key = (int(state["row_index"]), int(record["step_index"]))
                if key not in grouped:
                    continue
                try:
                    variants = grouped[key]
                    candidate_source_passages = _interleave_ranked(variants)
                    hints: list[dict[str, Any]] = []
                    hint_telemetry: dict[str, Any] | None = None
                    if record["consumers"]:
                        hints, hint_telemetry = select_bridge_query_hints_v6(
                            step=record["step"], consumers=record["consumers"],
                            target_type=state["target_type"],
                            query=" ; ".join(value["query"] for value in variants),
                            question=state["question"], passages=candidate_source_passages,
                            max_docs=BRIDGE_MAX_DOCS, max_hints=BRIDGE_MAX_HINTS,
                            max_body_chars=BRIDGE_MAX_BODY_CHARS,
                        )
                        summary = state["detail"]["query_hint_summary"]
                        summary["required_producers"] += 1
                        summary["raw_candidates"] += int(
                            hint_telemetry.get("v5_selector_telemetry", {}).get(
                                "raw_candidate_count", 0
                            )
                        )
                        summary["v5_admitted_hints"] += sum(
                            value.get("admission", {}).get("source") == "v5_accepted"
                            for value in hints
                        )
                        summary["exploratory_hints"] += sum(
                            value.get("admission", {}).get("source") == "raw_rank_fill"
                            for value in hints
                        )
                        summary["hard_rejected_candidates"] += len(
                            hint_telemetry.get("hard_rejected_candidates") or []
                        )
                        if hints:
                            summary["producers_with_hints"] += 1
                            _record_slot_values(
                                state["slot_values"], step=record["step"],
                                index=int(record["step_index"]), candidates=hints,
                            )
                    hop_id = _step_id(record["step"], int(record["step_index"]))
                    state["hop_results"].append({
                        "hop_id": hop_id,
                        "logical_hop_id": hop_id,
                        "step_index": int(record["step_index"]),
                        "dependencies": list(record["dependencies"]),
                        "dependency_depth": int(record["dependency_depth"]),
                        "is_dependent": bool(record["dependencies"]),
                        "query_variants": variants,
                    })
                    state["detail"]["hops"].append({
                        "hop_id": hop_id,
                        "step_index": int(record["step_index"]),
                        "dependencies": list(record["dependencies"]),
                        "dependency_depth": int(record["dependency_depth"]),
                        "is_dependent": bool(record["dependencies"]),
                        "query_renderer": render_by_task[key],
                        "query_variants": [
                            {name: value[name] for name in (
                                "query_variant_id", "query_variant_index", "query",
                                "query_sha256", "raw_count", "reranked_count", "retrieved_ids",
                            )}
                            for value in variants
                        ],
                        "hint_required": bool(record["consumers"]),
                        "query_hints": hints,
                        "query_hint_selector": hint_telemetry,
                    })
                except Exception as exc:
                    state["row_error"] = exc

    ready_to_score: list[dict[str, Any]] = []
    for state in states:
        state["hop_results"].sort(key=lambda value: int(value["step_index"]))
        state["detail"]["hops"].sort(key=lambda value: int(value["step_index"]))
        dependent_queries = [
            variant["query"]
            for hop in state["detail"]["hops"] if hop["dependencies"]
            for variant in hop["query_variants"]
        ]
        state["detail"]["dependent_query_count"] = len(dependent_queries)
        state["detail"]["second_hop_query_count"] = len(dependent_queries)
        state["detail"]["duplicate_dependent_queries"] = (
            len(dependent_queries) - len(set(dependent_queries))
        )
        state["detail"]["all_dependent_queries_start_with_exact_question"] = all(
            query.startswith(state["question"] + "\n") for query in dependent_queries
        )
        state["detail"]["max_query_variants_per_logical_hop"] = max(
            (len(hop["query_variants"]) for hop in state["detail"]["hops"] if hop["dependencies"]),
            default=0,
        )
        if state["row_error"] is not None:
            outcomes[state["row_index"]] = _fallback(
                state["arm_a"], state["detail"], status="fallback_execution_error",
                reason="execution_error", plan_executable=True, error=state["row_error"],
            )
        else:
            ready_to_score.append(state)

    scores_by_row, pair_counts = _score_document_unions_once(ready_to_score, cross_encoder=cross_encoder)
    for state in ready_to_score:
        state["detail"]["final_ce_pair_count"] = pair_counts.get(int(state["row_index"]), 0)
        merged, merge_telemetry = merge_dependent_passages_v6(
            state["arm_a"], state["hop_results"],
            scores_by_row.get(int(state["row_index"]), {}),
            protected_originals=PROTECTED_ORIGINALS,
            candidates_per_query_variant=CANDIDATES_PER_QUERY_VARIANT,
            total_passages=TOTAL_PASSAGES,
        )
        safety = _merge_safety(state["arm_a"], merged, state["hop_results"])
        if (
            len(merged) != TOTAL_PASSAGES or not safety["prefix8_exact"]
            or safety["unauthorized_original_displacements"] != 0
            or safety["root_passages_injected"] != 0
            or safety["duplicate_output_documents"] != 0
        ):
            raise RuntimeError(f"v6 safety invariant failed for {state['dataset']}::{state['qid']}: {safety}")
        changed = bool(merge_telemetry.get("changed"))
        state["detail"].update({
            "execution_status": "executed_changed" if changed else "fallback_no_candidate_strictly_better",
            "fallback_exact": not changed,
            "fallback_reason": None if changed else str(merge_telemetry.get("fallback_reason")),
            "merge": merge_telemetry,
            "safety": safety,
            "new_dependent_candidate_count": len(merge_telemetry.get("selected_new") or []),
            "arm_b_passages_sha256": _sha256_json(merged),
        })
        outcomes[state["row_index"]] = ([dict(value) for value in merged], state["detail"])
    if any(value is None for value in outcomes):
        raise RuntimeError("v6 execution did not produce every row outcome")
    return [value for value in outcomes if value is not None]


def _dry_run(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    report = dict(v5._dry_run(rows, args))
    report.update({
        "runner_version": RUNNER_VERSION,
        "query_policy_version": QUERY_POLICY_VERSION,
        "merge_policy_version": POLICY_VERSION,
        "v6_policy_not_executed": True,
    })
    return report


def _aggregate_dataset(
    dataset: str, execution_rows: Sequence[Mapping[str, Any]],
    arm_a_rows: Sequence[Mapping[str, Any]], arm_b_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    current = [row for row in execution_rows if row["dataset"] == dataset]
    arm_a = {str(row["qid"]): row for row in arm_a_rows if row["dataset"] == dataset}
    arm_b = {str(row["qid"]): row for row in arm_b_rows if row["dataset"] == dataset}
    fallbacks = [row for row in current if bool(row.get("fallback_exact"))]
    changed = sum(
        arm_a[str(row["qid"])]["passages_sha256"] != arm_b[str(row["qid"])]["passages_sha256"]
        for row in current
    )
    eligible = sum(bool(row["has_dependent_step"]) for row in current)
    nonempty = sum(
        bool(row["has_dependent_step"]) and int(row.get("dependent_query_count", 0)) > 0
        for row in current
    )
    return {
        "n": len(current),
        "executed": sum(row["execution_status"] == "executed_changed" for row in current),
        "fallback": len(fallbacks),
        "fallback_plan_invalid": sum(row["execution_status"] == "fallback_plan_invalid" for row in current),
        "fallback_invalid_hop_count": sum(row["execution_status"] == "fallback_invalid_hop_count" for row in current),
        "fallback_no_dependent_step": sum(row["execution_status"] == "fallback_no_dependent_step" for row in current),
        "fallback_no_candidate_strictly_better": sum(row["execution_status"] == "fallback_no_candidate_strictly_better" for row in current),
        "fallback_execution_error": sum(row["execution_status"] == "fallback_execution_error" for row in current),
        "runtime_errors": sum("error" in row for row in current),
        "arm_b_changed": changed,
        "retained_new_dependent_document_question_rate": changed / max(1, len(current)),
        "plan_executable": sum(bool(row["plan_executable"]) for row in current),
        "plan_executable_rate": sum(bool(row["plan_executable"]) for row in current) / max(1, len(current)),
        "dependent_step_eligible": eligible,
        "dependent_step_query_nonempty": nonempty,
        "dependent_hop_query_nonempty_rate": nonempty / max(1, eligible),
        "dependent_query_count": sum(int(row.get("dependent_query_count", 0)) for row in current),
        "duplicate_dependent_queries": sum(
            int(row.get("duplicate_dependent_queries", 0)) for row in current
        ),
        "max_query_variants_per_logical_hop": max(
            (int(row.get("max_query_variants_per_logical_hop", 0)) for row in current), default=0
        ),
        "all_dependent_queries_start_with_exact_question": all(
            bool(row.get("all_dependent_queries_start_with_exact_question", True)) for row in current
        ),
        "all_final_ce_pairs_use_exact_original_question": all(
            bool(row.get("all_final_ce_pairs_use_exact_original_question", True)) for row in current
        ),
        "final_ce_pair_count": sum(int(row.get("final_ce_pair_count", 0)) for row in current),
        "query_hint_required_producers": sum(int(row["query_hint_summary"]["required_producers"]) for row in current),
        "query_hint_producers_with_hints": sum(int(row["query_hint_summary"]["producers_with_hints"]) for row in current),
        "v5_admitted_hints": sum(int(row["query_hint_summary"]["v5_admitted_hints"]) for row in current),
        "exploratory_hints": sum(int(row["query_hint_summary"]["exploratory_hints"]) for row in current),
        "all_top10": all(int(row["safety"]["output_count"]) == 10 for row in current),
        "prefix8_exact": all(bool(row["safety"]["prefix8_exact"]) for row in current),
        "unauthorized_original_displacements": sum(int(row["safety"]["unauthorized_original_displacements"]) for row in current),
        "root_passages_injected": sum(int(row["safety"]["root_passages_injected"]) for row in current),
        "duplicate_output_documents": sum(int(row["safety"]["duplicate_output_documents"]) for row in current),
        "fallback_exact": all(
            arm_a[str(row["qid"])]["passages_sha256"] == arm_b[str(row["qid"])]["passages_sha256"]
            for row in fallbacks
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("outputs/audits/inference_proofkg_v1_n900_seed42_preregistration/pilot.question_only.jsonl"))
    parser.add_argument("--retrieval_contexts", type=Path, default=Path("outputs/audits/inference_proofkg_v1_n900_seed42_preregistration/retrieval_contexts.jsonl"))
    parser.add_argument("--musique_plans", type=Path, default=Path("outputs/audits/inference_proofkg_v1_pilot30x3_plans_v1/predictions.question_only.jsonl"))
    parser.add_argument("--hotpot_plans", type=Path, default=Path("outputs/audits/inference_proofkg_hotpot_relation_graph_pilot30_v2_plans/predictions.question_only.jsonl"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--n_per_dataset", type=int, default=30)
    parser.add_argument("--corpus_path", default="indexes_wiki18/corpus_flashrag.jsonl")
    parser.add_argument("--dense_index_path", default="indexes_wiki18/e5_fp16.dat")
    parser.add_argument("--bm25_index_path", default="indexes_wiki18/bm25")
    parser.add_argument("--expected_docs", type=int, default=21_015_324)
    parser.add_argument("--rrf_candidate_k", type=int, default=100)
    parser.add_argument("--step_rerank_topk", type=int, default=10)
    parser.add_argument("--cross_encoder_model", default="models/bge-reranker-v2-m3")
    parser.add_argument("--retrieval_encoder_path", default="models/e5-base-v2")
    parser.add_argument("--preregistration", type=Path, default=Path("outputs/audits/subquestion_dependent_retrieval_pilot30x2_seed42_v6_preregistration/protocol.json"))
    parser.add_argument("--max_hops", type=int, default=4)
    parser.add_argument("--max_query_variants", type=int, default=2)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--experiment_id")
    parser.add_argument("--dry_run", action="store_true", help="Validate joins/query rendering only; no retrieval, CE, or output files.")
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
        raise SystemExit("formal v6 materialisation is locked to HotpotQA30 + MuSiQue30")
    if args.output_dir is None or not str(args.experiment_id or "").strip():
        raise SystemExit("formal materialisation requires --output_dir and --experiment_id")

    preregistration, runtime_locks = _validate_preregistration_runtime(args)
    configured_retrieval_encoder = Path(model_path("e5")).expanduser().resolve()
    requested_retrieval_encoder = Path(args.retrieval_encoder_path).expanduser().resolve()
    if configured_retrieval_encoder != requested_retrieval_encoder:
        raise ValueError(
            "runtime E5 encoder differs from the explicitly locked retrieval encoder"
        )
    assets = _validate_full_wiki18_assets(
        args.corpus_path, args.dense_index_path, args.bm25_index_path,
        expected_docs=int(args.expected_docs),
    )
    _validate_preregistered_retrieval_assets(preregistration, assets)
    run_dir, experiment_id = prepare_new_run_dir(
        args.output_dir, experiment_id=args.experiment_id,
        extra={"phase": "plan_once_dependent_retrieval_v6_materialisation", "gold_access": False},
    )
    try:
        retriever = _build_retriever(
            args.datasets[0], int(args.rrf_candidate_k),
            corpus_path=args.corpus_path, dense_index_path=args.dense_index_path,
            bm25_index_path=args.bm25_index_path,
        )
        cross_encoder = get_cross_encoder(str(args.cross_encoder_model))
        outcomes = _execute_rows_batched_v6(rows, retriever, args, cross_encoder=cross_encoder)

        arm_a_rows: list[dict[str, Any]] = []
        arm_b_rows: list[dict[str, Any]] = []
        execution_rows: list[dict[str, Any]] = []
        for index, (row, outcome) in enumerate(zip(rows, outcomes), start=1):
            arm_b, detail = outcome
            common = {
                "row_id": f"dependent-retrieval-pilot::{row['dataset']}::{row['qid']}",
                "question_key": f"{row['dataset']}::{row['qid']}",
                "dataset": row["dataset"], "qid": row["qid"],
                "question": row["question"], "question_sha256": row["question_sha256"],
                "split": row["split"], "gold_access": False,
                "kg_subgraph": row["legacy_kg"], "legacy_kg_sha256": row["legacy_kg_sha256"],
            }
            arm_a = [dict(value) for value in row["arm_a_passages"]]
            arm_a_rows.append({
                **common, "arm": "A_question_only", "retrieved_passages": arm_a,
                "passages_sha256": _sha256_json(arm_a),
            })
            arm_b_rows.append({
                **common, "arm": "B_question_anchored_dependent", "retrieved_passages": arm_b,
                "passages_sha256": _sha256_json(arm_b), "fallback_to_a": bool(detail["fallback_exact"]),
                "retrieval_trace": {
                    "plan_executable": bool(detail["plan_executable"]),
                    "has_dependent_step": bool(detail["has_dependent_step"]),
                    "dependent_query_count": int(detail["dependent_query_count"]),
                    "duplicate_dependent_queries": int(
                        detail["duplicate_dependent_queries"]
                    ),
                    "new_dependent_candidate_count": int(detail["new_dependent_candidate_count"]),
                    "fallback_reason": detail["fallback_reason"],
                    "query_hint_required_producers": int(detail["query_hint_summary"]["required_producers"]),
                    "query_hint_producers_with_hints": int(detail["query_hint_summary"]["producers_with_hints"]),
                    "all_dependent_queries_start_with_exact_question": bool(detail.get("all_dependent_queries_start_with_exact_question", True)),
                    "max_query_variants_per_logical_hop": int(detail.get("max_query_variants_per_logical_hop", 0)),
                    "final_ce_pair_count": int(detail["final_ce_pair_count"]),
                    "prefix8_exact": bool(detail["safety"]["prefix8_exact"]),
                    "unauthorized_original_displacements": int(detail["safety"]["unauthorized_original_displacements"]),
                    "root_passages_injected": int(detail["safety"]["root_passages_injected"]),
                    "duplicate_output_documents": int(detail["safety"]["duplicate_output_documents"]),
                },
            })
            execution_rows.append(detail)
            print(f"v6 dependent retrieval {index}/{len(rows)} {row['dataset']}::{row['qid']} status={detail['execution_status']}", flush=True)

        arm_a_path, arm_b_path = run_dir / "arm_a.jsonl", run_dir / "arm_b.jsonl"
        execution_path = run_dir / "execution_details.jsonl"
        _write_jsonl(arm_a_path, arm_a_rows)
        _write_jsonl(arm_b_path, arm_b_rows)
        _write_jsonl(execution_path, execution_rows)
        by_dataset = {
            dataset: _aggregate_dataset(dataset, execution_rows, arm_a_rows, arm_b_rows)
            for dataset in args.datasets
        }
        safety_summary = {
            "all_top10": all(value["all_top10"] for value in by_dataset.values()),
            "prefix8_exact": all(value["prefix8_exact"] for value in by_dataset.values()),
            "unauthorized_original_displacements": sum(value["unauthorized_original_displacements"] for value in by_dataset.values()),
            "root_passages_injected": sum(value["root_passages_injected"] for value in by_dataset.values()),
            "duplicate_output_documents": sum(value["duplicate_output_documents"] for value in by_dataset.values()),
            "duplicate_dependent_queries": sum(value["duplicate_dependent_queries"] for value in by_dataset.values()),
            "fallback_exact": all(value["fallback_exact"] for value in by_dataset.values()),
            "runtime_errors": sum(value["runtime_errors"] for value in by_dataset.values()),
            "all_dependent_queries_start_with_exact_question": all(value["all_dependent_queries_start_with_exact_question"] for value in by_dataset.values()),
            "max_query_variants_per_logical_hop": max(value["max_query_variants_per_logical_hop"] for value in by_dataset.values()),
            "all_final_ce_pairs_use_exact_original_question": all(value["all_final_ce_pairs_use_exact_original_question"] for value in by_dataset.values()),
        }
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "runner_version": RUNNER_VERSION,
            "query_policy_version": QUERY_POLICY_VERSION,
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
                    "cohort": args.cohort, "retrieval_contexts": args.retrieval_contexts,
                    "musique_plans": args.musique_plans, "hotpot_plans": args.hotpot_plans,
                }.items()
            },
            "retrieval_assets": assets,
            "settings": runtime_locks["settings"],
            "by_dataset": by_dataset,
            "safety_summary": safety_summary,
            "outputs": {
                "arm_a": {"path": str(arm_a_path), "sha256": _sha256_file(arm_a_path)},
                "arm_b": {"path": str(arm_b_path), "sha256": _sha256_file(arm_b_path)},
                "execution_details": {"path": str(execution_path), "sha256": _sha256_file(execution_path)},
            },
            "scientific_boundary": "Development retrieval inputs only. No Gold, answer generation, or effectiveness scoring was used. v4/v5 artifacts remain unchanged.",
        }
        report_path = run_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        dump_manifest(
            run_dir, status=report["status"],
            extra={"experiment_id": experiment_id, "phase": "plan_once_dependent_retrieval_v6_materialisation", "report_sha256": _sha256_file(report_path)},
        )
        print(json.dumps({"status": report["status"], "by_dataset": by_dataset, "safety_summary": safety_summary}, ensure_ascii=False, indent=2))
    except Exception as exc:
        dump_manifest(
            run_dir, status="FAILED_RUNTIME",
            extra={"experiment_id": experiment_id, "phase": "plan_once_dependent_retrieval_v6_materialisation", "failure": {"type": type(exc).__name__, "message": str(exc)}},
        )
        raise


if __name__ == "__main__":
    main()
