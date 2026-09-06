#!/usr/bin/env python
"""Materialise a gold-free plan-once dependent-retrieval pilot.

This is an isolated development runner.  It does not modify the canonical
pipeline.  Arm A is the already frozen canonical passage list.  Arm B starts
from A, executes the frozen answer-free QueryPlan against the same local
Wiki18 retrieval assets, and replaces only a bounded number of passage slots.
Any plan/execution failure falls back to Arm A byte-for-byte for that question.

Gold answers, supporting-fact labels, and dataset decompositions are never
loaded by this program.  They belong in a later, separate scorer process after
these inputs have been frozen and hashed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_sha256
from kgproweight.retrieval.dependent import (
    extract_deterministic_bridge_candidates,
    instantiate_dependent_queries,
    merge_passages_with_provenance,
    render_root_query,
    validate_plan_for_dependent_retrieval,
)
from kgproweight.retrieval.reranker import rerank_passages
from kgproweight.utils.logging import dump_manifest, prepare_new_run_dir
from scripts.pilot.audit_iterative_bridge_retrieval import (
    _build_retriever,
    _validate_full_wiki18_assets,
)


RUNNER_VERSION = "plan-once-dependent-retrieval-pilot-2-layer-batched"
DATASETS = ("hotpotqa", "musique")
TARGET_TYPES = {"hotpotqa": "relation_graph", "musique": "subquery_graph"}
FORBIDDEN_EXECUTION_KEYS = frozenset({
    "answer", "answers", "gold_answer", "gold_answers", "golden_answers",
    "supporting_facts", "supporting_titles", "decomposition",
    "question_decomposition", "evidence", "evidences", "reasoning", "sp",
})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    # Must match freeze_inference_proofkg_preregistration.py.  Its historical
    # identity function used json.dumps defaults rather than compact separators.
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _question_key(row: Mapping[str, Any]) -> str:
    return f"{str(row.get('dataset') or '').strip()}::{str(row.get('qid') or '').strip()}"


def _assert_no_forbidden_keys(value: Any, *, where: str) -> None:
    """Reject execution inputs that contain Gold/decomposition fields.

    This checks field names, not generated text.  It deliberately does not load
    a Gold answer list merely to scan strings, because doing so would violate
    the execution/scoring separation this pilot is intended to test.
    """
    if isinstance(value, Mapping):
        bad = {str(key).casefold() for key in value} & FORBIDDEN_EXECUTION_KEYS
        if bad:
            raise ValueError(f"forbidden execution fields at {where}: {sorted(bad)}")
        for key, child in value.items():
            _assert_no_forbidden_keys(child, where=f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_keys(child, where=f"{where}[{index}]")


def _index_unique(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    datasets: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        dataset = str(row.get("dataset") or "")
        if datasets is not None and dataset not in datasets:
            continue
        key = _question_key(row)
        if key == "::":
            raise ValueError(f"{label} contains an empty dataset/qid")
        if key in indexed:
            raise ValueError(f"{label} contains duplicate key {key}")
        indexed[key] = row
    return indexed


def _load_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    wanted = set(args.datasets)
    cohort_rows = [
        row for row in _read_jsonl(args.cohort)
        if str(row.get("dataset") or "") in wanted
    ]
    contexts = _index_unique(
        _read_jsonl(args.retrieval_contexts), label="retrieval contexts", datasets=wanted
    )
    musique_plans = _index_unique(
        _read_jsonl(args.musique_plans), label="MuSiQue plans", datasets={"musique"}
    )
    hotpot_plans = _index_unique(
        _read_jsonl(args.hotpot_plans), label="HotpotQA plans", datasets={"hotpotqa"}
    )
    plans = {**musique_plans, **hotpot_plans}
    expected = int(args.n_per_dataset)
    counts = Counter(str(row.get("dataset") or "") for row in cohort_rows)
    if any(counts[dataset] != expected for dataset in wanted):
        raise ValueError(f"cohort counts differ from n_per_dataset={expected}: {dict(counts)}")

    assembled: list[dict[str, Any]] = []
    for row in cohort_rows:
        _assert_no_forbidden_keys(row, where=f"cohort.{_question_key(row)}")
        key = _question_key(row)
        if key not in contexts or key not in plans:
            raise ValueError(f"identity join is incomplete for {key}")
        context, plan_row = contexts[key], plans[key]
        _assert_no_forbidden_keys(context, where=f"context.{key}")
        _assert_no_forbidden_keys(plan_row, where=f"plan.{key}")
        if plan_row.get("gold_access") is not False:
            raise ValueError(f"planner record is not explicitly gold-free for {key}")
        question = str(row.get("question") or "").strip()
        expected_sha = str(row.get("question_sha256") or "")
        if not question or question_sha256(question) != expected_sha:
            raise ValueError(f"cohort question/hash mismatch for {key}")
        for label, joined in (("context", context), ("plan", plan_row)):
            if str(joined.get("question_sha256") or "") != expected_sha:
                raise ValueError(f"{label} question hash mismatch for {key}")
        if str(plan_row.get("question") or "").strip() != question:
            raise ValueError(f"plan question mismatch for {key}")
        passages = list(context.get("passages") or [])
        if not passages:
            raise ValueError(f"frozen Arm A has no passages for {key}")
        if str(context.get("passages_sha256") or "") != _sha256_json(passages):
            raise ValueError(f"frozen Arm A passage hash mismatch for {key}")
        legacy_kg = list(context.get("legacy_kg") or [])
        if str(context.get("legacy_kg_sha256") or "") != _sha256_json(legacy_kg):
            raise ValueError(f"frozen legacy KG hash mismatch for {key}")
        assembled.append({
            "dataset": str(row["dataset"]),
            "qid": str(row["qid"]),
            "question": question,
            "question_sha256": expected_sha,
            "split": str(row.get("split") or "pilot"),
            "arm_a_passages": passages,
            "arm_a_passages_sha256": str(context.get("passages_sha256") or _sha256_json(passages)),
            "legacy_kg": legacy_kg,
            "legacy_kg_sha256": str(context.get("legacy_kg_sha256")),
            "plan": dict(plan_row.get("predicted_target") or {}),
            "planner_row_sha256": _sha256_json(plan_row),
        })
    return assembled


def _step_id(step: Mapping[str, Any], index: int) -> str:
    return str(step.get("output_slot") or f"hop_{index}")


def _dependencies(step: Mapping[str, Any]) -> list[str]:
    return [str(value) for value in (step.get("dependencies") or []) if str(value)]


def _record_slot_values(
    slot_values: dict[str, list[str]],
    *,
    step: Mapping[str, Any],
    index: int,
    candidates: Sequence[Mapping[str, Any]],
) -> None:
    values = [str(row.get("surface") or "").strip() for row in candidates]
    values = list(dict.fromkeys(value for value in values if value))
    if not values:
        return
    output_slot = _step_id(step, index)
    aliases = {
        output_slot, f"hop_{index}", f"step_{index}",
        f"$hop_{index}", f"#{index}",
    }
    for alias in aliases:
        slot_values[alias] = values


def _rerank_step_results(
    query: str,
    results: Sequence[Mapping[str, Any]],
    *,
    topk: int,
    cross_encoder_model: str,
) -> list[dict[str, Any]]:
    if not results:
        return []
    ranked = rerank_passages(
        [query], [list(results)], topk=topk, method="cross-encoder",
        cross_encoder_model=cross_encoder_model,
    )
    return [dict(row) for row in ranked[0]]


def _execute_row(
    row: Mapping[str, Any],
    retriever: Any,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = str(row["dataset"])
    target_type = TARGET_TYPES[dataset]
    plan = dict(row["plan"])
    validation_errors = validate_plan_for_dependent_retrieval(plan, target_type)
    raw_steps = plan.get("steps")
    has_dependent_step = bool(
        not validation_errors
        and isinstance(raw_steps, list)
        and any(isinstance(step, Mapping) and bool(_dependencies(step)) for step in raw_steps)
    )
    detail: dict[str, Any] = {
        "dataset": dataset,
        "qid": str(row["qid"]),
        "question_sha256": str(row["question_sha256"]),
        "target_type": target_type,
        "plan_sha256": _sha256_json(plan),
        "plan_validation_errors": list(validation_errors),
        "has_dependent_step": has_dependent_step,
        "gold_access": False,
        "execution_status": "pending",
        "fallback_exact": False,
        "hops": [],
    }
    arm_a = [dict(passage) for passage in row["arm_a_passages"]]
    if validation_errors:
        detail.update({
            "execution_status": "fallback_plan_invalid", "fallback_exact": True,
            "plan_executable": False, "dependent_query_count": 0,
            "second_hop_query_count": 0, "new_dependent_candidate_count": 0,
            "fallback_reason": "plan_invalid",
        })
        return arm_a, detail

    steps = list(plan.get("steps") or [])
    if not steps or len(steps) > int(args.max_hops):
        detail.update({
            "execution_status": "fallback_invalid_hop_count", "fallback_exact": True,
            "plan_executable": False, "dependent_query_count": 0,
            "second_hop_query_count": 0, "new_dependent_candidate_count": 0,
            "fallback_reason": "invalid_hop_count",
        })
        return arm_a, detail

    slot_values: dict[str, list[str]] = {}
    hop_results: list[dict[str, Any]] = []
    used_surfaces: list[str] = []
    try:
        for index, step in enumerate(steps, start=1):
            dependencies = _dependencies(step)
            queries = (
                instantiate_dependent_queries(
                    step, target_type, slot_values, max_variants=int(args.max_query_variants)
                )
                if dependencies
                else [render_root_query(step, target_type)]
            )
            queries = list(dict.fromkeys(str(value).strip() for value in queries if str(value).strip()))
            if not queries:
                raise ValueError(f"no executable query for {_step_id(step, index)}")
            current_results: list[dict[str, Any]] = []
            query_details: list[dict[str, Any]] = []
            seen_documents: set[str] = set()
            for query in queries:
                raw = list(retriever.search(query))
                ranked = _rerank_step_results(
                    query, raw, topk=int(args.step_rerank_topk),
                    cross_encoder_model=str(args.cross_encoder_model),
                )
                for passage in ranked:
                    document_key = str(passage.get("id") or _sha256_json(passage))
                    if document_key not in seen_documents:
                        seen_documents.add(document_key)
                        current_results.append(passage)
                query_details.append({
                    "query": query,
                    "raw_count": len(raw),
                    "reranked_count": len(ranked),
                    "retrieved_ids": [str(doc.get("id") or "") for doc in ranked],
                })
            hop_results.append({
                "hop_id": _step_id(step, index),
                "query": " || ".join(queries),
                "passages": current_results,
            })
            candidates = extract_deterministic_bridge_candidates(
                " ; ".join(queries), current_results,
                exclude_surfaces=used_surfaces,
                max_docs=int(args.bridge_source_docs),
                max_candidates=int(args.max_bridge_candidates),
            )
            _record_slot_values(slot_values, step=step, index=index, candidates=candidates)
            used_surfaces.extend(
                str(candidate.get("surface") or "") for candidate in candidates
            )
            detail["hops"].append({
                "hop_id": _step_id(step, index),
                "dependencies": dependencies,
                "queries": query_details,
                "bridge_candidates": list(candidates),
            })

        merged, merge_telemetry = merge_passages_with_provenance(
            arm_a, hop_results,
            original_quota=int(args.original_quota),
            per_hop_quota=int(args.per_hop_quota),
            total=int(args.total_passages),
        )
        if not merged:
            raise ValueError("dependent retrieval produced an empty merged passage list")
        detail.update({
            "execution_status": "executed",
            "fallback_exact": False,
            "merge": merge_telemetry,
            "arm_b_passages_sha256": _sha256_json(merged),
            "plan_executable": True,
            "dependent_query_count": sum(
                len(hop["queries"]) for hop in detail["hops"] if hop["dependencies"]
            ),
            "second_hop_query_count": sum(
                len(hop["queries"]) for hop in detail["hops"] if hop["dependencies"]
            ),
            "new_dependent_candidate_count": sum(
                int(count) for source, count in
                dict(merge_telemetry.get("selected_by_source") or {}).items()
                if source not in {"original_prefix", "original_backfill"}
            ),
            "fallback_reason": None,
        })
        return [dict(passage) for passage in merged], detail
    except Exception as exc:
        detail.update({
            "execution_status": "fallback_execution_error",
            "fallback_exact": True,
            "plan_executable": True,
            "dependent_query_count": sum(
                len(hop["queries"]) for hop in detail["hops"] if hop["dependencies"]
            ),
            "second_hop_query_count": sum(
                len(hop["queries"]) for hop in detail["hops"] if hop["dependencies"]
            ),
            "new_dependent_candidate_count": 0,
            "fallback_reason": "execution_error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        })
        return arm_a, detail


def _execute_rows_batched(
    rows: Sequence[Mapping[str, Any]],
    retriever: Any,
    args: argparse.Namespace,
) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
    """Execute the same row logic with one full-index scan per plan layer.

    The previous implementation called ``retriever.search`` for every rendered
    query.  With the immutable 21M-document fp16 memmap this rescanned the whole
    index once per query.  This scheduler preserves row order and per-row query
    order, but flattens all queries that are ready at a given dependency layer
    into one ``batch_search`` call.  Bridge values are recorded only after the
    layer completes, so a later layer sees exactly the observations available to
    the sequential implementation.
    """

    if not hasattr(retriever, "batch_search"):
        raise TypeError("layer-batched dependent retrieval requires retriever.batch_search")

    outcomes: list[tuple[list[dict[str, Any]], dict[str, Any]] | None] = [
        None for _ in rows
    ]
    states: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        dataset = str(row["dataset"])
        target_type = TARGET_TYPES[dataset]
        plan = dict(row["plan"])
        validation_errors = validate_plan_for_dependent_retrieval(plan, target_type)
        raw_steps = plan.get("steps")
        has_dependent_step = bool(
            not validation_errors
            and isinstance(raw_steps, list)
            and any(
                isinstance(step, Mapping) and bool(_dependencies(step))
                for step in raw_steps
            )
        )
        detail: dict[str, Any] = {
            "dataset": dataset,
            "qid": str(row["qid"]),
            "question_sha256": str(row["question_sha256"]),
            "target_type": target_type,
            "plan_sha256": _sha256_json(plan),
            "plan_validation_errors": list(validation_errors),
            "has_dependent_step": has_dependent_step,
            "gold_access": False,
            "execution_status": "pending",
            "fallback_exact": False,
            "hops": [],
        }
        arm_a = [dict(passage) for passage in row["arm_a_passages"]]
        if validation_errors:
            detail.update({
                "execution_status": "fallback_plan_invalid", "fallback_exact": True,
                "plan_executable": False, "dependent_query_count": 0,
                "second_hop_query_count": 0, "new_dependent_candidate_count": 0,
                "fallback_reason": "plan_invalid",
            })
            outcomes[row_index] = (arm_a, detail)
            continue
        steps = list(plan.get("steps") or [])
        if not steps or len(steps) > int(args.max_hops):
            detail.update({
                "execution_status": "fallback_invalid_hop_count", "fallback_exact": True,
                "plan_executable": False, "dependent_query_count": 0,
                "second_hop_query_count": 0, "new_dependent_candidate_count": 0,
                "fallback_reason": "invalid_hop_count",
            })
            outcomes[row_index] = (arm_a, detail)
            continue
        states.append({
            "row_index": row_index,
            "target_type": target_type,
            "steps": steps,
            "arm_a": arm_a,
            "detail": detail,
            "slot_values": {},
            "hop_results": [],
            "used_surfaces": [],
            "error": None,
        })

    max_layers = max((len(state["steps"]) for state in states), default=0)
    for layer_index in range(max_layers):
        flat_queries: list[str] = []
        owners: list[dict[str, Any]] = []
        queries_by_state: dict[int, list[str]] = {}

        # Render only from observations produced by earlier completed layers.
        for state in states:
            if state["error"] is not None or layer_index >= len(state["steps"]):
                continue
            step = state["steps"][layer_index]
            try:
                dependencies = _dependencies(step)
                queries = (
                    instantiate_dependent_queries(
                        step,
                        state["target_type"],
                        state["slot_values"],
                        max_variants=int(args.max_query_variants),
                    )
                    if dependencies
                    else [render_root_query(step, state["target_type"])]
                )
                queries = list(dict.fromkeys(
                    str(value).strip() for value in queries if str(value).strip()
                ))
                if not queries:
                    raise ValueError(
                        f"no executable query for {_step_id(step, layer_index + 1)}"
                    )
                queries_by_state[state["row_index"]] = queries
                for query in queries:
                    flat_queries.append(query)
                    owners.append(state)
            except Exception as exc:
                state["error"] = exc

        if not flat_queries:
            continue
        # A full-batch retriever/reranker failure invalidates the materialisation
        # rather than silently turning the whole experimental arm into Arm A.
        flat_results = [list(result) for result in retriever.batch_search(flat_queries)]
        if len(flat_results) != len(flat_queries):
            raise ValueError(
                f"batch retriever returned {len(flat_results)}/{len(flat_queries)} rows"
            )
        flat_ranked = rerank_passages(
            flat_queries,
            flat_results,
            topk=int(args.step_rerank_topk),
            method="cross-encoder",
            cross_encoder_model=str(args.cross_encoder_model),
        )
        if len(flat_ranked) != len(flat_queries):
            raise ValueError(
                f"batch reranker returned {len(flat_ranked)}/{len(flat_queries)} rows"
            )

        results_by_state: dict[
            int, list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]
        ] = {}
        for state, query, raw, ranked in zip(
            owners, flat_queries, flat_results, flat_ranked
        ):
            results_by_state.setdefault(state["row_index"], []).append(
                (query, raw, [dict(item) for item in ranked])
            )

        for state in states:
            row_index = state["row_index"]
            if state["error"] is not None or row_index not in results_by_state:
                continue
            step = state["steps"][layer_index]
            queries = queries_by_state[row_index]
            current_results: list[dict[str, Any]] = []
            query_details: list[dict[str, Any]] = []
            seen_documents: set[str] = set()
            try:
                for query, raw, ranked in results_by_state[row_index]:
                    for passage in ranked:
                        document_key = str(passage.get("id") or _sha256_json(passage))
                        if document_key not in seen_documents:
                            seen_documents.add(document_key)
                            current_results.append(passage)
                    query_details.append({
                        "query": query,
                        "raw_count": len(raw),
                        "reranked_count": len(ranked),
                        "retrieved_ids": [str(doc.get("id") or "") for doc in ranked],
                    })
                state["hop_results"].append({
                    "hop_id": _step_id(step, layer_index + 1),
                    "query": " || ".join(queries),
                    "passages": current_results,
                })
                candidates = extract_deterministic_bridge_candidates(
                    " ; ".join(queries),
                    current_results,
                    exclude_surfaces=state["used_surfaces"],
                    max_docs=int(args.bridge_source_docs),
                    max_candidates=int(args.max_bridge_candidates),
                )
                _record_slot_values(
                    state["slot_values"],
                    step=step,
                    index=layer_index + 1,
                    candidates=candidates,
                )
                state["used_surfaces"].extend(
                    str(candidate.get("surface") or "") for candidate in candidates
                )
                state["detail"]["hops"].append({
                    "hop_id": _step_id(step, layer_index + 1),
                    "dependencies": _dependencies(step),
                    "queries": query_details,
                    "bridge_candidates": list(candidates),
                })
            except Exception as exc:
                state["error"] = exc

    for state in states:
        detail = state["detail"]
        arm_a = state["arm_a"]
        error = state["error"]
        if error is not None:
            detail.update({
                "execution_status": "fallback_execution_error",
                "fallback_exact": True,
                "plan_executable": True,
                "dependent_query_count": sum(
                    len(hop["queries"]) for hop in detail["hops"] if hop["dependencies"]
                ),
                "second_hop_query_count": sum(
                    len(hop["queries"]) for hop in detail["hops"] if hop["dependencies"]
                ),
                "new_dependent_candidate_count": 0,
                "fallback_reason": "execution_error",
                "error": {"type": type(error).__name__, "message": str(error)},
            })
            outcomes[state["row_index"]] = (arm_a, detail)
            continue
        try:
            merged, merge_telemetry = merge_passages_with_provenance(
                arm_a,
                state["hop_results"],
                original_quota=int(args.original_quota),
                per_hop_quota=int(args.per_hop_quota),
                total=int(args.total_passages),
            )
            if not merged:
                raise ValueError("dependent retrieval produced an empty merged passage list")
            detail.update({
                "execution_status": "executed",
                "fallback_exact": False,
                "merge": merge_telemetry,
                "arm_b_passages_sha256": _sha256_json(merged),
                "plan_executable": True,
                "dependent_query_count": sum(
                    len(hop["queries"]) for hop in detail["hops"] if hop["dependencies"]
                ),
                "second_hop_query_count": sum(
                    len(hop["queries"]) for hop in detail["hops"] if hop["dependencies"]
                ),
                "new_dependent_candidate_count": sum(
                    int(count)
                    for source, count in dict(
                        merge_telemetry.get("selected_by_source") or {}
                    ).items()
                    if source not in {"original_prefix", "original_backfill"}
                ),
                "fallback_reason": None,
            })
            outcomes[state["row_index"]] = (
                [dict(passage) for passage in merged], detail
            )
        except Exception as exc:
            detail.update({
                "execution_status": "fallback_execution_error",
                "fallback_exact": True,
                "plan_executable": True,
                "dependent_query_count": sum(
                    len(hop["queries"]) for hop in detail["hops"] if hop["dependencies"]
                ),
                "second_hop_query_count": sum(
                    len(hop["queries"]) for hop in detail["hops"] if hop["dependencies"]
                ),
                "new_dependent_candidate_count": 0,
                "fallback_reason": "execution_error",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            })
            outcomes[state["row_index"]] = (arm_a, detail)

    if any(outcome is None for outcome in outcomes):
        raise RuntimeError("layer-batched execution did not produce every row outcome")
    return [outcome for outcome in outcomes if outcome is not None]


def _dry_run(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    for row in rows:
        dataset = str(row["dataset"])
        target_type = TARGET_TYPES[dataset]
        plan = dict(row["plan"])
        errors = validate_plan_for_dependent_retrieval(plan, target_type)
        rendered: list[dict[str, Any]] = []
        for index, step in enumerate(list(plan.get("steps") or []), start=1):
            try:
                if _dependencies(step):
                    # Placeholder values exercise substitution without deriving any
                    # entity from Gold or from a retrieval result.
                    slots = {
                        f"hop_{number}": [f"DRY_BRIDGE_{number}"]
                        for number in range(1, index)
                    }
                    slots.update({
                        f"step_{number}": [f"DRY_BRIDGE_{number}"]
                        for number in range(1, index)
                    })
                    queries = instantiate_dependent_queries(
                        step, target_type, slots, max_variants=int(args.max_query_variants)
                    )
                else:
                    query = render_root_query(step, target_type)
                    queries = [query] if query else []
                rendered.append({"hop_id": _step_id(step, index), "queries": queries})
            except Exception as exc:
                rendered.append({
                    "hop_id": _step_id(step, index), "queries": [],
                    "render_error": {"type": type(exc).__name__, "message": str(exc)},
                })
        counters[f"{dataset}.n"] += 1
        counters[f"{dataset}.plan_valid"] += int(not errors)
        counters[f"{dataset}.has_dependent_step"] += int(
            not errors
            and any(
                isinstance(step, Mapping) and bool(_dependencies(step))
                for step in list(plan.get("steps") or [])
            )
        )
        counters[f"{dataset}.has_rendered_query"] += int(any(x["queries"] for x in rendered))
        details.append({
            "dataset": dataset, "qid": row["qid"], "errors": errors,
            "rendered": rendered,
        })
    return {
        "status": "PASS_DRY_RUN_NO_RETRIEVAL" if all(not x["errors"] for x in details) else "FAIL_DRY_RUN",
        "runner_version": RUNNER_VERSION,
        "gold_access": False,
        "retrieval_started": False,
        "counts": dict(counters),
        "details": details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", type=Path,
        default=Path("outputs/audits/inference_proofkg_v1_n900_seed42_preregistration/pilot.question_only.jsonl"),
    )
    parser.add_argument(
        "--retrieval_contexts", type=Path,
        default=Path("outputs/audits/inference_proofkg_v1_n900_seed42_preregistration/retrieval_contexts.jsonl"),
    )
    parser.add_argument(
        "--musique_plans", type=Path,
        default=Path("outputs/audits/inference_proofkg_v1_pilot30x3_plans_v1/predictions.question_only.jsonl"),
    )
    parser.add_argument(
        "--hotpot_plans", type=Path,
        default=Path("outputs/audits/inference_proofkg_hotpot_relation_graph_pilot30_v2_plans/predictions.question_only.jsonl"),
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
    parser.add_argument("--max_hops", type=int, default=4)
    parser.add_argument("--max_query_variants", type=int, default=2)
    parser.add_argument("--bridge_source_docs", type=int, default=5)
    parser.add_argument("--max_bridge_candidates", type=int, default=2)
    parser.add_argument("--original_quota", type=int, default=6)
    parser.add_argument("--per_hop_quota", type=int, default=2)
    parser.add_argument("--total_passages", type=int, default=10)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--experiment_id")
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Validate identity joins and query rendering only; no retrieval and no output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        raise SystemExit("--datasets contains duplicates")
    if args.n_per_dataset <= 0 or args.rrf_candidate_k <= 0:
        raise SystemExit("sample and retrieval sizes must be positive")
    if args.original_quota < 0 or args.per_hop_quota <= 0 or args.total_passages <= 0:
        raise SystemExit("invalid passage quotas")
    if args.original_quota > args.total_passages:
        raise SystemExit("original_quota cannot exceed total_passages")
    rows = _load_inputs(args)
    if args.dry_run:
        print(json.dumps(_dry_run(rows, args), ensure_ascii=False, indent=2))
        return
    if args.output_dir is None or not str(args.experiment_id or "").strip():
        raise SystemExit("formal materialisation requires --output_dir and --experiment_id")

    assets = _validate_full_wiki18_assets(
        args.corpus_path, args.dense_index_path, args.bm25_index_path,
        expected_docs=int(args.expected_docs),
    )
    run_dir, experiment_id = prepare_new_run_dir(
        args.output_dir, experiment_id=args.experiment_id,
        extra={"phase": "plan_once_dependent_retrieval_materialisation", "gold_access": False},
    )
    arm_a_rows: list[dict[str, Any]] = []
    arm_b_rows: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    try:
        # Both datasets query the same immutable Wiki18 corpus/index.  Loading
        # the 21M-row retriever twice needlessly duplicates its memory footprint.
        retriever = _build_retriever(
            args.datasets[0], int(args.rrf_candidate_k),
            corpus_path=args.corpus_path,
            dense_index_path=args.dense_index_path,
            bm25_index_path=args.bm25_index_path,
        )
        # Layer batching is an execution-only optimisation.  Query rendering,
        # bridge extraction and per-row merge semantics remain identical to
        # ``_execute_row``; only the number of 21M-index scans changes.
        batched_outcomes = _execute_rows_batched(rows, retriever, args)
        for index, (row, outcome) in enumerate(zip(rows, batched_outcomes), start=1):
            arm_b, detail = outcome
            common = {
                "row_id": f"dependent-retrieval-pilot::{row['dataset']}::{row['qid']}",
                "question_key": f"{row['dataset']}::{row['qid']}",
                "dataset": row["dataset"], "qid": row["qid"],
                "question": row["question"], "question_sha256": row["question_sha256"],
                "split": row["split"], "gold_access": False,
                # Frozen identically across A/B; no KG is constructed from the
                # newly retrieved B passages in this single-variable pilot.
                "kg_subgraph": row["legacy_kg"],
                "legacy_kg_sha256": row["legacy_kg_sha256"],
            }
            arm_a = [dict(passage) for passage in row["arm_a_passages"]]
            arm_a_rows.append({
                **common, "arm": "A_question_only",
                "retrieved_passages": arm_a, "passages_sha256": _sha256_json(arm_a),
            })
            arm_b_rows.append({
                **common, "arm": "B_dependent",
                "retrieved_passages": arm_b, "passages_sha256": _sha256_json(arm_b),
                "fallback_to_a": bool(detail["fallback_exact"]),
                "retrieval_trace": {
                    "plan_executable": bool(detail["plan_executable"]),
                    "has_dependent_step": bool(detail["has_dependent_step"]),
                    "dependent_query_count": int(detail["dependent_query_count"]),
                    "second_hop_query_count": int(detail["second_hop_query_count"]),
                    "new_dependent_candidate_count": int(detail["new_dependent_candidate_count"]),
                    "fallback_reason": detail["fallback_reason"],
                },
            })
            execution_rows.append(detail)
            print(
                f"dependent retrieval {index}/{len(rows)} {row['dataset']}::{row['qid']} "
                f"status={detail['execution_status']}", flush=True,
            )

        arm_a_path, arm_b_path = run_dir / "arm_a.jsonl", run_dir / "arm_b.jsonl"
        execution_path = run_dir / "execution_details.jsonl"
        _write_jsonl(arm_a_path, arm_a_rows)
        _write_jsonl(arm_b_path, arm_b_rows)
        _write_jsonl(execution_path, execution_rows)
        by_dataset: dict[str, Any] = {}
        for dataset in args.datasets:
            current = [row for row in execution_rows if row["dataset"] == dataset]
            by_dataset[dataset] = {
                "n": len(current),
                "executed": sum(row["execution_status"] == "executed" for row in current),
                "fallback": sum(bool(row["fallback_exact"]) for row in current),
                "fallback_execution_error": sum(
                    row["execution_status"] == "fallback_execution_error" for row in current
                ),
                "runtime_errors": sum("error" in row for row in current),
                "arm_b_changed": sum(
                    a["passages_sha256"] != b["passages_sha256"]
                    for a, b in zip(
                        [x for x in arm_a_rows if x["dataset"] == dataset],
                        [x for x in arm_b_rows if x["dataset"] == dataset],
                    )
                ),
                "root_queries": sum(
                    sum(not bool(hop["dependencies"]) for hop in row["hops"])
                    for row in current
                ),
                "dependent_queries": sum(
                    sum(bool(hop["dependencies"]) for hop in row["hops"])
                    for row in current
                ),
                "dependent_step_eligible": sum(
                    bool(row["has_dependent_step"]) for row in current
                ),
                "dependent_step_query_nonempty": sum(
                    bool(row["has_dependent_step"])
                    and int(row["second_hop_query_count"]) > 0
                    for row in current
                ),
                "fallback_exact": all(
                    (not row["fallback_exact"])
                    or next(x for x in arm_a_rows if x["dataset"] == dataset and x["qid"] == row["qid"])["passages_sha256"]
                    == next(x for x in arm_b_rows if x["dataset"] == dataset and x["qid"] == row["qid"])["passages_sha256"]
                    for row in current
                ),
            }
        report = {
            "schema_version": "plan-once-dependent-retrieval-report-1",
            "runner_version": RUNNER_VERSION,
            "experiment_id": experiment_id,
            "status": "COMPLETE_INPUTS_NOT_ANSWER_EVALUATED",
            "development_only": True,
            "gold_access": False,
            "canonical_pipeline_modified": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                name: {"path": str(path.resolve()), "sha256": _sha256_file(path.resolve())}
                for name, path in {
                    "cohort": args.cohort, "retrieval_contexts": args.retrieval_contexts,
                    "musique_plans": args.musique_plans, "hotpot_plans": args.hotpot_plans,
                }.items()
            },
            "retrieval_assets": assets,
            "settings": {
                "rrf_candidate_k": args.rrf_candidate_k,
                "step_rerank_topk": args.step_rerank_topk,
                "cross_encoder_model": args.cross_encoder_model,
                "max_hops": args.max_hops,
                "max_query_variants": args.max_query_variants,
                "original_quota": args.original_quota,
                "per_hop_quota": args.per_hop_quota,
                "total_passages": args.total_passages,
            },
            "by_dataset": by_dataset,
            "outputs": {
                "arm_a": {"path": str(arm_a_path), "sha256": _sha256_file(arm_a_path)},
                "arm_b": {"path": str(arm_b_path), "sha256": _sha256_file(arm_b_path)},
                "execution_details": {"path": str(execution_path), "sha256": _sha256_file(execution_path)},
            },
            "scientific_boundary": (
                "This materialises development inputs only. Gold scoring and answer generation "
                "have not run; no effectiveness claim follows from this report."
            ),
        }
        report_path = run_dir / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        dump_manifest(
            run_dir, status=report["status"],
            extra={"experiment_id": experiment_id, "phase": "plan_once_dependent_retrieval_materialisation", "report_sha256": _sha256_file(report_path)},
        )
        print(json.dumps({"status": report["status"], "by_dataset": by_dataset}, ensure_ascii=False, indent=2))
    except Exception as exc:
        dump_manifest(
            run_dir, status="FAILED_RUNTIME",
            extra={"experiment_id": experiment_id, "phase": "plan_once_dependent_retrieval_materialisation", "failure": {"type": type(exc).__name__, "message": str(exc)}},
        )
        raise


if __name__ == "__main__":
    main()
