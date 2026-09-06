#!/usr/bin/env python3
"""Materialize and exact-consumer replay all official-raw n1500 roots.

Resolution is Gold-free and precision first.  Every recognized planner root is
processed afresh in this order: exact Wikipedia title, unique v6 clean alias,
then question-context Wikidata search.  Only newly created title/entity caches
are written.  The continuation decision is computed by an offline replay using
the actual final consumer classes and precedence, never by adding new successes
to an old resolver count.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import requests

from kgproweight.kg.entity_linker import (
    DEFAULT_PROXY_HEADERS,
    EntityLinker,
    LinkCandidate,
    LinkResult,
    WIKIDATA_SEARCH_URL,
    WIKIDATA_USER_AGENT,
)
from kgproweight.kg.versioned_evidence_store import VersionedEvidenceStore
from kgproweight.kg.wikipedia_title_resolver import WikipediaTitleResolver
from kgproweight.utils.logging import dump_manifest
from scripts.pilot.build_automatic_proofkg_from_plans import (
    _ExactEntityCacheLinker,
    _ResolverChain,
)
from scripts.pilot.build_query_aware_proof_kg_pilot import _link_surface
from scripts.prepare.freeze_2wiki_official_raw_full_root_resolution_v2 import (
    PROTOCOL_SCHEMA,
    STATUS as FROZEN_STATUS,
    WORKLIST_FIELDS,
    WORKLIST_SCHEMA,
    file_identity,
    read_jsonl,
    sha256_file,
    write_jsonl,
)
from scripts.prepare.materialize_2wiki_root_anchor_resolution_v1 import (
    RateLimiter,
    abstain_cross_context_cache_conflicts,
    write_canonical_resolution_caches,
)


RESULT_SCHEMA = "2wiki-full-root-anchor-resolution-result-v2"
DRY_RUN_SCHEMA = "2wiki-full-root-anchor-consumer-dry-run-v2"
QID_RE = re.compile(r"^Q[1-9][0-9]*$")
PASS_STATUS = "PASS_ROOT_ANCHOR_CONTINUE_GATE_EXACT_CONSUMER"
FAIL_STATUS = "FAIL_ROOT_ANCHOR_CONTINUE_GATE_EXACT_CONSUMER"

RESULT_FIELDS = {
    "schema_version", "request_id", "question_key", "dataset", "qid",
    "question_sha256", "question_type", "root_position",
    "root_anchor_surface", "completed_root_anchor_surface",
    "resolution_method", "resolved_qid", "resolved_label", "score", "margin",
    "outcome", "abstain_reason", "title_abstain_reason",
    "fallback_candidates", "candidate_search_attempts", "gold_access",
}
DRY_RUN_FIELDS = {
    "schema_version", "request_id", "question_key", "dataset", "qid",
    "question_sha256", "root_position", "root_anchor_surface",
    "completed_root_anchor_surface", "completed_surface", "projected_qid", "dry_run_qid",
    "resolution_source", "matched", "abstained", "abstain_reason", "gold_access",
}


def _candidate_payload(candidate: LinkCandidate) -> dict[str, Any]:
    value = asdict(candidate)
    value["score"] = round(float(value.get("score") or 0.0), 6)
    return value


def _search_candidates(
    surface: str,
    *,
    limiter: RateLimiter,
    timeout: float,
    retries: int,
) -> tuple[list[LinkCandidate], int, str]:
    """Run a bounded Wikidata entity search with explicit retry telemetry."""

    last_error = ""
    for attempt in range(1, max(1, int(retries)) + 1):
        limiter.wait()
        try:
            response = requests.get(
                WIKIDATA_SEARCH_URL,
                params={
                    "action": "wbsearchentities",
                    "search": surface,
                    "language": "en",
                    "format": "json",
                    "limit": 10,
                    "props": "",
                },
                headers={"User-Agent": WIKIDATA_USER_AGENT, **DEFAULT_PROXY_HEADERS},
                timeout=float(timeout),
            )
            response.raise_for_status()
            payload = response.json()
            candidates = [
                LinkCandidate(
                    qid=str(item.get("id") or ""),
                    label=str(item.get("label") or surface),
                    description=str(item.get("description") or ""),
                )
                for item in payload.get("search") or []
                if item.get("id")
            ]
            return candidates, attempt, ""
        except (requests.RequestException, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return [], max(1, int(retries)), last_error


def resolve_request(
    row: Mapping[str, Any],
    *,
    title_resolver: WikipediaTitleResolver,
    alias_store: VersionedEvidenceStore,
    candidate_scorer: EntityLinker,
    limiter: RateLimiter,
    fallback_min_score: float,
    fallback_min_margin: float,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    surface = str(row["root_anchor_surface"])
    completed = str(row["completed_root_anchor_surface"])
    question = str(row["question"])
    base = {
        "schema_version": RESULT_SCHEMA,
        "request_id": str(row["request_id"]),
        "question_key": str(row["question_key"]),
        "dataset": str(row["dataset"]),
        "qid": str(row["qid"]),
        "question_sha256": str(row["question_sha256"]),
        "question_type": str(row["question_type"]),
        "root_position": int(row["root_position"]),
        "root_anchor_surface": surface,
        "completed_root_anchor_surface": completed,
        "gold_access": False,
    }

    limiter.wait()
    title = title_resolver.resolve(completed)
    title_qid = str(title.selected_qid or "")
    if not title.abstained and QID_RE.fullmatch(title_qid):
        return {
            **base,
            "resolution_method": "exact_wikipedia_title",
            "resolved_qid": title_qid,
            "resolved_label": str(title.selected_label or completed),
            "score": round(float(title.score), 6),
            "margin": round(float(title.margin), 6),
            "outcome": "positive",
            "abstain_reason": "",
            "title_abstain_reason": "",
            "fallback_candidates": [],
            "candidate_search_attempts": 0,
        }

    alias = alias_store.resolve(completed)
    alias_qid = str(alias.selected_qid or "")
    if not alias.abstained and QID_RE.fullmatch(alias_qid):
        return {
            **base,
            "resolution_method": "clean_v6_exact_alias",
            "resolved_qid": alias_qid,
            "resolved_label": str(alias.selected_label or completed),
            "score": round(float(alias.score), 6),
            "margin": round(float(alias.margin), 6),
            "outcome": "positive",
            "abstain_reason": "",
            "title_abstain_reason": str(title.abstain_reason or "title abstained"),
            "fallback_candidates": [],
            "candidate_search_attempts": 0,
        }

    candidates, attempts, network_error = _search_candidates(
        surface, limiter=limiter, timeout=timeout, retries=retries
    )
    scored = candidate_scorer._score_candidates(
        surface,
        candidates,
        question,
        expected_types=None,
        retrieved_titles=[],
        passage_text="",
    )
    top = scored[0] if scored else None
    second = scored[1] if len(scored) > 1 else None
    score = float(top.score) if top else 0.0
    margin = score - (float(second.score) if second else 0.0)
    qid = str(top.qid) if top else ""
    accepted = bool(
        top
        and QID_RE.fullmatch(qid)
        and score >= fallback_min_score
        and margin >= fallback_min_margin
    )
    if accepted:
        reason = ""
    elif network_error:
        reason = f"wikidata_search_failed_after_{attempts}_attempts: {network_error}"
    elif not top:
        reason = "no_wikidata_candidates"
    elif not QID_RE.fullmatch(qid):
        reason = "invalid_candidate_qid"
    elif score < fallback_min_score:
        reason = f"fallback_score_below_{fallback_min_score:.2f}"
    else:
        reason = f"fallback_margin_below_{fallback_min_margin:.2f}"
    return {
        **base,
        "resolution_method": "wikidata_question_context" if accepted else "abstain",
        "resolved_qid": qid if accepted else "",
        "resolved_label": str(top.label) if accepted and top else "",
        "score": round(score, 6),
        "margin": round(margin, 6),
        "outcome": "positive" if accepted else "abstain",
        "abstain_reason": reason,
        "title_abstain_reason": str(title.abstain_reason or "title abstained"),
        "fallback_candidates": [_candidate_payload(candidate) for candidate in scored],
        "candidate_search_attempts": attempts,
    }


def exact_consumer_dry_run(
    *,
    worklist: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    title_cache_path: Path,
    entity_cache_path: Path,
    v6_store_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay the literal final builder stack and compare every projected QID."""

    result_by_id = {str(row["request_id"]): row for row in results}
    if len(result_by_id) != len(results):
        raise ValueError("duplicate result request id")
    title = WikipediaTitleResolver(cache_path=title_cache_path, offline=True)
    alias = VersionedEvidenceStore(v6_store_dir)
    linker = _ExactEntityCacheLinker(entity_cache_path)
    chain = _ResolverChain(title, alias)
    rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    resolved_by_question: defaultdict[str, list[bool]] = defaultdict(list)
    for request in worklist:
        request_id = str(request["request_id"])
        result = result_by_id.get(request_id)
        if result is None:
            raise ValueError(f"missing materialized result for {request_id}")
        surface = str(request["root_anchor_surface"])
        completed = str(request["completed_root_anchor_surface"])
        question = str(request["question"])
        actual = _link_surface(linker, surface, question, title_resolver=chain)
        if str(actual.get("resolved_surface") or "") != completed:
            raise RuntimeError("consumer completion differs from frozen completed root")
        dry_qid = str(actual.get("qid") or "")
        projected_qid = (
            str(result.get("resolved_qid") or "")
            if result.get("outcome") == "positive"
            else ""
        )

        # Source naming is diagnostic; the authoritative QID above comes from
        # the exact production helper `_link_surface` and production resolver chain.
        title_result = title.resolve(completed)
        if not title_result.abstained and title_result.selected_qid:
            source = "new_exact_title_cache"
            source_qid = str(title_result.selected_qid)
        else:
            alias_result = alias.resolve(completed)
            if not alias_result.abstained and alias_result.selected_qid:
                source = "clean_v6_exact_alias"
                source_qid = str(alias_result.selected_qid)
            else:
                entity_result = linker.link_single(surface)
                if not entity_result.abstained and entity_result.selected_qid:
                    source = "new_exact_entity_cache"
                    source_qid = str(entity_result.selected_qid)
                else:
                    source = "abstain"
                    source_qid = ""
        if source_qid != dry_qid:
            raise RuntimeError("diagnostic source replay differs from production consumer")
        matched = projected_qid == dry_qid
        source_counts[source] += 1
        resolved_by_question[str(request["question_key"])].append(bool(dry_qid))
        rows.append(
            {
                "schema_version": DRY_RUN_SCHEMA,
                "request_id": request_id,
                "question_key": str(request["question_key"]),
                "dataset": str(request["dataset"]),
                "qid": str(request["qid"]),
                "question_sha256": str(request["question_sha256"]),
                "root_position": int(request["root_position"]),
                "root_anchor_surface": surface,
                "completed_root_anchor_surface": completed,
                "completed_surface": completed,
                "projected_qid": projected_qid,
                "dry_run_qid": dry_qid,
                "resolution_source": source,
                "matched": matched,
                "abstained": not bool(dry_qid),
                "abstain_reason": str(actual.get("abstain_reason") or "") if not dry_qid else "",
                "gold_access": False,
            }
        )
    rows.sort(key=lambda row: (str(row["question_key"]), int(row["root_position"])))
    if any(set(row) != DRY_RUN_FIELDS for row in rows):
        raise RuntimeError("consumer dry-run schema drift")
    resolved_occurrences = sum(bool(row["dry_run_qid"]) for row in rows)
    all_root_questions = sum(all(flags) for flags in resolved_by_question.values())
    matches = sum(bool(row["matched"]) for row in rows)
    return rows, {
        "resolved_anchor_occurrences": resolved_occurrences,
        "all_roots_resolved_questions": all_root_questions,
        "occurrence_matches": matches,
        "occurrence_mismatches": len(rows) - matches,
        "resolution_sources": dict(sorted(source_counts.items())),
    }


def _validate_inputs(
    protocol: Mapping[str, Any],
    worklist: Sequence[Mapping[str, Any]],
    worklist_path: Path,
    v6_store_dir: Path,
) -> None:
    if protocol.get("schema_version") != PROTOCOL_SCHEMA or protocol.get("status") != FROZEN_STATUS:
        raise ValueError("unexpected or unfrozen full-root protocol")
    expected_worklist = ((protocol.get("outputs") or {}).get("worklist") or {}).get("sha256")
    if expected_worklist != sha256_file(worklist_path):
        raise ValueError("worklist hash differs from protocol")
    expected_store = ((protocol.get("inputs") or {}).get("v6_store_manifest") or {}).get("sha256")
    expected_alias = ((protocol.get("inputs") or {}).get("v6_aliases") or {}).get("sha256")
    if expected_store != sha256_file(v6_store_dir / "store_manifest.json"):
        raise ValueError("v6 store manifest differs from protocol")
    if expected_alias != sha256_file(v6_store_dir / "aliases.jsonl"):
        raise ValueError("v6 alias file differs from protocol")
    expected_code = ((protocol.get("inputs") or {}).get("resolver_implementation") or {}).get("sha256")
    if expected_code != sha256_file(Path(__file__)):
        raise ValueError("resolver implementation differs from frozen protocol")
    if len(worklist) != int((protocol.get("counts") or {}).get("root_anchor_occurrences", -1)):
        raise ValueError("worklist occurrence count differs from protocol")
    if not worklist or any(set(row) != WORKLIST_FIELDS for row in worklist):
        raise ValueError("invalid full-root worklist schema")
    if any(row.get("schema_version") != WORKLIST_SCHEMA or row.get("gold_access") is not False for row in worklist):
        raise ValueError("invalid worklist schema version or gold boundary")
    ids = [str(row.get("request_id") or "") for row in worklist]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("missing or duplicate worklist request id")


def materialize(
    *,
    protocol_path: Path,
    worklist_path: Path,
    v6_store_dir: Path,
    output_dir: Path,
    workers: int,
    delay: float,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite append-only output: {output_dir}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    worklist = read_jsonl(worklist_path)
    _validate_inputs(protocol, worklist, worklist_path, v6_store_dir)
    policy = protocol["resolution_policy"]
    for key, actual in {
        "workers": workers,
        "request_delay_seconds": delay,
        "timeout_seconds": timeout,
        "max_retries": retries,
    }.items():
        if float(policy[key]) != float(actual):
            raise ValueError(f"runtime {key} differs from frozen policy")
    if workers != 2 or workers < 1:
        raise ValueError("full-root resolver requires frozen workers=2")

    output_dir.mkdir(parents=True, exist_ok=False)
    title_cache_path = output_dir / "title_cache.jsonl"
    entity_cache_path = output_dir / "entity_cache.jsonl"
    title_resolver = WikipediaTitleResolver(
        cache_path=title_cache_path,
        offline=False,
        timeout=timeout,
        max_retries=retries,
        request_delay=delay,
    )
    alias_store = VersionedEvidenceStore(v6_store_dir)
    # Used only for deterministic scoring. No local index, cache lookup, hard
    # coded fix, or link_single path is invoked.
    no_local_index = output_dir / "NO_LOCAL_ENTITY_INDEX_ALLOWED.json"
    candidate_scorer = EntityLinker(
        cache_path=str(entity_cache_path),
        confidence_threshold=100.0,
        use_genre=False,
        request_delay=delay,
        offline=True,
        entity_index_path=str(no_local_index),
    )
    limiter = RateLimiter(delay)
    min_score = float(policy["fallback_min_score"])
    min_margin = float(policy["fallback_min_margin"])

    def run_one(row: Mapping[str, Any]) -> dict[str, Any]:
        try:
            return resolve_request(
                row,
                title_resolver=title_resolver,
                alias_store=alias_store,
                candidate_scorer=candidate_scorer,
                limiter=limiter,
                fallback_min_score=min_score,
                fallback_min_margin=min_margin,
                timeout=timeout,
                retries=retries,
            )
        except Exception as exc:
            return {
                "schema_version": RESULT_SCHEMA,
                "request_id": str(row["request_id"]),
                "question_key": str(row["question_key"]),
                "dataset": str(row["dataset"]),
                "qid": str(row["qid"]),
                "question_sha256": str(row["question_sha256"]),
                "question_type": str(row["question_type"]),
                "root_position": int(row["root_position"]),
                "root_anchor_surface": str(row["root_anchor_surface"]),
                "completed_root_anchor_surface": str(row["completed_root_anchor_surface"]),
                "resolution_method": "runtime_error",
                "resolved_qid": "",
                "resolved_label": "",
                "score": 0.0,
                "margin": 0.0,
                "outcome": "fail",
                "abstain_reason": f"{type(exc).__name__}: {exc}",
                "title_abstain_reason": "",
                "fallback_candidates": [],
                "candidate_search_attempts": 0,
                "gold_access": False,
            }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(run_one, worklist), start=1):
            results.append(result)
            if index % 100 == 0 or index == len(worklist):
                print(f"resolved {index}/{len(worklist)} roots", flush=True)
    results, conflicts = abstain_cross_context_cache_conflicts(results)
    results.sort(key=lambda row: (str(row["question_key"]), int(row["root_position"])))
    if any(set(row) != RESULT_FIELDS for row in results):
        raise RuntimeError("resolution result schema drift")
    write_canonical_resolution_caches(
        title_cache_path=title_cache_path,
        entity_cache_path=entity_cache_path,
        results=results,
    )
    result_path = output_dir / "resolution_results.jsonl"
    write_jsonl(result_path, results)

    dry_rows, dry = exact_consumer_dry_run(
        worklist=worklist,
        results=results,
        title_cache_path=title_cache_path,
        entity_cache_path=entity_cache_path,
        v6_store_dir=v6_store_dir,
    )
    dry_path = output_dir / "consumer_dry_run.jsonl"
    write_jsonl(dry_path, dry_rows)

    outcomes = Counter(str(row["outcome"]) for row in results)
    methods = Counter(str(row["resolution_method"]) for row in results)
    request_ids = {str(row["request_id"]) for row in worklist}
    result_ids = {str(row["request_id"]) for row in results}
    frozen_counts = protocol["counts"]
    total_questions = int(frozen_counts["questions_total"])
    recognized_questions = int(frozen_counts["questions_recognized"])
    occurrences = int(frozen_counts["root_anchor_occurrences"])
    projected_occurrences = sum(
        row.get("outcome") == "positive" and bool(row.get("resolved_qid"))
        for row in results
    )
    projected_by_question: defaultdict[str, list[bool]] = defaultdict(list)
    for row in results:
        projected_by_question[str(row["question_key"])].append(
            row.get("outcome") == "positive" and bool(row.get("resolved_qid"))
        )
    projected_all = sum(all(flags) for flags in projected_by_question.values())
    failed = int(outcomes.get("fail", 0))
    question_join_rate = len(projected_by_question) / max(1, recognized_questions)
    request_join_rate = len(request_ids & result_ids) / max(1, len(request_ids))
    occurrence_rate = dry["resolved_anchor_occurrences"] / max(1, occurrences)
    all_question_rate = dry["all_roots_resolved_questions"] / max(1, total_questions)
    all_recognized_rate = dry["all_roots_resolved_questions"] / max(1, recognized_questions)
    match_rate = dry["occurrence_matches"] / max(1, occurrences)
    gates: dict[str, Any] = {
        "question_identity_join_eq_1": question_join_rate == 1.0,
        "request_result_join_eq_1": request_join_rate == 1.0 and len(result_ids) == len(request_ids),
        "recognized_plan_rate_ge_0_97": recognized_questions / total_questions >= 0.97,
        "runtime_errors_zero": failed == 0,
        "gold_access_false": all(row["gold_access"] is False for row in results + dry_rows),
        "v6_binding_exact": (
            sha256_file(v6_store_dir / "store_manifest.json")
            == protocol["inputs"]["v6_store_manifest"]["sha256"]
            and sha256_file(v6_store_dir / "aliases.jsonl")
            == protocol["inputs"]["v6_aliases"]["sha256"]
        ),
        "worklist_all_recognized_roots_exact": len(worklist) == occurrences,
        "projection_equals_dry_run_every_occurrence": dry["occurrence_mismatches"] == 0,
        "all_roots_resolved_question_rate_ge_0_80": all_question_rate >= 0.80,
        "anchor_occurrence_resolution_rate_ge_0_80": occurrence_rate >= 0.80,
    }
    all_pass = all(gates.values())
    gates["all_pass"] = all_pass
    gates["decision"] = "CONTINUE_TO_V6_PROPERTY_CLOSURE" if all_pass else "STOP_RETAIN_RESULT"
    counts = {
        "questions_total": total_questions,
        "questions_recognized": recognized_questions,
        "questions_unrecognized": int(frozen_counts["questions_unrecognized"]),
        "root_anchor_occurrences": occurrences,
        "requests": len(worklist),
        "results": len(results),
        "positive": int(outcomes.get("positive", 0)),
        "abstain": int(outcomes.get("abstain", 0)),
        "fail": failed,
        "cache_key_conflicts_abstained": conflicts,
        "by_method": dict(sorted(methods.items())),
        "projected_resolved_anchor_occurrences": projected_occurrences,
        "dry_run_resolved_anchor_occurrences": dry["resolved_anchor_occurrences"],
        "projected_all_roots_resolved_questions": projected_all,
        "dry_run_all_roots_resolved_questions": dry["all_roots_resolved_questions"],
        "projection_dry_run_occurrence_matches": dry["occurrence_matches"],
        "projection_dry_run_occurrence_mismatches": dry["occurrence_mismatches"],
    }
    rates = {
        "recognized_question_rate": recognized_questions / total_questions,
        "dry_run_all_roots_resolved_question_rate_all_questions": all_question_rate,
        "dry_run_all_roots_resolved_question_rate_recognized_questions": all_recognized_rate,
        "dry_run_anchor_occurrence_resolution_rate": occurrence_rate,
        "projection_dry_run_occurrence_match_rate": match_rate,
    }
    report = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": str(protocol["experiment_id"]).replace("-PREREGISTRATION", ""),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": PASS_STATUS if all_pass else FAIL_STATUS,
        "counts": counts,
        "rates": rates,
        "resolution_sources": dry["resolution_sources"],
        "gates": gates,
        "checks": {
            "question_identity_join_rate": question_join_rate,
            "request_result_join_rate": request_join_rate,
            "duplicate_result_ids": len(results) - len(result_ids),
            "projection_dry_run_mismatches": dry["occurrence_mismatches"],
            "old_resolved_bit_inherited": False,
            "old_cache_fallback": False,
            "wide_neighborhood_fallback": False,
            "runtime_errors": failed,
            "gold_access": False,
        },
        "policy": {
            "workers": workers,
            "request_delay_seconds": delay,
            "timeout_seconds": timeout,
            "max_retries": retries,
            "fallback_min_score": min_score,
            "fallback_min_margin": min_margin,
            "consumer_order": policy["consumer_order"],
        },
        "inputs": {
            "candidate_cohort": protocol["inputs"]["candidate_cohort"],
            "planner_predictions": protocol["inputs"]["planner_predictions"],
            "planner_postflight": protocol["inputs"]["planner_postflight"],
            "root_gap_audit": protocol["inputs"]["root_gap_audit"],
            "resolver_implementation": protocol["inputs"]["resolver_implementation"],
            "v6_store_manifest": file_identity(v6_store_dir / "store_manifest.json"),
            "v6_aliases": file_identity(v6_store_dir / "aliases.jsonl"),
            "protocol": file_identity(protocol_path),
            "worklist": file_identity(worklist_path),
        },
        "outputs": {
            "resolution_results": file_identity(result_path),
            "title_cache": file_identity(title_cache_path),
            "entity_cache": file_identity(entity_cache_path),
            "consumer_dry_run": file_identity(dry_path),
        },
        "network_access": True,
        "gold_access": False,
        "training_started": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(
        output_dir,
        status=report["status"],
        extra={
            "phase": "materialize_2wiki_official_raw_full_root_resolution_v2",
            "experiment_id": report["experiment_id"],
            "report": file_identity(report_path),
            "network_access": True,
            "gold_access": False,
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--worklist", type=Path, required=True)
    parser.add_argument("--v6-store", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    report = materialize(
        protocol_path=args.protocol,
        worklist_path=args.worklist,
        v6_store_dir=args.v6_store,
        output_dir=args.output_dir,
        workers=args.workers,
        delay=args.delay,
        timeout=args.timeout,
        retries=args.retries,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
