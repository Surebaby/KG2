#!/usr/bin/env python3
"""Materialize a clean, question-only 2Wiki root-anchor resolution cache.

This executable is intentionally separate from the preregistration/freezer.
It consumes the frozen worklist, uses a new empty cache directory, and resolves
each root in two precision-first stages:

1. exact Wikipedia-title -> Wikidata item resolution;
2. only after exact-title abstention, Wikidata candidate search scored with the
   original question text, followed by frozen score and margin gates.

It never accepts an old/expected Wikidata QID as input and has no CLI for an old
entity or title cache.  The script performs network access when executed; tests
exercise its decision logic with fakes and do not access the network.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Iterable, Mapping, Protocol, Sequence

from kgproweight.kg.entity_linker import EntityLinker, LinkCandidate, LinkResult
from kgproweight.kg.wikipedia_title_resolver import WikipediaTitleResolver
from kgproweight.utils.logging import dump_manifest


SCHEMA_VERSION = "2wiki-root-anchor-resolution-result-v1"
EXPECTED_PROTOCOL_SCHEMA = "2wiki-root-anchor-resolution-protocol-v1"
EXPECTED_WORKLIST_SCHEMA = "2wiki-root-anchor-resolution-worklist-v1"
QID_RE = re.compile(r"^Q[1-9][0-9]*$")


class TitleResolver(Protocol):
    def resolve(self, surface: str) -> LinkResult: ...


class CandidateLinker(Protocol):
    def _search_candidates(self, mention: str) -> list[LinkCandidate]: ...

    def _score_candidates(
        self,
        mention: str,
        candidates: list[LinkCandidate],
        question: str,
        expected_types: list[str] | None = None,
        retrieved_titles: list[str] | None = None,
        passage_text: str | None = None,
    ) -> list[LinkCandidate]: ...


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            result.append(value)
    return result


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _cache_key_for_result(row: Mapping[str, Any]) -> tuple[str, str] | None:
    if row.get("outcome") != "positive":
        return None
    method = str(row.get("resolution_method") or "")
    if method == "exact_wikipedia_title":
        label = str(row.get("completed_root_anchor_surface") or "").strip()
        return ("title", label.casefold()) if label else None
    if method == "wikidata_question_context":
        label = str(row.get("root_anchor_surface") or "").strip()
        return ("entity", label.casefold()) if label else None
    return None


def abstain_cross_context_cache_conflicts(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Fail closed when one global cache key received multiple accepted QIDs."""

    qids_by_key: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        key = _cache_key_for_result(row)
        if key is not None:
            qids_by_key.setdefault(key, set()).add(str(row.get("resolved_qid") or ""))
    conflicting = {key for key, qids in qids_by_key.items() if len(qids) > 1}
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if _cache_key_for_result(row) in conflicting:
            row.update(
                {
                    "resolution_method": "abstain",
                    "resolved_qid": "",
                    "resolved_label": "",
                    "outcome": "abstain",
                    "abstain_reason": "cross_context_cache_key_qid_conflict",
                }
            )
        output.append(row)
    return output, len(conflicting)


def write_canonical_resolution_caches(
    *,
    title_cache_path: Path,
    entity_cache_path: Path,
    results: Sequence[Mapping[str, Any]],
) -> None:
    """Rewrite new caches as deterministic, normalized, duplicate-free JSONL."""

    cache_rows: dict[str, dict[str, dict[str, str]]] = {"title": {}, "entity": {}}
    for row in results:
        key = _cache_key_for_result(row)
        if key is None:
            continue
        kind, normalized_label = key
        label_field = (
            "completed_root_anchor_surface"
            if kind == "title"
            else "root_anchor_surface"
        )
        label = str(row[label_field]).strip()
        qid = str(row["resolved_qid"])
        previous = cache_rows[kind].get(normalized_label)
        if previous is not None and previous["qid"] != qid:
            raise RuntimeError(f"unresolved {kind} cache QID conflict for {label!r}")
        candidate = {"label": label, "qid": qid}
        if previous is None or (label, qid) < (previous["label"], previous["qid"]):
            cache_rows[kind][normalized_label] = candidate
    for kind, path in (("title", title_cache_path), ("entity", entity_cache_path)):
        rows = sorted(
            cache_rows[kind].values(),
            key=lambda value: (value["label"].casefold(), value["label"], value["qid"]),
        )
        path.write_text(
            "".join(
                json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
                for value in rows
            ),
            encoding="utf-8",
        )


class RateLimiter:
    """Global deterministic request-start limiter shared by worker threads."""

    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = max(0.0, float(delay_seconds))
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_start - now)
            if wait_seconds:
                time.sleep(wait_seconds)
            self._next_start = time.monotonic() + self.delay_seconds


def _candidate_payload(candidate: LinkCandidate) -> dict[str, Any]:
    value = asdict(candidate)
    value["score"] = round(float(value.get("score") or 0.0), 6)
    return value


def resolve_request(
    row: Mapping[str, Any],
    *,
    title_resolver: TitleResolver,
    candidate_linker: CandidateLinker,
    limiter: RateLimiter,
    fallback_min_score: float,
    fallback_min_margin: float,
) -> dict[str, Any]:
    """Resolve one frozen request without consulting any prior QID target."""

    surface = str(row["root_anchor_surface"])
    completed_surface = str(row["completed_root_anchor_surface"])
    question = str(row["question"])

    limiter.wait()
    title_result = title_resolver.resolve(completed_surface)
    title_qid = str(title_result.selected_qid or "")
    if not title_result.abstained and QID_RE.fullmatch(title_qid):
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": str(row["request_id"]),
            "question_key": str(row["question_key"]),
            "dataset": str(row["dataset"]),
            "qid": str(row["qid"]),
            "question_sha256": str(row["question_sha256"]),
            "root_anchor_surface": surface,
            "completed_root_anchor_surface": completed_surface,
            "resolution_method": "exact_wikipedia_title",
            "resolved_qid": title_qid,
            "resolved_label": str(title_result.selected_label or completed_surface),
            "score": round(float(title_result.score), 6),
            "margin": round(float(title_result.margin), 6),
            "outcome": "positive",
            "abstain_reason": "",
            "title_abstain_reason": "",
            "fallback_candidates": [],
            "gold_access": False,
        }

    limiter.wait()
    candidates = candidate_linker._search_candidates(surface)
    scored = candidate_linker._score_candidates(
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
    second_score = float(second.score) if second else 0.0
    margin = score - second_score
    qid = str(top.qid) if top else ""
    accepted = bool(
        top
        and QID_RE.fullmatch(qid)
        and score >= float(fallback_min_score)
        and margin >= float(fallback_min_margin)
    )
    if accepted:
        reason = ""
    elif not top:
        reason = "no_wikidata_candidates"
    elif not QID_RE.fullmatch(qid):
        reason = "invalid_candidate_qid"
    elif score < float(fallback_min_score):
        reason = f"fallback_score_below_{float(fallback_min_score):.2f}"
    else:
        reason = f"fallback_margin_below_{float(fallback_min_margin):.2f}"

    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": str(row["request_id"]),
        "question_key": str(row["question_key"]),
        "dataset": str(row["dataset"]),
        "qid": str(row["qid"]),
        "question_sha256": str(row["question_sha256"]),
        "root_anchor_surface": surface,
        "completed_root_anchor_surface": completed_surface,
        "resolution_method": "wikidata_question_context" if accepted else "abstain",
        "resolved_qid": qid if accepted else "",
        "resolved_label": str(top.label) if accepted and top else "",
        "score": round(score, 6),
        "margin": round(margin, 6),
        "outcome": "positive" if accepted else "abstain",
        "abstain_reason": reason,
        "title_abstain_reason": str(
            title_result.abstain_reason or "exact_title_returned_no_valid_qid"
        ),
        "fallback_candidates": [_candidate_payload(candidate) for candidate in scored],
        "gold_access": False,
    }


def _validate_inputs(
    *, protocol: Mapping[str, Any], worklist: Sequence[Mapping[str, Any]], worklist_path: Path
) -> None:
    if protocol.get("schema_version") != EXPECTED_PROTOCOL_SCHEMA:
        raise ValueError("unexpected root-anchor resolution protocol schema")
    if protocol.get("status") != "FROZEN_BEFORE_NETWORK_NO_GOLD":
        raise ValueError("protocol is not frozen before network resolution")
    expected = ((protocol.get("outputs") or {}).get("worklist") or {}).get("sha256")
    if expected != sha256_file(worklist_path):
        raise ValueError("worklist hash does not match frozen protocol")
    if not worklist:
        raise ValueError("empty root-anchor resolution worklist")
    seen: set[str] = set()
    for row in worklist:
        if row.get("schema_version") != EXPECTED_WORKLIST_SCHEMA:
            raise ValueError("unexpected worklist row schema")
        if row.get("gold_access") is not False:
            raise ValueError("worklist gold_access must be false")
        request_id = str(row.get("request_id") or "")
        if not request_id or request_id in seen:
            raise ValueError("missing or duplicate resolver request_id")
        seen.add(request_id)


def materialize(
    *,
    protocol_path: Path,
    worklist_path: Path,
    output_dir: Path,
    workers: int,
    delay: float,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite resolver output: {output_dir}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    worklist = read_jsonl(worklist_path)
    _validate_inputs(protocol=protocol, worklist=worklist, worklist_path=worklist_path)
    frozen_policy = protocol["resolution_policy"]
    expected_policy = {
        "workers": workers,
        "request_delay_seconds": delay,
        "timeout_seconds": timeout,
        "max_retries": retries,
    }
    for key, actual in expected_policy.items():
        if float(frozen_policy[key]) != float(actual):
            raise ValueError(
                f"runtime {key}={actual!r} differs from frozen {frozen_policy[key]!r}"
            )

    output_dir.mkdir(parents=True, exist_ok=False)
    title_cache_path = output_dir / "title_cache.jsonl"
    entity_cache_path = output_dir / "entity_cache.jsonl"
    # Passing an explicit nonexistent path prevents EntityLinker from silently
    # auto-loading the project's old entity-description index.
    no_local_index = output_dir / "NO_LOCAL_ENTITY_INDEX_ALLOWED.json"
    title_resolver = WikipediaTitleResolver(
        cache_path=title_cache_path,
        offline=False,
        timeout=timeout,
        max_retries=retries,
        request_delay=delay,
    )
    candidate_linker = EntityLinker(
        cache_path=entity_cache_path,
        confidence_threshold=100.0,
        use_genre=False,
        request_delay=delay,
        offline=False,
        entity_index_path=str(no_local_index),
    )
    limiter = RateLimiter(delay)
    fallback_min_score = float(frozen_policy["fallback_min_score"])
    fallback_min_margin = float(frozen_policy["fallback_min_margin"])

    def resolve_one(row: Mapping[str, Any]) -> dict[str, Any]:
        try:
            return resolve_request(
                row,
                title_resolver=title_resolver,
                candidate_linker=candidate_linker,
                limiter=limiter,
                fallback_min_score=fallback_min_score,
                fallback_min_margin=fallback_min_margin,
            )
        except Exception as exc:  # preserve exact request identity and fail closed
            return {
                "schema_version": SCHEMA_VERSION,
                "request_id": str(row["request_id"]),
                "question_key": str(row["question_key"]),
                "dataset": str(row["dataset"]),
                "qid": str(row["qid"]),
                "question_sha256": str(row["question_sha256"]),
                "root_anchor_surface": str(row["root_anchor_surface"]),
                "completed_root_anchor_surface": str(
                    row["completed_root_anchor_surface"]
                ),
                "resolution_method": "runtime_error",
                "resolved_qid": "",
                "resolved_label": "",
                "score": 0.0,
                "margin": 0.0,
                "outcome": "fail",
                "abstain_reason": f"{type(exc).__name__}: {exc}",
                "title_abstain_reason": "",
                "fallback_candidates": [],
                "gold_access": False,
            }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(resolve_one, worklist))
    results, cache_key_conflicts = abstain_cross_context_cache_conflicts(results)
    results.sort(
        key=lambda row: (
            str(row["question_key"]),
            str(row["root_anchor_surface"]).casefold(),
        )
    )

    # Resolver workers append title-cache entries in completion order. Rebuild
    # both NEW caches from the accepted, conflict-checked result set so cache
    # bytes are deduplicated and deterministic regardless of worker scheduling.
    write_canonical_resolution_caches(
        title_cache_path=title_cache_path,
        entity_cache_path=entity_cache_path,
        results=results,
    )

    result_path = output_dir / "resolution_results.jsonl"
    write_jsonl(result_path, results)
    request_ids = {str(row["request_id"]) for row in worklist}
    result_ids = {str(row["request_id"]) for row in results}
    method_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    for row in results:
        method = str(row["resolution_method"])
        outcome = str(row["outcome"])
        method_counts[method] = method_counts.get(method, 0) + 1
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
    failed = int(outcome_counts.get("fail", 0))
    resolved = int(outcome_counts.get("positive", 0))
    by_question: dict[str, list[Mapping[str, Any]]] = {}
    for row in results:
        by_question.setdefault(str(row["question_key"]), []).append(row)
    newly_all_roots_resolved_questions = sum(
        all(row.get("outcome") == "positive" for row in rows)
        for rows in by_question.values()
    )
    frozen_counts = protocol["counts"]
    total_questions = int(frozen_counts["questions"])
    total_anchor_occurrences = int(frozen_counts["anchor_occurrences"])
    final_all_roots_resolved_questions = int(
        frozen_counts["all_roots_resolved_questions"]
    ) + newly_all_roots_resolved_questions
    final_resolved_anchor_occurrences = int(
        frozen_counts["resolved_anchor_occurrences"]
    ) + resolved
    all_roots_rate = final_all_roots_resolved_questions / total_questions
    occurrence_rate = final_resolved_anchor_occurrences / total_anchor_occurrences
    request_join_rate = len(request_ids & result_ids) / len(request_ids)
    gate = {
        "request_log_join_rate": request_join_rate == 1.0,
        "runtime_errors_zero": failed == 0,
        "all_roots_resolved_question_rate_ge_0_80": all_roots_rate >= 0.80,
        "anchor_occurrence_resolution_rate_ge_0_80": occurrence_rate >= 0.80,
        "gold_access_false": all(row["gold_access"] is False for row in results),
        "old_cache_fallback_false": True,
    }
    gate_pass = all(gate.values())
    run_status = (
        "ROOT_ANCHOR_RESOLUTION_COMPLETE_WITH_ERRORS"
        if failed
        else "PASS_ROOT_ANCHOR_CONTINUE_GATE"
        if gate_pass
        else "FAIL_ROOT_ANCHOR_CONTINUE_GATE"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": run_status,
        "counts": {
            "requests": len(worklist),
            "results": len(results),
            "positive": resolved,
            "abstain": int(outcome_counts.get("abstain", 0)),
            "fail": failed,
            "cache_key_conflicts_abstained": cache_key_conflicts,
            "by_method": dict(sorted(method_counts.items())),
        },
        "post_resolution_coverage": {
            "newly_all_roots_resolved_questions": newly_all_roots_resolved_questions,
            "final_all_roots_resolved_questions": final_all_roots_resolved_questions,
            "total_questions": total_questions,
            "all_roots_resolved_question_rate": all_roots_rate,
            "final_resolved_anchor_occurrences": final_resolved_anchor_occurrences,
            "total_anchor_occurrences": total_anchor_occurrences,
            "anchor_occurrence_resolution_rate": occurrence_rate,
        },
        "continue_gate_before_clean_closure_v2": {
            "checks": gate,
            "all_pass": gate_pass,
            "decision": "CONTINUE_TO_CLEAN_CLOSURE_V2" if gate_pass else "STOP_RETAIN_RESULT",
        },
        "checks": {
            "request_log_join_rate": request_join_rate,
            "duplicate_result_ids": len(results) - len(result_ids),
            "gold_access_false": all(row["gold_access"] is False for row in results),
            "old_cache_fallback": False,
            "runtime_errors": failed,
        },
        "policy": {
            "workers": workers,
            "request_delay_seconds": delay,
            "timeout_seconds": timeout,
            "max_retries": retries,
            "fallback_min_score": fallback_min_score,
            "fallback_min_margin": fallback_min_margin,
        },
        "inputs": {
            "protocol": file_identity(protocol_path),
            "worklist": file_identity(worklist_path),
        },
        "outputs": {
            "resolution_results": file_identity(result_path),
            "title_cache": file_identity(title_cache_path)
            if title_cache_path.is_file()
            else None,
            "entity_cache": file_identity(entity_cache_path)
            if entity_cache_path.is_file()
            else None,
        },
        "network_access": True,
        "gold_access": False,
        "training_started": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        output_dir,
        status=report["status"],
        extra={
            "phase": "materialize_clean_2wiki_root_anchor_resolution",
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    report = materialize(
        protocol_path=args.protocol,
        worklist_path=args.worklist,
        output_dir=args.output_dir,
        workers=args.workers,
        delay=args.delay,
        timeout=args.timeout,
        retries=args.retries,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
