#!/usr/bin/env python3
"""Audit the n=300 2Wiki root-resolution projection gap without Gold.

The historical root resolver reported a projected 281/300 questions with all
roots resolved, while the exact-cache-only closure-v2 execution observed
232/300.  This audit reconstructs the exact offline consumer lookup order and
separates a resolver-delta projection from consumer-realizable coverage.

It reads only frozen planner/runtime/cache artifacts.  It never reads Gold,
performs network access, changes an old result, or supplies old QIDs to a
resolver.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.versioned_evidence_store import normalize_alias
from kgproweight.utils.logging import dump_manifest


ROOT = Path(__file__).resolve().parents[2]
DATASET = "2wikimultihopqa"
SCHEMA_VERSION = "2wiki-n300-root-projection-gap-audit-v1"
DETAIL_SCHEMA_VERSION = "2wiki-n300-root-projection-state-v1"
EXPERIMENT_ID = "2WIKI-PROOFKG-EXTENSION-V1B-N300-ROOT-PROJECTION-GAP-AUDIT-V1"

DEFAULT_OLD_RUNTIME = ROOT / (
    "data/derived/2wiki_proofkg_extension_v1b_n300_closure_v1/round_2/"
    "runtime/runtime_details.jsonl"
)
DEFAULT_NEW_RUNTIME = ROOT / (
    "data/derived/2wiki_proofkg_extension_v1b_n300_closure_v2/round_2/"
    "runtime/runtime_details.jsonl"
)
DEFAULT_NEW_RUNTIME_REPORT = DEFAULT_NEW_RUNTIME.parent / "report.json"
DEFAULT_RESOLVER_DIR = ROOT / (
    "indexes/2wiki_proofkg_extension_v1b_n300_root_anchor_resolution_v1"
)
DEFAULT_RESOLVER_PROTOCOL = ROOT / (
    "outputs/audits/2wiki_proofkg_extension_v1b_n300_"
    "root_anchor_resolution_v1_preregistration/protocol.json"
)
DEFAULT_CLEAN_STORE = ROOT / (
    "indexes/versioned_2wiki_evidence_store_v5_mixed3_v4_seed42"
)
DEFAULT_LEGACY_ENTITY_INDEX = ROOT / (
    "data/silver_data/pilots/entity_candidates_local_roots_v2_20260828/"
    "entity_desc_index.expanded_local_roots.json"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/audits/2wiki_proofkg_extension_v1b_n300_"
    "root_projection_gap_audit_v1"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


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


def _resolved(entity: Mapping[str, Any]) -> bool:
    return bool(entity.get("qid")) and not bool(entity.get("abstained"))


def _runtime_index(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = str(row.get("question_key") or "")
        if (
            str(row.get("dataset") or "") != DATASET
            or not key.startswith(f"{DATASET}::")
            or not str(row.get("qid") or "")
            or key in result
        ):
            raise ValueError(f"{label}: invalid or duplicate identity {key!r}")
        anchors = list(((row.get("query_plan") or {}).get("anchors") or []))
        entities = (row.get("execution") or {}).get("anchor_entities") or {}
        if not anchors or len(anchors) != len(set(anchors)) or set(anchors) != set(entities):
            raise ValueError(f"{label}: root-anchor identity drift for {key}")
        result[key] = row
    return result


def _cache_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in read_jsonl(path):
        key = str(row.get("label") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        if not key or not qid:
            raise ValueError(f"invalid exact-cache row in {path}")
        if key in result and result[key] != qid:
            raise ValueError(f"conflicting exact-cache key {key!r} in {path}")
        result[key] = qid
    return result


def _alias_map(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for row in read_jsonl(path):
        key = str(row.get("normalized_alias") or "")
        qids = {
            str(candidate.get("qid") or "")
            for candidate in row.get("candidates") or []
            if candidate.get("qid")
        }
        if key in result and result[key] != qids:
            raise ValueError(f"duplicate conflicting alias key {key!r}")
        result[key] = qids
    return result


def consumer_resolution(
    *,
    surface: str,
    completed_surface: str,
    title_cache: Mapping[str, str],
    clean_aliases: Mapping[str, set[str]],
    entity_cache: Mapping[str, str],
) -> tuple[str, str | None]:
    """Replay the closure-v2 root lookup order exactly, without executing it."""

    title_qid = title_cache.get(completed_surface.strip().lower())
    if title_qid:
        return "new_exact_title_cache", title_qid
    alias_qids = clean_aliases.get(normalize_alias(completed_surface), set())
    if len(alias_qids) == 1:
        return "clean_v5_exact_alias", next(iter(alias_qids))
    entity_qid = entity_cache.get(surface.strip().lower())
    if entity_qid:
        return "new_exact_entity_cache", entity_qid
    return (
        "clean_v5_alias_ambiguous_then_exact_entity_miss"
        if alias_qids
        else "all_clean_consumer_sources_miss",
        None,
    )


def question_projection_state(
    *,
    old_entities: Mapping[str, Mapping[str, Any]],
    new_entities: Mapping[str, Mapping[str, Any]],
    resolver_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Contrast the historical delta projection with actual consumer state."""

    old_flags = [_resolved(entity) for entity in old_entities.values()]
    new_flags = [_resolved(entity) for entity in new_entities.values()]
    old_state = "all" if all(old_flags) else "partial" if any(old_flags) else "none"
    delta_all_positive = bool(resolver_rows) and all(
        row.get("outcome") == "positive" for row in resolver_rows
    )
    return {
        "old_resolution_state": old_state,
        # This mirrors the v1 report: baseline all-resolved questions plus
        # questions whose *previously unresolved* request subset all succeeded.
        "resolver_delta_projection_all_roots": all(old_flags) or delta_all_positive,
        "consumer_runtime_all_roots": all(new_flags),
    }


def run_audit(
    *,
    old_runtime_path: Path,
    new_runtime_path: Path,
    new_runtime_report_path: Path,
    resolver_dir: Path,
    resolver_protocol_path: Path,
    clean_store_dir: Path,
    legacy_entity_index_path: Path,
    output_dir: Path,
    experiment_id: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite append-only audit: {output_dir}")
    resolver_results_path = resolver_dir / "resolution_results.jsonl"
    resolver_report_path = resolver_dir / "report.json"
    title_cache_path = resolver_dir / "title_cache.jsonl"
    entity_cache_path = resolver_dir / "entity_cache.jsonl"
    aliases_path = clean_store_dir / "aliases.jsonl"
    required = (
        old_runtime_path,
        new_runtime_path,
        new_runtime_report_path,
        resolver_results_path,
        resolver_report_path,
        resolver_protocol_path,
        title_cache_path,
        entity_cache_path,
        aliases_path,
        legacy_entity_index_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    old = _runtime_index(read_jsonl(old_runtime_path), label="closure-v1")
    new = _runtime_index(read_jsonl(new_runtime_path), label="closure-v2")
    if set(old) != set(new):
        raise ValueError("closure-v1/v2 question identity join is not exact")

    resolver_results = read_jsonl(resolver_results_path)
    resolver_by_question: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    resolver_by_anchor: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in resolver_results:
        identity = (str(row.get("question_key") or ""), str(row.get("root_anchor_surface") or ""))
        if identity in resolver_by_anchor:
            raise ValueError(f"duplicate resolver question/anchor identity: {identity}")
        resolver_by_anchor[identity] = row
        resolver_by_question[identity[0]].append(row)

    resolver_report = json.loads(resolver_report_path.read_text(encoding="utf-8"))
    resolver_protocol = json.loads(resolver_protocol_path.read_text(encoding="utf-8"))
    runtime_report = json.loads(new_runtime_report_path.read_text(encoding="utf-8"))
    title_cache = _cache_map(title_cache_path)
    entity_cache = _cache_map(entity_cache_path)
    clean_aliases = _alias_map(aliases_path)
    legacy_index = json.loads(legacy_entity_index_path.read_text(encoding="utf-8"))

    source_counts: Counter[str] = Counter()
    old_resolution_counts: Counter[str] = Counter()
    question_states: Counter[tuple[str, bool, bool]] = Counter()
    detail_rows: list[dict[str, Any]] = []
    replay_matches = 0
    total_anchors = 0
    requested_old_resolved = 0
    requested_old_unresolved = 0
    lost_old_clean_unreproducible = 0
    lost_old_local_index = 0
    lost_old_legacy_cache = 0

    for key in sorted(old):
        old_row = old[key]
        new_row = new[key]
        if (
            str(old_row.get("question") or "") != str(new_row.get("question") or "")
            or str(old_row.get("question_sha256") or "")
            != str(new_row.get("question_sha256") or "")
            or (old_row.get("query_plan") or {}).get("anchors")
            != (new_row.get("query_plan") or {}).get("anchors")
        ):
            raise ValueError(f"closure-v1/v2 question or plan drift: {key}")
        old_entities = (old_row.get("execution") or {})["anchor_entities"]
        new_entities = (new_row.get("execution") or {})["anchor_entities"]
        per_question_sources: Counter[str] = Counter()
        lost_here = 0
        for surface, new_entity in new_entities.items():
            total_anchors += 1
            old_entity = old_entities[surface]
            old_is_resolved = _resolved(old_entity)
            old_resolution_counts["resolved" if old_is_resolved else "unresolved"] += 1
            request = resolver_by_anchor.get((key, surface))
            if request is not None:
                if old_is_resolved:
                    requested_old_resolved += 1
                else:
                    requested_old_unresolved += 1
            completed = str(new_entity.get("resolved_surface") or surface)
            source, expected_qid = consumer_resolution(
                surface=surface,
                completed_surface=completed,
                title_cache=title_cache,
                clean_aliases=clean_aliases,
                entity_cache=entity_cache,
            )
            source_counts[source] += 1
            per_question_sources[source] += 1
            actual_qid = str(new_entity.get("qid") or "") or None
            if expected_qid == actual_qid and _resolved(new_entity) == bool(expected_qid):
                replay_matches += 1
            if old_is_resolved and expected_qid is None:
                lost_old_clean_unreproducible += 1
                lost_here += 1
                candidates = legacy_index.get(surface.strip().lower()) or []
                candidate_qids = {str(candidate.get("qid") or "") for candidate in candidates}
                if str(old_entity.get("qid") or "") in candidate_qids:
                    lost_old_local_index += 1
                elif float(old_entity.get("score") or 0.0) == 0.85:
                    # In the old EntityLinker this exact score is emitted by
                    # _legacy_cache_lookup after the local candidate path misses.
                    lost_old_legacy_cache += 1

        projection = question_projection_state(
            old_entities=old_entities,
            new_entities=new_entities,
            resolver_rows=resolver_by_question.get(key, []),
        )
        projected = bool(projection["resolver_delta_projection_all_roots"])
        actual = bool(projection["consumer_runtime_all_roots"])
        question_states[(str(projection["old_resolution_state"]), projected, actual)] += 1
        detail_rows.append(
            {
                "schema_version": DETAIL_SCHEMA_VERSION,
                "question_key": key,
                "qid": str(new_row["qid"]),
                **projection,
                "lost_old_clean_unreproducible_roots": lost_here,
                "consumer_source_counts": dict(sorted(per_question_sources.items())),
                "contains_wikidata_qid_targets": False,
                "gold_access": False,
            }
        )

    resolver_request_set = set(resolver_by_anchor)
    initially_unresolved_set = {
        (key, surface)
        for key, row in old.items()
        for surface, entity in ((row.get("execution") or {}).get("anchor_entities") or {}).items()
        if not _resolved(entity)
    }
    projected_questions = sum(
        count for (_, projected, _), count in question_states.items() if projected
    )
    actual_questions = sum(
        count for (_, _, actual), count in question_states.items() if actual
    )
    actual_occurrences = sum(
        count
        for source, count in source_counts.items()
        if source
        in {"new_exact_title_cache", "clean_v5_exact_alias", "new_exact_entity_cache"}
    )
    reported_projected = int(
        (resolver_report.get("post_resolution_coverage") or {}).get(
            "final_all_roots_resolved_questions", -1
        )
    )
    reported_actual = int((runtime_report.get("counts") or {}).get("anchor_qid_resolved", -1))
    checks = {
        "closure_question_identity_join_exact": len(old) == len(new) == 300,
        "closure_plan_and_question_identity_unchanged": True,
        "resolver_worklist_equals_initially_unresolved_roots": resolver_request_set
        == initially_unresolved_set,
        "resolver_requested_zero_old_resolved_roots": requested_old_resolved == 0,
        "clean_consumer_replay_matches_every_anchor": replay_matches == total_anchors,
        "recomputed_projection_matches_resolver_report": projected_questions
        == reported_projected,
        "recomputed_actual_matches_closure_v2_report": actual_questions
        == reported_actual,
        "projection_gap_equals_clean_unreproducible_old_roots": projected_questions
        - actual_questions
        == lost_old_clean_unreproducible,
        "lost_old_root_source_decomposition_complete": lost_old_local_index
        + lost_old_legacy_cache
        == lost_old_clean_unreproducible,
        "gold_access_false": True,
        "network_access_false": True,
        "old_results_unchanged": True,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks})

    output_dir.mkdir(parents=True, exist_ok=False)
    detail_path = output_dir / "question_projection_states.jsonl"
    with detail_path.open("x", encoding="utf-8") as handle:
        for row in detail_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    rendered_states = {
        f"old_{old_state}__projected_{str(projected).lower()}__actual_{str(actual).lower()}": count
        for (old_state, projected, actual), count in sorted(question_states.items())
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_CAUSE_IDENTIFIED_OLD_RESULTS_UNCHANGED",
        "counts": {
            "questions": len(new),
            "root_anchor_occurrences": total_anchors,
            "closure_v1_resolved_anchor_occurrences": old_resolution_counts["resolved"],
            "closure_v1_all_roots_resolved_questions": int(
                (resolver_protocol.get("counts") or {}).get(
                    "all_roots_resolved_questions", -1
                )
            ),
            "resolver_requests": len(resolver_results),
            "resolver_positive": sum(row.get("outcome") == "positive" for row in resolver_results),
            "resolver_abstain": sum(row.get("outcome") == "abstain" for row in resolver_results),
            "resolver_projected_all_roots_resolved_questions": projected_questions,
            "clean_consumer_resolved_anchor_occurrences": actual_occurrences,
            "clean_consumer_all_roots_resolved_questions": actual_questions,
            "question_projection_gap": projected_questions - actual_questions,
            "old_resolved_roots_not_reproducible_by_clean_consumer": lost_old_clean_unreproducible,
        },
        "rates": {
            "resolver_projected_all_roots_question_rate": projected_questions / len(new),
            "clean_consumer_all_roots_question_rate": actual_questions / len(new),
            "clean_consumer_anchor_occurrence_rate": actual_occurrences / total_anchors,
        },
        "consumer_resolution_sources": dict(sorted(source_counts.items())),
        "lost_old_resolution_sources": {
            "legacy_local_entity_index_candidate": lost_old_local_index,
            "legacy_cache_lookup_after_candidate_miss": lost_old_legacy_cache,
        },
        "question_state_decomposition": rendered_states,
        "diagnosis": {
            "join_or_key_consumption_bug": False,
            "definition_bug": True,
            "cause": (
                "The resolver worklist contained only roots unresolved by closure-v1, "
                "but closure-v1 used a legacy local entity index/cache that closure-v2 "
                "intentionally removed. The resolver report added closure-v1 coverage "
                "to newly positive requests without replaying those old positives through "
                "the exact closure-v2 consumer stack."
            ),
            "closure_v2_behavior": (
                "Correctly fail-closed under exact title cache -> clean-v5 exact alias -> "
                "isolated exact entity cache -> abstain."
            ),
        },
        "successor_requirements": {
            "resolver_scope": (
                "Resolve every frozen planner root occurrence, or at minimum all roots "
                "of every question selected for clean re-execution; never inherit a "
                "resolved bit from a different resolver stack."
            ),
            "continue_gate": (
                "Compute all-roots-resolved from an offline replay using the exact final "
                "consumer caches, alias store, normalization, precedence, and ambiguity "
                "rules; do not use baseline_count + newly_resolved arithmetic."
            ),
            "postflight_gate": (
                "Keep the existing final-runtime all-root metric as authoritative and "
                "require preregistered projection == consumer dry-run == final runtime."
            ),
            "n1500_application": (
                "Use the full-root-per-failed-question worklist pattern already used by "
                "make_reresolution_rows, bind all cache/store hashes, and stop before "
                "property closure if the consumer-realizable root rate misses its gate."
            ),
            "per_question_repairs": False,
        },
        "checks": checks,
        "inputs": {
            "closure_v1_runtime": file_identity(old_runtime_path),
            "closure_v2_runtime": file_identity(new_runtime_path),
            "closure_v2_runtime_report": file_identity(new_runtime_report_path),
            "resolver_results": file_identity(resolver_results_path),
            "resolver_report": file_identity(resolver_report_path),
            "resolver_protocol": file_identity(resolver_protocol_path),
            "resolver_title_cache": file_identity(title_cache_path),
            "resolver_entity_cache": file_identity(entity_cache_path),
            "clean_v5_aliases": file_identity(aliases_path),
            "legacy_entity_index": file_identity(legacy_entity_index_path),
        },
        "outputs": {"question_projection_states": file_identity(detail_path)},
        "gold_access": False,
        "network_access": False,
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
            "phase": "audit_2wiki_n300_root_projection_gap",
            "experiment_id": experiment_id,
            "report": file_identity(report_path),
            "gold_access": False,
            "network_access": False,
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-runtime", type=Path, default=DEFAULT_OLD_RUNTIME)
    parser.add_argument("--new-runtime", type=Path, default=DEFAULT_NEW_RUNTIME)
    parser.add_argument(
        "--new-runtime-report", type=Path, default=DEFAULT_NEW_RUNTIME_REPORT
    )
    parser.add_argument("--resolver-dir", type=Path, default=DEFAULT_RESOLVER_DIR)
    parser.add_argument(
        "--resolver-protocol", type=Path, default=DEFAULT_RESOLVER_PROTOCOL
    )
    parser.add_argument("--clean-store", type=Path, default=DEFAULT_CLEAN_STORE)
    parser.add_argument(
        "--legacy-entity-index", type=Path, default=DEFAULT_LEGACY_ENTITY_INDEX
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    args = parser.parse_args()
    report = run_audit(
        old_runtime_path=args.old_runtime,
        new_runtime_path=args.new_runtime,
        new_runtime_report_path=args.new_runtime_report,
        resolver_dir=args.resolver_dir,
        resolver_protocol_path=args.resolver_protocol,
        clean_store_dir=args.clean_store,
        legacy_entity_index_path=args.legacy_entity_index,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
