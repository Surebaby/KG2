"""Precision-first passage merging for dependency-aware retrieval.

This module is deliberately independent from the v4 retrieval runner.  It
only decides whether already-retrieved *dependent-hop* passages are strong
enough to replace the unprotected tail of a frozen question-retrieval arm.
It does not inspect questions, answers, labels, or supporting facts.

The caller must provide one deterministic, comparable score per document
(for example, a full-question cross-encoder score over the union).  A new
document replaces an original document only when its score is *strictly*
higher.  Equal scores retain the original.  Root-hop passages may resolve a
bridge, but never enter the final context through this merge policy.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Dict, Mapping, Sequence, Tuple


POLICY_VERSION = "dependent-merge-v5-precision-first-1"
_WS_RE = re.compile(r"\s+")


class DependentMergeV5Error(ValueError):
    """The conservative merge contract is incomplete or inconsistent."""


def _normalise(value: object) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).split())


def passage_score_key(passage: Mapping[str, Any] | str) -> str:
    """Return the stable key expected by ``document_scores``.

    Wiki18 document ids are authoritative when present.  Synthetic or legacy
    passages without ids fall back to a hash of normalized title/body text.
    """

    if isinstance(passage, Mapping) and passage.get("id") is not None:
        return f"id:{passage['id']}"
    if isinstance(passage, Mapping):
        title = str(passage.get("title") or "")
        text = str(passage.get("contents") or passage.get("text") or "")
    else:
        title, text = "", str(passage)
    blob = f"{_normalise(title)}\n{_normalise(text)}"
    return "text:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _score(
    scores: Mapping[str, float],
    key: str,
    *,
    role: str,
) -> float:
    if key not in scores:
        raise DependentMergeV5Error(f"missing deterministic score for {role} {key}")
    value = scores[key]
    if isinstance(value, bool):
        raise DependentMergeV5Error(f"non-numeric deterministic score for {role} {key}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DependentMergeV5Error(
            f"non-numeric deterministic score for {role} {key}"
        ) from exc
    if not math.isfinite(result):
        raise DependentMergeV5Error(f"non-finite deterministic score for {role} {key}")
    return result


def _clean_query(value: object) -> str:
    return _WS_RE.sub(" ", str(value or "").strip())


def _dependency_depth(hop: Mapping[str, Any], hop_index: int) -> int:
    raw = hop.get("dependency_depth", hop.get("depth", hop_index))
    if isinstance(raw, bool):
        raise DependentMergeV5Error("dependency depth must be a positive integer")
    try:
        depth = int(raw)
    except (TypeError, ValueError) as exc:
        raise DependentMergeV5Error("dependency depth must be a positive integer") from exc
    if depth <= 0:
        raise DependentMergeV5Error("dependency depth must be a positive integer")
    return depth


def _is_dependent(hop: Mapping[str, Any]) -> bool:
    if "is_dependent" in hop:
        return bool(hop["is_dependent"])
    dependencies = hop.get("dependencies") or []
    if not isinstance(dependencies, Sequence) or isinstance(dependencies, (str, bytes)):
        raise DependentMergeV5Error("hop dependencies must be a sequence")
    return bool(dependencies)


def _event(
    hop: Mapping[str, Any],
    *,
    hop_index: int,
    depth: int,
    rerank_rank: int,
) -> Dict[str, Any]:
    hop_id = _clean_query(hop.get("hop_id")) or f"hop_{hop_index}"
    query = _clean_query(hop.get("query"))
    event: Dict[str, Any] = {
        "source": "dependent_query",
        "hop_id": hop_id,
        "dependency_depth": depth,
        "rank": rerank_rank,
    }
    if query:
        event["query"] = query
        event["query_sha256"] = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return event


def _event_identity(event: Mapping[str, Any]) -> str:
    return json.dumps(dict(event), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def merge_dependent_passages_v5(
    original_passages: Sequence[Mapping[str, Any] | str],
    hop_results: Sequence[Mapping[str, Any]],
    document_scores: Mapping[str, float],
    *,
    protected_originals: int = 8,
    candidates_per_hop: int = 2,
    total_passages: int = 10,
) -> Tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Conservatively replace at most the unprotected original tail.

    Rules are intentionally asymmetric in favor of the frozen Arm A:

    * root-hop results never enter the output;
    * when no dependent hop exists, Arm A is returned structurally unchanged;
    * only the first ``candidates_per_hop`` results of each dependent hop are
      considered (duplicates do not cause scanning farther down the list);
    * full-question score orders candidates; dependency depth only breaks ties;
    * at least ``protected_originals`` originals survive;
    * a candidate must strictly outscore the original it would replace;
      an exact tie keeps the original.

    ``document_scores`` must come from one caller-frozen scoring function so
    values are comparable across original and dependent documents.  Missing or
    non-finite scores fail closed instead of silently changing the experiment.
    """

    if total_passages <= 0:
        raise DependentMergeV5Error("total_passages must be positive")
    if protected_originals < 0 or protected_originals > total_passages:
        raise DependentMergeV5Error(
            "protected_originals must be between zero and total_passages"
        )
    if candidates_per_hop <= 0:
        raise DependentMergeV5Error("candidates_per_hop must be positive")
    if len(original_passages) != total_passages:
        raise DependentMergeV5Error(
            f"expected exactly {total_passages} frozen original passages, "
            f"got {len(original_passages)}"
        )

    original = [
        deepcopy(dict(passage)) if isinstance(passage, Mapping)
        else {"contents": str(passage)}
        for passage in original_passages
    ]
    dependent_hops: list[tuple[int, int, Mapping[str, Any]]] = []
    root_hop_count = 0
    for hop_index, hop in enumerate(hop_results, start=1):
        if not isinstance(hop, Mapping):
            raise DependentMergeV5Error(f"hop_{hop_index} is not an object")
        if _is_dependent(hop):
            dependent_hops.append((_dependency_depth(hop, hop_index), hop_index, hop))
        else:
            root_hop_count += 1

    base_telemetry: Dict[str, Any] = {
        "policy_version": POLICY_VERSION,
        "total_budget": total_passages,
        "protected_originals": protected_originals,
        "replaceable_originals": total_passages - protected_originals,
        "candidates_per_hop": candidates_per_hop,
        "root_hop_count": root_hop_count,
        "dependent_hop_count": len(dependent_hops),
    }
    if not dependent_hops:
        return original, {
            **base_telemetry,
            "changed": False,
            "fallback_exact": True,
            "fallback_reason": "no_dependent_hop",
            "candidate_occurrences_considered": 0,
            "unique_new_candidates": 0,
            "duplicates_with_original": 0,
            "duplicates_across_dependent_hops": 0,
            "rejected_not_strictly_better": [],
            "selected_new": [],
            "evicted_originals": [],
            "output_document_keys": [passage_score_key(row) for row in original],
        }

    original_keys = [passage_score_key(row) for row in original]
    original_key_set = set(original_keys)
    original_ranks_by_key: Dict[str, list[int]] = {}
    for index, key in enumerate(original_keys, start=1):
        original_ranks_by_key.setdefault(key, []).append(index)
    occurrence_count = 0
    duplicate_with_original = 0
    duplicate_original_observations: list[Dict[str, Any]] = []
    duplicate_across = 0
    candidates: Dict[str, Dict[str, Any]] = {}

    # Capture exactly top-k per dependent hop.  Do not scan rank k+1 to fill a
    # slot when a top-k result is already present in Arm A.
    for depth, hop_index, hop in dependent_hops:
        passages = hop.get("passages") or []
        if not isinstance(passages, Sequence) or isinstance(passages, (str, bytes)):
            raise DependentMergeV5Error(f"hop_{hop_index} passages must be a sequence")
        for rerank_rank, raw in enumerate(list(passages)[:candidates_per_hop], start=1):
            if not isinstance(raw, (Mapping, str)):
                raise DependentMergeV5Error(
                    f"hop_{hop_index} rank_{rerank_rank} passage has invalid type"
                )
            occurrence_count += 1
            passage = (
                deepcopy(dict(raw)) if isinstance(raw, Mapping)
                else {"contents": str(raw)}
            )
            key = passage_score_key(passage)
            event = _event(
                hop, hop_index=hop_index, depth=depth, rerank_rank=rerank_rank
            )
            if key in original_key_set:
                duplicate_with_original += 1
                duplicate_original_observations.append({
                    "document_key": key,
                    "original_ranks": list(original_ranks_by_key[key]),
                    "provenance": event,
                })
                continue
            if key in candidates:
                duplicate_across += 1
                known = {_event_identity(item) for item in candidates[key]["events"]}
                if _event_identity(event) not in known:
                    candidates[key]["events"].append(event)
                # A deeper occurrence is the authoritative priority record;
                # rank breaks ties, followed by earlier hop order.
                old_priority = candidates[key]["priority"]
                new_priority = (-depth, rerank_rank, hop_index, key)
                if new_priority < old_priority:
                    candidates[key].update({
                        "passage": passage,
                        "depth": depth,
                        "hop_index": hop_index,
                        "hop_id": event["hop_id"],
                        "rerank_rank": rerank_rank,
                        "priority": new_priority,
                    })
                continue
            candidates[key] = {
                "key": key,
                "passage": passage,
                "depth": depth,
                "hop_index": hop_index,
                "hop_id": event["hop_id"],
                "rerank_rank": rerank_rank,
                "priority": (-depth, rerank_rank, hop_index, key),
                "events": [event],
            }

    # Score validation happens only for documents that can participate in a
    # replacement.  Root-only and duplicate-with-A observations need no score.
    replaceable: list[Dict[str, Any]] = []
    for index in range(protected_originals, total_passages):
        key = original_keys[index]
        replaceable.append({
            "index": index,
            "rank": index + 1,
            "key": key,
            "score": _score(document_scores, key, role="original"),
        })
    for value in candidates.values():
        value["score"] = _score(document_scores, value["key"], role="candidate")

    ordered_candidates = sorted(
        candidates.values(),
        key=lambda row: (
            -float(row["score"]),
            -int(row["depth"]),
            int(row["rerank_rank"]),
            str(row["key"]),
            int(row["hop_index"]),
        ),
    )
    remaining_originals = list(replaceable)
    selected: list[Dict[str, Any]] = []
    rejected: list[Dict[str, Any]] = []
    evicted: list[Dict[str, Any]] = []
    max_new = total_passages - protected_originals

    for candidate in ordered_candidates:
        if len(selected) >= max_new or not remaining_originals:
            break
        weakest = min(
            remaining_originals,
            key=lambda row: (float(row["score"]), -int(row["index"])),
        )
        if float(candidate["score"]) <= float(weakest["score"]):
            rejected.append({
                "document_key": candidate["key"],
                "score": candidate["score"],
                "hop_id": candidate["hop_id"],
                "dependency_depth": candidate["depth"],
                "rerank_rank": candidate["rerank_rank"],
                "compared_original_key": weakest["key"],
                "compared_original_rank": weakest["rank"],
                "compared_original_score": weakest["score"],
                "reason": (
                    "score_tie_original_wins"
                    if float(candidate["score"]) == float(weakest["score"])
                    else "candidate_score_lower"
                ),
            })
            continue
        remaining_originals.remove(weakest)
        selected.append(candidate)
        evicted.append({
            "document_key": weakest["key"],
            "original_rank": weakest["rank"],
            "score": weakest["score"],
            "replaced_by": candidate["key"],
            "replacement_score": candidate["score"],
        })

    selected_keys = {str(row["key"]) for row in selected}
    rejected_by_key = {
        str(row["document_key"]): str(row["reason"])
        for row in rejected
    }
    candidate_inventory = [
        {
            "document_key": row["key"],
            "score": row["score"],
            "hop_id": row["hop_id"],
            "dependency_depth": row["depth"],
            "rerank_rank": row["rerank_rank"],
            "provenance": deepcopy(row["events"]),
            "decision": (
                "selected"
                if str(row["key"]) in selected_keys
                else rejected_by_key.get(str(row["key"]), "skipped_after_capacity")
            ),
        }
        for row in ordered_candidates
    ]

    if not selected:
        return original, {
            **base_telemetry,
            "changed": False,
            "fallback_exact": True,
            "fallback_reason": "no_candidate_strictly_outscored_original_tail",
            "candidate_occurrences_considered": occurrence_count,
            "unique_new_candidates": len(candidates),
            "duplicates_with_original": duplicate_with_original,
            "duplicate_original_observations": duplicate_original_observations,
            "duplicates_across_dependent_hops": duplicate_across,
            "candidate_inventory": candidate_inventory,
            "rejected_not_strictly_better": rejected,
            "selected_new": [],
            "evicted_originals": [],
            "output_document_keys": original_keys,
        }

    evicted_indices = {int(row["original_rank"]) - 1 for row in evicted}
    output = [row for index, row in enumerate(original) if index not in evicted_indices]
    selected_rows: list[Dict[str, Any]] = []
    selected_by_hop: Counter[str] = Counter()
    for candidate in selected:
        passage = deepcopy(candidate["passage"])
        existing = passage.get("retrieval_provenance") or []
        if not isinstance(existing, list):
            raise DependentMergeV5Error("candidate retrieval_provenance must be a list")
        provenance = [deepcopy(dict(item)) for item in existing if isinstance(item, Mapping)]
        known = {_event_identity(item) for item in provenance}
        for event in candidate["events"]:
            if _event_identity(event) not in known:
                provenance.append(deepcopy(event))
                known.add(_event_identity(event))
        passage["retrieval_provenance"] = provenance
        output.append(passage)
        selected_by_hop[str(candidate["hop_id"])] += 1
        selected_rows.append({
            "document_key": candidate["key"],
            "score": candidate["score"],
            "hop_id": candidate["hop_id"],
            "dependency_depth": candidate["depth"],
            "rerank_rank": candidate["rerank_rank"],
            "provenance": deepcopy(provenance),
        })

    output_keys = [passage_score_key(row) for row in output]
    if len(output) != total_passages or len(set(output_keys)) != len(output_keys):
        raise DependentMergeV5Error("merge violated fixed-budget deduplication contract")

    return output, {
        **base_telemetry,
        "changed": True,
        "fallback_exact": False,
        "fallback_reason": None,
        "candidate_occurrences_considered": occurrence_count,
        "unique_new_candidates": len(candidates),
        "duplicates_with_original": duplicate_with_original,
        "duplicate_original_observations": duplicate_original_observations,
        "duplicates_across_dependent_hops": duplicate_across,
        "candidate_inventory": candidate_inventory,
        "rejected_not_strictly_better": rejected,
        "selected_new": selected_rows,
        "selected_by_hop": dict(sorted(selected_by_hop.items())),
        "evicted_originals": evicted,
        "output_document_keys": output_keys,
    }


__all__ = [
    "DependentMergeV5Error",
    "POLICY_VERSION",
    "merge_dependent_passages_v5",
    "passage_score_key",
]
