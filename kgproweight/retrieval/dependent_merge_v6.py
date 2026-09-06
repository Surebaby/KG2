"""Query-variant adapter for the frozen v6 dependent-retrieval merge.

The v6 runner keeps related searches inside a logical hop as
``query_variants``.  The precision-first v5 merge already implements the
frozen Arm-A safety policy, so this module only:

* validates and flattens every query variant into an independent v5 hop;
* lets v5 inspect at most two reranked passages per variant;
* enriches v5 provenance with the logical-hop and query-variant identities;
* preserves v5's strict full-question-score comparison and fixed budget.

No question text, answer, label, supporting fact, or dataset-specific field is
read here.  ``full_question_scores`` must be computed and frozen by the caller.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import re
from typing import Any, Dict, Mapping, Sequence, Tuple

from kgproweight.retrieval.dependent_merge_v5 import (
    DependentMergeV5Error,
    merge_dependent_passages_v5,
    passage_score_key,
)


POLICY_VERSION = "dependent-merge-v6-query-variants-1"
_MAX_CANDIDATES_PER_QUERY_VARIANT = 2
_MAX_FINAL_REPLACEMENTS = 2
_WS_RE = re.compile(r"\s+")


class DependentMergeV6Error(DependentMergeV5Error):
    """The logical-hop/query-variant contract is invalid."""


def _clean(value: object) -> str:
    return _WS_RE.sub(" ", str(value or "").strip())


def _as_sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DependentMergeV6Error(f"{field} must be a sequence")
    return value


def _enrich_event(
    event: Mapping[str, Any],
    variant_metadata: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    enriched = deepcopy(dict(event))
    flat_hop_id = str(enriched.get("hop_id") or "")
    metadata = variant_metadata.get(flat_hop_id)
    if metadata is None or enriched.get("source") != "dependent_query":
        return enriched
    enriched["hop_id"] = metadata["logical_hop_id"]
    enriched["logical_hop_id"] = metadata["logical_hop_id"]
    enriched["query_variant_id"] = metadata["query_variant_id"]
    enriched["query"] = metadata["query"]
    enriched["hint"] = deepcopy(metadata["hint"])
    return enriched


def _enrich_row(
    row: Mapping[str, Any],
    variant_metadata: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    enriched = deepcopy(dict(row))
    flat_hop_id = str(enriched.get("hop_id") or "")
    metadata = variant_metadata.get(flat_hop_id)
    if metadata is not None:
        enriched["hop_id"] = metadata["logical_hop_id"]
        enriched["logical_hop_id"] = metadata["logical_hop_id"]
        enriched["query_variant_id"] = metadata["query_variant_id"]
        enriched["query"] = metadata["query"]
        enriched["hint"] = deepcopy(metadata["hint"])
    provenance = enriched.get("provenance")
    if isinstance(provenance, list):
        enriched["provenance"] = [
            _enrich_event(event, variant_metadata)
            if isinstance(event, Mapping)
            else deepcopy(event)
            for event in provenance
        ]
    return enriched


def _flatten_logical_hops(
    logical_hop_results: Sequence[Mapping[str, Any]],
) -> Tuple[list[Dict[str, Any]], Dict[str, Dict[str, Any]], int, int]:
    flattened: list[Dict[str, Any]] = []
    metadata: Dict[str, Dict[str, Any]] = {}
    logical_ids: set[str] = set()
    dependent_logical_hops = 0

    for logical_index, logical_hop in enumerate(logical_hop_results, start=1):
        if not isinstance(logical_hop, Mapping):
            raise DependentMergeV6Error(
                f"logical_hop_{logical_index} is not an object"
            )
        logical_hop_id = _clean(logical_hop.get("logical_hop_id"))
        if not logical_hop_id:
            raise DependentMergeV6Error(
                f"logical_hop_{logical_index} has no logical_hop_id"
            )
        if logical_hop_id in logical_ids:
            raise DependentMergeV6Error(
                f"duplicate logical_hop_id: {logical_hop_id}"
            )
        logical_ids.add(logical_hop_id)

        dependencies = list(
            _as_sequence(
                logical_hop.get("dependencies") or [],
                field=f"{logical_hop_id}.dependencies",
            )
        )
        is_dependent = (
            bool(logical_hop["is_dependent"])
            if "is_dependent" in logical_hop
            else bool(dependencies)
        )
        if is_dependent:
            dependent_logical_hops += 1
        dependency_depth = logical_hop.get(
            "dependency_depth", logical_hop.get("depth", logical_index)
        )
        variants = _as_sequence(
            logical_hop.get("query_variants"),
            field=f"{logical_hop_id}.query_variants",
        )
        if not variants:
            raise DependentMergeV6Error(
                f"{logical_hop_id}.query_variants must not be empty"
            )

        variant_ids: set[str] = set()
        for variant_index, variant in enumerate(variants, start=1):
            if not isinstance(variant, Mapping):
                raise DependentMergeV6Error(
                    f"{logical_hop_id}.query_variant_{variant_index} is not an object"
                )
            query_variant_id = _clean(variant.get("query_variant_id"))
            if not query_variant_id:
                raise DependentMergeV6Error(
                    f"{logical_hop_id}.query_variant_{variant_index} has no query_variant_id"
                )
            if query_variant_id in variant_ids:
                raise DependentMergeV6Error(
                    f"duplicate query_variant_id in {logical_hop_id}: {query_variant_id}"
                )
            variant_ids.add(query_variant_id)
            query = _clean(variant.get("query"))
            if not query:
                raise DependentMergeV6Error(
                    f"{logical_hop_id}.{query_variant_id} has no query"
                )
            passages = _as_sequence(
                variant.get("passages") or [],
                field=f"{logical_hop_id}.{query_variant_id}.passages",
            )

            flat_hop_id = f"__v6_logical_{logical_index}_variant_{variant_index}"
            metadata[flat_hop_id] = {
                "logical_hop_id": logical_hop_id,
                "query_variant_id": query_variant_id,
                "query": query,
                "hint": deepcopy(variant.get("hint")),
            }
            flattened.append({
                "hop_id": flat_hop_id,
                "query": query,
                "dependencies": deepcopy(dependencies),
                "is_dependent": is_dependent,
                "dependency_depth": dependency_depth,
                "passages": deepcopy(list(passages)),
            })

    return flattened, metadata, len(logical_ids), dependent_logical_hops


def merge_dependent_passages_v6(
    original_passages: Sequence[Mapping[str, Any] | str],
    logical_hop_results: Sequence[Mapping[str, Any]],
    full_question_scores: Mapping[str, float],
    protected_originals: int = 8,
    candidates_per_query_variant: int = 2,
    total_passages: int = 10,
) -> Tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Apply the v5 safety merge to independently ranked query variants.

    Every variant is presented to v5 as its own hop, so the candidate cutoff is
    applied separately rather than once to a pooled logical-hop passage list.
    The final number of replacements is capped at two by the protected-prefix
    contract.  Scores are caller-provided full-question scores; this wrapper
    performs no retrieval, model inference, or label access.
    """

    if not isinstance(logical_hop_results, Sequence) or isinstance(
        logical_hop_results, (str, bytes)
    ):
        raise DependentMergeV6Error("logical_hop_results must be a sequence")
    if (
        isinstance(candidates_per_query_variant, bool)
        or not 1 <= candidates_per_query_variant <= _MAX_CANDIDATES_PER_QUERY_VARIANT
    ):
        raise DependentMergeV6Error(
            "candidates_per_query_variant must be 1 or 2"
        )
    if (
        isinstance(protected_originals, bool)
        or isinstance(total_passages, bool)
        or total_passages <= 0
        or protected_originals < 0
        or protected_originals > total_passages
    ):
        raise DependentMergeV6Error("invalid protected/original passage budget")
    if total_passages - protected_originals > _MAX_FINAL_REPLACEMENTS:
        raise DependentMergeV6Error(
            "v6 permits at most two final passage replacements"
        )

    flattened, variant_metadata, logical_count, dependent_logical_count = (
        _flatten_logical_hops(logical_hop_results)
    )
    merged, v5_telemetry = merge_dependent_passages_v5(
        original_passages,
        flattened,
        full_question_scores,
        protected_originals=protected_originals,
        candidates_per_hop=candidates_per_query_variant,
        total_passages=total_passages,
    )

    enriched_merged: list[Dict[str, Any]] = []
    for passage in merged:
        enriched = deepcopy(passage)
        provenance = enriched.get("retrieval_provenance")
        if isinstance(provenance, list):
            enriched["retrieval_provenance"] = [
                _enrich_event(event, variant_metadata)
                if isinstance(event, Mapping)
                else deepcopy(event)
                for event in provenance
            ]
        enriched_merged.append(enriched)

    telemetry = deepcopy(v5_telemetry)
    base_policy_version = telemetry.get("policy_version")
    telemetry["policy_version"] = POLICY_VERSION
    telemetry["base_policy_version"] = base_policy_version
    telemetry["logical_hop_count"] = logical_count
    telemetry["dependent_logical_hop_count"] = dependent_logical_count
    telemetry["query_variant_count"] = len(flattened)
    telemetry["candidates_per_query_variant"] = candidates_per_query_variant
    telemetry["max_final_replacements"] = _MAX_FINAL_REPLACEMENTS

    for field in ("selected_new", "candidate_inventory", "rejected_not_strictly_better"):
        rows = telemetry.get(field)
        if isinstance(rows, list):
            telemetry[field] = [
                _enrich_row(row, variant_metadata)
                if isinstance(row, Mapping)
                else deepcopy(row)
                for row in rows
            ]
    duplicate_rows = telemetry.get("duplicate_original_observations")
    if isinstance(duplicate_rows, list):
        telemetry["duplicate_original_observations"] = [
            {
                **deepcopy(dict(row)),
                "provenance": _enrich_event(row["provenance"], variant_metadata),
            }
            if isinstance(row, Mapping) and isinstance(row.get("provenance"), Mapping)
            else deepcopy(row)
            for row in duplicate_rows
        ]

    selected_by_logical_hop: Counter[str] = Counter()
    selected_by_query_variant: Counter[tuple[str, str]] = Counter()
    for row in telemetry.get("selected_new") or []:
        variant_pairs = {
            (
                str(event.get("logical_hop_id") or ""),
                str(event.get("query_variant_id") or ""),
            )
            for event in row.get("provenance") or []
            if isinstance(event, Mapping)
            and event.get("logical_hop_id")
            and event.get("query_variant_id")
        }
        logical_ids = {logical_id for logical_id, _ in variant_pairs}
        for logical_id in logical_ids:
            selected_by_logical_hop[logical_id] += 1
        for logical_id, variant_id in variant_pairs:
            selected_by_query_variant[(logical_id, variant_id)] += 1
    telemetry.pop("selected_by_hop", None)
    telemetry["selected_by_logical_hop"] = dict(sorted(selected_by_logical_hop.items()))
    telemetry["selected_by_query_variant"] = [
        {
            "logical_hop_id": logical_id,
            "query_variant_id": variant_id,
            "selected": count,
        }
        for (logical_id, variant_id), count in sorted(selected_by_query_variant.items())
    ]

    if len(telemetry.get("selected_new") or []) > _MAX_FINAL_REPLACEMENTS:
        raise DependentMergeV6Error("v5 merge exceeded the frozen v6 replacement cap")
    return enriched_merged, telemetry


__all__ = [
    "DependentMergeV6Error",
    "POLICY_VERSION",
    "merge_dependent_passages_v6",
    "passage_score_key",
]
