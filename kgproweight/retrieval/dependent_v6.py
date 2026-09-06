"""Gold-free query-hint helpers for dependent-retrieval v6.

This module is an append-only development candidate.  It does not change the
v4 query renderer or the v5 bridge selector.  Instead, it reuses the complete
v5 selector trace while changing the semantic status of an extracted surface:
the surface is only a bounded *query hint*, never an asserted fact or a
prompt-visible evidence edge.

Dependent queries retain the original question as an exact byte prefix.  A
hint may specialize the frozen relation/subquery after the newline, but may
not replace or rewrite the question.  When no hint is available, one query is
still emitted from the original question and the frozen clause with unresolved
dependency placeholders removed.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import re
from typing import Any, Dict, Mapping, Sequence, Tuple

from kgproweight.retrieval.dependent import (
    DependentRetrievalError,
    dependency_refs,
    instantiate_dependent_queries,
)
from kgproweight.retrieval.dependent_v5 import select_bridge_candidates_v5


QUERY_HINT_POLICY_VERSION = "dependent-query-hints-v6-development-1"
QUERY_RENDERER_VERSION = "question-anchored-dependent-query-v6-development-1"

MAX_QUERY_HINTS = 2
MAX_QUERY_VARIANTS = 2

# These are the only v5 rejection reasons which prevent a surface from being
# used as a v6 query hint.  All semantic-confidence concerns remain visible in
# telemetry, but the final full-question CE passage gate—not this helper—owns
# document admission.
HARD_REJECTION_REASONS = frozenset(
    {
        "weak_singleton",
        "repeated_fragment",
        "strict_subject_echo",
        "explicit_subject_alias",
    }
)
SOFT_RISK_REASONS = frozenset(
    {
        "insufficient_gold_free_support",
        "producer_consumer_type_conflict",
        "high_confidence_type_conflict",
        "original_question_phrase",
    }
)

_SPACE_RE = re.compile(r"\s+")
_FALLBACK_SENTINEL = "KGPWDEPENDENCYPLACEHOLDER"


def _clean_clause(value: object) -> str:
    return _SPACE_RE.sub(" ", str(value or "").strip())


def _surface_key(value: object) -> str:
    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).split()
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _as_reason_set(decision: Mapping[str, Any]) -> set[str]:
    reasons = decision.get("reasons") or []
    if not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes)):
        raise DependentRetrievalError("v5 candidate decision reasons must be a sequence")
    return {str(value) for value in reasons}


def _hint_from_accepted(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    hint = deepcopy(dict(candidate))
    v5_admission = hint.pop("admission", None)
    hint["normalized_surface"] = str(
        hint.get("normalized_surface") or _surface_key(hint.get("surface"))
    )
    hint["semantic_role"] = "retrieval_query_hint"
    hint["admission"] = {
        "policy_version": QUERY_HINT_POLICY_VERSION,
        "semantic_role": "retrieval_query_hint_not_asserted_fact",
        "source": "v5_accepted",
        "v5_admission_telemetry": deepcopy(v5_admission),
        "hard_rejection_reasons": [],
        "soft_risk_flags": [],
    }
    return hint


def _hint_from_decision(decision: Mapping[str, Any]) -> Dict[str, Any]:
    reasons = _as_reason_set(decision)
    return {
        "surface": str(decision.get("surface") or "").strip(),
        "normalized_surface": str(
            decision.get("normalized_surface")
            or _surface_key(decision.get("surface"))
        ),
        "score": int(decision.get("base_score") or 0),
        "provenance": deepcopy(list(decision.get("base_provenance") or [])),
        "semantic_role": "retrieval_query_hint",
        "admission": {
            "policy_version": QUERY_HINT_POLICY_VERSION,
            "semantic_role": "retrieval_query_hint_not_asserted_fact",
            "source": "raw_rank_fill",
            "raw_rank": int(decision.get("raw_rank") or 0),
            "v5_decision": str(decision.get("decision") or ""),
            "v5_admission_basis": str(decision.get("admission_basis") or ""),
            "hard_rejection_reasons": sorted(reasons & HARD_REJECTION_REASONS),
            "soft_risk_flags": sorted(reasons & SOFT_RISK_REASONS),
            "other_v5_risk_flags": sorted(
                reasons - HARD_REJECTION_REASONS - SOFT_RISK_REASONS
            ),
        },
    }


def select_bridge_query_hints_v6(
    *,
    step: Mapping[str, Any],
    consumers: Sequence[Mapping[str, Any]],
    target_type: str,
    query: str,
    question: str,
    passages: Sequence[Mapping[str, Any]],
    max_hints: int = MAX_QUERY_HINTS,
    max_docs: int = 10,
    max_body_chars: int = 1200,
) -> Tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Select at most two retrieval-only hints using the frozen v5 trace.

    V5-accepted candidates retain their frozen v5 order and are considered
    first.  Empty positions are then filled from all remaining v5 candidate
    decisions in ``raw_rank`` order.  Only the four reasons in
    :data:`HARD_REJECTION_REASONS` block a raw-rank fill; type/profile/support
    concerns remain soft telemetry because a hint cannot itself enter the
    final context.
    """

    if not 1 <= max_hints <= MAX_QUERY_HINTS:
        raise DependentRetrievalError(
            f"max_hints must be between 1 and {MAX_QUERY_HINTS}"
        )

    accepted, v5_telemetry = select_bridge_candidates_v5(
        step=step,
        consumers=consumers,
        target_type=target_type,
        query=query,
        question=question,
        passages=passages,
        max_candidates=max_hints,
        max_docs=max_docs,
        max_body_chars=max_body_chars,
    )

    hints: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in accepted:
        hint = _hint_from_accepted(candidate)
        surface = str(hint.get("surface") or "").strip()
        key = _surface_key(surface)
        if not surface or not key or key in seen:
            continue
        hints.append(hint)
        seen.add(key)
        if len(hints) >= max_hints:
            break

    raw_decisions = v5_telemetry.get("candidate_decisions") or []
    if not isinstance(raw_decisions, Sequence) or isinstance(
        raw_decisions, (str, bytes)
    ):
        raise DependentRetrievalError("v5 candidate_decisions must be a sequence")
    ordered_decisions = sorted(
        (dict(value) for value in raw_decisions if isinstance(value, Mapping)),
        key=lambda value: (
            int(value.get("raw_rank") or 0),
            str(value.get("normalized_surface") or ""),
        ),
    )
    hard_rejected: list[Dict[str, Any]] = []
    for decision in ordered_decisions:
        if len(hints) >= max_hints:
            break
        surface = str(decision.get("surface") or "").strip()
        key = _surface_key(surface)
        if not surface or not key or key in seen:
            continue
        hard_reasons = sorted(_as_reason_set(decision) & HARD_REJECTION_REASONS)
        if hard_reasons:
            hard_rejected.append(
                {
                    "raw_rank": int(decision.get("raw_rank") or 0),
                    "surface": surface,
                    "reasons": hard_reasons,
                }
            )
            continue
        hint = _hint_from_decision(decision)
        hints.append(hint)
        seen.add(key)

    telemetry: Dict[str, Any] = {
        "policy_version": QUERY_HINT_POLICY_VERSION,
        "gold_access": False,
        "semantic_role": "retrieval_query_hint_not_asserted_fact",
        "question_sha256": _sha256_text(question),
        "max_hints": max_hints,
        "hint_count": len(hints),
        "hint_surfaces": [str(value["surface"]) for value in hints],
        "v5_accepted_first": True,
        "raw_rank_fill_enabled": True,
        "hard_rejection_reasons": sorted(HARD_REJECTION_REASONS),
        "soft_risk_reasons": sorted(SOFT_RISK_REASONS),
        "hard_rejected_candidates": hard_rejected,
        "hints": [deepcopy(value["admission"]) for value in hints],
        "fallback_recommended": not hints,
        "fallback_reason": "no_usable_query_hint" if not hints else None,
        "v5_selector_telemetry": deepcopy(dict(v5_telemetry)),
    }
    return hints, telemetry


def _dependent_source(step: Mapping[str, Any], target_type: str) -> str:
    if target_type == "relation_graph":
        return str(step.get("subject") or "")
    if target_type == "subquery_graph":
        return str(step.get("subquery_template") or "")
    raise DependentRetrievalError(f"unsupported target_type={target_type!r}")


def _fallback_clause(step: Mapping[str, Any], target_type: str) -> str:
    """Render the frozen dependent clause with every placeholder removed."""

    source = _dependent_source(step, target_type)
    refs = dependency_refs(source)
    if not refs:
        raise DependentRetrievalError("dependent query contains no dependency reference")
    if _FALLBACK_SENTINEL.casefold() in source.casefold():
        raise DependentRetrievalError("dependent query collides with fallback sentinel")
    sentinel_values = {ref: [_FALLBACK_SENTINEL] for ref in refs}
    rendered = instantiate_dependent_queries(
        step,
        target_type,
        sentinel_values,
        max_variants=1,
    )[0]
    clause = _clean_clause(rendered.replace(_FALLBACK_SENTINEL, " "))
    if not clause or dependency_refs(clause):
        raise DependentRetrievalError(
            "no safe frozen clause remains after removing unresolved placeholder"
        )
    return clause


def render_question_anchored_queries_v6(
    *,
    question: str,
    step: Mapping[str, Any],
    target_type: str,
    slot_values: Mapping[str, Sequence[str] | str],
    max_variants: int = MAX_QUERY_VARIANTS,
) -> Tuple[list[str], Dict[str, Any]]:
    """Render one branch per hint while preserving the full question bytes.

    The exact query template is ``{question}\n{instantiated frozen clause}``.
    At most two stable, deduplicated branches are returned.  If no usable slot
    value is available, a single fallback query removes unresolved dependency
    placeholders from the frozen relation/subquery clause instead of guessing
    a bridge.
    """

    if not isinstance(question, str) or not question.strip():
        raise DependentRetrievalError("original question must be a non-empty string")
    if not 1 <= max_variants <= MAX_QUERY_VARIANTS:
        raise DependentRetrievalError(
            f"max_variants must be between 1 and {MAX_QUERY_VARIANTS}"
        )

    clauses: list[str]
    mode = "hint_branches"
    try:
        clauses = instantiate_dependent_queries(
            step,
            target_type,
            slot_values,
            max_variants=max_variants,
        )
    except DependentRetrievalError as exc:
        # A missing/empty hint is the one intentional recovery path.  Invalid
        # schema, target type, or frozen relation text must still fail closed.
        message = str(exc)
        if not (
            message.startswith("unresolved dependencies:")
            or " has no usable values" in message
        ):
            raise
        clauses = [_fallback_clause(step, target_type)]
        mode = "no_hint_fallback"

    queries: list[str] = []
    for raw_clause in clauses:
        clause = _clean_clause(raw_clause)
        if not clause or dependency_refs(clause):
            raise DependentRetrievalError(
                "rendered dependent clause is empty or contains a placeholder"
            )
        value = f"{question}\n{clause}"
        if not value.startswith(question):
            raise DependentRetrievalError("dependent query lost the original question prefix")
        if value not in queries:
            queries.append(value)
        if len(queries) >= max_variants:
            break
    if not queries:
        queries = [f"{question}\n{_fallback_clause(step, target_type)}"]
        mode = "no_hint_fallback"
    query_telemetry = [
        {
            "query": value,
            "query_sha256": _sha256_text(value),
            "question_prefix_exact": value[: len(question)] == question,
        }
        for value in queries
    ]
    telemetry: Dict[str, Any] = {
        "renderer_version": QUERY_RENDERER_VERSION,
        "gold_access": False,
        "mode": mode,
        "question_sha256": _sha256_text(question),
        "max_variants": max_variants,
        "query_count": len(queries),
        "queries": query_telemetry,
        "all_question_prefix_exact": all(
            bool(value["question_prefix_exact"]) for value in query_telemetry
        ),
    }
    return queries, telemetry


__all__ = [
    "HARD_REJECTION_REASONS",
    "MAX_QUERY_HINTS",
    "MAX_QUERY_VARIANTS",
    "QUERY_HINT_POLICY_VERSION",
    "QUERY_RENDERER_VERSION",
    "SOFT_RISK_REASONS",
    "render_question_anchored_queries_v6",
    "select_bridge_query_hints_v6",
]
