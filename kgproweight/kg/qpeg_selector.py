"""Train-only feature extraction and fail-closed selection for QPEG-v2."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


QPEG_SELECTOR_FEATURE_VERSION = "qpeg-edge-features-v1"
QPEG_SELECTOR_EXTRACTOR_VERSION = "qpeg-v2-trainonly-selector-v1"
_WORD_RE = re.compile(r"[a-z0-9]+")
_YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_COMPARISON_WORDS = {"both", "earlier", "first", "later", "older", "same", "younger"}


def _tokens(value: object) -> set[str]:
    return set(_WORD_RE.findall(str(value or "").casefold()))


def _coverage(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(right))


def edge_features(*, dataset: str, question: str, edge: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic feature dictionary without answer/Gold fields."""
    q_tokens = _tokens(question)
    head_tokens = _tokens(edge.get("head_surface"))
    relation_tokens = _tokens(edge.get("relation_surface"))
    tail_tokens = _tokens(edge.get("tail_surface"))
    rule = str(edge.get("extraction_rule") or "unknown")
    return {
        "dataset": str(dataset).strip().lower(),
        "rule": rule,
        "relation": str(edge.get("relation_surface") or "").casefold(),
        "passage_rank": float(edge.get("passage_rank", 0)),
        "sentence_index": float(edge.get("sentence_index", 0)),
        "extractor_relevance": float(edge.get("relevance_score", 0.0)),
        "question_head_coverage": _coverage(q_tokens, head_tokens),
        "question_relation_coverage": _coverage(q_tokens, relation_tokens),
        "question_tail_coverage": _coverage(q_tokens, tail_tokens),
        "head_question_coverage": _coverage(head_tokens, q_tokens),
        "tail_question_coverage": _coverage(tail_tokens, q_tokens),
        "head_token_count": float(len(head_tokens)),
        "tail_token_count": float(len(tail_tokens)),
        "is_cross_passage": float(rule == "cross_passage_title_mention"),
        "is_copula": float(rule == "first_sentence_copula"),
        "is_surface_relation": float(rule.startswith("surface_pattern:")),
        "tail_has_year": float(bool(_YEAR_RE.search(str(edge.get("tail_surface") or "")))),
        "tail_has_number": float(bool(_NUMBER_RE.search(str(edge.get("tail_surface") or "")))),
        "question_is_comparison": float(bool(q_tokens & _COMPARISON_WORDS)),
    }


def select_edges(
    *,
    record: Mapping[str, Any],
    vectorizer: Any,
    classifier: Any,
    threshold: float,
    max_edges: int = 6,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Select high-confidence edges; return empty when none pass (fail closed)."""
    if not 1 <= max_edges <= 12:
        raise ValueError("max_edges must be in [1, 12]")
    edges = [dict(edge) for edge in record.get("edges") or []]
    if not edges:
        return [], []
    features = [
        edge_features(dataset=str(record["dataset"]), question=str(record["question"]), edge=edge)
        for edge in edges
    ]
    probabilities = classifier.predict_proba(vectorizer.transform(features))[:, 1]
    ranked = sorted(
        zip(edges, (float(value) for value in probabilities)),
        key=lambda value: (
            -value[1],
            int(value[0]["passage_rank"]),
            int(value[0]["sentence_index"]),
            str(value[0]["head_surface"]).casefold(),
            str(value[0]["relation_surface"]).casefold(),
            str(value[0]["tail_surface"]).casefold(),
        ),
    )
    selected = [(edge, score) for edge, score in ranked if score >= threshold][:max_edges]
    return [edge for edge, _ in selected], [score for _, score in selected]
