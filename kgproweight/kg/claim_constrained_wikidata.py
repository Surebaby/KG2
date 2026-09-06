"""Precision-first selection of passage-supported historical Wikidata claims.

The helpers in this module are deliberately independent of datasets and Gold
labels.  A caller may scan all claims on one *exactly resolved* Wikidata item,
but an edge is retained only when its tail is visible in the already-frozen
retrieval context and its property agrees with the planned relation (or is the
unique non-metadata Wikidata connection to a passage-title entity).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping, Sequence


SELECTOR_VERSION = "claim-constrained-wikidata-selector-1"

_STOP = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
    "is", "of", "on", "or", "the", "to", "was", "were", "what", "which",
    "who", "with",
}
_META_PIDS = {
    "P18", "P31", "P279", "P373", "P910", "P1343", "P1963",
}
_TOKEN_CANON = {
    "acted": "cast", "actor": "cast", "actors": "cast", "cast": "cast",
    "played": "cast", "plays": "cast", "starring": "cast",
    "authored": "author", "writer": "author", "writers": "author",
    "cowriter": "author", "screenwriter": "author",
    "directed": "director", "directors": "director",
    "founded": "founder", "founds": "founder",
    "located": "location", "place": "location",
    "members": "member", "membership": "member",
    "owned": "owner", "ownership": "owner",
    "performed": "performer", "performs": "performer",
    "published": "publisher", "publishes": "publisher",
    "teams": "team", "clubs": "team",
}


def normalise_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _tokens(value: object) -> set[str]:
    values = set()
    for token in normalise_text(value).split():
        if token in _STOP:
            continue
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]
        values.add(_TOKEN_CANON.get(token, token))
    return values


def relation_similarity(planned_relation: object, property_label: object) -> float:
    """Token F1 after a small, global morphology/synonym normalisation."""
    left, right = _tokens(planned_relation), _tokens(property_label)
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if not overlap:
        return 0.0
    precision = overlap / len(right)
    recall = overlap / len(left)
    return 2.0 * precision * recall / (precision + recall)


def tail_support(
    edge: Mapping[str, Any],
    *,
    passage_title_qids: set[str],
    passage_blob: str,
) -> str | None:
    """Return the visible support channel for an edge tail, else ``None``."""
    tail_qid = str(edge.get("tail_qid") or "")
    if tail_qid and tail_qid in passage_title_qids:
        return "passage_title_qid"

    blob = normalise_text(passage_blob)
    tail = normalise_text(edge.get("tail_value"))
    # Avoid treating very short/generic values as evidence.
    if tail and (len(tail) >= 4 or tail.isdigit()) and re.search(
        rf"(?<![a-z0-9]){re.escape(tail)}(?![a-z0-9])", blob
    ):
        return "passage_text_exact"

    raw = normalise_text(edge.get("tail_raw_value"))
    years = [token for token in raw.split() if len(token) == 4 and token.isdigit()]
    if years and all(re.search(rf"(?<!\d){re.escape(year)}(?!\d)", blob) for year in years):
        return "passage_text_year"
    return None


def select_claim_edges(
    edges: Sequence[Mapping[str, Any]],
    *,
    planned_pid: str | None,
    planned_relation: str,
    property_labels: Mapping[str, str],
    passage_title_qids: set[str],
    passage_blob: str,
    max_edges: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a small, deterministic set of supported standard KG edges.

    Returns ``(selected, rejected)`` with explicit scores/reasons.  The
    unique-connection fallback is allowed only for entity tails that are exact
    passage-title QIDs; free-text coincidence alone can never override a
    relation mismatch.
    """
    supported: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for raw in edges:
        edge = dict(raw)
        pid = str(edge.get("pid") or "")
        if not re.fullmatch(r"P[1-9][0-9]*", pid) or pid in _META_PIDS:
            rejected.append({"edge": edge, "reason": "metadata_or_invalid_pid"})
            continue
        support = tail_support(
            edge, passage_title_qids=passage_title_qids, passage_blob=passage_blob
        )
        if support is None:
            rejected.append({"edge": edge, "reason": "tail_not_supported_by_frozen_passages"})
            continue
        label = str(property_labels.get(pid) or edge.get("relation") or pid)
        similarity = relation_similarity(planned_relation, label)
        supported.append({
            **edge,
            "relation": label,
            "tail_support": support,
            "relation_similarity": similarity,
            "planned_pid_match": bool(planned_pid and pid == planned_pid),
        })

    title_supported_pids = {
        row["pid"] for row in supported
        if row["tail_support"] == "passage_title_qid"
    }
    unique_title_pid = next(iter(title_supported_pids)) if len(title_supported_pids) == 1 else None

    eligible: list[dict[str, Any]] = []
    for edge in supported:
        exact = edge["planned_pid_match"]
        semantic = float(edge["relation_similarity"]) >= 0.34
        unique_connection = (
            edge["tail_support"] == "passage_title_qid"
            and edge["pid"] == unique_title_pid
        )
        if not (exact or semantic or unique_connection):
            rejected.append({"edge": edge, "reason": "property_not_compatible_with_plan"})
            continue
        score = (
            (4.0 if exact else 0.0)
            + 2.0 * float(edge["relation_similarity"])
            + (2.0 if edge["tail_support"] == "passage_title_qid" else 1.0)
            + (0.5 if unique_connection else 0.0)
        )
        reason = "planned_pid" if exact else "semantic_property" if semantic else "unique_passage_claim"
        eligible.append({**edge, "selection_score": score, "selection_reason": reason})

    eligible.sort(key=lambda row: (
        -float(row["selection_score"]), str(row["pid"]),
        str(row.get("tail_qid") or ""), str(row.get("tail_value") or ""),
    ))
    selected: list[dict[str, Any]] = []
    seen = set()
    for edge in eligible:
        key = (edge.get("head_qid"), edge.get("pid"), edge.get("tail_qid"), edge.get("tail_value"))
        if key in seen:
            continue
        seen.add(key)
        selected.append(edge)
        if len(selected) >= max_edges:
            break
    return selected, rejected

