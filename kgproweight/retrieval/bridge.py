"""Deterministic, gold-free bridge-query helpers for retrieval audits."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Dict, List, Sequence

from kgproweight.data.entity_filter import clean_entities
from kgproweight.data.parsers import ENTITY_RE
from kgproweight.kg.entity_linker import passage_title


# Bridge-v2 is an abstention-only diagnostic: it may remove a weak query chosen
# by v1, but it must never add or reorder queries.  These categories are fixed
# before the fresh-seed audit and intentionally apply only to a BARE singleton;
# e.g. ``English`` is weak, while ``English Channel`` remains eligible.
_V2_WEAK_SINGLETONS: frozenset[str] = frozenset(
    """
    a an the this that these those
    he him his she her hers they them their theirs it its we us our ours
    you your yours i me my mine
    person people man men woman women
    monday tuesday wednesday thursday friday saturday sunday
    january february march april may june july august september october november december
    american australian british canadian chinese danish dutch english finnish french
    german greek indian iranian irish italian japanese korean malaysian norwegian
    pakistani polish portuguese russian scottish spanish swedish turkish welsh
    """.split()
)


def _normalise(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def _doc_title(doc: Dict[str, Any]) -> str:
    return passage_title(doc)


def extract_bridge_queries(
    question: str,
    ranked_docs: Sequence[Dict[str, Any]],
    *,
    max_docs: int = 5,
    max_bridges: int = 2,
    max_body_chars: int = 1200,
) -> List[str]:
    """Pick bridge entities from first-round evidence without dataset labels.

    Candidates are title-cased mentions from the top documents.  A mention gets
    two votes when it is a document title and one vote per document body in
    which it occurs.  Mentions already present in the original question are
    excluded.  Ties follow first occurrence, then normalised text, making the
    result independent of hash/random state.
    """
    question_key = _normalise(question)
    scores: Dict[str, int] = {}
    surfaces: Dict[str, str] = {}
    first_seen: Dict[str, int] = {}
    occurrence = 0

    def add(surface: str, weight: int) -> None:
        nonlocal occurrence
        surface = str(surface or "").strip()
        key = _normalise(surface)
        if not key or len(key) < 3 or len(surface) > 100:
            return
        if re.search(rf"\b{re.escape(key)}\b", question_key):
            return
        occurrence += 1
        scores[key] = scores.get(key, 0) + weight
        surfaces.setdefault(key, surface)
        first_seen.setdefault(key, occurrence)

    for doc in list(ranked_docs)[:max_docs]:
        title = _doc_title(doc)
        if title:
            add(title, 2)
        contents = str(doc.get("contents") or doc.get("text") or "")
        lines = contents.splitlines()
        body = "\n".join(lines[1:]) if title and lines else contents
        mentions = clean_entities(list(dict.fromkeys(ENTITY_RE.findall(body[:max_body_chars]))))
        # A body mention votes at most once per document.
        for mention in mentions:
            add(mention, 1)

    ordered = sorted(scores, key=lambda key: (-scores[key], first_seen[key], key))
    return [surfaces[key] for key in ordered[:max_bridges]]


def bridge_v2_rejection_reason(query: str) -> str | None:
    """Return a gold-free weak-query reason, or ``None`` when v2 keeps it.

    The v2 audit changes only abstention.  It removes bare generic singleton
    mentions and obvious duplicated extraction fragments while preserving
    potentially useful one-token named entities such as ``Mozart``.
    """
    tokens = _normalise(query).split()
    if not tokens:
        return "empty"
    if len(tokens) == 1 and tokens[0] in _V2_WEAK_SINGLETONS:
        return "weak_singleton"

    # Catch exact adjacent repetition ("Tetrisphere Tetrisphere" and
    # "Tamra Davis Tamra Davis") without rejecting ordinary repeated words in
    # longer legitimate titles by default.
    for width in range(1, len(tokens) // 2 + 1):
        for start in range(0, len(tokens) - 2 * width + 1):
            if tokens[start : start + width] == tokens[start + width : start + 2 * width]:
                return "repeated_fragment"

    # Extraction can also interleave a repeated name with one extra token,
    # e.g. "Kevin Tapani Kevin Ray Tapani".  Require at least two duplicate
    # tokens in a phrase of length >=5 so a single repeated preposition does
    # not trigger abstention.
    duplicate_count = sum(count - 1 for count in Counter(tokens).values() if count > 1)
    if len(tokens) >= 5 and duplicate_count >= 2:
        return "repeated_fragment"
    return None


def filter_bridge_queries_v2(queries: Sequence[str]) -> List[str]:
    """Keep the v1 order while abstaining from deterministic weak queries."""
    return [query for query in queries if bridge_v2_rejection_reason(query) is None]


def _document_key(doc: Dict[str, Any]) -> str:
    raw_id = doc.get("id")
    if raw_id is not None:
        return str(raw_id)
    blob = str(doc.get("contents") or doc.get("text") or "")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def reciprocal_rank_fuse(
    result_lists: Sequence[Sequence[Dict[str, Any]]],
    *,
    topk: int = 100,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """Equal-weight RRF across original and bridge-query ranked lists."""
    docs: Dict[str, Dict[str, Any]] = {}
    scores: Dict[str, float] = {}
    sources: Dict[str, int] = {}
    for results in result_lists:
        seen_in_source = set()
        for rank, doc in enumerate(results, 1):
            key = _document_key(doc)
            if key in seen_in_source:
                continue
            seen_in_source.add(key)
            docs.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            sources[key] = sources.get(key, 0) + 1
    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:topk]
    fused = []
    for key in ordered:
        doc = dict(docs[key])
        doc["bridge_rrf_score"] = scores[key]
        doc["bridge_query_sources"] = sources[key]
        fused.append(doc)
    return fused


def additive_bridge_candidates(
    original_candidates: Sequence[Dict[str, Any]],
    bridge_result_lists: Sequence[Sequence[Dict[str, Any]]],
    *,
    max_bridge_only: int = 50,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """Preserve every original candidate, then append bridge-only documents.

    Bridge result lists are ranked among themselves with equal-weight RRF. Any
    document already present in the original candidate pool is removed before
    the top ``max_bridge_only`` additions are appended.  The function therefore
    guarantees that bridge retrieval cannot lower candidate-level recall.
    """
    if max_bridge_only < 0:
        raise ValueError("max_bridge_only must be non-negative")
    preserved = [dict(doc) for doc in original_candidates]
    if max_bridge_only == 0 or not bridge_result_lists:
        return preserved
    original_keys = {_document_key(doc) for doc in original_candidates}
    max_fused = sum(len(results) for results in bridge_result_lists)
    bridge_ranked = reciprocal_rank_fuse(
        bridge_result_lists,
        topk=max(1, max_fused),
        rrf_k=rrf_k,
    )
    additions = [doc for doc in bridge_ranked if _document_key(doc) not in original_keys]
    return preserved + additions[:max_bridge_only]
