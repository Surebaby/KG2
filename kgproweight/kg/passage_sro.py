"""Validation helpers for model-extracted passage SRO evidence.

The extractor is allowed to propose a canonical relation, but every accepted
edge must retain an exact passage quote plus exact surface triggers.  This
keeps passage-derived evidence distinct from Wikidata claims while preventing
free-form model text from silently becoming a factual graph.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Mapping, Sequence


PASSAGE_SRO_SCHEMA_VERSION = "passage-sro-edge-1"
PASSAGE_SRO_VALIDATOR_VERSION = "passage-sro-validator-1"
MAX_EDGES = 4

# Frozen before generation.  These predicates cover the recurrent relation
# families in multi-hop QA without permitting arbitrary model-invented labels.
ALLOWED_RELATIONS = frozenset({
    "author", "award received", "based in", "birth date", "birth place",
    "capital of", "cast member", "child", "citizenship", "country",
    "creator", "death date", "death place", "director", "educated at",
    "employer", "founded by", "genre", "headquarters", "inception",
    "language", "located in", "member of", "named after", "occupation",
    "owned by", "parent", "part of", "performer", "position held",
    "producer", "publication date", "release date", "screenwriter",
    "sports team", "spouse",
})

_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _clean(value: object) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", str(value or ""))).strip()


def _norm(value: object) -> str:
    return " ".join(_WORD_RE.findall(_clean(value).casefold()))


def _passage_title(passage: Mapping[str, Any]) -> str:
    title = _clean(passage.get("title"))
    if title:
        return title.strip('"')
    contents = str(passage.get("contents") or passage.get("text") or "")
    return _clean(contents.splitlines()[0] if contents else "").strip('"')


def _passage_text(passage: Mapping[str, Any]) -> str:
    return _clean(passage.get("contents") or passage.get("text") or "")


def parse_extraction_json(generation: str) -> list[dict[str, Any]]:
    """Parse a JSON object/array without accepting prose as evidence."""
    value = str(generation or "").strip()
    if value.startswith("```json"):
        value = value[7:]
    elif value.startswith("```"):
        value = value[3:]
    if value.endswith("```"):
        value = value[:-3]
    value = value.strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        # A narrowly bounded recovery for models that surround one JSON value
        # with a short preamble.  We never parse individual prose fragments.
        start_obj, start_arr = value.find("{"), value.find("[")
        starts = [index for index in (start_obj, start_arr) if index >= 0]
        if not starts:
            raise
        start = min(starts)
        end = max(value.rfind("}"), value.rfind("]"))
        if end < start:
            raise
        parsed = json.loads(value[start : end + 1])
    if isinstance(parsed, Mapping):
        parsed = parsed.get("edges", [])
    if not isinstance(parsed, list):
        raise ValueError("model output must be a JSON list or an object with an edges list")
    if not all(isinstance(edge, Mapping) for edge in parsed):
        raise ValueError("every extracted edge must be a JSON object")
    return [dict(edge) for edge in parsed]


def validate_extracted_edges(
    raw_edges: Sequence[Mapping[str, Any]],
    passages: Sequence[Mapping[str, Any]],
    *,
    max_edges: int = MAX_EDGES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fail closed and return accepted edges plus rejection diagnostics."""
    if not 1 <= max_edges <= MAX_EDGES:
        raise ValueError(f"max_edges must be in [1, {MAX_EDGES}]")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for source_index, raw in enumerate(raw_edges):
        reason = ""
        head = _clean(raw.get("head"))
        relation = _clean(raw.get("relation")).casefold()
        tail = _clean(raw.get("tail"))
        trigger = _clean(raw.get("relation_trigger"))
        quote = _clean(raw.get("evidence_quote"))
        try:
            passage_rank = int(raw.get("passage_rank"))
        except (TypeError, ValueError):
            passage_rank = -1

        if passage_rank < 1 or passage_rank > len(passages):
            reason = "invalid_passage_rank"
        elif relation not in ALLOWED_RELATIONS:
            reason = "relation_not_in_frozen_vocabulary"
        elif not all((head, tail, trigger, quote)):
            reason = "missing_required_surface"
        elif len(_WORD_RE.findall(tail)) > 20 or len(tail) > 180:
            reason = "tail_too_long"
        else:
            passage = passages[passage_rank - 1]
            passage_text = _passage_text(passage)
            title = _passage_title(passage)
            quote_norm = _norm(quote)
            if not quote_norm or quote_norm not in _norm(passage_text):
                reason = "quote_not_in_passage"
            elif _norm(tail) not in quote_norm:
                reason = "tail_not_in_quote"
            elif _norm(trigger) not in quote_norm:
                reason = "relation_trigger_not_in_quote"
            elif _norm(head) not in quote_norm and _norm(head) != _norm(title):
                reason = "head_not_in_quote_or_title"
            elif _norm(head) == _norm(tail):
                reason = "self_loop"

        key = (_norm(head), relation, _norm(tail))
        if not reason and key in seen:
            reason = "duplicate_triple"
        if reason:
            rejected.append({"source_index": source_index, "reason": reason, "raw": dict(raw)})
            continue

        seen.add(key)
        passage = passages[passage_rank - 1]
        accepted.append({
            "schema_version": PASSAGE_SRO_SCHEMA_VERSION,
            "head": head,
            "relation": relation,
            "tail": tail,
            "relation_trigger": trigger,
            "evidence_quote": quote,
            "evidence_quote_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest(),
            "passage_rank": passage_rank,
            "passage_id": str(passage.get("id") or f"rank-{passage_rank}"),
            "passage_title": _passage_title(passage),
            "validator_version": PASSAGE_SRO_VALIDATOR_VERSION,
        })
        if len(accepted) >= max_edges:
            for skipped_index in range(source_index + 1, len(raw_edges)):
                rejected.append({
                    "source_index": skipped_index,
                    "reason": "over_frozen_edge_cap",
                    "raw": dict(raw_edges[skipped_index]),
                })
            break
    return accepted, rejected


def triples_from_edges(edges: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    return [
        [str(edge["head"]), str(edge["relation"]), str(edge["tail"])]
        for edge in edges
    ]
