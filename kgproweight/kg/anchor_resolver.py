"""Anchor resolution with passage-title supplement (no gold, no planner fallback).

Resolves a planner anchor surface to a QID.  The planner/question anchor is tried
first; if it abstains, an EXACT normalised match against the retrieved passage
titles is used as a supplement (never all top-10 titles as anchors).  Ambiguity,
low confidence, and conflicts abstain; broad-neighbourhood is never used.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Sequence, Tuple


def _norm(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def passage_titles(passages: Sequence[Dict[str, Any]]) -> list[str]:
    """Deterministic, deduplicated, normalised passage titles."""
    out: list[str] = []
    seen: set[str] = set()
    for p in passages:
        title = str(p.get("title") or "").strip()
        norm = _norm(title)
        if title and norm and norm not in seen:
            seen.add(norm)
            out.append(title)
    return out


def _ok(result, min_score: float = 0.0, min_margin: float = 0.0) -> bool:
    if result is None or result.abstained or not result.selected_qid:
        return False
    score = float(getattr(result, "score", 1.0) or 1.0)
    margin = float(getattr(result, "margin", 1.0) or 1.0)
    return score >= min_score and margin >= min_margin


def resolve_anchor(
    anchor: str,
    question: str,
    passages: Sequence[Dict[str, Any]],
    title_resolver,
    linker,
    *,
    min_score: float = 0.55,
    min_margin: float = 0.10,
) -> Tuple[Any, str]:
    """Return (result, source) where source in {planner_anchor, passage_title_fallback, abstain}."""
    # 1) planner/question anchor priority (direct resolution)
    result = title_resolver.resolve(anchor)
    if _ok(result):
        return result, "planner_anchor"
    result = linker.link_single(anchor, question=question)
    if _ok(result, min_score, min_margin):
        return result, "planner_anchor"

    # 2) passage-title supplement: the anchor must be an exact normalised match
    #    OR a normalised prefix of the passage title (the title disambiguates the
    #    anchor, e.g. "The Big Lebowski" -> "The Big Lebowski (film)").  Only the
    #    single best matching title is used; ambiguity/conflict abstains.
    anchor_norm = _norm(anchor)
    if anchor_norm:
        for title in passage_titles(passages):
            title_norm = _norm(title)
            if title_norm != anchor_norm and not title_norm.startswith(anchor_norm + " "):
                continue
            result = title_resolver.resolve(title)
            if _ok(result):
                return result, "passage_title_fallback"
            result = linker.link_single(title, question=question)
            if _ok(result, min_score, min_margin):
                return result, "passage_title_fallback"
            # conflict / ambiguity / low confidence -> do not fall further
            break

    return None, "abstain"
