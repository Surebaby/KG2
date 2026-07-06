"""EM / F1 / heuristic IHR metrics.

These mirror FlashRAG's own definitions so that the numbers we report in
the paper are comparable. The heuristic IHR is produced offline from a
saved trajectory; the GPT-4o LLM-as-Judge IHR lives in
:mod:`kgproweight.reward.ihr_judge`.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, List, Sequence


def _normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def compute_em(pred: str, golds: Sequence[str]) -> float:
    """Returns 1.0 if pred matches any gold after normalisation."""
    pn = _normalize(pred)
    return 1.0 if any(pn == _normalize(g) for g in golds) else 0.0


def compute_f1(pred: str, golds: Sequence[str]) -> float:
    """Token-level F1 vs the best-matching gold."""
    p = _normalize(pred).split()
    if not p:
        return 0.0
    best = 0.0
    for g in golds:
        g_tokens = _normalize(g).split()
        if not g_tokens:
            continue
        common = Counter(p) & Counter(g_tokens)
        n_same = sum(common.values())
        if n_same == 0:
            continue
        prec = n_same / len(p)
        rec = n_same / len(g_tokens)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def heuristic_ihr(ihr_flags: Sequence[Dict[str, Any]]) -> float:
    """Mean IHR (n_hallucinated / n_steps) over a list of flag dicts."""
    valid = [
        x.get("ihr_heuristic")
        for x in ihr_flags
        if isinstance(x, dict) and x.get("ihr_heuristic") is not None
    ]
    return float(sum(valid) / len(valid)) if valid else 0.0


def aggregate_metrics(
    predictions: Sequence[str],
    gold_lists: Sequence[Sequence[str]],
    ihr_flags: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, float]:
    """Return a dict with ``em``, ``f1``, and optionally ``ihr_heuristic``."""
    if len(predictions) != len(gold_lists):
        raise ValueError("predictions and gold_lists must have the same length")
    n = max(1, len(predictions))
    em = sum(compute_em(p, g) for p, g in zip(predictions, gold_lists)) / n
    f1 = sum(compute_f1(p, g) for p, g in zip(predictions, gold_lists)) / n
    out = {"em": em, "f1": f1}
    if ihr_flags:
        out["ihr_heuristic"] = heuristic_ihr(ihr_flags)
    return out
