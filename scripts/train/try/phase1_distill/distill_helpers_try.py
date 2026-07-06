"""Helpers for the `try` Phase-1 variant.

This module holds the *changed* logic only; everything unchanged is reused
from the original ``kgproweight`` package. Five improvements live here:

1. Lenient answer matching (``answer_match_score``): normalise gold/pred,
   handle the very common "gold is a substring of pred" case, and use a
   recall-leaning F1 so that correct-but-verbose answers are not rejected.
3. Robust mention extraction (``extract_mentions_robust``): use retrieved
   passage *titles* as additional anchors on top of the capitalised-phrase
   regex (optionally spaCy NER when installed). Coverage becomes a soft
   metric only.
2. Stratified acceptance (``StratifiedSilverFilter``): instead of hard
   rejecting low-triple-rate / low-coverage trajectories, bucket them and
   keep a configurable quota of the sparse bucket so the α-Gate can learn
   the α→0 fallback region.

The remaining two requested changes (SPARQL timeout/degrade, prewarm-first)
are wired in ``phase1_distill_try.py`` and ``phase1_generate_silver_try.py``.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# 1. Lenient answer matching
# ---------------------------------------------------------------------------

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_LEADIN_RE = re.compile(
    r"^(the\s+answer\s+is|answer\s*:|final\s+answer\s*:|it\s+is|this\s+is)\s+",
    re.IGNORECASE,
)


def normalize_answer(s: str) -> str:
    """SQuAD-style normalisation: lower, strip articles, strip punctuation."""
    s = s.lower().strip()
    s = _ARTICLE_RE.sub(" ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def clean_final_answer(pred: str) -> str:
    """Reduce a possibly-verbose final answer to its core entity phrase.

    - drop a leading "the answer is" / "answer:" lead-in,
    - keep only the first line,
    - keep only the part before the first sentence-final separator so that
      "Albert Einstein, a physicist" → "Albert Einstein".
    """
    if not pred:
        return ""
    text = pred.strip().splitlines()[0].strip()
    text = _LEADIN_RE.sub("", text).strip()
    # Cut at the first comma / semicolon / dash / "(" / " because"/" which".
    text = re.split(r"[,;(]| because | which | that | was | is ", text, maxsplit=1)[0]
    return text.strip().strip(".").strip()


def _token_f1(pred: str, gold: str) -> float:
    """Standard token-level F1 over normalised tokens."""
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p or not g:
        return 0.0
    common = set(p) & set(g)
    if not common:
        return 0.0
    n_same = sum(min(p.count(t), g.count(t)) for t in common)
    if n_same == 0:
        return 0.0
    prec = n_same / len(p)
    rec = n_same / len(g)
    return 2 * prec * rec / (prec + rec)


def _token_recall(pred: str, gold: str) -> float:
    """Fraction of gold tokens present in pred — robust to verbose preds."""
    p = set(normalize_answer(pred).split())
    g = normalize_answer(gold).split()
    if not g:
        return 0.0
    hit = sum(1 for t in g if t in p)
    return hit / len(g)


def answer_match_score(pred: str, gold: str) -> float:
    """Lenient match score in ``[0, 1]`` used for silver acceptance.

    The score is the maximum over three views so that a *correct* answer
    surrounded by extra words is not penalised by precision:

      * exact match after normalisation                       → 1.0
      * gold is a contiguous substring of the cleaned pred    → 1.0
      * token recall of gold inside pred (handles verbosity)
      * plain token-F1 on the cleaned pred (handles aliases)

    ``clean_final_answer`` is applied first to strip lead-ins / trailing
    clauses, which is where the original strict F1 lost most good traces.
    """
    if not pred or not gold:
        return 0.0
    cleaned = clean_final_answer(pred)
    n_pred_full = normalize_answer(pred)
    n_pred_clean = normalize_answer(cleaned)
    n_gold = normalize_answer(gold)
    if not n_gold:
        return 0.0

    # Exact (either on the cleaned or full pred).
    if n_gold == n_pred_clean or n_gold == n_pred_full:
        return 1.0
    # Substring: gold fully contained as a token sequence in pred.
    if n_gold and (f" {n_gold} " in f" {n_pred_full} " or f" {n_gold} " in f" {n_pred_clean} "):
        return 1.0

    recall = max(_token_recall(cleaned, gold), _token_recall(pred, gold))
    f1 = max(_token_f1(cleaned, gold), _token_f1(pred, gold))
    return max(recall, f1)


# ---------------------------------------------------------------------------
# 3. Robust mention extraction (passage-title anchors + optional spaCy NER)
# ---------------------------------------------------------------------------

_CAP_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
_MENTION_BLACKLIST = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "should", "would", "will", "the", "a", "an",
}

# Lazily-loaded spaCy pipeline (None means "not attempted yet", False means
# "tried and unavailable").
_SPACY_NLP: Any = None


def _maybe_spacy():
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP or None
    try:
        import spacy  # type: ignore

        _SPACY_NLP = spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger"])
    except Exception:
        _SPACY_NLP = False
    return _SPACY_NLP or None


def _passage_title(p: Any) -> Optional[str]:
    """Pull the title out of a FlashRAG passage dict (``contents = title\\ntext``)."""
    if isinstance(p, dict):
        title = p.get("title")
        if title:
            return str(title).strip()
        contents = str(p.get("contents") or "").strip()
        if contents:
            return contents.splitlines()[0].strip()
    return None


def extract_mentions_robust(
    question: str,
    passages: Optional[Sequence[Any]] = None,
    max_n: int = 8,
    title_anchor_top: int = 5,
) -> List[str]:
    """Best-effort mention set combining three sources.

    Order of preference (deduplicated, capped at ``max_n``):
      1. spaCy NER entities on the question (if spaCy is installed),
      2. capitalised noun phrases from the question (regex),
      3. titles of the top retrieved passages (strong anchors — the gold
         supporting docs in HotpotQA/2Wiki are titled by their key entity).
    """
    seen: Dict[str, None] = {}

    def _add(m: str) -> None:
        m = m.strip()
        if len(m) < 3:
            return
        if m.lower() in _MENTION_BLACKLIST:
            return
        seen.setdefault(m, None)

    nlp = _maybe_spacy()
    if nlp is not None and question:
        try:
            for ent in nlp(question).ents:
                if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT", "WORK_OF_ART", "NORP", "PRODUCT"}:
                    _add(ent.text)
        except Exception:
            pass

    for m in _CAP_PHRASE_RE.findall(question or ""):
        _add(m)

    if passages:
        for p in list(passages)[:title_anchor_top]:
            title = _passage_title(p)
            if title:
                _add(title)

    return list(seen.keys())[:max_n]


# ---------------------------------------------------------------------------
# 2. Stratified acceptance filter
# ---------------------------------------------------------------------------

@dataclass
class StratifiedDecision:
    accepted: bool
    bucket: str          # "kg_rich" | "kg_medium" | "kg_sparse" | "rejected_quality"
    triple_rate: float
    reason: str


@dataclass
class StratifiedSilverFilter:
    """Accept trajectories by KG-density bucket with per-bucket quotas.

    Unlike the original hard ``triple_rate``/``coverage`` rejection, low-KG
    trajectories are *not* discarded outright. They are routed to a
    ``kg_sparse`` bucket which keeps up to ``sparse_quota`` of the total
    accepted count, so the α-Gate sees genuine α→0 fallback examples.

    Quality gates that everyone must pass (regardless of bucket):
      * step count within ``[min_steps, max_steps]``,
      * lenient answer-match score ≥ ``min_answer_score``.

    ``coverage`` and ``triple_rate`` are recorded but never hard-reject; they
    only decide the bucket and are subject to quota.
    """

    min_steps: int = 3
    max_steps: int = 7
    min_answer_score: float = 0.3
    # bucket thresholds on triple_rate
    rich_triple_rate: float = 0.5
    medium_triple_rate: float = 0.15
    # fraction of the accepted pool allowed to come from the sparse bucket
    sparse_quota: float = 0.25
    medium_quota: float = 0.35

    # running counters (mutated as trajectories stream in)
    _counts: Dict[str, int] = field(default_factory=lambda: {"kg_rich": 0, "kg_medium": 0, "kg_sparse": 0})

    @property
    def total_accepted(self) -> int:
        return sum(self._counts.values())

    def _bucket_for(self, triple_rate: float) -> str:
        if triple_rate >= self.rich_triple_rate:
            return "kg_rich"
        if triple_rate >= self.medium_triple_rate:
            return "kg_medium"
        return "kg_sparse"

    def decide(self, steps, coverage: float, answer_score: float) -> StratifiedDecision:
        n = len(steps)
        n_with_triples = sum(1 for s in steps if s.cited_triples)
        triple_rate = n_with_triples / max(n, 1)

        # universal quality gates
        if n < self.min_steps or n > self.max_steps:
            return StratifiedDecision(False, "rejected_quality", triple_rate, f"step_count={n}")
        if answer_score < self.min_answer_score:
            return StratifiedDecision(False, "rejected_quality", triple_rate, f"answer_score={answer_score:.2f}")

        bucket = self._bucket_for(triple_rate)

        # quota check for the non-rich buckets (rich always accepted)
        total = self.total_accepted
        if bucket == "kg_sparse" and total > 0:
            if self._counts["kg_sparse"] >= self.sparse_quota * (total + 1):
                return StratifiedDecision(False, "kg_sparse", triple_rate, "sparse_quota_full")
        elif bucket == "kg_medium" and total > 0:
            if self._counts["kg_medium"] >= self.medium_quota * (total + 1):
                # demote to acceptance only if it would not break quota; else reject
                return StratifiedDecision(False, "kg_medium", triple_rate, "medium_quota_full")

        self._counts[bucket] += 1
        return StratifiedDecision(True, bucket, triple_rate, "ok")

    def stats(self) -> Dict[str, int]:
        return dict(self._counts)
