"""Phase 1 — Graph-Guided Trajectory Distillation.

Fixes bug #4: the legacy code's ``get_retrieved_text_placeholder`` always
returned a literal placeholder string. The Teacher now sees the actual
RRF top-K passages retrieved through :mod:`kgproweight.retrieval.hybrid`.

Entry-point: :func:`run_phase1`.
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import string
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import re

from kgproweight.data.parsers import (
    extract_final_answer,
    parse_steps,
)
from kgproweight.data.prompts import build_teacher_messages
from kgproweight.data.silver_dataset import SilverDatasetReader, SilverStepRecord, SilverTrajectory
from kgproweight.kg.coverage import coverage_score
from kgproweight.kg.entity_linker import (
    EntityLinker,
    build_passage_text,
    build_passage_titles,
    extract_mentions,
    passage_title,
)
from kgproweight.kg.wikidata_retriever import WikidataSubgraphRetriever
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.retrieval.bridge import additive_bridge_candidates, extract_bridge_queries
from kgproweight.retrieval.hybrid import DEFAULT_TOPK
from kgproweight.utils.logging import dump_manifest, get_logger
from kgproweight.utils.paths import data_dir, index_dir

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_f1(pred: str, gold: str) -> float:
    """Standard token-level F1, normalised."""
    import re
    import string

    def norm(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = s.translate(str.maketrans("", "", string.punctuation))
        return " ".join(s.split())

    p = norm(pred).split()
    g = norm(gold).split()
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


# ---------------------------------------------------------------------------
# Lenient answer matching (ported verbatim from the validated `try` variant)
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
# Robust mention extraction (passage-title anchors + optional spaCy NER)
# Ported verbatim from the validated `try` variant.
# ---------------------------------------------------------------------------

_CAP_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")

# Share the entity_linker blacklist + affix stripper so Phase 1 mentions and
# inference-time mentions are extracted identically (they feed the same linker
# and the same KG cache — divergence here means train/inference KG mismatch).
from kgproweight.kg.entity_linker import (  # noqa: E402
    _MENTION_BLACKLIST,
    _strip_stopword_affixes,
)

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
    title = passage_title(p)
    return title or None


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
        m = _strip_stopword_affixes(m.strip())
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


def _select_relevant_triples(
    question: str,
    passages: Sequence[Dict[str, Any]],
    triples: Sequence[Sequence[str]],
    top_n: int,
) -> List[Sequence[str]]:
    """Keep KG context focused so the teacher can cite it more reliably.

    Scores each triple by token overlap with question + top retrieved passages.
    Falls back to original order when overlap is not informative.
    """
    if not triples:
        return []
    if len(triples) <= top_n:
        return list(triples)

    text_blobs: List[str] = [question]
    for p in list(passages)[:8]:
        t = str(p.get("contents") or p.get("text") or "").strip()
        if t:
            text_blobs.append(t[:1200])
    context = " ".join(text_blobs).lower()
    ctx_tokens = set(re.findall(r"[a-z0-9]+", context))
    if not ctx_tokens:
        return list(triples)[:top_n]

    scored: List[tuple[int, int, Sequence[str]]] = []
    for i, tri in enumerate(triples):
        if len(tri) != 3:
            continue
        tri_text = " ".join(str(x) for x in tri).lower()
        tri_tokens = set(re.findall(r"[a-z0-9]+", tri_text))
        overlap = len(ctx_tokens & tri_tokens)
        scored.append((overlap, -i, tri))

    scored.sort(reverse=True)
    picked = [t for _, _, t in scored[:top_n]]
    if not any(s > 0 for s, _, _ in scored[:top_n]):
        return list(triples)[:top_n]
    return picked


# ---------------------------------------------------------------------------
# Teacher client
# ---------------------------------------------------------------------------

def _is_reasoning_model(model: str) -> bool:
    """Does this model emit a separate, max_tokens-billed reasoning channel?"""
    m = (model or "").lower()
    return (
        m.startswith(("o1", "o3", "o4"))
        or "reasoner" in m
        or "thinking" in m
        or any(m.startswith(f"deepseek-v{v}") for v in ("4", "5", "6"))
    )


@dataclass
class TeacherClient:
    """Tiny wrapper around an OpenAI-compatible chat client."""

    model: str = "deepseek-chat"
    backend: str = "deepseek"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 1500
    max_retries: int = 3
    # None preserves provider defaults. Formal V4 pilots must set this
    # explicitly so a model swap does not silently also swap reasoning mode.
    thinking: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    # 2026-08-24: the SDK default is 600 s, so a request the server never answers
    # blocked one worker for 600 s x (1 + max_retries) without ever raising. In
    # the teacher-swap pilot that silently lost 81/300 items: the futures were
    # abandoned when the iterator ended, no warning was logged, and the surviving
    # 219 were biased toward fast items. Bound every request.
    timeout: float = 90.0

    def __post_init__(self) -> None:
        from openai import OpenAI

        if self.backend == "deepseek":
            api_key = self.api_key or os.environ.get("DEEPSEEK_API_KEY")
            base_url = self.base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        else:
            api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            base_url = self.base_url or os.environ.get("OPENAI_BASE_URL")
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout)
        # Reasoning models (deepseek-v4-*, o-series) bill `reasoning_content`
        # against max_tokens and measured 3.4k on these prompts, so the 1500
        # default returns content='' with finish_reason='length' -- indistinguish-
        # able from teacher failure. Raise the floor rather than fail silently.
        if _is_reasoning_model(self.model) and self.thinking is not False and self.max_tokens < 12000:
            logger.warning(
                "%s is a reasoning model: raising max_tokens %d -> 12000 "
                "(reasoning_content is billed against max_tokens; measured ~3.4k "
                "on Phase-1 prompts, and 1500 returns EMPTY content).",
                self.model, self.max_tokens,
            )
            self.max_tokens = 12000

    def chat_with_metadata(
        self, messages: List[Dict[str, str]]
    ) -> tuple[str, Dict[str, Any]]:
        """Return visible content plus non-secret API accounting metadata."""
        last_exc: Optional[Exception] = None
        started = time.monotonic()
        for attempt in range(self.max_retries):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if self.thinking is not None:
                    kwargs["extra_body"] = {
                        "thinking": {"type": "enabled" if self.thinking else "disabled"}
                    }
                if self.thinking and self.reasoning_effort:
                    kwargs["reasoning_effort"] = self.reasoning_effort
                resp = self._client.chat.completions.create(
                    **kwargs,
                )
                msg = resp.choices[0].message
                content = msg.content or ""
                usage = getattr(resp, "usage", None)
                prompt_details = getattr(usage, "prompt_tokens_details", None)
                metadata = {
                    "requested_model": self.model,
                    "response_model": str(getattr(resp, "model", "") or ""),
                    "thinking": self.thinking,
                    "reasoning_effort": self.reasoning_effort,
                    "attempts": attempt + 1,
                    "latency_seconds": time.monotonic() - started,
                    "finish_reason": getattr(resp.choices[0], "finish_reason", None),
                    "reasoning_chars": len(getattr(msg, "reasoning_content", None) or ""),
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "cache_hit_tokens": getattr(prompt_details, "cached_tokens", None),
                }
                if not content.strip():
                    # Distinguish budget exhaustion from a genuine empty reply,
                    # instead of returning "" and looking like teacher failure.
                    fr = getattr(resp.choices[0], "finish_reason", None)
                    n_reason = len(getattr(msg, "reasoning_content", None) or "")
                    logger.warning(
                        "Teacher returned EMPTY content (finish_reason=%s, "
                        "reasoning_chars=%d, max_tokens=%d)%s",
                        fr, n_reason, self.max_tokens,
                        " -- raise max_tokens" if fr == "length" else "",
                    )
                return content, metadata
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("Teacher attempt %d/%d failed: %s", attempt + 1, self.max_retries, exc)
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Teacher generation failed after {self.max_retries} attempts: {last_exc}")

    def chat(self, messages: List[Dict[str, str]]) -> str:
        content, _ = self.chat_with_metadata(messages)
        return content


# ---------------------------------------------------------------------------
# Retrieval adapter
# ---------------------------------------------------------------------------

class _RetrievalAdapter:
    """Minimal adapter around FlashRAG's retriever for Phase 1.

    R9 v6: when ``rerank_topk > 0``, applies cross-encoder reranking after RRF
    retrieval. The retriever returns top-50 candidates, then the cross-encoder
    re-ranks and selects the top-K for the Teacher prompt.
    """

    def __init__(self, retriever: Any, top_k: int = DEFAULT_TOPK,
                 rerank_topk: int = 0,
                 cross_encoder_model: str = "models/bge-reranker-v2-m3",
                 bridge_mode: str = "off",
                 bridge_first_round_topk: int = 5,
                 bridge_max_queries: int = 2,
                 bridge_only_k: int = 50) -> None:
        self.retriever = retriever
        self.top_k = top_k
        self.rerank_topk = rerank_topk
        self.cross_encoder_model = cross_encoder_model
        self.bridge_mode = bridge_mode
        self.bridge_first_round_topk = bridge_first_round_topk
        self.bridge_max_queries = bridge_max_queries
        self.bridge_only_k = bridge_only_k
        if bridge_mode not in {"off", "additive_v3"}:
            raise ValueError(f"unknown Phase-1 bridge_mode={bridge_mode!r}")
        if bridge_mode == "additive_v3" and rerank_topk <= 0:
            raise ValueError("additive_v3 requires rerank_topk > 0")
        if bridge_first_round_topk <= 0 or bridge_max_queries <= 0 or bridge_only_k < 0:
            raise ValueError("invalid additive bridge retrieval limits")

    def _rerank(
        self,
        queries: Sequence[str],
        candidates: Sequence[Sequence[Dict[str, Any]]],
        *,
        topk: int,
    ) -> List[List[Dict[str, Any]]]:
        from kgproweight.retrieval.reranker import rerank_passages

        return rerank_passages(
            list(queries),
            [list(row) for row in candidates],
            topk=topk,
            method="cross-encoder",
            cross_encoder_model=self.cross_encoder_model,
        )

    def _batch_additive_v3(
        self,
        queries: Sequence[str],
        original: Sequence[Sequence[Dict[str, Any]]],
    ) -> List[List[Dict[str, Any]]]:
        if not hasattr(self.retriever, "batch_search"):
            raise TypeError("additive_v3 batch retrieval requires retriever.batch_search")
        first_round = self._rerank(
            queries,
            original,
            topk=self.bridge_first_round_topk,
        )
        bridge_queries = [
            extract_bridge_queries(
                question,
                docs,
                max_docs=self.bridge_first_round_topk,
                max_bridges=self.bridge_max_queries,
            )
            for question, docs in zip(queries, first_round)
        ]
        flat_queries: List[str] = []
        owners: List[int] = []
        for owner, values in enumerate(bridge_queries):
            for value in values:
                flat_queries.append(value)
                owners.append(owner)
        flat_results = (
            [list(row) for row in self.retriever.batch_search(flat_queries)]
            if flat_queries
            else []
        )
        if len(flat_results) != len(flat_queries):
            raise ValueError(
                f"bridge retriever returned {len(flat_results)} rows for "
                f"{len(flat_queries)} queries"
            )
        by_owner: List[List[List[Dict[str, Any]]]] = [[] for _ in queries]
        for owner, results in zip(owners, flat_results):
            by_owner[owner].append(results)
        augmented = [
            additive_bridge_candidates(
                original[i],
                by_owner[i],
                max_bridge_only=self.bridge_only_k,
            )
            for i in range(len(queries))
        ]
        logger.info(
            "Phase 1 additive_v3: questions=%d bridge_queries=%d "
            "candidate_size[min/mean/max]=%d/%.1f/%d",
            len(queries),
            len(flat_queries),
            min((len(row) for row in augmented), default=0),
            sum(len(row) for row in augmented) / max(1, len(augmented)),
            max((len(row) for row in augmented), default=0),
        )
        return self._rerank(queries, augmented, topk=self.rerank_topk)

    def __call__(self, query: str) -> List[Dict[str, Any]]:
        if self.bridge_mode == "additive_v3" and hasattr(self.retriever, "batch_search"):
            return self.batch([query])[0]
        if self.bridge_mode == "additive_v3":
            raise TypeError("additive_v3 requires retriever.batch_search")
        if hasattr(self.retriever, "search"):
            results = self.retriever.search(query)
        elif hasattr(self.retriever, "batch_search"):
            results = self.retriever.batch_search([query])[0]
        else:
            return []

        # R9 v6: rerank through the shared dispatcher so Phase 1 (silver data)
        # and inference use the SAME reranker. Previously Phase 1 loaded a
        # cross-encoder here while the eval pipeline used a hand-rolled BM25
        # scorer, so the passage distribution the student trained on differed
        # from the one it saw at test time.
        if self.rerank_topk > 0 and len(results) >= self.rerank_topk:
            return self._rerank([query], [list(results)], topk=self.rerank_topk)[0]

        return list(results)[: self.top_k]

    def batch(self, queries: Sequence[str]) -> List[List[Dict[str, Any]]]:
        """Retrieve a fixed query set in one dense-index pass when supported.

        The wiki18 fp16 memmap is an exact flat index. Calling ``search`` once
        per Teacher worker rescans all 21M vectors per question and lets four
        workers allocate multi-GB chunks concurrently. Batch search preserves
        the ranking but scans each database chunk once for the whole pilot.
        """
        if not queries:
            return []
        if not hasattr(self.retriever, "batch_search"):
            return [self(query) for query in queries]
        results = self.retriever.batch_search(list(queries))
        results = [list(row) for row in results]
        if self.bridge_mode == "additive_v3":
            return self._batch_additive_v3(queries, results)
        if self.rerank_topk > 0:
            return self._rerank(queries, results, topk=self.rerank_topk)
        return [row[: self.top_k] for row in results]


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

@dataclass
class SilverFilter:
    min_steps: int = 3
    max_steps: int = 7
    min_triple_rate: float = 0.4
    min_coverage: float = 0.5
    min_token_f1: float = 0.5

    def accepts(
        self,
        steps,
        coverage: float,
        token_f1: float,
    ) -> bool:
        n = len(steps)
        if n < self.min_steps or n > self.max_steps:
            return False
        if coverage < self.min_coverage:
            return False
        if token_f1 < self.min_token_f1:
            return False
        n_with_triples = sum(1 for s in steps if s.cited_triples)
        if n_with_triples / max(n, 1) < self.min_triple_rate:
            return False
        return True


# ---------------------------------------------------------------------------
# Stratified acceptance filter (ported verbatim from the validated `try`
# variant). This is the new default; the hard ``SilverFilter`` above is kept
# for back-compat.
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

    def assess_quality(
        self,
        steps,
        answer_score: float,
        hard_reject_reason: str = "",
    ) -> StratifiedDecision:
        """Judge intrinsic trajectory quality without applying composition quotas."""
        n = len(steps)
        n_with_triples = sum(1 for s in steps if s.cited_triples)
        triple_rate = n_with_triples / max(n, 1)
        if hard_reject_reason:
            return StratifiedDecision(
                False, "rejected_quality", triple_rate, hard_reject_reason
            )
        if n < self.min_steps or n > self.max_steps:
            return StratifiedDecision(
                False, "rejected_quality", triple_rate, f"step_count={n}"
            )
        if answer_score < self.min_answer_score:
            return StratifiedDecision(
                False,
                "rejected_quality",
                triple_rate,
                f"answer_score={answer_score:.2f}",
            )
        return StratifiedDecision(True, self._bucket_for(triple_rate), triple_rate, "ok")

    def decide(self, steps, coverage: float, answer_score: float) -> StratifiedDecision:
        quality = self.assess_quality(steps, answer_score)
        if not quality.accepted:
            return quality
        triple_rate = quality.triple_rate
        bucket = quality.bucket

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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@dataclass
class Phase1Config:
    dataset_name: str
    items: Sequence[Dict[str, Any]]
    output_path: str
    teacher_client: TeacherClient
    retriever_factory: Any  # callable returning a retriever, or pre-built object
    entity_linker: EntityLinker
    kg_retriever: WikidataSubgraphRetriever
    append_output: bool = False
    prm_annotator: Optional[PRMAnnotator] = None
    top_k: int = DEFAULT_TOPK
    # 2026-08-24 (retraining_plan §12.3): 50 -> 12 to match the student. The
    # Teacher used to see a top-50 KG block while PPO/inference render only 12
    # (ppo_max_kg_triples=12, pipeline max_kg_triples=12), so 44.5% of teacher
    # citations pointed at triples the student can never see. Measured: accepted
    # trajectories cite 1.66 distinct triples on average (p99=6, 99.9% <= 12),
    # so 12 is ~3x the actual demand -- the binding constraint is the ranker,
    # not this budget (quota_filter alone caps citation recall at 83.6%).
    max_kg_triples: int = 12
    # min_keep for filter_and_rank_triples. Was never passed here (defaulting
    # to 0) while every student path passes 5, so silver was distilled through a
    # STRICTER filter than inference uses -- a second train/serve mismatch with
    # the same root cause as max_kg_triples. min_keep>0 relaxes only the score
    # threshold; hard-delete and quota are never relaxed (kg_filter.py:770).
    min_kg_keep: int = 5
    rerank_topk: int = 0  # R9 v6: 0=disabled, >0 enables cross-encoder rerank
    cross_encoder_model: str = "models/bge-reranker-v2-m3"
    bridge_mode: str = "off"
    bridge_first_round_topk: int = 5
    bridge_max_queries: int = 2
    bridge_only_k: int = 50
    max_workers: int = 1
    accept_filter: StratifiedSilverFilter = field(default_factory=StratifiedSilverFilter)
    seed: int = 42
    teacher_temperature: float = 0.3
    extra_metadata: Optional[Dict[str, Any]] = None


def _annotate_steps(
    raw_output: str,
    kg_subgraph,
    annotator: PRMAnnotator,
) -> List[SilverStepRecord]:
    parsed = parse_steps(raw_output, known_kg=kg_subgraph)
    labels = annotator.annotate_trajectory(parsed, list(kg_subgraph))
    out: List[SilverStepRecord] = []
    for step, label in zip(parsed, labels):
        out.append(
            SilverStepRecord(
                index=step.index,
                text=step.raw_text,
                label=float(label),
                cited_triples=list(step.cited_triples),
                token_logprobs=None,
            )
        )
    return out


def _parsed_contract_errors(raw_output: str, kg_subgraph) -> List[str]:
    """Return stable per-step citation-schema errors for audit/rejection."""
    errors: List[str] = []
    for step in parse_steps(raw_output, known_kg=kg_subgraph):
        for error in step.citation_contract_errors:
            errors.append(f"step_{step.index}:{error}")
    return errors


def _needs_format_retry(
    steps: List[SilverStepRecord],
    kg_subgraph: Sequence[tuple | list],
    min_steps: int,
    *,
    raw_output: str = "",
    max_steps: int = 7,
) -> bool:
    """Heuristic: retry once when format quality is clearly below Phase1 expectations."""
    if len(steps) < min_steps or len(steps) > max_steps:
        return True
    if raw_output and _parsed_contract_errors(raw_output, kg_subgraph):
        return True
    return False


def _build_retry_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    retry_hint = (
        "Regenerate strictly with the required schema. "
        "You MUST output 3-7 steps. "
        "Every step must contain exactly one single-line `Knowledge Used: [...]` field. "
        "Cite only relevant triples copied VERBATIM from the supplied KG block. "
        "If no supplied triple supports a step, use `Knowledge Used: []` even when the KG is non-empty. "
        "Do not write example, hypothetical, or absent triples anywhere in the response. "
        "Keep [Final Answer] concise."
    )
    return [*messages, {"role": "user", "content": retry_hint}]


def _process_one(
    item: Dict[str, Any],
    cfg: Phase1Config,
    retriever_call: _RetrievalAdapter,
) -> Optional["_Candidate"]:
    qid = str(item.get("id") or item.get("qid") or "")
    question = str(item.get("question", "")).strip()
    gold_list = item.get("golden_answers") or ([item.get("answer", "")] if item.get("answer") else [])
    gold = str(gold_list[0]) if gold_list else ""

    if not question:
        return None

    # ---- Hybrid retrieval first: passages double as mention anchors -------
    if "_retrieved_passages" in item:
        passages = list(item["_retrieved_passages"])
    else:
        try:
            passages = retriever_call(question)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Retrieval failed for qid=%s: %s", qid, exc)
            passages = []

    # ---- robust mentions: NER/regex + passage titles ---------------------
    # R9 v6: pass question context to entity linker for disambiguation.
    # R9 v7: also pass passage titles + bodies so silver-data linking matches
    # the inference/reward paths (train/eval alignment).
    mentions = extract_mentions_robust(question, passages=passages, max_n=8)
    qids = []
    linked = {}  # mention → QID (for coverage + metadata)
    if mentions:
        _titles = build_passage_titles(passages)
        _ptext = build_passage_text(passages)
        for m in mentions:
            result = cfg.entity_linker.link_single(
                m, question=question,
                retrieved_titles=_titles,
                passage_text=_ptext,
            )
            if result.selected_qid and not result.abstained:
                qids.append(result.selected_qid)
                linked[m] = result.selected_qid

    # ---- SPARQL degrade: empty subgraph is allowed -----------------------
    triples: List = []
    if qids:
        try:
            triples = cfg.kg_retriever.fetch(qids)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SPARQL fetch failed for qid=%s (%d qids): %s", qid, len(qids), exc)
            triples = []

    # coverage is now a *soft* signal only (recorded, never rejects)
    coverage = coverage_score(mentions, linked) if mentions else 0.0

    # R9 v6: apply v2 scoring (3-layer filter) before Teacher sees KG
    from kgproweight.kg.kg_filter import filter_and_rank_triples
    teacher_kg = filter_and_rank_triples(
        triples, question=question,
        max_keep=cfg.max_kg_triples, min_keep=cfg.min_kg_keep,
    )

    messages = build_teacher_messages(
        question=question,
        retrieved_passages=passages,
        kg_triples=teacher_kg,
        top_k=cfg.top_k,
        max_kg_triples=cfg.max_kg_triples,
    )

    raw_output = cfg.teacher_client.chat(messages)
    if not raw_output.strip():
        return None
    annotator = cfg.prm_annotator or PRMAnnotator(entity_linker=cfg.entity_linker, verbose=False)
    # 2026-08-24: verify citations against `teacher_kg` (what the Teacher was
    # actually shown and what is stored as kg_subgraph), NOT the unfiltered
    # `triples`. Annotating against `triples` labelled a citation POSITIVE even
    # when the cited triple had been filtered out of the prompt -- so the PRM
    # labels were computed against one KG and persisted beside a different one.
    parsed_steps = _annotate_steps(raw_output, teacher_kg, annotator)

    # One-shot corrective retry when the model ignores required structure or
    # cites anything outside the exact KG block it saw.
    min_steps = cfg.accept_filter.min_steps
    retry_attempted = _needs_format_retry(
        parsed_steps,
        teacher_kg,
        min_steps,
        raw_output=raw_output,
        max_steps=cfg.accept_filter.max_steps,
    )
    retry_succeeded = False
    if retry_attempted:
        retry_messages = _build_retry_messages(messages)
        retry_output = cfg.teacher_client.chat(retry_messages)
        if retry_output.strip():
            retry_steps = _annotate_steps(retry_output, teacher_kg, annotator)
            if not _needs_format_retry(
                retry_steps,
                teacher_kg,
                min_steps,
                raw_output=retry_output,
                max_steps=cfg.accept_filter.max_steps,
            ):
                raw_output = retry_output
                parsed_steps = retry_steps
                retry_succeeded = True

    citation_contract_errors = _parsed_contract_errors(raw_output, teacher_kg)

    final_answer = extract_final_answer(raw_output) or ""
    # ---- lenient answer match --------------------------------------------
    answer_score = answer_match_score(final_answer, gold) if gold else 0.0

    traj = SilverTrajectory(
        qid=qid,
        question=question,
        answer=final_answer,
        dataset=cfg.dataset_name,
        steps=parsed_steps,
        kg_subgraph=teacher_kg,
        retrieved_passages=passages,
        teacher_output=raw_output,
        teacher_model=cfg.teacher_client.model,
        accepted=False,  # set in run_phase1 after the stratified decision
        metadata={
            "gold_answer": gold,
            "answer_score": answer_score,
            "coverage": coverage,
            "linked_entities": {m: q for m, q in linked.items() if q},
            "n_mentions": len(mentions),
            # Describes the STORED subgraph (== what the Teacher saw). `triples`
            # is pre-filter, so a question whose every triple was dropped by
            # hard-delete/quota used to be recorded kg_empty=False with an
            # empty kg_subgraph.
            "kg_empty": len(teacher_kg) == 0,
            "n_triples_prefilter": len(triples),
            "n_triples_teacher": len(teacher_kg),
            "format_retried": retry_attempted,
            "retry_succeeded": retry_succeeded,
            "citation_contract_errors": citation_contract_errors,
            "extra": cfg.extra_metadata or {},
        },
    )
    hard_reject_reason = ""
    if citation_contract_errors:
        hard_reject_reason = "citation_contract:" + "|".join(citation_contract_errors)
    return _Candidate(
        trajectory=traj,
        coverage=coverage,
        answer_score=answer_score,
        hard_reject_reason=hard_reject_reason,
    )


# ---------------------------------------------------------------------------
# Per-item candidate + serialised accept/write step. Generation can be
# parallel; the stratified accept decision and the write are serialised so the
# shared quota counter is race-free.
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    trajectory: SilverTrajectory
    coverage: float
    answer_score: float
    hard_reject_reason: str = ""


def phase1_candidate_path(output_path: str | Path) -> Path:
    """Raw quality-assessed sidecar retained before composition selection."""
    path = Path(output_path)
    return path.with_name(f"{path.stem}.candidates{path.suffix}")


def _assess_and_write_candidate(
    cand: "_Candidate", cfg: Phase1Config, fh
) -> Dict[str, Any]:
    """Persist intrinsic quality without applying dataset-composition quotas."""
    decision = cfg.accept_filter.assess_quality(
        steps=cand.trajectory.steps,
        answer_score=cand.answer_score,
        hard_reject_reason=cand.hard_reject_reason,
    )
    cand.trajectory.accepted = False
    cand.trajectory.metadata["bucket"] = decision.bucket
    cand.trajectory.metadata["kg_bucket"] = (
        cfg.accept_filter._bucket_for(decision.triple_rate)
    )
    cand.trajectory.metadata["triple_rate"] = decision.triple_rate
    cand.trajectory.metadata["quality_pass"] = decision.accepted
    cand.trajectory.metadata["quality_reject_reason"] = (
        "" if decision.accepted else decision.reason
    )
    cand.trajectory.metadata["selection_pass"] = False
    cand.trajectory.metadata["selection_reason"] = "pending_post_generation_selection"
    cand.trajectory.metadata["reject_reason"] = (
        "" if decision.accepted else decision.reason
    )
    fh.write(json.dumps(cand.trajectory.to_dict(), ensure_ascii=False) + "\n")
    fh.flush()
    return {"quality_pass": int(decision.accepted), "bucket": decision.bucket}


def _quota_selection_counts(
    n_rich: int,
    n_medium: int,
    n_sparse: int,
    medium_quota: float,
    sparse_quota: float,
) -> Dict[str, int]:
    """Maximise selected rows while enforcing exact post-hoc bucket caps."""
    available_total = n_rich + n_medium + n_sparse
    for total in range(available_total, n_rich - 1, -1):
        max_medium = min(n_medium, int(medium_quota * total), total - n_rich)
        min_medium = max(
            0,
            total - n_rich - min(n_sparse, int(sparse_quota * total)),
        )
        if min_medium <= max_medium:
            # Prefer medium over sparse at an otherwise identical total: it is
            # the stronger KG-supervision stratum.
            selected_medium = max_medium
            selected_sparse = total - n_rich - selected_medium
            if selected_sparse <= min(n_sparse, int(sparse_quota * total)):
                return {
                    "kg_rich": n_rich,
                    "kg_medium": selected_medium,
                    "kg_sparse": selected_sparse,
                }
    return {"kg_rich": n_rich, "kg_medium": 0, "kg_sparse": 0}


def _selection_rank(row: Dict[str, Any], seed: int, lineno: int) -> str:
    identity = "\0".join(
        (
            str(seed),
            str(row.get("dataset") or ""),
            str(row.get("qid") or row.get("id") or ""),
            str(lineno),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _finalize_stratified_candidates(
    candidate_path: Path,
    output_path: Path,
    accept_filter: StratifiedSilverFilter,
    seed: int,
) -> Dict[str, Any]:
    """Create the selected dataset from the immutable candidate sidecar."""
    eligible: Dict[str, List[tuple[str, int]]] = {
        "kg_rich": [],
        "kg_medium": [],
        "kg_sparse": [],
    }
    bucket_counts: Dict[str, int] = {}
    written = 0
    with candidate_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            written += 1
            metadata = row.get("metadata") or {}
            bucket = str(metadata.get("bucket") or "UNKNOWN")
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            if not metadata.get("quality_pass"):
                continue
            kg_bucket = str(metadata.get("kg_bucket") or "")
            if kg_bucket not in eligible:
                raise ValueError(f"{candidate_path}:{lineno}: invalid kg_bucket={kg_bucket!r}")
            eligible[kg_bucket].append((_selection_rank(row, seed, lineno), lineno))

    selected_counts = _quota_selection_counts(
        len(eligible["kg_rich"]),
        len(eligible["kg_medium"]),
        len(eligible["kg_sparse"]),
        accept_filter.medium_quota,
        accept_filter.sparse_quota,
    )
    selected_lines: set[int] = set()
    for bucket, rows in eligible.items():
        rows.sort()
        selected_lines.update(lineno for _, lineno in rows[: selected_counts[bucket]])

    tmp_path = output_path.with_name(f".{output_path.name}.selecting")
    with candidate_path.open(encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for lineno, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.setdefault("metadata", {})
            quality_pass = bool(metadata.get("quality_pass"))
            selected = quality_pass and lineno in selected_lines
            row["accepted"] = selected
            metadata["selection_pass"] = selected
            if selected:
                metadata["selection_reason"] = "selected"
                metadata["reject_reason"] = ""
            elif quality_pass:
                reason = f"{metadata.get('kg_bucket', 'unknown')}_quota_full"
                metadata["selection_reason"] = reason
                metadata["reject_reason"] = reason
            else:
                metadata["selection_reason"] = "quality_rejected"
                metadata["reject_reason"] = metadata.get("quality_reject_reason") or "quality_rejected"
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, output_path)
    return {
        "written": written,
        "quality_passed": sum(len(rows) for rows in eligible.values()),
        "accepted": len(selected_lines),
        "bucket_counts": bucket_counts,
        "quality_bucket_counts": {key: len(value) for key, value in eligible.items()},
        "accepted_bucket_counts": selected_counts,
    }


def run_phase1(cfg: Phase1Config) -> Dict[str, Any]:
    """Generate silver trajectories for ``cfg.items`` and write to ``cfg.output_path``.

    Returns a small stats dict.
    """
    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path = phase1_candidate_path(out_path)
    if cfg.append_output and not candidate_path.exists() and out_path.exists():
        # Upgrade a completed new-format run into resumable form without
        # discarding the already-persisted records.
        shutil.copyfile(out_path, candidate_path)

    # Build a single retriever object once (heavy).
    retriever = cfg.retriever_factory() if callable(cfg.retriever_factory) else cfg.retriever_factory
    retrieval_call = _RetrievalAdapter(
        retriever, top_k=cfg.top_k,
        rerank_topk=cfg.rerank_topk,
        cross_encoder_model=cfg.cross_encoder_model,
        bridge_mode=cfg.bridge_mode,
        bridge_first_round_topk=cfg.bridge_first_round_topk,
        bridge_max_queries=cfg.bridge_max_queries,
        bridge_only_k=cfg.bridge_only_k,
    )

    # Retrieve before Teacher concurrency so a flat/memmap dense index is
    # scanned in batches instead of once per item per worker. If batch retrieval
    # fails, preserve the old per-item path and log the degradation explicitly.
    items = [dict(item) for item in cfg.items]
    questions = [str(item.get("question", "")).strip() for item in items]
    try:
        prefetched = retrieval_call.batch(questions)
        if len(prefetched) != len(items):
            raise ValueError(
                f"batch retriever returned {len(prefetched)} rows for {len(items)} queries"
            )
        for item, passages in zip(items, prefetched):
            item["_retrieved_passages"] = passages
        logger.info("Phase 1 prefetched retrieval for %d items in batch mode.", len(items))
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("KGPW_REQUIRE_BATCH_RETRIEVAL", "").lower() in {
            "1", "true", "yes", "on",
        }:
            raise RuntimeError(
                "Phase 1 batch retrieval is required for this run; refusing "
                "the per-item full-index fallback"
            ) from exc
        logger.warning(
            "Phase 1 batch retrieval failed (%s); falling back to per-item retrieval.",
            exc,
        )

    total = 0
    written = 0
    bucket_counts: Dict[str, int] = {}
    write_mode = "a" if cfg.append_output else "w"
    with open(candidate_path, write_mode, encoding="utf-8") as fh:
        if cfg.max_workers <= 1:
            for item in items:
                total += 1
                cand = _process_one(item, cfg, retrieval_call)
                if cand is None:
                    continue
                res = _assess_and_write_candidate(cand, cfg, fh)
                written += 1
                bucket_counts[res["bucket"]] = bucket_counts.get(res["bucket"], 0) + 1
        else:
            # Teacher/SPARQL calls run in parallel; decide+write stays serial.
            with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
                futures = [ex.submit(_process_one, item, cfg, retrieval_call) for item in items]
                # Consume in INPUT order.  Worker calls still execute in
                # parallel, but quota-based acceptance must not depend on API
                # response timing: as_completed() made the accepted subset and
                # output ordering change across otherwise identical seeded runs.
                for fut in futures:
                    total += 1
                    try:
                        cand = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Teacher worker failed: %s", exc)
                        continue
                    if cand is None:
                        continue
                    res = _assess_and_write_candidate(cand, cfg, fh)
                    written += 1
                    bucket_counts[res["bucket"]] = bucket_counts.get(res["bucket"], 0) + 1

    selection = _finalize_stratified_candidates(
        candidate_path,
        out_path,
        cfg.accept_filter,
        cfg.seed,
    )
    accepted = int(selection["accepted"])
    dump_manifest(
        out_path.parent / f"{out_path.stem}_run",
        extra={
            "phase": "phase1_distill",
            "dataset": cfg.dataset_name,
            "teacher_model": cfg.teacher_client.model,
            "teacher_temperature": cfg.teacher_temperature,
            "seed": cfg.seed,
            "retrieval": {
                "rrf_candidate_topk": cfg.top_k,
                "rerank_topk": cfg.rerank_topk,
                "cross_encoder_model": cfg.cross_encoder_model,
                "bridge_mode": cfg.bridge_mode,
                "bridge_first_round_topk": cfg.bridge_first_round_topk,
                "bridge_max_queries": cfg.bridge_max_queries,
                "bridge_only_k": cfg.bridge_only_k,
            },
            "kg_budget": {
                "max_kg_triples": cfg.max_kg_triples,
                "min_kg_keep": cfg.min_kg_keep,
            },
            "extra_metadata": cfg.extra_metadata or {},
            "output_path": str(out_path),
            "candidate_path": str(candidate_path),
            "total_attempts": total,
            "written_this_invocation": written,
            "written": selection["written"],
            "quality_passed": selection["quality_passed"],
            "accepted": accepted,
            "bucket_counts_this_invocation": bucket_counts,
            "bucket_counts": selection["bucket_counts"],
            "quality_bucket_counts": selection["quality_bucket_counts"],
            "accepted_bucket_counts": selection["accepted_bucket_counts"],
            "accept_filter": vars(cfg.accept_filter),
        },
    )
    logger.info(
        "Phase 1 finished: wrote %d trajectories this invocation (candidates=%d quality=%d accepted=%d / attempts=%d) buckets=%s to %s",
        written,
        selection["written"],
        selection["quality_passed"],
        accepted,
        total,
        selection["accepted_bucket_counts"],
        out_path,
    )
    return {
        "total": total,
        "written": selection["written"],
        "written_this_invocation": written,
        "quality_passed": selection["quality_passed"],
        "accepted": accepted,
        "bucket_counts": selection["bucket_counts"],
        "accepted_bucket_counts": selection["accepted_bucket_counts"],
        "output": str(out_path),
        "candidates": str(candidate_path),
    }


# ---------------------------------------------------------------------------
# Optional helper: build everything from a YAML
# ---------------------------------------------------------------------------

def build_components_from_config(cfg) -> Dict[str, Any]:
    """Convenience builder used by the CLI wrapper.

    Returns a dict with ``entity_linker``, ``kg_retriever``, ``annotator``.
    Caller still supplies items and retriever_factory.
    """
    from kgproweight.retrieval.bootstrap import resolve_entity_cache_path, resolve_kg_cache_dir

    entity_cache = cfg.get("entity_cache_path") or resolve_entity_cache_path()
    kg_cache_dir = cfg.get("kg_cache_dir") or resolve_kg_cache_dir()

    linker = EntityLinker(cache_path=entity_cache)
    retriever = WikidataSubgraphRetriever(
        max_hops=cfg.get("kg_max_hops", 2),
        max_neighbors=cfg.get("kg_max_neighbors", 30),
        cache_dir=kg_cache_dir,
    )
    annotator = PRMAnnotator(entity_linker=linker, verbose=False)
    return {
        "entity_linker": linker,
        "kg_retriever": retriever,
        "annotator": annotator,
        "data_dir": str(data_dir()),
        "index_dir": str(index_dir()),
    }
