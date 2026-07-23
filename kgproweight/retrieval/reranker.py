"""Two-stage retrieval: RRF candidate pool → reranker → top-K prompt.

Architecture:
  dense top-100 + sparse top-100 → RRF merge → top-50 candidates
  → Cross-encoder (or BM25 fallback) rerank → top-10/15 prompt

Config fields (set in hybrid.py or YAML):
  dense_candidate_topk: 100
  sparse_candidate_topk: 100
  rrf_candidate_topk: 50
  rerank_topk: 10
  prompt_passage_token_budget: 3860
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from kgproweight.utils.logging import get_logger

logger = get_logger(__name__)


def rerank_with_bm25(
    questions: List[str],
    candidates: List[List[Dict[str, Any]]],
    topk: int = 10,
) -> List[List[Dict[str, Any]]]:
    """Lightweight BM25-based reranker. Zero external dependencies.

    Uses the question as a query against candidate passage texts via
    a simple BM25-like term frequency scoring.
    """
    import math
    from collections import defaultdict

    results: List[List[Dict[str, Any]]] = []

    for q, cands in zip(questions, candidates):
        if not cands:
            results.append([])
            continue

        q_terms = q.lower().split()

        # Simple TF-IDF-like scoring
        N = len(cands)
        df: Dict[str, int] = defaultdict(int)
        for c in cands:
            text = _passage_text(c).lower()
            for t in set(q_terms):
                if t in text:
                    df[t] += 1

        scored = []
        for c in cands:
            text = _passage_text(c).lower()
            score = 0.0
            for t in q_terms:
                if t not in text:
                    continue
                tf = text.count(t) / max(1, len(text.split()))
                idf = math.log((N + 1) / (df.get(t, 0) + 1)) + 1
                score += tf * idf
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        results.append([c for _, c in scored[:topk]])

    return results


def _passage_text(passage: Dict[str, Any]) -> str:
    """Extract text from a passage dict."""
    return passage.get("contents", "") or passage.get("text", "") or ""


class RetrievalConfig:
    """Two-stage retrieval configuration."""

    def __init__(
        self,
        dense_candidate_topk: int = 100,
        sparse_candidate_topk: int = 100,
        rrf_candidate_topk: int = 50,
        rerank_topk: int = 10,
        prompt_passage_token_budget: int = 3860,
        rerank_method: str = "bm25",  # or "cross-encoder"
        cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
    ):
        self.dense_candidate_topk = dense_candidate_topk
        self.sparse_candidate_topk = sparse_candidate_topk
        self.rrf_candidate_topk = rrf_candidate_topk
        self.rerank_topk = rerank_topk
        self.prompt_passage_token_budget = prompt_passage_token_budget
        self.rerank_method = rerank_method
        self.cross_encoder_model = cross_encoder_model


def pack_passages_by_token_budget(
    passages: List[Dict[str, Any]],
    max_tokens: int,
    chars_per_token: int = 4,
) -> List[Dict[str, Any]]:
    """Pack top passages into prompt budget by token count.

    Passages are assumed to be pre-sorted by relevance (reranker output).
    Each passage is truncated to 1200 chars (~300 tokens).
    """
    budget = 0
    packed = []
    for p in passages:
        text = _passage_text(p)[:1200]
        tokens = len(text) // chars_per_token
        if budget + tokens > max_tokens:
            # Try to fit a truncated version
            remaining = max_tokens - budget
            if remaining > 50 * chars_per_token:  # at least 50 tokens = ~200 chars
                p_truncated = dict(p)
                p_truncated["contents"] = text[:remaining * chars_per_token] + " ..."
                packed.append(p_truncated)
            break
        budget += tokens
        packed.append(p)
    if not packed and passages:
        # Force at least 1 passage
        text = _passage_text(passages[0])[:max_tokens * chars_per_token]
        p = dict(passages[0])
        p["contents"] = text
        packed.append(p)
    return packed
