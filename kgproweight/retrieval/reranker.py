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

from pathlib import Path
from typing import Any, Dict, List

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


# Cross-encoder models are ~600 MB and take seconds to load; cache per path so a
# per-question or per-batch rerank does not reload the weights every call.
_CE_CACHE: Dict[str, Any] = {}


def resolve_cross_encoder_path(model_name: str) -> str:
    """Resolve a cross-encoder to a local dir when available, else an HF id."""
    if Path(model_name).exists():
        return str(Path(model_name))
    from kgproweight.utils.paths import project_root

    local = Path(project_root()) / "models" / model_name.split("/")[-1]
    if local.exists():
        return str(local)
    return model_name  # let sentence-transformers download it


def get_cross_encoder(model_name: str):
    """Load (and memoise) a sentence-transformers CrossEncoder."""
    path = resolve_cross_encoder_path(model_name)
    if path not in _CE_CACHE:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder reranker from %s", path)
        _CE_CACHE[path] = CrossEncoder(path)
    return _CE_CACHE[path]


def rerank_with_cross_encoder(
    questions: List[str],
    candidates: List[List[Dict[str, Any]]],
    topk: int = 10,
    model_name: str = "models/bge-reranker-v2-m3",
    max_chars: int = 1200,
) -> List[List[Dict[str, Any]]]:
    """Cross-encoder rerank. Falls back to BM25 if the model cannot load."""
    try:
        model = get_cross_encoder(model_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Cross-encoder %s unavailable (%s) — falling back to the BM25 reranker. "
            "Results will NOT match a cross-encoder run; note this in any comparison.",
            model_name, exc,
        )
        return rerank_with_bm25(questions, candidates, topk=topk)

    results: List[List[Dict[str, Any]]] = []
    for q, cands in zip(questions, candidates):
        if not cands:
            results.append([])
            continue
        pairs = [(q, _passage_text(c)[:max_chars]) for c in cands]
        scores = model.predict(pairs, show_progress_bar=False)
        order = sorted(range(len(cands)), key=lambda i: float(scores[i]), reverse=True)
        results.append([cands[i] for i in order[:topk]])
    return results


def rerank_passages(
    questions: List[str],
    candidates: List[List[Dict[str, Any]]],
    topk: int = 10,
    method: str = "cross-encoder",
    cross_encoder_model: str = "models/bge-reranker-v2-m3",
) -> List[List[Dict[str, Any]]]:
    """Single entry point for reranking, so every call site agrees on the method.

    Phase 1 used a cross-encoder while inference used the hand-rolled BM25
    scorer, which silently made the silver-data and eval passage distributions
    different.
    """
    if method == "bm25":
        return rerank_with_bm25(questions, candidates, topk=topk)
    if method in ("cross-encoder", "cross_encoder", "ce"):
        return rerank_with_cross_encoder(
            questions, candidates, topk=topk, model_name=cross_encoder_model,
        )
    logger.warning("Unknown rerank method %r — truncating to top-%d instead.", method, topk)
    return [c[:topk] for c in candidates]


class RetrievalConfig:
    """Two-stage retrieval configuration.

    All top-k values are independent — no single value controls multiple stages.
    """

    def __init__(
        self,
        dense_candidate_topk: int = 100,
        sparse_candidate_topk: int = 100,
        rrf_k: int = 60,
        rrf_candidate_topk: int = 50,
        rerank_topk: int = 10,
        prompt_passage_token_budget: int = 3860,
        rerank_method: str = "cross-encoder",
        cross_encoder_model: str = "models/bge-reranker-v2-m3",
    ):
        self.dense_candidate_topk = dense_candidate_topk
        self.sparse_candidate_topk = sparse_candidate_topk
        self.rrf_k = rrf_k
        self.rrf_candidate_topk = rrf_candidate_topk
        self.rerank_topk = rerank_topk
        self.prompt_passage_token_budget = prompt_passage_token_budget
        self.rerank_method = rerank_method
        self.cross_encoder_model = cross_encoder_model

    def log_string(self) -> str:
        return (
            f"dense@{self.dense_candidate_topk} + sparse@{self.sparse_candidate_topk} "
            f"→ RRF(k={self.rrf_k})@{self.rrf_candidate_topk} "
            f"→ {self.rerank_method}@{self.rerank_topk} "
            f"→ prompt≤{self.prompt_passage_token_budget}tok"
        )


class RRFRerankRetriever:
    """Two-stage retriever wrapping FlashRAG dense + sparse retrievers.

    Stage 1: dense@100 + sparse@100 → RRF merge → top-50 candidates
    Stage 2: reranker (BM25 or cross-encoder) → top-10 for prompt
    """

    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        config: RetrievalConfig,
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.config = config

    def batch_search(self, questions: List[str]) -> List[List[Dict[str, Any]]]:
        """Two-stage search: dense+sparse → RRF → rerank."""
        # Stage 1: dense + sparse retrieval
        dense_results = self.dense.batch_search(questions)
        sparse_results = self.sparse.batch_search(questions)

        # RRF merge (handles dedup + score accumulation)
        all_candidates = []
        for d_res, s_res in zip(dense_results, sparse_results):
            # Simple RRF: combine with source tracking, dedup by id
            id2doc = {}
            scores: Dict[str, float] = {}
            k = self.config.rrf_k

            for rank, doc in enumerate(d_res[:self.config.dense_candidate_topk], start=1):
                did = doc.get("id", str(rank))
                id2doc[did] = doc
                scores[did] = scores.get(did, 0) + 1.0 / (k + rank)

            for rank, doc in enumerate(s_res[:self.config.sparse_candidate_topk], start=1):
                did = doc.get("id", str(rank))
                if did not in id2doc:
                    id2doc[did] = doc
                scores[did] = scores.get(did, 0) + 1.0 / (k + rank)

            # Sort by RRF score, take top candidates
            sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
            candidate_ids = sorted_ids[:self.config.rrf_candidate_topk]
            candidates = [id2doc[did] for did in candidate_ids]
            all_candidates.append(candidates)

        # Stage 2: rerank
        if self.config.rerank_method == "bm25":
            return rerank_with_bm25(questions, all_candidates, topk=self.config.rerank_topk)

        # Cross-encoder rerank
        if self.config.rerank_method == "cross-encoder":
            return self._cross_encoder_rerank(questions, all_candidates)

        logger.warning("Unknown rerank method '%s', falling back to top-K truncation",
                       self.config.rerank_method)
        return [c[:self.config.rerank_topk] for c in all_candidates]

    def _cross_encoder_rerank(
        self, questions: List[str], candidates: List[List[Dict[str, Any]]],
    ) -> List[List[Dict[str, Any]]]:
        """Rerank candidates using a cross-encoder model."""
        return rerank_with_cross_encoder(
            questions, candidates,
            topk=self.config.rerank_topk,
            model_name=self.config.cross_encoder_model,
        )


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
