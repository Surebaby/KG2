"""Hybrid RRF top-50 retrieval setting (paper §5.1).

This is the canonical retrieval configuration used by:

- Phase 1 silver-data generation (Teacher prompt context).
- Every baseline in ``scripts/eval/run_baselines.py``.
- The KG-ProWeight inference pipeline.

The function returns a FlashRAG-ready config dict; callers either pass it
to ``flashrag.config.Config`` directly or merge it into their per-dataset
config.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from kgproweight.retrieval.bootstrap import (
    resolve_bm25_index_path,
    resolve_corpus_path,
    resolve_dense_index_path,
)
from kgproweight.utils.paths import data_dir, model_path

# Two-stage retrieval architecture (R9 v6).
# Stage 1: dense + sparse → RRF → candidate pool
# Stage 2: reranker → top-K → token-budgeted prompt
DEFAULT_DENSE_CANDIDATE_TOPK = 100
DEFAULT_SPARSE_CANDIDATE_TOPK = 100
DEFAULT_RRF_CANDIDATE_TOPK = 50
DEFAULT_RERANK_TOPK = 10
DEFAULT_PROMPT_TOKEN_BUDGET = 3860

# Legacy: single retrieval_topk (used when two-stage is disabled)
DEFAULT_TOPK = 15
DEFAULT_RRF_K = 60
DEFAULT_PER_RETRIEVER_TOPK = 100

# Prompt budget reserved for non-passage content.
PROMPT_RESERVED_TOKENS = (
    700   # instruction + question
    + 1200  # KG block (30 triples × ~40 chars/triple ÷ 4 chars/token)
    + 384   # generation budget
)

# Shared eval token budget (paper Appendix A / FlashRAG defaults).
EVAL_GENERATOR_MAX_INPUT_LEN = 4096
EVAL_GENERATION_MAX_TOKENS = 512
EVAL_RETRIEVAL_QUERY_MAX_LENGTH = 128


def build_rrf_setting(
    topk: int = DEFAULT_TOPK,
    rrf_k: int = DEFAULT_RRF_K,
    per_retriever_topk: int = DEFAULT_PER_RETRIEVER_TOPK,
    dense_index_path: Optional[str] = None,
    sparse_index_path: Optional[str] = None,
    dense_model_path: Optional[str] = None,
    corpus_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the ``multi_retriever_setting`` dict for FlashRAG."""
    resolved_corpus_path = corpus_path or resolve_corpus_path()
    return {
        "merge_method": "rrf",
        "rrf_k": rrf_k,
        "topk": topk,
        "retriever_list": [
            {
                "retrieval_method": "e5",
                "retrieval_model_path": dense_model_path or model_path("e5"),
                "index_path": dense_index_path or resolve_dense_index_path(),
                "corpus_path": resolved_corpus_path,
                "retrieval_topk": per_retriever_topk,
            },
            {
                "retrieval_method": "bm25",
                "index_path": sparse_index_path or resolve_bm25_index_path(),
                "corpus_path": resolved_corpus_path,
                "bm25_backend": "bm25s",
                "retrieval_topk": per_retriever_topk,
            },
        ],
    }


def apply_retrieval_overrides(cfg: Dict[str, Any], retrieval: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a ``retrieval:`` block from YAML onto a FlashRAG config dict."""
    if not retrieval:
        return cfg
    use_multi = retrieval.get("use_multi_retriever", True)
    topk = retrieval.get("retrieval_topk", DEFAULT_TOPK)
    cfg["retrieval_topk"] = topk
    cfg["use_multi_retriever"] = use_multi
    if use_multi:
        cfg["multi_retriever_setting"] = build_rrf_setting(
            topk=topk,
            rrf_k=retrieval.get("rrf_k", DEFAULT_RRF_K),
            corpus_path=cfg.get("corpus_path"),
        )
    else:
        cfg["retrieval_method"] = retrieval.get("dense_model", "e5")
        cfg["index_path"] = resolve_dense_index_path()
    return cfg


def build_flashrag_config(
    dataset_name: str,
    save_note: str,
    save_dir: str,
    *,
    method_name: str = "kg_proweight",
    pipeline_class: str = "KGProWeightPipeline",
    generator_model: str = "llama3-8B-instruct",
    framework: str = "hf",
    topk: int = DEFAULT_TOPK,
    rrf_k: int = DEFAULT_RRF_K,
    use_multi_retriever: bool = True,
    corpus_path: Optional[str] = None,
    split: str = "dev",
    test_sample_num: Optional[int] = None,
    seed: int = 42,
    gpu_id: str = "0",
    is_reasoning: bool = True,
    max_retrieval_num: int = 5,
    generator_lora_path: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a complete FlashRAG ``config_dict`` baked with RRF top-50."""
    cfg: Dict[str, Any] = {
        "dataset_name": dataset_name,
        # FlashRAG's Config overwrites dataset_path as os.path.join(data_dir, dataset_name),
        # so we must set data_dir to keep local project datasets effective.
        "data_dir": str(data_dir()),
        "split": [split],
        "save_dir": save_dir,
        "save_note": save_note,
        "corpus_path": corpus_path or resolve_corpus_path(),
        # Models
        "model2path": {
            "e5": model_path("e5"),
            "llama3-8B-instruct": model_path("llama3-8B-instruct"),
            "rearag": model_path("rearag"),
            "r1-searcher": model_path("r1-searcher"),
            "selfrag": model_path("selfrag"),
            "corag": model_path("corag"),
        },
        "model2pooling": {"e5": "mean", "bge": "cls", "contriever": "mean"},
        # Default single retriever (used when use_multi_retriever=False)
        "retrieval_method": "e5",
        "index_path": resolve_dense_index_path(),
        "retrieval_topk": topk,
        "retrieval_batch_size": 256,
        "retrieval_use_fp16": True,
        "retrieval_query_max_length": EVAL_RETRIEVAL_QUERY_MAX_LENGTH,
        "pooling_method": "mean",
        "bm25_index_path": resolve_bm25_index_path(),
        "bm25_backend": "bm25s",
        # RRF
        "use_multi_retriever": use_multi_retriever,
        # Generator
        "framework": framework,
        "generator_model": generator_model,
        "generator_max_input_len": EVAL_GENERATOR_MAX_INPUT_LEN,
        "generator_batch_size": 1,
        "generation_params": {
            "max_tokens": EVAL_GENERATION_MAX_TOKENS,
            "temperature": 0.0,
            "do_sample": False,
        },
        "gpu_memory_utilization": 0.80,
        # Metrics
        "metrics": ["em", "f1", "input_tokens"],
        "metric_setting": {"retrieval_recall_topk": 5, "tokenizer_name": "gpt-4"},
        # IO
        "save_intermediate_data": True,
        "save_metric_score": True,
        # Reproducibility
        "seed": seed,
        "gpu_id": gpu_id,
        "random_sample": False,
        "test_sample_num": test_sample_num,
        # Reasoning
        "is_reasoning": is_reasoning,
        "max_retrieval_num": max_retrieval_num,
        # KG-ProWeight bookkeeping
        "method_name": method_name,
        "pipeline_class": pipeline_class,
    }
    if use_multi_retriever:
        cfg["multi_retriever_setting"] = build_rrf_setting(
            topk=topk,
            rrf_k=rrf_k,
            corpus_path=cfg["corpus_path"],
        )
    if generator_lora_path:
        cfg["generator_lora_path"] = generator_lora_path
    if extra:
        cfg.update(extra)
    return cfg
