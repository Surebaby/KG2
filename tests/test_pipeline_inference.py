"""Tests for KG-ProWeight inference pipeline and answer extraction."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kgproweight.data.parsers import extract_final_answer
from kgproweight.data.prompts import build_inference_messages
from kgproweight.eval.pred_processing import extract_kg_proweight_answer, kg_proweight_pred_process
from kgproweight.retrieval.hybrid import DEFAULT_TOPK


_SAMPLE_TRACE = """\
[Step 1]
Reasoning: Both directors are American.
Knowledge Used: [(Scott Derrickson, country, United States)]
Conclusion: Scott Derrickson is American.

[Final Answer]
yes
"""


def test_extract_kg_proweight_answer_from_schema():
    assert extract_kg_proweight_answer(_SAMPLE_TRACE) == "yes"


def test_extract_final_answer_not_reasoning_pipeline_tag():
    raw = " long reasoning \n\n<answer>no</answer>"
    assert extract_final_answer(raw) is None
    assert extract_kg_proweight_answer(raw) == raw.strip()


def test_kg_proweight_pred_process_dataset():
    item = MagicMock()
    item.pred = _SAMPLE_TRACE
    kg_proweight_pred_process([item])
    assert item.pred == "yes"


def test_build_inference_messages_uses_default_top_k():
    msgs = build_inference_messages(
        question="Who?",
        retrieved_passages=[{"contents": "Title\nBody text"}],
        kg_triples=[("A", "rel", "B")],
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "[Step 1]" in msgs[0]["content"]
    assert "Who?" in msgs[1]["content"]
    assert "(A, rel, B)" in msgs[1]["content"]


def test_default_top_k_is_prompt_budget_not_candidate_pool():
    """``DEFAULT_TOPK`` is the PROMPT passage count, not the candidate pool.

    Two-stage retrieval split the old single "50" into independent stages, so 50
    now lives in ``DEFAULT_RRF_CANDIDATE_TOPK`` while ``DEFAULT_TOPK`` is the
    number of passages that reach the prompt. Asserting ``DEFAULT_TOPK == 50``
    would put ~5.9k tokens of passages in a 6144-token budget and squeeze out
    the KG block.
    """
    from kgproweight.retrieval.hybrid import (
        DEFAULT_DENSE_CANDIDATE_TOPK,
        DEFAULT_RERANK_TOPK,
        DEFAULT_RRF_CANDIDATE_TOPK,
    )

    assert DEFAULT_RRF_CANDIDATE_TOPK == 50
    assert DEFAULT_DENSE_CANDIDATE_TOPK == 100
    assert DEFAULT_RERANK_TOPK <= DEFAULT_TOPK <= DEFAULT_RRF_CANDIDATE_TOPK


def test_naive_rag_prompt_includes_reference_placeholder():
    from kgproweight.eval.baselines import BASELINES

    naive = next(b for b in BASELINES if b.name == "naive_rag")
    assert "{reference}" in naive.user_prompt
    # `trace` runs FlashRAG's IRCOTPipeline, which builds its own prompts and so
    # carries no user_prompt template — it used to be a plain prompted baseline.
    trace = next(b for b in BASELINES if b.name == "trace")
    assert trace.pipeline_class == "IRCOTPipeline"
    assert trace.user_prompt is None


def test_extracted_answer_matches_gold_for_em():
    """Regression: [Final Answer] extraction must yield matchable short answers."""
    parsed = extract_kg_proweight_answer(_SAMPLE_TRACE)
    gold = "yes"
    assert parsed == gold
    assert parsed.lower().strip() == gold.lower().strip()


def test_build_flashrag_config_includes_input_tokens_metric():
    from kgproweight.retrieval.hybrid import build_flashrag_config

    cfg = build_flashrag_config("hotpotqa", "t", "/tmp/out")
    assert "input_tokens" in cfg["metrics"]

