"""CPU-only tests for the v9 canonical subquestion-answer interface."""

from __future__ import annotations

import pytest

from kgproweight.retrieval.canonical_subqa_v9 import (
    build_canonical_subqa_messages,
    parse_and_bind_canonical_subanswer,
)
from scripts.pilot.run_canonical_subqa_v9_phase0 import summarize


SUBQUESTION = "Who directed Film Alpha?"


def _passages(*, answer_in: tuple[int, ...] = (3,)) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for rank in range(1, 11):
        body = f"Ordinary evidence in document {rank}."
        if rank in answer_in:
            body = "Film Alpha was directed by Jane Smith in 1998."
        result.append(
            {
                "doc_id": f"doc-{rank}",
                "title": f"Title {rank}",
                "text": body,
            }
        )
    return result


def test_prompt_reuses_canonical_sft_schema_with_empty_kg() -> None:
    messages = build_canonical_subqa_messages(
        subquestion=SUBQUESTION,
        retrieved_passages=_passages(),
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "[Step 1]" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert f"Question: {SUBQUESTION}" in messages[1]["content"]
    assert "[Knowledge Graph Context]\n  (empty)" in messages[1]["content"]
    assert "Film Alpha was directed by Jane Smith" in messages[1]["content"]


def test_full_trace_final_answer_is_bound_by_unchanged_v8_binder() -> None:
    generation = """[Step 1]
Reasoning: The passage names the director.
Knowledge Used: []
Conclusion: Jane Smith directed Film Alpha.

[Final Answer]
Jane Smith"""
    result = parse_and_bind_canonical_subanswer(
        generation,
        subquestion=SUBQUESTION,
        retrieved_passages=_passages(),
    )
    assert result["final_answer_parsed"] is True
    assert result["final_answer"] == "Jane Smith"
    assert result["parsed_step_count"] == 1
    assert result["binding"]["verified"] is True
    assert result["binding"]["supporting_doc_id"] == "doc-3"


def test_missing_final_answer_fails_closed() -> None:
    result = parse_and_bind_canonical_subanswer(
        "[Step 1]\nReasoning: Jane Smith.\nKnowledge Used: []\nConclusion: Jane Smith.",
        subquestion=SUBQUESTION,
        retrieved_passages=_passages(),
    )
    assert result["final_answer_parsed"] is False
    assert result["binding"]["verified"] is False
    assert result["binding"]["reason"] == "parse_error:empty_response"


def test_ambiguous_surface_remains_rejected() -> None:
    result = parse_and_bind_canonical_subanswer(
        "[Final Answer]\nJane Smith",
        subquestion=SUBQUESTION,
        retrieved_passages=_passages(answer_in=(2, 3)),
    )
    assert result["final_answer_parsed"] is True
    assert result["binding"]["verified"] is False
    assert result["binding"]["reason"] == "answer_surface_ambiguous_across_documents"


def test_prompt_requires_exactly_ten_safe_passages() -> None:
    with pytest.raises(ValueError, match="exactly 10"):
        build_canonical_subqa_messages(
            subquestion=SUBQUESTION,
            retrieved_passages=_passages()[:9],
        )


def _summary_row(dataset: str, *, parsed: bool, trace: bool, verified: bool) -> dict:
    return {
        "dataset": dataset,
        "old_binding_verified": False,
        "canonical_subanswer": {
            "final_answer_parsed": parsed,
            "has_step_trace": trace,
            "binding": {
                "verified": verified,
                "reason": "verified_unique_document_surface" if verified else "no_match",
            },
        },
    }


def test_summary_applies_each_dataset_gates_not_pooled_average() -> None:
    rows = []
    for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
        rows.extend(
            _summary_row(dataset, parsed=True, trace=True, verified=index < 12)
            for index in range(30)
        )
    summary = summarize(rows)
    assert summary["all_pass"] is True
    assert summary["by_dataset"]["hotpotqa"]["admissible_rate"] == 0.4

    rows[-1]["canonical_subanswer"]["final_answer_parsed"] = False
    rows[-2]["canonical_subanswer"]["final_answer_parsed"] = False
    # 28/30 = .933, so one failing dataset cannot be hidden by pooling.
    summary = summarize(rows)
    assert summary["gates"]["final_answer_parse_rate_min_each_dataset"] is False
    assert summary["all_pass"] is False
