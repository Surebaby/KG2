"""CPU-only tests for canonical sub-QA v9.1 rank-first binding."""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from kgproweight.retrieval.canonical_subqa_v9_1 import (
    PROVENANCE_BINDER_VERSION,
    SELECTION_POLICY,
    VERIFICATION_SCOPE,
    bind_subanswer_rank_first,
    parse_and_bind_canonical_subanswer,
    project_rank_first_binding_for_runtime,
)
from kgproweight.retrieval.dynamic_decomposition_v8 import (
    DynamicDecompositionV8Error,
    parse_and_bind_subanswer,
)


SUBQUESTION = "Who directed Film Alpha?"


def _passages(*, answer_in: tuple[int, ...] = (3,)) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank in range(1, 11):
        text = f"Ordinary evidence in document {rank}."
        if rank in answer_in:
            text = f"Rank {rank} says Film Alpha was directed by Jane Smith in 1998."
        rows.append(
            {
                "doc_id": f"doc-{rank}",
                "title": f"Title {rank}",
                "text": text,
                "rerank_score": float(100 - rank),
            }
        )
    return rows


def test_multiple_matching_documents_select_minimum_input_rank() -> None:
    passages = _passages(answer_in=(2, 4, 9))
    result = bind_subanswer_rank_first(
        "Jane Smith",
        subquestion=SUBQUESTION,
        retrieved_passages=passages,
    )
    assert result["verified"] is True
    assert result["supporting_doc_id"] == "doc-2"
    assert result["supporting_doc_rank"] == 2
    assert result["matching_document_count"] == 3
    assert result["matching_unit_count"] == 3
    assert result["selected_document_matching_unit_count"] == 1
    assert result["selection_reason"] == (
        "minimum_rank_selected_from_multiple_surface_matching_documents"
    )


def test_input_rank_not_score_controls_selection() -> None:
    passages = _passages(answer_in=(2, 8))
    passages[1]["rerank_score"] = -999.0
    passages[7]["rerank_score"] = 999.0
    result = bind_subanswer_rank_first(
        "Jane Smith",
        subquestion=SUBQUESTION,
        retrieved_passages=passages,
    )
    assert result["supporting_doc_rank"] == 2
    assert result["selection_policy"] == SELECTION_POLICY


def test_selected_document_reuses_v8_unit_and_offset_logic() -> None:
    passages = _passages(answer_in=(2, 8))
    passages[1]["title"] = "Jane Smith"
    passages[1]["text"] = (
        "Jane Smith\nFilm Alpha was directed by Jane Smith. "
        "A later sentence also credits Jane Smith."
    )
    v91 = bind_subanswer_rank_first(
        "Jane Smith",
        subquestion=SUBQUESTION,
        retrieved_passages=passages,
    )

    unique = deepcopy(passages)
    unique[7]["text"] = "No director is named in this document."
    v8 = parse_and_bind_subanswer(
        "Jane Smith",
        q1_query=SUBQUESTION,
        q1_passages=unique,
    )
    for field in (
        "supporting_sentence",
        "support_location",
        "support_unit_index",
        "surface_start",
        "surface_end",
        "bound_evidence_excerpt",
        "bound_excerpt_surface_start",
        "bound_excerpt_surface_end",
        "supporting_document_prompt_sha256",
    ):
        assert v91[field] == v8[field]
    assert v91["selected_document_matching_unit_count"] == 3
    assert v91["matching_unit_count"] == 4


def test_unique_document_is_preserved_under_new_explicit_contract() -> None:
    result = bind_subanswer_rank_first(
        "Jane Smith",
        subquestion=SUBQUESTION,
        retrieved_passages=_passages(),
    )
    assert result["verified"] is True
    assert result["binder_version"] == PROVENANCE_BINDER_VERSION
    assert result["verification_scope"] == VERIFICATION_SCOPE
    assert result["selection_reason"] == "only_surface_matching_document"
    assert result["matching_document_count"] == 1


def test_runtime_projection_keeps_new_version_and_complete_provenance() -> None:
    binding = bind_subanswer_rank_first(
        "Jane Smith",
        subquestion=SUBQUESTION,
        retrieved_passages=_passages(answer_in=(2, 6)),
    )
    projection = project_rank_first_binding_for_runtime(binding)
    assert projection["binder_version"] == PROVENANCE_BINDER_VERSION
    assert projection["binder_version"] != (
        "dynamic-decomposition-v8-unique-doc-surface-binder-1"
    )
    assert projection["supporting_document_key"] == "id:doc-2"
    for field in (
        "supporting_doc_id",
        "supporting_doc_rank",
        "supporting_sentence",
        "supporting_sentence_sha256",
        "support_location",
        "support_unit_index",
        "bound_evidence_excerpt",
        "bound_evidence_excerpt_sha256",
        "supporting_document_prompt_sha256",
    ):
        assert projection[field] is not None


def test_runtime_projection_rejects_old_or_unverified_binding() -> None:
    old = parse_and_bind_subanswer(
        "Jane Smith", q1_query=SUBQUESTION, q1_passages=_passages()
    )
    with pytest.raises(DynamicDecompositionV8Error, match="unexpected binder version"):
        project_rank_first_binding_for_runtime(old)

    unverified = bind_subanswer_rank_first(
        "Nobody Here", subquestion=SUBQUESTION, retrieved_passages=_passages()
    )
    with pytest.raises(DynamicDecompositionV8Error, match="verified Gold-free"):
        project_rank_first_binding_for_runtime(unverified)


@pytest.mark.parametrize(
    ("answer", "subquestion", "reason"),
    [
        ("yes", SUBQUESTION, "non_extractive_boolean_answer"),
        ("unknown", SUBQUESTION, "null_like_answer"),
        ("Film Alpha", SUBQUESTION, "q1_surface_echo"),
        ("Nobody Here", SUBQUESTION, "answer_surface_not_found"),
    ],
)
def test_prior_fail_closed_cases_remain_fail_closed(answer, subquestion, reason) -> None:
    result = bind_subanswer_rank_first(
        answer,
        subquestion=subquestion,
        retrieved_passages=_passages(),
    )
    assert result["verified"] is False
    assert result["reason"] == reason
    assert result["selection_reason"] == f"fail_closed:{reason}"
    assert result["verification_scope"] == VERIFICATION_SCOPE


def test_canonical_final_answer_is_parsed_and_rank_first_bound() -> None:
    generation = """[Step 1]
Reasoning: The passages identify the director.
Knowledge Used: []
Conclusion: Jane Smith directed Film Alpha.

[Final Answer]
Jane Smith"""
    result = parse_and_bind_canonical_subanswer(
        generation,
        subquestion=SUBQUESTION,
        retrieved_passages=_passages(answer_in=(4, 7)),
    )
    assert result["gold_access"] is False
    assert result["final_answer_parsed"] is True
    assert result["final_answer"] == "Jane Smith"
    assert result["parsed_step_count"] == 1
    assert result["binding"]["verified"] is True
    assert result["binding"]["supporting_doc_rank"] == 4


def test_missing_final_answer_remains_fail_closed() -> None:
    result = parse_and_bind_canonical_subanswer(
        "[Step 1]\nReasoning: Jane Smith.\nKnowledge Used: []\nConclusion: Jane Smith.",
        subquestion=SUBQUESTION,
        retrieved_passages=_passages(),
    )
    assert result["final_answer_parsed"] is False
    assert result["binding"]["verified"] is False
    assert result["binding"]["reason"] == "parse_error:empty_response"


def test_gold_like_extra_fields_are_neither_read_nor_emitted() -> None:
    passages = _passages(answer_in=(2, 6))
    for passage in passages:
        passage["gold_answer"] = "DO-NOT-LEAK"
        passage["supporting_facts"] = ["DO-NOT-LEAK"]
    original = deepcopy(passages)
    result = bind_subanswer_rank_first(
        "Jane Smith",
        subquestion=SUBQUESTION,
        retrieved_passages=passages,
    )
    assert result["gold_access"] is False
    assert "DO-NOT-LEAK" not in json.dumps(result)
    assert passages == original


def test_unicode_and_numeric_boundary_behavior_stays_v8_owned() -> None:
    passages = _passages(answer_in=())
    passages[0]["text"] = "Bamboo Man\u0303alac fronted the group."
    passages[1]["text"] = "The measured value was 38.5."
    passages[2]["text"] = "The event happened in 1990."
    unicode_result = bind_subanswer_rank_first(
        "Bamboo Mañalac",
        subquestion="Who fronted the group?",
        retrieved_passages=passages,
    )
    decimal_subspan = bind_subanswer_rank_first(
        "38",
        subquestion="What was the integer value?",
        retrieved_passages=passages,
    )
    year = bind_subanswer_rank_first(
        "1990",
        subquestion="When did the event happen?",
        retrieved_passages=passages,
    )
    assert unicode_result["verified"] is True
    assert decimal_subspan["reason"] == "answer_surface_not_found"
    assert year["verified"] is True


def test_binding_is_deterministic_and_does_not_mutate_inputs() -> None:
    passages = _passages(answer_in=(2, 3, 10))
    original = deepcopy(passages)
    first = bind_subanswer_rank_first(
        "Jane Smith", subquestion=SUBQUESTION, retrieved_passages=passages
    )
    second = bind_subanswer_rank_first(
        "Jane Smith", subquestion=SUBQUESTION, retrieved_passages=passages
    )
    assert first == second
    assert passages == original


def test_broken_top10_and_duplicate_identity_remain_caller_errors() -> None:
    with pytest.raises(DynamicDecompositionV8Error, match="exactly 10"):
        bind_subanswer_rank_first(
            "Jane Smith",
            subquestion=SUBQUESTION,
            retrieved_passages=_passages()[:9],
        )
    duplicate = _passages()
    duplicate[1]["doc_id"] = duplicate[0]["doc_id"]
    with pytest.raises(DynamicDecompositionV8Error, match="duplicate document"):
        bind_subanswer_rank_first(
            "Jane Smith",
            subquestion=SUBQUESTION,
            retrieved_passages=duplicate,
        )
