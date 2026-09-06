"""Unit tests for the isolated v6 query-hint and rendering policy."""

from copy import deepcopy
import json

import pytest

from kgproweight.retrieval.dependent import DependentRetrievalError
import kgproweight.retrieval.dependent_v6 as v6


def _v5_trace():
    accepted = [
        {
            "surface": "Accepted Person",
            "normalized_surface": "accepted person",
            "score": 7,
            "provenance": [{"document_key": "id:1", "rank": 1}],
            "admission": {"tier": 3, "basis": "relation_local_type_compatible"},
        }
    ]
    telemetry = {
        "selector_version": "v5-test",
        "accepted_count": 1,
        "candidate_decisions": [
            {
                "raw_rank": 1,
                "surface": "Accepted Person",
                "normalized_surface": "accepted person",
                "base_score": 7,
                "base_provenance": [{"document_key": "id:1", "rank": 1}],
                "decision": "accept",
                "admission_basis": "relation_local_type_compatible",
                "reasons": [],
            },
            {
                "raw_rank": 2,
                "surface": "Soft Risk Work",
                "normalized_surface": "soft risk work",
                "base_score": 6,
                "base_provenance": [{"document_key": "id:2", "rank": 2}],
                "decision": "reject",
                "admission_basis": "rejected",
                "reasons": [
                    "high_confidence_type_conflict",
                    "original_question_phrase",
                ],
            },
            {
                "raw_rank": 3,
                "surface": "A A",
                "normalized_surface": "a a",
                "base_score": 5,
                "base_provenance": [],
                "decision": "reject",
                "admission_basis": "rejected",
                "reasons": ["repeated_fragment"],
            },
        ],
    }
    return accepted, telemetry


def _select(monkeypatch, accepted, telemetry):
    def fake_v5(**kwargs):
        assert kwargs["question"] == "Original question?"
        assert kwargs["max_candidates"] == 2
        return deepcopy(accepted), deepcopy(telemetry)

    monkeypatch.setattr(v6, "select_bridge_candidates_v5", fake_v5)
    return v6.select_bridge_query_hints_v6(
        step={"subject": "Root", "relation_label": "author"},
        consumers=[{"subject": "$hop_1", "relation_label": "place of birth"}],
        target_type="relation_graph",
        query="Root author",
        question="Original question?",
        passages=[{"id": "1", "contents": "Root\nbody"}],
    )


def test_v5_accepted_is_first_then_soft_risk_raw_rank_fills(monkeypatch):
    accepted, telemetry = _v5_trace()
    hints, trace = _select(monkeypatch, accepted, telemetry)

    assert [value["surface"] for value in hints] == [
        "Accepted Person",
        "Soft Risk Work",
    ]
    assert [value["admission"]["source"] for value in hints] == [
        "v5_accepted",
        "raw_rank_fill",
    ]
    assert hints[0]["admission"]["v5_admission_telemetry"] == {
        "tier": 3,
        "basis": "relation_local_type_compatible",
    }
    assert hints[0]["semantic_role"] == "retrieval_query_hint"
    assert hints[1]["admission"]["soft_risk_flags"] == [
        "high_confidence_type_conflict",
        "original_question_phrase",
    ]
    assert hints[1]["admission"]["hard_rejection_reasons"] == []
    assert trace["semantic_role"] == "retrieval_query_hint_not_asserted_fact"
    assert trace["v5_selector_telemetry"] == telemetry
    assert trace["gold_access"] is False
    json.dumps((hints, trace), sort_keys=True)


@pytest.mark.parametrize(
    "hard_reason",
    [
        "weak_singleton",
        "repeated_fragment",
        "strict_subject_echo",
        "explicit_subject_alias",
    ],
)
def test_only_frozen_hard_reasons_block_raw_fill(monkeypatch, hard_reason):
    accepted, telemetry = _v5_trace()
    accepted = []
    telemetry["candidate_decisions"] = [
        {
            "raw_rank": 1,
            "surface": "Blocked Surface",
            "normalized_surface": "blocked surface",
            "base_score": 9,
            "decision": "reject",
            "admission_basis": "rejected",
            "reasons": [hard_reason, "insufficient_gold_free_support"],
        },
        {
            "raw_rank": 2,
            "surface": "Usable Surface",
            "normalized_surface": "usable surface",
            "base_score": 8,
            "decision": "reject",
            "admission_basis": "insufficient_gold_free_support",
            "reasons": ["insufficient_gold_free_support"],
        },
    ]
    hints, trace = _select(monkeypatch, accepted, telemetry)
    assert [value["surface"] for value in hints] == ["Usable Surface"]
    assert trace["hard_rejected_candidates"][0]["reasons"] == [hard_reason]


def test_raw_fill_obeys_raw_rank_not_v5_semantic_score(monkeypatch):
    accepted, telemetry = _v5_trace()
    accepted = []
    telemetry["candidate_decisions"] = [
        {
            "raw_rank": 2,
            "surface": "Second Raw",
            "normalized_surface": "second raw",
            "base_score": 100,
            "decision": "reject",
            "admission_basis": "rejected",
            "reasons": ["producer_consumer_type_conflict"],
        },
        {
            "raw_rank": 1,
            "surface": "First Raw",
            "normalized_surface": "first raw",
            "base_score": 1,
            "decision": "reject",
            "admission_basis": "rejected",
            "reasons": ["high_confidence_type_conflict"],
        },
        {
            "raw_rank": 3,
            "surface": "Third Raw",
            "normalized_surface": "third raw",
            "base_score": 200,
            "decision": "reject",
            "admission_basis": "rejected",
            "reasons": [],
        },
    ]
    hints, _ = _select(monkeypatch, accepted, telemetry)
    assert [value["surface"] for value in hints] == ["First Raw", "Second Raw"]


def test_relation_graph_renders_two_atomic_question_prefixed_branches():
    question = "Where was the author of Root Work born?"
    step = {
        "subject": "$hop_1",
        "relation_label": "place of birth",
        "output_slot": "hop_2",
        "dependencies": ["hop_1"],
    }
    queries, telemetry = v6.render_question_anchored_queries_v6(
        question=question,
        step=step,
        target_type="relation_graph",
        slot_values={"hop_1": ["Ada Lovelace", "Wrong Bridge", "Ignored Third"]},
    )
    assert queries == [
        f"{question}\nAda Lovelace place of birth",
        f"{question}\nWrong Bridge place of birth",
    ]
    assert all(value.startswith(question) for value in queries)
    assert all("Ada Lovelace Wrong Bridge" not in value for value in queries)
    assert telemetry["mode"] == "hint_branches"
    assert telemetry["query_count"] == 2
    assert all(value["question_prefix_exact"] for value in telemetry["queries"])


def test_subquery_graph_preserves_question_bytes_and_stable_deduplication():
    question = "  Which school did the writer attend?  "
    step = {
        "subquery_template": "Which school did #1 attend?",
        "output_slot": "step_2",
        "dependencies": ["step_1"],
    }
    queries, telemetry = v6.render_question_anchored_queries_v6(
        question=question,
        step=step,
        target_type="subquery_graph",
        slot_values={"step_1": ["Ada Lovelace", "Ada Lovelace"]},
    )
    assert queries == [f"{question}\nWhich school did Ada Lovelace attend?"]
    assert queries[0][: len(question)].encode("utf-8") == question.encode("utf-8")
    assert telemetry["all_question_prefix_exact"] is True


@pytest.mark.parametrize(
    ("target_type", "step", "expected_clause"),
    [
        (
            "relation_graph",
            {
                "subject": "$hop_1",
                "relation_label": "place of birth",
                "dependencies": ["hop_1"],
            },
            "place of birth",
        ),
        (
            "subquery_graph",
            {
                "subquery_template": "Who did #1 portray in True Grit?",
                "dependencies": ["step_1"],
            },
            "Who did portray in True Grit?",
        ),
        (
            "subquery_graph",
            {
                "subquery_template": "#1 >> educated at",
                "dependencies": ["step_1"],
            },
            "educated at",
        ),
    ],
)
def test_no_hint_fallback_removes_placeholder_but_keeps_frozen_clause(
    target_type, step, expected_clause
):
    question = "Original complete question?"
    queries, telemetry = v6.render_question_anchored_queries_v6(
        question=question,
        step=step,
        target_type=target_type,
        slot_values={},
    )
    assert queries == [f"{question}\n{expected_clause}"]
    assert "$hop" not in queries[0]
    assert "#1" not in queries[0]
    assert telemetry["mode"] == "no_hint_fallback"
    assert telemetry["queries"][0]["question_prefix_exact"] is True


def test_renderer_rejects_more_than_two_branches_and_malformed_frozen_clause():
    question = "Question?"
    step = {
        "subject": "$hop_1",
        "relation_label": "place of birth",
        "dependencies": ["hop_1"],
    }
    with pytest.raises(DependentRetrievalError, match="max_variants"):
        v6.render_question_anchored_queries_v6(
            question=question,
            step=step,
            target_type="relation_graph",
            slot_values={"hop_1": ["A"]},
            max_variants=3,
        )
    with pytest.raises(DependentRetrievalError, match="textual relation"):
        v6.render_question_anchored_queries_v6(
            question=question,
            step={"subject": "$hop_1", "dependencies": ["hop_1"]},
            target_type="relation_graph",
            slot_values={},
        )
