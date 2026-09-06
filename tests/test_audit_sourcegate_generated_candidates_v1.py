"""Synthetic CPU diagnostics: paired ranking must not be mistaken for evaluation."""
from __future__ import annotations

import pytest

from scripts.pilot.audit_sourcegate_generated_candidates_v1 import pair_summary, summarize


def candidate(question, index, em, **overrides):
    row = {
        "question_key": question, "family_sha256": f"family-{question}",
        "valid": True, "m_graph": 1, "length_capped": False,
        "em": em, "f1": em, "ppo_em": em, "ppo_f1": em,
        "response_tokens": 20, "steps": 3, "violations": [],
        "raw_graph": 0.5, "structural_component": 0.5, "answer_component": 0.5,
        "generation_sha256": f"{question}-{index}", "answer": "yes" if em else "no",
        "features": {"synthetic_feature": index},
    }
    row.update(overrides)
    return row


def test_ranking_uses_correct_candidate_independent_of_generation_order():
    rows = [
        candidate("win", 0, 0, raw_graph=0.2, structural_component=0.2, answer_component=0.8),
        candidate("win", 1, 1, raw_graph=0.8, structural_component=0.8, answer_component=0.2),
        candidate("tie", 0, 1, raw_graph=0.5 + 5e-11, structural_component=0.8, answer_component=0.5),
        candidate("tie", 1, 0, raw_graph=0.5, structural_component=0.2, answer_component=0.5),
        candidate("loss", 0, 0, raw_graph=0.8, structural_component=0.2, answer_component=0.8),
        candidate("loss", 1, 1, raw_graph=0.2, structural_component=0.8, answer_component=0.2),
    ]
    result = pair_summary(rows)
    assert result["eligible_both_valid_mixed_pairs"] == 3
    ranking = result["graph_ranking_on_eligible_both_valid_mixed_pairs"]
    assert ranking["raw_graph"] == {"win": 1, "tie": 1, "loss": 1, "tie_adjusted_accuracy": 0.5}
    assert ranking["structural_component"] == {"win": 3, "tie": 0, "loss": 0, "tie_adjusted_accuracy": 1.0}
    assert ranking["answer_component"] == {"win": 0, "tie": 1, "loss": 2, "tie_adjusted_accuracy": pytest.approx(1 / 6)}


def test_ranking_requires_both_candidates_valid_and_graph_eligible_and_mixed():
    rows = [
        candidate("eligible", 0, 1, raw_graph=0.8), candidate("eligible", 1, 0, raw_graph=0.2),
        candidate("invalid", 0, 1, valid=False, raw_graph=0.1), candidate("invalid", 1, 0, raw_graph=0.9),
        candidate("no_graph", 0, 1, raw_graph=0.1), candidate("no_graph", 1, 0, m_graph=0, raw_graph=0.9),
        candidate("both_correct", 0, 1), candidate("both_correct", 1, 1),
    ]
    result = pair_summary(rows)
    assert result["questions"] == 4
    assert result["mixed_correct_wrong"] == 3
    assert result["both_valid"] == 3
    assert result["eligible_both_valid_mixed_pairs"] == 1
    assert result["graph_ranking_on_eligible_both_valid_mixed_pairs"]["raw_graph"] == {
        "win": 1, "tie": 0, "loss": 0, "tie_adjusted_accuracy": 1.0,
    }
    no_eligible = pair_summary(rows[2:])
    assert no_eligible["eligible_both_valid_mixed_pairs"] == 0
    assert no_eligible["graph_ranking_on_eligible_both_valid_mixed_pairs"]["raw_graph"] == {
        "win": 0, "tie": 0, "loss": 0, "tie_adjusted_accuracy": None,
    }


def test_identical_gate_features_count_only_eligible_mixed_pairs():
    rows = [
        candidate("same", 0, 1, features={"x": 0.3, "y": 1}),
        candidate("same", 1, 0, features={"y": 1, "x": 0.3}),
        candidate("different", 0, 1), candidate("different", 1, 0),
        candidate("invalid_same", 0, 1, valid=False, features={"x": 1}),
        candidate("invalid_same", 1, 0, features={"x": 1}),
        candidate("both_wrong", 0, 0, features={"x": 1}),
        candidate("both_wrong", 1, 0, features={"x": 1}),
    ]
    result = pair_summary(rows)
    assert result["eligible_both_valid_mixed_pairs"] == 2
    assert result["mixed_pairs_identical_gate_features"] == 1
    # Correct and incorrect traces can be indistinguishable to a gate receiving
    # only these features, even when their generated answers differ.
    assert rows[0]["answer"] != rows[1]["answer"]


def test_oracle_at_two_is_not_single_candidate_answer_em():
    rows = [
        candidate("both_correct", 0, 1), candidate("both_correct", 1, 1),
        candidate("mixed", 0, 0), candidate("mixed", 1, 1),
        candidate("both_wrong", 0, 0), candidate("both_wrong", 1, 0),
    ]
    result = pair_summary(rows)
    assert result["both_correct"] == result["both_wrong"] == result["mixed_correct_wrong"] == 1
    assert result["oracle_answer_em_at_2"] == pytest.approx(2 / 3)
    assert summarize(rows)["answer_em"] == 0.5
    assert result["oracle_answer_em_at_2"] > summarize(rows)["answer_em"]


@pytest.mark.parametrize("bad_count", [1, 3])
def test_rejects_non_k2_question_even_when_total_includes_valid_pair(bad_count):
    rows = [candidate("valid_pair", 0, 1), candidate("valid_pair", 1, 0)]
    rows += [candidate("wrong_k", index, 0) for index in range(bad_count)]
    with pytest.raises(ValueError, match="exactly K2"):
        pair_summary(rows)
