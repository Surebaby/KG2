"""Small CPU regression tests for actual process math and pair denominators."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.pilot.audit_source_credit_reward_utility_v1 import (
    RANK_FIELDS, paired_summary, process_components,
)


def _row(text=(0.5, 0.5), graph=0.65, eligible=1, valid=True):
    return {"features": {"m_graph": eligible}, "raw_text": list(text), "raw_graph": graph,
            "trajectory_valid": valid, "proof_result": {"telemetry": {"structural_component": .4}}}


def _gate(alpha=.25, text_center=0.0, text_scale=.5):
    return SimpleNamespace(normalization={"text_center": text_center, "text_scale": text_scale,
        "graph_center": .45, "graph_scale": .2, "fixed_alpha": .4},
        predict=lambda features: alpha if features["m_graph"] else 0.0)


def test_actual_A_F_T_weights_no_outcome_and_no_double_text_weight():
    result = process_components(_row(), _gate())
    assert result["text_T_process"] == pytest.approx(.3)
    assert result["learned_A_process"] == pytest.approx(.275)
    assert result["fixed_F_process"] == pytest.approx(.26)
    assert result["learned_A_graph_component"] == pytest.approx(.05)
    assert result["text_clipped_step_count"] == 0  # Exactly one is not clipped.


def test_text_clip_is_per_step_before_mean():
    result = process_components(_row(text=(-.8, .6)), _gate(text_center=.2, text_scale=.2))
    assert result["text_normalized_step_mean"] == pytest.approx(0)
    assert result["text_T_process"] == 0
    assert result["text_clipped_step_count"] == 2
    assert result["learned_A_process"] == pytest.approx(.05)


def test_masked_graph_credit_exactly_zero_and_invalid_has_no_process():
    masked = process_components(_row(eligible=0), _gate())
    assert masked["learned_A_graph_component"] == masked["fixed_F_graph_component"] == 0
    assert masked["learned_A_process"] == masked["fixed_F_process"] == masked["text_T_process"]
    invalid = process_components(_row(text=(), valid=False), _gate())
    assert invalid["learned_A_process"] == invalid["fixed_F_process"] == invalid["text_T_process"] == 0
    assert invalid["alpha_A"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1.1])
def test_bad_text_scores_are_rejected(value):
    with pytest.raises(ValueError, match="invalid raw text"):
        process_components(_row(text=(value,)), _gate())


def test_pair_summary_uses_one_valid_EM_hit_nonhit_pair_per_question():
    pairs = []
    for index in range(3):
        pair = [{"question_key": f"q{index}", "valid": True, "em": em,
                 "credit_eligible": True, "feature_vector": [1, 1, 1, 1],
                 **{field: score for field in RANK_FIELDS}}
                for em, score in ((1, .6), (0, .2))]
        pairs.extend(pair)
    pairs[2]["valid"] = False  # Excluded even though the EM values differ.
    pairs[4]["em"] = 0  # Both non-hits: excluded.
    pairs[0]["fixed_F_process"] = 0
    pairs[0]["graph_structure_raw_diagnostic"] = .2  # Structural tie.
    result = paired_summary(pairs)
    assert result["question_pairs"] == 3
    assert result["valid_em_hit_nonhit_pairs"] == 1
    assert result["rankings"]["learned_A_process"]["win"] == 1
    assert result["rankings"]["fixed_F_process"]["loss"] == 1
    assert result["rankings"]["graph_structure_raw_diagnostic"]["tie_adjusted_accuracy"] == .5
    assert result["A_minus_F"]["improved_pairs"] == 1
    assert result["A_minus_F"]["paired_ranking_delta"]["mean"] == 1
    assert result["credit_eligible_mixed_pairs_identical_four_features"] == 1
    broken = deepcopy(pairs)
    broken.pop()
    with pytest.raises(ValueError, match="exact K2"):
        paired_summary(broken)
