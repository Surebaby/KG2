"""Statistical-unit, hierarchical-mass and bounded-map regression coverage."""
from copy import deepcopy
import math

import pytest

from kgproweight.reward.source_reward_normalization_v2 import (
    APPLICATION_CONTRACT, VERSION, fit_text_normalization_v2,
    normalize_text_steps_v2, validate_text_normalization_v2,
)


def row(qid, candidate, scores, *, valid=True, dataset="synthetic"):
    return {"dataset": dataset, "qid": qid, "candidate_id": candidate,
            "trajectory_valid": valid, "raw_text": scores, "family_split": "train"}


def example_rows():
    return [row("a", "a0", [0.0, .8]), row("a", "a1", [-.4]), row("b", "b0", [.2, .2, .2])]


def test_fit_uses_step_variance_not_variance_of_trajectory_means():
    stats = fit_text_normalization_v2([row("a", "a0", [-1., 1.])])
    assert stats["text_center"] == 0
    assert stats["raw_step_std"] == stats["text_scale"] == 1
    assert stats["variance_components"] == {"within_trajectory": 1., "between_trajectory_means": 0.}


def test_question_candidate_step_weights_match_analytic_population():
    stats = fit_text_normalization_v2(example_rows())
    # Question a gets 1/2, each of its candidates 1/4; question b gets 1/2.
    assert stats["text_center"] == pytest.approx(.1)
    assert stats["raw_step_std"] == pytest.approx(math.sqrt(.13))
    assert stats["text_scale"] == pytest.approx(math.sqrt(.13))
    assert stats["counts"] == {"input_candidates": 3, "valid_candidates": 3,
        "input_questions": 2, "questions_with_valid_scores": 2, "steps": 6}
    assert stats["weight_sum"] == 1


def test_steps_or_identical_candidates_do_not_reweight_their_parent_question():
    original = example_rows()
    baseline = fit_text_normalization_v2(original)
    repeated_steps = deepcopy(original)
    repeated_steps[0]["raw_text"] *= 5
    duplicated_candidate = deepcopy(original) + [row("b", "b1", [.2, .2, .2])]
    for changed in (repeated_steps, duplicated_candidate):
        stats = fit_text_normalization_v2(changed)
        assert stats["text_center"] == baseline["text_center"]
        assert stats["raw_step_std"] == pytest.approx(baseline["raw_step_std"])
    assert fit_text_normalization_v2(list(reversed(original))) == baseline


def test_invalid_rows_stay_counted_but_add_no_observation_mass():
    original = example_rows()
    baseline = fit_text_normalization_v2(original)
    stats = fit_text_normalization_v2(original + [row("c", "c0", [], valid=False)])
    assert stats["text_center"] == baseline["text_center"]
    assert stats["raw_step_std"] == baseline["raw_step_std"]
    assert stats["counts"]["input_questions"] == 3
    assert stats["counts"]["questions_with_valid_scores"] == 2
    assert stats["counts"]["input_candidates"] == 4


def test_application_is_per_step_softsign_then_mean_with_no_hard_plateau():
    stats = fit_text_normalization_v2([row("a", "a0", [.2, .2])])
    assert stats["text_scale"] == .1
    result = normalize_text_steps_v2([-.8, .6], stats)
    assert result["version"] == VERSION and result["application_contract"] == APPLICATION_CONTRACT
    assert result["normalized_unclipped_step_scores"] == pytest.approx([-10, 4])
    assert result["bounded_step_scores"] == pytest.approx([-10 / 11, 4 / 5])
    assert result["mean_bounded"] == pytest.approx((-10 / 11 + 4 / 5) / 2)
    assert result["hard_clip_frac"] == 0
    assert result["raw_z_outside_unit_frac"] == 1
    increasing = normalize_text_steps_v2([-.9, -.8, .5, .6], stats)["bounded_step_scores"]
    assert increasing == sorted(set(increasing))
    assert all(-1 < value < 1 for value in increasing)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.01, True, "0.2"])
def test_bad_raw_scores_rejected_at_fit_and_apply(bad):
    stats = fit_text_normalization_v2(example_rows())
    with pytest.raises(ValueError):
        fit_text_normalization_v2([row("a", "a0", [bad])])
    with pytest.raises(ValueError):
        normalize_text_steps_v2([bad], stats)


def test_duplicate_ids_and_explicit_nontrain_rows_are_rejected():
    duplicated = example_rows()
    duplicated[1]["candidate_id"] = duplicated[0]["candidate_id"]
    with pytest.raises(ValueError, match="duplicate"):
        fit_text_normalization_v2(duplicated)
    nontrain = example_rows()
    nontrain[0]["family_split"] = "confirmation"
    with pytest.raises(ValueError, match="non-train"):
        fit_text_normalization_v2(nontrain)
    with pytest.raises(ValueError, match="at least one step"):
        fit_text_normalization_v2([row("a", "a0", [])])
    with pytest.raises(ValueError, match="no valid train"):
        fit_text_normalization_v2([row("a", "a0", [], valid=False)])


@pytest.mark.parametrize("field,value", [
    ("text_scale", .02), ("text_scale", float("nan")), ("scale_floor", .2),
    ("text_center", 2.), ("fit_split", "confirmation"), ("fit_unit", "trajectory_mean"),
    ("application_contract", "hard_clip"), ("hard_clipping_used", True),
    ("soft_saturation_threshold", .99), ("weight_sum", 2.),
])
def test_statistics_are_strictly_versioned_and_numerically_validated(field, value):
    stats = fit_text_normalization_v2(example_rows())
    stats[field] = value
    with pytest.raises(ValueError):
        validate_text_normalization_v2(stats)
