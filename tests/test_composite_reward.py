"""CompositeRewardModel — the central place where bug #1 was fixed."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from kgproweight.data.parsers import ParsedStep
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.reward.alpha_gate import AlphaGate
from kgproweight.reward.composite_reward import CompositeRewardModel
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.reward.text_reward_model import TextRewardModel, _DummyTextReward


class _ConstantTextReward(_DummyTextReward):
    def __init__(self, value: float) -> None:
        self.value = value

    def score_step(self, prompt: str, step_text: str) -> float:  # noqa: ARG002
        return self.value


def _build(text_value: float, tmp_path, **kw):
    linker = EntityLinker(cache_path=str(tmp_path / "entity_cache.jsonl"), use_genre=False)
    return CompositeRewardModel(
        alpha_gate=AlphaGate(),
        prm_annotator=PRMAnnotator(entity_linker=linker),
        text_reward_model=TextRewardModel(_ConstantTextReward(text_value), name="const"),
        outcome_weight=1.0,
        discount=0.95,
        **kw,
    )


def _step_positive():
    return ParsedStep(
        index=0,
        raw_text="(Barack Obama, spouse, Michelle Obama).",
        cited_triples=[("Barack Obama", "spouse", "Michelle Obama")],
        mentioned_entities=["Barack Obama"],
        intermediate_conclusion="Michelle Obama",
    )


def _step_negative():
    return ParsedStep(
        index=1,
        raw_text="(Barack Obama, spouse, Hillary Clinton).",
        cited_triples=[("Barack Obama", "spouse", "Hillary Clinton")],
        mentioned_entities=["Barack Obama"],
        intermediate_conclusion="Wrong",
    )


def test_text_reward_is_actually_mixed_in(tmp_path):
    """Regression for bug #1: r_text must influence r_total."""
    # >= min_subgraph_for_verify (3) triples: a sparser graph cannot verify or
    # refute a citation, so every label would be NEUTRAL and r_kg = 0 (C2).
    kg = [("Barack Obama", "spouse", "Michelle Obama"),
          ("Barack Obama", "occupation", "politician"),
          ("Barack Obama", "country of citizenship", "United States")]
    model_a = _build(text_value=0.5, tmp_path=tmp_path)
    model_b = _build(text_value=-1.0, tmp_path=tmp_path)
    step = _step_positive()
    a = model_a.compute_step_reward(step, kg, "prompt", logprobs=None, prev_conclusions=[])
    b = model_b.compute_step_reward(step, kg, "prompt", logprobs=None, prev_conclusions=[])
    assert a.r_total != pytest.approx(b.r_total)


def test_alpha_override_zero_drops_kg(tmp_path):
    """α=0 must reduce R_total to (1-α) · R_Text = R_Text."""
    # >= min_subgraph_for_verify (3) triples: a sparser graph cannot verify or
    # refute a citation, so every label would be NEUTRAL and r_kg = 0 (C2).
    kg = [("Barack Obama", "spouse", "Michelle Obama"),
          ("Barack Obama", "occupation", "politician"),
          ("Barack Obama", "country of citizenship", "United States")]
    model = _build(text_value=0.3, tmp_path=tmp_path)
    recs = model.compute_trajectory_rewards(
        steps=[_step_positive()],
        kg_subgraph=kg,
        text_reward_prompts=["prompt"],
        logprobs_list=[None],
        alpha_override=0.0,
    )
    assert recs[0].alpha == 0.0
    assert recs[0].r_total == pytest.approx(0.3)


def test_alpha_override_one_drops_text(tmp_path):
    """α=1 must reduce R_total to α · R_KG = R_KG."""
    # >= min_subgraph_for_verify (3) triples: a sparser graph cannot verify or
    # refute a citation, so every label would be NEUTRAL and r_kg = 0 (C2).
    kg = [("Barack Obama", "spouse", "Michelle Obama"),
          ("Barack Obama", "occupation", "politician"),
          ("Barack Obama", "country of citizenship", "United States")]
    model = _build(text_value=0.3, tmp_path=tmp_path)
    recs = model.compute_trajectory_rewards(
        steps=[_step_positive()],
        kg_subgraph=kg,
        text_reward_prompts=["prompt"],
        logprobs_list=[None],
        alpha_override=1.0,
    )
    assert recs[0].alpha == 1.0
    assert recs[0].r_total == pytest.approx(1.0)  # POSITIVE = +1


def test_outcome_added_to_last_step(tmp_path):
    # >= min_subgraph_for_verify (3) triples: a sparser graph cannot verify or
    # refute a citation, so every label would be NEUTRAL and r_kg = 0 (C2).
    kg = [("Barack Obama", "spouse", "Michelle Obama"),
          ("Barack Obama", "occupation", "politician"),
          ("Barack Obama", "country of citizenship", "United States")]
    model = _build(text_value=0.0, tmp_path=tmp_path)
    steps = [_step_positive(), _step_positive()]
    no_outcome = model.compute_trajectory_rewards(
        steps=steps,
        kg_subgraph=kg,
        text_reward_prompts=["", ""],
        logprobs_list=[None, None],
        alpha_override=1.0,
    )
    with_outcome = model.compute_trajectory_rewards(
        steps=steps,
        kg_subgraph=kg,
        text_reward_prompts=["", ""],
        logprobs_list=[None, None],
        alpha_override=1.0,
        predicted_answer="Michelle Obama",
        gold_answer="michelle obama",
    )
    assert with_outcome[-1].r_total == pytest.approx(no_outcome[-1].r_total + 1.0)


# ---------------------------------------------------------------------------
# Step-shortfall penalty (retraining_plan §9.4-3 / R-1b)
# ---------------------------------------------------------------------------

_KG3 = [("Barack Obama", "spouse", "Michelle Obama"),
        ("Barack Obama", "occupation", "politician"),
        ("Barack Obama", "country of citizenship", "United States")]


def _traj(model, n_steps, **kw):
    steps = [_step_positive() for _ in range(n_steps)]
    return model.compute_trajectory_rewards(
        steps=steps,
        kg_subgraph=_KG3,
        text_reward_prompts=[""] * n_steps,
        logprobs_list=[None] * n_steps,
        alpha_override=1.0,
        **kw,
    )


def test_shortfall_disabled_by_default(tmp_path):
    """coef=0 must reproduce the pre-2026-08-22 reward exactly."""
    model = _build(text_value=0.0, tmp_path=tmp_path)
    assert model.shortfall_coef == 0.0
    two = _traj(model, 2)
    assert two[-1].r_total == pytest.approx(1.0)  # POSITIVE only, no penalty


def test_shortfall_penalises_short_trajectory(tmp_path):
    """A 2-step trajectory is 1 step short of target_steps=3."""
    model = _build(text_value=0.0, tmp_path=tmp_path,
                   shortfall_coef=0.25, target_steps=3)
    # outcome_weight=1.0 here, so penalty = 0.25 * 1.0 * 1/3.
    expected = 1.0 - 0.25 * 1.0 * (1.0 / 3.0)
    assert _traj(model, 2)[-1].r_total == pytest.approx(expected)


def test_shortfall_scales_with_the_gap(tmp_path):
    model = _build(text_value=0.0, tmp_path=tmp_path,
                   shortfall_coef=0.25, target_steps=3)
    one, two, three = (_traj(model, n)[-1].r_total for n in (1, 2, 3))
    # Deeper shortfall must cost strictly more, and meeting the target costs zero.
    assert one < two < three
    assert three == pytest.approx(1.0)


def test_shortfall_not_charged_at_or_above_target(tmp_path):
    model = _build(text_value=0.0, tmp_path=tmp_path,
                   shortfall_coef=0.25, target_steps=3)
    assert _traj(model, 4)[-1].r_total == pytest.approx(1.0)


def test_shortfall_applies_to_invalid_trajectories_too(tmp_path):
    """The collapse used *valid* 2-step traces, so validity must not exempt it.

    Gating the penalty on trajectory_valid would leave exactly the escape route
    R-1b exists to close.
    """
    model = _build(text_value=0.0, tmp_path=tmp_path,
                   shortfall_coef=0.25, target_steps=3)
    penalty = 0.25 * 1.0 * (1.0 / 3.0)
    invalid = _traj(model, 2, trajectory_valid=False)
    # invalid_penalty (-outcome_weight) AND the shortfall both land on the last step.
    assert invalid[-1].r_total == pytest.approx(1.0 - 1.0 - penalty)


def test_shortfall_anchors_to_outcome_weight(tmp_path):
    """The penalty is defined relative to outcome_weight, not as an absolute."""
    a = _build(text_value=0.0, tmp_path=tmp_path, shortfall_coef=0.25, target_steps=3)
    b = _build(text_value=0.0, tmp_path=tmp_path, shortfall_coef=0.25, target_steps=3)
    b.outcome_weight = 4.0  # the production value
    pen_a = 1.0 - _traj(a, 2)[-1].r_total
    pen_b = 1.0 - _traj(b, 2)[-1].r_total
    assert pen_b == pytest.approx(pen_a * 4.0)
    # Production magnitude check: missing 1 of 3 steps costs 0.33 against the
    # ~0.33 total process-reward budget of a collapsed 2-step trajectory.
    assert pen_b == pytest.approx(0.25 * 4.0 / 3.0)


# ---------------------------------------------------------------------------
# R_Text DC removal / 量纲统一 (retraining_plan §9.4-1, D2)
# ---------------------------------------------------------------------------

def _one(model, text_prompt="", **kw):
    return model.compute_trajectory_rewards(
        steps=[_step_positive()],
        kg_subgraph=_KG3,
        text_reward_prompts=[text_prompt],
        logprobs_list=[None],
        **kw,
    )[0]


def test_centering_off_is_bit_for_bit_identical(tmp_path):
    """The default must reproduce every pre-2026-08-23 run exactly.

    This is the guarantee that lets the change ship without re-baselining the
    r9/R10 history: with center_text_reward=False the mixed value is the raw
    value and r_total is the same float.
    """
    off = _build(text_value=0.6284, tmp_path=tmp_path, text_reward_scale=0.3,
                 step_reward_scale=1.5)
    assert off.center_text_reward is False
    rec = _one(off, alpha_override=0.8)
    # α·r_kg + (1-α)·r_text·0.3, all ×1.5, with r_kg = +1 (POSITIVE citation).
    expected = (0.8 * 1.0 + 0.2 * 0.6284 * 0.3) * 1.5
    assert rec.r_total == pytest.approx(expected)
    # r_text_used mirrors r_text when centering is off, so downstream diagnostics
    # stay meaningful on old configs instead of reading a default 0.0.
    assert rec.r_text_used == pytest.approx(rec.r_text)
    assert rec.text_baseline == 0.0


def test_centering_removes_the_dc_offset(tmp_path):
    """A constant text reward must center to ~0 once the baseline has warmed."""
    on = _build(text_value=0.6284, tmp_path=tmp_path, text_reward_scale=0.3,
                step_reward_scale=1.5, center_text_reward=True,
                text_baseline_momentum=0.99)
    # First observation: baseline initialises AT the data, so the offset is gone
    # immediately rather than after a warm-up carrying the full +0.63.
    first = _one(on, alpha_override=0.8)
    assert first.r_text == pytest.approx(0.6284)
    assert first.r_text_used == pytest.approx(0.0, abs=1e-9)
    # A constant input keeps the baseline pinned, so it stays centered.
    for _ in range(20):
        rec = _one(on, alpha_override=0.8)
    assert rec.r_text_used == pytest.approx(0.0, abs=1e-9)
    assert on.text_baseline == pytest.approx(0.6284)


def test_centering_preserves_variation_around_the_baseline(tmp_path):
    """Centering must remove the mean, not the signal."""
    on = _build(text_value=0.6, tmp_path=tmp_path, center_text_reward=True,
                text_baseline_momentum=0.99)
    for _ in range(50):
        _one(on, alpha_override=0.5)
    baseline = on.text_baseline
    # Now feed a step that scores clearly above the established baseline.
    on.text_reward_model.backend.value = 0.9
    high = _one(on, alpha_override=0.5)
    assert high.r_text_used == pytest.approx(0.9 - baseline, abs=1e-6)
    assert high.r_text_used > 0.0
    on.text_reward_model.backend.value = 0.2
    low = _one(on, alpha_override=0.5)
    assert low.r_text_used < 0.0


def test_centering_flips_the_sign_of_dreward_dalpha(tmp_path):
    """The actual bug §9.4-1 exists to fix.

    d r_total/d alpha = (r_kg - c_text·r_text_used)·c_step. With the MEASURED
    r_kg=0.0896 and r_text=0.6284 this is (0.0896 - 0.3·0.6284)·1.5 = -0.148:
    the reward paid the policy to LOWER alpha, i.e. to make the cited subgraph
    look sparser, because alpha rises with f_density = |E|/(|V|+eps).
    """
    C_TEXT, C_STEP = 0.3, 1.5
    R_KG, R_TEXT = 0.0896, 0.6284

    def dr_dalpha(r_kg, r_text_used):
        return (r_kg - C_TEXT * r_text_used) * C_STEP

    # Before: strictly negative, and this is where the -0.148 in the docs comes from.
    before = dr_dalpha(R_KG, R_TEXT)
    assert before == pytest.approx(-0.14838, abs=1e-5)
    assert before < 0
    # After: the DC offset is gone, so a step scoring AT the baseline leaves the
    # sensitivity equal to r_kg·c_step > 0 -- the gate now points toward the KG.
    after = dr_dalpha(R_KG, 0.0)
    assert after > 0
    assert after == pytest.approx(R_KG * C_STEP)

    # And the same arithmetic through the real model, not just the formula.
    on = _build(text_value=R_TEXT, tmp_path=tmp_path, text_reward_scale=C_TEXT,
                step_reward_scale=C_STEP, center_text_reward=True)
    for _ in range(10):
        _one(on, alpha_override=0.5)
    lo = _one(on, alpha_override=0.2)
    hi = _one(on, alpha_override=0.9)
    # r_kg is +1 for this step, so raising alpha must now RAISE reward.
    assert hi.r_total > lo.r_total


def test_alpha_override_uses_the_centered_value(tmp_path):
    """The ablation arms must run the same dimensioned channel as the main arm.

    If this branch read the raw r_text, the α ∈ {0, 0.5, 1} arms would be the
    only ones with an uncentered text channel and the ablation would be
    measuring the centering as well as α.
    """
    on = _build(text_value=0.6284, tmp_path=tmp_path, text_reward_scale=0.3,
                step_reward_scale=1.5, center_text_reward=True)
    for _ in range(10):
        _one(on, alpha_override=0.5)
    rec = _one(on, alpha_override=0.0)
    # α=0 => r_total is entirely the text channel, which is centered to ~0.
    assert rec.r_text == pytest.approx(0.6284)
    assert rec.r_total == pytest.approx(0.0, abs=1e-6)


def test_baseline_observed_once_per_step_not_twice(tmp_path):
    """alpha_override must not re-observe the same r_text into the EMA.

    compute_step_reward already consumed the observation; the override branch
    recombines from the recorded value. A second observation would double the
    EMA's effective sample rate on the ablation path only.
    """
    on = _build(text_value=0.6, tmp_path=tmp_path, center_text_reward=True)
    _one(on, alpha_override=0.5)
    assert on.text_baseline_n_obs == 1
    _one(on)  # no override
    assert on.text_baseline_n_obs == 2


def test_reset_text_baseline(tmp_path):
    on = _build(text_value=0.6, tmp_path=tmp_path, center_text_reward=True)
    for _ in range(5):
        _one(on, alpha_override=0.5)
    assert on.text_baseline_n_obs == 5
    on.reset_text_baseline()
    assert on.text_baseline_n_obs == 0
    assert on.text_baseline == 0.0


def test_empty_trajectory_does_not_move_the_baseline(tmp_path):
    """A no-step invalid trajectory has no r_text observation to record.

    Letting it push the baseline toward 0 would re-introduce part of the offset
    on exactly the batches where the policy emitted nothing parseable.
    """
    on = _build(text_value=0.6, tmp_path=tmp_path, center_text_reward=True)
    for _ in range(10):
        _one(on, alpha_override=0.5)
    base_before, n_before = on.text_baseline, on.text_baseline_n_obs
    recs = on.compute_trajectory_rewards(
        steps=[], kg_subgraph=_KG3, text_reward_prompts=[], logprobs_list=[],
        trajectory_valid=False,
    )
    assert len(recs) == 1
    assert recs[0].r_total == pytest.approx(-on.outcome_weight)
    assert on.text_baseline == pytest.approx(base_before)
    assert on.text_baseline_n_obs == n_before


def test_outcome_and_penalties_preserve_the_new_fields(tmp_path):
    """The last-step reconstructions must not silently reset r_text_used.

    r_total is rebuilt in four places (outcome, invalid, shortfall); each does a
    field-by-field StepReward copy, so an omitted field reads back as the
    dataclass default 0.0 and the diagnostic goes quietly wrong.
    """
    on = _build(text_value=0.6, tmp_path=tmp_path, center_text_reward=True,
                shortfall_coef=0.25, target_steps=3)
    for _ in range(10):
        _one(on, alpha_override=0.5)
    baseline = on.text_baseline
    steps = [_step_positive(), _step_positive()]
    recs = on.compute_trajectory_rewards(
        steps=steps, kg_subgraph=_KG3, text_reward_prompts=["", ""],
        logprobs_list=[None, None], alpha_override=1.0,
        predicted_answer="Michelle Obama", gold_answer="michelle obama",
    )
    # Last step carries outcome + shortfall AND must still report the channel.
    assert recs[-1].text_baseline == pytest.approx(baseline, abs=1e-6)
    assert recs[-1].r_text == pytest.approx(0.6)
    assert recs[-1].r_text_used == pytest.approx(0.6 - baseline, abs=1e-6)
