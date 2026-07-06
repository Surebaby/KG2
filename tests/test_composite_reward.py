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


def _build(text_value: float, tmp_path):
    linker = EntityLinker(cache_path=str(tmp_path / "entity_cache.jsonl"), use_genre=False)
    return CompositeRewardModel(
        alpha_gate=AlphaGate(),
        prm_annotator=PRMAnnotator(entity_linker=linker),
        text_reward_model=TextRewardModel(_ConstantTextReward(text_value), name="const"),
        outcome_weight=1.0,
        discount=0.95,
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
    kg = [("Barack Obama", "spouse", "Michelle Obama")]
    model_a = _build(text_value=0.5, tmp_path=tmp_path)
    model_b = _build(text_value=-1.0, tmp_path=tmp_path)
    step = _step_positive()
    a = model_a.compute_step_reward(step, kg, "prompt", logprobs=None, prev_conclusions=[])
    b = model_b.compute_step_reward(step, kg, "prompt", logprobs=None, prev_conclusions=[])
    assert a.r_total != pytest.approx(b.r_total)


def test_alpha_override_zero_drops_kg(tmp_path):
    """α=0 must reduce R_total to (1-α) · R_Text = R_Text."""
    kg = [("Barack Obama", "spouse", "Michelle Obama")]
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
    kg = [("Barack Obama", "spouse", "Michelle Obama")]
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
    kg = [("Barack Obama", "spouse", "Michelle Obama")]
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


def test_discounted_returns_monotone(tmp_path):
    model = _build(text_value=0.0, tmp_path=tmp_path)
    returns = model.discounted_returns([1.0, 1.0, 1.0])
    # All rewards equal → returns must be strictly decreasing over time.
    assert returns[0] > returns[1] > returns[2]
    assert returns[-1] == pytest.approx(1.0)
