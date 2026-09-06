from types import SimpleNamespace

import pytest

from kgproweight.reward.proofkg_process import is_automatic_proofkg
from kgproweight.training.reward_function import KGProWeightRewardFunction, RewardSpec


class _Tokenizer:
    def __call__(self, text, **_kwargs):
        return {"input_ids": list(text.encode("utf-8"))}

    def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
        return bytes(int(value) for value in ids).decode("utf-8")


def _reward():
    return KGProWeightRewardFunction(
        alpha_gate=SimpleNamespace(),
        prm_annotator=SimpleNamespace(),
        text_reward_model=SimpleNamespace(),
        tokenizer=_Tokenizer(),
        outcome_weight=4.0,
        min_valid_steps=3,
        min_reasoning_chars=20,
        proofkg_process_reward=True,
        proofkg_process_weight=1.0,
        proofkg_f1_weight=0.10,
        proofkg_dynamic_validity=True,
    )


KG = [("Alpha", "links to", "Beta"), ("Beta", "links to", "Gamma")]
RUNTIME = {
    "query_plan": {"hops": [{"pid": "P1"}, {"pid": "P2"}]},
    "provenance": {"gold_access": False, "complete_plan_execution": True},
}


def _response(answer="Gamma", two_steps=True):
    first = (
        "[Step 1]\nReasoning: Alpha connects to Beta according to the supplied proof edge.\n"
        "Knowledge Used: [(Alpha, links to, Beta)]\nConclusion: Alpha connects to Beta.\n"
    )
    second = (
        "[Step 2]\nReasoning: Beta then connects to Gamma according to the next proof edge.\n"
        "Knowledge Used: [(Beta, links to, Gamma)]\nConclusion: Beta connects to Gamma.\n"
    )
    return first + (second if two_steps else "") + f"[Final Answer] {answer}"


def _spec(runtime=RUNTIME):
    return RewardSpec(
        query="What does Alpha ultimately link to?",
        gold_answer="Gamma",
        kg_subgraph=list(KG),
        metadata={"question_kg_runtime": runtime},
    )


def test_two_hop_automatic_proof_is_valid_and_receives_continuous_reward():
    result = _reward()(prompt="", response=_response(), spec=_spec())
    proof = result["proofkg_process"]
    assert result["trajectory_valid"] is True
    assert proof["required_steps"] == 2
    assert proof["process_score"] == pytest.approx(1.0)
    assert proof["outcome_em"] == 1.0
    assert proof["outcome_f1"] == pytest.approx(1.0)
    assert result["trajectory_reward"] == pytest.approx(5.4)
    assert result["token_rewards"].sum().item() == pytest.approx(5.4)


def test_wrong_answer_is_ranked_below_correct_answer_without_process_collapse():
    correct = _reward()(prompt="", response=_response("Gamma"), spec=_spec())
    wrong = _reward()(prompt="", response=_response("Delta"), spec=_spec())
    assert wrong["trajectory_valid"] is True
    assert wrong["proofkg_process"]["process_score"] == pytest.approx(0.8)
    assert wrong["trajectory_reward"] < correct["trajectory_reward"]


def test_incomplete_trace_gets_exact_invalid_penalty_even_with_two_hop_target():
    result = _reward()(prompt="", response=_response(two_steps=False), spec=_spec())
    assert result["trajectory_valid"] is False
    assert result["trajectory_reward"] == pytest.approx(-4.0)
    assert result["token_rewards"].sum().item() == pytest.approx(-4.0)


def test_partial_or_gold_derived_question_kg_is_not_eligible():
    assert not is_automatic_proofkg(RUNTIME, KG[:1])
    assert not is_automatic_proofkg(
        {**RUNTIME, "provenance": {
            "gold_access": False, "complete_plan_execution": False,
        }}, KG
    )
    assert not is_automatic_proofkg(
        {**RUNTIME, "provenance": {"gold_access": True}}, KG
    )
