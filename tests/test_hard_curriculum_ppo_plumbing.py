from types import SimpleNamespace

import pytest
import torch

from kgproweight.reward.proofkg_process_v2 import build_execution_trace, score_proofkg_v2
from kgproweight.training.phase3_ppo import _sample_rollout_indices
from kgproweight.training.reward_function import KGProWeightRewardFunction, RewardSpec


class _Tokenizer:
    def __call__(self, text, **_kwargs):
        return {"input_ids": list(text.encode("utf-8"))}

    def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
        return bytes(int(value) for value in ids).decode("utf-8")


KG = [("Alpha", "links to", "Beta"), ("Beta", "links to", "Gamma")]
PLAN = {
    "hops": [
        {"subject": "Alpha", "pids": ["P1"], "output_slot": "hop_1", "relation_role": "bridge"},
        {"subject": "$hop_1", "pids": ["P2"], "output_slot": "hop_2", "relation_role": "answer_operand"},
    ]
}
EXECUTION = {
    "hops": [
        {"hop_index": 1, "matches": [KG[0]], "output_entities": [{"label": "Beta"}]},
        {"hop_index": 2, "matches": [KG[1]], "output_entities": [{"label": "Gamma"}]},
    ]
}
RUNTIME = {
    "query_plan": PLAN,
    "execution": EXECUTION,
    "provenance": {"gold_access": False, "complete_plan_execution": True},
}
RESPONSE = (
    "[Step 1]\nReasoning: Alpha connects to Beta according to the supplied exact proof edge.\n"
    "Knowledge Used: [(Alpha, links to, Beta)]\nConclusion: Alpha connects to Beta.\n"
    "[Step 2]\nReasoning: Beta connects to Gamma according to the supplied second proof edge.\n"
    "Knowledge Used: [(Beta, links to, Gamma)]\nConclusion: Beta connects to Gamma.\n"
    "[Final Answer] Gamma"
)


def _reward(*, process: bool):
    return KGProWeightRewardFunction(
        alpha_gate=SimpleNamespace(),
        prm_annotator=SimpleNamespace(),
        text_reward_model=SimpleNamespace(),
        tokenizer=_Tokenizer(),
        outcome_weight=4.0,
        min_reasoning_chars=20,
        proofkg_outcome_only_reward=True,
        proofkg_process_reward=process,
        proofkg_process_version="v2_1",
        proofkg_process_weight=0.2,
        proofkg_f1_weight=0.1,
        proofkg_dynamic_validity=True,
    )


def test_weighted_k4_sampler_is_grouped_deterministic_and_uses_weights():
    left = _sample_rollout_indices(
        2, 400, 4, torch.Generator().manual_seed(42), sampling_weights=[0.9, 0.1]
    )
    right = _sample_rollout_indices(
        2, 400, 4, torch.Generator().manual_seed(42), sampling_weights=[0.9, 0.1]
    )
    assert left == right
    assert all(len(set(left[start:start + 4])) == 1 for start in range(0, len(left), 4))
    assert left.count(0) > left.count(1) * 4


def test_paired_reward_diff_is_only_v21_process_term():
    spec = RewardSpec(
        query="What does Alpha ultimately link to?",
        gold_answer="Gamma",
        kg_subgraph=list(KG),
        metadata={"question_kg_runtime": RUNTIME},
    )
    outcome = _reward(process=False)(prompt="", response=RESPONSE, spec=spec)
    process = _reward(process=True)(prompt="", response=RESPONSE, spec=spec)
    direct = score_proofkg_v2(
        question=spec.query,
        generation=RESPONSE,
        kg_triples=KG,
        execution_trace=build_execution_trace(PLAN, EXECUTION),
        planned_hops=2,
    )
    assert outcome["trajectory_reward"] == pytest.approx(4.4)
    assert process["proofkg_process"]["process_score"] == pytest.approx(direct["score"])
    assert process["trajectory_reward"] - outcome["trajectory_reward"] == pytest.approx(
        0.2 * direct["score"]
    )


def test_paired_invalid_penalty_is_identical():
    spec = RewardSpec(
        query="What does Alpha ultimately link to?", gold_answer="Gamma",
        kg_subgraph=list(KG), metadata={"question_kg_runtime": RUNTIME},
    )
    invalid = "[Final Answer] Gamma"
    assert _reward(process=False)(prompt="", response=invalid, spec=spec)["trajectory_reward"] == -4.0
    assert _reward(process=True)(prompt="", response=invalid, spec=spec)["trajectory_reward"] == -4.0


def test_v21_fails_closed_without_execution_trace():
    spec = RewardSpec(
        query="What does Alpha ultimately link to?", gold_answer="Gamma",
        kg_subgraph=list(KG),
        metadata={"question_kg_runtime": {k: v for k, v in RUNTIME.items() if k != "execution"}},
    )
    with pytest.raises(ValueError, match="execution.hops"):
        _reward(process=True)(prompt="", response=RESPONSE, spec=spec)
