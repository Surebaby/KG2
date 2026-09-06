"""Outcome-first PPO reward and opt-in output-format contract regressions.

Scorers are forbidden in PPO-O tests. These tests compute rewards on synthetic
trajectories only; they do not initialize or update a research checkpoint.
"""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch

import kgproweight.training.reward_function as reward_module
from kgproweight.reward.proofkg_process_v2_3 import (
    SCORER_VERSION,
    build_execution_trace_v2_3,
    score_proofkg_v2_3,
)
from kgproweight.training.reward_function import KGProWeightRewardFunction, RewardSpec


class _Tokenizer:
    eos_token_id = 0

    def __call__(self, text, **_kwargs):
        return {"input_ids": [ord(char) for char in text]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(
            chr(int(token)) if int(token) else ("" if skip_special_tokens else "<eos>")
            for token in ids
        )


class _ForbiddenComponent:
    def __init__(self, name):
        self.name = name

    def __call__(self, *_args, **_kwargs):
        raise AssertionError(f"PPO-O unexpectedly called {self.name}")

    def __getattr__(self, attr):
        raise AssertionError(f"PPO-O unexpectedly accessed {self.name}.{attr}")


KG = [("Alpha", "links to", "Beta"), ("Beta", "links to", "Gamma")]
PLAN = {"hops": [
    {"subject": "Alpha", "pids": ["P1"], "output_slot": "hop_1", "relation_role": "bridge"},
    {"subject": "$hop_1", "pids": ["P2"], "output_slot": "hop_2", "relation_role": "answer_operand"},
]}
EXECUTION = {"hops": [
    {"hop_index": 1, "matches": [KG[0]], "output_entities": [{"label": "Beta"}]},
    {"hop_index": 2, "matches": [KG[1]], "output_entities": [{"label": "Gamma"}]},
]}
RUNTIME = {
    "question_key": "2wikimultihopqa::proof-emf1-contract",
    "query_plan": PLAN,
    "execution": EXECUTION,
    "provenance": {"gold_access": False, "complete_plan_execution": True},
}


def _response(answer="Gamma", *, steps=3):
    blocks = []
    for index in range(1, steps + 1):
        evidence = (
            "[(Alpha, links to, Beta)]" if index == 1 else
            "[(Beta, links to, Gamma)]" if index == 2 else "[]"
        )
        conclusion = "Alpha links to Beta." if index == 1 else "Beta links to Gamma."
        blocks.append(
            f"[Step {index}]\n"
            "Reasoning: The supplied evidence connects the entities needed to answer the question.\n"
            f"Knowledge Used: {evidence}\n"
            f"Conclusion: {conclusion}\n"
        )
    return "\n".join(blocks) + f"\n[Final Answer]\n{answer}"


def _spec(*, dataset="hotpotqa", eligible=False, gold="Gamma", aliases=()):
    return RewardSpec(
        query="What does Alpha ultimately link to?",
        gold_answer=gold,
        gold_answer_aliases=list(aliases),
        kg_subgraph=list(KG) if eligible else [],
        metadata={
            "dataset": "2wikimultihopqa" if eligible else dataset,
            "qid": "proof-emf1-contract" if eligible else "ordinary-emf1-contract",
            "question_kg_runtime": deepcopy(RUNTIME) if eligible else {},
        },
    )


def _reward(*, runtime="v2", process=False):
    return KGProWeightRewardFunction(
        alpha_gate=_ForbiddenComponent("alpha"),
        prm_annotator=_ForbiddenComponent("PRM"),
        text_reward_model=_ForbiddenComponent("ReaRAG"),
        tokenizer=_Tokenizer(),
        outcome_weight=4.,
        max_steps=5,
        min_valid_steps=3,
        min_reasoning_chars=20,
        proofkg_process_reward=process,
        proofkg_process_version="v2_3",
        proofkg_process_weight=.2,
        proofkg_f1_weight=.1,
        proofkg_dynamic_validity=True,
        mixed_outcome_reward=True,
        mixed_text_reward=False,
        runtime_contract_version=runtime,
    )


def _score(response, *, spec=None, runtime="v2", process=False):
    ids = torch.tensor([ord(char) for char in response] + [_Tokenizer.eos_token_id])
    return _reward(runtime=runtime, process=process)(
        prompt="", response=response, response_ids=ids, spec=spec or _spec(),
    )


def _assert_terminal_reward(result, expected):
    rewards = result["token_rewards"]
    assert result["trajectory_reward"] == pytest.approx(expected)
    assert float(rewards.double().sum()) == pytest.approx(expected, abs=1e-6)
    assert float(rewards[-1]) == pytest.approx(expected, abs=1e-6)
    assert bool((rewards[:-1] == 0).all())
    assert sum(result["per_step_rewards"]) == pytest.approx(expected)


def _forbid_proof_calls(monkeypatch, *, allow_v23=False):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("PPO-O unexpectedly invoked a ProofKG scorer")

    names = ["score_grounded_process", "score_proofkg_v2", "score_proofkg_v2_2"]
    if not allow_v23:
        names.append("score_proofkg_v2_3")
    for name in names:
        monkeypatch.setattr(reward_module, name, forbidden)


@pytest.mark.parametrize("dataset", ["hotpotqa", "2wikimultihopqa", "musique"])
@pytest.mark.parametrize(
    "answer,gold,expected_em,expected_f1,expected_reward",
    [
        ("Gamma", "Gamma", 1., 1., 4.4),
        ("New York", "New York City", 0., .8, .32),
        ("Unrelated", "Gamma", 0., 0., 0.),
    ],
)
def test_ppo_o_uses_only_canonical_em_plus_f1_for_every_dataset(
    monkeypatch, dataset, answer, gold, expected_em, expected_f1, expected_reward,
):
    _forbid_proof_calls(monkeypatch)
    result = _score(_response(answer), spec=_spec(dataset=dataset, gold=gold))
    assert result["trajectory_valid"] is True
    assert result["proofkg_process"]["outcome_em"] == pytest.approx(expected_em)
    assert result["proofkg_process"]["outcome_f1"] == pytest.approx(expected_f1)
    assert result["proofkg_process"]["process_applied"] is False
    assert result["mixed_reward"]["text"] == result["mixed_reward"]["process"] == 0.
    assert result["mixed_reward"]["text_ema_n_obs"] == 0
    _assert_terminal_reward(result, expected_reward)


def test_ppo_o_eligible_graph_still_never_calls_any_process_scorer(monkeypatch):
    _forbid_proof_calls(monkeypatch)
    result = _score(_response(steps=2), spec=_spec(eligible=True))
    assert result["trajectory_valid"] is True
    assert result["proofkg_process"]["eligible"] is True
    assert result["proofkg_process"]["required_steps"] == 2
    assert result["proofkg_process"]["process_applied"] is False
    _assert_terminal_reward(result, 4.4)


def test_ppo_o_nonprimary_alias_is_normalized_deduplicated_and_rewarded(monkeypatch):
    _forbid_proof_calls(monkeypatch)
    result = _score(
        _response("NEW YORK."),
        spec=_spec(gold="New York City", aliases=["New York", " new york "]),
    )
    assert result["proofkg_process"]["gold_alias_count"] == 2
    assert result["mixed_reward"]["outcome_em_matched_nonprimary"] is True
    assert result["mixed_reward"]["outcome_f1_matched_nonprimary"] is True
    _assert_terminal_reward(result, 4.4)


@pytest.mark.parametrize("eligible", [False, True])
def test_v2_checks_max_steps_before_historical_scoring_cap(monkeypatch, eligible):
    _forbid_proof_calls(monkeypatch)
    response = _response(steps=6)
    result = _score(response, spec=_spec(eligible=eligible))
    assert result["trajectory_valid"] is False
    assert "too_many_steps" in result["format_contract_violations"]
    _assert_terminal_reward(result, -4.)
    legacy = _score(response, spec=_spec(eligible=eligible), runtime="legacy")
    assert legacy["trajectory_valid"] is True
    assert "format_contract_violations" not in legacy
    _assert_terminal_reward(legacy, 4.4)


@pytest.mark.parametrize("label", ["Reasoning", "Knowledge Used", "Conclusion"])
@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_v2_requires_each_step_field_exactly_once(monkeypatch, label, mutation):
    _forbid_proof_calls(monkeypatch)
    lines = _response().splitlines()
    index = next(index for index, line in enumerate(lines) if line.startswith(label + ":"))
    if mutation == "missing":
        del lines[index]
    else:
        lines.insert(index + 1, lines[index])
    response = "\n".join(lines)
    result = _score(response)
    assert result["trajectory_valid"] is False
    assert f"step_1_{label.lower().replace(' ', '_')}_count_not_one" in result["format_contract_violations"]
    _assert_terminal_reward(result, -4.)
    legacy = _score(response, runtime="legacy")
    legacy_valid = not (label == "Reasoning" and mutation == "missing")
    assert legacy["trajectory_valid"] is legacy_valid
    _assert_terminal_reward(legacy, 4.4 if legacy_valid else -4.)


@pytest.mark.parametrize("extra_final", [
    "[Final Answer]\nGamma",
    "Final Answer: Gamma",
    "**Final Answer**: Gamma",
])
def test_v2_rejects_duplicate_final_answer_in_parser_supported_forms(monkeypatch, extra_final):
    _forbid_proof_calls(monkeypatch)
    response = _response() + "\n" + extra_final
    result = _score(response)
    assert result["trajectory_valid"] is False
    assert "final_field_count_not_one" in result["format_contract_violations"]
    _assert_terminal_reward(result, -4.)
    legacy = _score(response, runtime="legacy")
    assert legacy["trajectory_valid"] is True
    _assert_terminal_reward(legacy, 4.4)


def test_v2_missing_final_answer_is_invalid_and_does_not_score(monkeypatch):
    _forbid_proof_calls(monkeypatch)
    result = _score(_response().split("[Final Answer]", 1)[0])
    assert result["trajectory_valid"] is False
    assert "final_field_count_not_one" in result["format_contract_violations"]
    _assert_terminal_reward(result, -4.)


def test_valid_five_step_outcome_is_unchanged_between_legacy_and_v2():
    response = _response(steps=5)
    current = _score(response)
    legacy = _score(response, runtime="legacy")
    assert current["trajectory_valid"] is legacy["trajectory_valid"] is True
    assert current["format_contract_violations"] == []
    assert current["proofkg_process"] == legacy["proofkg_process"]
    assert current["mixed_reward"] == legacy["mixed_reward"]
    assert torch.equal(current["token_rewards"], legacy["token_rewards"])
    _assert_terminal_reward(current, 4.4)


def test_v23_reward_dispatch_invokes_new_scorer_with_gold_free_arguments(monkeypatch):
    _forbid_proof_calls(monkeypatch, allow_v23=True)
    calls = []

    def observed_scorer(**kwargs):
        calls.append(kwargs)
        return score_proofkg_v2_3(**kwargs)

    monkeypatch.setattr(reward_module, "score_proofkg_v2_3", observed_scorer)
    response = _response(steps=2)
    spec = _spec(eligible=True)
    result = _score(response, spec=spec, process=True)
    direct = score_proofkg_v2_3(
        question=spec.query,
        generation=response,
        kg_triples=KG,
        execution_trace=build_execution_trace_v2_3(PLAN, EXECUTION),
        planned_hops=2,
    )
    assert len(calls) == 1
    assert set(calls[0]) == {"question", "generation", "kg_triples", "execution_trace", "planned_hops"}
    assert result["proofkg_process"]["scorer_version"] == SCORER_VERSION
    assert result["proofkg_process"]["process_applied"] is True
    assert result["proofkg_process"]["process_score"] == pytest.approx(direct["score"])
    _assert_terminal_reward(result, 4.4 + .2 * direct["score"])
