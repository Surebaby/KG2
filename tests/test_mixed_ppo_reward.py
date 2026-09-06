from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from kgproweight.config.schemas import PPOConfig
from kgproweight.data.silver_dataset import SilverTrajectory
from kgproweight.reward.proofkg_process import canonical_token_f1, token_f1
from kgproweight.reward.proofkg_process_v2 import build_execution_trace, score_proofkg_v2
from kgproweight.training.phase3_ppo import (
    Phase3PPOConfig,
    _assert_mixed_text_backend,
    _load_fixed_rollout_schedule,
    _mixed_reward_dataset_diagnostics,
    _mixed_text_batch_diagnostics,
    _prepare_prompts,
    _select_rollout_batch_indices,
    _validate_mixed_reward_config,
    _validate_v21_execution_preflight,
)
from kgproweight.training.reward_function import KGProWeightRewardFunction, RewardSpec


class _Tokenizer:
    def __call__(self, text, **_kwargs):
        return {"input_ids": list(text.encode("utf-8"))}

    def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
        return bytes(int(value) for value in ids).decode("utf-8")


class _FakeRearag:
    name = "rearag"
    is_dummy = False

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0

    def score_steps(self, prompts, step_texts):
        self.calls += 1
        assert len(prompts) == len(step_texts)
        assert len(self.scores) >= len(step_texts)
        return self.scores[: len(step_texts)]


KG = [("Alpha", "links to", "Beta"), ("Beta", "links to", "Gamma")]
PLAN = {
    "hops": [
        {
            "subject": "Alpha", "pids": ["P1"], "output_slot": "hop_1",
            "relation_role": "bridge",
        },
        {
            "subject": "$hop_1", "pids": ["P2"], "output_slot": "hop_2",
            "relation_role": "answer_operand",
        },
    ]
}
EXECUTION = {
    "hops": [
        {"hop_index": 1, "matches": [KG[0]], "output_entities": [{"label": "Beta"}]},
        {"hop_index": 2, "matches": [KG[1]], "output_entities": [{"label": "Gamma"}]},
    ]
}
RUNTIME = {
    "question_key": "2wikimultihopqa::proof-1",
    "query_plan": PLAN,
    "execution": EXECUTION,
    "provenance": {"gold_access": False, "complete_plan_execution": True},
}


def _response(answer: str = "Gamma", *, steps: int = 3) -> str:
    blocks = [
        "[Step 1]\nReasoning: Alpha connects to Beta using the supplied exact proof edge.\n"
        "Knowledge Used: [(Alpha, links to, Beta)]\nConclusion: Alpha connects to Beta.\n",
        "[Step 2]\nReasoning: Beta connects to Gamma using the supplied exact proof edge.\n"
        "Knowledge Used: [(Beta, links to, Gamma)]\nConclusion: Beta connects to Gamma.\n",
        "[Step 3]\nReasoning: The preceding evidence determines the requested final answer clearly.\n"
        "Knowledge Used: []\nConclusion: The answer follows from the evidence.\n",
    ]
    return "".join(blocks[:steps]) + f"[Final Answer] {answer}"


def _reward(
    *, process: bool, mixed_text: bool = False, text_model=None,
) -> KGProWeightRewardFunction:
    # The three legacy components intentionally have no callable methods. Any
    # accidental fallback from the mixed route therefore makes the test fail.
    return KGProWeightRewardFunction(
        alpha_gate=SimpleNamespace(),
        prm_annotator=SimpleNamespace(),
        text_reward_model=text_model or SimpleNamespace(),
        tokenizer=_Tokenizer(),
        outcome_weight=4.0,
        min_valid_steps=3,
        min_reasoning_chars=20,
        proofkg_process_reward=process,
        proofkg_process_version="v2_1",
        proofkg_process_weight=0.2,
        proofkg_f1_weight=0.1,
        proofkg_dynamic_validity=True,
        mixed_outcome_reward=True,
        mixed_text_reward=mixed_text,
        center_text_reward=mixed_text,
        text_reward_scale=0.3,
        text_baseline_momentum=0.99,
    )


def _spec(
    *,
    dataset: str = "2wikimultihopqa",
    qid: str = "proof-1",
    kg=KG,
    runtime=RUNTIME,
    gold: str = "Gamma",
    aliases=None,
) -> RewardSpec:
    return RewardSpec(
        query="What does Alpha ultimately link to?",
        gold_answer=gold,
        kg_subgraph=list(kg),
        metadata={"dataset": dataset, "qid": qid, "question_kg_runtime": runtime},
        gold_answer_aliases=list(aliases or []),
    )


def test_mixed_ppo_k_differs_from_o_only_by_v21_on_identity_safe_proof():
    spec = _spec()
    response = _response(steps=2)  # dynamic validity follows the two-hop proof
    outcome = _reward(process=False)(prompt="", response=response, spec=spec)
    process = _reward(process=True)(prompt="", response=response, spec=spec)
    direct = score_proofkg_v2(
        question=spec.query,
        generation=response,
        kg_triples=KG,
        execution_trace=build_execution_trace(PLAN, EXECUTION),
        planned_hops=2,
    )

    assert outcome["trajectory_reward"] == pytest.approx(4.4)
    assert process["trajectory_reward"] - outcome["trajectory_reward"] == pytest.approx(
        0.2 * direct["score"]
    )
    assert process["proofkg_process"]["eligible"] is True
    assert process["proofkg_process"]["identity_safe"] is True
    assert process["proofkg_process"]["process_applied"] is True
    nonzero = torch.nonzero(process["token_rewards"], as_tuple=False).flatten().tolist()
    assert nonzero == [process["token_rewards"].numel() - 1]


@pytest.mark.parametrize("dataset", ["hotpotqa", "musique", "2wikimultihopqa"])
def test_mixed_ineligible_rows_get_f1_outcome_and_never_legacy_fallback(dataset):
    # Two-token prediction against a three-token answer gives token F1=0.8.
    spec = _spec(
        dataset=dataset,
        qid="ordinary-1",
        kg=[("legacy", "relation", "distractor")],
        runtime={},
        gold="New York City",
    )
    response = _response("New York", steps=3)
    outcome = _reward(process=False)(prompt="", response=response, spec=spec)
    process = _reward(process=True)(prompt="", response=response, spec=spec)

    assert outcome["trajectory_reward"] == pytest.approx(4.0 * (0.0 + 0.1 * 0.8))
    assert process["trajectory_reward"] == pytest.approx(outcome["trajectory_reward"])
    assert process["proofkg_process"]["eligible"] is False
    assert process["proofkg_process"]["process_applied"] is False
    assert process["proofkg_process"]["process_weight"] == 0.0


def test_mixed_identity_mismatch_cannot_receive_process_reward():
    stale = dict(RUNTIME)
    stale["question_key"] = "2wikimultihopqa::another-qid"
    spec = _spec(runtime=stale)
    outcome = _reward(process=False)(prompt="", response=_response(steps=3), spec=spec)
    process = _reward(process=True)(prompt="", response=_response(steps=3), spec=spec)
    assert process["trajectory_reward"] == pytest.approx(outcome["trajectory_reward"])
    assert process["proofkg_process"]["eligible"] is False


def test_mixed_invalid_is_exact_minus_four_in_both_arms_and_hits_final_token():
    response = "[Final Answer] Gamma"
    response_ids = list(response.encode("utf-8"))
    results = [
        _reward(process=process)(
            prompt="", response=response, spec=_spec(), response_ids=response_ids
        )
        for process in (False, True)
    ]
    for result in results:
        assert result["trajectory_valid"] is False
        assert result["trajectory_reward"] == pytest.approx(-4.0)
        assert result["token_rewards"].sum().item() == pytest.approx(-4.0)
        nonzero = torch.nonzero(result["token_rewards"], as_tuple=False).flatten().tolist()
        assert nonzero == [len(response_ids) - 1]


def test_mixed_text_uses_preupdate_causal_ema_and_first_observation_centers_to_zero():
    model = _FakeRearag([0.6, 0.8, 0.4])
    reward = _reward(process=False, mixed_text=True, text_model=model)
    result = reward(prompt="", response=_response(steps=3), spec=_spec(runtime={}))
    telemetry = result["mixed_reward"]

    assert model.calls == 1
    assert telemetry["text_baseline_before_step"] == pytest.approx([0.6, 0.6, 0.602])
    assert telemetry["text_centered_clipped_step_scores"] == pytest.approx(
        [0.0, 0.2, -0.202]
    )
    assert telemetry["text"] == pytest.approx(0.3 * (0.0 + 0.2 - 0.202) / 3)
    assert reward.composite.text_baseline == pytest.approx(0.59998)


@pytest.mark.parametrize("n_steps", [2, 3])
def test_mixed_text_is_clipped_length_normalized_and_placed_at_each_step_end(n_steps):
    model = _FakeRearag([1.0] * n_steps)
    reward = _reward(process=False, mixed_text=True, text_model=model)
    # Force a two-unit residual so every per-step residual exercises clip(+1).
    reward.composite._text_baseline = -1.0
    result = reward(prompt="", response=_response(steps=n_steps), spec=_spec())
    telemetry = result["mixed_reward"]

    assert telemetry["text_centered_clipped_step_scores"] == [1.0] * n_steps
    assert telemetry["text_weighted_step_rewards"] == pytest.approx(
        [0.3 / n_steps] * n_steps
    )
    assert telemetry["text"] == pytest.approx(0.3)
    assert sum(result["per_step_rewards"]) == pytest.approx(result["trajectory_reward"])
    assert result["token_rewards"].sum().item() == pytest.approx(
        result["trajectory_reward"]
    )
    # Every ReaRAG step receives credit at its reasoning span end. The last
    # reasoning credit is strictly before the final-token outcome/KG credit.
    nonzero = torch.nonzero(result["token_rewards"], as_tuple=False).flatten().tolist()
    text_positions = [end - 1 for _start, end in result["text_step_spans"]]
    assert text_positions[-1] < result["token_rewards"].numel() - 1
    assert nonzero == text_positions + [result["token_rewards"].numel() - 1]


def test_mixed_invalid_never_calls_rearag_or_updates_ema():
    model = _FakeRearag([0.9])
    reward = _reward(process=True, mixed_text=True, text_model=model)
    result = reward(prompt="", response="[Final Answer] Gamma", spec=_spec())

    assert result["trajectory_reward"] == -4.0
    assert result["mixed_reward"]["text"] == 0.0
    assert result["mixed_reward"]["process"] == 0.0
    assert model.calls == 0
    assert reward.composite.text_baseline_n_obs == 0
    assert result["proofkg_process"]["process_applied"] is False
    assert result["proofkg_process"]["process_weight"] == 0.0


def test_mixed_ineligible_rows_still_receive_shared_rearag_text_without_legacy_path():
    model = _FakeRearag([0.5, 0.5, 0.5])
    reward = _reward(process=True, mixed_text=True, text_model=model)
    reward.composite._text_baseline = 0.0
    spec = _spec(dataset="hotpotqa", qid="ordinary", kg=[], runtime={})
    result = reward(prompt="", response=_response(steps=3), spec=spec)

    assert model.calls == 1
    assert result["mixed_reward"]["text"] > 0.0
    assert result["mixed_reward"]["process"] == 0.0
    assert result["proofkg_process"]["eligible"] is False
    assert result["proofkg_process"]["process_applied"] is False


def test_paired_text_and_kg_arms_differ_exactly_by_eligible_v21_term():
    response = _response(steps=2)
    text_model = _FakeRearag([0.2, 0.4])
    kg_model = _FakeRearag([0.2, 0.4])
    text_arm = _reward(process=False, mixed_text=True, text_model=text_model)(
        prompt="", response=response, spec=_spec()
    )
    kg_arm = _reward(process=True, mixed_text=True, text_model=kg_model)(
        prompt="", response=response, spec=_spec()
    )
    direct = score_proofkg_v2(
        question=_spec().query,
        generation=response,
        kg_triples=KG,
        execution_trace=build_execution_trace(PLAN, EXECUTION),
        planned_hops=2,
    )

    assert text_arm["mixed_reward"]["outcome"] == kg_arm["mixed_reward"]["outcome"]
    assert text_arm["mixed_reward"]["text"] == kg_arm["mixed_reward"]["text"]
    assert kg_arm["trajectory_reward"] - text_arm["trajectory_reward"] == pytest.approx(
        0.2 * direct["score"]
    )
    assert kg_arm["mixed_reward"]["process"] == pytest.approx(0.2 * direct["score"])


def test_mixed_outcome_uses_canonical_yes_no_f1_guard():
    assert token_f1("not yes", "yes") > 0.0
    assert canonical_token_f1("not yes", "yes") == 0.0
    result = _reward(process=False)(
        prompt="", response=_response("not yes", steps=3),
        spec=_spec(runtime={}, gold="yes"),
    )
    assert result["proofkg_process"]["outcome_em"] == 0.0
    assert result["proofkg_process"]["outcome_f1"] == 0.0
    assert result["trajectory_reward"] == 0.0


@pytest.mark.parametrize(
    ("primary", "aliases", "prediction"),
    [
        (
            "Federal Bureau of Investigation",
            ["Federal Bureau of Investigation", "FBI"],
            "FBI",
        ),
        (
            "People's Republic of China",
            ["People's Republic of China", "PRC", "China"],
            "PRC",
        ),
    ],
)
def test_mixed_outcome_accepts_frozen_canonical_answer_aliases(
    primary, aliases, prediction,
):
    result = _reward(process=False)(
        prompt="",
        response=_response(prediction, steps=3),
        spec=_spec(runtime={}, gold=primary, aliases=aliases),
    )

    assert result["proofkg_process"]["outcome_em"] == 1.0
    assert result["proofkg_process"]["outcome_f1"] == 1.0
    assert result["trajectory_reward"] == pytest.approx(4.4)
    assert result["mixed_reward"]["outcome_em_matched_alias"] == prediction
    assert result["mixed_reward"]["outcome_em_matched_nonprimary"] is True


def test_mixed_outcome_maximises_em_and_f1_independently_across_aliases():
    primary = "completely unrelated"
    f1_alias = "Federal Bureau of Investigation"
    prediction = "Federal Bureau agency"
    result = _reward(process=False)(
        prompt="",
        response=_response(prediction, steps=3),
        spec=_spec(runtime={}, gold=primary, aliases=[f1_alias]),
    )

    # No alias is an exact match, so deterministic EM tie-breaking retains the
    # primary.  F1 independently selects the useful frozen alias.
    assert result["proofkg_process"]["outcome_em"] == 0.0
    assert result["proofkg_process"]["outcome_f1"] == pytest.approx(
        canonical_token_f1(prediction, f1_alias)
    )
    assert result["mixed_reward"]["outcome_em_matched_alias"] == primary
    assert result["mixed_reward"]["outcome_f1_matched_alias"] == f1_alias
    assert result["mixed_reward"]["outcome_em_matched_nonprimary"] is False
    assert result["mixed_reward"]["outcome_f1_matched_nonprimary"] is True


@pytest.mark.parametrize("malformed_aliases", [None, {}, [None, 7, ""]])
def test_mixed_outcome_malformed_or_empty_aliases_fail_safe_to_primary(
    malformed_aliases,
):
    spec = _spec(runtime={}, gold="FBI")
    spec.gold_answer_aliases = malformed_aliases
    result = _reward(process=False)(
        prompt="", response=_response("FBI", steps=3), spec=spec,
    )

    assert result["proofkg_process"]["outcome_em"] == 1.0
    assert result["proofkg_process"]["gold_alias_count"] == 1
    assert result["mixed_reward"]["outcome_em_matched_alias"] == "FBI"


def test_aliases_are_identical_between_paired_mixed_arms():
    spec = _spec(
        dataset="hotpotqa", qid="ordinary", kg=[], runtime={},
        gold="Federal Bureau of Investigation", aliases=["FBI"],
    )
    response = _response("FBI", steps=3)
    text = _reward(process=False)(prompt="", response=response, spec=spec)
    text_kg = _reward(process=True)(prompt="", response=response, spec=spec)

    assert text["trajectory_reward"] == text_kg["trajectory_reward"]
    assert text["proofkg_process"]["outcome_em"] == 1.0
    assert text["mixed_reward"]["outcome_em_matched_alias"] == "FBI"
    assert text_kg["mixed_reward"]["outcome_em_matched_alias"] == "FBI"


def test_historical_proofkg_fast_path_remains_single_gold_when_aliases_exist():
    reward = KGProWeightRewardFunction(
        alpha_gate=SimpleNamespace(),
        prm_annotator=SimpleNamespace(),
        text_reward_model=SimpleNamespace(),
        tokenizer=_Tokenizer(),
        outcome_weight=4.0,
        min_valid_steps=3,
        min_reasoning_chars=20,
        proofkg_outcome_only_reward=True,
        proofkg_dynamic_validity=True,
        proofkg_f1_weight=0.1,
        mixed_outcome_reward=False,
    )
    spec = _spec(
        gold="Federal Bureau of Investigation", aliases=["FBI"],
    )
    result = reward(prompt="", response=_response("FBI", steps=2), spec=spec)

    assert result["proofkg_process"]["outcome_em"] == 0.0
    assert result["proofkg_process"]["outcome_f1"] == 0.0
    assert result["proofkg_process"]["gold_alias_count"] == 1
    assert "mixed_reward" not in result


def test_prepare_prompts_forwards_frozen_alias_metadata_with_primary_fallback():
    with_aliases = _trajectory("with-aliases", eligible=False)
    with_aliases.metadata["gold_answer"] = "Federal Bureau of Investigation"
    with_aliases.metadata["gold_answer_aliases"] = ["FBI", "", 7]
    fallback = _trajectory("fallback", eligible=False)
    fallback.metadata["gold_answer"] = "People's Republic of China"
    fallback.metadata["gold_answer_aliases"] = {"malformed": "PRC"}
    reader = SimpleNamespace(accepted=lambda: [with_aliases, fallback])
    cfg = Phase3PPOConfig(
        silver_path="unused", output_dir="unused", max_input_length=100_000,
    )

    rows = _prepare_prompts(reader, _Tokenizer(), cfg)

    assert rows[0]["spec"].gold_answer_aliases == ["FBI"]
    assert rows[1]["spec"].gold_answer_aliases == ["People's Republic of China"]


def _trajectory(qid: str, *, eligible: bool, execution: bool = True) -> SilverTrajectory:
    runtime = {}
    kg = []
    if eligible:
        kg = list(KG)
        runtime = {
            **RUNTIME,
            "question_key": f"2wikimultihopqa::{qid}",
            **({} if execution else {"execution": {}}),
        }
    return SilverTrajectory(
        qid=qid,
        question=f"Question {qid}?",
        answer="answer",
        dataset="2wikimultihopqa" if eligible else "hotpotqa",
        steps=[],
        kg_subgraph=kg,
        metadata={"gold_answer": "answer", "question_kg_runtime": runtime},
    )


def test_v21_preflight_requires_execution_only_for_eligible_rows():
    partial = _trajectory("partial", eligible=False)
    partial.dataset = "2wikimultihopqa"
    partial.kg_subgraph = [KG[0]]
    partial.metadata["question_kg_runtime"] = {
        "question_key": "2wikimultihopqa::partial",
        "query_plan": PLAN,
        "provenance": {"gold_access": False, "complete_plan_execution": False},
    }
    stats = _validate_v21_execution_preflight(
        [_trajectory("proof", eligible=True), _trajectory("ordinary", eligible=False), partial]
    )
    assert stats == {"eligible_rows": 1, "missing_execution_rows": 0}

    with pytest.raises(ValueError, match="eligible rows"):
        _validate_v21_execution_preflight(
            [_trajectory("proof", eligible=True, execution=False),
             _trajectory("ordinary", eligible=False)]
        )


def test_mixed_config_is_default_off_and_frozen_outcome_is_enforced():
    assert PPOConfig().mixed_outcome_reward is False
    assert PPOConfig().mixed_text_reward is False
    assert Phase3PPOConfig(silver_path="s", output_dir="o").mixed_outcome_reward is False
    assert Phase3PPOConfig(silver_path="s", output_dir="o").mixed_text_reward is False

    good = Phase3PPOConfig(
        silver_path="s", output_dir="o", mixed_outcome_reward=True,
        outcome_weight=4.0, proofkg_f1_weight=0.1,
        proofkg_dynamic_validity=True, proofkg_process_version="v2_1",
        question_kg_records_path="records.jsonl",
    )
    _validate_mixed_reward_config(good)

    with pytest.raises(ValueError, match="alpha_gate_path=null"):
        _validate_mixed_reward_config(
            Phase3PPOConfig(**{**good.__dict__, "alpha_gate_path": "legacy-alpha.pt"})
        )
    with pytest.raises(ValueError, match="alpha_override=null"):
        _validate_mixed_reward_config(
            Phase3PPOConfig(**{**good.__dict__, "alpha_override": 0.5})
        )

    good.outcome_weight = 8.0
    with pytest.raises(ValueError, match="outcome_weight=4.0"):
        _validate_mixed_reward_config(good)

    treatment = Phase3PPOConfig(
        **{
            **good.__dict__,
            "outcome_weight": 4.0,
            "proofkg_process_reward": True,
            "proofkg_process_weight": 0.3,
        }
    )
    with pytest.raises(ValueError, match="proofkg_process_weight=0.20"):
        _validate_mixed_reward_config(treatment)


def test_mixed_text_config_and_built_backend_are_fail_hard():
    good = Phase3PPOConfig(
        silver_path="s", output_dir="o", mixed_outcome_reward=True,
        mixed_text_reward=True, text_reward_backend="rearag",
        outcome_weight=4.0, proofkg_f1_weight=0.1,
        proofkg_dynamic_validity=True, proofkg_process_version="v2_1",
        text_reward_scale=0.3, center_text_reward=True,
        text_baseline_momentum=0.99,
        question_kg_records_path="records.jsonl",
    )
    _validate_mixed_reward_config(good)
    _assert_mixed_text_backend(good, _FakeRearag([0.0]))

    for backend in ("auto", "llama_head", "dummy"):
        bad = Phase3PPOConfig(**{**good.__dict__, "text_reward_backend": backend})
        with pytest.raises(ValueError, match="text_reward_backend=rearag"):
            _validate_mixed_reward_config(bad)
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        _assert_mixed_text_backend(
            good, SimpleNamespace(name="dummy", is_dummy=True)
        )


def test_mixed_reward_telemetry_is_aggregated_by_dataset_and_component():
    rows = [
        {
            "trajectory_valid": True,
            "mixed_reward": {
                "dataset": "hotpotqa", "outcome": 4.0, "text": 0.2,
                "process": 0.0, "total": 4.2, "proofkg_eligible": False,
                "text_raw_step_scores": [1.5, 0.5],
                "text_baseline_before_step": [0.0, 0.0],
                "text_centered_clipped_step_scores": [1.0, 0.5],
                "text_ema_baseline": 0.6, "text_ema_n_obs": 2,
                "outcome_em_matched_nonprimary": True,
                "outcome_f1_matched_nonprimary": True,
            },
        },
        {
            "trajectory_valid": False,
            "mixed_reward": {
                "dataset": "hotpotqa", "outcome": -4.0, "text": 0.0,
                "process": 0.0, "total": -4.0, "proofkg_eligible": False,
                "text_raw_step_scores": [], "text_baseline_before_step": [],
                "text_centered_clipped_step_scores": [],
                "text_ema_baseline": 0.6, "text_ema_n_obs": 2,
            },
        },
        {
            "trajectory_valid": True,
            "mixed_reward": {
                "dataset": "2wikimultihopqa", "outcome": 4.4, "text": 0.1,
                "process": 0.2, "total": 4.7, "proofkg_eligible": True,
                "text_raw_step_scores": [-0.2],
                "text_baseline_before_step": [0.0],
                "text_centered_clipped_step_scores": [-0.2],
                "text_ema_baseline": -0.1, "text_ema_n_obs": 3,
            },
        },
    ]
    telemetry = _mixed_reward_dataset_diagnostics(rows)

    assert list(telemetry) == ["2wikimultihopqa", "hotpotqa"]
    hotpot = telemetry["hotpotqa"]
    assert hotpot["count"] == 2
    assert hotpot["valid_count"] == 1
    assert hotpot["valid_rate"] == pytest.approx(0.5)
    assert hotpot["proofkg_eligible_count"] == 0
    assert hotpot["proofkg_eligible_rate"] == 0.0
    assert hotpot["outcome_mean"] == 0.0
    assert hotpot["text_mean"] == pytest.approx(0.1)
    assert hotpot["process_mean"] == 0.0
    assert hotpot["total_mean"] == pytest.approx(0.1)
    assert hotpot["text_step_count"] == 2
    assert hotpot["text_raw_step_mean"] == pytest.approx(1.0)
    assert hotpot["text_centered_step_mean"] == pytest.approx(0.75)
    assert hotpot["text_centered_abs_mean"] == pytest.approx(0.75)
    assert hotpot["text_clip_frac"] == pytest.approx(0.5)
    assert hotpot["text_ema_baseline"] == pytest.approx(0.6)
    assert hotpot["text_ema_n_obs"] == 2
    assert hotpot["em_matched_nonprimary_count"] == 1
    assert hotpot["f1_matched_nonprimary_count"] == 1
    assert telemetry["2wikimultihopqa"]["process_mean"] == pytest.approx(0.2)

    batch = _mixed_text_batch_diagnostics(telemetry)
    assert batch["mixed_text_step_count"] == 3
    assert batch["mixed_text_raw_step_mean"] == pytest.approx(0.6)
    assert batch["mixed_text_centered_step_mean"] == pytest.approx(1.3 / 3)
    assert batch["mixed_text_centered_abs_mean"] == pytest.approx(1.7 / 3)
    assert batch["mixed_text_clip_frac"] == pytest.approx(1 / 3)
    assert batch["mixed_text_ema_baseline"] == pytest.approx(-0.1)
    assert batch["mixed_text_ema_n_obs"] == 3


def test_fixed_schedule_is_resolved_and_selected_in_exact_k4_batches(tmp_path):
    population = [
        _trajectory("proof", eligible=True),
        _trajectory("ordinary", eligible=False),
    ]
    rows = []
    for rollout_index, qid in enumerate(["proof"] * 4 + ["ordinary"] * 4, start=1):
        rows.append({
            "rollout_index": rollout_index,
            "dataset": "2wikimultihopqa" if qid == "proof" else "hotpotqa",
            "qid": qid,
        })
    path = tmp_path / "schedule.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    indices, loaded = _load_fixed_rollout_schedule(
        path, population, total_steps=8, rollouts_per_prompt=4,
    )
    assert indices == [0, 0, 0, 0, 1, 1, 1, 1]
    assert loaded == rows
    rng = torch.Generator().manual_seed(42)
    assert _select_rollout_batch_indices(
        population_size=2, batch_size=4, rollouts_per_prompt=4,
        generator=rng, fixed_indices=indices, offset=4,
    ) == [1, 1, 1, 1]


def test_fixed_schedule_rejects_broken_k4_group(tmp_path):
    population = [_trajectory("proof", eligible=True), _trajectory("ordinary", eligible=False)]
    identities = [
        ("2wikimultihopqa", "proof"),
        ("2wikimultihopqa", "proof"),
        ("hotpotqa", "ordinary"),
        ("2wikimultihopqa", "proof"),
    ]
    path = tmp_path / "broken.jsonl"
    path.write_text(
        "".join(
            json.dumps({"rollout_index": i, "dataset": dataset, "qid": qid}) + "\n"
            for i, (dataset, qid) in enumerate(identities, start=1)
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="breaks K=4 grouping"):
        _load_fixed_rollout_schedule(
            path, population, total_steps=4, rollouts_per_prompt=4,
        )
