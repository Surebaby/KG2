"""Opt-in PPO generation/EOS/reward contracts; no research-model updates."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kgproweight.training.phase3_ppo import (
    Phase3PPOConfig,
    _align_token_rewards,
    _generate,
    _response_is_length_capped_v2,
    _rollout_eos_token_ids,
    _runtime_contract_v2,
    _trim_response_v2,
)
from kgproweight.training.step_reward_ppo_trainer import StepRewardPPOTrainer


@pytest.fixture(scope="module")
def strong_sft_tokenizer():
    from transformers import AutoTokenizer

    path = Path(__file__).resolve().parents[1] / (
        "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"
    )
    if not (path / "tokenizer.json").exists():
        pytest.skip("local frozen Strong SFT tokenizer is unavailable")
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


class _GenerationProbe(torch.nn.Module):
    def __init__(self, tokenizer, *, fail=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.dropout = torch.nn.Dropout(0.05)
        self.selectively_frozen = torch.nn.Dropout(0.1)
        self.selectively_frozen.eval()
        self.tokenizer = tokenizer
        other_eos = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
        self.generation_config = SimpleNamespace(eos_token_id=[tokenizer.eos_token_id, other_eos])
        self.fail = fail
        self.calls = []

    def generate(self, input_ids, **kwargs):
        self.calls.append({
            "training": self.training,
            "dropout_training": self.dropout.training,
            "grad_enabled": torch.is_grad_enabled(),
            "pad_token_id": kwargs["pad_token_id"],
            "eos_token_id": kwargs.get("eos_token_id"),
        })
        if self.fail:
            raise RuntimeError("synthetic generation failure")
        continuations = []
        for row in range(input_ids.shape[0]):
            text = "Yes." if row == 0 else "A longer answer."
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            ids.append(self.generation_config.eos_token_id[row % 2])
            continuations.append(ids)
        width = max(map(len, continuations))
        suffix = torch.tensor([
            ids + [kwargs["pad_token_id"]] * (width - len(ids)) for ids in continuations
        ])
        return torch.cat([input_ids, suffix], dim=1)


def _cfg(**kwargs):
    return Phase3PPOConfig(
        silver_path="unused", output_dir="unused", runtime_contract_version="v2",
        rollout_chunk_size=2, max_input_length=256, max_new_tokens=16,
        use_real_logprobs=False, **kwargs,
    )


@pytest.mark.parametrize("initial_training", [False, True])
def test_real_tokenizer_generation_preserves_eos_and_restores_modes(strong_sft_tokenizer, initial_training):
    tokenizer = strong_sft_tokenizer
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    policy = _GenerationProbe(tokenizer)
    policy.train(initial_training)
    policy.selectively_frozen.eval()
    original_padding = tokenizer.padding_side
    prompts = ["Hello", "A longer prompt for the same reader."]
    queries, responses, texts, _ = _generate(policy, tokenizer, prompts, _cfg(), "cpu")
    assert policy.training is initial_training
    assert policy.dropout.training is initial_training
    assert policy.selectively_frozen.training is False
    assert tokenizer.padding_side == original_padding
    assert policy.calls[0]["training"] is False
    assert policy.calls[0]["dropout_training"] is False
    assert policy.calls[0]["grad_enabled"] is False
    assert policy.calls[0]["eos_token_id"] == policy.generation_config.eos_token_id
    for index, (query, response) in enumerate(zip(queries, responses)):
        expected_query = tokenizer(prompts[index], add_special_tokens=False)["input_ids"]
        assert query.tolist() == expected_query
        assert int(response[-1]) == policy.generation_config.eos_token_id[index]
        assert not _response_is_length_capped_v2(
            response, max_new_tokens=16, eos_token_ids=policy.generation_config.eos_token_id,
        )
    assert texts == ["Yes.", "A longer answer."]


def test_generation_failure_restores_modes(strong_sft_tokenizer):
    policy = _GenerationProbe(strong_sft_tokenizer, fail=True)
    assert policy.training is True and policy.selectively_frozen.training is False
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        _generate(policy, strong_sft_tokenizer, ["Hello"], _cfg(), "cpu")
    assert policy.training is True and policy.selectively_frozen.training is False


@pytest.mark.parametrize(
    "ids,eos,pad,expected,capped",
    [
        ([4, 5, 2, 2], [2], 2, [4, 5, 2], False),
        ([2, 2, 2, 2], [2], 2, [2], False),
        ([4, 5, 8, 0], [7, 8], 0, [4, 5, 8], False),
        ([4, 5, 6, 8], [7, 8], 0, [4, 5, 6, 8], False),
        ([4, 5, 6, 9], [7, 8], 0, [4, 5, 6, 9], True),
        ([4, 5, 6, 0], [7, 8], 0, [4, 5, 6, 0], True),
    ],
)
def test_eos_and_length_cap_are_distinct(ids, eos, pad, expected, capped):
    response = _trim_response_v2(torch.tensor(ids), eos_token_ids=eos, pad_token_id=pad, max_new_tokens=4)
    assert response.tolist() == expected
    assert _response_is_length_capped_v2(response, max_new_tokens=4, eos_token_ids=eos) is capped


def test_nonpadding_after_true_eos_fails():
    with pytest.raises(ValueError, match="after the first EOS"):
        _trim_response_v2(torch.tensor([4, 2, 5]), eos_token_ids=[2], pad_token_id=2, max_new_tokens=4)


def test_effective_eos_prefers_generation_config_and_runtime_is_opt_in():
    policy = SimpleNamespace(pretrained_model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[7, 8])))
    assert _rollout_eos_token_ids(policy, SimpleNamespace(eos_token_id=2)) == (7, 8)
    assert _runtime_contract_v2(Phase3PPOConfig(silver_path="", output_dir="")) is False
    with pytest.raises(ValueError, match="runtime_contract_version"):
        _runtime_contract_v2(SimpleNamespace(runtime_contract_version="typo"))


def _trainer():
    trainer = object.__new__(StepRewardPPOTrainer)
    trainer._runtime_contract_version = "v2"
    trainer._require_step_rewards = True
    trainer._pending_step_rewards = None
    trainer.config = SimpleNamespace(batch_size=2)
    trainer.kl_ctl = SimpleNamespace(value=0.25)
    trainer._kl_penalty = lambda policy_lp, reference_lp: policy_lp - reference_lp
    return trainer


def test_scatter_conserves_every_response_reward_and_preserves_kl_channel():
    trainer = _trainer()
    pending = [torch.tensor([0.1, -0.2, 4.4]), torch.tensor([0.3, -4.0])]
    trainer.set_pending_step_rewards(pending)
    logprobs = torch.arange(12).float().reshape(2, 6) / 10
    ref = torch.ones_like(logprobs) * 0.2
    masks = torch.tensor([[0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 0, 0]])
    rewards, non_score, kls = trainer.compute_rewards([0., 0.], logprobs, ref, masks)
    assert torch.equal(kls, logprobs - ref)
    assert torch.equal(non_score, -0.25 * kls)
    for index in range(2):
        added = rewards[index] - non_score[index]
        assert torch.allclose(added[masks[index].bool()], pending[index], atol=1e-6)
        assert float(added.sum()) == pytest.approx(float(pending[index].sum()), abs=1e-6)
        assert bool((added[~masks[index].bool()] == 0).all())
    assert trainer._pending_step_rewards is None
    with pytest.raises(RuntimeError, match="without pending"):
        trainer.compute_rewards([0., 0.], logprobs, ref, masks)


@pytest.mark.parametrize(
    "first_mask,match",
    [([0, 0, 1, 1, 0], "length mismatch"), ([0, 1, 0, 1, 1], "not contiguous"),
     ([0, 0.5, 1, 1, 0], "not binary"), ([0, 0, 0, 0, 0], "no generated tokens")],
)
def test_bad_response_masks_fail_hard(first_mask, match):
    trainer = _trainer()
    trainer.set_pending_step_rewards([torch.ones(3), torch.ones(3)])
    masks = torch.tensor([first_mask, [0, 1, 1, 1, 0]])
    with pytest.raises(ValueError, match=match):
        trainer.compute_rewards([0., 0.], torch.zeros(2, 5), torch.zeros(2, 5), masks)


def test_batch_and_pending_shapes_cannot_be_silently_truncated():
    trainer = _trainer()
    with pytest.raises(ValueError, match="nonempty vector"):
        trainer.set_pending_step_rewards([torch.ones(1, 3), torch.ones(3)])
    with pytest.raises(ValueError, match="non-finite"):
        trainer.set_pending_step_rewards([torch.tensor([float("nan")]), torch.ones(3)])
    trainer.set_pending_step_rewards([torch.ones(3), torch.ones(3)])
    with pytest.raises(ValueError, match="batch length mismatch"):
        trainer.compute_rewards([0.], torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2, 3))


def test_reference_broadcast_and_nonfinite_kl_penalty_fail_before_gae():
    trainer = _trainer()
    trainer.set_pending_step_rewards([torch.ones(3), torch.ones(3)])
    with pytest.raises(ValueError, match="policy/reference token shape mismatch"):
        trainer.compute_rewards([0., 0.], torch.zeros(2, 3), torch.zeros(2, 1), torch.ones(2, 3))
    trainer.kl_ctl.value = float("nan")
    with pytest.raises(ValueError, match="non-score reward"):
        trainer.compute_rewards([0., 0.], torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2, 3))


def test_reward_response_lengths_and_trajectory_sum_are_strict_only_in_v2():
    response = torch.tensor([4, 5, 2])
    token_rewards = torch.tensor([0.1, 0.2, 4.4])
    assert _align_token_rewards(token_rewards, response, trajectory_reward=4.7, runtime_contract_version="v2") is token_rewards
    with pytest.raises(ValueError, match="length mismatch"):
        _align_token_rewards(torch.zeros(2), response, trajectory_reward=0., runtime_contract_version="v2")
    with pytest.raises(ValueError, match="conservation"):
        _align_token_rewards(token_rewards, response, trajectory_reward=5., runtime_contract_version="v2")
    with pytest.raises(ValueError, match="non-finite"):
        _align_token_rewards(torch.tensor([0., 0., float("inf")]), response, trajectory_reward=4., runtime_contract_version="v2")
    assert _align_token_rewards(torch.zeros(2), response, trajectory_reward=0.).numel() == 3
