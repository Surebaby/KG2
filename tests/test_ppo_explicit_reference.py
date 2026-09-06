"""Regression tests for the explicit SFT reference under TRL 0.11.4 + PEFT."""

from __future__ import annotations

import torch

from kgproweight.training.phase3_ppo import _measure_explicit_reference_kl
from kgproweight.training.step_reward_ppo_trainer import StepRewardPPOTrainer


def _tiny_peft_policy_and_tokenizer():
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from trl import AutoModelForCausalLMWithValueHead

    backend = Tokenizer(models.WordLevel(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.add_tokens(["<unk>", "<pad>", "<eos>"] + [f"w{i}" for i in range(61)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        pad_token_id=tokenizer.pad_token_id,
    )
    base = LlamaForCausalLM(config)
    peft_model = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    # LoRA B initializes to zero, which would make enabled and disabled adapters
    # identical and let the historical bare-base shortcut escape this test.
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_A" in name:
                param.fill_(0.05)
            elif "lora_B" in name:
                param.fill_(0.10)
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model)
    return policy, tokenizer


def test_peft_policy_uses_explicit_reference_instead_of_bare_base():
    from trl import PPOConfig, create_reference_model

    torch.manual_seed(7)
    policy, tokenizer = _tiny_peft_policy_and_tokenizer()
    reference = create_reference_model(policy)
    config = PPOConfig(
        learning_rate=1e-5,
        batch_size=2,
        mini_batch_size=1,
        ppo_epochs=1,
        adap_kl_ctrl=True,
        init_kl_coef=0.15,
        target=40.0,
        log_with=None,
        seed=42,
    )
    trainer = StepRewardPPOTrainer(
        config=config,
        model=policy,
        ref_model=reference,
        tokenizer=tokenizer,
    )

    assert trainer._policy_is_peft_model is True
    assert trainer._uses_explicit_reference is True
    assert trainer.ref_model is not None
    # This is TRL's reference-dispatch switch. The wrapped model remains PEFT,
    # so LoRA parameters are still trainable and save_pretrained still saves it.
    assert trainer.is_peft_model is False
    assert trainer.model.is_peft_model is True

    queries = [torch.tensor([3, 4]), torch.tensor([5, 6])]
    responses = [torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])]
    model_inputs = trainer.prepare_model_inputs(queries, responses)
    with torch.no_grad():
        policy_lp, _, _, masks = trainer.batched_forward_pass(
            trainer.model, queries, responses, model_inputs,
        )
        reference_lp, _, _, _ = trainer.batched_forward_pass(
            trainer.ref_model, queries, responses, model_inputs,
        )
        with trainer.model.pretrained_model.disable_adapter():
            bare_base_lp, _, _, _ = trainer.batched_forward_pass(
                trainer.model, queries, responses, model_inputs,
            )

    response_mask = masks.bool()
    assert torch.equal(policy_lp[response_mask], reference_lp[response_mask])
    assert not torch.allclose(
        policy_lp[response_mask], bare_base_lp[response_mask], atol=1e-6, rtol=0.0,
    )
    assert _measure_explicit_reference_kl(trainer, queries, responses) == 0.0

    # objective/kl is measured before PPO's optimiser update. It therefore must
    # be zero on the first step when policy and explicit reference are copies.
    trainer.set_pending_step_rewards([torch.zeros(3), torch.zeros(3)])
    stats = trainer.step(queries, responses, [torch.zeros(()), torch.zeros(())])
    assert abs(float(stats["objective/kl"])) < 1e-6


def test_explicit_reference_config_reverts_kl_and_replay_to_ce010_baseline():
    from pathlib import Path

    from kgproweight.config import ProjectConfig, load_config

    root = Path(__file__).resolve().parents[1]
    baseline = load_config(
        root / "configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600.yaml",
        validate=ProjectConfig,
    )
    fixed = load_config(
        root / "configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_explicit_sft_ref.yaml",
        validate=ProjectConfig,
    )
    old = baseline.model_dump() if hasattr(baseline, "model_dump") else baseline.dict()
    new = fixed.model_dump() if hasattr(fixed, "model_dump") else fixed.dict()

    assert new["training"]["ppo"]["kl_coef"] == 0.15
    assert new["training"]["ppo"]["sft_replay_ratio"] == 0.10
    assert new["training"]["ppo"]["sft_anchor_weight"] == 0.10
    new["training"]["output_dir"] = old["training"]["output_dir"]
    assert new == old


def test_v2_trl_mask_includes_true_eos_when_eos_equals_padding():
    from trl import PPOConfig, create_reference_model

    policy, tokenizer = _tiny_peft_policy_and_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    trainer = StepRewardPPOTrainer(
        config=PPOConfig(batch_size=2, mini_batch_size=1, ppo_epochs=1, log_with=None),
        model=policy, ref_model=create_reference_model(policy), tokenizer=tokenizer,
        runtime_contract_version="v2",
    )
    eos = tokenizer.eos_token_id
    queries = [torch.tensor([3, 4]), torch.tensor([5, 6, 7])]
    responses = [torch.tensor([8, 9, eos]), torch.tensor([10, eos])]
    inputs = trainer.prepare_model_inputs(queries, responses)
    with torch.no_grad():
        logprobs, _, _, masks = trainer.batched_forward_pass(
            trainer.model, queries, responses, inputs,
        )
        ref_logprobs, _, _, _ = trainer.batched_forward_pass(
            trainer.ref_model, queries, responses, inputs,
        )
    assert masks.sum(dim=1).tolist() == [3, 2]
    trainer.set_pending_step_rewards([torch.tensor([0., 0., 4.4]), torch.tensor([0., -4.])])
    rewards, non_score, _ = trainer.compute_rewards([0., 0.], logprobs, ref_logprobs, masks)
    for index, expected in enumerate([4.4, -4.]):
        positions = masks[index].nonzero().flatten()
        assert torch.isclose(rewards[index, positions[-1]] - non_score[index, positions[-1]], torch.tensor(expected))
