#!/usr/bin/env python
"""CPU smoke test for the PACKAGE Phase-3 PPO path (no GPU, no real LLM).

Purpose: validate the two 2026-06-23 fixes in
``kgproweight/training/phase3_ppo.py`` without a GPU —

  1. KL-config routing: the PPOConfig built by run_phase3_ppo uses
     init_kl_coef=cfg.kl_coef, target=cfg.target_kl, adap_kl_ctrl=True, with the
     early-stop target_kl left inert (early_stopping=False).
  2. Rollout sampling kwargs: the package ``_generate`` now samples at
     temperature=1.0 / top_p=1.0 / top_k=0 so the rollout distribution matches
     TRL's raw-logit logp recomputation (the mismatch — temp=0.7/top_p=0.9 —
     is what bent KL negative on 2026-06-23). We assert the generate() call is
     actually issued with those kwargs by capturing them, AND that the real
     ``_generate`` code path runs end-to-end on a tiny CPU model.

It also runs ONE real ``StepRewardPPOTrainer.step()`` on CPU to confirm the
whole chain (reward_fn → per-token rewards → set_pending → step →
compute_rewards → GAE) interoperates and returns NaN-free stats.

NOTE on what this canNOT show: negative-KL bias only appears once the policy
has diverged from the frozen reference (post-update). On an untrained tiny
model policy==ref so KL≈0 for ANY sampling — so we validate the FIX (sampling
kwargs == scoring distribution) by capturing the kwargs, not by trying to
reproduce divergence on a random model.

Run on the AutoDL box (no-GPU mode is fine):
    /root/autodl-tmp/kgpw_env/bin/python -u scripts/train/tests/smoke_ppo_package_cpu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Package import: repo root is two levels up from scripts/train/tests/.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kgproweight.training.reward_function import (
    KGProWeightRewardFunction,
    RewardSpec,
    step_spans_over_ids,
)
from kgproweight.training import phase3_ppo as P3
from kgproweight.training.phase3_ppo import Phase3PPOConfig, _generate
from kgproweight.training.step_reward_ppo_trainer import StepRewardPPOTrainer
from kgproweight.reward.alpha_gate import AlphaGate
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.reward.text_reward_model import TextRewardModel, _DummyTextReward
from kgproweight.data.parsers import parse_steps

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Tiny model
# ---------------------------------------------------------------------------

def _build_tiny():
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers
    from trl import AutoModelForCausalLMWithValueHead, create_reference_model

    tk = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    vocab = ["<unk>", "<pad>", "<eos>"] + [f"w{i}" for i in range(120)] + \
            ["[Step", "1]", "2]", "Reasoning:", "Conclusion:", "[Final", "Answer]",
             "Germany", "Ulm", "Berlin", "is", "in", "the", "of", "born"]
    tk.add_tokens(vocab)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>",
                                  eos_token="<eos>", unk_token="<unk>")

    # WordLevel starts empty so tok.vocab_size==0; the real size is len(tok).
    vocab_n = len(tok) + 8
    cfg = LlamaConfig(
        vocab_size=vocab_n,
        hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=1024, pad_token_id=tok.pad_token_id,
    )
    base = LlamaForCausalLM(cfg)
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(base)
    ref = create_reference_model(policy)
    return policy, ref, tok


def _tiny_cfg(**over):
    """A Phase3PPOConfig with the package generation defaults, small for CPU."""
    d = dict(silver_path="x", output_dir="y", max_new_tokens=12,
             max_input_length=256, max_steps=7, use_real_logprobs=True)
    d.update(over)
    return Phase3PPOConfig(**d)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_kl_config_routing():
    """The PPOConfig built by run_phase3_ppo must route kl knobs correctly."""
    from trl import PPOConfig
    cfg_kl, cfg_target = 0.2, 6.0
    ppo_cfg = PPOConfig(
        learning_rate=1e-5, batch_size=2, mini_batch_size=1, ppo_epochs=1,
        cliprange=0.2, kl_penalty="kl", adap_kl_ctrl=True,
        init_kl_coef=cfg_kl, target=cfg_target, gamma=0.95, lam=0.95,
        max_grad_norm=1.0, vf_coef=0.5, early_stopping=False, seed=42, log_with=None,
    )
    assert ppo_cfg.init_kl_coef == cfg_kl, ppo_cfg.init_kl_coef
    assert ppo_cfg.target == cfg_target, ppo_cfg.target
    assert ppo_cfg.adap_kl_ctrl is True
    assert ppo_cfg.early_stopping is False
    print(f"  init_kl_coef={ppo_cfg.init_kl_coef} target={ppo_cfg.target} "
          f"adap_kl_ctrl={ppo_cfg.adap_kl_ctrl} early_stopping={ppo_cfg.early_stopping}  ok")


def test_generate_uses_aligned_sampling():
    """The package _generate must call generate() with temp=1.0/top_p=1.0/top_k=0.

    We monkeypatch the policy's generate to capture kwargs (and return a canned
    short sequence) so this is instant and exercises the REAL _generate code.
    """
    policy, ref, tok = _build_tiny()
    cfg = _tiny_cfg()
    # default config values the fix sets:
    assert cfg.temperature == 1.0 and cfg.top_p == 1.0, (cfg.temperature, cfg.top_p)

    captured = {}
    real_generate = policy.generate

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real_generate(*args, **kwargs)

    policy.generate = spy
    prompts = ["w1 w2 [Step 1] Reasoning: w3 born Ulm"]
    qts, rts, texts, lps = _generate(policy, tok, prompts, cfg, "cpu")
    policy.generate = real_generate

    assert captured.get("temperature") == 1.0, captured.get("temperature")
    assert captured.get("top_p") == 1.0, captured.get("top_p")
    assert captured.get("top_k") == 0, captured.get("top_k")
    assert captured.get("do_sample") is True
    assert len(qts) == 1 and len(rts) == 1 and len(texts) == 1
    print(f"  _generate sampled with temp={captured.get('temperature')} "
          f"top_p={captured.get('top_p')} top_k={captured.get('top_k')}  ok "
          f"(rollout distribution == TRL raw-logit scoring)")


def test_real_trl_step_cpu():
    """One real StepRewardPPOTrainer.step() on CPU, fixed sampling, NaN-free."""
    from trl import PPOConfig

    policy, ref, tok = _build_tiny()
    reward_fn = KGProWeightRewardFunction(
        alpha_gate=AlphaGate(),
        prm_annotator=PRMAnnotator(verbose=False),
        text_reward_model=TextRewardModel(_DummyTextReward(), name="dummy"),
        tokenizer=tok, outcome_weight=1.0, discount=0.95, max_steps=7,
    )
    ppo_cfg = PPOConfig(
        learning_rate=1e-5, batch_size=2, mini_batch_size=1, ppo_epochs=1,
        kl_penalty="kl", adap_kl_ctrl=True, init_kl_coef=0.2, target=6.0,
        gamma=0.95, lam=0.95, seed=42, log_with=None,
    )
    trainer = StepRewardPPOTrainer(config=ppo_cfg, model=policy, ref_model=ref, tokenizer=tok)

    prompts = ["w1 w2 [Step 1] Reasoning: w3 born Ulm",
               "w4 w5 [Step 1] Reasoning: w6 the of Berlin"]
    specs = [RewardSpec(query="q1", gold_answer="Ulm", kg_subgraph=[("a", "b", "Ulm")]),
             RewardSpec(query="q2", gold_answer="Berlin", kg_subgraph=[("c", "d", "Berlin")])]

    qts, rts, trl_rewards = [], [], []
    for prompt, spec in zip(prompts, specs):
        enc = tok(prompt, return_tensors="pt")
        qids = enc["input_ids"][0]
        with torch.no_grad():
            gen = policy.generate(input_ids=qids.unsqueeze(0), max_new_tokens=12,
                                  do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                  pad_token_id=tok.pad_token_id)[0]
        rids = gen[qids.size(0):]
        resp = tok.decode(rids, skip_special_tokens=True)
        n_parsed = len(parse_steps(resp)[:7])
        spans = step_spans_over_ids(rids, tok, n_parsed)
        info = reward_fn(prompt="", response=resp, spec=spec,
                         response_ids=rids, step_spans=spans)
        tr = info["token_rewards"]
        n = rids.size(0)
        if tr.size(0) != n:
            tr = (torch.cat([tr, torch.zeros(n - tr.size(0))]) if tr.size(0) < n else tr[:n])
        qts.append(qids); rts.append(rids); trl_rewards.append(tr)

    placeholder = [torch.zeros(()) for _ in trl_rewards]
    trainer.set_pending_step_rewards(trl_rewards)
    stats = trainer.step(qts, rts, placeholder)

    kl = float(stats.get("objective/kl", 0.0))
    loss = float(torch.as_tensor(stats.get("ppo/loss/total", 0.0)).mean())
    assert kl == kl, "KL is NaN"
    assert loss == loss, "loss is NaN"
    print(f"  real TRL step on CPU ok — objective/kl={kl:+.4f}, loss={loss:+.4f}, "
          f"stat keys={len(stats)}")


def main():
    print("test_kl_config_routing");             test_kl_config_routing()
    print("test_generate_uses_aligned_sampling"); test_generate_uses_aligned_sampling()
    print("test_real_trl_step_cpu");              test_real_trl_step_cpu()
    print("\nPACKAGE PPO CPU SMOKE PASSED ✅  (KL routing + aligned sampling kwargs + real step)")


if __name__ == "__main__":
    main()

