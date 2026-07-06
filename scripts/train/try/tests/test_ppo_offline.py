#!/usr/bin/env python
"""Offline tests for the try-variant PPO reward + trainer — NO API/GPU/model.

Covers:
  (a) per-token reward tensor places each step's R_total on the step-end token;
  (b) the EM outcome bonus lands on the last step (vs real gold);
  (c) gold is read from metadata, not the teacher answer;
  (d) StepRewardPPOTrainer.compute_rewards distributes per-token step rewards
      over the masked region AND still adds the per-token KL penalty, with the
      right output shape.

Run:
    python scripts/train/try/test_ppo_offline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make sibling try-modules importable regardless of CWD/subdir layout.
_TRY_ROOT = Path(__file__).resolve().parent.parent
for _d in (_TRY_ROOT, _TRY_ROOT / "shared", _TRY_ROOT / "phase1_distill",
           _TRY_ROOT / "phase2_prm", _TRY_ROOT / "phase3_sft", _TRY_ROOT / "phase3_ppo"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from ppo_reward_try import ImprovedRewardFunction, RewardSpec
from prm_annotator_try import ImprovedPRMAnnotator
from kgproweight.reward.alpha_gate import AlphaGate


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _WordTokenizer:
    """Reversible whitespace tokenizer with offset mapping.

    decode() must round-trip (extract_step_token_spans re-finds ``[Step N]``
    markers in the *decoded* text), so we keep an id→token vocab and join with
    single spaces. That reproduces the markers and lets spans split per step.
    """

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self._vocab = {}      # token -> id
        self._inv = {}        # id -> token
        self._next = 2

    def _id(self, tok):
        if tok not in self._vocab:
            self._vocab[tok] = self._next
            self._inv[self._next] = tok
            self._next += 1
        return self._vocab[tok]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kw):
        ids, offs = [], []
        i = 0
        for tok in text.split(" "):
            if tok == "":
                i += 1
                continue
            start = text.index(tok, i)
            end = start + len(tok)
            ids.append(self._id(tok))
            offs.append((start, end))
            i = end
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self._inv.get(int(i), "") for i in ids)


class _FakeTextReward:
    """Constant positive text reward so R_total is non-trivial and predictable."""

    def score_step(self, prompt, step_text):  # noqa: ARG002
        return 0.5


_RESPONSE = """[Step 1]
Reasoning: Einstein was born somewhere.
Knowledge Used: [(Albert Einstein, place of birth, Ulm)]
Conclusion: Einstein was born in Ulm.

[Step 2]
Reasoning: Ulm is in a country.
Knowledge Used: [(Ulm, country, Germany)]
Conclusion: Ulm is in Germany.

[Final Answer]
Germany"""

_KG = [
    ("Albert Einstein", "place of birth", "Ulm"),
    ("Ulm", "country", "Germany"),
    ("Germany", "capital", "Berlin"),
]


def _make_reward_fn():
    return ImprovedRewardFunction(
        alpha_gate=AlphaGate(),
        prm_annotator=ImprovedPRMAnnotator(verbose=False),
        text_reward_model=_FakeTextReward(),
        tokenizer=_WordTokenizer(),
        outcome_weight=1.0,
        discount=0.95,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_token_rewards_and_outcome():
    fn = _make_reward_fn()
    spec = RewardSpec(query="Where was Einstein born?", gold_answer="Germany", kg_subgraph=_KG)
    info = fn(prompt="", response=_RESPONSE, spec=spec)

    per_step = info["per_step_rewards"]
    tok = info["token_rewards"]
    spans = info["step_spans"]
    assert len(per_step) == 2, per_step
    # (a) non-zero reward only on step-end tokens.
    nz = (tok != 0).nonzero().squeeze(-1).tolist()
    end_tokens = [min(e - 1, tok.size(0) - 1) for (_s, e) in spans]
    assert set(nz).issubset(set(end_tokens)), (nz, end_tokens)
    # (b) outcome: pred "Germany" == gold "Germany" → last step gets +1 outcome.
    #     last step r_total should exceed a plausible no-outcome ceiling (α·1+1).
    assert per_step[-1] > 1.0, per_step
    print(f"  per_step={[round(x,3) for x in per_step]}  end_tokens={end_tokens}  ok")


def test_gold_from_metadata_not_teacher():
    """Outcome must use the spec.gold_answer the entrypoint fills from metadata."""
    fn = _make_reward_fn()
    # Wrong gold → no EM → last-step reward stays modest; correct gold → +1.
    spec_wrong = RewardSpec(query="q", gold_answer="France", kg_subgraph=_KG)
    spec_right = RewardSpec(query="q", gold_answer="Germany", kg_subgraph=_KG)
    r_wrong = fn(prompt="", response=_RESPONSE, spec=spec_wrong)["per_step_rewards"][-1]
    r_right = fn(prompt="", response=_RESPONSE, spec=spec_right)["per_step_rewards"][-1]
    assert r_right - r_wrong == 1.0 or r_right > r_wrong, (r_wrong, r_right)
    print(f"  last-step reward: wrong-gold={r_wrong:.3f}  right-gold={r_right:.3f}  ok")


def test_compute_rewards_override():
    """Exercise the override's math directly (no real PPOTrainer needed)."""
    from ppo_trainer_try import StepRewardPPOTrainer

    # Build an unbound-style call: make a bare object with the attributes the
    # method touches (kl_ctl.value, _kl_penalty, config.kl_penalty).
    class _KLCtl:
        value = 0.2

    class _Cfg:
        kl_penalty = "kl"

    fake = StepRewardPPOTrainer.__new__(StepRewardPPOTrainer)
    fake.kl_ctl = _KLCtl()
    fake.config = _Cfg()

    T = 5
    logprob = torch.full((T,), -1.0)
    ref_logprob = torch.full((T,), -1.2)  # kl = logprob-ref = +0.2 per token
    mask = torch.tensor([0, 1, 1, 1, 0])  # response tokens at positions 1,2,3
    # per-token step rewards aligned to the 3 masked positions: step1 end, step2 end, outcome
    step_rewards = torch.tensor([0.0, 1.3, 2.1])
    fake.set_pending_step_rewards([step_rewards])

    rewards, non_score, kls = fake.compute_rewards(
        scores=[torch.zeros(())], logprobs=[logprob], ref_logprobs=[ref_logprob], masks=[mask]
    )
    assert rewards.shape == (1, T), rewards.shape
    # KL penalty present on every token: non_score = -0.2 * 0.2 = -0.04
    assert torch.allclose(non_score[0], torch.full((T,), -0.04), atol=1e-6), non_score
    # step rewards landed on masked positions 1,2,3 on top of the KL penalty
    expected = torch.full((T,), -0.04)
    expected[torch.tensor([1, 2, 3])] += step_rewards
    assert torch.allclose(rewards[0], expected, atol=1e-6), (rewards[0], expected)
    # buffer consumed
    assert fake._pending_step_rewards is None
    print("  compute_rewards override: shape, KL channel, span placement ok")


def test_fallback_scalar_placement():
    """With no pending rewards, override must reproduce TRL last-token scalar."""
    from ppo_trainer_try import StepRewardPPOTrainer

    class _KLCtl:
        value = 0.0  # zero KL to isolate the score placement

    class _Cfg:
        kl_penalty = "kl"

    fake = StepRewardPPOTrainer.__new__(StepRewardPPOTrainer)
    fake.kl_ctl = _KLCtl()
    fake.config = _Cfg()
    fake._pending_step_rewards = None

    T = 4
    mask = torch.tensor([0, 1, 1, 0])  # last response token = index 2
    rewards, _, _ = fake.compute_rewards(
        scores=[torch.tensor(0.9)],
        logprobs=[torch.zeros(T)],
        ref_logprobs=[torch.zeros(T)],
        masks=[mask],
    )
    expected = torch.zeros(T)
    expected[2] = 0.9
    assert torch.allclose(rewards[0], expected, atol=1e-6), rewards[0]
    print("  fallback scalar placement (last masked token) ok")


def main():
    print("test_token_rewards_and_outcome"); test_token_rewards_and_outcome()
    print("test_gold_from_metadata_not_teacher"); test_gold_from_metadata_not_teacher()
    print("test_compute_rewards_override"); test_compute_rewards_override()
    print("test_fallback_scalar_placement"); test_fallback_scalar_placement()
    print("\nALL PPO OFFLINE TESTS PASSED ✅")


if __name__ == "__main__":
    main()
