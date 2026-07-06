"""Improved PPO reward function (try variant).

Differences vs the package's ``kgproweight.training.reward_function``:

* **P0-2** — uses ``ImprovedPRMAnnotator`` (filler-citation + abstention guards)
  for ``R_KG`` instead of the original ``PRMAnnotator`` whose 24% filler-+1 and
  -1 misfires would otherwise leak into the PPO reward as a reward-hacking
  signal.
* **P1-1** — accepts *real* per-step token logprobs so the α-gate's
  ``f_entropy`` feature is the genuine token-level entropy (the package path
  passes ``logprobs=[None]`` which collapses entropy to a constant 1.0).

Everything else (the composite ``R_total = α·R_KG + (1-α)·R_Text``, the EM
outcome bonus on the last step, the per-step→token-span mapping) is reused
unchanged from the package via ``CompositeRewardModel`` and the parsers.

This module is import-flat (run via a CLI that inserts ``scripts/train/try`` on
``sys.path``); it does not touch the package.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# --- reused, unchanged, from the original package --------------------------
from kgproweight.data.parsers import (
    extract_final_answer,
    extract_step_token_spans,
    parse_steps,
)
from kgproweight.data.prompts import build_sft_messages
from kgproweight.reward.alpha_gate import AlphaGate
from kgproweight.reward.composite_reward import CompositeRewardModel
from kgproweight.reward.text_reward_model import TextRewardModel
from kgproweight.utils.logging import get_logger

# --- changed logic, local to the try variant -------------------------------
from prm_annotator_try import ImprovedPRMAnnotator
from entity_filter_try import clean_entities

logger = get_logger(__name__)


def _decode_len(tokenizer, ids: List[int]) -> int:
    """Char length of the decoded prefix ``ids`` (special tokens kept so the
    offset is measured in the SAME stream the trainer scatters onto)."""
    if not ids:
        return 0
    try:
        return len(tokenizer.decode(ids, skip_special_tokens=False))
    except TypeError:  # fakes / tokenizers without the kwarg
        return len(tokenizer.decode(ids))


def step_spans_over_ids(
    response_ids: Sequence[int],
    tokenizer,
    n_steps: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Step ``(start, end)`` token spans in **response_ids coordinates** (#6).

    The previous path re-tokenised ``decode(response_ids, skip_special_tokens=
    True)`` and computed spans in that re-tokenised space, which does NOT align
    with the raw ``response_ids`` the PPO trainer scatters rewards onto (special
    tokens stripped + decode∘encode drift). Each per-step reward therefore
    landed a few tokens off its true position, smearing the per-step credit
    assignment that the whole StepRewardPPOTrainer exists to provide.

    Here we locate each ``[Step N]`` header directly in the decode of the
    actual ``response_ids`` stream and map its char offset back to a token index
    by binary-searching the monotonically-increasing prefix-decode length. Only
    the handful of step boundaries are searched (≈n_steps·log₂T decodes), not
    every token.
    """
    ids = [int(x) for x in (response_ids.tolist() if hasattr(response_ids, "tolist") else response_ids)]
    n = len(ids)
    if n == 0:
        return []
    full_text = tokenizer.decode(ids, skip_special_tokens=False) if _supports_skip(tokenizer) else tokenizer.decode(ids)
    headers = [m.start() for m in re.finditer(r"\[Step\s+\d+\]", full_text)]
    if not headers:
        return [(0, n)]

    def char_to_token(char_pos: int) -> int:
        """Smallest token index k such that decode(ids[:k]) reaches char_pos."""
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if _decode_len(tokenizer, ids[:mid]) >= char_pos:
                hi = mid
            else:
                lo = mid + 1
        return lo

    bounds = [char_to_token(h) for h in headers] + [n]
    spans: List[Tuple[int, int]] = []
    for i in range(len(headers)):
        start, end = bounds[i], bounds[i + 1]
        spans.append((start, max(end, start + 1)))
    if n_steps is not None:
        spans = spans[:n_steps]
    return spans


def _supports_skip(tokenizer) -> bool:
    try:
        tokenizer.decode([], skip_special_tokens=False)
        return True
    except TypeError:
        return False
    except Exception:
        return True


@dataclass
class RewardSpec:
    """Per-sample inputs (same shape as the package's RewardSpec)."""

    query: str
    gold_answer: str
    kg_subgraph: List[Tuple[str, str, str]]
    retrieved_passages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImprovedRewardFunction:
    """PPO reward callable producing per-token reward tensors.

    Parameters
    ----------
    alpha_gate, text_reward_model:
        Composed inside a :class:`CompositeRewardModel`.
    prm_annotator:
        An ``ImprovedPRMAnnotator`` instance (the whole point of the try
        variant). Kept as a parameter so the offline test can inject a fake.
    tokenizer:
        Aligns per-step reward to ``[Step N]`` token spans for PPO.
    outcome_weight, discount:
        Forwarded to :class:`CompositeRewardModel`.
    alpha_override:
        ``None`` or one of ``{0.0, 0.5, 1.0}`` for the α-ablations.
    """

    def __init__(
        self,
        alpha_gate: AlphaGate,
        prm_annotator: ImprovedPRMAnnotator,
        text_reward_model: TextRewardModel,
        tokenizer,
        outcome_weight: float = 1.0,
        discount: float = 0.95,
        alpha_override: Optional[float] = None,
        max_steps: int = 7,
    ) -> None:
        self.composite = CompositeRewardModel(
            alpha_gate=alpha_gate,
            prm_annotator=prm_annotator,
            text_reward_model=text_reward_model,
            outcome_weight=outcome_weight,
            discount=discount,
        )
        self.tokenizer = tokenizer
        self.alpha_override = alpha_override
        self.max_steps = max_steps

    def __call__(
        self,
        prompt: str,
        response: str,
        spec: RewardSpec,
        logprobs_per_step: Optional[Sequence[Optional[Sequence[float]]]] = None,
        response_ids: Optional[Sequence[int]] = None,
        step_spans: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        """Compute per-step + per-token rewards for one rollout.

        ``logprobs_per_step`` (P1-1): optional list aligned to the parsed steps,
        each entry the token logprobs of that step (or ``None``). When absent we
        fall back to ``None`` per step, matching the inference path.

        ``response_ids`` / ``step_spans`` (#6): when the caller passes the raw
        generated token ids (and, optionally, precomputed spans in those same
        coordinates), the per-token reward tensor is built in ``response_ids``
        space so it aligns EXACTLY with what the PPO trainer scatters onto. When
        omitted (e.g. the offline test), we fall back to re-tokenising the
        decoded ``response`` — correct in isolation, but only used outside the
        trainer loop.
        """
        steps = parse_steps(response)[: self.max_steps]
        # Finding-2 follow-up: strip reasoning-scaffold mentions ("Reasoning",
        # "Conclusion", …) so link_confidence reflects real entities only. MUST
        # match Phase 2's _build_samples_accepted_only, which applies the same
        # clean_entities to the same parser output.
        for _s in steps:
            _s.mentioned_entities = clean_entities(_s.mentioned_entities)
        predicted_answer = extract_final_answer(response) or ""

        # Build per-step text-reward prompts aligned with the SFT prompt so the
        # text reward model scores each step in its actual context.
        msgs = build_sft_messages(
            question=spec.query,
            retrieved_passages=spec.retrieved_passages,
            kg_triples=spec.kg_subgraph,
        )
        rendered_prompt = "\n\n".join(m["content"] for m in msgs)
        text_reward_prompts: List[str] = []
        for i, _ in enumerate(steps):
            prefix = "\n".join(s.raw_text for s in steps[:i])
            text_reward_prompts.append(rendered_prompt + ("\n\n" + prefix if prefix else ""))

        # P1-1: pass real per-step logprobs through to the α-gate's entropy
        # feature. If none supplied, use None per step (entropy→1.0 fallback).
        if logprobs_per_step is None:
            logprobs_list: List[Optional[Sequence[float]]] = [None] * len(steps)
        else:
            logprobs_list = list(logprobs_per_step[: len(steps)])
            if len(logprobs_list) < len(steps):
                logprobs_list += [None] * (len(steps) - len(logprobs_list))

        records = self.composite.compute_trajectory_rewards(
            steps=steps,
            kg_subgraph=spec.kg_subgraph,
            text_reward_prompts=text_reward_prompts,
            logprobs_list=logprobs_list,
            predicted_answer=predicted_answer,
            gold_answer=spec.gold_answer,
            alpha_override=self.alpha_override,
        )

        per_step_rewards = [r.r_total for r in records]
        returns = self.composite.discounted_returns(per_step_rewards)

        # Map each step's reward to the last token of its [Step N] span.
        # #6: prefer response_ids coordinates so placement aligns with the
        # trainer's scatter; fall back to re-tokenising the decoded response.
        if response_ids is not None:
            ids = [int(x) for x in (response_ids.tolist() if hasattr(response_ids, "tolist") else response_ids)]
            n_tokens = len(ids)
            spans = list(step_spans) if step_spans is not None else step_spans_over_ids(ids, self.tokenizer, len(steps))
        else:
            spans = extract_step_token_spans(response, self.tokenizer)
            n_tokens = len(self.tokenizer(response, add_special_tokens=False)["input_ids"])

        token_rewards = torch.zeros(n_tokens, dtype=torch.float32)
        for span, r in zip(spans, per_step_rewards):
            start, end = span
            if end <= 0 or start >= n_tokens:
                continue
            token_rewards[min(end - 1, n_tokens - 1)] += float(r)

        # #6b: when the rollout emits NO parseable [Step N] markers (common early
        # in training), `records` is empty and the package drops the EM outcome
        # entirely — a correct final answer would then get pure-KL reward (zero
        # task signal) exactly when the policy most needs it. Attach the EM
        # outcome to the last response token so the answer is still rewarded.
        outcome_fallback = 0.0
        if not records and predicted_answer and spec.gold_answer and n_tokens > 0:
            outcome_fallback = float(self.composite._em(predicted_answer, spec.gold_answer))
            if outcome_fallback:
                token_rewards[n_tokens - 1] += self.composite.outcome_weight * outcome_fallback

        return {
            "per_step_rewards": per_step_rewards,
            "per_step_records": records,
            "returns": returns,
            "token_rewards": token_rewards,
            "step_spans": spans,
            "predicted_answer": predicted_answer,
            "trajectory_reward": float(sum(per_step_rewards) + outcome_fallback),
        }
