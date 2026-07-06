"""PPO / GRPO reward function — shared.

Given a Student-generated trajectory string and the per-trajectory context
(query, gold answer, KG subgraph, retrieved passages), produce:

  - per-step rewards ``R_total(t) = α_t · R_KG(t) + (1 - α_t) · R_Text(t)``,
  - the outcome bonus ``R_outcome = EM(answer, gold)`` added to the last step,
  - step token-boundary indices so PPO can place per-step rewards on the
    correct token positions.

The function is intentionally self-contained: it instantiates the
``CompositeRewardModel`` on first use and reuses it across calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from kgproweight.data.entity_filter import clean_entities
from kgproweight.data.parsers import extract_final_answer, extract_step_token_spans, parse_steps
from kgproweight.data.prompts import build_sft_messages
from kgproweight.reward.alpha_gate import AlphaGate
from kgproweight.reward.composite_reward import CompositeRewardModel
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.reward.text_reward_model import TextRewardModel
from kgproweight.utils.logging import get_logger

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


def _supports_skip(tokenizer) -> bool:
    try:
        tokenizer.decode([], skip_special_tokens=False)
        return True
    except TypeError:
        return False
    except Exception:
        return True


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


@dataclass
class RewardSpec:
    """Per-sample inputs to :func:`KGProWeightRewardFunction.__call__`."""

    query: str
    gold_answer: str
    kg_subgraph: List[Tuple[str, str, str]]
    retrieved_passages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class KGProWeightRewardFunction:
    """PPO/GRPO reward callable.

    Parameters
    ----------
    alpha_gate, prm_annotator, text_reward_model:
        Components composed inside a :class:`CompositeRewardModel`.
    tokenizer:
        Used to align per-step reward to token spans for PPO.
    outcome_weight, discount:
        Hyperparameters from :class:`CompositeRewardModel`.
    alpha_override:
        ``None`` (default) or one of ``{0.0, 0.5, 1.0}`` for the alpha
        ablations. The trained α-gate is bypassed if set.
    """

    def __init__(
        self,
        alpha_gate: AlphaGate,
        prm_annotator: PRMAnnotator,
        text_reward_model: TextRewardModel,
        tokenizer,
        outcome_weight: float = 1.0,
        discount: float = 0.95,
        alpha_override: Optional[float] = None,
        max_steps: int = 7,
        pure_em: bool = False,
        text_reward_scale: float = 1.0,
        # R7: minimum number of parsed [Step N] blocks for a trajectory to be
        # considered "valid" and eligible for the outcome reward.
        min_valid_steps: int = 3,
        # R8: minimum characters of actual reasoning content per step.
        min_reasoning_chars: int = 20,
    ) -> None:
        self.composite = CompositeRewardModel(
            alpha_gate=alpha_gate,
            prm_annotator=prm_annotator,
            text_reward_model=text_reward_model,
            outcome_weight=outcome_weight,
            discount=discount,
            text_reward_scale=text_reward_scale,
        )
        self.tokenizer = tokenizer
        self.alpha_override = alpha_override
        self.max_steps = max_steps
        # R7: minimum valid step count for trajectory validity gating.
        self.min_valid_steps = min_valid_steps
        # R8: minimum reasoning content per step (content-aware gate).
        self.min_reasoning_chars = min_reasoning_chars
        # Pure EM reward mode (ablation): ignore R_KG and R_text entirely;
        # reward is EM × outcome_weight on the final step (no per-step bonuses).
        # This is the upper bound for "what PPO can achieve when reward is
        # perfectly aligned with the evaluation metric".
        self.pure_em = pure_em

    @staticmethod
    def _is_valid_trajectory(
        steps: list,
        response: str,
        min_steps: int = 3,
        min_reasoning_chars: int = 20,
    ) -> bool:
        """R7: Check whether a generated trajectory meets format requirements.

        A trajectory is "valid" (eligible for the outcome reward) when ALL of
        the following hold:

        1. At least ``min_steps`` parseable ``[Step N]`` blocks.
        2. A ``Final Answer`` can be extracted.
        3. Step indices are sequential (1, 2, 3, …).
        4. Every step has non-empty text.
        5. (R8) Every step's ``Reasoning:`` section contains at least
           ``min_reasoning_chars`` characters of actual content (excluding
           whitespace and subsequent Knowledge/Conclusion/Final Answer sections).

        This is a FORMAT constraint — it does NOT judge factual correctness.
        Its sole purpose is to make the outcome reward *conditional* on
        producing a well-structured reasoning trace, so PPO cannot collect the
        "grand prize" by emitting a bare answer or an empty "Reasoning:" block.
        """
        if not steps or len(steps) < min_steps:
            return False
        if extract_final_answer(response) is None:
            return False
        expected = 1
        for s in steps:
            if s.index != expected:
                return False
            if not s.raw_text or not s.raw_text.strip():
                return False
            # R8: content-aware gate — ``Reasoning:`` must not be empty.
            # PPO was exploiting the old check (just non-empty raw_text) by
            # writing ``[Step 1]\\nReasoning: \\nFinal Answer: X``, which
            # parses as raw_text="Reasoning: \\nFinal Answer: X" → non-empty
            # → gate passes. Now we extract the actual reasoning body and
            # require >= min_reasoning_chars of substantive content.
            body = s.raw_text.strip()
            if "Reasoning:" in body:
                after = body.split("Reasoning:", 1)[1]
                # Stop at the next structural label.
                reasoning = re.split(
                    r'Knowledge Used:|Conclusion:|Final Answer:', after
                )[0].strip()
                if len(reasoning) < min_reasoning_chars:
                    return False
            expected += 1
        return True

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

        # R7: gate the outcome reward on trajectory validity.
        # Per-step composite rewards are still computed regardless — PPO gets
        # signal from step-level KG/Text quality even for incomplete traces.
        trajectory_valid = self._is_valid_trajectory(
            steps, response, min_steps=self.min_valid_steps,
            min_reasoning_chars=self.min_reasoning_chars,
        )

        # ── Pure EM reward fast-path (ablation) ──
        # When enabled, skip the entire composite reward pipeline (R_KG, R_text,
        # α-gate). Reward = EM × outcome_weight on the last step ONLY when the
        # trajectory is valid. (R7: outcome now gated on trajectory validity.)
        if self.pure_em:
            per_step_rewards: List[float] = (
                [0.0] * len(steps) if steps else [0.0] * max(len(steps), 1)
            )
            outcome = 0.0
            if trajectory_valid and predicted_answer and spec.gold_answer:
                outcome = float(self.composite._em(predicted_answer, spec.gold_answer))
            if per_step_rewards and outcome:
                per_step_rewards[-1] += self.composite.outcome_weight * outcome
            returns = self.composite.discounted_returns(per_step_rewards)

            # token mapping (shared with the main path)
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

            return {
                "per_step_rewards": per_step_rewards,
                "per_step_records": [],
                "returns": returns,
                "token_rewards": token_rewards,
                "step_spans": spans,
                "predicted_answer": predicted_answer,
                "trajectory_reward": float(sum(per_step_rewards)),
            }

        # Build per-step text-reward prompts that align with the SFT prompt
        # so the ReaRAG/Llama-head reward model evaluates each step in its
        # actual context.
        text_reward_prompts = []
        msgs = build_sft_messages(
            question=spec.query,
            retrieved_passages=spec.retrieved_passages,
            kg_triples=spec.kg_subgraph,
        )
        rendered_prompt = "\n\n".join(m["content"] for m in msgs)
        for i, _ in enumerate(steps):
            prefix = "\n".join(s.raw_text for s in steps[:i])
            text_reward_prompts.append(rendered_prompt + ("\n\n" + prefix if prefix else ""))

        # P1-1: pass real per-step logprobs through to the α-gate's entropy
        # feature. If none supplied, use None per step (entropy→1.0 fallback,
        # matching the inference path).
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
            trajectory_valid=trajectory_valid,
        )

        per_step_rewards = [r.r_total for r in records]
        # R7: no per-step format bonus. Format is a CONSTRAINT (enforced via
        # trajectory_valid gating the outcome reward), not a reward target.
        # See problem_and_solutions.md for the rationale.
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

        # R7: outcome fallback is now also gated on trajectory_valid.
        # When the policy emits a bare correct answer with no [Step N] markers,
        # it does NOT receive the outcome reward — the "grand prize" requires
        # a well-structured reasoning trace.  (The #6b fallback was originally
        # added to prevent zero task signal early in training; R7 replaces it
        # with the combination of valid-trajectory gating + SFT anchor.)
        outcome_fallback = 0.0
        if (
            trajectory_valid
            and not records
            and predicted_answer
            and spec.gold_answer
            and n_tokens > 0
        ):
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
