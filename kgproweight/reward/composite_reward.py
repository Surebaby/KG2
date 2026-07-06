"""Composite per-step reward.

  R_total(t) = α_t · R_KG(t) + (1 - α_t) · R_Text(t)
  R_outcome  = EM(answer, gold)              # added to the LAST step only

Bug-fix #1 from :doc:`docs/refactor_notes`: ``R_Text`` is now actually
mixed in — the legacy PPO path silently dropped it.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from kgproweight.reward.alpha_gate import AlphaGate, compute_features
from kgproweight.reward.prm_annotator import ParsedStep, PRMAnnotator
from kgproweight.reward.prm_value_head import PRMValueHead
from kgproweight.reward.text_reward_model import TextRewardModel


# ---------------------------------------------------------------------------
# Step-level record
# ---------------------------------------------------------------------------

@dataclass
class StepReward:
    step_index: int
    alpha: float
    r_kg: float
    r_text: float
    r_total: float
    graph_density: float
    link_confidence: float
    semantic_entropy: float


# ---------------------------------------------------------------------------
# Composite model
# ---------------------------------------------------------------------------

class CompositeRewardModel(nn.Module):
    """Composes α-gate, PRM-annotator-derived R_KG, text reward, and EM outcome."""

    def __init__(
        self,
        alpha_gate: AlphaGate,
        prm_annotator: PRMAnnotator,
        text_reward_model: TextRewardModel,
        prm_value_head: Optional[PRMValueHead] = None,
        outcome_weight: float = 1.0,
        discount: float = 0.95,
        text_reward_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha_gate = alpha_gate
        self.prm_annotator = prm_annotator
        self.text_reward_model = text_reward_model
        self.prm_value_head = prm_value_head
        self.outcome_weight = outcome_weight
        self.discount = discount
        # R5: scale down R_text to prevent it dominating reward
        self.text_reward_scale = text_reward_scale

    # ------------------------------------------------------------------
    # Single-step
    # ------------------------------------------------------------------

    def compute_step_reward(
        self,
        step: ParsedStep,
        kg_subgraph: List[Tuple[str, str, str]],
        prompt_for_text_reward: str,
        logprobs: Optional[Sequence[float]],
        prev_conclusions: List[str],
        kg_embedding_model=None,
        context_vector: Optional[torch.Tensor] = None,
    ) -> StepReward:
        f_density, f_confidence, f_entropy = compute_features(
            step_entities=step.mentioned_entities,
            kg_subgraph=kg_subgraph,
            logprobs=logprobs,
            entity_linker=self.prm_annotator.entity_linker,
            kg_embedding_model=kg_embedding_model,
            context_vector=context_vector,
        )
        alpha = self.alpha_gate.forward_single(f_density, f_confidence, f_entropy)
        r_kg = float(self.prm_annotator.label(step, kg_subgraph, prev_conclusions))
        r_text = float(
            self.text_reward_model.score_step(prompt_for_text_reward, step.raw_text)
        )
        r_total = alpha * r_kg + (1.0 - alpha) * r_text * self.text_reward_scale
        return StepReward(
            step_index=step.index,
            alpha=alpha,
            r_kg=r_kg,
            r_text=r_text,
            r_total=r_total,
            graph_density=f_density,
            link_confidence=f_confidence,
            semantic_entropy=f_entropy,
        )

    # ------------------------------------------------------------------
    # Trajectory
    # ------------------------------------------------------------------

    def compute_trajectory_rewards(
        self,
        steps: List[ParsedStep],
        kg_subgraph: List[Tuple[str, str, str]],
        text_reward_prompts: List[str],
        logprobs_list: Sequence[Optional[Sequence[float]]],
        predicted_answer: Optional[str] = None,
        gold_answer: Optional[str] = None,
        alpha_override: Optional[float] = None,
        kg_embedding_model=None,
        context_vectors: Optional[List[torch.Tensor]] = None,
        trajectory_valid: bool = True,
    ) -> List[StepReward]:
        """Return one :class:`StepReward` per step.

        Ablations can fix ``alpha_override`` ∈ {0.0, 0.5, 1.0}; in that case
        the trained α-gate is bypassed for *this trajectory* but the model
        itself remains the same checkpoint.

        R7: ``trajectory_valid`` gates the outcome reward. When False, the
        per-step composite rewards are still computed (so PPO has signal on
        step quality), but the +outcome_weight·EM bonus is withheld — the
        model only receives the "grand prize" for complete, well-formatted
        reasoning traces.
        """
        records: List[StepReward] = []
        prev_conclusions: List[str] = []
        ctx = context_vectors or [None] * len(steps)
        for i, step in enumerate(steps):
            sr = self.compute_step_reward(
                step=step,
                kg_subgraph=kg_subgraph,
                prompt_for_text_reward=text_reward_prompts[i]
                if i < len(text_reward_prompts)
                else "",
                logprobs=logprobs_list[i] if i < len(logprobs_list) else None,
                prev_conclusions=prev_conclusions,
                kg_embedding_model=kg_embedding_model,
                context_vector=ctx[i] if i < len(ctx) else None,
            )
            if alpha_override is not None:
                a = float(alpha_override)
                sr = StepReward(
                    step_index=sr.step_index,
                    alpha=a,
                    r_kg=sr.r_kg,
                    r_text=sr.r_text,
                    r_total=a * sr.r_kg + (1.0 - a) * sr.r_text * self.text_reward_scale,
                    graph_density=sr.graph_density,
                    link_confidence=sr.link_confidence,
                    semantic_entropy=sr.semantic_entropy,
                )
            records.append(sr)
            if step.intermediate_conclusion:
                prev_conclusions.append(step.intermediate_conclusion)

        # R7: Outcome reward is conditional on trajectory validity.
        # When the policy emits an incomplete or malformed trace, the per-step
        # composite rewards still provide a learning signal, but the large
        # outcome bonus is withheld — the model must earn it by generating
        # well-structured reasoning traces, not just correct final answers.
        if predicted_answer is not None and gold_answer is not None and trajectory_valid:
            outcome = float(self._em(predicted_answer, gold_answer))
            if records:
                last = records[-1]
                records[-1] = StepReward(
                    step_index=last.step_index,
                    alpha=last.alpha,
                    r_kg=last.r_kg,
                    r_text=last.r_text,
                    r_total=last.r_total + self.outcome_weight * outcome,
                    graph_density=last.graph_density,
                    link_confidence=last.link_confidence,
                    semantic_entropy=last.semantic_entropy,
                )
        return records

    def discounted_returns(self, rewards: List[float]) -> List[float]:
        returns = [0.0] * len(rewards)
        cum = 0.0
        for t in reversed(range(len(rewards))):
            cum = rewards[t] + self.discount * cum
            returns[t] = cum
        return returns

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _em(pred: str, gold: str) -> bool:
        def normalize(s: str) -> str:
            s = s.lower().strip()
            s = re.sub(r"\b(a|an|the)\b", " ", s)
            s = s.translate(str.maketrans("", "", string.punctuation))
            return " ".join(s.split())

        return normalize(pred) == normalize(gold)
