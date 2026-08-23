"""Composite per-step reward.

  R_total(t) = (α_t · R_KG(t) + (1 - α_t) · R_Text(t) · text_reward_scale)
               · step_reward_scale
  R_outcome  = outcome_weight · EM(answer, gold)   # LAST step only, and only
                                                   # when trajectory_valid
  R_invalid  = -outcome_weight                     # LAST step, when not valid
  R_short    = -shortfall_coef · outcome_weight
               · max(0, target_steps - n_steps) / target_steps

There is NO return/discount computation here. The per-step rewards produced by
this module are scattered onto response tokens by the caller and discounted by
TRL's GAE (``PPOConfig(gamma=..., lam=...)``). A ``discounted_returns`` helper
used to live here and was removed 2026-08-22 (retraining_plan §13-3): nothing
consumed its output, so the paper's G_t = Σ γ^(k-t) R_k formula described code
that never influenced training. ``self.discount`` is retained only because it
mirrors the GAE gamma actually in use.

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
from kgproweight.reward.citation_features import citation_features
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
    # 2026-08-23 (retraining_plan §9.4-1, the 量纲 fix): the value of R_Text that
    # was actually MIXED INTO ``r_total``. With ``center_text_reward=False`` it
    # equals ``r_text`` exactly, so every pre-2026-08-23 record is unchanged.
    # With centering on it is ``r_text - text_baseline``.
    #
    # Both are kept because they answer different questions: ``r_text`` is the
    # raw scorer output and is what §9.1's measured statistics refer to, while
    # ``r_text_used`` is what the policy is actually being paid. Logging only one
    # of them makes the reward un-auditable in one direction or the other.
    r_text_used: float = 0.0
    text_baseline: float = 0.0
    # §14: the α-gate's citation features, recorded so the PPO diagnostics can
    # report them without recomputing (and so a step record is self-describing).
    cite_any: float = 0.0
    cite_match: float = 0.0


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
        step_reward_scale: float = 1.0,
        # 2026-08-22 (retraining_plan §9.4-3 / R-1b): step-shortfall penalty.
        # Both measured PPO runs collapsed to exactly min_valid_steps steps per
        # trajectory within ~150 updates (2.84 -> 2.04 and 3.13 -> 2.01), against
        # silver's 3.36. The arithmetic: a 2-step trajectory earns ~2 x 0.163 =
        # 0.33 of process reward against outcome_weight 4.0 -- a 1:12 imbalance --
        # so the marginal ~0.163 from writing a third step is worth less than the
        # KL cost of the ~60 extra tokens. The policy was behaving optimally; the
        # reward was mis-scaled. This penalty prices the missing steps.
        #
        # shortfall_coef is anchored to outcome_weight (not an absolute number) so
        # the two stay in proportion if either is retuned. 0.25 => missing one of
        # three steps costs 0.25 * 4.0 * (1/3) = 0.33, i.e. the same order as the
        # whole process-reward budget of a collapsed trajectory. Deliberately NOT
        # larger: at high values PPO optimises step count over EM, which is how
        # kg_reward_share fell to 0.009 under outcome_weight=10.
        shortfall_coef: float = 0.0,
        # Step count the penalty measures against. Should equal the silver
        # min_steps (3), NOT min_valid_steps -- the gap between those two is the
        # room the collapse exploited.
        target_steps: int = 3,
        # ------------------------------------------------------------------
        # 2026-08-23 (retraining_plan §9.4-1): R_Text DC removal — the 量纲 fix.
        # ------------------------------------------------------------------
        # MEASURED over PPO① (`ppo_r10_split`, 750 updates):
        #     r_kg    mean 0.0896  sd 0.1166  range [-0.2222, 0.7500]
        #     r_text  mean 0.6284  sd 0.1353  range [ 0.0994, 0.8489]
        # Both channels are nominally [-1, 1], but they occupy intervals whose
        # MEANS differ by 7x while their SDs differ by 16%. R_Text is therefore
        # not a signal riding on a shared scale -- it is a near-constant +0.63
        # bias with a small signal on top. `RearagPromptScorer.score_step` makes
        # this structural: tanh((2.5 - nll)/1.5) with typical nll ~1.5 gives
        # tanh(0.67) = 0.585, which is the measured 0.628 up to noise.
        #
        # Why a constant is not harmless. Two distinct consequences:
        #
        # (1) It is INVISIBLE to the policy gradient. GAE subtracts a state
        #     baseline and TRL whitens advantages, so any additive term that
        #     does not vary with the action contributes nothing. The 0.63 is
        #     simply deleted -- the channel spends its budget on nothing.
        #
        # (2) It is NOT invisible where it interacts with alpha, and there it
        #     points the WRONG WAY. With the measured numbers the reward's
        #     sensitivity to the gate is
        #        d r_total / d alpha = (r_kg - c_text * r_text) * c_step
        #                            = (0.0896 - 0.3*0.6284) * 1.5 = -0.148
        #     i.e. STRICTLY NEGATIVE: the policy is paid to push alpha DOWN.
        #     alpha rises with f_density = |E|/(|V|+eps), so "push alpha down"
        #     means "make the cited subgraph look sparser". The reward function
        #     of a KG-grounding method was rewarding LESS KG grounding, through
        #     the very gate the method is named after.
        #
        # Removing the DC offset fixes both: the mixed value becomes zero-mean,
        # so d r_total / d alpha averages to 0 and alpha stops carrying a
        # systematic direction, while the surviving variation (sd 0.135, of the
        # same order as r_kg's 0.117) is the part that was always the signal.
        #
        # DEVIATION FROM THE PLAN, stated openly: §9.4-1 writes the fix as
        # `r_text - mean_batch(r_text)`. We use a causal EMA baseline instead of
        # the literal within-batch mean, for two reasons.
        #   - Noise. batch_size=4 at ~3 steps is ~12 samples, so a batch mean has
        #     se = 0.135/sqrt(12) = 0.039, which is 29% of the signal's own sd --
        #     it would inject nearly a third of the channel back as noise. An EMA
        #     at momentum 0.99 has an effective window of ~100 samples, se 0.014
        #     (10%).
        #   - Algorithm identity. A within-batch mean makes each trajectory's
        #     reward depend on the peers it happened to be batched with. That is
        #     a GRPO-style relative reward, and the critic cannot represent it
        #     (it sees the state, not the batch), so it appears to the value
        #     function as pure noise. A slowly-varying global constant is instead
        #     a reparameterisation the critic tracks for free.
        #
        # Default False so every pre-2026-08-23 run reproduces bit-for-bit; the
        # YAML turns it on. The baseline is NOT checkpointed -- a resumed run
        # re-warms it over its first ~100 steps (see reset_text_baseline).
        center_text_reward: bool = False,
        text_baseline_momentum: float = 0.99,
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
        # R9: scale up R_step so each good step (~0.8 raw) covers KL cost (~5).
        self.step_reward_scale = step_reward_scale
        self.shortfall_coef = shortfall_coef
        self.target_steps = max(1, int(target_steps))
        # §9.4-1: R_Text DC removal. `_text_baseline` is None until the first
        # observation so the baseline starts AT the data rather than at 0 -- an
        # 0-init would leave the whole warm-up window carrying the +0.63 offset
        # this exists to remove.
        self.center_text_reward = bool(center_text_reward)
        self.text_baseline_momentum = float(text_baseline_momentum)
        self._text_baseline: Optional[float] = None
        # Counts observations that updated the baseline. Exposed so the trainer
        # can log whether the baseline is still warming up when a metric is read.
        self._text_baseline_n: int = 0

    # ------------------------------------------------------------------
    # §9.4-1: baseline management
    # ------------------------------------------------------------------

    def reset_text_baseline(self) -> None:
        """Reset the text baseline to its cold-start state (next obs inits it).

        For resuming from a checkpoint or starting a new stage. The baseline is
        deliberately NOT checkpointed: it converges in ~100 updates (momentum
        0.99) against runs of 1000+ updates, so re-warming costs < 10% per run
        and keeping it out of the checkpoint state avoids the trap where an old
        baseline is silently carried forward into a new reward configuration.
        """
        self._text_baseline = None
        self._text_baseline_n = 0

    def _update_text_baseline(self, r_text: float) -> float:
        """Observe one r_text value, update the EMA, return the baseline to subtract.

        Returns the baseline BEFORE the update so the current observation is
        centered on past data, not including itself (causality). This keeps the
        definition consistent with the batch-mean version in the plan: the mean
        of a batch does not include "the future".
        """
        baseline = self._text_baseline if self._text_baseline is not None else r_text
        if self._text_baseline is None:
            self._text_baseline = r_text
        else:
            m = self.text_baseline_momentum
            self._text_baseline = m * self._text_baseline + (1.0 - m) * r_text
        self._text_baseline_n += 1
        return baseline

    @property
    def text_baseline(self) -> float:
        """Current baseline value (0 if not yet initialized)."""
        return self._text_baseline if self._text_baseline is not None else 0.0

    @property
    def text_baseline_n_obs(self) -> int:
        """Number of observations that have updated the baseline."""
        return self._text_baseline_n

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
        # §14: the α-gate's two per-step citation features. Computed from the SAME
        # ``step`` and ``kg_subgraph`` Phase 2 uses, through the same shared
        # helper, so the gate sees one feature definition at train and inference.
        f_cite_any, f_cite_match = citation_features(step.cited_triples, kg_subgraph)
        alpha = self.alpha_gate.forward_single(
            f_density, f_confidence, f_entropy, f_cite_any, f_cite_match
        )
        r_kg = float(self.prm_annotator.label(step, kg_subgraph, prev_conclusions))
        r_text = float(
            self.text_reward_model.score_step(prompt_for_text_reward, step.raw_text)
        )
        # §9.4-1 (量纲): subtract the running DC offset from R_Text before mixing.
        # Only the CENTERED value enters r_total; the raw value is still recorded
        # so the §9.1 statistics remain comparable across the change.
        if self.center_text_reward:
            baseline = self._update_text_baseline(r_text)
            r_text_used = r_text - baseline
        else:
            baseline = 0.0
            r_text_used = r_text
        r_total = (
            alpha * r_kg + (1.0 - alpha) * r_text_used * self.text_reward_scale
        ) * self.step_reward_scale
        return StepReward(
            step_index=step.index,
            alpha=alpha,
            r_kg=r_kg,
            r_text=r_text,
            r_total=r_total,
            graph_density=f_density,
            link_confidence=f_confidence,
            semantic_entropy=f_entropy,
            r_text_used=r_text_used,
            text_baseline=baseline,
            cite_any=f_cite_any,
            cite_match=f_cite_match,
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
                # §9.4-1: recombine from `r_text_used`, NOT from `r_text`. Two
                # things would break if this branch read the raw value:
                #   - the alpha ablation arms would be the only arms running an
                #     UNCENTERED text channel, so their rewards would not be on
                #     the same scale as the main arm and the ablation would
                #     measure the centering as well as alpha;
                #   - `_update_text_baseline` has already consumed this
                #     observation inside compute_step_reward, so recomputing
                #     here must not observe it a second time (it would double
                #     the EMA's effective sample rate on this path only).
                # With center_text_reward=False, r_text_used == r_text and this
                # is identical to the pre-2026-08-23 expression.
                sr = StepReward(
                    step_index=sr.step_index,
                    alpha=a,
                    r_kg=sr.r_kg,
                    r_text=sr.r_text,
                    r_total=(a * sr.r_kg + (1.0 - a) * sr.r_text_used * self.text_reward_scale) * self.step_reward_scale,
                    graph_density=sr.graph_density,
                    link_confidence=sr.link_confidence,
                    semantic_entropy=sr.semantic_entropy,
                    r_text_used=sr.r_text_used,
                    text_baseline=sr.text_baseline,
                    cite_any=sr.cite_any,
                    cite_match=sr.cite_match,
                )
            records.append(sr)
            if step.intermediate_conclusion:
                prev_conclusions.append(step.intermediate_conclusion)

        # R7: Outcome reward is conditional on trajectory validity.
        # R9: invalid trajectories receive a NEGATIVE penalty so PPO cannot escape
        # the gate by "shutting up" (short outputs have low KL cost, and without
        # penalty invalid→0 > long valid→negative from KL). The penalty must be
        # larger than the KL cost of a valid trajectory (~15 for 300 tokens).
        if trajectory_valid:
            if predicted_answer is not None and gold_answer is not None:
                outcome = float(self._em(predicted_answer, gold_answer))
                if records and outcome > 0:
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
                        r_text_used=last.r_text_used,
                        text_baseline=last.text_baseline,
                    )
        else:
            invalid_penalty = -self.outcome_weight  # = -10.0
            if records:
                last = records[-1]
                records[-1] = StepReward(
                    step_index=last.step_index,
                    alpha=last.alpha,
                    r_kg=last.r_kg,
                    r_text=last.r_text,
                    r_total=last.r_total + invalid_penalty,
                    graph_density=last.graph_density,
                    link_confidence=last.link_confidence,
                    semantic_entropy=last.semantic_entropy,
                    r_text_used=last.r_text_used,
                    text_baseline=last.text_baseline,
                    cite_any=last.cite_any,
                    cite_match=last.cite_match,
                )
            else:
                # No parsed steps at all: there is no r_text observation to
                # record, so r_text_used/text_baseline are genuinely 0 here (not
                # a dropped field). Deliberately does NOT touch the EMA -- a
                # trajectory with no steps must not pull the baseline toward 0.
                records.append(StepReward(
                    step_index=0, alpha=0.0, r_kg=0.0, r_text=0.0,
                    r_total=invalid_penalty,
                    graph_density=0.0, link_confidence=0.0, semantic_entropy=0.0,
                    r_text_used=0.0, text_baseline=self.text_baseline,
                ))

        # ── Step-shortfall penalty (retraining_plan §9.4-3 / R-1b) ──
        # Applied to BOTH valid and invalid trajectories, and measured on the
        # number of PARSED steps rather than on trajectory_valid. Gating it on
        # validity would leave the exact escape route the collapse used: a 2-step
        # trajectory that passes min_valid_steps is "valid", so a validity-gated
        # penalty would never fire on it -- which is the case this exists for.
        #
        # Charged on the LAST step so it lands on the same token as the outcome
        # reward: the decision being priced is "stop here vs. write another step",
        # and GAE propagates it back over the response either way.
        if self.shortfall_coef > 0.0 and records:
            shortfall = max(0, self.target_steps - len(steps))
            if shortfall:
                penalty = -(
                    self.shortfall_coef
                    * self.outcome_weight
                    * shortfall
                    / self.target_steps
                )
                last = records[-1]
                records[-1] = StepReward(
                    step_index=last.step_index,
                    alpha=last.alpha,
                    r_kg=last.r_kg,
                    r_text=last.r_text,
                    r_total=last.r_total + penalty,
                    graph_density=last.graph_density,
                    link_confidence=last.link_confidence,
                    semantic_entropy=last.semantic_entropy,
                    r_text_used=last.r_text_used,
                    text_baseline=last.text_baseline,
                    cite_any=last.cite_any,
                    cite_match=last.cite_match,
                )
        return records

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
