"""StepRewardPPOTrainer (try variant) — per-step reward into GAE.

The package's PPO path sums per-step rewards into a single scalar before
handing them to TRL, which (in 0.11.4) places that scalar on the *last*
response token only. GAE then runs on an essentially outcome-only signal, so
the dynamic-α *per-step* structure — the mechanism Theorem 2 depends on — never
reaches the advantage estimator.

This subclass overrides exactly one method, ``compute_rewards``, to place each
step's ``R_total(t)`` on the last token of its ``[Step N]`` span (the last step
also carrying the EM outcome). Everything downstream — ``compute_advantages``
(GAE), minibatching, PPO clipping, the adaptive KL controller — is reused
unchanged from TRL.

Coupling note (accepted by design): the override reproduces TRL 0.11.4's
per-token KL-penalty bookkeeping (``_kl_penalty`` + ``kl_ctl.value``) so the
non-score reward channel is identical to upstream. If TRL is upgraded this
method must be re-synced with the new ``compute_rewards``.

Usage::

    trainer = StepRewardPPOTrainer(config=ppo_cfg, model=policy,
                                   ref_model=ref, tokenizer=tok)
    ...
    trainer.set_pending_step_rewards(token_reward_tensors)  # one per response
    trainer.step(query_tensors, response_tensors, placeholder_scores)
"""

from __future__ import annotations

from typing import List, Optional

import torch
from trl import PPOTrainer

from kgproweight.utils.logging import get_logger

logger = get_logger(__name__)


class StepRewardPPOTrainer(PPOTrainer):
    """PPOTrainer that injects per-token step rewards via a side channel."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Per-batch buffer: list (len == batch) of 1-D float tensors, each the
        # per-response-token reward (step rewards on step-end tokens). Set right
        # before every ``step`` call; consumed (and cleared) by compute_rewards.
        self._pending_step_rewards: Optional[List[torch.Tensor]] = None
        # When True (default), compute_rewards REQUIRES pending step rewards and
        # raises if they are missing or the batch size mismatches — instead of
        # silently falling back to the (zero placeholder) scalar path, which
        # would train on KL-only reward with no error (#8). Set False only to
        # use this subclass like a vanilla PPOTrainer.
        self._require_step_rewards: bool = True

    def set_pending_step_rewards(self, token_rewards: List[torch.Tensor]) -> None:
        """Register this batch's per-token step rewards (response order).

        #8: validate up front so a wrong-length buffer can't silently misalign
        (extra samples would otherwise take the zero-placeholder path).
        """
        if token_rewards is None:
            raise ValueError("set_pending_step_rewards got None")
        bs = getattr(getattr(self, "config", None), "batch_size", None)
        if bs is not None and len(token_rewards) != int(bs):
            raise ValueError(
                f"pending step rewards length {len(token_rewards)} != batch_size {bs}; "
                "every sample in the batch must have a per-token reward tensor."
            )
        self._pending_step_rewards = token_rewards

    def compute_rewards(self, scores, logprobs, ref_logprobs, masks):
        """Per-token rewards = KL penalty (per token) + step rewards (on spans).

        Mirrors TRL 0.11.4's ``compute_rewards`` exactly for the KL channel, but
        replaces ``reward[last_non_masked] += score`` with our per-token step
        rewards aligned to the masked (response) region.

        #8: when ``_require_step_rewards`` is True (default) and no pending
        rewards are set, this RAISES rather than silently scattering the (zero)
        placeholder scalar — so a forgotten ``set_pending_step_rewards`` is a
        loud failure, not a run that trains on KL-only reward.

        Returns the same 3-tuple ``(rewards, non_score_rewards, kls)`` of shape
        ``(batch, response_len)`` that ``compute_advantages`` consumes.
        """
        pending = self._pending_step_rewards
        if getattr(self, "_require_step_rewards", False) and pending is None:
            raise RuntimeError(
                "StepRewardPPOTrainer.compute_rewards called without pending step "
                "rewards. Call set_pending_step_rewards(token_reward_list) before "
                "every trainer.step(...). (Set _require_step_rewards=False to allow "
                "the vanilla last-token scalar fallback.)"
            )
        rewards, non_score_rewards, kls = [], [], []
        for i, (score, logprob, ref_logprob, mask) in enumerate(
            zip(scores, logprobs, ref_logprobs, masks)
        ):
            # --- identical to TRL: per-token KL penalty as the non-score reward
            kl = self._kl_penalty(logprob, ref_logprob)
            kls.append(kl)
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()

            mask_idx = mask.nonzero()
            if pending is not None and i < len(pending) and mask_idx.numel() > 0:
                # --- our change: scatter step rewards onto the masked region.
                # The masked positions (mask==1) are exactly the response
                # tokens; align our per-response-token rewards to them in order.
                resp_positions = mask_idx.squeeze(-1)  # ascending token indices
                step_r = pending[i].to(reward.device, reward.dtype)
                n = min(resp_positions.numel(), step_r.numel())
                if n > 0:
                    reward[resp_positions[:n]] += step_r[:n]
            elif mask_idx.numel() > 0:
                # --- fallback: upstream behaviour (scalar on last token).
                reward[mask_idx[-1]] += score
            rewards.append(reward)

        # Consume the buffer so a missing set_pending_step_rewards on the next
        # step is caught by the guard above rather than silently reusing stale
        # rewards.
        self._pending_step_rewards = None
        return torch.stack(rewards), torch.stack(non_score_rewards), torch.stack(kls)
