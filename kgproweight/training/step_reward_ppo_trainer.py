"""StepRewardPPOTrainer — per-step reward into GAE.

The package's legacy PPO path summed per-step rewards into a single scalar
before handing them to TRL, which (in 0.11.4) places that scalar on the *last*
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

from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
from trl import PPOTrainer

from kgproweight.utils.logging import get_logger

logger = get_logger(__name__)


class StepRewardPPOTrainer(PPOTrainer):
    """PPOTrainer that injects per-token step rewards via a side channel."""

    def __init__(self, *args, **kwargs) -> None:
        self._runtime_contract_version = kwargs.pop("runtime_contract_version", "legacy")
        if self._runtime_contract_version not in {"legacy", "v2"}:
            raise ValueError(
                f"unknown PPO runtime_contract_version: {self._runtime_contract_version!r}"
            )
        # TRL 0.11.4 has a surprising PEFT shortcut: even when callers pass an
        # explicit frozen ``ref_model``, ``PPOTrainer.step`` checks only
        # ``self.is_peft_model`` and computes reference logprobs by disabling the
        # policy adapter.  For an SFT LoRA policy that anchors KL to the BARE base
        # model, not to the SFT snapshot.  The symptom in both quota70 smokes was
        # objective/kl=65.44 on the very first batch although policy and the
        # requested SFT reference started from identical weights.
        explicit_ref = kwargs.get("ref_model")
        if explicit_ref is None and len(args) >= 3:
            explicit_ref = args[2]
        super().__init__(*args, **kwargs)
        self._policy_is_peft_model = bool(getattr(self.model, "is_peft_model", False))
        self._uses_explicit_reference = explicit_ref is not None
        if self._uses_explicit_reference:
            if self.ref_model is None:
                raise RuntimeError(
                    "An explicit PPO reference was provided but TRL discarded it. "
                    "Refusing to fall back to the bare base model."
                )
            if self._policy_is_peft_model:
                # This flag is a TRL reference-dispatch switch here; it does not
                # change the wrapped model's own ``is_peft_model`` property, LoRA
                # trainability, or adapter saving.  With False, TRL's step() uses
                # self.ref_model instead of self.model under disable_adapter().
                self.is_peft_model = False
                self.optional_peft_ctx = nullcontext
            logger.info(
                "PPO reference mode: explicit frozen model snapshot "
                "(policy_is_peft=%s; TRL bare-base PEFT shortcut disabled)",
                self._policy_is_peft_model,
            )
        self._pending_step_rewards: Optional[List[torch.Tensor]] = None
        self._require_step_rewards: bool = True
        # Store PRE-whitening statistics. Whitened variance is ~1 by
        # construction and therefore cannot diagnose exploding/oscillating GAE.
        self._last_adv_var: float = 0.0
        self._last_adv_stats: Dict[str, float] = {}

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
        if getattr(self, "_runtime_contract_version", "legacy") == "v2":
            for i, reward in enumerate(token_rewards):
                if not isinstance(reward, torch.Tensor) or reward.ndim != 1 or reward.numel() == 0:
                    raise ValueError(f"PPO v2 pending reward {i} must be a nonempty vector")
                if not bool(torch.isfinite(reward).all()):
                    raise ValueError(f"PPO v2 pending reward {i} is non-finite")
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
        strict = getattr(self, "_runtime_contract_version", "legacy") == "v2"
        if (strict or getattr(self, "_require_step_rewards", False)) and pending is None:
            raise RuntimeError(
                "StepRewardPPOTrainer.compute_rewards called without pending step "
                "rewards. Call set_pending_step_rewards(token_reward_list) before "
                "every trainer.step(...). (Set _require_step_rewards=False to allow "
                "the vanilla last-token scalar fallback.)"
            )
        if strict:
            batch_sizes = [len(scores), len(logprobs), len(ref_logprobs), len(masks), len(pending)]
            if len(set(batch_sizes)) != 1:
                raise ValueError(f"PPO v2 reward batch length mismatch: {batch_sizes}")
        rewards, non_score_rewards, kls = [], [], []
        for i, (score, logprob, ref_logprob, mask) in enumerate(
            zip(scores, logprobs, ref_logprobs, masks)
        ):
            if strict and logprob.shape != ref_logprob.shape:
                raise ValueError(f"PPO v2 policy/reference token shape mismatch on sample {i}")
            # --- identical to TRL: per-token KL penalty as the non-score reward
            kl = self._kl_penalty(logprob, ref_logprob)
            if strict:
                if mask.ndim != 1 or kl.shape != mask.shape:
                    raise ValueError(f"PPO v2 KL/mask shape mismatch on sample {i}")
                if not bool(((mask == 0) | (mask == 1)).all()):
                    raise ValueError(f"PPO v2 response mask {i} is not binary")
                if not bool(torch.isfinite(kl).all()):
                    raise ValueError(f"PPO v2 KL reward {i} is non-finite")
            kls.append(kl)
            non_score_reward = -self.kl_ctl.value * kl
            if strict and not bool(torch.isfinite(non_score_reward).all()):
                raise ValueError(f"PPO v2 non-score reward {i} is non-finite")
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()

            mask_idx = mask.nonzero()
            if pending is not None and i < len(pending) and mask_idx.numel() > 0:
                # --- our change: scatter step rewards onto the masked region.
                # The masked positions (mask==1) are exactly the response
                # tokens; align our per-response-token rewards to them in order.
                resp_positions = mask_idx.squeeze(-1)  # ascending token indices
                step_r = pending[i].to(reward.device, reward.dtype)
                if strict:
                    if step_r.ndim != 1 or step_r.numel() != resp_positions.numel():
                        raise ValueError(
                            f"PPO v2 reward/mask token length mismatch on sample {i}: "
                            f"{step_r.numel()} != {resp_positions.numel()}"
                        )
                    if not bool(torch.isfinite(step_r).all()):
                        raise ValueError(f"PPO v2 token reward {i} is non-finite after dtype conversion")
                    if resp_positions.numel() > 1 and not bool((resp_positions.diff() == 1).all()):
                        raise ValueError(f"PPO v2 response mask {i} is not contiguous")
                n = min(resp_positions.numel(), step_r.numel())
                if n > 0:
                    reward[resp_positions[:n]] += step_r[:n]
            elif mask_idx.numel() > 0:
                # --- fallback: upstream behaviour (scalar on last token).
                reward[mask_idx[-1]] += score
            elif strict:
                raise ValueError(f"PPO v2 response mask {i} has no generated tokens")
            if strict and not bool(torch.isfinite(reward).all()):
                raise ValueError(f"PPO v2 combined token reward {i} is non-finite")
            rewards.append(reward)

        # Consume the buffer so a missing set_pending_step_rewards on the next
        # step is caught by the guard above rather than silently reusing stale
        # rewards.
        self._pending_step_rewards = None
        return torch.stack(rewards), torch.stack(non_score_rewards), torch.stack(kls)

    def compute_advantages(
        self,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        mask: torch.FloatTensor,
    ):
        """Override to log advantage statistics before/after whitening."""
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1]

        values = values * mask
        rewards = rewards * mask

        if self.config.whiten_rewards:
            rewards = self._masked_whiten_debug(rewards, mask, shift_mean=False, label="rewards")

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values

        # Diagnostics before whitening. These are the quantities whose scale can
        # reveal unstable rewards/value estimates; the old persisted metric was
        # measured after whitening and was therefore exactly ~1 every update.
        raw_adv = advantages[mask.bool()]
        raw_values = values[mask.bool()]
        raw_returns = returns[mask.bool()]
        if raw_adv.numel() == 0:
            raise ValueError("compute_advantages received a mask with no valid tokens")

        def _summary(prefix: str, x: torch.Tensor) -> Dict[str, float]:
            x = x.detach().float()
            return {
                f"{prefix}_mean": float(x.mean().item()),
                f"{prefix}_std": float(x.std(unbiased=False).item()),
                f"{prefix}_min": float(x.min().item()),
                f"{prefix}_max": float(x.max().item()),
            }

        adv_stats = _summary("raw", raw_adv)
        adv_stats.update(_summary("value", raw_values))
        adv_stats.update(_summary("return", raw_returns))
        return_var = raw_returns.detach().float().var(unbiased=False)
        if return_var.item() > 0:
            residual_var = (raw_returns - raw_values).detach().float().var(unbiased=False)
            adv_stats["explained_variance"] = float((1.0 - residual_var / return_var).item())
        else:
            adv_stats["explained_variance"] = float("nan")
        for q, name in ((0.50, "p50"), (0.90, "p90"), (0.95, "p95"), (0.99, "p99")):
            adv_stats[f"raw_{name}"] = float(torch.quantile(raw_adv.detach().float(), q).item())
        logger.info(
            "ADV_DEBUG before_whiten: n=%d mean=%.6f std=%.6f min=%.4f max=%.4f",
            raw_adv.numel(), raw_adv.mean().item(), raw_adv.std(unbiased=False).item(),
            raw_adv.min().item(), raw_adv.max().item(),
        )

        advantages = self._masked_whiten_debug(advantages, mask, label="advantages")
        advantages = advantages.detach()

        # Debug: after whitening
        wh_adv = advantages[mask.bool()]
        logger.info(
            "ADV_DEBUG after_whiten:  n=%d mean=%.6f std=%.6f min=%.4f max=%.4f",
            wh_adv.numel(), wh_adv.mean().item(), wh_adv.std(unbiased=False).item(),
            wh_adv.min().item(), wh_adv.max().item(),
        )
        adv_stats.update(
            whitened_mean=float(wh_adv.mean().item()),
            # TRL's masked_whiten normalises with the unbiased/sample variance,
            # so these are ~1 by construction (for n > 1).
            whitened_std=float(wh_adv.std().item()),
            whitened_var=float(wh_adv.var().item()),
            whitened_max_abs=float(wh_adv.abs().max().item()),
        )
        self._last_adv_stats = adv_stats
        self._last_adv_var = adv_stats["raw_std"] ** 2

        return values, advantages, returns

    @staticmethod
    def _masked_whiten_debug(tensor, mask, shift_mean=True, label=""):
        """Whiten with debug logging."""
        from trl.core import masked_whiten as _mw
        result = _mw(tensor, mask, shift_mean=shift_mean)
        # only log first call per batch
        return result
