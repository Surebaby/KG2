"""Regression tests for pre-whitening PPO advantage telemetry."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from kgproweight.training.step_reward_ppo_trainer import StepRewardPPOTrainer


def test_advantage_variance_is_recorded_before_whitening():
    # Avoid constructing a full TRL trainer; compute_advantages only needs its
    # scalar config and the diagnostic attributes under test.
    trainer = object.__new__(StepRewardPPOTrainer)
    trainer.config = types.SimpleNamespace(whiten_rewards=False, gamma=0.95, lam=0.95)
    trainer._last_adv_var = 0.0
    trainer._last_adv_stats = {}

    values = torch.tensor([[0.2, -0.1, 0.3], [0.0, 0.4, -0.2]])
    rewards = torch.tensor([[3.0, -1.0, 0.5], [-2.0, 4.0, 1.5]])
    mask = torch.ones_like(rewards)
    _, whitened, _ = trainer.compute_advantages(values, rewards, mask)

    stats = trainer._last_adv_stats
    assert trainer._last_adv_var == pytest.approx(stats["raw_std"] ** 2)
    assert stats["whitened_var"] == pytest.approx(1.0, rel=1e-5)
    assert trainer._last_adv_var != pytest.approx(stats["whitened_var"], rel=1e-3)
    assert stats["raw_p99"] >= stats["raw_p95"] >= stats["raw_p90"]
    assert torch.isfinite(torch.tensor(stats["explained_variance"]))
    assert torch.isfinite(whitened).all()
