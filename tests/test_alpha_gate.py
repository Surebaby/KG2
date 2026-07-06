"""α-Gate unit tests.

Covers:
  * Sigmoid output stays in (0, 1).
  * Higher density + higher confidence → higher α (gate trusts KG).
  * Higher semantic entropy → lower α (gate distrusts KG).
  * ``forward_single`` is deterministic and matches the tensorised forward.
  * ``entropy_from_logprobs`` returns 0 for empty input and a positive value otherwise.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from kgproweight.reward.alpha_gate import (
    AlphaCalibrationLoss,
    AlphaGate,
    compute_semantic_entropy,
    entropy_from_logprobs,
)


def _gate():
    return AlphaGate(init_weights=(1.0, 1.5, -0.8), init_bias=-2.0, init_tau=0.5)


def test_alpha_in_unit_interval():
    gate = _gate()
    for d in [0.0, 0.5, 1.0]:
        for c in [0.0, 0.5, 1.0]:
            for e in [0.0, 1.0, 3.0]:
                a = gate.forward_single(d, c, e)
                assert 0.0 < a < 1.0


def test_density_increases_alpha():
    gate = _gate()
    low = gate.forward_single(0.1, 0.5, 0.5)
    high = gate.forward_single(0.9, 0.5, 0.5)
    assert high > low


def test_confidence_increases_alpha():
    gate = _gate()
    low = gate.forward_single(0.5, 0.1, 0.5)
    high = gate.forward_single(0.5, 0.9, 0.5)
    assert high > low


def test_entropy_decreases_alpha():
    gate = _gate()
    low_e = gate.forward_single(0.5, 0.5, 0.1)
    high_e = gate.forward_single(0.5, 0.5, 3.0)
    assert low_e > high_e


def test_forward_matches_single():
    gate = _gate()
    d = torch.tensor([0.3, 0.7])
    c = torch.tensor([0.2, 0.9])
    e = torch.tensor([1.0, 0.4])
    batch = gate(d, c, e)
    assert batch.shape == (2,)
    for i in range(2):
        single = gate.forward_single(float(d[i]), float(c[i]), float(e[i]))
        assert abs(single - float(batch[i])) < 1e-5


def test_entropy_from_logprobs():
    assert entropy_from_logprobs([]) == pytest.approx(1.0)  # documented default
    val = entropy_from_logprobs([-0.5, -1.0, -1.5])
    assert val == pytest.approx(1.0)
    val2 = entropy_from_logprobs([-3.0, -3.0])
    assert val2 > val
    # backward-compat alias
    assert compute_semantic_entropy([-0.5, -1.0]) == pytest.approx(entropy_from_logprobs([-0.5, -1.0]))


def test_calibration_loss():
    gate = _gate()
    loss_fn = AlphaCalibrationLoss(weight=0.2)
    alpha = torch.tensor([0.3, 0.8])
    targets = torch.tensor([0.5, 0.5])
    loss = loss_fn(alpha, targets)
    assert loss.item() > 0
    # weight scales the BCE term linearly
    loss_fn2 = AlphaCalibrationLoss(weight=0.4)
    assert loss_fn2(alpha, targets).item() == pytest.approx(2.0 * loss.item(), rel=1e-5)
