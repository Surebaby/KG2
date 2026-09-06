"""α-Gate unit tests.

Covers:
  * Sigmoid output stays in (0, 1).
  * Higher density + higher confidence → higher α (gate trusts KG).
  * Higher semantic entropy → lower α (gate distrusts KG).
  * ``forward_single`` is deterministic and matches the tensorised forward.
  * ``entropy_from_logprobs`` returns 0 for empty input and a positive value otherwise.
"""

from __future__ import annotations

import math

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
    loss_fn = AlphaCalibrationLoss(weight=0.2)
    logits = torch.tensor([-0.7, 1.4])
    targets = torch.tensor([0.5, 0.5])
    loss = loss_fn(logits, targets)
    assert loss.item() > 0
    # weight scales the BCE term linearly
    loss_fn2 = AlphaCalibrationLoss(weight=0.4)
    assert loss_fn2(logits, targets).item() == pytest.approx(2.0 * loss.item(), rel=1e-5)


def test_logits_calibration_is_equivalent_to_probability_bce():
    import torch.nn.functional as F

    logits = torch.tensor([-2.0, -0.1, 0.5, 3.0])
    targets = torch.tensor([0.0, 0.25, 0.75, 1.0])
    got = AlphaCalibrationLoss(weight=1.0)(logits, targets)
    expected = F.binary_cross_entropy(torch.sigmoid(logits), targets)
    assert got.item() == pytest.approx(expected.item(), rel=1e-6)


def test_logits_calibration_remains_finite_when_gate_saturates():
    logits = torch.tensor([-1000.0, 1000.0], requires_grad=True)
    targets = torch.tensor([1.0, 0.0])
    loss = AlphaCalibrationLoss(weight=1.0)(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_alpha_schema_matches_runtime_feature_order():
    from kgproweight.config.schemas import AlphaGateConfig

    cfg = AlphaGateConfig()
    assert cfg.feature_dim == AlphaGate.N_FEATURES == len(cfg.initial_W)


# ---------------------------------------------------------------------------
# alpha_bias_correction default (2026-08-23)
# ---------------------------------------------------------------------------
# The pipeline used to add +0.78 to the loaded gate bias by default. That value
# was never derived: b_trained = -1.7818 and -1.7818 + 0.78 = -1.0018, i.e. it
# was picked to land b_effective on a round -1.0. These tests pin the new
# default (0.0) and the arithmetic that rules the old one out, so neither can
# drift back silently.


def test_bias_correction_default_is_zero():
    """`None` must resolve to NO correction, not to +0.78."""
    import inspect

    from kgproweight.pipeline.kg_proweight_pipeline import KGProWeightPipeline

    src = inspect.getsource(KGProWeightPipeline)
    assert "_correction = 0.0 if self.alpha_bias_correction is None" in src, (
        "the default resolved when alpha_bias_correction is None must be 0.0; "
        "+0.78 was reverse-engineered to make b_effective a round -1.0"
    )


def test_078_exceeds_what_the_entropy_mechanism_allows():
    """+0.78 needs an f_entropy shift larger than f_entropy's own ceiling.

    The correction exists to compensate a regime shift Δe = e_inf - e_tr in
    f_entropy, entering the α-logit through W2. So the compensating additive
    bias is -W2·Δe. With the shipped W2 = -0.5833, +0.78 implies Δe = 1.337 --
    impossible, since the measured e_inf is ~0.62 and entropy_from_logprobs
    floors at 0, bounding Δe by 0.62.
    """
    W2 = -0.583338737487793
    e_inf = 0.6215  # mean back-solved from the three bias=0 runs (P6)

    implied_delta = 0.78 / -W2
    assert implied_delta > e_inf, "the premise of this test has changed"

    max_defensible = -W2 * e_inf  # Δe at its ceiling, i.e. e_tr = 0
    assert max_defensible == pytest.approx(0.3626, abs=5e-4)
    assert 0.78 > 2.0 * max_defensible, "+0.78 should be >2x the defensible ceiling"


# ---------------------------------------------------------------------------
# §14 (2026-08-23): the gate grew two per-step citation features.
#
# The three original features hit a measured R^2 ceiling of +0.038 on the gate's
# own BCE target; cite_any alone reaches +0.300 and both together +0.439. These
# tests pin the two properties that make the change safe to ship: an OLD 3-feature
# checkpoint must keep producing exactly the alpha it did before, and the new
# features must actually be able to move alpha once they carry weight.
# ---------------------------------------------------------------------------


def test_old_3feature_checkpoint_loads_and_reproduces_old_alpha(tmp_path):
    """A 3-weight checkpoint zero-pads, which IS the old gate exactly."""
    import torch
    from kgproweight.reward.alpha_gate import AlphaGate

    old = {
        "W": torch.tensor([1.2096, 1.7088, -0.5833]),
        "b": torch.tensor(-1.7818),
        "log_tau": torch.tensor(math.log(0.5994)),
    }
    path = tmp_path / "old_gate.pt"
    torch.save(old, path)

    g = AlphaGate()
    g.load_state_dict(torch.load(path, map_location="cpu"))
    assert g.W.numel() == AlphaGate.N_FEATURES
    assert float(g.W[3]) == 0.0 and float(g.W[4]) == 0.0

    d, c, e = 0.8764, 0.9070, 0.603
    tau = 0.5994
    expect = 1.0 / (1.0 + math.exp(-((1.2096 * d + 1.7088 * c - 0.5833 * e) - 1.7818) / tau))
    assert g.forward_single(d, c, e) == pytest.approx(expect, abs=1e-4)
    # And the padded weights make the new features inert on an old checkpoint.
    assert g.forward_single(d, c, e, 1.0, 1.0) == pytest.approx(
        g.forward_single(d, c, e, 0.0, 0.0), abs=1e-6
    )


def test_wider_checkpoint_is_refused_not_truncated():
    """Truncating a trained weight would change alpha with no warning."""
    import torch
    from kgproweight.reward.alpha_gate import AlphaGate

    too_wide = {
        "W": torch.zeros(6),
        "b": torch.tensor(-2.0),
        "log_tau": torch.tensor(math.log(0.5)),
    }
    with pytest.raises(ValueError, match="Refusing to truncate"):
        AlphaGate().load_state_dict(too_wide)


def test_citation_features_can_move_alpha():
    from kgproweight.reward.alpha_gate import AlphaGate

    g = AlphaGate(init_weights=(1.0, 1.5, -0.8, 0.9, 1.0), init_bias=-2.0, init_tau=0.5)
    no_cite = g.forward_single(0.5, 0.5, 0.6, 0.0, 0.0)
    cited = g.forward_single(0.5, 0.5, 0.6, 1.0, 1.0)
    assert cited > no_cite, "citing grounded triples must raise the KG weight"


def test_forward_defaults_citation_features_to_zero():
    """A legacy 3-argument call must behave as 'no citations observed'."""
    import torch
    from kgproweight.reward.alpha_gate import AlphaGate

    g = AlphaGate()
    t = lambda v: torch.tensor([v])  # noqa: E731
    three = g.forward(t(0.5), t(0.5), t(0.6))
    five = g.forward(t(0.5), t(0.5), t(0.6), t(0.0), t(0.0))
    assert float(three) == pytest.approx(float(five), abs=1e-7)
