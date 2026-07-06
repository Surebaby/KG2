"""EM / F1 / heuristic IHR / paired statistics."""

from __future__ import annotations

import pytest

from kgproweight.eval.metrics import (
    aggregate_metrics,
    compute_em,
    compute_f1,
    heuristic_ihr,
)
from kgproweight.eval.stats import mean_std_ci, paired_bootstrap, paired_t_test


def test_em_normalisation():
    assert compute_em("the Michelle Obama", ["Michelle Obama"]) == 1.0
    assert compute_em("michelle obama!", ["Michelle Obama"]) == 1.0
    assert compute_em("Hillary Clinton", ["Michelle Obama"]) == 0.0
    assert compute_em("Michelle  Obama", ["michelle obama"]) == 1.0


def test_f1_partial_overlap():
    # 'Barack Hussein Obama' shares 2 tokens with the gold 'Barack Obama'.
    assert compute_f1("Barack Hussein Obama", ["Barack Obama"]) == pytest.approx(2 * 2 / 5)
    # Exact match scores 1.
    assert compute_f1("Michelle Obama", ["Michelle Obama"]) == 1.0
    # No overlap → 0.
    assert compute_f1("Joe Biden", ["Michelle Obama"]) == 0.0


def test_heuristic_ihr():
    flags = [{"ihr_heuristic": 0.0}, {"ihr_heuristic": 1.0}, {"ihr_heuristic": 0.5}]
    assert heuristic_ihr(flags) == pytest.approx(0.5)
    assert heuristic_ihr([{}]) == 0.0
    assert heuristic_ihr([]) == 0.0


def test_aggregate_metrics():
    preds = ["Michelle Obama", "Hillary Clinton"]
    golds = [["Michelle Obama"], ["Michelle Obama"]]
    out = aggregate_metrics(preds, golds)
    assert out["em"] == pytest.approx(0.5)
    assert 0 < out["f1"] <= 1.0


def test_paired_bootstrap_handles_identical_inputs():
    a = [0.7, 0.8, 0.6]
    out = paired_bootstrap(a, a, n_resamples=200)
    assert out["diff_mean"] == pytest.approx(0.0)
    assert out["n"] == 3


def test_paired_bootstrap_detects_difference():
    a = [0.9, 0.8, 0.85, 0.95]
    b = [0.6, 0.55, 0.65, 0.7]
    out = paired_bootstrap(a, b, n_resamples=2000, seed=1)
    assert out["diff_mean"] > 0.2
    assert out["lower"] > 0.0


def test_paired_t_test_empty():
    out = paired_t_test([], [])
    assert out["p_value"] == 1.0


def test_mean_std_ci_single_value():
    out = mean_std_ci([0.5])
    assert out["mean"] == 0.5
    assert out["std"] == 0.0
    assert out["n"] == 1


def test_mean_std_ci_multi_value():
    out = mean_std_ci([0.4, 0.5, 0.6])
    assert out["mean"] == pytest.approx(0.5)
    assert out["std"] > 0
    assert out["lower"] < out["upper"]
    assert out["n"] == 3
