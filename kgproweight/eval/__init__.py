"""Evaluation utilities, metrics, and runners."""

from kgproweight.eval.metrics import (
    compute_em,
    compute_f1,
    heuristic_ihr,
    aggregate_metrics,
)
from kgproweight.eval.stats import paired_bootstrap, paired_t_test, mean_std_ci
from kgproweight.eval.alpha_analysis import compare_alpha_distributions, summarise_alpha
from kgproweight.eval.baselines import BASELINES, baseline_config, BaselineSpec

__all__ = [
    "compute_em",
    "compute_f1",
    "heuristic_ihr",
    "aggregate_metrics",
    "paired_bootstrap",
    "paired_t_test",
    "mean_std_ci",
    "compare_alpha_distributions",
    "summarise_alpha",
    "BASELINES",
    "baseline_config",
    "BaselineSpec",
]
