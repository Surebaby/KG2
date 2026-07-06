"""Paired statistical tests for rigorous baselines comparison.

Provides:
  • :func:`paired_bootstrap` — non-parametric 95% CI for the difference
    in per-item F1 (or any per-item metric) between two methods.
  • :func:`paired_t_test` — paired t-test wrapper.
  • :func:`mean_std_ci` — mean / std / 95 % CI for a single metric across
    seeds.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def paired_bootstrap(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, float]:
    """Bootstrap CI of ``mean(scores_a) - mean(scores_b)``.

    Returns ``{"diff_mean", "lower", "upper", "p_value", "n"}``.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired bootstrap requires equal-length samples.")
    n = a.shape[0]
    if n == 0:
        return {"diff_mean": 0.0, "lower": 0.0, "upper": 0.0, "p_value": 1.0, "n": 0}

    diffs = a - b
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot[i] = diffs[idx].mean()
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(boot, alpha))
    upper = float(np.quantile(boot, 1.0 - alpha))
    p_value = float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return {
        "diff_mean": float(diffs.mean()),
        "lower": lower,
        "upper": upper,
        "p_value": p_value,
        "n": int(n),
    }


def paired_t_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
) -> Dict[str, float]:
    """Standard paired t-test on per-item differences."""
    from scipy import stats

    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        return {"t": 0.0, "p_value": 1.0, "df": 0}
    res = stats.ttest_rel(a, b)
    return {"t": float(res.statistic), "p_value": float(res.pvalue), "df": int(a.size - 1)}


def mean_std_ci(values: Sequence[float], ci: float = 0.95) -> Dict[str, float]:
    """Return mean, std, and a normal-approx CI for a small set of seeds."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if arr.size > 1:
        from scipy import stats

        sem = std / np.sqrt(arr.size)
        half = stats.t.ppf(0.5 + ci / 2.0, df=arr.size - 1) * sem
    else:
        half = 0.0
    return {"mean": mean, "std": std, "lower": mean - half, "upper": mean + half, "n": int(arr.size)}
