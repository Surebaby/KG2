"""paired_stats.py — the tests that matter are the ones pinning WHY it exists.

The script replaces independent two-proportion tests, whose insensitivity is the
reason every KG ablation currently reads "NS". So the central test is
test_mcnemar_beats_independent_test: a consistent one-sided effect that an
independent test cannot see must come out significant here.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "paired_stats", Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "paired_stats.py"
)
ps = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ps)


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------


def test_mcnemar_identical_arms_is_p1():
    x = [1.0, 0.0, 1.0, 1.0, 0.0]
    r = ps.mcnemar(x, x)
    assert r["discordant"] == 0
    assert r["p_value"] == 1.0


def test_mcnemar_counts_discordant_pairs_directionally():
    a = [1.0, 0.0, 0.0, 1.0]
    b = [0.0, 1.0, 1.0, 1.0]
    r = ps.mcnemar(a, b)
    assert r["b_only"] == 2   # a wrong -> b right
    assert r["a_only"] == 1   # a right -> b wrong
    assert r["discordant"] == 3


def test_mcnemar_matches_hand_computed_exact_binomial():
    # 10 discordant, all favouring b: two-sided exact p = 2 * (1/2)^10.
    a = [0.0] * 10
    b = [1.0] * 10
    r = ps.mcnemar(a, b)
    assert r["p_value"] == pytest.approx(2.0 / 1024.0)


def test_mcnemar_ignores_concordant_pairs():
    """Only disagreements carry information — 1000 agreeing pairs must not
    dilute the p-value, which is exactly the sensitivity the independent test
    lacks."""
    a = [1.0] * 500 + [0.0] * 500 + [0.0] * 10
    b = [1.0] * 500 + [0.0] * 500 + [1.0] * 10
    assert ps.mcnemar(a, b)["p_value"] == pytest.approx(2.0 / 1024.0)


def test_mcnemar_rejects_non_binary():
    with pytest.raises(ValueError):
        ps.mcnemar([0.5, 1.0], [1.0, 0.0])


def test_mcnemar_beats_independent_test():
    """The whole point of P7. A +2.3pp EM effect at n=300 that is consistently
    one-sided is significant under McNemar while the independent two-proportion
    test cannot resolve it."""
    n, disc = 300, 12
    a = [1.0] * 100 + [0.0] * (n - 100)
    b = list(a)
    for i in range(100, 100 + disc):   # 12 questions a got wrong, b got right
        b[i] = 1.0
    r = ps.mcnemar(a, b)
    assert r["discordant"] == disc and r["a_only"] == 0
    assert r["p_value"] < 0.05

    # Independent two-proportion z-test on the same data: NOT significant.
    pa, pb = sum(a) / n, sum(b) / n
    pool = (sum(a) + sum(b)) / (2 * n)
    se = math.sqrt(pool * (1 - pool) * 2 / n)
    p_ind = math.erfc(abs(pb - pa) / se / math.sqrt(2))
    assert p_ind > 0.05, "fixture no longer demonstrates the sensitivity gap"


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_delta_is_the_paired_mean_difference():
    a = [0.1, 0.5, 0.9]
    b = [0.2, 0.5, 1.0]
    r = ps.paired_bootstrap(a, b, n_boot=200, seed=1)
    assert r["delta"] == pytest.approx((0.1 + 0.0 + 0.1) / 3)


def test_bootstrap_zero_variance_gives_degenerate_ci():
    """Constant difference => every resample has the same mean => CI collapses
    onto the point estimate. Guards against resampling the two arms
    independently, which would manufacture spread that isn't there."""
    a = [0.0, 0.0, 0.0, 0.0]
    b = [0.5, 0.5, 0.5, 0.5]
    r = ps.paired_bootstrap(a, b, n_boot=500, seed=7)
    lo, hi = r["ci95"]
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)


def test_bootstrap_is_seed_deterministic():
    a, b = [0.0, 1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0, 1.0]
    r1 = ps.paired_bootstrap(a, b, n_boot=300, seed=42)
    r2 = ps.paired_bootstrap(a, b, n_boot=300, seed=42)
    assert r1["ci95"] == r2["ci95"]


def test_bootstrap_ci_includes_zero_for_a_null_effect():
    a = [0.0, 1.0] * 50
    b = [0.0, 1.0] * 50
    lo, hi = ps.paired_bootstrap(a, b, n_boot=500, seed=3)["ci95"]
    assert lo <= 0.0 <= hi


# ---------------------------------------------------------------------------
# Wilcoxon
# ---------------------------------------------------------------------------


def test_wilcoxon_drops_zero_differences():
    a = [0.3, 0.3, 0.4]
    b = [0.3, 0.3, 0.9]
    assert ps.wilcoxon_signed_rank(a, b)["n_nonzero"] == 1


def test_wilcoxon_too_few_nonzero_returns_none_not_a_number():
    """A p-value from a 3-pair normal approximation would be worse than no
    p-value, because it would get quoted."""
    r = ps.wilcoxon_signed_rank([0.1, 0.2, 0.3], [0.2, 0.3, 0.4])
    assert r["p_value"] is None and "note" in r


def test_wilcoxon_detects_a_consistent_shift():
    a = [i / 100 for i in range(30)]
    b = [x + 0.05 for x in a]
    r = ps.wilcoxon_signed_rank(a, b)
    assert r["p_value"] < 0.05


def test_wilcoxon_all_ties_is_not_a_crash():
    a = b = [0.5] * 20
    r = ps.wilcoxon_signed_rank(a, b)
    assert r["p_value"] is None


# ---------------------------------------------------------------------------
# Loading / pairing
# ---------------------------------------------------------------------------


def _write_intermediate(path, rows):
    path.write_text(json.dumps([
        {"id": i, "question": "q", "golden_answers": ["g"],
         "output": {"metric_score": {"em": em, "f1": f1}}}
        for i, em, f1 in rows
    ]), encoding="utf-8")


def _write_ihr(path, rows, judge="deepseek-v4-pro"):
    path.write_text(json.dumps({
        "method": "m", "judge_model": judge, "n_items": len(rows),
        "mean_ihr": sum(r[1] for r in rows) / len(rows),
        "items": [{"id": i, "ihr": v, "steps": []} for i, v in rows],
    }), encoding="utf-8")


def test_loads_em_from_intermediate_data(tmp_path):
    f = tmp_path / "intermediate_data.json"
    _write_intermediate(f, [("dev_0", 1.0, 0.8), ("dev_1", 0.0, 0.1)])
    assert ps._load_scores(f, "em") == {"dev_0": 1.0, "dev_1": 0.0}
    assert ps._load_scores(f, "f1") == {"dev_0": 0.8, "dev_1": 0.1}


def test_loads_ihr_from_judge_result(tmp_path):
    f = tmp_path / "ihr_result_x.json"
    _write_ihr(f, [("dev_0", 0.0), ("dev_1", 0.5)])
    assert ps._load_scores(f, "ihr") == {"dev_0": 0.0, "dev_1": 0.5}


def test_wrong_metric_for_file_type_raises_not_returns_empty(tmp_path):
    """An empty dict would surface as '0 paired items', which reads like a
    pairing bug rather than a wrong --metric."""
    ihr = tmp_path / "ihr_result_x.json"
    _write_ihr(ihr, [("dev_0", 0.1)])
    with pytest.raises(ValueError, match="IHR result file"):
        ps._load_scores(ihr, "em")

    inter = tmp_path / "intermediate_data.json"
    _write_intermediate(inter, [("dev_0", 1.0, 1.0)])
    with pytest.raises(ValueError, match="no IHR"):
        ps._load_scores(inter, "ihr")


def test_pairing_is_by_id_not_by_position():
    """Row order differs between runs, so positional pairing would silently
    compare unrelated questions."""
    a = {"dev_0": 1.0, "dev_1": 0.0}
    b = {"dev_1": 0.0, "dev_0": 1.0}
    ids, xa, xb = ps._pair(a, b)
    assert ids == ["dev_0", "dev_1"]
    assert xa == [1.0, 0.0] and xb == [1.0, 0.0]


def test_pairing_drops_unshared_ids():
    ids, xa, xb = ps._pair({"a": 1.0, "b": 0.0}, {"b": 1.0, "c": 1.0})
    assert ids == ["b"] and xa == [0.0] and xb == [1.0]


def test_judge_model_is_read_for_the_mismatch_warning(tmp_path):
    f = tmp_path / "ihr_result_x.json"
    _write_ihr(f, [("dev_0", 0.1)], judge="gpt-4o-2024-08-06")
    assert ps._judge_model(f) == "gpt-4o-2024-08-06"
