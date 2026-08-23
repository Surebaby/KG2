"""alpha_diagnose.py — the P6 forensic reconstruction.

These tests build runs with KNOWN (density, conf, entropy) by pushing them
through the real sigmoid, then check the script recovers what was put in. That
is the only honest way to test a reconstruction: a hand-written expected number
would just re-encode the same algebra.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "alpha_diagnose",
    Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "alpha_diagnose.py",
)
ad = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ad)

# The shipped gate, so the tests exercise the parameters that produced the
# numbers in statistics.md rather than round toy values.
W = [1.2096, 1.7088, -0.5833]
B = -1.7818
TAU = 0.5994


def _alpha(dens: float, conf: float, ent: float, b: float = B) -> float:
    z = (W[0] * dens + W[1] * conf + W[2] * ent + b) / TAU
    return 1.0 / (1.0 + math.exp(-z))


def _make_run(tmp_path, rows, name="intermediate_data.json"):
    """rows: list of (n_triples_chain, conf, ent). A chain of n triples over
    n+1 distinct nodes gives density n/(n+1), so density is controllable."""
    items = []
    for i, (n_tri, conf, ent) in enumerate(rows):
        tri = [[f"e{i}_{k}", "r", f"e{i}_{k+1}"] for k in range(n_tri)]
        dens = ad.graph_density([tuple(t) for t in tri]) if tri else 0.0
        a = _alpha(dens, conf, ent)
        items.append({
            "id": f"q{i}",
            "output": {
                "kg_subgraphs": tri,
                "alpha_stats": {"num_steps": 1, "alpha_mean": a, "alpha_values": [a]},
            },
        })
    p = tmp_path / name
    p.write_text(json.dumps(items), encoding="utf-8")
    return p


GATE = {"W": W, "b": B, "tau": TAU}


# ---------------------------------------------------------------------------
# density
# ---------------------------------------------------------------------------


def test_density_matches_the_shipped_formula():
    # 2 triples over 3 nodes -> 2/(3+eps)
    tri = [("a", "r", "b"), ("b", "r", "c")]
    assert ad.graph_density(tri) == pytest.approx(2 / 3, rel=1e-5)
    assert ad.graph_density([]) == 0.0


def test_flatten_handles_both_subgraph_layouts():
    flat = [["a", "r", "b"], ["b", "r", "c"]]
    nested = [[["a", "r", "b"]], [["b", "r", "c"]]]
    assert ad._flatten_subgraphs(flat) == [("a", "r", "b"), ("b", "r", "c")]
    assert ad._flatten_subgraphs(nested) == [("a", "r", "b"), ("b", "r", "c")]
    assert ad._flatten_subgraphs([]) == []


# ---------------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------------


def test_recovers_the_residual_it_was_built_from(tmp_path):
    conf, ent = 0.4, 0.7
    run = _make_run(tmp_path, [(3, conf, ent), (5, conf, ent), (9, conf, ent)])
    rep = ad.analyse(run, GATE, bias_correction=0.0)
    assert rep["residual_mean"] == pytest.approx(W[1] * conf + W[2] * ent, abs=1e-6)
    assert rep["residual_sd"] == pytest.approx(0.0, abs=1e-6)


def test_bias_correction_shifts_the_residual_by_exactly_itself(tmp_path):
    run = _make_run(tmp_path, [(4, 0.5, 0.5), (8, 0.5, 0.5)])
    a = ad.analyse(run, GATE, bias_correction=0.0)["residual_mean"]
    b = ad.analyse(run, GATE, bias_correction=0.78)["residual_mean"]
    assert a - b == pytest.approx(0.78, abs=1e-6)


def test_boundary_solutions_are_exact():
    """Both boundary solves must satisfy the equation they claim to solve."""
    r = 0.42
    s = ad.solve_boundaries(r, W)
    assert W[1] * s["conf_at_ent1"] + W[2] * 1.0 == pytest.approx(r, abs=1e-9)
    assert W[1] * 1.0 + W[2] * s["ent_at_conf1"] == pytest.approx(r, abs=1e-9)


def test_empty_subgraphs_are_excluded_and_counted(tmp_path):
    """density=0 is a different input regime; mixing it in would bias the mean."""
    run = _make_run(tmp_path, [(3, 0.5, 0.5), (0, 0.5, 0.5), (4, 0.5, 0.5)])
    rep = ad.analyse(run, GATE, bias_correction=0.0)
    assert rep["n"] == 2 and rep["n_empty_kg"] == 1


# ---------------------------------------------------------------------------
# the two verdicts this script exists to deliver
# ---------------------------------------------------------------------------


def test_flags_the_empty_cache_signature(tmp_path):
    """conf==0, ent==1 for every question is the HotpotQA-0.292 signature: an
    EntityLinker with an empty cache. Must be named, not merely reported."""
    run = _make_run(tmp_path, [(3, 0.0, 1.0), (6, 0.0, 1.0), (12, 0.0, 1.0)])
    rep = ad.analyse(run, GATE, bias_correction=0.0)
    joined = " ".join(rep["notes"])
    assert "DEGENERATE" in joined
    assert "cache is EMPTY" in joined or "cache being populated" in joined
    assert rep["distinct_step_residuals"] == 1


def test_does_not_flag_a_healthy_run_as_degenerate(tmp_path):
    run = _make_run(tmp_path, [(3, 0.2, 0.4), (6, 0.7, 0.5), (12, 0.5, 0.9)])
    rep = ad.analyse(run, GATE, bias_correction=0.0)
    assert "DEGENERATE" not in " ".join(rep["notes"])
    assert rep["distinct_step_residuals"] > 1


def test_infeasible_ent1_is_reported_as_real_logprobs(tmp_path):
    """A residual above W1+W2 cannot come from ent=1.0, which is exactly how the
    MuSiQue run is identified as post-D1."""
    run = _make_run(tmp_path, [(4, 0.95, 0.3), (9, 0.95, 0.3)])
    rep = ad.analyse(run, GATE, bias_correction=0.0)
    assert not rep["boundaries"]["conf_at_ent1_feasible"]
    assert "REAL LOGPROBS" in " ".join(rep["notes"])


def test_alpha_is_monotone_in_density_within_a_fixed_regime(tmp_path):
    """Sanity check on the fixture itself: with conf/ent held fixed the gate is
    monotone in density, which is what makes the cross-run ORDERING INVERSION
    diagnostic in the first place."""
    run = _make_run(tmp_path, [(2, 0.5, 0.5), (5, 0.5, 0.5), (20, 0.5, 0.5)])
    rep = ad.analyse(run, GATE, bias_correction=0.0)
    assert rep["r_alpha_density"] > 0.9


def test_no_alpha_stats_items_are_counted_not_crashed(tmp_path):
    p = tmp_path / "intermediate_data.json"
    p.write_text(json.dumps([
        {"id": "q0", "output": {"kg_subgraphs": [["a", "r", "b"]], "alpha_stats": {}}},
    ]), encoding="utf-8")
    rep = ad.analyse(p, GATE, bias_correction=0.0)
    assert rep["n"] == 0 and rep["n_no_alpha"] == 1


# ---------------------------------------------------------------------------
# BIAS MISMATCH: the residual escaping [W2, W1] proves the assumed bias is wrong
# ---------------------------------------------------------------------------


def test_flags_residual_outside_feasible_range_as_bias_mismatch(tmp_path):
    """statistics.md's HotpotQA α=0.292 is the real instance of this.

    That α needs a residual of -0.647 while the floor is W2 = -0.583, so no
    (conf, entropy) pair in [0,1]^2 produces it under bias=+0.78. The verdict
    must say so WITHOUT a second run to compare against -- which is what makes
    it stronger than the cross-run NOT COMPARABLE heuristic.

    Construct it the way it actually arose: alphas generated with NO bias
    correction, then diagnosed as though +0.78 had been applied.
    """
    # 12 triples over 13 nodes -> density ~0.923, matching the measured run.
    run = _make_run(tmp_path, [(12, 0.0, 1.0), (12, 0.05, 1.0)])
    rep = ad.analyse(run, GATE, bias_correction=0.78)
    joined = " ".join(rep["notes"])
    assert "BIAS MISMATCH" in joined, joined
    # Must name the direction so the reader knows which way to re-diagnose.
    assert "lower" in joined
    # And it must come first: it invalidates every other verdict about this run.
    assert "BIAS MISMATCH" in rep["notes"][0]


def test_no_bias_mismatch_when_the_bias_is_right(tmp_path):
    """Must not fire on a correctly-diagnosed run."""
    run = _make_run(tmp_path, [(12, 0.6, 0.6), (12, 0.55, 0.62)])
    rep = ad.analyse(run, GATE, bias_correction=0.0)
    assert not any("BIAS MISMATCH" in n for n in rep["notes"]), rep["notes"]
