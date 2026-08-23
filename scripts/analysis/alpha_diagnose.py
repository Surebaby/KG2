#!/usr/bin/env python
"""Reconstruct the α-gate's hidden inputs from a finished eval run (P6).

WHY THIS EXISTS
---------------
``statistics.md`` §2 compared alpha_mean across datasets (HotpotQA 0.292 /
2Wiki 0.804 / MuSiQue 0.854) and the paper read that spread as the gate adapting
to intrinsic dataset properties. It is not: those three runs were produced by
different code states and a different entity-linker cache, so they are three
different functions, not three points on one curve. Two independent signs:

* α correlates with graph density at r=+0.96..0.98 WITHIN every dataset, yet
  ACROSS datasets the ordering inverts (MuSiQue has the LOWEST density and the
  HIGHEST α). One monotone gate cannot produce both.
* the HotpotQA run's reconstructed ``W1*conf + W2*ent`` term has sd = 0.000 over
  all 267 questions and equals W2 exactly, i.e. f_confidence was identically 0
  and f_entropy identically 1.0.

α has only three inputs and the gate is a single sigmoid, so given the run's
``kg_subgraphs`` (from which density is recomputable) and the gate checkpoint,
the remaining term is solvable in closed form:

    α = σ((W0·dens + W1·conf + W2·ent + b) / τ)
    =>  W1·conf + W2·ent = logit(α)·τ − b − W0·dens        (`residual` below)

That residual is what this script reports, plus the two boundary solutions
(assume ent=1.0 → solve conf; assume conf=1.0 → solve ent) and whether each is
inside [0,1]. An infeasible ent=1.0 solution is positive evidence that the run
used REAL logprobs (D1, commit e6b2198); a residual with sd=0 is positive
evidence that the entity cache was empty and link_confidence returned a constant.

USAGE
    python scripts/analysis/alpha_diagnose.py \
        --run outputs/wiki18_eval/musique_kg/.../intermediate_data.json \
        --gate checkpoints/prm_alpha_gate_v1reann_negfix/alpha_gate.pt

    # compare several runs; adds a cross-run verdict
    python scripts/analysis/alpha_diagnose.py --run A/intermediate_data.json \
        --run B/intermediate_data.json --gate ... --bias_correction 0.78

``--bias_correction`` mirrors the +0.78 that kg_proweight_pipeline.py adds to the
gate bias at load time when f_entropy is forced to 1.0. Pass the value the run
actually used; the residual shifts by exactly this amount, and the ent=1.0
feasibility test is only meaningful with it applied.

Comparability rule (AGENTS.md §5): alpha_mean from two runs may be compared ONLY
when their gate checkpoint, bias correction, logprobs mode and entity-cache state
all match. This script cannot see the cache, so it infers its effect from the
residual's spread and says so rather than claiming to have measured it.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def graph_density(triples: Sequence[Sequence[str]]) -> float:
    """|E| / (|V| + eps), matching kgproweight.kg.coverage.graph_density."""
    if not triples:
        return 0.0
    nodes = set()
    for h, _, t in triples:
        nodes.add(h)
        nodes.add(t)
    return len(triples) / (len(nodes) + 1e-6)


def _flatten_subgraphs(kg) -> List[Tuple[str, str, str]]:
    """``kg_subgraphs`` is written either as a flat triple list or as one list
    per step. Detect which by looking at the first element's shape."""
    if not kg:
        return []
    first = kg[0]
    if isinstance(first, (list, tuple)) and len(first) == 3 and isinstance(first[0], str):
        return [tuple(t) for t in kg]
    return [tuple(t) for g in kg for t in g]


def load_gate(path: Path) -> Dict[str, float]:
    import torch  # local: the script is useful without torch only for --W/--b

    sd = torch.load(path, map_location="cpu")
    tau = float(torch.clamp(torch.exp(sd["log_tau"]), min=0.1))
    return {"W": [float(x) for x in sd["W"]], "b": float(sd["b"]), "tau": tau}


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def residuals(run: Path, W: Sequence[float], b: float, tau: float) -> dict:
    """Per-question residual W1*conf + W2*ent, plus density and alpha."""
    data = json.loads(run.read_text(encoding="utf-8"))
    alphas: List[float] = []
    dens: List[float] = []
    res: List[float] = []
    n_empty_kg = 0
    n_no_alpha = 0
    step_res: List[float] = []
    for item in data:
        out = item.get("output") or {}
        stats = out.get("alpha_stats") or {}
        am = stats.get("alpha_mean")
        if am is None or not stats.get("num_steps"):
            n_no_alpha += 1
            continue
        tri = _flatten_subgraphs(out.get("kg_subgraphs") or [])
        if not tri:
            # density=0 is a different regime (the gate sees a different input),
            # so these are counted and excluded rather than mixed in.
            n_empty_kg += 1
            continue
        d = graph_density(tri)
        a = min(max(float(am), 1e-9), 1.0 - 1e-9)
        res.append(math.log(a / (1 - a)) * tau - b - W[0] * d)
        alphas.append(float(am))
        dens.append(d)
        for av in stats.get("alpha_values") or []:
            av = min(max(float(av), 1e-9), 1.0 - 1e-9)
            step_res.append(math.log(av / (1 - av)) * tau - b - W[0] * d)
    return {
        "n": len(res), "n_empty_kg": n_empty_kg, "n_no_alpha": n_no_alpha,
        "alpha": alphas, "density": dens, "residual": res, "step_residual": step_res,
    }


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((c - my) ** 2 for c in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (c - my) for a, c in zip(x, y)) / (sx * sy)


def solve_boundaries(residual: float, W: Sequence[float]) -> dict:
    """The two one-unknown solutions of W1*conf + W2*ent = residual."""
    conf_at_ent1 = (residual - W[2]) / W[1] if W[1] else float("nan")
    ent_at_conf1 = (residual - W[1]) / W[2] if W[2] else float("nan")
    return {
        "conf_at_ent1": conf_at_ent1,
        "conf_at_ent1_feasible": -1e-3 <= conf_at_ent1 <= 1 + 1e-3,
        "ent_at_conf1": ent_at_conf1,
        "ent_at_conf1_feasible": -1e-3 <= ent_at_conf1 <= 1 + 1e-3,
    }


# Two constants below are thresholds for the two degeneracy verdicts. Both are
# stated as measured quantities rather than tuned: DEGENERATE_SD is far below any
# real run's spread (the healthy runs measured 0.044-0.058), and the "== W2"
# check is exact arithmetic, not a fit.
DEGENERATE_SD = 1e-6


def classify(rep: dict, W: Sequence[float]) -> List[str]:
    """Verdicts about this run's α, most serious first."""
    notes: List[str] = []
    sd = rep["residual_sd"]
    if rep["n"] < 1:
        return ["NO QUESTIONS with a non-empty subgraph"]
    # The DEGENERATE verdict is about SPREAD, so it needs several questions; the
    # boundary-feasibility verdict is exact algebra on the mean and stays valid
    # at n=1. Guarding both at n>=3 silently withheld the logprobs verdict from
    # small runs, which is the one that survives a tiny sample.
    if rep["n"] >= 3 and sd <= DEGENERATE_SD:
        notes.append(
            "DEGENERATE: the residual is CONSTANT across every question "
            f"(sd={sd:.2e}), so f_confidence and f_entropy did not vary at all. "
            "α is then a pure function of graph density and carries no "
            "link-confidence or entropy information."
        )
        if abs(rep["residual_mean"] - W[2]) < 1e-2:
            notes.append(
                f"  residual == W2 ({W[2]:+.4f}) exactly => f_confidence == 0.0 and "
                "f_entropy == 1.0. f_confidence=0 is what "
                "EntityLinker.link_confidence returns when its cache is EMPTY "
                "(`if not cache_items: return 0.0`), so this run predates the "
                "cache being populated. Its alpha_mean is NOT comparable to a "
                "run with a warm cache."
            )
    # STRONGEST verdict available, and it needs no comparison run: the residual
    # W1*conf + W2*ent is bounded by [W2, W1] for conf,ent in [0,1] (W1>0, W2<0).
    # A reconstructed residual OUTSIDE that interval cannot be produced by any
    # (conf, entropy) whatsoever, so the assumed bias_correction is not the one
    # that run used -- the alpha_mean and the bias are mutually inconsistent.
    # This is what identifies statistics.md's HotpotQA α=0.292: it needs a
    # residual of -0.647 against a floor of -0.583, and is only reachable with
    # bias_correction=0.0, contradicting §3.4's claim that +0.78 was applied.
    lo, hi = W[2], W[1]
    rm = rep["residual_mean"]
    if rm < lo - 1e-9 or rm > hi + 1e-9:
        need = "lower" if rm < lo else "higher"
        notes.insert(0,
            f"BIAS MISMATCH: the reconstructed residual {rm:+.4f} is OUTSIDE the "
            f"feasible range [{lo:+.4f}, {hi:+.4f}], so NO (f_confidence, "
            f"f_entropy) in [0,1]^2 produces this α at this density under "
            f"bias_correction={rep['gate']['bias_correction']:+.2f}. This run used a "
            f"{need} effective bias than assumed -- its α and the bias correction "
            "reported alongside it cannot both be right. Re-diagnose with the "
            "bias this run actually ran under before using its α for anything."
        )
    b = rep["boundaries"]
    if not b["conf_at_ent1_feasible"] and b["ent_at_conf1_feasible"]:
        notes.append(
            f"REAL LOGPROBS: no solution with f_entropy=1.0 (conf would be "
            f"{b['conf_at_ent1']:+.3f}, outside [0,1]); consistent with "
            f"f_entropy≈{b['ent_at_conf1']:.3f} from actual per-token logprobs "
            "(D1, commit e6b2198). Runs with forced f_entropy=1.0 are a "
            "different function -- W2 is negative, so the difference shifts α."
        )
    elif b["conf_at_ent1_feasible"] and not b["ent_at_conf1_feasible"]:
        notes.append(
            f"NO LOGPROBS: consistent with f_entropy=1.0 forced and "
            f"f_confidence≈{b['conf_at_ent1']:.3f}; the f_entropy=1.0 branch is "
            "the pre-D1 behaviour."
        )
    if rep["n"] >= 3 and rep["r_alpha_density"] > 0.9:
        notes.append(
            f"α is essentially a function of graph density alone within this run "
            f"(r={rep['r_alpha_density']:+.3f})."
        )
    return notes


def analyse(run: Path, gate: dict, bias_correction: float) -> dict:
    W, b, tau = gate["W"], gate["b"] + bias_correction, gate["tau"]
    raw = residuals(run, W, b, tau)
    if raw["n"] == 0:
        return {"run": str(run), "n": 0, "n_empty_kg": raw["n_empty_kg"],
                "n_no_alpha": raw["n_no_alpha"], "notes": ["no usable questions"]}
    res = raw["residual"]
    rep = {
        "run": str(run),
        "n": raw["n"], "n_empty_kg": raw["n_empty_kg"], "n_no_alpha": raw["n_no_alpha"],
        "gate": {"W": W, "b_effective": b, "tau": tau, "bias_correction": bias_correction},
        "alpha_mean": st.mean(raw["alpha"]), "alpha_sd": st.pstdev(raw["alpha"]),
        "density_mean": st.mean(raw["density"]), "density_sd": st.pstdev(raw["density"]),
        "residual_mean": st.mean(res), "residual_sd": st.pstdev(res),
        "distinct_step_residuals": len({round(v, 6) for v in raw["step_residual"]}),
        "n_steps": len(raw["step_residual"]),
        "r_alpha_density": pearson(raw["alpha"], raw["density"]),
    }
    rep["boundaries"] = solve_boundaries(rep["residual_mean"], W)
    rep["notes"] = classify(rep, W)
    return rep


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True,
                   help="intermediate_data.json from an eval run (repeatable).")
    p.add_argument("--gate", default="checkpoints/prm_alpha_gate_v1reann_negfix/alpha_gate.pt",
                   help="alpha_gate.pt whose W/b/tau produced these alphas.")
    p.add_argument("--bias_correction", type=float, default=0.0,
                   help="Additive gate-bias correction the ANALYSED RUN used. "
                        "Must match that run, not your preference. Default 0.0 "
                        "tracks the pipeline default as of 2026-08-23; pass 0.78 "
                        "for runs produced before that date. A wrong value is "
                        "usually caught by the feasibility check below (that is "
                        "how the P6 mismatch was found), but not always.")
    p.add_argument("--output", default=None, help="Write the full report as JSON.")
    args = p.parse_args()

    gate = load_gate(Path(args.gate))
    print(f"gate {args.gate}")
    print(f"  W={[round(x,4) for x in gate['W']]}  b={gate['b']:+.4f}"
          f"  tau={gate['tau']:.4f}  bias_correction={args.bias_correction:+.2f}"
          f"  => b_effective={gate['b']+args.bias_correction:+.4f}")
    print(f"  feasible range of W1*conf + W2*ent for conf,ent in [0,1]: "
          f"[{gate['W'][2]:+.3f}, {gate['W'][1]:+.3f}]")
    print()

    reps = [analyse(Path(r), gate, args.bias_correction) for r in args.run]
    for rep in reps:
        print(f"=== {rep['run']}")
        if not rep["n"]:
            print(f"    no usable questions (empty KG {rep['n_empty_kg']}, "
                  f"no alpha {rep['n_no_alpha']})")
            continue
        print(f"    n={rep['n']} (excluded: {rep['n_empty_kg']} empty subgraph, "
              f"{rep['n_no_alpha']} no alpha_stats)")
        print(f"    alpha   = {rep['alpha_mean']:.4f} (sd {rep['alpha_sd']:.4f})")
        print(f"    density = {rep['density_mean']:.4f} (sd {rep['density_sd']:.4f})"
              f"   r(alpha,density) = {rep['r_alpha_density']:+.3f}")
        print(f"    W1*conf + W2*ent = {rep['residual_mean']:+.4f} "
              f"(sd {rep['residual_sd']:.4f});  distinct per-step values: "
              f"{rep['distinct_step_residuals']}/{rep['n_steps']}")
        b = rep["boundaries"]
        print(f"      if ent=1.0 -> conf={b['conf_at_ent1']:+.3f} "
              f"{'FEASIBLE' if b['conf_at_ent1_feasible'] else 'INFEASIBLE'}"
              f"   |  if conf=1.0 -> ent={b['ent_at_conf1']:+.3f} "
              f"{'FEASIBLE' if b['ent_at_conf1_feasible'] else 'INFEASIBLE'}")
        for note in rep["notes"]:
            print(f"    ! {note}")
        print()

    usable = [r for r in reps if r.get("n")]
    if len(usable) > 1:
        print("=== CROSS-RUN COMPARABILITY ===")
        # The residual is the whole non-density part of the gate input. Runs whose
        # residual distributions do not overlap were driven by different conf/ent
        # regimes, so their alpha_means measure different functions.
        spread = max(r["residual_mean"] for r in usable) - min(r["residual_mean"] for r in usable)
        worst_sd = max(r["residual_sd"] for r in usable)
        print(f"  residual spread across runs = {spread:.3f}; "
              f"largest within-run sd = {worst_sd:.3f}")
        if spread > 3 * max(worst_sd, 1e-6):
            print("  ⚠️  NOT COMPARABLE: the between-run shift in (conf, entropy) is")
            print("      larger than the within-run variation, so these alpha_means")
            print("      come from different input regimes -- different code state,")
            print("      bias correction, logprobs mode, or entity-cache warmth.")
            print("      Do NOT tabulate them as one cross-dataset comparison")
            print("      (AGENTS.md §5). Re-run all arms under one fixed state.")
        else:
            print("  comparable: residuals overlap within run-level noise.")
        # A density/alpha ordering inversion is independently diagnostic and does
        # not depend on the threshold above.
        by_d = sorted(usable, key=lambda r: r["density_mean"])
        by_a = sorted(usable, key=lambda r: r["alpha_mean"])
        if [r["run"] for r in by_d] != [r["run"] for r in by_a]:
            print("  ⚠️  ORDERING INVERSION: ranking by density and by alpha disagree,")
            print("      yet within each run r(alpha,density) is near +1. A single")
            print("      monotone gate cannot produce both -- further proof these")
            print("      runs are not one curve.")

    if args.output:
        Path(args.output).write_text(json.dumps(reps, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
