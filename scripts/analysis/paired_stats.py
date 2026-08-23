#!/usr/bin/env python
"""Paired significance tests for EM/F1 and IHR (retraining_plan P7).

Every KG-ablation comparison in the paper is between two runs over the SAME
question set, so the right test is a PAIRED one. The independent two-proportion
test that produced the current "all NS" verdicts throws away that pairing and is
much less sensitive: at EM ~0.34 with n=300 the independent 95% CI half-width is
~5.3pp, which no +1.3-2.7pp effect can clear regardless of how consistent it is.
McNemar looks only at the questions where the two arms DISAGREE, so a small but
one-sided effect can be significant even when the marginal CIs overlap heavily.

Two input shapes are accepted, dispatched on file content:

* ``intermediate_data.json`` (FlashRAG) — a list of items with ``id`` and
  ``output.metric_score.{em,f1}``. EM is binary per question -> McNemar;
  F1 is continuous -> paired bootstrap + Wilcoxon signed-rank.
* ``ihr_result_*.json`` (run_ihr_judge.py / run_baseline_ihr.py) — ``items[]``
  with ``id`` and a continuous ``ihr`` in [0,1] -> paired bootstrap + Wilcoxon.
  Lower IHR is better, so the reported delta is b - a and "improvement" means
  the delta is negative; the script says so explicitly rather than leaving the
  sign to the reader.

Usage:
    python scripts/analysis/paired_stats.py --a noKG/intermediate_data.json \
                                            --b withKG/intermediate_data.json
    python scripts/analysis/paired_stats.py --a ihr_result_ircot.json \
                                            --b ihr_result_kgpw.json --metric ihr

Pairing is by ``id``. Questions present in only one arm are DROPPED and counted
in the report -- silently intersecting would let two runs over different subsets
look comparable, which is the same class of error as the judge-model split
(AGENTS.md §5).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_scores(path: Path, metric: str) -> Dict[str, float]:
    """Return {question_id: score} for the requested metric.

    Raises rather than guessing when the file does not carry the metric: an
    empty dict here would surface downstream as "0 paired items", which reads
    like a pairing problem rather than a wrong --metric.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    # ihr_result_*.json: {"items": [{"id", "ihr", ...}], "judge_model": ...}
    if isinstance(raw, dict) and "items" in raw:
        if metric != "ihr":
            raise ValueError(
                f"{path} is an IHR result file but --metric is {metric!r}. "
                "Pass --metric ihr, or point --a/--b at intermediate_data.json."
            )
        out = {}
        for it in raw["items"]:
            if it.get("ihr") is not None:
                out[str(it["id"])] = float(it["ihr"])
        return out

    # intermediate_data.json: [{"id", "output": {"metric_score": {"em","f1"}}}]
    if isinstance(raw, list):
        if metric == "ihr":
            raise ValueError(
                f"{path} is a FlashRAG intermediate_data file, which carries no "
                "IHR. Judge it first with run_ihr_judge.py."
            )
        out = {}
        for it in raw:
            ms = (it.get("output") or {}).get("metric_score") or {}
            if metric in ms and ms[metric] is not None:
                out[str(it["id"])] = float(ms[metric])
        if not out:
            raise ValueError(f"{path}: no output.metric_score.{metric} in any item")
        return out

    raise ValueError(f"{path}: unrecognised shape (expected a list or a dict with 'items')")


def _judge_model(path: Path) -> Optional[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("judge_model") if isinstance(raw, dict) else None


def _pair(a: Dict[str, float], b: Dict[str, float]) -> Tuple[List[str], List[float], List[float]]:
    ids = sorted(set(a) & set(b))
    return ids, [a[i] for i in ids], [b[i] for i in ids]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def mcnemar(xa: List[float], xb: List[float]) -> dict:
    """Exact two-sided McNemar on binary outcomes.

    Uses the exact binomial rather than the chi-square approximation: the
    discordant count here is typically small (a +2pp effect over n=300 means
    ~10-30 discordant pairs), which is exactly where the chi-square version is
    unreliable even with continuity correction.
    """
    for v in xa + xb:
        if v not in (0.0, 1.0):
            raise ValueError(f"McNemar needs binary values, got {v}")
    b = sum(1 for p, q in zip(xa, xb) if p == 0.0 and q == 1.0)  # a wrong, b right
    c = sum(1 for p, q in zip(xa, xb) if p == 1.0 and q == 0.0)  # a right, b wrong
    n = b + c
    if n == 0:
        return {"b_only": 0, "a_only": 0, "discordant": 0, "p_value": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return {"b_only": b, "a_only": c, "discordant": n, "p_value": min(1.0, 2.0 * tail)}


def paired_bootstrap(xa: List[float], xb: List[float], n_boot: int, seed: int) -> dict:
    """Bootstrap CI for mean(b) - mean(a), resampling QUESTIONS (not arms).

    Resampling the paired index keeps each question's two scores together, which
    is what makes the CI narrower than an independent one when the arms are
    correlated -- and they are strongly correlated here, since both arms see the
    same retrieval for most questions.
    """
    rng = random.Random(seed)
    n = len(xa)
    diffs = [q - p for p, q in zip(xa, xb)]
    point = sum(diffs) / n
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return {"delta": point, "ci95": (lo, hi), "n_boot": n_boot}


def wilcoxon_signed_rank(xa: List[float], xb: List[float]) -> dict:
    """Two-sided Wilcoxon signed-rank via a normal approximation with tie
    correction. Zero differences are dropped (Wilcoxon convention).

    Reported alongside the bootstrap because the two answer different
    questions: the bootstrap gives the effect SIZE with uncertainty, Wilcoxon
    gives a p-value that does not assume the mean is the right summary of a
    bounded, heavily-tied score like F1.
    """
    diffs = [q - p for p, q in zip(xa, xb) if q != p]
    n = len(diffs)
    if n < 6:
        return {"n_nonzero": n, "p_value": None,
                "note": "n<6 nonzero differences; normal approximation not used"}
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    tie_groups = []
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        tie_groups.append(j - i + 1)
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, diffs) if d > 0)
    mean = n * (n + 1) / 4.0
    tie_corr = sum(t ** 3 - t for t in tie_groups) / 48.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr
    if var <= 0:
        return {"n_nonzero": n, "p_value": None, "note": "zero variance (all ties)"}
    z = (w_plus - mean) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"n_nonzero": n, "w_plus": w_plus, "z": z, "p_value": min(1.0, p)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="Baseline arm (e.g. noKG).")
    p.add_argument("--b", required=True, help="Treatment arm (e.g. +KG).")
    p.add_argument("--metric", default="em", choices=["em", "f1", "ihr"])
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None, help="Write the report as JSON here.")
    args = p.parse_args()

    pa, pb = Path(args.a), Path(args.b)
    sa = _load_scores(pa, args.metric)
    sb = _load_scores(pb, args.metric)
    ids, xa, xb = _pair(sa, sb)
    if not ids:
        raise SystemExit(
            f"No shared ids between the two arms ({len(sa)} vs {len(sb)} items). "
            "They were evaluated on different question sets, so no paired test applies."
        )

    lower_is_better = args.metric == "ihr"
    rep = {
        "metric": args.metric,
        "a": str(pa), "b": str(pb),
        "n_paired": len(ids),
        "dropped_a_only": len(set(sa) - set(sb)),
        "dropped_b_only": len(set(sb) - set(sa)),
        "mean_a": sum(xa) / len(xa),
        "mean_b": sum(xb) / len(xb),
        "lower_is_better": lower_is_better,
    }

    binary = all(v in (0.0, 1.0) for v in xa + xb)
    if binary:
        rep["mcnemar"] = mcnemar(xa, xb)
    rep["bootstrap"] = paired_bootstrap(xa, xb, args.n_boot, args.seed)
    rep["wilcoxon"] = wilcoxon_signed_rank(xa, xb)

    if args.metric == "ihr":
        ja, jb = _judge_model(pa), _judge_model(pb)
        rep["judge_model_a"], rep["judge_model_b"] = ja, jb
        if ja != jb:
            rep["judge_mismatch"] = True

    # ---- report ----
    d = rep["bootstrap"]["delta"]
    lo, hi = rep["bootstrap"]["ci95"]
    print(f"metric={args.metric}  n_paired={rep['n_paired']}"
          f"  (dropped: {rep['dropped_a_only']} a-only, {rep['dropped_b_only']} b-only)")
    print(f"  mean A = {rep['mean_a']:.4f}   mean B = {rep['mean_b']:.4f}")
    print(f"  paired delta (B-A) = {d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"   ({'lower is better' if lower_is_better else 'higher is better'})")
    if binary:
        m = rep["mcnemar"]
        print(f"  McNemar (exact): discordant={m['discordant']}"
              f"  (B-only right {m['b_only']} / A-only right {m['a_only']})"
              f"  p={m['p_value']:.4g}")
    w = rep["wilcoxon"]
    if w.get("p_value") is not None:
        print(f"  Wilcoxon signed-rank: n_nonzero={w['n_nonzero']}  p={w['p_value']:.4g}")
    else:
        print(f"  Wilcoxon signed-rank: {w.get('note')}")

    ps = [t.get("p_value") for t in (rep.get("mcnemar"), rep.get("wilcoxon")) if t]
    ps = [x for x in ps if x is not None]
    sig = any(x < 0.05 for x in ps)
    ci_excludes_zero = (lo > 0) or (hi < 0)
    print(f"  VERDICT: {'SIGNIFICANT (p<0.05)' if sig else 'not significant (p>=0.05)'}"
          f"; bootstrap CI {'excludes' if ci_excludes_zero else 'includes'} zero")
    if not sig:
        print("           Do not write 'improves' / 'significant' in the paper for this pair.")
    if rep.get("judge_mismatch"):
        print(f"  ⚠️  JUDGE MISMATCH: {rep['judge_model_a']} vs {rep['judge_model_b']}"
              " — these IHR numbers are not comparable (AGENTS.md §5).")

    if args.output:
        Path(args.output).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
