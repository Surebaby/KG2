#!/usr/bin/env python
"""Aggregate the rerank10 seed sweep: per-seed EM/F1 plus a pooled paired bootstrap.

Two levels of uncertainty are reported because they answer different questions:

  per-seed paired CI  — "on this question set, does the checkpoint help?"
  pooled paired CI    — same, resampling (seed, question) pairs together, which
                        is the number to quote; it folds in seed-to-seed spread
                        instead of pretending one seed's question set is the
                        population.

Seed spread is also printed raw, because a +13pp effect that flips sign across
seeds is a different claim from one that holds in all three.
"""
from __future__ import annotations

import glob
import json
import random
import re
import sys

sys.path.insert(0, "flashrag_src")
from flashrag.evaluator.utils import normalize_answer  # noqa: E402

SEEDS = [13, 42, 2024]
ARMS = {
    "split": {13: "outputs/split_sft_rerank10",
              42: "outputs/split_rerank10_s42",
              2024: "outputs/split_rerank10_s2024"},
    "elite": {13: "outputs/elite_rerank10_s13",
              42: "outputs/elite_rerank10_s42",
              2024: "outputs/elite_rerank10_s2024"},
}
N_BOOT = 20000


def _strip_role_artifact(pred: str) -> str:
    s = re.sub(r"assistant\s*$", "", pred.strip())
    return re.sub(r"^(assistant|system|user)\s*", "", s).strip()


def _f1(pred: str, gold: str) -> float:
    p, g = pred.split(), gold.split()
    if not p or not g:
        return float(p == g)
    gg, common = list(g), 0
    for t in p:
        if t in gg:
            gg.remove(t)
            common += 1
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def load(root: str):
    f = glob.glob(f"{root}/**/intermediate_data.json", recursive=True)
    if not f:
        return None
    rows = json.load(open(f[0]))
    ms = glob.glob(f"{root}/**/metric_score.json", recursive=True)
    ait = json.load(open(ms[0])).get("avg_input_tokens") if ms else None
    per = {}
    for r in rows:
        o = r.get("output")
        pred = (o.get("pred") if isinstance(o, dict) else o) or ""
        golds = r.get("golden_answers") or []
        np_ = normalize_answer(_strip_role_artifact(pred))
        qid = str(r.get("id"))
        em = 1 if any(normalize_answer(g) == np_ for g in golds) else 0
        f1 = max((_f1(np_, normalize_answer(g)) for g in golds), default=0.0)
        per[qid] = (em, f1)
    return per, ait


def paired_ci(diffs, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    bs = []
    for _ in range(N_BOOT):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        bs.append(s / n)
    bs.sort()
    return bs[int(N_BOOT * 0.025)], bs[int(N_BOOT * 0.975)]


def main() -> int:
    data = {}
    for arm, roots in ARMS.items():
        for sd, root in roots.items():
            r = load(root)
            if r is None:
                print(f"{arm:6s} seed {sd:<5} NOT FINISHED  ({root})")
                continue
            data[(arm, sd)] = r[0]
            per = r[0]
            n = len(per)
            em = sum(v[0] for v in per.values()) / n
            f1 = sum(v[1] for v in per.values()) / n
            print(f"{arm:6s} seed {sd:<5} n={n:3d}  EM={em:.3f}  F1={f1:.3f}  "
                  f"avg_input_tokens={r[1]}")

    print()
    pooled_em, pooled_f1, per_seed = [], [], []
    for sd in SEEDS:
        a, b = data.get(("split", sd)), data.get(("elite", sd))
        if not a or not b:
            continue
        common = sorted(set(a) & set(b))
        dem = [a[q][0] - b[q][0] for q in common]
        df1 = [a[q][1] - b[q][1] for q in common]
        pooled_em += dem
        pooled_f1 += df1
        lo, hi = paired_ci(dem, seed=sd)
        mean = sum(dem) / len(dem)
        per_seed.append(mean)
        flag = "" if lo <= 0 <= hi else "  sig"
        print(f"seed {sd:<5} dEM={mean:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]{flag}"
              f"   (n={len(common)})")

    if not pooled_em:
        print("\nnothing to pool yet")
        return 0

    print(f"\nseed spread of dEM: {[f'{x:+.3f}' for x in per_seed]}")
    lo, hi = paired_ci(pooled_em, seed=99)
    mean = sum(pooled_em) / len(pooled_em)
    lo1, hi1 = paired_ci(pooled_f1, seed=98)
    mean1 = sum(pooled_f1) / len(pooled_f1)
    sig = "SIGNIFICANT" if not (lo <= 0 <= hi) else "not significant"
    print(f"\nPOOLED  split - elite   (n={len(pooled_em)} seed-question pairs)")
    print(f"  dEM = {mean:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]   {sig}")
    print(f"  dF1 = {mean1:+.3f}  95% CI [{lo1:+.3f}, {hi1:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
