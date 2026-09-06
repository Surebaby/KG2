#!/usr/bin/env python
"""Paired IHR over two replay-validation JSONLs (same question set).

Each JSONL row has ``qid``, ``question``, ``gold``, ``generation`` (the full
reasoning trace).  We parse the trace into steps and judge every step with the
LLM-as-judge IHR, so the two arms are compared on identical questions and the
hallucination flags are paired per step.

    python scripts/pilot/paired_ihr_replay.py \
        --arm_a <sft.jsonl> --arm_b <combined.jsonl> \
        --judge_model deepseek-v4-pro --out <out.json>

Uses OPENAI_API_KEY / OPENAI_BASE_URL from the environment (source .env first).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from kgproweight.data.parsers import parse_steps
from kgproweight.reward.ihr_judge import IHRJudge


def load(path: str) -> dict:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["qid"]: r for r in rows}


def paired_mcnemar(flags_a, flags_b):
    """McNemar on paired per-step hallucination flags (b/c/d as bool counts)."""
    b = c = 0
    for fa, fb in zip(flags_a, flags_b):
        if fa and not fb:
            b += 1
        elif not fa and fb:
            c += 1
    n_disc = b + c
    if n_disc == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p": 1.0}
    # exact binomial two-sided
    from math import comb
    p = 0.0
    for k in range(0, min(b, c) + 1):
        p += comb(n_disc, k) * (0.5 ** n_disc)
    for k in range(max(b, c), n_disc + 1):
        p += comb(n_disc, k) * (0.5 ** n_disc)
    return {"b": b, "c": c, "n_discordant": n_disc, "p": min(1.0, p)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm_a", required=True)
    ap.add_argument("--arm_b", required=True)
    ap.add_argument("--judge_model", default="deepseek-v4-pro")
    ap.add_argument("--n", type=int, default=None, help="cap the paired set (seeded)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a = load(args.arm_a)
    b = load(args.arm_b)
    common = sorted(set(a) & set(b))
    if args.n and args.n < len(common):
        common = sorted(random.Random(args.seed).sample(common, args.n))
    print(f"paired qids: {len(common)}", flush=True)

    judge = IHRJudge(model=args.judge_model)

    def run(rows, arm):
        out = []
        done = 0
        for q in common:
            r = rows[q]
            question = r["question"]
            gold = (r.get("gold") or [""])[0]
            gen = r.get("generation") or ""
            steps = parse_steps(gen)
            if not steps:
                continue
            per_step = judge.judge_trajectory(question, gold, [s.raw_text for s in steps])
            out.append({
                "qid": q,
                "ihr": IHRJudge.aggregate_ihr(per_step),
                "n_steps": len(per_step),
                "n_halluc": sum(1 for j in per_step if j.is_hallucination),
                "flags": [j.is_hallucination for j in per_step],
            })
            done += 1
            if done % 10 == 0:
                print(f"  [{arm}] {done}/{len(common)} judged", flush=True)
        return out

    a_out = run(a, "sft")
    b_out = run(b, "combined")
    a_by_q = {x["qid"]: x for x in a_out}
    b_by_q = {x["qid"]: x for x in b_out}
    paired_q = sorted(set(a_by_q) & set(b_by_q))

    a_ihr = [a_by_q[q]["ihr"] for q in paired_q]
    b_ihr = [b_by_q[q]["ihr"] for q in paired_q]
    a_flags = [f for q in paired_q for f in a_by_q[q]["flags"]]
    b_flags = [f for q in paired_q for f in b_by_q[q]["flags"]]

    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    diffs = [ba - aa for aa, ba in zip(a_ihr, b_ihr)]

    # paired t-test on per-item IHR rates
    try:
        from statistics import mean as _m, stdev as _sd
        from math import sqrt
        n = len(diffs)
        d_mean = _m(diffs)
        d_sd = _sd(diffs) if n > 1 else 0.0
        t = d_mean / (d_sd / sqrt(n)) if d_sd > 0 else float("nan")
        from math import erf
        # two-sided normal approx p
        from statistics import NormalDist
        p_paired_t = 2 * (1 - NormalDist().cdf(abs(t))) if n > 1 and d_sd > 0 else 1.0
    except Exception:
        t = d_mean = d_sd = p_paired_t = float("nan")

    report = {
        "arm_a": args.arm_a,
        "arm_b": args.arm_b,
        "judge_model": args.judge_model,
        "n_paired_items": len(paired_q),
        "mean_ihr_a": mean(a_ihr),
        "mean_ihr_b": mean(b_ihr),
        "mean_paired_diff_b_minus_a": d_mean,
        "paired_t": t,
        "paired_t_p": p_paired_t,
        "per_step_mcnemar": paired_mcnemar(a_flags, b_flags),
        "n_steps_a": sum(a_by_q[q]["n_steps"] for q in paired_q),
        "n_steps_b": sum(b_by_q[q]["n_steps"] for q in paired_q),
        "n_halluc_a": sum(a_by_q[q]["n_halluc"] for q in paired_q),
        "n_halluc_b": sum(b_by_q[q]["n_halluc"] for q in paired_q),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
