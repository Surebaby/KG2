"""Score a teacher-swap pilot against the deepseek-chat baseline, paired on qid.

Both arms are measured by the SAME code so the numbers are comparable; the
baseline's published 14.53% / 51.1% are recomputed here rather than quoted, in
case the definitions drifted.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def norm(t: Any) -> tuple:
    if isinstance(t, dict):
        parts = [t.get("head", ""), t.get("relation", ""), t.get("tail", "")]
    else:
        parts = list(t)[:3]
    return tuple(str(p).strip().lower() for p in parts)


def stats(recs: List[Dict[str, Any]], label: str, max_keep: int) -> Dict[str, Any]:
    from kgproweight.kg.kg_filter import filter_and_rank_triples

    n_steps = n_citing = 0
    cites = halluc = 0
    survive = 0            # citations still visible under the student's top-K budget
    all_intact = 0         # citing steps whose every citation survives
    n_traj_nocite = 0
    for r in recs:
        kg = [tuple(t) if not isinstance(t, dict) else t for t in (r.get("kg_subgraph") or [])]
        kgset = {norm(t) for t in kg}
        kept = {norm(t) for t in filter_and_rank_triples(
            kg, question=r["question"], max_keep=max_keep, min_keep=5)} if kg else set()
        steps = r.get("steps") or []
        n_steps += len(steps)
        traj_cited = False
        for s in steps:
            ct = s.get("cited_triples") or []
            if ct:
                n_citing += 1
                traj_cited = True
            ok = 0
            for t in ct:
                cites += 1
                k = norm(t)
                if k not in kgset:
                    halluc += 1
                else:
                    if k in kept:
                        survive += 1
                        ok += 1
            if ct and ok == len(ct):
                all_intact += 1
        if not traj_cited:
            n_traj_nocite += 1
    pct = lambda a, b: (100.0 * a / b) if b else float("nan")
    return {
        "arm": label, "n_traj": len(recs), "n_steps": n_steps,
        "steps_per_traj": n_steps / max(len(recs), 1),
        "citation_rate_%": pct(n_citing, n_steps),
        "traj_no_citation_%": pct(n_traj_nocite, len(recs)),
        "n_citations": cites,
        "halluc_rate_%": pct(halluc, cites),
        f"survive_top{max_keep}_%": pct(survive, cites),
        "steps_all_intact_%": pct(all_intact, n_citing),
    }


def accept_rate(recs: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """Replay StratifiedSilverFilter over an arm, in file order.

    This is the metric the rebuild decision turns on: the filter gates on
    ``answer_score >= min_answer_score`` BEFORE bucketing, so a teacher that
    cites more but answers worse loses here even while its citation_rate looks
    better. The sparse/medium quotas are stateful, hence one fresh filter per
    arm and a fixed order.
    """
    from kgproweight.training.phase1_distill import StratifiedSilverFilter

    class _S:  # decide() only reads .cited_triples
        __slots__ = ("cited_triples",)
        def __init__(self, ct): self.cited_triples = ct

    f = StratifiedSilverFilter()
    n_acc = 0
    reasons: Dict[str, int] = {}
    for r in recs:
        steps = [_S(s.get("cited_triples") or []) for s in (r.get("steps") or [])]
        md = r.get("metadata") or {}
        d = f.decide(steps=steps, coverage=md.get("coverage", 0.0),
                     answer_score=md.get("answer_score", 0.0))
        n_acc += bool(d.accepted)
        if not d.accepted:
            key = d.reason.split("=")[0]
            reasons[key] = reasons.get(key, 0) + 1
    return {"arm": label, "accepted": n_acc, "n": len(recs),
            "accepted_%": 100.0 * n_acc / max(len(recs), 1),
            "buckets": f.stats(), "reject_reasons": reasons}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--baseline", default="data/silver_data/silver_v1_reannotated.jsonl")
    ap.add_argument("--max_keep", type=int, default=12,
                    help="Student's inference/PPO KG budget (ppo_max_kg_triples).")
    args = ap.parse_args()

    pilot = [json.loads(l) for l in open(args.pilot, encoding="utf-8") if l.strip()]
    pilot = [r for r in pilot if r.get("_status", "ok") == "ok"]
    # A pilot that dropped items is not a random subsample: hard/slow questions
    # fail first, so every metric below is optimistic. Refuse to print a
    # comparison table that would be read as a teacher verdict.
    fp = pathlib.Path(args.pilot).with_suffix(".failures.jsonl")
    n_fail = sum(1 for _ in open(fp, encoding="utf-8")) if fp.exists() else 0
    if n_fail:
        tot = len(pilot) + n_fail
        print(f"WARNING: {n_fail}/{tot} items failed ({100.0*n_fail/tot:.1f}%). The written "
              f"subset is biased toward items the model answered quickly; treat the table "
              f"below as diagnostic only, NOT as a teacher comparison.\n")
    qids = {r["qid"] for r in pilot}
    base = [json.loads(l) for l in open(args.baseline, encoding="utf-8") if l.strip()]
    base_paired = [r for r in base if r.get("qid") in qids]

    rows = [stats(base_paired, f"deepseek-chat (paired, n={len(base_paired)})", args.max_keep),
            stats(pilot, f"{pilot[0].get('teacher_model')} (pilot)", args.max_keep)]
    keys = [k for k in rows[0] if k != "arm"]
    w = max(len(k) for k in keys) + 2
    print(f"{'metric':<{w}}" + "".join(f"{r['arm']:>42}" for r in rows))
    for k in keys:
        cells = ""
        for r in rows:
            v = r[k]
            cells += f"{v:>42.2f}" if isinstance(v, float) else f"{v:>42}"
        print(f"{k:<{w}}" + cells)
    print()
    for recs, lab in ((base_paired, "deepseek-chat"), (pilot, pilot[0].get("teacher_model"))):
        a = accept_rate(recs, lab)
        print(f"{lab:>22}: accepted {a['accepted']}/{a['n']} ({a['accepted_%']:.1f}%)  "
              f"buckets={a['buckets']}  rejects={a['reject_reasons']}")

    nr = sum(1 for r in pilot if r.get("retried"))
    print(f"\nformat-retry triggered: {nr}/{len(pilot)} ({100.0*nr/max(len(pilot),1):.1f}%)")
    ans = [r["metadata"]["answer_score"] for r in pilot if r.get("metadata")]
    if ans:
        print(f"pilot answer_match mean: {sum(ans)/len(ans):.3f}")
    ba = [(r.get("metadata") or {}).get("answer_score", 0.0) for r in base_paired]
    if ba:
        print(f"baseline answer_match mean (same qids): {sum(ba)/len(ba):.3f}")


if __name__ == "__main__":
    main()
