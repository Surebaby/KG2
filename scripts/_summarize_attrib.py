#!/usr/bin/env python
"""Summarise the three attribution runs with paired bootstrap CIs.

A bare point EM is not reportable at n=100: the noise floor measured on
ablation_v6 is ~7pp (single-arm 95% CI [0.280, 0.470], paired diff
[-0.070, +0.070]). So every comparison here is a *paired* bootstrap over the
per-question hit vectors, which cancels question difficulty and is far tighter
than comparing two independent CIs.

Runs compared:
  split_sft_pipeline_eval   new ckpt, default topk 50   (prompts truncated)
  elite_baseline_curenv     old ckpt, same env+config   (checkpoint control)
  split_sft_rerank10        new ckpt, --rerank 10       (no truncation)

  elite -> split   : effect of the checkpoint, env held fixed
  topk50 -> rerank : cost of the 6144 right-truncation, checkpoint held fixed
"""
from __future__ import annotations

import glob
import json
import random
import re
import sys

sys.path.insert(0, "flashrag_src")
from flashrag.evaluator.utils import normalize_answer  # noqa: E402

RUNS = [
    ("split_sft(topk50)", "outputs/split_sft_pipeline_eval"),
    ("elite(topk50)", "outputs/elite_baseline_curenv"),
    ("split_sft(rerank10)", "outputs/split_sft_rerank10"),
]
N_BOOT = 20000


def load(root: str):
    """Return (hit_vector, f1_vector, qids, avg_input_tokens) or None."""
    f = glob.glob(f"{root}/**/intermediate_data.json", recursive=True)
    if not f:
        return None
    rows = json.load(open(f[0]))
    ms = glob.glob(f"{root}/**/metric_score.json", recursive=True)
    ait = json.load(open(ms[0])).get("avg_input_tokens") if ms else None

    hits, f1s, qids = [], [], []
    raw_hits = []
    for r in rows:
        o = r.get("output")
        pred = (o.get("pred") if isinstance(o, dict) else o) or ""
        golds = r.get("golden_answers") or []
        np_raw = normalize_answer(pred)
        np_ = normalize_answer(_strip_role_artifact(pred))
        raw_hits.append(1 if any(normalize_answer(g) == np_raw for g in golds) else 0)
        hits.append(1 if any(normalize_answer(g) == np_ for g in golds) else 0)
        f1s.append(max((_f1(np_, normalize_answer(g)) for g in golds), default=0.0))
        qids.append(str(r.get("id", len(qids))))
    n = len(hits)
    lost = (sum(hits) - sum(raw_hits)) / max(1, n)
    if lost > 0.01:
        print(f"    [{root}] role-artifact stripping recovers {lost:+.3f} EM "
              f"(as-scored {sum(raw_hits)/n:.3f} -> {sum(hits)/n:.3f})")
    return hits, f1s, qids, ait


def _strip_role_artifact(pred: str) -> str:
    """Remove a literal trailing/leading ``assistant`` left by chat-template truncation.

    When the prompt exceeds ``generator_max_input_len``, right-truncation cuts the
    special tokens of ``<|start_header_id|>assistant<|end_header_id|>`` and leaves the
    bare word ``assistant`` at the end of the prompt. The model continues from it, so
    every prediction comes back as e.g. ``Animorphsassistant``. ``normalize_answer``
    does not remove it, and ``ExactMatch`` is strict full-string equality, so a fully
    correct answer scores 0. Measured cost on the topk-50 run: 38pp of EM.
    """
    s = re.sub(r"assistant\s*$", "", pred.strip())
    return re.sub(r"^(assistant|system|user)\s*", "", s).strip()


def _f1(pred: str, gold: str) -> float:
    p, g = pred.split(), gold.split()
    if not p or not g:
        return float(p == g)
    common = 0
    gg = list(g)
    for t in p:
        if t in gg:
            gg.remove(t)
            common += 1
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def paired_ci(a, b, seed=0):
    """95% CI of mean(a) - mean(b) by paired bootstrap over question index."""
    rng = random.Random(seed)
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    bs = []
    for _ in range(N_BOOT):
        s = 0.0
        for _ in range(n):
            s += d[rng.randrange(n)]
        bs.append(s / n)
    bs.sort()
    return bs[int(N_BOOT * 0.025)], bs[int(N_BOOT * 0.975)]


def main() -> int:
    loaded = {}
    for label, root in RUNS:
        r = load(root)
        if r is None:
            print(f"{label:22s} NOT FINISHED ({root})")
            continue
        hits, f1s, qids, ait = r
        loaded[label] = (hits, f1s, qids)
        n = len(hits)
        print(f"{label:22s} n={n:3d}  EM={sum(hits)/n:.3f}  "
              f"F1={sum(f1s)/n:.3f}  avg_input_tokens={ait}")

    def cmp(x, y, what):
        if x not in loaded or y not in loaded:
            return
        hx, fx, qx = loaded[x]
        hy, fy, qy = loaded[y]
        if qx != qy:
            # Align on qid so a reordered run does not silently pair the wrong
            # questions against each other.
            common = [q for q in qx if q in set(qy)]
            ix = {q: i for i, q in enumerate(qx)}
            iy = {q: i for i, q in enumerate(qy)}
            hx = [hx[ix[q]] for q in common]
            hy = [hy[iy[q]] for q in common]
            fx = [fx[ix[q]] for q in common]
            fy = [fy[iy[q]] for q in common]
            print(f"  (aligned on {len(common)} shared qids)")
        dem = sum(hx) / len(hx) - sum(hy) / len(hy)
        lo, hi = paired_ci(hx, hy)
        df1 = sum(fx) / len(fx) - sum(fy) / len(fy)
        lo1, hi1 = paired_ci(fx, fy)
        sig = "" if lo <= 0 <= hi else "  SIGNIFICANT"
        print(f"\n{what}")
        print(f"  dEM  = {dem:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]{sig}")
        print(f"  dF1  = {df1:+.3f}  95% CI [{lo1:+.3f}, {hi1:+.3f}]")
        nx = sum(1 for p, q in zip(hx, hy) if p and not q)
        ny = sum(1 for p, q in zip(hx, hy) if q and not p)
        print(f"  only-{x.split('(')[0]}={nx}  only-{y.split('(')[0]}={ny}")

    print()
    cmp("split_sft(topk50)", "elite(topk50)",
        "checkpoint effect (env + config held fixed):")
    cmp("split_sft(rerank10)", "split_sft(topk50)",
        "cost of 6144 right-truncation (checkpoint held fixed):")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
