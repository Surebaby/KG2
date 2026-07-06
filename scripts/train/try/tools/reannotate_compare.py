#!/usr/bin/env python
"""Offline re-annotation: original vs improved PRM annotator — NO API/network.

Reads an existing silver jsonl, rebuilds each ParsedStep, then relabels every
step with both the original ``PRMAnnotator`` and the new
``ImprovedPRMAnnotator`` and prints the label distributions side by side.

Usage:
    python scripts/train/try/reannotate_compare.py [silver.jsonl]
    (defaults to /tmp/kgpw_try/silver_try_smoke.jsonl)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# Make sibling try-modules importable regardless of CWD/subdir layout.
_TRY_ROOT = Path(__file__).resolve().parent.parent
for _d in (_TRY_ROOT, _TRY_ROOT / "shared", _TRY_ROOT / "phase1_distill",
           _TRY_ROOT / "phase2_prm", _TRY_ROOT / "phase3_sft", _TRY_ROOT / "phase3_ppo"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from kgproweight.data.parsers import parsed_step_from_silver_dict
from kgproweight.reward.prm_annotator import PRMAnnotator
from prm_annotator_try import ImprovedPRMAnnotator

LABEL_NAMES = {1: "+1", 0: " 0", -1: "-1"}


def _dist(counter: Counter) -> str:
    total = sum(counter.values()) or 1
    return "  ".join(
        f"{LABEL_NAMES[l]}={counter.get(l, 0):3d}({100*counter.get(l,0)/total:4.1f}%)"
        for l in (1, 0, -1)
    )


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/kgpw_try/silver_try_smoke.jsonl")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    orig = PRMAnnotator(verbose=False)
    impr = ImprovedPRMAnnotator(verbose=False)

    orig_all, impr_all = Counter(), Counter()
    orig_correct, impr_correct = Counter(), Counter()  # answer_score>=1.0 traj
    flips = Counter()  # (orig_label -> impr_label)
    no_triple_orig, no_triple_impr = Counter(), Counter()

    for r in rows:
        kg = [tuple(t) for t in r.get("kg_subgraph", []) if len(t) == 3]
        steps = [parsed_step_from_silver_dict(s, i) for i, s in enumerate(r.get("steps", []))]
        o_labels = orig.annotate_trajectory(steps, kg)
        i_labels = impr.annotate_trajectory(steps, kg)
        correct = r["metadata"].get("answer_score", 0) >= 1.0
        for st, ol, il in zip(steps, o_labels, i_labels):
            orig_all[ol] += 1
            impr_all[il] += 1
            flips[(ol, il)] += 1
            if correct:
                orig_correct[ol] += 1
                impr_correct[il] += 1
            if not st.cited_triples:
                no_triple_orig[ol] += 1
                no_triple_impr[il] += 1

    print("=" * 78)
    print(f"Re-annotation comparison on {len(rows)} trajectories ({sum(orig_all.values())} steps)")
    print(f"  file: {path}")
    print("=" * 78)
    print(f"ALL steps      original : {_dist(orig_all)}")
    print(f"ALL steps      improved : {_dist(impr_all)}")
    print("-" * 78)
    nc = sum(orig_correct.values())
    print(f"answer-correct original : {_dist(orig_correct)}   ({nc} steps)")
    print(f"answer-correct improved : {_dist(impr_correct)}")
    print("   ^ -1 here = false negatives (correct answer but step flagged hallucination)")
    print("-" * 78)
    nt = sum(no_triple_orig.values())
    print(f"no-cited-triple original : {_dist(no_triple_orig)}   ({nt} steps)")
    print(f"no-cited-triple improved : {_dist(no_triple_impr)}")
    print("   ^ these SHOULD be NEUTRAL(0), not -1")
    print("-" * 78)
    print("Label flips (original -> improved):")
    for (o, i), c in sorted(flips.items(), key=lambda kv: -kv[1]):
        tag = "  <- rescued" if (o == -1 and i in (0, 1)) else ""
        print(f"  {LABEL_NAMES[o]} -> {LABEL_NAMES[i]} : {c:3d}{tag}")


if __name__ == "__main__":
    main()
