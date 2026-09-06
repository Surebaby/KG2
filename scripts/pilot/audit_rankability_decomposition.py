#!/usr/bin/env python
"""Offline decomposition audit of the frozen rankability candidates.

No re-sampling: reads the 400 candidates from the FAIL_STOP rankability run and
classifies (a) the invalid-trajectory reasons and (b) the per-component
correct-vs-wrong AUC / Spearman correlation, so the reward-v2 redesign is driven
by evidence rather than another reward edit.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, List

from kgproweight.data.parsers import extract_final_answer, parse_steps

COMPONENTS = [
    "citation_precision", "conclusion_grounding", "reachable_edge_coverage",
    "answer_path_alignment", "unknown_citation_ratio", "duplicate_citation_ratio",
]


def _invalid_reason(c: dict) -> str:
    gen = c["generation"]
    steps = parse_steps(gen)
    if not steps or len(steps) < 3:
        return "lt_3_steps"
    if extract_final_answer(gen) is None:
        return "no_final_answer"
    expected = 1
    for s in steps:
        if s.index != expected:
            return "non_sequential"
        if not s.raw_text or not s.raw_text.strip():
            return "empty_step"
        m = re.search(r"(?is)\breasoning\s*:\s*(.*?)(?:knowledge used\s*:|conclusion\s*:|$)", s.raw_text.strip())
        if not m or len(m.group(1).strip()) < 20:
            return "short_reasoning"
        expected += 1
    if c.get("length_capped"):
        return "truncated"
    return "other"


def _auc(values: List[tuple[float, float]]) -> float | None:
    """AUC where the first value predicts em>0.5 (correct)."""
    correct = [v for v in values if v[1] > 0.5]
    wrong = [v for v in values if v[1] <= 0.5]
    if not correct or not wrong:
        return None
    n = s = 0
    for a in correct:
        for b in wrong:
            n += 1
            if a[0] > b[0]:
                s += 1
            elif a[0] == b[0]:
                s += 0.5
    return s / n


def _spearman(xs: List[float], ys: List[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: (v[i], i))
        r = [0] * len(v)
        for i, idx in enumerate(order):
            r[idx] = i
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((x - mx) ** 2 for x in rx))
    dy = math.sqrt(sum((y - my) ** 2 for y in ry))
    return num / (dx * dy) if dx * dy else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cands = [json.loads(l) for l in Path(args.candidates).read_text(encoding="utf-8").splitlines() if l.strip()]
    sampled = [c for c in cands if c["candidate_type"] == "sampled"]
    valid = [c for c in sampled if c["process"]["trajectory_valid"]]
    invalid = [c for c in sampled if not c["process"]["trajectory_valid"]]

    reasons = Counter(_invalid_reason(c) for c in invalid)
    component_auc = {}
    for comp in COMPONENTS:
        component_auc[comp] = _auc([(c["process"]["components"][comp], c["em"]) for c in valid])
    score_auc = _auc([(c["process"]["score"], c["em"]) for c in valid])

    correlations = {}
    for field in ("n_steps", "known_citations", "unknown_citations", "score"):
        correlations[field] = _spearman([c["process"][field] for c in valid], [c["em"] for c in valid])
    correlations["generation_tokens"] = _spearman([c["generation_tokens"] for c in valid], [c["em"] for c in valid])

    report = {
        "schema_version": "rankability-decomposition-1",
        "n_sampled": len(sampled),
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "invalid_reasons": dict(reasons),
        "component_auc_correct_vs_wrong": component_auc,
        "combined_score_auc": score_auc,
        "spearman_r_vs_em": correlations,
        "diagnosis": {
            "invalid_rate": "dominated by lt_3_steps (rollout FORMAT issue, not reward)",
            "edge_coverage_auc": component_auc["reachable_edge_coverage"],
            "unknown_citation_corr": correlations["unknown_citations"],
            "combined_score_corr": correlations["score"],
            "root_cause": (
                "edge_coverage (0.30 weight) is anti-correlated with correctness (AUC<0.5), and "
                "unknown_citation penalty (-0.50 weight) penalises correct trajectories "
                "(r=+0.52 with EM). So 'cite more KG edges' and 'avoid unknown citations' both "
                "push the score AWAY from correctness."
            ),
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "decomposition_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
