#!/usr/bin/env python
"""Offline re-score + re-rank the frozen 400 candidates with reward v2.

Uses the PRODUCTION dynamic step gate (min_steps from planned hops), the reward
v2 required-hop formula, and a tie-aware Spearman for component diagnostics.
No re-sampling, no gold in the scorer (gold used only for EM evaluation).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from kgproweight.eval.metrics import compute_em, compute_f1
from kgproweight.reward.proofkg_process_v2 import build_execution_trace, score_proofkg_v2

COMPONENTS = [
    "P_precise_citation", "H_hop_coverage", "O_dependency_order",
    "G_conclusion_grounding", "A_answer_consistency",
]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _tie_aware_spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: (v[i], i))
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((x - mx) ** 2 for x in rx))
    dy = math.sqrt(sum((y - my) ** 2 for y in ry))
    return num / (dx * dy) if dx * dy else 0.0


def _bootstrap_delta(left, right, seed):
    import numpy as np
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(10_000)
    for i in range(len(draws)):
        draws[i] = delta[rng.integers(0, len(delta), len(delta))].mean()
    return {
        "diff_mean": float(delta.mean()),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "p_value": min(1.0, float(2 * min((draws <= 0).mean(), (draws >= 0).mean()))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--question_kg_records", required=True)
    parser.add_argument("--runtime_details", required=True)
    parser.add_argument("--proof_input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cands = _read_jsonl(Path(args.candidates))
    kg_by_qid = {str(r["qid"]): r for r in _read_jsonl(Path(args.question_kg_records))}
    detail_by_qid = {str(r["qid"]): r for r in _read_jsonl(Path(args.runtime_details))}
    gold_by_qid = {str(r["qid"]): r.get("gold_answers") or [] for r in _read_jsonl(Path(args.proof_input))}

    # build execution traces once per qid
    trace_by_qid: Dict[str, List[Dict[str, Any]]] = {}
    planned_by_qid: Dict[str, int] = {}
    for qid, kg in kg_by_qid.items():
        plan = kg.get("query_plan") or {}
        detail = detail_by_qid.get(qid, {})
        planned_by_qid[qid] = len(plan.get("hops") or [])
        trace_by_qid[qid] = build_execution_trace(plan, detail.get("execution") or {})

    scored = []
    for c in cands:
        qid = str(c["qid"])
        kg = kg_by_qid[qid]
        proc = score_proofkg_v2(
            question=str(c["question"]),
            generation=str(c["generation"]),
            kg_triples=kg.get("kg_subgraph") or [],
            execution_trace=trace_by_qid[qid],
            planned_hops=planned_by_qid[qid],
        )
        golds = [str(g) for g in gold_by_qid.get(qid, []) if str(g).strip()]
        c["process"] = proc
        c["em"] = compute_em(proc["prediction"], golds) if proc["prediction"] and golds else 0.0
        c["f1"] = compute_f1(proc["prediction"], golds) if proc["prediction"] and golds else 0.0
        scored.append(c)

    sampled = [c for c in scored if c["candidate_type"] == "sampled"]
    greedy = {str(c["qid"]): c for c in scored if c["candidate_type"] == "greedy"}
    valid = [c for c in sampled if c["process"]["trajectory_valid"]]
    qids = sorted(set(greedy).intersection({str(c["qid"]) for c in sampled}))

    greedy_em = [greedy[q]["em"] for q in qids]
    oracle_em = [max(c["em"] for c in sampled if str(c["qid"]) == q) for q in qids]
    selected = [
        max((c for c in sampled if str(c["qid"]) == q),
            key=lambda c: (c["process"]["score"], -c["candidate_index"]))
        for q in qids
    ]
    selected_em = [c["em"] for c in selected]

    # pairwise accuracy (tie-aware)
    by_qid = defaultdict(list)
    for c in sampled:
        by_qid[str(c["qid"])].append(c)
    wins = ties = comparisons = 0
    for rows in by_qid.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if a["em"] == b["em"]:
                    continue
                corr, wrong = (a, b) if a["em"] > b["em"] else (b, a)
                comparisons += 1
                if corr["process"]["score"] > wrong["process"]["score"]:
                    wins += 1
                elif corr["process"]["score"] == wrong["process"]["score"]:
                    ties += 1
    pairwise = (wins + 0.5 * ties) / comparisons if comparisons else None

    valid_rate = len(valid) / len(sampled)
    report = {
        "schema_version": "rankability-rescore-v2-1",
        "n_qids": len(qids),
        "greedy_em": sum(greedy_em) / len(greedy_em),
        "oracle_at_4_em": sum(oracle_em) / len(oracle_em),
        "process_top1_em": sum(selected_em) / len(selected_em),
        "sample_valid_rate": valid_rate,
        "process_pairwise_accuracy": pairwise,
        "oracle_minus_greedy_ci": _bootstrap_delta(oracle_em, greedy_em, 20260901),
        "process_top1_minus_greedy_ci": _bootstrap_delta(selected_em, greedy_em, 20260902),
        "gates": {
            "exploration_headroom": (sum(oracle_em) / len(oracle_em)) - (sum(greedy_em) / len(greedy_em)) >= 0.05,
            "process_selected_gain": (sum(selected_em) / len(selected_em)) - (sum(greedy_em) / len(greedy_em)) >= 0.02,
            "process_pairwise_accuracy": pairwise is not None and pairwise >= 0.60,
            "sample_valid_rate": valid_rate >= 0.90,
        },
        "component_spearman_r_vs_em": {
            comp: _tie_aware_spearman([c["process"]["components"][comp] for c in valid], [c["em"] for c in valid])
            for comp in COMPONENTS
        },
        "combined_score_spearman_r": _tie_aware_spearman([c["process"]["score"] for c in valid], [c["em"] for c in valid]),
    }
    report["all_pass"] = all(report["gates"].values())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "rescore_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
