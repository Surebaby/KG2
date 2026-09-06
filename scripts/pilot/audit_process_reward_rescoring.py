#!/usr/bin/env python
"""Exploratory offline re-ranking of saved PPO reward-rankability rollouts.

No model is loaded and the production reward implementation is not changed.
The audit compares a small, theory-motivated set of gold-free process scores.
Gold EM is read only after scores are fixed, to measure ranking alignment.

This first artifact is explicitly exploratory: aggregate outcomes from all 100
qids were inspected during diagnosis before the formulas were frozen.  It is
therefore not an independent confirmation set and cannot by itself authorise a
core Reward change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from kgproweight.data.parsers import parse_steps
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir


TEXT_SCALE = 0.3
STEP_SCALE = 1.5


def _norm(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def _phrase_in(phrase: str, text: str) -> bool:
    needle, haystack = _norm(phrase), _norm(text)
    return bool(needle and re.search(rf"(?:^| ){re.escape(needle)}(?: |$)", haystack))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_qkg(path: Path) -> Dict[str, List[Tuple[str, str, str]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    is_v2 = bool(raw and "builder_version" in raw[0])
    out: Dict[str, List[Tuple[str, str, str]]] = {}
    for row in raw:
        question = str(row.get("question", row.get("q", "")))
        triples = row.get("triples", []) if is_v2 else row.get("t", [])
        if is_v2:
            out[question] = [(str(t["h"]), str(t["r"]), str(t["t"])) for t in triples]
        else:
            out[question] = [tuple(map(str, t)) for t in triples if len(t) == 3]
    return out


def _candidate_split(qid: str) -> str:
    # Frozen only for descriptive stability. It is not a clean held-out split:
    # simple aggregate formula outcomes were inspected before this script.
    value = int(hashlib.sha256(("20260828:" + qid).encode()).hexdigest(), 16)
    return "diagnostic_a" if value % 2 == 0 else "diagnostic_b"


def _answer_evidence_support(
    answer: str,
    passages: Sequence[Mapping[str, Any]],
    kg: Sequence[Tuple[str, str, str]],
) -> float:
    key = _norm(answer)
    if not key or key in {"yes", "no"}:
        return 0.0
    if any(_phrase_in(answer, str(p.get("contents") or p.get("text") or "")) for p in passages):
        return 1.0
    if any(_phrase_in(answer, h) or _phrase_in(answer, t) for h, _, t in kg):
        return 1.0
    return 0.0


def _citation_gates(step, question: str) -> Tuple[float, float]:
    """Return conclusion-connected and question→conclusion bridge fractions."""

    cited = list(step.cited_triples)
    conclusion = str(step.intermediate_conclusion or "")
    if not cited or not conclusion:
        return 0.0, 0.0
    conclusion_connected = 0
    bridge_connected = 0
    for h, _, t in cited:
        h_c, t_c = _phrase_in(h, conclusion), _phrase_in(t, conclusion)
        h_q, t_q = _phrase_in(h, question), _phrase_in(t, question)
        conclusion_connected += h_c or t_c
        bridge_connected += (h_q and t_c) or (t_q and h_c) or (h_c and t_c)
    return conclusion_connected / len(cited), bridge_connected / len(cited)


def score_candidate(
    row: Mapping[str, Any],
    *,
    passages: Sequence[Mapping[str, Any]],
    kg: Sequence[Tuple[str, str, str]],
) -> Dict[str, float]:
    records = [r for r in row["per_step_records"] if int(r.get("step_index", 0)) > 0]
    steps = parse_steps(str(row["response"]), known_kg=kg)
    n = max(1, len(records))

    current = float(row["process_reward"])
    raw_text_mean = sum(float(r["r_text"]) for r in records) / n
    centered_text_mean = sum(
        (1.0 - float(r["alpha"])) * float(r["r_text_used"]) * TEXT_SCALE * STEP_SCALE
        for r in records
    ) / n
    conclusion_score = 0.0
    bridge_score = 0.0
    for index, record in enumerate(records):
        step = steps[index] if index < len(steps) else None
        conclusion_gate, bridge_gate = _citation_gates(step, str(row["question"])) if step else (0.0, 0.0)
        alpha = float(record["alpha"])
        r_kg = float(record["r_kg"])
        text = float(record["r_text_used"])
        conclusion_score += (
            alpha * r_kg * conclusion_gate + (1.0 - alpha) * text * TEXT_SCALE
        ) * STEP_SCALE
        bridge_score += (
            alpha * r_kg * bridge_gate + (1.0 - alpha) * text * TEXT_SCALE
        ) * STEP_SCALE
    conclusion_score /= n
    bridge_score /= n
    evidence = _answer_evidence_support(str(row["predicted_answer"]), passages, kg)

    # Formula names encode the exact intervention.  All are gold-free.
    return {
        "current_process_sum": current,
        "current_process_step_mean": current / n,
        "raw_text_step_mean": raw_text_mean,
        "centered_text_step_mean": centered_text_mean,
        "conclusion_gated_kg_step_mean": conclusion_score,
        "question_bridge_gated_kg_step_mean": bridge_score,
        # Fixed +1 support bonus on the native [-1, 1] raw-text scale.
        "answer_evidence_plus_raw_text": evidence + raw_text_mean,
    }


def _pairwise(rows_by_qid: Mapping[str, Sequence[Mapping[str, Any]]], score: str, qids: Sequence[str]) -> Dict[str, Any]:
    wins = ties = comparisons = 0
    per_qid: Dict[str, Tuple[float, int]] = {}
    for qid in qids:
        local_wins = local_ties = local_n = 0
        rows = rows_by_qid[qid]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if float(rows[i]["em"]) == float(rows[j]["em"]):
                    continue
                correct, wrong = (rows[i], rows[j]) if rows[i]["em"] else (rows[j], rows[i])
                local_n += 1
                local_wins += correct["offline_scores"][score] > wrong["offline_scores"][score]
                local_ties += correct["offline_scores"][score] == wrong["offline_scores"][score]
        wins += local_wins
        ties += local_ties
        comparisons += local_n
        if local_n:
            per_qid[qid] = ((local_wins + 0.5 * local_ties) / local_n, local_n)
    return {
        "accuracy": (wins + 0.5 * ties) / comparisons if comparisons else None,
        "wins": wins, "ties": ties, "comparisons": comparisons,
        "qids_with_mixed_outcomes": len(per_qid),
    }


def _bootstrap_top1_delta(
    selected: Sequence[float], greedy: Sequence[float], *, seed: int, n_boot: int = 10000,
) -> Dict[str, Any]:
    a, b = np.asarray(selected, dtype=float), np.asarray(greedy, dtype=float)
    diff = a - b
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(diff), len(diff))
        boot[i] = diff[idx].mean()
    return {
        "diff_mean": float(diff.mean()),
        "lower": float(np.quantile(boot, 0.025)),
        "upper": float(np.quantile(boot, 0.975)),
        "p_value": min(1.0, float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))),
        "n": len(diff),
    }


def summarise(
    sampled: Mapping[str, Sequence[Mapping[str, Any]]],
    greedy: Mapping[str, Mapping[str, Any]],
    formulas: Sequence[str],
) -> Dict[str, Any]:
    qids = sorted(sampled)
    result: Dict[str, Any] = {}
    for formula_index, formula in enumerate(formulas):
        selected_rows = [
            max(sampled[qid], key=lambda row: (row["offline_scores"][formula], -int(row["candidate_index"])))
            for qid in qids
        ]
        selected_em = [float(row["em"]) for row in selected_rows]
        greedy_em = [float(greedy[qid]["em"]) for qid in qids]
        pair = _pairwise(sampled, formula, qids)
        result[formula] = {
            "top1_em": sum(selected_em) / len(selected_em),
            "top1_f1": sum(float(row["f1"]) for row in selected_rows) / len(selected_rows),
            "pairwise": pair,
            "top1_em_minus_greedy_ci": _bootstrap_top1_delta(
                selected_em, greedy_em, seed=42 + formula_index,
            ),
            "gates": {
                "pairwise_at_least_0_60": pair["accuracy"] is not None and pair["accuracy"] >= 0.60,
                "top1_not_below_greedy_0_69": sum(selected_em) / len(selected_em) >= 0.69,
            },
            "selected_qids": [row["qid"] for row in selected_rows],
        }
    return result


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankability-dir", required=True)
    parser.add_argument("--hybrid-overrides", required=True)
    parser.add_argument("--question-kg-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank_dir = Path(args.rankability_dir)
    paths = {
        "rollouts": rank_dir / "rollouts.jsonl",
        "rankability_summary": rank_dir / "summary.json",
        "rankability_manifest": rank_dir / "manifest.json",
        "hybrid_overrides": Path(args.hybrid_overrides),
        "question_kg_index": Path(args.question_kg_index),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    rollouts = _read_jsonl(paths["rollouts"])
    overrides = {str(row["qid"]): row for row in _read_jsonl(paths["hybrid_overrides"])}
    qkg = _load_qkg(paths["question_kg_index"])
    sampled: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    greedy: Dict[str, Dict[str, Any]] = {}
    scored_rows: List[Dict[str, Any]] = []
    for source in rollouts:
        row = dict(source)
        qid = str(row["qid"])
        if row["candidate_type"] == "greedy":
            greedy[qid] = row
            continue
        if row["candidate_type"] != "sampled":
            continue
        if qid not in overrides:
            raise KeyError(f"qid={qid} missing hybrid passages")
        kg = qkg.get(str(row["question"]), [])
        row["offline_scores"] = score_candidate(
            row, passages=overrides[qid]["retrieved_passages"], kg=kg,
        )
        row["diagnostic_split"] = _candidate_split(qid)
        sampled[qid].append(row)
        scored_rows.append(row)
    if set(sampled) != set(greedy) or len(sampled) != 100:
        raise ValueError(f"Expected matching 100-qid greedy/sampled sets, got {len(greedy)}/{len(sampled)}")
    if any(len(rows) != 4 for rows in sampled.values()):
        raise ValueError("Every qid must have exactly K=4 sampled candidates")

    formulas = list(scored_rows[0]["offline_scores"])
    formula_summary = summarise(sampled, greedy, formulas)
    split_counts = Counter(row["diagnostic_split"] for qid, rows in sampled.items() for row in rows[:1])
    summary = {
        "experiment_id": args.experiment_id,
        "status": "EXPLORATORY_NOT_INDEPENDENT_CONFIRMATION",
        "n_qids": len(sampled),
        "n_candidates": len(scored_rows),
        "greedy_em": sum(float(row["em"]) for row in greedy.values()) / len(greedy),
        "oracle_at_4_em": sum(max(float(row["em"]) for row in rows) for rows in sampled.values()) / len(sampled),
        "diagnostic_split_counts": dict(sorted(split_counts.items())),
        "formulas": formula_summary,
        "decision_rule": {
            "pairwise_accuracy": 0.60,
            "top1_em_floor": 0.69,
            "independent_confirmation_required_before_core_reward_change": True,
        },
    }
    run_record = {
        "phase": "offline_process_reward_rescoring",
        "protocol": {
            "version": "process_reward_rescoring_exploratory_v1",
            "training": "none",
            "model_generation": "none; saved candidates only",
            "production_reward_changed": False,
            "gold_usage": "evaluation only after each gold-free score is computed",
            "independence_warning": summary["status"],
            "formula_definitions": formulas,
        },
        "input_artifacts": {key: artifact_identity(path) for key, path in paths.items()},
    }
    out_dir, experiment_id = prepare_new_run_dir(
        args.output_dir, experiment_id=args.experiment_id, extra=run_record,
    )
    try:
        _write_jsonl(out_dir / "rescored_candidates.jsonl", scored_rows)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        dump_manifest(out_dir, extra={**run_record, "experiment_id": experiment_id, "summary": summary})
        # Avoid printing the 100 selected qids per formula to the terminal.
        compact = {
            key: {
                "top1_em": value["top1_em"],
                "pairwise_accuracy": value["pairwise"]["accuracy"],
                "gates": value["gates"],
                "delta_ci": value["top1_em_minus_greedy_ci"],
            }
            for key, value in formula_summary.items()
        }
        print(json.dumps({"status": summary["status"], "formulas": compact}, indent=2))
    except Exception as exc:
        dump_manifest(
            out_dir,
            extra={**run_record, "experiment_id": experiment_id, "error": repr(exc), "traceback": traceback.format_exc()},
            status="FAILED",
        )
        raise


if __name__ == "__main__":
    main()
