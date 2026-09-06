#!/usr/bin/env python
"""Score a preregistered, zero-update ProofKG reward confirmation.

The input candidates are generated before this scorer sees Gold answers.  The
ProofKG process score is Gold-free; Gold is opened only to measure rankability
and the outcome-reward channel.  This file does not modify production reward.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from kgproweight.data.parsers import extract_final_answer, parse_steps
from kgproweight.eval.metrics import compute_em, compute_f1
from kgproweight.training.reward_function import KGProWeightRewardFunction
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir
from scripts.pilot.audit_proofkg_process_rankability import (
    COMPARISON_CUES,
    _phrase_in,
    _reachable_cited_edges,
)


SCORER_VERSION = "proofkg-grounded-process-score-2-dynamic-validity"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def required_steps(record: Mapping[str, Any]) -> int:
    """Input-conditioned validity target; never inferred from model output."""

    if not record.get("kg_subgraph"):
        return 3
    planned = len((record.get("query_plan") or {}).get("hops") or [])
    return max(2, min(3, planned or 2))


def score_process_candidate(
    *, question: str, record: Mapping[str, Any], generation: str
) -> dict[str, Any]:
    """Frozen Gold-free score for an automatically supplied ProofKG path."""

    kg = record.get("kg_subgraph") or []
    triples = [tuple(str(value).strip() for value in edge) for edge in kg if len(edge) == 3]
    steps = parse_steps(generation, known_kg=triples)
    answer = (extract_final_answer(generation) or "").split("\n", 1)[0].strip()
    minimum = required_steps(record)
    valid = KGProWeightRewardFunction._is_valid_trajectory(
        steps, generation, min_steps=minimum, min_reasoning_chars=20
    )

    # Empty ProofKG rows are not evidence-scored.  They retain the legacy
    # three-step format gate and receive process=0 when valid.
    if not triples:
        return {
            "scorer_version": SCORER_VERSION,
            "eligible_proofkg": False,
            "required_steps": minimum,
            "trajectory_valid": bool(valid),
            "prediction": answer,
            "n_steps": len(steps),
            "score": 0.0 if valid else -1.0,
            "components": {},
        }

    known = [triple for step in steps for triple in step.cited_triples]
    unknown_count = sum(len(step.unknown_citation_surfaces) for step in steps)
    attempts = len(known) + unknown_count
    citation_precision = _ratio(len(known), attempts)

    grounded = 0
    for step in steps:
        conclusion = step.intermediate_conclusion or ""
        for head, _, tail in step.cited_triples:
            grounded += int(_phrase_in(head, conclusion) or _phrase_in(tail, conclusion))
    conclusion_grounding = _ratio(grounded, len(known))

    reachable_edges, reachable_nodes = _reachable_cited_edges(
        question, triples, [step.cited_triples for step in steps]
    )
    edge_coverage = _ratio(len(reachable_edges), len(set(triples)))
    outgoing = {head for head, _, _ in triples}
    terminal_nodes = {tail for _, _, tail in triples if tail not in outgoing}
    if COMPARISON_CUES.search(question):
        supported_answers = {
            head for head, _, _ in reachable_edges if _phrase_in(head, question)
        }
    else:
        supported_answers = terminal_nodes.intersection(reachable_nodes)
    answer_alignment = float(
        bool(answer)
        and any(
            _phrase_in(answer, value) or _phrase_in(value, answer)
            for value in supported_answers
        )
    )
    unknown_ratio = _ratio(unknown_count, attempts)
    duplicate_ratio = _ratio(len(known) - len(set(known)), len(known))
    score = (
        0.25 * citation_precision
        + 0.25 * conclusion_grounding
        + 0.30 * edge_coverage
        + 0.20 * answer_alignment
        - 0.50 * unknown_ratio
        - 0.15 * duplicate_ratio
        if valid
        else -1.0
    )
    return {
        "scorer_version": SCORER_VERSION,
        "eligible_proofkg": True,
        "required_steps": minimum,
        "trajectory_valid": bool(valid),
        "prediction": answer,
        "n_steps": len(steps),
        "score": float(score),
        "components": {
            "citation_precision": citation_precision,
            "conclusion_grounding": conclusion_grounding,
            "reachable_edge_coverage": edge_coverage,
            "answer_path_alignment": answer_alignment,
            "unknown_citation_ratio": unknown_ratio,
            "duplicate_citation_ratio": duplicate_ratio,
        },
    }


def _bootstrap_delta(left: Sequence[float], right: Sequence[float], seed: int) -> dict[str, Any]:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(10_000, dtype=float)
    for index in range(len(draws)):
        positions = rng.integers(0, len(delta), len(delta))
        draws[index] = delta[positions].mean()
    return {
        "diff_mean": float(delta.mean()),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "p_value": min(1.0, float(2 * min((draws <= 0).mean(), (draws >= 0).mean()))),
        "n": len(delta),
    }


def _pairwise(rows: Sequence[Mapping[str, Any]], score_key: str) -> dict[str, Any]:
    wins = ties = comparisons = 0
    by_qid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["candidate_type"] == "sampled" and row["process"]["eligible_proofkg"]:
            by_qid[str(row["qid"])].append(row)
    for local in by_qid.values():
        for left_index in range(len(local)):
            for right_index in range(left_index + 1, len(local)):
                left, right = local[left_index], local[right_index]
                if float(left["em"]) == float(right["em"]):
                    continue
                correct, wrong = (left, right) if left["em"] > right["em"] else (right, left)
                comparisons += 1
                if float(correct[score_key]) > float(wrong[score_key]):
                    wins += 1
                elif float(correct[score_key]) == float(wrong[score_key]):
                    ties += 1
    return {
        "accuracy": _ratio(wins + 0.5 * ties, comparisons) if comparisons else None,
        "wins": wins,
        "ties": ties,
        "comparisons": comparisons,
        "eligible_qids": len(by_qid),
    }


def summarize(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    greedy = {str(row["qid"]): row for row in rows if row["candidate_type"] == "greedy"}
    sampled: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["candidate_type"] == "sampled":
            sampled[str(row["qid"])].append(row)
    qids = sorted(set(greedy).intersection(sampled))
    if len(qids) != 100 or any(len(sampled[qid]) != 4 for qid in qids):
        raise ValueError("expected 100 qids with one greedy and four sampled candidates")
    greedy_em = [float(greedy[qid]["em"]) for qid in qids]
    oracle_em = [max(float(row["em"]) for row in sampled[qid]) for qid in qids]
    process_selected = [
        max(sampled[qid], key=lambda row: (float(row["process_score"]), -int(row["candidate_index"])))
        for qid in qids
    ]
    full_selected = [
        max(sampled[qid], key=lambda row: (float(row["full_reward"]), -int(row["candidate_index"])))
        for qid in qids
    ]
    sampled_rows = [row for qid in qids for row in sampled[qid]]
    eligible = [row for row in sampled_rows if row["process"]["eligible_proofkg"]]
    process_pair = _pairwise(rows, "process_score")
    full_pair = _pairwise(rows, "full_reward")
    metrics = {
        "n_qids": len(qids),
        "sampled_candidates": len(sampled_rows),
        "proofkg_eligible_qids": process_pair["eligible_qids"],
        "greedy_em": float(np.mean(greedy_em)),
        "oracle_at_4_em": float(np.mean(oracle_em)),
        "process_top1_em": float(np.mean([row["em"] for row in process_selected])),
        "full_reward_top1_em": float(np.mean([row["em"] for row in full_selected])),
        "full_reward_top1_f1": float(np.mean([row["f1"] for row in full_selected])),
        "proofkg_sample_valid_rate": float(np.mean([row["process"]["trajectory_valid"] for row in eligible])) if eligible else None,
        "process_pairwise": process_pair,
        "full_reward_pairwise": full_pair,
        "oracle_minus_greedy_ci": _bootstrap_delta(oracle_em, greedy_em, seed),
        "full_reward_minus_greedy_ci": _bootstrap_delta(
            [float(row["em"]) for row in full_selected], greedy_em, seed + 1
        ),
    }
    gates = {
        "exploration_headroom": metrics["oracle_at_4_em"] - metrics["greedy_em"] >= 0.05,
        "full_reward_selected_gain": metrics["full_reward_top1_em"] - metrics["greedy_em"] >= 0.02,
        "process_pairwise_accuracy": process_pair["accuracy"] is not None and process_pair["accuracy"] >= 0.60,
        "proofkg_sample_valid_rate": metrics["proofkg_sample_valid_rate"] is not None and metrics["proofkg_sample_valid_rate"] >= 0.90,
    }
    return {"metrics": metrics, "gates": gates, "all_pass": all(gates.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--proof_input", required=True)
    parser.add_argument("--question_kg_records", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol).resolve()
    candidates_path = Path(args.candidates).resolve()
    proof_path = Path(args.proof_input).resolve()
    records_path = Path(args.question_kg_records).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = {"candidates": candidates_path, "proof_input": proof_path, "question_kg_records": records_path}
    for name, path in paths.items():
        expected = protocol["inputs"][name]["sha256"]
        # Candidate bytes do not exist at preregistration time.  Their generator,
        # inputs, decoding settings, and destination are frozen separately; this
        # scorer records the produced hash without mutating the protocol.
        if expected is not None and _sha256(path) != expected:
            raise ValueError(f"{name} SHA256 mismatch")
    if protocol["process_scorer"]["version"] != SCORER_VERSION:
        raise ValueError("scorer version mismatch")
    if _sha256(Path(__file__).resolve()) != protocol["process_scorer"]["implementation_sha256"]:
        raise ValueError("scorer implementation SHA256 mismatch")

    proof = {str(row["qid"]): row for row in _read_jsonl(proof_path)}
    records = {str(row["qid"]): row for row in _read_jsonl(records_path)}
    output_rows: list[dict[str, Any]] = []
    for raw in _read_jsonl(candidates_path):
        qid = str(raw["qid"])
        spec, record = proof[qid], records[qid]
        process = score_process_candidate(
            question=str(spec["question"]), record=record, generation=str(raw["generation"])
        )
        # Gold access starts here, after the process score is fixed.
        golds = [str(value) for value in spec["gold_answers"]]
        em = float(compute_em(process["prediction"], golds))
        f1 = float(compute_f1(process["prediction"], golds))
        if process["trajectory_valid"]:
            full_reward = float(process["score"] + 4.0 * (em + 0.10 * f1))
        else:
            full_reward = -4.0
        output_rows.append({
            "qid": qid,
            "candidate_type": raw["candidate_type"],
            "candidate_index": int(raw["candidate_index"]),
            "generation": raw["generation"],
            "process": process,
            "process_score": float(process["score"]),
            "full_reward": full_reward,
            "em": em,
            "f1": f1,
        })

    run_dir, experiment_id = prepare_new_run_dir(
        args.run_dir,
        experiment_id=args.experiment_id,
        extra={"phase": "proofkg_dynamic_validity_confirmation_zero_update"},
    )
    scored_path = run_dir / "scored_candidates.jsonl"
    with scored_path.open("x", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(output_rows, seed=int(protocol["bootstrap_seed"]))
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "status": "PASS" if summary["all_pass"] else "FAIL_STOP",
        "zero_update": True,
        "production_reward_changed": False,
        "summary": summary,
        "scientific_boundary": protocol["scientific_boundary"],
        "inputs": {name: artifact_identity(path) for name, path in paths.items()},
        "outputs": {"scored_candidates": artifact_identity(scored_path)},
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(run_dir, status=report["status"], extra=report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
