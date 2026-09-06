#!/usr/bin/env python
"""Apply the preregistered within-rollout rankability gates to reserve82."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from kgproweight.utils.logging import dump_manifest
from scripts.pilot.rescore_rankability_v2 import _tie_aware_spearman


COMPONENTS = (
    "P_precise_citation", "H_hop_coverage", "O_dependency_order",
    "G_conclusion_grounding", "A_answer_consistency",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_candidates(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_qid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_qid[str(row["qid"])].append(row)
    greedy = {}
    sampled = {}
    for qid, values in by_qid.items():
        greedy_rows = [row for row in values if row["candidate_type"] == "greedy"]
        sampled_rows = [row for row in values if row["candidate_type"] == "sampled"]
        if len(greedy_rows) != 1 or len(sampled_rows) != 4:
            raise ValueError(f"expected greedy+K4 for {qid}")
        greedy[qid] = greedy_rows[0]
        sampled[qid] = sampled_rows

    mixed_qids = sorted(qid for qid, values in sampled.items() if {float(row["em"]) for row in values} == {0.0, 1.0})
    mixed_rows = [row for qid in mixed_qids for row in sampled[qid]]
    wins = ties = comparisons = 0
    for qid in mixed_qids:
        correct = [row for row in sampled[qid] if float(row["em"]) == 1]
        wrong = [row for row in sampled[qid] if float(row["em"]) == 0]
        for left in correct:
            for right in wrong:
                comparisons += 1
                left_score = float(left["process"]["score"])
                right_score = float(right["process"]["score"])
                wins += int(left_score > right_score)
                ties += int(left_score == right_score)
    pairwise = (wins + 0.5 * ties) / comparisons if comparisons else 0.0
    reward_top1 = [
        max(sampled[qid], key=lambda row: (float(row["process"]["score"]), -int(row["candidate_index"])))
        for qid in mixed_qids
    ]
    top1_em = sum(float(row["em"]) for row in reward_top1) / max(1, len(reward_top1))
    random_em = sum(
        sum(float(row["em"]) for row in sampled[qid]) / 4.0 for qid in mixed_qids
    ) / max(1, len(mixed_qids))
    all_sampled = [row for values in sampled.values() for row in values]
    valid_rate = sum(bool(row["process"]["trajectory_valid"]) for row in all_sampled) / max(1, len(all_sampled))
    all_qids = sorted(by_qid)
    greedy_em = sum(float(greedy[qid]["em"]) for qid in all_qids) / len(all_qids)
    oracle_em = sum(max(float(row["em"]) for row in sampled[qid]) for qid in all_qids) / len(all_qids)
    all_top1_em = sum(
        float(max(sampled[qid], key=lambda row: (float(row["process"]["score"]), -int(row["candidate_index"]))) ["em"])
        for qid in all_qids
    ) / len(all_qids)
    recovery = sum(float(greedy[qid]["em"]) == 0 and max(float(row["em"]) for row in sampled[qid]) == 1 for qid in all_qids)
    valid_mixed = [row for row in mixed_rows if row["process"]["trajectory_valid"]]
    return {
        "n_qids": len(all_qids),
        "n_sampled": len(all_sampled),
        "sample_valid_rate": valid_rate,
        "mixed_outcome_qids": len(mixed_qids),
        "recovery_qids": recovery,
        "reward_pairwise_accuracy": pairwise,
        "pairwise_comparisons": comparisons,
        "pairwise_tie_rate": ties / max(1, comparisons),
        "mixed_reward_top1_em": top1_em,
        "mixed_random_sampled_em": random_em,
        "reward_top1_minus_random_sampled_em": top1_em - random_em,
        "greedy_em_all": greedy_em,
        "oracle_at_4_em_all": oracle_em,
        "reward_top1_em_all": all_top1_em,
        "component_spearman_on_valid_mixed": {
            component: _tie_aware_spearman(
                [float(row["process"]["components"][component]) for row in valid_mixed],
                [float(row["em"]) for row in valid_mixed],
            ) if valid_mixed else 0.0
            for component in COMPONENTS
        },
        "combined_spearman_on_valid_mixed": _tie_aware_spearman(
            [float(row["process"]["score"]) for row in valid_mixed],
            [float(row["em"]) for row in valid_mixed],
        ) if valid_mixed else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--generation_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite: {args.output_dir}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    generation_report = json.loads((args.generation_dir / "report.json").read_text(encoding="utf-8"))
    if generation_report["protocol"]["sha256"] != _sha256(args.protocol):
        raise SystemExit("generation did not use the frozen protocol")
    candidates_path = args.generation_dir / "candidates.jsonl"
    if generation_report["candidate_sha256"] != _sha256(candidates_path):
        raise SystemExit("candidate hash mismatch")
    metrics = score_candidates(_read_jsonl(candidates_path))
    gates = protocol["promotion_gates"]
    decisions = {
        "sample_valid_rate": metrics["sample_valid_rate"] >= gates["sample_valid_rate_min"],
        "mixed_outcome_qids": metrics["mixed_outcome_qids"] >= gates["mixed_outcome_qids_min"],
        "reward_pairwise_accuracy": metrics["reward_pairwise_accuracy"] >= gates["reward_pairwise_accuracy_min"],
        "reward_top1_minus_random_sampled_em": metrics["reward_top1_minus_random_sampled_em"] >= gates["reward_top1_minus_random_sampled_em_min"],
        "runtime_errors": int(generation_report["runtime_errors"]) == int(gates["runtime_errors"]),
    }
    all_pass = all(decisions.values())
    args.output_dir.mkdir(parents=True)
    report = {
        "schema_version": "proofkg-hard-curriculum-reserve-result-1",
        "experiment_id": protocol["experiment_id"],
        "status": "PASS_READY_TO_PREPARE_PAIRED_PPO" if all_pass else "FAIL_STOP_BEFORE_PPO",
        "development_only": True,
        "prior_top1_vs_greedy_failure_unchanged": True,
        "protocol": {"path": str(args.protocol), "sha256": _sha256(args.protocol)},
        "candidates": {"path": str(candidates_path), "sha256": _sha256(candidates_path)},
        "metrics": metrics,
        "gates": decisions,
        "all_pass": all_pass,
        "decision": "Prepare but do not launch paired PPO-O/PPO-K configs." if all_pass else "Do not prepare or launch PPO from this curriculum.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    result_path = args.output_dir / "result_record.json"
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.output_dir, status=report["status"], extra={
        "experiment_id": report["experiment_id"],
        "phase": "proofkg_hard_curriculum_reserve_result",
        "result_record_sha256": _sha256(result_path),
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
