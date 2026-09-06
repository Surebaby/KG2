#!/usr/bin/env python
"""Freeze a contrastive ProofKG curriculum and untouched reserve evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.utils.logging import dump_manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gold_answers(row: Mapping[str, Any]) -> list[str]:
    value = row.get("gold_answers", row.get("answer"))
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if value is not None and str(value).strip() else []


def select_candidate_strata(
    by_qid: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, str]:
    """Select stochastic contrast qids; greedy outcome only names the stratum."""
    selected: dict[str, str] = {}
    for qid, rows in by_qid.items():
        greedy = [row for row in rows if row.get("candidate_type") == "greedy"]
        sampled = [row for row in rows if row.get("candidate_type") == "sampled"]
        if len(greedy) != 1 or len(sampled) != 4:
            raise ValueError(f"expected greedy+K4 for {qid}, got {len(greedy)}+{len(sampled)}")
        outcomes = {int(float(row.get("em", 0))) for row in sampled}
        if outcomes == {0, 1}:
            selected[str(qid)] = "recovery" if float(greedy[0]["em"]) == 0 else "stability"
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_sources", type=Path, nargs="+", required=True)
    parser.add_argument("--full_cohort", type=Path, required=True)
    parser.add_argument("--full_question_kg", type=Path, required=True)
    parser.add_argument("--full_runtime_details", type=Path, required=True)
    parser.add_argument("--full_silver", type=Path, required=True)
    parser.add_argument("--reserve", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base_model", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    cohort = {str(row["qid"]): row for row in _read_jsonl(args.full_cohort)}
    question_kg = {str(row["qid"]): row for row in _read_jsonl(args.full_question_kg)}
    runtime = {str(row["qid"]): row for row in _read_jsonl(args.full_runtime_details)}
    silver = {str(row["qid"]): row for row in _read_jsonl(args.full_silver)}

    by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_by_qid: dict[str, str] = {}
    source_counts: Counter[str] = Counter()
    for source in args.candidate_sources:
        for row in _read_jsonl(source):
            qid = str(row["qid"])
            if qid in source_by_qid and source_by_qid[qid] != str(source):
                raise ValueError(f"candidate qid appears in multiple sources: {qid}")
            source_by_qid[qid] = str(source)
            by_qid[qid].append(row)
        source_counts[str(source)] = len({str(row["qid"]) for row in _read_jsonl(source)})

    selected_strata = select_candidate_strata(by_qid)
    curriculum: list[dict[str, Any]] = []
    recovery = retention = 0
    greedy_wrong_sample_solvable = 0
    for qid in sorted(by_qid):
        rows = by_qid[qid]
        greedy = [row for row in rows if row.get("candidate_type") == "greedy"]
        sampled = [row for row in rows if row.get("candidate_type") == "sampled"]
        greedy_wrong_sample_solvable += int(
            float(greedy[0]["em"]) == 0 and any(float(row["em"]) == 1 for row in sampled)
        )
        if qid not in selected_strata:
            continue
        if qid not in cohort or qid not in question_kg or qid not in runtime or qid not in silver:
            raise ValueError(f"missing complete source assets for {qid}")
        kg = question_kg[qid]
        if not kg.get("kg_subgraph") or not kg.get("provenance", {}).get("complete_plan_execution"):
            raise ValueError(f"contrastive qid does not have complete ProofKG: {qid}")
        stratum = selected_strata[qid]
        recovery += int(stratum == "recovery")
        retention += int(stratum == "stability")
        curriculum.append({
            "schema_version": "proofkg-hard-curriculum-qid-1",
            "dataset": "2wikimultihopqa",
            "qid": qid,
            "question": str(cohort[qid]["question"]),
            "question_sha256": str(cohort[qid]["question_sha256"]),
            "family_sha256": str(cohort[qid]["family_sha256"]),
            "question_type": str(cohort[qid]["question_type"]),
            "stratum": stratum,
            "greedy_em": float(greedy[0]["em"]),
            "sampled_correct": sum(float(row["em"]) == 1 for row in sampled),
            "sampled_wrong": sum(float(row["em"]) == 0 for row in sampled),
            "candidate_source": source_by_qid[qid],
            "proofkg_complete": True,
        })
    if not curriculum or recovery == 0 or retention == 0:
        raise ValueError("both recovery and stability strata must be non-empty")
    for row in curriculum:
        denominator = recovery if row["stratum"] == "recovery" else retention
        row["sampling_probability"] = 0.5 / denominator
    if abs(sum(float(row["sampling_probability"]) for row in curriculum) - 1.0) > 1e-9:
        raise ValueError("curriculum sampling weights do not sum to one")

    reserve_rows = _read_jsonl(args.reserve)
    reserve_qids = {str(row["qid"]) for row in reserve_rows}
    curriculum_qids = {str(row["qid"]) for row in curriculum}
    curriculum_families = {str(row["family_sha256"]) for row in curriculum}
    reserve_families = {str(row["family_sha256"]) for row in reserve_rows}
    if curriculum_qids & reserve_qids or curriculum_families & reserve_families:
        raise ValueError("reserve overlaps curriculum by qid or family")
    if len(reserve_rows) != 82:
        raise ValueError(f"expected 82 untouched reserve qids, got {len(reserve_rows)}")

    reserve_proof: list[dict[str, Any]] = []
    reserve_kg: list[dict[str, Any]] = []
    reserve_runtime: list[dict[str, Any]] = []
    for identity in reserve_rows:
        qid = str(identity["qid"])
        if qid not in silver or qid not in question_kg or qid not in runtime:
            raise ValueError(f"reserve qid lacks materialized ProofKG assets: {qid}")
        kg = question_kg[qid]
        if not kg.get("kg_subgraph") or not kg.get("provenance", {}).get("complete_plan_execution"):
            raise ValueError(f"reserve qid does not have complete ProofKG: {qid}")
        source = silver[qid]
        reserve_proof.append({
            "dataset": "2wikimultihopqa",
            "qid": qid,
            "question": str(source["question"]),
            "gold_answers": _gold_answers(source),
            "retrieved_passages": list(source.get("retrieved_passages") or []),
            "kg_subgraph": list(kg.get("kg_subgraph") or []),
        })
        reserve_kg.append(kg)
        reserve_runtime.append(runtime[qid])
    if any(not row["gold_answers"] for row in reserve_proof):
        raise ValueError("reserve scorer input is missing a train answer")

    outputs = {
        "curriculum": args.output_dir / "train_contrastive_qids.jsonl",
        "reserve_proof": args.output_dir / "reserve_proof_input.scorer_only.jsonl",
        "reserve_kg": args.output_dir / "reserve_question_kg_records.jsonl",
        "reserve_runtime": args.output_dir / "reserve_runtime_details.jsonl",
    }
    _write_jsonl(outputs["curriculum"], curriculum)
    _write_jsonl(outputs["reserve_proof"], reserve_proof)
    _write_jsonl(outputs["reserve_kg"], reserve_kg)
    _write_jsonl(outputs["reserve_runtime"], reserve_runtime)

    protocol = {
        "schema_version": "proofkg-hard-curriculum-protocol-1",
        "experiment_id": "PROOFKG-2WIKI-HARD-CONTRASTIVE-CURRICULUM-V1",
        "status": "FROZEN_BEFORE_UNTOUCHED_RESERVE_GENERATION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "research_question": "Does reward-v2.1 rank correct above wrong stochastic rollouts on a fresh family-disjoint complete-ProofKG reserve, strongly enough to justify a paired hard-curriculum PPO smoke?",
        "prior_result_not_reclassified": "The earlier reward-v2.1 top1-vs-greedy gate remains failed. This protocol tests within-rollout contrast needed by PPO and does not overwrite that result.",
        "curriculum": {
            "selection": "train-only qids with one greedy and K=4 sampled candidates, complete automatic ProofKG, and both correct and wrong sampled outcomes",
            "n": len(curriculum),
            "recovery": recovery,
            "stability": retention,
            "sampling": "50% recovery / 50% stability; uniform within stratum",
            "gold_usage": "train answers used only to define train-only outcome strata",
        },
        "reserve": {
            "n": len(reserve_rows),
            "families": len(reserve_families),
            "previous_candidate_generation": False,
            "qid_overlap_with_curriculum": 0,
            "family_overlap_with_curriculum": 0,
            "role": "single-use rankability validation only; never train",
        },
        "generation": {
            "checkpoint": str(args.adapter),
            "base_model": str(args.base_model),
            "adapter_model_sha256": _sha256(args.adapter / "adapter_model.safetensors"),
            "adapter_config_sha256": _sha256(args.adapter / "adapter_config.json"),
            "base_config_sha256": _sha256(args.base_model / "config.json"),
            "base_index_sha256": _sha256(args.base_model / "model.safetensors.index.json"),
            "greedy_per_qid": 1,
            "sampled_per_qid": 4,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 512,
            "seed": 42,
        },
        "promotion_gates": {
            "sample_valid_rate_min": 0.90,
            "mixed_outcome_qids_min": 25,
            "reward_pairwise_accuracy_min": 0.65,
            "reward_top1_minus_random_sampled_em_min": 0.10,
            "runtime_errors": 0,
            "decision": "all gates must pass before preparing paired PPO-O/PPO-K configs",
        },
        "mandatory_reporting_not_a_gate": [
            "greedy EM", "oracle@4 EM", "reward top1 EM on all reserve qids",
            "recovery qid count", "component correlations", "tie rate",
        ],
        "forbidden": [
            "reward formula or weight changes", "reserve qid replacement",
            "per-qid patches", "using reserve for PPO training",
            "opening any other sealed confirmation", "starting PPO before all gates pass",
        ],
        "inputs": {
            "candidate_sources": [{"path": str(path), "sha256": _sha256(path)} for path in args.candidate_sources],
            "full_cohort": {"path": str(args.full_cohort), "sha256": _sha256(args.full_cohort)},
            "full_question_kg": {"path": str(args.full_question_kg), "sha256": _sha256(args.full_question_kg)},
            "full_runtime_details": {"path": str(args.full_runtime_details), "sha256": _sha256(args.full_runtime_details)},
            "full_silver": {"path": str(args.full_silver), "sha256": _sha256(args.full_silver)},
            "reserve_identity": {"path": str(args.reserve), "sha256": _sha256(args.reserve)},
        },
        "outputs": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in outputs.items()},
        "availability_audit": {
            "candidate_qids": len(by_qid),
            "candidate_source_qids": dict(source_counts),
            "complete_proofkg_candidate_qids": sum(
                bool(question_kg[qid].get("provenance", {}).get("complete_plan_execution")) for qid in by_qid
            ),
            "greedy_wrong_with_any_correct_sample": greedy_wrong_sample_solvable,
            "contrastive_recovery_qids": recovery,
            "contrastive_sampled_outcomes": len(curriculum),
        },
    }
    protocol_path = args.output_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.output_dir, status=protocol["status"], extra={
        "experiment_id": protocol["experiment_id"],
        "phase": "proofkg_hard_curriculum_preregistration",
        "protocol_sha256": _sha256(protocol_path),
    })
    print(json.dumps({
        "status": protocol["status"], "curriculum": protocol["curriculum"],
        "reserve": protocol["reserve"], "promotion_gates": protocol["promotion_gates"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
