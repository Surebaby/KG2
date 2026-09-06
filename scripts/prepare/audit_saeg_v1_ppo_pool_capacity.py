#!/usr/bin/env python
"""Audit train-only capacity for a balanced SAEG-v1 on-policy PPO pool.

This is an identity/count audit.  It reads questions to recompute the frozen
answer-free lexical family hash, but writes no question, answer, passage, graph,
or target text.  It does not materialise PPO data or authorize training.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
EXPERIMENT_ID = "SAEG-V1-PPO-POOL-CAPACITY-AUDIT-SEED42"
TARGET_PROMPTS_PER_DATASET = 1800
K = 4


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qid(row: Mapping[str, Any]) -> str:
    return str(row.get("source_qid") or row.get("qid") or row.get("id") or "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument(
        "--sft_release",
        type=Path,
        default=Path("data/silver_data/saeg_v1_train4860_family_disjoint_seed42_v2/silver_train.jsonl"),
    )
    parser.add_argument(
        "--eval_dir",
        type=Path,
        default=Path("data/derived/saeg_v1_evaluation_inputs_seed42_v1"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/audits/saeg_v1_ppo_pool_capacity_v1"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite PPO capacity audit: {args.out}")

    eval_paths = [
        args.eval_dir / "development.answer_free.jsonl",
        args.eval_dir / "confirmation.answer_free.jsonl",
        args.eval_dir / "canonical_reporting.answer_free.jsonl",
    ]
    consumed_paths = [
        Path("outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_train.question_only.jsonl"),
        Path("outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_dev.question_only.jsonl"),
        Path("outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_confirmation.question_only.jsonl"),
        Path("outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_reserve.question_only.jsonl"),
        Path("outputs/audits/2wiki_train_only_rankability_n150_v1/cohort.question_only.jsonl"),
        Path("outputs/audits/2wiki_train_only_rankability_confirmation_n100_v1/cohort.question_only.jsonl"),
    ]
    raw_paths = {dataset: args.data_root / dataset / "train.jsonl" for dataset in DATASETS}
    required = [args.sft_release, *eval_paths, *consumed_paths, *raw_paths.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing capacity-audit inputs: {missing}")

    heldout_qids: set[tuple[str, str]] = set()
    heldout_families: set[str] = set()
    for path in eval_paths:
        for row in read_jsonl(path):
            dataset = str(row["dataset"])
            heldout_qids.add((dataset, qid(row)))
            heldout_families.add(family_sha256(str(row["question"])))

    consumed_qids: set[tuple[str, str]] = set()
    consumed_families: set[str] = set()
    for path in consumed_paths:
        for row in read_jsonl(path):
            dataset = str(row.get("dataset") or "2wikimultihopqa")
            consumed_qids.add((dataset, qid(row)))
            consumed_families.add(family_sha256(str(row["question"])))

    existing_modes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in read_jsonl(args.sft_release):
        existing_modes[(str(row["dataset"]), qid(row))].add(str(row["evidence_mode"]))

    raw_counts: dict[str, Counter[str]] = {}
    eligible_new: dict[str, int] = {}
    existing_by_dataset = Counter(dataset for dataset, _ in existing_modes)
    for dataset, path in raw_paths.items():
        counts: Counter[str] = Counter()
        seen_qids: set[str] = set()
        for row in read_jsonl(path):
            row_qid = qid(row)
            if not row_qid or row_qid in seen_qids:
                counts["missing_or_duplicate_qid"] += 1
                continue
            seen_qids.add(row_qid)
            counts["raw_unique_qids"] += 1
            key = (dataset, row_qid)
            family = family_sha256(str(row["question"]))
            if key in heldout_qids or family in heldout_families:
                counts["excluded_saeg_eval_qid_or_family"] += 1
                continue
            if key in consumed_qids or family in consumed_families:
                counts["excluded_historical_reward_qid_or_family"] += 1
                continue
            counts["safe_after_all_exclusions"] += 1
            if key not in existing_modes:
                counts["safe_new_outside_sft_release"] += 1
        raw_counts[dataset] = counts
        eligible_new[dataset] = counts["safe_new_outside_sft_release"]

    target = TARGET_PROMPTS_PER_DATASET
    additions_needed = {
        dataset: max(0, target - int(existing_by_dataset[dataset]))
        for dataset in DATASETS
    }
    # Existing 2Wiki has 1,824 qids: use 1,800 and deterministically leave 24
    # unused.  Hotpot/MuSiQue each need 1,200 fresh qids.  The exact identities
    # and route assignment are deliberately deferred to a separately frozen
    # materialisation protocol.
    capacity_checks = {
        f"{dataset}_can_fill_balanced_1800": eligible_new[dataset] >= additions_needed[dataset]
        for dataset in DATASETS
    }
    capacity_checks.update({
        "target_prompts_5400": target * len(DATASETS) == 5400,
        "formal_rollouts_per_arm_at_least_20000": target * len(DATASETS) * K >= 20000,
        "sft_release_eval_qid_overlap_zero": not (set(existing_modes) & heldout_qids),
        "no_gold_or_content_written": True,
    })
    status = "PASS_CAPACITY_ONLY_NOT_MATERIALIZED" if all(capacity_checks.values()) else "FAIL_CAPACITY"
    report = {
        "schema_version": "saeg-v1-ppo-pool-capacity-audit-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "proposed_budget_not_yet_frozen": {
            "unique_prompts": 5400,
            "unique_prompts_per_dataset": target,
            "rollouts_per_prompt": K,
            "trajectories_per_arm": 5400 * K,
            "smoke_prefix": {"prompts": 300, "trajectories_per_arm": 1200},
            "mid_prefix": {"prompts": 2000, "trajectories_per_arm": 8000},
            "formal_full_pool": {"prompts": 5400, "trajectories_per_arm": 21600},
        },
        "existing_sft_release_unique_qids": {
            "total": len(existing_modes),
            "by_dataset": dict(sorted(existing_by_dataset.items())),
            "note": "P/W/fused variants sharing one source qid count once.",
        },
        "additions_needed_for_balanced_5400": additions_needed,
        "eligible_new_capacity": eligible_new,
        "raw_capacity_counts": {key: dict(value) for key, value in raw_counts.items()},
        "exclusion_sets": {
            "saeg_eval_qids": len(heldout_qids),
            "saeg_eval_lexical_families": len(heldout_families),
            "historical_reward_qids": len(consumed_qids),
            "historical_reward_lexical_families": len(consumed_families),
        },
        "checks": capacity_checks,
        "inputs": {
            "sft_release": {"path": str(args.sft_release), "sha256": sha256_file(args.sft_release)},
            "eval": [{"path": str(path), "sha256": sha256_file(path)} for path in eval_paths],
            "historical_consumed": [
                {"path": str(path), "sha256": sha256_file(path)} for path in consumed_paths
            ],
            "raw_train": {
                dataset: {"path": str(path), "sha256": sha256_file(path)}
                for dataset, path in raw_paths.items()
            },
        },
        "next_required_stage": (
            "After the development utility gate, freeze exact qids and source routes, reserve fresh "
            "family-disjoint reward dev/confirmation, then materialise missing P graphs and canonical passages."
        ),
        "scientific_boundary": (
            "Capacity-only CPU audit. It does not freeze identities, construct graph evidence, expose Gold, "
            "modify reward/evaluation, authorize PPO, or claim that 5,400 prompts are usable yet."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(args.out, extra=report, status=status)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status != "PASS_CAPACITY_ONLY_NOT_MATERIALIZED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
