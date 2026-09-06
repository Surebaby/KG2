#!/usr/bin/env python
"""Freeze an untouched, question-only cohort for automatic Proof-KG evaluation.

The source planner dev file contains structural targets, but selection does not
inspect them and the frozen runtime cohort deliberately omits them.  One item
per family is sampled after excluding every family used by the historical
dev-600 evaluation and any explicitly supplied cohorts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

from kgproweight.training.query_planner import balanced_sample
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir


SCHEMA_VERSION = "automatic-proofkg-confirmation-cohort-1"


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256_text(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def freeze_cohort(
    *,
    dev_rows: list[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
    old_evaluated: list[Mapping[str, Any]],
    excluded_keys: set[str],
    per_dataset: int,
    seed: int,
    datasets: tuple[str, ...] = ("2wikimultihopqa", "musique"),
    source_split: str = "planner_dev_untouched_family",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("datasets must be a non-empty unique sequence")
    allowed = {"2wikimultihopqa", "musique"}
    if not set(datasets).issubset(allowed):
        raise ValueError(f"unsupported datasets: {sorted(set(datasets) - allowed)}")
    old_families = {
        dataset: {
            str(assignments[str(row["question_key"])]["family_sha256"])
            for row in old_evaluated
            if str(row["dataset"]) == dataset
        }
        for dataset in datasets
    }
    excluded_families = {
        str(assignments[key]["family_sha256"])
        for key in excluded_keys
        if key in assignments
    }
    selected: list[dict[str, Any]] = []
    availability: dict[str, Any] = {}
    for offset, dataset in enumerate(datasets):
        by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in dev_rows:
            key = str(row["question_key"])
            if str(row["dataset"]) != dataset or key in excluded_keys:
                continue
            family = str(assignments[key]["family_sha256"])
            if family in old_families[dataset] or family in excluded_families:
                continue
            by_family[family].append(row)
        if len(by_family) < per_dataset:
            raise ValueError(
                f"{dataset} has only {len(by_family)} untouched families; need {per_dataset}"
            )
        rng = random.Random(seed + offset)
        chosen_families = rng.sample(sorted(by_family), per_dataset)
        for family in chosen_families:
            candidates = sorted(by_family[family], key=lambda value: str(value["question_key"]))
            row = rng.choice(candidates)
            question = str(row["question"]).strip()
            selected.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "question_key": str(row["question_key"]),
                    "dataset": dataset,
                    "qid": str(row["qid"]),
                    "question": question,
                    "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                    "target_type": str(row["target_type"]),
                    "family_sha256": family,
                    "source_split": source_split,
                }
            )
        availability[dataset] = {
            "eligible_families_before_sampling": len(by_family),
            "historical_dev_evaluated_families": len(old_families[dataset]),
            "selected_families": per_dataset,
            "explicitly_excluded_families": len(excluded_families),
        }
    random.Random(seed).shuffle(selected)
    for index, row in enumerate(selected, start=1):
        row["row_id"] = f"AUTO-PROOF-{index:03d}"
    return selected, availability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--per_dataset", type=int, default=50)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=("2wikimultihopqa", "musique"),
        help="Dataset to include; repeat for multiple. Defaults to both.",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--old_dev_per_dataset", type=int, default=300)
    parser.add_argument("--old_dev_seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument(
        "--source_split_label",
        default="planner_dev_untouched_family",
    )
    parser.add_argument(
        "--scope",
        default=(
            "planner-dev untouched families; question-only automatic KG construction; "
            "zero training; exploratory confirmation"
        ),
    )
    parser.add_argument(
        "--supply_mode",
        choices=("live_wikidata", "training_partition_versioned_store"),
        default="live_wikidata",
        help=(
            "Freeze the scientific boundary for the supply under test. The "
            "training-partition store may use official annotations from other "
            "training families, but never the selected confirmation families."
        ),
    )
    args = parser.parse_args()

    dev_path = Path(args.dev).resolve()
    assignment_path = Path(args.assignments).resolve()
    dev_rows = list(_read_jsonl(dev_path))
    assignments = {
        str(row["question_key"]): row for row in _read_jsonl(assignment_path)
    }
    old_evaluated = balanced_sample(
        dev_path, per_dataset=args.old_dev_per_dataset, seed=args.old_dev_seed
    )
    excluded_keys = {
        str(row["question_key"])
        for path in args.exclude
        for row in _read_jsonl(path)
    }
    selected, availability = freeze_cohort(
        dev_rows=dev_rows,
        assignments=assignments,
        old_evaluated=old_evaluated,
        excluded_keys=excluded_keys,
        per_dataset=args.per_dataset,
        seed=args.seed,
        datasets=tuple(args.dataset or ("2wikimultihopqa", "musique")),
        source_split=args.source_split_label,
    )
    output_dir, experiment_id = prepare_new_run_dir(
        args.output_dir,
        experiment_id=args.experiment_id,
        extra={
            "phase": "freeze_automatic_proofkg_unseen_confirmation",
            "dev": artifact_identity(dev_path),
            "assignments": artifact_identity(assignment_path),
        },
    )
    cohort_path = output_dir / "cohort.question_only.jsonl"
    with cohort_path.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(str(row["dataset"]) for row in selected)
    training_store = args.supply_mode == "training_partition_versioned_store"
    protocol = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_PLANNER_AND_KG_BUILD",
        "experiment_family": args.experiment_id.casefold().replace("_", "-"),
        "scope": args.scope,
        "n": len(selected),
        "by_dataset": dict(counts),
        "qid_order_sha256": _sha256_text(str(row["question_key"]) for row in selected),
        "runtime_allowed_fields": [
            "dataset", "qid", "question", "question_sha256", "target_type"
        ],
        "runtime_prohibited_fields": [
            "answer", "golden_answers", "target", "supporting_facts", "evidences",
            "question_decomposition", "paragraph_text"
        ],
        "stages": [
            "learned planner greedy generation from question only",
            (
                "title/QID resolution and exact planned PID execution against a "
                "versioned store built only from non-confirmation training families"
                if training_store else
                "title/QID resolution and exact planned PID property execution"
            ),
            "freeze runtime KG and hashes",
            "post-freeze structural coverage audit using dataset annotations",
            "standard-retrieval paired legacy-KG versus Proof-KG model evaluation",
        ],
        "engineering_gates": {
            "planner_schema_valid_rate_min": 0.97,
            "anchor_qid_resolved_rate_min": 0.85,
            "proof_kg_nonempty_rate_min": 0.80,
            "complete_plan_execution_rate_min": 0.65,
            "full_relation_value_chain_rate_evaluable_min": 0.65,
            "runtime_exception_count_max": 0,
        },
        "model_utility_gates": {
            "each_arm_parse_rate_min": 0.90,
            "proof_minus_legacy_em_min": 0.03,
            "proof_minus_legacy_net_correct_min": 3,
            "proof_known_citation_response_rate_min": 0.50,
            "citation_contract_error_rate_increase_max": 0.05,
        },
        "decision_rule": (
            "Do not train on or deploy the new supply unless every engineering gate passes "
            "and the frozen SFT paired evaluation passes every model-utility gate."
        ),
        "scientific_boundary": {
            "supply_mode": args.supply_mode,
            "selected_confirmation_gold_used_for_runtime_build": False,
            "other_training_partition_annotations_used_for_store": training_store,
            "selected_confirmation_families_must_be_excluded_from_store": training_store,
            "selected_confirmation_gold_used_only_after_runtime_freeze": True,
            # Kept for compatibility with the original live-Wikidata protocol.
            "gold_used_for_runtime_build": False if not training_store else None,
            "gold_used_only_after_runtime_freeze": True if not training_store else None,
            "new_main_table_claim_allowed": False,
            "passing_allows": (
                "2Wiki-specific train-derived supply utility evaluation; it does not "
                "establish a dataset-independent external-Wikidata result"
                if training_store else
                "versioned automatic supply integration followed by continued-SFT smoke"
            ),
            "failing_requires": "stop and attribute planner, entity/QID, PID/value, or model-use failure",
        },
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "status": protocol["status"],
        "selection": {
            "seed": args.seed,
            "per_dataset": args.per_dataset,
            "one_question_per_family": True,
            "historical_dev_sample_excluded": True,
            "historical_reference_sample_per_dataset": args.old_dev_per_dataset,
            "source_split_label": args.source_split_label,
            "explicit_excluded_question_count": len(excluded_keys),
            "availability": availability,
            "historical_family_overlap": 0,
            "explicit_question_overlap": 0,
        },
        "inputs": {
            "dev": artifact_identity(dev_path),
            "assignments": artifact_identity(assignment_path),
            "excludes": [artifact_identity(path) for path in args.exclude],
        },
        "outputs": {
            "cohort": artifact_identity(cohort_path),
            "protocol": artifact_identity(protocol_path),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(output_dir, status=report["status"], extra=report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
