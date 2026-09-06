#!/usr/bin/env python
"""Write an append-only family-scope/code-lock addendum for mixed PPO v2."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_mixed_ppo_three_dataset_v1 import read_jsonl, sha256_file
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256


PROTOCOL_DIR = Path("outputs/audits/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_protocol")
DATA_DIR = Path("data/silver_data/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42")
V1_PROTOCOL_DIR = Path("outputs/audits/mixed_ppo_three_dataset_v1_n1799_k4_seed42_protocol")
PREVIOUS_ADDENDUM = Path("outputs/audits/mixed_ppo_three_dataset_v2_proof400_family_scope_addendum_v1/addendum.json")
OUT = Path("outputs/audits/mixed_ppo_three_dataset_v2_proof400_family_scope_addendum_v2")
STATUS = "COMPLETE_APPEND_ONLY_CLARIFICATION_DATA_UNCHANGED"


def ref(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def identities(rows):
    return (
        {(row["dataset"], row["qid"]) for row in rows},
        {(row["dataset"], family_sha256(row["question"])) for row in rows},
    )


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite addendum: {OUT}")
    population = read_jsonl(PROTOCOL_DIR / "population.question_only.jsonl")
    protected = [
        *read_jsonl(PROTOCOL_DIR / "protected_a_canonical_main.question_only.jsonl"),
        *read_jsonl(PROTOCOL_DIR / "protected_a_unopened_confirmation.question_only.jsonl"),
    ]
    pop_qids, pop_families = identities(population)
    protected_qids, protected_families = identities(protected)

    by_family = defaultdict(list)
    for row in population:
        by_family[family_sha256(row["question"])].append((row["dataset"], row["qid"]))
    cross = {family: values for family, values in by_family.items() if len({dataset for dataset, _ in values}) > 1}
    protected_family_datasets = defaultdict(set)
    for row in protected:
        protected_family_datasets[family_sha256(row["question"])].add(row["dataset"])
    cross_to_a_rows = [
        row for row in population
        if family_sha256(row["question"]) in protected_family_datasets
        and row["dataset"] not in protected_family_datasets[family_sha256(row["question"])]
    ]

    v1 = read_jsonl(V1_PROTOCOL_DIR / "population.question_only.jsonl")
    _v1_qids, v1_families = identities(v1)
    overlap_pairs = v1_families & protected_families
    v1_overlap_rows = sum(
        (row["dataset"], family_sha256(row["question"])) in protected_families for row in v1
    )
    if pop_qids & protected_qids or pop_families & protected_families:
        raise ValueError("v2 protected A-class isolation no longer holds")

    payload = {
        "schema_version": "mixed-ppo-v2-family-scope-addendum-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "clarification": {
            "family_version": FAMILY_VERSION,
            "family_isolation_scope": "dataset-scoped (dataset, family_sha256)",
            "reason": "The same lexical template in different datasets is not an evaluation-family leak.",
            "protected_a_qid_overlap": 0,
            "protected_a_dataset_scoped_family_overlap": 0,
            "population_internal_cross_dataset_same_template_family_count": len(cross),
            "population_internal_cross_dataset_same_template_row_count": sum(len(values) for values in cross.values()),
            "population_internal_cross_dataset_same_template_rows_by_dataset": dict(sorted(Counter(
                dataset for values in cross.values() for dataset, _qid in values
            ).items())),
            "population_to_A_cross_dataset_same_template_family_count": len({
                family_sha256(row["question"]) for row in cross_to_a_rows
            }),
            "population_to_A_cross_dataset_same_template_row_count": len(cross_to_a_rows),
            "population_to_A_cross_dataset_same_template_rows_by_dataset": dict(sorted(Counter(
                row["dataset"] for row in cross_to_a_rows
            ).items())),
            "train_side_family_repeats_allowed": True,
        },
        "supersession": {
            "previous_addendum": ref(PREVIOUS_ADDENDUM),
            "superseded_fields": [
                "clarification.cross_dataset_same_template_family_count",
                "clarification.cross_dataset_same_template_row_count",
                "clarification.cross_dataset_same_template_rows_by_dataset",
            ],
            "reason": (
                "The previous telemetry counted only cross-dataset duplicates internal to the training population. "
                "This version reports that quantity separately from raw-family collisions between population and A."
            ),
            "underlying_data_or_isolation_decision_changed": False,
        },
        "v1_supersession_evidence": {
            "stored_namespace_claim": "population_eval_family_overlap_zero",
            "status": "SUPERSEDED_NAMESPACE_INCOMPARABLE",
            "v1_recomputed_dataset_scoped_overlap_family_count": len(overlap_pairs),
            "v1_recomputed_overlap_row_count": v1_overlap_rows,
            "v1_data_or_results_modified": False,
        },
        "code_lock": {
            "freeze_v2": ref(Path("scripts/prepare/freeze_mixed_ppo_three_dataset_v2_proof400.py")),
            "materialize_v2": ref(Path("scripts/prepare/materialize_mixed_ppo_three_dataset_v2_proof400.py")),
            "family_implementation": ref(Path("scripts/prepare/freeze_qpeg_v1_protocol.py")),
        },
        "bound_assets": {
            "protocol": ref(PROTOCOL_DIR / "protocol.json"),
            "protocol_manifest": ref(PROTOCOL_DIR / "manifest.json"),
            "data_report": ref(DATA_DIR / "report.json"),
            "data_manifest": ref(DATA_DIR / "manifest.json"),
            "silver_train": ref(DATA_DIR / "silver_train.jsonl"),
            "question_kg_records": ref(DATA_DIR / "question_kg_records.jsonl"),
            "fixed_rollout_schedule": ref(DATA_DIR / "fixed_rollout_schedule.jsonl"),
        },
        "scientific_boundary": {
            "append_only_clarification": True,
            "underlying_protocol_or_data_modified": False,
            "training_started": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=False)
    report = OUT / "addendum.json"
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(OUT, status=STATUS, extra={
        "phase": "mixed_ppo_v2_family_scope_and_code_lock",
        "addendum_sha256": sha256_file(report),
    })
    print(json.dumps(payload["clarification"] | payload["v1_supersession_evidence"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
