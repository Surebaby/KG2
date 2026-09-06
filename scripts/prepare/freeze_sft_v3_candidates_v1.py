#!/usr/bin/env python3
"""Freeze isolated question-only three-domain SFT candidates before labeling.

Family assignment, within-family representative selection, and reserve order
use question identities only. Gold-bearing raw JSON is decoded, but answer
values are first accessed after the complete question-only selection is saved
and hash-bound. Labels copy the original selected raw row without alteration.
No teacher, retriever, model training, or GPU is invoked by this preparation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.prepare import freeze_sft_v3_protected_ledger_v1 as ledger

ROOT = Path(__file__).resolve().parents[2]
VERSION = "sft-v3-three-domain-candidate-pool-v1"
DATASETS = ledger.DATASETS
DEFAULT_OUTPUT = ROOT / "data/silver_data/sft_v3_three_domain_candidate_pool_n16500_seed42_20260906_v1"
SPLITS = ("train", "validation")


def rank(namespace: str, seed: int, *parts: str) -> str:
    return hashlib.sha256(ledger.canonical_json([namespace, seed, *parts]).encode()).hexdigest()


def family_split(family: str, seed: int) -> str:
    bucket = int(rank("sft-v3-family-split-v1", seed, family)[:8], 16) % 10000
    return "validation" if bucket < 1000 else "train"


def load_ledger(directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    required = {"protocol.json", "report.json", "protected_identities.question_only.jsonl", "exact_question_aliases.question_only.jsonl"}
    if (manifest.get("schema_version") != ledger.VERSION
            or manifest.get("status") != "COMPLETE_FROZEN_IDENTITY_ONLY_NOT_TRAINING_DATA"
            or manifest.get("complete") is not True
            or not required <= set(manifest.get("outputs", {}))
            or (directory / "FAILED.json").exists()):
        raise ValueError("incomplete or failed protected ledger")
    for name, binding in manifest["outputs"].items():
        if Path(name).name != name:
            raise ValueError("unsafe protected-ledger output name")
        ledger._validate_bound(directory / name, binding["sha256"])
    protocol = json.loads((directory / "protocol.json").read_text())
    if protocol.get("identity_scope") != "dataset-scoped qid; GLOBAL exact stripped-question SHA256 and GLOBAL current lexical family SHA256":
        raise ValueError("candidate selection requires global question/family exclusion")
    rows = [row for _, row, _ in ledger.read_rows(directory / "protected_identities.question_only.jsonl")]
    if len(rows) != manifest["protected_dataset_qids"]:
        raise ValueError("protected-ledger row count mismatch")
    return rows, manifest


def collect_safe_candidates(*, raw_sources: Mapping[str, Path], protected: list[dict[str, Any]], seed: int):
    index = ledger.make_index(protected)
    candidates = []
    telemetry = {}
    for dataset in DATASETS:
        path = raw_sources[dataset]
        source_binding = ledger.bind(path)
        counts: Counter[str] = Counter()
        seen_qids, question_qids = {}, defaultdict(set)
        for line_number, raw, line_sha in ledger.read_rows(path):
            counts["raw_rows"] += 1
            item = ledger.identity(raw, dataset)
            qid = item["qid"]
            if qid in seen_qids:
                raise ValueError(f"duplicate dataset/qid in raw train source: {dataset}::{qid}")
            seen_qids[qid] = item["question_sha256"]
            question_qids[item["question_sha256"]].add(qid)
            reasons = ledger.overlap_reasons(item, index)
            counts.update("protected_" + field for field in reasons)
            if reasons:
                counts["protected_any"] += 1
                continue
            counts["safe_rows_before_global_family_dedup"] += 1
            family = item["family_sha256"]
            candidates.append({**item, "source_split": "train", "split": family_split(family, seed),
                               "family_dataset_owner_rank": rank("sft-v3-family-dataset-owner-v1", seed, family, dataset),
                               "within_family_question_rank": rank("sft-v3-family-question-v1", seed, item["question_sha256"], qid),
                               "selection_rank": rank("sft-v3-candidate-order-v1", seed, dataset, family),
                               "source": {**source_binding, "line_number": line_number, "line_bytes_sha256": line_sha}})
        ledger._validate_bound(path, source_binding["sha256"])
        counts["raw_exact_question_multi_qid_groups"] = sum(len(ids) > 1 for ids in question_qids.values())
        telemetry[dataset] = dict(counts)
    return candidates, telemetry


def select_candidates(candidates: list[dict[str, Any]], *, train_per_dataset: int, validation_per_dataset: int):
    """Select exactly one question globally per family without reading labels."""
    if min(train_per_dataset, validation_per_dataset) < 1:
        raise ValueError("candidate quotas must be positive")
    by_family = defaultdict(list)
    for row in candidates:
        by_family[row["family_sha256"]].append(row)
    representatives = []
    counts = {dataset: Counter() for dataset in DATASETS}
    cross_dataset_family_groups = 0
    for family in sorted(by_family):
        rows = by_family[family]
        owners = {row["dataset"] for row in rows}
        cross_dataset_family_groups += len(owners) > 1
        selected = min(rows, key=lambda row: (row["family_dataset_owner_rank"], row["within_family_question_rank"], row["question_sha256"], row["qid"]))
        representatives.append(selected)
        counts[selected["dataset"]]["globally_owned_families"] += 1
        counts[selected["dataset"]]["owned_" + selected["split"] + "_families"] += 1
        for row in rows:
            if row is selected:
                continue
            cause = "same_dataset_family_representative_not_selected" if row["dataset"] == selected["dataset"] else "family_owned_by_other_dataset"
            counts[row["dataset"]][cause] += 1
    quotas = {"train": train_per_dataset, "validation": validation_per_dataset}
    selected_by_cell = {}
    for split in SPLITS:
        for dataset in DATASETS:
            pool = sorted((row for row in representatives if row["dataset"] == dataset and row["split"] == split),
                          key=lambda row: (row["selection_rank"], row["family_sha256"], row["qid"]))
            if len(pool) < quotas[split]:
                raise ValueError(f"insufficient frozen {dataset}/{split} families: {len(pool)} < {quotas[split]}; no redraw allowed")
            chosen = pool[:quotas[split]]
            counts[dataset]["unselected_" + split + "_family_reserve_outside_release"] = len(pool) - len(chosen)
            selected_by_cell[(split, dataset)] = chosen
    ordered = []
    for split in SPLITS:
        for within_cell_index in range(quotas[split]):
            for dataset in DATASETS:
                row = selected_by_cell[(split, dataset)][within_cell_index]
                ordered.append({"schema_version": VERSION, **row,
                                "question_key": f"{dataset}::{row['qid']}",
                                "role": f"sft_v3_{split}_candidate_reserve",
                                "within_split_dataset_rank": within_cell_index + 1,
                                "consumption_order": len(ordered) + 1,
                                "gold_access": False, "evaluation_eligible": False,
                                "teacher_acceptance_pending": True})
    return ordered, {"by_dataset": {dataset: dict(value) for dataset, value in counts.items()},
                     "cross_dataset_family_groups_before_owner_assignment": cross_dataset_family_groups,
                     "candidate_quotas_per_dataset": quotas}


def verify_isolation(selected: list[dict[str, Any]], protected: list[dict[str, Any]]) -> dict[str, Any]:
    index = ledger.make_index(protected)
    hits = Counter()
    for row in selected:
        hits.update(ledger.overlap_reasons(row, index))
    def identities(rows, field):
        return {(row["dataset"], row[field]) if field == "qid" else row[field] for row in rows}
    train = [row for row in selected if row["split"] == "train"]
    validation = [row for row in selected if row["split"] == "validation"]
    cross = {field: len(identities(train, field) & identities(validation, field)) for field in index}
    duplicates = {field: len(selected) - len(identities(selected, field)) for field in index}
    gates = {"protected_dataset_qid_overlap_zero": hits["qid"] == 0,
             "protected_global_question_hash_overlap_zero": hits["question_sha256"] == 0,
             "protected_global_family_overlap_zero": hits["family_sha256"] == 0,
             "train_validation_all_identity_overlap_zero": not any(cross.values()),
             "all_candidates_globally_question_family_unique": not any(duplicates.values()),
             "all_sources_are_raw_train": all(row["source_split"] == "train" for row in selected),
             "all_candidates_explicitly_non_evaluation": all(row["evaluation_eligible"] is False for row in selected)}
    if not all(gates.values()):
        raise ValueError(f"SFT candidate isolation failed: {gates}")
    return {"gates": gates, "protected_overlap": {field: hits[field] for field in index},
            "train_validation_overlap": cross, "duplicate_counts": duplicates}


def write_rows(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(ledger.canonical_json(row) + "\n")


def copy_selected_labels(*, selected: list[dict[str, Any]], raw_sources: Mapping[str, Path]):
    """Run ONLY after the selected question-only requests have been frozen."""
    expected = {(row["dataset"], row["source"]["line_number"]): row for row in selected}
    results = {}
    for dataset in DATASETS:
        for line_number, raw, line_sha in ledger.read_rows(raw_sources[dataset]):
            item = expected.get((dataset, line_number))
            if item is None:
                continue
            ident = ledger.identity(raw, dataset)
            if any(ident[field] != item[field] for field in ident) or line_sha != item["source"]["line_bytes_sha256"]:
                raise ValueError("label join disagrees with frozen same-line question identity")
            # First semantic Gold access: faithfully copy, never normalise,
            # repair, choose aliases, or condition identity selection on it.
            answers = raw.get("golden_answers")
            if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
                raise ValueError("selected raw row lacks an unchanged nonempty Golden-answer list")
            results[item["question_key"]] = {"schema_version": VERSION + "-labels",
                                            "dataset": dataset, "qid": item["qid"],
                                            "question_key": item["question_key"],
                                            "question_sha256": item["question_sha256"],
                                            "family_sha256": item["family_sha256"],
                                            "split": item["split"], "golden_answers": answers,
                                            "source": item["source"],
                                            "role": "post_generation_checker_only_never_teacher_input"}
    if len(results) != len(selected):
        raise ValueError("not every selected question has exactly one same-line label")
    return [results[row["question_key"]] for row in selected]


def freeze_candidates(*, output_dir: Path, protected_ledger_dir: Path, raw_sources: Mapping[str, Path],
                      experiment_id: str, seed: int = 42, train_per_dataset: int = 5000,
                      validation_per_dataset: int = 500) -> dict[str, Any]:
    if set(raw_sources) != set(DATASETS):
        raise ValueError("exactly three raw-train source domains required")
    protected, authority = load_ledger(protected_ledger_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {"schema_version": VERSION, "experiment_id": experiment_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
                "code": ledger.bind(Path(__file__)), "ledger_code": ledger.bind(Path(ledger.__file__)),
                "family_code": ledger.bind(ROOT / "scripts/prepare/freeze_qpeg_v1_protocol.py"),
                "protected_ledger": ledger.bind(protected_ledger_dir / "manifest.json"),
                "protected_identities": ledger.bind(protected_ledger_dir / "protected_identities.question_only.jsonl"),
                "raw_train_sources": {dataset: ledger.bind(path) for dataset, path in raw_sources.items()},
                "candidate_quotas_per_dataset": {"train": train_per_dataset, "validation": validation_per_dataset},
                "family_partition": "SHA256(canonicalJSON(['sft-v3-family-split-v1',seed,GLOBAL_family])) first8hex %10000: <1000 validation, else train",
                "cross_dataset_family_owner": "lowest SHA256(canonicalJSON(['sft-v3-family-dataset-owner-v1',seed,family,dataset]))",
                "within_family_question": "lowest SHA256(canonicalJSON(['sft-v3-family-question-v1',seed,question_sha256,qid])); no labels or evidence quality",
                "selection_order": "lowest SHA256(canonicalJSON(['sft-v3-candidate-order-v1',seed,dataset,family])) within fixed split/dataset",
                "consumption_order": "train then validation; within split round-robin hotpotqa,2wikimultihopqa,musique by frozen within-cell rank",
                "reserve_policy": "all frozen candidates are ordered reserves; no outcome-conditioned redraw or external replacement; teacher/quality rejection preserved",
                "candidate_shortage_policy": "fail closed; preserve failure; no split redraw or family relaxation",
                "goal_is_qualified_train6000_validation300_not_guaranteed_by_candidate_capacity": True,
                "labels": "copy complete golden_answers list from same selected raw line AFTER immutable question-only selection; checker-only, never retrieval or teacher input",
                "identity_scope": "dataset qid; global question SHA and lexical family; every family contributes at most one question",
                "training_started": False, "teacher_api_called": False, "retrieval_started": False,
                "future_KG_input_version_must_not_change_this_random_identity_selection": True}
    ledger.write_json(output_dir / "protocol.json", protocol)
    try:
        for dataset, binding in protocol["raw_train_sources"].items():
            expected = authority["raw_train_capacity"][dataset]["source"]
            if binding["sha256"] != expected["sha256"] or binding["bytes"] != expected["bytes"]:
                raise ValueError("raw training source differs from protected-ledger capacity audit")
        candidates, source_stats = collect_safe_candidates(raw_sources=raw_sources, protected=protected, seed=seed)
        selected, selection_stats = select_candidates(candidates, train_per_dataset=train_per_dataset, validation_per_dataset=validation_per_dataset)
        checks = verify_isolation(selected, protected)
        selection_files = ["candidates.question_only.jsonl", "retrieval_requests.question_only.jsonl",
                           "train_candidates.question_only.jsonl", "validation_candidates.question_only.jsonl"]
        write_rows(output_dir / selection_files[0], selected)
        write_rows(output_dir / selection_files[1], selected)
        for split in SPLITS:
            write_rows(output_dir / f"{split}_candidates.question_only.jsonl", (row for row in selected if row["split"] == split))
        ledger.write_json(output_dir / "before_gold_labels.json", {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "status": "QUESTION_ONLY_SELECTION_FROZEN_BEFORE_LABEL_COPY",
            "source_json_objects_with_gold_decoded": True, "gold_values_used_for_selection": False,
            "checks": checks, "selection_outputs": {name: ledger.bind(output_dir / name) for name in selection_files}})
        labels = copy_selected_labels(selected=selected, raw_sources=raw_sources)
        write_rows(output_dir / "labels.checker_only.jsonl", labels)
        for binding in [*protocol["raw_train_sources"].values(), protocol["code"], protocol["ledger_code"],
                        protocol["family_code"], protocol["protected_ledger"], protocol["protected_identities"]]:
            ledger._validate_bound(Path(binding["path"]), binding["sha256"])
        report = {"schema_version": VERSION, "experiment_id": experiment_id,
                  "status": "COMPLETE_CANDIDATE_IDENTITIES_AND_CHECKER_LABELS_FROZEN_NOT_SFT_DATA", "complete": True,
                  "candidate_questions": len(selected), "candidate_global_families": len({r["family_sha256"] for r in selected}),
                  "by_dataset_split": {dataset: {split: sum(r["dataset"] == dataset and r["split"] == split for r in selected) for split in SPLITS} for dataset in DATASETS},
                  "source_filtering": source_stats, "family_selection": selection_stats,
                  **checks, "protocol": ledger.bind(output_dir / "protocol.json"),
                  "boundary": {"all_raw_sources_train": True, "gold_used_to_choose_candidates": False,
                               "gold_labels_faithfully_copied_after_question_only_freeze": True,
                               "labels_not_reader_teacher_or_retrieval_inputs": True,
                               "answer_surface_in_context_not_tested_or_used_for_selection": True,
                               "candidate_acceptance_quality_unknown": True, "teacher_trajectories_exist": False,
                               "evidence_passages_not_yet_materialized": True, "kg_not_yet_materialized": True,
                               "training_started": False, "family_exclusion_changes_2wiki_template_distribution": True,
                               "original_strong_sft_train_questions_allowed_unless_otherwise_protected": True}}
        ledger.write_json(output_dir / "report.json", report)
        outputs = selection_files + ["protocol.json", "before_gold_labels.json", "labels.checker_only.jsonl", "report.json"]
        ledger.write_json(output_dir / "manifest.json", {**report, "outputs": {name: ledger.bind(output_dir / name) for name in outputs}})
        return report
    except BaseException as exc:
        ledger.write_json(output_dir / "FAILED.json", {"status": "FAILED_NOT_TRAINED", "type": type(exc).__name__,
                                                       "error": str(exc), "partial_outputs_retained": True})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protected-ledger", type=Path, default=ledger.DEFAULT_OUT)
    parser.add_argument("--experiment-id", default="SFT-V3-THREE-DOMAIN-CANDIDATE-POOL-N16500-SEED42-20260906-V1")
    args = parser.parse_args()
    report = freeze_candidates(output_dir=args.output_dir, protected_ledger_dir=args.protected_ledger,
                               raw_sources={dataset: ROOT / "data" / dataset / "train.jsonl" for dataset in DATASETS},
                               experiment_id=args.experiment_id)
    print(json.dumps({"status": report["status"], "candidate_questions": report["candidate_questions"],
                      "by_dataset_split": report["by_dataset_split"], "gates": report["gates"],
                      "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
