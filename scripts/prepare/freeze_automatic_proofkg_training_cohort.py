#!/usr/bin/env python
"""Freeze a balanced, confirmation-disjoint 2Wiki PPO rollout cohort.

The source silver contains Gold-derived traces, but this selector never copies
them.  Its output is question-only and is used solely by the planner/executor;
the later PPO silver materializer must also strip those source traces.
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

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir


SCHEMA_VERSION = "automatic-proofkg-training-cohort-1"
QUESTION_TYPES = ("bridge_comparison", "comparison", "compositional", "inference")


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def select_training_rows(
    source_rows: Iterable[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
    *,
    excluded_keys: set[str],
    excluded_families: set[str],
    per_type: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_type: dict[str, list[tuple[Mapping[str, Any], str]]] = defaultdict(list)
    for row in source_rows:
        if str(row.get("dataset")) != "2wikimultihopqa":
            continue
        qid = str(row.get("qid") or "")
        key = question_key("2wikimultihopqa", qid)
        assignment = assignments.get(key)
        if assignment is None or str(assignment.get("split")) != "train":
            continue
        family = str(assignment.get("family_sha256") or "")
        if key in excluded_keys or family in excluded_families:
            continue
        metadata = row.get("metadata") or {}
        qtype = str(metadata.get("question_type") or "")
        if qtype not in QUESTION_TYPES:
            continue
        if metadata.get("train_only") is not True:
            raise ValueError(f"source row is not explicitly train_only: {key}")
        by_type[qtype].append((row, family))

    selected: list[dict[str, Any]] = []
    for offset, qtype in enumerate(QUESTION_TYPES):
        candidates = sorted(by_type[qtype], key=lambda item: str(item[0]["qid"]))
        if len(candidates) < per_type:
            raise ValueError(
                f"{qtype} has {len(candidates)} eligible rows, need {per_type}"
            )
        for row, family in random.Random(seed + offset).sample(candidates, per_type):
            question = str(row["question"]).strip()
            qid = str(row["qid"])
            selected.append({
                "schema_version": SCHEMA_VERSION,
                "question_key": question_key("2wikimultihopqa", qid),
                "dataset": "2wikimultihopqa",
                "qid": qid,
                "question": question,
                "question_sha256": question_sha256(question),
                "target_type": "relation_graph",
                "question_type": qtype,
                "family_sha256": family,
                "source_split": "train",
            })
    random.Random(seed).shuffle(selected)
    for index, row in enumerate(selected, start=1):
        row["row_id"] = f"AUTO-PROOF-TRAIN-{index:04d}"
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--per_type", type=int, default=375)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    args = parser.parse_args()

    silver_path = Path(args.silver).resolve()
    assignments_path = Path(args.assignments).resolve()
    assignments = {
        str(row["question_key"]): row for row in _read_jsonl(assignments_path)
    }
    excluded_keys: set[str] = set()
    excluded_families: set[str] = set()
    exclusion_inputs = []
    for value in args.exclude:
        path = Path(value).resolve()
        exclusion_inputs.append(artifact_identity(path))
        for row in _read_jsonl(path):
            key = str(row.get("question_key") or "")
            if key:
                excluded_keys.add(key)
                family = str(
                    row.get("family_sha256")
                    or (assignments.get(key) or {}).get("family_sha256")
                    or ""
                )
                if family:
                    excluded_families.add(family)
    selected = select_training_rows(
        _read_jsonl(silver_path), assignments,
        excluded_keys=excluded_keys,
        excluded_families=excluded_families,
        per_type=args.per_type,
        seed=args.seed,
    )
    out_dir, experiment_id = prepare_new_run_dir(
        args.output_dir,
        experiment_id=args.experiment_id,
        extra={"phase": "freeze_automatic_proofkg_training_cohort"},
    )
    cohort_path = out_dir / "cohort.question_only.jsonl"
    with cohort_path.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    qids = [str(row["question_key"]) for row in selected]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "status": "FROZEN_TRAIN_ONLY_BEFORE_PLANNER",
        "schema_version": SCHEMA_VERSION,
        "scientific_boundary": {
            "train_only": True,
            "planner_runtime_question_only": True,
            "source_gold_traces_copied": False,
            "excluded_confirmation_questions": len(excluded_keys),
            "excluded_confirmation_families": len(excluded_families),
            "must_rebuild_evidence_store_excluding_selected_families": True,
            "ppo_launch_authorized": False,
        },
        "counts": {
            "n": len(selected),
            "by_question_type": dict(Counter(row["question_type"] for row in selected)),
            "unique_qids": len(set(qids)),
            "unique_families": len({row["family_sha256"] for row in selected}),
        },
        "qid_order_sha256": hashlib.sha256("\n".join(qids).encode()).hexdigest(),
        "inputs": {
            "silver": artifact_identity(silver_path),
            "assignments": artifact_identity(assignments_path),
            "excluded_cohorts": exclusion_inputs,
        },
        "outputs": {"cohort": artifact_identity(cohort_path)},
    }
    report_path = out_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(out_dir, status=report["status"], extra=report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
