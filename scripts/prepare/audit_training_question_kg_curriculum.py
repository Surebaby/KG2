#!/usr/bin/env python
"""CPU-only fail-fast audit for a silver curriculum and question-KG JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from kgproweight.data.parsers import ParsedStep
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.kg.training_question_kg import (
    apply_training_question_kg,
    read_question_kg_records,
)
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--question_kg_records", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir, experiment_id = prepare_new_run_dir(args.output_dir)
    reader = SilverDatasetReader(args.silver)
    records = read_question_kg_records(args.question_kg_records)
    stats = apply_training_question_kg(reader.accepted(), records, min_coverage=1.0)

    keys = [(traj.dataset.lower(), traj.qid) for traj in reader.accepted()]
    proof_rows = [
        traj for traj in reader.accepted()
        if bool(traj.metadata.get("gold_derived", False))
    ]
    errors = []
    proof_citing_steps = 0
    for traj in proof_rows:
        if not traj.metadata.get("train_only") or traj.metadata.get("evaluation_eligible") is not False:
            errors.append(f"train-only boundary missing: {traj.dataset}::{traj.qid}")
        if len(traj.steps) < 3:
            errors.append(f"fewer than 3 steps: {traj.dataset}::{traj.qid}")
        if not traj.kg_subgraph:
            errors.append(f"empty Proof-KG: {traj.dataset}::{traj.qid}")
        for source_step in traj.steps[:-1]:
            parsed = ParsedStep.from_text(
                source_step.index, source_step.text, known_kg=traj.kg_subgraph
            )
            proof_citing_steps += 1
            if not parsed.knowledge_used_valid:
                errors.append(f"invalid citation contract: {traj.dataset}::{traj.qid}")
            if parsed.cited_triples != source_step.cited_triples:
                errors.append(f"citation parse mismatch: {traj.dataset}::{traj.qid}")
    duplicate_keys = len(keys) - len(set(keys))
    if duplicate_keys:
        errors.append(f"duplicate trajectory keys: {duplicate_keys}")
    if len(records) != len(reader.accepted()):
        errors.append(
            f"record cardinality mismatch: records={len(records)} accepted={len(reader.accepted())}"
        )

    report = {
        "experiment_id": experiment_id,
        "status": "PASS" if not errors else "FAIL_STOP",
        "counts": {
            "trajectories": len(reader.trajectories),
            "accepted": len(reader.accepted()),
            "records": len(records),
            "by_dataset": dict(Counter(t.dataset for t in reader.accepted())),
            "proof_rows": len(proof_rows),
            "proof_citing_steps_checked": proof_citing_steps,
            "duplicate_keys": duplicate_keys,
        },
        "question_kg_join": stats.to_dict(),
        "errors": errors[:100],
        "inputs": {
            "silver": artifact_identity(args.silver),
            "question_kg_records": artifact_identity(args.question_kg_records),
        },
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    dump_manifest(
        out_dir,
        status="COMPLETE" if not errors else "FAILED",
        extra={
            "experiment_id": experiment_id,
            "phase": "data_audit",
            "report": artifact_identity(report_path),
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
