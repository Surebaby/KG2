#!/usr/bin/env python
"""Create the family-disjoint SAEG-v1 training release from the master pool."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256


EXPERIMENT_ID = "SAEG-V1-TRAIN4860-FAMILY-DISJOINT-SEED42"
STATUS = "COMPLETE_TRAIN_ONLY_FAMILY_DISJOINT_NOT_TRAINED"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=Path(
        "data/silver_data/saeg_v1_train4862_seed42_v1/silver_train.jsonl"))
    parser.add_argument("--development", type=Path, default=Path(
        "data/derived/saeg_v1_evaluation_inputs_seed42_v1/development.answer_free.jsonl"))
    parser.add_argument("--confirmation", type=Path, default=Path(
        "data/derived/saeg_v1_evaluation_inputs_seed42_v1/confirmation.answer_free.jsonl"))
    parser.add_argument("--out", type=Path, default=Path(
        "data/silver_data/saeg_v1_train4860_family_disjoint_seed42_v2"))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite family-disjoint train release: {args.out}")
    for path in (args.master, args.development, args.confirmation):
        if not path.is_file():
            raise FileNotFoundError(path)
    master = read_jsonl(args.master)
    heldout = read_jsonl(args.development) + read_jsonl(args.confirmation)
    heldout_families = {str(row["family_sha256"]) for row in heldout}
    kept, excluded = [], []
    for row in master:
        family = family_sha256(str(row["question"]))
        if family in heldout_families:
            excluded.append({
                "qid": str(row["qid"]),
                "source_qid": str(row["source_qid"]),
                "dataset": str(row["dataset"]),
                "evidence_mode": str(row["evidence_mode"]),
                "family_sha256": family,
                "reason": "cross_dataset_family_overlap_with_fresh_development_or_confirmation",
            })
        else:
            kept.append(row)
    total_probability = sum(float(row["metadata"]["sampling_probability"]) for row in kept)
    for row in kept:
        row["metadata"] = dict(row["metadata"])
        row["metadata"]["sampling_probability_before_family_filter"] = float(
            row["metadata"]["sampling_probability"]
        )
        row["metadata"]["sampling_probability"] /= total_probability
        row["metadata"]["family_disjoint_release"] = True
    if len(kept) != 4860 or len(excluded) != 2:
        raise ValueError(f"expected 4860 kept/2 excluded, got {len(kept)}/{len(excluded)}")
    if abs(sum(float(row["metadata"]["sampling_probability"]) for row in kept) - 1.0) > 1e-9:
        raise ValueError("renormalized sampling probabilities do not sum to 1")
    if {family_sha256(row["question"]) for row in kept} & heldout_families:
        raise ValueError("held-out family remains in training release")
    args.out.mkdir(parents=True, exist_ok=False)
    output_path = args.out / "silver_train.jsonl"
    excluded_path = args.out / "excluded.identity_only.jsonl"
    write_jsonl(output_path, kept)
    write_jsonl(excluded_path, excluded)
    counts = Counter(f"{row['dataset']}::{row['evidence_mode']}" for row in kept)
    report = {
        "schema_version": "saeg-family-disjoint-training-release-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "counts": {"kept": len(kept), "excluded": len(excluded), "by_dataset_mode": dict(sorted(counts.items()))},
        "integrity": {
            "train_development_confirmation_family_overlap": 0,
            "sampling_probability_sum": sum(float(row["metadata"]["sampling_probability"]) for row in kept),
            "source_master_preserved": True,
            "answers_are_train_only": True,
        },
        "input": {"path": str(args.master), "sha256": sha256_file(args.master)},
        "outputs": {
            "silver_train": {"path": str(output_path), "sha256": sha256_file(output_path)},
            "excluded": {"path": str(excluded_path), "sha256": sha256_file(excluded_path)},
        },
        "scientific_boundary": "Only leakage filtering and probability renormalization; no target, answer, or evidence content was edited.",
    }
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra=report, status=STATUS)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
