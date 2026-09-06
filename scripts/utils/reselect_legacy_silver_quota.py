#!/usr/bin/env python
"""Create a new accepted subset from an immutable repaired-silver candidate file."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from kgproweight.training.phase1_distill import StratifiedSilverFilter
from kgproweight.utils.logging import dump_manifest
from scripts.utils.repair_legacy_silver_v2 import _finalize_per_dataset


PROTOCOL_VERSION = "legacy_repaired_v2.quota_reselection_1"


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--medium_quota", type=float, default=0.35)
    parser.add_argument("--sparse_quota", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidates).resolve()
    output_path = Path(args.output).resolve()
    report_path = output_path.with_name(f"{output_path.stem}.report.json")
    run_dir = output_path.with_name(f"{output_path.stem}_run")
    for path in (output_path, report_path, run_dir):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing path: {path}")
    if args.medium_quota != 0.35:
        raise SystemExit("approved quota-70 protocol keeps medium_quota fixed at 0.35")
    if args.sparse_quota != 0.70:
        raise SystemExit("approved quota-70 protocol requires sparse_quota=0.70")

    candidate_md5 = _md5(candidate_path)
    accept_filter = StratifiedSilverFilter(
        medium_quota=args.medium_quota,
        sparse_quota=args.sparse_quota,
    )
    selection = _finalize_per_dataset(
        candidate_path,
        output_path,
        accept_filter,
        args.seed,
        selection_metadata={
            "selection_protocol_version": PROTOCOL_VERSION,
            "selection_seed": args.seed,
            "selection_medium_quota": args.medium_quota,
            "selection_sparse_quota": args.sparse_quota,
            "selection_source_candidate_md5": candidate_md5,
        },
    )
    output_md5 = _md5(output_path)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "status": "DERIVED_SELECTION_NOT_TRAINED",
        "protocol_version": PROTOCOL_VERSION,
        "single_variable_change": "sparse_quota: 0.25 -> 0.70",
        "labels_or_kg_modified": False,
        "source": {
            "candidate_path": str(candidate_path),
            "candidate_md5": candidate_md5,
        },
        "config": {
            "seed": args.seed,
            "medium_quota": args.medium_quota,
            "sparse_quota": args.sparse_quota,
            "quota_scope": "per_dataset_posthoc",
        },
        "selection": selection,
        "output": {
            "path": str(output_path),
            "md5": output_md5,
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(run_dir, extra={
        "experiment_id": args.experiment_id,
        "phase": "legacy_silver_quota_reselection",
        "protocol_version": PROTOCOL_VERSION,
        "candidate": str(candidate_path),
        "candidate_md5": candidate_md5,
        "output": str(output_path),
        "output_md5": output_md5,
        "total": selection["total"],
        "accepted": selection["accepted"],
        "medium_quota": args.medium_quota,
        "sparse_quota": args.sparse_quota,
        "seed": args.seed,
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
