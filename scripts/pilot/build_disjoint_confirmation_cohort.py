#!/usr/bin/env python
"""Freeze a gold-free silver validation cohort disjoint from a prior cohort."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Sequence

from kgproweight.data.silver_dataset import SilverDatasetReader, SilverTrajectory
from kgproweight.data.silver_split import SplitSpec
from kgproweight.utils.logging import dump_manifest


BUILDER_VERSION = "disjoint-confirmation-cohort-1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_qids(path: Path) -> List[str]:
    qids: List[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            qids.append(str(row.get("qid") or row.get("id") or ""))
    if not qids or any(not qid for qid in qids) or len(qids) != len(set(qids)):
        raise ValueError("excluded cohort requires unique non-empty qids")
    return qids


def select_disjoint(
    trajectories: Sequence[SilverTrajectory],
    excluded_qids: Iterable[str],
    *,
    n: int,
    seed: int,
) -> List[SilverTrajectory]:
    excluded = set(excluded_qids)
    candidates = sorted(
        (item for item in trajectories if item.qid not in excluded),
        key=lambda item: (item.qid, item.question),
    )
    if n <= 0:
        raise ValueError("n must be positive")
    if len(candidates) < n:
        raise ValueError(f"only {len(candidates)} disjoint candidates remain for n={n}")
    return random.Random(seed).sample(candidates, n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--exclude", required=True)
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    silver_path = Path(args.silver).resolve()
    exclude_path = Path(args.exclude).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    run_dir = Path(args.run_dir).resolve()
    for path in (output_path, report_path, run_dir):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing path: {path}")

    try:
        excluded_qids = _read_qids(exclude_path)
        reader = SilverDatasetReader(
            silver_path,
            split=args.split,
            split_spec=SplitSpec(
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.split_seed,
            ),
        )
        accepted = reader.accepted()
        selected = select_disjoint(
            accepted, excluded_qids, n=args.n, seed=args.seed
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    selected_qids = [item.qid for item in selected]
    overlap = sorted(set(selected_qids).intersection(excluded_qids))
    if overlap:
        raise SystemExit(f"internal error: confirmation overlap detected: {overlap}")
    output_rows: List[Dict[str, Any]] = [
        {"qid": item.qid, "question": item.question, "dataset": item.dataset}
        for item in selected
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as fh:
        for row in output_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    qid_sha256 = hashlib.sha256("\n".join(selected_qids).encode()).hexdigest()
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "status": "FROZEN_NOT_EVALUATED",
        "builder_version": BUILDER_VERSION,
        "protocol": {
            "split": args.split,
            "n": args.n,
            "selection_seed": args.seed,
            "split_seed": args.split_seed,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "gold_fields_written": False,
            "selection_order": "seeded sample from qid-sorted remaining accepted fold",
        },
        "inputs": {
            "silver": str(silver_path),
            "silver_sha256": _sha256(silver_path),
            "silver_read_only": True,
            "excluded_cohort": str(exclude_path),
            "excluded_cohort_sha256": _sha256(exclude_path),
            "excluded_n": len(excluded_qids),
        },
        "fold_counts": {
            "accepted": len(accepted),
            "remaining_after_exclusion": len(
                [item for item in accepted if item.qid not in set(excluded_qids)]
            ),
            "selected": len(selected),
            "overlap": len(overlap),
        },
        "output": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "qid_sha256": qid_sha256,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        run_dir,
        extra={
            "experiment_id": args.experiment_id,
            "phase": "freeze_disjoint_confirmation_cohort",
            "builder_version": BUILDER_VERSION,
            "protocol": report["protocol"],
            "inputs": report["inputs"],
            "fold_counts": report["fold_counts"],
            "output": report["output"],
        },
    )
    print(json.dumps(report["fold_counts"] | report["output"], indent=2))


if __name__ == "__main__":
    main()
