#!/usr/bin/env python
"""Freeze a consumed, identity-only 4x3 engineering-smoke cohort for v8.

The source is an already consumed QPEG-v4 development artifact which was part
of the historical registry excluded when the fresh v8 development/prospective
cohorts were selected.  Only dataset/qid/question are read and emitted.  The
sealed prospective file is never opened or hashed by this command.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "dynamic-decomposition-v8-consumed-smoke-cohort-1"
MANIFEST_SCHEMA_VERSION = "dynamic-decomposition-v8-consumed-smoke-manifest-1"
EXPERIMENT_ID = "SUBQUESTION-DECOMPOSITION-V8-CONSUMED-SMOKE4X3-SEED20260904-V1"
STATUS = "COMPLETE_FROZEN_CONSUMED_IDENTITY_ONLY_SMOKE4X3"
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
N_PER_DATASET = 4
FIELDS = ("dataset", "qid", "question")
SOURCE_RELATIVE = Path(
    "outputs/audits/qpeg_v4_schema_adaptation_protocol_v1/"
    "development.question_only.jsonl"
)
SOURCE_SHA256 = "b0047f8a3ea304652f314c3287d8e3be7200304212bfb085a03fa02f16696358"
CAPACITY_INVENTORY_RELATIVE = Path(
    "outputs/audits/subquestion_decomposition_v8_cohort_capacity_audit_v1/inventory.json"
)
FRESH_FREEZE_REPORT_RELATIVE = Path(
    "outputs/audits/"
    "subquestion_decomposition_v8_cohort_freeze_dev30_prospective300_"
    "seed20260904_v1/report.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/audits/"
    "subquestion_decomposition_v8_consumed_smoke4x3_seed20260904_v1"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if tuple(row) != FIELDS or set(row) != set(FIELDS):
                raise ValueError("smoke row violates exact identity allowlist")
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def select_consumed_smoke_rows(source: Path) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"source line {line_number} is not an object")
            # Exact projection: no answer/support/decomposition field is read.
            dataset = row.get("dataset")
            qid = row.get("qid")
            question = row.get("question")
            if dataset not in DATASETS or counts[str(dataset)] >= N_PER_DATASET:
                continue
            if not all(isinstance(value, str) and value and value == value.strip() for value in (qid, question)):
                raise ValueError(f"invalid identity at source line {line_number}")
            identity = f"{dataset}::{qid}"
            if identity in seen:
                raise ValueError(f"duplicate source identity: {identity}")
            seen.add(identity)
            counts[str(dataset)] += 1
            selected.append(
                {"dataset": str(dataset), "qid": str(qid), "question": str(question)}
            )
    expected = {dataset: N_PER_DATASET for dataset in DATASETS}
    if dict(counts) != expected:
        raise ValueError(f"smoke cohort count mismatch: {dict(counts)}")
    return selected


def freeze(*, project_root: Path, output_dir: Path) -> dict[str, Any]:
    output = output_dir if output_dir.is_absolute() else project_root / output_dir
    if output.exists():
        raise FileExistsError(f"refusing to overwrite append-only smoke cohort: {output}")
    source = project_root / SOURCE_RELATIVE
    inventory_path = project_root / CAPACITY_INVENTORY_RELATIVE
    freeze_report_path = project_root / FRESH_FREEZE_REPORT_RELATIVE
    if _sha256_file(source) != SOURCE_SHA256:
        raise ValueError("consumed smoke source SHA mismatch")
    inventory = _load_json(inventory_path)
    registries = inventory.get("historical_evaluation_protocol_registries")
    if not isinstance(registries, list):
        raise ValueError("capacity inventory lacks historical registry list")
    matching = [row for row in registries if row.get("path") == SOURCE_RELATIVE.as_posix()]
    if len(matching) != 1 or matching[0].get("sha256") != SOURCE_SHA256:
        raise ValueError("smoke source is not locked as a consumed historical registry")
    fresh_report = _load_json(freeze_report_path)
    checks = fresh_report.get("checks") or {}
    if (
        fresh_report.get("status")
        != "COMPLETE_FROZEN_SCOPE_A_DEV30_PROSPECTIVE300_NO_RESERVE"
        or checks.get("historical_registry_qid_overlap") != 0
        or checks.get("historical_registry_family_overlap") != 0
    ):
        raise ValueError("fresh v8 freeze does not prove historical-registry isolation")

    rows = select_consumed_smoke_rows(source)
    output.mkdir(parents=True, exist_ok=False)
    cohort_path = output / "smoke.identity_only.jsonl"
    _write_jsonl(cohort_path, rows)
    now = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": STATUS,
        "created_at_utc": now,
        "scope": "CONSUMED_ENGINEERING_SMOKE_ONLY_NOT_FRESH_DEVELOPMENT",
        "gold_access": False,
        "source_may_contain_nonidentity_metadata": True,
        "source_fields_accessed": list(FIELDS),
        "output_fields_exact": list(FIELDS),
        "row_count": len(rows),
        "per_dataset_counts": dict(Counter(row["dataset"] for row in rows)),
        "fresh_development_or_prospective_overlap": 0,
        "overlap_evidence": (
            "source is a locked historical registry and the fresh Scope-A freeze "
            "reports zero qid/family overlap with all locked historical registries"
        ),
        "prospective_opened_or_hashed": False,
        "scientific_boundary": (
            "runtime/interface engineering only; not a fresh mechanism or outcome result"
        ),
        "inputs": {
            "consumed_source": {
                "path": SOURCE_RELATIVE.as_posix(),
                "sha256": SOURCE_SHA256,
            },
            "capacity_inventory": {
                "path": CAPACITY_INVENTORY_RELATIVE.as_posix(),
                "sha256": _sha256_file(inventory_path),
            },
            "fresh_freeze_report": {
                "path": FRESH_FREEZE_REPORT_RELATIVE.as_posix(),
                "sha256": _sha256_file(freeze_report_path),
            },
        },
    }
    _write_json(output / "report.json", report)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": STATUS,
        "gold_access": False,
        "prospective_opened_or_hashed": False,
        "outputs": [
            {
                "path": "smoke.identity_only.jsonl",
                "sha256": _sha256_file(cohort_path),
                "size_bytes": cohort_path.stat().st_size,
            },
            {
                "path": "report.json",
                "sha256": _sha256_file(output / "report.json"),
                "size_bytes": (output / "report.json").stat().st_size,
            },
        ],
        "implementation": {
            "path": Path(__file__).resolve().relative_to(project_root).as_posix(),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
    }
    _write_json(output / "manifest.json", manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = freeze(project_root=PROJECT_ROOT, output_dir=args.output_dir)
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": report["status"]}))


if __name__ == "__main__":
    main()
