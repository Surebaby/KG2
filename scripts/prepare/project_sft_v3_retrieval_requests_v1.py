#!/usr/bin/env python3
"""Append-only strict reader-interface projection of frozen SFT-v3 candidates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.prepare import freeze_sft_v3_candidates_v1 as pool
from scripts.prepare import freeze_sft_v3_protected_ledger_v1 as ledger

VERSION = "sft-v3-retrieval-request-projection-v1"
FIELDS = frozenset({"dataset", "qid", "question", "question_key", "question_sha256",
                    "family_sha256", "family_version", "role", "gold_access", "split",
                    "selection_rank", "schema_version"})


def project(row):
    ident = ledger.identity(row)
    if any(row[field] != ident[field] for field in ident):
        raise ValueError("projection identity differs from current question/family")
    n = row.get("within_split_dataset_rank")
    if type(n) is not int or n < 1 or row.get("gold_access") is not False or row.get("split") not in pool.SPLITS:
        raise ValueError("projection requires explicit positive ordinal rank, false Gold access, and known split")
    projected = {field: row[field] for field in FIELDS if field != "selection_rank"}
    projected["selection_rank"] = n
    if set(projected) != FIELDS:
        raise ValueError("projection field contract differs")
    return projected


def freeze_projection(parent_dir: Path, output_dir: Path | None = None):
    output_dir = output_dir or parent_dir / "retrieval_projection_v1"
    parent_manifest = parent_dir / "manifest.json"
    authority = json.loads(parent_manifest.read_text())
    if (authority.get("schema_version") != pool.VERSION or authority.get("complete") is not True
            or authority.get("status") != "COMPLETE_CANDIDATE_IDENTITIES_AND_CHECKER_LABELS_FROZEN_NOT_SFT_DATA"
            or not authority.get("gates") or not all(authority["gates"].values())
            or (parent_dir / "FAILED.json").exists()):
        raise ValueError("candidate parent is incomplete/failed")
    source = parent_dir / "candidates.question_only.jsonl"
    ledger._validate_bound(source, authority["outputs"][source.name]["sha256"])
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {"schema_version": VERSION, "experiment_id": "SFT-V3-RETRIEVAL-REQUEST-PROJECTION-20260906-V1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "parent_manifest": ledger.bind(parent_manifest), "parent_candidates": ledger.bind(source),
                "code": ledger.bind(Path(__file__)), "identity_code": ledger.bind(Path(ledger.__file__)),
                "fields": sorted(FIELDS),
                "selection_rank_semantics": "positive integer within frozen dataset/split; copied from parent within_split_dataset_rank; NOT parent's SHA selection_rank",
                "ordering": "preserve exact parent row order (train then validation, within split dataset round-robin)",
                "metadata": "provenance/seed ranking/consumption ordinal remain in unchanged parent only",
                "gold_value_access": False, "parent_modified": False, "retrieval_started": False}
    ledger.write_json(output_dir / "protocol.json", protocol)
    try:
        rows = [project(row) for _, row, _ in ledger.read_rows(source)]
        if len(rows) != authority["candidate_questions"]:
            raise ValueError("projection row count differs from parent")
        if len({r["question_key"] for r in rows}) != len(rows):
            raise ValueError("duplicate question identity in projection")
        pool.write_rows(output_dir / "requests.question_only.jsonl", rows)
        for binding in (protocol["parent_manifest"], protocol["parent_candidates"], protocol["code"], protocol["identity_code"]):
            ledger._validate_bound(Path(binding["path"]), binding["sha256"])
        report = {"schema_version": VERSION, "experiment_id": protocol["experiment_id"],
                  "status": "COMPLETE_STRICT_12_FIELD_PROJECTION_NOT_RETRIEVED", "complete": True,
                  "rows": len(rows), "all_rows_have_exactly_12_allowed_fields": all(set(r) == FIELDS for r in rows),
                  "all_selection_ranks_positive_integers": all(type(r["selection_rank"]) is int and r["selection_rank"] > 0 for r in rows),
                  "parent_question_order_and_identity_preserved": True, "gold_value_access": False,
                  "selection_rank_semantics": protocol["selection_rank_semantics"]}
        ledger.write_json(output_dir / "report.json", report)
        ledger.write_json(output_dir / "manifest.json", {**report, "outputs": {name: ledger.bind(output_dir / name)
                           for name in ("requests.question_only.jsonl", "protocol.json", "report.json")}})
        return report
    except BaseException as exc:
        ledger.write_json(output_dir / "FAILED.json", {"status": "FAILED_NOT_RETRIEVED", "type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, default=pool.DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(freeze_projection(args.parent_dir), ensure_ascii=False, indent=2))
