#!/usr/bin/env python
"""Prepare the identity-only SAEG-v1 training candidate pool.

This stage does not materialise graph text or SFT targets.  It records which
existing train-only assets could contribute P-only, W-only, fused P+W, and
no-graph variants after the overlap and leakage audit.  The output is a
candidate inventory, not a frozen training protocol.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.utils.logging import dump_manifest


EXPERIMENT_ID = "SAEG-V1-TRAINING-CANDIDATE-POOL"
STATUS = "PREPARED_CANDIDATE_POOL_NOT_FROZEN_NOT_TRAINED"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_candidate_rows(
    qpeg_graph: Sequence[Mapping[str, Any]],
    qpeg_replay: Sequence[Mapping[str, Any]],
    proof_audit: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in qpeg_graph:
        metadata = row.get("metadata") or {}
        source_qid = str(metadata.get("source_qid") or "")
        if not source_qid:
            raise ValueError("QPEG graph row missing source_qid")
        rows.append({
            "candidate_id": f"{row['dataset']}::{source_qid}::P_ONLY",
            "dataset": str(row["dataset"]),
            "qid": source_qid,
            "question_sha256": str(metadata["source_question_sha256"]),
            "source_mode": "P_ONLY",
            "gold_train_only": True,
            "materialized": False,
        })
    for row in qpeg_replay:
        metadata = row.get("metadata") or {}
        source_qid = str(metadata.get("source_qid") or "")
        if not source_qid:
            raise ValueError("QPEG replay row missing source_qid")
        rows.append({
            "candidate_id": f"{row['dataset']}::{source_qid}::N_REPLAY",
            "dataset": str(row["dataset"]),
            "qid": source_qid,
            "question_sha256": str(metadata["source_question_sha256"]),
            "source_mode": "N_REPLAY",
            "gold_train_only": True,
            "materialized": False,
        })
    for row in proof_audit:
        if not row.get("passage_branch_rebuildable"):
            continue
        if row.get("excluded_by_current_qpeg_v4_eval_family"):
            continue
        common = {
            "dataset": str(row["dataset"]),
            "qid": str(row["qid"]),
            "question_sha256": str(row["question_sha256"]),
            "family_sha256": str(row["family_sha256"]),
            "gold_train_only": True,
            "proof_gold_access_false": bool(row.get("proof_gold_access_false")),
            "context_title_set_exact": bool(row.get("context_title_set_exact")),
            "context_body_set_exact": bool(row.get("context_body_set_exact")),
            "materialized": False,
        }
        for mode in ("W_ONLY", "P_W_FUSED"):
            rows.append({
                "candidate_id": f"{row['dataset']}::{row['qid']}::{mode}",
                **common,
                "source_mode": mode,
            })
    candidate_ids = [str(row["candidate_id"]) for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("duplicate candidate_id")
    return sorted(rows, key=lambda row: str(row["candidate_id"]))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qpeg_silver",
        type=Path,
        default=Path("data/silver_data/qpeg_v4_schema_adaptation_n2400_seed42_v2/silver_curriculum.jsonl"),
    )
    parser.add_argument(
        "--overlap_audit",
        type=Path,
        default=Path("outputs/audits/saeg_v1_training_asset_overlap_v2/audit_rows.jsonl"),
    )
    parser.add_argument(
        "--overlap_report",
        type=Path,
        default=Path("outputs/audits/saeg_v1_training_asset_overlap_v2/report.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/audits/saeg_v1_training_candidate_pool_v1"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite candidate pool: {args.out}")
    for path in (args.qpeg_silver, args.overlap_audit, args.overlap_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    overlap_report = json.loads(args.overlap_report.read_text(encoding="utf-8"))
    if overlap_report.get("status") != "COMPLETE_DATASET_PREPARATION_AUDIT_NOT_MATERIALIZED":
        raise ValueError("overlap audit is not complete")
    integrity = overlap_report.get("integrity") or {}
    if integrity.get("cross_protocol_family_hash") != "recomputed with answer-free lexical family v1":
        raise ValueError("refusing candidate preparation without unified family hashes")

    qpeg = read_jsonl(args.qpeg_silver)
    graph = [
        row for row in qpeg
        if (row.get("metadata") or {}).get("curriculum_variant") == "qpeg"
    ]
    replay = [
        row for row in qpeg
        if (row.get("metadata") or {}).get("curriculum_variant") == "no_graph_replay"
    ]
    proof_audit = read_jsonl(args.overlap_audit)
    rows = build_candidate_rows(graph, replay, proof_audit)

    args.out.mkdir(parents=True, exist_ok=False)
    pool_path = args.out / "candidate_pool.identity_only.jsonl"
    write_jsonl(pool_path, rows)
    by_mode = Counter(str(row["source_mode"]) for row in rows)
    by_dataset_mode = Counter(
        f"{row['dataset']}::{row['source_mode']}" for row in rows
    )
    report = {
        "schema_version": "saeg-training-candidate-pool-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "counts": {
            "candidate_variants": len(rows),
            "by_mode": dict(sorted(by_mode.items())),
            "by_dataset_mode": dict(sorted(by_dataset_mode.items())),
            "unique_dataset_qid": len({(row["dataset"], row["qid"]) for row in rows}),
        },
        "integrity": {
            "unique_candidate_id": True,
            "all_materialized_false": all(row["materialized"] is False for row in rows),
            "excluded_current_qpeg_v4_eval_families_from_W_and_fused": True,
            "proof_context_semantically_identical_for_fused_candidates": all(
                row.get("context_title_set_exact") and row.get("context_body_set_exact")
                for row in rows if row["source_mode"] == "P_W_FUSED"
            ),
        },
        "inputs": {
            "qpeg_silver": {"path": str(args.qpeg_silver), "sha256": sha256_file(args.qpeg_silver)},
            "overlap_audit": {"path": str(args.overlap_audit), "sha256": sha256_file(args.overlap_audit)},
            "overlap_report": {"path": str(args.overlap_report), "sha256": sha256_file(args.overlap_report)},
        },
        "output": {"path": str(pool_path), "sha256": sha256_file(pool_path)},
        "next_decision_required": (
            "Approve the source/dataset sampling policy before materialising graph records and SFT targets. "
            "Keeping every candidate would over-represent 2Wiki because W and fused variants exist only there."
        ),
        "scientific_boundary": (
            "Identity-only candidate inventory. It contains no new graph or target materialization, "
            "does not authorize training, and is not evidence of model utility."
        ),
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra=report, status=STATUS)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
