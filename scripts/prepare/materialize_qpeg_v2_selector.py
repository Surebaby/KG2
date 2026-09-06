#!/usr/bin/env python
"""Apply the frozen train-only selector to QPEG-v1.1 candidates."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib

from kgproweight.kg.qpeg import _sha_json, validate_qpeg_record
from kgproweight.kg.qpeg_selector import (
    QPEG_SELECTOR_EXTRACTOR_VERSION,
    QPEG_SELECTOR_FEATURE_VERSION,
    select_edges,
)
from kgproweight.utils.logging import dump_manifest


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path,
        default=Path("data/derived/qpeg_v1_1_precision_n1350_seed42/question_graph_records.jsonl"),
    )
    parser.add_argument(
        "--selector_run", type=Path,
        default=Path("outputs/training/qpeg_v2_selector_n1000x3_seed42"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/derived/qpeg_v2_selector_n1350_seed42"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite QPEG-v2 materialization: {args.out}")
    args.out.mkdir(parents=True)
    training_report = json.loads((args.selector_run / "report.json").read_text(encoding="utf-8"))
    if training_report.get("status") != "PASS_TRAIN_HOLDOUT_ADVANCE":
        raise ValueError("selector training run did not pass holdout gates")
    model_path = args.selector_run / "selector.joblib"
    if _sha(model_path) != training_report["outputs"]["model"]["sha256"]:
        raise ValueError("selector model hash differs from training report")
    bundle = joblib.load(model_path)
    if bundle.get("feature_version") != QPEG_SELECTOR_FEATURE_VERSION:
        raise ValueError("selector feature version mismatch")
    threshold = float(bundle["threshold"])
    max_edges = int(bundle["max_selected_edges"])

    records: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for source in _read_jsonl(args.candidates):
        validate_qpeg_record(source)
        selected, scores = select_edges(
            record=source,
            vectorizer=bundle["vectorizer"],
            classifier=bundle["classifier"],
            threshold=threshold,
            max_edges=max_edges,
        )
        record = dict(source)
        record["extractor_version"] = QPEG_SELECTOR_EXTRACTOR_VERSION
        record["max_edges"] = max_edges
        record["edges"] = selected
        record["kg_subgraph"] = [
            [edge["head_surface"], edge["relation_surface"], edge["tail_surface"]]
            for edge in selected
        ]
        record["provenance_complete"] = bool(selected)
        record["build_status"] = "nonempty" if selected else "empty"
        record["selector"] = {
            "feature_version": QPEG_SELECTOR_FEATURE_VERSION,
            "model_sha256": _sha(model_path),
            "threshold": threshold,
            "max_selected_edges": max_edges,
            "selected_scores": scores,
            "source_qpeg_sha256": source["qpeg_sha256"],
            "source_edge_count": len(source["edges"]),
        }
        record["qpeg_sha256"] = _sha_json({
            "question_key": record["question_key"],
            "question_sha256": record["question_sha256"],
            "passages_sha256": record["passages_sha256"],
            "edges": selected,
        })
        validate_qpeg_record(record)
        records.append(record)
        dataset = str(record["dataset"])
        role = str(record.get("role") or "unknown")
        counters[f"n::{dataset}::{role}"] += 1
        counters[f"nonempty::{dataset}::{role}"] += bool(selected)
        counters[f"edges::{dataset}::{role}"] += len(selected)
        details.append({
            "question_key": record["question_key"],
            "dataset": dataset,
            "role": role,
            "source_edge_count": len(source["edges"]),
            "selected_edge_count": len(selected),
            "selected_scores": scores,
            "routing": "qpeg_v2" if selected else "no_graph",
        })
    _write_jsonl(args.out / "question_graph_records.jsonl", records)
    _write_jsonl(args.out / "selection_details.jsonl", details)
    metrics: dict[str, Any] = {}
    for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
        metrics[dataset] = {}
        for role in ("pilot", "confirmation", "final"):
            n = counters[f"n::{dataset}::{role}"]
            nonempty = counters[f"nonempty::{dataset}::{role}"]
            metrics[dataset][role] = {
                "n": n,
                "nonempty": nonempty,
                "nonempty_rate": nonempty / max(1, n),
                "mean_selected_edges": counters[f"edges::{dataset}::{role}"] / max(1, n),
            }
    report = {
        "schema_version": "qpeg-v2-materialization-report-v1",
        "experiment_id": "QPEG-V2-SELECTOR-N1350-SEED42-MATERIALIZATION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_NOT_EVALUATED",
        "n": len(records),
        "selector": {
            "model": str(model_path),
            "model_sha256": _sha(model_path),
            "threshold": threshold,
            "max_selected_edges": max_edges,
            "training_report_sha256": _sha(args.selector_run / "report.json"),
        },
        "metrics": metrics,
        "gates": {
            "unique_question_key": len({row["question_key"] for row in records}) == len(records),
            "gold_access_false": all(row["gold_access"] is False for row in records),
            "max_edges_le_6": all(len(row["edges"]) <= 6 for row in records),
            "provenance_consistent": all(bool(row["edges"]) == bool(row["provenance_complete"]) for row in records),
        },
        "inputs": {"candidates": {"path": str(args.candidates), "sha256": _sha(args.candidates)}},
        "scientific_boundary": "Selector output materialized; downstream answer utility not yet evaluated.",
    }
    if not all(report["gates"].values()):
        raise RuntimeError(f"QPEG-v2 materialization integrity gate failed: {report['gates']}")
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra={"phase": "qpeg_v2_materialization", **report}, status=report["status"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
