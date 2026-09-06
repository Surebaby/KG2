#!/usr/bin/env python
"""Materialize answer-free QPEG-v3 full-sentence graphs for frozen contexts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib

from kgproweight.kg.qpeg import validate_qpeg_record
from kgproweight.kg.qpeg_sentence_selector import build_selected_sentence_record
from kgproweight.utils.logging import dump_manifest


def _sha256(path: Path) -> str:
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
        "--contexts", type=Path,
        default=Path("outputs/audits/qpeg_v1_n1350_seed42_preregistration/final.retrieval_contexts.jsonl"),
    )
    parser.add_argument(
        "--selector", type=Path,
        default=Path("outputs/training/qpeg_v3_sentence_selector_n1000x3_seed42/selector.joblib"),
    )
    parser.add_argument(
        "--gate_addendum", type=Path,
        default=Path("outputs/audits/qpeg_v3_sentence_selector_runtime_cap_correction_v1/decision_addendum.json"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/derived/qpeg_v3_sentence_selector_final900_seed42"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite materialization: {args.out}")
    args.out.mkdir(parents=True)
    gate = json.loads(args.gate_addendum.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS_TRAIN_HOLDOUT_ADVANCE_CORRECTED_RUNTIME_CAP":
        raise ValueError("QPEG-v3 train-only runtime gate did not pass")
    model = joblib.load(args.selector)
    if _sha256(args.selector) != gate["model"]["sha256"]:
        raise ValueError("selector hash differs from gate addendum")
    contexts = _read_jsonl(args.contexts)
    if len(contexts) != 900 or any(row.get("role") != "final" for row in contexts):
        raise ValueError("expected frozen final900 contexts")

    records: list[dict[str, Any]] = []
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    seen: set[str] = set()
    for context in contexts:
        key = str(context["question_key"])
        if key in seen:
            raise ValueError(f"duplicate question key: {key}")
        seen.add(key)
        record = build_selected_sentence_record(
            dataset=str(context["dataset"]),
            qid=str(context["qid"]),
            question=str(context["question"]),
            passages=list(context["passages"]),
            passages_sha256=str(context["passages_sha256"]),
            vectorizer=model["vectorizer"],
            classifier=model["classifier"],
            threshold=float(model["threshold"]),
            max_edges=int(model["max_selected_edges"]),
        )
        record["role"] = "final"
        record["family_sha256"] = context["family_sha256"]
        record["selector"]["model_sha256"] = _sha256(args.selector)
        validate_qpeg_record(record, passages=context["passages"])
        records.append(record)
        counter = counters[record["dataset"]]
        counter["n"] += 1
        counter["nonempty"] += bool(record["edges"])
        counter["edges"] += len(record["edges"])
        counter["candidate_sentences"] += int(record["candidate_count"])

    records_path = args.out / "question_graph_records.jsonl"
    _write_jsonl(records_path, records)
    datasets = {
        dataset: {
            **dict(counter),
            "nonempty_rate": counter["nonempty"] / max(1, counter["n"]),
            "mean_selected_edges": counter["edges"] / max(1, counter["n"]),
            "mean_candidate_sentences": counter["candidate_sentences"] / max(1, counter["n"]),
        }
        for dataset, counter in counters.items()
    }
    report = {
        "schema_version": "qpeg-v3-sentence-materialization-v1",
        "experiment_id": "QPEG-V3-SENTENCE-SELECTOR-FINAL900-SEED42",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_NOT_EVALUATED",
        "n": len(records),
        "datasets": datasets,
        "integrity": {
            "identity_unique": len(seen) == len(records),
            "gold_access_false": all(row["gold_access"] is False for row in records),
            "provenance_validated": True,
            "max_edges": max(len(row["edges"]) for row in records),
        },
        "inputs": {
            "contexts": {"path": str(args.contexts), "sha256": _sha256(args.contexts)},
            "selector": {"path": str(args.selector), "sha256": _sha256(args.selector)},
            "gate_addendum": {"path": str(args.gate_addendum), "sha256": _sha256(args.gate_addendum)},
        },
        "outputs": {"records": {"path": str(records_path), "sha256": _sha256(records_path)}},
        "scientific_boundary": "Answer-free graph materialization only; downstream EM/F1 remains UNKNOWN.",
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra={"phase": "qpeg_v3_sentence_materialization", **report}, status=report["status"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
