#!/usr/bin/env python
"""Correct QPEG-v3 holdout gates using the frozen runtime top-k semantics.

The original training report counted every sentence above the threshold even
though the frozen protocol and saved selector both cap runtime output at four
sentences per question.  This append-only audit does not refit the model or
change its threshold; it evaluates the exact frozen runtime selection.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import joblib
from sklearn.metrics import precision_score, recall_score

from kgproweight.utils.logging import dump_manifest


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _runtime_metrics(
    rows: list[Mapping[str, Any]], probabilities: list[float], threshold: float, max_edges: int
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[tuple[float, Mapping[str, Any]]]] = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        grouped[(str(row["dataset"]), str(row["question_key"]))].append((probability, row))
    selected: list[tuple[float, Mapping[str, Any]]] = []
    qid_selected: dict[tuple[str, str], bool] = {}
    for key, values in grouped.items():
        values.sort(key=lambda value: (-value[0], int(value[1]["candidate_index"])))
        current = [value for value in values if value[0] >= threshold][:max_edges]
        selected.extend(current)
        qid_selected[key] = bool(current)

    def aggregate(dataset: str | None = None) -> dict[str, Any]:
        current = [
            row for _, row in selected if dataset is None or str(row["dataset"]) == dataset
        ]
        qids = [key for key in grouped if dataset is None or key[0] == dataset]
        labels = [int(row["label"]) for row in current]
        return {
            "selected_sentences": len(current),
            "positive_selected_sentences": sum(labels),
            "sentence_precision": sum(labels) / max(1, len(labels)),
            "qid_selected_rate": sum(qid_selected[key] for key in qids) / max(1, len(qids)),
            "qids": len(qids),
        }

    return {
        **aggregate(),
        "per_dataset": {dataset: aggregate(dataset) for dataset in DATASETS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path,
        default=Path("outputs/training/qpeg_v3_sentence_selector_n1000x3_seed42"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/audits/qpeg_v3_sentence_selector_runtime_cap_correction_v1"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite audit: {args.out}")
    args.out.mkdir(parents=True)

    original_path = args.run / "report.json"
    examples_path = args.run / "sentence_examples.jsonl"
    model_path = args.run / "selector.joblib"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    protocol_path = Path(original["protocol"]["path"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    model = joblib.load(model_path)
    if _sha256(model_path) != original["outputs"]["model"]["sha256"]:
        raise ValueError("selector model hash mismatch")
    if float(model["threshold"]) != float(original["selected_threshold_from_dev"]["threshold"]):
        raise ValueError("frozen threshold mismatch")
    if int(model["max_selected_edges"]) != int(protocol["representation"]["max_selected_edges"]):
        raise ValueError("runtime cap differs from protocol")

    rows = [row for row in _read_jsonl(examples_path) if row["split"] == "holdout"]
    probabilities = [
        float(value) for value in model["classifier"].predict_proba(
            model["vectorizer"].transform([row["features"] for row in rows])
        )[:, 1]
    ]
    metrics = _runtime_metrics(
        rows, probabilities, float(model["threshold"]), int(model["max_selected_edges"])
    )
    gates = {
        "roc_auc_ge_0_65": original["metrics"]["holdout"]["roc_auc"] >= protocol["holdout_gates"]["roc_auc_min"],
        "runtime_selected_sentence_precision_ge_0_55": metrics["sentence_precision"] >= protocol["holdout_gates"]["selected_sentence_precision_min"],
        "qid_selected_rate_ge_0_65": metrics["qid_selected_rate"] >= protocol["holdout_gates"]["qid_selected_rate_min"],
        "each_dataset_precision_ge_0_45": all(
            value["sentence_precision"] >= protocol["holdout_gates"]["each_dataset_precision_min"]
            for value in metrics["per_dataset"].values()
        ),
        "each_dataset_qid_selected_rate_ge_0_55": all(
            value["qid_selected_rate"] >= protocol["holdout_gates"]["each_dataset_qid_selected_rate_min"]
            for value in metrics["per_dataset"].values()
        ),
    }
    report = {
        "schema_version": "qpeg-v3-runtime-cap-gate-correction-v1",
        "experiment_id": "QPEG-V3-SENTENCE-SELECTOR-RUNTIME-CAP-CORRECTION-V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_TRAIN_HOLDOUT_ADVANCE_CORRECTED_RUNTIME_CAP" if all(gates.values()) else "FAIL_STOP_SELECTOR_CORRECTED_RUNTIME_CAP",
        "original_report": {"path": str(original_path), "sha256": _sha256(original_path), "status": original["status"]},
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "model": {"path": str(model_path), "sha256": _sha256(model_path)},
        "examples": {"path": str(examples_path), "sha256": _sha256(examples_path)},
        "frozen_selection": {
            "threshold": model["threshold"],
            "max_selected_edges": model["max_selected_edges"],
            "model_refit": False,
            "threshold_changed": False,
        },
        "holdout_roc_auc": original["metrics"]["holdout"]["roc_auc"],
        "runtime_holdout_metrics": metrics,
        "gates": {"checks": gates, "all_pass": all(gates.values())},
        "correction": {
            "bug": "original gate aggregated all above-threshold sentences without applying frozen runtime top-4 cap",
            "original_uncapped_precision": original["metrics"]["holdout"]["sentence_precision"],
            "correct_runtime_capped_precision": metrics["sentence_precision"],
            "original_artifacts_overwritten": False,
        },
        "scientific_boundary": (
            "This is a train-only implementation correction. It permits proposing, not running, a new final "
            "A/B evaluation. It does not establish downstream EM/F1 utility."
        ),
    }
    path = args.out / "decision_addendum.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra={"phase": "qpeg_v3_runtime_cap_gate_correction", **report}, status=report["status"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
