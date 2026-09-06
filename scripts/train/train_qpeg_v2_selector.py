#!/usr/bin/env python
"""Fit the frozen train-only QPEG-v2 edge selector and evaluate its train holdout."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from kgproweight.kg.qpeg import build_qpeg_record
from kgproweight.kg.qpeg_selector import QPEG_SELECTOR_FEATURE_VERSION, edge_features
from kgproweight.utils.logging import dump_manifest
from scripts.diagnose.audit_qpeg_v2_train_supervision import (
    DATASETS,
    _context_and_supports,
    _hash_sample,
    _is_positive,
    _sha256,
)
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _split(question: str) -> str:
    bucket = int(family_sha256(question)[:8], 16) % 10
    if bucket <= 6:
        return "train"
    if bucket == 7:
        return "dev"
    return "holdout"


def _threshold_metrics(
    examples: list[dict[str, Any]], probabilities: list[float], threshold: float
) -> dict[str, Any]:
    labels = [int(row["label"]) for row in examples]
    predictions = [int(value >= threshold) for value in probabilities]
    selected_by_qid: dict[str, int] = Counter()
    qids_by_dataset: dict[str, set[str]] = defaultdict(set)
    selected_by_dataset: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row, prediction in zip(examples, predictions):
        qkey = str(row["question_key"])
        dataset = str(row["dataset"])
        qids_by_dataset[dataset].add(qkey)
        selected_by_qid[qkey] += prediction
        selected_by_dataset[dataset].append((int(row["label"]), prediction))
    per_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        pairs = selected_by_dataset[dataset]
        ds_labels = [left for left, _ in pairs]
        ds_predictions = [right for _, right in pairs]
        qids = qids_by_dataset[dataset]
        per_dataset[dataset] = {
            "edge_precision": precision_score(ds_labels, ds_predictions, zero_division=0),
            "edge_recall": recall_score(ds_labels, ds_predictions, zero_division=0),
            "qid_selected_rate": sum(selected_by_qid[qid] > 0 for qid in qids) / max(1, len(qids)),
            "selected_edges": sum(ds_predictions),
            "qids": len(qids),
        }
    return {
        "threshold": threshold,
        "edge_precision": precision_score(labels, predictions, zero_division=0),
        "edge_recall": recall_score(labels, predictions, zero_division=0),
        "edge_f1": f1_score(labels, predictions, zero_division=0),
        "selected_edges": sum(predictions),
        "qid_selected_rate": sum(value > 0 for value in selected_by_qid.values()) / max(1, len(selected_by_qid)),
        "per_dataset": per_dataset,
    }


def _choose_threshold(examples: list[dict[str, Any]], probabilities: list[float]) -> dict[str, Any]:
    candidates = [
        _threshold_metrics(examples, probabilities, value / 100)
        for value in range(5, 96)
    ]
    feasible = [
        row for row in candidates
        if row["edge_precision"] >= 0.55
        and row["qid_selected_rate"] >= 0.70
        and all(value["qid_selected_rate"] >= 0.60 for value in row["per_dataset"].values())
    ]
    if not feasible:
        raise RuntimeError("no dev threshold satisfies the frozen precision/coverage constraints")
    return max(feasible, key=lambda row: (row["edge_f1"], row["edge_precision"], row["threshold"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("outputs/audits/qpeg_v2_selector_protocol_v1_run2/protocol.json"),
    )
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/training/qpeg_v2_selector_n1000x3_seed42"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite selector run: {args.out}")
    args.out.mkdir(parents=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_SELECTOR_DATASET_OR_FIT":
        raise ValueError("unexpected selector protocol status")

    examples: list[dict[str, Any]] = []
    family_sets: dict[str, set[str]] = defaultdict(set)
    qid_sets: dict[str, set[str]] = defaultdict(set)
    for dataset in DATASETS:
        rows = _hash_sample(args.data_root / dataset / "train.jsonl", dataset, 1000)
        for row in rows:
            question = str(row["question"])
            split = _split(question)
            family = family_sha256(question)
            question_key = f"{dataset}::{row['id']}"
            family_sets[split].add(family)
            qid_sets[split].add(question_key)
            passages, support = _context_and_supports(dataset, row)
            if not passages:
                continue
            graph = build_qpeg_record(
                dataset=dataset, qid=str(row["id"]), question=question, passages=passages
            )
            for edge_index, edge in enumerate(graph["edges"]):
                examples.append({
                    "question_key": question_key,
                    "dataset": dataset,
                    "split": split,
                    "family_sha256": family,
                    "edge_index": edge_index,
                    "label": int(_is_positive(dataset, edge, passages, support)),
                    "features": edge_features(dataset=dataset, question=question, edge=edge),
                })
    if any(family_sets[left] & family_sets[right] for left, right in (("train", "dev"), ("train", "holdout"), ("dev", "holdout"))):
        raise RuntimeError("family leakage across selector splits")

    by_split = {split: [row for row in examples if row["split"] == split] for split in ("train", "dev", "holdout")}
    vectorizer = DictVectorizer(sparse=True)
    train_x = vectorizer.fit_transform([row["features"] for row in by_split["train"]])
    train_y = [int(row["label"]) for row in by_split["train"]]
    classifier = LogisticRegression(
        C=1.0, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=42
    )
    classifier.fit(train_x, train_y)

    probabilities: dict[str, list[float]] = {}
    auc: dict[str, float] = {}
    for split, rows in by_split.items():
        values = classifier.predict_proba(vectorizer.transform([row["features"] for row in rows]))[:, 1]
        probabilities[split] = [float(value) for value in values]
        auc[split] = float(roc_auc_score([row["label"] for row in rows], values))
    selected_dev = _choose_threshold(by_split["dev"], probabilities["dev"])
    threshold = float(selected_dev["threshold"])
    split_metrics = {
        split: {"roc_auc": auc[split], **_threshold_metrics(rows, probabilities[split], threshold)}
        for split, rows in by_split.items()
    }
    holdout = split_metrics["holdout"]
    gates = {
        "roc_auc_ge_0_65": holdout["roc_auc"] >= 0.65,
        "selected_edge_precision_ge_0_55": holdout["edge_precision"] >= 0.55,
        "qid_selected_rate_ge_0_65": holdout["qid_selected_rate"] >= 0.65,
        "each_dataset_precision_ge_0_45": all(value["edge_precision"] >= 0.45 for value in holdout["per_dataset"].values()),
        "each_dataset_qid_selected_rate_ge_0_55": all(value["qid_selected_rate"] >= 0.55 for value in holdout["per_dataset"].values()),
    }

    model_path = args.out / "selector.joblib"
    joblib.dump({
        "feature_version": QPEG_SELECTOR_FEATURE_VERSION,
        "vectorizer": vectorizer,
        "classifier": classifier,
        "threshold": threshold,
        "max_selected_edges": 6,
    }, model_path)
    example_path = args.out / "edge_examples.jsonl"
    _write_jsonl(example_path, examples)
    report = {
        "schema_version": "qpeg-v2-selector-training-result-v1",
        "experiment_id": protocol["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_TRAIN_HOLDOUT_ADVANCE" if all(gates.values()) else "FAIL_STOP_SELECTOR",
        "protocol": {"path": str(args.protocol), "sha256": _sha256(args.protocol)},
        "split": {
            split: {
                "qids": len(qid_sets[split]),
                "families": len(family_sets[split]),
                "edges": len(by_split[split]),
                "positive_edges": sum(row["label"] for row in by_split[split]),
            }
            for split in by_split
        },
        "family_overlap": {
            "train_dev": len(family_sets["train"] & family_sets["dev"]),
            "train_holdout": len(family_sets["train"] & family_sets["holdout"]),
            "dev_holdout": len(family_sets["dev"] & family_sets["holdout"]),
        },
        "selected_threshold_from_dev": selected_dev,
        "metrics": split_metrics,
        "gates": {"checks": gates, "all_pass": all(gates.values())},
        "outputs": {
            "model": {"path": str(model_path), "sha256": _sha256(model_path)},
            "examples": {"path": str(example_path), "sha256": _sha256(example_path)},
        },
        "scientific_boundary": "Train split and family-disjoint train holdout only; no evaluation confirmation labels accessed.",
    }
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra={"phase": "qpeg_v2_selector_training", **report}, status=report["status"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
