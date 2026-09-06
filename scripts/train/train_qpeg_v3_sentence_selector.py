#!/usr/bin/env python
"""Fit and evaluate the frozen train-only QPEG-v3 full-sentence selector."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from kgproweight.kg.qpeg import passage_sentences, passage_title
from kgproweight.kg.qpeg_sentence_selector import (
    QPEG_SENTENCE_FEATURE_VERSION,
    sentence_candidates,
)
from kgproweight.utils.logging import dump_manifest
from scripts.diagnose.audit_qpeg_v2_train_supervision import DATASETS, _hash_sample
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256


_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(value: object) -> str:
    return " ".join(_WORD_RE.findall(str(value or "").casefold()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _training_passages_and_labels(
    dataset: str, row: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], set[tuple[int, int]]]:
    metadata = row.get("metadata") or {}
    if dataset in {"hotpotqa", "2wikimultihopqa"}:
        context = metadata.get("context") or {}
        titles = list(context.get("title") or [])
        sentence_key = "sentences" if dataset == "hotpotqa" else "content"
        contents = list(context.get(sentence_key) or [])
        supporting = metadata.get("supporting_facts") or {}
        support_pairs = {
            (_norm(title), int(sentence_index))
            for title, sentence_index in zip(
                supporting.get("title") or [], supporting.get("sent_id") or []
            )
        }
        passages: list[dict[str, Any]] = []
        positive: set[tuple[int, int]] = set()
        for passage_rank, (title, sentences) in enumerate(zip(titles, contents)):
            clean_sentences = [str(sentence) for sentence in sentences]
            passages.append({
                "id": f"train-context-{passage_rank}",
                "title": str(title),
                "contents": f'"{title}"\n' + " ".join(clean_sentences),
                "_sentences": clean_sentences,
            })
            for sentence_index in range(len(clean_sentences)):
                if (_norm(title), sentence_index) in support_pairs:
                    positive.add((passage_rank, sentence_index))
        return passages, positive

    decomposition = ((metadata.get("metadata") or {}).get("question_decomposition") or [])
    passages = []
    positive: set[tuple[int, int]] = set()
    for passage_rank, step in enumerate(decomposition):
        support = step.get("support_paragraph") or {}
        title = str(support.get("title") or "")
        text = str(support.get("paragraph_text") or "")
        passage = {
            "id": f"train-support-{support.get('idx', passage_rank)}",
            "title": title,
            "contents": f'"{title}"\n{text}',
        }
        passages.append(passage)
        answer = _norm(step.get("answer"))
        for sentence_index, sentence in enumerate(passage_sentences(passage)):
            if answer and answer in _norm(sentence):
                positive.add((passage_rank, sentence_index))
    return passages, positive


def _threshold_metrics(
    examples: list[dict[str, Any]], probabilities: list[float], threshold: float
) -> dict[str, Any]:
    labels = [int(row["label"]) for row in examples]
    predictions = [int(value >= threshold) for value in probabilities]
    selected_by_qid: Counter[str] = Counter()
    qids_by_dataset: dict[str, set[str]] = defaultdict(set)
    pairs_by_dataset: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row, prediction in zip(examples, predictions):
        qkey = str(row["question_key"])
        dataset = str(row["dataset"])
        selected_by_qid[qkey] += prediction
        qids_by_dataset[dataset].add(qkey)
        pairs_by_dataset[dataset].append((int(row["label"]), prediction))
    per_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        pairs = pairs_by_dataset[dataset]
        ds_labels = [left for left, _ in pairs]
        ds_predictions = [right for _, right in pairs]
        qids = qids_by_dataset[dataset]
        per_dataset[dataset] = {
            "sentence_precision": precision_score(ds_labels, ds_predictions, zero_division=0),
            "sentence_recall": recall_score(ds_labels, ds_predictions, zero_division=0),
            "qid_selected_rate": sum(selected_by_qid[qid] > 0 for qid in qids) / max(1, len(qids)),
            "selected_sentences": sum(ds_predictions),
            "qids": len(qids),
        }
    return {
        "threshold": threshold,
        "sentence_precision": precision_score(labels, predictions, zero_division=0),
        "sentence_recall": recall_score(labels, predictions, zero_division=0),
        "sentence_f1": f1_score(labels, predictions, zero_division=0),
        "selected_sentences": sum(predictions),
        "qid_selected_rate": sum(value > 0 for value in selected_by_qid.values()) / max(1, len(selected_by_qid)),
        "per_dataset": per_dataset,
    }


def _choose_threshold(examples: list[dict[str, Any]], probabilities: list[float]) -> dict[str, Any]:
    candidates = [
        _threshold_metrics(examples, probabilities, value / 100) for value in range(5, 96)
    ]
    feasible = [
        row for row in candidates
        if row["sentence_precision"] >= 0.55
        and row["qid_selected_rate"] >= 0.65
        and all(value["qid_selected_rate"] >= 0.55 for value in row["per_dataset"].values())
    ]
    if not feasible:
        raise RuntimeError("no dev threshold satisfies frozen precision/coverage constraints")
    return max(feasible, key=lambda row: (row["sentence_f1"], row["sentence_precision"], row["threshold"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("outputs/audits/qpeg_v3_sentence_selector_protocol_v1/protocol.json"),
    )
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/training/qpeg_v3_sentence_selector_n1000x3_seed42"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite selector run: {args.out}")
    args.out.mkdir(parents=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_FIT":
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
            qkey = f"{dataset}::{row['id']}"
            family_sets[split].add(family)
            qid_sets[split].add(qkey)
            passages, positive = _training_passages_and_labels(dataset, row)
            candidates = sentence_candidates(dataset=dataset, question=question, passages=passages)
            for candidate_index, candidate in enumerate(candidates):
                location = (int(candidate["passage_rank"]), int(candidate["sentence_index"]))
                examples.append({
                    "question_key": qkey,
                    "dataset": dataset,
                    "split": split,
                    "family_sha256": family,
                    "candidate_index": candidate_index,
                    "label": int(location in positive),
                    "features": candidate["features"],
                })
    if any(
        family_sets[left] & family_sets[right]
        for left, right in (("train", "dev"), ("train", "holdout"), ("dev", "holdout"))
    ):
        raise RuntimeError("family leakage across selector splits")

    by_split = {
        split: [row for row in examples if row["split"] == split]
        for split in ("train", "dev", "holdout")
    }
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
        "selected_sentence_precision_ge_0_55": holdout["sentence_precision"] >= 0.55,
        "qid_selected_rate_ge_0_65": holdout["qid_selected_rate"] >= 0.65,
        "each_dataset_precision_ge_0_45": all(
            value["sentence_precision"] >= 0.45 for value in holdout["per_dataset"].values()
        ),
        "each_dataset_qid_selected_rate_ge_0_55": all(
            value["qid_selected_rate"] >= 0.55 for value in holdout["per_dataset"].values()
        ),
    }

    model_path = args.out / "selector.joblib"
    joblib.dump({
        "feature_version": QPEG_SENTENCE_FEATURE_VERSION,
        "vectorizer": vectorizer,
        "classifier": classifier,
        "threshold": threshold,
        "max_selected_edges": 4,
    }, model_path)
    examples_path = args.out / "sentence_examples.jsonl"
    _write_jsonl(examples_path, examples)
    report = {
        "schema_version": "qpeg-v3-sentence-selector-training-result-v1",
        "experiment_id": protocol["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_TRAIN_HOLDOUT_ADVANCE" if all(gates.values()) else "FAIL_STOP_SELECTOR",
        "protocol": {"path": str(args.protocol), "sha256": _sha256(args.protocol)},
        "split": {
            split: {
                "qids": len(qid_sets[split]),
                "families": len(family_sets[split]),
                "sentences": len(by_split[split]),
                "positive_sentences": sum(row["label"] for row in by_split[split]),
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
            "examples": {"path": str(examples_path), "sha256": _sha256(examples_path)},
        },
        "scientific_boundary": (
            "Train-only and family-disjoint train holdout only. No evaluation confirmation/final labels "
            "were used. Passing these gates does not establish downstream EM/F1 utility."
        ),
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra={"phase": "qpeg_v3_sentence_selector_training", **report}, status=report["status"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
