#!/usr/bin/env python
"""Freeze the train-only QPEG-v3 sentence-selector protocol before fitting."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from kgproweight.utils.logging import dump_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    out = Path("outputs/audits/qpeg_v3_sentence_selector_protocol_v1")
    if out.exists():
        raise SystemExit(f"refusing to overwrite protocol: {out}")
    out.mkdir(parents=True)
    code = [
        Path("kgproweight/kg/qpeg.py"),
        Path("kgproweight/kg/qpeg_sentence_selector.py"),
        Path("scripts/train/train_qpeg_v3_sentence_selector.py"),
    ]
    data = [Path(f"data/{dataset}/train.jsonl") for dataset in (
        "hotpotqa", "2wikimultihopqa", "musique"
    )]
    protocol = {
        "schema_version": "qpeg-v3-sentence-selector-protocol-v1",
        "experiment_id": "QPEG-V3-SENTENCE-SELECTOR-TRAINONLY-N1000X3-SEED42",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_FIT",
        "hypothesis": (
            "Selecting complete provenance sentences fixes QPEG-v2's supervision-representation mismatch "
            "without adding retrieval or external knowledge resources."
        ),
        "scope": "train-only selector development; evaluation final remains unopened for QPEG-v3",
        "data": {
            "sample_per_dataset": 1000,
            "selection": "sha256(42::dataset::qid), first 1000",
            "split_unit": "answer-free question family",
            "split": "family_sha256 modulo 10: train=0..6, dev=7, holdout=8..9",
            "labels": {
                "hotpotqa": "exact train supporting-fact title/sentence index",
                "2wikimultihopqa": "exact train supporting-fact title/sentence index",
                "musique": "train decomposition support sentence contains the decomposition-step answer",
            },
        },
        "representation": {
            "edge": "(passage_title, evidence sentence, full_source_sentence)",
            "semantics": "typed passage evidence edge; not asserted as a Wikidata factual relation",
            "provenance": "passage id/rank, sentence index, sentence SHA256",
            "max_selected_edges": 4,
            "fallback": "empty graph; no legacy or Wikidata fallback",
        },
        "features": (
            "answer-free dataset/rank/index/lexical coverage/question-cue/numeric/temporal/"
            "capitalization/cross-title features"
        ),
        "model": {
            "type": "DictVectorizer + LogisticRegression",
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 1000,
            "random_state": 42,
        },
        "threshold_selection": {
            "source": "train-only family-disjoint dev",
            "grid": "0.05..0.95 step 0.01",
            "constraints": {
                "sentence_precision_min": 0.55,
                "qid_selected_rate_min": 0.65,
                "each_dataset_qid_selected_rate_min": 0.55,
            },
            "objective": "maximum sentence F1; ties higher precision then threshold",
        },
        "holdout_gates": {
            "roc_auc_min": 0.65,
            "selected_sentence_precision_min": 0.55,
            "qid_selected_rate_min": 0.65,
            "each_dataset_precision_min": 0.45,
            "each_dataset_qid_selected_rate_min": 0.55,
        },
        "next_gate": (
            "Only if train-only holdout passes may a new final300x3 A/B evaluation protocol be proposed; "
            "final evaluation requires researcher approval and no post-final tuning."
        ),
        "forbidden": [
            "evaluation answer/support/decomposition features",
            "reuse of consumed confirmation for fitting or threshold selection",
            "per-qid patches",
            "silent legacy/Wikidata fallback",
            "claiming typed evidence sentences are Wikidata facts",
        ],
        "inputs": {str(path): _sha256(path) for path in data},
        "code_sha256": {str(path): _sha256(path) for path in code},
    }
    path = out / "protocol.json"
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(out, extra={"phase": "qpeg_v3_sentence_selector_protocol", **protocol}, status=protocol["status"])
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
