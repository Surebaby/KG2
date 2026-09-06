#!/usr/bin/env python
"""Freeze the QPEG-v2 confirmation stop decision and representation audit.

This script does not rescore or alter predictions.  It binds the already
consumed confirmation result to its frozen protocol and measures whether the
selected semantic triples retained answer-bearing text from their provenance
sentences.  The output is append-only and corrects the inherited pilot-only
``scientific_boundary`` string in the original generic evaluator report.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from kgproweight.kg.qpeg import passage_sentences
from kgproweight.utils.logging import dump_manifest


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _norm(value: object) -> str:
    return " ".join(_WORD_RE.findall(str(value or "").casefold()))


def _visible(golds: Iterable[object], values: Iterable[object]) -> bool:
    blob = _norm(" ".join(str(value or "") for value in values))
    return any(_norm(gold) and _norm(gold) in blob for gold in golds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result_dir", type=Path,
        default=Path("outputs/validation/qpeg_v2_selector_confirmation100x3_strong_sft_ab_seed42"),
    )
    parser.add_argument(
        "--qpeg_records", type=Path,
        default=Path("data/derived/qpeg_v2_selector_n1350_seed42/question_graph_records.jsonl"),
    )
    parser.add_argument(
        "--qpeg_inputs", type=Path,
        default=Path("outputs/audits/qpeg_v2_selector_confirmation100x3_ab_inputs_seed42/arm_qpeg.jsonl"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/audits/qpeg_v2_confirmation_failure_audit_v1"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite audit: {args.out}")
    args.out.mkdir(parents=True)

    report_path = args.result_dir / "report.json"
    predictions_path = args.result_dir / "predictions.jsonl"
    result = json.loads(report_path.read_text(encoding="utf-8"))
    protocol_path = Path(result["protocol"]["path"])
    if _sha256(protocol_path) != result["protocol"]["sha256"]:
        raise ValueError("result protocol hash mismatch")
    if result.get("gates", {}).get("all_pass") is not False:
        raise ValueError("expected failed QPEG-v2 confirmation result")

    inputs = {str(row["question_key"]): row for row in _read_jsonl(args.qpeg_inputs)}
    records = {
        str(row["question_key"]): row
        for row in _read_jsonl(args.qpeg_records)
        if row.get("role") == "confirmation"
    }
    predictions = [
        row for row in _read_jsonl(predictions_path) if row.get("arm") == "qpeg"
    ]
    if len(predictions) != 300 or set(inputs) != {str(row["question_key"]) for row in predictions}:
        raise ValueError("confirmation prediction/input identity mismatch")

    counters: dict[str, Counter[str]] = {dataset: Counter() for dataset in DATASETS}
    details: list[dict[str, Any]] = []
    for prediction in predictions:
        key = str(prediction["question_key"])
        dataset = str(prediction["dataset"])
        source = inputs[key]
        record = records[key]
        passages = list(source["retrieved_passages"])
        selected_sentences: list[str] = []
        for edge in record.get("edges") or []:
            rank = int(edge["passage_rank"])
            sentence_index = int(edge["sentence_index"])
            sentences = passage_sentences(passages[rank])
            if sentence_index >= len(sentences):
                raise ValueError(f"provenance points outside passage sentence list: {key}")
            sentence = sentences[sentence_index]
            expected = hashlib.sha256(sentence.encode("utf-8")).hexdigest()
            if expected != edge["sentence_sha256"]:
                raise ValueError(f"provenance sentence hash mismatch: {key}")
            selected_sentences.append(sentence)
        golds = list(source.get("gold_answers") or [])
        passage_visible = _visible(golds, [row.get("contents") for row in passages])
        tail_visible = _visible(golds, [edge.get("tail_surface") for edge in record.get("edges") or []])
        sentence_visible = _visible(golds, selected_sentences)
        counter = counters[dataset]
        counter["n"] += 1
        counter["passage_visible"] += passage_visible
        counter["selected_tail_visible"] += tail_visible
        counter["selected_provenance_sentence_visible"] += sentence_visible
        counter["nonempty"] += bool(record.get("edges"))
        details.append({
            "question_key": key,
            "dataset": dataset,
            "qid": prediction["qid"],
            "passage_visible": passage_visible,
            "selected_tail_visible": tail_visible,
            "selected_provenance_sentence_visible": sentence_visible,
            "selected_edge_count": len(record.get("edges") or []),
        })

    detail_path = args.out / "per_question.jsonl"
    with detail_path.open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    datasets: dict[str, Any] = {}
    for dataset, counter in counters.items():
        n = counter["n"]
        datasets[dataset] = {
            **dict(counter),
            "passage_visible_rate": counter["passage_visible"] / n,
            "selected_tail_visible_rate": counter["selected_tail_visible"] / n,
            "selected_provenance_sentence_visible_rate": counter["selected_provenance_sentence_visible"] / n,
        }
    audit = {
        "schema_version": "qpeg-v2-confirmation-failure-audit-v1",
        "experiment_id": "QPEG-V2-CONFIRMATION100X3-FAILURE-AUDIT-V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL_STOP_FINAL",
        "decision": "QPEG-v2 does not advance to SFT or PPO; no confirmation retuning is allowed.",
        "result": {"path": str(report_path), "sha256": _sha256(report_path)},
        "predictions": {"path": str(predictions_path), "sha256": _sha256(predictions_path)},
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "qpeg_records": {"path": str(args.qpeg_records), "sha256": _sha256(args.qpeg_records)},
        "qpeg_inputs": {"path": str(args.qpeg_inputs), "sha256": _sha256(args.qpeg_inputs)},
        "datasets": datasets,
        "observed_result": {
            "macro_delta_em": result["macro_delta_em"],
            "overall_delta_f1": result["overall"]["paired"]["delta_f1"],
            "per_dataset": {
                dataset: result["by_dataset"][dataset]["paired"] for dataset in DATASETS
            },
        },
        "metadata_correction": {
            "original_scientific_boundary": result.get("scientific_boundary"),
            "correct_scientific_boundary": "family-disjoint confirmation100x3 was opened and consumed; this is final stop evidence for QPEG-v2",
            "numbers_or_predictions_changed": False,
        },
        "diagnosis": (
            "The train-only selector learned relevance of provenance sentences, but inference injected only "
            "regex-derived semantic triple tails. Answer-bearing text was frequently present in the selected "
            "source sentence but absent from the injected tail. This is a representation mismatch, not evidence "
            "that the confirmation may be retuned."
        ),
        "scientific_boundary": (
            "This audit explains the frozen negative result. It does not validate a replacement method and "
            "must not be used to change the consumed confirmation decision."
        ),
        "outputs": {"per_question": {"path": str(detail_path), "sha256": _sha256(detail_path)}},
    }
    audit_path = args.out / "decision_addendum.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra={"phase": "qpeg_v2_confirmation_failure_audit", **audit}, status=audit["status"])
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
