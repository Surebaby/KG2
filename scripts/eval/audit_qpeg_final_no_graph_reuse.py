#!/usr/bin/env python
"""Verify whether historical no-KG predictions exactly match frozen QPEG final prompts."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from kgproweight.data.prompts import build_rl_messages
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contexts", type=Path,
        default=Path("outputs/audits/qpeg_v1_n1350_seed42_preregistration/final.retrieval_contexts.jsonl"),
    )
    parser.add_argument(
        "--old_inputs_dir", type=Path,
        default=Path("outputs/audits/legacy_kg_utilization_2x2_nokg_inputs"),
    )
    parser.add_argument(
        "--old_predictions_root", type=Path,
        default=Path("outputs/validation"),
    )
    parser.add_argument("--base_model", type=Path, default=Path("models/llama3-8b"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/audits/qpeg_final_no_graph_prompt_reuse_v1"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite audit: {args.out}")
    args.out.mkdir(parents=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    contexts = _read_jsonl(args.contexts)
    by_dataset = {dataset: [] for dataset in DATASETS}
    for row in contexts:
        by_dataset[str(row["dataset"])].append(row)

    reports: dict[str, Any] = {}
    detail_rows: list[dict[str, Any]] = []
    input_assets: dict[str, Any] = {
        "contexts": {"path": str(args.contexts), "sha256": _sha256(args.contexts)}
    }
    total = Counter()
    for dataset in DATASETS:
        old_input_path = args.old_inputs_dir / f"{dataset}_nokg.jsonl"
        prediction_path = args.old_predictions_root / f"legacy_2x2_SFT_nokg_{dataset}" / "predictions.jsonl"
        old_rows = {str(row["qid"]): row for row in _read_jsonl(old_input_path)}
        predictions = {str(row["qid"]): row for row in _read_jsonl(prediction_path)}
        current = by_dataset[dataset]
        counters = Counter()
        for row in current:
            qid = str(row["qid"])
            old = old_rows[qid]
            prediction = predictions[qid]
            passages_equal = old["retrieved_passages"] == row["passages"]
            messages = build_rl_messages(
                question=str(row["question"]),
                retrieved_passages=list(row["passages"]),
                kg_triples=[],
                top_k=10,
            )
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            prompt_equal = prompt_sha256 == prediction.get("prompt_sha256")
            question_equal = old["question"] == row["question"]
            counters["n"] += 1
            counters["qid_join"] += 1
            counters["question_exact"] += question_equal
            counters["passages_exact"] += passages_equal
            counters["prompt_sha256_exact"] += prompt_equal
            detail_rows.append({
                "dataset": dataset,
                "qid": qid,
                "question_exact": question_equal,
                "passages_exact": passages_equal,
                "prompt_sha256_exact": prompt_equal,
                "historical_prompt_sha256": prediction.get("prompt_sha256"),
                "reconstructed_prompt_sha256": prompt_sha256,
            })
        reports[dataset] = dict(counters)
        total.update(counters)
        input_assets[dataset] = {
            "old_input": {"path": str(old_input_path), "sha256": _sha256(old_input_path)},
            "predictions": {"path": str(prediction_path), "sha256": _sha256(prediction_path)},
        }

    all_pass = (
        total["n"] == 900
        and total["qid_join"] == 900
        and total["passages_exact"] == 900
        and total["prompt_sha256_exact"] == 900
    )
    details_path = args.out / "details.jsonl"
    with details_path.open("w", encoding="utf-8") as handle:
        for row in detail_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": "qpeg-final-no-graph-prompt-reuse-audit-v1",
        "experiment_id": "QPEG-FINAL-NO-GRAPH-PROMPT-REUSE-V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_REUSE_EXACT_PROMPTS" if all_pass else "FAIL_NO_REUSE",
        "datasets": reports,
        "totals": dict(total),
        "gates": {
            "n_900": total["n"] == 900,
            "qid_join_900": total["qid_join"] == 900,
            "passages_exact_900": total["passages_exact"] == 900,
            "prompt_sha256_exact_900": total["prompt_sha256_exact"] == 900,
            "all_pass": all_pass,
        },
        "inputs": input_assets,
        "base_model": {
            "path": str(args.base_model),
            "tokenizer_config_sha256": _sha256(args.base_model / "tokenizer_config.json"),
            "tokenizer_sha256": _sha256(args.base_model / "tokenizer.json"),
        },
        "scientific_boundary": (
            "Only exact prompt identity permits reuse. Question wrapper strings may differ while the canonical "
            "prompt is identical. This audit does not validate any future graph arm."
        ),
        "outputs": {"details": {"path": str(details_path), "sha256": _sha256(details_path)}},
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra={"phase": "qpeg_final_no_graph_prompt_reuse", **report}, status=report["status"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
