#!/usr/bin/env python
"""Audit exact-generation reuse and canonical re-scoring for QPEG-v3 arm A."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from transformers import AutoTokenizer

from kgproweight.data.prompts import build_rl_messages
from kgproweight.utils.logging import dump_manifest
from scripts.eval.evaluate_a1_fixed_context_kg import _score_generation


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
    protocol_path = Path("outputs/audits/qpeg_v3_final300x3_ab_inputs_seed42/protocol.json")
    out = Path("outputs/audits/qpeg_v3_historical_a_canonical_rescore_v1")
    if out.exists():
        raise SystemExit(f"refusing to overwrite audit: {out}")
    out.mkdir(parents=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    arm_path = Path(protocol["inputs"]["arm_no_graph"]["path"])
    arm_rows = _read_jsonl(arm_path)
    tokenizer = AutoTokenizer.from_pretrained(protocol["models"]["base_model"]["path"])
    historical: dict[tuple[str, str], dict[str, Any]] = {}
    historical_assets: dict[str, Any] = {}
    for dataset in DATASETS:
        asset = protocol["inputs"]["historical_no_graph_predictions"][dataset]["predictions"]
        path = Path(asset["path"])
        if _sha256(path) != asset["sha256"]:
            raise ValueError(f"historical prediction hash mismatch: {dataset}")
        historical_assets[dataset] = asset
        for row in _read_jsonl(path):
            historical[(dataset, str(row["qid"]))] = row

    counters: dict[str, Counter[str]] = {dataset: Counter() for dataset in DATASETS}
    canonical_rows: list[dict[str, Any]] = []
    for row in arm_rows:
        dataset = str(row["dataset"])
        prior = historical[(dataset, str(row["qid"]))]
        messages = build_rl_messages(
            question=str(row["question"]), retrieved_passages=list(row["retrieved_passages"]),
            kg_triples=[], top_k=10,
        )
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        if prompt_sha != prior["prompt_sha256"]:
            raise ValueError(f"prompt mismatch: {row['question_key']}")
        scored = _score_generation(
            row=row, generation=str(prior["generation"]), prompt_sha256=prompt_sha,
            prompt_tokens=len(tokenizer(prompt, add_special_tokens=False)["input_ids"]),
            model_label="strong_sft", arm="no_graph",
            input_sha256=protocol["inputs"]["arm_no_graph"]["sha256"],
        )
        scored["question_key"] = row["question_key"]
        scored["qpeg_edge_count"] = row["qpeg_edge_count"]
        scored["reused_generation"] = True
        scored["historical_prediction"] = prior["prediction"]
        scored["historical_em"] = prior["em"]
        scored["historical_f1"] = prior["f1"]
        canonical_rows.append(scored)
        counter = counters[dataset]
        counter["n"] += 1
        counter["prompt_exact"] += 1
        counter["prediction_exact"] += scored["prediction"] == prior["prediction"]
        counter["em_exact"] += float(scored["em"]) == float(prior["em"])
        counter["f1_exact"] += abs(float(scored["f1"]) - float(prior["f1"])) <= 1e-12

    rows_path = out / "canonical_a_predictions.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in canonical_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        current = [row for row in canonical_rows if row["dataset"] == dataset]
        old = [historical[(dataset, str(row["qid"]))] for row in current]
        datasets[dataset] = {
            **dict(counters[dataset]),
            "historical_em": sum(float(row["em"]) for row in old) / 300,
            "canonical_rescore_em": sum(float(row["em"]) for row in current) / 300,
            "historical_f1": sum(float(row["f1"]) for row in old) / 300,
            "canonical_rescore_f1": sum(float(row["f1"]) for row in current) / 300,
        }
        datasets[dataset]["f1_mean_drift"] = (
            datasets[dataset]["canonical_rescore_f1"] - datasets[dataset]["historical_f1"]
        )
    all_pass = (
        len(canonical_rows) == 900
        and sum(value["prompt_exact"] for value in counters.values()) == 900
        and sum(value["em_exact"] for value in counters.values()) == 900
    )
    report = {
        "schema_version": "qpeg-v3-historical-a-canonical-rescore-audit-v1",
        "experiment_id": "QPEG-V3-HISTORICAL-A-CANONICAL-RESCORE-V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_REUSE_GENERATION_CANONICAL_RESCORE" if all_pass else "FAIL_NO_REUSE",
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "arm_input": {"path": str(arm_path), "sha256": _sha256(arm_path)},
        "historical_assets": historical_assets,
        "datasets": datasets,
        "gates": {
            "n_900": len(canonical_rows) == 900,
            "prompt_exact_900": sum(value["prompt_exact"] for value in counters.values()) == 900,
            "em_exact_900": sum(value["em_exact"] for value in counters.values()) == 900,
            "all_pass": all_pass,
        },
        "interpretation": (
            "Reuse exact historical generations, not historical parsed prediction/score fields. Both arms are "
            "scored by the current frozen canonical scorer. Differences are limited to legacy handling of "
            "truncated outputs without a Final Answer block."
        ),
        "outputs": {"canonical_a_predictions": {"path": str(rows_path), "sha256": _sha256(rows_path)}},
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(out, extra={"phase": "qpeg_v3_historical_a_canonical_rescore", **report}, status=report["status"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
