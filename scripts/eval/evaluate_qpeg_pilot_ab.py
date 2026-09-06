#!/usr/bin/env python
"""Run the frozen strong-SFT QPEG pilot A/B in one model-loading pass."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import torch

from kgproweight.data.prompts import build_rl_messages
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir
from scripts.eval.evaluate_a1_fixed_context_kg import _aggregate, _score_generation
from scripts.pilot.score_a1_fixed_context_kg import _bootstrap_ci, _mcnemar_exact


EVALUATOR_VERSION = "qpeg-pilot-matched-ab-v1"
ARMS = ("no_qpeg", "qpeg")


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


def _projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "kg_subgraph"}


def _adapter_hashes(path: Path) -> dict[str, str]:
    return {
        "adapter_config_sha256": _sha256(path / "adapter_config.json"),
        "adapter_model_sha256": _sha256(path / "adapter_model.safetensors"),
    }


def _paired(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {(str(row["question_key"]), str(row["arm"])): row for row in rows}
    keys = [str(row["question_key"]) for row in rows if row["arm"] == "no_qpeg"]
    pairs = [(by_key[(key, "no_qpeg")], by_key[(key, "qpeg")]) for key in keys]
    em_diffs = [float(right["em"]) - float(left["em"]) for left, right in pairs]
    f1_diffs = [float(right["f1"]) - float(left["f1"]) for left, right in pairs]
    gained = sum(left["em"] < right["em"] for left, right in pairs)
    lost = sum(left["em"] > right["em"] for left, right in pairs)
    return {
        "n": len(pairs),
        "no_qpeg_em": sum(float(left["em"]) for left, _ in pairs) / max(1, len(pairs)),
        "qpeg_em": sum(float(right["em"]) for _, right in pairs) / max(1, len(pairs)),
        "delta_em": sum(em_diffs) / max(1, len(em_diffs)),
        "delta_em_bootstrap_95ci": _bootstrap_ci(em_diffs, seed=20260902),
        "no_qpeg_f1": sum(float(left["f1"]) for left, _ in pairs) / max(1, len(pairs)),
        "qpeg_f1": sum(float(right["f1"]) for _, right in pairs) / max(1, len(pairs)),
        "delta_f1": sum(f1_diffs) / max(1, len(f1_diffs)),
        "delta_f1_bootstrap_95ci": _bootstrap_ci(f1_diffs, seed=20260903),
        "gained_correct": gained,
        "lost_correct": lost,
        "tied_correctness": len(pairs) - gained - lost,
        "net_correct": gained - lost,
        "mcnemar_exact_p": _mcnemar_exact(gained, lost),
        "prediction_changed": sum(left["prediction"] != right["prediction"] for left, right in pairs),
        "no_qpeg_parse_rate": sum(bool(left["well_formed"]) for left, _ in pairs) / max(1, len(pairs)),
        "qpeg_parse_rate": sum(bool(right["well_formed"]) for _, right in pairs) / max(1, len(pairs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument(
        "--reuse_no_qpeg_predictions",
        help="Optional prior matched-run predictions.jsonl; only its validated no_qpeg rows are reused.",
    )
    parser.add_argument(
        "--reuse_no_qpeg_report",
        help="Required with --reuse_no_qpeg_predictions; binds prior model and generation identity.",
    )
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    adapter_path = Path(args.adapter).resolve()
    base_path = Path(args.base_model).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    arm_paths = {
        arm: Path(protocol["inputs"][f"arm_{arm}"]["path"]).resolve()
        for arm in ARMS
    }
    arm_rows = {arm: _read_jsonl(path) for arm, path in arm_paths.items()}
    for arm in ARMS:
        if _sha256(arm_paths[arm]) != protocol["inputs"][f"arm_{arm}"]["sha256"]:
            raise ValueError(f"{arm} input hash differs from frozen protocol")
        if len(arm_rows[arm]) != int(protocol["n"]):
            raise ValueError(f"{arm} row count differs from frozen protocol")
    if any(_projection(left) != _projection(right) for left, right in zip(arm_rows["no_qpeg"], arm_rows["qpeg"])):
        raise ValueError("paired inputs differ outside kg_subgraph")
    qid_order = [str(row["question_key"]) for row in arm_rows["no_qpeg"]]
    if hashlib.sha256("\n".join(qid_order).encode()).hexdigest() != protocol["qid_order_sha256"]:
        raise ValueError("question-key order differs from frozen protocol")
    if _adapter_hashes(adapter_path) != {
        key: protocol["models"]["strong_sft"][key]
        for key in ("adapter_config_sha256", "adapter_model_sha256")
    }:
        raise ValueError("adapter hashes differ from frozen protocol")
    base_hashes = {
        "config_sha256": _sha256(base_path / "config.json"),
        "model_index_sha256": _sha256(base_path / "model.safetensors.index.json"),
    }
    if base_hashes != {
        key: protocol["base_model"][key] for key in base_hashes
    }:
        raise ValueError("base model hashes differ from frozen protocol")

    run_dir, experiment_id = prepare_new_run_dir(
        args.run_dir,
        experiment_id=args.experiment_id,
        extra={"phase": "qpeg_pilot50x3_matched_ab", "protocol_sha256": _sha256(protocol_path)},
    )
    predictions_path = run_dir / "predictions.jsonl"
    report_path = run_dir / "report.json"
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        seed = int(protocol["generation"]["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        predictions: list[dict[str, Any]] = []
        reused_no_qpeg_path = None
        reused_no_qpeg_report_path = None
        if args.reuse_no_qpeg_predictions:
            if not args.reuse_no_qpeg_report:
                raise ValueError("--reuse_no_qpeg_report is required when reusing predictions")
            reused_no_qpeg_path = Path(args.reuse_no_qpeg_predictions).resolve()
            reused_no_qpeg_report_path = Path(args.reuse_no_qpeg_report).resolve()
            prior_report = json.loads(reused_no_qpeg_report_path.read_text(encoding="utf-8"))
            prior_protocol_path = Path(prior_report["protocol"]["path"])
            if _sha256(prior_protocol_path) != prior_report["protocol"]["sha256"]:
                raise ValueError("prior protocol hash no longer matches its report")
            prior_protocol = json.loads(prior_protocol_path.read_text(encoding="utf-8"))
            if prior_protocol["generation"] != protocol["generation"]:
                raise ValueError("prior generation settings differ from current frozen protocol")
            if prior_report["inputs"]["adapter"]["inventory_sha256"] != artifact_identity(adapter_path)["inventory_sha256"]:
                raise ValueError("prior adapter identity differs from current adapter")
            if prior_report["inputs"]["base_model"]["inventory_sha256"] != artifact_identity(base_path)["inventory_sha256"]:
                raise ValueError("prior base-model identity differs from current base model")
            reused = [
                row for row in _read_jsonl(reused_no_qpeg_path)
                if row.get("arm") == "no_qpeg"
            ]
            if len(reused) != int(protocol["n"]):
                raise ValueError("reused no_qpeg prediction count differs from frozen protocol")
            if [str(row.get("question_key")) for row in reused] != qid_order:
                raise ValueError("reused no_qpeg question order differs from frozen protocol")
            if any(row.get("model_label") != "strong_sft" for row in reused):
                raise ValueError("reused no_qpeg predictions have wrong model identity")
            # Hash the exact prompts again with the current tokenizer. This is
            # stronger than comparing wrapper JSON files whose QPEG telemetry
            # legitimately changes between extractor revisions.
            rebound: list[dict[str, Any]] = []
            for prior, current in zip(reused, arm_rows["no_qpeg"]):
                messages = build_rl_messages(
                    question=str(current["question"]),
                    retrieved_passages=list(current["retrieved_passages"]),
                    kg_triples=[],
                    top_k=int(protocol["generation"]["top_k_passages"]),
                )
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if hashlib.sha256(prompt.encode()).hexdigest() != prior.get("prompt_sha256"):
                    raise ValueError(f"reused no_qpeg prompt differs for {current['question_key']}")
                copied = dict(prior)
                copied["input_sha256"] = protocol["inputs"]["arm_no_qpeg"]["sha256"]
                copied["reused_prediction"] = True
                copied["reused_from_input_sha256"] = prior.get("input_sha256")
                rebound.append(copied)
            predictions.extend(rebound)
        for index in range(int(protocol["n"])):
            generation_arms = ("qpeg",) if reused_no_qpeg_path is not None else ARMS
            for arm in generation_arms:
                row = arm_rows[arm][index]
                messages = build_rl_messages(
                    question=str(row["question"]),
                    retrieved_passages=list(row["retrieved_passages"]),
                    kg_triples=list(row["kg_subgraph"]),
                    top_k=int(protocol["generation"]["top_k_passages"]),
                )
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    output = model.generate(
                        **encoded,
                        max_new_tokens=int(protocol["generation"]["max_new_tokens"]),
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                generation = tokenizer.decode(
                    output[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True
                )
                scored = _score_generation(
                    row=row,
                    generation=generation,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    prompt_tokens=int(encoded["input_ids"].shape[1]),
                    model_label="strong_sft",
                    arm=arm,
                    input_sha256=protocol["inputs"][f"arm_{arm}"]["sha256"],
                )
                scored["question_key"] = row["question_key"]
                scored["qpeg_edge_count"] = row["qpeg_edge_count"]
                predictions.append(scored)
            print(f"QPEG pilot paired inference {index + 1}/{protocol['n']}", flush=True)
        _write_jsonl(predictions_path, predictions)

        by_dataset: dict[str, Any] = {}
        for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
            current = [row for row in predictions if row["dataset"] == dataset]
            by_dataset[dataset] = {
                "by_arm": {arm: _aggregate([row for row in current if row["arm"] == arm]) for arm in ARMS},
                "paired": _paired(current),
            }
        overall = {
            "by_arm": {arm: _aggregate([row for row in predictions if row["arm"] == arm]) for arm in ARMS},
            "paired": _paired(predictions),
        }
        macro_delta_em = sum(value["paired"]["delta_em"] for value in by_dataset.values()) / 3
        gates = {
            "macro_delta_em_positive": macro_delta_em > float(protocol["decision_gates"]["macro_delta_em_gt"]),
            "no_dataset_net_loss_gt_2": all(
                value["paired"]["net_correct"] >= -int(protocol["decision_gates"]["max_net_correct_loss_per_dataset"])
                for value in by_dataset.values()
            ),
        }
        report = {
            "schema_version": "qpeg-pilot-ab-result-v1",
            "experiment_id": experiment_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS_ADVANCE_TO_CONFIRMATION" if all(gates.values()) else "FAIL_STOP_OR_ONE_GENERIC_REVISION",
            "evaluator_version": EVALUATOR_VERSION,
            "scope": protocol["scope"],
            "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
            "inputs": {
                "base_model": artifact_identity(base_path),
                "adapter": artifact_identity(adapter_path),
                **{arm: {"path": str(arm_paths[arm]), "sha256": _sha256(arm_paths[arm])} for arm in ARMS},
                "reused_no_qpeg_predictions": (
                    {"path": str(reused_no_qpeg_path), "sha256": _sha256(reused_no_qpeg_path)}
                    if reused_no_qpeg_path is not None else None
                ),
                "reused_no_qpeg_report": (
                    {"path": str(reused_no_qpeg_report_path), "sha256": _sha256(reused_no_qpeg_report_path)}
                    if reused_no_qpeg_report_path is not None else None
                ),
            },
            "by_dataset": by_dataset,
            "overall": overall,
            "macro_delta_em": macro_delta_em,
            "gates": {"checks": gates, "all_pass": all(gates.values())},
            "scientific_boundary": "pilot engineering evidence only; confirmation remains unopened",
            "outputs": {"predictions": {"path": str(predictions_path), "sha256": _sha256(predictions_path)}},
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        dump_manifest(run_dir, extra={"phase": "qpeg_pilot50x3_matched_ab", **report}, status=report["status"])
        print(json.dumps({
            "status": report["status"],
            "macro_delta_em": macro_delta_em,
            "gates": gates,
            "datasets": {dataset: value["paired"] for dataset, value in by_dataset.items()},
        }, ensure_ascii=False, indent=2))
    except Exception as exc:
        dump_manifest(
            run_dir,
            extra={"phase": "qpeg_pilot50x3_matched_ab", "failure": {"type": type(exc).__name__, "message": str(exc)}},
            status="FAILED_RUNTIME",
        )
        raise


if __name__ == "__main__":
    main()
