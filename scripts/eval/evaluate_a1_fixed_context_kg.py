#!/usr/bin/env python
"""Greedy paired legacy/proof-KG inference on frozen fixed-context inputs."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Mapping

import torch

from kgproweight.data.parsers import extract_final_answer, parse_steps
from kgproweight.data.prompts import build_rl_messages
from kgproweight.eval.metrics import compute_em, compute_f1
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir
from scripts.pilot.score_a1_fixed_context_kg import paired_metrics


EVALUATOR_VERSION = "a1-fixed-context-kg-eval-2-paired-stats"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: List[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _norm(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _visible(golds: List[str], values: List[str]) -> bool:
    blob = _norm(" ".join(values))
    return any(_norm(gold) and _norm(gold) in blob for gold in golds)


def _common_without_kg(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if key != "kg_subgraph"}


def _adapter_sha(adapter: Path) -> Dict[str, str]:
    return {
        "adapter_config_sha256": _sha256(adapter / "adapter_config.json"),
        "adapter_model_sha256": _sha256(adapter / "adapter_model.safetensors"),
    }


def _validate_frozen_inputs(
    protocol: Mapping[str, Any],
    legacy_path: Path,
    proof_path: Path,
    adapter: Path,
    model_label: str,
    max_new_tokens: int,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    expected_inputs = protocol["inputs"]
    if _sha256(legacy_path) != expected_inputs["arm_legacy"]["sha256"]:
        raise ValueError("legacy input hash differs from frozen protocol")
    if _sha256(proof_path) != expected_inputs["arm_proof"]["sha256"]:
        raise ValueError("proof input hash differs from frozen protocol")
    if model_label not in protocol["models"]:
        raise ValueError(f"model_label not frozen: {model_label}")
    if _adapter_sha(adapter) != {
        key: protocol["models"][model_label][key]
        for key in ("adapter_config_sha256", "adapter_model_sha256")
    }:
        raise ValueError("adapter hashes differ from frozen protocol")
    generation = protocol["generation"]
    if max_new_tokens != int(generation["max_new_tokens"]) or seed != int(generation["seed"]):
        raise ValueError("generation settings differ from frozen protocol")

    legacy = _read_jsonl(legacy_path)
    proof = _read_jsonl(proof_path)
    if len(legacy) != int(protocol["n"]) or len(proof) != len(legacy):
        raise ValueError("input row count differs from frozen protocol")
    if any(_common_without_kg(left) != _common_without_kg(right) for left, right in zip(legacy, proof)):
        raise ValueError("paired inputs differ outside kg_subgraph")
    qid_sha = hashlib.sha256("\n".join(str(row["qid"]) for row in legacy).encode()).hexdigest()
    if qid_sha != protocol["qid_order_sha256"]:
        raise ValueError("qid order hash differs from frozen protocol")
    return legacy, proof


def _score_generation(
    *,
    row: Mapping[str, Any],
    generation: str,
    prompt_sha256: str,
    prompt_tokens: int,
    model_label: str,
    arm: str,
    input_sha256: str,
) -> Dict[str, Any]:
    kg = list(row.get("kg_subgraph") or [])
    steps = parse_steps(generation, known_kg=kg)
    answer = (extract_final_answer(generation) or "").split("\n", 1)[0].strip()
    golds = [str(value) for value in row.get("gold_answers") or [] if str(value).strip()]
    indices = [step.index for step in steps]
    cited = sum(len(step.cited_triples) for step in steps)
    unknown = sum(len(step.unknown_citation_surfaces) for step in steps)
    contract_error = any(step.citation_contract_errors for step in steps)
    passage_values = [str(value.get("contents") or "") for value in row["retrieved_passages"]]
    kg_values = [str(triple[2]) for triple in kg if len(triple) == 3]
    return {
        "row_id": row["row_id"],
        "dataset": row["dataset"],
        "qid": row["qid"],
        "question": row["question"],
        "gold_answers": golds,
        "model_label": model_label,
        "arm": arm,
        "input_sha256": input_sha256,
        "prompt_sha256": prompt_sha256,
        "prompt_tokens": prompt_tokens,
        "kg_triple_count": len(kg),
        "gold_in_passages": _visible(golds, passage_values),
        "gold_in_kg_tail": _visible(golds, kg_values),
        "prediction": answer,
        "em": compute_em(answer, golds) if answer and golds else 0.0,
        "f1": compute_f1(answer, golds) if answer and golds else 0.0,
        "n_steps": len(steps),
        "well_formed": bool(steps and answer),
        "contiguous": indices == list(range(1, len(indices) + 1)),
        "known_citation_count": cited,
        "unknown_citation_count": unknown,
        "known_citation_response": cited > 0,
        "citation_contract_error": contract_error,
        "generation": generation,
    }


def _aggregate(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "em": sum(float(row["em"]) for row in rows) / max(1, n),
        "f1": sum(float(row["f1"]) for row in rows) / max(1, n),
        "parse_rate": sum(bool(row["well_formed"]) for row in rows) / max(1, n),
        "contiguous_rate": sum(bool(row["contiguous"]) for row in rows) / max(1, n),
        "known_citation_response_rate": sum(
            bool(row["known_citation_response"]) for row in rows
        ) / max(1, n),
        "citation_contract_error_rate": sum(
            bool(row["citation_contract_error"]) for row in rows
        ) / max(1, n),
        "gold_in_passages_rate": sum(bool(row["gold_in_passages"]) for row in rows) / max(1, n),
        "gold_in_kg_tail_rate": sum(bool(row["gold_in_kg_tail"]) for row in rows) / max(1, n),
        "step_histogram": dict(Counter(int(row["n_steps"]) for row in rows)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--legacy_input", required=True)
    parser.add_argument("--proof_input", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol).resolve()
    legacy_path = Path(args.legacy_input).resolve()
    proof_path = Path(args.proof_input).resolve()
    adapter_path = Path(args.adapter).resolve()
    base_path = Path(args.base_model).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    legacy, proof = _validate_frozen_inputs(
        protocol,
        legacy_path,
        proof_path,
        adapter_path,
        args.model_label,
        args.max_new_tokens,
        args.seed,
    )
    base_hashes = {
        "config_sha256": _sha256(base_path / "config.json"),
        "model_index_sha256": _sha256(base_path / "model.safetensors.index.json"),
    }
    if base_hashes != {
        key: protocol["base_model"][key] for key in ("config_sha256", "model_index_sha256")
    }:
        raise SystemExit("base model identity differs from frozen protocol")

    run_dir, experiment_id = prepare_new_run_dir(
        args.run_dir,
        experiment_id=args.experiment_id,
        extra={
            "phase": "a1_fixed_context_zero_training_kg_utilization",
            "model_label": args.model_label,
            "protocol_sha256": _sha256(protocol_path),
        },
    )
    predictions_path = run_dir / "predictions.jsonl"
    report_path = run_dir / "report.json"
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

        input_hashes = {"legacy": _sha256(legacy_path), "proof": _sha256(proof_path)}
        predictions: List[Dict[str, Any]] = []
        for index, (legacy_row, proof_row) in enumerate(zip(legacy, proof), start=1):
            for arm, row in (("legacy", legacy_row), ("proof", proof_row)):
                messages = build_rl_messages(
                    question=str(row["question"]),
                    retrieved_passages=list(row["retrieved_passages"]),
                    kg_triples=list(row["kg_subgraph"]),
                    top_k=int(protocol["generation"]["top_k_passages"]),
                )
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                encoded = tokenizer(
                    prompt, return_tensors="pt", add_special_tokens=False
                ).to(model.device)
                with torch.no_grad():
                    output = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                generated = tokenizer.decode(
                    output[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
                )
                predictions.append(
                    _score_generation(
                        row=row,
                        generation=generated,
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                        prompt_tokens=int(encoded["input_ids"].shape[1]),
                        model_label=args.model_label,
                        arm=arm,
                        input_sha256=input_hashes[arm],
                    )
                )
            print(f"{args.model_label} paired inference {index}/{len(legacy)}", flush=True)

        _write_jsonl(predictions_path, predictions)
        by_arm = {
            arm: _aggregate([row for row in predictions if row["arm"] == arm])
            for arm in ("legacy", "proof")
        }
        paired = paired_metrics(predictions, args.model_label)
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": experiment_id,
            "status": "COMPLETE_ZERO_TRAINING_SEEN_DIAGNOSTIC",
            "scope": protocol["scope"],
            "model_label": args.model_label,
            "evaluator_version": EVALUATOR_VERSION,
            "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
            "generation": protocol["generation"],
            "inputs": {
                "legacy": {"path": str(legacy_path), "sha256": input_hashes["legacy"]},
                "proof": {"path": str(proof_path), "sha256": input_hashes["proof"]},
                "base_model": artifact_identity(base_path),
                "adapter": artifact_identity(adapter_path),
            },
            "by_arm": by_arm,
            "paired": paired,
            "outputs": {
                "predictions": {
                    "path": str(predictions_path), "sha256": _sha256(predictions_path)
                }
            },
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        dump_manifest(
            run_dir,
            extra={
                "experiment_id": experiment_id,
                "phase": "a1_fixed_context_zero_training_kg_utilization",
                "scope": protocol["scope"],
                "model_label": args.model_label,
                "inputs": report["inputs"],
                "outputs": report["outputs"],
                "by_arm": by_arm,
                "paired": paired,
            },
        )
        print(json.dumps({
            "status": report["status"],
            "model": args.model_label,
            "by_arm": by_arm,
            "paired": paired,
        }, indent=2))
    except Exception as exc:
        dump_manifest(
            run_dir,
            extra={
                "experiment_id": experiment_id,
                "phase": "a1_fixed_context_zero_training_kg_utilization",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            },
            status="FAILED_RUNTIME",
        )
        raise


if __name__ == "__main__":
    main()
