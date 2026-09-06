#!/usr/bin/env python
"""Run the frozen strong-SFT exploration-headroom audit (zero updates).

The prompt file is Gold-free.  Train-only outcome labels are loaded separately
and are used only after greedy and sampled generations have been produced.  No
reward model, critic, optimiser, or PPO update is constructed by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
from pathlib import Path
import platform
import random
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from kgproweight.data.parsers import extract_final_answer, parse_steps
from kgproweight.data.prompts import build_rl_messages
from kgproweight.eval.metrics import compute_em, compute_f1
from kgproweight.kg.question_kg import question_sha256, validate_question_kg_record
from kgproweight.reward.proofkg_process import is_automatic_proofkg
from kgproweight.training.reward_function import KGProWeightRewardFunction
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_STATUS = "FROZEN_NOT_RUN_TRAIN_SIDE_DEVELOPMENT_CONSUMED"
ENVIRONMENT_PACKAGES = (
    "torch", "transformers", "peft", "accelerate", "numpy", "safetensors", "trl"
)


def software_environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ENVIRONMENT_PACKAGES:
        try:
            packages[name] = package_version(name)
        except PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "packages": packages,
        "torch_cuda_build": str(torch.version.cuda),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_ref(ref: Mapping[str, Any], label: str) -> Path:
    path = ROOT / str(ref["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    actual = _sha256(path)
    if actual != str(ref["sha256"]):
        raise ValueError(f"{label} SHA256 mismatch: {actual}")
    return path


def _generate(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> list[dict[str, Any]]:
    """Generate with the same left-padded sampling semantics as the old audit."""

    results: list[dict[str, Any]] = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=False,
        ).to(model.device)
        prompt_length = int(encoded["input_ids"].shape[1])
        kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        if do_sample:
            kwargs.update(temperature=temperature, top_p=top_p, top_k=0)
        with torch.inference_mode():
            generated = model.generate(**kwargs)
        continuation = generated[:, prompt_length:]
        texts = tokenizer.batch_decode(continuation, skip_special_tokens=True)
        for text, token_ids in zip(texts, continuation):
            ids = token_ids.detach().cpu().tolist()
            if tokenizer.eos_token_id in ids:
                ids = ids[: ids.index(tokenizer.eos_token_id) + 1]
            results.append({
                "generation": text,
                "generation_tokens": len(ids),
                "length_capped": len(ids) >= max_new_tokens,
            })
    return results


def _score_candidate(
    prompt_row: Mapping[str, Any],
    label_row: Mapping[str, Any],
    generated: Mapping[str, Any],
    *,
    candidate_type: str,
    candidate_index: int,
    min_valid_steps: int,
    min_reasoning_chars: int,
) -> dict[str, Any]:
    generation = str(generated["generation"])
    kg = list(prompt_row["kg_subgraph"])
    steps = parse_steps(generation, known_kg=kg)
    valid = KGProWeightRewardFunction._is_valid_trajectory(
        steps,
        generation,
        min_steps=min_valid_steps,
        min_reasoning_chars=min_reasoning_chars,
    )
    prediction = (extract_final_answer(generation) or "").split("\n", 1)[0].strip()
    gold_answers = [str(value) for value in label_row["gold_answers"]]
    return {
        "schema_version": "proof400-fill275-sft-headroom-candidate-v1",
        "dataset": str(prompt_row["dataset"]),
        "qid": str(prompt_row["qid"]),
        "question_sha256": str(prompt_row["question_sha256"]),
        "question_type": str(prompt_row["question_type"]),
        "candidate_type": candidate_type,
        "candidate_index": int(candidate_index),
        "generation": generation,
        "generation_tokens": int(generated["generation_tokens"]),
        "length_capped": bool(generated["length_capped"]),
        "prediction": prediction,
        "em": float(compute_em(prediction, gold_answers)),
        "f1": float(compute_f1(prediction, gold_answers)),
        "trajectory_valid": bool(valid),
        "n_steps": len(steps),
    }


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def summarize_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    qids_expected: int,
    k: int,
    gates: Mapping[str, float],
) -> dict[str, Any]:
    greedy: dict[str, Mapping[str, Any]] = {}
    sampled: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        qid = str(row["qid"])
        if row["candidate_type"] == "greedy":
            if qid in greedy:
                raise ValueError(f"duplicate greedy row: {qid}")
            greedy[qid] = row
        elif row["candidate_type"] == "sampled":
            sampled[qid].append(row)
        else:
            raise ValueError(f"unknown candidate type: {row['candidate_type']}")
    qids = sorted(set(greedy) | set(sampled))
    if len(qids) != qids_expected or set(qids) != set(greedy) or set(qids) != set(sampled):
        raise ValueError("greedy/sampled qid coverage mismatch")
    if any(len(sampled[qid]) != k for qid in qids):
        raise ValueError(f"every qid must have exactly K={k} sampled candidates")

    greedy_em = [float(greedy[qid]["em"]) for qid in qids]
    greedy_f1 = [float(greedy[qid]["f1"]) for qid in qids]
    oracle_em = [max(float(row["em"]) for row in sampled[qid]) for qid in qids]
    oracle_f1 = [max(float(row["f1"]) for row in sampled[qid]) for qid in qids]
    sampled_rows = [row for qid in qids for row in sampled[qid]]
    mixed_qids = [
        qid for qid in qids
        if min(float(row["em"]) for row in sampled[qid])
        < max(float(row["em"]) for row in sampled[qid])
    ]
    metrics = {
        "n_qids": len(qids),
        "sampled_candidates": len(sampled_rows),
        "greedy_em": _mean(greedy_em),
        "greedy_f1": _mean(greedy_f1),
        "oracle_at_4_em": _mean(oracle_em),
        "oracle_at_4_f1": _mean(oracle_f1),
        "oracle_at_4_minus_greedy_em": _mean(oracle_em) - _mean(greedy_em),
        "sample_valid_rate": _mean(float(row["trajectory_valid"]) for row in sampled_rows),
        "mixed_outcome_qids": len(mixed_qids),
        "mixed_outcome_qid_rate": len(mixed_qids) / len(qids),
    }
    decisions = {
        "sample_valid_rate": {
            "threshold": float(gates["sample_valid_rate_min"]),
            "observed": metrics["sample_valid_rate"],
        },
        "oracle_at_4_minus_greedy_em": {
            "threshold": float(gates["oracle_at_4_minus_greedy_em_min"]),
            "observed": metrics["oracle_at_4_minus_greedy_em"],
        },
        "mixed_outcome_qid_rate": {
            "threshold": float(gates["mixed_outcome_qid_rate_min"]),
            "observed": metrics["mixed_outcome_qid_rate"],
        },
    }
    for decision in decisions.values():
        decision["passed"] = bool(
            math.isfinite(float(decision["observed"]))
            and float(decision["observed"]) >= float(decision["threshold"])
        )
    return {
        "metrics": metrics,
        "gates": decisions,
        "all_pass": all(bool(row["passed"]) for row in decisions.values()),
        "mixed_outcome_qids": mixed_qids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != EXPECTED_STATUS:
        raise ValueError("headroom protocol is not frozen/not-run")
    if args.experiment_id != protocol["runtime"]["experiment_id"]:
        raise ValueError("Experiment ID differs from preregistration")
    if args.output_dir.resolve() != (ROOT / protocol["runtime"]["output_dir"]).resolve():
        raise ValueError("output directory differs from preregistration")
    for label, identity in protocol["inputs"].items():
        _verify_ref(identity, label)
    for label, identity in protocol["code_closure"].items():
        _verify_ref(identity, f"code:{label}")
    for label, identity in protocol["model"]["locked_files"].items():
        _verify_ref(identity, f"model:{label}")
    actual_environment = software_environment()
    if actual_environment != protocol.get("software_environment"):
        raise ValueError(
            "software environment differs from preregistration: "
            f"expected={protocol.get('software_environment')} actual={actual_environment}"
        )

    prompt_path = _verify_ref(protocol["outputs"]["prompt_inputs"], "prompt_inputs")
    label_path = _verify_ref(protocol["outputs"]["outcome_labels"], "outcome_labels")
    qkg_path = _verify_ref(protocol["outputs"]["question_kg_records"], "question_kg_records")
    prompt_rows = _read_jsonl(prompt_path)
    label_rows = {str(row["qid"]): row for row in _read_jsonl(label_path)}
    qkg_rows = {str(row["qid"]): row for row in _read_jsonl(qkg_path)}
    if len(prompt_rows) != protocol["cohort"]["n"] or len(label_rows) != len(prompt_rows):
        raise ValueError("frozen cohort/label count mismatch")
    qids = [str(row["qid"]) for row in prompt_rows]
    if len(set(qids)) != len(qids) or set(qids) != set(label_rows) or set(qids) != set(qkg_rows):
        raise ValueError("frozen prompt/label/question-KG identity mismatch")
    for row in prompt_rows:
        qid = str(row["qid"])
        if question_sha256(str(row["question"])) != row["question_sha256"]:
            raise ValueError(f"question hash mismatch: {qid}")
        record = qkg_rows[qid]
        validate_question_kg_record(record)
        if record["question_sha256"] != row["question_sha256"] or record["kg_subgraph"] != row["kg_subgraph"]:
            raise ValueError(f"ProofKG identity mismatch: {qid}")
        if not is_automatic_proofkg(record, record["kg_subgraph"]):
            raise ValueError(f"not a complete automatic ProofKG: {qid}")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    generation = protocol["generation"]
    seed = int(generation["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    base_path = (ROOT / protocol["model"]["base_model_path"]).resolve()
    adapter_path = (ROOT / protocol["model"]["adapter_path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if generation["dtype"] == "bf16" else torch.float16
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()

    prompts = []
    prompt_hashes: dict[str, str] = {}
    for row in prompt_rows:
        messages = build_rl_messages(
            question=str(row["question"]),
            retrieved_passages=list(row["retrieved_passages"]),
            kg_triples=list(row["kg_subgraph"]),
            top_k=int(generation["top_k_passages"]),
        )
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if token_count > int(generation["max_input_length"]):
            raise ValueError(f"prompt exceeds frozen budget: {row['qid']} {token_count}")
        prompts.append(prompt)
        prompt_hashes[str(row["qid"])] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    run_record = {
        "phase": "proof400_fill275_strong_sft_headroom_zero_update",
        "protocol": artifact_identity(protocol_path),
        "optimizer_updates": 0,
        "reward_model_loaded": False,
    }
    run_dir, experiment_id = prepare_new_run_dir(
        args.output_dir, experiment_id=args.experiment_id, extra=run_record
    )
    try:
        greedy_generated = _generate(
            model, tokenizer, prompts,
            batch_size=args.batch_size,
            max_new_tokens=int(generation["max_new_tokens"]),
            do_sample=False,
            temperature=float(generation["sampled"]["temperature"]),
            top_p=float(generation["sampled"]["top_p"]),
        )
        expanded_rows = [row for row in prompt_rows for _ in range(int(generation["rollouts_per_qid"]))]
        expanded_prompts = [prompt for prompt in prompts for _ in range(int(generation["rollouts_per_qid"]))]
        # Reset only the sampled stream to the preregistered generation seed.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        sampled_generated = _generate(
            model, tokenizer, expanded_prompts,
            batch_size=args.batch_size,
            max_new_tokens=int(generation["max_new_tokens"]),
            do_sample=True,
            temperature=float(generation["sampled"]["temperature"]),
            top_p=float(generation["sampled"]["top_p"]),
        )

        candidates: list[dict[str, Any]] = []
        for row, generated in zip(prompt_rows, greedy_generated):
            scored = _score_candidate(
                row, label_rows[str(row["qid"])], generated,
                candidate_type="greedy", candidate_index=0,
                min_valid_steps=int(generation["min_valid_steps"]),
                min_reasoning_chars=int(generation["min_reasoning_chars"]),
            )
            scored["prompt_sha256"] = prompt_hashes[str(row["qid"])]
            candidates.append(scored)
        counters: Counter[str] = Counter()
        for row, generated in zip(expanded_rows, sampled_generated):
            qid = str(row["qid"])
            counters[qid] += 1
            scored = _score_candidate(
                row, label_rows[qid], generated,
                candidate_type="sampled", candidate_index=counters[qid] - 1,
                min_valid_steps=int(generation["min_valid_steps"]),
                min_reasoning_chars=int(generation["min_reasoning_chars"]),
            )
            scored["prompt_sha256"] = prompt_hashes[qid]
            candidates.append(scored)
        _write_jsonl(run_dir / "candidates.jsonl", candidates)
        summary = summarize_candidates(
            candidates,
            qids_expected=int(protocol["cohort"]["n"]),
            k=int(generation["rollouts_per_qid"]),
            gates=protocol["decision_gates"],
        )
        status = "PASS_HEADROOM_GATES" if summary["all_pass"] else "FAIL_RESELECT_OR_REQUOTA_ONLY"
        report = {
            "schema_version": "proof400-fill275-strong-sft-headroom-result-v1",
            "experiment_id": experiment_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "zero_update": True,
            "train_side_development_consumed": True,
            "summary": summary,
            "scientific_boundary": protocol["scientific_boundary"],
            "outputs": {"candidates": artifact_identity(run_dir / "candidates.jsonl")},
        }
        report_path = run_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        dump_manifest(run_dir, status=status, extra={**run_record, **report})
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception as exc:
        dump_manifest(run_dir, status="FAILED_RUNTIME", extra={
            **run_record,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
