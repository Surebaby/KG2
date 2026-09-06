#!/usr/bin/env python
"""Evaluate frozen SAEG N/P/W/F arms on development with a fixed checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import torch

from kgproweight.data.parsers import extract_final_answer
from kgproweight.data.prompts import build_saeg_inference_messages
from kgproweight.data.saeg_dataset import ARMS, assert_role_allowed, iter_saeg_eval_inputs, route_eval_arm
from kgproweight.data.saeg_parsers import parse_saeg_steps
from kgproweight.eval.metrics import compute_em, compute_f1
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir
from scripts.pilot.score_a1_fixed_context_kg import _bootstrap_ci, _mcnemar_exact


EVALUATOR_VERSION = "saeg-v1-development-utility-v2"
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
GENERATED_ARMS = ("A_no_graph", "B_passage", "D_fused")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0, "status": "NOT_EVALUABLE"}
    return {
        "n": n,
        "em": sum(float(row["em"]) for row in rows) / n,
        "f1": sum(float(row["f1"]) for row in rows) / n,
        "parse_rate": sum(bool(row["well_formed"]) for row in rows) / n,
        "citation_contract_rate": sum(bool(row["citation_contract_valid"]) for row in rows) / n,
        "known_wikidata_citation_rate": sum(int(row["known_wikidata_citations"]) > 0 for row in rows) / n,
        "known_passage_citation_rate": sum(int(row["known_passage_citations"]) > 0 for row in rows) / n,
        "mean_prompt_tokens": sum(int(row["prompt_tokens"]) for row in rows) / n,
    }


def paired(rows: Sequence[Mapping[str, Any]], left_arm: str, right_arm: str) -> dict[str, Any]:
    by_key = {(str(row["question_key"]), str(row["arm"])): row for row in rows}
    keys = [str(row["question_key"]) for row in rows if row["arm"] == left_arm]
    pairs = [
        (by_key[(key, left_arm)], by_key[(key, right_arm)])
        for key in keys if (key, right_arm) in by_key
    ]
    em_diffs = [float(right["em"]) - float(left["em"]) for left, right in pairs]
    f1_diffs = [float(right["f1"]) - float(left["f1"]) for left, right in pairs]
    gained = sum(float(right["em"]) > float(left["em"]) for left, right in pairs)
    lost = sum(float(right["em"]) < float(left["em"]) for left, right in pairs)
    return {
        "n": len(pairs),
        "left_arm": left_arm,
        "right_arm": right_arm,
        "left_em": sum(float(left["em"]) for left, _ in pairs) / max(1, len(pairs)),
        "right_em": sum(float(right["em"]) for _, right in pairs) / max(1, len(pairs)),
        "delta_em": sum(em_diffs) / max(1, len(em_diffs)),
        "delta_em_bootstrap_95ci": _bootstrap_ci(em_diffs, seed=20260903),
        "left_f1": sum(float(left["f1"]) for left, _ in pairs) / max(1, len(pairs)),
        "right_f1": sum(float(right["f1"]) for _, right in pairs) / max(1, len(pairs)),
        "delta_f1": sum(f1_diffs) / max(1, len(f1_diffs)),
        "delta_f1_bootstrap_95ci": _bootstrap_ci(f1_diffs, seed=20260904),
        "gained_correct": gained,
        "lost_correct": lost,
        "tied_correctness": len(pairs) - gained - lost,
        "net_correct": gained - lost,
        "mcnemar_exact_p": _mcnemar_exact(gained, lost),
        "left_parse_rate": sum(bool(left["well_formed"]) for left, _ in pairs) / max(1, len(pairs)),
        "right_parse_rate": sum(bool(right["well_formed"]) for _, right in pairs) / max(1, len(pairs)),
    }


def score_generation(
    row: Mapping[str, Any],
    routed: Mapping[str, Any],
    golds: Sequence[str],
    *,
    arm: str,
    generation: str,
    prompt_sha256: str,
    prompt_tokens: int,
) -> dict[str, Any]:
    known_passage_ids = [str(item["passage_id"]) for item in routed["passage_evidence"]]
    steps = parse_saeg_steps(
        generation,
        known_kg=routed["kg_triples"],
        known_passage_ids=known_passage_ids,
    )
    answer = (extract_final_answer(generation) or "").split("\n", 1)[0].strip()
    indices = [step.index for step in steps]
    contract_valid = bool(steps) and all(step.citation_contract_valid for step in steps)
    return {
        "question_key": str(row["question_key"]),
        "dataset": str(row["dataset"]),
        "qid": str(row["qid"]),
        "arm": arm,
        "fallback_no_graph": bool(routed["fallback_no_graph"]),
        "passage_evidence_count": len(routed["passage_evidence"]),
        "wikidata_triple_count": len(routed["kg_triples"]),
        "prompt_sha256": prompt_sha256,
        "prompt_tokens": prompt_tokens,
        "gold_answers": list(golds),
        "prediction": answer,
        "em": compute_em(answer, list(golds)) if answer else 0.0,
        "f1": compute_f1(answer, list(golds)) if answer else 0.0,
        "n_steps": len(steps),
        "well_formed": bool(answer and contract_valid and indices == list(range(1, len(indices) + 1))),
        "citation_contract_valid": contract_valid,
        "known_wikidata_citations": sum(len(step.cited_triples) for step in steps),
        "known_passage_citations": sum(len(step.cited_passage_ids) for step in steps),
        "generation": generation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="outputs/audits/saeg_v1_development_zero_training_protocol_v2/protocol.json",
    )
    parser.add_argument(
        "--run_dir",
        default="outputs/validation/saeg_v1_development_strong_sft_npdf_v2",
    )
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_DEVELOPMENT_GENERATION":
        raise ValueError("development utility protocol is not frozen")

    inputs = protocol["inputs"]
    eval_path = Path(inputs["eval_input"]["path"]).resolve()
    gold_path = Path(inputs["scorer_gold"]["path"]).resolve()
    adapter_path = Path(inputs["adapter"]["path"]).resolve()
    base_path = Path(inputs["base_model"]["path"]).resolve()
    if sha256_file(eval_path) != inputs["eval_input"]["sha256"]:
        raise ValueError("development input hash differs from protocol")
    if sha256_file(gold_path) != inputs["scorer_gold"]["sha256"]:
        raise ValueError("development Gold hash differs from protocol")
    if artifact_identity(adapter_path) != inputs["adapter"]:
        raise ValueError("adapter identity differs from protocol")
    if artifact_identity(base_path) != inputs["base_model"]:
        raise ValueError("base model identity differs from protocol")

    rows = list(iter_saeg_eval_inputs(eval_path))
    gold_rows = read_jsonl(gold_path)
    gold = {str(row["question_key"]): [str(x) for x in row["golden_answers"]] for row in gold_rows}
    for row in rows:
        assert_role_allowed(row)
    if set(gold) != {str(row["question_key"]) for row in rows}:
        raise ValueError("development Gold identity join is not 1.0")

    run_dir, experiment_id = prepare_new_run_dir(
        args.run_dir,
        experiment_id=protocol["experiment_id"],
        extra={"phase": "saeg_v1_development_zero_training_utility", "protocol_sha256": sha256_file(protocol_path)},
    )
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        generation_cfg = protocol["generation"]
        random.seed(int(generation_cfg["seed"]))
        torch.manual_seed(int(generation_cfg["seed"]))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing silent CPU fallback for SAEG development inference")
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        execution_device = str(model.device)
        if not execution_device.startswith("cuda"):
            raise RuntimeError(f"model loaded on {execution_device}; refusing silent CPU fallback")
        print(f"SAEG evaluator device={execution_device}", flush=True)

        predictions: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            prompt_cache: dict[str, dict[str, Any]] = {}
            for arm in GENERATED_ARMS:
                # A/B/D are a strict paired population.  An empty P source is a
                # fail-closed A-equivalent input, not a reason to drop the qid.
                routed = route_eval_arm(row, arm)
                messages = build_saeg_inference_messages(
                    question=routed["question"],
                    retrieved_passages=routed["retrieved_passages"],
                    kg_triples=routed["kg_triples"],
                    passage_evidence=routed["passage_evidence"],
                    top_k=int(generation_cfg["top_k_passages"]),
                    max_kg_triples=int(generation_cfg["max_wikidata_triples"]),
                    max_passage_evidence=int(generation_cfg["max_passage_evidence"]),
                )
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
                if prompt_sha in prompt_cache:
                    cached = deepcopy(prompt_cache[prompt_sha])
                    cached.update({"arm": arm, "reused_identical_prompt_from_arm": cached["arm"]})
                    predictions.append(cached)
                    continue
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    output = model.generate(
                        **encoded,
                        max_new_tokens=int(generation_cfg["max_new_tokens"]),
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                generation = tokenizer.decode(
                    output[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True
                )
                scored = score_generation(
                    row,
                    routed,
                    gold[str(row["question_key"])],
                    arm=arm,
                    generation=generation,
                    prompt_sha256=prompt_sha,
                    prompt_tokens=int(encoded["input_ids"].shape[1]),
                )
                predictions.append(scored)
                prompt_cache[prompt_sha] = scored
            print(f"SAEG development strong-SFT inference {index}/{len(rows)}", flush=True)

        predictions_path = run_dir / "predictions.jsonl"
        write_jsonl(predictions_path, predictions)
        expected_predictions = len(rows) * len(GENERATED_ARMS)
        if len(predictions) != expected_predictions:
            raise ValueError(
                f"strict paired population broken: {len(predictions)} != {expected_predictions}"
            )
        arm_keys = {
            arm: {str(row["question_key"]) for row in predictions if row["arm"] == arm}
            for arm in GENERATED_ARMS
        }
        if len(set(map(frozenset, arm_keys.values()))) != 1:
            raise ValueError("A/B/D question-key populations differ")
        by_dataset: dict[str, Any] = {}
        for dataset in DATASETS:
            current = [row for row in predictions if row["dataset"] == dataset]
            by_dataset[dataset] = {
                "arms": {
                    arm: aggregate([row for row in current if row["arm"] == arm])
                    for arm in sorted(ARMS)
                },
                "D_minus_A": paired(current, "A_no_graph", "D_fused"),
            }
        overall = {
            "arms": {
                arm: aggregate([row for row in predictions if row["arm"] == arm])
                for arm in sorted(ARMS)
            },
            "D_minus_A": paired(predictions, "A_no_graph", "D_fused"),
        }
        covered = [
            row for row in predictions
            if row["arm"] in {"A_no_graph", "D_fused"}
            and next(x for x in rows if x["question_key"] == row["question_key"])["source_status"]["passage"] == "nonempty"
        ]
        fallback_pairs = [
            (left, right)
            for key in {row["question_key"] for row in predictions if row["arm"] == "A_no_graph"}
            for left in [next(row for row in predictions if row["question_key"] == key and row["arm"] == "A_no_graph")]
            for right in [next(row for row in predictions if row["question_key"] == key and row["arm"] == "D_fused")]
            if right["fallback_no_graph"]
        ]
        covered_pair = paired(covered, "A_no_graph", "D_fused")
        fallback_exact = all(
            left["prompt_sha256"] == right["prompt_sha256"]
            and left["generation"] == right["generation"]
            and left["prediction"] == right["prediction"]
            for left, right in fallback_pairs
        )
        macro_delta_em = sum(by_dataset[name]["D_minus_A"]["delta_em"] for name in DATASETS) / 3
        macro_delta_f1 = sum(by_dataset[name]["D_minus_A"]["delta_f1"] for name in DATASETS) / 3
        gate_cfg = protocol["decision_gates"]
        gates = {
            "macro_delta_em_positive": macro_delta_em > 0,
            "macro_delta_f1_positive": macro_delta_f1 > 0,
            "at_least_two_datasets_positive_em": sum(
                by_dataset[name]["D_minus_A"]["delta_em"] > 0 for name in DATASETS
            ) >= int(gate_cfg["min_datasets_with_positive_delta_em"]),
            "no_dataset_net_loss_gt_2": all(
                by_dataset[name]["D_minus_A"]["net_correct"]
                >= -int(gate_cfg["max_net_correct_loss_per_dataset"])
                for name in DATASETS
            ),
            "covered_subset_em_positive": covered_pair["delta_em"] > 0,
            "parse_not_degraded": all(
                by_dataset[name]["D_minus_A"]["right_parse_rate"]
                >= by_dataset[name]["D_minus_A"]["left_parse_rate"]
                - float(gate_cfg["max_parse_rate_drop_per_dataset"])
                for name in DATASETS
            ),
            "fallback_exact": fallback_exact,
            "wikidata_not_evaluable_reported": overall["arms"]["C_wikidata"]["status"] == "NOT_EVALUABLE",
        }
        status = "PASS_ZERO_TRAINING_UTILITY" if all(gates.values()) else "FAIL_STOP_BEFORE_SFT"
        report = {
            "schema_version": "saeg-development-zero-training-utility-result-v2",
            "experiment_id": experiment_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "evaluator_version": EVALUATOR_VERSION,
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            "by_dataset": by_dataset,
            "overall": overall,
            "covered_D_minus_A": covered_pair,
            "fallback": {"n": len(fallback_pairs), "exact": fallback_exact},
            "macro_delta_em": macro_delta_em,
            "macro_delta_f1": macro_delta_f1,
            "gates": {"checks": gates, "all_pass": all(gates.values())},
            "inputs": {"adapter": artifact_identity(adapter_path), "base_model": artifact_identity(base_path)},
            "outputs": {"predictions": {"path": str(predictions_path), "sha256": sha256_file(predictions_path)}},
            "scientific_boundary": (
                "Development-only zero-training evidence. C/W is NOT_EVALUABLE because fresh Wikidata supply failed "
                "its frozen structural gate; D therefore equals B on this cohort. Confirmation remains sealed."
            ),
        }
        (run_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        dump_manifest(run_dir, extra={"phase": "saeg_v1_development_zero_training_utility", **report}, status=status)
        print(json.dumps({
            "status": status,
            "macro_delta_em": macro_delta_em,
            "macro_delta_f1": macro_delta_f1,
            "gates": gates,
            "by_dataset": {name: value["D_minus_A"] for name, value in by_dataset.items()},
        }, ensure_ascii=False, indent=2))
    except BaseException as exc:
        dump_manifest(
            run_dir,
            extra={"phase": "saeg_v1_development_zero_training_utility", "failure": {"type": type(exc).__name__, "message": str(exc)}},
            status="ABORTED" if isinstance(exc, KeyboardInterrupt) else "FAILED_RUNTIME",
        )
        raise


if __name__ == "__main__":
    main()
