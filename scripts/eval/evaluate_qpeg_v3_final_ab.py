#!/usr/bin/env python
"""Run approved QPEG-v3 final A/B, reusing only exact historical A prompts."""

from __future__ import annotations

import argparse
from collections import defaultdict
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


ARMS = ("no_graph", "qpeg_v3")


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


def _without_kg(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "kg_subgraph"}


def _paired(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {(str(row["question_key"]), str(row["arm"])): row for row in rows}
    keys = [str(row["question_key"]) for row in rows if row["arm"] == "no_graph"]
    pairs = [(by_key[(key, "no_graph")], by_key[(key, "qpeg_v3")]) for key in keys]
    em_diffs = [float(right["em"]) - float(left["em"]) for left, right in pairs]
    f1_diffs = [float(right["f1"]) - float(left["f1"]) for left, right in pairs]
    gained = sum(left["em"] < right["em"] for left, right in pairs)
    lost = sum(left["em"] > right["em"] for left, right in pairs)
    return {
        "n": len(pairs),
        "no_graph_em": sum(float(left["em"]) for left, _ in pairs) / max(1, len(pairs)),
        "qpeg_v3_em": sum(float(right["em"]) for _, right in pairs) / max(1, len(pairs)),
        "delta_em": sum(em_diffs) / max(1, len(em_diffs)),
        "delta_em_bootstrap_95ci": _bootstrap_ci(em_diffs, seed=20260903),
        "no_graph_f1": sum(float(left["f1"]) for left, _ in pairs) / max(1, len(pairs)),
        "qpeg_v3_f1": sum(float(right["f1"]) for _, right in pairs) / max(1, len(pairs)),
        "delta_f1": sum(f1_diffs) / max(1, len(f1_diffs)),
        "delta_f1_bootstrap_95ci": _bootstrap_ci(f1_diffs, seed=20260904),
        "gained_correct": gained,
        "lost_correct": lost,
        "tied_correctness": len(pairs) - gained - lost,
        "net_correct": gained - lost,
        "mcnemar_exact_p": _mcnemar_exact(gained, lost),
        "no_graph_parse_rate": sum(bool(left["well_formed"]) for left, _ in pairs) / max(1, len(pairs)),
        "qpeg_v3_parse_rate": sum(bool(right["well_formed"]) for _, right in pairs) / max(1, len(pairs)),
    }


def _decision_gates(by_dataset: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, bool]:
    frozen = protocol["decision_gates"]
    macro_em = sum(value["paired"]["delta_em"] for value in by_dataset.values()) / 3
    macro_f1 = sum(value["paired"]["delta_f1"] for value in by_dataset.values()) / 3
    return {
        "macro_delta_em_positive": macro_em > float(frozen["macro_delta_em_gt"]),
        "macro_delta_f1_positive": macro_f1 > float(frozen["macro_delta_f1_gt"]),
        "no_dataset_net_loss_gt_6": all(
            value["paired"]["net_correct"] >= -int(frozen["max_net_correct_loss_per_dataset"])
            for value in by_dataset.values()
        ),
        "parse_rate_drop_le_0_01": all(
            value["paired"]["qpeg_v3_parse_rate"]
            >= value["paired"]["no_graph_parse_rate"] - float(frozen["max_parse_rate_drop"])
            for value in by_dataset.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--experiment_id", required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_INPUTS_NOT_EVALUATED":
        raise ValueError("final A/B protocol is not frozen and unopened")
    if not protocol.get("researcher_approval_id"):
        raise ValueError("final A/B protocol lacks researcher approval identifier")

    arm_paths = {
        "no_graph": Path(protocol["inputs"]["arm_no_graph"]["path"]),
        "qpeg_v3": Path(protocol["inputs"]["arm_qpeg_v3"]["path"]),
    }
    arm_rows = {arm: _read_jsonl(path) for arm, path in arm_paths.items()}
    for arm, path in arm_paths.items():
        if _sha256(path) != protocol["inputs"][f"arm_{arm}"]["sha256"]:
            raise ValueError(f"{arm} input hash mismatch")
        if len(arm_rows[arm]) != 900:
            raise ValueError(f"{arm} does not contain 900 rows")
    if any(_without_kg(left) != _without_kg(right) for left, right in zip(arm_rows["no_graph"], arm_rows["qpeg_v3"])):
        raise ValueError("paired inputs differ outside QPEG block")
    qid_order = [str(row["question_key"]) for row in arm_rows["no_graph"]]
    if hashlib.sha256("\n".join(qid_order).encode()).hexdigest() != protocol["qid_order_sha256"]:
        raise ValueError("final qid order hash mismatch")

    adapter = Path(protocol["models"]["strong_sft"]["path"])
    base_model = Path(protocol["models"]["base_model"]["path"])
    if _sha256(adapter / "adapter_config.json") != protocol["models"]["strong_sft"]["adapter_config_sha256"]:
        raise ValueError("adapter config hash mismatch")
    if _sha256(adapter / "adapter_model.safetensors") != protocol["models"]["strong_sft"]["adapter_model_sha256"]:
        raise ValueError("adapter model hash mismatch")
    for name in ("config", "model_index", "tokenizer"):
        filename = {"config": "config.json", "model_index": "model.safetensors.index.json", "tokenizer": "tokenizer.json"}[name]
        if _sha256(base_model / filename) != protocol["models"]["base_model"][f"{name}_sha256"]:
            raise ValueError(f"base model {name} hash mismatch")

    run_dir, experiment_id = prepare_new_run_dir(
        args.run_dir,
        experiment_id=args.experiment_id,
        extra={"phase": "qpeg_v3_final300x3_matched_ab", "protocol_sha256": _sha256(args.protocol)},
    )
    predictions_path = run_dir / "predictions.jsonl"
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        historical: dict[tuple[str, str], dict[str, Any]] = {}
        reuse_assets = protocol["inputs"]["historical_no_graph_predictions"]
        for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
            asset = reuse_assets[dataset]["predictions"]
            path = Path(asset["path"])
            if _sha256(path) != asset["sha256"]:
                raise ValueError(f"historical A prediction hash mismatch: {dataset}")
            for row in _read_jsonl(path):
                historical[(dataset, str(row["qid"]))] = row

        predictions: list[dict[str, Any]] = []
        for row in arm_rows["no_graph"]:
            prior = historical[(str(row["dataset"]), str(row["qid"]))]
            messages = build_rl_messages(
                question=str(row["question"]), retrieved_passages=list(row["retrieved_passages"]),
                kg_triples=[], top_k=10,
            )
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            if prompt_sha != prior.get("prompt_sha256"):
                raise ValueError(f"historical A prompt mismatch: {row['question_key']}")
            rebound = _score_generation(
                row=row, generation=str(prior["generation"]), prompt_sha256=prompt_sha,
                prompt_tokens=len(tokenizer(prompt, add_special_tokens=False)["input_ids"]),
                model_label="strong_sft", arm="no_graph",
                input_sha256=protocol["inputs"]["arm_no_graph"]["sha256"],
            )
            rebound["question_key"] = row["question_key"]
            rebound["qpeg_edge_count"] = row["qpeg_edge_count"]
            rebound["reused_generation"] = True
            rebound["historical_prediction"] = prior["prediction"]
            rebound["historical_em"] = prior["em"]
            rebound["historical_f1"] = prior["f1"]
            rebound["historical_prediction_exact"] = rebound["prediction"] == prior["prediction"]
            rebound["historical_em_exact"] = float(rebound["em"]) == float(prior["em"])
            rebound["historical_f1_exact"] = abs(float(rebound["f1"]) - float(prior["f1"])) <= 1e-12
            predictions.append(rebound)

        seed = int(protocol["generation"]["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        for index, row in enumerate(arm_rows["qpeg_v3"]):
            messages = build_rl_messages(
                question=str(row["question"]), retrieved_passages=list(row["retrieved_passages"]),
                kg_triples=list(row["kg_subgraph"]), top_k=10,
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
                row=row, generation=generation,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                prompt_tokens=int(encoded["input_ids"].shape[1]), model_label="strong_sft",
                arm="qpeg_v3", input_sha256=protocol["inputs"]["arm_qpeg_v3"]["sha256"],
            )
            scored["question_key"] = row["question_key"]
            scored["qpeg_edge_count"] = row["qpeg_edge_count"]
            predictions.append(scored)
            print(f"QPEG-v3 final B inference {index + 1}/900", flush=True)
        _write_jsonl(predictions_path, predictions)

        by_dataset: dict[str, Any] = {}
        for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
            current = [row for row in predictions if row["dataset"] == dataset]
            paired = _paired(current)
            nonempty_keys = {
                str(row["question_key"]) for row in arm_rows["qpeg_v3"]
                if row["dataset"] == dataset and row["kg_subgraph"]
            }
            by_dataset[dataset] = {
                "by_arm": {
                    arm: _aggregate([row for row in current if row["arm"] == arm]) for arm in ARMS
                },
                "paired": paired,
                "strata": {
                    "nonempty": _paired([row for row in current if row["question_key"] in nonempty_keys]),
                    "empty": _paired([row for row in current if row["question_key"] not in nonempty_keys]),
                },
            }
        gates = _decision_gates(by_dataset, protocol)
        macro_delta_em = sum(value["paired"]["delta_em"] for value in by_dataset.values()) / 3
        macro_delta_f1 = sum(value["paired"]["delta_f1"] for value in by_dataset.values()) / 3
        report = {
            "schema_version": "qpeg-v3-final-matched-ab-result-v1",
            "experiment_id": experiment_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS_ADVANCE_TO_SFT_DATA_PROPOSAL" if all(gates.values()) else "FAIL_STOP_FINAL",
            "protocol": {"path": str(args.protocol), "sha256": _sha256(args.protocol)},
            "inputs": {
                "base_model": artifact_identity(base_model),
                "adapter": artifact_identity(adapter),
                "historical_a_reused": True,
                "historical_a_reuse_semantics": "exact generation reused; both arms canonical-rescored",
                "new_generations": 900,
            },
            "by_dataset": by_dataset,
            "macro_delta_em": macro_delta_em,
            "macro_delta_f1": macro_delta_f1,
            "gates": {"checks": gates, "all_pass": all(gates.values())},
            "outputs": {"predictions": {"path": str(predictions_path), "sha256": _sha256(predictions_path)}},
            "scientific_boundary": (
                "Single-seed n300 matched-passage final evidence. Passing engineering gates does not imply "
                "statistical significance; no post-final selector tuning is permitted."
            ),
        }
        report_path = run_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        dump_manifest(run_dir, extra={"phase": "qpeg_v3_final_matched_ab", **report}, status=report["status"])
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception as exc:
        dump_manifest(
            run_dir,
            extra={"phase": "qpeg_v3_final_matched_ab", "failure": {"type": type(exc).__name__, "message": str(exc)}},
            status="FAILED_RUNTIME",
        )
        raise


if __name__ == "__main__":
    main()
