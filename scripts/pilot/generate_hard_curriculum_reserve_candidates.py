#!/usr/bin/env python
"""Generate one greedy and K=4 sampled candidates on the frozen reserve."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import torch

from kgproweight.data.prompts import build_inference_messages
from kgproweight.eval.metrics import compute_em, compute_f1
from kgproweight.reward.proofkg_process_v2 import build_execution_trace, score_proofkg_v2
from kgproweight.utils.logging import dump_manifest, prepare_new_run_dir


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prompt(tokenizer, row: Mapping[str, Any]) -> str:
    messages = build_inference_messages(
        question=str(row["question"]),
        retrieved_passages=list(row["retrieved_passages"]),
        kg_triples=list(row["kg_subgraph"]),
        top_k=10,
        max_kg_triples=12,
    )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    proof_path = Path(protocol["outputs"]["reserve_proof"]["path"])
    kg_path = Path(protocol["outputs"]["reserve_kg"]["path"])
    runtime_path = Path(protocol["outputs"]["reserve_runtime"]["path"])
    for name, path in (("reserve_proof", proof_path), ("reserve_kg", kg_path), ("reserve_runtime", runtime_path)):
        if _sha256(path) != protocol["outputs"][name]["sha256"]:
            raise SystemExit(f"frozen input hash mismatch: {name}")
    generation = protocol["generation"]
    adapter = Path(generation["checkpoint"])
    base_model = Path(generation["base_model"])
    if _sha256(adapter / "adapter_model.safetensors") != generation["adapter_model_sha256"]:
        raise SystemExit("adapter model hash mismatch")
    if _sha256(adapter / "adapter_config.json") != generation["adapter_config_sha256"]:
        raise SystemExit("adapter config hash mismatch")
    if _sha256(base_model / "config.json") != generation["base_config_sha256"]:
        raise SystemExit("base config hash mismatch")
    if _sha256(base_model / "model.safetensors.index.json") != generation["base_index_sha256"]:
        raise SystemExit("base model index hash mismatch")

    proof = _read_jsonl(proof_path)
    kg_by_qid = {str(row["qid"]): row for row in _read_jsonl(kg_path)}
    runtime_by_qid = {str(row["qid"]): row for row in _read_jsonl(runtime_path)}
    if len(proof) != protocol["reserve"]["n"]:
        raise SystemExit("reserve row count mismatch")

    run_dir, experiment_id = prepare_new_run_dir(
        args.run_dir,
        experiment_id=protocol["experiment_id"] + "-RESERVE82-K4",
        extra={
            "phase": "hard_curriculum_reserve_candidate_generation",
            "protocol_sha256": _sha256(args.protocol),
            "generator_sha256": _sha256(Path(__file__)),
        },
    )
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        seed = int(generation["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()

        traces: dict[str, list[dict[str, Any]]] = {}
        planned: dict[str, int] = {}
        prompts: dict[str, str] = {}
        for row in proof:
            qid = str(row["qid"])
            kg = kg_by_qid[qid]
            plan = kg.get("query_plan") or {}
            planned[qid] = len(plan.get("hops") or [])
            traces[qid] = build_execution_trace(
                plan, (runtime_by_qid[qid].get("execution") or {})
            )
            prompts[qid] = _prompt(tokenizer, row)

        def score_one(row: Mapping[str, Any], text: str, candidate_type: str, candidate_index: int) -> dict[str, Any]:
            qid = str(row["qid"])
            kg = kg_by_qid[qid]
            process = score_proofkg_v2(
                question=str(row["question"]), generation=text,
                kg_triples=kg.get("kg_subgraph") or [],
                execution_trace=traces[qid], planned_hops=planned[qid],
            )
            golds = [str(value) for value in row.get("gold_answers") or [] if str(value).strip()]
            return {
                "qid": qid,
                "question": row["question"],
                "candidate_type": candidate_type,
                "candidate_index": candidate_index,
                "generation": text,
                "process": process,
                "em": compute_em(process["prediction"], golds) if process["prediction"] and golds else 0.0,
                "f1": compute_f1(process["prediction"], golds) if process["prediction"] and golds else 0.0,
            }

        candidates: list[dict[str, Any]] = []
        specs = [
            (row, "greedy", 0) for row in proof
        ] + [
            (row, "sampled", index)
            for row in proof for index in range(int(generation["sampled_per_qid"]))
        ]
        batch_size = 2
        for start in range(0, len(specs), batch_size):
            batch = specs[start : start + batch_size]
            encoded = tokenizer(
                [prompts[str(row["qid"])] for row, _, _ in batch],
                return_tensors="pt", add_special_tokens=False, padding=True,
            ).to(model.device)
            prompt_len = int(encoded["input_ids"].shape[1])
            is_sampled = batch[0][1] == "sampled"
            if any((kind == "sampled") != is_sampled for _, kind, _ in batch):
                raise RuntimeError("greedy/sampled boundary must align to the batch")
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    max_new_tokens=int(generation["max_new_tokens"]),
                    do_sample=is_sampled,
                    temperature=float(generation["temperature"]) if is_sampled else None,
                    top_p=float(generation["top_p"]) if is_sampled else None,
                    top_k=int(generation["top_k"]) if is_sampled else None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            texts = tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True)
            for (row, kind, index), text in zip(batch, texts):
                candidates.append(score_one(row, text, kind, index))
            done = min(start + len(batch), len(specs))
            if done % 20 == 0 or done == len(specs):
                print(f"reserve candidates {done}/{len(specs)}", flush=True)

        candidate_path = run_dir / "candidates.jsonl"
        _write_jsonl(candidate_path, candidates)
        report = {
            "schema_version": "hard-curriculum-reserve-generation-report-1",
            "experiment_id": experiment_id,
            "status": "COMPLETE_NOT_SCORED",
            "n_qids": len(proof),
            "n_greedy": sum(row["candidate_type"] == "greedy" for row in candidates),
            "n_sampled": sum(row["candidate_type"] == "sampled" for row in candidates),
            "runtime_errors": 0,
            "protocol": {"path": str(args.protocol), "sha256": _sha256(args.protocol)},
            "candidate_sha256": _sha256(candidate_path),
            "generator_sha256": _sha256(Path(__file__)),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        report_path = run_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        dump_manifest(run_dir, status=report["status"], extra={
            "experiment_id": experiment_id,
            "phase": "hard_curriculum_reserve_candidate_generation",
            "report_sha256": _sha256(report_path),
        })
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception as exc:
        dump_manifest(run_dir, status="FAILED_RUNTIME", extra={
            "experiment_id": experiment_id,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
        })
        raise


if __name__ == "__main__":
    main()
