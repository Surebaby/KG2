#!/usr/bin/env python
"""Generate and score learned query plans on frozen dev or confirmation data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from kgproweight.eval.query_planner import (
    build_scored_row,
    evaluate_gates,
    load_source_rows,
    resolve_dev_gates,
    score_predictions,
)
from kgproweight.training.query_planner import balanced_sample, planner_messages
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir
from kgproweight.utils.paths import model_path


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--split", choices=["dev", "confirmation"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--per_dataset", type=int, help="Dev-only engineering subset override")
    parser.add_argument("--unlock_confirmation", action="store_true")
    args = parser.parse_args()
    if args.split == "confirmation" and not args.unlock_confirmation:
        raise SystemExit("confirmation is locked; pass --unlock_confirmation only after dev gates PASS")
    if args.split == "confirmation" and args.per_dataset is not None:
        raise SystemExit("confirmation must always be evaluated in full")
    if not torch.cuda.is_available():
        raise SystemExit("query planner evaluation requires CUDA")

    import yaml
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    split_path = Path(config["data"]["split_root"]) / f"{args.split}.jsonl"
    if args.split == "dev":
        records = balanced_sample(
            split_path,
            per_dataset=int(args.per_dataset or config["data"]["sampling"]["dev_per_dataset"]),
            seed=int(config["data"]["sampling"]["seed"]),
        )
        gates = resolve_dev_gates(protocol)
    else:
        records = list(_read_jsonl(split_path))
        gates = protocol["independent_confirmation_gates"]

    output_dir, experiment_id = prepare_new_run_dir(
        args.output_dir,
        extra={
            "phase": "query_planner_structural_evaluation",
            "split": args.split,
            "adapter": artifact_identity(args.adapter),
            "split_data": artifact_identity(split_path),
            "protocol": artifact_identity(args.protocol),
        },
    )
    base_id = model_path(str(config["model"]["base_model"]))
    dtype = torch.bfloat16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=dtype,
        quantization_config=quantization,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    sources = load_source_rows(args.data_root)
    scored = []
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    planner_messages(record, include_target=False),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for record in batch
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for record, text in zip(batch, texts):
                scored.append(build_scored_row(record, text, sources.get(record["question_key"])))

    metrics = score_predictions(scored)
    gate_result = evaluate_gates(metrics, gates)
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("x", encoding="utf-8") as fh:
        for row in scored:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "status": "PASS" if gate_result["pass"] else "FAIL_STOP",
        "split": args.split,
        "decode": {"greedy": True, "max_new_tokens": args.max_new_tokens},
        "metrics": metrics,
        "gates": gate_result,
        "inputs": {
            "adapter": artifact_identity(args.adapter),
            "split_data": artifact_identity(split_path),
            "protocol": artifact_identity(args.protocol),
        },
        "predictions": artifact_identity(predictions_path),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(output_dir, status=report["status"], extra=report)
    print(json.dumps({"status": report["status"], "metrics": metrics, "gates": gate_result}, indent=2))


if __name__ == "__main__":
    main()
