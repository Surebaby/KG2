#!/usr/bin/env python
"""Generate query plans from a frozen question-only cohort without gold access."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from kgproweight.eval.query_planner import parse_plan, plan_validation_errors
from kgproweight.training.query_planner import planner_messages
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir
from kgproweight.utils.paths import model_path


_PROHIBITED = {
    "answer", "answers", "golden_answers", "target", "supporting_facts",
    "evidences", "question_decomposition", "paragraph_text",
}


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _assert_question_only(value: Any, *, location: str = "row") -> None:
    if isinstance(value, Mapping):
        present = _PROHIBITED.intersection(str(key) for key in value)
        if present:
            raise ValueError(f"prohibited runtime fields at {location}: {sorted(present)}")
        for key, child in value.items():
            _assert_question_only(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_question_only(child, location=f"{location}[{index}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument(
        "--scope",
        default=None,
        help="Auditable cohort description; defaults to the observed row count.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("query planner generation requires CUDA")

    import yaml
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cohort_path = Path(args.cohort).resolve()
    records = list(_read_jsonl(cohort_path))
    if not records:
        raise SystemExit("empty cohort")
    for row in records:
        _assert_question_only(row)
    if len({str(row["question_key"]) for row in records}) != len(records):
        raise SystemExit("duplicate question_key in cohort")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_dir, experiment_id = prepare_new_run_dir(
        args.output_dir,
        experiment_id=args.experiment_id,
        extra={
            "phase": "generate_question_only_query_plans",
            "cohort": artifact_identity(cohort_path),
            "adapter": artifact_identity(args.adapter),
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

    generated_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    planner_messages(row, include_target=False),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch
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
            for row, generated_text in zip(batch, texts):
                parsed, parse_error = parse_plan(generated_text)
                validation_record = {
                    "schema_version": "query-planner-supervision-1",
                    "question_key": row["question_key"],
                    "dataset": row["dataset"],
                    "qid": row["qid"],
                    "question": row["question"],
                    "question_sha256": row["question_sha256"],
                    "target_type": row["target_type"],
                }
                errors = (
                    plan_validation_errors(validation_record, parsed)
                    if parsed is not None else [str(parse_error)]
                )
                generated_rows.append(
                    {
                        "row_id": row["row_id"],
                        "question_key": row["question_key"],
                        "dataset": row["dataset"],
                        "qid": row["qid"],
                        "question": row["question"],
                        "question_sha256": row["question_sha256"],
                        "generated_text": generated_text,
                        "predicted_target": parsed,
                        "schema_valid": not errors,
                        "validation_errors": errors,
                        "gold_access": False,
                    }
                )
            print(f"generated {len(generated_rows)}/{len(records)}", flush=True)

    predictions_path = output_dir / "predictions.question_only.jsonl"
    with predictions_path.open("x", encoding="utf-8") as handle:
        for row in generated_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    valid = sum(bool(row["schema_valid"]) for row in generated_rows)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "status": "RUNTIME_PLANS_FROZEN_NO_GOLD_AUDIT",
        "scope": args.scope
        or f"question-only planner generation; n={len(generated_rows)}; zero training",
        "generation": {
            "greedy": True,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        },
        "counts": {
            "n": len(generated_rows),
            "by_dataset": dict(Counter(str(row["dataset"]) for row in generated_rows)),
            "schema_valid": valid,
        },
        "rates": {"schema_valid": valid / len(generated_rows)},
        "inputs": {
            "cohort": artifact_identity(cohort_path),
            "adapter": artifact_identity(args.adapter),
            "config": artifact_identity(args.config),
            "protocol": artifact_identity(args.protocol),
        },
        "outputs": {"predictions": artifact_identity(predictions_path)},
        "gold_access": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(output_dir, status=report["status"], extra=report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
