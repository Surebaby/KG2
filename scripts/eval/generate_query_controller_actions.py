#!/usr/bin/env python
"""Greedily generate and mechanically validate Query Controller actions."""

from __future__ import annotations

import argparse
import json

from kgproweight.eval.query_controller_runner import run_greedy_controller


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Frozen canonical action records")
    parser.add_argument("--adapter", required=True, help="Controller LoRA final directory")
    parser.add_argument("--protocol", required=True, help="Frozen parent training protocol JSON")
    parser.add_argument(
        "--eval_protocol", required=True, help="Frozen eval-only successor protocol JSON"
    )
    parser.add_argument("--training_manifest", required=True, help="Completed probe manifest")
    parser.add_argument("--expected_protocol_sha256", required=True)
    parser.add_argument("--expected_eval_protocol_sha256", required=True)
    parser.add_argument("--expected_training_manifest_sha256", required=True)
    parser.add_argument("--expected_adapter_sha256", required=True)
    parser.add_argument("--cohort_role", required=True, choices=("dev",))
    parser.add_argument("--output_dir", required=True, help="New append-only run directory")
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--base_model", default="llama3-8B-instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_input_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--load_in_4bit", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    result = run_greedy_controller(
        input_path=args.input,
        adapter_path=args.adapter,
        protocol_path=args.protocol,
        eval_protocol_path=args.eval_protocol,
        training_manifest_path=args.training_manifest,
        expected_protocol_sha256=args.expected_protocol_sha256,
        expected_eval_protocol_sha256=args.expected_eval_protocol_sha256,
        expected_training_manifest_sha256=args.expected_training_manifest_sha256,
        expected_adapter_sha256=args.expected_adapter_sha256,
        cohort_role=args.cohort_role,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
        base_model=args.base_model,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
