#!/usr/bin/env python
"""Train or dry-run the answer-free learned query planner."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json

from transformers import AutoTokenizer

from kgproweight.training.query_planner import (
    load_smoke_config,
    prepare_data,
    run_query_planner_sft,
)
from kgproweight.utils.paths import model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--probe_steps", type=int)
    parser.add_argument("--train_per_dataset", type=int)
    parser.add_argument("--dev_per_dataset", type=int)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.probe_steps is not None and args.probe_steps <= 0:
        raise SystemExit("--probe_steps must be positive")
    cfg = load_smoke_config(
        args.config, output_override=args.output_dir, max_steps=args.probe_steps
    )
    cfg = replace(
        cfg,
        train_per_dataset=args.train_per_dataset or cfg.train_per_dataset,
        dev_per_dataset=args.dev_per_dataset or cfg.dev_per_dataset,
    )
    if args.dry_run:
        tokenizer = AutoTokenizer.from_pretrained(model_path(cfg.base_model))
        _, _, report = prepare_data(cfg, tokenizer)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.probe_steps is not None and not args.output_dir:
        raise SystemExit("probe requires a unique --output_dir; the formal smoke path is reserved")
    result = run_query_planner_sft(cfg, probe=args.probe_steps is not None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
