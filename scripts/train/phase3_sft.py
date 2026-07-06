#!/usr/bin/env python
"""Phase 3a CLI — supervised fine-tuning of the Student."""

from __future__ import annotations

import argparse
from pathlib import Path

from kgproweight.config import ProjectConfig, load_config
from kgproweight.training.phase3_sft import Phase3SFTConfig, run_phase3_sft
from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import checkpoint_dir, data_dir

configure_logging("INFO")
logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--silver_data", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--base_model", default="llama3-8B-instruct")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--no_lora", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.config:
        cfg_doc = load_config(args.config, validate=ProjectConfig)
        tcfg = cfg_doc.training
        silver = args.silver_data or str(
            Path(checkpoint_dir()) / "prm_alpha_gate" / "silver_with_logprobs.jsonl"
        )
        out_dir = args.output_dir or str(Path(checkpoint_dir()) / "sft_student")
        cfg = Phase3SFTConfig(
            silver_path=silver,
            output_dir=out_dir,
            base_model=tcfg.base_model,
            dtype=tcfg.dtype,
            seed=tcfg.seed,
            epochs=tcfg.sft_epochs,
            batch_size=tcfg.sft_batch_size,
            grad_accum=tcfg.sft_grad_accum,
            lr=tcfg.sft_lr,
            max_length=tcfg.sft_max_length,
            use_lora=not args.no_lora,
            lora_r=tcfg.lora_r,
            lora_alpha=tcfg.lora_alpha,
            lora_dropout=tcfg.lora_dropout,
        )
    else:
        silver = args.silver_data or str(
            Path(checkpoint_dir()) / "prm_alpha_gate" / "silver_with_logprobs.jsonl"
        )
        if not Path(silver).exists():
            silver = str(Path(data_dir()) / "silver_data" / "silver_trajectories.jsonl")
        out_dir = args.output_dir or str(Path(checkpoint_dir()) / "sft_student")
        cfg = Phase3SFTConfig(
            silver_path=silver,
            output_dir=out_dir,
            base_model=args.base_model,
            dtype=args.dtype,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            use_lora=not args.no_lora,
        )

    result = run_phase3_sft(cfg)
    logger.info("Phase 3a result: %s", result)


if __name__ == "__main__":
    main()
