#!/usr/bin/env python
"""Phase 2 CLI — train the PRM head + α-gate jointly with real logprobs."""

from __future__ import annotations

import argparse
from pathlib import Path

from kgproweight.config import ProjectConfig, load_config
from kgproweight.training.phase2_prm import Phase2Config, run_phase2
from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import checkpoint_dir, data_dir

configure_logging("INFO")
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--silver_data", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--base_model", default="llama3-8B-instruct")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--no_lora", action="store_true")
    p.add_argument("--no_text_head", action="store_true")
    p.add_argument("--binary_labels_only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        cfg_doc = load_config(args.config, validate=ProjectConfig)
        tcfg = cfg_doc.training
        silver = args.silver_data or str(Path(data_dir()) / "silver_data" / "silver_trajectories.jsonl")
        out_dir = args.output_dir or str(Path(checkpoint_dir()) / "prm_alpha_gate")
        p2 = Phase2Config(
            silver_path=silver,
            output_dir=out_dir,
            base_model=tcfg.base_model,
            dtype=tcfg.dtype,
            seed=tcfg.seed,
            epochs=tcfg.prm_epochs,
            batch_size=tcfg.prm_batch_size,
            grad_accum=tcfg.prm_grad_accum,
            lr=tcfg.prm_lr,
            max_length=tcfg.prm_max_length,
            use_lora=not args.no_lora,
            lora_r=tcfg.lora_r,
            lora_alpha=tcfg.lora_alpha,
            lora_dropout=tcfg.lora_dropout,
            calibration_weight=cfg_doc.reward.alpha_gate.calibration_weight,
            train_text_reward_head=not args.no_text_head,
            binary_labels_only=args.binary_labels_only,
        )
    else:
        silver = args.silver_data or str(Path(data_dir()) / "silver_data" / "silver_trajectories.jsonl")
        out_dir = args.output_dir or str(Path(checkpoint_dir()) / "prm_alpha_gate")
        p2 = Phase2Config(
            silver_path=silver,
            output_dir=out_dir,
            base_model=args.base_model,
            dtype=args.dtype,
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            use_lora=not args.no_lora,
            train_text_reward_head=not args.no_text_head,
            binary_labels_only=args.binary_labels_only,
        )

    result = run_phase2(p2)
    logger.info("Phase 2 result: %s", result)


if __name__ == "__main__":
    main()
