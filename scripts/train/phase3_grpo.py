#!/usr/bin/env python
"""Phase 3b alternative — GRPO (fallback for memory-constrained machines)."""

from __future__ import annotations

import argparse
from pathlib import Path

from kgproweight.training.phase3_grpo import Phase3GRPOConfig, run_phase3_grpo
from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import checkpoint_dir

configure_logging("INFO")
logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--silver_data", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--sft_checkpoint", default=None)
    p.add_argument("--alpha_gate_path", default=None)
    p.add_argument("--text_reward_backend", default="auto", choices=["rearag", "llama_head", "auto", "dummy"])
    p.add_argument("--text_reward_fallback_path", default=None)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--total_steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha_override", type=float, default=None)
    p.add_argument("--binary_labels_only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir or str(Path(checkpoint_dir()) / "kg_proweight_grpo")
    silver = args.silver_data or str(Path(checkpoint_dir()) / "prm_alpha_gate" / "silver_with_logprobs.jsonl")
    sft = args.sft_checkpoint or str(Path(checkpoint_dir()) / "sft_student" / "final")
    alpha = args.alpha_gate_path or str(Path(checkpoint_dir()) / "prm_alpha_gate" / "alpha_gate.pt")
    cfg = Phase3GRPOConfig(
        silver_path=silver,
        output_dir=out_dir,
        sft_checkpoint=sft,
        alpha_gate_path=alpha,
        text_reward_backend=args.text_reward_backend,
        text_reward_fallback_path=args.text_reward_fallback_path,
        group_size=args.group_size,
        total_steps=args.total_steps,
        seed=args.seed,
        alpha_override=args.alpha_override,
        binary_labels_only=args.binary_labels_only,
    )
    result = run_phase3_grpo(cfg)
    logger.info("Phase 3b GRPO result: %s", result)


if __name__ == "__main__":
    main()
