#!/usr/bin/env python
"""Phase 2 CLI — train the PRM head + α-gate jointly with real logprobs."""

from __future__ import annotations

import argparse
from pathlib import Path

from kgproweight.config import ProjectConfig, load_config
from kgproweight.data.silver_split import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_VAL_RATIO,
)
from kgproweight.training.phase2_prm import Phase2Config, run_phase2
from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import checkpoint_dir, data_dir

configure_logging("INFO")
logger = get_logger(__name__)

# Argparse defaults, kept as named constants so the --config branch can tell
# "user passed this" from "argparse filled it in" and only override YAML on the
# former.
# Fallbacks for the no-config branch. The --config branch must NOT compare
# against these to detect "user passed it" — a user who legitimately types the
# same value as the default would be ignored. We use argparse.SUPPRESS instead,
# so the attribute only exists when the flag was actually given.
_DEF_EPOCHS = 3
_DEF_LR = 5e-5
_DEF_BATCH = 8
_DEF_ACCUM = 2
# Step-level, not prompt-level, but the input also carries up to 6 prior
# conclusions (the evidence the NEG label depends on): p99=242, p99.9=701.
# Keep in sync with Phase2Config.max_length and configs/training/phase2_prm.yaml.
_DEF_MAXLEN = 1024


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--silver_data", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--base_model", default="llama3-8B-instruct")
    p.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    p.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    p.add_argument("--batch_size", type=int, default=argparse.SUPPRESS)
    p.add_argument("--grad_accum", type=int, default=argparse.SUPPRESS)
    p.add_argument(
        "--max_length", type=int, default=argparse.SUPPRESS,
        help="Per-step token cap. Memory-critical: batches pad to their longest "
             "member, so this bounds peak activation. Step p99 is ~211 tokens.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--split", choices=["train", "val", "test"], default=argparse.SUPPRESS,
        help="Fold to train on. Omit to train on the whole file (the behaviour of "
             "every run before the split existed). Pass 'train' to hold val/test "
             "back so a same-distribution held-out number can be reported.",
    )
    p.add_argument(
        "--val_ratio", type=float, default=argparse.SUPPRESS,
        help="Fraction of question groups held out for validation (default 0.10).",
    )
    p.add_argument(
        "--test_ratio", type=float, default=argparse.SUPPRESS,
        help="Fraction of question groups held out for test (default 0.10).",
    )
    p.add_argument(
        "--split_seed", type=int, default=argparse.SUPPRESS,
        help="Seed for fold assignment, kept separate from --seed so a seed sweep "
             "over training randomness does not also reshuffle the held-out set "
             "(default 42).",
    )
    p.add_argument(
        "--split_allow_none", action="store_true", default=argparse.SUPPRESS,
        help="Deliberately train on the whole silver file. Only for reproducing "
             "historical pre-split runs; results are not held-out.",
    )
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--no_lora", action="store_true")
    p.add_argument("--no_text_head", action="store_true")
    p.add_argument("--binary_labels_only", action="store_true")
    p.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable activation recomputation. Checkpointing is on by default "
             "because a 24 GB card cannot fit bf16 Llama-3-8B plus full "
             "activations; on a 96 GB card turn it off to regain ~30%% step speed.",
    )
    p.add_argument(
        "--logprob_batch_size", type=int, default=None,
        help="Batch for the logprob pre-pass (default 4, safe for 24 GB). "
             "On a 96 GB card 16-32 is much faster.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        cfg_doc = load_config(args.config, validate=ProjectConfig)
        tcfg = cfg_doc.training
        silver = args.silver_data or getattr(tcfg, "silver_path", None) or str(
            Path(data_dir()) / "silver_data" / "silver_trajectories.jsonl"
        )
        out_dir = args.output_dir or tcfg.output_dir or str(
            Path(checkpoint_dir()) / "prm_alpha_gate"
        )
        p2 = Phase2Config(
            silver_path=silver,
            output_dir=out_dir,
            base_model=tcfg.base_model,
            dtype=tcfg.dtype,
            seed=tcfg.seed,
            # CLI wins over YAML: these are the knobs that have to change per
            # card (24 GB vs 96 GB), and silently ignoring them meant a run
            # launched with --batch_size 8 actually trained at the YAML's 4.
            epochs=getattr(args, "epochs", tcfg.prm_epochs),
            batch_size=getattr(args, "batch_size", tcfg.prm_batch_size),
            grad_accum=getattr(args, "grad_accum", tcfg.prm_grad_accum),
            lr=getattr(args, "lr", tcfg.prm_lr),
            max_length=getattr(args, "max_length", tcfg.prm_max_length),
            use_lora=not args.no_lora,
            lora_r=tcfg.lora_r,
            lora_alpha=tcfg.lora_alpha,
            lora_dropout=tcfg.lora_dropout,
            calibration_weight=cfg_doc.reward.alpha_gate.calibration_weight,
            alpha_target=tcfg.alpha_target,
            train_text_reward_head=not args.no_text_head,
            binary_labels_only=args.binary_labels_only,
            gradient_checkpointing=not args.no_gradient_checkpointing,
            # Same CLI-wins-over-YAML rule as the knobs above.
            split=getattr(args, "split", tcfg.split),
            val_ratio=getattr(args, "val_ratio", tcfg.val_ratio),
            test_ratio=getattr(args, "test_ratio", tcfg.test_ratio),
            split_seed=getattr(args, "split_seed", tcfg.split_seed),
            split_allow_none=getattr(
                args, "split_allow_none", getattr(tcfg, "split_allow_none", False)
            ),
            **({"logprob_batch_size": args.logprob_batch_size}
               if getattr(args, "logprob_batch_size", None) else {}),
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
            epochs=getattr(args, "epochs", _DEF_EPOCHS),
            lr=getattr(args, "lr", _DEF_LR),
            batch_size=getattr(args, "batch_size", _DEF_BATCH),
            grad_accum=getattr(args, "grad_accum", _DEF_ACCUM),
            max_length=getattr(args, "max_length", _DEF_MAXLEN),
            use_lora=not args.no_lora,
            train_text_reward_head=not args.no_text_head,
            binary_labels_only=args.binary_labels_only,
            gradient_checkpointing=not args.no_gradient_checkpointing,
            split=getattr(args, "split", None),
            val_ratio=getattr(args, "val_ratio", DEFAULT_VAL_RATIO),
            test_ratio=getattr(args, "test_ratio", DEFAULT_TEST_RATIO),
            split_seed=getattr(args, "split_seed", DEFAULT_SPLIT_SEED),
            split_allow_none=getattr(args, "split_allow_none", False),
            **({"logprob_batch_size": args.logprob_batch_size}
               if getattr(args, "logprob_batch_size", None) else {}),
        )

    logger.info(
        "Phase 2 effective config: batch=%d x accum=%d (eff %d) | epochs=%d | lr=%g | "
        "max_len=%d | grad_ckpt=%s | logprob_bs=%d",
        p2.batch_size, p2.grad_accum, p2.batch_size * p2.grad_accum,
        p2.epochs, p2.lr, p2.max_length, p2.gradient_checkpointing, p2.logprob_batch_size,
    )
    if p2.split is None:
        logger.warning(
            "Phase 2 split: NONE — training on the whole file. val/test are not held "
            "back, so nothing measured on this data is out-of-sample. Pass "
            "--split train for a real held-out set."
        )
    else:
        logger.info(
            "Phase 2 split: fold=%s val=%.3f test=%.3f split_seed=%d",
            p2.split, p2.val_ratio, p2.test_ratio,
            p2.seed if p2.split_seed is None else p2.split_seed,
        )
    result = run_phase2(p2)
    logger.info("Phase 2 result: %s", result)


if __name__ == "__main__":
    main()
