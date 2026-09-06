#!/usr/bin/env python
"""Phase 3a CLI — supervised fine-tuning of the Student."""

from __future__ import annotations

import argparse
from pathlib import Path

from kgproweight.config import ProjectConfig, load_config
from kgproweight.training.phase3_sft import Phase3SFTConfig, run_phase3_sft
from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import checkpoint_dir, data_dir

try:  # installed / -m invocation
    from scripts.train._split_args import add_split_args, log_split, split_kwargs
except ModuleNotFoundError:  # `python scripts/train/phase3_sft.py` — sys.path[0]
    # is this file's directory, and `scripts` is not an installed package.
    from _split_args import add_split_args, log_split, split_kwargs

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
    p.add_argument(
        "--init_adapter_path",
        default=None,
        help="Existing LoRA adapter to continue SFT from.",
    )
    p.add_argument(
        "--question_kg_records_path",
        default=None,
        help="dataset::qid + question-hash JSONL used to override prompt KG.",
    )
    p.add_argument("--min_question_kg_record_coverage", type=float, default=None)
    p.add_argument("--require_nonempty_question_kg_records", action="store_true")
    add_split_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    if args.config:
        cfg_doc = load_config(args.config, validate=ProjectConfig)
        tcfg = cfg_doc.training
        silver = args.silver_data or getattr(tcfg, "silver_path", None) or str(
            Path(checkpoint_dir()) / "prm_alpha_gate" / "silver_with_logprobs.jsonl"
        )
        out_dir = args.output_dir or tcfg.output_dir or str(
            Path(checkpoint_dir()) / "sft_student"
        )
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
            save_strategy=tcfg.sft_save_strategy,
            save_steps=tcfg.sft_save_steps,
            save_total_limit=tcfg.sft_save_total_limit,
            save_only_model=tcfg.sft_save_only_model,
            log_with=getattr(tcfg, "sft_log_with", None),
            logging_dir=getattr(tcfg, "sft_logging_dir", None),
            use_lora=not args.no_lora,
            lora_r=tcfg.lora_r,
            lora_alpha=tcfg.lora_alpha,
            lora_dropout=tcfg.lora_dropout,
            init_adapter_path=(
                args.init_adapter_path
                if args.init_adapter_path is not None
                else getattr(tcfg, "sft_init_adapter_path", None)
            ),
            question_kg_records_path=(
                args.question_kg_records_path
                if args.question_kg_records_path is not None
                else getattr(tcfg, "question_kg_records_path", None)
            ),
            min_question_kg_record_coverage=(
                args.min_question_kg_record_coverage
                if args.min_question_kg_record_coverage is not None
                else getattr(tcfg, "min_question_kg_record_coverage", 1.0)
            ),
            require_nonempty_question_kg_records=(
                args.require_nonempty_question_kg_records
                or getattr(tcfg, "require_nonempty_question_kg_records", False)
            ),
            split_allow_none=(
                bool(getattr(args, "split_allow_none", False))
                or bool(getattr(tcfg, "split_allow_none", False))
            ),
            **split_kwargs(args, tcfg),
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
            init_adapter_path=args.init_adapter_path,
            question_kg_records_path=args.question_kg_records_path,
            min_question_kg_record_coverage=(
                1.0
                if args.min_question_kg_record_coverage is None
                else args.min_question_kg_record_coverage
            ),
            require_nonempty_question_kg_records=(
                args.require_nonempty_question_kg_records
            ),
            split_allow_none=getattr(args, "split_allow_none", False),
            **split_kwargs(args),
        )

    log_split(logger, "Phase 3a", cfg)
    result = run_phase3_sft(cfg)
    logger.info("Phase 3a result: %s", result)


if __name__ == "__main__":
    main()
