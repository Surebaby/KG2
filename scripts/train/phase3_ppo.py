#!/usr/bin/env python
"""Phase 3b CLI — PPO + GAE + Critic + Reference Model (default on Pro 6000)."""

from __future__ import annotations

import argparse
from pathlib import Path

from kgproweight.config import ProjectConfig, load_config
from kgproweight.training.phase3_ppo import Phase3PPOConfig, run_phase3_ppo
from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import checkpoint_dir

configure_logging("INFO")
logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--silver_data", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--sft_checkpoint", default=None)
    p.add_argument("--alpha_gate_path", default=None)
    p.add_argument("--text_reward_backend", default="auto", choices=["rearag", "llama_head", "auto", "dummy"])
    p.add_argument("--text_reward_fallback_path", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total_steps", type=int, default=5000)
    p.add_argument("--alpha_override", type=float, default=None)
    p.add_argument("--binary_labels_only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir or str(Path(checkpoint_dir()) / "kg_proweight_final")
    silver = args.silver_data or str(Path(checkpoint_dir()) / "prm_alpha_gate" / "silver_with_logprobs.jsonl")
    sft = args.sft_checkpoint or str(Path(checkpoint_dir()) / "sft_student" / "final")
    alpha = args.alpha_gate_path or str(Path(checkpoint_dir()) / "prm_alpha_gate" / "alpha_gate.pt")

    if args.config:
        cfg_doc = load_config(args.config, validate=ProjectConfig)
        tcfg = cfg_doc.training
        ppo_cfg = tcfg.ppo
        cfg = Phase3PPOConfig(
            silver_path=silver,
            output_dir=out_dir,
            base_model=tcfg.base_model,
            sft_checkpoint=sft,
            alpha_gate_path=alpha,
            text_reward_backend=args.text_reward_backend,
            text_reward_fallback_path=args.text_reward_fallback_path,
            dtype=tcfg.dtype,
            seed=tcfg.seed,
            learning_rate=ppo_cfg.learning_rate,
            batch_size=ppo_cfg.batch_size,
            mini_batch_size=ppo_cfg.mini_batch_size,
            ppo_epochs=ppo_cfg.ppo_epochs,
            cliprange=ppo_cfg.cliprange,
            cliprange_value=ppo_cfg.cliprange_value,
            kl_coef=ppo_cfg.kl_coef,
            gamma=ppo_cfg.gamma,
            lam=ppo_cfg.lam,
            total_steps=ppo_cfg.total_ppo_steps,
            vf_coef=ppo_cfg.vf_coef,
            target_kl=ppo_cfg.target_kl,
            kl_horizon=ppo_cfg.kl_horizon,
            early_stopping=ppo_cfg.early_stopping,
            save_every_steps=ppo_cfg.save_every_steps,
            outcome_weight=ppo_cfg.outcome_weight,
            text_reward_scale=ppo_cfg.text_reward_scale,
            step_reward_scale=getattr(ppo_cfg, "step_reward_scale", 1.0),
            pure_em_reward=ppo_cfg.pure_em_reward,
            # R7: format-as-constraint (replaces step_format_bonus)
            min_valid_steps=getattr(ppo_cfg, "min_valid_steps", 3),
            min_reasoning_chars=getattr(ppo_cfg, "min_reasoning_chars", 20),
            sft_anchor_weight=getattr(ppo_cfg, "sft_anchor_weight", 0.02),
            sft_anchor_interval=getattr(ppo_cfg, "sft_anchor_interval", 50),
            sft_replay_ratio=getattr(ppo_cfg, "sft_replay_ratio", 0.15),
            log_with=ppo_cfg.log_with,
            use_lora=True,
            lora_r=tcfg.lora_r,
            lora_alpha=tcfg.lora_alpha,
            lora_dropout=tcfg.lora_dropout,
            alpha_override=args.alpha_override if args.alpha_override is not None else tcfg.alpha_override,
            binary_labels_only=args.binary_labels_only or tcfg.binary_labels_only,
            max_input_length=getattr(tcfg, "max_input_length", 4096),
        )
    else:
        cfg = Phase3PPOConfig(
            silver_path=silver,
            output_dir=out_dir,
            sft_checkpoint=sft,
            alpha_gate_path=alpha,
            text_reward_backend=args.text_reward_backend,
            text_reward_fallback_path=args.text_reward_fallback_path,
            seed=args.seed,
            total_steps=args.total_steps,
            alpha_override=args.alpha_override,
            binary_labels_only=args.binary_labels_only,
        )

    result = run_phase3_ppo(cfg)
    logger.info("Phase 3b PPO result: %s", result)


if __name__ == "__main__":
    main()
