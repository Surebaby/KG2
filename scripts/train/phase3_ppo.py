#!/usr/bin/env python
"""Phase 3b CLI — PPO + GAE + Critic + Reference Model (default on Pro 6000)."""

from __future__ import annotations

import argparse
from pathlib import Path

from kgproweight.config import ProjectConfig, load_config
from kgproweight.training.phase3_ppo import Phase3PPOConfig, run_phase3_ppo
from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import checkpoint_dir

try:  # installed / -m invocation
    from scripts.train._split_args import add_split_args, log_split, split_kwargs
except ModuleNotFoundError:  # `python scripts/train/phase3_ppo.py` — sys.path[0]
    # is this file's directory, and `scripts` is not an installed package.
    from _split_args import add_split_args, log_split, split_kwargs

configure_logging("INFO")
logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--silver_data", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--sft_checkpoint", default=None)
    p.add_argument("--alpha_gate_path", default=None)
    p.add_argument("--text_reward_backend", default=None, choices=["rearag", "llama_head", "auto", "dummy"])
    p.add_argument("--text_reward_fallback_path", default=None)
    # R10: default=None so "was it passed?" is distinguishable from "it happens
    # to equal the YAML value". With --config these used to be dropped on the
    # floor (total_steps came from ppo_cfg.total_ppo_steps, seed from
    # tcfg.seed), so `--config ... --total_steps 64` silently ran the full
    # 16000-trajectory schedule — a smoke test that burns a GPU session.
    p.add_argument("--seed", type=int, default=None,
                   help="Overrides training.seed from --config when given.")
    p.add_argument("--total_steps", type=int, default=None,
                   help="TRAJECTORIES to roll out, not optimiser steps "
                        "(n_seen += batch_size per update). Overrides "
                        "training.ppo.total_ppo_steps from --config.")
    p.add_argument("--save_every_steps", type=int, default=None,
                   help="Also in trajectories. Overrides the YAML value; use a "
                        "small number for smoke tests so a checkpoint is "
                        "actually written before the run ends.")
    p.add_argument("--question_kg_index_path", default=None,
                   help="Pre-built question->KG index for PROMPT-side injection. "
                        "MUST be built over the same silver file+fold this run "
                        "rolls out on (06_build_question_kg_index.py --silver ...); "
                        "a mismatched index misses every question and silently "
                        "degrades both the prompt KG and r_kg. Overrides "
                        "training.question_kg_index_path.")
    p.add_argument("--max_kg_index_miss_rate", type=float, default=None,
                   help="Abort if the index misses more than this fraction of "
                        "prompts (default from YAML; 1.0 = warn only).")
    p.add_argument(
        "--question_kg_records_path",
        default=None,
        help="Identity-safe dataset::qid + question-hash KG JSONL. Mutually "
             "exclusive with --question_kg_index_path.",
    )
    p.add_argument("--min_question_kg_record_coverage", type=float, default=None)
    p.add_argument("--require_nonempty_question_kg_records", action="store_true")
    p.add_argument(
        "--require_exact_kg_index_alignment", action="store_true",
        help="Abort unless every indexed triple list exactly equals the stored "
             "silver KG for that accepted trajectory.",
    )
    p.add_argument(
        "--passage_overrides_path",
        default=None,
        help="Versioned qid->retrieved_passages JSONL for PPO rollout prompts. "
             "Must be paired with --rollout_schedule_path.",
    )
    p.add_argument(
        "--rollout_schedule_path",
        default=None,
        help="Frozen rollout qid schedule used as a fail-fast RNG/data guard. "
             "Must be paired with --passage_overrides_path.",
    )
    p.add_argument("--alpha_override", type=float, default=None)
    p.add_argument("--binary_labels_only", action="store_true")
    add_split_args(p)
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
        silver = args.silver_data or tcfg.silver_path or silver
        out_dir = args.output_dir or tcfg.output_dir or out_dir
        sft = args.sft_checkpoint or tcfg.sft_checkpoint or sft
        # The mixed PPO-T/PPO-TK route deliberately disables the historical
        # learned alpha gate.  Preserve an explicit YAML null instead of
        # silently replacing it with the legacy default checkpoint.  Legacy
        # configurations retain the historical fallback behaviour.
        if args.alpha_gate_path is not None:
            alpha = args.alpha_gate_path
        elif getattr(ppo_cfg, "mixed_outcome_reward", False):
            alpha = tcfg.alpha_gate_path
        else:
            alpha = tcfg.alpha_gate_path or alpha
        cfg = Phase3PPOConfig(
            silver_path=silver,
            output_dir=out_dir,
            base_model=tcfg.base_model,
            sft_checkpoint=sft,
            sft_selection_report_path=getattr(tcfg, "sft_selection_report_path", None),
            sft_replay_silver_path=getattr(tcfg, "sft_replay_silver_path", None),
            sft_replay_split=getattr(tcfg, "sft_replay_split", None),
            alpha_gate_path=alpha,
            text_reward_backend=(
                args.text_reward_backend or cfg_doc.reward.text_reward_backend
            ),
            text_reward_fallback_path=args.text_reward_fallback_path,
            dtype=tcfg.dtype,
            seed=tcfg.seed if args.seed is None else args.seed,
            learning_rate=ppo_cfg.learning_rate,
            batch_size=ppo_cfg.batch_size,
            mini_batch_size=ppo_cfg.mini_batch_size,
            ppo_epochs=ppo_cfg.ppo_epochs,
            cliprange=ppo_cfg.cliprange,
            cliprange_value=ppo_cfg.cliprange_value,
            kl_coef=ppo_cfg.kl_coef,
            gamma=ppo_cfg.gamma,
            lam=ppo_cfg.lam,
            max_grad_norm=ppo_cfg.max_grad_norm,
            total_steps=(
                ppo_cfg.total_ppo_steps if args.total_steps is None else args.total_steps
            ),
            vf_coef=ppo_cfg.vf_coef,
            value_head_init=ppo_cfg.value_head_init,
            value_head_dropout=ppo_cfg.value_head_dropout,
            runtime_contract_version=getattr(ppo_cfg, "runtime_contract_version", "legacy"),
            health_guard_after_steps=ppo_cfg.health_guard_after_steps,
            health_guard_window=ppo_cfg.health_guard_window,
            health_guard_min_valid_rate=ppo_cfg.health_guard_min_valid_rate,
            health_guard_max_length_capped_frac=ppo_cfg.health_guard_max_length_capped_frac,
            health_guard_max_mean_kl=ppo_cfg.health_guard_max_mean_kl,
            target_kl=ppo_cfg.target_kl,
            kl_horizon=ppo_cfg.kl_horizon,
            early_stopping=ppo_cfg.early_stopping,
            save_every_steps=(
                ppo_cfg.save_every_steps
                if args.save_every_steps is None
                else args.save_every_steps
            ),
            outcome_weight=ppo_cfg.outcome_weight,
            text_reward_scale=ppo_cfg.text_reward_scale,
            step_reward_scale=getattr(ppo_cfg, "step_reward_scale", 1.0),
            pure_em_reward=ppo_cfg.pure_em_reward,
            proofkg_process_reward=ppo_cfg.proofkg_process_reward,
            proofkg_outcome_only_reward=ppo_cfg.proofkg_outcome_only_reward,
            proofkg_process_version=ppo_cfg.proofkg_process_version,
            proofkg_process_weight=ppo_cfg.proofkg_process_weight,
            proofkg_f1_weight=ppo_cfg.proofkg_f1_weight,
            proofkg_dynamic_validity=ppo_cfg.proofkg_dynamic_validity,
            mixed_outcome_reward=getattr(ppo_cfg, "mixed_outcome_reward", False),
            mixed_text_reward=getattr(ppo_cfg, "mixed_text_reward", False),
            source_gated_reward_version=ppo_cfg.source_gated_reward_version,
            source_gate_format_version=ppo_cfg.source_gate_format_version,
            answer_format_reward_version=ppo_cfg.answer_format_reward_version,
            source_gate_credit_version=ppo_cfg.source_gate_credit_version,  # Explicit v2 opt-in reaches the runtime loader.
            source_gate_mode=ppo_cfg.source_gate_mode,
            source_gate_calibration_path=ppo_cfg.source_gate_calibration_path,
            proofkg_require_all_eligible=ppo_cfg.proofkg_require_all_eligible,
            rollouts_per_prompt=ppo_cfg.rollouts_per_prompt,
            # R7: format-as-constraint (replaces step_format_bonus)
            min_valid_steps=getattr(ppo_cfg, "min_valid_steps", 3),
            # §9.4-3 / R-1b: explicit forwarding is mandatory -- schemas.py sets
            # extra="allow", so a YAML key that is not named here is accepted and
            # silently ignored (the ppo_max_kg_triples trap).
            shortfall_coef=getattr(ppo_cfg, "shortfall_coef", 0.0),
            target_steps=getattr(ppo_cfg, "target_steps", 3),
            min_reasoning_chars=getattr(ppo_cfg, "min_reasoning_chars", 20),
            # §9.4-1 (量纲): R_Text DC removal. Must be forwarded explicitly or
            # the YAML setting is silently dropped (schemas.py extra="allow").
            center_text_reward=getattr(ppo_cfg, "center_text_reward", False),
            text_baseline_momentum=getattr(ppo_cfg, "text_baseline_momentum", 0.99),
            sft_anchor_weight=getattr(ppo_cfg, "sft_anchor_weight", 0.02),
            sft_anchor_interval=getattr(ppo_cfg, "sft_anchor_interval", 0),
            sft_replay_ratio=getattr(ppo_cfg, "sft_replay_ratio", 0.10),
            log_with=ppo_cfg.log_with,
            use_lora=True,
            lora_r=tcfg.lora_r,
            lora_alpha=tcfg.lora_alpha,
            lora_dropout=tcfg.lora_dropout,
            alpha_override=args.alpha_override if args.alpha_override is not None else tcfg.alpha_override,
            binary_labels_only=args.binary_labels_only or tcfg.binary_labels_only,
            use_real_logprobs=cfg_doc.reward.use_real_logprobs,
            max_new_tokens=ppo_cfg.max_new_tokens,
            temperature=ppo_cfg.temperature,
            top_p=ppo_cfg.top_p,
            max_input_length=getattr(tcfg, "max_input_length", 4096),
            # R10 speed: must be forwarded EXPLICITLY. schemas.py sets
            # extra="allow", so an unrecognised YAML key is accepted silently and
            # then never reaches the dataclass -- the same trap that left
            # ppo_max_kg_triples pinned at its default while the YAML "set" it.
            rollout_chunk_size=ppo_cfg.rollout_chunk_size,
            max_steps=ppo_cfg.max_steps,
            ppo_max_passages=ppo_cfg.ppo_max_passages,
            ppo_min_kg_triples=ppo_cfg.ppo_min_kg_triples,
            ppo_max_kg_triples=ppo_cfg.ppo_max_kg_triples,
            prm_min_subgraph_for_verify=ppo_cfg.prm_min_subgraph_for_verify,
            # §10.3 / R-2: must be forwarded EXPLICITLY -- schemas.py sets
            # extra="allow", so these YAML keys would otherwise be accepted and
            # silently dropped, exactly like ppo_max_kg_triples was.
            question_kg_index_path=(
                args.question_kg_index_path
                if args.question_kg_index_path is not None
                else getattr(tcfg, "question_kg_index_path", None)
            ),
            max_kg_index_miss_rate=(
                args.max_kg_index_miss_rate
                if args.max_kg_index_miss_rate is not None
                else getattr(tcfg, "max_kg_index_miss_rate", 1.0)
            ),
            require_exact_kg_index_alignment=(
                args.require_exact_kg_index_alignment
                or getattr(tcfg, "require_exact_kg_index_alignment", False)
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
            passage_overrides_path=(
                args.passage_overrides_path
                if args.passage_overrides_path is not None
                else tcfg.passage_overrides_path
            ),
            rollout_schedule_path=(
                args.rollout_schedule_path
                if args.rollout_schedule_path is not None
                else tcfg.rollout_schedule_path
            ),
            rollout_sampling_weights_path=getattr(
                tcfg, "rollout_sampling_weights_path", None
            ),
            fixed_rollout_schedule_path=getattr(
                tcfg, "fixed_rollout_schedule_path", None
            ),
            # §13-1: passed explicitly rather than via split_kwargs(), because
            # phase3_sft.py shares that helper and its config has no such field.
            split_allow_none=(
                bool(getattr(args, "split_allow_none", False))
                or bool(getattr(tcfg, "split_allow_none", False))
            ),
            **split_kwargs(args, tcfg),
        )
    else:
        cfg = Phase3PPOConfig(
            silver_path=silver,
            output_dir=out_dir,
            sft_checkpoint=sft,
            alpha_gate_path=alpha,
            text_reward_backend=args.text_reward_backend or "auto",
            text_reward_fallback_path=args.text_reward_fallback_path,
            # No --config: fall back to the dataclass defaults when the flag is
            # absent, since args.* is now None rather than a hardcoded number.
            **({} if args.seed is None else {"seed": args.seed}),
            **({} if args.total_steps is None else {"total_steps": args.total_steps}),
            **({} if args.save_every_steps is None
               else {"save_every_steps": args.save_every_steps}),
            alpha_override=args.alpha_override,
            binary_labels_only=args.binary_labels_only,
            split_allow_none=getattr(args, "split_allow_none", False),
            **({} if args.question_kg_index_path is None
               else {"question_kg_index_path": args.question_kg_index_path}),
            **({} if args.max_kg_index_miss_rate is None
               else {"max_kg_index_miss_rate": args.max_kg_index_miss_rate}),
            require_exact_kg_index_alignment=args.require_exact_kg_index_alignment,
            **({} if args.question_kg_records_path is None
               else {"question_kg_records_path": args.question_kg_records_path}),
            **({} if args.min_question_kg_record_coverage is None
               else {"min_question_kg_record_coverage": args.min_question_kg_record_coverage}),
            require_nonempty_question_kg_records=(
                args.require_nonempty_question_kg_records
            ),
            **({} if args.passage_overrides_path is None
               else {"passage_overrides_path": args.passage_overrides_path}),
            **({} if args.rollout_schedule_path is None
               else {"rollout_schedule_path": args.rollout_schedule_path}),
            **split_kwargs(args),
        )

    log_split(logger, "Phase 3b PPO", cfg)
    # total_steps is in trajectories; report the update count so a run's length
    # is never misread as "N optimiser steps" the way the r9 runs were.
    logger.info(
        "Phase 3b PPO schedule: %d trajectories / batch_size %d = %d optimiser "
        "updates; checkpoint every %d trajectories (= %d updates)",
        cfg.total_steps, cfg.batch_size, cfg.total_steps // max(1, cfg.batch_size),
        cfg.save_every_steps, cfg.save_every_steps // max(1, cfg.batch_size),
    )
    result = run_phase3_ppo(cfg)
    logger.info("Phase 3b PPO result: %s", result)


if __name__ == "__main__":
    main()
