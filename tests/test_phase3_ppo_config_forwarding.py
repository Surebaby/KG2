"""PPO YAML controls must reach the runtime dataclass explicitly."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

import scripts.train.phase3_ppo as phase3_cli
from scripts.prepare.resolve_phase3_ppo_runtime_config import (
    resolve_phase3_ppo_runtime_config,
)


def test_rollout_and_optimizer_controls_are_forwarded(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "ppo.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "reward": {"use_real_logprobs": False},
                "training": {
                    "phase": "phase3_ppo",
                    "split": None,
                    "split_allow_none": True,
                    "max_input_length": 5678,
                    "question_kg_records_path": "data/proofkg.jsonl",
                    "sft_selection_report_path": "outputs/selection.json",
                    "sft_replay_silver_path": "data/hotpot_replay.jsonl",
                    "sft_replay_split": "train",
                    "min_question_kg_record_coverage": 0.95,
                    "require_nonempty_question_kg_records": True,
                    "passage_overrides_path": "data/hybrid.jsonl",
                    "rollout_schedule_path": "data/schedule.jsonl",
                    "rollout_sampling_weights_path": "data/weights.jsonl",
                    "fixed_rollout_schedule_path": "data/fixed_schedule.jsonl",
                    "ppo": {
                        "max_grad_norm": 0.73,
                        "value_head_init": "zero",
                        "value_head_dropout": 0.0,
                        "runtime_contract_version": "v2",
                        "health_guard_after_steps": 120,
                        "health_guard_window": 11,
                        "health_guard_min_valid_rate": 0.45,
                        "health_guard_max_length_capped_frac": 0.55,
                        "health_guard_max_mean_kl": 17.0,
                        "max_new_tokens": 333,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "rollout_chunk_size": 3,
                        "max_steps": 7,
                        "ppo_max_passages": 11,
                        "prm_min_subgraph_for_verify": 1,
                        "proofkg_process_reward": True,
                        "proofkg_outcome_only_reward": True,
                        "proofkg_process_version": "v2_1",
                        "proofkg_process_weight": 1.25,
                        "proofkg_f1_weight": 0.10,
                        "proofkg_dynamic_validity": True,
                        "mixed_outcome_reward": True,
                        "mixed_text_reward": True,
                        "proofkg_require_all_eligible": True,
                        "rollouts_per_prompt": 4,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    captured = {}
    monkeypatch.setattr(
        phase3_cli,
        "run_phase3_ppo",
        lambda cfg: captured.setdefault("cfg", cfg),
    )
    monkeypatch.setattr(sys, "argv", ["phase3_ppo.py", "--config", str(config_path)])

    phase3_cli.main()
    cfg = captured["cfg"]

    assert cfg.max_grad_norm == 0.73
    assert cfg.value_head_init == "zero"
    assert cfg.value_head_dropout == 0.0
    assert cfg.runtime_contract_version == "v2"
    assert cfg.health_guard_after_steps == 120
    assert cfg.health_guard_window == 11
    assert cfg.health_guard_min_valid_rate == 0.45
    assert cfg.health_guard_max_length_capped_frac == 0.55
    assert cfg.health_guard_max_mean_kl == 17.0
    assert cfg.max_new_tokens == 333
    assert cfg.temperature == 1.0
    assert cfg.top_p == 1.0
    assert cfg.rollout_chunk_size == 3
    assert cfg.max_steps == 7
    assert cfg.ppo_max_passages == 11
    assert cfg.use_real_logprobs is False
    assert cfg.max_input_length == 5678
    assert cfg.passage_overrides_path == "data/hybrid.jsonl"
    assert cfg.rollout_schedule_path == "data/schedule.jsonl"
    assert cfg.rollout_sampling_weights_path == "data/weights.jsonl"
    assert cfg.fixed_rollout_schedule_path == "data/fixed_schedule.jsonl"
    assert cfg.split is None
    assert cfg.split_allow_none is True
    assert cfg.question_kg_records_path == "data/proofkg.jsonl"
    assert cfg.sft_selection_report_path == "outputs/selection.json"
    assert cfg.sft_replay_silver_path == "data/hotpot_replay.jsonl"
    assert cfg.sft_replay_split == "train"
    assert cfg.min_question_kg_record_coverage == 0.95
    assert cfg.require_nonempty_question_kg_records is True
    assert cfg.prm_min_subgraph_for_verify == 1
    assert cfg.proofkg_process_reward is True
    assert cfg.proofkg_outcome_only_reward is True
    assert cfg.proofkg_process_version == "v2_1"
    assert cfg.proofkg_process_weight == 1.25
    assert cfg.proofkg_f1_weight == 0.10
    assert cfg.proofkg_dynamic_validity is True
    assert cfg.mixed_outcome_reward is True
    assert cfg.mixed_text_reward is True
    assert cfg.alpha_gate_path is None
    assert cfg.proofkg_require_all_eligible is True
    assert cfg.rollouts_per_prompt == 4


def test_sampling_distribution_modifiers_are_rejected(tmp_path: Path):
    from kgproweight.config.schemas import PPOConfig

    for field, value in (("temperature", 0.7), ("top_p", 0.9)):
        try:
            PPOConfig(**{field: value})
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe PPO sampling modifier accepted: {field}={value}")


def test_formal_mixed_pair_exact_cli_contract_disables_alpha():
    root = Path(__file__).resolve().parents[1]
    control = resolve_phase3_ppo_runtime_config(
        root / "configs/training/phase3_ppo_mixed3_rearag_v1_text7200_seed42.yaml"
    )
    treatment = resolve_phase3_ppo_runtime_config(
        root / "configs/training/phase3_ppo_mixed3_rearag_v1_text_kg_v2_1_7200_seed42.yaml"
    )

    assert control["alpha_gate_path"] is None
    assert control["runtime_contract_version"] == "legacy"
    assert control["alpha_override"] is None
    assert treatment["alpha_gate_path"] is None
    assert treatment["alpha_override"] is None
    differences = sorted(
        key for key in set(control) | set(treatment)
        if control.get(key) != treatment.get(key)
    )
    assert differences == ["output_dir", "proofkg_process_reward"]
