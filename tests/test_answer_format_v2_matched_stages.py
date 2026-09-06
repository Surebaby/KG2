"""Prevent probe-to-smoke objective drift or accidental source-gate release."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from kgproweight.config import ProjectConfig, load_config


ROOT = Path(__file__).resolve().parents[1]
MODES = {"a": "learned", "f": "fixed", "t": "text"}
SFT = "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"
GATE = "outputs/calibration/source_credit_gate_v2_representative_local_seed42_20260906_v1/features_v2/gate.json"


def config(arm, stage):
    path = ROOT / f"configs/training/phase3_ppo_mixed4_answer_format_v2_{arm}_{stage}_seed42.yaml"
    return load_config(str(path), validate=ProjectConfig)


@pytest.mark.parametrize("arm", MODES)
def test_matched_smoke_preserves_probe_objective_inputs_and_training_start(arm):
    probe, smoke = config(arm, "probe"), config(arm, "smoke")
    for cfg in (probe, smoke):
        training, ppo = cfg.training, cfg.training.ppo
        assert training.sft_checkpoint == training.reference_model == SFT
        assert training.sft_replay_silver_path == "data/silver_data/sft_replay_legacy_strong_train_rendered3to5_n2000_seed42_v2/silver_train.jsonl"
        assert ppo.source_gate_calibration_path == GATE
        assert ppo.answer_format_reward_version == "v2"
        assert ppo.source_gate_credit_version == ppo.source_gate_format_version == "v2"
        assert ppo.source_gated_reward_version == "v1"
        assert ppo.source_gate_mode == MODES[arm]
        assert ppo.mixed_outcome_reward and ppo.mixed_text_reward and ppo.proofkg_process_reward
        assert cfg.reward.text_reward_backend == "rearag"
        assert ppo.outcome_weight == 4 and ppo.proofkg_f1_weight == .1
        assert ppo.text_reward_scale == .3 and ppo.proofkg_process_weight == .2
        assert ppo.sft_replay_ratio == ppo.sft_anchor_weight == .1
        assert ppo.batch_size == ppo.rollouts_per_prompt == 4
        assert ppo.mini_batch_size == 1 and ppo.learning_rate == 1e-6
    assert probe.training.ppo.total_ppo_steps == 12
    assert smoke.training.ppo.total_ppo_steps == 600
    assert probe.training.ppo.save_every_steps == 12
    assert smoke.training.ppo.save_every_steps == 200
    left, right = deepcopy(probe.model_dump()), deepcopy(smoke.model_dump())
    for cfg in (left, right):
        cfg["training"].pop("output_dir")
        cfg["training"].pop("fixed_rollout_schedule_path")
        cfg["training"]["ppo"].pop("total_ppo_steps")
        cfg["training"]["ppo"].pop("save_every_steps")
    assert left == right


@pytest.mark.parametrize("stage", ["probe", "smoke"])
def test_a_f_t_differ_only_in_mode_and_output_directory(stage):
    rows = [deepcopy(config(arm, stage).model_dump()) for arm in MODES]
    for row in rows:
        row["training"].pop("output_dir")
        row["training"]["ppo"].pop("source_gate_mode")
    assert rows[0] == rows[1] == rows[2]


def test_shared_gate_still_requires_independent_confirmation():
    gate = json.loads((ROOT / GATE).read_text())
    assert gate["source_credit_clearance"] is True
    assert gate["training_clearance"] is False
    assert gate["independent_confirmation_clearance"] is False
    assert gate["ppo_launch_clearance"] is False
    # The new config files expose no diagnostic/allow-unvalidated override.
    for arm in MODES:
        cfg = config(arm, "smoke").model_dump()
        assert "allow_unvalidated" not in json.dumps(cfg)


@pytest.mark.parametrize("stage,count", [("probe", 12), ("smoke", 600)])
def test_stage_schedule_has_exact_k4_budget(stage, count):
    cfg = config("a", stage)
    rows = [json.loads(line) for line in (ROOT / cfg.training.fixed_rollout_schedule_path).read_text().splitlines()]
    assert len(rows) == count
    for offset in range(0, count, 4):
        group = rows[offset:offset + 4]
        assert len({(row["dataset"], row["qid"], row["question_sha256"]) for row in group}) == 1
