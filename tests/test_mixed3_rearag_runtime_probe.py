from __future__ import annotations

import json
from pathlib import Path

from kgproweight.config import ProjectConfig, load_config
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.kg.training_question_kg import (
    apply_training_question_kg,
    read_question_kg_records,
)
from kgproweight.reward.proofkg_process import is_identity_safe_automatic_proofkg
from kgproweight.training.phase3_ppo import (
    _load_fixed_rollout_schedule,
    _load_rollout_sampling_weights,
    _validate_v21_execution_preflight,
)
from scripts.prepare.materialize_mixed3_rearag_runtime_probe_v1 import (
    ARM_SPECS,
    choose_probe_groups,
    read_jsonl,
)
from scripts.prepare.preflight_mixed3_rearag_runtime_probe_v1 import (
    EXPECTED_CONFIG_DIFF,
    config_diff,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/silver_data/mixed3_rearag_runtime_wiring_probe_v1_seed42"
PROTOCOL = (
    ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v1_seed42_freeze/protocol.json"
)


def test_source_selection_is_earliest_k4_group_in_each_eligibility_class():
    source = read_jsonl(
        ROOT
        / "data/silver_data/mixed_ppo_three_dataset_v1_n1799_k4_seed42/fixed_rollout_schedule.jsonl"
    )
    selected = choose_probe_groups(source)
    assert [row["rollout_index"] for row in selected["ppo_t_noneligible_k4"]] == [1, 2, 3, 4]
    assert [row["rollout_index"] for row in selected["ppo_tk_eligible_k4"]] == [17, 18, 19, 20]
    assert not any(row["process_reward_eligible"] for row in selected["ppo_t_noneligible_k4"])
    assert all(row["process_reward_eligible"] for row in selected["ppo_tk_eligible_k4"])


def test_probe_configs_change_only_budget_identity_and_micro_inputs():
    for arm, spec in ARM_SPECS.items():
        probe = load_config(ROOT / spec["config"], validate=ProjectConfig)
        formal = load_config(ROOT / spec["formal_config"], validate=ProjectConfig)
        assert config_diff(formal, probe) == EXPECTED_CONFIG_DIFF, arm
        assert probe.training.ppo.total_ppo_steps == 4
        assert probe.training.ppo.save_every_steps == 4
        assert probe.training.ppo.batch_size == 4
        assert probe.training.ppo.rollouts_per_prompt == 4
        assert probe.training.ppo.proofkg_process_reward is spec["expected_eligible"]
        assert Path(probe.training.output_dir).name == spec["experiment_id"]


def test_probe_assets_use_real_identity_join_and_fixed_schedule_loader():
    observed = {}
    for arm, spec in ARM_SPECS.items():
        arm_dir = DATA_DIR / arm
        trajectories = list(SilverDatasetReader(arm_dir / "silver_train.jsonl", split=None).accepted())
        assert len(trajectories) == 1
        stats = apply_training_question_kg(
            trajectories,
            read_question_kg_records(arm_dir / "question_kg_records.jsonl"),
            min_coverage=1.0,
            require_nonempty=False,
        )
        assert stats.covered == 1 and stats.absent == 0
        row = trajectories[0]
        eligible = is_identity_safe_automatic_proofkg(
            row.metadata["question_kg_runtime"], row.kg_subgraph,
            dataset=row.dataset, qid=row.qid,
        )
        assert eligible is spec["expected_eligible"]
        if eligible:
            assert _validate_v21_execution_preflight(trajectories) == {
                "eligible_rows": 1,
                "missing_execution_rows": 0,
            }
            assert row.kg_subgraph
        else:
            assert not row.kg_subgraph

        weights, records = _load_rollout_sampling_weights(
            arm_dir / "sampling_weights.jsonl", trajectories
        )
        indices, schedule = _load_fixed_rollout_schedule(
            arm_dir / "fixed_rollout_schedule.jsonl", trajectories,
            total_steps=4, rollouts_per_prompt=4, sampling_records=records,
        )
        assert weights == [1.0]
        assert indices == [0, 0, 0, 0]
        assert [row["within_group_rollout"] for row in schedule] == [1, 2, 3, 4]
        observed[arm] = f"{row.dataset}::{row.qid}"
    assert len(set(observed.values())) == 2


def test_frozen_protocol_limits_claims_and_total_budget():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "FROZEN_NOT_RUN"
    assert protocol["counts"]["scheduled_trajectories_total"] == 8
    assert protocol["scientific_boundary"]["training_started"] is False
    assert protocol["scientific_boundary"]["gpu_invoked"] is False
    assert protocol["scientific_boundary"]["training_effect_estimation"] is False
    assert protocol["scientific_boundary"]["paired_effect_comparison"] is False
    assert protocol["scientific_boundary"]["formal_pair_modified"] is False
    assert protocol["scientific_boundary"]["formal_data_modified"] is False


def test_launcher_is_probe_only_and_uses_distinct_log_and_tensorboard_targets():
    text = (ROOT / "launch_ppo_mixed3_rearag_runtime_probe_v1_remote.sh").read_text(
        encoding="utf-8"
    )
    assert "phase3_ppo_mixed3_rearag_runtime_probe_v1_t_noneligible_k4_seed42.yaml" in text
    assert "phase3_ppo_mixed3_rearag_runtime_probe_v1_tk_eligible_k4_seed42.yaml" in text
    assert "phase3_ppo_mixed3_rearag_v1_text7200_seed42.yaml" not in text
    assert "phase3_ppo_mixed3_rearag_v1_text_kg_v2_1_7200_seed42.yaml" not in text
    assert text.count("run_arm \"") == 2
    assert "KGPW_TB_DIR" in text
    assert "verify_mixed3_rearag_runtime_probe_v1.py" in text

