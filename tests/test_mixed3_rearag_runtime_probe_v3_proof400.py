from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from kgproweight.config import ProjectConfig, load_config
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.kg.training_question_kg import apply_training_question_kg, read_question_kg_records
from kgproweight.reward.proofkg_process import is_identity_safe_automatic_proofkg
from kgproweight.training.phase3_ppo import (
    _load_fixed_rollout_schedule, _load_rollout_sampling_weights,
    _mixed_reward_dataset_diagnostics, _mixed_text_batch_diagnostics,
    _validate_v21_execution_preflight,
)
from scripts.prepare.materialize_mixed3_rearag_runtime_probe_v3_proof400 import ARM_SPECS_V3
from scripts.prepare.preflight_mixed3_rearag_runtime_probe_v1 import EXPECTED_CONFIG_DIFF, config_diff
from scripts.prepare.preflight_mixed3_rearag_runtime_probe_v2 import EXPECTED_RUNTIME_DIFF
from scripts.prepare.resolve_phase3_ppo_runtime_config import resolve_phase3_ppo_runtime_config
from scripts.prepare.verify_mixed3_rearag_runtime_probe_v3_proof400 import (
    required_finite, verify_process, verify_rearag,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/silver_data/mixed3_rearag_runtime_wiring_probe_v3_proof400_seed42"
PROTOCOL = ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v3_proof400_seed42_freeze/protocol.json"


def test_v3_configs_are_exact_one_k4_subsets_of_current_proof400_pair():
    for arm, spec in ARM_SPECS_V3.items():
        probe = load_config(ROOT / spec["config"], validate=ProjectConfig)
        formal = load_config(ROOT / spec["formal_config"], validate=ProjectConfig)
        assert config_diff(formal, probe) == EXPECTED_CONFIG_DIFF, arm
        runtime = resolve_phase3_ppo_runtime_config(ROOT / spec["config"])
        formal_runtime = resolve_phase3_ppo_runtime_config(ROOT / spec["formal_config"])
        assert sorted(
            key for key in set(runtime) | set(formal_runtime)
            if runtime.get(key) != formal_runtime.get(key)
        ) == EXPECTED_RUNTIME_DIFF
        assert runtime["total_steps"] == runtime["batch_size"] == runtime["rollouts_per_prompt"] == 4
        assert runtime["save_every_steps"] == 4
        assert runtime["proofkg_process_reward"] is spec["expected_eligible"]
        assert runtime["alpha_gate_path"] is None and runtime["alpha_override"] is None


def test_v3_real_probe_assets_are_distinct_and_follow_production_loaders():
    identities = set()
    for arm, spec in ARM_SPECS_V3.items():
        arm_dir = DATA / arm
        rows = list(SilverDatasetReader(arm_dir / "silver_train.jsonl", split=None).accepted())
        assert len(rows) == 1
        stats = apply_training_question_kg(
            rows, read_question_kg_records(arm_dir / "question_kg_records.jsonl"),
            min_coverage=1.0, require_nonempty=False,
        )
        row = rows[0]
        identities.add((row.dataset, row.qid))
        eligible = is_identity_safe_automatic_proofkg(
            row.metadata["question_kg_runtime"], row.kg_subgraph,
            dataset=row.dataset, qid=row.qid,
        )
        assert eligible is spec["expected_eligible"]
        assert stats.coverage_rate == 1.0
        if eligible:
            assert _validate_v21_execution_preflight(rows)["eligible_rows"] == 1
        weights, sampling = _load_rollout_sampling_weights(arm_dir / "sampling_weights.jsonl", rows)
        indices, schedule = _load_fixed_rollout_schedule(
            arm_dir / "fixed_rollout_schedule.jsonl", rows,
            total_steps=4, rollouts_per_prompt=4, sampling_records=sampling,
        )
        assert weights == [1.0] and indices == [0, 0, 0, 0] and len(schedule) == 4
    assert len(identities) == 2


def test_v3_protocol_binds_proof400_formal_assets_and_current_runtime_closure():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "FROZEN_NOT_RUN"
    assert protocol["counts"]["scheduled_trajectories_total"] == 8
    assert protocol["scientific_boundary"]["maximum_trajectories"] == 8
    assert protocol["scientific_boundary"]["gpu_invoked"] is False
    assert protocol["postflight_contract"]["initial_reference_kl_abs_max"] == 1.0
    assert protocol["postflight_contract"]["ppo_tk_valid_process_applied_min"] == 1
    closure = protocol["runtime_code_closure"]
    assert {
        "scripts/train/phase3_ppo.py", "kgproweight/training/phase3_ppo.py",
        "kgproweight/training/reward_function.py",
        "scripts/prepare/verify_mixed3_rearag_runtime_probe_v3_proof400.py",
        "launch_ppo_mixed3_rearag_runtime_probe_v3_proof400_remote.sh",
    } <= set(closure)
    assert "config_comparison" in protocol["inputs"]
    assert protocol["arms"]["ppo_tk_eligible_k4"]["process_reward_eligible"] is True


def test_new_telemetry_exposes_preupdate_centering_and_process_applied():
    info = {
        "trajectory_valid": True,
        "proofkg_process": {"process_applied": True},
        "mixed_reward": {
            "dataset": "2wikimultihopqa", "outcome": 4.0, "text": 0.1,
            "process": 0.2, "total": 4.3, "proofkg_eligible": True,
            "text_raw_step_scores": [2.0, 0.0],
            "text_baseline_before_step": [0.5, 0.25],
            "text_centered_clipped_step_scores": [1.0, -0.25],
            "text_ema_baseline": 0.4, "text_ema_n_obs": 2,
        },
    }
    by_dataset = _mixed_reward_dataset_diagnostics([info])
    diag = by_dataset["2wikimultihopqa"]
    assert diag["process_applied_count"] == 1
    assert diag["text_baseline_preupdate_step_mean"] == pytest.approx(0.375)
    assert diag["text_centered_unclipped_step_mean"] == pytest.approx(0.625)
    batch = _mixed_text_batch_diagnostics(by_dataset)
    assert batch["mixed_text_baseline_preupdate_step_mean"] == pytest.approx(0.375)
    assert batch["mixed_text_centered_unclipped_step_mean"] == pytest.approx(0.625)


def _postflight_row(*, eligible: bool):
    return {
        "n_valid": 2, "mixed_text_step_count": 4,
        "mixed_text_raw_step_mean": 0.5,
        "mixed_text_baseline_preupdate_step_mean": 0.25,
        "mixed_text_centered_unclipped_step_mean": 0.25,
        "mixed_text_centered_step_mean": 0.25,
        "mixed_text_clip_frac": 0.0, "mixed_text_ema_baseline": 0.3,
        "mixed_text_ema_n_obs": 4,
        "proofkg_eligible_count": 4 if eligible else 0,
        "proofkg_process_applied_count": 2 if eligible else 0,
        "proofkg_process_mean": 0.5 if eligible else None,
        "mixed_reward_by_dataset": {"2wikimultihopqa": {
            "valid_count": 2, "text_step_count": 4,
            "text_raw_step_mean": 0.5,
            "text_baseline_preupdate_step_mean": 0.25,
            "text_centered_unclipped_step_mean": 0.25,
            "text_centered_step_mean": 0.25, "text_clip_frac": 0.0,
            "text_ema_baseline": 0.3, "text_ema_n_obs": 4,
            "process_applied_count": 2 if eligible else 0,
            "process_mean": 0.1 if eligible else 0.0,
        }},
    }


def test_v3_postflight_helpers_fail_closed_on_missing_text_or_process():
    assert required_finite({"x": 0.0}, "x", "arm") == 0.0
    with pytest.raises(ValueError, match="missing/null"):
        required_finite({"x": None}, "x", "arm")
    with pytest.raises(ValueError, match="non-finite"):
        required_finite({"x": math.inf}, "x", "arm")
    eligible = _postflight_row(eligible=True)
    assert verify_rearag(eligible, "tk")["n_obs"] == 4
    assert verify_process(eligible, True, "tk")["process_applied_count"] == 2
    control = _postflight_row(eligible=False)
    assert verify_process(control, False, "t")["weighted_process_mean"] == 0.0
    eligible["mixed_reward_by_dataset"]["2wikimultihopqa"]["process_mean"] = 0.0
    with pytest.raises(ValueError, match="contribution is zero"):
        verify_process(eligible, True, "tk")
    control["mixed_text_step_count"] = 0
    with pytest.raises(ValueError, match="no valid ReaRAG"):
        verify_rearag(control, "t")


def test_v3_launcher_uses_only_probe_configs_and_runs_postflight():
    text = (ROOT / "launch_ppo_mixed3_rearag_runtime_probe_v3_proof400_remote.sh").read_text(encoding="utf-8")
    assert text.count("run_arm \"") == 2
    assert "runtime_probe_v3_proof400_t_noneligible_k4_seed42.yaml" in text
    assert "runtime_probe_v3_proof400_tk_eligible_k4_seed42.yaml" in text
    assert "phase3_ppo_mixed3_rearag_v2_proof400_text7200_seed42.yaml" not in text
    assert "phase3_ppo_mixed3_rearag_v2_proof400_text_kg_v2_1_7200_seed42.yaml" not in text
    assert "verify_mixed3_rearag_runtime_probe_v3_proof400.py" in text
