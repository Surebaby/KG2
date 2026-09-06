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
    _load_fixed_rollout_schedule,
    _load_rollout_sampling_weights,
    _validate_v21_execution_preflight,
)
from scripts.prepare.materialize_mixed3_rearag_runtime_probe_v2 import ARM_SPECS_V2
from scripts.prepare.preflight_mixed3_rearag_runtime_probe_v1 import EXPECTED_CONFIG_DIFF, config_diff
from scripts.prepare.verify_mixed3_rearag_runtime_probe_v2 import required_finite
from scripts.prepare.resolve_phase3_ppo_runtime_config import resolve_phase3_ppo_runtime_config
from scripts.prepare.preflight_mixed3_rearag_runtime_probe_v2 import EXPECTED_RUNTIME_DIFF


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/silver_data/mixed3_rearag_runtime_wiring_probe_v2_seed42"
PROTOCOL = ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v2_seed42_freeze/protocol.json"


def test_v2_configs_are_one_batch_subsets_of_formal_arms():
    for arm, spec in ARM_SPECS_V2.items():
        probe = load_config(ROOT / spec["config"], validate=ProjectConfig)
        formal = load_config(ROOT / spec["formal_config"], validate=ProjectConfig)
        assert config_diff(formal, probe) == EXPECTED_CONFIG_DIFF, arm
        assert probe.training.ppo.total_ppo_steps == 4
        assert probe.training.ppo.save_every_steps == 4
        assert probe.training.ppo.batch_size == 4
        assert probe.training.ppo.rollouts_per_prompt == 4
        assert probe.training.alpha_gate_path is None
        assert probe.training.alpha_override is None
        assert probe.training.ppo.proofkg_process_reward is spec["expected_eligible"]
        probe_runtime = resolve_phase3_ppo_runtime_config(ROOT / spec["config"])
        formal_runtime = resolve_phase3_ppo_runtime_config(ROOT / spec["formal_config"])
        assert sorted(
            key for key in set(probe_runtime) | set(formal_runtime)
            if probe_runtime.get(key) != formal_runtime.get(key)
        ) == EXPECTED_RUNTIME_DIFF
        assert probe_runtime["alpha_gate_path"] is None
        assert probe_runtime["alpha_override"] is None


def test_v2_real_data_join_schedule_and_eligibility_contract():
    identities = set()
    for arm, spec in ARM_SPECS_V2.items():
        arm_dir = DATA / arm
        rows = list(SilverDatasetReader(arm_dir / "silver_train.jsonl", split=None).accepted())
        assert len(rows) == 1
        stats = apply_training_question_kg(
            rows, read_question_kg_records(arm_dir / "question_kg_records.jsonl"),
            min_coverage=1.0, require_nonempty=False,
        )
        assert stats.covered == 1 and stats.absent == 0
        row = rows[0]
        identities.add((row.dataset, row.qid))
        eligible = is_identity_safe_automatic_proofkg(
            row.metadata["question_kg_runtime"], row.kg_subgraph,
            dataset=row.dataset, qid=row.qid,
        )
        assert eligible is spec["expected_eligible"]
        if eligible:
            assert _validate_v21_execution_preflight(rows)["eligible_rows"] == 1
        weights, sampling = _load_rollout_sampling_weights(
            arm_dir / "sampling_weights.jsonl", rows
        )
        indices, schedule = _load_fixed_rollout_schedule(
            arm_dir / "fixed_rollout_schedule.jsonl", rows,
            total_steps=4, rollouts_per_prompt=4, sampling_records=sampling,
        )
        assert weights == [1.0]
        assert indices == [0, 0, 0, 0]
        assert len(schedule) == 4
    assert len(identities) == 2


def test_v2_protocol_locks_cli_and_broad_runtime_closure():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "FROZEN_NOT_RUN"
    assert protocol["counts"]["scheduled_trajectories_total"] == 8
    closure = protocol["runtime_code_closure"]
    required = {
        "scripts/train/phase3_ppo.py",
        "scripts/train/_split_args.py",
        "scripts/prepare/resolve_phase3_ppo_runtime_config.py",
        "kgproweight/training/phase3_ppo.py",
        "kgproweight/training/reward_function.py",
        "kgproweight/training/step_reward_ppo_trainer.py",
        "kgproweight/reward/proofkg_process_v2.py",
        "kgproweight/reward/text_reward_model.py",
        "kgproweight/data/parsers.py",
        "kgproweight/data/silver_dataset.py",
        "kgproweight/data/silver_split.py",
        "launch_ppo_mixed3_rearag_runtime_probe_v2_remote.sh",
    }
    assert required <= set(closure)
    assert len(closure) >= 30
    assert protocol["scientific_boundary"]["training_started"] is False
    assert protocol["scientific_boundary"]["gpu_invoked"] is False
    assert protocol["supersedes"]["protocol"].endswith("probe_v1_seed42_freeze/protocol.json")


def test_v2_postflight_required_statistics_reject_missing_null_and_nonfinite():
    assert required_finite({"ppo_mean_kl": 0.0}, "ppo_mean_kl", "arm") == 0.0
    with pytest.raises(ValueError, match="missing/null"):
        required_finite({}, "ppo_mean_kl", "arm")
    with pytest.raises(ValueError, match="missing/null"):
        required_finite({"ppo_mean_kl": None}, "ppo_mean_kl", "arm")
    with pytest.raises(ValueError, match="non-finite"):
        required_finite({"ppo_mean_kl": math.nan}, "ppo_mean_kl", "arm")


def test_v2_launcher_never_references_formal_arm_configs():
    text = (ROOT / "launch_ppo_mixed3_rearag_runtime_probe_v2_remote.sh").read_text(encoding="utf-8")
    assert "runtime_probe_v2_t_noneligible_k4_seed42.yaml" in text
    assert "runtime_probe_v2_tk_eligible_k4_seed42.yaml" in text
    assert "phase3_ppo_mixed3_rearag_v1_text7200_seed42.yaml" not in text
    assert "phase3_ppo_mixed3_rearag_v1_text_kg_v2_1_7200_seed42.yaml" not in text
    assert text.count("run_arm \"") == 2
    assert "verify_mixed3_rearag_runtime_probe_v2.py" in text
