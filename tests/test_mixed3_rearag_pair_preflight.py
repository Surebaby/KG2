from __future__ import annotations

from pathlib import Path

from kgproweight.config import ProjectConfig, load_config
from kgproweight.data.silver_dataset import SilverDatasetReader

from scripts.prepare.preflight_mixed3_rearag_ppo_pair import (
    EXPECTED_ALIAS_AUDIT,
    _alias_reward_probe,
    audit_frozen_answer_aliases,
    flatten,
    nearest_rank,
)


ROOT = Path(__file__).resolve().parents[1]


def test_flatten_reports_only_leaf_differences():
    left = {"training": {"output_dir": "a", "ppo": {"flag": False, "same": 4}}}
    right = {"training": {"output_dir": "b", "ppo": {"flag": True, "same": 4}}}
    flat_left, flat_right = flatten(left), flatten(right)
    assert sorted(
        key for key in set(flat_left) | set(flat_right)
        if flat_left.get(key) != flat_right.get(key)
    ) == ["training.output_dir", "training.ppo.flag"]


def test_nearest_rank_is_deterministic():
    assert nearest_rank([5, 1, 4, 2, 3], .5) == 3
    assert nearest_rank([5, 1, 4, 2, 3], .95) == 5


def test_frozen_alias_population_matches_registered_contract():
    rows = list(
        SilverDatasetReader(
            ROOT
            / "data/silver_data/mixed_ppo_three_dataset_v1_n1799_k4_seed42/silver_train.jsonl",
            split=None,
        ).accepted()
    )
    audit = audit_frozen_answer_aliases(rows)
    for field, expected in EXPECTED_ALIAS_AUDIT.items():
        assert audit[field] == expected


def test_alias_reward_probes_match_nonprimary_fbi_and_prc_answers():
    fbi = _alias_reward_probe(
        "The FBI refused",
        ["The FBI refused", "FBI", "fbi", "Federal Bureau of Investigation"],
        "FBI",
    )
    china = _alias_reward_probe(
        "PRC", ["PRC", "China", "People's Republic of China"], "China"
    )
    unrelated = _alias_reward_probe(
        "The FBI refused",
        ["The FBI refused", "FBI", "Federal Bureau of Investigation"],
        "unrelated zebra sentinel",
    )
    for result in (fbi, china):
        assert result["trajectory_valid"] is True
        assert result["outcome_em"] == 1.0
        assert result["outcome_f1"] == 1.0
        assert result["trajectory_reward"] == 4.4
    assert unrelated["outcome_em"] == 0.0
    assert unrelated["outcome_f1"] == 0.0
    assert unrelated["trajectory_reward"] == 0.0


def test_frozen_pair_differs_only_in_output_and_proofkg_process_flag():
    control = load_config(
        ROOT / "configs/training/phase3_ppo_mixed3_rearag_v1_text7200_seed42.yaml",
        validate=ProjectConfig,
    )
    treatment = load_config(
        ROOT / "configs/training/phase3_ppo_mixed3_rearag_v1_text_kg_v2_1_7200_seed42.yaml",
        validate=ProjectConfig,
    )
    left, right = flatten(control.model_dump()), flatten(treatment.model_dump())
    differences = sorted(
        key for key in set(left) | set(right) if left.get(key) != right.get(key)
    )
    assert differences == [
        "training.output_dir",
        "training.ppo.proofkg_process_reward",
    ]
    for cfg in (control, treatment):
        assert cfg.reward.text_reward_backend == "rearag"
        assert cfg.training.alpha_gate_path is None
        assert cfg.training.ppo.mixed_outcome_reward is True
        assert cfg.training.ppo.mixed_text_reward is True
        assert cfg.training.ppo.total_ppo_steps == 7200
        assert cfg.training.ppo.rollouts_per_prompt == 4
        assert cfg.training.ppo.sft_replay_ratio == 0.10
