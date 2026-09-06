"""Telemetry-only regression tests for the replay10 CE=0.30 PPO smoke."""

from types import SimpleNamespace
from pathlib import Path

import pytest

from kgproweight.config import ProjectConfig, load_config
from kgproweight.data.parsers import parse_steps
from kgproweight.training.phase3_ppo import (
    Phase3PPOConfig,
    _citation_contract_diagnostics,
    _citation_reward_diagnostics,
    _configure_fresh_value_head,
    _nonfinite_training_state_reason,
    _smoke_health_guard_reason,
)


ROOT = Path(__file__).resolve().parents[1]


def test_zero_value_head_is_neutral_trainable_and_deterministic():
    import torch

    policy = SimpleNamespace(
        v_head=SimpleNamespace(
            summary=torch.nn.Linear(8, 1),
            dropout=torch.nn.Dropout(0.1),
        )
    )
    telemetry = _configure_fresh_value_head(
        policy, init_strategy="zero", dropout=0.0,
    )

    assert isinstance(policy.v_head.dropout, torch.nn.Identity)
    assert telemetry["weight_norm"] == 0.0
    assert telemetry["bias_norm"] == 0.0
    assert policy.v_head.summary.weight.requires_grad
    assert torch.equal(policy.v_head.summary(torch.randn(3, 8)), torch.zeros(3, 1))


def test_value_head_configuration_rejects_unknown_or_invalid_settings():
    import torch

    policy = SimpleNamespace(
        v_head=SimpleNamespace(
            summary=torch.nn.Linear(4, 1),
            dropout=torch.nn.Dropout(0.1),
        )
    )
    with pytest.raises(ValueError, match="value_head_init"):
        _configure_fresh_value_head(policy, init_strategy="mystery", dropout=0.0)
    with pytest.raises(ValueError, match="value_head_dropout"):
        _configure_fresh_value_head(policy, init_strategy="zero", dropout=1.0)


def _health_cfg(**overrides):
    values = dict(
        silver_path="unused.jsonl",
        output_dir="unused-output",
        health_guard_after_steps=200,
        health_guard_window=3,
        health_guard_min_valid_rate=0.5,
        health_guard_max_length_capped_frac=0.5,
        health_guard_max_mean_kl=20.0,
    )
    values.update(overrides)
    return Phase3PPOConfig(**values)


def test_smoke_health_guard_uses_rolling_window_and_waits_for_start_step():
    rows = [
        {"step": 192, "valid_rate": 0.25, "length_capped_frac": 0.75, "ppo_mean_kl": 30.0},
        {"step": 196, "valid_rate": 0.50, "length_capped_frac": 0.25, "ppo_mean_kl": 8.0},
        {"step": 200, "valid_rate": 0.50, "length_capped_frac": 0.25, "ppo_mean_kl": 8.0},
    ]
    reason = _smoke_health_guard_reason(rows, _health_cfg())
    assert reason is not None and "valid_rate" in reason
    assert _smoke_health_guard_reason(rows[:-1], _health_cfg()) is None


def test_smoke_health_guard_accepts_healthy_window_and_flags_cap_or_kl():
    healthy = [
        {"step": step, "valid_rate": 0.75, "length_capped_frac": 0.25, "ppo_mean_kl": 8.0}
        for step in (192, 196, 200)
    ]
    assert _smoke_health_guard_reason(healthy, _health_cfg()) is None

    capped = [dict(row, length_capped_frac=0.75) for row in healthy]
    assert "length_capped_frac" in _smoke_health_guard_reason(capped, _health_cfg())
    high_kl = [dict(row, ppo_mean_kl=21.0) for row in healthy]
    assert "mean KL" in _smoke_health_guard_reason(high_kl, _health_cfg())


@pytest.mark.parametrize(
    "field",
    [
        "mean_reward", "ppo_mean_kl", "policy_approxkl", "loss_total",
        "loss_policy", "loss_value", "advantage_raw_mean", "return_mean",
        "value_mean",
    ],
)
def test_nonfinite_training_state_fails_immediately_without_rolling_window(field):
    row = {field: float("nan")}
    reason = _nonfinite_training_state_reason(row)
    assert reason is not None
    assert field in reason
    # The public health guard must check this before both the start-step and
    # rolling-window gates.
    guarded = _smoke_health_guard_reason(
        [{"step": 4, field: float("inf")}], _health_cfg()
    )
    assert guarded is not None and field in guarded
    assert _nonfinite_training_state_reason({field: 0.0}) is None


def test_combined_stability_smoke_has_only_declared_combination_changes():
    baseline = load_config(
        ROOT / "configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_explicit_sft_ref.yaml",
        validate=ProjectConfig,
    )
    combined = load_config(
        ROOT / "configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_combined_stability_v1.yaml",
        validate=ProjectConfig,
    )
    old = baseline.model_dump() if hasattr(baseline, "model_dump") else baseline.dict()
    new = combined.model_dump() if hasattr(combined, "model_dump") else combined.dict()
    q = new["training"]["ppo"]

    assert q["max_new_tokens"] == 384
    assert q["kl_coef"] == pytest.approx(0.25)
    assert q["target_kl"] == pytest.approx(8.0)
    assert q["value_head_init"] == "zero"
    assert q["value_head_dropout"] == pytest.approx(0.0)
    assert q["ppo_epochs"] == 2
    assert q["vf_coef"] == pytest.approx(0.5)
    assert q["sft_replay_ratio"] == pytest.approx(0.10)
    assert q["sft_anchor_weight"] == pytest.approx(0.10)

    # Normalize exactly the approved combination variables and operational
    # guard. Any remaining difference means inheritance drifted unexpectedly.
    new["training"]["output_dir"] = old["training"]["output_dir"]
    for key in (
        "max_new_tokens", "kl_coef", "target_kl", "value_head_init",
        "value_head_dropout", "ppo_epochs", "health_guard_after_steps",
        "health_guard_window", "health_guard_min_valid_rate",
        "health_guard_max_length_capped_frac", "health_guard_max_mean_kl",
    ):
        new["training"]["ppo"][key] = old["training"]["ppo"][key]
    assert new == old


def _record(*, cite_any, cite_match, alpha, r_kg):
    return SimpleNamespace(
        cite_any=cite_any,
        cite_match=cite_match,
        alpha=alpha,
        r_kg=r_kg,
    )


def test_citation_diagnostics_separate_known_partial_and_unknown_steps():
    rows = [
        _record(cite_any=0.0, cite_match=0.0, alpha=0.2, r_kg=0.0),
        _record(cite_any=1.0, cite_match=0.0, alpha=0.8, r_kg=0.0),
        _record(cite_any=1.0, cite_match=0.5, alpha=0.6, r_kg=0.2),
        _record(cite_any=1.0, cite_match=1.0, alpha=0.7, r_kg=1.0),
    ]

    out = _citation_reward_diagnostics(rows)

    assert out["cite_any_step_frac"] == pytest.approx(0.75)
    assert out["cite_match_mean_citing_step"] == pytest.approx(0.5)
    assert out["cite_unknown_only_step_frac_citing"] == pytest.approx(1 / 3)
    assert out["cite_partial_match_step_frac_citing"] == pytest.approx(1 / 3)
    assert out["cite_all_matched_step_frac_citing"] == pytest.approx(1 / 3)
    assert out["alpha_mean_no_cite_step"] == pytest.approx(0.2)
    assert out["alpha_mean_unknown_cite_step"] == pytest.approx(0.8)
    assert out["alpha_mean_known_cite_step"] == pytest.approx(0.65)
    assert out["r_kg_zero_frac_unknown_cite_step"] == pytest.approx(1.0)
    assert out["r_kg_zero_frac_known_cite_step"] == pytest.approx(0.0)


def test_citation_diagnostics_do_not_fabricate_missing_group_means():
    out = _citation_reward_diagnostics([
        _record(cite_any=0.0, cite_match=0.0, alpha=0.3, r_kg=0.0),
    ])

    assert out["cite_any_step_frac"] == 0.0
    assert out["cite_match_mean_citing_step"] is None
    assert out["alpha_mean_known_cite_step"] is None
    assert out["alpha_mean_unknown_cite_step"] is None


def test_raw_contract_diagnostics_do_not_hide_unknown_citations():
    kg = [("Known", "relation", "Entity")]
    responses = [
        parse_steps(
            """[Step 1]
Reasoning: first
Knowledge Used: [(Known, relation, Entity), (Unknown, relation, Entity)]
Conclusion: first
[Final Answer] x
""",
            known_kg=kg,
        ),
        parse_steps(
            """[Step 1]
Reasoning: second
Knowledge Used: [(Known, relation, Entity), broken]
Conclusion: second
[Final Answer] x
""",
            known_kg=kg,
        ),
    ]

    out = _citation_contract_diagnostics(responses)

    assert out["citation_raw_citing_step_frac"] == 1.0
    assert out["citation_known_citing_step_frac"] == 1.0
    assert out["citation_unknown_citing_step_frac"] == pytest.approx(0.5)
    assert out["citation_malformed_content_step_frac"] == pytest.approx(0.5)
    assert out["citation_known_surface_count"] == 2
    assert out["citation_unknown_surface_count"] == 1
    assert out["citation_known_frac_recognized_surfaces"] == pytest.approx(2 / 3)
    assert out["citation_contract_invalid_response_frac"] == 1.0


def test_ce030_smoke_is_a_single_training_variable_change():
    old = load_config(
        ROOT / "configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600.yaml",
        validate=ProjectConfig,
    )
    new = load_config(
        ROOT / "configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_replay10_ce030.yaml",
        validate=ProjectConfig,
    )
    old_doc = old.model_dump() if hasattr(old, "model_dump") else old.dict()
    new_doc = new.model_dump() if hasattr(new, "model_dump") else new.dict()

    assert old_doc["training"]["ppo"]["sft_replay_ratio"] == pytest.approx(0.10)
    assert old_doc["training"]["ppo"]["sft_anchor_weight"] == pytest.approx(0.10)
    assert new_doc["training"]["ppo"]["sft_anchor_weight"] == pytest.approx(0.30)

    # Normalize the approved loss strength and unique Experiment ID; every
    # remaining resolved setting must be identical.
    new_doc["training"]["output_dir"] = old_doc["training"]["output_dir"]
    new_doc["training"]["ppo"]["sft_anchor_weight"] = 0.10
    assert new_doc == old_doc
