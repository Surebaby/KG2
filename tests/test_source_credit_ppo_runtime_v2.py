"""Explicit runtime-v2 source features and step-softsign reward contracts."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch
import yaml

import kgproweight.training.phase3_ppo as ppo
from kgproweight.reward import source_credit_gate_v2 as gate_v2
from kgproweight.reward.source_quality_gate_v1 import canonical_sha256
from kgproweight.reward.source_reward_normalization_v2 import fit_text_normalization_v2, normalize_text_steps_v2
from kgproweight.training.reward_function import KGProWeightRewardFunction
from scripts.prepare.resolve_phase3_ppo_runtime_config import resolve_phase3_ppo_runtime_config
from tests.test_source_credit_ppo_runtime_v1 import _bound_gate, _reward as old_reward
from tests.test_source_gated_ppo_reward_v1 import _Text, _score
from tests.test_ppo_emf1_reward_contract_v1 import _ForbiddenComponent, _Tokenizer, _response


def _gate(tmp_path, status="PASS"):
    old, spec = _bound_gate(tmp_path, status)
    data = deepcopy(old.artifact)
    stats = fit_text_normalization_v2([{"dataset": "synthetic", "qid": "unit",
        "candidate_id": "unit0", "trajectory_valid": True, "raw_text": [0., .4]}])
    data.update(schema_version=gate_v2.ARTIFACT_SCHEMA, gate_version=gate_v2.GATE_VERSION,
                training_clearance=False, independent_confirmation_clearance=False)
    data["normalization"].update(input_contract=gate_v2.NORMALIZATION_CONTRACT,
        text_center=stats["text_center"], text_scale=stats["text_scale"], text_v2=stats,
        text_application_scope=stats["application_contract"])
    data.pop("payload_sha256")
    data["payload_sha256"] = canonical_sha256(data)
    gate = gate_v2.SourceCreditGateV2(data, mask=old.mask, allow_synthetic=True, allow_unvalidated=True)
    return gate, spec, old


def _reward(gate, *, mode="learned", text=None, version="v2", format_version="v2"):
    return KGProWeightRewardFunction(
        alpha_gate=_ForbiddenComponent("legacy alpha"), prm_annotator=_ForbiddenComponent("PRM"),
        text_reward_model=text or _Text(), tokenizer=_Tokenizer(),
        outcome_weight=4., text_reward_scale=.3, max_steps=5,
        proofkg_process_reward=True, proofkg_process_version="v2_3", proofkg_process_weight=.2,
        proofkg_f1_weight=.1, proofkg_dynamic_validity=True, mixed_outcome_reward=True,
        mixed_text_reward=True, runtime_contract_version="v2", source_gated_reward_version="v1",
        source_gate_format_version=format_version, source_gate_credit_version=version,
        source_gate_mode=mode, source_quality_gate=gate, center_text_reward=False,
    )


@pytest.mark.parametrize("mode,alpha", [("learned", .5), ("fixed", .25), ("text", 0.)])
def test_v2_uses_step_softsign_with_unchanged_graph_outcome_and_token_conservation(tmp_path, mode, alpha):
    gate, spec, old = _gate(tmp_path)
    result = _score(_reward(gate, mode=mode), spec)
    reference = _score(old_reward(old, mode=mode), spec)
    expected = normalize_text_steps_v2([.8, .3, -.1], gate.normalization["text_v2"])
    text = .3 * (1-alpha) * expected["mean_bounded"]
    assert result["mixed_reward"]["text"] == pytest.approx(text)
    for field in ("outcome", "process"):
        assert result["mixed_reward"][field] == reference["mixed_reward"][field]
    assert result["source_gate"]["alpha_effective"] == alpha
    assert result["source_gate"]["text_aggregation"] == expected["application_contract"]
    assert result["source_gate"]["text_normalization_v2"] == expected
    assert result["mixed_reward"]["text_centered_clipped_step_scores"] == expected["bounded_step_scores"]
    assert result["mixed_reward"]["text_clip_frac"] == 0.
    assert result["token_rewards"].sum().item() == pytest.approx(result["trajectory_reward"], abs=1e-6)
    assert sum(result["per_step_rewards"]) == pytest.approx(result["trajectory_reward"])


@pytest.mark.parametrize("status", ["FAIL", "UNVERIFIED"])
@pytest.mark.parametrize("mode", ["learned", "fixed", "text"])
def test_v2_preserves_exact_source_exclusion_and_full_text_credit(tmp_path, status, mode):
    gate, spec, _old = _gate(tmp_path, status)
    text = _Text()
    result = _score(_reward(gate, mode=mode, text=text), spec)
    expected = normalize_text_steps_v2(text.scores, gate.normalization["text_v2"])
    assert result["source_gate"]["alpha_effective"] == result["mixed_reward"]["process"] == 0.
    assert result["mixed_reward"]["text"] == pytest.approx(.3 * expected["mean_bounded"])
    assert result["source_gate"]["source_credit_mask"]["status"] == status
    assert len(text.calls) == 1


def test_v2_invalid_format_stays_minus_four_without_text_or_alpha_calls(tmp_path, monkeypatch):
    gate, spec, _old = _gate(tmp_path, "UNVERIFIED")
    text = _Text()
    monkeypatch.setattr(text, "score_steps", _ForbiddenComponent("invalid Text score"))
    monkeypatch.setattr(gate, "predict", _ForbiddenComponent("invalid alpha prediction"))
    result = _score(_reward(gate, text=text), spec, _response(answer=""))
    assert result["trajectory_valid"] is False
    assert result["trajectory_reward"] == -4
    assert result["source_gate"]["source_credit_mask"]["status"] == "UNVERIFIED"
    assert torch.count_nonzero(result["token_rewards"]) == 1
    assert result["token_rewards"][-1] == -4
    two_steps = _score(_reward(gate, mode="fixed"), spec, _response(steps=2))
    assert two_steps["trajectory_valid"] and two_steps["proofkg_process"]["required_steps"] == 2


def test_v2_delegates_feature_computation_to_declared_gate_contract(tmp_path, monkeypatch):
    gate, spec, _old = _gate(tmp_path)
    calls = []
    compute = gate.compute_features
    def record(specification, steps, proof):
        calls.append({"n_steps": len(steps), "scorer": proof.get("scorer_version")})
        return compute(specification, steps, proof)
    monkeypatch.setattr(gate, "compute_features", record)
    _score(_reward(gate), spec)
    _score(_reward(gate), spec, _response(answer=""))
    assert len(calls) == 2
    assert calls[0]["scorer"] is not None and calls[1]["scorer"] is None


def test_v2_telemetry_separates_softsign_saturation_from_hard_clipping(tmp_path):
    gate, spec, _old = _gate(tmp_path)
    result = _score(_reward(gate), spec)
    dataset = ppo._mixed_reward_dataset_diagnostics([result])["2wikimultihopqa"]
    assert dataset["text_clip_frac"] == 0.
    assert dataset["text_raw_z_outside_unit_frac"] == pytest.approx(2/3)
    assert dataset["text_soft_saturation_frac"] == 0.
    batch = ppo._source_gate_batch_diagnostics([result])
    assert batch["source_gate_text_v2_hard_clip_frac"] == 0.
    assert batch["source_gate_text_v2_raw_z_outside_unit_frac"] == pytest.approx(2/3)


def test_batch_feature_telemetry_includes_new_registered_dimensions(tmp_path):
    gate, spec, _old = _gate(tmp_path)
    result = _score(_reward(gate), spec)
    # A telemetry fixture, not a prediction: verify aggregation has no 4D cap.
    result["source_gate"]["features"]["values"]["registered_future_feature"] = .75
    batch = ppo._source_gate_batch_diagnostics([result])
    assert batch["source_gate_feature_registered_future_feature_mean"] == .75


def test_real_six_dimensional_v2_features_reach_alpha_and_invalid_telemetry(tmp_path):
    from kgproweight.reward.source_trajectory_features_v2 import FEATURE_NAMES, FEATURE_VERSION
    gate, spec, _old = _gate(tmp_path)
    data = deepcopy(gate.artifact)
    data.update(feature_version=FEATURE_VERSION, feature_names=list(FEATURE_NAMES),
                weights=[0., 0., 0., 0., .75, -.25])
    data["feature_standardization"] = {
        "mean": dict.fromkeys(FEATURE_NAMES, 0.), "scale": dict.fromkeys(FEATURE_NAMES, 1.)}
    data.pop("payload_sha256")
    data["payload_sha256"] = canonical_sha256(data)
    gate = gate_v2.SourceCreditGateV2(data, mask=gate.mask, allow_synthetic=True, allow_unvalidated=True)
    valid = _score(_reward(gate), spec)
    invalid = _score(_reward(gate), spec, _response(answer=""))
    features = valid["source_gate"]["features"]["values"]
    assert tuple(features) == FEATURE_NAMES
    assert features["source_edge_coverage"] == 1.
    assert features["min_step_citation_precision"] == 0.
    assert valid["source_gate"]["alpha_effective"] == pytest.approx(1/(1+math.exp(-.75)))
    assert invalid["source_gate"]["features"]["feature_version"] == FEATURE_VERSION
    assert tuple(invalid["source_gate"]["features"]["values"]) == FEATURE_NAMES
    batch = ppo._source_gate_batch_diagnostics([valid, invalid])
    assert batch["source_gate_feature_source_edge_coverage_mean"] == 1.
    assert batch["source_gate_feature_min_step_citation_precision_mean"] == 0.


def _config(tmp_path):
    path = tmp_path / "v2.yaml"
    base = Path("configs/training/phase3_ppo_mixed4_sourcegate_v1_a_probe_seed42.yaml").resolve()
    path.write_text(yaml.safe_dump({"includes": [str(base)], "training": {"ppo": {
        "source_gate_format_version": "v2", "source_gate_credit_version": "v2"}}}))
    return ppo.Phase3PPOConfig(**resolve_phase3_ppo_runtime_config(path))


def test_v2_schema_cli_and_runtime_config_forward_explicit_opt_in(tmp_path):
    cfg = _config(tmp_path)
    assert cfg.source_gate_credit_version == "v2"
    ppo._validate_mixed_reward_config(cfg)
    cfg.source_gate_format_version = "v1"
    with pytest.raises(ValueError, match="source credit v2 requires"):
        ppo._validate_mixed_reward_config(cfg)


@pytest.mark.parametrize("configured", ["disabled", "v1"])
def test_v2_gate_is_never_accepted_by_older_runtime_modes(tmp_path, configured):
    gate, _specification, _old = _gate(tmp_path)
    with pytest.raises(TypeError):
        _reward(gate, version=configured)


def test_v2_normal_cli_refuses_unconfirmed_artifact_before_any_model_allocation(tmp_path, monkeypatch):
    gate, _specification, _old = _gate(tmp_path / "mask")
    data = deepcopy(gate.artifact)
    data["bank_source"] = "real_frozen_policy_rollouts"  # Loader-only synthetic fixture.
    data.pop("payload_sha256")
    data["payload_sha256"] = canonical_sha256(data)
    artifact_path = tmp_path / "unconfirmed.json"
    artifact_path.write_text(json.dumps(data))
    cfg = _config(tmp_path)
    cfg.source_gate_calibration_path = str(artifact_path)
    monkeypatch.setattr(ppo, "_build_models", _ForbiddenComponent("model allocation"))
    with pytest.raises(ValueError, match="fresh confirmation"):
        ppo.run_phase3_ppo(cfg)
