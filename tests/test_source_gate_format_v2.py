"""Versioned empty-Final repair; CPU fixtures never update research artifacts."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from kgproweight.data.parsers import extract_final_answer
from kgproweight.reward.source_quality_gate_v1 import SourceQualityGateV1, canonical_sha256
import kgproweight.training.phase3_ppo as ppo
from kgproweight.training.reward_function import (
    KGProWeightRewardFunction, source_gate_format_contract_version,
    validate_source_gate_trajectory, validate_source_gate_trajectory_v1,
    validate_source_gate_trajectory_v2,
)
from scripts.prepare.resolve_phase3_ppo_runtime_config import resolve_phase3_ppo_runtime_config
from scripts.train.calibrate_source_quality_gate_v1 import calibrate, validate_bank
from tests.test_ppo_emf1_reward_contract_v1 import _ForbiddenComponent, _Tokenizer, _response
from tests.test_source_gated_ppo_reward_v1 import _gate, _score, _spec, _Text
from tests.test_source_quality_gate_v1 import _refresh_bindings, _write_bank


@pytest.mark.parametrize("suffix", ["", " \n\t", "]", "[]", "( )", "**[]**", "： ... {} 【】"])
def test_empty_or_decorated_final_is_invalid_without_changing_legacy_parser(suffix):
    response = _response(answer=suffix)
    legacy = validate_source_gate_trajectory_v1(_spec(), response)
    assert validate_source_gate_trajectory(_spec(), response) == legacy
    if suffix == "":
        assert extract_final_answer(response) == "]"  # Actual observed parser failure.
        assert legacy["valid"] is True
    repaired = validate_source_gate_trajectory(_spec(), response, format_version="v2")
    assert repaired["valid"] is False
    assert "final_answer_empty_or_decoration_only" in repaired["violations"]
    assert repaired["contract_version"] == "source-gate-runtime-v2-format-v2"


@pytest.mark.parametrize("answer", ["a", "0", "1", "yes", "no", "中", "é", "Москва", "نعم", "[A]"])
@pytest.mark.parametrize("heading", ["[Final Answer]", "Final Answer:", "**Final Answer**："])
def test_short_unicode_and_colon_final_answers_remain_valid(answer, heading):
    response = _response(answer=answer).replace("[Final Answer]", heading)
    assert validate_source_gate_trajectory_v1(_spec(), response)["valid"]
    assert validate_source_gate_trajectory_v2(_spec(), response)["valid"]


@pytest.mark.parametrize("version", ["v0", "v3", "source-gate-runtime-v2-format-v2", None])
def test_unknown_dispatch_version_rejected(version):
    with pytest.raises(ValueError, match="source_gate_format_version"):
        validate_source_gate_trajectory(_spec(), _response(), format_version=version)


def _versioned_gate(version="v2"):
    artifact = deepcopy(_gate().artifact)
    artifact["format_contract_version"] = source_gate_format_contract_version(version)
    if version == "v2":
        artifact.update(source_integrity_clearance=True, source_integrity_status="SYNTHETIC_UNIT_CLEARED")
    artifact.pop("payload_sha256")
    artifact["payload_sha256"] = canonical_sha256(artifact)
    return SourceQualityGateV1(artifact, allow_synthetic=True, allow_unvalidated=True)


def _reward(*, format_version="v2", gate=None, text=None, mode="learned"):
    return KGProWeightRewardFunction(
        alpha_gate=_ForbiddenComponent("legacy alpha"),
        prm_annotator=_ForbiddenComponent("PRM"),
        text_reward_model=text or _Text(), tokenizer=_Tokenizer(),
        outcome_weight=4., text_reward_scale=.3, max_steps=5,
        proofkg_process_reward=True, proofkg_process_version="v2_3",
        proofkg_process_weight=.2, proofkg_f1_weight=.1,
        proofkg_dynamic_validity=True, mixed_outcome_reward=True,
        mixed_text_reward=True, runtime_contract_version="v2",
        source_gated_reward_version="v1", source_gate_mode=mode,
        source_gate_format_version=format_version,
        source_quality_gate=gate or _versioned_gate(format_version),
        center_text_reward=False,
    )


@pytest.mark.parametrize("mode", ["text", "fixed", "learned"])
@pytest.mark.parametrize("eligible", [False, True])
def test_v2_invalid_reward_is_minus_four_and_never_calls_text_or_learned_gate(monkeypatch, mode, eligible):
    text = _Text()
    gate = _versioned_gate()
    monkeypatch.setattr(text, "score_steps", _ForbiddenComponent("ReaRAG score_steps"))
    monkeypatch.setattr(gate, "predict", _ForbiddenComponent("learned alpha"))
    result = _score(_reward(mode=mode, gate=gate, text=text), _spec(eligible), _response(answer=""))
    assert result["trajectory_valid"] is False
    assert result["trajectory_reward"] == -4.
    assert torch.count_nonzero(result["token_rewards"]) == 1
    assert result["token_rewards"][-1] == -4
    assert result["mixed_reward"]["text"] == result["mixed_reward"]["process"] == 0
    assert result["source_gate"]["invalid_not_scored"] is True
    assert result["source_gate"]["format_contract_version"] == source_gate_format_contract_version("v2")


def test_valid_v2_reward_preserves_numeric_v1_components_and_records_format():
    legacy = _score(_reward(format_version="v1", gate=_gate()))
    repaired = _score(_reward())
    for key in ("trajectory_valid", "trajectory_reward", "mixed_reward", "per_step_rewards"):
        assert repaired[key] == legacy[key]
    assert torch.equal(repaired["token_rewards"], legacy["token_rewards"])
    assert repaired["source_gate"]["format_contract_version"] == source_gate_format_contract_version("v2")


@pytest.mark.parametrize(("selected", "artifact"), [("v2", None), ("v2", "v1"), ("v1", "v2")])
def test_reward_rejects_mismatched_gate_format(selected, artifact):
    gate = _gate() if artifact is None else _versioned_gate(artifact)
    with pytest.raises(ValueError, match="format contract mismatch"):
        _reward(format_version=selected, gate=gate)


def test_cli_forwards_v2_and_legacy_config_defaults_v1(tmp_path):
    old = Path("configs/training/phase3_ppo_mixed4_sourcegate_v1_a_probe_seed42.yaml")
    assert resolve_phase3_ppo_runtime_config(old)["source_gate_format_version"] == "v1"
    config = yaml.safe_load(old.read_text())
    config["includes"] = [str((old.parent / path).resolve()) for path in config["includes"]]
    config["training"]["ppo"]["source_gate_format_version"] = "v2"
    path = tmp_path / "v2.yaml"
    path.write_text(yaml.safe_dump(config))
    runtime = resolve_phase3_ppo_runtime_config(path)
    assert runtime["source_gate_format_version"] == "v2"
    ppo._validate_mixed_reward_config(ppo.Phase3PPOConfig(**runtime))


def test_ppo_rejects_old_gate_before_any_model_allocation(monkeypatch):
    cfg = ppo.Phase3PPOConfig(**resolve_phase3_ppo_runtime_config(
        "configs/training/phase3_ppo_mixed4_sourcegate_v1_a_probe_seed42.yaml"))
    cfg.source_gate_format_version = "v2"
    monkeypatch.setattr(SourceQualityGateV1, "load", lambda _path: _gate())
    monkeypatch.setattr(ppo, "_build_models", _ForbiddenComponent("policy load"))
    with pytest.raises(ValueError, match="format contract mismatch"):
        ppo.run_phase3_ppo(cfg)


def _upgrade_fixture_bank(manifest_path, isolation_path):
    manifest = json.loads(manifest_path.read_text())
    manifest["format_contract_version"] = source_gate_format_contract_version("v2")
    manifest.update(source_integrity_clearance=False, source_integrity_status="LABEL_PROJECTION_REPAIR_PENDING")
    manifest_path.write_text(json.dumps(manifest))
    path = manifest_path.parent / "rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    # Row 0 is the same concrete boundary as the observed real 384-token cap.
    rows[0]["generation"] = rows[0]["generation"].rsplit("Gamma", 1)[0]
    rows[0]["raw_text"] = []
    for row in rows:
        record = row["source_quality_record"]
        spec = SimpleNamespace(query=row["question"], kg_subgraph=record.get("kg_subgraph") or [],
                               retrieved_passages=row["retrieved_passages"],
                               metadata={"dataset": row["dataset"], "qid": row["qid"],
                                         "source_quality_record": record})
        validation = validate_source_gate_trajectory_v2(spec, row["generation"])
        row["trajectory_valid"] = validation["valid"]
        row["format_validation"] = {key: validation[key] for key in (
            "valid", "violations", "all_step_count", "required_steps", "contract_version")}
        if validation["source_features"]["m_graph"]:
            from kgproweight.reward.proofkg_process_v2_3 import build_execution_trace_v2_3, score_proofkg_v2_3
            row["proof_result"] = score_proofkg_v2_3(
                question=spec.query, generation=row["generation"], kg_triples=spec.kg_subgraph,
                execution_trace=build_execution_trace_v2_3(record["query_plan"], record["execution"]),
                planned_hops=len(record["query_plan"]["hops"]),
            )
            row["raw_graph"] = row["proof_result"]["score"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_bindings(manifest_path, isolation_path)


def test_calibrator_selects_manifest_format_and_records_same_artifact_contract(tmp_path):
    manifest, isolation = _write_bank(tmp_path / "bank")
    _upgrade_fixture_bank(manifest, isolation)
    rows, binding = validate_bank(manifest, isolation, synthetic_test_only=True)
    assert rows[0]["trajectory_valid"] is False
    assert rows[0]["quality"]["target"] is None
    assert binding["format_contract_version"] == source_gate_format_contract_version("v2")
    output = tmp_path / "gate"
    calibrate(manifest, isolation, output, experiment_id="UNIT-FORMAT-V2", epochs=20,
              synthetic_test_only=True)
    for name in ("gate.json", "report.json", "manifest.json"):
        result = json.loads((output / name).read_text())
        assert result["format_contract_version"] == source_gate_format_contract_version("v2")
        assert result["source_integrity_clearance"] is False
        assert result["source_integrity_status"] == "LABEL_PROJECTION_REPAIR_PENDING"


def test_manifest_cannot_claim_v2_for_unmodified_v1_validity_rows(tmp_path):
    manifest, isolation = _write_bank(tmp_path / "bank")
    data = json.loads(manifest.read_text())
    data["format_contract_version"] = source_gate_format_contract_version("v2")
    data.update(source_integrity_clearance=False, source_integrity_status="LABEL_PROJECTION_REPAIR_PENDING")
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="shared PPO format contract"):
        validate_bank(manifest, isolation, synthetic_test_only=True)


@pytest.mark.parametrize("clearance", [False, None, 1, "true"])
def test_v2_source_integrity_blocks_reward_and_ppo_before_allocation(monkeypatch, clearance):
    artifact = deepcopy(_versioned_gate().artifact)
    artifact["source_integrity_clearance"] = clearance
    artifact["source_integrity_status"] = "LABEL_PROJECTION_REPAIR_PENDING"
    artifact.pop("payload_sha256")
    artifact["payload_sha256"] = canonical_sha256(artifact)
    gate = SourceQualityGateV1(artifact, allow_synthetic=True, allow_unvalidated=True)
    with pytest.raises(ValueError, match="source_integrity_clearance=true"):
        _reward(gate=gate)
    cfg = ppo.Phase3PPOConfig(**resolve_phase3_ppo_runtime_config(
        "configs/training/phase3_ppo_mixed4_sourcegate_v1_a_probe_seed42.yaml"))
    cfg.source_gate_format_version = "v2"
    monkeypatch.setattr(SourceQualityGateV1, "load", lambda _path: gate)
    monkeypatch.setattr(ppo, "_build_models", _ForbiddenComponent("policy load"))
    with pytest.raises(ValueError, match="source_integrity_clearance=true"):
        ppo.run_phase3_ppo(cfg)


def test_v2_bank_cannot_omit_source_integrity_contract(tmp_path):
    manifest, isolation = _write_bank(tmp_path / "bank")
    data = json.loads(manifest.read_text())
    data["format_contract_version"] = source_gate_format_contract_version("v2")
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="explicit source_integrity"):
        validate_bank(manifest, isolation, synthetic_test_only=True)
