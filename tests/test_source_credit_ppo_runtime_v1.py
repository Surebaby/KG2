"""CPU source-credit PPO contracts; immutable synthetic masks, no model loads."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

import kgproweight.training.phase3_ppo as ppo
from kgproweight.reward import source_credit_gate_v1 as credit
from kgproweight.reward import source_integrity_v1 as integrity
from kgproweight.reward.source_quality_gate_v1 import canonical_sha256
from kgproweight.training.reward_function import KGProWeightRewardFunction
from scripts.prepare.resolve_phase3_ppo_runtime_config import resolve_phase3_ppo_runtime_config
from tests.test_source_credit_gate_v1 import _artifact, _bind, _signed
from tests.test_source_gated_ppo_reward_v1 import _spec, _Text, _score
from tests.test_source_gate_format_v2 import _versioned_gate
from tests.test_ppo_emf1_reward_contract_v1 import _ForbiddenComponent, _Tokenizer, _response


def _bound_gate(tmp_path, status="PASS"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    spec = _spec()
    record = spec.metadata["source_quality_record"]
    record["execution"]["anchor_entities"] = {"Alpha": {"qid": "Q1", "surface": "Alpha"}}
    source = tmp_path / "synthetic_source.txt"
    source.write_text("Synthetic typed source; never research data.")
    bindings = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    entities = {f"Q{i}": {"labels": [name], "aliases": [], "demonyms": [],
                            "bindings": bindings, "typed_edges": []}
                for i, name in enumerate(("Alpha", "Beta", "Gamma"), 1)}
    for i, hop in enumerate(record["execution"]["hops"], 1):
        hop.update(pids=[f"P{i}"], match_sources=["store"])
        head, relation, tail = hop["matches"][0]
        entities[f"Q{i}"]["typed_edges"] = [{"head_qid": f"Q{i}", "pid": f"P{i}",
            "relation": relation, "head_label": head, "tail_qid": f"Q{i+1}",
            "tail_value": tail, "source": "store", "bindings": bindings}]
    if status == "UNVERIFIED":
        entities["Q3"]["labels"] = ["Unconfirmed terminal name"]
    elif status == "FAIL":
        entities["Q1"]["typed_edges"][0]["head_qid"] = "Q999"
    evidence = {"schema_version": "qid-source-evidence-v1", "bindings": bindings, "entities": entities}
    verdict = integrity.validate_source_integrity_v1(record, evidence)
    assert verdict["status"] == status
    key = "2wikimultihopqa::q1"
    row = {"dataset": "2wikimultihopqa", "qid": "q1", "question_key": key,
           "question": spec.query, "question_sha256": record["question_sha256"],
           "input_sha256": "c" * 64, "m_graph": 1,
           "source_record_sha256": canonical_sha256(record), "fullsource_record": record}
    check = {**verdict, "question_key": key, "original_m_graph": 1, "input_sha256": row["input_sha256"]}
    inputs, checks, evidence_path = (tmp_path / p for p in ("inputs.jsonl", "checks.jsonl", "evidence.json"))
    inputs.write_text(json.dumps(row) + "\n")
    checks.write_text(json.dumps(check) + "\n")
    evidence_path.write_text(json.dumps(evidence))
    manifest = _signed({"schema_version": credit.MASK_SCHEMA, "mask_version": credit.MASK_VERSION,
       "experiment_id": "SYNTHETIC-PPO-SOURCE-CREDIT", "inputs": _bind(inputs),
       "question_checks": _bind(checks), "source_evidence": _bind(evidence_path),
       "verifier_code": _bind(Path(integrity.__file__))})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    mask = credit.FrozenSourceCreditMask.load(manifest_path)
    artifact = _artifact(mask)
    artifact["format_contract_version"] = "source-gate-runtime-v2-format-v2"
    gate = credit.SourceCreditGateV1(_signed(artifact), mask=mask,
                                   allow_synthetic=True, allow_unvalidated=True)
    return gate, spec


def _reward(gate, text=None, mode="learned", credit_version="v1", format_version="v2"):
    return KGProWeightRewardFunction(
        alpha_gate=_ForbiddenComponent("legacy alpha"), prm_annotator=_ForbiddenComponent("PRM"),
        text_reward_model=text or _Text(), tokenizer=_Tokenizer(),
        outcome_weight=4., text_reward_scale=.3, max_steps=5,
        proofkg_process_reward=True, proofkg_process_version="v2_3",
        proofkg_process_weight=.2, proofkg_f1_weight=.1, proofkg_dynamic_validity=True,
        mixed_outcome_reward=True, mixed_text_reward=True, runtime_contract_version="v2",
        source_gated_reward_version="v1", source_gate_format_version=format_version,
        source_gate_credit_version=credit_version, source_gate_mode=mode,
        source_quality_gate=gate, center_text_reward=False,
    )


@pytest.mark.parametrize("mode", ["learned", "fixed", "text"])
@pytest.mark.parametrize("status", ["UNVERIFIED", "FAIL"])
def test_excluded_graph_has_zero_credit_but_retains_text_and_outcome(tmp_path, mode, status):
    gate, spec = _bound_gate(tmp_path, status)
    before = deepcopy(spec)
    text = _Text()
    result = _score(_reward(gate, text, mode), spec)
    details = result["source_gate"]
    assert details["m_graph"] == 0 and details["alpha_effective"] == 0.
    assert details["source_credit_version"] == "v1"
    assert details["source_credit_mask"]["status"] == status
    assert details["source_credit_mask"]["mask_payload_sha256"] == gate.mask.payload_sha256
    assert result["mixed_reward"]["process"] == 0.
    assert result["mixed_reward"]["text"] == pytest.approx(.05)
    assert result["mixed_reward"]["outcome"] == pytest.approx(4.4)
    assert len(text.calls) == 1 and spec == before
    assert gate.artifact["source_integrity_clearance"] is False  # Input was not repaired.


@pytest.mark.parametrize("mode", ["learned", "fixed", "text"])
def test_pass_preserves_legacy_reward_numbers_and_token_placement(tmp_path, mode):
    gate, spec = _bound_gate(tmp_path)
    new = _score(_reward(gate, mode=mode), spec)
    old = _score(_reward(_versioned_gate(), mode=mode, credit_version="disabled"), spec)
    for field in ("trajectory_valid", "trajectory_reward", "mixed_reward", "per_step_rewards"):
        assert new[field] == old[field]
    assert torch.equal(new["token_rewards"], old["token_rewards"])
    assert new["source_gate"]["source_credit_mask"]["status"] == "PASS"


@pytest.mark.parametrize("mutation", ["query", "visible_graph", "source_record", "missing_record"])
def test_changed_runtime_input_loses_graph_credit(tmp_path, mutation):
    gate, spec = _bound_gate(tmp_path)
    if mutation == "query": spec.query += " changed"
    elif mutation == "visible_graph": spec.kg_subgraph = []
    elif mutation == "source_record": spec.metadata["source_quality_record"]["extra"] = "changed"
    else: spec.metadata.pop("source_quality_record")
    result = _score(_reward(gate, mode="fixed"), spec)
    assert result["source_gate"]["alpha_effective"] == 0.
    assert result["mixed_reward"]["process"] == 0.
    assert result["source_gate"]["source_credit_mask"]["source_credit_pass"] is False


def test_ordinary_no_graph_cannot_gain_credit_from_missing_mask_identity(tmp_path):
    gate, spec = _bound_gate(tmp_path)
    spec.kg_subgraph = []
    spec.metadata = {"dataset": "hotpotqa", "qid": "ordinary"}
    result = _score(_reward(gate, mode="fixed"), spec)
    assert result["source_gate"]["m_graph"] == 0
    assert result["source_gate"]["alpha_effective"] == 0
    assert result["mixed_reward"]["text"] == pytest.approx(.05)


def test_mask_does_not_change_format_required_steps_or_invalid_penalty(tmp_path):
    gate, spec = _bound_gate(tmp_path, "UNVERIFIED")
    two_steps = _score(_reward(gate), spec, _response(steps=2))
    assert two_steps["trajectory_valid"] is True
    assert two_steps["source_gate"]["m_graph"] == 0
    invalid = _score(_reward(gate), spec, _response(answer=""))
    assert invalid["trajectory_reward"] == -4.
    assert invalid["source_gate"]["source_credit_mask"]["status"] == "UNVERIFIED"


def test_new_gate_and_legacy_configuration_reject_each_other(tmp_path):
    gate, _specification = _bound_gate(tmp_path)
    with pytest.raises(TypeError, match="disabled rejects"):
        _reward(gate, credit_version="disabled")
    with pytest.raises(TypeError, match="requires a validated SourceCreditGateV1"):
        _reward(_versioned_gate())
    with pytest.raises(ValueError, match="source credit v1 requires"):
        _reward(gate, format_version="v1")


@pytest.mark.parametrize("clearance", [False, None, 1, "true"])
def test_runtime_requires_strict_credit_clearance(tmp_path, clearance):
    gate, _specification = _bound_gate(tmp_path)
    gate.artifact["source_credit_clearance"] = clearance
    with pytest.raises(ValueError, match="source credit clearance"):
        _reward(gate)


def test_runtime_rechecks_gate_mask_binding(tmp_path):
    gate, _specification = _bound_gate(tmp_path)
    gate.artifact["source_credit_mask"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen mask binding"):
        _reward(gate)


def _config(tmp_path):
    base = Path("configs/training/phase3_ppo_mixed4_sourcegate_v1_a_probe_seed42.yaml").resolve()
    path = tmp_path / "source_credit.yaml"
    path.write_text(yaml.safe_dump({"includes": [str(base)], "training": {"ppo": {
        "source_gate_format_version": "v2", "source_gate_credit_version": "v1"}}}))
    return ppo.Phase3PPOConfig(**resolve_phase3_ppo_runtime_config(path))


def test_yaml_schema_cli_and_runtime_forward_explicit_credit_opt_in(tmp_path):
    cfg = _config(tmp_path)
    assert cfg.source_gate_credit_version == "v1"
    assert cfg.source_gate_format_version == "v2"
    ppo._validate_mixed_reward_config(cfg)
    for changes in ({"source_gate_format_version": "v1"},
                    {"source_gated_reward_version": "disabled"},
                    {"source_gate_credit_version": "unknown_future_version"}):
        with pytest.raises(ValueError, match="source.*credit"):
            ppo._validate_mixed_reward_config(replace(cfg, **changes))
    old = resolve_phase3_ppo_runtime_config("configs/training/phase3_ppo_mixed4_sourcegate_v1_a_probe_seed42.yaml")
    assert old["source_gate_credit_version"] == "disabled"


def test_credit_loader_selected_before_any_model_allocation(monkeypatch, tmp_path):
    cfg = _config(tmp_path)
    def selected(_path): raise RuntimeError("SOURCE_CREDIT_LOADER_SELECTED")
    monkeypatch.setattr(credit.SourceCreditGateV1, "load", selected)
    monkeypatch.setattr(ppo, "_build_models", _ForbiddenComponent("policy load"))
    with pytest.raises(RuntimeError, match="SOURCE_CREDIT_LOADER_SELECTED"):
        ppo.run_phase3_ppo(cfg)
