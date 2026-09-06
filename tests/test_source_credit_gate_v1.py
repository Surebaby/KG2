"""Synthetic immutable source masks; no baseline or research labels are read."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.reward import source_credit_gate_v1 as credit
from kgproweight.reward import source_integrity_v1 as integrity
from kgproweight.reward.source_quality_gate_v1 import FEATURE_NAMES, FEATURE_VERSION, SourceQualityGateV1, canonical_sha256
from tests.test_source_gated_ppo_reward_v1 import _gate
from tests.test_source_integrity_v1 import _fixture


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bind(path):
    return {"path": str(path), "sha256": _sha(path)}


def _signed(data):
    data = deepcopy(data)
    data.pop("payload_sha256", None)
    data["payload_sha256"] = canonical_sha256(data)
    return data


def _write_release(tmp_path, status="PASS"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    record, evidence = _fixture()
    record.update(question_sha256=question_sha256("Where does Ada lead?"))
    source = tmp_path / "source.json"
    source.write_text("Synthetic names and source bytes")
    def rebind(value):
        if isinstance(value, dict):
            if "bindings" in value:
                value["bindings"] = {str(source): _sha(source)}
            for child in value.values():
                rebind(child)
        elif isinstance(value, list):
            for child in value:
                rebind(child)
    rebind(evidence)
    if status == "UNVERIFIED":
        evidence["entities"]["Q201"]["labels"] = ["Unsupported country"]
    elif status == "FAIL":
        evidence["entities"]["Q101"]["typed_edges"][0]["head_qid"] = "Q999"
    key = question_key(record["dataset"], record["qid"])
    spec = SimpleNamespace(query="Where does Ada lead?", kg_subgraph=deepcopy(record["kg_subgraph"]),
                           metadata={"dataset": record["dataset"], "qid": record["qid"],
                                     "source_quality_record": deepcopy(record)})
    row = {"dataset": record["dataset"], "qid": record["qid"], "question_key": key,
           "question": spec.query, "question_sha256": question_sha256(spec.query),
           "input_sha256": "c" * 64, "m_graph": 1,
           "source_record_sha256": canonical_sha256(record), "fullsource_record": record}
    result = integrity.validate_source_integrity_v1(record, evidence)
    assert result["status"] == status
    check = {**result, "question_key": key, "original_m_graph": 1, "input_sha256": row["input_sha256"]}
    inputs, checks, evidence_file = (tmp_path / name for name in ("inputs.jsonl", "checks.jsonl", "evidence.json"))
    inputs.write_text(json.dumps(row) + "\n")
    checks.write_text(json.dumps(check) + "\n")
    evidence_file.write_text(json.dumps(evidence))
    manifest = _signed({"schema_version": credit.MASK_SCHEMA, "mask_version": credit.MASK_VERSION,
                        "experiment_id": "SYNTHETIC-CREDIT-MASK",
                        "inputs": _bind(inputs), "question_checks": _bind(checks),
                        "source_evidence": _bind(evidence_file),
                        "verifier_code": _bind(Path(integrity.__file__))})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, spec


def _features(m_graph=1):
    return {"feature_version": FEATURE_VERSION,
            "values": dict(zip(FEATURE_NAMES, [.5, 1., 1., 1.])), "m_graph": m_graph,
            "hard_gate": {"m_graph": m_graph, "graph_eligible": bool(m_graph),
                          "checks": {"legacy_check": True}}, "telemetry": {"policy_entropy_used": False}}


def _artifact(mask):
    artifact = deepcopy(_gate().artifact)
    artifact.update(schema_version=credit.ARTIFACT_SCHEMA,
                    source_credit_version=credit.MASK_VERSION,
                    source_credit_scope=credit.CREDIT_SCOPE,
                    source_credit_clearance=True, source_integrity_clearance=False,
                    source_credit_mask={"path": str(mask.manifest_path), "sha256": mask.manifest_sha256,
                                        "payload_sha256": mask.payload_sha256})
    return _signed(artifact)


@pytest.mark.parametrize("status", ["PASS", "UNVERIFIED", "FAIL"])
def test_credit_changes_only_eligibility_not_four_features_or_original_inputs(tmp_path, status):
    path, spec = _write_release(tmp_path, status)
    mask = credit.FrozenSourceCreditMask.load(path)
    features = _features()
    before = deepcopy((spec.__dict__, features))
    result = mask.mask_features(spec, features)
    assert result["m_graph"] == (1 if status == "PASS" else 0)
    assert result["values"] == features["values"]
    assert result["source_credit_mask"]["status"] == status
    assert result["source_credit_mask"]["parent_m_graph"] == 1
    assert result["hard_gate"]["checks"]["source_credit_pass"] is (status == "PASS")
    assert (spec.__dict__, features) == before


@pytest.mark.parametrize("mutation", ["missing_record", "record_change", "query", "qid", "dataset", "visible_graph"])
def test_missing_or_changed_identity_record_and_prompt_graph_always_lose_credit(tmp_path, mutation):
    path, spec = _write_release(tmp_path)
    mask = credit.FrozenSourceCreditMask.load(path)
    if mutation == "missing_record":
        spec.metadata.pop("source_quality_record")
    elif mutation == "record_change":
        spec.metadata["source_quality_record"]["additional_unbound_field"] = "changed"
    elif mutation == "query":
        spec.query = "Different question"
    elif mutation in {"qid", "dataset"}:
        spec.metadata[mutation] = "not_in_mask"
    else:
        spec.kg_subgraph = []
    assert mask.mask_features(spec, _features())["m_graph"] == 0


def test_original_graph_ineligibility_cannot_be_overridden_by_source_pass(tmp_path):
    path, spec = _write_release(tmp_path)
    mask = credit.FrozenSourceCreditMask.load(path)
    assert mask.mask_features(spec, _features(0))["m_graph"] == 0


@pytest.mark.parametrize("file", ["inputs.jsonl", "checks.jsonl", "evidence.json", "source.json"])
def test_any_frozen_input_or_nested_evidence_byte_change_is_rejected(tmp_path, file):
    path, _spec = _write_release(tmp_path)
    target = tmp_path / file
    target.write_text(target.read_text() + " ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        credit.FrozenSourceCreditMask.load(path)


@pytest.mark.parametrize("mutation", ["status", "input_sha256", "question_key"])
def test_rehashing_a_forged_audit_cannot_bypass_record_reproduction(tmp_path, mutation):
    path, _spec = _write_release(tmp_path)
    checks_path = tmp_path / "checks.jsonl"
    check = json.loads(checks_path.read_text())
    check[mutation] = "FAIL" if mutation == "status" else "different"
    checks_path.write_text(json.dumps(check) + "\n")
    manifest = json.loads(path.read_text())
    manifest["question_checks"] = _bind(checks_path)
    path.write_text(json.dumps(_signed(manifest)))
    with pytest.raises(ValueError, match="identity|reproduce"):
        credit.FrozenSourceCreditMask.load(path)


def test_new_gate_rejects_unmasked_wrong_mask_or_modified_feature_views(tmp_path):
    path, spec = _write_release(tmp_path)
    mask = credit.FrozenSourceCreditMask.load(path)
    gate = credit.SourceCreditGateV1(_artifact(mask), mask=mask, allow_synthetic=True, allow_unvalidated=True)
    assert gate.predict(gate.mask_features(spec, _features())) == .5
    with pytest.raises(ValueError, match="processed by its frozen mask"):
        gate.predict(_features())
    masked = gate.mask_features(spec, _features())
    masked["source_credit_mask"]["mask_payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="processed by its frozen mask"):
        gate.predict(masked)
    masked = gate.mask_features(spec, _features())
    masked["values"]["cite_match"] = 0.
    with pytest.raises(ValueError, match="changed after masking"):
        gate.predict(masked)


@pytest.mark.parametrize("status", ["FAIL", "UNVERIFIED"])
def test_excluded_sources_predict_exact_zero_alpha(tmp_path, status):
    path, spec = _write_release(tmp_path, status)
    mask = credit.FrozenSourceCreditMask.load(path)
    gate = credit.SourceCreditGateV1(_artifact(mask), mask=mask, allow_synthetic=True, allow_unvalidated=True)
    assert gate.predict(gate.mask_features(spec, _features())) == 0.


@pytest.mark.parametrize("clearance", [False, None, 1, "true"])
def test_new_gate_requires_strict_credit_clearance_without_claiming_source_repair(tmp_path, clearance):
    path, _spec = _write_release(tmp_path)
    mask = credit.FrozenSourceCreditMask.load(path)
    artifact = _artifact(mask)
    artifact["source_credit_clearance"] = clearance
    with pytest.raises(ValueError, match="clearance mismatch"):
        credit.SourceCreditGateV1(_signed(artifact), mask=mask, allow_synthetic=True, allow_unvalidated=True)


def test_old_loader_rejects_new_schema_and_factory_dispatches_without_losing_identity(tmp_path):
    path, _spec = _write_release(tmp_path)
    mask = credit.FrozenSourceCreditMask.load(path)
    artifact = _artifact(mask)
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="legacy/unknown"):
        SourceQualityGateV1.load(gate_path, allow_synthetic=True, allow_unvalidated=True)
    gate = credit.load_source_quality_gate(gate_path, allow_synthetic=True, allow_unvalidated=True)
    assert isinstance(gate, credit.SourceCreditGateV1)
    assert gate.artifact == artifact
    assert gate.artifact["source_integrity_clearance"] is False
    old = tmp_path / "old_gate.json"
    old.write_text(json.dumps(_gate().artifact))
    assert type(credit.load_source_quality_gate(old, allow_synthetic=True, allow_unvalidated=True)) is SourceQualityGateV1
