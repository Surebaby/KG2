"""Frozen-protocol and train/development-boundary regression checks."""
from copy import deepcopy
import json

import pytest

from kgproweight.reward.source_trajectory_features_v2 import FEATURE_NAMES, FEATURE_VERSION
from scripts.train import calibrate_source_credit_gate_v2 as module


def _protocol(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    manifest = parent / "manifest.json"
    manifest.write_text("{}\n")
    (parent / "gate.json").write_text("{}\n")
    protocol = {"schema_version": module.PROTOCOL_SCHEMA,
                "experiment_id": "SYNTHETIC-V2-BOUNDARY-CHECK",
                "feature_names": list(FEATURE_NAMES), "feature_version": FEATURE_VERSION,
                "seed": 42, "epochs": 800, "variants": ["norm_only", "features_v2"],
                "parent_manifest": module.identity(manifest),
                "code_bindings": {name: module.identity(module.ROOT / name) for name in module.CODE_FILES}}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    return parent, path, protocol


@pytest.mark.parametrize("mutation", ["empty_code", "missing_code", "wrong_code_path", "changed_code_sha", "feature_design", "budget", "variant_scan"])
def test_incomplete_or_changed_protocol_is_rejected_before_loading_rows(tmp_path, monkeypatch, mutation):
    parent, path, protocol = _protocol(tmp_path)
    if mutation == "empty_code":
        protocol["code_bindings"] = {}
    elif mutation == "missing_code":
        protocol["code_bindings"].pop(next(iter(protocol["code_bindings"])))
    elif mutation == "wrong_code_path":
        name = next(iter(protocol["code_bindings"]))
        other = tmp_path / "same_bytes.py"
        other.write_bytes((module.ROOT / name).read_bytes())
        protocol["code_bindings"][name] = module.identity(other)
    elif mutation == "changed_code_sha":
        protocol["code_bindings"][next(iter(protocol["code_bindings"]))]["sha256"] = "0" * 64
    elif mutation == "feature_design":
        protocol["feature_names"].append("answer_consistency")
    elif mutation == "budget":
        protocol["epochs"] += 1
    else:
        protocol["variants"].append("scan_more_on_confirmation")
    path.write_text(json.dumps(protocol))
    def forbidden(*args):
        raise AssertionError("invalid protocol must fail before parent candidate access")
    monkeypatch.setattr(module, "load_parent", forbidden)
    with pytest.raises(ValueError):
        module.calibrate(parent, path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def _fit_rows():
    return [{"split": "train", "features": {"values": {"x": x}}, "quality": {"target": x}}
            for x in (0., 0., 1., 1.)]


@pytest.mark.parametrize("field", ["split", "family_split"])
def test_fit_rejects_explicit_consumed_holdout_membership(field):
    rows = _fit_rows()
    rows[0][field] = "confirmation"
    with pytest.raises(ValueError, match="train-only"):
        module.fit_logistic(rows, ("x",))


@pytest.mark.parametrize("epochs", [0, -1, True, 2.5])
def test_fit_requires_a_positive_fixed_integer_budget(epochs):
    with pytest.raises(ValueError, match="epochs"):
        module.fit_logistic(_fit_rows(), ("x",), epochs=epochs)


def test_train_fit_is_reproducible_and_learns_a_synthetic_direction_without_answer_labels():
    first = module.fit_logistic(_fit_rows(), ("x",))
    assert first == module.fit_logistic(_fit_rows(), ("x",))
    assert first["weights"][0] > 1
    assert first["feature_standardization"]["fit_split"] == "train_eligible_nonabstain"


def _flow_rows():
    rows = []
    splits = {"train-family": "train", "calibration-family": "calibration", "confirmation-family": "confirmation"}
    for split in ("train", "calibration", "confirmation"):
        for index in range(2):
            rows.append({"candidate_id": f"{split}-{index}", "dataset": "synthetic", "qid": split,
                         "family_sha256": f"{split}-family", "trajectory_valid": True,
                         "raw_text": [.2], "raw_graph": .4,
                         "features": {"m_graph": 1, "values": {"x": index}},
                         "quality": {"target": .5, "q_graph": .5, "q_text": .5, "abstain_reason": None}})
    return rows, splits


class _StopAtFit(Exception):
    pass


@pytest.mark.parametrize("stage", ["text", "gate"])
def test_orchestrator_supplies_only_train_members_to_each_fit(tmp_path, monkeypatch, stage):
    parent, protocol, _ = _protocol(tmp_path)
    rows, splits = _flow_rows()
    parent_artifact = {"payload_sha256": "synthetic", "normalization": {}, "fit": {}}
    monkeypatch.setattr(module, "load_parent", lambda path: (deepcopy(rows), splits, object(), parent_artifact))
    observed = []
    def inspect_train(selected, *args, **kwargs):
        observed.extend(selected)
        assert {r["family_sha256"] for r in selected} == {"train-family"}
        assert all(r["split"] == "train" for r in selected)
        raise _StopAtFit()
    if stage == "text":
        monkeypatch.setattr(module, "fit_text_normalization_v2", inspect_train)
    else:
        monkeypatch.setattr(module, "derive_feature_rows", lambda rows, mask: deepcopy(rows))
        monkeypatch.setattr(module, "reanalysis_metrics", lambda *args: {})
        class DiagnosticGate:
            def __init__(self, artifact, **kwargs):
                assert artifact["training_clearance"] is False
                assert artifact["independent_confirmation_clearance"] is False
                assert artifact["ppo_launch_clearance"] is False
                assert kwargs["allow_unvalidated"] is True
            def predict(self, features):
                return .5
        monkeypatch.setattr(module, "SourceCreditGateV2", DiagnosticGate)
        monkeypatch.setattr(module, "fit_logistic", inspect_train)
    with pytest.raises(_StopAtFit):
        module.calibrate(parent, protocol, tmp_path / "out")
    assert len(observed) == 2
    assert (tmp_path / "out" / "FAILED.json").exists()
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_reanalysis_marks_consumed_splits_as_development_without_clearance():
    rows, splits = _flow_rows()
    class Gate:
        def predict(self, features):
            return .5
    report = module.reanalysis_metrics(rows, splits, Gate())
    assert report["train"]["interpretation"] == "train_fit"
    for split in ("calibration", "confirmation"):
        assert report[split]["interpretation"] == "already_consumed_split_development_reanalysis"
        assert "training_clearance" not in report[split]


def test_invalid_six_step_derivation_uses_the_same_five_step_feature_view_as_runtime():
    from kgproweight.reward.source_trajectory_features_v2 import compute_gate_features_v2
    from kgproweight.training.reward_function import validate_source_gate_trajectory
    from tests.test_source_trajectory_features_v2 import _fixture
    spec, _, proof = _fixture()
    generation = "\n".join(
        f"[Step {index}]\nReasoning: The frozen source provides a concrete relational fact.\n"
        f"Knowledge Used: [{'(Alpha, links, Beta)' if index < 6 else '(Unsupported, links, Fiction)'}]\n"
        "Conclusion: This records the cited relation."
        for index in range(1, 7)
    ) + "\n[Final Answer]\nGamma"
    validity = validate_source_gate_trajectory(spec, generation, format_version="v2")
    assert validity["valid"] is False and "too_many_steps" in validity["violations"]
    assert len(validity["steps"]) == 5 and validity["all_step_count"] == 6
    expected = compute_gate_features_v2(spec, validity["steps"], proof)
    parent = {"candidate_id": "synthetic-six-step", "dataset": spec.metadata["dataset"],
              "qid": spec.metadata["qid"], "question": spec.query,
              "source_quality_record": spec.metadata["source_quality_record"],
              "retrieved_passages": [], "generation": generation, "proof_result": proof,
              "features": expected, "trajectory_valid": False}
    # This spy isolates step selection; immutable mask verification is tested
    # separately with real source-bound release fixtures in gate-v2 tests.
    class FeatureViewSpy:
        def mask_features(self, spec, features):
            return deepcopy(features)
    result = module.derive_feature_rows([parent], FeatureViewSpy())
    assert result[0]["features"] == expected
    assert result[0]["generation"] == generation
    assert result[0]["trajectory_valid"] is False


@pytest.mark.parametrize("changed", ["protocol", "parent_manifest"])
def test_midrun_frozen_binding_change_retains_failure_and_cannot_publish_manifest(tmp_path, monkeypatch, changed):
    parent, protocol, _ = _protocol(tmp_path)
    rows, splits = _flow_rows()
    (parent / "assignments.jsonl").write_text("".join(
        json.dumps({"candidate_id": row["candidate_id"], "alpha": .5}) + "\n" for row in rows))
    artifact = {"payload_sha256": "synthetic", "normalization": {}, "fit": {}, "source_credit_mask": {}}
    monkeypatch.setattr(module, "load_parent", lambda path: (deepcopy(rows), splits, object(), artifact))
    def mutate_binding(selected, mask):
        target = protocol if changed == "protocol" else parent / "manifest.json"
        target.write_text(target.read_text() + " ")
        return deepcopy(selected)
    monkeypatch.setattr(module, "derive_feature_rows", mutate_binding)
    monkeypatch.setattr(module, "reanalysis_metrics", lambda *args: {})
    monkeypatch.setattr(module, "fit_logistic", lambda *args, **kwargs: {})
    class DiagnosticGate:
        def __init__(self, artifact, **kwargs):
            assert artifact["training_clearance"] is False
            assert artifact["independent_confirmation_clearance"] is False
        def predict(self, features):
            return .5
    monkeypatch.setattr(module, "SourceCreditGateV2", DiagnosticGate)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="changed"):
        module.calibrate(parent, protocol, output)
    assert not (output / "manifest.json").exists()
    assert (output / "FAILED.json").exists()
    for variant in ("norm_only", "features_v2"):
        written = json.loads((output / variant / "gate.json").read_text())
        assert written["training_clearance"] is False
        assert written["ppo_launch_clearance"] is False
