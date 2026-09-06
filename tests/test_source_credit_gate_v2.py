"""Independent schema, frozen-mask and normalization-control checks."""
from copy import deepcopy
import json

import pytest

from kgproweight.reward.source_credit_gate_v1 import FrozenSourceCreditMask, SourceCreditGateV1
from kgproweight.reward.source_credit_gate_v2 import (
    ARTIFACT_SCHEMA, GATE_VERSION, NORMALIZATION_CONTRACT, SourceCreditGateV2,
)
from kgproweight.reward.source_quality_gate_v1 import (
    FEATURE_NAMES as V1_NAMES, FEATURE_VERSION as V1_VERSION, SourceQualityGateV1,
)
from kgproweight.reward.source_reward_normalization_v2 import fit_text_normalization_v2, normalize_text_steps_v2
from kgproweight.reward.source_trajectory_features_v2 import FEATURE_NAMES, FEATURE_VERSION
from tests.test_source_credit_gate_v1 import _artifact, _features, _signed, _write_release
from tests.test_source_trajectory_features_v2 import _fixture


def _v2_artifact(mask, version=V1_VERSION):
    artifact = _artifact(mask)
    names = V1_NAMES if version == V1_VERSION else FEATURE_NAMES
    artifact.update(schema_version=ARTIFACT_SCHEMA, gate_version=GATE_VERSION,
                    format_contract_version="source-gate-runtime-v2-format-v2",
                    feature_version=version, feature_names=list(names),
                    training_clearance=False, independent_confirmation_clearance=False,
                    ppo_launch_clearance=False,
                    weights=[.11 * (i + 1) for i in range(len(names))], bias=.17,
                    feature_standardization={"mean": dict.fromkeys(names, .2),
                                             "scale": dict.fromkeys(names, .3)})
    text = fit_text_normalization_v2([
        {"dataset": "synthetic", "qid": "a", "candidate_id": "a0", "split": "train",
         "trajectory_valid": True, "raw_text": [-.4, .2, .8]},
    ])
    artifact["normalization"].pop("application_clip", None)
    artifact["normalization"].update(input_contract=NORMALIZATION_CONTRACT, text_v2=text,
        text_center=text["text_center"], text_scale=text["text_scale"],
        text_application_scope=text["application_contract"])
    return _signed(artifact)


def _load(mask, artifact):
    return SourceCreditGateV2(artifact, mask=mask, allow_synthetic=True, allow_unvalidated=True)


def _v2_features(version):
    features = _features()
    if version == FEATURE_VERSION:
        features["feature_version"] = version
        features["values"].update(source_edge_coverage=.75, min_step_citation_precision=.5)
    return features


@pytest.mark.parametrize("version", [V1_VERSION, FEATURE_VERSION])
def test_diagnostic_load_and_explicit_registered_feature_dispatch(tmp_path, version):
    path, _ = _write_release(tmp_path)
    mask = FrozenSourceCreditMask.load(path)
    artifact = _v2_artifact(mask, version)
    gate = _load(mask, artifact)
    spec, steps, proof = _fixture()
    features = gate.compute_features(spec, steps, proof)
    assert features["feature_version"] == version
    assert tuple(features["values"]) == tuple(artifact["feature_names"])
    assert gate.artifact == artifact
    assert gate.artifact["training_clearance"] is False
    assert gate.artifact["source_integrity_clearance"] is False


@pytest.mark.parametrize("status", ["PASS", "FAIL", "UNVERIFIED"])
@pytest.mark.parametrize("version", [V1_VERSION, FEATURE_VERSION])
def test_mask_population_decision_is_independent_of_feature_schema(tmp_path, status, version):
    path, spec = _write_release(tmp_path, status)
    mask = FrozenSourceCreditMask.load(path)
    gate = _load(mask, _v2_artifact(mask, version))
    masked = gate.mask_features(spec, _v2_features(version))
    assert masked["m_graph"] == int(status == "PASS")
    prediction = gate.predict(masked)
    assert (0 < prediction < 1) if status == "PASS" else prediction == 0


def test_norm_only_predictor_is_exactly_the_old_predictor_for_each_feature_view(tmp_path):
    path, spec = _write_release(tmp_path)
    mask = FrozenSourceCreditMask.load(path)
    old_artifact = _artifact(mask)
    old_artifact.update(weights=[.11, -.2, .3, -.47], bias=.19,
                        feature_standardization={"mean": dict.fromkeys(V1_NAMES, .3),
                                                 "scale": dict.fromkeys(V1_NAMES, .4)})
    old = SourceCreditGateV1(_signed(old_artifact), mask=mask, allow_synthetic=True, allow_unvalidated=True)
    artifact = _v2_artifact(mask)
    for key in ("weights", "bias", "feature_standardization"):
        artifact[key] = deepcopy(old.artifact[key])
    new = _load(mask, _signed(artifact))
    for value in (0., .1, .49, .75, 1.):
        features = _features()
        features["values"].update(density=value * 2, link_confidence=1 - value,
                                  cite_any=float(value > 0), cite_match=value)
        masked = mask.mask_features(spec, features)
        assert new.predict(masked) == old.predict(masked)


def test_unconfirmed_gate_cannot_load_in_production_even_from_real_bank(tmp_path):
    path, _ = _write_release(tmp_path)
    mask = FrozenSourceCreditMask.load(path)
    artifact = _v2_artifact(mask)
    artifact["bank_source"] = "real_frozen_policy_rollouts"
    artifact = _signed(artifact)
    file = tmp_path / "gate.json"
    file.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="fresh confirmation"):
        SourceCreditGateV2.load(file)
    assert SourceCreditGateV2.load(file, allow_unvalidated=True).artifact == artifact


@pytest.mark.parametrize("value", [False, None, 1, "true"])
def test_training_claim_requires_strict_independent_confirmation_even_for_diagnostics(tmp_path, value):
    path, _ = _write_release(tmp_path)
    mask = FrozenSourceCreditMask.load(path)
    artifact = _v2_artifact(mask)
    artifact.update(training_clearance=True, independent_confirmation_clearance=value)
    with pytest.raises(ValueError, match="development diagnostics"):
        _load(mask, _signed(artifact))


@pytest.mark.parametrize("value", [False, None, 1, "true"])
def test_source_credit_clearance_is_strict_boolean(tmp_path, value):
    path, _ = _write_release(tmp_path)
    mask = FrozenSourceCreditMask.load(path)
    artifact = _v2_artifact(mask)
    artifact["source_credit_clearance"] = value
    with pytest.raises(ValueError, match="clearance mismatch"):
        _load(mask, _signed(artifact))


@pytest.mark.parametrize("field", ["sha256", "payload_sha256"])
def test_gate_cannot_bind_different_mask_hash(tmp_path, field):
    path, _ = _write_release(tmp_path)
    mask = FrozenSourceCreditMask.load(path)
    artifact = _v2_artifact(mask)
    artifact["source_credit_mask"][field] = "0" * 64
    with pytest.raises(ValueError, match="mask binding mismatch"):
        _load(mask, _signed(artifact))


def test_unmasked_changed_or_different_schema_features_cannot_predict(tmp_path):
    path, spec = _write_release(tmp_path)
    mask = FrozenSourceCreditMask.load(path)
    gate = _load(mask, _v2_artifact(mask, FEATURE_VERSION))
    with pytest.raises(ValueError, match="frozen mask"):
        gate.predict(_v2_features(FEATURE_VERSION))
    masked = gate.mask_features(spec, _v2_features(FEATURE_VERSION))
    masked["values"]["source_edge_coverage"] = 0.
    with pytest.raises(ValueError, match="changed after masking"):
        gate.predict(masked)
    with pytest.raises(ValueError, match="feature version or fields"):
        gate.predict(gate.mask_features(spec, _features()))


def test_old_and_new_artifact_schemas_are_bidirectionally_isolated(tmp_path):
    path, _ = _write_release(tmp_path)
    mask = FrozenSourceCreditMask.load(path)
    old = _artifact(mask)
    new = _v2_artifact(mask)
    with pytest.raises(ValueError, match="schema"):
        _load(mask, old)
    with pytest.raises(ValueError, match="schema"):
        SourceCreditGateV1(new, mask=mask, allow_synthetic=True, allow_unvalidated=True)
    with pytest.raises(ValueError, match="legacy/unknown"):
        SourceQualityGateV1(new, allow_synthetic=True, allow_unvalidated=True)


@pytest.mark.parametrize("mutation", ["text_center", "text_scale", "scope", "step_unit", "hard_clip", "feature_order", "weights"])
def test_normalization_mirror_and_feature_bindings_are_strict(tmp_path, mutation):
    path, _ = _write_release(tmp_path)
    mask = FrozenSourceCreditMask.load(path)
    artifact = _v2_artifact(mask, FEATURE_VERSION)
    norm = artifact["normalization"]
    if mutation in {"text_center", "text_scale"}:
        norm[mutation] += .01
    elif mutation == "scope":
        norm["text_application_scope"] = "trajectory_mean_then_clip"
    elif mutation == "step_unit":
        norm["text_v2"]["fit_unit"] = "trajectory_mean"
    elif mutation == "hard_clip":
        norm["text_v2"]["hard_clipping_used"] = True
    elif mutation == "feature_order":
        artifact["feature_names"].reverse()
    else:
        artifact["weights"].pop()
    with pytest.raises(ValueError):
        _load(mask, _signed(artifact))


def test_normalizer_snapshot_is_private_and_softsign_is_applied_per_step(tmp_path):
    path, _ = _write_release(tmp_path)
    mask = FrozenSourceCreditMask.load(path)
    gate = _load(mask, _v2_artifact(mask))
    snapshot = gate.normalization
    scores = [-.9, .1, .8]
    result = normalize_text_steps_v2(scores, snapshot["text_v2"])
    center, scale = snapshot["text_center"], snapshot["text_scale"]
    manual = [(score - center) / scale for score in scores]
    assert result["bounded_step_scores"] == [z / (1 + abs(z)) for z in manual]
    snapshot["text_v2"]["text_center"] = -.9
    assert gate.normalization["text_v2"]["text_center"] == center
