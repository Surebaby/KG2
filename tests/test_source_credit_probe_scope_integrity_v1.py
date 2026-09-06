"""Independent bounded-probe attack tests; synthetic CPU fixtures, no training."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kgproweight.reward import source_gate_probe_scope_v1 as scope
from kgproweight.reward.source_credit_gate_v1 import FrozenSourceCreditMask
from kgproweight.reward.source_credit_gate_v2 import SourceCreditGateV2
from kgproweight.training import phase3_ppo as training
from scripts.prepare.resolve_phase3_ppo_runtime_config import resolve_phase3_ppo_runtime_config
from tests.test_source_credit_gate_v1 import _signed, _write_release
from tests.test_source_credit_gate_v2 import _v2_artifact
from tests.test_source_gate_probe_scope_v1 import release


@pytest.mark.parametrize("diagnostic", [False, True])
@pytest.mark.parametrize("retained_scope_marker", [False, True])
def test_recomputed_payload_cannot_promote_parent_by_stripping_scope(tmp_path, diagnostic, retained_scope_marker):
    manifest, _ = _write_release(tmp_path / "synthetic-mask")
    mask = FrozenSourceCreditMask.load(manifest)
    artifact = _v2_artifact(mask)
    artifact.update(bank_source="real_frozen_policy_rollouts", training_clearance=True,
                    independent_confirmation_clearance=True)
    if retained_scope_marker:
        artifact["training_clearance_scope"] = "complete_A_probe12_only"
    with pytest.raises(ValueError, match="registered bounded execution scope"):
        SourceCreditGateV2(_signed(artifact), mask=mask, allow_unvalidated=diagnostic)


@pytest.mark.parametrize("diagnostic", [False, True])
def test_diagnostic_switch_cannot_skip_scope_without_runtime_config(tmp_path, diagnostic):
    manifest, _ = _write_release(tmp_path / "synthetic-mask")
    mask = FrozenSourceCreditMask.load(manifest)
    artifact = _v2_artifact(mask)
    artifact["execution_scope"] = {"path": "missing-scope.json", "sha256": "0" * 64}
    with pytest.raises(ValueError, match="exact runtime configuration"):
        SourceCreditGateV2(_signed(artifact), mask=mask, allow_synthetic=True,
                          allow_unvalidated=diagnostic, artifact_path=tmp_path / "gate.json")
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(_signed(artifact)))
    with pytest.raises(ValueError, match="exact runtime configuration"):
        SourceCreditGateV2.load(gate_path, allow_synthetic=True, allow_unvalidated=diagnostic)


class ScopeObservedBeforeCUDA(RuntimeError):
    pass


@pytest.mark.parametrize("changes", [
    {"total_steps": 600},
    {"source_gate_mode": "fixed"},
    {"source_gate_mode": "text"},
    {"source_gated_reward_version": "disabled", "source_gate_credit_version": "disabled",
     "answer_format_reward_version": "legacy", "center_text_reward": True},
])
def test_real_training_dispatch_checks_referenced_scope_before_models(tmp_path, monkeypatch, changes):
    """Changing CLI versions cannot route around a referenced limited child."""
    cfg = training.Phase3PPOConfig(**resolve_phase3_ppo_runtime_config(
        Path(__file__).resolve().parents[1] /
        "configs/training/phase3_ppo_mixed4_answer_format_v2_a_probe_seed42.yaml"))
    gate_path = tmp_path / "scope-marker-only.json"
    gate_path.write_text(json.dumps({"execution_scope": {"path": "synthetic-only"}}))
    cfg = replace(cfg, source_gate_calibration_path=str(gate_path), **changes)
    training._validate_mixed_reward_config(cfg)
    observed = []

    def inspect(path, *, runtime_config, **kwargs):
        observed.append((Path(path), runtime_config))
        raise ScopeObservedBeforeCUDA("reached scope validation before any model allocation")

    def forbidden(*args, **kwargs):
        pytest.fail("model/CUDA work must not precede scope validation")

    monkeypatch.setattr(SourceCreditGateV2, "load", inspect)
    monkeypatch.setattr(training, "set_seed", lambda seed: None)
    monkeypatch.setattr(training, "_build_models", forbidden)
    monkeypatch.setattr(training.torch.cuda, "is_available", forbidden)
    with pytest.raises(ScopeObservedBeforeCUDA):
        training.run_phase3_ppo(cfg)
    assert observed == [(gate_path, cfg)]


@pytest.fixture
def execution_paths(tmp_path, monkeypatch):
    for directory in ("checkpoints/policy", "models/base", "models/rearag", "models/other"):
        (tmp_path / directory).mkdir(parents=True)
    adapter_path = tmp_path / "checkpoints/policy/adapter_config.json"
    adapter_path.write_text(json.dumps({"base_model_name_or_path": "models/base"}))
    frozen = {"models": {"base_model": {"path": "models/base"},
                          "rearag_model": {"path": "models/rearag"}}}
    monkeypatch.setattr(scope, "project_root", lambda: tmp_path)
    monkeypatch.setattr(scope, "read_scope", lambda *args, **kwargs: frozen)
    monkeypatch.setattr(scope, "model_path", lambda role: str(tmp_path / "models/rearag"))
    cfg = SimpleNamespace(sft_checkpoint="checkpoints/policy", source_gate_calibration_path="gate.json")
    return SimpleNamespace(artifact={"execution_scope": {}}), cfg, adapter_path, tmp_path


@pytest.mark.parametrize("absolute", [False, True])
def test_execution_path_check_accepts_exact_relocated_models(execution_paths, absolute):
    gate, cfg, adapter, root = execution_paths
    adapter.write_text(json.dumps({"base_model_name_or_path": str(root / "models/base") if absolute else "models/base"}))
    scope.validate_probe_execution_paths(gate, cfg)


@pytest.mark.parametrize("base", ["models/other", "models/missing"])
def test_adapter_cannot_redirect_actual_reference_and_policy_base(execution_paths, base):
    gate, cfg, adapter, _ = execution_paths
    adapter.write_text(json.dumps({"base_model_name_or_path": base}))
    with pytest.raises(ValueError, match="base model path"):
        scope.validate_probe_execution_paths(gate, cfg)


def test_rearag_environment_cannot_redirect_bound_model(execution_paths, monkeypatch):
    gate, cfg, _, root = execution_paths
    monkeypatch.setattr(scope, "model_path", lambda role: str(root / "models/other"))
    with pytest.raises(ValueError, match="ReaRAG environment"):
        scope.validate_probe_execution_paths(gate, cfg)


@pytest.mark.parametrize("value", ["../other/gate.json", "/tmp/outside-project-gate.json", "."])
def test_scope_portability_rejects_paths_outside_bound_checkout(value):
    with pytest.raises(ValueError):
        scope.portable_path(value)


@pytest.mark.parametrize("name", [
    "kgproweight/training/unbound_loader.py",
    "kgproweight/reward/unbound_transform.py",
    "kgproweight/utils/unbound_path_resolver.py",
])
def test_new_package_code_requires_binding_even_outside_original_core_four(release, name):
    release.put(name, {"synthetic": "new executable file"})
    with pytest.raises(ValueError, match="code closure"):
        release.validate()


@pytest.mark.parametrize("name", [
    "scripts/train/phase3_ppo.py",
    "scripts/train/_split_args.py",
    "scripts/prepare/resolve_phase3_ppo_runtime_config.py",
    "scripts/prepare/freeze_source_credit_v2_probe12_scope_v1.py",
    "scripts/sourcegate_python.sh",
])
def test_removing_execution_entry_binding_cannot_be_resigned(release, name):
    release.scope["code_bindings"].pop(name)
    release.rescope()
    with pytest.raises(ValueError, match="code closure"):
        release.validate()


@pytest.mark.parametrize("name", [
    "models/base/model.safetensors",
    "models/base/model.safetensors.index.json",
    "models/rearag/model.safetensors",
    "models/rearag/model.safetensors.index.json",
    "models/rearag/tokenization_changed.py",
    "models/policy/added_tokens.json",
])
def test_new_loader_visible_file_cannot_override_frozen_model_identity(release, name):
    """An added unbound single-file weight can override intact frozen shards."""
    release.put(name, {"synthetic": "unbound loader input"})
    with pytest.raises(ValueError):
        release.validate()


def test_nonloading_readme_does_not_change_model_identity(release):
    release.put("models/base/README.md", {"synthetic": "documentation only"})
    assert release.validate()["trajectory_limit"] == 12
