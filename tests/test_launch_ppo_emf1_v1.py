"""Strict outcome-control launcher tests using only small temporary files."""
import json
from pathlib import Path

import pytest

import scripts.train.launch_ppo_emf1_v1 as launcher


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def release(tmp_path, monkeypatch):
    root = tmp_path / "project"
    paths = [
        "pyproject.toml", "scripts/train/phase3_ppo.py", "scripts/train/_split_args.py",
        "scripts/train/launch_ppo_emf1_v1.py", "scripts/prepare/resolve_phase3_ppo_runtime_config.py",
        "kgproweight/training/source.py", "models/base/config.json", "models/base/weights.bin",
        "data/silver.jsonl", "data/replay.jsonl", "data/kg.jsonl", "data/weights.jsonl",
    ]
    paths += [row[0] for row in launcher.STAGES.values()]
    paths += [f"data/{stage}_schedule.jsonl" for stage in launcher.STAGES]
    paths += [f"{launcher.SFT}/{name}" for name in ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]]
    for relative in paths:
        _write(root / relative, {})
    _write(root / launcher.SFT / "adapter_config.json", {"base_model_name_or_path": str(root / "models/base")})
    files = {relative: {"sha256": launcher._sha(root / relative), "size_bytes": (root / relative).stat().st_size} for relative in paths}
    _write(root / launcher.RELEASE, {"files": files})
    configs = {}
    for stage, (_, total, _) in launcher.STAGES.items():
        configs[stage] = {
            "runtime_contract_version": "v2", "total_steps": total, "output_dir": launcher.OUTPUTS[stage],
            "sft_checkpoint": launcher.SFT, "seed": 42, "batch_size": 4, "rollouts_per_prompt": 4,
            "mixed_outcome_reward": True, "mixed_text_reward": False, "proofkg_process_reward": False,
            "alpha_gate_path": None, "alpha_override": None, "sft_replay_ratio": .1, "sft_anchor_weight": .1,
            "silver_path": "data/silver.jsonl", "sft_replay_silver_path": "data/replay.jsonl",
            "question_kg_records_path": "data/kg.jsonl", "rollout_sampling_weights_path": "data/weights.jsonl",
            "fixed_rollout_schedule_path": f"data/{stage}_schedule.jsonl",
            "health_guard_after_steps": 200, "health_guard_window": 15,
            "health_guard_min_valid_rate": .7, "health_guard_max_length_capped_frac": .2,
            "health_guard_max_mean_kl": 10.,
        }
    monkeypatch.setattr(launcher, "_resolve_config", lambda _root, relative: next(dict(configs[stage]) for stage, (path, _, _) in launcher.STAGES.items() if path == relative))
    return root, configs


def _completed(root, configs, stage, *, changes=None):
    output = root / launcher.OUTPUTS[stage]
    _write(output / "manifest.json", {"status": "COMPLETE", "run": {"config": configs[stage], "reference_mode": "explicit_frozen_sft_snapshot"}})
    _write(root / launcher.LAUNCHES / stage / "manifest.json", {"release_sha256": launcher._sha(root / launcher.RELEASE)})
    rows = []
    for step in range(4, launcher.STAGES[stage][1] + 1, 4):
        replay = int(step * .1 + 1e-9)
        rows.append({
            "step": step, "mean_reward": 1., "ppo_mean_kl": 0., "loss_total": .1,
            "loss_policy": .05, "loss_value": .1, "return_std": .5, "explained_variance": 0.,
            "valid_rate": 1., "length_capped_frac": 0., "uses_explicit_sft_reference": True,
            "sft_replay_items_seen": replay, "sft_replay_items": replay - int((step - 4) * .1 + 1e-9),
            "sft_replay_actual_ratio": replay / step,
        })
    if changes:
        for row in rows:
            row.update(changes)
    (output / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def test_check_only_is_read_only_and_not_gpu_clearance(release):
    root, _ = release
    before = {str(path.relative_to(root)) for path in root.rglob("*")}
    result = launcher.launch("probe", root=root, environment={})
    assert result["status"] == "CPU_LOCK_CHECK_PASS_NOT_GPU_CLEARANCE"
    assert result["predecessor"] is None
    assert result["command"][1:] == ["-m", "scripts.train.phase3_ppo", "--config", launcher.STAGES["probe"][0]]
    assert before == {str(path.relative_to(root)) for path in root.rglob("*")}


@pytest.mark.parametrize("relative", ["kgproweight/training/source.py", "data/silver.jsonl", "models/base/weights.bin", f"{launcher.SFT}/adapter_model.safetensors"])
def test_changed_source_data_or_weights_are_rejected(release, relative):
    root, _ = release
    (root / relative).write_text("tampered")
    with pytest.raises(launcher.LaunchError, match="mismatch"):
        launcher.check_stage("probe", root=root, environment={})


def test_unlocked_new_python_source_is_rejected(release):
    root, _ = release
    (root / "kgproweight/new.py").write_text("# unbound")
    with pytest.raises(launcher.LaunchError, match="not bound"):
        launcher.check_stage("probe", root=root, environment={})


@pytest.mark.parametrize("name", ["KGPW_PROJECT_ROOT", "KGPW_DATA_DIR", "KGPW_INDEX_DIR", "KGPW_CHECKPOINT_DIR", "KGPW_TB_DIR", "KGPW_LLAMA3_PATH"])
def test_environment_cannot_redirect_outside_bound_assets(release, name):
    root, _ = release
    with pytest.raises(launcher.LaunchError):
        launcher.check_stage("probe", root=root, environment={name: "/unbound/location"})


def test_bound_model_environment_is_allowed(release):
    root, _ = release
    launcher.check_stage("probe", root=root, environment={"KGPW_LLAMA3_PATH": str(root / "models/base")})


def test_existing_training_output_refuses_resume_or_overwrite(release):
    root, _ = release
    (root / launcher.OUTPUTS["probe"]).mkdir(parents=True)
    with pytest.raises(launcher.LaunchError, match="refusing overwrite/resume"):
        launcher.check_stage("probe", root=root, environment={})


def test_smoke_requires_complete_probe_and_exact_replay_delivery(release):
    root, configs = release
    with pytest.raises(FileNotFoundError):
        launcher.check_stage("smoke", root=root, environment={})
    rows = _completed(root, configs, "probe")
    result = launcher.check_stage("smoke", root=root, environment={})
    assert result["predecessor"]["updates"] == 3
    assert result["predecessor"]["trajectories"] == 12
    assert result["predecessor"]["replay_items"] == 1
    rows[-1]["sft_replay_items_seen"] = 0
    (root / launcher.OUTPUTS["probe"] / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(launcher.LaunchError, match="replay delivery"):
        launcher.check_stage("smoke", root=root, environment={})


def test_full_requires_complete_healthy_smoke(release):
    root, configs = release
    _completed(root, configs, "smoke")
    assert launcher.check_stage("full", root=root, environment={})["predecessor"]["trajectories"] == 600
    _completed(root, configs, "smoke", changes={"valid_rate": .1})
    with pytest.raises(launcher.LaunchError, match="health guard"):
        launcher.check_stage("full", root=root, environment={})


def test_nonfinite_state_rejected_but_zero_variance_ev_reported(release):
    root, configs = release
    _completed(root, configs, "probe", changes={"loss_total": float("nan")})
    with pytest.raises(launcher.LaunchError, match="non-finite"):
        launcher.check_stage("smoke", root=root, environment={})
    _completed(root, configs, "probe", changes={"explained_variance": float("nan"), "return_std": 0.})
    result = launcher.check_stage("smoke", root=root, environment={})
    assert result["predecessor"]["undefined_ev_zero_variance_updates"] == 3


def test_execute_without_cuda_creates_no_launch_or_training_output(release, monkeypatch):
    root, _ = release
    def no_cuda():
        raise launcher.LaunchError("CUDA unavailable")
    monkeypatch.setattr(launcher, "_cuda_status", no_cuda)
    with pytest.raises(launcher.LaunchError, match="CUDA"):
        launcher.launch("probe", execute=True, root=root, environment={})
    assert not (root / launcher.LAUNCHES).exists()
    assert not (root / launcher.OUTPUTS["probe"]).exists()


def test_cli_defaults_to_check_only(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(launcher, "launch", lambda stage, execute: calls.append((stage, execute)) or {"status": "checked"})
    assert launcher.main(["--stage", "probe"]) == 0
    assert calls == [("probe", False)]
    assert "checked" in capsys.readouterr().out
