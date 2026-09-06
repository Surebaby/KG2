"""Locked launcher for the optional EM/F1 outcome-only control experiment.

Checks are read-only by default. --execute requires CUDA and launches the exact
production CLI from the same frozen SFT; it never resumes a predecessor. This
control's stage dependencies do not apply to the separate full-method study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

RELEASE = "outputs/audits/ppo_mixed4_emf1_v1_release/manifest.json"
STAGES = {
    "probe": ("configs/training/phase3_ppo_mixed4_emf1_v1_outcome_probe12_seed42.yaml", 12, None),
    "smoke": ("configs/training/phase3_ppo_mixed4_emf1_v1_outcome_smoke600_seed42.yaml", 600, "probe"),
    "full": ("configs/training/phase3_ppo_mixed4_emf1_v1_outcome_seed42.yaml", 12000, "smoke"),
}
OUTPUTS = {
    "probe": "outputs/ppo_mixed4_emf1_v1_outcome_probe12_seed42",
    "smoke": "outputs/ppo_mixed4_emf1_v1_outcome_smoke600_seed42",
    "full": "outputs/ppo_mixed4_emf1_v1_outcome12000_seed42",
}
SFT = "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"
LAUNCHES = "outputs/ppo_mixed4_emf1_v1_launches"


class LaunchError(RuntimeError):
    pass


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _inside(path, root):
    return path.resolve().is_relative_to(root.resolve())


def _relative(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise LaunchError(f"release path must be canonical and project-relative: {value}")
    return path


def verify_release(root):
    manifest_path = root / RELEASE
    manifest = _json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise LaunchError("release manifest has no files mapping")
    for relative, identity in files.items():
        _relative(relative)
        path = root / relative
        # Base model directories may be symlinks to a separately stored model.
        if not _inside(path, root) and not relative.startswith("models/"):
            raise LaunchError(f"bound research/source file escapes project root: {relative}")
        if not path.is_file() or path.stat().st_size != identity.get("size_bytes"):
            raise LaunchError(f"missing file or size mismatch: {relative}")
        if _sha(path) != identity.get("sha256"):
            raise LaunchError(f"SHA256 mismatch: {relative}")
    required = {
        "pyproject.toml", "scripts/train/phase3_ppo.py", "scripts/train/_split_args.py",
        "scripts/train/launch_ppo_emf1_v1.py",
        "scripts/prepare/resolve_phase3_ppo_runtime_config.py",
        *(str(path.relative_to(root)) for path in (root / "kgproweight").rglob("*.py")),
    }
    missing = required - files.keys()
    if missing:
        raise LaunchError(f"production source is not bound by release: {sorted(missing)}")
    return manifest, _sha(manifest_path)


def _verify_environment(root, files, environment):
    model_dirs = {
        (root / relative).parent.resolve() for relative in files
        if relative.startswith("models/") and PurePosixPath(relative).name == "config.json"
    }
    for name, value in environment.items():
        if not name.startswith("KGPW_") or not value:
            continue
        if name == "KGPW_LLAMA3_PATH":
            if Path(value).expanduser().resolve() not in model_dirs:
                raise LaunchError("KGPW_LLAMA3_PATH is not a release-bound base model")
        elif name.endswith(("_ROOT", "_DIR", "_PATH")):
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = root / path
            if not _inside(path, root):
                raise LaunchError(f"{name} redirects outside the locked project root")
            if name == "KGPW_PROJECT_ROOT" and path.resolve() != root:
                raise LaunchError("KGPW_PROJECT_ROOT differs from launcher project root")
    return model_dirs


def _resolve_config(root, relative):
    # Import project code only after the source inventory passed hash checks.
    sys.path.insert(0, str(root))
    from scripts.prepare.resolve_phase3_ppo_runtime_config import resolve_phase3_ppo_runtime_config
    return resolve_phase3_ppo_runtime_config(root / relative)


def _config_sources(root, relative, files, seen=None):
    import yaml

    seen = set() if seen is None else seen
    path = (root / relative).resolve()
    if not _inside(path, root):
        raise LaunchError("config include escapes project root")
    key = path.relative_to(root).as_posix()
    if key not in files:
        raise LaunchError(f"config/include is not release-bound: {key}")
    if key in seen:
        return
    seen.add(key)
    for include in (yaml.safe_load(path.read_text()) or {}).get("includes", []):
        child = (path.parent / include).resolve()
        if not _inside(child, root):
            raise LaunchError("config include escapes project root")
        _config_sources(root, child.relative_to(root).as_posix(), files, seen)


def _stage_config(root, stage, files, model_dirs):
    relative, trajectories, _ = STAGES[stage]
    _config_sources(root, relative, files)
    config = _resolve_config(root, relative)
    expected = {
        "runtime_contract_version": "v2", "total_steps": trajectories,
        "output_dir": OUTPUTS[stage], "sft_checkpoint": SFT, "seed": 42,
        "batch_size": 4, "rollouts_per_prompt": 4,
        "mixed_outcome_reward": True, "mixed_text_reward": False,
        "proofkg_process_reward": False, "alpha_gate_path": None, "alpha_override": None,
        "sft_replay_ratio": .1, "sft_anchor_weight": .1,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise LaunchError(f"{stage} config violates the outcome-control contract: {key}")
    for key in ["silver_path", "sft_replay_silver_path", "question_kg_records_path", "rollout_sampling_weights_path", "fixed_rollout_schedule_path"]:
        if config.get(key) not in files:
            raise LaunchError(f"{stage} input is not release-bound: {key}")
    for name in ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        if f"{SFT}/{name}" not in files:
            raise LaunchError(f"SFT adapter/tokenizer file is not release-bound: {name}")
    base = _json(root / SFT / "adapter_config.json").get("base_model_name_or_path")
    if not base:
        raise LaunchError("SFT adapter has no base model identity")
    actual_base = Path(base).expanduser()
    if not actual_base.is_absolute():
        actual_base = root / actual_base
    if actual_base.resolve() not in model_dirs:
        raise LaunchError("PEFT adapter's actual base path is not the release-bound model")
    return config


def _health_failure(rows, config):
    from kgproweight.training.phase3_ppo import _smoke_health_guard_reason
    return _smoke_health_guard_reason(rows, SimpleNamespace(**config))


def _completed_stage(root, stage, config, release_sha):
    output = root / OUTPUTS[stage]
    manifest_path, history_path = output / "manifest.json", output / "history.jsonl"
    manifest = _json(manifest_path)
    run = manifest.get("run", {})
    if manifest.get("status") != "COMPLETE" or run.get("config") != config:
        raise LaunchError(f"{stage} predecessor is not COMPLETE under the locked resolved config")
    if run.get("reference_mode") != "explicit_frozen_sft_snapshot":
        raise LaunchError(f"{stage} lacks an explicit SFT reference")
    launch = _json(root / LAUNCHES / stage / "manifest.json")
    if launch.get("release_sha256") != release_sha:
        raise LaunchError(f"{stage} was launched under a different release")
    rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    expected_steps = list(range(4, STAGES[stage][1] + 1, 4))
    if [row.get("step") for row in rows] != expected_steps:
        raise LaunchError(f"{stage} history does not cover its exact trajectory/update budget")
    undefined_ev = 0
    for index, row in enumerate(rows):
        for key in ["mean_reward", "ppo_mean_kl", "loss_total", "loss_policy", "loss_value", "return_std", "valid_rate", "length_capped_frac"]:
            if row.get(key) is None or not math.isfinite(float(row[key])):
                raise LaunchError(f"{stage} has missing/non-finite {key} at update {index + 1}")
        for key, value in row.items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                if key == "explained_variance" and row["return_std"] == 0 and math.isnan(value):
                    undefined_ev += 1
                else:
                    raise LaunchError(f"{stage} has non-finite {key} at update {index + 1}")
        expected_replay = int(row["step"] * .1 + 1e-9)
        previous = int((row["step"] - 4) * .1 + 1e-9)
        if row.get("sft_replay_items_seen") != expected_replay or row.get("sft_replay_items") != expected_replay - previous:
            raise LaunchError(f"{stage} replay delivery differs from its frozen ratio")
        if not math.isclose(float(row.get("sft_replay_actual_ratio", -1)), expected_replay / row["step"], abs_tol=1e-9):
            raise LaunchError(f"{stage} replay ratio telemetry is inconsistent")
        if row.get("uses_explicit_sft_reference") is not True:
            raise LaunchError(f"{stage} history lost the explicit SFT reference")
        if stage != "probe":
            failure = _health_failure(rows[: index + 1], config)
            if failure:
                raise LaunchError(f"{stage} failed its frozen health guard: {failure}")
    if abs(rows[0]["ppo_mean_kl"]) > 1.:
        raise LaunchError(f"{stage} initial reference KL invariant failed")
    return {
        "stage": stage, "manifest_sha256": _sha(manifest_path), "history_sha256": _sha(history_path),
        "updates": len(rows), "trajectories": rows[-1]["step"],
        "replay_items": rows[-1]["sft_replay_items_seen"], "undefined_ev_zero_variance_updates": undefined_ev,
    }


def check_stage(stage, *, root=None, environment=None):
    if stage not in STAGES:
        raise LaunchError(f"unknown outcome-control stage: {stage}")
    root = (Path(root) if root is not None else Path(__file__).resolve().parents[2]).resolve()
    environment = dict(os.environ if environment is None else environment)
    release, release_sha = verify_release(root)
    model_dirs = _verify_environment(root, release["files"], environment)
    config = _stage_config(root, stage, release["files"], model_dirs)
    output, launch_dir = root / OUTPUTS[stage], root / LAUNCHES / stage
    if not _inside(output, root) or not _inside(launch_dir, root):
        raise LaunchError("output/launch directory escapes the locked project")
    if output.exists() or output.is_symlink() or launch_dir.exists() or launch_dir.is_symlink():
        raise LaunchError("output or stage launch directory already exists; refusing overwrite/resume")
    predecessor = STAGES[stage][2]
    evidence = None
    if predecessor:
        prior = _stage_config(root, predecessor, release["files"], model_dirs)
        evidence = _completed_stage(root, predecessor, prior, release_sha)
    return {
        "status": "CPU_LOCK_CHECK_PASS_NOT_GPU_CLEARANCE", "stage": stage,
        "project_root": str(root), "release_sha256": release_sha,
        "release_manifest": RELEASE, "verified_files": len(release["files"]),
        "config_path": STAGES[stage][0], "resolved_config": config,
        "predecessor": evidence, "launch_dir": str(launch_dir),
        "command": [sys.executable, "-m", "scripts.train.phase3_ppo", "--config", STAGES[stage][0]],
    }


def _cuda_status():
    import torch
    if not torch.cuda.is_available():
        raise LaunchError("--execute requires CUDA; CPU lock checks are not GPU clearance")
    return {"name": torch.cuda.get_device_name(0), "torch_cuda": torch.version.cuda}


def launch(stage, *, execute=False, root=None, environment=None):
    report = check_stage(stage, root=root, environment=environment)
    if not execute:
        return report
    report["gpu"] = _cuda_status()
    root = Path(report["project_root"])
    launch_dir = Path(report["launch_dir"])
    launch_dir.mkdir(parents=True, exist_ok=False)
    report.update(status="STARTED", started_at_utc=datetime.now(timezone.utc).isoformat())
    with (launch_dir / "manifest.json").open("x") as handle:
        json.dump(report, handle, indent=2)
    child_environment = dict(os.environ if environment is None else environment)
    child_environment.update(PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1", KGPW_PROJECT_ROOT=str(root))
    try:
        with (launch_dir / "training.log").open("xb") as log:
            result = subprocess.run(report["command"], cwd=root, env=child_environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise LaunchError(f"production PPO CLI exited {result.returncode}; see {launch_dir / 'training.log'}")
        completed = _completed_stage(root, stage, report["resolved_config"], report["release_sha256"])
        completion = {"status": "COMPLETE", "training": completed}
    except BaseException as exc:
        completion = {"status": "FAILED", "error": str(exc)}
        raise
    finally:
        completion["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        with (launch_dir / "completion.json").open("x") as handle:
            json.dump(completion, handle, indent=2)
    return completion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Default: verify locks/dependencies without writing outputs.")
    mode.add_argument("--execute", action="store_true", help="Run the locked outcome-only control stage on CUDA.")
    args = parser.parse_args(argv)
    try:
        report = launch(args.stage, execute=args.execute)
    except (LaunchError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"PPO outcome-control launch blocked: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
