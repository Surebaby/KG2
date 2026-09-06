#!/usr/bin/env python
"""CPU-only fail-closed preflight for corrected runtime wiring probe v2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

from kgproweight.config import ProjectConfig, load_config
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.kg.question_kg import question_key
from kgproweight.kg.training_question_kg import apply_training_question_kg, read_question_kg_records
from kgproweight.reward.proofkg_process import is_identity_safe_automatic_proofkg
from kgproweight.training.phase3_ppo import (
    Phase3PPOConfig,
    _load_fixed_rollout_schedule,
    _load_rollout_sampling_weights,
    _prepare_prompts,
    _validate_mixed_reward_config,
    _validate_v21_execution_preflight,
)
from kgproweight.utils.logging import dump_manifest
from kgproweight.utils.paths import model_path
from scripts.prepare.materialize_mixed3_rearag_runtime_probe_v2 import (
    ARM_SPECS_V2,
    sha256_file,
)
from scripts.prepare.resolve_phase3_ppo_runtime_config import (
    resolve_phase3_ppo_runtime_config,
)
from scripts.prepare.preflight_mixed3_rearag_runtime_probe_v1 import (
    EXPECTED_CONFIG_DIFF,
    _MemoryReader,
    _PromptTokenizer,
    config_diff,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v2_seed42_freeze/protocol.json"
DEFAULT_REPORT_DIR = ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v2_seed42_local_preflight"
TESTS = [
    "tests/test_mixed3_rearag_runtime_probe_v2.py",
    "tests/test_ppo_rollout_schedule.py",
    "tests/test_mixed_ppo_reward.py",
    "tests/test_phase3_ppo_config_forwarding.py",
    "tests/test_ppo_sft_replay.py",
    "tests/test_ppo_explicit_reference.py",
    "tests/test_ppo_diagnostics.py",
]
EXPECTED_RUNTIME_DIFF = [
    "fixed_rollout_schedule_path",
    "output_dir",
    "question_kg_records_path",
    "rollout_sampling_weights_path",
    "save_every_steps",
    "silver_path",
    "total_steps",
]


def verify_ref(spec: dict[str, Any], label: str) -> None:
    path = Path(str(spec["path"]))
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    if path.stat().st_size != int(spec["size_bytes"]):
        raise ValueError(f"{label}: size mismatch")
    if sha256_file(path) != str(spec["sha256"]):
        raise ValueError(f"{label}: SHA256 mismatch")


def target_absence(spec: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for label in ("output_dir", "log_path"):
        path = Path(str(spec[label]))
        path = path if path.is_absolute() else ROOT / path
        if path.exists():
            raise FileExistsError(f"probe target already exists: {path}")
        checks[label] = {"path": str(path), "exists": False}
    checks["tensorboard_dir"] = {
        "path": spec["tensorboard_dir"],
        "status": "REMOTE_LAUNCHER_RECHECK_REQUIRED",
    }
    return checks


def validate_arm(arm: str, spec: dict[str, Any]) -> dict[str, Any]:
    probe = load_config(ROOT / spec["config"], validate=ProjectConfig)
    formal = load_config(ROOT / spec["formal_config"], validate=ProjectConfig)
    diff = config_diff(formal, probe)
    if diff != EXPECTED_CONFIG_DIFF:
        raise ValueError(f"{arm}: effective config drift={diff}")
    # Resolve through the real CLI instead of trusting the YAML/schema or a
    # hand-copied forwarding subset. This is what caught the alpha-null bug.
    probe_runtime = resolve_phase3_ppo_runtime_config(ROOT / spec["config"])
    formal_runtime = resolve_phase3_ppo_runtime_config(ROOT / spec["formal_config"])
    runtime_diff = sorted(
        key for key in set(probe_runtime) | set(formal_runtime)
        if probe_runtime.get(key) != formal_runtime.get(key)
    )
    if runtime_diff != EXPECTED_RUNTIME_DIFF:
        raise ValueError(f"{arm}: real CLI runtime drift={runtime_diff}")
    cfg = Phase3PPOConfig(**probe_runtime)
    _validate_mixed_reward_config(cfg)
    expected = bool(spec["expected_eligible"])
    if cfg.alpha_gate_path is not None or cfg.alpha_override is not None:
        raise ValueError(f"{arm}: mixed route must retain alpha=null")
    if cfg.proofkg_process_reward is not expected:
        raise ValueError(f"{arm}: process flag does not match route")
    if (cfg.total_steps, cfg.batch_size, cfg.rollouts_per_prompt, cfg.save_every_steps) != (4, 4, 4, 4):
        raise ValueError(f"{arm}: not exactly one saved K=4 update")
    if Path(cfg.output_dir).name != spec["experiment_id"]:
        raise ValueError(f"{arm}: Experiment ID/output mismatch")
    trajectories = list(SilverDatasetReader(cfg.silver_path, split=None).accepted())
    if len(trajectories) != 1:
        raise ValueError(f"{arm}: population is not one row")
    join = apply_training_question_kg(
        trajectories, read_question_kg_records(cfg.question_kg_records_path),
        min_coverage=1.0, require_nonempty=False,
    ).to_dict()
    trajectory = trajectories[0]
    eligible = is_identity_safe_automatic_proofkg(
        trajectory.metadata.get("question_kg_runtime") or {}, trajectory.kg_subgraph,
        dataset=trajectory.dataset, qid=trajectory.qid,
    )
    if eligible is not expected:
        raise ValueError(f"{arm}: production eligibility mismatch")
    execution = _validate_v21_execution_preflight(trajectories) if expected else None
    weights, sampling = _load_rollout_sampling_weights(cfg.rollout_sampling_weights_path, trajectories)
    indices, schedule = _load_fixed_rollout_schedule(
        cfg.fixed_rollout_schedule_path, trajectories, total_steps=4, rollouts_per_prompt=4,
        sampling_records=sampling,
    )
    if weights != [1.0] or indices != [0, 0, 0, 0]:
        raise ValueError(f"{arm}: production fixed schedule resolution failed")
    if any(bool(row.get("process_reward_eligible")) is not expected for row in schedule):
        raise ValueError(f"{arm}: schedule route annotation mismatch")
    prepared = _prepare_prompts(_MemoryReader(trajectories), _PromptTokenizer(), cfg)
    if len(prepared) != 1:
        raise ValueError(f"{arm}: production prompt builder row count mismatch")
    runtime = prepared[0]["spec"].metadata.get("question_kg_runtime") or {}
    identity = question_key(str(trajectory.dataset), str(trajectory.qid))
    if runtime.get("question_key") != identity:
        raise ValueError(f"{arm}: RewardSpec lost identity-safe runtime")
    return {
        "experiment_id": spec["experiment_id"],
        "identity": identity,
        "effective_config_diff": diff,
        "real_cli_runtime_diff": runtime_diff,
        "alpha_gate_path": cfg.alpha_gate_path,
        "alpha_override": cfg.alpha_override,
        "mixed_outcome_reward": cfg.mixed_outcome_reward,
        "mixed_text_reward": cfg.mixed_text_reward,
        "text_reward_backend": cfg.text_reward_backend,
        "proofkg_process_reward": cfg.proofkg_process_reward,
        "process_reward_eligible": eligible,
        "question_kg_join": join,
        "v2_1_execution": execution,
        "kg_triples": len(trajectory.kg_subgraph),
        "fixed_indices": indices,
        "sampling_mass": sum(weights),
        "prompt_tokens_whitespace_proxy": prepared[0]["prompt_tokens"],
        "targets": target_absence(spec),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--run_tests", action="store_true")
    args = parser.parse_args()
    protocol_path, report_dir = args.protocol.resolve(), args.report_dir.resolve()
    if report_dir.exists():
        raise FileExistsError(f"append-only preflight exists: {report_dir}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_NOT_RUN" or protocol.get("counts", {}).get("scheduled_trajectories_total") != 8:
        raise ValueError("wrong v2 frozen protocol/budget")
    if protocol.get("scientific_boundary", {}).get("training_started") is not False:
        raise ValueError("protocol does not assert training_started=false")
    for section in ("inputs", "runtime_code_closure", "config_dependency_closure"):
        for label, identity in protocol.get(section, {}).items():
            verify_ref(identity, f"{section}:{label}")
    for arm, refs in protocol.get("outputs", {}).items():
        for label, identity in refs.items():
            verify_ref(identity, f"output:{arm}:{label}")
    arms = {arm: validate_arm(arm, spec) for arm, spec in ARM_SPECS_V2.items()}

    models = {}
    for logical in ("llama3-8B-instruct", "rearag"):
        path = Path(model_path(logical)).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"local {logical} missing: {path}")
        models[logical] = str(path.resolve())
    adapter = ROOT / "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / filename).is_file():
            raise FileNotFoundError(adapter / filename)

    tests = None
    if args.run_tests:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *TESTS], cwd=ROOT,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        tests = {"returncode": proc.returncode, "output_tail": proc.stdout[-8000:]}
        if proc.returncode:
            raise RuntimeError("v2 probe regression tests failed")
    report = {
        "schema_version": "mixed3-rearag-runtime-wiring-probe-preflight-v2",
        "experiment_id": "MIXED3-REARAG-RUNTIME-WIRING-PROBE-V2-SEED42-PREFLIGHT",
        "status": "PASS_CPU_PREFLIGHT_NOT_RUN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": str(protocol_path.relative_to(ROOT)), "sha256": sha256_file(protocol_path)},
        "cuda_allocated": False,
        "training_started": False,
        "checks": {
            "arms": arms, "total_trajectories": 8, "models": models,
            "sft_adapter": str(adapter.relative_to(ROOT)),
            "runtime_code_files_hashed": len(protocol["runtime_code_closure"]),
            "config_dependency_files_hashed": len(protocol["config_dependency_closure"]),
            "formal_pair_unchanged": True, "formal_data_unchanged": True,
            "tensorboard": "Remote launcher repeats per-arm absence checks before CUDA.",
            "replay": "One batch accrues 0.4 replay credit and therefore tests zero replay updates.",
        },
        "tests": tests,
        "scientific_boundary": "Runtime wiring only; no effect or convergence evidence.",
    }
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = report_dir / "preflight.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(report_dir, status=report["status"], extra={
        "phase": "mixed3_rearag_runtime_probe_v2_preflight",
        "experiment_id": report["experiment_id"],
        "cuda_allocated": False, "training_started": False,
        "preflight_sha256": sha256_file(report_path),
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
