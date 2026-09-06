#!/usr/bin/env python
"""CPU fail-closed preflight for the formal Proof400 PPO-T/PPO-TK pair.

Missing GPU postflight is an expected, explicitly reported blocking state, not
a CPU-preflight PASS.  Formal training is allowed only when the exact frozen
report/manifest bundle has the exact registered runtime-only success status,
the manifest binds the report hash, and every CPU lock recheck also succeeds.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from kgproweight.config import ProjectConfig, load_config
from kgproweight.training.phase3_ppo import Phase3PPOConfig, _validate_mixed_reward_config
from scripts.prepare.finalize_mixed3_rearag_proof400_ppo_pair import (
    ARMS,
    BOUND_PROTOCOL_PATHS,
    EXPECTED_COUNTS,
    GPU_POSTFLIGHT_MANIFEST_PATH,
    GPU_POSTFLIGHT_PATH,
    GPU_POSTFLIGHT_REQUIRED_STATUS,
    ROOT,
    _assert_config_contract,
    assert_bound_protocol_contracts,
    file_identity,
    flatten,
    inspect_data_contract,
    inspect_gpu_postflight,
    model_fingerprint,
    relative,
    sha256,
    software_environment,
)
from scripts.prepare.resolve_phase3_ppo_runtime_config import (
    resolve_phase3_ppo_runtime_config,
)


TESTS = [
    "tests/test_mixed3_rearag_proof400_pair_wiring.py",
    "tests/test_mixed_ppo_three_dataset_v2_proof400.py",
    "tests/test_mixed_ppo_reward.py",
    "tests/test_phase3_ppo_config_forwarding.py",
    "tests/test_training_question_kg.py",
    "tests/test_proofkg_production_reward.py",
    "tests/test_ppo_rollout_schedule.py",
    "tests/test_ppo_sft_replay.py",
    "tests/test_ppo_explicit_reference.py",
    "tests/test_ppo_diagnostics.py",
]


def verify_file_identity(
    spec: dict[str, Any], failures: list[str], label: str
) -> None:
    path = Path(str(spec.get("path") or ""))
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        failures.append(f"missing {label}: {path}")
        return
    if path.stat().st_size != int(spec.get("size_bytes", -1)):
        failures.append(f"size mismatch {label}: {path}")
    if sha256(path) != spec.get("sha256"):
        failures.append(f"SHA256 mismatch {label}: {path}")


def _verify_materialization_report_outputs(failures: list[str]) -> dict[str, Any]:
    report_path = ROOT / (
        "data/silver_data/"
        "mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42/report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "COMPLETE_DATA_NOT_TRAINED":
        failures.append("materialization report status is not COMPLETE_DATA_NOT_TRAINED")
    gates = report.get("gates", {})
    if not gates or not all(gates.values()):
        failures.append("materialization report has a false/missing gate")
    outputs = report.get("outputs", {})
    expected_names = {
        "silver_train", "question_kg_records", "sampling_weights",
        "prompt_groups", "fixed_rollout_schedule",
    }
    if set(outputs) != expected_names:
        failures.append(
            f"materialization report outputs={sorted(outputs)}, expected={sorted(expected_names)}"
        )
    for label, spec in outputs.items():
        verify_file_identity(spec, failures, f"materialization output {label}")
    return {
        "status": report.get("status"),
        "all_gates_pass": bool(gates) and all(gates.values()),
        "output_names": sorted(outputs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair_manifest", type=Path, required=True)
    parser.add_argument("--report_path", type=Path, required=True)
    parser.add_argument("--run_tests", action="store_true")
    args = parser.parse_args()
    pair_manifest_path = args.pair_manifest.resolve()
    report_path = args.report_path.resolve()
    if report_path.exists():
        raise FileExistsError(f"append-only report already exists: {report_path}")

    failures: list[str] = []
    blockers: list[str] = []
    if not pair_manifest_path.is_file():
        raise FileNotFoundError(pair_manifest_path)
    manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {
        "PREPARED_BLOCKED_GPU_PROBE", "READY_GPU_POSTFLIGHT_BOUND_NOT_STARTED",
    }:
        failures.append(f"unexpected pair manifest status: {manifest.get('status')!r}")
    if manifest.get("execution", {}).get("training_started") is not False:
        failures.append("pair manifest does not say training_started=false")
    expected_postflight_rel = relative(GPU_POSTFLIGHT_PATH)
    if manifest.get("execution", {}).get("required_gpu_postflight_path") != expected_postflight_rel:
        failures.append("pair manifest GPU-postflight path differs from frozen exact path")
    if manifest.get("execution", {}).get(
        "required_gpu_postflight_manifest_path"
    ) != relative(GPU_POSTFLIGHT_MANIFEST_PATH):
        failures.append("pair manifest GPU-postflight manifest path differs from exact path")
    if manifest.get("execution", {}).get(
        "required_gpu_postflight_status"
    ) != GPU_POSTFLIGHT_REQUIRED_STATUS:
        failures.append("pair manifest GPU-postflight required status changed")

    locks: dict[str, dict[str, Any]] = {}
    cfg_docs: dict[str, ProjectConfig] = {}
    runtime_configs: dict[str, dict[str, Any]] = {}
    target_checks: list[dict[str, Any]] = []
    if manifest.get("arm_order") != ["ppo_t", "ppo_tk"]:
        failures.append("arm order is not the frozen PPO-T then PPO-TK order")
    for arm in manifest.get("arm_order", []):
        ref = manifest.get("arm_locks", {}).get(arm)
        if not isinstance(ref, dict):
            failures.append(f"missing arm lock ref: {arm}")
            continue
        verify_file_identity(ref, failures, f"{arm} lock")
        lock_path = ROOT / str(ref.get("path") or "")
        if not lock_path.is_file():
            continue
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        locks[arm] = lock
        if lock.get("status") != manifest.get("status"):
            failures.append(f"{arm}: arm-lock status differs from pair manifest")
        if lock.get("experiment_id") != ARMS[arm]["experiment_id"]:
            failures.append(f"{arm}: Experiment ID differs from source contract")
        if lock.get("proofkg_process_reward") is not ARMS[arm]["process_reward"]:
            failures.append(f"{arm}: process-reward flag differs from source contract")
        verify_file_identity(lock.get("config", {}), failures, f"{arm} config")
        for label, spec in lock.get("code", {}).items():
            verify_file_identity(spec, failures, f"code {label}")
        for label, spec in lock.get("config_dependencies", {}).items():
            verify_file_identity(spec, failures, f"config dependency {label}")
        for label, spec in lock.get("inputs", {}).items():
            verify_file_identity(spec, failures, f"input {label}")
        for label, spec in lock.get("bound_protocols", {}).items():
            verify_file_identity(spec, failures, f"bound protocol {label}")
        try:
            cfg_docs[arm] = _assert_config_contract(arm, ARMS[arm])
            runtime = resolve_phase3_ppo_runtime_config(ARMS[arm]["config"])
            runtime_configs[arm] = runtime
            _validate_mixed_reward_config(Phase3PPOConfig(**runtime))
            if runtime != lock.get("resolved_cli_runtime_config"):
                failures.append(f"{arm}: real CLI runtime differs from frozen lock")
        except Exception as exc:
            failures.append(
                f"{arm}: config/runtime contract failed: {type(exc).__name__}: {exc}"
            )

        for field in ("output_dir", "log_path"):
            target = ROOT / str(lock.get(field) or "")
            exists = target.exists()
            target_checks.append(
                {"arm": arm, "field": field, "path": str(target), "exists": exists}
            )
            if exists:
                failures.append(f"{arm}: {field} already exists: {target}")
        tb_path = Path(str(lock.get("tensorboard_dir") or ""))
        try:
            tb_exists = tb_path.exists()
        except PermissionError as exc:
            target_checks.append(
                {
                    "arm": arm,
                    "field": "tensorboard_dir",
                    "path": str(tb_path),
                    "exists": None,
                    "status": "DEFERRED_TO_REMOTE_FAIL_CLOSED_CHECK",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            target_checks.append(
                {
                    "arm": arm, "field": "tensorboard_dir",
                    "path": str(tb_path), "exists": tb_exists,
                }
            )
            if tb_exists:
                failures.append(f"{arm}: tensorboard_dir already exists: {tb_path}")

    effective_diff: list[str] = []
    if set(cfg_docs) == {"ppo_t", "ppo_tk"}:
        left = flatten(cfg_docs["ppo_t"].model_dump())
        right = flatten(cfg_docs["ppo_tk"].model_dump())
        effective_diff = sorted(
            key for key in set(left) | set(right) if left.get(key) != right.get(key)
        )
        if effective_diff != [
            "training.output_dir", "training.ppo.proofkg_process_reward",
        ]:
            failures.append(f"effective config diff is {effective_diff!r}")
    runtime_diff: list[str] = []
    if set(runtime_configs) == {"ppo_t", "ppo_tk"}:
        left = flatten(runtime_configs["ppo_t"])
        right = flatten(runtime_configs["ppo_tk"])
        runtime_diff = sorted(
            key for key in set(left) | set(right) if left.get(key) != right.get(key)
        )
        if runtime_diff != ["output_dir", "proofkg_process_reward"]:
            failures.append(f"real CLI runtime diff is {runtime_diff!r}")

    try:
        current_data_contract = inspect_data_contract()
        if current_data_contract != manifest.get("shared_data_contract"):
            failures.append("current independently computed data contract differs from pair lock")
    except Exception as exc:
        current_data_contract = {"error": f"{type(exc).__name__}: {exc}"}
        failures.append("independent proof400 data-contract audit failed")
    try:
        current_protocol_contracts = assert_bound_protocol_contracts()
    except Exception as exc:
        current_protocol_contracts = {"error": f"{type(exc).__name__}: {exc}"}
        failures.append("bound protocol semantic audit failed")
    materialization_outputs = _verify_materialization_report_outputs(failures)

    if locks:
        reference = locks.get("ppo_t") or next(iter(locks.values()))
        expected_environment = reference.get("software_environment")
        actual_environment = software_environment()
        if expected_environment is None:
            failures.append("software environment absent from arm lock")
        else:
            for key in ("python", "packages", "torch_cuda_build", "cudnn_version"):
                if actual_environment.get(key) != expected_environment.get(key):
                    failures.append(f"software environment mismatch: {key}")
        for label, expected in reference.get("models", {}).items():
            try:
                actual = model_fingerprint(expected["logical_name"])
            except Exception as exc:
                failures.append(f"model {label} unavailable: {type(exc).__name__}: {exc}")
                continue
            for key in (
                "critical_files", "weight_inventory_sha256",
                "weight_count", "weight_total_bytes",
            ):
                if actual.get(key) != expected.get(key):
                    failures.append(f"model fingerprint mismatch: {label}.{key}")

    gpu_postflight = inspect_gpu_postflight(GPU_POSTFLIGHT_PATH)
    if gpu_postflight["state"] == "MISSING":
        blockers.append(
            "required Proof400 v3 GPU postflight is missing; formal training remains blocked"
        )
    elif not gpu_postflight["gate_pass"]:
        failures.append(
            "required Proof400 v3 GPU postflight report/manifest bundle failed exact status/hash binding"
        )

    test_result = None
    if args.run_tests:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *TESTS],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        test_result = {"returncode": proc.returncode, "output_tail": proc.stdout[-12000:]}
        if proc.returncode:
            failures.append("regression tests failed")

    if failures:
        status = "FAIL_CPU_PREFLIGHT"
    elif blockers:
        status = "PREPARED_BLOCKED_GPU_PROBE"
    else:
        status = "PASS_CPU_PREFLIGHT_GPU_POSTFLIGHT_BOUND"
    report = {
        "schema_version": "mixed3-rearag-proof400-ppo-pair-preflight-v1",
        "experiment_family": manifest.get("experiment_family"),
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_allocated": False,
        "training_started": False,
        "pair_manifest": file_identity(pair_manifest_path),
        "checks": {
            "effective_config_diff": effective_diff,
            "real_cli_runtime_diff": runtime_diff,
            "target_absence": target_checks,
            "data_contract": current_data_contract,
            "expected_counts": EXPECTED_COUNTS,
            "protocol_contracts": current_protocol_contracts,
            "bound_protocol_paths": {
                key: relative(path) for key, path in BOUND_PROTOCOL_PATHS.items()
            },
            "materialization_report_outputs": materialization_outputs,
            "gpu_postflight": gpu_postflight,
            "gpu_gate_required_for_training": True,
        },
        "tests": test_result,
        "blockers": blockers,
        "failures": failures,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
