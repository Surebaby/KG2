#!/usr/bin/env python
"""Strict GPU postflight for corrected mixed3 runtime wiring probe v2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from kgproweight.utils.logging import dump_manifest
from scripts.prepare.materialize_mixed3_rearag_runtime_probe_v2 import ARM_SPECS_V2, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v2_seed42_freeze/protocol.json"
DEFAULT_REPORT_DIR = ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v2_seed42_gpu_postflight"
REQUIRED_FINITE = (
    "mean_reward", "ppo_mean_kl", "loss_total", "loss_policy", "loss_value",
    "advantage_var", "value_mean", "return_mean",
)


def required_finite(row: dict[str, Any], field: str, arm: str) -> float:
    if field not in row or row[field] is None:
        raise ValueError(f"{arm}: required runtime statistic {field} is missing/null")
    try:
        value = float(row[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{arm}: required runtime statistic {field} is not numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{arm}: required runtime statistic {field} is non-finite")
    return value


def verify_arm(arm: str, spec: dict[str, Any]) -> dict[str, Any]:
    out_dir = ROOT / str(spec["output_dir"])
    history_path, manifest_path = out_dir / "history.jsonl", out_dir / "manifest.json"
    final_dir = out_dir / "final"
    for path in (history_path, manifest_path, final_dir / "adapter_config.json", final_dir / "adapter_model.safetensors"):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{arm}: missing completed artifact: {path}")
    rows = [json.loads(line) for line in history_path.open(encoding="utf-8") if line.strip()]
    if len(rows) != 1 or int(rows[0].get("step", -1)) != 4:
        raise ValueError(f"{arm}: expected exactly one completed K=4 update")
    row = rows[0]
    finite = {field: required_finite(row, field, arm) for field in REQUIRED_FINITE}
    if row.get("uses_explicit_sft_reference") is not True:
        raise ValueError(f"{arm}: explicit frozen SFT reference was not used")
    if int(row.get("sft_replay_items", -1)) != 0:
        raise ValueError(f"{arm}: one-batch replay boundary changed")
    qids = row.get("rollout_qids") or []
    if len(qids) != 4 or len(set(qids)) != 1:
        raise ValueError(f"{arm}: runtime fixed K=4 group mismatch")
    expected_eligible = 4 if spec["expected_eligible"] else 0
    if int(row.get("proofkg_eligible_count", -1)) != expected_eligible:
        raise ValueError(f"{arm}: process eligibility telemetry mismatch")
    by_dataset = row.get("mixed_reward_by_dataset") or {}
    if len(by_dataset) != 1:
        raise ValueError(f"{arm}: mixed route telemetry missing")
    diag = next(iter(by_dataset.values()))
    if int(diag.get("count", -1)) != 4 or int(diag.get("proofkg_eligible_count", -1)) != expected_eligible:
        raise ValueError(f"{arm}: dataset route telemetry mismatch")
    if "process_mean" not in diag or diag["process_mean"] is None or not math.isfinite(float(diag["process_mean"])):
        raise ValueError(f"{arm}: process component telemetry missing/non-finite")
    if not spec["expected_eligible"] and not math.isclose(float(diag["process_mean"]), 0.0, abs_tol=1e-12):
        raise ValueError(f"{arm}: PPO-T process component must be exactly zero")
    if spec["expected_eligible"]:
        if row.get("proofkg_process_mean") is None or not math.isfinite(float(row["proofkg_process_mean"])):
            raise ValueError(f"{arm}: PPO-TK ProofKG-v2.1 telemetry missing/non-finite")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = manifest.get("run") or {}
    if run.get("experiment_id") != spec["experiment_id"]:
        raise ValueError(f"{arm}: runtime manifest Experiment ID mismatch")
    cfg = run.get("config") or {}
    required_manifest = {
        "alpha_gate_path": None,
        "alpha_override": None,
        "mixed_outcome_reward": True,
        "mixed_text_reward": True,
        "text_reward_backend": "rearag",
        "proofkg_process_reward": bool(spec["expected_eligible"]),
        "proofkg_process_version": "v2_1",
    }
    for key, expected in required_manifest.items():
        if key not in cfg or cfg[key] != expected:
            raise ValueError(f"{arm}: runtime manifest config {key}={cfg.get(key)!r}, expected={expected!r}")
    rearag = (run.get("input_artifacts") or {}).get("rearag")
    if not isinstance(rearag, dict):
        raise ValueError(f"{arm}: runtime manifest did not bind the ReaRAG artifact")

    log_path = ROOT / str(spec["log_path"])
    if not log_path.is_file() or log_path.stat().st_size <= 0:
        raise FileNotFoundError(f"{arm}: log missing: {log_path}")
    tb_dir = Path(str(spec["tensorboard_dir"]))
    events = sorted(tb_dir.glob("events.out.tfevents.*")) if tb_dir.is_dir() else []
    if not events or not any(path.stat().st_size > 0 for path in events):
        raise FileNotFoundError(f"{arm}: TensorBoard event missing: {tb_dir}")
    return {
        "experiment_id": spec["experiment_id"],
        "status": "PASS_ONE_K4_RUNTIME_UPDATE",
        "rollout_qids": qids,
        "proofkg_eligible_count": expected_eligible,
        "required_finite_statistics": finite,
        "process_component_mean": float(diag["process_mean"]),
        "runtime_config_contract": required_manifest,
        "history_sha256": sha256_file(history_path),
        "manifest_sha256": sha256_file(manifest_path),
        "adapter_sha256": sha256_file(final_dir / "adapter_model.safetensors"),
        "log_sha256": sha256_file(log_path),
        "tensorboard_events": [
            {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in events
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    protocol_path, report_dir = args.protocol.resolve(), args.report_dir.resolve()
    if report_dir.exists():
        raise FileExistsError(f"append-only postflight exists: {report_dir}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_NOT_RUN":
        raise ValueError("wrong v2 protocol status")
    arms = {arm: verify_arm(arm, spec) for arm, spec in ARM_SPECS_V2.items()}
    report = {
        "schema_version": "mixed3-rearag-runtime-wiring-probe-postflight-v2",
        "experiment_id": "MIXED3-REARAG-RUNTIME-WIRING-PROBE-V2-SEED42-POSTFLIGHT",
        "status": "PASS_RUNTIME_WIRING_NOT_EFFECT_EVIDENCE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "arms": arms, "total_trajectories": 8,
        "scientific_boundary": "One-update wiring only; no effect/convergence claim is valid.",
    }
    report_dir.mkdir(parents=True, exist_ok=False)
    path = report_dir / "postflight.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(report_dir, status=report["status"], extra={
        "phase": "mixed3_rearag_runtime_probe_v2_postflight",
        "experiment_id": report["experiment_id"], "total_trajectories": 8,
        "effect_evidence": False, "postflight_sha256": sha256_file(path),
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

