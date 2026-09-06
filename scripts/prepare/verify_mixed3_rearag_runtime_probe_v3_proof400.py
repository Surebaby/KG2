#!/usr/bin/env python
"""Strict GPU postflight for the Proof400 runtime probe v3 (not run locally)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from kgproweight.utils.logging import dump_manifest
from scripts.prepare.materialize_mixed3_rearag_runtime_probe_v1 import file_ref
from scripts.prepare.materialize_mixed3_rearag_runtime_probe_v3_proof400 import ARM_SPECS_V3


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v3_proof400_seed42_freeze/protocol.json"
DEFAULT_REPORT_DIR = ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v3_proof400_seed42_gpu_postflight"
REQUIRED_FINITE = (
    "mean_reward", "ppo_mean_kl", "loss_total", "loss_policy", "loss_value",
    "advantage_var", "value_mean", "return_mean",
)
REARAG_FINITE = (
    "mixed_text_raw_step_mean",
    "mixed_text_baseline_preupdate_step_mean",
    "mixed_text_centered_unclipped_step_mean",
    "mixed_text_centered_step_mean",
    "mixed_text_clip_frac",
    "mixed_text_ema_baseline",
)


def required_finite(row: dict[str, Any], field: str, arm: str) -> float:
    if field not in row or row[field] is None:
        raise ValueError(f"{arm}: required statistic {field} missing/null")
    try:
        value = float(row[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{arm}: required statistic {field} not numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{arm}: required statistic {field} non-finite")
    return value


def verify_rearag(row: dict[str, Any], arm: str) -> dict[str, Any]:
    if int(row.get("n_valid", 0)) < 1 or int(row.get("mixed_text_step_count", 0)) < 1:
        raise ValueError(f"{arm}: no valid ReaRAG-scored reasoning step")
    values = {field: required_finite(row, field, arm) for field in REARAG_FINITE}
    n_obs = int(row.get("mixed_text_ema_n_obs", 0))
    if n_obs < 1:
        raise ValueError(f"{arm}: ReaRAG EMA has no observations")
    by_dataset = row.get("mixed_reward_by_dataset") or {}
    if len(by_dataset) != 1:
        raise ValueError(f"{arm}: expected one dataset telemetry group")
    diag = next(iter(by_dataset.values()))
    if int(diag.get("valid_count", 0)) < 1 or int(diag.get("text_step_count", 0)) < 1:
        raise ValueError(f"{arm}: dataset telemetry has no valid text step")
    for field in (
        "text_raw_step_mean", "text_baseline_preupdate_step_mean",
        "text_centered_unclipped_step_mean", "text_centered_step_mean",
        "text_clip_frac", "text_ema_baseline",
    ):
        required_finite(diag, field, arm)
    if int(diag.get("text_ema_n_obs", 0)) < 1:
        raise ValueError(f"{arm}: dataset EMA observation count is zero")
    return {"step_count": int(row["mixed_text_step_count"]), "n_obs": n_obs,
            "finite_means": values}


def verify_process(row: dict[str, Any], expected_eligible: bool, arm: str) -> dict[str, Any]:
    by_dataset = row.get("mixed_reward_by_dataset") or {}
    if len(by_dataset) != 1:
        raise ValueError(f"{arm}: process dataset telemetry absent")
    diag = next(iter(by_dataset.values()))
    applied = int(row.get("proofkg_process_applied_count", -1))
    diag_applied = int(diag.get("process_applied_count", -1))
    weighted = required_finite(diag, "process_mean", arm)
    if expected_eligible:
        if int(row.get("proofkg_eligible_count", -1)) != 4:
            raise ValueError(f"{arm}: eligible count is not four")
        if applied < 1 or diag_applied < 1:
            raise ValueError(f"{arm}: no valid process_applied rollout")
        if math.isclose(weighted, 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{arm}: weighted ProofKG process contribution is zero")
        required_finite(row, "proofkg_process_mean", arm)
    else:
        if int(row.get("proofkg_eligible_count", -1)) != 0:
            raise ValueError(f"{arm}: PPO-T must have zero eligible rows")
        if applied != 0 or diag_applied != 0 or not math.isclose(weighted, 0.0, abs_tol=1e-12):
            raise ValueError(f"{arm}: PPO-T process must be strictly zero")
    return {"eligible": expected_eligible, "process_applied_count": applied,
            "weighted_process_mean": weighted}


def verify_arm(arm: str, spec: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / spec["output_dir"]
    history_path, manifest_path = out / "history.jsonl", out / "manifest.json"
    final = out / "final"
    required_files = [history_path, manifest_path, final / "adapter_config.json", final / "adapter_model.safetensors"]
    for path in required_files:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{arm}: missing artifact {path}")
    rows = [json.loads(line) for line in history_path.open(encoding="utf-8") if line.strip()]
    if len(rows) != 1 or int(rows[0].get("step", -1)) != 4:
        raise ValueError(f"{arm}: expected exactly one completed K4 update")
    row = rows[0]
    finite = {field: required_finite(row, field, arm) for field in REQUIRED_FINITE}
    if row.get("uses_explicit_sft_reference") is not True:
        raise ValueError(f"{arm}: explicit frozen SFT reference not used")
    history_kl = finite["ppo_mean_kl"]
    if abs(history_kl) > 1.0:
        raise ValueError(f"{arm}: initial reference KL exceeds 1: {history_kl}")
    if int(row.get("sft_replay_items", -1)) != 0:
        raise ValueError(f"{arm}: one-batch replay boundary changed")
    qids = row.get("rollout_qids") or []
    if len(qids) != 4 or len(set(qids)) != 1:
        raise ValueError(f"{arm}: runtime group is not K4 same-qid")
    rearag = verify_rearag(row, arm)
    process = verify_process(row, bool(spec["expected_eligible"]), arm)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = manifest.get("run") or {}
    if run.get("experiment_id") != spec["experiment_id"]:
        raise ValueError(f"{arm}: manifest Experiment ID mismatch")
    if run.get("reference_mode") != "explicit_frozen_sft_snapshot":
        raise ValueError(f"{arm}: manifest reference mode mismatch")
    manifest_kl = required_finite(run, "initial_reference_kl", arm)
    if abs(manifest_kl) > 1.0 or not math.isclose(manifest_kl, history_kl, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{arm}: manifest/history initial KL mismatch")
    cfg = run.get("config") or {}
    contract = {
        "alpha_gate_path": None, "alpha_override": None,
        "mixed_outcome_reward": True, "mixed_text_reward": True,
        "text_reward_backend": "rearag",
        "proofkg_process_reward": bool(spec["expected_eligible"]),
        "proofkg_process_version": "v2_1", "total_steps": 4,
        "rollouts_per_prompt": 4,
    }
    for key, expected in contract.items():
        if key not in cfg or cfg[key] != expected:
            raise ValueError(f"{arm}: manifest config {key} mismatch")
    if not isinstance((run.get("input_artifacts") or {}).get("rearag"), dict):
        raise ValueError(f"{arm}: ReaRAG artifact not bound")

    log_path = ROOT / spec["log_path"]
    if not log_path.is_file() or log_path.stat().st_size <= 0:
        raise FileNotFoundError(f"{arm}: training log missing")
    tb = Path(spec["tensorboard_dir"])
    events = sorted(tb.glob("events.out.tfevents.*")) if tb.is_dir() else []
    if not events or not any(path.stat().st_size > 0 for path in events):
        raise FileNotFoundError(f"{arm}: TensorBoard event missing")
    return {
        "experiment_id": spec["experiment_id"], "status": "PASS_ONE_K4_RUNTIME_UPDATE",
        "rollout_qids": qids, "required_finite_statistics": finite,
        "initial_reference_kl": manifest_kl, "rearag": rearag, "process": process,
        "runtime_config_contract": contract,
        "artifacts": {
            "history": file_ref(history_path), "manifest": file_ref(manifest_path),
            "adapter": file_ref(final / "adapter_model.safetensors"),
            "adapter_config": file_ref(final / "adapter_config.json"),
            "log": file_ref(log_path), "tensorboard_events": [file_ref(path) for path in events],
        },
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
    if protocol.get("status") != "FROZEN_NOT_RUN" or protocol.get("counts", {}).get("scheduled_trajectories_total") != 8:
        raise ValueError("wrong v3 protocol")
    arms = {arm: verify_arm(arm, spec) for arm, spec in ARM_SPECS_V3.items()}
    report = {
        "schema_version": "mixed3-rearag-runtime-wiring-probe-postflight-v3-proof400",
        "experiment_id": "MIXED3-REARAG-RUNTIME-WIRING-PROBE-V3-PROOF400-SEED42-POSTFLIGHT",
        "status": "PASS_RUNTIME_WIRING_NOT_EFFECT_EVIDENCE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": file_ref(protocol_path), "arms": arms, "total_trajectories": 8,
        "scientific_boundary": "One K4 update per route; no effect or convergence claim is valid.",
    }
    report_dir.mkdir(parents=True, exist_ok=False)
    path = report_dir / "postflight.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(report_dir, status=report["status"], extra={
        "phase": "mixed3_rearag_runtime_probe_v3_proof400_postflight",
        "experiment_id": report["experiment_id"], "total_trajectories": 8,
        "effect_evidence": False, "postflight_sha256": file_ref(path)["sha256"],
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
