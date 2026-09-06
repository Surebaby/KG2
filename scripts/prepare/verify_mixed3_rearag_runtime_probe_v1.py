#!/usr/bin/env python
"""Verify completed GPU artifacts from both runtime probes, append-only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from kgproweight.utils.logging import dump_manifest
from scripts.prepare.materialize_mixed3_rearag_runtime_probe_v1 import ARM_SPECS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v1_seed42_freeze/protocol.json"
)
DEFAULT_REPORT_DIR = (
    ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v1_seed42_gpu_postflight"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_or_none(value: Any) -> bool:
    return value is None or (
        isinstance(value, (int, float)) and math.isfinite(float(value))
    )


def verify_arm(arm: str, spec: dict[str, Any]) -> dict[str, Any]:
    out_dir = ROOT / str(spec["output_dir"])
    history_path = out_dir / "history.jsonl"
    manifest_path = out_dir / "manifest.json"
    final_dir = out_dir / "final"
    required = [
        history_path,
        manifest_path,
        final_dir / "adapter_config.json",
        final_dir / "adapter_model.safetensors",
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{arm}: missing completed runtime artifact: {path}")
    history = [
        json.loads(line) for line in history_path.open(encoding="utf-8") if line.strip()
    ]
    if len(history) != 1 or int(history[0].get("step", -1)) != 4:
        raise ValueError(f"{arm}: expected exactly one completed K=4 update")
    row = history[0]
    expected_eligible = 4 if bool(spec["expected_eligible"]) else 0
    if int(row.get("proofkg_eligible_count", -1)) != expected_eligible:
        raise ValueError(
            f"{arm}: runtime eligibility count={row.get('proofkg_eligible_count')}, "
            f"expected={expected_eligible}"
        )
    if len(row.get("rollout_qids") or []) != 4 or len(set(row["rollout_qids"])) != 1:
        raise ValueError(f"{arm}: runtime did not preserve one contiguous K=4 qid group")
    dataset_rows = row.get("mixed_reward_by_dataset") or {}
    if len(dataset_rows) != 1:
        raise ValueError(f"{arm}: mixed reward route telemetry is absent or ambiguous")
    dataset_diag = next(iter(dataset_rows.values()))
    if int(dataset_diag.get("count", -1)) != 4:
        raise ValueError(f"{arm}: mixed reward route did not score all four trajectories")
    if int(dataset_diag.get("proofkg_eligible_count", -1)) != expected_eligible:
        raise ValueError(f"{arm}: dataset-level process eligibility telemetry is wrong")
    for field in (
        "mean_reward", "ppo_mean_kl", "loss_total", "loss_policy", "loss_value",
        "policy_clipfrac", "advantage_var", "value_mean", "return_mean",
    ):
        if not finite_or_none(row.get(field)):
            raise ValueError(f"{arm}: non-finite {field}={row.get(field)!r}")
    if row.get("uses_explicit_sft_reference") is not True:
        raise ValueError(f"{arm}: explicit frozen SFT reference was not used")
    if int(row.get("sft_replay_items", -1)) != 0:
        raise ValueError(f"{arm}: one-batch replay boundary unexpectedly changed")

    tb_dir = Path(str(spec["tensorboard_dir"]))
    events = sorted(tb_dir.glob("events.out.tfevents.*")) if tb_dir.is_dir() else []
    if not events or not any(path.stat().st_size > 0 for path in events):
        raise FileNotFoundError(f"{arm}: TensorBoard event file was not emitted under {tb_dir}")
    log_path = ROOT / str(spec["log_path"])
    if not log_path.is_file() or log_path.stat().st_size <= 0:
        raise FileNotFoundError(f"{arm}: runtime log missing: {log_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != spec["experiment_id"]:
        raise ValueError(f"{arm}: runtime manifest Experiment ID mismatch")
    return {
        "experiment_id": spec["experiment_id"],
        "status": "PASS_ONE_K4_RUNTIME_UPDATE",
        "output_dir": spec["output_dir"],
        "history": {
            "path": str(history_path.relative_to(ROOT)),
            "sha256": sha256_file(history_path),
            "step": row["step"],
            "rollout_qids": row["rollout_qids"],
            "proofkg_eligible_count": row["proofkg_eligible_count"],
            "valid_rate": row.get("valid_rate"),
            "ppo_mean_kl": row.get("ppo_mean_kl"),
            "mean_reward": row.get("mean_reward"),
        },
        "final_adapter_sha256": sha256_file(final_dir / "adapter_model.safetensors"),
        "runtime_manifest_sha256": sha256_file(manifest_path),
        "log_sha256": sha256_file(log_path),
        "tensorboard_event_files": [
            {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in events
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    report_dir = args.report_dir.resolve()
    if report_dir.exists():
        raise FileExistsError(f"append-only GPU postflight already exists: {report_dir}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_NOT_RUN":
        raise ValueError("runtime probe protocol is not the frozen not-run version")
    arms = {arm: verify_arm(arm, spec) for arm, spec in ARM_SPECS.items()}
    report = {
        "schema_version": "mixed3-rearag-runtime-wiring-probe-postflight-v1",
        "experiment_id": "MIXED3-REARAG-RUNTIME-WIRING-PROBE-V1-SEED42-POSTFLIGHT",
        "status": "PASS_RUNTIME_WIRING_NOT_EFFECT_EVIDENCE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "arms": arms,
        "total_trajectories": 8,
        "scientific_boundary": (
            "The probe proves one-update runtime wiring only. The qids differ across "
            "arms, sample size is one prompt per route, and no effect/convergence claim is valid."
        ),
    }
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = report_dir / "postflight.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(report_dir, status=report["status"], extra={
        "phase": "mixed3_rearag_runtime_wiring_probe_postflight",
        "experiment_id": report["experiment_id"],
        "total_trajectories": 8,
        "effect_evidence": False,
        "postflight_sha256": sha256_file(report_path),
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

