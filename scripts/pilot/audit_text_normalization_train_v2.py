"""Train-only statistical diagnostics for the fixed text-normalization repair.

No outcome labels or held-out reward/answer metrics are read or computed.
Changing the bounded map guarantees zero hard clipping by construction; that
fact must never be described as improved answer accuracy or process utility.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from kgproweight.reward.source_reward_normalization_v2 import (
    fit_text_normalization_v2, normalize_text_steps_v2,
)


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def distribution(values):
    arr = np.asarray(values, dtype=float)
    return {"n": len(arr), "mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "p05": float(np.quantile(arr, .05)),
            "median": float(np.median(arr)), "p95": float(np.quantile(arr, .95)), "max": float(arr.max())} if len(arr) else {"n": 0}


def audit(calibration_dir, output_dir, experiment_id):
    manifest = json.loads((calibration_dir / "manifest.json").read_text())
    if (calibration_dir / "FAILED_CALIBRATION.json").exists():
        raise ValueError("failed source-credit calibration")
    for name in ("gate.json", "assignments.jsonl", "candidates.credit_masked.jsonl"):
        if identity(calibration_dir / name)["sha256"] != manifest["outputs"][name]["sha256"]:
            raise ValueError(f"frozen input hash mismatch: {name}")
    train_ids = set()
    # Only membership fields are used from assignments; no held-out fit metrics.
    for line in (calibration_dir / "assignments.jsonl").open():
        row = json.loads(line)
        if row["split"] == "train":
            train_ids.add(row["candidate_id"])
    rows = []
    for line in (calibration_dir / "candidates.credit_masked.jsonl").open():
        row = json.loads(line)
        if row["candidate_id"] not in train_ids:
            continue
        # Whitelist only train identity, validity and raw text observations.
        rows.append({name: row[name] for name in ("candidate_id", "dataset", "qid", "trajectory_valid", "raw_text")})
        rows[-1]["family_split"] = "train"
    if {row["candidate_id"] for row in rows} != train_ids:
        raise ValueError("frozen train membership missing from scored rows")
    stats = fit_text_normalization_v2(rows)
    # These historical parameters were fitted solely from this same train split.
    old = json.loads((calibration_dir / "gate.json").read_text())["normalization"]
    valid = [row for row in rows if row["trajectory_valid"]]
    old_means = [float(np.mean(row["raw_text"])) for row in valid]
    if not np.isclose(np.mean(old_means), old["text_center"], atol=1e-12, rtol=0):
        raise ValueError("train filter does not reproduce the frozen legacy center")
    raw = [value for row in valid for value in row["raw_text"]]
    old_z = [(value - old["text_center"]) / old["text_scale"] for value in raw]
    results = [normalize_text_steps_v2(row["raw_text"], stats) for row in valid]
    new_bounded = [value for result in results for value in result["bounded_step_scores"]]
    new_z = [value for result in results for value in result["normalized_unclipped_step_scores"]]
    by_dataset = defaultdict(list)
    for row in valid:
        by_dataset[row["dataset"]].append(row)
    report = {
        "schema_version": "train-only-text-normalization-statistical-diagnostic-v2",
        "experiment_id": experiment_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "TRAIN_ONLY_STATISTICAL_CONTRACT_DIAGNOSTIC_NOT_UTILITY_CLEARANCE",
        "scope": "frozen train membership only; no Gold, EM/F1, or new calibration/confirmation outcome metrics",
        "new_fixed_stats": stats,
        "legacy_train_mean_fit": {"text_center": old["text_center"], "text_scale": old["text_scale"],
                                  "trajectory_mean_distribution": distribution(old_means)},
        "train_step_distribution": distribution(raw),
        "train_map_telemetry": {
            "n_steps": len(raw), "old_hard_clip_fraction": float(np.mean(np.abs(old_z) > 1)),
            "new_hard_clip_fraction": 0.0,
            "new_raw_z_outside_unit_fraction": float(np.mean(np.abs(new_z) > 1)),
            "new_soft_saturation_fraction_abs_ge_0_95": float(np.mean(np.abs(new_bounded) >= .95)),
            "new_bounded_distribution": distribution(new_bounded),
            "new_mean_bounded_trajectory_distribution": distribution([result["mean_bounded"] for result in results]),
            "interpretation": "The mapping changed from hard clip to softsign. Zero hard clipping follows by definition; it is not an empirical utility improvement.",
        },
        "by_dataset_train_support": {dataset: {
            "questions": len({row["qid"] for row in items}), "valid_candidates": len(items),
            "raw_step_distribution": distribution([value for row in items for value in row["raw_text"]])}
            for dataset, items in sorted(by_dataset.items())},
        "source_mix_rebalanced": False, "gold_used": False,
        "new_confirmation_metrics_computed": False, "hyperparameter_search_performed": False,
        "core_runtime_modified": False, "policy_optimizer_updates": 0,
        "scientific_boundary": "The fixed step-unit, hierarchical-weight, softsign combination is a new reward contract. This train bank remains graph-heavy; this audit cannot establish multi-dataset representativeness or PPO/alpha benefit.",
        "bindings": {name: identity(calibration_dir / name) for name in (
            "manifest.json", "gate.json", "assignments.jsonl", "candidates.credit_masked.jsonl")},
        "code": identity(Path(__file__)),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("report.json", report), ("normalization.stats.json", stats)):
        with (output_dir / name).open("x") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    report = audit(**vars(parser.parse_args()))
    print(json.dumps({key: report[key] for key in (
        "status", "new_fixed_stats", "train_map_telemetry", "by_dataset_train_support")}, indent=2))
