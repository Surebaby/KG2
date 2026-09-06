#!/usr/bin/env python
"""Freeze two one-K4 Proof400 GPU-wiring probes without invoking CUDA."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from kgproweight.kg.question_kg import question_sha256
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.materialize_mixed3_rearag_runtime_probe_v1 import (
    choose_probe_groups, file_ref, read_jsonl, row_key, unique_index,
    validate_arm_assets, write_jsonl,
)
from scripts.prepare.materialize_mixed3_rearag_runtime_probe_v2 import (
    config_dependency_closure,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "data/silver_data/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42"
DATA_DIR = ROOT / "data/silver_data/mixed3_rearag_runtime_wiring_probe_v3_proof400_seed42"
AUDIT_DIR = ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v3_proof400_seed42_freeze"
EXPERIMENT_ID = "MIXED3-REARAG-RUNTIME-WIRING-PROBE-V3-PROOF400-SEED42-FREEZE"

ARM_SPECS_V3: dict[str, dict[str, Any]] = {
    "ppo_t_noneligible_k4": {
        "expected_eligible": False,
        "experiment_id": "ppo_mixed3_rearag_runtime_probe_v3_proof400_t_noneligible_k4_seed42",
        "config": "configs/training/phase3_ppo_mixed3_rearag_runtime_probe_v3_proof400_t_noneligible_k4_seed42.yaml",
        "formal_config": "configs/training/phase3_ppo_mixed3_rearag_v2_proof400_text7200_seed42.yaml",
        "output_dir": "outputs/ppo_mixed3_rearag_runtime_probe_v3_proof400_t_noneligible_k4_seed42",
        "log_path": "logs/training/ppo_mixed3_rearag_runtime_probe_v3_proof400_t_noneligible_k4_seed42.log",
        "tensorboard_dir": "/root/tf-logs/ppo_mixed3_rearag_runtime_probe_v3_proof400_t_noneligible_k4_seed42",
    },
    "ppo_tk_eligible_k4": {
        "expected_eligible": True,
        "experiment_id": "ppo_mixed3_rearag_runtime_probe_v3_proof400_tk_eligible_k4_seed42",
        "config": "configs/training/phase3_ppo_mixed3_rearag_runtime_probe_v3_proof400_tk_eligible_k4_seed42.yaml",
        "formal_config": "configs/training/phase3_ppo_mixed3_rearag_v2_proof400_text_kg_v2_1_7200_seed42.yaml",
        "output_dir": "outputs/ppo_mixed3_rearag_runtime_probe_v3_proof400_tk_eligible_k4_seed42",
        "log_path": "logs/training/ppo_mixed3_rearag_runtime_probe_v3_proof400_tk_eligible_k4_seed42.log",
        "tensorboard_dir": "/root/tf-logs/ppo_mixed3_rearag_runtime_probe_v3_proof400_tk_eligible_k4_seed42",
    },
}


def runtime_code_closure() -> list[Path]:
    paths = sorted(path for path in (ROOT / "kgproweight").rglob("*.py") if path.is_file())
    paths.extend([
        ROOT / "scripts/train/phase3_ppo.py",
        ROOT / "scripts/train/_split_args.py",
        ROOT / "scripts/prepare/resolve_phase3_ppo_runtime_config.py",
        Path(__file__).resolve(),
        ROOT / "scripts/prepare/preflight_mixed3_rearag_runtime_probe_v3_proof400.py",
        ROOT / "scripts/prepare/verify_mixed3_rearag_runtime_probe_v3_proof400.py",
        ROOT / "launch_ppo_mixed3_rearag_runtime_probe_v3_proof400_remote.sh",
        ROOT / "tests/test_mixed3_rearag_runtime_probe_v3_proof400.py",
    ])
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return sorted(set(path.resolve() for path in paths))


def make_schedule(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "schema_version": "mixed3-rearag-runtime-probe-fixed-schedule-v3-proof400",
        "rollout_index": index, "prompt_group_index": 1,
        "within_group_rollout": index, "dataset": source["dataset"],
        "qid": source["qid"], "question_sha256": source["question_sha256"],
        "stratum": source["stratum"],
        "process_reward_eligible": bool(source["process_reward_eligible"]),
    } for index, source in enumerate(group, start=1)]


def materialize(data_dir: Path = DATA_DIR, audit_dir: Path = AUDIT_DIR) -> dict[str, Any]:
    if data_dir.exists() or audit_dir.exists():
        raise FileExistsError(f"append-only v3 target exists: {data_dir} or {audit_dir}")
    source_paths = {
        name: SOURCE_DIR / name for name in (
            "silver_train.jsonl", "question_kg_records.jsonl", "sampling_weights.jsonl",
            "prompt_groups.jsonl", "fixed_rollout_schedule.jsonl", "report.json", "manifest.json",
        )
    }
    bound_reports = {
        "formal_protocol": ROOT / "outputs/audits/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_protocol/protocol.json",
        "family_scope_lock": ROOT / "outputs/audits/mixed_ppo_three_dataset_v2_proof400_family_scope_addendum_v2/addendum.json",
        "config_comparison": ROOT / "outputs/audits/mixed3_rearag_ppo_pair_proof400_7200_seed42_config_comparison_v2/report.json",
    }
    for path in [*source_paths.values(), *bound_reports.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    configs = [ROOT / spec[key] for spec in ARM_SPECS_V3.values() for key in ("config", "formal_config")]
    config_closure = config_dependency_closure(configs)
    code_closure = runtime_code_closure()

    silver = unique_index(read_jsonl(source_paths["silver_train.jsonl"]), "Proof400 silver")
    qkg = unique_index(read_jsonl(source_paths["question_kg_records.jsonl"]), "Proof400 qKG")
    groups = choose_probe_groups(read_jsonl(source_paths["fixed_rollout_schedule.jsonl"]))
    prepared = {}
    identities = set()
    for arm, spec in ARM_SPECS_V3.items():
        group = groups[arm]
        key = row_key(group[0])
        if key in identities or key not in silver or key not in qkg:
            raise ValueError(f"invalid/distinct source identity: {key}")
        identities.add(key)
        source_silver = copy.deepcopy(silver[key])
        if question_sha256(source_silver["question"]) != group[0]["question_sha256"]:
            raise ValueError(f"schedule hash mismatch: {key}")
        prepared[arm] = {
            "silver": [source_silver], "qkg": [copy.deepcopy(qkg[key])],
            "sampling": [{
                "schema_version": "mixed3-rearag-runtime-probe-sampling-weight-v3-proof400",
                "dataset": source_silver["dataset"], "qid": source_silver["qid"],
                "question_sha256": group[0]["question_sha256"],
                "sampling_probability": 1.0, "stratum": group[0]["stratum"],
                "process_reward_eligible": bool(spec["expected_eligible"]),
            }],
            "schedule": make_schedule(group),
            "source_prompt_group_index": group[0]["prompt_group_index"],
        }

    data_dir.mkdir(parents=True, exist_ok=False)
    arm_reports, output_refs = {}, {}
    for arm, payload in prepared.items():
        arm_dir = data_dir / arm
        arm_dir.mkdir(exist_ok=False)
        files = {
            "silver_train": ("silver_train.jsonl", payload["silver"]),
            "question_kg_records": ("question_kg_records.jsonl", payload["qkg"]),
            "sampling_weights": ("sampling_weights.jsonl", payload["sampling"]),
            "fixed_rollout_schedule": ("fixed_rollout_schedule.jsonl", payload["schedule"]),
        }
        for _label, (filename, rows) in files.items():
            write_jsonl(arm_dir / filename, rows)
        validated = validate_arm_assets(
            arm=arm, expected_eligible=ARM_SPECS_V3[arm]["expected_eligible"], arm_dir=arm_dir,
        )
        arm_reports[arm] = {
            **ARM_SPECS_V3[arm], **validated,
            "source_prompt_group_index": payload["source_prompt_group_index"],
            "prompt_groups": 1, "scheduled_trajectories": 4,
        }
        output_refs[arm] = {
            label: file_ref(arm_dir / filename) for label, (filename, _rows) in files.items()
        }

    report = {
        "schema_version": "mixed3-rearag-runtime-wiring-probe-freeze-v3-proof400",
        "experiment_id": EXPERIMENT_ID, "status": "FROZEN_NOT_RUN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "supersedes": {
            "v2_protocol": file_ref(ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v2_seed42_freeze/protocol.json"),
            "reason": "Formal data/config moved append-only to Proof400 v2; old v2 probe remains unrun and preserved.",
        },
        "selection_rule": "Earliest complete formal K4 group in each eligibility class; no answer/reward/output inspected.",
        "counts": {"arms": 2, "unique_questions": 2, "prompt_groups": 2, "scheduled_trajectories_total": 8},
        "arms": arm_reports,
        "inputs": {**{name: file_ref(path) for name, path in source_paths.items()},
                   **{name: file_ref(path) for name, path in bound_reports.items()}},
        "runtime_code_closure": {str(path.relative_to(ROOT)): file_ref(path) for path in code_closure},
        "config_dependency_closure": {str(path.relative_to(ROOT)): file_ref(path) for path in config_closure},
        "outputs": output_refs,
        "postflight_contract": {
            "explicit_sft_reference": True, "initial_reference_kl_abs_max": 1.0,
            "valid_rearag_steps_min": 1, "rearag_ema_observations_min": 1,
            "ppo_t_process_applied": 0, "ppo_t_weighted_process": 0.0,
            "ppo_tk_valid_process_applied_min": 1, "ppo_tk_weighted_process_nonzero": True,
            "required_finite_ppo_critic": [
                "mean_reward", "ppo_mean_kl", "loss_total", "loss_policy", "loss_value",
                "advantage_var", "value_mean", "return_mean",
            ],
        },
        "scientific_boundary": {
            "purpose": "GPU runtime wiring only", "effect_evidence": False,
            "formal_pair_modified": False, "formal_data_modified": False,
            "training_started": False, "gpu_invoked": False,
            "maximum_trajectories": 8,
        },
    }
    report_path = data_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(data_dir, status="FROZEN_NOT_RUN", extra={
        "phase": "mixed3_rearag_runtime_probe_v3_proof400",
        "experiment_id": EXPERIMENT_ID, "training_started": False, "gpu_invoked": False,
        "report_sha256": file_ref(report_path)["sha256"],
    })
    audit_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = audit_dir / "protocol.json"
    protocol_path.write_text(json.dumps({
        **report, "data_report": file_ref(report_path),
        "data_manifest": file_ref(data_dir / "manifest.json"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(audit_dir, status="FROZEN_NOT_RUN", extra={
        "phase": "mixed3_rearag_runtime_probe_v3_proof400_freeze",
        "experiment_id": EXPERIMENT_ID, "training_started": False, "gpu_invoked": False,
        "protocol_sha256": file_ref(protocol_path)["sha256"],
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--audit_dir", type=Path, default=AUDIT_DIR)
    args = parser.parse_args()
    report = materialize(args.data_dir.resolve(), args.audit_dir.resolve())
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "arms": {arm: {key: row[key] for key in ("identity", "process_reward_eligible", "kg_triples")}
                               for arm, row in report["arms"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
