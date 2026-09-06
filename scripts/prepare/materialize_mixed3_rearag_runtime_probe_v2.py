#!/usr/bin/env python
"""Freeze corrected v2 one-batch PPO-T/PPO-TK GPU wiring-probe assets.

No model is loaded and no training is started.  V2 supersedes the unrun v1
freeze because it binds the real CLI entry point and a broad runtime import
closure after the alpha-CLI forwarding fix.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from kgproweight.kg.question_kg import question_sha256
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.materialize_mixed3_rearag_runtime_probe_v1 import (
    SOURCE_DIR,
    choose_probe_groups,
    file_ref,
    read_jsonl,
    row_key,
    unique_index,
    validate_arm_assets,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data/silver_data/mixed3_rearag_runtime_wiring_probe_v2_seed42"
DEFAULT_AUDIT_DIR = ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v2_seed42_freeze"
DEFAULT_SUPERSESSION_DIR = ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v1_seed42_supersession"
EXPERIMENT_ID = "MIXED3-REARAG-RUNTIME-WIRING-PROBE-V2-SEED42-FREEZE"

ARM_SPECS_V2: dict[str, dict[str, Any]] = {
    "ppo_t_noneligible_k4": {
        "expected_eligible": False,
        "experiment_id": "ppo_mixed3_rearag_runtime_probe_v2_t_noneligible_k4_seed42",
        "config": "configs/training/phase3_ppo_mixed3_rearag_runtime_probe_v2_t_noneligible_k4_seed42.yaml",
        "formal_config": "configs/training/phase3_ppo_mixed3_rearag_v1_text7200_seed42.yaml",
        "output_dir": "outputs/ppo_mixed3_rearag_runtime_probe_v2_t_noneligible_k4_seed42",
        "log_path": "logs/training/ppo_mixed3_rearag_runtime_probe_v2_t_noneligible_k4_seed42.log",
        "tensorboard_dir": "/root/tf-logs/ppo_mixed3_rearag_runtime_probe_v2_t_noneligible_k4_seed42",
    },
    "ppo_tk_eligible_k4": {
        "expected_eligible": True,
        "experiment_id": "ppo_mixed3_rearag_runtime_probe_v2_tk_eligible_k4_seed42",
        "config": "configs/training/phase3_ppo_mixed3_rearag_runtime_probe_v2_tk_eligible_k4_seed42.yaml",
        "formal_config": "configs/training/phase3_ppo_mixed3_rearag_v1_text_kg_v2_1_7200_seed42.yaml",
        "output_dir": "outputs/ppo_mixed3_rearag_runtime_probe_v2_tk_eligible_k4_seed42",
        "log_path": "logs/training/ppo_mixed3_rearag_runtime_probe_v2_tk_eligible_k4_seed42.log",
        "tensorboard_dir": "/root/tf-logs/ppo_mixed3_rearag_runtime_probe_v2_tk_eligible_k4_seed42",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_dependency_closure(config_paths: list[Path]) -> list[Path]:
    """Return every recursively included YAML, fail-closed on missing files."""

    seen: set[Path] = set()

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in seen:
            return
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(path)
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for include in doc.get("includes") or []:
            visit(path.parent / str(include))

    for path in config_paths:
        visit(path)
    return sorted(seen)


def runtime_code_closure() -> list[Path]:
    """Broadly bind the runtime package plus every executable probe entry."""

    paths = sorted(path for path in (ROOT / "kgproweight").rglob("*.py") if path.is_file())
    paths.extend([
        ROOT / "scripts/train/phase3_ppo.py",
        ROOT / "scripts/train/_split_args.py",
        ROOT / "scripts/prepare/resolve_phase3_ppo_runtime_config.py",
        ROOT / "scripts/prepare/materialize_mixed3_rearag_runtime_probe_v1.py",
        ROOT / "scripts/prepare/preflight_mixed3_rearag_runtime_probe_v1.py",
        Path(__file__).resolve(),
        ROOT / "scripts/prepare/preflight_mixed3_rearag_runtime_probe_v2.py",
        ROOT / "scripts/prepare/verify_mixed3_rearag_runtime_probe_v2.py",
        ROOT / "launch_ppo_mixed3_rearag_runtime_probe_v2_remote.sh",
    ])
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"runtime closure paths missing: {missing}")
    return sorted(set(path.resolve() for path in paths))


def make_schedule(source_group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "schema_version": "mixed3-rearag-runtime-probe-fixed-schedule-v2",
        "rollout_index": index,
        "prompt_group_index": 1,
        "within_group_rollout": index,
        "dataset": str(source["dataset"]),
        "qid": str(source["qid"]),
        "question_sha256": str(source["question_sha256"]),
        "stratum": str(source["stratum"]),
        "process_reward_eligible": bool(source["process_reward_eligible"]),
    } for index, source in enumerate(source_group, start=1)]


def materialize(data_dir: Path, audit_dir: Path, supersession_dir: Path) -> dict[str, Any]:
    if data_dir.exists() or audit_dir.exists() or supersession_dir.exists():
        raise FileExistsError(
            "v2 append-only destination exists: "
            f"{data_dir}, {audit_dir}, or {supersession_dir}"
        )
    source_paths = {
        "silver_train": SOURCE_DIR / "silver_train.jsonl",
        "question_kg_records": SOURCE_DIR / "question_kg_records.jsonl",
        "sampling_weights": SOURCE_DIR / "sampling_weights.jsonl",
        "fixed_rollout_schedule": SOURCE_DIR / "fixed_rollout_schedule.jsonl",
        "formal_data_manifest": SOURCE_DIR / "manifest.json",
        "formal_pair_manifest": ROOT / "outputs/audits/mixed3_rearag_ppo_pair_7200_seed42_v2/pair_manifest.json",
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    config_paths = [ROOT / spec["config"] for spec in ARM_SPECS_V2.values()]
    config_paths += [ROOT / spec["formal_config"] for spec in ARM_SPECS_V2.values()]
    config_closure = config_dependency_closure(config_paths)
    code_closure = runtime_code_closure()

    silver = unique_index(read_jsonl(source_paths["silver_train"]), "source silver")
    qkg = unique_index(read_jsonl(source_paths["question_kg_records"]), "source qKG")
    groups = choose_probe_groups(read_jsonl(source_paths["fixed_rollout_schedule"]))
    prepared: dict[str, dict[str, Any]] = {}
    for arm, spec in ARM_SPECS_V2.items():
        source_group = groups[arm]
        key = row_key(source_group[0])
        if key not in silver or key not in qkg:
            raise ValueError(f"{arm}: selected source is incomplete: {key}")
        source_silver = copy.deepcopy(silver[key])
        if question_sha256(str(source_silver.get("question") or "")) != str(source_group[0]["question_sha256"]):
            raise ValueError(f"{arm}: source question hash mismatch")
        prepared[arm] = {
            "silver": [source_silver],
            "qkg": [copy.deepcopy(qkg[key])],
            "sampling": [{
                "schema_version": "mixed3-rearag-runtime-probe-sampling-weight-v2",
                "dataset": source_silver["dataset"],
                "qid": source_silver["qid"],
                "question_sha256": source_group[0]["question_sha256"],
                "sampling_probability": 1.0,
                "stratum": source_group[0]["stratum"],
                "process_reward_eligible": bool(spec["expected_eligible"]),
            }],
            "schedule": make_schedule(source_group),
            "source_prompt_group_index": int(source_group[0]["prompt_group_index"]),
        }

    data_dir.mkdir(parents=True, exist_ok=False)
    arm_reports: dict[str, Any] = {}
    output_refs: dict[str, Any] = {}
    for arm, payload in prepared.items():
        arm_dir = data_dir / arm
        arm_dir.mkdir(exist_ok=False)
        for filename, rows in (
            ("silver_train.jsonl", payload["silver"]),
            ("question_kg_records.jsonl", payload["qkg"]),
            ("sampling_weights.jsonl", payload["sampling"]),
            ("fixed_rollout_schedule.jsonl", payload["schedule"]),
        ):
            write_jsonl(arm_dir / filename, rows)
        validated = validate_arm_assets(
            arm=arm, expected_eligible=bool(ARM_SPECS_V2[arm]["expected_eligible"]), arm_dir=arm_dir,
        )
        arm_reports[arm] = {
            **ARM_SPECS_V2[arm], **validated,
            "source_prompt_group_index": payload["source_prompt_group_index"],
            "prompt_groups": 1, "scheduled_trajectories": 4,
        }
        output_refs[arm] = {
            label: file_ref(arm_dir / filename)
            for label, filename in (
                ("silver_train", "silver_train.jsonl"),
                ("question_kg_records", "question_kg_records.jsonl"),
                ("sampling_weights", "sampling_weights.jsonl"),
                ("fixed_rollout_schedule", "fixed_rollout_schedule.jsonl"),
            )
        }

    report = {
        "schema_version": "mixed3-rearag-runtime-wiring-probe-freeze-v2",
        "experiment_id": EXPERIMENT_ID,
        "status": "FROZEN_NOT_RUN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "supersedes": {
            "protocol": "outputs/audits/mixed3_rearag_runtime_wiring_probe_v1_seed42_freeze/protocol.json",
            "reason": (
                "Unrun v1 omitted scripts/train/phase3_ppo.py and its broad import closure; "
                "its postflight also read the runtime Experiment ID at the wrong manifest level."
            ),
        },
        "selection_rule": (
            "Earliest complete formal K=4 group in each process-eligibility class; "
            "no answer, reward, rollout, or prediction inspected."
        ),
        "counts": {"arms": 2, "unique_questions": 2, "prompt_groups": 2, "scheduled_trajectories_total": 8},
        "arms": arm_reports,
        "inputs": {name: file_ref(path) for name, path in source_paths.items()},
        "runtime_code_closure": {
            str(path.relative_to(ROOT)): file_ref(path) for path in code_closure
        },
        "config_dependency_closure": {
            str(path.relative_to(ROOT)): file_ref(path) for path in config_closure
        },
        "outputs": output_refs,
        "scientific_boundary": {
            "purpose": "GPU runtime wiring only",
            "training_effect_estimation": False,
            "paired_effect_comparison": False,
            "formal_pair_modified": False,
            "formal_data_modified": False,
            "training_started": False,
            "gpu_invoked": False,
            "gold_in_prompt_fields": False,
            "train_gold_use": "outcome label only",
            "replay": "Formal 10% setting retained; one batch accrues 0.4 credit and executes zero replay items.",
        },
    }
    report_path = data_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(data_dir, status="FROZEN_NOT_RUN", extra={
        "phase": "mixed3_rearag_runtime_wiring_probe_v2",
        "experiment_id": EXPERIMENT_ID,
        "training_started": False,
        "gpu_invoked": False,
        "report_sha256": sha256_file(report_path),
    })
    audit_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = audit_dir / "protocol.json"
    protocol_path.write_text(json.dumps({
        **report,
        "data_report": file_ref(report_path),
        "data_manifest": file_ref(data_dir / "manifest.json"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(audit_dir, status="FROZEN_NOT_RUN", extra={
        "phase": "mixed3_rearag_runtime_wiring_probe_v2_freeze",
        "experiment_id": EXPERIMENT_ID,
        "training_started": False,
        "gpu_invoked": False,
        "protocol_sha256": sha256_file(protocol_path),
    })

    # Append-only record beside (never inside) the preserved v1 freeze/data.
    v1_protocol = ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v1_seed42_freeze/protocol.json"
    v1_data_manifest = ROOT / "data/silver_data/mixed3_rearag_runtime_wiring_probe_v1_seed42/manifest.json"
    supersession = {
        "schema_version": "mixed3-rearag-runtime-wiring-probe-supersession-v1",
        "experiment_id": "MIXED3-REARAG-RUNTIME-WIRING-PROBE-V1-SUPERSESSION",
        "status": "SUPERSEDED_NOT_RUN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "superseded": {
            "protocol": file_ref(v1_protocol),
            "data_manifest": file_ref(v1_data_manifest),
        },
        "superseded_by": {
            "protocol": file_ref(protocol_path),
            "data_manifest": file_ref(data_dir / "manifest.json"),
        },
        "reasons": [
            "v1 did not bind scripts/train/phase3_ppo.py or the broad runtime import closure",
            "v1 was frozen before the mixed-route alpha-null CLI forwarding correction",
            "v1 postflight looked for experiment_id at the manifest root instead of run.experiment_id",
        ],
        "preservation": {
            "v1_protocol_modified": False,
            "v1_data_modified": False,
            "v1_training_started": False,
            "v1_gpu_invoked": False,
        },
        "execution_rule": "Use only launch_ppo_mixed3_rearag_runtime_probe_v2_remote.sh.",
    }
    supersession_dir.mkdir(parents=True, exist_ok=False)
    supersession_path = supersession_dir / "supersession.json"
    supersession_path.write_text(
        json.dumps(supersession, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(supersession_dir, status=supersession["status"], extra={
        "phase": "mixed3_rearag_runtime_probe_v1_supersession",
        "experiment_id": supersession["experiment_id"],
        "training_started": False,
        "gpu_invoked": False,
        "supersession_sha256": sha256_file(supersession_path),
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--audit_dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--supersession_dir", type=Path, default=DEFAULT_SUPERSESSION_DIR)
    args = parser.parse_args()
    report = materialize(
        args.data_dir.resolve(), args.audit_dir.resolve(), args.supersession_dir.resolve()
    )
    print(json.dumps({
        "status": report["status"], "experiment_id": report["experiment_id"],
        "counts": report["counts"],
        "runtime_code_files": len(report["runtime_code_closure"]),
        "config_dependency_files": len(report["config_dependency_closure"]),
        "arms": {arm: {k: row[k] for k in ("identity", "process_reward_eligible", "kg_triples")}
                 for arm, row in report["arms"].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
