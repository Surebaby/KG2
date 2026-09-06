#!/usr/bin/env python
"""Freeze the two one-batch mixed-ReaRAG PPO runtime-wiring probes.

This script does not load a model or start PPO.  It selects rows by a frozen,
outcome-independent rule from the formal 7,200-trajectory assets:

* PPO-T: the earliest fixed-schedule K=4 group that is process-ineligible;
* PPO-TK: the earliest fixed-schedule K=4 group that is identity-safe,
  complete ProofKG-v2.1 eligible.

The resulting assets are deliberately too small to estimate a training effect.
They exist only to exercise the production CUDA/runtime wiring once, before the
formal paired runs are allowed to consume a remote GPU reservation.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.kg.training_question_kg import (
    apply_training_question_kg,
    read_question_kg_records,
)
from kgproweight.reward.proofkg_process import (
    is_identity_safe_automatic_proofkg,
)
from kgproweight.training.phase3_ppo import _validate_v21_execution_preflight
from kgproweight.utils.logging import dump_manifest


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "data/silver_data/mixed_ppo_three_dataset_v1_n1799_k4_seed42"
DEFAULT_DATA_DIR = (
    ROOT / "data/silver_data/mixed3_rearag_runtime_wiring_probe_v1_seed42"
)
DEFAULT_AUDIT_DIR = (
    ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v1_seed42_freeze"
)
EXPERIMENT_ID = "MIXED3-REARAG-RUNTIME-WIRING-PROBE-V1-SEED42-FREEZE"

ARM_SPECS: dict[str, dict[str, Any]] = {
    "ppo_t_noneligible_k4": {
        "expected_eligible": False,
        "experiment_id": "ppo_mixed3_rearag_runtime_probe_v1_t_noneligible_k4_seed42",
        "config": "configs/training/phase3_ppo_mixed3_rearag_runtime_probe_v1_t_noneligible_k4_seed42.yaml",
        "formal_config": "configs/training/phase3_ppo_mixed3_rearag_v1_text7200_seed42.yaml",
        "output_dir": "outputs/ppo_mixed3_rearag_runtime_probe_v1_t_noneligible_k4_seed42",
        "log_path": "logs/training/ppo_mixed3_rearag_runtime_probe_v1_t_noneligible_k4_seed42.log",
        "tensorboard_dir": "/root/tf-logs/ppo_mixed3_rearag_runtime_probe_v1_t_noneligible_k4_seed42",
    },
    "ppo_tk_eligible_k4": {
        "expected_eligible": True,
        "experiment_id": "ppo_mixed3_rearag_runtime_probe_v1_tk_eligible_k4_seed42",
        "config": "configs/training/phase3_ppo_mixed3_rearag_runtime_probe_v1_tk_eligible_k4_seed42.yaml",
        "formal_config": "configs/training/phase3_ppo_mixed3_rearag_v1_text_kg_v2_1_7200_seed42.yaml",
        "output_dir": "outputs/ppo_mixed3_rearag_runtime_probe_v1_tk_eligible_k4_seed42",
        "log_path": "logs/training/ppo_mixed3_rearag_runtime_probe_v1_tk_eligible_k4_seed42.log",
        "tensorboard_dir": "/root/tf-logs/ppo_mixed3_rearag_runtime_probe_v1_tk_eligible_k4_seed42",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        label = str(path.relative_to(ROOT))
    except ValueError:
        label = str(path)
    return {"path": label, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def row_key(row: Mapping[str, Any]) -> str:
    return question_key(str(row.get("dataset") or ""), str(row.get("qid") or ""))


def unique_index(rows: Iterable[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in rows:
        row = dict(value)
        key = row_key(row)
        if key in result:
            raise ValueError(f"{label} duplicate identity: {key}")
        result[key] = row
    return result


def choose_probe_groups(schedule: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Choose the earliest complete K=4 group in each eligibility class."""

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in schedule:
        grouped.setdefault(int(row.get("prompt_group_index", -1)), []).append(row)
    candidates: dict[bool, list[tuple[int, list[dict[str, Any]]]]] = {False: [], True: []}
    for group_index, rows in grouped.items():
        rows = sorted(rows, key=lambda value: int(value.get("within_group_rollout", -1)))
        if len(rows) != 4 or [int(row.get("within_group_rollout", -1)) for row in rows] != [1, 2, 3, 4]:
            raise ValueError(f"source schedule group {group_index} is not one complete K=4 group")
        identities = {row_key(row) for row in rows}
        eligibility = {bool(row.get("process_reward_eligible")) for row in rows}
        if len(identities) != 1 or len(eligibility) != 1:
            raise ValueError(f"source schedule group {group_index} is internally inconsistent")
        eligible = next(iter(eligibility))
        candidates[eligible].append((group_index, rows))
    if not candidates[False] or not candidates[True]:
        raise ValueError("source schedule lacks an eligible or non-eligible K=4 group")
    return {
        "ppo_t_noneligible_k4": min(candidates[False], key=lambda pair: pair[0])[1],
        "ppo_tk_eligible_k4": min(candidates[True], key=lambda pair: pair[0])[1],
    }


def make_probe_schedule(source_group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for index, source in enumerate(source_group, start=1):
        output.append({
            "schema_version": "mixed3-rearag-runtime-probe-fixed-schedule-v1",
            "rollout_index": index,
            "prompt_group_index": 1,
            "within_group_rollout": index,
            "dataset": str(source["dataset"]),
            "qid": str(source["qid"]),
            "question_sha256": str(source["question_sha256"]),
            "stratum": str(source["stratum"]),
            "process_reward_eligible": bool(source["process_reward_eligible"]),
        })
    return output


def validate_arm_assets(
    *, arm: str, expected_eligible: bool, arm_dir: Path,
) -> dict[str, Any]:
    trajectories = list(SilverDatasetReader(arm_dir / "silver_train.jsonl", split=None).accepted())
    if len(trajectories) != 1:
        raise ValueError(f"{arm}: expected exactly one accepted trajectory")
    trajectory = trajectories[0]
    stats = apply_training_question_kg(
        trajectories,
        read_question_kg_records(arm_dir / "question_kg_records.jsonl"),
        min_coverage=1.0,
        require_nonempty=False,
    )
    runtime = trajectory.metadata.get("question_kg_runtime") or {}
    actual_eligible = is_identity_safe_automatic_proofkg(
        runtime,
        trajectory.kg_subgraph,
        dataset=trajectory.dataset,
        qid=trajectory.qid,
    )
    if actual_eligible is not expected_eligible:
        raise ValueError(
            f"{arm}: production eligibility={actual_eligible}, expected={expected_eligible}"
        )
    execution = None
    if expected_eligible:
        execution = _validate_v21_execution_preflight(trajectories)
    if trajectory.steps:
        raise ValueError(f"{arm}: rollout-only probe must not contain supervised Gold steps")
    if str(trajectory.teacher_output or "").strip():
        raise ValueError(f"{arm}: rollout-only probe must not contain a teacher trace")
    if trajectory.metadata.get("gold_use") != "outcome_reward_label_only":
        raise ValueError(f"{arm}: train Gold may be used only as the outcome label")
    if trajectory.metadata.get("failed_qpeg_or_saeg_p_edges_included") is not False:
        raise ValueError(f"{arm}: failed QPEG/P edges are forbidden")
    return {
        "identity": question_key(str(trajectory.dataset), str(trajectory.qid)),
        "dataset": str(trajectory.dataset),
        "qid": str(trajectory.qid),
        "question_sha256": question_sha256(str(trajectory.question)),
        "kg_triples": len(trajectory.kg_subgraph),
        "process_reward_eligible": actual_eligible,
        "question_kg_join": stats.to_dict(),
        "v2_1_execution": execution,
    }


def materialize(*, data_dir: Path, audit_dir: Path) -> dict[str, Any]:
    if data_dir.exists():
        raise FileExistsError(f"append-only probe data already exists: {data_dir}")
    if audit_dir.exists():
        raise FileExistsError(f"append-only probe freeze already exists: {audit_dir}")

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

    configs_and_code = {
        "launcher": ROOT / "launch_ppo_mixed3_rearag_runtime_probe_v1_remote.sh",
        "materializer": Path(__file__).resolve(),
        "preflight": ROOT / "scripts/prepare/preflight_mixed3_rearag_runtime_probe_v1.py",
        "postflight": ROOT / "scripts/prepare/verify_mixed3_rearag_runtime_probe_v1.py",
        "runtime": ROOT / "kgproweight/training/phase3_ppo.py",
        **{
            f"{arm}_config": ROOT / spec["config"] for arm, spec in ARM_SPECS.items()
        },
        **{
            f"{arm}_formal_config": ROOT / spec["formal_config"]
            for arm, spec in ARM_SPECS.items()
        },
    }
    for path in configs_and_code.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    silver = unique_index(read_jsonl(source_paths["silver_train"]), "source silver")
    qkg = unique_index(read_jsonl(source_paths["question_kg_records"]), "source qKG")
    selected_groups = choose_probe_groups(read_jsonl(source_paths["fixed_rollout_schedule"]))

    prepared: dict[str, dict[str, Any]] = {}
    selected_keys: set[str] = set()
    for arm, spec in ARM_SPECS.items():
        group = selected_groups[arm]
        key = row_key(group[0])
        if key in selected_keys:
            raise ValueError("probe arms must use distinct identities")
        selected_keys.add(key)
        if key not in silver or key not in qkg:
            raise ValueError(f"{arm}: selected source identity is not fully materialized: {key}")
        source_silver = copy.deepcopy(silver[key])
        source_qkg = copy.deepcopy(qkg[key])
        if question_sha256(str(source_silver.get("question") or "")) != str(group[0]["question_sha256"]):
            raise ValueError(f"{arm}: selected schedule question hash is stale")
        schedule = make_probe_schedule(group)
        sampling = [{
            "schema_version": "mixed3-rearag-runtime-probe-sampling-weight-v1",
            "dataset": source_silver["dataset"],
            "qid": source_silver["qid"],
            "question_sha256": group[0]["question_sha256"],
            "sampling_probability": 1.0,
            "stratum": group[0]["stratum"],
            "process_reward_eligible": bool(spec["expected_eligible"]),
        }]
        prepared[arm] = {
            "silver": [source_silver],
            "qkg": [source_qkg],
            "sampling": sampling,
            "schedule": schedule,
            "source_prompt_group_index": int(group[0]["prompt_group_index"]),
        }

    # Create only after all source/config contracts above have passed.
    data_dir.mkdir(parents=True, exist_ok=False)
    arm_reports: dict[str, Any] = {}
    output_refs: dict[str, Any] = {}
    for arm, payload in prepared.items():
        arm_dir = data_dir / arm
        arm_dir.mkdir(exist_ok=False)
        write_jsonl(arm_dir / "silver_train.jsonl", payload["silver"])
        write_jsonl(arm_dir / "question_kg_records.jsonl", payload["qkg"])
        write_jsonl(arm_dir / "sampling_weights.jsonl", payload["sampling"])
        write_jsonl(arm_dir / "fixed_rollout_schedule.jsonl", payload["schedule"])
        validation = validate_arm_assets(
            arm=arm,
            expected_eligible=bool(ARM_SPECS[arm]["expected_eligible"]),
            arm_dir=arm_dir,
        )
        arm_reports[arm] = {
            **ARM_SPECS[arm],
            **validation,
            "source_prompt_group_index": payload["source_prompt_group_index"],
            "prompt_groups": 1,
            "scheduled_trajectories": 4,
        }
        output_refs[arm] = {
            name: file_ref(arm_dir / filename)
            for name, filename in {
                "silver_train": "silver_train.jsonl",
                "question_kg_records": "question_kg_records.jsonl",
                "sampling_weights": "sampling_weights.jsonl",
                "fixed_rollout_schedule": "fixed_rollout_schedule.jsonl",
            }.items()
        }

    created_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": "mixed3-rearag-runtime-wiring-probe-freeze-v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "FROZEN_NOT_RUN",
        "created_at_utc": created_at,
        "selection_rule": (
            "Earliest complete formal K=4 fixed-schedule group in each production "
            "process-eligibility class; no outcome, reward, or prediction was inspected."
        ),
        "counts": {
            "arms": 2,
            "unique_questions": 2,
            "prompt_groups": 2,
            "scheduled_trajectories_total": 8,
            "ppo_t_noneligible_trajectories": 4,
            "ppo_tk_eligible_trajectories": 4,
        },
        "arms": arm_reports,
        "inputs": {name: file_ref(path) for name, path in source_paths.items()},
        "code_and_configs": {
            name: file_ref(path) for name, path in configs_and_code.items()
        },
        "outputs": output_refs,
        "scientific_boundary": {
            "purpose": "GPU runtime wiring only",
            "training_effect_estimation": False,
            "paired_effect_comparison": False,
            "reason_not_paired": "The two arms intentionally exercise different eligibility routes and different qids.",
            "formal_pair_modified": False,
            "formal_data_modified": False,
            "training_started": False,
            "gpu_invoked": False,
            "gold_in_prompt_fields": False,
            "train_gold_use": "outcome label only",
            "sft_replay_boundary": (
                "The formal 10% replay configuration is retained, but one batch gives "
                "only 0.4 replay credit and therefore does not exercise a replay update."
            ),
            "pass_condition": (
                "Each arm must finish one K=4 update, pass initial reference-KL and "
                "finite diagnostic guards, write final/history/manifest, emit the "
                "expected route telemetry, and create TensorBoard event data."
            ),
        },
    }
    report_path = data_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(data_dir, status="FROZEN_NOT_RUN", extra={
        "phase": "mixed3_rearag_runtime_wiring_probe",
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
        "phase": "mixed3_rearag_runtime_wiring_probe_freeze",
        "experiment_id": EXPERIMENT_ID,
        "training_started": False,
        "gpu_invoked": False,
        "protocol_sha256": sha256_file(protocol_path),
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--audit_dir", type=Path, default=DEFAULT_AUDIT_DIR)
    args = parser.parse_args()
    report = materialize(data_dir=args.data_dir.resolve(), audit_dir=args.audit_dir.resolve())
    print(json.dumps({
        "status": report["status"],
        "experiment_id": report["experiment_id"],
        "counts": report["counts"],
        "selected": {
            arm: {key: value for key, value in details.items() if key in {
                "identity", "source_prompt_group_index", "process_reward_eligible", "kg_triples"
            }}
            for arm, details in report["arms"].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
