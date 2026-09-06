#!/usr/bin/env python
"""Freeze paired PPO-O/PPO-K configs, inputs, code and expected qid schedule."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from kgproweight.config import ProjectConfig, load_config
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.training.phase3_ppo import _advance_replay_credit, _sample_rollout_indices
from kgproweight.utils.logging import dump_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome_config", type=Path, required=True)
    parser.add_argument("--process_config", type=Path, required=True)
    parser.add_argument("--data_report", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--lock_suffix", default=".lock.json")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite: {args.output_dir}")

    outcome = load_config(args.outcome_config, validate=ProjectConfig)
    process = load_config(args.process_config, validate=ProjectConfig)
    left = _jsonable(outcome.training)
    right = _jsonable(process.training)
    left["output_dir"] = right["output_dir"] = "<ARM_OUTPUT_DIR>"
    left["ppo"]["proofkg_process_reward"] = right["ppo"]["proofkg_process_reward"] = "<ARM>"
    if left != right:
        raise SystemExit("paired configs differ outside output_dir/proofkg_process_reward")

    ot, pt = outcome.training, process.training
    for label, training, expected_process in (("PPO-O", ot, False), ("PPO-K", pt, True)):
        ppo = training.ppo
        expected = {
            "total_ppo_steps": 1200,
            "batch_size": 4,
            "mini_batch_size": 1,
            "ppo_epochs": 2,
            "rollouts_per_prompt": 4,
            "proofkg_outcome_only_reward": True,
            "proofkg_process_reward": expected_process,
            "proofkg_process_version": "v2_1",
            "proofkg_process_weight": 0.20,
            "proofkg_f1_weight": 0.10,
            "proofkg_dynamic_validity": True,
            "proofkg_require_all_eligible": True,
            "pure_em_reward": False,
            "kl_coef": 0.25,
            "target_kl": 8.0,
            "value_head_init": "zero",
            "value_head_dropout": 0.0,
            "sft_replay_ratio": 0.10,
            "sft_anchor_weight": 0.10,
            "log_with": "tensorboard",
        }
        for key, wanted in expected.items():
            actual = getattr(ppo, key)
            if actual != wanted:
                raise SystemExit(f"{label} {key}={actual!r}, expected {wanted!r}")
        if training.split is not None or not training.split_allow_none:
            raise SystemExit(f"{label} must use the curated whole-file split contract")

    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    if data_report.get("status") != "COMPLETE_NOT_TRAINED" or data_report.get("n_train_qids") != 208:
        raise SystemExit("hard curriculum data report is not complete n=208")
    input_paths = {
        name: Path(value["path"])
        for name, value in data_report["outputs"].items()
    }
    for name, path in input_paths.items():
        if _sha256(path) != data_report["outputs"][name]["sha256"]:
            raise SystemExit(f"materialized input hash mismatch: {name}")
    if Path(ot.silver_path) != input_paths["silver_train"]:
        raise SystemExit("config silver path is not the materialized 208-qid asset")
    if Path(ot.question_kg_records_path) != input_paths["question_kg_records"]:
        raise SystemExit("config question-KG path mismatch")
    if Path(ot.rollout_sampling_weights_path) != input_paths["sampling_weights"]:
        raise SystemExit("config sampling weights path mismatch")

    trajectories = [
        row for row in SilverDatasetReader(input_paths["silver_train"]).accepted()
        if str(row.metadata.get("gold_answer") or "").strip()
    ]
    weights_rows = _read_jsonl(input_paths["sampling_weights"])
    weights_by_qid = {str(row["qid"]): row for row in weights_rows}
    if len(trajectories) != 208 or len(weights_by_qid) != 208:
        raise SystemExit("paired sampling population is not exactly 208 unique qids")
    weights = [float(weights_by_qid[row.qid]["sampling_probability"]) for row in trajectories]
    rng = torch.Generator().manual_seed(int(ot.seed))
    replay_credit = 0.0
    schedule: list[dict[str, Any]] = []
    for start in range(0, int(ot.ppo.total_ppo_steps), int(ot.ppo.batch_size)):
        indices = _sample_rollout_indices(
            len(trajectories), int(ot.ppo.batch_size), int(ot.ppo.rollouts_per_prompt),
            rng, sampling_weights=weights,
        )
        for offset, index in enumerate(indices, start=1):
            traj = trajectories[index]
            schedule.append({
                "rollout_index": start + offset,
                "dataset": traj.dataset,
                "qid": traj.qid,
                "stratum": str(weights_by_qid[traj.qid]["stratum"]),
            })
        due, replay_credit = _advance_replay_credit(
            replay_credit,
            batch_size=int(ot.ppo.batch_size),
            replay_ratio=float(ot.ppo.sft_replay_ratio),
        )
        if due:
            torch.randint(0, 2000, (due,), generator=rng)
    if len(schedule) != 1200:
        raise SystemExit("expected 1200 scheduled trajectories")
    for start in range(0, len(schedule), 4):
        if len({row["qid"] for row in schedule[start:start + 4]}) != 1:
            raise SystemExit("K=4 same-prompt grouping failed")

    args.output_dir.mkdir(parents=True)
    schedule_path = args.output_dir / "expected_paired_rollout_schedule.jsonl"
    _write_jsonl(schedule_path, schedule)
    shared_code = [
        Path("kgproweight/training/phase3_ppo.py"),
        Path("kgproweight/training/reward_function.py"),
        Path("kgproweight/reward/proofkg_process_v2.py"),
        Path("kgproweight/kg/training_question_kg.py"),
        Path("kgproweight/config/schemas.py"),
        Path("scripts/train/phase3_ppo.py"),
        Path("scripts/prepare/materialize_2wiki_hard_curriculum_ppo.py"),
        Path("scripts/prepare/preflight_2wiki_hard_curriculum_ppo.py"),
    ]
    experiments = {
        "outcome": {
            "id": "PROOFKG-2WIKI-HARD-V1-PPO-O-1200-SEED42",
            "config": args.outcome_config,
            "tensorboard_dir": "/root/tf-logs/PROOFKG-2WIKI-HARD-V1-PPO-O-1200-SEED42",
        },
        "process": {
            "id": "PROOFKG-2WIKI-HARD-V1-PPO-K-V2.1-1200-SEED42",
            "config": args.process_config,
            "tensorboard_dir": "/root/tf-logs/PROOFKG-2WIKI-HARD-V1-PPO-K-V2.1-1200-SEED42",
        },
    }
    common = {
        "status": "CONFIGURED_NOT_STARTED",
        "data_report": {"path": str(args.data_report), "sha256": _sha256(args.data_report)},
        "inputs": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in input_paths.items()},
        "expected_schedule": {"path": str(schedule_path), "sha256": _sha256(schedule_path)},
        "expected_schedule_counts": {
            "recovery": sum(row["stratum"] == "recovery" for row in schedule),
            "stability": sum(row["stratum"] == "stability" for row in schedule),
            "unique_qids": len({row["qid"] for row in schedule}),
        },
        "code": {str(path): _sha256(path) for path in shared_code},
        "sft_adapter": {
            "path": str(Path(ot.sft_checkpoint) / "adapter_model.safetensors"),
            "sha256": _sha256(Path(ot.sft_checkpoint) / "adapter_model.safetensors"),
        },
        "alpha_gate": {"path": str(ot.alpha_gate_path), "sha256": _sha256(Path(ot.alpha_gate_path))},
        "single_research_variable": "proofkg_process_reward false (PPO-O) vs true (PPO-K)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    for arm, spec in experiments.items():
        lock = {
            "schema_version": "proofkg-hard-curriculum-paired-ppo-lock-1",
            "experiment_id": spec["id"],
            "arm": arm,
            "config": {"path": str(spec["config"]), "sha256": _sha256(spec["config"])},
            "tensorboard_dir": spec["tensorboard_dir"],
            **common,
        }
        lock_path = Path(str(spec["config"]) + args.lock_suffix)
        if lock_path.exists():
            raise SystemExit(f"refusing to overwrite lock: {lock_path}")
        lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "schema_version": "proofkg-hard-curriculum-paired-ppo-config-report-1",
        **common,
        "experiments": {
            arm: {
                **{key: str(value) if isinstance(value, Path) else value for key, value in spec.items()},
                "config_sha256": _sha256(spec["config"]),
                "lock_sha256": _sha256(Path(str(spec["config"]) + args.lock_suffix)),
            }
            for arm, spec in experiments.items()
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.output_dir, status=report["status"], extra={
        "experiment_ids": [spec["id"] for spec in experiments.values()],
        "report_sha256": _sha256(report_path),
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
