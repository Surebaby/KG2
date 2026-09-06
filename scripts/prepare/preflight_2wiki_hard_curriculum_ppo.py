#!/usr/bin/env python
"""CPU-only preflight for one frozen hard-curriculum PPO arm."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

from kgproweight.config import ProjectConfig, load_config
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.kg.training_question_kg import apply_training_question_kg, read_question_kg_records
from kgproweight.reward.proofkg_process import is_automatic_proofkg
from kgproweight.training.phase3_ppo import (
    _advance_replay_credit,
    _load_rollout_sampling_weights,
    _sample_rollout_indices,
)


TESTS = [
    "tests/test_hard_curriculum_ppo_plumbing.py",
    "tests/test_training_question_kg.py",
    "tests/test_phase3_ppo_config_forwarding.py",
    "tests/test_proofkg_production_reward.py",
    "tests/test_ppo_rollout_schedule.py",
    "tests/test_ppo_sft_replay.py",
    "tests/test_ppo_explicit_reference.py",
    "tests/test_ppo_diagnostics.py",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--report_path", type=Path, required=True)
    parser.add_argument("--run_tests", action="store_true")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    failures: list[str] = []
    if lock.get("status") != "CONFIGURED_NOT_STARTED":
        failures.append("lock status is not CONFIGURED_NOT_STARTED")
    if lock.get("config", {}).get("sha256") != _sha256(args.config):
        failures.append("config SHA256 does not match lock")
    for path_text, expected in (lock.get("code") or {}).items():
        path = Path(path_text)
        if not path.is_file() or _sha256(path) != expected:
            failures.append(f"code hash mismatch: {path}")
    for name, spec in (lock.get("inputs") or {}).items():
        path = Path(spec["path"])
        if not path.is_file() or _sha256(path) != spec["sha256"]:
            failures.append(f"input hash mismatch: {name}")

    cfg = load_config(args.config, validate=ProjectConfig).training
    ppo = cfg.ppo
    if ppo.total_ppo_steps != 1200 or ppo.batch_size != 4 or ppo.rollouts_per_prompt != 4:
        failures.append("runtime budget/grouping is not 1200 trajectories, batch4, K4")
    if not ppo.proofkg_outcome_only_reward or ppo.proofkg_process_version != "v2_1":
        failures.append("paired automatic-ProofKG reward contract is not active")
    if ppo.proofkg_process_reward != (lock.get("arm") == "process"):
        failures.append("process flag disagrees with arm lock")
    if ppo.proofkg_process_weight != 0.2 or ppo.proofkg_f1_weight != 0.1:
        failures.append("reward weights are not frozen at process=.2, F1=.1")
    if ppo.kl_coef != 0.25 or ppo.target_kl != 8.0:
        failures.append("KL controls differ from frozen stability settings")
    if ppo.sft_replay_ratio != 0.1 or ppo.sft_anchor_weight != 0.1:
        failures.append("10% replay contract is not active")
    if ppo.log_with != "tensorboard" or not str(lock.get("tensorboard_dir") or "").startswith("/root/tf-logs/"):
        failures.append("AutoDL TensorBoard contract is missing")

    reader = SilverDatasetReader(cfg.silver_path, split=None)
    trajectories = [row for row in reader.accepted() if str(row.metadata.get("gold_answer") or "").strip()]
    if len(trajectories) != 208 or len({(row.dataset, row.qid) for row in trajectories}) != 208:
        failures.append("training population is not 208 unique accepted gold-bearing qids")
    records = read_question_kg_records(cfg.question_kg_records_path)
    try:
        stats = apply_training_question_kg(
            trajectories, records, min_coverage=1.0, require_nonempty=True,
        ).to_dict()
    except Exception as exc:  # fail report remains machine readable
        stats = {"error": f"{type(exc).__name__}: {exc}"}
        failures.append("question-KG identity join failed")
    eligible = 0
    execution_ready = 0
    for row in trajectories:
        runtime = row.metadata.get("question_kg_runtime") or {}
        eligible += int(is_automatic_proofkg(runtime, row.kg_subgraph))
        planned = list((runtime.get("query_plan") or {}).get("hops") or [])
        executed = list((runtime.get("execution") or {}).get("hops") or [])
        execution_ready += int(bool(planned) and len(executed) >= len(planned))
    if eligible != 208 or execution_ready != 208:
        failures.append(f"ProofKG eligibility/execution is {eligible}/{execution_ready}, expected 208/208")

    weights, rows_by_key = _load_rollout_sampling_weights(
        cfg.rollout_sampling_weights_path, trajectories,
    )
    recovery_mass = sum(
        float(row["sampling_probability"])
        for row in rows_by_key.values() if row.get("stratum") == "recovery"
    )
    stability_mass = sum(
        float(row["sampling_probability"])
        for row in rows_by_key.values() if row.get("stratum") == "stability"
    )
    if abs(recovery_mass - 0.5) > 1e-9 or abs(stability_mass - 0.5) > 1e-9:
        failures.append("hard curriculum sampling mass is not 50/50")

    expected_schedule = _read_jsonl(Path(lock["expected_schedule"]["path"]))
    if _sha256(Path(lock["expected_schedule"]["path"])) != lock["expected_schedule"]["sha256"]:
        failures.append("expected paired schedule hash mismatch")
    rng = torch.Generator().manual_seed(int(cfg.seed))
    replay_credit = 0.0
    actual: list[str] = []
    for _ in range(0, int(ppo.total_ppo_steps), int(ppo.batch_size)):
        indices = _sample_rollout_indices(
            len(trajectories), int(ppo.batch_size), int(ppo.rollouts_per_prompt), rng,
            sampling_weights=weights,
        )
        actual.extend(trajectories[index].qid for index in indices)
        due, replay_credit = _advance_replay_credit(
            replay_credit, batch_size=int(ppo.batch_size), replay_ratio=float(ppo.sft_replay_ratio),
        )
        if due:
            torch.randint(0, 2000, (due,), generator=rng)
    if actual != [str(row["qid"]) for row in expected_schedule]:
        failures.append("runtime sampler does not reproduce the frozen paired qid schedule")

    test_result = None
    if args.run_tests:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *TESTS], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        test_result = {"returncode": proc.returncode, "output_tail": proc.stdout[-4000:]}
        if proc.returncode:
            failures.append("regression tests failed")

    report = {
        "schema_version": "proofkg-hard-curriculum-ppo-preflight-1",
        "experiment_id": lock.get("experiment_id"),
        "status": "PASS_NO_GPU_PREFLIGHT" if not failures else "FAIL_NO_GPU_PREFLIGHT",
        "cuda_required": False,
        "config": {"path": str(args.config), "sha256": _sha256(args.config)},
        "lock": {"path": str(args.lock), "sha256": _sha256(args.lock)},
        "checks": {
            "training_qids": len(trajectories),
            "question_kg": stats,
            "proofkg_eligible": eligible,
            "v2_execution_ready": execution_ready,
            "sampling_mass": {"recovery": recovery_mass, "stability": stability_mass},
            "schedule_rows": len(actual),
            "schedule_exact_match": actual == [str(row["qid"]) for row in expected_schedule],
            "tensorboard_dir": lock.get("tensorboard_dir"),
        },
        "tests": test_result,
        "failures": failures,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
