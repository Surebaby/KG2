#!/usr/bin/env python
"""Freeze the exact seeded PPO smoke rollout qids and unique retrieval cohort."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Sequence, Tuple

import torch

from kgproweight.config import ProjectConfig, load_config
from kgproweight.data.silver_dataset import SilverDatasetReader, SilverTrajectory
from kgproweight.data.silver_split import SplitSpec
from kgproweight.training.phase3_ppo import _advance_replay_credit, _sample_rollout_indices
from kgproweight.utils.logging import dump_manifest


BUILDER_VERSION = "ppo-rollout-schedule-1"
SFT_REPLAY_POOL_SIZE = 2000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_schedule(
    samples: Sequence[SilverTrajectory],
    *,
    seed: int,
    batch_size: int,
    total_steps: int,
    replay_ratio: float,
    rollouts_per_prompt: int = 1,
    replay_pool_size: int = SFT_REPLAY_POOL_SIZE,
) -> Tuple[List[SilverTrajectory], int]:
    """Mirror Phase3 PPO's explore/replay RNG consumption without generation."""
    if not samples:
        raise ValueError("cannot build a rollout schedule from zero samples")
    if batch_size <= 0 or total_steps <= 0 or total_steps % batch_size:
        raise ValueError("total_steps must be positive and divisible by batch_size")
    generator = torch.Generator().manual_seed(seed)
    replay_credit = 0.0
    scheduled: List[SilverTrajectory] = []
    replay_draws = 0
    for _ in range(0, total_steps, batch_size):
        indices = _sample_rollout_indices(
            len(samples), batch_size, rollouts_per_prompt, generator
        )
        scheduled.extend(samples[index] for index in indices)
        due, replay_credit = _advance_replay_credit(
            replay_credit,
            batch_size=batch_size,
            replay_ratio=replay_ratio,
        )
        if due:
            # The training loop shares this generator with replay sampling.
            # Consume the same draws so every later explore batch is identical.
            torch.randint(0, replay_pool_size, (due,), generator=generator)
            replay_draws += due
    return scheduled, replay_draws


def _verify_sample_log(path: Path, schedule: Sequence[SilverTrajectory]) -> Dict[str, Any]:
    match = re.search(r"step_(\d+)\.txt$", path.name)
    if not match:
        raise ValueError("verification sample log must be named step_NNNNN.txt")
    step = int(match.group(1))
    actual = re.findall(
        r"^--- Sample \d+ qid=(\S+)", path.read_text(encoding="utf-8"), re.MULTILINE
    )
    if not actual:
        raise ValueError(f"no sample qids found in {path}")
    expected = [item.qid for item in schedule[max(0, step - len(actual)) : step]]
    if actual != expected:
        raise ValueError(
            f"schedule does not reproduce {path}: actual={actual}, expected={expected}"
        )
    return {"path": str(path), "sha256": _sha256(path), "step": step, "n": len(actual)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--verify_sample_log")
    parser.add_argument("--schedule_output", required=True)
    parser.add_argument("--cohort_output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    schedule_path = Path(args.schedule_output).resolve()
    cohort_path = Path(args.cohort_output).resolve()
    report_path = Path(args.report).resolve()
    run_dir = Path(args.run_dir).resolve()
    for path in (schedule_path, cohort_path, report_path, run_dir):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing path: {path}")

    config = load_config(config_path, validate=ProjectConfig)
    training = config.training
    ppo = training.ppo
    silver_path = Path(training.silver_path).resolve()
    reader = SilverDatasetReader(
        silver_path,
        split=training.split,
        split_spec=SplitSpec(
            val_ratio=training.val_ratio,
            test_ratio=training.test_ratio,
            seed=training.split_seed,
        ),
    )
    samples = [
        item
        for item in reader.accepted()
        if str(item.metadata.get("gold_answer") or "").strip()
    ]
    try:
        schedule, replay_draws = build_schedule(
            samples,
            seed=training.seed,
            batch_size=ppo.batch_size,
            total_steps=ppo.total_ppo_steps,
            replay_ratio=ppo.sft_replay_ratio,
            rollouts_per_prompt=ppo.rollouts_per_prompt,
        )
        verification = (
            _verify_sample_log(Path(args.verify_sample_log).resolve(), schedule)
            if args.verify_sample_log
            else None
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    schedule_rows = [
        {
            "rollout_index": index,
            "qid": item.qid,
            "question": item.question,
            "dataset": item.dataset,
        }
        for index, item in enumerate(schedule, start=1)
    ]
    unique: Dict[str, SilverTrajectory] = {}
    for item in schedule:
        unique.setdefault(item.qid, item)
    cohort_rows = [
        {"qid": item.qid, "question": item.question, "dataset": item.dataset}
        for item in unique.values()
    ]
    for path, rows in ((schedule_path, schedule_rows), (cohort_path, cohort_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    qids = [item.qid for item in schedule]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "status": "FROZEN_NOT_RETRIEVED",
        "builder_version": BUILDER_VERSION,
        "protocol": {
            "gold_fields_written": False,
            "seed": training.seed,
            "split": training.split,
            "split_seed": training.split_seed,
            "batch_size": ppo.batch_size,
            "total_steps": ppo.total_ppo_steps,
            "sft_replay_ratio": ppo.sft_replay_ratio,
            "rollouts_per_prompt": ppo.rollouts_per_prompt,
            "sft_replay_pool_size": SFT_REPLAY_POOL_SIZE,
            "rng": "torch.Generator CPU; shared explore/replay consumption mirrored",
        },
        "inputs": {
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "silver": str(silver_path),
            "silver_sha256": _sha256(silver_path),
            "silver_read_only": True,
        },
        "counts": {
            "accepted_train": len(reader.accepted()),
            "ppo_samples": len(samples),
            "scheduled_rollouts": len(schedule),
            "unique_qids": len(unique),
            "duplicate_rollouts": len(schedule) - len(unique),
            "replay_rng_draws": replay_draws,
        },
        "verification": verification,
        "outputs": {
            "schedule": str(schedule_path),
            "schedule_sha256": _sha256(schedule_path),
            "schedule_qid_sha256": hashlib.sha256("\n".join(qids).encode()).hexdigest(),
            "cohort": str(cohort_path),
            "cohort_sha256": _sha256(cohort_path),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        run_dir,
        extra={
            "experiment_id": args.experiment_id,
            "phase": "freeze_ppo_rollout_schedule",
            "builder_version": BUILDER_VERSION,
            "protocol": report["protocol"],
            "inputs": report["inputs"],
            "counts": report["counts"],
            "verification": verification,
            "outputs": report["outputs"],
        },
    )
    print(json.dumps({"counts": report["counts"], "outputs": report["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
