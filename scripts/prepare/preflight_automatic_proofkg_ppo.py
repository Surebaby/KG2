#!/usr/bin/env python
"""Fail-fast preflight for the finalized automatic-ProofKG PPO smoke."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

from kgproweight.config import ProjectConfig, load_config
from kgproweight.kg.question_kg import question_sha256
from kgproweight.reward.proofkg_process import is_automatic_proofkg
from kgproweight.utils.logging import artifact_identity


TESTS = (
    "tests/test_proofkg_production_reward.py",
    "tests/test_ppo_rollout_schedule.py",
    "tests/test_phase3_ppo_config_forwarding.py",
    "tests/test_training_question_kg.py",
    "tests/test_ppo_sft_replay.py",
    "tests/test_ppo_explicit_reference.py",
    "tests/test_run_preflight_manifest.py",
)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (Path.cwd() / value).resolve()


def _same_artifact(expected: dict[str, Any], path: Path) -> bool:
    actual = artifact_identity(path)
    return bool(actual.get("exists") and actual.get("md5") == expected.get("md5"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--run_tests", action="store_true")
    parser.add_argument("--report_path")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    lock_path = Path(args.lock).resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    cfg = load_config(config_path, validate=ProjectConfig)
    training = cfg.training
    ppo = training.ppo
    checks: dict[str, bool] = {}

    checks["finalized_config_hash"] = _same_artifact(lock["config"], config_path)
    silver_path = _resolve(str(training.silver_path))
    question_kg_path = _resolve(str(training.question_kg_records_path))
    # Lock paths record the machine where finalization happened.  Resolve the
    # versioned report from the runnable config so the same content hashes can
    # be verified after rsync to a different project root.
    report_path = silver_path.parent / "report.json"
    checks["materialization_report_hash"] = _same_artifact(
        lock["materialization_report"], report_path
    )
    checks["silver_hash"] = _same_artifact(lock["silver_train"], silver_path)
    checks["question_kg_hash"] = _same_artifact(
        lock["question_kg_records"], question_kg_path
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks["materialization_gates"] = bool(
        report.get("materialization_gates")
        and all(v.get("passed") for v in report["materialization_gates"].values())
    )

    expected_protocol = lock["protocol"]
    actual_protocol = {
        "total_ppo_steps": int(ppo.total_ppo_steps),
        "rollouts_per_prompt": int(ppo.rollouts_per_prompt),
        "batch_size": int(ppo.batch_size),
        "ppo_epochs": int(ppo.ppo_epochs),
        "sft_replay_ratio": float(ppo.sft_replay_ratio),
        "kl_coef": float(ppo.kl_coef),
        "target_kl": float(ppo.target_kl),
        "proofkg_process_reward": bool(ppo.proofkg_process_reward),
        "proofkg_require_all_eligible": bool(ppo.proofkg_require_all_eligible),
    }
    checks["config_protocol"] = actual_protocol == expected_protocol
    checks["k4_batch_grouping"] = (
        ppo.rollouts_per_prompt == 4
        and ppo.batch_size % ppo.rollouts_per_prompt == 0
        and ppo.total_ppo_steps % ppo.rollouts_per_prompt == 0
    )
    checks["reward_protocol"] = bool(
        ppo.proofkg_process_reward
        and ppo.proofkg_dynamic_validity
        and ppo.proofkg_require_all_eligible
        and float(ppo.proofkg_process_weight) == 1.0
        and float(ppo.proofkg_f1_weight) == 0.10
    )
    replay_path = _resolve(str(training.sft_replay_silver_path))
    checks["replay_isolated"] = (
        float(ppo.sft_replay_ratio) == 0.10
        and replay_path != silver_path
        and replay_path.is_file()
        and str(training.sft_replay_split) == "train"
    )
    checks["split_and_input_protocol"] = bool(
        training.split is None
        and training.split_allow_none
        and training.question_kg_index_path is None
        and training.passage_overrides_path is None
        and training.rollout_schedule_path is None
        and float(training.min_question_kg_record_coverage) == 1.0
        and training.require_nonempty_question_kg_records
    )

    records = {str(row["question_key"]): row for row in _read_jsonl(question_kg_path)}
    silver_rows = list(_read_jsonl(silver_path))
    eligible = 0
    joined = 0
    no_gold_steps = 0
    for row in silver_rows:
        key = f"{row.get('dataset')}::{row.get('qid')}"
        record = records.get(key)
        if record is None:
            continue
        joined += int(
            question_sha256(str(row.get("question") or ""))
            == str(record.get("question_sha256") or "")
            == question_sha256(str(record.get("question") or ""))
            and list(record.get("kg_subgraph") or []) == list(row.get("kg_subgraph") or [])
        )
        no_gold_steps += int(not row.get("steps") and not row.get("teacher_output"))
        eligible += int(is_automatic_proofkg(
            {
                "query_plan": record.get("query_plan") or {},
                "provenance": record.get("provenance") or {},
            },
            record.get("kg_subgraph") or [],
        ))
    n = len(silver_rows)
    checks["eligible_rollout_supply"] = n >= 600 and eligible == n == len(records)
    checks["identity_hash_join"] = joined == n
    checks["no_gold_process_trace"] = no_gold_steps == n
    checks["local_model_inputs"] = bool(
        _resolve(str(training.sft_checkpoint)).is_dir()
        and _resolve(str(training.alpha_gate_path)).is_file()
    )
    checks["output_is_new"] = not _resolve(str(training.output_dir)).exists()

    if args.run_tests:
        completed = subprocess.run(
            [str(Path(__import__("sys").executable)), "-m", "pytest", "-q", *TESTS],
            check=False,
        )
        checks["targeted_tests"] = completed.returncode == 0
    failed = sorted(key for key, passed in checks.items() if not passed)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": lock["experiment_id"],
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "counts": {"rollout_rows": n, "eligible": eligible, "identity_joined": joined},
        "failed": failed,
        "remote_checks_deferred_to_launcher": ["remote_files", "GPU", "free_disk"],
    }
    if args.report_path:
        out = Path(args.report_path).resolve()
        if out.exists():
            raise SystemExit(f"refusing to overwrite preflight report: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(f"preflight failed: {failed}")


if __name__ == "__main__":
    main()
