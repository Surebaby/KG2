#!/usr/bin/env python
"""Finalize the automatic-ProofKG PPO config and bind frozen inputs by hash."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from kgproweight.config import ProjectConfig, load_config
from kgproweight.utils.logging import artifact_identity


def _verify_artifact(identity: dict[str, Any], expected_path: Path) -> None:
    actual = artifact_identity(expected_path)
    if not actual.get("exists") or actual.get("md5") != identity.get("md5"):
        raise SystemExit(f"frozen artifact missing or changed: {expected_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--materialized_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment_id", required=True)
    args = parser.parse_args()

    template = Path(args.template).resolve()
    materialized_dir = Path(args.materialized_dir).resolve()
    output = Path(args.output).resolve()
    lock_path = Path(str(output) + ".lock.json")
    if output.exists() or lock_path.exists():
        raise SystemExit(f"refusing to overwrite finalized config or lock: {output}")

    report_path = materialized_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "COMPLETE_NOT_TRAINED":
        raise SystemExit(f"materialization is not frozen for training: {report.get('status')}")
    gates = report.get("materialization_gates") or {}
    if not gates or not all(bool(value.get("passed")) for value in gates.values()):
        raise SystemExit("one or more materialization gates did not pass")
    silver = materialized_dir / "silver_train.jsonl"
    question_kg = materialized_dir / "question_kg_records.jsonl"
    _verify_artifact(report["outputs"]["silver_train"], silver)
    _verify_artifact(report["outputs"]["question_kg_records"], question_kg)

    # Validate the fully inherited template before converting it into a runnable config.
    cfg = load_config(template, validate=ProjectConfig)
    training = cfg.training
    ppo = training.ppo
    expected = {
        "silver_path": str(silver.relative_to(Path.cwd())),
        "question_kg_records_path": str(question_kg.relative_to(Path.cwd())),
        "output_dir": f"outputs/{args.experiment_id}",
        "total_ppo_steps": 600,
        "rollouts_per_prompt": 4,
    }
    actual = {
        "silver_path": training.silver_path,
        "question_kg_records_path": training.question_kg_records_path,
        "output_dir": training.output_dir,
        "total_ppo_steps": ppo.total_ppo_steps,
        "rollouts_per_prompt": ppo.rollouts_per_prompt,
    }
    if actual != expected:
        raise SystemExit(f"template identity/protocol mismatch: expected={expected}, actual={actual}")
    if int(report["counts"]["complete_automatic_proofs"]) < int(ppo.total_ppo_steps):
        raise SystemExit("not enough complete automatic proofs for the requested smoke")

    body = template.read_text(encoding="utf-8")
    marker = "# NOT RUNNABLE UNTIL THE TWO VERSIONED INPUTS BELOW ARE MATERIALISED."
    if marker not in body:
        raise SystemExit("template finalization marker is missing")
    body = body.replace(
        marker,
        "# FINALIZED: frozen automatic-ProofKG inputs are bound by the adjacent .lock.json.",
        1,
    )
    header = (
        f"# experiment_id: {args.experiment_id}\n"
        f"# materialization_report_md5: {artifact_identity(report_path)['md5']}\n"
        f"# silver_train_md5: {artifact_identity(silver)['md5']}\n"
        f"# question_kg_records_md5: {artifact_identity(question_kg)['md5']}\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(header + body)

    lock = {
        "schema_version": "automatic-proofkg-ppo-config-lock-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "status": "FINALIZED_NOT_TRAINED",
        "config": artifact_identity(output),
        "template": artifact_identity(template),
        "materialization_report": artifact_identity(report_path),
        "silver_train": artifact_identity(silver),
        "question_kg_records": artifact_identity(question_kg),
        "protocol": {
            "total_ppo_steps": int(ppo.total_ppo_steps),
            "rollouts_per_prompt": int(ppo.rollouts_per_prompt),
            "batch_size": int(ppo.batch_size),
            "ppo_epochs": int(ppo.ppo_epochs),
            "sft_replay_ratio": float(ppo.sft_replay_ratio),
            "kl_coef": float(ppo.kl_coef),
            "target_kl": float(ppo.target_kl),
            "proofkg_process_reward": bool(ppo.proofkg_process_reward),
            "proofkg_require_all_eligible": bool(ppo.proofkg_require_all_eligible),
        },
        "materialization_gates": gates,
    }
    with lock_path.open("x", encoding="utf-8") as handle:
        json.dump(lock, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
