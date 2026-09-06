#!/usr/bin/env python
"""Append-only real-CLI comparison for the current Proof400 PPO pair."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_mixed_ppo_three_dataset_v1 import sha256_file
from scripts.prepare.resolve_phase3_ppo_runtime_config import resolve_phase3_ppo_runtime_config


CONTROL = Path("configs/training/phase3_ppo_mixed3_rearag_v2_proof400_text7200_seed42.yaml")
TREATMENT = Path("configs/training/phase3_ppo_mixed3_rearag_v2_proof400_text_kg_v2_1_7200_seed42.yaml")
DATA = Path("data/silver_data/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42")
OLD = Path("outputs/audits/mixed3_rearag_ppo_pair_proof400_7200_seed42_config_comparison_v1/report.json")
OUT = Path("outputs/audits/mixed3_rearag_ppo_pair_proof400_7200_seed42_config_comparison_v2")
STATUS = "PASS_CONFIG_ONLY_NOT_GPU_PROBED_NOT_TRAINED"


def ref(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite comparison: {OUT}")
    control = resolve_phase3_ppo_runtime_config(CONTROL)
    treatment = resolve_phase3_ppo_runtime_config(TREATMENT)
    differences = {
        key: [control.get(key), treatment.get(key)]
        for key in sorted(set(control) | set(treatment))
        if control.get(key) != treatment.get(key)
    }
    expected = {"output_dir", "proofkg_process_reward"}
    if set(differences) != expected:
        raise ValueError(f"unexpected real-CLI pair differences: {differences}")
    report = json.loads((DATA / "report.json").read_text(encoding="utf-8"))
    if not all(report.get("gates", {}).values()):
        raise ValueError("materialization gates are not all true")
    expected_paths = {
        "silver_path": DATA / "silver_train.jsonl",
        "question_kg_records_path": DATA / "question_kg_records.jsonl",
        "rollout_sampling_weights_path": DATA / "sampling_weights.jsonl",
        "fixed_rollout_schedule_path": DATA / "fixed_rollout_schedule.jsonl",
    }
    for key, path in expected_paths.items():
        if control.get(key) != str(path) or treatment.get(key) != str(path) or not path.is_file():
            raise ValueError(f"runtime data path mismatch: {key}")
    gates = {
        "real_cli_pair_diff_exact": set(differences) == expected,
        "control_process_false": control["proofkg_process_reward"] is False,
        "treatment_process_true": treatment["proofkg_process_reward"] is True,
        "total_steps_7200": control["total_steps"] == treatment["total_steps"] == 7200,
        "k4": control["rollouts_per_prompt"] == treatment["rollouts_per_prompt"] == 4,
        "mixed_reward_enabled": all(
            cfg["mixed_outcome_reward"] and cfg["mixed_text_reward"] for cfg in (control, treatment)
        ),
        "old_alpha_disabled": control["alpha_gate_path"] is None and treatment["alpha_gate_path"] is None,
        "rearag_shared": control["text_reward_backend"] == treatment["text_reward_backend"] == "rearag",
        "data_gates_pass": all(report["gates"].values()),
        "output_dirs_absent": not Path(control["output_dir"]).exists() and not Path(treatment["output_dir"]).exists(),
    }
    if not all(gates.values()):
        raise ValueError(gates)

    payload = {
        "schema_version": "mixed3-rearag-proof400-config-comparison-2",
        "experiment_id": "MIXED3-REARAG-PROOF400-PAIRED-PPO-CONFIGS-7200-SEED42-V2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "supersession": {
            "superseded_report": ref(OLD),
            "superseded_fields": [
                "configs.proof400_control.sha256", "configs.proof400_treatment.sha256"
            ],
            "reason": "The config files received explicit status comments/control=false after v1 was written.",
            "v1_report_or_configs_overwritten": False,
        },
        "configs": {
            "control": ref(CONTROL), "treatment": ref(TREATMENT),
            "control_output_dir": control["output_dir"],
            "treatment_output_dir": treatment["output_dir"],
        },
        "real_cli": {
            "control_sha256": canonical_hash(control),
            "treatment_sha256": canonical_hash(treatment),
            "pair_differences": differences,
            "sole_scientific_variable": "proofkg_process_reward false -> true",
        },
        "data": {
            "materialization_report": ref(DATA / "report.json"),
            "files": {name: ref(DATA / name) for name in (
                "silver_train.jsonl", "question_kg_records.jsonl", "sampling_weights.jsonl",
                "prompt_groups.jsonl", "fixed_rollout_schedule.jsonl", "manifest.json",
            )},
            "unique_population": report["counts"]["unique_population"],
            "process_reward_eligible_unique": report["counts"]["process_reward_eligible_unique"],
            "scheduled_trajectories": report["counts"]["scheduled_trajectories"],
            "scheduled_process_eligible_trajectories": report["counts"]["scheduled_process_eligible_trajectories"],
        },
        "gates": gates,
        "boundary": {
            "config_and_data_pass_only": True,
            "gpu_probe_started": False,
            "training_started": False,
            "formal_pair_lock_created": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=False)
    path = OUT / "report.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(OUT, status=STATUS, extra={
        "phase": "mixed3_rearag_proof400_config_comparison",
        "experiment_id": payload["experiment_id"],
        "report_sha256": sha256_file(path),
    })
    print(json.dumps({"status": STATUS, "configs": payload["configs"],
                      "pair_differences": differences, "gates": gates}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
