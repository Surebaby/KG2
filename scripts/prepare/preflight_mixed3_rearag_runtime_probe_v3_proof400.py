#!/usr/bin/env python
"""CPU-only fail-closed preflight for the Proof400 runtime probe v3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from kgproweight.config import ProjectConfig, load_config
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.kg.question_kg import question_key
from kgproweight.kg.training_question_kg import apply_training_question_kg, read_question_kg_records
from kgproweight.reward.proofkg_process import is_identity_safe_automatic_proofkg
from kgproweight.training.phase3_ppo import (
    Phase3PPOConfig, _load_fixed_rollout_schedule, _load_rollout_sampling_weights,
    _prepare_prompts, _validate_mixed_reward_config, _validate_v21_execution_preflight,
)
from kgproweight.utils.logging import dump_manifest
from kgproweight.utils.paths import model_path
from scripts.prepare.materialize_mixed3_rearag_runtime_probe_v1 import file_ref
from scripts.prepare.materialize_mixed3_rearag_runtime_probe_v3_proof400 import ARM_SPECS_V3
from scripts.prepare.preflight_mixed3_rearag_runtime_probe_v1 import (
    EXPECTED_CONFIG_DIFF, _MemoryReader, _PromptTokenizer, config_diff,
)
from scripts.prepare.preflight_mixed3_rearag_runtime_probe_v2 import (
    EXPECTED_RUNTIME_DIFF, target_absence, verify_ref,
)
from scripts.prepare.resolve_phase3_ppo_runtime_config import resolve_phase3_ppo_runtime_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v3_proof400_seed42_freeze/protocol.json"
DEFAULT_REPORT_DIR = ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v3_proof400_seed42_local_preflight"
TESTS = [
    "tests/test_mixed3_rearag_runtime_probe_v3_proof400.py",
    "tests/test_mixed_ppo_three_dataset_v2_proof400.py",
    "tests/test_mixed_ppo_reward.py", "tests/test_ppo_rollout_schedule.py",
    "tests/test_phase3_ppo_config_forwarding.py", "tests/test_ppo_sft_replay.py",
    "tests/test_ppo_explicit_reference.py", "tests/test_ppo_diagnostics.py",
]


def validate_arm(arm: str, spec: dict[str, Any]) -> dict[str, Any]:
    probe_doc = load_config(ROOT / spec["config"], validate=ProjectConfig)
    formal_doc = load_config(ROOT / spec["formal_config"], validate=ProjectConfig)
    doc_diff = config_diff(formal_doc, probe_doc)
    if doc_diff != EXPECTED_CONFIG_DIFF:
        raise ValueError(f"{arm}: ProjectConfig drift={doc_diff}")
    probe_runtime = resolve_phase3_ppo_runtime_config(ROOT / spec["config"])
    formal_runtime = resolve_phase3_ppo_runtime_config(ROOT / spec["formal_config"])
    runtime_diff = sorted(
        key for key in set(probe_runtime) | set(formal_runtime)
        if probe_runtime.get(key) != formal_runtime.get(key)
    )
    if runtime_diff != EXPECTED_RUNTIME_DIFF:
        raise ValueError(f"{arm}: real CLI drift={runtime_diff}")
    cfg = Phase3PPOConfig(**probe_runtime)
    _validate_mixed_reward_config(cfg)
    expected = bool(spec["expected_eligible"])
    if cfg.alpha_gate_path is not None or cfg.alpha_override is not None:
        raise ValueError(f"{arm}: alpha must be null on mixed route")
    if cfg.proofkg_process_reward is not expected:
        raise ValueError(f"{arm}: process switch mismatch")
    if (cfg.total_steps, cfg.batch_size, cfg.rollouts_per_prompt, cfg.save_every_steps) != (4, 4, 4, 4):
        raise ValueError(f"{arm}: probe is not one saved K4 update")
    if Path(cfg.output_dir).name != spec["experiment_id"]:
        raise ValueError(f"{arm}: output/Experiment ID mismatch")
    rows = list(SilverDatasetReader(cfg.silver_path, split=None).accepted())
    if len(rows) != 1:
        raise ValueError(f"{arm}: expected one source prompt")
    join = apply_training_question_kg(
        rows, read_question_kg_records(cfg.question_kg_records_path),
        min_coverage=1.0, require_nonempty=False,
    ).to_dict()
    row = rows[0]
    eligible = is_identity_safe_automatic_proofkg(
        row.metadata.get("question_kg_runtime") or {}, row.kg_subgraph,
        dataset=row.dataset, qid=row.qid,
    )
    if eligible is not expected:
        raise ValueError(f"{arm}: production eligibility mismatch")
    execution = _validate_v21_execution_preflight(rows) if expected else None
    weights, sampling = _load_rollout_sampling_weights(cfg.rollout_sampling_weights_path, rows)
    indices, schedule = _load_fixed_rollout_schedule(
        cfg.fixed_rollout_schedule_path, rows, total_steps=4, rollouts_per_prompt=4,
        sampling_records=sampling,
    )
    if weights != [1.0] or indices != [0, 0, 0, 0] or len(schedule) != 4:
        raise ValueError(f"{arm}: fixed K4 loader mismatch")
    if any(bool(item.get("process_reward_eligible")) is not expected for item in schedule):
        raise ValueError(f"{arm}: schedule eligibility mismatch")
    prompts = _prepare_prompts(_MemoryReader(rows), _PromptTokenizer(), cfg)
    if len(prompts) != 1:
        raise ValueError(f"{arm}: prompt builder mismatch")
    identity = question_key(str(row.dataset), str(row.qid))
    if (prompts[0]["spec"].metadata.get("question_kg_runtime") or {}).get("question_key") != identity:
        raise ValueError(f"{arm}: RewardSpec identity lost")
    return {
        "experiment_id": spec["experiment_id"], "identity": identity,
        "project_config_diff": doc_diff, "real_cli_runtime_diff": runtime_diff,
        "alpha_gate_path": cfg.alpha_gate_path, "alpha_override": cfg.alpha_override,
        "mixed_outcome_reward": cfg.mixed_outcome_reward,
        "mixed_text_reward": cfg.mixed_text_reward,
        "text_reward_backend": cfg.text_reward_backend,
        "proofkg_process_reward": cfg.proofkg_process_reward,
        "process_reward_eligible": eligible, "v2_1_execution": execution,
        "question_kg_join": join, "kg_triples": len(row.kg_subgraph),
        "fixed_indices": indices, "targets": target_absence(spec),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--run_tests", action="store_true")
    args = parser.parse_args()
    protocol_path, report_dir = args.protocol.resolve(), args.report_dir.resolve()
    if report_dir.exists():
        raise FileExistsError(f"append-only preflight exists: {report_dir}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_NOT_RUN" or protocol.get("counts", {}).get("scheduled_trajectories_total") != 8:
        raise ValueError("wrong v3 Proof400 protocol/budget")
    if protocol.get("scientific_boundary", {}).get("training_started") is not False:
        raise ValueError("protocol training boundary missing")
    for section in ("inputs", "runtime_code_closure", "config_dependency_closure"):
        for label, identity in protocol.get(section, {}).items():
            verify_ref(identity, f"{section}:{label}")
    for arm, refs in protocol["outputs"].items():
        for label, identity in refs.items():
            verify_ref(identity, f"output:{arm}:{label}")
    arms = {arm: validate_arm(arm, spec) for arm, spec in ARM_SPECS_V3.items()}

    models = {}
    for logical in ("llama3-8B-instruct", "rearag"):
        path = Path(model_path(logical)).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"local model missing: {logical} -> {path}")
        models[logical] = str(path.resolve())
    adapter = ROOT / "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / filename).is_file():
            raise FileNotFoundError(adapter / filename)

    tests = None
    if args.run_tests:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *TESTS], cwd=ROOT,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        tests = {"returncode": proc.returncode, "output_tail": proc.stdout[-8000:]}
        if proc.returncode:
            raise RuntimeError("v3 probe tests failed")
    report = {
        "schema_version": "mixed3-rearag-runtime-wiring-probe-preflight-v3-proof400",
        "experiment_id": "MIXED3-REARAG-RUNTIME-WIRING-PROBE-V3-PROOF400-SEED42-PREFLIGHT",
        "status": "PASS_CPU_PREFLIGHT_NOT_RUN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": file_ref(protocol_path), "cuda_allocated": False, "training_started": False,
        "checks": {
            "arms": arms, "total_trajectories": 8, "models": models,
            "sft_adapter": str(adapter.relative_to(ROOT)),
            "runtime_code_files_hashed": len(protocol["runtime_code_closure"]),
            "config_dependency_files_hashed": len(protocol["config_dependency_closure"]),
            "formal_pair_unchanged": True, "formal_data_unchanged": True,
            "postflight_contract": protocol["postflight_contract"],
        },
        "tests": tests,
        "scientific_boundary": "CPU wiring only; no CUDA, training, effect, or convergence evidence.",
    }
    report_dir.mkdir(parents=True, exist_ok=False)
    path = report_dir / "preflight.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(report_dir, status=report["status"], extra={
        "phase": "mixed3_rearag_runtime_probe_v3_proof400_preflight",
        "experiment_id": report["experiment_id"], "cuda_allocated": False,
        "training_started": False, "preflight_sha256": file_ref(path)["sha256"],
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
