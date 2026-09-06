#!/usr/bin/env python
"""CPU-only, fail-closed preflight for the two K=4 runtime-wiring probes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

from kgproweight.config import ProjectConfig, load_config
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.kg.question_kg import question_key
from kgproweight.kg.training_question_kg import (
    apply_training_question_kg,
    read_question_kg_records,
)
from kgproweight.reward.proofkg_process import is_identity_safe_automatic_proofkg
from kgproweight.training.phase3_ppo import (
    Phase3PPOConfig,
    _load_fixed_rollout_schedule,
    _load_rollout_sampling_weights,
    _prepare_prompts,
    _validate_mixed_reward_config,
    _validate_v21_execution_preflight,
)
from kgproweight.utils.logging import dump_manifest
from kgproweight.utils.paths import model_path
from scripts.prepare.materialize_mixed3_rearag_runtime_probe_v1 import ARM_SPECS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v1_seed42_freeze/protocol.json"
)
DEFAULT_REPORT_DIR = (
    ROOT / "outputs/audits/mixed3_rearag_runtime_wiring_probe_v1_seed42_local_preflight"
)
TESTS = [
    "tests/test_mixed3_rearag_runtime_probe.py",
    "tests/test_ppo_rollout_schedule.py",
    "tests/test_mixed_ppo_reward.py",
    "tests/test_phase3_ppo_config_forwarding.py",
]


class _MemoryReader:
    def __init__(self, rows):
        self.rows = rows

    def accepted(self):
        yield from self.rows


class _PromptTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if tokenize is not False or add_generation_prompt is not True:
            raise ValueError("unexpected chat-template options")
        return "\n".join(str(row["content"]) for row in messages)

    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(len(str(text).split())))}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_ref(spec: dict[str, Any]) -> Path:
    path = Path(str(spec["path"]))
    return path if path.is_absolute() else ROOT / path


def verify_ref(spec: dict[str, Any], label: str) -> None:
    path = resolve_ref(spec)
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    if path.stat().st_size != int(spec["size_bytes"]):
        raise ValueError(f"{label}: size changed: {path}")
    if sha256_file(path) != str(spec["sha256"]):
        raise ValueError(f"{label}: SHA256 changed: {path}")


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(item, child))
        return result
    return {prefix: value}


def config_diff(formal: ProjectConfig, probe: ProjectConfig) -> list[str]:
    left, right = flatten(formal.model_dump()), flatten(probe.model_dump())
    return sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))


EXPECTED_CONFIG_DIFF = [
    "training.fixed_rollout_schedule_path",
    "training.output_dir",
    "training.ppo.save_every_steps",
    "training.ppo.total_ppo_steps",
    "training.question_kg_records_path",
    "training.rollout_sampling_weights_path",
    "training.silver_path",
]


def runtime_contract(doc: ProjectConfig) -> Phase3PPOConfig:
    training, ppo = doc.training, doc.training.ppo
    cfg = Phase3PPOConfig(
        silver_path=str(training.silver_path),
        output_dir=str(training.output_dir),
        base_model=str(training.base_model),
        sft_checkpoint=str(training.sft_checkpoint),
        sft_replay_silver_path=str(training.sft_replay_silver_path),
        sft_replay_split=training.sft_replay_split,
        alpha_gate_path=training.alpha_gate_path,
        text_reward_backend=str(doc.reward.text_reward_backend),
        dtype=str(training.dtype),
        seed=int(training.seed),
        learning_rate=float(ppo.learning_rate),
        batch_size=int(ppo.batch_size),
        mini_batch_size=int(ppo.mini_batch_size),
        ppo_epochs=int(ppo.ppo_epochs),
        total_steps=int(ppo.total_ppo_steps),
        save_every_steps=int(ppo.save_every_steps),
        outcome_weight=float(ppo.outcome_weight),
        text_reward_scale=float(ppo.text_reward_scale),
        center_text_reward=bool(ppo.center_text_reward),
        text_baseline_momentum=float(ppo.text_baseline_momentum),
        proofkg_process_reward=bool(ppo.proofkg_process_reward),
        proofkg_process_version=str(ppo.proofkg_process_version),
        proofkg_process_weight=float(ppo.proofkg_process_weight),
        proofkg_f1_weight=float(ppo.proofkg_f1_weight),
        proofkg_dynamic_validity=bool(ppo.proofkg_dynamic_validity),
        mixed_outcome_reward=bool(ppo.mixed_outcome_reward),
        mixed_text_reward=bool(ppo.mixed_text_reward),
        proofkg_require_all_eligible=bool(ppo.proofkg_require_all_eligible),
        rollouts_per_prompt=int(ppo.rollouts_per_prompt),
        max_new_tokens=int(ppo.max_new_tokens),
        max_input_length=int(training.max_input_length),
        ppo_max_passages=int(ppo.ppo_max_passages),
        ppo_max_kg_triples=int(ppo.ppo_max_kg_triples),
        question_kg_records_path=str(training.question_kg_records_path),
        min_question_kg_record_coverage=float(training.min_question_kg_record_coverage),
        require_nonempty_question_kg_records=bool(training.require_nonempty_question_kg_records),
        rollout_sampling_weights_path=str(training.rollout_sampling_weights_path),
        fixed_rollout_schedule_path=str(training.fixed_rollout_schedule_path),
        split=training.split,
        split_allow_none=bool(training.split_allow_none),
        alpha_override=training.alpha_override,
    )
    _validate_mixed_reward_config(cfg)
    return cfg


def check_targets(spec: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in ("output_dir", "log_path"):
        path = Path(str(spec[label]))
        if not path.is_absolute():
            path = ROOT / path
        present = path.exists()
        result[label] = {"path": str(path), "exists": present}
        if present:
            raise FileExistsError(f"probe target already exists: {path}")
    # /root/tf-logs is a remote-only target.  The launcher repeats this check
    # with the actual remote identity immediately before CUDA is invoked.
    result["tensorboard_dir"] = {
        "path": spec["tensorboard_dir"],
        "status": "DEFERRED_TO_REMOTE_LAUNCHER_FAIL_CLOSED_CHECK",
    }
    return result


def validate_arm(arm: str, spec: dict[str, Any]) -> dict[str, Any]:
    probe_doc = load_config(ROOT / spec["config"], validate=ProjectConfig)
    formal_doc = load_config(ROOT / spec["formal_config"], validate=ProjectConfig)
    diff = config_diff(formal_doc, probe_doc)
    if diff != EXPECTED_CONFIG_DIFF:
        raise ValueError(f"{arm}: effective config drift={diff}")
    cfg = runtime_contract(probe_doc)
    expected_eligible = bool(spec["expected_eligible"])
    if cfg.proofkg_process_reward is not expected_eligible:
        raise ValueError(f"{arm}: ProofKG process flag does not match route under test")
    if cfg.total_steps != 4 or cfg.batch_size != 4 or cfg.rollouts_per_prompt != 4:
        raise ValueError(f"{arm}: probe must be exactly one batch / one K=4 group")
    if cfg.save_every_steps != 4:
        raise ValueError(f"{arm}: probe must checkpoint after its only batch")
    if Path(cfg.output_dir).name != spec["experiment_id"]:
        raise ValueError(f"{arm}: output basename is not its independent Experiment ID")

    trajectories = list(SilverDatasetReader(cfg.silver_path, split=None).accepted())
    if len(trajectories) != 1:
        raise ValueError(f"{arm}: probe population must contain one accepted row")
    join = apply_training_question_kg(
        trajectories,
        read_question_kg_records(cfg.question_kg_records_path),
        min_coverage=1.0,
        require_nonempty=False,
    ).to_dict()
    trajectory = trajectories[0]
    actual_eligible = is_identity_safe_automatic_proofkg(
        trajectory.metadata.get("question_kg_runtime") or {},
        trajectory.kg_subgraph,
        dataset=trajectory.dataset,
        qid=trajectory.qid,
    )
    if actual_eligible is not expected_eligible:
        raise ValueError(f"{arm}: production eligibility route is wrong")
    v21 = _validate_v21_execution_preflight(trajectories) if expected_eligible else None

    weights, records = _load_rollout_sampling_weights(
        cfg.rollout_sampling_weights_path, trajectories
    )
    indices, schedule = _load_fixed_rollout_schedule(
        cfg.fixed_rollout_schedule_path,
        trajectories,
        total_steps=cfg.total_steps,
        rollouts_per_prompt=cfg.rollouts_per_prompt,
        sampling_records=records,
    )
    if weights != [1.0] or indices != [0, 0, 0, 0]:
        raise ValueError(f"{arm}: fixed production loader did not resolve exactly one K=4 group")
    if any(bool(row.get("process_reward_eligible")) is not expected_eligible for row in schedule):
        raise ValueError(f"{arm}: schedule eligibility annotation disagrees with production route")

    prepared = _prepare_prompts(_MemoryReader(trajectories), _PromptTokenizer(), cfg)
    if len(prepared) != 1:
        raise ValueError(f"{arm}: production prompt builder did not yield exactly one prompt")
    prompt = prepared[0]["prompt"]
    if "[Knowledge Graph Context]" not in prompt or "[End of Knowledge Graph]" not in prompt:
        raise ValueError(f"{arm}: production KG prompt block is absent")
    if expected_eligible and not trajectory.kg_subgraph:
        raise ValueError(f"{arm}: eligible probe has no ProofKG triples")
    if not expected_eligible and trajectory.kg_subgraph:
        raise ValueError(f"{arm}: non-eligible wiring probe unexpectedly carries KG triples")
    spec_runtime = prepared[0]["spec"].metadata.get("question_kg_runtime") or {}
    if str(spec_runtime.get("question_key") or "") != question_key(
        str(trajectory.dataset), str(trajectory.qid)
    ):
        raise ValueError(f"{arm}: prompt RewardSpec lost identity-safe qKG runtime")

    return {
        "experiment_id": spec["experiment_id"],
        "effective_config_diff_from_formal_arm": diff,
        "config": spec["config"],
        "identity": question_key(str(trajectory.dataset), str(trajectory.qid)),
        "question_kg_join": join,
        "process_reward_eligible": actual_eligible,
        "v2_1_execution": v21,
        "kg_triples": len(trajectory.kg_subgraph),
        "fixed_schedule_indices": indices,
        "schedule_rows": len(schedule),
        "sampling_mass": sum(weights),
        "prompt_tokens_whitespace_proxy": int(prepared[0]["prompt_tokens"]),
        "targets": check_targets(spec),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--run_tests", action="store_true")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    report_dir = args.report_dir.resolve()
    if report_dir.exists():
        raise FileExistsError(f"append-only preflight directory already exists: {report_dir}")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_NOT_RUN":
        raise ValueError("probe freeze protocol is not FROZEN_NOT_RUN")
    if protocol.get("counts", {}).get("scheduled_trajectories_total") != 8:
        raise ValueError("probe freeze is not exactly eight trajectories")
    if protocol.get("scientific_boundary", {}).get("training_started") is not False:
        raise ValueError("probe freeze does not explicitly say training_started=false")
    for label, ref in protocol.get("inputs", {}).items():
        verify_ref(ref, f"input {label}")
    for label, ref in protocol.get("code_and_configs", {}).items():
        verify_ref(ref, f"code/config {label}")
    for arm, refs in protocol.get("outputs", {}).items():
        for label, ref in refs.items():
            verify_ref(ref, f"{arm} output {label}")

    arm_results = {arm: validate_arm(arm, spec) for arm, spec in ARM_SPECS.items()}
    if sum(result["schedule_rows"] for result in arm_results.values()) != 8:
        raise ValueError("combined probe budget exceeded or fell below eight trajectories")

    model_checks: dict[str, Any] = {}
    for logical in ("llama3-8B-instruct", "rearag"):
        resolved = Path(model_path(logical)).expanduser()
        model_checks[logical] = {"resolved": str(resolved), "is_local_dir": resolved.is_dir()}
        if not resolved.is_dir():
            raise FileNotFoundError(f"probe requires a complete local {logical} checkout: {resolved}")
    adapter = ROOT / "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / filename).is_file():
            raise FileNotFoundError(adapter / filename)

    test_result = None
    if args.run_tests:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *TESTS],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        test_result = {"returncode": proc.returncode, "output_tail": proc.stdout[-8000:]}
        if proc.returncode:
            raise RuntimeError("runtime-probe regression tests failed")

    report = {
        "schema_version": "mixed3-rearag-runtime-wiring-probe-preflight-v1",
        "experiment_id": "MIXED3-REARAG-RUNTIME-WIRING-PROBE-V1-SEED42-PREFLIGHT",
        "status": "PASS_CPU_PREFLIGHT_NOT_RUN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)),
            "sha256": sha256_file(protocol_path),
        },
        "cuda_allocated": False,
        "training_started": False,
        "checks": {
            "arm_results": arm_results,
            "total_scheduled_trajectories": 8,
            "models": model_checks,
            "sft_adapter": str(adapter.relative_to(ROOT)),
            "formal_pair_unchanged": True,
            "formal_data_unchanged": True,
            "tensorboard_boundary": (
                "Local preflight does not create /root/tf-logs. The remote launcher "
                "fail-closes on pre-existing per-arm TB targets before nvidia-smi/CUDA."
            ),
            "replay_boundary": (
                "One batch accrues 0.4 sample of 10% replay credit, hence this probe "
                "does not validate an actual replay optimizer update."
            ),
        },
        "tests": test_result,
        "scientific_boundary": (
            "Passing means only that production CPU contracts are wired. It is not "
            "evidence of PPO convergence, model improvement, or PPO-T/PPO-TK effect."
        ),
    }
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = report_dir / "preflight.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(report_dir, status=report["status"], extra={
        "phase": "mixed3_rearag_runtime_wiring_probe_preflight",
        "experiment_id": report["experiment_id"],
        "cuda_allocated": False,
        "training_started": False,
        "preflight_sha256": sha256_file(report_path),
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

