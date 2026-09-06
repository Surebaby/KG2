#!/usr/bin/env python
"""Freeze append-only locks for the approved mixed PPO-T/PPO-TK pair.

This script never starts training and never mutates an existing audit folder.
Run it only after the reward/data/config regression suite is green.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

from kgproweight.config import ProjectConfig, load_config
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.training.phase3_ppo import Phase3PPOConfig, _validate_mixed_reward_config
from kgproweight.utils.paths import model_path
from scripts.prepare.preflight_mixed3_rearag_ppo_pair import (
    EXPECTED_ALIAS_AUDIT,
    audit_frozen_answer_aliases,
)
from scripts.prepare.resolve_phase3_ppo_runtime_config import (
    resolve_phase3_ppo_runtime_config,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_DIR = ROOT / "outputs/audits/mixed3_rearag_ppo_pair_7200_seed42_v3"
ARMS = {
    "ppo_t": {
        "experiment_id": "ppo_mixed3_rearag_v1_text7200_seed42",
        "config": ROOT / "configs/training/phase3_ppo_mixed3_rearag_v1_text7200_seed42.yaml",
        "output_dir": "outputs/ppo_mixed3_rearag_v1_text7200_seed42",
        "tensorboard_dir": "/root/tf-logs/ppo_mixed3_rearag_v1_text7200_seed42",
        "log_path": "logs/training/ppo_mixed3_rearag_v1_text7200_seed42.log",
        "process_reward": False,
    },
    "ppo_tk": {
        "experiment_id": "ppo_mixed3_rearag_v1_text_kg_v2_1_7200_seed42",
        "config": ROOT / "configs/training/phase3_ppo_mixed3_rearag_v1_text_kg_v2_1_7200_seed42.yaml",
        "output_dir": "outputs/ppo_mixed3_rearag_v1_text_kg_v2_1_7200_seed42",
        "tensorboard_dir": "/root/tf-logs/ppo_mixed3_rearag_v1_text_kg_v2_1_7200_seed42",
        "log_path": "logs/training/ppo_mixed3_rearag_v1_text_kg_v2_1_7200_seed42.log",
        "process_reward": True,
    },
}

CODE_PATHS = [
    "kgproweight/config/__init__.py",
    "kgproweight/config/loader.py",
    "kgproweight/config/schemas.py",
    "kgproweight/data/prompts.py",
    "kgproweight/data/parsers.py",
    "kgproweight/data/silver_dataset.py",
    "kgproweight/data/entity_filter.py",
    "kgproweight/data/silver_split.py",
    "kgproweight/kg/training_question_kg.py",
    "kgproweight/kg/question_kg.py",
    "kgproweight/kg/kg_filter.py",
    "kgproweight/reward/composite_reward.py",
    "kgproweight/reward/proofkg_process.py",
    "kgproweight/reward/proofkg_process_v2.py",
    "kgproweight/reward/text_reward_model.py",
    "kgproweight/training/phase3_ppo.py",
    "kgproweight/training/phase3_sft.py",
    "kgproweight/training/reward_function.py",
    "kgproweight/training/step_reward_ppo_trainer.py",
    "kgproweight/utils/logging.py",
    "kgproweight/utils/paths.py",
    "kgproweight/utils/seed.py",
    "scripts/train/phase3_ppo.py",
    "scripts/train/_split_args.py",
    "scripts/prepare/finalize_mixed3_rearag_ppo_pair.py",
    "scripts/prepare/preflight_mixed3_rearag_ppo_pair.py",
    "scripts/prepare/resolve_phase3_ppo_runtime_config.py",
    "launch_ppo_mixed3_rearag_v1_paired7200_remote.sh",
]

CONFIG_DEPENDENCIES = [
    "configs/base.yaml",
    "configs/training/phase3_ppo.yaml",
    "configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600.yaml",
    "configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_explicit_sft_ref.yaml",
    "configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_combined_stability_v1.yaml",
]

INPUT_PATHS = {
    "data_manifest": "data/silver_data/mixed_ppo_three_dataset_v1_n1799_k4_seed42/manifest.json",
    "data_report": "data/silver_data/mixed_ppo_three_dataset_v1_n1799_k4_seed42/report.json",
    "silver_train": "data/silver_data/mixed_ppo_three_dataset_v1_n1799_k4_seed42/silver_train.jsonl",
    "question_kg_records": "data/silver_data/mixed_ppo_three_dataset_v1_n1799_k4_seed42/question_kg_records.jsonl",
    "sampling_weights": "data/silver_data/mixed_ppo_three_dataset_v1_n1799_k4_seed42/sampling_weights.jsonl",
    "fixed_rollout_schedule": "data/silver_data/mixed_ppo_three_dataset_v1_n1799_k4_seed42/fixed_rollout_schedule.jsonl",
    "sft_replay_silver": "checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42_no_text_head/silver_with_logprobs.jsonl",
    "sft_adapter_config": "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final/adapter_config.json",
    "sft_adapter_weights": "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final/adapter_model.safetensors",
    "sft_tokenizer_json": "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final/tokenizer.json",
    "sft_tokenizer_config": "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final/tokenizer_config.json",
    "sft_special_tokens_map": "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final/special_tokens_map.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(item, child))
        return result
    return {prefix: value}


def model_fingerprint(logical_name: str) -> dict[str, Any]:
    resolved = Path(model_path(logical_name)).expanduser()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"{logical_name} must resolve to a complete local directory, got {resolved}"
        )
    critical_names = [
        "config.json", "generation_config.json", "tokenizer_config.json",
        "tokenizer.json", "tokenizer.model", "special_tokens_map.json",
        "added_tokens.json", "vocab.json", "merges.txt",
        "model.safetensors.index.json", "pytorch_model.bin.index.json",
        "configuration_chatglm.py", "modeling_chatglm.py", "tokenization_chatglm.py",
    ]
    critical: dict[str, dict[str, Any]] = {}
    for name in critical_names:
        path = resolved / name
        if path.is_file():
            critical[name] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    if "config.json" not in critical:
        raise FileNotFoundError(f"model config missing: {resolved / 'config.json'}")
    weights = sorted(
        path for path in resolved.iterdir()
        if path.is_file() and path.suffix in {".bin", ".safetensors"}
    )
    if not weights:
        raise FileNotFoundError(f"model weights missing under {resolved}")
    inventory = [
        {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in weights
    ]
    inventory_sha = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "logical_name": logical_name,
        "resolved_path_at_freeze": str(resolved.resolve()),
        "critical_files": critical,
        "weight_inventory": inventory,
        "weight_inventory_sha256": inventory_sha,
        "weight_count": len(weights),
        "weight_total_bytes": sum(row["size_bytes"] for row in inventory),
        "boundary": "All listed local model files, including weight shards, are byte-hashed.",
    }


def git_state() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).splitlines()
    return {"commit": commit, "dirty": bool(status), "status_short": status}


def software_environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "torch", "transformers", "trl", "peft", "accelerate",
        "tokenizers", "safetensors", "tensorboard", "numpy",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    import torch

    return {
        "python": platform.python_version(),
        "packages": packages,
        "torch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "hardware_boundary": (
            "GPU model/driver are recorded by the remote preflight and need not "
            "equal the local freeze host; software versions are exact."
        ),
    }


def _assert_config_contract(arm: str, spec: dict[str, Any]) -> None:
    cfg = load_config(spec["config"], validate=ProjectConfig)
    training, ppo = cfg.training, cfg.training.ppo
    expected = {
        "silver_path": INPUT_PATHS["silver_train"],
        "question_kg_records_path": INPUT_PATHS["question_kg_records"],
        "fixed_rollout_schedule_path": INPUT_PATHS["fixed_rollout_schedule"],
        "rollout_sampling_weights_path": INPUT_PATHS["sampling_weights"],
        "sft_checkpoint": "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final",
        "output_dir": spec["output_dir"],
    }
    for field, value in expected.items():
        if getattr(training, field) != value:
            raise ValueError(f"{arm}: training.{field}={getattr(training, field)!r}, expected {value!r}")
    if cfg.reward.text_reward_backend != "rearag":
        raise ValueError(f"{arm}: ReaRAG backend is not explicit/fail-hard")
    if (
        training.alpha_gate_path is not None
        or training.alpha_override is not None
        or training.prm_checkpoint is not None
    ):
        raise ValueError(f"{arm}: historical alpha gate/PRM must be disabled")
    if training.question_kg_index_path is not None:
        raise ValueError(f"{arm}: legacy question-text KG index must be disabled")
    if training.rollout_schedule_path is not None:
        raise ValueError(f"{arm}: superseded non-fixed rollout schedule must be disabled")
    expected_ppo = {
        "learning_rate": 1e-6, "batch_size": 4, "mini_batch_size": 1,
        "rollouts_per_prompt": 4, "ppo_epochs": 2, "total_ppo_steps": 7200,
        "save_every_steps": 600, "kl_coef": .25, "target_kl": 8.0,
        "kl_horizon": 2000.0, "outcome_weight": 4.0,
        "proofkg_f1_weight": .1, "text_reward_scale": .3,
        "text_baseline_momentum": .99, "proofkg_process_weight": .2,
        "health_guard_after_steps": 200, "health_guard_window": 15,
        "health_guard_min_valid_rate": .70,
        "health_guard_max_length_capped_frac": .20,
        "health_guard_max_mean_kl": 10.0,
    }
    for field, value in expected_ppo.items():
        actual = getattr(ppo, field)
        if actual != value:
            raise ValueError(f"{arm}: training.ppo.{field}={actual!r}, expected {value!r}")
    boolean_contract = {
        "mixed_outcome_reward": True,
        "mixed_text_reward": True,
        "proofkg_process_reward": spec["process_reward"],
        "proofkg_dynamic_validity": True,
        "proofkg_require_all_eligible": False,
        "center_text_reward": True,
        "pure_em_reward": False,
        "proofkg_outcome_only_reward": False,
    }
    for field, value in boolean_contract.items():
        if getattr(ppo, field) is not value:
            raise ValueError(f"{arm}: training.ppo.{field} violates paired contract")
    if ppo.proofkg_process_version != "v2_1":
        raise ValueError(f"{arm}: ProofKG scorer must be v2_1")
    if training.split is not None or not training.split_allow_none:
        raise ValueError(f"{arm}: frozen train-only cohort split contract is wrong")
    if training.max_input_length != 6144 or ppo.max_new_tokens != 384:
        raise ValueError(f"{arm}: prompt/response length contract is wrong")
    if ppo.sft_replay_ratio != .1 or ppo.sft_anchor_weight != .1 or ppo.sft_anchor_interval != 0:
        raise ValueError(f"{arm}: 10% replay contract is wrong")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_dir", type=Path, default=DEFAULT_AUDIT_DIR)
    args = parser.parse_args()
    audit_dir = args.audit_dir.resolve()
    if audit_dir.exists():
        raise FileExistsError(f"append-only audit directory already exists: {audit_dir}")

    for arm, spec in ARMS.items():
        _assert_config_contract(arm, spec)
        if Path(spec["output_dir"]).name != spec["experiment_id"]:
            raise ValueError(
                f"{arm}: runtime Experiment ID is output basename and must equal locked ID"
            )
    for rel in [*CODE_PATHS, *CONFIG_DEPENDENCIES, *INPUT_PATHS.values()]:
        if not (ROOT / rel).is_file():
            raise FileNotFoundError(ROOT / rel)

    trajectories = list(
        SilverDatasetReader(ROOT / INPUT_PATHS["silver_train"], split=None).accepted()
    )
    alias_audit = audit_frozen_answer_aliases(trajectories)
    for field, expected in EXPECTED_ALIAS_AUDIT.items():
        if alias_audit.get(field) != expected:
            raise ValueError(
                f"frozen alias contract {field}={alias_audit.get(field)!r}, "
                f"expected {expected!r}"
            )

    # Do every expensive/read-only operation before creating the append-only
    # directory. A missing model shard or hash error therefore cannot leave a
    # half-created audit version that looks like a completed lock.
    runtime_configs = {
        arm: resolve_phase3_ppo_runtime_config(spec["config"])
        for arm, spec in ARMS.items()
    }
    for arm, runtime in runtime_configs.items():
        _validate_mixed_reward_config(Phase3PPOConfig(**runtime))
        if runtime["alpha_gate_path"] is not None or runtime["alpha_override"] is not None:
            raise ValueError(f"{arm}: actual CLI runtime unexpectedly enabled alpha")
    runtime_left = flatten(runtime_configs["ppo_t"])
    runtime_right = flatten(runtime_configs["ppo_tk"])
    runtime_diff = sorted(
        key for key in set(runtime_left) | set(runtime_right)
        if runtime_left.get(key) != runtime_right.get(key)
    )
    expected_runtime_diff = ["output_dir", "proofkg_process_reward"]
    if runtime_diff != expected_runtime_diff:
        raise ValueError(
            f"actual CLI runtime diff={runtime_diff}, expected {expected_runtime_diff}"
        )

    code = {path: file_identity(ROOT / path) for path in CODE_PATHS}
    config_dependencies = {
        path: file_identity(ROOT / path) for path in CONFIG_DEPENDENCIES
    }
    inputs = {name: file_identity(ROOT / path) for name, path in INPUT_PATHS.items()}
    models = {
        "base_policy": model_fingerprint("llama3-8B-instruct"),
        "text_reward": model_fingerprint("rearag"),
    }
    environment = software_environment()
    state = git_state()
    audit_dir.mkdir(parents=True, exist_ok=False)
    created_at = datetime.now(timezone.utc).isoformat()
    arm_locks: dict[str, dict[str, Any]] = {}
    for arm, spec in ARMS.items():
        config_id = file_identity(spec["config"])
        lock = {
            "schema_version": "mixed3-rearag-ppo-arm-lock-v1",
            "status": "CONFIGURED_NOT_STARTED",
            "arm": arm,
            "experiment_id": spec["experiment_id"],
            "created_at_utc": created_at,
            "config": config_id,
            "output_dir": spec["output_dir"],
            "tensorboard_dir": spec["tensorboard_dir"],
            "log_path": spec["log_path"],
            "proofkg_process_reward": spec["process_reward"],
            "resolved_cli_runtime_config": runtime_configs[arm],
            "git": state,
            "code": code,
            "config_dependencies": config_dependencies,
            "inputs": inputs,
            "models": models,
            "software_environment": environment,
            "reward_contract": {
                "invalid": "-4.0 exactly; no ReaRAG or ProofKG term",
                "valid_ppo_t": (
                    "4*(max_frozen_alias_canonical_EM + "
                    "0.1*max_frozen_alias_canonical_token_F1) + "
                    "0.3*mean_t(clip(ReaRAG_step_score - causal_EMA_preupdate, -1, 1))"
                ),
                "valid_ppo_tk_delta": (
                    "identity_safe_complete_ProofKG_eligible * 0.2 * ProofKG-v2.1"
                ),
                "reward_placement": {
                    "rearag": (
                        "for n valid steps, each step-end receives "
                        "0.3/n*clip(score_t-causal_EMA_preupdate,-1,1)"
                    ),
                    "outcome": "final generated token only",
                    "proofkg_v2_1": "global process-derived scalar on final generated token only",
                    "invalid": "-4.0 on final generated token only; no scorer call",
                },
                "rearag": "frozen, explicit backend, load failure is fatal",
                "historical_alpha_or_prm": "not consumed",
                "answer_aliases": {
                    "policy": (
                        "primary plus versioned gold_answer_aliases; canonical "
                        "EM and token F1 take independent maxima over canonical-unique surfaces"
                    ),
                    "frozen_population_audit": alias_audit,
                },
            },
        }
        lock_path = audit_dir / f"{arm}.lock.json"
        lock_path.write_text(
            json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        arm_locks[arm] = {
            "path": str(lock_path.relative_to(ROOT)),
            "sha256": sha256(lock_path),
            "size_bytes": lock_path.stat().st_size,
        }

    pair_manifest = {
        "schema_version": "mixed3-rearag-ppo-pair-manifest-v1",
        "status": "CONFIGURED_NOT_STARTED",
        "experiment_family": "MIXED3-V1-REARAG-PAIRED-PPO-7200-SEED42",
        "researcher_approval": "USER_APPROVED_2026-09-03_REARAG_PPO_T_VS_PPO_TK",
        "created_at_utc": created_at,
        "arm_order": ["ppo_t", "ppo_tk"],
        "arm_locks": arm_locks,
        "single_variable": {
            "description": "PPO-TK adds eligible*0.2*ProofKG-v2.1 to PPO-T",
            "allowed_effective_config_differences": [
                "training.output_dir", "training.ppo.proofkg_process_reward"
            ],
            "actual_cli_runtime_differences": runtime_diff,
            "allowed_cli_runtime_differences": expected_runtime_diff,
        },
        "shared_schedule": {
            "trajectories": 7200,
            "prompt_groups": 1800,
            "rollouts_per_prompt": 4,
            "dataset_prompt_groups": {
                "hotpotqa": 600, "2wikimultihopqa": 600, "musique": 600
            },
            "process_eligible_prompt_groups": 300,
            "process_eligible_trajectories": 1200,
        },
        "shared_answer_alias_contract": alias_audit,
        "execution": {
            "training_started": False,
            "remote_order": ["ppo_t", "ppo_tk"],
            "large_training_must_not_start_until_preflight_status": "PASS_NO_GPU_PREFLIGHT",
        },
        "scientific_boundary": {
            "ppo_t_minus_sft": "outcome plus ReaRAG post-training effect",
            "ppo_tk_minus_ppo_t": "net eligible ProofKG-v2.1 process-reward effect",
            "hotpot_musique_process_supervision": "none; differences are shared-policy transfer/retention",
            "sft_replay": "10% HotpotQA-only anti-forgetting anchor, shared by both arms",
            "number_of_seeds": 1,
        },
    }
    manifest_path = audit_dir / "pair_manifest.json"
    manifest_path.write_text(
        json.dumps(pair_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(manifest_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
