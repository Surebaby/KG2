#!/usr/bin/env python
"""Freeze the append-only Proof400 PPO-T/PPO-TK formal training pair.

This is a CPU-only finalizer.  It never starts model inference or training and
never mutates an existing audit directory.  The formal launcher remains
fail-closed until the separately versioned Proof400 v3 GPU postflight bundle
exists at the exact frozen paths, its report and manifest both have status
``PASS_RUNTIME_WIRING_NOT_EFFECT_EVIDENCE``, and the manifest binds the report
hash.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
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
from kgproweight.kg.training_question_kg import (
    apply_training_question_kg,
    read_question_kg_records,
)
from kgproweight.reward.proofkg_process import (
    canonical_answer_normalize,
    is_identity_safe_automatic_proofkg,
)
from kgproweight.training.phase3_ppo import (
    Phase3PPOConfig,
    _load_fixed_rollout_schedule,
    _load_rollout_sampling_weights,
    _validate_mixed_reward_config,
    _validate_v21_execution_preflight,
)
from kgproweight.utils.paths import model_path
from scripts.prepare.resolve_phase3_ppo_runtime_config import (
    resolve_phase3_ppo_runtime_config,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_DIR = (
    ROOT / "outputs/audits/mixed3_rearag_proof400_ppo_pair_7200_seed42_v2"
)
GPU_POSTFLIGHT_PATH = (
    ROOT
    / "outputs/audits/"
    "mixed3_rearag_runtime_wiring_probe_v3_proof400_seed42_gpu_postflight/"
    "postflight.json"
)
GPU_POSTFLIGHT_MANIFEST_PATH = GPU_POSTFLIGHT_PATH.parent / "manifest.json"
GPU_POSTFLIGHT_REQUIRED_STATUS = "PASS_RUNTIME_WIRING_NOT_EFFECT_EVIDENCE"
DATA_DIR = (
    ROOT
    / "data/silver_data/"
    "mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42"
)

ARMS = {
    "ppo_t": {
        "experiment_id": "ppo_mixed3_rearag_v2_proof400_text7200_seed42",
        "config": (
            ROOT
            / "configs/training/"
            "phase3_ppo_mixed3_rearag_v2_proof400_text7200_seed42.yaml"
        ),
        "output_dir": "outputs/ppo_mixed3_rearag_v2_proof400_text7200_seed42",
        "tensorboard_dir": "/root/tf-logs/ppo_mixed3_rearag_v2_proof400_text7200_seed42",
        "log_path": "logs/training/ppo_mixed3_rearag_v2_proof400_text7200_seed42.log",
        "process_reward": False,
    },
    "ppo_tk": {
        "experiment_id": "ppo_mixed3_rearag_v2_proof400_text_kg_v2_1_7200_seed42",
        "config": (
            ROOT
            / "configs/training/"
            "phase3_ppo_mixed3_rearag_v2_proof400_text_kg_v2_1_7200_seed42.yaml"
        ),
        "output_dir": (
            "outputs/ppo_mixed3_rearag_v2_proof400_text_kg_v2_1_7200_seed42"
        ),
        "tensorboard_dir": (
            "/root/tf-logs/"
            "ppo_mixed3_rearag_v2_proof400_text_kg_v2_1_7200_seed42"
        ),
        "log_path": (
            "logs/training/"
            "ppo_mixed3_rearag_v2_proof400_text_kg_v2_1_7200_seed42.log"
        ),
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
    "kgproweight/kg/training_question_kg.py",
    "kgproweight/reward/proofkg_process.py",
    "kgproweight/reward/proofkg_process_v2.py",
    "kgproweight/reward/text_reward_model.py",
    "kgproweight/training/phase3_ppo.py",
    "kgproweight/training/reward_function.py",
    "kgproweight/training/step_reward_ppo_trainer.py",
    "scripts/train/phase3_ppo.py",
    "scripts/train/_split_args.py",
    "scripts/prepare/resolve_phase3_ppo_runtime_config.py",
    "scripts/prepare/finalize_mixed3_rearag_proof400_ppo_pair.py",
    "scripts/prepare/preflight_mixed3_rearag_proof400_ppo_pair.py",
    "launch_ppo_mixed3_rearag_v2_proof400_paired7200_remote.sh",
    "tests/test_mixed3_rearag_proof400_pair_wiring.py",
]

CONFIG_DEPENDENCIES = [
    "configs/base.yaml",
    "configs/training/phase3_ppo.yaml",
    "configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600.yaml",
    "configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_explicit_sft_ref.yaml",
    "configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_combined_stability_v1.yaml",
    "configs/training/phase3_ppo_mixed3_rearag_v1_text7200_seed42.yaml",
]

INPUT_PATHS = {
    "data_manifest": DATA_DIR / "manifest.json",
    "data_report": DATA_DIR / "report.json",
    "silver_train": DATA_DIR / "silver_train.jsonl",
    "question_kg_records": DATA_DIR / "question_kg_records.jsonl",
    "sampling_weights": DATA_DIR / "sampling_weights.jsonl",
    "prompt_groups": DATA_DIR / "prompt_groups.jsonl",
    "fixed_rollout_schedule": DATA_DIR / "fixed_rollout_schedule.jsonl",
    "sft_replay_silver": (
        ROOT
        / "checkpoints/"
        "prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42_no_text_head/"
        "silver_with_logprobs.jsonl"
    ),
    "sft_adapter_config": (
        ROOT
        / "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/"
        "final/adapter_config.json"
    ),
    "sft_adapter_weights": (
        ROOT
        / "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/"
        "final/adapter_model.safetensors"
    ),
    "sft_tokenizer_json": (
        ROOT
        / "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/"
        "final/tokenizer.json"
    ),
    "sft_tokenizer_config": (
        ROOT
        / "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/"
        "final/tokenizer_config.json"
    ),
    "sft_special_tokens_map": (
        ROOT
        / "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/"
        "final/special_tokens_map.json"
    ),
}

BOUND_PROTOCOL_PATHS = {
    "v2_data_protocol": (
        ROOT
        / "outputs/audits/"
        "mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_protocol/"
        "protocol.json"
    ),
    "v2_data_protocol_manifest": (
        ROOT
        / "outputs/audits/"
        "mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_protocol/"
        "manifest.json"
    ),
    "v2_family_scope_addendum": (
        ROOT
        / "outputs/audits/"
        "mixed_ppo_three_dataset_v2_proof400_family_scope_addendum_v2/"
        "addendum.json"
    ),
    "v2_family_scope_addendum_manifest": (
        ROOT
        / "outputs/audits/"
        "mixed_ppo_three_dataset_v2_proof400_family_scope_addendum_v2/"
        "manifest.json"
    ),
    "config_comparison_v2": (
        ROOT
        / "outputs/audits/"
        "mixed3_rearag_ppo_pair_proof400_7200_seed42_config_comparison_v2/"
        "report.json"
    ),
    "config_comparison_v2_manifest": (
        ROOT
        / "outputs/audits/"
        "mixed3_rearag_ppo_pair_proof400_7200_seed42_config_comparison_v2/"
        "manifest.json"
    ),
    "standard_legacy_eval_protocol": (
        ROOT
        / "outputs/audits/mixed3_rearag_legacy_n300_eval_protocol_v1/"
        "protocol.json"
    ),
    "standard_legacy_eval_qids": (
        ROOT
        / "outputs/audits/mixed3_rearag_legacy_n300_eval_protocol_v1/"
        "qids.jsonl"
    ),
    "standard_legacy_eval_manifest": (
        ROOT
        / "outputs/audits/mixed3_rearag_legacy_n300_eval_protocol_v1/"
        "manifest.json"
    ),
    "v3_runtime_probe_protocol": (
        ROOT
        / "outputs/audits/"
        "mixed3_rearag_runtime_wiring_probe_v3_proof400_seed42_freeze/"
        "protocol.json"
    ),
    "v3_runtime_probe_protocol_manifest": (
        ROOT
        / "outputs/audits/"
        "mixed3_rearag_runtime_wiring_probe_v3_proof400_seed42_freeze/"
        "manifest.json"
    ),
    "v3_runtime_probe_local_preflight": (
        ROOT
        / "outputs/audits/"
        "mixed3_rearag_runtime_wiring_probe_v3_proof400_seed42_local_preflight/"
        "preflight.json"
    ),
    "v3_runtime_probe_local_preflight_manifest": (
        ROOT
        / "outputs/audits/"
        "mixed3_rearag_runtime_wiring_probe_v3_proof400_seed42_local_preflight/"
        "manifest.json"
    ),
}

EXPECTED_ALIAS_AUDIT = {
    "rows_with_multiple_raw_aliases": 163,
    "rows_with_multiple_normalized_aliases": 149,
    "rows_collapsed_to_one_normalized_alias": 14,
    "raw_aliases_in_multi_alias_rows": 443,
    "normalized_unique_aliases_in_multi_alias_rows": 377,
    "normalized_unique_count_distribution": {
        "1": 14, "2": 102, "3": 32, "4": 12, "5": 3,
    },
    "multi_alias_rows_by_dataset": {"musique": 163},
}

EXPECTED_COUNTS = {
    "unique_population": 1799,
    "unique_by_dataset": {
        "hotpotqa": 600, "2wikimultihopqa": 600, "musique": 599,
    },
    "eligible_unique": 400,
    "prompt_groups": 1800,
    "trajectories": 7200,
    "dataset_prompt_groups": {
        "hotpotqa": 600, "2wikimultihopqa": 600, "musique": 600,
    },
    "eligible_prompt_groups": 400,
    "eligible_trajectories": 1600,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative(path),
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


def audit_frozen_answer_aliases(trajectories: list[Any]) -> dict[str, Any]:
    unique_distribution: Counter[int] = Counter()
    by_dataset: Counter[str] = Counter()
    multi_rows = 0
    raw_alias_count = 0
    normalized_unique_count = 0
    for row in trajectories:
        aliases = row.metadata.get("gold_answer_aliases")
        if not isinstance(aliases, list) or not aliases:
            raise ValueError(f"{row.dataset}::{row.qid}: missing frozen alias list")
        primary = str(row.metadata.get("gold_answer") or "").strip()
        if str(aliases[0]).strip() != primary:
            raise ValueError(f"{row.dataset}::{row.qid}: primary alias is not first")
        if len(aliases) <= 1:
            continue
        normalized = {
            canonical_answer_normalize(value)
            for value in aliases
            if isinstance(value, str) and canonical_answer_normalize(value)
        }
        multi_rows += 1
        by_dataset[str(row.dataset)] += 1
        raw_alias_count += len(aliases)
        normalized_unique_count += len(normalized)
        unique_distribution[len(normalized)] += 1
    return {
        "rows_with_multiple_raw_aliases": multi_rows,
        "rows_with_multiple_normalized_aliases": sum(
            count for n_unique, count in unique_distribution.items() if n_unique > 1
        ),
        "rows_collapsed_to_one_normalized_alias": unique_distribution.get(1, 0),
        "raw_aliases_in_multi_alias_rows": raw_alias_count,
        "normalized_unique_aliases_in_multi_alias_rows": normalized_unique_count,
        "normalized_unique_count_distribution": {
            str(key): unique_distribution[key] for key in sorted(unique_distribution)
        },
        "multi_alias_rows_by_dataset": dict(sorted(by_dataset.items())),
        "normalizer": "kgproweight.reward.proofkg_process.canonical_answer_normalize",
    }


def inspect_gpu_postflight(path: Path = GPU_POSTFLIGHT_PATH) -> dict[str, Any]:
    manifest_path = path.parent / "manifest.json"
    result: dict[str, Any] = {
        "path": relative(path),
        "manifest_path": relative(manifest_path),
        "required_top_level_status": GPU_POSTFLIGHT_REQUIRED_STATUS,
        "gate_pass": False,
    }
    if not path.is_file():
        result.update(
            {
                "state": "MISSING",
                "observed_status": None,
                "manifest_present": manifest_path.is_file(),
            }
        )
        return result
    result["postflight_identity"] = file_identity(path)
    if not manifest_path.is_file():
        result.update(
            {
                "state": "INVALID_BUNDLE",
                "observed_status": None,
                "manifest_present": False,
                "error": "postflight.json exists but manifest.json is missing",
            }
        )
        return result
    result["manifest_present"] = True
    result["manifest_identity"] = file_identity(manifest_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result.update(
            {"state": "INVALID_JSON", "observed_status": None,
             "error": f"{type(exc).__name__}: {exc}"}
        )
        return result
    observed = payload.get("status")
    manifest_status = manifest.get("status")
    bound_hash = (manifest.get("run") or {}).get("postflight_sha256")
    actual_hash = result["postflight_identity"]["sha256"]
    expected_protocol = BOUND_PROTOCOL_PATHS["v3_runtime_probe_protocol"]
    protocol_ref = payload.get("protocol") or {}
    protocol_path = Path(str(protocol_ref.get("path") or ""))
    if not protocol_path.is_absolute():
        protocol_path = ROOT / protocol_path
    protocol_bound = (
        protocol_path.resolve() == expected_protocol.resolve()
        and protocol_ref.get("sha256") == sha256(expected_protocol)
        and int(protocol_ref.get("size_bytes", -1)) == expected_protocol.stat().st_size
    )
    bundle_pass = (
        observed == GPU_POSTFLIGHT_REQUIRED_STATUS
        and manifest_status == GPU_POSTFLIGHT_REQUIRED_STATUS
        and bound_hash == actual_hash
        and (manifest.get("run") or {}).get("effect_evidence") is False
        and protocol_bound
    )
    result.update(
        {
            "state": "PASS" if bundle_pass else "NON_PASS_BUNDLE",
            "observed_status": observed,
            "manifest_status": manifest_status,
            "manifest_bound_postflight_sha256": bound_hash,
            "postflight_actual_sha256": actual_hash,
            "manifest_hash_matches": bound_hash == actual_hash,
            "effect_evidence_false": (
                (manifest.get("run") or {}).get("effect_evidence") is False
            ),
            "probe_protocol_bound": protocol_bound,
            "gate_pass": bundle_pass,
        }
    )
    return result


def _assert_config_contract(arm: str, spec: dict[str, Any]) -> ProjectConfig:
    cfg = load_config(spec["config"], validate=ProjectConfig)
    training, ppo = cfg.training, cfg.training.ppo
    expected_training = {
        "silver_path": relative(INPUT_PATHS["silver_train"]),
        "question_kg_records_path": relative(INPUT_PATHS["question_kg_records"]),
        "rollout_sampling_weights_path": relative(INPUT_PATHS["sampling_weights"]),
        "fixed_rollout_schedule_path": relative(INPUT_PATHS["fixed_rollout_schedule"]),
        "sft_checkpoint": (
            "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"
        ),
        "output_dir": spec["output_dir"],
    }
    for field, expected in expected_training.items():
        actual = getattr(training, field)
        if actual != expected:
            raise ValueError(f"{arm}: training.{field}={actual!r}, expected {expected!r}")
    if cfg.reward.text_reward_backend != "rearag":
        raise ValueError(f"{arm}: ReaRAG must be the explicit fail-hard backend")
    if any(
        value is not None
        for value in (
            training.alpha_gate_path, training.alpha_override,
            training.prm_checkpoint, training.question_kg_index_path,
            training.rollout_schedule_path,
        )
    ):
        raise ValueError(f"{arm}: old alpha/PRM/legacy-index/schedule path is enabled")
    expected_ppo = {
        "learning_rate": 1e-6,
        "batch_size": 4,
        "mini_batch_size": 1,
        "rollouts_per_prompt": 4,
        "ppo_epochs": 2,
        "total_ppo_steps": 7200,
        "save_every_steps": 600,
        "kl_coef": 0.25,
        "target_kl": 8.0,
        "kl_horizon": 2000.0,
        "outcome_weight": 4.0,
        "proofkg_f1_weight": 0.1,
        "text_reward_scale": 0.3,
        "text_baseline_momentum": 0.99,
        "proofkg_process_weight": 0.2,
        "health_guard_after_steps": 200,
        "health_guard_window": 15,
        "health_guard_min_valid_rate": 0.70,
        "health_guard_max_length_capped_frac": 0.20,
        "health_guard_max_mean_kl": 10.0,
    }
    for field, expected in expected_ppo.items():
        actual = getattr(ppo, field)
        if actual != expected:
            raise ValueError(f"{arm}: training.ppo.{field}={actual!r}, expected {expected!r}")
    expected_flags = {
        "mixed_outcome_reward": True,
        "mixed_text_reward": True,
        "proofkg_process_reward": spec["process_reward"],
        "proofkg_dynamic_validity": True,
        "proofkg_require_all_eligible": False,
        "center_text_reward": True,
        "pure_em_reward": False,
        "proofkg_outcome_only_reward": False,
    }
    for field, expected in expected_flags.items():
        if getattr(ppo, field) is not expected:
            raise ValueError(f"{arm}: training.ppo.{field} violates paired contract")
    if ppo.proofkg_process_version != "v2_1":
        raise ValueError(f"{arm}: ProofKG process scorer is not v2_1")
    if training.split is not None or training.split_allow_none is not True:
        raise ValueError(f"{arm}: split contract is not train-only split=None")
    if training.max_input_length != 6144 or ppo.max_new_tokens != 384:
        raise ValueError(f"{arm}: prompt/response length contract changed")
    if (
        ppo.sft_replay_ratio != 0.1
        or ppo.sft_anchor_weight != 0.1
        or ppo.sft_anchor_interval != 0
    ):
        raise ValueError(f"{arm}: shared 10% replay contract changed")
    return cfg


def inspect_data_contract() -> dict[str, Any]:
    report = json.loads(INPUT_PATHS["data_report"].read_text(encoding="utf-8"))
    if report.get("status") != "COMPLETE_DATA_NOT_TRAINED":
        raise ValueError(f"unexpected data status: {report.get('status')!r}")
    if not all(report.get("gates", {}).values()):
        raise ValueError("not all materialization gates are true")
    expected_output_names = {
        "silver_train", "question_kg_records", "sampling_weights",
        "prompt_groups", "fixed_rollout_schedule",
    }
    if set(report.get("outputs", {})) != expected_output_names:
        raise ValueError("materialization report output inventory changed")
    for label, spec in report["outputs"].items():
        path = ROOT / str(spec.get("path") or "")
        if not path.is_file():
            raise FileNotFoundError(f"materialization output {label}: {path}")
        if (
            path.stat().st_size != int(spec.get("size_bytes", -1))
            or sha256(path) != spec.get("sha256")
        ):
            raise ValueError(f"materialization output identity mismatch: {label}")
    expected_report_counts = {
        "unique_population": EXPECTED_COUNTS["unique_population"],
        "unique_by_dataset": {
            "2wikimultihopqa": 600, "hotpotqa": 600, "musique": 599,
        },
        "process_reward_eligible_unique": EXPECTED_COUNTS["eligible_unique"],
        "scheduled_prompt_groups": EXPECTED_COUNTS["prompt_groups"],
        "scheduled_trajectories": EXPECTED_COUNTS["trajectories"],
        "scheduled_process_eligible_prompt_groups": EXPECTED_COUNTS[
            "eligible_prompt_groups"
        ],
        "scheduled_process_eligible_trajectories": EXPECTED_COUNTS[
            "eligible_trajectories"
        ],
    }
    for field, expected in expected_report_counts.items():
        if report.get("counts", {}).get(field) != expected:
            raise ValueError(
                f"materialization count {field}={report.get('counts', {}).get(field)!r}, "
                f"expected {expected!r}"
            )

    trajectories = list(
        SilverDatasetReader(INPUT_PATHS["silver_train"], split=None).accepted()
    )
    by_dataset = Counter(str(row.dataset) for row in trajectories)
    if len(trajectories) != EXPECTED_COUNTS["unique_population"]:
        raise ValueError(f"population={len(trajectories)}, expected 1799")
    if dict(by_dataset) != EXPECTED_COUNTS["unique_by_dataset"]:
        raise ValueError(f"dataset counts={dict(by_dataset)!r}")
    keys = [(str(row.dataset), str(row.qid)) for row in trajectories]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate dataset::qid in frozen population")
    alias_audit = audit_frozen_answer_aliases(trajectories)
    for field, expected in EXPECTED_ALIAS_AUDIT.items():
        if alias_audit.get(field) != expected:
            raise ValueError(
                f"alias contract {field}={alias_audit.get(field)!r}, expected {expected!r}"
            )

    join = apply_training_question_kg(
        trajectories,
        read_question_kg_records(INPUT_PATHS["question_kg_records"]),
        min_coverage=1.0,
        require_nonempty=False,
    ).to_dict()
    eligible_keys = {
        (str(row.dataset), str(row.qid))
        for row in trajectories
        if is_identity_safe_automatic_proofkg(
            row.metadata.get("question_kg_runtime") or {},
            row.kg_subgraph,
            dataset=row.dataset,
            qid=row.qid,
        )
    }
    if len(eligible_keys) != EXPECTED_COUNTS["eligible_unique"]:
        raise ValueError(f"eligible unique={len(eligible_keys)}, expected 400")
    eligible_rows = [
        row for row in trajectories
        if (str(row.dataset), str(row.qid)) in eligible_keys
    ]
    eligible_qtypes = Counter(
        str(row.metadata.get("question_type") or "") for row in eligible_rows
    )
    expected_qtypes = {
        "bridge_comparison": 100,
        "comparison": 100,
        "compositional": 100,
        "inference": 100,
    }
    if dict(eligible_qtypes) != expected_qtypes:
        raise ValueError(f"eligible question types={dict(eligible_qtypes)!r}")
    ineligible_rows = [
        row for row in trajectories
        if (str(row.dataset), str(row.qid)) not in eligible_keys
    ]
    if len(ineligible_rows) != 1399 or any(row.kg_subgraph for row in ineligible_rows):
        raise ValueError("the 1399 outcome/text-only rows are not all empty-KG")
    execution = _validate_v21_execution_preflight(trajectories)
    weights, sampling_records = _load_rollout_sampling_weights(
        INPUT_PATHS["sampling_weights"], trajectories
    )
    indices, schedule = _load_fixed_rollout_schedule(
        INPUT_PATHS["fixed_rollout_schedule"],
        trajectories,
        total_steps=EXPECTED_COUNTS["trajectories"],
        rollouts_per_prompt=4,
        sampling_records=sampling_records,
    )
    groups = schedule[::4]
    by_group_dataset = Counter(str(row.get("dataset") or "") for row in groups)
    eligible_groups = sum(bool(row.get("process_reward_eligible")) for row in groups)
    scheduled_eligible_group_keys = [
        (str(row.get("dataset") or ""), str(row.get("qid") or ""))
        for row in groups
        if bool(row.get("process_reward_eligible"))
    ]
    eligible_trajectories = sum(
        (str(trajectories[index].dataset), str(trajectories[index].qid))
        in eligible_keys
        for index in indices
    )
    if len(schedule) != EXPECTED_COUNTS["trajectories"]:
        raise ValueError(f"schedule rows={len(schedule)}, expected 7200")
    if len(groups) != EXPECTED_COUNTS["prompt_groups"]:
        raise ValueError(f"schedule groups={len(groups)}, expected 1800")
    if dict(by_group_dataset) != EXPECTED_COUNTS["dataset_prompt_groups"]:
        raise ValueError(f"schedule dataset groups={dict(by_group_dataset)!r}")
    if eligible_groups != EXPECTED_COUNTS["eligible_prompt_groups"]:
        raise ValueError(f"eligible groups={eligible_groups}, expected 400")
    if (
        len(set(scheduled_eligible_group_keys)) != 400
        or set(scheduled_eligible_group_keys) != eligible_keys
    ):
        raise ValueError("the schedule does not expose every eligible question exactly once")
    if eligible_trajectories != EXPECTED_COUNTS["eligible_trajectories"]:
        raise ValueError(f"eligible trajectories={eligible_trajectories}, expected 1600")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("rollout weights do not sum to one")
    return {
        "unique_population": len(trajectories),
        "unique_by_dataset": dict(by_dataset),
        "identity_join": join,
        "eligible_unique": len(eligible_keys),
        "eligible_by_question_type": dict(eligible_qtypes),
        "outcome_text_only_empty_kg": len(ineligible_rows),
        "prompt_groups": len(groups),
        "trajectories": len(schedule),
        "dataset_prompt_groups": dict(by_group_dataset),
        "eligible_prompt_groups": eligible_groups,
        "eligible_trajectories": eligible_trajectories,
        "answer_aliases": alias_audit,
        "v2_1_execution": execution,
    }


def assert_bound_protocol_contracts() -> dict[str, Any]:
    documents = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in BOUND_PROTOCOL_PATHS.items()
        if path.suffix == ".json" and not path.name.endswith(".jsonl")
    }
    protocol = documents["v2_data_protocol"]
    family = documents["v2_family_scope_addendum"]
    comparison = documents["config_comparison_v2"]
    evaluation = documents["standard_legacy_eval_protocol"]
    probe = documents["v3_runtime_probe_protocol"]
    probe_preflight = documents["v3_runtime_probe_local_preflight"]
    if protocol.get("population", {}).get("unique_total") != 1799:
        raise ValueError("v2 protocol does not lock 1799 unique questions")
    if protocol.get("population", {}).get("2wiki_complete_proofkg") != 400:
        raise ValueError("v2 protocol does not lock 400 complete ProofKG questions")
    schedule = protocol.get("schedule", {})
    if (
        schedule.get("prompt_groups") != 1800
        or schedule.get("trajectories") != 7200
        or schedule.get("process_eligible_groups") != 400
        or schedule.get("process_eligible_trajectories") != 1600
    ):
        raise ValueError("v2 protocol schedule contract changed")
    if family.get("status") != "COMPLETE_APPEND_ONLY_CLARIFICATION_DATA_UNCHANGED":
        raise ValueError("family-scope addendum v2 is not complete")
    if family.get("clarification", {}).get("protected_a_qid_overlap") != 0:
        raise ValueError("family addendum reports protected-A QID overlap")
    if family.get("clarification", {}).get(
        "protected_a_dataset_scoped_family_overlap"
    ) != 0:
        raise ValueError("family addendum reports protected-A family overlap")
    if comparison.get("status") != "PASS_CONFIG_ONLY_NOT_GPU_PROBED_NOT_TRAINED":
        raise ValueError("config-comparison-v2 status is not PASS")
    if not all(comparison.get("gates", {}).values()):
        raise ValueError("config-comparison-v2 contains a failed gate")
    if set(comparison.get("real_cli", {}).get("pair_differences", {})) != {
        "output_dir", "proofkg_process_reward",
    }:
        raise ValueError("config-comparison-v2 real CLI diff changed")
    if evaluation.get("status") != "FROZEN_SCHEME_A_READY_NOT_YET_RUN_ON_PPO":
        raise ValueError("standard legacy evaluation protocol status changed")
    scope = evaluation.get("scope", {})
    if (
        scope.get("datasets_in_order")
        != ["hotpotqa", "2wikimultihopqa", "musique"]
        or scope.get("n_per_dataset") != 300
        or scope.get("n_total") != 900
        or scope.get("seed") != 42
    ):
        raise ValueError("standard legacy evaluation scope changed")
    if probe.get("status") != "FROZEN_NOT_RUN":
        raise ValueError("Proof400 v3 runtime-probe protocol is not frozen/not-run")
    if probe.get("counts", {}).get("scheduled_trajectories_total") != 8:
        raise ValueError("Proof400 v3 runtime probe is not the frozen 4+4 trajectory probe")
    if probe_preflight.get("status") != "PASS_CPU_PREFLIGHT_NOT_RUN":
        raise ValueError("Proof400 v3 runtime-probe local preflight is not PASS")
    return {
        "v2_protocol_status": protocol.get("status"),
        "family_addendum_status": family.get("status"),
        "config_comparison_status": comparison.get("status"),
        "standard_eval_status": evaluation.get("status"),
        "standard_eval_scope": scope,
        "v3_runtime_probe_status": probe.get("status"),
        "v3_runtime_probe_local_preflight_status": probe_preflight.get("status"),
    }


def model_fingerprint(logical_name: str) -> dict[str, Any]:
    resolved = Path(model_path(logical_name)).expanduser()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{logical_name} resolved to missing {resolved}")
    critical_names = [
        "config.json", "generation_config.json", "tokenizer_config.json",
        "tokenizer.json", "tokenizer.model", "special_tokens_map.json",
        "added_tokens.json", "vocab.json", "merges.txt",
        "model.safetensors.index.json", "pytorch_model.bin.index.json",
        "configuration_chatglm.py", "modeling_chatglm.py", "tokenization_chatglm.py",
    ]
    critical = {
        name: {"sha256": sha256(resolved / name), "size_bytes": (resolved / name).stat().st_size}
        for name in critical_names
        if (resolved / name).is_file()
    }
    if "config.json" not in critical:
        raise FileNotFoundError(resolved / "config.json")
    weights = sorted(
        path for path in resolved.iterdir()
        if path.is_file() and path.suffix in {".bin", ".safetensors"}
    )
    if not weights:
        raise FileNotFoundError(f"no model weights under {resolved}")
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
    }


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
        "boundary": "GPU allocation is not performed by this CPU finalizer.",
    }


def git_state() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).splitlines()
    return {"commit": commit, "dirty": bool(status), "status_short": status}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_dir", type=Path, default=DEFAULT_AUDIT_DIR)
    args = parser.parse_args()
    audit_dir = args.audit_dir.resolve()
    if audit_dir.exists():
        raise FileExistsError(f"append-only audit directory exists: {audit_dir}")

    for path in [
        *[spec["config"] for spec in ARMS.values()],
        *[ROOT / path for path in CODE_PATHS],
        *[ROOT / path for path in CONFIG_DEPENDENCIES],
        *INPUT_PATHS.values(),
        *BOUND_PROTOCOL_PATHS.values(),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    for arm, spec in ARMS.items():
        _assert_config_contract(arm, spec)
        if Path(spec["output_dir"]).name != spec["experiment_id"]:
            raise ValueError(f"{arm}: output basename differs from Experiment ID")
        if (ROOT / spec["output_dir"]).exists():
            raise FileExistsError(ROOT / spec["output_dir"])
        if (ROOT / spec["log_path"]).exists():
            raise FileExistsError(ROOT / spec["log_path"])

    runtime_configs = {
        arm: resolve_phase3_ppo_runtime_config(spec["config"])
        for arm, spec in ARMS.items()
    }
    for arm, runtime in runtime_configs.items():
        _validate_mixed_reward_config(Phase3PPOConfig(**runtime))
        if runtime["proofkg_process_reward"] is not ARMS[arm]["process_reward"]:
            raise ValueError(f"{arm}: real CLI process flag differs from arm lock")
    left, right = (
        flatten(runtime_configs["ppo_t"]), flatten(runtime_configs["ppo_tk"])
    )
    runtime_diff = sorted(
        key for key in set(left) | set(right) if left.get(key) != right.get(key)
    )
    if runtime_diff != ["output_dir", "proofkg_process_reward"]:
        raise ValueError(f"real CLI diff={runtime_diff!r}")
    cfg_left = flatten(
        load_config(ARMS["ppo_t"]["config"], validate=ProjectConfig).model_dump()
    )
    cfg_right = flatten(
        load_config(ARMS["ppo_tk"]["config"], validate=ProjectConfig).model_dump()
    )
    effective_diff = sorted(
        key for key in set(cfg_left) | set(cfg_right)
        if cfg_left.get(key) != cfg_right.get(key)
    )
    if effective_diff != [
        "training.output_dir", "training.ppo.proofkg_process_reward",
    ]:
        raise ValueError(f"effective config diff={effective_diff!r}")

    data_contract = inspect_data_contract()
    protocol_contracts = assert_bound_protocol_contracts()
    postflight = inspect_gpu_postflight()
    if postflight["state"] not in {"MISSING", "PASS"}:
        raise ValueError(f"existing GPU postflight is not PASS: {postflight!r}")
    pair_status = (
        "READY_GPU_POSTFLIGHT_BOUND_NOT_STARTED"
        if postflight["gate_pass"]
        else "PREPARED_BLOCKED_GPU_PROBE"
    )

    # Complete all expensive hashes before creating the append-only directory.
    code = {path: file_identity(ROOT / path) for path in CODE_PATHS}
    config_dependencies = {
        path: file_identity(ROOT / path) for path in CONFIG_DEPENDENCIES
    }
    inputs = {name: file_identity(path) for name, path in INPUT_PATHS.items()}
    protocols = {
        name: file_identity(path) for name, path in BOUND_PROTOCOL_PATHS.items()
    }
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
        lock = {
            "schema_version": "mixed3-rearag-proof400-ppo-arm-lock-v1",
            "status": pair_status,
            "arm": arm,
            "experiment_id": spec["experiment_id"],
            "created_at_utc": created_at,
            "config": file_identity(spec["config"]),
            "output_dir": spec["output_dir"],
            "tensorboard_dir": spec["tensorboard_dir"],
            "log_path": spec["log_path"],
            "proofkg_process_reward": spec["process_reward"],
            "resolved_cli_runtime_config": runtime_configs[arm],
            "git": state,
            "code": code,
            "config_dependencies": config_dependencies,
            "inputs": inputs,
            "bound_protocols": protocols,
            "protocol_contracts": protocol_contracts,
            "models": models,
            "software_environment": environment,
            "gpu_postflight_at_freeze": postflight,
            "reward_contract": {
                "invalid": "-4.0 exactly; ReaRAG and ProofKG scorers are not called",
                "valid_ppo_t": (
                    "4*(alias-aware canonical EM + 0.1*alias-aware token F1) + "
                    "step-level 0.3*mean_t(clip(ReaRAG_t-causal_EMA_preupdate,-1,1))"
                ),
                "valid_ppo_tk_delta": (
                    "identity-safe complete ProofKG eligible * 0.2 * ProofKG-v2.1"
                ),
                "rearag_placement": (
                    "Each valid reasoning-step score is centered by the causal "
                    "pre-update EMA and placed at that step's end token."
                ),
                "outcome_and_proofkg_placement": "final generated token only",
                "rearag_backend": "shared, frozen, explicit, fail-hard",
                "historical_alpha_or_prm": "disabled and not consumed",
                "answer_aliases": data_contract["answer_aliases"],
            },
        }
        lock_path = audit_dir / f"{arm}.lock.json"
        lock_path.write_text(
            json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        arm_locks[arm] = file_identity(lock_path)

    manifest = {
        "schema_version": "mixed3-rearag-proof400-ppo-pair-manifest-v1",
        "status": pair_status,
        "experiment_family": "MIXED3-V2-PROOF400-REARAG-PAIRED-PPO-7200-SEED42",
        "researcher_approval": "USER_APPROVED_PROOF400_FORMAL_WIRING_2026-09-03",
        "created_at_utc": created_at,
        "arm_order": ["ppo_t", "ppo_tk"],
        "arm_locks": arm_locks,
        "single_variable": {
            "description": "PPO-TK adds eligible*0.2*ProofKG-v2.1 to PPO-T",
            "sole_scientific_variable": "proofkg_process_reward false -> true",
            "allowed_effective_config_differences": [
                "training.output_dir", "training.ppo.proofkg_process_reward",
            ],
            "actual_effective_config_differences": effective_diff,
            "allowed_cli_runtime_differences": [
                "output_dir", "proofkg_process_reward",
            ],
            "actual_cli_runtime_differences": runtime_diff,
        },
        "shared_data_contract": data_contract,
        "shared_schedule": {
            "unique_population": 1799,
            "prompt_groups": 1800,
            "trajectories": 7200,
            "rollouts_per_prompt": 4,
            "dataset_prompt_groups": EXPECTED_COUNTS["dataset_prompt_groups"],
            "process_eligible_unique": 400,
            "process_eligible_prompt_groups": 400,
            "process_eligible_trajectories": 1600,
        },
        "bound_protocols": protocols,
        "execution": {
            "training_started": False,
            "gpu_postflight": postflight,
            "required_gpu_postflight_path": relative(GPU_POSTFLIGHT_PATH),
            "required_gpu_postflight_manifest_path": relative(
                GPU_POSTFLIGHT_MANIFEST_PATH
            ),
            "required_gpu_postflight_status": GPU_POSTFLIGHT_REQUIRED_STATUS,
            "formal_launcher": "launch_ppo_mixed3_rearag_v2_proof400_paired7200_remote.sh",
            "remote_order": ["ppo_t", "ppo_tk"],
            "large_training_requires_preflight_status": (
                "PASS_CPU_PREFLIGHT_GPU_POSTFLIGHT_BOUND"
            ),
        },
        "scientific_boundary": {
            "ppo_t_minus_strong_sft": "outcome plus step-level ReaRAG post-training effect",
            "ppo_tk_minus_ppo_t": "net eligible ProofKG-v2.1 process-reward effect",
            "hotpot_musique_process_supervision": (
                "none; both receive shared outcome/text reward only"
            ),
            "sft_replay": "10% HotpotQA-only anti-forgetting replay, shared",
            "standard_main_table": (
                "frozen legacy n=300 standard pipeline; fresh SFT/PPO-T/PPO-TK rerun"
            ),
            "extra_resource_table": (
                "2Wiki ProofKG evaluation is separate because it consumes extra Wikidata resources"
            ),
            "number_of_training_seeds": 1,
            "replication_gate": (
                "No important superiority claim may be generalized beyond seed42 until an "
                "independent training seed reproduces it."
            ),
        },
    }
    manifest_path = audit_dir / "pair_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "pair_manifest": relative(manifest_path),
                "status": pair_status,
                "gpu_postflight": postflight,
                "real_cli_diff": runtime_diff,
                "counts": manifest["shared_schedule"],
                "training_started": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
