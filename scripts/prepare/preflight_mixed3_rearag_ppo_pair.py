#!/usr/bin/env python
"""CPU-only fail-closed preflight for the mixed ReaRAG PPO-T/PPO-TK pair."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

from kgproweight.config import ProjectConfig, load_config
from kgproweight.data.prompts import build_sft_messages
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.kg.training_question_kg import (
    apply_training_question_kg,
    read_question_kg_records,
)
from kgproweight.reward.proofkg_process import is_identity_safe_automatic_proofkg
from kgproweight.reward.proofkg_process import canonical_answer_normalize
from kgproweight.training.phase3_ppo import (
    Phase3PPOConfig,
    _load_fixed_rollout_schedule,
    _load_rollout_sampling_weights,
    _prepare_prompts,
    _validate_mixed_reward_config,
    _validate_v21_execution_preflight,
)
from kgproweight.training.reward_function import (
    KGProWeightRewardFunction,
    RewardSpec,
    _canonical_gold_surfaces,
)
from kgproweight.utils.paths import model_path
from scripts.prepare.resolve_phase3_ppo_runtime_config import (
    resolve_phase3_ppo_runtime_config,
)


ROOT = Path(__file__).resolve().parents[2]
TESTS = [
    "tests/test_mixed_ppo_reward.py",
    "tests/test_mixed_ppo_three_dataset_v1.py",
    "tests/test_phase3_ppo_config_forwarding.py",
    "tests/test_training_question_kg.py",
    "tests/test_proofkg_production_reward.py",
    "tests/test_ppo_rollout_schedule.py",
    "tests/test_ppo_sft_replay.py",
    "tests/test_ppo_explicit_reference.py",
    "tests/test_ppo_diagnostics.py",
    "tests/test_mixed3_rearag_pair_preflight.py",
]

EXPECTED_ALIAS_AUDIT = {
    "rows_with_multiple_raw_aliases": 163,
    "rows_with_multiple_normalized_aliases": 149,
    "rows_collapsed_to_one_normalized_alias": 14,
    "raw_aliases_in_multi_alias_rows": 443,
    "normalized_unique_aliases_in_multi_alias_rows": 377,
    "normalized_unique_count_distribution": {"1": 14, "2": 102, "3": 32, "4": 12, "5": 3},
    "multi_alias_rows_by_dataset": {"musique": 163},
}


class _MemoryReader:
    """Small reader adapter used to exercise the real prompt-preparation path."""

    def __init__(self, rows):
        self._rows = rows

    def accepted(self):
        yield from self._rows


class _PromptAuditTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if tokenize is not False or add_generation_prompt is not True:
            raise ValueError("unexpected prompt-audit chat-template arguments")
        return "\n".join(str(message["content"]) for message in messages)

    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(len(str(text).split())))}


class _CharTokenizer:
    def __call__(self, text, **_kwargs):
        return {"input_ids": list(str(text).encode("utf-8"))}

    def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
        return bytes(int(value) for value in ids).decode("utf-8")


def audit_frozen_answer_aliases(trajectories) -> dict[str, Any]:
    """Audit the versioned raw aliases with the exact reward normalizer."""

    multi_rows = []
    unique_distribution: Counter[int] = Counter()
    raw_alias_count = 0
    normalized_unique_count = 0
    by_dataset: Counter[str] = Counter()
    for row in trajectories:
        aliases = row.metadata.get("gold_answer_aliases")
        if not isinstance(aliases, list):
            raise ValueError(
                f"{row.dataset}::{row.qid} gold_answer_aliases is not a frozen list"
            )
        if not aliases or str(aliases[0]).strip() != str(
            row.metadata.get("gold_answer") or ""
        ).strip():
            raise ValueError(
                f"{row.dataset}::{row.qid} aliases do not start with primary Gold"
            )
        if len(aliases) <= 1:
            continue
        normalized = {
            canonical_answer_normalize(value)
            for value in aliases
            if isinstance(value, str) and canonical_answer_normalize(value)
        }
        multi_rows.append(row)
        by_dataset[str(row.dataset)] += 1
        raw_alias_count += len(aliases)
        normalized_unique_count += len(normalized)
        unique_distribution[len(normalized)] += 1
    result = {
        "rows_with_multiple_raw_aliases": len(multi_rows),
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
        "normalizer": (
            "kgproweight.reward.proofkg_process.canonical_answer_normalize"
        ),
    }
    return result


def _valid_answer_response(answer: str) -> str:
    blocks = []
    for index in range(1, 4):
        blocks.append(
            f"[Step {index}]\n"
            "Reasoning: The supplied evidence is checked carefully before answering.\n"
            "Knowledge Used: []\n"
            "Conclusion: The evidence check is complete.\n"
        )
    return "".join(blocks) + f"[Final Answer] {answer}"


def _alias_reward_probe(primary: str, aliases: list[str], prediction: str) -> dict[str, Any]:
    """Exercise the production mixed-outcome branch without model inference."""

    reward = KGProWeightRewardFunction(
        alpha_gate=SimpleNamespace(),
        prm_annotator=SimpleNamespace(),
        text_reward_model=SimpleNamespace(),
        tokenizer=_CharTokenizer(),
        outcome_weight=4.0,
        min_valid_steps=3,
        min_reasoning_chars=20,
        proofkg_process_reward=False,
        proofkg_process_version="v2_1",
        proofkg_process_weight=0.2,
        proofkg_f1_weight=0.1,
        proofkg_dynamic_validity=True,
        mixed_outcome_reward=True,
        mixed_text_reward=False,
    )
    response = _valid_answer_response(prediction)
    result = reward(
        prompt="",
        response=response,
        spec=RewardSpec(
            query="Alias reward dry-run question",
            gold_answer=primary,
            gold_answer_aliases=aliases,
            kg_subgraph=[],
            metadata={"dataset": "musique", "qid": "alias-dry-run"},
        ),
    )
    return {
        "primary": primary,
        "aliases": aliases,
        "prediction": prediction,
        "canonical_gold_surfaces": _canonical_gold_surfaces(primary, aliases),
        "trajectory_valid": result["trajectory_valid"],
        "outcome_em": result["proofkg_process"]["outcome_em"],
        "outcome_f1": result["proofkg_process"]["outcome_f1"],
        "matched_em_alias": result["proofkg_process"]["outcome_em_matched_alias"],
        "matched_f1_alias": result["proofkg_process"]["outcome_f1_matched_alias"],
        "trajectory_reward": result["trajectory_reward"],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(item, child))
        return result
    return {prefix: value}


def nearest_rank(values: list[int], quantile: float) -> int:
    if not values:
        raise ValueError("cannot compute a quantile of an empty list")
    ordered = sorted(values)
    index = max(0, math.ceil(float(quantile) * len(ordered)) - 1)
    return int(ordered[index])


def verify_file_identity(spec: dict[str, Any], failures: list[str], label: str) -> None:
    path = ROOT / str(spec["path"])
    if not path.is_file():
        failures.append(f"missing {label}: {path}")
        return
    if path.stat().st_size != int(spec["size_bytes"]):
        failures.append(f"size mismatch {label}: {path}")
    if sha256(path) != spec["sha256"]:
        failures.append(f"SHA256 mismatch {label}: {path}")


def current_model_fingerprint(logical_name: str) -> dict[str, Any]:
    resolved = Path(model_path(logical_name)).expanduser()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{logical_name} is not a local model directory: {resolved}")
    critical_names = [
        "config.json", "generation_config.json", "tokenizer_config.json",
        "tokenizer.json", "tokenizer.model", "special_tokens_map.json",
        "added_tokens.json", "vocab.json", "merges.txt",
        "model.safetensors.index.json", "pytorch_model.bin.index.json",
        "configuration_chatglm.py", "modeling_chatglm.py", "tokenization_chatglm.py",
    ]
    critical = {}
    for name in critical_names:
        path = resolved / name
        if path.is_file():
            critical[name] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    weights = sorted(
        path for path in resolved.iterdir()
        if path.is_file() and path.suffix in {".bin", ".safetensors"}
    )
    inventory = [
        {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in weights
    ]
    inventory_sha = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "critical_files": critical,
        "weight_inventory_sha256": inventory_sha,
        "weight_count": len(weights),
        "weight_total_bytes": sum(row["size_bytes"] for row in inventory),
        "resolved_path": str(resolved.resolve()),
    }


def current_software_environment() -> dict[str, Any]:
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
    }


def _runtime_contract(cfg_doc: ProjectConfig, process: bool) -> Phase3PPOConfig:
    training, ppo = cfg_doc.training, cfg_doc.training.ppo
    cfg = Phase3PPOConfig(
        silver_path=str(training.silver_path),
        output_dir=str(training.output_dir),
        sft_checkpoint=str(training.sft_checkpoint),
        sft_replay_silver_path=str(training.sft_replay_silver_path),
        sft_replay_split=training.sft_replay_split,
        alpha_gate_path=training.alpha_gate_path,
        text_reward_backend=cfg_doc.reward.text_reward_backend,
        learning_rate=ppo.learning_rate,
        batch_size=ppo.batch_size,
        mini_batch_size=ppo.mini_batch_size,
        ppo_epochs=ppo.ppo_epochs,
        total_steps=ppo.total_ppo_steps,
        save_every_steps=ppo.save_every_steps,
        outcome_weight=ppo.outcome_weight,
        text_reward_scale=ppo.text_reward_scale,
        center_text_reward=ppo.center_text_reward,
        text_baseline_momentum=ppo.text_baseline_momentum,
        proofkg_process_reward=ppo.proofkg_process_reward,
        proofkg_process_version=ppo.proofkg_process_version,
        proofkg_process_weight=ppo.proofkg_process_weight,
        proofkg_f1_weight=ppo.proofkg_f1_weight,
        proofkg_dynamic_validity=ppo.proofkg_dynamic_validity,
        mixed_outcome_reward=ppo.mixed_outcome_reward,
        mixed_text_reward=ppo.mixed_text_reward,
        proofkg_require_all_eligible=ppo.proofkg_require_all_eligible,
        rollouts_per_prompt=ppo.rollouts_per_prompt,
        question_kg_records_path=training.question_kg_records_path,
        rollout_sampling_weights_path=training.rollout_sampling_weights_path,
        fixed_rollout_schedule_path=training.fixed_rollout_schedule_path,
        split=training.split,
        split_allow_none=training.split_allow_none,
        alpha_override=training.alpha_override,
    )
    _validate_mixed_reward_config(cfg)
    if cfg.proofkg_process_reward is not process:
        raise ValueError("runtime process flag differs from arm lock")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair_manifest", type=Path, required=True)
    parser.add_argument("--report_path", type=Path, required=True)
    parser.add_argument("--run_tests", action="store_true")
    args = parser.parse_args()
    if args.report_path.exists():
        raise FileExistsError(f"append-only report already exists: {args.report_path}")

    failures: list[str] = []
    manifest = json.loads(args.pair_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "CONFIGURED_NOT_STARTED":
        failures.append("pair manifest status is not CONFIGURED_NOT_STARTED")
    if manifest.get("execution", {}).get("training_started") is not False:
        failures.append("pair manifest does not explicitly say training_started=false")

    locks: dict[str, dict[str, Any]] = {}
    cfg_docs: dict[str, ProjectConfig] = {}
    cli_runtime_configs: dict[str, dict[str, Any]] = {}
    target_absence_checks: list[dict[str, Any]] = []
    for arm in manifest.get("arm_order", []):
        lock_ref = manifest["arm_locks"][arm]
        lock_path = ROOT / lock_ref["path"]
        if not lock_path.is_file() or sha256(lock_path) != lock_ref["sha256"]:
            failures.append(f"arm lock hash mismatch: {arm}")
            continue
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        locks[arm] = lock
        if lock.get("status") != "CONFIGURED_NOT_STARTED":
            failures.append(f"{arm}: lock status is not CONFIGURED_NOT_STARTED")
        verify_file_identity(lock["config"], failures, f"{arm} config")
        for label, spec in lock.get("code", {}).items():
            verify_file_identity(spec, failures, f"code {label}")
        for label, spec in lock.get("config_dependencies", {}).items():
            verify_file_identity(spec, failures, f"config dependency {label}")
        for label, spec in lock.get("inputs", {}).items():
            verify_file_identity(spec, failures, f"input {label}")
        config_path = ROOT / lock["config"]["path"]
        try:
            cfg_docs[arm] = load_config(config_path, validate=ProjectConfig)
            _runtime_contract(cfg_docs[arm], bool(lock["proofkg_process_reward"]))
            cli_runtime_configs[arm] = resolve_phase3_ppo_runtime_config(config_path)
            if cli_runtime_configs[arm] != lock.get("resolved_cli_runtime_config"):
                failures.append(f"{arm}: exact CLI runtime config differs from frozen lock")
            _validate_mixed_reward_config(
                Phase3PPOConfig(**cli_runtime_configs[arm])
            )
        except Exception as exc:
            failures.append(f"{arm}: config/runtime contract failed: {type(exc).__name__}: {exc}")
        for target_field in ("output_dir", "log_path"):
            target = ROOT / lock[target_field]
            exists = target.exists()
            target_absence_checks.append(
                {
                    "arm": arm,
                    "field": target_field,
                    "path": str(target),
                    "status": "EXISTS" if exists else "ABSENT",
                    "checked_on": "current_host",
                }
            )
            if exists:
                failures.append(f"{arm}: {target_field} already exists: {target}")
        tensorboard_target = Path(lock["tensorboard_dir"])
        try:
            tensorboard_exists = tensorboard_target.exists()
        except PermissionError as exc:
            # The versioned lock deliberately names the AutoDL absolute path.
            # A non-root local preflight cannot stat /root; the remote launcher
            # repeats this same preflight as root *and* has a shell-level
            # fail-closed `test ! -e` before either arm starts.
            target_absence_checks.append(
                {
                    "arm": arm,
                    "field": "tensorboard_dir",
                    "path": str(tensorboard_target),
                    "status": "DEFERRED_TO_REMOTE_PREFLIGHT",
                    "checked_on": "current_host",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "remote_fail_closed": True,
                }
            )
        else:
            target_absence_checks.append(
                {
                    "arm": arm,
                    "field": "tensorboard_dir",
                    "path": str(tensorboard_target),
                    "status": "EXISTS" if tensorboard_exists else "ABSENT",
                    "checked_on": "current_host",
                }
            )
            if tensorboard_exists:
                failures.append(
                    f"{arm}: tensorboard_dir already exists: {lock['tensorboard_dir']}"
                )

    if set(cfg_docs) == {"ppo_t", "ppo_tk"}:
        left = flatten(cfg_docs["ppo_t"].model_dump())
        right = flatten(cfg_docs["ppo_tk"].model_dump())
        diff = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        allowed = sorted(manifest["single_variable"]["allowed_effective_config_differences"])
        if diff != allowed:
            failures.append(f"effective config diff={diff}, expected exactly {allowed}")
    else:
        diff = []

    if set(cli_runtime_configs) == {"ppo_t", "ppo_tk"}:
        runtime_left = flatten(cli_runtime_configs["ppo_t"])
        runtime_right = flatten(cli_runtime_configs["ppo_tk"])
        runtime_diff = sorted(
            key for key in set(runtime_left) | set(runtime_right)
            if runtime_left.get(key) != runtime_right.get(key)
        )
        allowed_runtime = sorted(
            manifest["single_variable"]["allowed_cli_runtime_differences"]
        )
        if runtime_diff != allowed_runtime:
            failures.append(
                f"exact CLI runtime diff={runtime_diff}, expected {allowed_runtime}"
            )
    else:
        runtime_diff = []

    if locks:
        reference_lock = locks[manifest["arm_order"][0]]
        expected_environment = reference_lock.get("software_environment")
        actual_environment = current_software_environment()
        if expected_environment is None:
            failures.append("software environment is absent from arm lock")
        else:
            for key in ("python", "packages", "torch_cuda_build", "cudnn_version"):
                if actual_environment.get(key) != expected_environment.get(key):
                    failures.append(f"software environment mismatch: {key}")
        for logical_name, expected in reference_lock.get("models", {}).items():
            model_name = expected["logical_name"]
            try:
                actual = current_model_fingerprint(model_name)
            except Exception as exc:
                failures.append(f"model {logical_name} unavailable: {type(exc).__name__}: {exc}")
                continue
            for key in ("critical_files", "weight_inventory_sha256", "weight_count", "weight_total_bytes"):
                if actual[key] != expected[key]:
                    failures.append(f"model fingerprint mismatch: {logical_name}.{key}")

        data_manifest_spec = reference_lock["inputs"]["data_manifest"]
        data_manifest = json.loads(
            (ROOT / data_manifest_spec["path"]).read_text(encoding="utf-8")
        )
        if not str(data_manifest.get("status", "")).startswith("COMPLETE_DATA_NOT_TRAINED"):
            failures.append("mixed data manifest is not frozen/not-trained")
        for label, spec in data_manifest.get("outputs", {}).items():
            verify_file_identity(spec, failures, f"data manifest output {label}")

    data_dir = ROOT / "data/silver_data/mixed_ppo_three_dataset_v1_n1799_k4_seed42"
    trajectories = list(SilverDatasetReader(data_dir / "silver_train.jsonl", split=None).accepted())
    trajectories = [row for row in trajectories if str(row.metadata.get("gold_answer") or "").strip()]
    population_counts = Counter(row.dataset for row in trajectories)
    if len(trajectories) != 1799 or population_counts != Counter(
        {"hotpotqa": 600, "2wikimultihopqa": 600, "musique": 599}
    ):
        failures.append(f"mixed population mismatch: n={len(trajectories)} counts={dict(population_counts)}")
    if len({(row.dataset, row.qid) for row in trajectories}) != len(trajectories):
        failures.append("mixed population contains duplicate dataset::qid")

    try:
        alias_audit = audit_frozen_answer_aliases(trajectories)
        for field, expected in EXPECTED_ALIAS_AUDIT.items():
            if alias_audit.get(field) != expected:
                failures.append(
                    f"frozen alias audit {field}={alias_audit.get(field)!r}, "
                    f"expected {expected!r}"
                )
    except Exception as exc:
        alias_audit = {"error": f"{type(exc).__name__}: {exc}"}
        failures.append("frozen answer-alias audit failed")

    # Exercise the exact production prompt-preparation function, rather than
    # merely searching source text for the RewardSpec field.  This verifies
    # that every versioned raw alias list survives SilverTrajectory parsing and
    # reaches RewardSpec unchanged in both arms' shared path.
    alias_forwarding: dict[str, Any]
    try:
        if "ppo_t" not in cfg_docs:
            raise ValueError("PPO-T config unavailable")
        prompt_cfg = _runtime_contract(cfg_docs["ppo_t"], False)
        prepared = _prepare_prompts(
            _MemoryReader(trajectories), _PromptAuditTokenizer(), prompt_cfg
        )
        prepared_by_key = {
            (str(row["spec"].metadata["dataset"]), str(row["spec"].metadata["qid"])):
            row["spec"]
            for row in prepared
        }
        forwarding_mismatches = []
        for trajectory in trajectories:
            key = (str(trajectory.dataset), str(trajectory.qid))
            spec = prepared_by_key.get(key)
            expected_aliases = [
                value for value in trajectory.metadata["gold_answer_aliases"]
                if isinstance(value, str) and value.strip()
            ]
            if spec is None or spec.gold_answer_aliases != expected_aliases:
                forwarding_mismatches.append(
                    {
                        "question_key": "::".join(key),
                        "expected": expected_aliases,
                        "actual": None if spec is None else spec.gold_answer_aliases,
                    }
                )
        alias_forwarding = {
            "production_function": "kgproweight.training.phase3_ppo._prepare_prompts",
            "prepared_specs": len(prepared),
            "exact_alias_lists_forwarded": len(trajectories) - len(forwarding_mismatches),
            "mismatch_count": len(forwarding_mismatches),
            "mismatch_examples": forwarding_mismatches[:5],
        }
        if len(prepared) != 1799 or forwarding_mismatches:
            failures.append(f"RewardSpec alias forwarding failed: {alias_forwarding}")
    except Exception as exc:
        alias_forwarding = {"error": f"{type(exc).__name__}: {exc}"}
        failures.append("RewardSpec alias forwarding audit failed")

    # Code-level outcome probes use real frozen rows where the primary answer
    # is not evaluation-equivalent to another accepted alias.  These examples
    # would be falsely marked wrong by the pre-fix single-primary reward.
    alias_reward_dry_examples: list[dict[str, Any]] = []
    try:
        trajectory_by_qid = {str(row.qid): row for row in trajectories}
        fbi = trajectory_by_qid["train_553"]
        prc = trajectory_by_qid["train_18524"]
        probes = [
            (fbi, "FBI", "fbi_short_alias"),
            (prc, "China", "prc_china_alias"),
            (prc, "People's Republic of China", "prc_long_alias"),
            (fbi, "unrelated zebra sentinel", "negative_control"),
        ]
        for trajectory, prediction, label in probes:
            result = _alias_reward_probe(
                str(trajectory.metadata["gold_answer"]),
                list(trajectory.metadata["gold_answer_aliases"]),
                prediction,
            )
            result["label"] = label
            result["source_question_key"] = f"{trajectory.dataset}::{trajectory.qid}"
            alias_reward_dry_examples.append(result)
        for result in alias_reward_dry_examples[:3]:
            if (
                result["trajectory_valid"] is not True
                or result["outcome_em"] != 1.0
                or result["outcome_f1"] != 1.0
                or not math.isclose(result["trajectory_reward"], 4.4)
            ):
                failures.append(f"positive alias reward dry-run failed: {result}")
        negative = alias_reward_dry_examples[-1]
        if (
            negative["outcome_em"] != 0.0
            or negative["outcome_f1"] != 0.0
            or negative["trajectory_reward"] != 0.0
        ):
            failures.append(f"negative alias reward dry-run failed: {negative}")
    except Exception as exc:
        alias_reward_dry_examples = [{"error": f"{type(exc).__name__}: {exc}"}]
        failures.append("alias-aware production reward dry-runs failed")

    # ReaRAG uses a different tokenizer from the policy. Audit the exact base
    # prompts that _score_mixed_text_process sends to that scorer. This is a
    # CPU-only context-safety check; generated step prefixes are runtime data,
    # so the report explicitly does not claim their token lengths are known.
    rearag_prompt_audit: dict[str, Any]
    try:
        from transformers import AutoTokenizer

        rearag_path = Path(model_path("rearag"))
        rearag_tokenizer = AutoTokenizer.from_pretrained(
            rearag_path, trust_remote_code=True
        )
        prompt_lengths: list[int] = []
        for row in trajectories:
            messages = build_sft_messages(
                question=row.question,
                retrieved_passages=row.retrieved_passages,
                kg_triples=row.kg_subgraph,
            )
            rendered = "\n\n".join(message["content"] for message in messages)
            prompt_lengths.append(
                len(rearag_tokenizer(rendered, add_special_tokens=False)["input_ids"])
            )
        rearag_prompt_audit = {
            "scope": "base prompt before generated step prefix",
            "n": len(prompt_lengths),
            "mean": sum(prompt_lengths) / max(1, len(prompt_lengths)),
            "p95_nearest_rank": nearest_rank(prompt_lengths, .95),
            "p99_nearest_rank": nearest_rank(prompt_lengths, .99),
            "max": max(prompt_lengths),
            "count_gt_3500": sum(length > 3500 for length in prompt_lengths),
            "count_gt_rearag_window_4096": sum(length > 4096 for length in prompt_lengths),
            "rearag_context_window_used_by_scorer": 4096,
            "configured_policy_max_new_tokens": 384,
            "boundary": (
                "The frozen base prompts fit with >=596 ReaRAG tokens of margin. "
                "Generated Llama-token prefixes are not assumed to have a 1:1 ReaRAG "
                "token ratio; RearagPromptScorer left-truncates at 4096 if required."
            ),
        }
        if len(prompt_lengths) != 1799 or rearag_prompt_audit["count_gt_3500"] != 0:
            failures.append(f"ReaRAG base-prompt context gate failed: {rearag_prompt_audit}")
    except Exception as exc:
        rearag_prompt_audit = {"error": f"{type(exc).__name__}: {exc}"}
        failures.append("ReaRAG tokenizer prompt audit failed")

    try:
        join_stats = apply_training_question_kg(
            trajectories,
            read_question_kg_records(data_dir / "question_kg_records.jsonl"),
            min_coverage=1.0,
            require_nonempty=False,
        ).to_dict()
    except Exception as exc:
        join_stats = {"error": f"{type(exc).__name__}: {exc}"}
        failures.append("question-KG identity join failed")

    eligible_keys = {
        (row.dataset, row.qid)
        for row in trajectories
        if is_identity_safe_automatic_proofkg(
            row.metadata.get("question_kg_runtime") or {}, row.kg_subgraph,
            dataset=row.dataset, qid=row.qid,
        )
    }
    if len(eligible_keys) != 208:
        failures.append(f"identity-safe complete ProofKG population is {len(eligible_keys)}, expected 208")
    try:
        execution_stats = _validate_v21_execution_preflight(trajectories)
    except Exception as exc:
        execution_stats = {"error": f"{type(exc).__name__}: {exc}"}
        failures.append("ProofKG-v2.1 execution preflight failed")

    try:
        weights, sampling_records = _load_rollout_sampling_weights(
            data_dir / "sampling_weights.jsonl", trajectories
        )
        indices, schedule = _load_fixed_rollout_schedule(
            data_dir / "fixed_rollout_schedule.jsonl", trajectories,
            total_steps=7200, rollouts_per_prompt=4,
            sampling_records=sampling_records,
        )
    except Exception as exc:
        weights, sampling_records, indices, schedule = [], {}, [], []
        failures.append(f"fixed schedule preflight failed: {type(exc).__name__}: {exc}")

    schedule_groups = schedule[::4]
    scheduled_dataset_groups = Counter(str(row.get("dataset") or "") for row in schedule_groups)
    scheduled_process_groups = sum(bool(row.get("process_reward_eligible")) for row in schedule_groups)
    actual_process_trajectories = sum(
        (trajectories[index].dataset, trajectories[index].qid) in eligible_keys for index in indices
    )
    if len(schedule) != 7200 or len(schedule_groups) != 1800:
        failures.append("schedule is not 7200 trajectories / 1800 prompt groups")
    if scheduled_dataset_groups != Counter(
        {"hotpotqa": 600, "2wikimultihopqa": 600, "musique": 600}
    ):
        failures.append(f"scheduled dataset groups mismatch: {dict(scheduled_dataset_groups)}")
    if scheduled_process_groups != 300 or actual_process_trajectories != 1200:
        failures.append(
            "scheduled process eligibility mismatch: "
            f"groups={scheduled_process_groups}, trajectories={actual_process_trajectories}"
        )
    if weights and not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        failures.append("sampling weights do not sum to 1.0")

    # Algebraic sentinels for the frozen formula. Code-level behaviour is
    # separately exercised by test_mixed_ppo_reward.py.
    em, f1, centered_text, process_score = 1.0, 0.5, 0.2, 0.6
    dry_t = 4.0 * (em + .1 * f1) + .3 * centered_text
    dry_tk = dry_t + .2 * process_score
    if not math.isclose(dry_t, 4.26) or not math.isclose(dry_tk - dry_t, .12):
        failures.append("frozen reward algebra sentinel failed")

    test_result = None
    if args.run_tests:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *TESTS],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        test_result = {"returncode": proc.returncode, "output_tail": proc.stdout[-8000:]}
        if proc.returncode:
            failures.append("regression tests failed")

    report = {
        "schema_version": "mixed3-rearag-ppo-pair-preflight-v1",
        "experiment_family": manifest.get("experiment_family"),
        "status": "PASS_NO_GPU_PREFLIGHT" if not failures else "FAIL_NO_GPU_PREFLIGHT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_allocated": False,
        "training_started": False,
        "pair_manifest": {
            "path": str(args.pair_manifest), "sha256": sha256(args.pair_manifest)
        },
        "checks": {
            "effective_config_diff": diff,
            "exact_cli_runtime_diff": runtime_diff,
            "target_absence": target_absence_checks,
            "remote_tensorboard_boundary": (
                "A local non-root PermissionError is explicitly deferred. The remote "
                "launcher and its repeated preflight both require each locked /root/tf-logs "
                "directory to be absent before any training process starts."
            ),
            "software_environment": (
                current_software_environment() if locks else None
            ),
            "population_rows": len(trajectories),
            "population_by_dataset": dict(population_counts),
            "frozen_answer_aliases": alias_audit,
            "reward_spec_alias_forwarding": alias_forwarding,
            "alias_reward_dry_examples": alias_reward_dry_examples,
            "question_kg_join": join_stats,
            "proofkg_eligible_unique": len(eligible_keys),
            "v2_1_execution": execution_stats,
            "schedule_rows": len(schedule),
            "schedule_prompt_groups": len(schedule_groups),
            "schedule_groups_by_dataset": dict(scheduled_dataset_groups),
            "schedule_process_eligible_groups": scheduled_process_groups,
            "schedule_process_eligible_trajectories": actual_process_trajectories,
            "reward_dry_sentinel": {
                "inputs": {"EM": em, "F1": f1, "centered_text": centered_text, "process": process_score},
                "ppo_t": dry_t, "ppo_tk": dry_tk, "delta": dry_tk - dry_t,
                "placement": {
                    "rearag": "0.3/n centered/clipped credit at each valid step end",
                    "outcome_and_optional_proofkg": "final generated token only",
                    "invalid": "-4 final only; text/process scorers not called",
                },
            },
            "rearag_prompt_tokens": rearag_prompt_audit,
        },
        "tests": test_result,
        "failures": failures,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
