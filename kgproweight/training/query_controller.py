"""Dedicated observation-conditioned query-controller SFT utilities.

This module intentionally does not reuse the query-planner prompt.  A
controller is trained to emit *one* canonical retrieval action from the
currently visible state; it is not trained to answer the QA question or to
emit an entire proof plan.  The v1 experiment keeps ``source_action=text`` so
that observation conditioning is the only method variable under test.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch

from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir
from kgproweight.utils.paths import model_path
from kgproweight.utils.seed import set_seed


CONTROLLER_SCHEMA_VERSION = "query-controller-action-v1"
CONTROLLER_SLOTS = ("q1", "q2_dynamic")
TARGET_KEYS = (
    "action",
    "query",
    "anchor",
    "relation_intent",
    "pid",
    "dependencies",
    "output_slot",
    "source_action",
)

CONTROLLER_SYSTEM_PROMPT = """You are a retrieval-query controller for multi-hop question answering.
Given only the current model-visible state, emit exactly one JSON retrieval action and nothing else.
Do not answer the original question. Do not invent an observation, entity identifier, or evidence.
The JSON keys, in order, must be: action, query, anchor, relation_intent, pid, dependencies, output_slot, source_action.
For this experiment action must be \"retrieve\" and source_action must be \"text\"."""

_PLACEHOLDER_RE = re.compile(
    r"(?:<[^<>]+>|\{[^{}]+\}|\$(?:hop|step)_?\d+|#\d+|"
    r"\b(?:TBD|TODO|PLACEHOLDER|UNKNOWN_ENTITY)\b)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryControllerTrainConfig:
    experiment_id: str
    train_path: str
    dev_path: str
    output_dir: str
    config_path: str | None = None
    protocol_path: str | None = None
    protocol_report_path: str | None = None
    protocol_manifest_path: str | None = None
    release_report_path: str | None = None
    release_manifest_path: str | None = None
    expected_protocol_sha256: str | None = None
    expected_protocol_report_sha256: str | None = None
    expected_protocol_manifest_sha256: str | None = None
    expected_train_sha256: str | None = None
    expected_dev_sha256: str | None = None
    expected_release_report_sha256: str | None = None
    expected_release_manifest_sha256: str | None = None
    expected_protocol_schema_version: str | None = None
    expected_protocol_report_schema_version: str | None = None
    expected_protocol_manifest_schema_version: str | None = None
    expected_protocol_status: str | None = None
    expected_protocol_experiment_id: str | None = None
    expected_release_schema_version: str | None = None
    expected_release_status: str | None = None
    expected_release_experiment_id: str | None = None
    schema_version: str = CONTROLLER_SCHEMA_VERSION
    base_model: str = "llama3-8B-instruct"
    method: str = "qlora"
    initialization: str = "base_instruct"
    init_adapter_path: str | None = None
    allowed_source_actions: tuple[str, ...] = ("text",)
    train_quotas: Mapping[str, Mapping[str, int]] | None = None
    dev_quotas: Mapping[str, Mapping[str, int]] | None = None
    require_all_records_selected: bool = False
    seed: int = 42
    max_seq_length: int = 1024
    batch_size: int = 1
    eval_batch_size: int = 1
    grad_accum: int = 16
    learning_rate: float = 1.0e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    epochs: float = 1.0
    max_steps: int = 20
    logging_steps: int = 1
    eval_strategy: str = "no"
    eval_steps: int = 20
    save_strategy: str = "no"
    save_steps: int = 20
    save_total_limit: int = 1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    load_in_4bit: bool = True
    dtype: str = "bf16"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    report_to: tuple[str, ...] = ("tensorboard",)
    logging_dir: str | None = None
    verify_saved_adapter_reload: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)


def _read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"controller action file does not exist: {path}")
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"controller row must be an object at {path}:{line_number}")
            yield row


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: str | Path, *, role: str) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"frozen {role} file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"frozen {role} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"frozen {role} must be a JSON object: {path}")
    return value


def _require_sha256(value: Any, *, role: str) -> str:
    digest = str(value or "")
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"missing or malformed expected SHA256 for {role}")
    return digest


def _verify_hash(path: str | Path, expected: Any, *, role: str) -> str:
    expected_digest = _require_sha256(expected, role=role)
    actual_digest = _sha256_file(path)
    if actual_digest != expected_digest:
        raise ValueError(
            f"frozen asset hash drift for {role}: expected={expected_digest}, "
            f"actual={actual_digest}, path={path}"
        )
    return actual_digest


def _manifest_run(document: Mapping[str, Any]) -> Mapping[str, Any]:
    run = document.get("run")
    return run if isinstance(run, Mapping) else document


def _identity_lock(
    document: Mapping[str, Any], *, role: str
) -> Mapping[str, Any]:
    values = document.get("identity_locks")
    if not isinstance(values, list):
        raise ValueError("release identity_locks must be a list")
    matches = [row for row in values if isinstance(row, Mapping) and row.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"release must contain exactly one identity lock for role={role}")
    return matches[0]


def _normalise_runtime_quotas(
    value: Mapping[str, Mapping[str, int]] | None,
) -> Dict[str, Dict[str, int]] | None:
    """Render quota mappings in the protocol's JSON-compatible shape."""

    if value is None:
        return None
    return {
        str(dataset): {str(slot): int(count) for slot, count in slots.items()}
        for dataset, slots in value.items()
    }


def _runtime_config_from_cfg(cfg: "QueryControllerTrainConfig") -> Dict[str, Any]:
    """Return every protocol-controlled probe runtime value.

    Artifact and logging paths are bound separately.  This object contains all
    values that can change model initialization, sample selection,
    optimization, or the scientific interpretation of the probe.
    """

    return {
        "phase": "probe",
        "experiment_id": cfg.experiment_id,
        "model": {
            "base_model": cfg.base_model,
            "method": cfg.method,
            "initialization": cfg.initialization,
            "init_adapter_path": cfg.init_adapter_path,
            "load_in_4bit": cfg.load_in_4bit,
            "dtype": cfg.dtype,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "target_modules": list(cfg.target_modules),
        },
        "data": {
            "allowed_source_actions": list(cfg.allowed_source_actions),
            "train_quotas": _normalise_runtime_quotas(cfg.train_quotas),
            "dev_quotas": _normalise_runtime_quotas(cfg.dev_quotas),
        },
        "training": {
            "seed": cfg.seed,
            "max_seq_length": cfg.max_seq_length,
            "per_device_train_batch_size": cfg.batch_size,
            "per_device_eval_batch_size": cfg.eval_batch_size,
            "gradient_accumulation_steps": cfg.grad_accum,
            "learning_rate": cfg.learning_rate,
            "warmup_ratio": cfg.warmup_ratio,
            "lr_scheduler_type": cfg.lr_scheduler_type,
            "num_train_epochs": cfg.epochs,
            "max_steps": cfg.max_steps,
            "weight_decay": cfg.weight_decay,
            "max_grad_norm": cfg.max_grad_norm,
            "logging_steps": cfg.logging_steps,
            "eval_strategy": cfg.eval_strategy,
            "eval_steps": cfg.eval_steps,
            "save_strategy": cfg.save_strategy,
            "save_steps": cfg.save_steps,
            "save_total_limit": cfg.save_total_limit,
            "verify_saved_adapter_reload": cfg.verify_saved_adapter_reload,
        },
    }


def _verify_implementation_locks(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    """Hash every implementation file declared by the frozen protocol."""

    locks = protocol.get("implementation_locks")
    if not isinstance(locks, Mapping) or not locks:
        raise ValueError("frozen Controller protocol has no implementation_locks")
    project_root = Path(__file__).resolve().parents[2]
    verified: Dict[str, Dict[str, Any]] = {}
    locked_relative_paths: set[str] = set()
    for role, raw in sorted(locks.items()):
        if not isinstance(raw, Mapping):
            raise ValueError(f"implementation lock {role!r} must be an object")
        raw_path = str(raw.get("path") or "")
        if not raw_path:
            raise ValueError(f"implementation lock {role!r} has no path")
        candidate = Path(raw_path)
        path = candidate if candidate.is_absolute() else project_root / candidate
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"locked implementation file is missing: role={role}, path={path}")
        try:
            relative_path = path.relative_to(project_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"locked implementation must be inside the project root: role={role}, path={path}"
            ) from exc
        digest = _verify_hash(path, raw.get("sha256"), role=f"implementation:{role}")
        locked_relative_paths.add(relative_path)
        verified[str(role)] = {"path": relative_path, "sha256": digest}
    required_paths = {
        "kgproweight/training/query_controller.py",
        "kgproweight/eval/query_controller_v1.py",
        "scripts/train/query_controller.py",
    }
    missing = sorted(required_paths - locked_relative_paths)
    if missing:
        raise ValueError(
            "frozen Controller protocol is missing required implementation locks: "
            f"{missing}"
        )
    return {"status": "PASS", "verified": verified}


def verify_frozen_assets(cfg: "QueryControllerTrainConfig") -> Dict[str, Any]:
    """Bind the trainer to the exact preregistered protocol and data release."""

    path_fields = {
        "protocol": cfg.protocol_path,
        "protocol_report": cfg.protocol_report_path,
        "protocol_manifest": cfg.protocol_manifest_path,
        "release_report": cfg.release_report_path,
        "release_manifest": cfg.release_manifest_path,
        "train": cfg.train_path,
        "dev": cfg.dev_path,
    }
    missing = [role for role, value in path_fields.items() if not value]
    if missing:
        raise ValueError(f"frozen Controller asset paths are required: missing={missing}")
    paths = {role: Path(str(value)) for role, value in path_fields.items()}
    expected_names = {
        "protocol": "protocol.json",
        "protocol_report": "report.json",
        "protocol_manifest": "manifest.json",
        "release_report": "report.json",
        "release_manifest": "manifest.json",
        "train": "train.jsonl",
        "dev": "dev.jsonl",
    }
    for role, expected_name in expected_names.items():
        if paths[role].name != expected_name:
            raise ValueError(
                f"frozen {role} path must end in {expected_name}, got {paths[role]}"
            )
    if paths["protocol_report"].parent.resolve() != paths["protocol"].parent.resolve():
        raise ValueError("protocol report must be a sibling of the frozen protocol")
    if paths["protocol_manifest"].parent.resolve() != paths["protocol"].parent.resolve():
        raise ValueError("protocol manifest must be a sibling of the frozen protocol")
    release_dir = paths["release_report"].parent.resolve()
    if paths["release_manifest"].parent.resolve() != release_dir:
        raise ValueError("release manifest must be a sibling of the frozen release report")
    if paths["train"].resolve() != release_dir / "train.jsonl":
        raise ValueError("train_path must be the exact train.jsonl beside the release report")
    if paths["dev"].resolve() != release_dir / "dev.jsonl":
        raise ValueError("dev_path must be the exact dev.jsonl beside the release report")

    expected_hashes = {
        "protocol": cfg.expected_protocol_sha256,
        "protocol_report": cfg.expected_protocol_report_sha256,
        "protocol_manifest": cfg.expected_protocol_manifest_sha256,
        "train": cfg.expected_train_sha256,
        "dev": cfg.expected_dev_sha256,
        "release_report": cfg.expected_release_report_sha256,
        "release_manifest": cfg.expected_release_manifest_sha256,
    }
    actual_hashes = {
        role: _verify_hash(paths[role], expected_hashes[role], role=role)
        for role in paths
    }
    protocol = _load_json_object(paths["protocol"], role="protocol")
    protocol_report = _load_json_object(paths["protocol_report"], role="protocol report")
    protocol_manifest = _load_json_object(paths["protocol_manifest"], role="protocol manifest")
    release_report = _load_json_object(paths["release_report"], role="release report")
    release_manifest = _load_json_object(paths["release_manifest"], role="release manifest")
    release_manifest_run = _manifest_run(release_manifest)

    protocol_status = str(cfg.expected_protocol_status or "")
    if not protocol_status.startswith("FROZEN_") or not protocol_status.endswith("_NOT_TRAINED"):
        raise ValueError("expected protocol status must be a frozen NOT_TRAINED status")
    if (
        protocol.get("schema_version") != cfg.expected_protocol_schema_version
        or protocol.get("status") != protocol_status
        or protocol.get("experiment_id") != cfg.expected_protocol_experiment_id
    ):
        raise ValueError("frozen Controller protocol schema/status gate failed")
    implementation_lock_report = _verify_implementation_locks(protocol)
    action_contract = protocol.get("action_contract") or {}
    if (
        action_contract.get("schema_version") != CONTROLLER_SCHEMA_VERSION
        or action_contract.get("source_action") != "text"
        or action_contract.get("dual_source_routing") is not False
        or action_contract.get("trainer_allowed_splits") != ["train", "dev"]
    ):
        raise ValueError("frozen Controller protocol action contract drifted")
    if protocol.get("enabled_training_datasets") != ["2wikimultihopqa", "musique"]:
        raise ValueError("frozen Controller protocol enabled datasets drifted")
    training_contract = protocol.get("training_contract") or {}
    probe_optimizer_steps = training_contract.get("probe_optimizer_steps")
    if (
        probe_optimizer_steps != 20
        or training_contract.get("confirmation_read_by_trainer") is not False
        or "assistant target JSON tokens only" not in str(training_contract.get("objective", ""))
        or "base Llama-3-8B instruct" not in str(training_contract.get("initialization", ""))
        or training_contract.get("target_truncation") != "forbidden"
    ):
        raise ValueError("frozen Controller training contract gate failed")
    probe_gates = training_contract.get("probe_gates") or {}
    if (
        probe_gates.get("adapter_save_reload_exact") is not True
        or probe_gates.get("adapter_save_fidelity_exact") is not True
        or probe_gates.get("adapter_clean_reload_single_adapter") is not True
        or probe_gates.get("adapter_clean_reload_tensor_exact") is not True
        or probe_gates.get("adapter_dtype_inventories_recorded") is not True
        or probe_gates.get(
            "adapter_saved_live_clean_reload_dtype_inventories_equal"
        )
        is not True
    ):
        raise ValueError(
            "frozen Controller protocol must require exact adapter save fidelity and "
            "clean single-adapter reload verification"
        )
    if cfg.max_steps != probe_optimizer_steps:
        raise ValueError(
            "runtime max_steps differs from frozen probe_optimizer_steps: "
            f"cfg={cfg.max_steps}, protocol={probe_optimizer_steps}"
        )
    if cfg.verify_saved_adapter_reload is not True:
        raise ValueError(
            "frozen Controller probe requires verify_saved_adapter_reload=true"
        )
    forbidden_initialization = set(training_contract.get("forbidden_initialization") or [])
    if cfg.initialization != "base_instruct" or (
        "historical query-planner adapter" not in forbidden_initialization
    ):
        raise ValueError(
            "this frozen probe protocol authorizes only an independent base_instruct LoRA"
        )
    frozen_runtime_config = training_contract.get("runtime_config")
    actual_runtime_config = _runtime_config_from_cfg(cfg)
    if not isinstance(frozen_runtime_config, Mapping):
        raise ValueError("frozen Controller protocol is missing training_contract.runtime_config")
    if dict(frozen_runtime_config) != actual_runtime_config:
        differing_sections = [
            key
            for key in sorted(set(frozen_runtime_config) | set(actual_runtime_config))
            if frozen_runtime_config.get(key) != actual_runtime_config.get(key)
        ]
        raise ValueError(
            "Controller runtime config differs from frozen protocol: "
            f"sections={differing_sections}"
        )
    data_gates = protocol.get("data_release_gates") or {}
    expected_protocol_gates = {
        "action_schema_valid_rate": 1.0,
        "dependency_closed_rate": 1.0,
        "duplicate_example_id_count": 0,
        "exact_actions_per_qid": 2,
        "gold_boundary_valid_rate": 1.0,
        "placeholder_free_rate": 1.0,
        "query_nonrepeat_rate": 1.0,
        "question_identity_join_rate": 1.0,
        "source_action_text_rate": 1.0,
        "state_use_valid_rate": 1.0,
    }
    if any(data_gates.get(key) != value for key, value in expected_protocol_gates.items()):
        raise ValueError("frozen Controller protocol data-release gates are incomplete or changed")
    if (
        protocol_report.get("status") != protocol_status
        or protocol_report.get("schema_version") != cfg.expected_protocol_report_schema_version
        or protocol_report.get("experiment_id") != cfg.expected_protocol_experiment_id
    ):
        raise ValueError("frozen Controller protocol report status gate failed")
    protocol_outputs = protocol_manifest.get("outputs") or {}
    if (
        protocol_manifest.get("status") != protocol_status
        or protocol_manifest.get("schema_version")
        != cfg.expected_protocol_manifest_schema_version
        or protocol_manifest.get("experiment_id") != cfg.expected_protocol_experiment_id
        or protocol_manifest.get("training_started") is not False
        or protocol_outputs.get("protocol.json") != actual_hashes["protocol"]
        or protocol_outputs.get("report.json") != actual_hashes["protocol_report"]
    ):
        raise ValueError("frozen Controller protocol manifest binding failed")

    if (
        release_report.get("schema_version") != cfg.expected_release_schema_version
        or release_report.get("status") != cfg.expected_release_status
        or release_report.get("experiment_id") != cfg.expected_release_experiment_id
        or release_report.get("all_release_gates_pass") is not True
    ):
        raise ValueError("Controller release status/all-gates gate failed")
    selection = release_report.get("selection") or {}
    if (
        selection.get("identity_authority") != "frozen_protocol_exact_join"
        or selection.get("source_action") != "text"
        or selection.get("one_qid_per_dataset_scoped_family") is not True
    ):
        raise ValueError("Controller release identity authority gate failed")
    release_checks = release_report.get("checks") or {}
    expected_release_checks = {
        "all_records_schema_valid": True,
        "two_actions_per_qid": True,
        "cross_split_family_overlap": 0,
        "cross_split_qid_overlap": 0,
        "excluded_family_overlap": 0,
        "excluded_qid_overlap": 0,
        "gold_final_answer_visible_count": 0,
        "evaluation_gold_access_count": 0,
        "source_action_values": ["text"],
    }
    if any(release_checks.get(key) != value for key, value in expected_release_checks.items()):
        raise ValueError("Controller release checks did not all pass")
    release_outputs = release_report.get("outputs") or {}
    if (
        (release_outputs.get("train.jsonl") or {}).get("sha256") != actual_hashes["train"]
        or (release_outputs.get("dev.jsonl") or {}).get("sha256") != actual_hashes["dev"]
    ):
        raise ValueError("Controller release report does not bind exact train/dev hashes")
    protocol_lock = _identity_lock(release_report, role="protocol")
    if protocol_lock.get("sha256") != actual_hashes["protocol"]:
        raise ValueError("Controller release protocol identity lock hash mismatch")
    stored_protocol_path = str(protocol_lock.get("path") or "").replace("\\", "/")
    expected_protocol_suffix = "/".join(paths["protocol"].parts[-2:])
    if not stored_protocol_path.endswith(expected_protocol_suffix):
        raise ValueError("Controller release protocol identity lock path mismatch")

    if (
        release_manifest.get("status") != cfg.expected_release_status
        or release_manifest_run.get("status") != cfg.expected_release_status
        or release_manifest_run.get("schema_version") != cfg.expected_release_schema_version
        or release_manifest_run.get("experiment_id") != cfg.expected_release_experiment_id
        or release_manifest_run.get("all_release_gates_pass") is not True
        or release_manifest_run.get("outputs") != release_outputs
        or release_manifest_run.get("selection") != selection
        or release_manifest_run.get("checks") != release_checks
        or release_manifest_run.get("identity_locks")
        != release_report.get("identity_locks")
        or release_manifest_run.get("gold_boundary")
        != release_report.get("gold_boundary")
    ):
        raise ValueError("Controller release manifest does not exactly bind the release report")
    manifest_protocol_lock = _identity_lock(release_manifest_run, role="protocol")
    if manifest_protocol_lock.get("sha256") != actual_hashes["protocol"]:
        raise ValueError("Controller release manifest protocol lock mismatch")

    return {
        "status": "PASS",
        "identity_authority": "frozen_protocol_exact_join",
        "protocol_status": protocol_status,
        "release_status": cfg.expected_release_status,
        "all_release_gates_pass": True,
        "runtime_config": actual_runtime_config,
        "implementation_locks": implementation_lock_report,
        "hashes": actual_hashes,
        "paths": {role: str(path.resolve()) for role, path in paths.items()},
    }


def _canonical_target(target: Mapping[str, Any]) -> str:
    missing = [key for key in TARGET_KEYS if key not in target]
    extras = sorted(set(target) - set(TARGET_KEYS))
    if missing or extras:
        raise ValueError(f"target keys mismatch: missing={missing}, extras={extras}")
    ordered = {key: target[key] for key in TARGET_KEYS}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def controller_messages(
    record: Mapping[str, Any], *, include_target: bool
) -> list[Dict[str, str]]:
    """Render only model-visible state plus, optionally, the exact JSON target."""

    visible = {
        "dataset": record["dataset"],
        "slot": record["slot"],
        "turn_index": record["turn_index"],
        "state": record["state"],
    }
    messages = [
        {"role": "system", "content": CONTROLLER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Current model-visible state:\n"
                + json.dumps(visible, ensure_ascii=False, separators=(",", ":"))
                + "\nEmit the next retrieval action JSON."
            ),
        },
    ]
    if include_target:
        messages.append({"role": "assistant", "content": _canonical_target(record["target"])})
    return messages


def encode_record(
    record: Mapping[str, Any], tokenizer: Any, *, max_seq_length: int
) -> Dict[str, Any]:
    """Tokenize one action while masking every non-assistant token.

    Sequences are rejected rather than truncated.  This makes it impossible
    for a long observation to silently clip the supervised JSON action.
    """

    prompt = tokenizer.apply_chat_template(
        controller_messages(record, include_target=False),
        tokenize=False,
        add_generation_prompt=True,
    )
    full = tokenizer.apply_chat_template(
        controller_messages(record, include_target=True),
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"chat-template prompt is not a prefix for example_id={record.get('example_id')}"
        )
    if len(full_ids) > max_seq_length:
        raise ValueError(
            f"target-preserving length gate failed for example_id={record.get('example_id')}: "
            f"{len(full_ids)} tokens > max_seq_length={max_seq_length}; no truncation allowed"
        )
    labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids) :])
    if not labels or not any(label != -100 for label in labels):
        raise ValueError(f"no assistant target tokens for example_id={record.get('example_id')}")
    return {
        "input_ids": list(full_ids),
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "length": len(full_ids),
        "prompt_length": len(prompt_ids),
        "supervised_length": len(full_ids) - len(prompt_ids),
    }


def _call_canonical_validator(record: Mapping[str, Any]) -> None:
    """Use the sole canonical schema validator owned by the eval protocol."""

    try:
        from kgproweight.eval.query_controller_v1 import validate_action_record
    except ImportError as exc:
        raise RuntimeError(
            "canonical Controller-v1 validator is unavailable: "
            "kgproweight.eval.query_controller_v1.validate_action_record"
        ) from exc
    result = validate_action_record(record)
    # The canonical validator normally raises on failure.  Supporting an
    # explicit boolean/error-list return keeps this consumer fail-closed if the
    # validator's reporting API evolves without duplicating its rules here.
    if result is False:
        raise ValueError(f"canonical schema validation failed: {record.get('example_id')}")
    if isinstance(result, (list, tuple)) and result:
        raise ValueError(
            f"canonical schema validation failed for {record.get('example_id')}: {list(result)}"
        )
    if isinstance(result, Mapping) and result.get("valid") is False:
        raise ValueError(
            f"canonical schema validation failed for {record.get('example_id')}: "
            f"{result.get('errors', result)}"
        )


def _normalise_query(text: Any) -> str:
    return " ".join(str(text or "").strip().casefold().split())


def _validate_training_contract(record: Mapping[str, Any], *, expected_split: str,
                                allowed_source_actions: Sequence[str]) -> None:
    _call_canonical_validator(record)
    if record.get("schema_version") != CONTROLLER_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version for {record.get('example_id')}: "
            f"{record.get('schema_version')!r}"
        )
    if record.get("split") != expected_split:
        raise ValueError(
            f"split mismatch for {record.get('example_id')}: expected {expected_split!r}, "
            f"got {record.get('split')!r}"
        )
    target = record["target"]
    query = str(target.get("query") or "").strip()
    if not query:
        raise ValueError(f"empty target query for {record.get('example_id')}")
    if _PLACEHOLDER_RE.search(query):
        raise ValueError(f"placeholder target query for {record.get('example_id')}: {query!r}")
    normalised = _normalise_query(query)
    state = record["state"]
    seen_queries = {_normalise_query(state.get("original_question"))}
    for previous in state.get("previous_actions", []):
        if isinstance(previous, Mapping):
            seen_queries.add(_normalise_query(previous.get("query")))
        elif isinstance(previous, str):
            seen_queries.add(_normalise_query(previous))
    seen_queries.discard("")
    if normalised in seen_queries:
        raise ValueError(
            f"target query repeats the original/prior query for {record.get('example_id')}"
        )
    source_action = str(target.get("source_action") or "")
    if source_action not in set(allowed_source_actions):
        raise ValueError(
            f"source_action={source_action!r} is outside the frozen v1 choices "
            f"{sorted(set(allowed_source_actions))} for {record.get('example_id')}"
        )


def _stable_rank(record: Mapping[str, Any], seed: int) -> str:
    key = f"{seed}\0{record.get('example_id')}\0{record.get('question_key')}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _pair_rank(dataset: str, qid: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{dataset}\0{qid}".encode("utf-8")).hexdigest()


def _complete_pairs(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, Dict[str, Any]]] = defaultdict(dict)
    for item in records:
        row = dict(item)
        identity = (str(row.get("dataset")), str(row.get("qid")))
        slot = str(row.get("slot"))
        if slot in grouped[identity]:
            raise ValueError(f"duplicate slot={slot} for dataset::qid={identity}")
        grouped[identity][slot] = row
    return {
        identity: by_slot
        for identity, by_slot in grouped.items()
        if set(by_slot) == set(CONTROLLER_SLOTS)
    }


def select_by_quotas(
    records: Sequence[Mapping[str, Any]],
    quotas: Mapping[str, Mapping[str, int]] | None,
    *,
    seed: int,
    require_all_records_selected: bool,
) -> list[Dict[str, Any]]:
    """Deterministically freeze arbitrary dataset-by-slot quotas."""

    rows = [dict(row) for row in records]
    complete = _complete_pairs(rows)
    if quotas is None:
        selected = [
            by_slot[slot]
            for identity in sorted(complete, key=lambda key: _pair_rank(*key, seed))
            for slot in CONTROLLER_SLOTS
            for by_slot in (complete[identity],)
        ]
        if require_all_records_selected and len(selected) != len(rows):
            raise ValueError(
                "require_all_records_selected=true but the release contains "
                "unpaired q1/q2_dynamic records"
            )
        return selected
    normalised_quotas: dict[tuple[str, str], int] = {}
    for dataset, slot_counts in quotas.items():
        if not isinstance(slot_counts, Mapping):
            raise ValueError(f"quota for {dataset!r} must map slot to count")
        for slot, value in slot_counts.items():
            if str(slot) not in CONTROLLER_SLOTS:
                raise ValueError(f"unsupported Controller slot in quota: {dataset}/{slot}")
            count = int(value)
            if count < 0:
                raise ValueError(f"negative quota for {dataset}/{slot}: {count}")
            normalised_quotas[(str(dataset), str(slot))] = count
    selected: list[Dict[str, Any]] = []
    datasets = sorted({dataset for dataset, _ in normalised_quotas})
    for dataset in datasets:
        dataset_quotas = {
            slot: normalised_quotas.get((dataset, slot)) for slot in CONTROLLER_SLOTS
        }
        if None in dataset_quotas.values() or len(set(dataset_quotas.values())) != 1:
            raise ValueError(
                f"paired Controller quotas require equal q1/q2_dynamic counts for {dataset}: "
                f"{dataset_quotas}"
            )
        need = int(dataset_quotas["q1"])
        candidates = sorted(
            (identity for identity in complete if identity[0] == dataset),
            key=lambda identity: _pair_rank(*identity, seed),
        )
        if len(candidates) < need:
            raise ValueError(
                f"paired quota underflow for dataset={dataset}: "
                f"available={len(candidates)}, required={need}"
            )
        for identity in candidates[:need]:
            selected.extend(complete[identity][slot] for slot in CONTROLLER_SLOTS)
    if require_all_records_selected and len(selected) != len(rows):
        selected_ids = {str(row.get("example_id")) for row in selected}
        omitted = [str(row.get("example_id")) for row in rows if str(row.get("example_id")) not in selected_ids]
        raise ValueError(
            f"require_all_records_selected=true but quotas omitted {len(omitted)} records; "
            f"examples={omitted[:5]}"
        )
    return selected


def sample_identity(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ids = [str(row["example_id"]) for row in records]
    return {
        "n": len(ids),
        "by_dataset": dict(sorted(Counter(str(row["dataset"]) for row in records).items())),
        "by_slot": dict(sorted(Counter(str(row["slot"]) for row in records).items())),
        "by_dataset_slot": dict(sorted(Counter(
            f"{row['dataset']}::{row['slot']}" for row in records
        ).items())),
        "example_id_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
    }


def _validate_unique(records: Sequence[Mapping[str, Any]], *, split: str) -> None:
    # q1 and q2_dynamic intentionally share a question_key; example_id includes
    # the slot and is the unique action identity.
    counts = Counter(str(row.get("example_id")) for row in records)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate example_id in {split}: {duplicates[:5]}")


def _check_split_isolation(
    train_records: Sequence[Mapping[str, Any]], dev_records: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    train_qids = {(str(row["dataset"]), str(row["qid"])) for row in train_records}
    dev_qids = {(str(row["dataset"]), str(row["qid"])) for row in dev_records}
    qid_overlap = sorted(train_qids & dev_qids)
    train_families = {
        (str(row["dataset"]), str(row["family_sha256"])) for row in train_records
    }
    dev_families = {
        (str(row["dataset"]), str(row["family_sha256"])) for row in dev_records
    }
    family_overlap = sorted(train_families & dev_families)
    if qid_overlap:
        raise ValueError(f"train/dev dataset::qid overlap: {qid_overlap[:5]}")
    if family_overlap:
        raise ValueError(f"train/dev family_sha256 overlap: {family_overlap[:5]}")
    return {
        "qid_overlap_count": 0,
        "family_overlap_count": 0,
        "train_unique_qids": len(train_qids),
        "dev_unique_qids": len(dev_qids),
        "train_unique_families": len(train_families),
        "dev_unique_families": len(dev_families),
    }


def _check_selected_pairs(records: Sequence[Mapping[str, Any]], *, split: str) -> Dict[str, Any]:
    complete = _complete_pairs(records)
    unique_qids = {(str(row["dataset"]), str(row["qid"])) for row in records}
    if len(complete) != len(unique_qids) or len(records) != 2 * len(unique_qids):
        broken = []
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for row in records:
            grouped[(str(row["dataset"]), str(row["qid"]))].append(str(row["slot"]))
        for identity, slots in sorted(grouped.items()):
            if sorted(slots) != sorted(CONTROLLER_SLOTS):
                broken.append({"dataset_qid": identity, "slots": sorted(slots)})
        raise ValueError(f"selected {split} records are not paired q1/q2_dynamic: {broken[:5]}")
    return {"paired_qids": len(unique_qids), "actions_per_qid": 2}


def _adapter_config_preflight(cfg: QueryControllerTrainConfig) -> Dict[str, Any] | None:
    if cfg.initialization not in {"base_instruct", "planner_adapter"}:
        raise ValueError(
            "initialization must be exactly 'base_instruct' or 'planner_adapter', "
            f"got {cfg.initialization!r}"
        )
    if cfg.initialization == "base_instruct":
        if cfg.init_adapter_path:
            raise ValueError("base_instruct initialization forbids init_adapter_path")
        return None
    if not cfg.init_adapter_path:
        raise ValueError("planner_adapter initialization requires init_adapter_path")
    path = Path(cfg.init_adapter_path)
    adapter_config_path = path / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise FileNotFoundError(f"planner adapter_config.json is missing: {adapter_config_path}")
    document = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    expected_targets = set(cfg.target_modules)
    actual_targets = set(document.get("target_modules") or [])
    if (
        int(document.get("r", -1)) != cfg.lora_r
        or int(document.get("lora_alpha", -1)) != cfg.lora_alpha
        or actual_targets != expected_targets
    ):
        raise ValueError(
            "planner adapter LoRA configuration differs from Controller config: "
            f"r={document.get('r')}, alpha={document.get('lora_alpha')}, "
            f"targets={sorted(actual_targets)}"
        )
    return {
        "path": str(path.resolve()),
        "r": int(document["r"]),
        "lora_alpha": int(document["lora_alpha"]),
        "target_modules": sorted(actual_targets),
    }


def preflight_records(
    cfg: QueryControllerTrainConfig,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, Any]]:
    """Run all data/scientific gates without loading a model or using CUDA."""

    if cfg.schema_version != CONTROLLER_SCHEMA_VERSION:
        raise ValueError(
            f"config schema_version must be {CONTROLLER_SCHEMA_VERSION!r}, got {cfg.schema_version!r}"
        )
    if tuple(cfg.allowed_source_actions) != ("text",):
        raise ValueError(
            "Controller-v1 is a single-variable text-action experiment; "
            "allowed_source_actions must be exactly ['text']"
        )
    asset_lock_report = verify_frozen_assets(cfg)
    adapter_report = _adapter_config_preflight(cfg)
    raw_train = list(_read_jsonl(cfg.train_path))
    raw_dev = list(_read_jsonl(cfg.dev_path))
    if not raw_train or not raw_dev:
        raise ValueError("controller train and dev files must both be non-empty")
    for row in raw_train:
        _validate_training_contract(
            row, expected_split="train", allowed_source_actions=cfg.allowed_source_actions
        )
    for row in raw_dev:
        _validate_training_contract(
            row, expected_split="dev", allowed_source_actions=cfg.allowed_source_actions
        )
    _validate_unique(raw_train, split="train")
    _validate_unique(raw_dev, split="dev")
    raw_isolation = _check_split_isolation(raw_train, raw_dev)
    train = select_by_quotas(
        raw_train,
        cfg.train_quotas,
        seed=cfg.seed,
        require_all_records_selected=cfg.require_all_records_selected,
    )
    dev = select_by_quotas(
        raw_dev,
        cfg.dev_quotas,
        seed=cfg.seed + 1,
        require_all_records_selected=cfg.require_all_records_selected,
    )
    if not train or not dev:
        raise ValueError("quota selection must retain at least one paired qid in train and dev")
    selected_isolation = _check_split_isolation(train, dev)
    selected_pairs = {
        "train": _check_selected_pairs(train, split="train"),
        "dev": _check_selected_pairs(dev, split="dev"),
    }
    return train, dev, {
        "status": "PASS",
        "schema_version": CONTROLLER_SCHEMA_VERSION,
        "initialization": cfg.initialization,
        "asset_lock": asset_lock_report,
        "adapter_preflight": adapter_report,
        "allowed_source_actions": list(cfg.allowed_source_actions),
        "raw": {
            "train": sample_identity(raw_train),
            "dev": sample_identity(raw_dev),
            "isolation": raw_isolation,
        },
        "selected": {
            "train": sample_identity(train),
            "dev": sample_identity(dev),
            "isolation": selected_isolation,
            "pairing": selected_pairs,
        },
        "gates": {
            "canonical_schema_valid": True,
            "query_nonrepetition_and_no_placeholder": True,
            "gold_boundary_valid": True,
            "source_action_text_only": True,
            "train_dev_qid_overlap_zero": True,
            "train_dev_family_overlap_zero": True,
            "selected_q1_q2_paired": True,
        },
    }


def _encode_records(records: Sequence[Mapping[str, Any]], tokenizer: Any,
                    *, max_seq_length: int):
    import datasets

    rows = [
        encode_record(record, tokenizer, max_seq_length=max_seq_length)
        for record in records
    ]
    return datasets.Dataset.from_list(rows)


def _length_summary(dataset: Any) -> Dict[str, Any]:
    lengths = sorted(int(value) for value in dataset["length"])
    supervised = sorted(int(value) for value in dataset["supervised_length"])

    def percentile(values: Sequence[int], fraction: float) -> int:
        return values[min(len(values) - 1, int((len(values) - 1) * fraction))]

    return {
        "n": len(lengths),
        "total_tokens": sum(lengths),
        "length": {
            "min": lengths[0],
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "max": lengths[-1],
        },
        "supervised_length": {
            "min": supervised[0],
            "p50": percentile(supervised, 0.50),
            "p95": percentile(supervised, 0.95),
            "max": supervised[-1],
        },
    }


def prepare_data(
    cfg: QueryControllerTrainConfig, tokenizer: Any
) -> tuple[Any, Any, Dict[str, Any]]:
    train, dev, report = preflight_records(cfg)
    train_dataset = _encode_records(train, tokenizer, max_seq_length=cfg.max_seq_length)
    dev_dataset = _encode_records(dev, tokenizer, max_seq_length=cfg.max_seq_length)
    report["token_lengths"] = {
        "train": _length_summary(train_dataset),
        "dev": _length_summary(dev_dataset),
    }
    report["gates"]["target_not_truncated"] = True
    return train_dataset, dev_dataset, report


def _quota_document(value: Any, *, label: str) -> Mapping[str, Mapping[str, int]] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a dataset -> slot -> count mapping")
    return {
        str(dataset): {str(slot): int(count) for slot, count in slots.items()}
        for dataset, slots in value.items()
    }


def load_config(
    path: str | Path,
    *,
    output_override: str | None = None,
    max_steps_override: int | None = None,
) -> QueryControllerTrainConfig:
    import yaml

    path = Path(path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiment = document["experiment"]
    data = document["data"]
    model = document["model"]
    training = document["training"]
    logging = document.get("logging", {})
    asset_lock = document.get("asset_lock", {})
    expected_sha256 = asset_lock.get("expected_sha256", {})
    expected_identity = asset_lock.get("expected_identity", {})
    max_steps = int(
        max_steps_override if max_steps_override is not None else training.get("max_steps", -1)
    )
    report_to_value = logging.get("report_to", ["tensorboard"])
    if isinstance(report_to_value, str):
        report_to_value = [report_to_value]
    configured_output = str(training["output_dir"])
    configured_logging_dir = (
        str(logging["logging_dir"]) if logging.get("logging_dir") else None
    )
    if (
        output_override
        and configured_logging_dir
        and Path(configured_logging_dir) == Path(configured_output) / "tensorboard"
    ):
        configured_logging_dir = str(Path(output_override) / "tensorboard")
    cfg = QueryControllerTrainConfig(
        experiment_id=str(experiment["id"]),
        train_path=str(data["train_path"]),
        dev_path=str(data["dev_path"]),
        output_dir=str(output_override or configured_output),
        config_path=str(path),
        protocol_path=(str(asset_lock["protocol_path"]) if asset_lock.get("protocol_path") else None),
        protocol_report_path=(
            str(asset_lock["protocol_report_path"])
            if asset_lock.get("protocol_report_path") else None
        ),
        protocol_manifest_path=(
            str(asset_lock["protocol_manifest_path"])
            if asset_lock.get("protocol_manifest_path") else None
        ),
        release_report_path=(
            str(asset_lock["release_report_path"])
            if asset_lock.get("release_report_path") else None
        ),
        release_manifest_path=(
            str(asset_lock["release_manifest_path"])
            if asset_lock.get("release_manifest_path") else None
        ),
        expected_protocol_sha256=expected_sha256.get("protocol"),
        expected_protocol_report_sha256=expected_sha256.get("protocol_report"),
        expected_protocol_manifest_sha256=expected_sha256.get("protocol_manifest"),
        expected_train_sha256=expected_sha256.get("train"),
        expected_dev_sha256=expected_sha256.get("dev"),
        expected_release_report_sha256=expected_sha256.get("release_report"),
        expected_release_manifest_sha256=expected_sha256.get("release_manifest"),
        expected_protocol_schema_version=expected_identity.get("protocol_schema_version"),
        expected_protocol_report_schema_version=expected_identity.get(
            "protocol_report_schema_version"
        ),
        expected_protocol_manifest_schema_version=expected_identity.get(
            "protocol_manifest_schema_version"
        ),
        expected_protocol_status=expected_identity.get("protocol_status"),
        expected_protocol_experiment_id=expected_identity.get("protocol_experiment_id"),
        expected_release_schema_version=expected_identity.get("release_schema_version"),
        expected_release_status=expected_identity.get("release_status"),
        expected_release_experiment_id=expected_identity.get("release_experiment_id"),
        schema_version=str(data.get("schema_version", CONTROLLER_SCHEMA_VERSION)),
        base_model=str(model["base_model"]),
        method=str(model.get("method", "")),
        initialization=str(model["initialization"]),
        init_adapter_path=(
            str(model["init_adapter_path"]) if model.get("init_adapter_path") else None
        ),
        allowed_source_actions=tuple(data.get("allowed_source_actions", ["text"])),
        train_quotas=_quota_document(data.get("quotas", {}).get("train"), label="train quotas"),
        dev_quotas=_quota_document(data.get("quotas", {}).get("dev"), label="dev quotas"),
        require_all_records_selected=bool(data.get("require_all_records_selected", False)),
        seed=int(training.get("seed", 42)),
        max_seq_length=int(training["max_seq_length"]),
        batch_size=int(training["per_device_train_batch_size"]),
        eval_batch_size=int(training.get("per_device_eval_batch_size", 1)),
        grad_accum=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(training.get("lr_scheduler_type", "")),
        epochs=float(training.get("num_train_epochs", 1.0)),
        max_steps=max_steps,
        logging_steps=int(training.get("logging_steps", 1)),
        eval_strategy=str(training.get("eval_strategy", "no")),
        eval_steps=int(training.get("eval_steps", 20)),
        save_strategy=str(training.get("save_strategy", "no")),
        save_steps=int(training.get("save_steps", 20)),
        save_total_limit=int(training.get("save_total_limit", 1)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        load_in_4bit=bool(model.get("load_in_4bit", True)),
        dtype=str(model.get("dtype", "bf16")),
        lora_r=int(model["lora_r"]),
        lora_alpha=int(model["lora_alpha"]),
        lora_dropout=float(model["lora_dropout"]),
        target_modules=tuple(model["target_modules"]),
        report_to=tuple(str(value) for value in report_to_value),
        logging_dir=configured_logging_dir,
        verify_saved_adapter_reload=bool(training.get("verify_saved_adapter_reload", True)),
        extra={"experiment": dict(experiment), "guardrails": dict(document.get("guardrails", {}))},
    )
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: QueryControllerTrainConfig) -> None:
    if not cfg.experiment_id.strip():
        raise ValueError("experiment.id must be non-empty")
    if cfg.max_steps == 0 or cfg.max_steps < -1:
        raise ValueError("max_steps must be -1 or a positive integer")
    if cfg.max_seq_length <= 0 or cfg.batch_size <= 0 or cfg.grad_accum <= 0:
        raise ValueError("sequence length, batch size, and gradient accumulation must be positive")
    if cfg.dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"unsupported dtype: {cfg.dtype}")
    if cfg.method != "qlora":
        raise ValueError(f"Controller probe method must be exactly 'qlora', got {cfg.method!r}")
    if cfg.lr_scheduler_type != "cosine":
        raise ValueError(
            "Controller probe lr_scheduler_type must be exactly 'cosine', "
            f"got {cfg.lr_scheduler_type!r}"
        )
    if cfg.verify_saved_adapter_reload is not True:
        raise ValueError("Controller probe requires verify_saved_adapter_reload=true")
    if cfg.eval_strategy not in {"no", "steps", "epoch"}:
        raise ValueError(f"unsupported eval_strategy: {cfg.eval_strategy}")
    if cfg.save_strategy not in {"no", "steps", "epoch"}:
        raise ValueError(f"unsupported save_strategy: {cfg.save_strategy}")
    if not cfg.report_to:
        raise ValueError("at least one logging backend is required")
    if "tensorboard" in cfg.report_to and not cfg.logging_dir:
        raise ValueError("TensorBoard reporting requires an explicit logging.logging_dir")
    for role, value in {
        "protocol_path": cfg.protocol_path,
        "protocol_report_path": cfg.protocol_report_path,
        "protocol_manifest_path": cfg.protocol_manifest_path,
        "release_report_path": cfg.release_report_path,
        "release_manifest_path": cfg.release_manifest_path,
        "expected_protocol_sha256": cfg.expected_protocol_sha256,
        "expected_protocol_report_sha256": cfg.expected_protocol_report_sha256,
        "expected_protocol_manifest_sha256": cfg.expected_protocol_manifest_sha256,
        "expected_train_sha256": cfg.expected_train_sha256,
        "expected_dev_sha256": cfg.expected_dev_sha256,
        "expected_release_report_sha256": cfg.expected_release_report_sha256,
        "expected_release_manifest_sha256": cfg.expected_release_manifest_sha256,
        "expected_protocol_schema_version": cfg.expected_protocol_schema_version,
        "expected_protocol_report_schema_version": cfg.expected_protocol_report_schema_version,
        "expected_protocol_manifest_schema_version": cfg.expected_protocol_manifest_schema_version,
        "expected_protocol_status": cfg.expected_protocol_status,
        "expected_protocol_experiment_id": cfg.expected_protocol_experiment_id,
        "expected_release_schema_version": cfg.expected_release_schema_version,
        "expected_release_status": cfg.expected_release_status,
        "expected_release_experiment_id": cfg.expected_release_experiment_id,
    }.items():
        if not value:
            raise ValueError(f"frozen asset lock config is missing {role}")
    _adapter_config_preflight(cfg)


def dry_run(cfg: QueryControllerTrainConfig, tokenizer: Any | None = None) -> Dict[str, Any]:
    """CPU preflight; optionally include the no-truncation token gate."""

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path(cfg.base_model))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    _, _, report = prepare_data(cfg, tokenizer)
    return report


def _validate_training_logs(log_history: Sequence[Mapping[str, Any]], train_loss: float) -> Dict[str, Any]:
    loss_values = [float(row["loss"]) for row in log_history if "loss" in row]
    grad_values = [float(row["grad_norm"]) for row in log_history if row.get("grad_norm") is not None]
    if not math.isfinite(float(train_loss)) or not loss_values or not all(map(math.isfinite, loss_values)):
        raise RuntimeError("non-finite or missing training loss in Controller training history")
    if not grad_values or not all(map(math.isfinite, grad_values)):
        raise RuntimeError("non-finite or missing gradient norm in Controller training history")
    if not any(value > 0.0 for value in grad_values):
        raise RuntimeError("no nonzero trainable gradient was observed in Controller training")
    return {
        "finite_train_loss": True,
        "finite_logged_losses": True,
        "finite_gradient_norms": True,
        "nonzero_trainable_gradient_observed": True,
        "logged_loss_rows": len(loss_values),
        "logged_gradient_rows": len(grad_values),
    }


def _validate_optimizer_steps(global_step: int, cfg: QueryControllerTrainConfig) -> Dict[str, Any]:
    """Require the probe to reach exactly its frozen optimizer-step count."""

    actual = int(global_step)
    expected = int(cfg.max_steps)
    if actual != expected:
        raise RuntimeError(
            "Controller probe optimizer-step mismatch: "
            f"expected={expected}, actual={actual}"
        )
    return {
        "expected_optimizer_steps": expected,
        "actual_optimizer_steps": actual,
        "exact_optimizer_steps_pass": True,
    }


def _assert_adapter_state_exact(
    expected: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
    *,
    gate: str,
) -> None:
    """Fail with stable diagnostics unless two adapter states are byte-exact."""

    if set(expected) != set(actual):
        raise RuntimeError(
            f"{gate} adapter tensor keys differ: "
            f"expected_only={sorted(set(expected) - set(actual))[:3]}, "
            f"actual_only={sorted(set(actual) - set(expected))[:3]}"
        )
    unequal = [
        key for key in sorted(expected) if not torch.equal(expected[key], actual[key])
    ]
    if unequal:
        raise RuntimeError(f"{gate} adapter tensors differ: {unequal[:3]}")


def _validate_saved_adapter(final_dir: Path, *, live_model: Any | None = None) -> Dict[str, Any]:
    """Verify save fidelity, then reload into a clean single-adapter base.

    Loading a second named adapter beside the trained adapter is intentionally
    forbidden here: PEFT's shared module namespaces can make that comparison
    report false tensor mismatches.  The trained adapter is first compared to
    the on-disk state, then unloaded; only the saved adapter is attached to the
    resulting clean base for an independent disk-reload comparison.
    """

    from peft import PeftConfig, PeftModel, get_peft_model_state_dict
    from safetensors import safe_open
    from safetensors.torch import load_file

    peft_config = PeftConfig.from_pretrained(str(final_dir))
    weights = final_dir / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"saved Controller adapter is missing: {weights}")
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if not keys:
            raise RuntimeError("saved Controller adapter has no tensors")
        first_shape = list(handle.get_tensor(keys[0]).shape)
    save_fidelity_exact = False
    clean_reload_tensor_exact = False
    clean_reload_single_adapter = False
    saved_dtype_inventory: Dict[str, int] = {}
    live_dtype_inventory: Dict[str, int] = {}
    clean_reload_dtype_inventory: Dict[str, int] = {}
    if live_model is not None:
        saved = {
            key: value.detach().cpu()
            for key, value in load_file(str(weights), device="cpu").items()
        }
        saved_dtype_inventory = dict(
            sorted(Counter(str(value.dtype) for value in saved.values()).items())
        )
        active_adapter = live_model.active_adapter
        if isinstance(active_adapter, (list, tuple)):
            if len(active_adapter) != 1:
                raise RuntimeError(
                    "save-fidelity check requires exactly one active trained adapter"
                )
            active_adapter = active_adapter[0]
        if not isinstance(active_adapter, str) or not active_adapter:
            raise RuntimeError("save-fidelity check could not identify the active adapter")
        live_state = get_peft_model_state_dict(
            live_model, adapter_name=active_adapter, save_embedding_layers=False
        )
        live_state = {key: value.detach().cpu() for key, value in live_state.items()}
        live_dtype_inventory = dict(
            sorted(Counter(str(value.dtype) for value in live_state.values()).items())
        )
        if live_dtype_inventory != saved_dtype_inventory:
            raise RuntimeError(
                "save_fidelity adapter dtype inventory differs: "
                f"saved={saved_dtype_inventory}, live={live_dtype_inventory}"
            )
        _assert_adapter_state_exact(saved, live_state, gate="save_fidelity")
        save_fidelity_exact = True

        # ``unload`` removes every PEFT adapter from the trained wrapper and
        # returns its underlying base.  Reloading from disk on that base gives
        # a clean, single-adapter path without allocating a second 8B base.
        clean_base = live_model.unload()
        reloaded_model = PeftModel.from_pretrained(
            clean_base,
            str(final_dir),
            is_trainable=False,
            # Match the actual inference runner and PEFT's safe default:
            # half/bfloat16 adapter parameters are promoted to float32.  The
            # saved Controller adapter is float32, so this preserves exact
            # on-disk values instead of introducing a validation-only BF16
            # round trip.
            autocast_adapter_dtype=True,
        )
        adapter_names = list(reloaded_model.peft_config)
        if len(adapter_names) != 1:
            raise RuntimeError(
                "clean reload must contain exactly one adapter, got "
                f"{adapter_names}"
            )
        clean_reload_single_adapter = True
        reloaded = get_peft_model_state_dict(
            reloaded_model,
            adapter_name=adapter_names[0],
            save_embedding_layers=False,
        )
        reloaded = {key: value.detach().cpu() for key, value in reloaded.items()}
        clean_reload_dtype_inventory = dict(
            sorted(Counter(str(value.dtype) for value in reloaded.values()).items())
        )
        if clean_reload_dtype_inventory != saved_dtype_inventory:
            raise RuntimeError(
                "clean_reload adapter dtype inventory differs: "
                f"saved={saved_dtype_inventory}, reloaded={clean_reload_dtype_inventory}"
            )
        _assert_adapter_state_exact(saved, reloaded, gate="clean_reload")
        clean_reload_tensor_exact = True
    return {
        "reloadable_peft_config": True,
        "adapter_tensor_count": len(keys),
        "first_tensor_shape": first_shape,
        "peft_type": str(peft_config.peft_type),
        "save_fidelity_exact": save_fidelity_exact,
        "clean_reload_single_adapter": clean_reload_single_adapter,
        "clean_reload_tensor_exact": clean_reload_tensor_exact,
        "saved_dtype_inventory": saved_dtype_inventory,
        "live_dtype_inventory": live_dtype_inventory,
        "clean_reload_dtype_inventory": clean_reload_dtype_inventory,
        "dtype_inventories_recorded": bool(
            saved_dtype_inventory
            and live_dtype_inventory
            and clean_reload_dtype_inventory
        ),
        "saved_live_clean_reload_dtype_inventories_equal": bool(
            saved_dtype_inventory
            and saved_dtype_inventory
            == live_dtype_inventory
            == clean_reload_dtype_inventory
        ),
        # Backwards-compatible aggregate name; v4.3 additionally binds the
        # two explicit gates above.
        "tensor_reload_exact": save_fidelity_exact and clean_reload_tensor_exact,
        "adapter_file_sha256": _sha256_file(weights),
    }


def run_query_controller_sft(
    cfg: QueryControllerTrainConfig, *, probe: bool = False
) -> Dict[str, Any]:
    """Run dedicated Controller QLoRA after all CPU preflight gates pass."""

    _validate_config(cfg)
    if probe is not True:
        raise ValueError(
            "this frozen Controller training entry point authorizes the 20-step probe only; "
            "run_query_controller_sft(..., probe=True) is required"
        )
    # All immutable protocol/release/hash/schema/isolation gates must pass
    # before even querying CUDA state.  This keeps a malformed or tampered run
    # from touching the accelerator and makes preflight genuinely fail-closed.
    preflight_records(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("query Controller SFT requires a CUDA-visible process")
    if cfg.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("active GPU does not support bf16")

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    set_seed(cfg.seed)
    base_id = model_path(cfg.base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # This includes schema/isolation/query/gold gates and rejects target
    # truncation before the output directory or CUDA model is created.
    train_dataset, dev_dataset, data_report = prepare_data(cfg, tokenizer)

    out_dir, experiment_id = prepare_new_run_dir(
        cfg.output_dir,
        experiment_id=cfg.experiment_id,
        extra={
            "phase": "query_controller_sft_probe" if probe else "query_controller_sft",
            "config": asdict(cfg),
            "input_artifacts": {
                "config": artifact_identity(cfg.config_path) if cfg.config_path else None,
                "protocol": artifact_identity(cfg.protocol_path),
                "protocol_report": artifact_identity(cfg.protocol_report_path),
                "protocol_manifest": artifact_identity(cfg.protocol_manifest_path),
                "release_report": artifact_identity(cfg.release_report_path),
                "release_manifest": artifact_identity(cfg.release_manifest_path),
                "train": artifact_identity(cfg.train_path),
                "dev": artifact_identity(cfg.dev_path),
                "base_model": artifact_identity(base_id),
                "init_adapter": (
                    artifact_identity(cfg.init_adapter_path) if cfg.init_adapter_path else None
                ),
            },
        },
    )
    try:
        (out_dir / "data_report.json").write_text(
            json.dumps(data_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.dtype]
        quantization_config = None
        if cfg.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            device_map={"": 0},
        )
        if cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )
        if cfg.initialization == "planner_adapter":
            model = PeftModel.from_pretrained(
                model, str(cfg.init_adapter_path), is_trainable=True
            )
        else:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=cfg.lora_r,
                    lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    target_modules=list(cfg.target_modules),
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                ),
            )
        model.config.use_cache = False
        model.enable_input_require_grads()
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        if trainable_parameters <= 0:
            raise RuntimeError("Controller model has no trainable parameters")

        arguments = TrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            num_train_epochs=cfg.epochs,
            max_steps=cfg.max_steps,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            bf16=cfg.dtype == "bf16",
            fp16=cfg.dtype == "fp16",
            logging_steps=cfg.logging_steps,
            logging_first_step=True,
            eval_strategy=cfg.eval_strategy,
            eval_steps=cfg.eval_steps,
            save_strategy=cfg.save_strategy,
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            report_to=list(cfg.report_to),
            logging_dir=cfg.logging_dir,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_grad_norm=cfg.max_grad_norm,
            seed=cfg.seed,
            data_seed=cfg.seed,
            group_by_length=True,
        )
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding="longest", label_pad_token_id=-100
        )
        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset if cfg.eval_strategy != "no" else None,
            data_collator=collator,
        )
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        train_result = trainer.train()
        elapsed = time.perf_counter() - started

        history_path = out_dir / "training_history.jsonl"
        with history_path.open("w", encoding="utf-8") as fh:
            for row in trainer.state.log_history:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log_gates = _validate_training_logs(
            trainer.state.log_history, float(train_result.training_loss)
        )
        optimizer_step_gate = _validate_optimizer_steps(trainer.state.global_step, cfg)
        final_dir = out_dir / "final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(final_dir)
        reload_report = (
            _validate_saved_adapter(final_dir, live_model=model)
            if cfg.verify_saved_adapter_reload
            else {"reload_check": "disabled_by_config"}
        )
        throughput = {
            "elapsed_seconds": elapsed,
            "global_steps": int(trainer.state.global_step),
            "seconds_per_optimizer_step": elapsed / max(1, trainer.state.global_step),
            "train_loss": float(train_result.training_loss),
            "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
            "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 1024 ** 3,
            "log_gates": log_gates,
            "optimizer_step_gate": optimizer_step_gate,
            "exact_optimizer_steps_pass": True,
            "saved_adapter": reload_report,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
        }
        throughput_path = out_dir / "throughput.json"
        throughput_path.write_text(
            json.dumps(throughput, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        dump_manifest(
            out_dir,
            status="COMPLETE",
            extra={
                "phase": "query_controller_sft_probe" if probe else "query_controller_sft",
                "experiment_id": experiment_id,
                "config": asdict(cfg),
                "data_report": data_report,
                "throughput": throughput,
                "output_artifacts": {
                    "final": artifact_identity(final_dir),
                    "history": artifact_identity(history_path),
                    "tensorboard": (
                        artifact_identity(cfg.logging_dir) if cfg.logging_dir else None
                    ),
                },
            },
        )
        return {"output_dir": str(out_dir), "final": str(final_dir), **throughput}
    except Exception as exc:
        dump_manifest(
            out_dir,
            status="FAIL_STOP",
            extra={
                "phase": "query_controller_sft_probe" if probe else "query_controller_sft",
                "experiment_id": experiment_id,
                "config": asdict(cfg),
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "data_report": data_report,
            },
        )
        raise
