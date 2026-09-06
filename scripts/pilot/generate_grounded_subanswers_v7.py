#!/usr/bin/env python
"""Generate passage-local, fail-closed v7 subanswers without reading Gold.

The input is an append-only JSONL of C-arm producer tasks emitted by the v7
retrieval runner.  This program validates every task before loading a model,
uses only the dedicated :mod:`kgproweight.retrieval.subanswer_v7` prompt and
parser/verifier, and writes a new run directory.  It never opens a dataset or
scorer-Gold artifact.

The production path loads the base Llama-3 model and the frozen strong-SFT
adapter in BF16 on CUDA.  Tests can inject a fake generator through
``run_generation`` or call ``generate_subanswer_rows`` directly; the CLI never
offers a fake or CPU fallback.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from kgproweight.kg.question_kg import question_sha256
from kgproweight.retrieval.subanswer_v7 import (
    PARSER_VERSION,
    PROMPT_VERSION,
    VERIFIER_VERSION,
    SubanswerParseError,
    build_subanswer_reader_messages,
    parse_and_verify_subanswer,
    parse_subanswer_response,
)
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir


RUNNER_VERSION = "grounded-subanswers-v7-1"
OUTPUT_SCHEMA_VERSION = "dependent-retrieval-v7-subanswer-output-1"
MAX_NEW_TOKENS = 96
MAX_PRODUCER_PASSAGES = 10
MAX_PASSAGE_TEXT_CHARS = 1200
DEFAULT_SEED = 42
DEFAULT_BASE_MODEL = Path("models/llama3-8b")
DEFAULT_ADAPTER = Path(
    "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"
)
DEFAULT_IMPLEMENTATION_LOCK = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_implementation_lock_v1/protocol.json"
)
DEFAULT_PLAN_LOCK = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_plans_lock_v1/protocol.json"
)
IMPLEMENTATION_LOCK_SCHEMA = "subquestion-dependent-retrieval-v7-implementation-lock-1"
IMPLEMENTATION_LOCK_STATUS = "AUTHORIZED_PLANNER_ONLY"
PLAN_LOCK_SCHEMA = "subquestion-dependent-retrieval-v7-plan-lock-1"
PLAN_LOCK_STATUS = "AUTHORIZED_GOLD_FREE_MATERIALIZATION"
EXECUTION_SCOPE = "DEVELOPMENT_ONLY_GLOBALLY_CONSUMED_HOTPOT20_MUSIQUE20"
MODEL_ARTIFACT_SCHEMA = "dependent-retrieval-v7-strong-sft-model-content-lock-1"
STAGE_DESCRIPTOR_SCHEMA = "paired-dependent-retrieval-v7-stage-descriptor-2"

TARGET_TYPES = {
    "hotpotqa": "relation_graph",
    "musique": "subquery_graph",
}

# These are exactly the keys the retrieval runner is authorized to place in a
# C task.  An exact outer schema prevents a superficially harmless extra field
# from becoming a label or decomposition side channel.
TASK_KEYS = frozenset(
    {
        "task_id",
        "question_key",
        "dataset",
        "qid",
        "question",
        "question_sha256",
        "target_type",
        "producer_slot",
        "step",
        "step_sha256",
        "producer_passages",
        "producer_passages_sha256",
        "gold_access",
    }
)

OUTPUT_IDENTITY_KEYS = (
    "task_id",
    "question_key",
    "dataset",
    "qid",
    "question_sha256",
    "target_type",
    "producer_slot",
    "step_sha256",
    "producer_passages_sha256",
)

# Frozen by the v7 Gold policy.  Matching is case-insensitive and recursive.
# Text values are deliberately not scanned against a Gold answer list: loading
# such a list here would itself violate the stage boundary.
FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "answers",
        "decomposition",
        "evidence",
        "evidences",
        "gold_answer",
        "gold_answers",
        "gold_target",
        "golden_answers",
        "paragraph_text",
        "question_decomposition",
        "reasoning",
        "sp",
        "supporting_facts",
        "target",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TaskValidationError(ValueError):
    """A caller-owned C task violates the frozen input contract."""


class _DuplicateJSONKey(ValueError):
    pass


@dataclass(frozen=True)
class GenerationOutput:
    """One response and the token accounting returned by a generator backend."""

    prompt: str
    response_text: str
    prompt_tokens: int
    generation_tokens: int
    runtime_telemetry: Mapping[str, Any] = field(default_factory=dict)


class GeneratorBackend(Protocol):
    """Small injectable boundary used by the production and fake generators."""

    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int,
        do_sample: bool,
    ) -> GenerationOutput: ...


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(key)
        result[key] = value
    return result


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical bytes used for v7 step and producer-passage commitments."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _assert_prompt_document_projection(
    messages: Sequence[Mapping[str, str]],
    producer_passages: Sequence[Mapping[str, Any]],
) -> None:
    """Prove that the reader prompt contains the exact verifier documents.

    ``build_subanswer_reader_messages`` deliberately performs its own narrow
    document projection.  The v7 truncation addendum requires that projection
    to be byte-identical to the already frozen producer-passage artifact, not
    merely semantically similar.  Recompute the visible payload here so a
    title-normalisation or field-selection drift fails before model loading.
    """

    if len(messages) != 2 or messages[1].get("role") != "user":
        raise TaskValidationError("reader messages differ from the frozen two-message schema")
    content = messages[1].get("content")
    if not isinstance(content, str):
        raise TaskValidationError("reader user message has no JSON content")
    try:
        payload = json.loads(
            content,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (_DuplicateJSONKey, json.JSONDecodeError, ValueError) as exc:
        raise TaskValidationError(f"reader user message is not strict JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TaskValidationError("reader user payload is not an object")
    visible = payload.get("retrieved_documents")
    expected = list(producer_passages)
    if visible != expected:
        raise TaskValidationError(
            "reader prompt document projection differs from verifier input bytes"
        )
    if canonical_json_sha256(visible) != canonical_json_sha256(expected):
        raise TaskValidationError("reader/verifier document projection hash mismatch")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_lock(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"required non-empty file is missing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _assert_file_lock(
    expected: Mapping[str, Any], path: Path, *, label: str
) -> dict[str, Any]:
    if not isinstance(expected, Mapping):
        raise TaskValidationError(f"missing file content lock: {label}")
    current = _file_lock(path)
    try:
        expected_path = Path(str(expected["path"])).expanduser().resolve()
        expected_size = int(expected["size_bytes"])
        expected_hash = str(expected["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskValidationError(f"malformed file content lock: {label}") from exc
    if expected_path != Path(current["path"]):
        raise TaskValidationError(f"{label} path differs from its frozen lock")
    if expected_size != current["size_bytes"]:
        raise TaskValidationError(f"{label} size differs from its frozen lock")
    if expected_hash != current["sha256"]:
        raise TaskValidationError(f"{label} SHA256 differs from its frozen lock")
    return current


def _tree_lock(path: Path) -> dict[str, Any]:
    """Hash an entire model directory with the implementation-lock encoding."""

    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"required model directory is missing: {resolved}")
    files: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for child in sorted(item for item in resolved.rglob("*") if item.is_file()):
        relative = child.relative_to(resolved).as_posix()
        size = child.stat().st_size
        child_hash = sha256_file(child)
        files.append({"path": relative, "size_bytes": size, "sha256": child_hash})
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(str(size).encode("ascii") + b"\0")
        digest.update(child_hash.encode("ascii") + b"\n")
    if not files:
        raise TaskValidationError(f"required model directory has no files: {resolved}")
    return {
        "path": str(resolved),
        "file_count": len(files),
        "size_bytes": sum(int(item["size_bytes"]) for item in files),
        "tree_sha256": digest.hexdigest(),
        "files": files,
    }


def _assert_tree_lock(
    expected: Mapping[str, Any], path: Path, *, label: str
) -> dict[str, Any]:
    if not isinstance(expected, Mapping):
        raise TaskValidationError(f"missing model content lock: {label}")
    current = _tree_lock(path)
    expected_path = Path(str(expected.get("path") or "")).expanduser().resolve()
    if expected_path != Path(current["path"]):
        raise TaskValidationError(f"{label} path differs from its frozen lock")
    for field_name in ("file_count", "size_bytes", "tree_sha256", "files"):
        if current[field_name] != expected.get(field_name):
            raise TaskValidationError(
                f"{label} model content differs from its frozen lock: {field_name}"
            )
    return current


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (_DuplicateJSONKey, json.JSONDecodeError, ValueError) as exc:
        raise TaskValidationError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise TaskValidationError(f"{label} must be a JSON object")
    return value, _file_lock(resolved)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TaskValidationError(f"{field_name} must be a lowercase SHA256")
    return value


def _require_identity_string(
    value: object,
    *,
    field_name: str,
    maximum_chars: int = 512,
) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TaskValidationError(f"{field_name} must be a non-empty unpadded string")
    if len(value) > maximum_chars:
        raise TaskValidationError(f"{field_name} exceeds {maximum_chars} characters")
    if any(ord(character) < 32 for character in value):
        raise TaskValidationError(f"{field_name} contains a control character")
    return value


def assert_answer_free(value: Any, *, location: str = "task") -> None:
    """Recursively reject all frozen Gold/decomposition field names."""

    if isinstance(value, Mapping):
        folded = {str(key).casefold() for key in value}
        bad = folded & FORBIDDEN_KEYS
        if bad:
            raise TaskValidationError(
                f"forbidden Gold/answer fields at {location}: {sorted(bad)}"
            )
        for key, child in value.items():
            assert_answer_free(child, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_answer_free(child, location=f"{location}[{index}]")


def read_tasks_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read strict JSONL, rejecting duplicate keys and non-finite constants."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_json_constant,
                )
            except (_DuplicateJSONKey, json.JSONDecodeError, ValueError) as exc:
                raise TaskValidationError(
                    f"{path}:{line_number}: invalid strict JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise TaskValidationError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    if not rows:
        raise TaskValidationError(f"C task file is empty: {path}")
    return rows


def _validate_passage_limits(passages: object, *, location: str) -> list[Mapping[str, Any]]:
    if not isinstance(passages, list) or not passages:
        raise TaskValidationError(f"{location} must be a non-empty list")
    if len(passages) > MAX_PRODUCER_PASSAGES:
        raise TaskValidationError(
            f"{location} has {len(passages)} passages; maximum is {MAX_PRODUCER_PASSAGES}"
        )
    checked: list[Mapping[str, Any]] = []
    for rank, passage in enumerate(passages, start=1):
        if not isinstance(passage, Mapping):
            raise TaskValidationError(f"{location}[{rank - 1}] is not an object")
        has_visible_text = False
        for field_name in ("contents", "text"):
            if field_name not in passage or passage[field_name] is None:
                continue
            text = passage[field_name]
            if not isinstance(text, str):
                raise TaskValidationError(
                    f"{location}[{rank - 1}].{field_name} must be a string"
                )
            if text.strip():
                has_visible_text = True
                if len(text) > MAX_PASSAGE_TEXT_CHARS:
                    raise TaskValidationError(
                        f"{location}[{rank - 1}].{field_name} exceeds the frozen "
                        f"{MAX_PASSAGE_TEXT_CHARS}-Unicode-character prefix"
                    )
        if not has_visible_text:
            raise TaskValidationError(
                f"{location}[{rank - 1}] has no non-empty contents/text"
            )
        checked.append(passage)
    return checked


def _passage_character_telemetry(
    passages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for rank, passage in enumerate(passages, start=1):
        lengths = {
            field_name: len(passage[field_name])
            for field_name in ("contents", "text")
            if isinstance(passage.get(field_name), str) and passage[field_name].strip()
        }
        result.append({"rank": rank, "text_character_counts": lengths})
    return result


def validate_task_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate all tasks before any model/generator invocation."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise TaskValidationError("C tasks must be a non-empty sequence")

    validated: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    question_slots: set[tuple[str, str]] = set()
    for index, source in enumerate(rows):
        location = f"tasks[{index}]"
        if not isinstance(source, Mapping):
            raise TaskValidationError(f"{location} is not an object")
        actual_keys = frozenset(source)
        if actual_keys != TASK_KEYS:
            missing = sorted(TASK_KEYS - actual_keys)
            extra = sorted(actual_keys - TASK_KEYS)
            raise TaskValidationError(
                f"{location} fields differ from exact C-task schema; "
                f"missing={missing}, extra={extra}"
            )
        assert_answer_free(source, location=location)
        if source["gold_access"] is not False:
            raise TaskValidationError(f"{location}.gold_access must be exactly false")

        task_id = _require_identity_string(source["task_id"], field_name=f"{location}.task_id")
        dataset = _require_identity_string(
            source["dataset"], field_name=f"{location}.dataset", maximum_chars=64
        )
        qid = _require_identity_string(
            source["qid"], field_name=f"{location}.qid", maximum_chars=256
        )
        question_key = _require_identity_string(
            source["question_key"], field_name=f"{location}.question_key"
        )
        producer_slot = _require_identity_string(
            source["producer_slot"],
            field_name=f"{location}.producer_slot",
            maximum_chars=256,
        )
        if dataset not in TARGET_TYPES:
            raise TaskValidationError(f"{location}.dataset is unsupported: {dataset!r}")
        if question_key != f"{dataset}::{qid}":
            raise TaskValidationError(f"{location}.question_key identity mismatch")
        if task_id in task_ids:
            raise TaskValidationError(f"duplicate task_id: {task_id}")
        question_slot = (question_key, producer_slot)
        if question_slot in question_slots:
            raise TaskValidationError(
                f"duplicate question_key/producer_slot: {question_key}/{producer_slot}"
            )

        question = source["question"]
        if not isinstance(question, str) or not question.strip():
            raise TaskValidationError(f"{location}.question must be non-empty text")
        observed_question_hash = _require_sha256(
            source["question_sha256"], field_name=f"{location}.question_sha256"
        )
        if question_sha256(question) != observed_question_hash:
            raise TaskValidationError(f"{location}.question_sha256 mismatch")

        target_type = _require_identity_string(
            source["target_type"],
            field_name=f"{location}.target_type",
            maximum_chars=64,
        )
        if target_type != TARGET_TYPES[dataset]:
            raise TaskValidationError(
                f"{location}.target_type mismatch for dataset {dataset}"
            )
        step = source["step"]
        if not isinstance(step, Mapping):
            raise TaskValidationError(f"{location}.step must be an object")
        expected_step_hash = _require_sha256(
            source["step_sha256"], field_name=f"{location}.step_sha256"
        )
        if canonical_json_sha256(step) != expected_step_hash:
            raise TaskValidationError(f"{location}.step_sha256 mismatch")

        passages = _validate_passage_limits(
            source["producer_passages"], location=f"{location}.producer_passages"
        )
        expected_passages_hash = _require_sha256(
            source["producer_passages_sha256"],
            field_name=f"{location}.producer_passages_sha256",
        )
        if canonical_json_sha256(source["producer_passages"]) != expected_passages_hash:
            raise TaskValidationError(
                f"{location}.producer_passages_sha256 byte commitment mismatch"
            )

        # This performs the module's independent step/document validation now,
        # before a backend can be loaded or called.  The exact same passage list
        # object is later passed to the builder and integrated verifier.
        messages = build_subanswer_reader_messages(
            question,
            step,
            passages,
            target_type=target_type,
        )
        _assert_prompt_document_projection(messages, passages)
        task_ids.add(task_id)
        question_slots.add(question_slot)
        validated.append(dict(source))
    return validated


def _validate_generation_output(value: object) -> GenerationOutput:
    if not isinstance(value, GenerationOutput):
        raise TypeError("generator must return GenerationOutput")
    if not isinstance(value.prompt, str) or not value.prompt:
        raise ValueError("generator returned an empty/non-string prompt")
    if not isinstance(value.response_text, str):
        raise ValueError("generator returned a non-string response")
    for field_name, count in (
        ("prompt_tokens", value.prompt_tokens),
        ("generation_tokens", value.generation_tokens),
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"generator returned invalid {field_name}")
    if value.prompt_tokens <= 0:
        raise ValueError("generator returned zero prompt tokens")
    if value.generation_tokens > MAX_NEW_TOKENS:
        raise ValueError("generator exceeded the frozen max_new_tokens=96")
    if value.response_text and value.generation_tokens == 0:
        raise ValueError("non-empty response cannot have zero generation tokens")
    if not isinstance(value.runtime_telemetry, Mapping):
        raise ValueError("generator runtime_telemetry must be an object")
    assert_answer_free(value.runtime_telemetry, location="generator.runtime_telemetry")
    return value


def generate_subanswer_rows(
    tasks: Sequence[Mapping[str, Any]],
    *,
    generator: GeneratorBackend,
    input_file_sha256: str,
    model_artifact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Generate all rows using an injected production or fake backend."""

    validated = validate_task_rows(tasks)
    input_hash = _require_sha256(input_file_sha256, field_name="input_file_sha256")
    if not isinstance(model_artifact, Mapping) or not model_artifact:
        raise ValueError("model_artifact must be a non-empty object")
    assert_answer_free(model_artifact, location="model_artifact")

    output_rows: list[dict[str, Any]] = []
    for task in validated:
        passages = task["producer_passages"]
        messages = build_subanswer_reader_messages(
            task["question"],
            task["step"],
            passages,
            target_type=task["target_type"],
        )
        _assert_prompt_document_projection(messages, passages)
        generated = _validate_generation_output(
            generator(
                messages,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        )
        raw_response_sha256 = _sha256_text(generated.response_text)
        parse_valid = False
        parse_error_code: str | None = None
        try:
            parse_subanswer_response(generated.response_text)
            parse_valid = True
        except SubanswerParseError as exc:
            parse_error_code = exc.code

        verification = parse_and_verify_subanswer(
            generated.response_text,
            task["question"],
            task["step"],
            passages,
            target_type=task["target_type"],
        )
        if verification.get("response_sha256") != raw_response_sha256:
            raise RuntimeError("subanswer module response hash mismatch")
        if bool(verification.get("verified")) and not parse_valid:
            raise RuntimeError("unparseable response was marked verified")

        passage_hash = str(task["producer_passages_sha256"])
        telemetry = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "runner_version": RUNNER_VERSION,
            "input_file_sha256": input_hash,
            "input_task_sha256": canonical_json_sha256(task),
            "messages_sha256": canonical_json_sha256(messages),
            "prompt_sha256": _sha256_text(generated.prompt),
            "prompt_tokens": generated.prompt_tokens,
            "generation_tokens": generated.generation_tokens,
            "raw_response": generated.response_text,
            "raw_response_sha256": raw_response_sha256,
            "raw_response_utf8_bytes": len(generated.response_text.encode("utf-8")),
            "strict_parse": {
                "valid": parse_valid,
                "error_code": parse_error_code,
            },
            "verification": verification,
            "producer_passage_count": len(passages),
            "producer_passage_character_counts": _passage_character_telemetry(passages),
            "prompt_passages_sha256": passage_hash,
            "verifier_passages_sha256": passage_hash,
            "same_passage_bytes_for_prompt_and_verifier": True,
            "generation": {
                "decode": "greedy",
                "do_sample": False,
                "max_new_tokens": MAX_NEW_TOKENS,
                "retry_count": 0,
            },
            "model_artifact": dict(model_artifact),
            "runtime": dict(generated.runtime_telemetry),
            "gold_access": False,
            "network_access": False,
        }
        output = {
            key: task[key]
            for key in OUTPUT_IDENTITY_KEYS
        }
        output.update(
            {
                "verified": bool(verification["verified"]),
                "verified_answer": verification.get("verified_answer"),
                "telemetry": telemetry,
                "gold_access": False,
            }
        )
        assert_answer_free(output, location=f"output.{task['task_id']}")
        output_rows.append(output)
    return output_rows


def _critical_file(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _locked_model_artifact(
    *,
    base_model_lock: Mapping[str, Any],
    strong_sft_lock: Mapping[str, Any],
    implementation_lock: Mapping[str, Any],
    plan_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonical model identity shared with the staged retrieval consumer."""

    return {
        "schema_version": MODEL_ARTIFACT_SCHEMA,
        "base_model": dict(base_model_lock),
        "strong_sft_adapter": dict(strong_sft_lock),
        "load_contract": {
            "torch_dtype": "bfloat16",
            "device": "cuda:0",
            "local_files_only": True,
            "separate_process_required": True,
        },
        "authorization_locks": {
            "implementation_lock": dict(implementation_lock),
            "plan_lock": dict(plan_lock),
        },
    }


def _validate_task_stage_descriptor(
    descriptor_path: Path,
    *,
    tasks_path: Path,
    tasks_sha256: str,
    implementation_lock: Mapping[str, Any],
    plan_lock: Mapping[str, Any],
) -> dict[str, Any]:
    descriptor, descriptor_lock = _read_json_object(
        descriptor_path, label="v7 producer-stage descriptor"
    )
    assert_answer_free(descriptor, location="producer_stage_descriptor")
    if descriptor.get("gold_access") is not False:
        raise TaskValidationError("producer-stage descriptor is not Gold-free")
    if descriptor.get("schema_version") != STAGE_DESCRIPTOR_SCHEMA:
        raise TaskValidationError("producer-stage descriptor schema differs")
    if descriptor.get("runner_version") != "paired-dependent-retrieval-v7-staged-gold-free-1":
        raise TaskValidationError("producer-stage descriptor runner version differs")
    stage = descriptor.get("stage")
    if stage not in {"roots", "dependents"}:
        raise TaskValidationError("producer-stage descriptor has an invalid stage")
    if stage == "roots" and (
        descriptor.get("state_depth") != 1
        or descriptor.get("parent_stage_descriptor") is not None
    ):
        raise TaskValidationError("root producer-stage descriptor depth/parent differs")
    if stage == "dependents":
        producer_depth = descriptor.get("producer_depth")
        target_depth = descriptor.get("target_depth")
        if (
            isinstance(producer_depth, bool)
            or not isinstance(producer_depth, int)
            or producer_depth < 1
            or target_depth != producer_depth + 1
        ):
            raise TaskValidationError("producer-stage descriptor depth chain is invalid")
    runtime_locks = descriptor.get("runtime_locks")
    if not isinstance(runtime_locks, Mapping):
        raise TaskValidationError("producer-stage descriptor lacks runtime locks")
    if runtime_locks.get("implementation_lock") != dict(implementation_lock):
        raise TaskValidationError("producer-stage implementation lock differs")
    if runtime_locks.get("post_plan_execution_lock") != dict(plan_lock):
        raise TaskValidationError("producer-stage plan lock differs")
    outputs = descriptor.get("outputs")
    if not isinstance(outputs, Mapping):
        raise TaskValidationError("producer-stage descriptor lacks outputs")
    task_output = outputs.get("c_tasks")
    if not isinstance(task_output, Mapping):
        raise TaskValidationError("producer-stage descriptor lacks C-task output lock")
    current = _assert_file_lock(task_output, tasks_path, label="producer-stage C tasks")
    if current["sha256"] != tasks_sha256:
        raise TaskValidationError("caller task SHA256 differs from producer descriptor")
    return descriptor_lock


def load_generation_authorization(
    *,
    implementation_lock_path: Path,
    plan_lock_path: Path,
    stage_descriptor_path: Path,
    tasks_path: Path,
    tasks_sha256: str,
    base_model: Path,
    adapter: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate both v7 execution locks and model content before CUDA loading."""

    implementation, implementation_lock = _read_json_object(
        implementation_lock_path, label="v7 implementation lock"
    )
    plan, plan_lock = _read_json_object(plan_lock_path, label="v7 plan lock")
    if (
        implementation.get("schema_version") != IMPLEMENTATION_LOCK_SCHEMA
        or implementation.get("status") != IMPLEMENTATION_LOCK_STATUS
        or implementation.get("scope") != EXECUTION_SCOPE
        or implementation.get("gold_access") is not False
    ):
        raise TaskValidationError("v7 implementation lock is not the frozen authorized lock")
    if implementation.get("authorization") != {
        "planner": True,
        "gold_free_materialization": False,
        "gold_attachment": False,
        "answer_evaluation": False,
        "training": False,
    }:
        raise TaskValidationError("v7 implementation authorization flags differ")
    if (
        plan.get("schema_version") != PLAN_LOCK_SCHEMA
        or plan.get("status") != PLAN_LOCK_STATUS
        or plan.get("scope") != EXECUTION_SCOPE
        or plan.get("gold_access") is not False
    ):
        raise TaskValidationError("v7 plan lock is not the frozen materialization lock")
    if plan.get("authorization") != {
        "planner_complete": True,
        "gold_free_materialization": True,
        "gold_attachment": False,
        "answer_evaluation": False,
        "training": False,
    }:
        raise TaskValidationError("v7 plan-lock authorization flags differ")

    plan_parents = plan.get("parents")
    if not isinstance(plan_parents, Mapping) or plan_parents.get(
        "implementation_lock"
    ) != implementation_lock:
        raise TaskValidationError("v7 plan lock is not bound to this implementation lock")
    for document_name, document in (("implementation", implementation), ("plan", plan)):
        parents = document.get("parents")
        if not isinstance(parents, Mapping):
            raise TaskValidationError(f"v7 {document_name} lock lacks parent locks")
        for name, lock in parents.items():
            if not isinstance(lock, Mapping):
                raise TaskValidationError(f"v7 {document_name} parent lock is malformed: {name}")
            _assert_file_lock(
                lock,
                Path(str(lock.get("path") or "")),
                label=f"{document_name}.parents.{name}",
            )

    implementation_runtime = implementation.get("runtime_code")
    plan_runtime = plan.get("runtime_code")
    if (
        not isinstance(implementation_runtime, Mapping)
        or dict(plan_runtime or {}) != dict(implementation_runtime)
        or "subanswer_generator" not in implementation_runtime
    ):
        raise TaskValidationError("v7 runtime-code locks differ between implementation/plan")
    for name, lock in implementation_runtime.items():
        _assert_file_lock(
            lock,
            Path(str(lock.get("path") or "")),
            label=f"runtime_code.{name}",
        )
    if Path(
        str(implementation_runtime["subanswer_generator"].get("path") or "")
    ).resolve() != Path(__file__).resolve():
        raise TaskValidationError("subanswer-generator runtime lock points to another file")

    for name, lock in (plan.get("inputs") or {}).items():
        if not isinstance(lock, Mapping):
            raise TaskValidationError(f"v7 plan input lock is malformed: {name}")
        _assert_file_lock(
            lock,
            Path(str(lock.get("path") or "")),
            label=f"plan.inputs.{name}",
        )

    verified = (implementation.get("content_reverification") or {}).get("verified")
    models = verified.get("models") if isinstance(verified, Mapping) else None
    if not isinstance(models, Mapping):
        raise TaskValidationError("implementation lock lacks verified model content")
    base_lock = models.get("base_model")
    strong_sft_lock = models.get("strong_sft")
    if not isinstance(base_lock, Mapping) or not isinstance(strong_sft_lock, Mapping):
        raise TaskValidationError("implementation lock lacks base/strong-SFT content locks")
    base = base_model.expanduser().resolve()
    strong_sft = adapter.expanduser().resolve()
    if base != Path(str(base_lock.get("path") or "")).expanduser().resolve():
        raise TaskValidationError("CLI --base_model differs from frozen base model")
    if strong_sft != Path(str(strong_sft_lock.get("path") or "")).expanduser().resolve():
        raise TaskValidationError("CLI --adapter differs from frozen strong-SFT adapter")
    current_base = _assert_tree_lock(base_lock, base, label="base_model")
    current_sft = _assert_tree_lock(strong_sft_lock, strong_sft, label="strong_sft")

    stage_descriptor_lock = _validate_task_stage_descriptor(
        stage_descriptor_path,
        tasks_path=tasks_path,
        tasks_sha256=tasks_sha256,
        implementation_lock=implementation_lock,
        plan_lock=plan_lock,
    )
    artifact = _locked_model_artifact(
        base_model_lock=current_base,
        strong_sft_lock=current_sft,
        implementation_lock=implementation_lock,
        plan_lock=plan_lock,
    )
    authorization = {
        "implementation_lock": implementation_lock,
        "plan_lock": plan_lock,
        "producer_stage_descriptor": stage_descriptor_lock,
    }
    return artifact, authorization


def build_model_artifact(base_model: Path, adapter: Path) -> dict[str, Any]:
    """Build content-bearing identities before the CUDA model is loaded."""

    base = base_model.expanduser().resolve()
    strong_sft = adapter.expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"base model directory is missing: {base}")
    if not strong_sft.is_dir():
        raise FileNotFoundError(f"strong-SFT adapter directory is missing: {strong_sft}")

    base_config = base / "config.json"
    base_index = base / "model.safetensors.index.json"
    adapter_config = strong_sft / "adapter_config.json"
    adapter_weights = strong_sft / "adapter_model.safetensors"
    for path in (base_config, base_index, adapter_config, adapter_weights):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"required model artifact is missing/empty: {path}")

    return {
        "base_model": {
            "identity": artifact_identity(base),
            "config": _critical_file(base_config),
            "weight_index": _critical_file(base_index),
        },
        "strong_sft_adapter": {
            "identity": artifact_identity(strong_sft),
            "config": _critical_file(adapter_config),
            "weights": _critical_file(adapter_weights),
        },
        "load_contract": {
            "torch_dtype": "bfloat16",
            "device": "cuda:0",
            "local_files_only": True,
            "separate_process_required": True,
        },
    }


class HuggingFaceGenerator:
    """Single-row greedy BF16 generator used only by the production CLI."""

    def __init__(self, base_model: Path, adapter: Path, *, seed: int) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing a CPU fallback")
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            local_files_only=True,
        )
        if self._tokenizer.eos_token_id is None:
            raise RuntimeError("base tokenizer has no EOS token")
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self._model = PeftModel.from_pretrained(
            self._model,
            adapter,
            is_trainable=False,
            local_files_only=True,
        )
        self._model.eval()
        self._device = next(self._model.parameters()).device
        if self._device.type != "cuda":
            raise RuntimeError(
                f"model loaded on {self._device}; refusing a CPU/disk fallback"
            )

    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int,
        do_sample: bool,
    ) -> GenerationOutput:
        if max_new_tokens != MAX_NEW_TOKENS or do_sample is not False:
            raise ValueError("generation settings differ from the frozen v7 contract")
        prompt = self._tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self._device)
        prompt_tokens = int(encoded["input_ids"].shape[1])
        self._torch.cuda.reset_peak_memory_stats(self._device)
        with self._torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        new_token_ids = generated[0, prompt_tokens:]
        response = self._tokenizer.decode(new_token_ids, skip_special_tokens=True)
        return GenerationOutput(
            prompt=prompt,
            response_text=response,
            prompt_tokens=prompt_tokens,
            generation_tokens=int(new_token_ids.shape[0]),
            runtime_telemetry={
                "execution_device": str(self._device),
                "torch_dtype": "bfloat16",
                "cuda_max_memory_allocated_bytes": int(
                    self._torch.cuda.max_memory_allocated(self._device)
                ),
                "cuda_max_memory_reserved_bytes": int(
                    self._torch.cuda.max_memory_reserved(self._device)
                ),
            },
        )


def _write_jsonl_exclusive(
    path: Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, allow_nan=False, sort_keys=True)
                + "\n"
            )


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            dict(value),
            handle,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, Any] = {}
    for dataset in TARGET_TYPES:
        selected = [row for row in rows if row["dataset"] == dataset]
        reasons = Counter(
            str(row["telemetry"]["verification"]["reason"])
            for row in selected
        )
        parse_valid = sum(bool(row["telemetry"]["strict_parse"]["valid"]) for row in selected)
        verified = sum(bool(row["verified"]) for row in selected)
        by_dataset[dataset] = {
            "tasks": len(selected),
            "strict_parse_valid": parse_valid,
            "strict_parse_rate": parse_valid / max(1, len(selected)),
            "mechanically_verified": verified,
            "mechanically_verified_rate": verified / max(1, len(selected)),
            "verification_reasons": dict(sorted(reasons.items())),
        }
    return {
        "tasks": len(rows),
        "strict_parse_valid": sum(
            bool(row["telemetry"]["strict_parse"]["valid"]) for row in rows
        ),
        "mechanically_verified": sum(bool(row["verified"]) for row in rows),
        "by_dataset": by_dataset,
    }


def run_generation(
    *,
    tasks_path: Path,
    expected_tasks_sha256: str,
    output_dir: Path,
    experiment_id: str,
    base_model: Path = DEFAULT_BASE_MODEL,
    adapter: Path = DEFAULT_ADAPTER,
    implementation_lock_path: Path | None = None,
    plan_lock_path: Path | None = None,
    stage_descriptor_path: Path | None = None,
    seed: int = DEFAULT_SEED,
    generator: GeneratorBackend | None = None,
    model_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one append-only generation stage.

    ``generator`` and ``model_artifact`` are injectable only through this Python
    API for CPU tests.  The CLI always constructs ``HuggingFaceGenerator``.
    """

    tasks_path = tasks_path.expanduser().resolve()
    if not tasks_path.is_file():
        raise FileNotFoundError(f"C task file is missing: {tasks_path}")
    expected_hash = _require_sha256(
        expected_tasks_sha256, field_name="expected_tasks_sha256"
    )
    observed_hash = sha256_file(tasks_path)
    if observed_hash != expected_hash:
        raise TaskValidationError("C task file SHA256 differs from the caller lock")
    tasks = validate_task_rows(read_tasks_jsonl(tasks_path))

    authorization_locks: dict[str, Any] | None = None
    if generator is None:
        if model_artifact is not None:
            raise ValueError("model_artifact override requires an injected generator")
        if (
            implementation_lock_path is None
            or plan_lock_path is None
            or stage_descriptor_path is None
        ):
            raise ValueError(
                "production generation requires implementation, plan, and producer-stage locks"
            )
        frozen_model_artifact, authorization_locks = load_generation_authorization(
            implementation_lock_path=implementation_lock_path,
            plan_lock_path=plan_lock_path,
            stage_descriptor_path=stage_descriptor_path,
            tasks_path=tasks_path,
            tasks_sha256=observed_hash,
            base_model=base_model,
            adapter=adapter,
        )
    else:
        if model_artifact is None:
            raise ValueError("an injected generator requires an explicit model_artifact")
        frozen_model_artifact = dict(model_artifact)
        assert_answer_free(frozen_model_artifact, location="model_artifact")

    run_dir, reserved_experiment_id = prepare_new_run_dir(
        output_dir,
        experiment_id=experiment_id,
        extra={
            "phase": "dependent_retrieval_v7_grounded_subanswer_generation",
            "runner_version": RUNNER_VERSION,
            "input_tasks_sha256": observed_hash,
            "authorization_locks": authorization_locks,
            "gold_access": False,
            "network_access": False,
        },
    )
    try:
        active_generator = generator or HuggingFaceGenerator(
            base_model.expanduser().resolve(),
            adapter.expanduser().resolve(),
            seed=seed,
        )
        output_rows = generate_subanswer_rows(
            tasks,
            generator=active_generator,
            input_file_sha256=observed_hash,
            model_artifact=frozen_model_artifact,
        )
        if len(output_rows) != len(tasks):
            raise RuntimeError("subanswer output cardinality differs from input tasks")
        if [row["task_id"] for row in output_rows] != [row["task_id"] for row in tasks]:
            raise RuntimeError("subanswer output order/identity differs from input tasks")

        output_path = run_dir / "subanswers.jsonl"
        _write_jsonl_exclusive(output_path, output_rows)
        counts = _summarize(output_rows)
        report = {
            "schema_version": "dependent-retrieval-v7-subanswer-report-1",
            "experiment_id": reserved_experiment_id,
            "status": "COMPLETE_GOLD_FREE_SUBANSWERS",
            "runner_version": RUNNER_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": {
                "path": str(tasks_path),
                "sha256": observed_hash,
                "rows": len(tasks),
            },
            "generation": {
                "decode": "greedy",
                "do_sample": False,
                "max_new_tokens": MAX_NEW_TOKENS,
                "seed": seed,
                "retry_count": 0,
                "torch_dtype": "bfloat16",
            },
            "model_artifact": frozen_model_artifact,
            "authorization_locks": authorization_locks,
            "counts": counts,
            "output": {
                "path": str(output_path.resolve()),
                "sha256": sha256_file(output_path),
                "rows": len(output_rows),
            },
            "gold_access": False,
            "network_access": False,
        }
        assert_answer_free(report, location="report")
        report_path = run_dir / "report.json"
        _write_json_exclusive(report_path, report)
        dump_manifest(
            run_dir,
            status=report["status"],
            extra={
                "experiment_id": reserved_experiment_id,
                "phase": "dependent_retrieval_v7_grounded_subanswer_generation",
                "runner_version": RUNNER_VERSION,
                "input_tasks_sha256": observed_hash,
                "output_sha256": report["output"]["sha256"],
                "report_sha256": sha256_file(report_path),
                "authorization_locks": authorization_locks,
                "gold_access": False,
                "network_access": False,
            },
        )
        return report
    except Exception as exc:
        dump_manifest(
            run_dir,
            status="FAILED_RUNTIME_GOLD_FREE",
            extra={
                "experiment_id": reserved_experiment_id,
                "phase": "dependent_retrieval_v7_grounded_subanswer_generation",
                "runner_version": RUNNER_VERSION,
                "input_tasks_sha256": observed_hash,
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "gold_access": False,
                "network_access": False,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--expected_tasks_sha256", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--base_model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument(
        "--implementation_lock", type=Path, default=DEFAULT_IMPLEMENTATION_LOCK
    )
    parser.add_argument("--plan_lock", type=Path, default=DEFAULT_PLAN_LOCK)
    parser.add_argument("--stage_descriptor", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    report = run_generation(
        tasks_path=args.tasks,
        expected_tasks_sha256=args.expected_tasks_sha256,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
        base_model=args.base_model,
        adapter=args.adapter,
        implementation_lock_path=args.implementation_lock,
        plan_lock_path=args.plan_lock,
        stage_descriptor_path=args.stage_descriptor,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
