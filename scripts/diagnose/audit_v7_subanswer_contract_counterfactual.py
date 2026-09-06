#!/usr/bin/env python
"""Gold-free Phase-0 counterfactual audit of the frozen v7 reader contract.

This script is deliberately specific to the already-consumed v7 depth-1
development artifacts.  It does not load a dataset, Gold answer, supporting
fact, decomposition, model, retriever, or GPU.  It authenticates the frozen
task/response chain and evaluates three read-only contracts:

* P0: re-execute the original strict v7 parser and verifier;
* P1: keep the strict parser, but infer the effective extractive surface class
  with the frozen v7 date -> number -> entity order before re-running the
  original verifier; and
* P2: P1 plus one optimistic parser exception: ``abstain: ""`` may be changed
  to ``false`` only when the answer is non-empty and there is exactly one
  citation.  The patched object must then pass the original strict parser.

P2 is an upper-bound diagnostic, not a proposed production parser.  None of
the three conditions measures semantic entailment, support-chain recall,
EM/F1, or answer quality.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_sha256
from kgproweight.retrieval import subanswer_v7


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TASKS = Path(
    "outputs/validation/subquestion_dependent_retrieval_v7_development_"
    "materialization_v1_retry1/c_tasks.depth_1.jsonl"
)
DEFAULT_SUBANSWERS = Path(
    "outputs/validation/subquestion_dependent_retrieval_v7_subanswers_"
    "depth1_v1_retry1/subanswers.jsonl"
)
DEFAULT_GENERATOR_REPORT = Path(
    "outputs/validation/subquestion_dependent_retrieval_v7_subanswers_"
    "depth1_v1_retry1/report.json"
)
DEFAULT_GENERATOR_MANIFEST = Path(
    "outputs/validation/subquestion_dependent_retrieval_v7_subanswers_"
    "depth1_v1_retry1/manifest.json"
)
DEFAULT_STAGE_DESCRIPTOR = Path(
    "outputs/validation/subquestion_dependent_retrieval_v7_development_"
    "materialization_v1_retry1/roots_stage.json"
)
DEFAULT_IMPLEMENTATION_LOCK = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_"
    "implementation_lock_v1_retry1/protocol.json"
)
DEFAULT_PLAN_LOCK = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_"
    "plans_lock_v1_retry1/protocol.json"
)
DEFAULT_PREREGISTRATION = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_"
    "preregistration/protocol.json"
)
DEFAULT_PREREGISTRATION_MANIFEST = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_"
    "preregistration/manifest.json"
)
DEFAULT_TRAJECTORY_LOCK = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_"
    "preregistration_addendum_recursive_trajectory_v1/protocol.json"
)
DEFAULT_TRAJECTORY_MANIFEST = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_"
    "preregistration_addendum_recursive_trajectory_v1/manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/audits/subquestion_decomposition_v8_phase0_v7_contract_"
    "counterfactual_v1"
)
DEFAULT_EXPERIMENT_ID = (
    "SUBQUESTION-DECOMPOSITION-V8-PHASE0-V7-CONTRACT-"
    "COUNTERFACTUAL-DEV41-SEED20260904-V1"
)

FROZEN_SHA256 = {
    "tasks": "c0391c06c7827e08e39072767620c4b53ccffc379eeb296798243fbe3e4eaf8c",
    "subanswers": "6fb92342a11a0923a37d45e5d8c0c32b1d86db34f072a90cf938c254984f7665",
    "generator_report": "667d70cc8f84c34e58e8ab4dd3ac354d1ac4392d4734356df7886de25bc4942c",
    "generator_manifest": "abbd0da33c7a706d270e2cd846d5acf35299ffef98dc27940420dbde06207465",
    "stage_descriptor": "e06bbf48271df87ccbe1229f48c588b19e9dd958a7d95d9cee75e6ff54516eae",
    "implementation_lock": "47259e05cebd1771da3022c5ae79f25214ae5010a3bdc834075ecb47fc576bdc",
    "plan_lock": "ac688cac440d90fec6bb20427c8bc4e2b141b14e3ff21477855000a8db4b0efa",
    "preregistration": "c7f1674f62a191671a22844e5589c3f9b80a990aae3c0344cd4001e47a50395d",
    "preregistration_manifest": "01ab042c1f824dc649cddc5aa929393efe17aab9631ec07de70fcb5f8dda19cb",
    "trajectory_lock": "53738e0474e677af89a08ba2cc16e98f6b0ecd3613dbd45608566065e46bfe2d",
    "trajectory_manifest": "a51d05348753e747a9d1955642b1dedefaa26a8bb8ceedf335f043e28d4801d2",
}

DATASETS = ("hotpotqa", "musique")
EXPECTED_TASKS_BY_DATASET = {"hotpotqa": 19, "musique": 22}
EXPECTED_P0_VERIFIED_BY_DATASET = {"hotpotqa": 3, "musique": 3}
FROZEN_GATE = 0.40

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
SUBANSWER_KEYS = frozenset(
    {
        "task_id",
        "question_key",
        "dataset",
        "qid",
        "question_sha256",
        "target_type",
        "producer_slot",
        "step_sha256",
        "producer_passages_sha256",
        "verified",
        "verified_answer",
        "telemetry",
        "gold_access",
    }
)
IDENTITY_FIELDS = (
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
FORBIDDEN_TASK_KEYS = frozenset(
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
_EXPERIMENT_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{15,255}$")
_CITATION_MISS_REASONS = frozenset(
    {"cited_document_not_in_input", "answer_surface_not_in_cited_document"}
)

REPORT_SCHEMA = "subquestion-decomposition-v8-phase0-contract-audit-1"
ROW_SCHEMA = "subquestion-decomposition-v8-phase0-contract-row-1"
MANIFEST_SCHEMA = "subquestion-decomposition-v8-phase0-contract-manifest-1"
PROTOCOL_SCHEMA = "subquestion-decomposition-v8-phase0-contract-protocol-1"
PROTOCOL_STATUS = "FROZEN_BEFORE_PHASE0_EXECUTION"
RESEARCHER_AUTHORIZATION = "approved_in_thread_2026-09-04"
SCOPE = "CONSUMED_DEVELOPMENT_ONLY_HOTPOT19_MUSIQUE22"


class CounterfactualAuditError(RuntimeError):
    """An input or output violates the fail-closed audit contract."""


class _DuplicateJSONKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_lock(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise CounterfactualAuditError(f"required non-empty file is missing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _safe_input(path: Path, *, project_root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    root = project_root.expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise CounterfactualAuditError(
            f"{label} is outside the allowed project root: {resolved}"
        ) from exc
    if relative.parts[:2] == ("data", "raw"):
        raise CounterfactualAuditError(f"{label} attempts to read data/raw: {resolved}")
    if not resolved.is_file():
        raise CounterfactualAuditError(f"{label} is missing: {resolved}")
    return resolved


def _safe_new_output(path: Path, *, project_root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    root = project_root.expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise CounterfactualAuditError(
            f"{label} is outside the allowed project root: {resolved}"
        ) from exc
    if relative.parts[:2] == ("data", "raw"):
        raise CounterfactualAuditError(f"{label} attempts to write data/raw: {resolved}")
    if resolved.exists():
        raise FileExistsError(f"append-only {label} already exists: {resolved}")
    return resolved


def _loads_json(text: str, *, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (_DuplicateJSONKey, json.JSONDecodeError, ValueError) as exc:
        raise CounterfactualAuditError(f"invalid strict JSON in {label}: {exc}") from exc


def _read_json(path: Path, *, project_root: Path, label: str) -> dict[str, Any]:
    resolved = _safe_input(path, project_root=project_root, label=label)
    value = _loads_json(resolved.read_text(encoding="utf-8"), label=label)
    if not isinstance(value, dict):
        raise CounterfactualAuditError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, *, project_root: Path, label: str) -> list[dict[str, Any]]:
    resolved = _safe_input(path, project_root=project_root, label=label)
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = _loads_json(line, label=f"{label}:{line_number}")
            if not isinstance(value, dict):
                raise CounterfactualAuditError(
                    f"{label}:{line_number} must be a JSON object"
                )
            rows.append(value)
    if not rows:
        raise CounterfactualAuditError(f"{label} is empty")
    return rows


def _assert_no_forbidden_task_keys(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().casefold()
            if key in FORBIDDEN_TASK_KEYS:
                raise CounterfactualAuditError(
                    f"forbidden Gold/decomposition key at {location}: {raw_key}"
                )
            _assert_no_forbidden_task_keys(child, location=f"{location}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_task_keys(child, location=f"{location}[{index}]")


def _require_sha256(value: object, *, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CounterfactualAuditError(f"{location} is not a lowercase SHA256")
    return value


def _validate_task(task: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    location = f"tasks[{index}]"
    if frozenset(task) != TASK_KEYS:
        raise CounterfactualAuditError(f"{location} differs from the exact v7 schema")
    _assert_no_forbidden_task_keys(task, location=location)
    if task.get("gold_access") is not False:
        raise CounterfactualAuditError(f"{location}.gold_access must be false")
    dataset = task.get("dataset")
    qid = task.get("qid")
    task_id = task.get("task_id")
    if dataset not in DATASETS or not isinstance(qid, str) or not qid.strip():
        raise CounterfactualAuditError(f"{location} has invalid dataset/qid")
    if not isinstance(task_id, str) or not task_id or task_id != task_id.strip():
        raise CounterfactualAuditError(f"{location} has invalid task_id")
    if task.get("question_key") != f"{dataset}::{qid}":
        raise CounterfactualAuditError(f"{location} question identity mismatch")
    question = task.get("question")
    if not isinstance(question, str) or not question.strip():
        raise CounterfactualAuditError(f"{location}.question must be non-empty")
    if question_sha256(question) != _require_sha256(
        task.get("question_sha256"), location=f"{location}.question_sha256"
    ):
        raise CounterfactualAuditError(f"{location} question/hash mismatch")
    step = task.get("step")
    passages = task.get("producer_passages")
    if not isinstance(step, Mapping):
        raise CounterfactualAuditError(f"{location}.step must be an object")
    if not isinstance(passages, list) or not passages or len(passages) > 10:
        raise CounterfactualAuditError(f"{location}.producer_passages is invalid")
    if sha256_json(step) != _require_sha256(
        task.get("step_sha256"), location=f"{location}.step_sha256"
    ):
        raise CounterfactualAuditError(f"{location} step/hash mismatch")
    if sha256_json(passages) != _require_sha256(
        task.get("producer_passages_sha256"),
        location=f"{location}.producer_passages_sha256",
    ):
        raise CounterfactualAuditError(f"{location} passage/hash mismatch")
    # Reuse the frozen module's caller-input validation without generating.
    subanswer_v7.build_subanswer_reader_messages(
        question,
        step,
        passages,
        target_type=str(task.get("target_type")),
    )
    return dict(task)


def _validate_subanswer_identity(
    row: Mapping[str, Any], task: Mapping[str, Any], *, index: int
) -> dict[str, Any]:
    location = f"subanswers[{index}]"
    if frozenset(row) != SUBANSWER_KEYS:
        raise CounterfactualAuditError(f"{location} differs from the exact v7 schema")
    if row.get("gold_access") is not False:
        raise CounterfactualAuditError(f"{location}.gold_access must be false")
    for field in IDENTITY_FIELDS:
        if row.get(field) != task.get(field):
            raise CounterfactualAuditError(f"{location}.{field} identity mismatch")
    telemetry = row.get("telemetry")
    if not isinstance(telemetry, Mapping):
        raise CounterfactualAuditError(f"{location}.telemetry must be an object")
    if telemetry.get("gold_access") is not False or telemetry.get("network_access") is not False:
        raise CounterfactualAuditError(f"{location} is not Gold-free/offline")
    if telemetry.get("same_passage_bytes_for_prompt_and_verifier") is not True:
        raise CounterfactualAuditError(f"{location} passage binding is not exact")
    passage_hash = str(task["producer_passages_sha256"])
    if telemetry.get("prompt_passages_sha256") != passage_hash:
        raise CounterfactualAuditError(f"{location} prompt passage hash mismatch")
    if telemetry.get("verifier_passages_sha256") != passage_hash:
        raise CounterfactualAuditError(f"{location} verifier passage hash mismatch")
    if telemetry.get("input_task_sha256") != sha256_json(task):
        raise CounterfactualAuditError(f"{location} task hash mismatch")
    raw_response = telemetry.get("raw_response")
    if not isinstance(raw_response, str):
        raise CounterfactualAuditError(f"{location}.raw_response must be text")
    if telemetry.get("raw_response_sha256") != sha256_text(raw_response):
        raise CounterfactualAuditError(f"{location} response hash mismatch")
    if telemetry.get("raw_response_utf8_bytes") != len(raw_response.encode("utf-8")):
        raise CounterfactualAuditError(f"{location} response byte count mismatch")
    if not isinstance(telemetry.get("strict_parse"), Mapping):
        raise CounterfactualAuditError(f"{location} lacks strict_parse telemetry")
    if not isinstance(telemetry.get("verification"), Mapping):
        raise CounterfactualAuditError(f"{location} lacks verification telemetry")
    return dict(row)


def validate_and_join_rows(
    tasks: Sequence[Mapping[str, Any]],
    subanswers: Sequence[Mapping[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if not tasks or len(tasks) != len(subanswers):
        raise CounterfactualAuditError("task/subanswer cardinality mismatch")
    result: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    for index, (raw_task, raw_row) in enumerate(zip(tasks, subanswers)):
        if not isinstance(raw_task, Mapping) or not isinstance(raw_row, Mapping):
            raise CounterfactualAuditError(f"row {index} is not an object")
        task = _validate_task(raw_task, index=index)
        row = _validate_subanswer_identity(raw_row, task, index=index)
        task_id = str(task["task_id"])
        if task_id in seen:
            raise CounterfactualAuditError(f"duplicate task_id: {task_id}")
        seen.add(task_id)
        result.append((task, row))
    return result


def infer_surface_class(answer: str) -> str:
    """Frozen P1/P2 effective class: date, then number, else entity-like."""

    if not isinstance(answer, str):
        raise CounterfactualAuditError("surface-class inference requires a string")
    if subanswer_v7._looks_like_date(answer):
        return "date"
    if subanswer_v7._looks_like_number(answer):
        return "number"
    return "entity"


def _raw_response_object(response_text: str) -> dict[str, Any]:
    value = _loads_json(response_text, label="raw_response")
    if not isinstance(value, dict):
        raise CounterfactualAuditError("raw_response must be a JSON object for P2")
    return value


def _parse_candidate(
    response_text: str, *, allow_empty_string_abstain: bool
) -> tuple[dict[str, Any] | None, str | None, bool, str | None]:
    try:
        return subanswer_v7.parse_subanswer_response(response_text), None, False, None
    except subanswer_v7.SubanswerParseError as exc:
        if not allow_empty_string_abstain or exc.code != "abstain_not_boolean":
            return None, exc.code, False, exc.code
        original_error = exc.code

    try:
        raw = _raw_response_object(response_text)
    except CounterfactualAuditError:
        return None, "invalid_json", False, original_error
    answer = raw.get("answer")
    citations = raw.get("cited_doc_ids")
    if (
        raw.get("abstain") != ""
        or not isinstance(answer, str)
        or not answer.strip()
        or not isinstance(citations, list)
        or len(citations) != 1
    ):
        return None, "p2_empty_abstain_coercion_precondition", False, original_error
    patched = dict(raw)
    patched["abstain"] = False
    try:
        # Re-enter the original parser so every non-abstain constraint remains
        # exactly frozen after the one permitted coercion.
        candidate = subanswer_v7.parse_subanswer_response(canonical_json(patched))
    except subanswer_v7.SubanswerParseError as exc:
        return None, f"p2_post_coercion:{exc.code}", False, original_error
    return candidate, None, True, original_error


def _compact_result(
    *,
    verification: Mapping[str, Any],
    condition_parse_admitted: bool,
    original_strict_parse_valid: bool,
    original_parse_error_code: str | None,
    condition_parse_error_code: str | None,
    parser_semantics: str,
    abstain_coerced: bool,
    original_answer_type: str | None,
    effective_answer_type: str | None,
    answer_sha256: str | None,
) -> dict[str, Any]:
    reason = str(verification.get("reason") or "")
    return {
        "condition_parse_admitted": condition_parse_admitted,
        "original_strict_parse_valid": original_strict_parse_valid,
        "original_parse_error_code": original_parse_error_code,
        "condition_parse_error_code": condition_parse_error_code,
        "parser_semantics": parser_semantics,
        "abstain_empty_string_coerced": abstain_coerced,
        "original_answer_type": original_answer_type,
        "effective_answer_type": effective_answer_type,
        "answer_sha256": answer_sha256,
        "verified": verification.get("verified") is True,
        "reason": reason,
        "locality_pass": verification.get("supporting_sentence_sha256") is not None,
        "subject_echo": reason == "subject_echo",
        "citation_or_locality_miss": reason in _CITATION_MISS_REASONS,
        "supporting_doc_id": verification.get("supporting_doc_id"),
        "supporting_sentence_sha256": verification.get("supporting_sentence_sha256"),
        "surface_match_mode": verification.get("surface_match_mode"),
    }


def _evaluate_inferred_contract(
    response_text: str,
    task: Mapping[str, Any],
    *,
    allow_empty_string_abstain: bool,
) -> dict[str, Any]:
    candidate, error_code, coerced, original_error = _parse_candidate(
        response_text,
        allow_empty_string_abstain=allow_empty_string_abstain,
    )
    if candidate is None:
        verification = {
            "verified": False,
            "reason": f"parse_error:{error_code}",
            "supporting_sentence_sha256": None,
            "supporting_doc_id": None,
            "surface_match_mode": None,
        }
        return _compact_result(
            verification=verification,
            condition_parse_admitted=False,
            original_strict_parse_valid=False,
            original_parse_error_code=original_error,
            condition_parse_error_code=error_code,
            parser_semantics=(
                "p2_registered_coercion_then_original_strict_v7_parser"
                if allow_empty_string_abstain
                else "original_strict_v7_parser"
            ),
            abstain_coerced=False,
            original_answer_type=None,
            effective_answer_type=None,
            answer_sha256=None,
        )

    answer = str(candidate["answer"])
    original_type = str(candidate["answer_type"])
    effective_type = infer_surface_class(answer)
    effective = dict(candidate)
    effective["answer_type"] = effective_type
    verification = subanswer_v7.verify_subanswer(
        effective,
        str(task["question"]),
        task["step"],
        task["producer_passages"],
        target_type=str(task["target_type"]),
    )
    return _compact_result(
        verification=verification,
        condition_parse_admitted=True,
        original_strict_parse_valid=original_error is None,
        original_parse_error_code=original_error,
        condition_parse_error_code=None,
        parser_semantics=(
            "p2_registered_coercion_then_original_strict_v7_parser"
            if allow_empty_string_abstain
            else "original_strict_v7_parser"
        ),
        abstain_coerced=coerced,
        original_answer_type=original_type,
        effective_answer_type=effective_type,
        answer_sha256=sha256_text(answer) if answer else None,
    )


def audit_counterfactual_rows(
    tasks: Sequence[Mapping[str, Any]],
    subanswers: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Re-execute P0/P1/P2 in memory without opening any dataset or Gold."""

    joined = validate_and_join_rows(tasks, subanswers)
    # Complete and authenticate the entire P0 replay first.  P1/P2 are not
    # evaluated for even one row if any frozen P0 row differs.
    p0_rows: list[dict[str, Any]] = []
    for task, row in joined:
        telemetry = row["telemetry"]
        response_text = str(telemetry["raw_response"])
        p0_verification = subanswer_v7.parse_and_verify_subanswer(
            response_text,
            str(task["question"]),
            task["step"],
            task["producer_passages"],
            target_type=str(task["target_type"]),
        )
        if dict(p0_verification) != dict(telemetry["verification"]):
            raise CounterfactualAuditError(
                f"P0 verification drift for task_id={task['task_id']}"
            )
        parse_valid = False
        parse_error: str | None = None
        original_type: str | None = None
        answer_hash: str | None = None
        try:
            p0_candidate = subanswer_v7.parse_subanswer_response(response_text)
            parse_valid = True
            original_type = str(p0_candidate["answer_type"])
            if p0_candidate["answer"]:
                answer_hash = sha256_text(str(p0_candidate["answer"]))
        except subanswer_v7.SubanswerParseError as exc:
            parse_error = exc.code
        recorded_parse = telemetry["strict_parse"]
        if recorded_parse.get("valid") is not parse_valid or recorded_parse.get(
            "error_code"
        ) != parse_error:
            raise CounterfactualAuditError(
                f"P0 strict-parser telemetry drift for task_id={task['task_id']}"
            )
        if row["verified"] is not bool(p0_verification["verified"]):
            raise CounterfactualAuditError(
                f"P0 top-level verified drift for task_id={task['task_id']}"
            )
        if row.get("verified_answer") != p0_verification.get("verified_answer"):
            raise CounterfactualAuditError(
                f"P0 top-level verified_answer drift for task_id={task['task_id']}"
            )
        p0 = _compact_result(
            verification=p0_verification,
            condition_parse_admitted=parse_valid,
            original_strict_parse_valid=parse_valid,
            original_parse_error_code=parse_error,
            condition_parse_error_code=parse_error,
            parser_semantics="original_strict_v7_parser",
            abstain_coerced=False,
            original_answer_type=original_type,
            effective_answer_type=original_type,
            answer_sha256=answer_hash,
        )
        p0_rows.append(p0)

    output: list[dict[str, Any]] = []
    for (task, row), p0 in zip(joined, p0_rows):
        telemetry = row["telemetry"]
        response_text = str(telemetry["raw_response"])
        p1 = _evaluate_inferred_contract(
            response_text, task, allow_empty_string_abstain=False
        )
        p2 = _evaluate_inferred_contract(
            response_text, task, allow_empty_string_abstain=True
        )
        output.append(
            {
                "schema_version": ROW_SCHEMA,
                "task_id": task["task_id"],
                "question_key": task["question_key"],
                "dataset": task["dataset"],
                "qid": task["qid"],
                "question_sha256": task["question_sha256"],
                "producer_slot": task["producer_slot"],
                "step_sha256": task["step_sha256"],
                "producer_passages_sha256": task["producer_passages_sha256"],
                "raw_response_sha256": telemetry["raw_response_sha256"],
                "p0_current": p0,
                "p1_surface_class_inferred": p1,
                "p2_contract_upper_bound": p2,
                "gold_access": False,
            }
        )
    return output


def _condition_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    reasons = Counter(str(row[key]["reason"]) for row in rows)
    parse_admitted = sum(bool(row[key]["condition_parse_admitted"]) for row in rows)
    original_strict_valid = sum(
        bool(row[key]["original_strict_parse_valid"]) for row in rows
    )
    verified = sum(bool(row[key]["verified"]) for row in rows)
    return {
        "tasks": len(rows),
        "condition_parse_admitted": parse_admitted,
        "condition_parse_admitted_rate": parse_admitted / len(rows),
        "original_strict_parse_valid": original_strict_valid,
        "original_strict_parse_rate": original_strict_valid / len(rows),
        "parser_semantics": rows[0][key]["parser_semantics"],
        "mechanically_verified": verified,
        "mechanically_verified_rate": verified / len(rows),
        "locality_pass": sum(bool(row[key]["locality_pass"]) for row in rows),
        "subject_echo": sum(bool(row[key]["subject_echo"]) for row in rows),
        "citation_or_locality_miss": sum(
            bool(row[key]["citation_or_locality_miss"]) for row in rows
        ),
        "verification_reasons": dict(sorted(reasons.items())),
    }


def summarize_counterfactual(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise CounterfactualAuditError("cannot summarize zero audit rows")
    by_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        selected = [row for row in rows if row["dataset"] == dataset]
        if not selected:
            raise CounterfactualAuditError(f"audit has no rows for {dataset}")
        p1_recovered = Counter(
            str(row["p0_current"]["reason"])
            for row in selected
            if not row["p0_current"]["verified"]
            and row["p1_surface_class_inferred"]["verified"]
        )
        p2_recovered = Counter(
            str(row["p1_surface_class_inferred"]["reason"])
            for row in selected
            if not row["p1_surface_class_inferred"]["verified"]
            and row["p2_contract_upper_bound"]["verified"]
        )
        by_dataset[dataset] = {
            "p0_current": _condition_summary(selected, "p0_current"),
            "p1_surface_class_inferred": _condition_summary(
                selected, "p1_surface_class_inferred"
            ),
            "p2_contract_upper_bound": _condition_summary(
                selected, "p2_contract_upper_bound"
            ),
            "p1_recovered_from_p0_reason": dict(sorted(p1_recovered.items())),
            "p2_recovered_from_p1_reason": dict(sorted(p2_recovered.items())),
            "p1_answer_type_changed": sum(
                row["p1_surface_class_inferred"]["condition_parse_admitted"]
                and row["p1_surface_class_inferred"]["original_answer_type"]
                != row["p1_surface_class_inferred"]["effective_answer_type"]
                for row in selected
            ),
            "p2_abstain_empty_string_coerced": sum(
                bool(row["p2_contract_upper_bound"]["abstain_empty_string_coerced"])
                for row in selected
            ),
        }
    overall = {
        key: _condition_summary(rows, key)
        for key in (
            "p0_current",
            "p1_surface_class_inferred",
            "p2_contract_upper_bound",
        )
    }
    return {"tasks": len(rows), "by_dataset": by_dataset, "overall": overall}


def _generator_p0_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        selected = [row for row in rows if row["dataset"] == dataset]
        condition = _condition_summary(selected, "p0_current")
        by_dataset[dataset] = {
            "tasks": condition["tasks"],
            "strict_parse_valid": condition["original_strict_parse_valid"],
            "strict_parse_rate": condition["original_strict_parse_rate"],
            "mechanically_verified": condition["mechanically_verified"],
            "mechanically_verified_rate": condition["mechanically_verified_rate"],
            "verification_reasons": condition["verification_reasons"],
        }
    return {
        "tasks": len(rows),
        "strict_parse_valid": sum(
            bool(row["p0_current"]["original_strict_parse_valid"]) for row in rows
        ),
        "mechanically_verified": sum(
            bool(row["p0_current"]["verified"]) for row in rows
        ),
        "by_dataset": by_dataset,
    }


def _assert_file_sha(path: Path, expected: str, *, label: str) -> dict[str, Any]:
    lock = file_lock(path)
    if lock["sha256"] != expected:
        raise CounterfactualAuditError(
            f"{label} SHA256 drift: expected={expected}, observed={lock['sha256']}"
        )
    return lock


def authenticate_frozen_inputs(
    *,
    tasks_path: Path = DEFAULT_TASKS,
    subanswers_path: Path = DEFAULT_SUBANSWERS,
    generator_report_path: Path = DEFAULT_GENERATOR_REPORT,
    generator_manifest_path: Path = DEFAULT_GENERATOR_MANIFEST,
    stage_descriptor_path: Path = DEFAULT_STAGE_DESCRIPTOR,
    implementation_lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK,
    plan_lock_path: Path = DEFAULT_PLAN_LOCK,
    preregistration_path: Path = DEFAULT_PREREGISTRATION,
    preregistration_manifest_path: Path = DEFAULT_PREREGISTRATION_MANIFEST,
    trajectory_lock_path: Path = DEFAULT_TRAJECTORY_LOCK,
    trajectory_manifest_path: Path = DEFAULT_TRAJECTORY_MANIFEST,
    project_root: Path = PROJECT_ROOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Authenticate the exact frozen v7 chain before evaluating any response."""

    paths = {
        "tasks": tasks_path,
        "subanswers": subanswers_path,
        "generator_report": generator_report_path,
        "generator_manifest": generator_manifest_path,
        "stage_descriptor": stage_descriptor_path,
        "implementation_lock": implementation_lock_path,
        "plan_lock": plan_lock_path,
        "preregistration": preregistration_path,
        "preregistration_manifest": preregistration_manifest_path,
        "trajectory_lock": trajectory_lock_path,
        "trajectory_manifest": trajectory_manifest_path,
    }
    resolved = {
        name: _safe_input(path, project_root=project_root, label=name)
        for name, path in paths.items()
    }
    locks = {
        name: _assert_file_sha(resolved[name], FROZEN_SHA256[name], label=name)
        for name in paths
    }
    implementation = _read_json(
        resolved["implementation_lock"], project_root=project_root, label="implementation"
    )
    plan = _read_json(resolved["plan_lock"], project_root=project_root, label="plan")
    preregistration = _read_json(
        resolved["preregistration"], project_root=project_root, label="preregistration"
    )
    preregistration_manifest = _read_json(
        resolved["preregistration_manifest"],
        project_root=project_root,
        label="preregistration_manifest",
    )
    trajectory = _read_json(
        resolved["trajectory_lock"], project_root=project_root, label="trajectory"
    )
    trajectory_manifest = _read_json(
        resolved["trajectory_manifest"],
        project_root=project_root,
        label="trajectory_manifest",
    )
    stage = _read_json(
        resolved["stage_descriptor"], project_root=project_root, label="stage"
    )
    generator_report = _read_json(
        resolved["generator_report"], project_root=project_root, label="generator_report"
    )
    generator_manifest = _read_json(
        resolved["generator_manifest"],
        project_root=project_root,
        label="generator_manifest",
    )

    if (
        implementation.get("schema_version")
        != "subquestion-dependent-retrieval-v7-implementation-lock-1"
        or implementation.get("status") != "AUTHORIZED_PLANNER_ONLY"
        or implementation.get("gold_access") is not False
    ):
        raise CounterfactualAuditError("implementation lock contract mismatch")
    if (
        plan.get("schema_version") != "subquestion-dependent-retrieval-v7-plan-lock-1"
        or plan.get("status") != "AUTHORIZED_GOLD_FREE_MATERIALIZATION"
        or plan.get("gold_access") is not False
    ):
        raise CounterfactualAuditError("plan lock contract mismatch")
    if (plan.get("parents") or {}).get("implementation_lock") != locks[
        "implementation_lock"
    ]:
        raise CounterfactualAuditError("plan does not bind the implementation lock")
    if (plan.get("parents") or {}).get("preregistration") != locks["preregistration"]:
        raise CounterfactualAuditError("plan does not bind the v7 preregistration")
    if (plan.get("parents") or {}).get("trajectory_semantics_addendum") != locks[
        "trajectory_lock"
    ]:
        raise CounterfactualAuditError("plan does not bind the recursive trajectory lock")
    if (
        preregistration.get("schema_version")
        != "subquestion-dependent-retrieval-v7-preregistration-1"
        or preregistration.get("status")
        != "FROZEN_COHORT_AND_RULES_BLOCKED_UNTIL_IMPLEMENTATION_HASH_LOCK"
        or preregistration.get("gold_access") is not False
    ):
        raise CounterfactualAuditError("v7 preregistration contract mismatch")
    frozen_threshold = (
        (preregistration.get("decision_gates") or {})
        .get("gold_free_mechanism", {})
        .get("mechanically_verified_subanswer_rate_min_each_dataset")
    )
    if frozen_threshold != FROZEN_GATE:
        raise CounterfactualAuditError("v7 preregistration threshold drift")
    if (
        preregistration_manifest.get("schema_version")
        != "dependent-retrieval-v7-preregistration-manifest-1"
        or preregistration_manifest.get("status") != preregistration.get("status")
        or preregistration_manifest.get("gold_access") is not False
        or (preregistration_manifest.get("artifacts") or {}).get("protocol")
        != locks["preregistration"]
    ):
        raise CounterfactualAuditError("v7 preregistration manifest mismatch")
    if (
        trajectory.get("schema_version")
        != "subquestion-dependent-retrieval-v7-recursive-trajectory-addendum-1"
        or trajectory.get("status") != "FROZEN_APPEND_ONLY_BEFORE_V7_EXECUTION"
        or trajectory.get("gold_access") is not False
        or (trajectory.get("parents") or {}).get("parent_preregistration")
        != locks["preregistration"]
        or (trajectory.get("parents") or {}).get("parent_preregistration_manifest")
        != locks["preregistration_manifest"]
    ):
        raise CounterfactualAuditError("recursive trajectory lock mismatch")
    if (
        trajectory_manifest.get("schema_version")
        != "subquestion-dependent-retrieval-v7-recursive-trajectory-addendum-manifest-1"
        or trajectory_manifest.get("status") != trajectory.get("status")
        or trajectory_manifest.get("gold_access") is not False
        or trajectory_manifest.get("protocol") != locks["trajectory_lock"]
    ):
        raise CounterfactualAuditError("recursive trajectory manifest mismatch")
    if (
        stage.get("schema_version") != "paired-dependent-retrieval-v7-stage-descriptor-2"
        or stage.get("runner_version")
        != "paired-dependent-retrieval-v7-staged-gold-free-1"
        or stage.get("stage") != "roots"
        or stage.get("state_depth") != 1
        or stage.get("gold_access") is not False
    ):
        raise CounterfactualAuditError("producer stage descriptor mismatch")
    runtime_locks = stage.get("runtime_locks") or {}
    if runtime_locks.get("implementation_lock") != locks["implementation_lock"]:
        raise CounterfactualAuditError("stage implementation lock mismatch")
    if runtime_locks.get("post_plan_execution_lock") != locks["plan_lock"]:
        raise CounterfactualAuditError("stage plan lock mismatch")
    if runtime_locks.get("preregistration") != locks["preregistration"]:
        raise CounterfactualAuditError("stage preregistration lock mismatch")
    if runtime_locks.get("trajectory_semantics_addendum") != locks["trajectory_lock"]:
        raise CounterfactualAuditError("stage trajectory lock mismatch")
    if (stage.get("outputs") or {}).get("c_tasks") != locks["tasks"]:
        raise CounterfactualAuditError("stage task output lock mismatch")

    closure = implementation.get("actual_local_import_closure") or {}
    verifier_lock = closure.get("kgproweight/retrieval/subanswer_v7.py")
    current_verifier = file_lock(Path(subanswer_v7.__file__))
    if verifier_lock != current_verifier:
        raise CounterfactualAuditError("frozen v7 verifier source has drifted")
    runtime_code = implementation.get("runtime_code") or {}
    generator_code_lock = runtime_code.get("subanswer_generator")
    if not isinstance(generator_code_lock, Mapping):
        raise CounterfactualAuditError("implementation lacks generator source lock")
    if file_lock(Path(str(generator_code_lock.get("path") or ""))) != generator_code_lock:
        raise CounterfactualAuditError("frozen v7 generator source has drifted")

    if (
        generator_report.get("schema_version")
        != "dependent-retrieval-v7-subanswer-report-1"
        or generator_report.get("status") != "COMPLETE_GOLD_FREE_SUBANSWERS"
        or generator_report.get("gold_access") is not False
        or generator_report.get("network_access") is not False
    ):
        raise CounterfactualAuditError("generator report contract mismatch")
    if generator_report.get("input") != {
        "path": str(resolved["tasks"]),
        "sha256": locks["tasks"]["sha256"],
        "rows": 41,
    }:
        raise CounterfactualAuditError("generator report input commitment mismatch")
    if generator_report.get("output") != {
        "path": str(resolved["subanswers"]),
        "sha256": locks["subanswers"]["sha256"],
        "rows": 41,
    }:
        raise CounterfactualAuditError("generator report output commitment mismatch")
    expected_authorization = {
        "implementation_lock": locks["implementation_lock"],
        "plan_lock": locks["plan_lock"],
        "producer_stage_descriptor": locks["stage_descriptor"],
    }
    if generator_report.get("authorization_locks") != expected_authorization:
        raise CounterfactualAuditError("generator authorization-lock mismatch")
    run = generator_manifest.get("run") or {}
    if (
        generator_manifest.get("status") != "COMPLETE_GOLD_FREE_SUBANSWERS"
        or run.get("experiment_id") != generator_report.get("experiment_id")
        or run.get("input_tasks_sha256") != locks["tasks"]["sha256"]
        or run.get("output_sha256") != locks["subanswers"]["sha256"]
        or run.get("report_sha256") != locks["generator_report"]["sha256"]
        or run.get("authorization_locks") != expected_authorization
        or run.get("gold_access") is not False
        or run.get("network_access") is not False
    ):
        raise CounterfactualAuditError("generator manifest commitment mismatch")

    tasks = _read_jsonl(resolved["tasks"], project_root=project_root, label="tasks")
    subanswers = _read_jsonl(
        resolved["subanswers"], project_root=project_root, label="subanswers"
    )
    validated = validate_and_join_rows(tasks, subanswers)
    tasks = [task for task, _ in validated]
    subanswers = [row for _, row in validated]
    return tasks, subanswers, {
        "input_locks": locks,
        "generator_report": generator_report,
        "generator_manifest": generator_manifest,
        "v7_verifier_source": current_verifier,
        "v7_generator_source": dict(generator_code_lock),
    }


def audit_frozen_inputs_in_memory(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Authenticate and evaluate the frozen artifacts without writing output."""

    tasks, subanswers, provenance = authenticate_frozen_inputs(**kwargs)
    rows = audit_counterfactual_rows(tasks, subanswers)
    summary = summarize_counterfactual(rows)
    _assert_frozen_reproduction(rows, summary, provenance)
    return rows, summary, provenance


def _assert_frozen_reproduction(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Require exact P0 cardinality/count reproduction for the pinned artifacts."""

    if summary["tasks"] != 41:
        raise CounterfactualAuditError("frozen audit must contain exactly 41 tasks")
    for dataset in DATASETS:
        observed = summary["by_dataset"][dataset]
        if observed["p0_current"]["tasks"] != EXPECTED_TASKS_BY_DATASET[dataset]:
            raise CounterfactualAuditError(f"frozen {dataset} task count drift")
        if (
            observed["p0_current"]["mechanically_verified"]
            != EXPECTED_P0_VERIFIED_BY_DATASET[dataset]
        ):
            raise CounterfactualAuditError(f"frozen {dataset} P0 count drift")
    if _generator_p0_counts(rows) != provenance["generator_report"].get("counts"):
        raise CounterfactualAuditError("P0 counts do not reproduce the generator report")


def _decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    p1_pass = all(
        summary["by_dataset"][dataset]["p1_surface_class_inferred"][
            "mechanically_verified_rate"
        ]
        >= FROZEN_GATE
        for dataset in DATASETS
    )
    p2_pass = all(
        summary["by_dataset"][dataset]["p2_contract_upper_bound"][
            "mechanically_verified_rate"
        ]
        >= FROZEN_GATE
        for dataset in DATASETS
    )
    if p1_pass:
        status = "PASS_P1_TYPE_ONLY_INTERFACE_DIAGNOSIS"
    elif p2_pass:
        status = "FAIL_P1_PASS_P2_UPPER_BOUND_ONLY"
    else:
        status = "FAIL_P2_INTERFACE_FIX_INSUFFICIENT"
    return {
        "status": status,
        "frozen_verified_rate_min_each_dataset": FROZEN_GATE,
        "p1_all_datasets_pass": p1_pass,
        "p2_all_datasets_pass": p2_pass,
        "gold_attachment_authorized": False,
        "answer_evaluation_authorized": False,
        "training_authorized": False,
        "v8_dynamic_route_authorized_by_phase0": False,
        "boundary": (
            "Phase 0 is a consumed-development interface diagnosis only; it is "
            "not an advancement gate for fresh v8 dynamic retrieval."
        ),
    }


def _condition_definitions() -> dict[str, str]:
    return {
        "p0_current": "unaltered strict v7 parser and verifier",
        "p1_surface_class_inferred": (
            "strict v7 parser; only verifier admission type becomes "
            "date -> number -> entity-like inferred from the unchanged answer surface"
        ),
        "p2_contract_upper_bound": (
            "P1 plus only abstain empty-string -> false when answer is non-empty "
            "and exactly one citation exists; optimistic diagnostic, not production parser"
        ),
    }


def build_protocol_document(
    *, experiment_id: str, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the exact critical fields required in the pre-execution protocol.

    The caller may add a timestamp or explanatory notes before freezing the
    file, but every field returned here is validated for exact equality by the
    formal runner.
    """

    return {
        "schema_version": PROTOCOL_SCHEMA,
        "experiment_id": experiment_id,
        "status": PROTOCOL_STATUS,
        "researcher_authorization": RESEARCHER_AUTHORIZATION,
        "scope": SCOPE,
        "gold_access": False,
        "authorization": {
            "phase0_gold_free_counterfactual": True,
            "gold_attachment": False,
            "answer_evaluation": False,
            "training": False,
        },
        "input_locks": provenance["input_locks"],
        "runtime_source_locks": {
            "audit": file_lock(Path(__file__)),
            "v7_verifier": provenance["v7_verifier_source"],
            "v7_generator": provenance["v7_generator_source"],
        },
        "conditions": _condition_definitions(),
        "decision_gate": {
            "datasets": list(DATASETS),
            "verified_rate_min_each_dataset": FROZEN_GATE,
            "phase0_is_not_v8_advancement_gate": True,
        },
        "scientific_boundary": (
            "Gold-free consumed-development interface diagnosis only; no semantic "
            "entailment, support recall, EM/F1/IHR, confirmation, or training claim."
        ),
    }


def authenticate_protocol(
    protocol_path: Path,
    *,
    experiment_id: str,
    provenance: Mapping[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Fail closed unless a protocol frozen against this exact code/input exists."""

    protocol = _read_json(protocol_path, project_root=project_root, label="protocol")
    required = build_protocol_document(
        experiment_id=experiment_id,
        provenance=provenance,
    )
    for field, expected in required.items():
        if protocol.get(field) != expected:
            raise CounterfactualAuditError(f"protocol critical field drift: {field}")
    result = dict(provenance)
    result["protocol"] = file_lock(protocol_path)
    return result


def freeze_protocol(
    *,
    protocol_path: Path,
    experiment_id: str,
    provenance: Mapping[str, Any] | None = None,
    project_root: Path = PROJECT_ROOT,
    **authentication_kwargs: Any,
) -> dict[str, Any]:
    """Exclusively freeze the Phase-0 protocol without executing the audit."""

    if not isinstance(experiment_id, str) or _EXPERIMENT_ID_RE.fullmatch(experiment_id) is None:
        raise CounterfactualAuditError("experiment_id must be a long uppercase stable identifier")
    if provenance is None:
        _, _, provenance = authenticate_frozen_inputs(
            project_root=project_root,
            **authentication_kwargs,
        )
    target = _safe_new_output(
        protocol_path,
        project_root=project_root,
        label="Phase-0 protocol",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    document = build_protocol_document(
        experiment_id=experiment_id,
        provenance=provenance,
    )
    document["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json_exclusive(target, document)
    # Re-read and authenticate the just-written bytes before reporting success.
    authenticate_protocol(
        target,
        experiment_id=experiment_id,
        provenance=provenance,
        project_root=project_root,
    )
    return document


def _write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True))
            handle.write("\n")


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        handle.write("\n")


def write_integrity_failure(
    *,
    output_dir: Path,
    experiment_id: str,
    provenance: Mapping[str, Any],
    error: Exception,
) -> None:
    """Persist a fail-stop record after authenticated P0 replay divergence."""

    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    manifest_path = output / "manifest.json"
    report = {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": experiment_id,
        "status": "FAIL_STOP_INTEGRITY_REPLAY_MISMATCH",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": SCOPE,
        "development_only": True,
        "gold_access": False,
        "network_access": False,
        "gpu_calls": 0,
        "model_calls": 0,
        "retrieval_calls": 0,
        "protocol": provenance["protocol"],
        "input_locks": provenance["input_locks"],
        "failure": {"type": type(error).__name__, "message": str(error)},
        "p0_replay_complete": False,
        "p1_p2_executed": False,
        "scientific_boundary": (
            "Authenticated P0 replay diverged. No P1/P2 count or scientific "
            "interpretation is emitted."
        ),
    }
    _write_json_exclusive(report_path, report)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "experiment_id": experiment_id,
        "status": "FAIL_STOP_INTEGRITY_REPLAY_MISMATCH",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_access": False,
        "audit_code": file_lock(Path(__file__)),
        "protocol": provenance["protocol"],
        "inputs": provenance["input_locks"],
        "report": file_lock(report_path),
    }
    _write_json_exclusive(manifest_path, manifest)


def write_audit_artifacts(
    *,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
    output_dir: Path,
    experiment_id: str,
) -> dict[str, Any]:
    """Write a new append-only report.  Existing directories are never reused."""

    if not isinstance(experiment_id, str) or _EXPERIMENT_ID_RE.fullmatch(experiment_id) is None:
        raise CounterfactualAuditError("experiment_id must be a long uppercase stable identifier")
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows_path = output / "per_task_counterfactual.jsonl"
    report_path = output / "report.json"
    manifest_path = output / "manifest.json"
    try:
        _write_jsonl_exclusive(rows_path, rows)
        decision = _decision(summary)
        report = {
            "schema_version": REPORT_SCHEMA,
            "experiment_id": experiment_id,
            "status": decision["status"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": SCOPE,
            "development_only": True,
            "gold_access": False,
            "network_access": False,
            "gpu_calls": 0,
            "model_calls": 0,
            "retrieval_calls": 0,
            "input_locks": provenance["input_locks"],
            "runtime_source_locks": {
                "audit": file_lock(Path(__file__)),
                "v7_verifier": provenance["v7_verifier_source"],
                "v7_generator": provenance["v7_generator_source"],
            },
            "protocol": provenance["protocol"],
            "conditions": _condition_definitions(),
            "summary": dict(summary),
            "gate_decision": decision,
            "output": file_lock(rows_path),
            "scientific_boundary": (
                "Gold-free, already-consumed development interface counterfactual. "
                "Mechanical verification means surface locality, not semantic entailment. "
                "No EM, F1, IHR, support-chain recall, confirmation, or training claim."
            ),
        }
        _write_json_exclusive(report_path, report)
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "experiment_id": experiment_id,
            "status": decision["status"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "gold_access": False,
            "gpu_calls": 0,
            "model_calls": 0,
            "retrieval_calls": 0,
            "audit_code": file_lock(Path(__file__)),
            "protocol": provenance["protocol"],
            "inputs": provenance["input_locks"],
            "per_task_counterfactual": file_lock(rows_path),
            "report": file_lock(report_path),
        }
        _write_json_exclusive(manifest_path, manifest)
        return report
    except Exception as exc:
        if not manifest_path.exists():
            failure = {
                "schema_version": MANIFEST_SCHEMA,
                "experiment_id": experiment_id,
                "status": "FAILED_RUNTIME_GOLD_FREE",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "gold_access": False,
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
            _write_json_exclusive(manifest_path, failure)
        raise


def run_audit(
    *,
    protocol_path: Path,
    output_dir: Path,
    experiment_id: str,
    tasks_path: Path = DEFAULT_TASKS,
    subanswers_path: Path = DEFAULT_SUBANSWERS,
    generator_report_path: Path = DEFAULT_GENERATOR_REPORT,
    generator_manifest_path: Path = DEFAULT_GENERATOR_MANIFEST,
    stage_descriptor_path: Path = DEFAULT_STAGE_DESCRIPTOR,
    implementation_lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK,
    plan_lock_path: Path = DEFAULT_PLAN_LOCK,
    preregistration_path: Path = DEFAULT_PREREGISTRATION,
    preregistration_manifest_path: Path = DEFAULT_PREREGISTRATION_MANIFEST,
    trajectory_lock_path: Path = DEFAULT_TRAJECTORY_LOCK,
    trajectory_manifest_path: Path = DEFAULT_TRAJECTORY_MANIFEST,
) -> dict[str, Any]:
    tasks, subanswers, provenance = authenticate_frozen_inputs(
        tasks_path=tasks_path,
        subanswers_path=subanswers_path,
        generator_report_path=generator_report_path,
        generator_manifest_path=generator_manifest_path,
        stage_descriptor_path=stage_descriptor_path,
        implementation_lock_path=implementation_lock_path,
        plan_lock_path=plan_lock_path,
        preregistration_path=preregistration_path,
        preregistration_manifest_path=preregistration_manifest_path,
        trajectory_lock_path=trajectory_lock_path,
        trajectory_manifest_path=trajectory_manifest_path,
    )
    provenance = authenticate_protocol(
        protocol_path,
        experiment_id=experiment_id,
        provenance=provenance,
    )
    try:
        rows = audit_counterfactual_rows(tasks, subanswers)
        summary = summarize_counterfactual(rows)
        _assert_frozen_reproduction(rows, summary, provenance)
    except CounterfactualAuditError as exc:
        write_integrity_failure(
            output_dir=output_dir,
            experiment_id=experiment_id,
            provenance=provenance,
            error=exc,
        )
        raise
    return write_audit_artifacts(
        rows=rows,
        summary=summary,
        provenance=provenance,
        output_dir=output_dir,
        experiment_id=experiment_id,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--subanswers", type=Path, default=DEFAULT_SUBANSWERS)
    parser.add_argument("--generator_report", type=Path, default=DEFAULT_GENERATOR_REPORT)
    parser.add_argument("--generator_manifest", type=Path, default=DEFAULT_GENERATOR_MANIFEST)
    parser.add_argument("--stage_descriptor", type=Path, default=DEFAULT_STAGE_DESCRIPTOR)
    parser.add_argument("--implementation_lock", type=Path, default=DEFAULT_IMPLEMENTATION_LOCK)
    parser.add_argument("--plan_lock", type=Path, default=DEFAULT_PLAN_LOCK)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument(
        "--preregistration_manifest",
        type=Path,
        default=DEFAULT_PREREGISTRATION_MANIFEST,
    )
    parser.add_argument("--trajectory_lock", type=Path, default=DEFAULT_TRAJECTORY_LOCK)
    parser.add_argument(
        "--trajectory_manifest", type=Path, default=DEFAULT_TRAJECTORY_MANIFEST
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--protocol",
        type=Path,
        help="append-only Phase-0 protocol frozen against exact input/code hashes",
    )
    mode.add_argument(
        "--freeze_protocol",
        type=Path,
        help="exclusively freeze the protocol and exit without running the audit",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment_id", default=DEFAULT_EXPERIMENT_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.freeze_protocol is not None:
        document = freeze_protocol(
            protocol_path=args.freeze_protocol,
            experiment_id=args.experiment_id,
            tasks_path=args.tasks,
            subanswers_path=args.subanswers,
            generator_report_path=args.generator_report,
            generator_manifest_path=args.generator_manifest,
            stage_descriptor_path=args.stage_descriptor,
            implementation_lock_path=args.implementation_lock,
            plan_lock_path=args.plan_lock,
            preregistration_path=args.preregistration,
            preregistration_manifest_path=args.preregistration_manifest,
            trajectory_lock_path=args.trajectory_lock,
            trajectory_manifest_path=args.trajectory_manifest,
        )
        print(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2))
        return
    report = run_audit(
        protocol_path=args.protocol,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
        tasks_path=args.tasks,
        subanswers_path=args.subanswers,
        generator_report_path=args.generator_report,
        generator_manifest_path=args.generator_manifest,
        stage_descriptor_path=args.stage_descriptor,
        implementation_lock_path=args.implementation_lock,
        plan_lock_path=args.plan_lock,
        preregistration_path=args.preregistration,
        preregistration_manifest_path=args.preregistration_manifest,
        trajectory_lock_path=args.trajectory_lock,
        trajectory_manifest_path=args.trajectory_manifest,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
