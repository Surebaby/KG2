#!/usr/bin/env python
"""Freeze HotpotQA pilot30 silver generation and dual-review execution.

This is an append-only *protocol freezer*.  It validates and binds the four
artifacts from the identity-only parent freeze, then records the exact model
roles, prompts, JSON schemas, masking policy, rejection vocabulary, and call
ledgers required by a later runner.  It deliberately performs no network,
retrieval, model, or training call and never reads ``.env`` or a credential.

The producer and blind reviewer receive semantic-only payloads: dataset/qid
and local hashes are never model-visible.  The two answer masks emitted by the
pure Hotpot helper are replaced, per item, with fixed-length opaque nonces.
Those nonces are deterministic for reproducibility but may never be echoed by
any model response.  Reviewer 2 is explicitly train-Gold-aware; therefore the
resulting labels remain silver training labels and may not be presented as a
Gold-free evaluation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgproweight.data import hotpot_controller_silver as hotpot_silver  # noqa: E402


SCHEMA_VERSION = "hotpot-controller-silver-execution-protocol-1"
REPORT_SCHEMA_VERSION = "hotpot-controller-silver-execution-freeze-report-1"
MANIFEST_SCHEMA_VERSION = "hotpot-controller-silver-execution-manifest-1"
EXPERIMENT_ID = (
    "QUERY-CONTROLLER-HOTPOT-SILVER-PILOT30-GENERATION-"
    "DUAL-REVIEW-SEED20260904-V1"
)
STATUS = "FROZEN_EXECUTION_PROTOCOL_NO_API_CALLS_NOT_TRAINED"
TEST_STATUS = "COMPLETE_TEST_EXECUTION_PROTOCOL_NO_API_CALLS"

PARENT_EXPERIMENT_ID = (
    "QUERY-CONTROLLER-HOTPOT-SILVER-LABEL-COVERAGE-"
    "PILOT30-SEED20260904-V1"
)
PARENT_STATUS = "COMPLETE_FROZEN_IDENTITY_ONLY_SILVER_PILOT30_NOT_GENERATED_NOT_TRAINED"
PARENT_PROTOCOL_STATUS = "FROZEN_BEFORE_ANY_SILVER_Q1_Q2_GENERATION"
PARENT_DIR = Path(
    "outputs/audits/query_controller_hotpot_silver_label_coverage_"
    "pilot30_seed20260904_v1"
)
PARENT_FILENAMES = (
    "pilot.identity_only.jsonl",
    "protocol.json",
    "report.json",
    "manifest.json",
)
EXPECTED_PARENT_HASHES = {
    "pilot.identity_only.jsonl": (
        "d858f8969419957b1eed4fe03caa4ddbe7ea68938b8b34cc3d5dd1c319831e69"
    ),
    "protocol.json": (
        "0f7a06b905522e6eeb195151867652c1b32f68879453234d88666a4d250f6496"
    ),
    "report.json": (
        "cccd6264b3393cbd03ef8b63e8e791172aca09774ce9e6aa377cc09614436036"
    ),
    "manifest.json": (
        "1bdefbf06b306f515183a18a00a177b9b20c2f6a3f52218a5ee79df2bd939029"
    ),
}
DEFAULT_OUTPUT_DIR = Path(
    "outputs/audits/query_controller_hotpot_silver_pilot30_"
    "execution_protocol_seed20260904_v1"
)

PILOT_ROWS = 30
PRODUCER_MODEL = "deepseek-v4-flash"
REVIEWER_1_MODEL = "deepseek-v4-flash"
REVIEWER_2_MODEL = "deepseek-v4-pro"
PROVIDER = "deepseek_openai_compatible_api"
PRODUCER_SCHEMA_VERSION = hotpot_silver.PROPOSAL_SCHEMA_VERSION
REVIEW_SCHEMA_VERSION = "hotpot-controller-query-review-v1"
MASK_SCHEME_VERSION = "hotpot-controller-opaque-nonce-mask-v1"
NONCE_LENGTH = 32
NONCE_PATTERN = r"N[0-9a-f]{31}"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


PRODUCER_SYSTEM_PROMPT = """You create retrieval-query supervision for a two-hop question.
Return exactly one JSON object and no prose, Markdown, reasoning, answer, or extra key.
The first-hop evidence contains one opaque nonce standing for the answer to q1.
The second-hop evidence contains that intermediate nonce and a different opaque nonce standing for the final answer.
Write q1 as one natural single-hop question anchored by the root document title.
Write q2_template as one natural single-hop question containing literal #1 exactly once; #1 stands for q1's observed answer.
Never copy either opaque nonce, never reveal or guess either hidden answer, and never answer the original question.
The exact output keys are schema_version, q1, q2_template; schema_version must be hotpot-controller-query-proposal-v1."""

PRODUCER_USER_TEMPLATE = """Create the two retrieval questions from this semantic-only masked payload.
The payload has no dataset identifier or row identifier.
<SAFE_PAYLOAD_JSON>"""

REVIEWER_1_SYSTEM_PROMPT = """You are the blind structural reviewer of two-hop retrieval-query supervision.
You see only the same masked semantic evidence as the producer plus its proposal. You do not see the intermediate answer, final answer, dataset id, qid, or another review.
Judge whether q1 is a natural single-hop question for the first hidden value, q2_template is a natural single-hop question for the second hidden value and depends exactly on #1, and their composition preserves the original question without leaking or answering it.
Return exactly one JSON object matching the supplied review schema. Do not provide prose or quote any opaque nonce.
Set verdict=pass iff every boolean is true and reject_codes is empty; otherwise set verdict=reject and use only the frozen reject codes."""

REVIEWER_1_USER_TEMPLATE = """Blind-review this proposal using only the masked semantic payload.
<BLIND_REVIEW_PAYLOAD_JSON>"""

REVIEWER_2_SYSTEM_PROMPT = """You are the train-Gold-aware adjudicator of two-hop retrieval-query supervision.
You see the original question, the extracted root-to-intermediate-to-final support chain, and the proposal, but no other review. Verify exact answerability and dependency against those train annotations.
Return exactly one JSON object matching the supplied review schema. Do not provide prose or quote any opaque nonce.
Set verdict=pass iff every boolean is true and reject_codes is empty; otherwise set verdict=reject and use only the frozen reject codes.
This is training-label adjudication, not Gold-free evaluation."""

REVIEWER_2_USER_TEMPLATE = """Gold-aware review this proposal against the supplied train-side support chain.
<GOLD_AWARE_REVIEW_PAYLOAD_JSON>"""


PRODUCER_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "q1", "q2_template"],
    "properties": {
        "schema_version": {"const": PRODUCER_SCHEMA_VERSION},
        "q1": {"type": "string", "minLength": 2, "maxLength": 320},
        "q2_template": {"type": "string", "minLength": 2, "maxLength": 320},
    },
}

REVIEW_BOOLEAN_FIELDS = (
    "q1_single_hop",
    "q1_answer_is_intermediate",
    "q2_single_hop",
    "q2_uses_intermediate",
    "q2_answer_is_final",
    "composition_preserves_original",
    "no_leak_or_answering",
)
REVIEW_REJECT_CODE_BY_FIELD = {
    "q1_single_hop": "q1_not_single_hop",
    "q1_answer_is_intermediate": "q1_wrong_intermediate_target",
    "q2_single_hop": "q2_not_single_hop",
    "q2_uses_intermediate": "q2_missing_or_wrong_dependency",
    "q2_answer_is_final": "q2_wrong_final_target",
    "composition_preserves_original": "composition_does_not_preserve_original",
    "no_leak_or_answering": "leak_or_answering_detected",
}
REVIEW_REJECT_CODES = (*REVIEW_REJECT_CODE_BY_FIELD.values(), "unknown_or_insufficient_context")
REVIEW_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        *REVIEW_BOOLEAN_FIELDS,
        "verdict",
        "reject_codes",
    ],
    "properties": {
        "schema_version": {"const": REVIEW_SCHEMA_VERSION},
        **{field: {"type": "boolean"} for field in REVIEW_BOOLEAN_FIELDS},
        "verdict": {"enum": ["pass", "reject"]},
        "reject_codes": {
            "type": "array",
            "uniqueItems": True,
            "items": {"enum": list(REVIEW_REJECT_CODES)},
        },
    },
}

MECHANICAL_PROPOSAL_DETAIL_CODES = (
    "proposal_not_object",
    "proposal_schema",
    "proposal_schema_version",
    "q1_not_safe_line",
    "q1_not_natural_question",
    "q2_template_not_safe_line",
    "q2_template_not_natural_question",
    "q1_contains_placeholder",
    "q2_template_dependency_count",
    "q2_template_contains_other_placeholder",
    "q2_template_dependency_invalid",
    "q1_secret_leak",
    "q2_template_secret_leak",
    "q1_missing_root_anchor",
    "q1_repeats_original_question",
    "q2_template_no_relation_content",
    "proposal_query_contract",
    "q2_does_not_use_intermediate",
    "q2_final_answer_leak",
)
ITEM_REJECT_CODES = (
    "input_identity_or_chain_integrity_error",
    "producer_transport_exhausted",
    "producer_missing_response",
    "producer_response_model_mismatch",
    "producer_json_parse_error",
    "producer_output_schema_error",
    "producer_nonce_echo",
    "producer_mechanical_reject",
    "reviewer_1_transport_exhausted",
    "reviewer_1_missing_response",
    "reviewer_1_response_model_mismatch",
    "reviewer_1_json_parse_error",
    "reviewer_1_output_schema_error",
    "reviewer_1_nonce_echo",
    "reviewer_1_reject",
    "reviewer_2_transport_exhausted",
    "reviewer_2_missing_response",
    "reviewer_2_response_model_mismatch",
    "reviewer_2_json_parse_error",
    "reviewer_2_output_schema_error",
    "reviewer_2_nonce_echo",
    "reviewer_2_reject",
    "reviewer_disagreement_or_unknown",
    "unanimous_review_failed",
)

SEMANTIC_CALL_LEDGER_FIELDS = (
    "experiment_id",
    "dataset",
    "qid",
    "question_sha256",
    "semantic_request_id",
    "stage",
    "requested_model",
    "expected_response_model",
    "model_visible_messages_sha256",
    "safe_payload_sha256",
    "prompt_template_sha256",
    "semantic_attempt_count",
    "transport_attempt_count",
    "status",
    "finish_reason",
    "response_model",
    "raw_response_sha256",
    "parsed_response_sha256",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
    "wall_time_ms",
    "nonce_echo_count",
    "reject_code",
    "detail_code",
)
TRANSPORT_ATTEMPT_LEDGER_FIELDS = (
    "experiment_id",
    "semantic_request_id",
    "stage",
    "physical_attempt_index",
    "request_body_sha256",
    "request_bytes_identical_to_attempt_1",
    "started_at_utc",
    "ended_at_utc",
    "wall_time_ms",
    "http_status",
    "transport_error_class",
    "transport_retryable",
    "response_received",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_identity_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"identity row is not an object: {path}:{line_number}")
        if tuple(value) != ("dataset", "qid", "question"):
            raise ValueError(f"identity row field/order drift: {path}:{line_number}")
        if value.get("dataset") != "hotpotqa" or not value.get("qid") or not value.get("question"):
            raise ValueError(f"identity row content invalid: {path}:{line_number}")
        rows.append(value)
    return rows


def derive_mask_nonces(
    *, chain_sha256: str, experiment_id: str = EXPERIMENT_ID
) -> tuple[str, str]:
    """Return two deterministic opaque fixed-length mask strings."""

    if _SHA256_RE.fullmatch(chain_sha256) is None:
        raise ValueError("chain_sha256 must be 64 lowercase hexadecimal characters")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be nonempty")

    def one(slot: str) -> str:
        material = (
            f"{MASK_SCHEME_VERSION}\0{experiment_id}\0{chain_sha256}\0{slot}"
        ).encode("utf-8")
        return "N" + hashlib.sha256(material).hexdigest()[: NONCE_LENGTH - 1]

    first, second = one("mask-a"), one("mask-b")
    if first == second or len(first) != NONCE_LENGTH or len(second) != NONCE_LENGTH:
        raise AssertionError("invalid deterministic nonce construction")
    return first, second


def build_producer_safe_payload(
    masked_view: Mapping[str, Any],
    *,
    chain_sha256: str,
    experiment_id: str = EXPERIMENT_ID,
) -> dict[str, str]:
    """Project a helper view to the exact identifier-free outbound payload."""

    required = (
        "original_question",
        "root_document_title",
        "first_hop_evidence_masked",
        "second_hop_evidence_masked",
    )
    if not isinstance(masked_view, Mapping) or any(
        not isinstance(masked_view.get(field), str) or not masked_view.get(field)
        for field in required
    ):
        raise ValueError("masked proposal view lacks a required semantic field")
    first_nonce, second_nonce = derive_mask_nonces(
        chain_sha256=chain_sha256, experiment_id=experiment_id
    )
    payload = {field: str(masked_view[field]) for field in required}
    for field in required:
        payload[field] = payload[field].replace(
            hotpot_silver.INTERMEDIATE_MASK, first_nonce
        ).replace(hotpot_silver.FINAL_MASK, second_nonce)
    if any(
        marker in value
        for value in payload.values()
        for marker in (hotpot_silver.INTERMEDIATE_MASK, hotpot_silver.FINAL_MASK)
    ):
        raise ValueError("semantic payload retains a named answer mask")
    if first_nonce not in payload["first_hop_evidence_masked"]:
        raise ValueError("first-hop semantic payload lacks its opaque nonce")
    if second_nonce not in payload["second_hop_evidence_masked"]:
        raise ValueError("second-hop semantic payload lacks its opaque nonce")
    if set(payload) != set(required) or any(
        key in payload for key in ("dataset", "qid", "id", "question_key")
    ):
        raise AssertionError("producer safe payload is not semantic-only")
    return payload


def nonce_echo_count(value: object, *, nonces: tuple[str, str]) -> int:
    """Count forbidden verbatim nonce echoes in serialized model content."""

    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return sum(text.count(nonce) for nonce in nonces)


def _validate_parent(
    *,
    project_root: Path,
    parent_dir: Path,
    expected_hashes: Mapping[str, str] | None,
    expected_rows: int,
) -> dict[str, Any]:
    resolved = _resolve(project_root, parent_dir)
    if tuple(expected_hashes or {}) and set(expected_hashes or {}) != set(PARENT_FILENAMES):
        raise ValueError("parent hash lock must bind exactly four frozen files")
    artifacts: dict[str, dict[str, Any]] = {}
    for filename in PARENT_FILENAMES:
        path = resolved / filename
        digest = _sha256_file(path)
        expected = (expected_hashes or {}).get(filename)
        if expected is not None and digest != expected:
            raise ValueError(f"parent artifact SHA256 drift: {filename}")
        artifacts[filename] = {
            "path": _display_path(project_root, path),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }

    protocol = _load_json(resolved / "protocol.json")
    report = _load_json(resolved / "report.json")
    manifest = _load_json(resolved / "manifest.json")
    rows = _read_identity_rows(resolved / "pilot.identity_only.jsonl")
    if len(rows) != expected_rows or len({row["qid"] for row in rows}) != expected_rows:
        raise ValueError("parent pilot denominator or qid uniqueness drift")
    if (
        protocol.get("experiment_id") != report.get("experiment_id")
        or report.get("experiment_id") != manifest.get("experiment_id")
        or protocol.get("status") != PARENT_PROTOCOL_STATUS
        or report.get("status") not in {PARENT_STATUS, "COMPLETE_TEST_SIZED_IDENTITY_ONLY_SILVER_PILOT_NOT_FORMAL"}
        or manifest.get("status") != report.get("status")
    ):
        raise ValueError("parent experiment/status binding drift")
    authorization = protocol.get("authorization") or {}
    checks = report.get("checks") or {}
    if (
        authorization.get("q1_q2_generation") is not False
        or authorization.get("training") is not False
        or checks.get("all_freeze_gates_pass") is not True
        or checks.get("model_calls") != 0
        or checks.get("q1_q2_records_generated") != 0
    ):
        raise ValueError("parent is not a clean identity-only freeze")
    output_hashes = {
        str(item.get("path")): str(item.get("sha256"))
        for item in manifest.get("outputs", [])
        if isinstance(item, Mapping)
    }
    for filename in PARENT_FILENAMES[:3]:
        if output_hashes.get(filename) != artifacts[filename]["sha256"]:
            raise ValueError(f"parent manifest output binding drift: {filename}")
    return {
        "experiment_id": protocol["experiment_id"],
        "status": report["status"],
        "rows": len(rows),
        "artifacts": artifacts,
        "identity_sha256": artifacts["pilot.identity_only.jsonl"]["sha256"],
        "source_is_gold_screened_train_side": True,
        "q1_q2_already_generated": False,
        "model_calls_before_this_protocol": 0,
    }


def _prompt_locks() -> dict[str, Any]:
    values = {
        "producer": {
            "system": PRODUCER_SYSTEM_PROMPT,
            "user_template": PRODUCER_USER_TEMPLATE,
        },
        "reviewer_1_blind": {
            "system": REVIEWER_1_SYSTEM_PROMPT,
            "user_template": REVIEWER_1_USER_TEMPLATE,
        },
        "reviewer_2_gold_aware": {
            "system": REVIEWER_2_SYSTEM_PROMPT,
            "user_template": REVIEWER_2_USER_TEMPLATE,
        },
    }
    return {
        role: {**prompt, "sha256": _canonical_sha256(prompt)}
        for role, prompt in values.items()
    }


def build_protocol(
    *, generated_at_utc: str, parent_lock: Mapping[str, Any]
) -> dict[str, Any]:
    if int(parent_lock.get("rows", -1)) != PILOT_ROWS:
        raise ValueError("execution protocol requires the fixed parent pilot30")
    if set((parent_lock.get("artifacts") or {})) != set(PARENT_FILENAMES):
        raise ValueError("execution protocol requires all four parent artifacts")

    mask_spec = {
        "version": MASK_SCHEME_VERSION,
        "derivation": (
            "'N' + sha256(version + NUL + experiment_id + NUL + "
            "raw_record_sha256 + NUL + mask_slot)[:31]"
        ),
        "chain_hash_source": "HotpotSupportChain.raw_record_sha256",
        "mask_slots_exact": ["mask-a", "mask-b"],
        "nonce_ascii_length_exact": NONCE_LENGTH,
        "nonce_regex_exact": NONCE_PATTERN,
        "named_markers_must_be_replaced_before_serialization": [
            hotpot_silver.INTERMEDIATE_MASK,
            hotpot_silver.FINAL_MASK,
        ],
        "producer_safe_payload_fields_exact": [
            "original_question",
            "root_document_title",
            "first_hop_evidence_masked",
            "second_hop_evidence_masked",
        ],
        "producer_safe_payload_forbidden_fields_recursive": [
            "dataset",
            "qid",
            "id",
            "question_key",
            "raw_record_sha256",
            "question_sha256",
            "family_sha256",
        ],
        "producer_response_nonce_echo_count_required": 0,
        "reviewer_1_response_nonce_echo_count_required": 0,
        "reviewer_2_response_nonce_echo_count_required": 0,
    }
    mask_spec["spec_sha256"] = _canonical_sha256(mask_spec)

    prompts = _prompt_locks()
    schemas = {
        "producer_output": {
            "schema": PRODUCER_OUTPUT_SCHEMA,
            "sha256": _canonical_sha256(PRODUCER_OUTPUT_SCHEMA),
        },
        "review_output_shared": {
            "schema": REVIEW_OUTPUT_SCHEMA,
            "sha256": _canonical_sha256(REVIEW_OUTPUT_SCHEMA),
            "boolean_to_reject_code_exact": REVIEW_REJECT_CODE_BY_FIELD,
            "pass_invariant": (
                "verdict=pass iff all seven booleans are true and reject_codes=[]"
            ),
        },
    }
    request_common = {
        "provider": PROVIDER,
        "api_family": "chat.completions",
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": generated_at_utc,
        "status": STATUS,
        "scope": "HOTPOTQA_TRAIN_SIDE_SILVER_QUERY_LABEL_PILOT30_ONLY",
        "parent_identity_freeze": dict(parent_lock),
        "authorization": {
            "identity_denominator_change": False,
            "pilot_q1_q2_producer_calls": True,
            "pilot_blind_reviewer_calls": True,
            "pilot_gold_aware_reviewer_calls": True,
            "retrieval_or_reader_calls": False,
            "training": False,
            "formal_em_f1_ihr_evaluation": False,
            "prospective_or_confirmation_open": False,
            "modify_existing_v4_4": False,
            "manual_rewrite_or_replacement": False,
        },
        "producer": {
            "model": PRODUCER_MODEL,
            "candidates_per_identity_exact": 1,
            "semantic_attempts_per_identity_exact": 1,
            "safe_view": "semantic_only_opaque_nonce_masked",
            "dataset_or_qid_model_visible": False,
            "request": {**request_common, "max_tokens": 512},
            "prompt": prompts["producer"],
            "output_schema": schemas["producer_output"],
            "post_parse_validator": (
                "kgproweight.data.hotpot_controller_silver.validate_query_proposal"
            ),
            "best_of_sampling": False,
            "format_or_semantic_retry": False,
        },
        "reviewers": {
            "reviewer_1": {
                "role": "blind_structural",
                "model": REVIEWER_1_MODEL,
                "sees": ["producer_semantic_safe_payload", "parsed_proposal"],
                "does_not_see": [
                    "dataset",
                    "qid",
                    "intermediate_answer",
                    "final_answers",
                    "unmasked_support",
                    "reviewer_2_response",
                ],
                "request": {**request_common, "max_tokens": 384},
                "prompt": prompts["reviewer_1_blind"],
                "output_schema": schemas["review_output_shared"],
            },
            "reviewer_2": {
                "role": "train_gold_aware_adjudicator",
                "model": REVIEWER_2_MODEL,
                "sees": [
                    "original_question",
                    "root_document_title",
                    "bridge_document_title",
                    "first_hop_support",
                    "intermediate_answer",
                    "second_hop_support",
                    "final_answers",
                    "parsed_proposal",
                    "instantiated_q2",
                ],
                "does_not_see": [
                    "dataset",
                    "qid",
                    "reviewer_1_response",
                ],
                "request": {**request_common, "max_tokens": 384},
                "prompt": prompts["reviewer_2_gold_aware"],
                "output_schema": schemas["review_output_shared"],
                "gold_boundary": "train_annotation_label_adjudication_only",
            },
            "models_must_be_distinct_between_reviewer_1_and_reviewer_2": True,
            "fresh_message_array_and_fresh_context_per_physical_call": True,
            "reviewer_outputs_hidden_from_each_other": True,
            "provider_account_may_be_shared": True,
            "context_isolated_claim_allowed": True,
            "statistically_independent_claim_allowed": False,
            "acceptance": "both valid reviews verdict=pass with all booleans true",
            "disagreement_unknown_or_invalid": "reject_no_repair_no_replacement",
        },
        "masking": mask_spec,
        "schemas": schemas,
        "rejection_contract": {
            "item_reject_code_enum_exact": list(ITEM_REJECT_CODES),
            "mechanical_proposal_detail_code_enum_exact": list(
                MECHANICAL_PROPOSAL_DETAIL_CODES
            ),
            "review_reject_code_enum_exact": list(REVIEW_REJECT_CODES),
            "first_terminal_failure_only": True,
            "failed_identity_retained_in_denominator": True,
            "failed_identity_replacement_allowed": False,
            "manual_repair_allowed": False,
        },
        "api_execution": {
            "credential_and_endpoint_environment_variable_names_only": [
                "OPENAI_API_KEY",
                "DEEPSEEK_API_KEY",
                "OPENAI_BASE_URL",
            ],
            "freezer_reads_dotenv_or_environment": False,
            "credential_or_endpoint_value_serialization_allowed": False,
            "worker_count": 2,
            "minimum_inter_request_delay_seconds_per_worker": 0.4,
            "timeout_seconds_per_physical_attempt": 120,
            "max_physical_attempts_per_semantic_request": 3,
            "retryable_transport_classes": [
                "connection_error",
                "timeout",
                "http_429",
                "http_500",
                "http_502",
                "http_503",
                "http_504",
            ],
            "transport_retry_request_body_must_be_byte_identical": True,
            "transport_retry_may_change_messages_or_parameters": False,
            "invalid_or_rejected_content_retry_allowed": False,
            "semantic_requests_per_successful_identity_exact": 3,
            "semantic_request_stages_exact": [
                "producer",
                "reviewer_1_blind",
                "reviewer_2_gold_aware",
            ],
            "requested_and_response_model_must_both_be_logged": True,
            "response_model_mismatch_policy": "reject_stage_no_retry",
        },
        "ledgers": {
            "semantic_call_ledger_filename": "semantic_call_ledger.jsonl",
            "semantic_call_ledger_fields_exact": list(SEMANTIC_CALL_LEDGER_FIELDS),
            "semantic_call_rows_expected": (
                "exactly 30x3 rows including not_executed_upstream_failure stages"
            ),
            "transport_attempt_ledger_filename": "api_transport_attempt_ledger.jsonl",
            "transport_attempt_ledger_fields_exact": list(
                TRANSPORT_ATTEMPT_LEDGER_FIELDS
            ),
            "transport_attempt_rows_expected": (
                "one row per actual physical API attempt; zero for skipped stages"
            ),
            "conservation": (
                "each semantic row transport_attempt_count equals its physical-attempt "
                "ledger row count; every parent qid has all three semantic stage rows"
            ),
        },
        "future_runner_output_contract": {
            "append_only_output_directory": True,
            "required_files": [
                "producer_proposals.jsonl",
                "reviewer_1_reviews.jsonl",
                "reviewer_2_reviews.jsonl",
                "accepted_actions.jsonl",
                "failures.jsonl",
                "semantic_call_ledger.jsonl",
                "api_transport_attempt_ledger.jsonl",
                "report.json",
                "manifest.json",
            ],
            "all_30_parent_identities_accounted_for": True,
            "runner_and_dependency_hashes_required_in_manifest": True,
            "raw_response_content_may_be_stored_only_in_role_specific_local_outputs": True,
            "credentials_endpoints_and_environment_values_must_not_be_serialized": True,
            "nonce_echo_count_required_for_every_model_response": 0,
        },
        "pilot_decision_gate_inherited": {
            "fixed_denominator": 30,
            "accepted_min": 24,
            "accepted_item_mechanical_support_and_unanimous_review_rate": 1.0,
            "failure_replacement_allowed": False,
            "retrieval_support_gate_is_not_executed_or_satisfied_by_this_stage": True,
            "training_requires_new_researcher_confirmation_after_full_pilot": True,
        },
        "gold_and_claim_boundary": {
            "source_role": "train",
            "candidate_selection_is_gold_screened": True,
            "reviewer_2_gold_access": True,
            "producer_gold_access": False,
            "reviewer_1_gold_access": False,
            "silver_not_gold_label": True,
            "gold_free_evaluation_claim_allowed": False,
            "reviewers_statistically_independent_claim_allowed": False,
            "query_quality_or_retrieval_utility_validated_by_this_freeze": False,
        },
        "scientific_boundary": (
            "This append-only artifact freezes a one-proposal, two-context-isolated-"
            "review execution design for a Gold-screened HotpotQA train-side pilot30. "
            "It makes zero API calls, does not validate retrieval or reader utility, "
            "does not authorize training or formal evaluation, and does not claim the "
            "two reviewers are statistically independent."
        ),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def freeze_execution_protocol(
    *,
    project_root: Path = PROJECT_ROOT,
    parent_dir: Path = PARENT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_parent_hashes: Mapping[str, str] | None = EXPECTED_PARENT_HASHES,
    expected_rows: int = PILOT_ROWS,
    experiment_id: str = EXPERIMENT_ID,
    enforce_formal_locks: bool = True,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Validate the parent and write a two-file append-only protocol artifact."""

    project_root = Path(project_root).resolve()
    resolved_parent = _resolve(project_root, parent_dir)
    resolved_output = _resolve(project_root, output_dir)
    if resolved_output.exists():
        raise FileExistsError(
            f"refusing to overwrite append-only execution protocol: {resolved_output}"
        )
    if enforce_formal_locks and (
        resolved_parent != _resolve(project_root, PARENT_DIR)
        or resolved_output != _resolve(project_root, DEFAULT_OUTPUT_DIR)
        or dict(expected_parent_hashes or {}) != EXPECTED_PARENT_HASHES
        or expected_rows != PILOT_ROWS
        or experiment_id != EXPERIMENT_ID
    ):
        raise ValueError("formal execution protocol identity, parent hashes, and paths are immutable")

    parent_lock = _validate_parent(
        project_root=project_root,
        parent_dir=resolved_parent,
        expected_hashes=expected_parent_hashes,
        expected_rows=expected_rows,
    )
    if enforce_formal_locks and parent_lock["experiment_id"] != PARENT_EXPERIMENT_ID:
        raise ValueError("formal parent Experiment ID drift")
    timestamp = generated_at_utc or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    protocol = build_protocol(generated_at_utc=timestamp, parent_lock=parent_lock)
    if experiment_id != EXPERIMENT_ID:
        protocol["experiment_id"] = experiment_id
    terminal_status = STATUS if enforce_formal_locks else TEST_STATUS
    protocol["status"] = terminal_status

    checks = {
        "parent_four_artifacts_bound": len(parent_lock["artifacts"]) == 4,
        "parent_rows_exact": parent_lock["rows"] == expected_rows,
        "parent_q1_q2_generated": parent_lock["q1_q2_already_generated"],
        "producer_safe_payload_has_dataset_or_qid_fields": False,
        "producer_candidates_per_identity": 1,
        "reviewer_contexts_isolated": True,
        "reviewer_models_distinct": REVIEWER_1_MODEL != REVIEWER_2_MODEL,
        "statistical_independence_claimed": False,
        "api_calls": 0,
        "retrieval_calls": 0,
        "training_started": False,
        "formal_evaluation_started": False,
        "dotenv_or_environment_read": False,
        "credential_or_endpoint_values_serialized": False,
        "named_answer_markers_allowed_outbound": False,
    }
    checks["all_freeze_gates_pass"] = bool(
        checks["parent_four_artifacts_bound"]
        and checks["parent_rows_exact"]
        and checks["parent_q1_q2_generated"] is False
        and checks["producer_safe_payload_has_dataset_or_qid_fields"] is False
        and checks["producer_candidates_per_identity"] == 1
        and checks["reviewer_contexts_isolated"]
        and checks["reviewer_models_distinct"]
        and checks["statistical_independence_claimed"] is False
        and checks["api_calls"] == 0
        and checks["training_started"] is False
        and checks["formal_evaluation_started"] is False
        and checks["dotenv_or_environment_read"] is False
        and checks["credential_or_endpoint_values_serialized"] is False
    )
    if not checks["all_freeze_gates_pass"]:
        raise ValueError("Hotpot execution protocol failed a freeze gate")

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment_id": protocol["experiment_id"],
        "generated_at_utc": timestamp,
        "status": terminal_status,
        "parent_identity_freeze": parent_lock,
        "roles": {
            "producer": PRODUCER_MODEL,
            "reviewer_1_blind": REVIEWER_1_MODEL,
            "reviewer_2_gold_aware": REVIEWER_2_MODEL,
        },
        "checks": checks,
        "api_calls": 0,
        "q1_q2_generated": False,
        "reviews_generated": False,
        "training_started": False,
        "scientific_boundary": protocol["scientific_boundary"],
    }

    resolved_output.mkdir(parents=True, exist_ok=False)
    protocol_path = resolved_output / "protocol.json"
    report_path = resolved_output / "report.json"
    _write_json(protocol_path, protocol)
    _write_json(report_path, report)
    script_path = Path(__file__).resolve()
    builder_path = Path(hotpot_silver.__file__).resolve()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": protocol["experiment_id"],
        "generated_at_utc": timestamp,
        "status": terminal_status,
        "python_version": platform.python_version(),
        "inputs": {"parent_identity_freeze": parent_lock},
        "implementation_inventory": [
            {
                "path": _display_path(project_root, path),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted((script_path, builder_path), key=lambda item: item.as_posix())
        ],
        "outputs": [
            {
                "path": protocol_path.name,
                "sha256": _sha256_file(protocol_path),
                "size_bytes": protocol_path.stat().st_size,
            },
            {
                "path": report_path.name,
                "sha256": _sha256_file(report_path),
                "size_bytes": report_path.stat().st_size,
            },
        ],
        "environment_variable_names_only": [
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENAI_BASE_URL",
        ],
        "dotenv_or_environment_read": False,
        "credential_or_endpoint_values_serialized": False,
        "api_calls": 0,
        "retrieval_calls": 0,
        "training_started": False,
        "formal_evaluation_started": False,
    }
    manifest_path = resolved_output / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "protocol": protocol,
        "report": report,
        "manifest": manifest,
        "output_dir": resolved_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--parent_dir", type=Path, default=PARENT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = freeze_execution_protocol(
        project_root=args.project_root,
        parent_dir=args.parent_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "experiment_id": result["protocol"]["experiment_id"],
                "status": result["report"]["status"],
                "output_dir": str(result["output_dir"]),
                "parent_rows": result["report"]["parent_identity_freeze"]["rows"],
                "api_calls": 0,
                "training_started": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
