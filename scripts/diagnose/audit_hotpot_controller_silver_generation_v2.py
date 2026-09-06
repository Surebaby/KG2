#!/usr/bin/env python
"""Fail-closed, CPU-only audit of a completed Hotpot silver-generation V2 run.

This is deliberately independent from the generation runner: it never imports
or invokes the provider client, and it does not reconstruct a request by calling
runner helpers.  It validates the frozen protocol, the completed run manifest,
all role/ledger files, and the train-side action pairs as one closed artifact.

The command is non-executing unless ``--execute_audit`` is supplied.  A
successful audit prints a JSON report to stdout; it does not modify or freeze
the generation directory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
import unicodedata


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgproweight.data import hotpot_controller_silver as silver  # noqa: E402
from kgproweight.kg.question_kg import question_sha256  # noqa: E402


AUDIT_SCHEMA_VERSION = "hotpot-controller-silver-generation-v2-independent-audit-1"
PROTOCOL_SCHEMA_VERSION = "hotpot-controller-silver-execution-protocol-2"
PROTOCOL_STATUS = "FROZEN_V2_EXECUTION_PROTOCOL_NO_API_CALLS_NOT_TRAINED"
PROTOCOL_MANIFEST_SCHEMA_VERSION = "hotpot-controller-silver-execution-manifest-2"
GENERATION_SCHEMA_VERSION = "hotpot-controller-silver-generation-run-2"
GENERATION_REPORT_SCHEMA_VERSION = "hotpot-controller-silver-generation-report-2"
GENERATION_MANIFEST_SCHEMA_VERSION = "hotpot-controller-silver-generation-manifest-2"
GENERATION_STATUS = "COMPLETE_V2_GENERATION_DUAL_REVIEW_RETRIEVAL_NOT_RUN_NOT_TRAINED"
WAL_SCHEMA_VERSION = "hotpot-controller-api-wal-event-1"
EXPERIMENT_ID = (
    "QUERY-CONTROLLER-HOTPOT-SILVER-PILOT30-GENERATION-"
    "DUAL-REVIEW-SEED20260904-V2"
)

DEFAULT_PROTOCOL = Path(
    "outputs/audits/query_controller_hotpot_silver_pilot30_"
    "execution_protocol_seed20260904_v2/protocol.json"
)
DEFAULT_GENERATION_DIR = Path(
    "outputs/audits/query_controller_hotpot_silver_pilot30_"
    "generation_dual_review_seed20260904_v2"
)

FIXED_DENOMINATOR = 30
EXPECTED_PRECALL_CONSTRUCTIBLE = 29
EXPECTED_PRECALL_REJECTED = 1
EXPECTED_SEMANTIC_SLOTS = 90
ACCEPTED_MIN = 24
STAGES = ("producer", "reviewer_1_blind", "reviewer_2_gold_aware")
ROLE_FILES = {
    "producer": "producer_proposals.jsonl",
    "reviewer_1_blind": "reviewer_1_reviews.jsonl",
    "reviewer_2_gold_aware": "reviewer_2_reviews.jsonl",
}
OUTPUT_FILES = (
    "producer_proposals.jsonl",
    "reviewer_1_reviews.jsonl",
    "reviewer_2_reviews.jsonl",
    "accepted_actions.jsonl",
    "failures.jsonl",
    "semantic_call_ledger.jsonl",
    "api_transport_attempt_ledger.jsonl",
    "api_call_wal.jsonl",
    "report.json",
    "manifest.json",
)
MANIFEST_BOUND_OUTPUTS = OUTPUT_FILES[:-1]

SEMANTIC_FIELDS = {
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
}
TRANSPORT_FIELDS = {
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
}
ROLE_BASE_FIELDS = {
    "schema_version",
    "dataset",
    "qid",
    "question",
    "question_sha256",
    "stage",
    "status",
    "semantic_request_id",
    "requested_model",
    "response_model",
    "finish_reason",
    "raw_response_content",
    "raw_response_sha256",
    "parsed_response",
    "parsed_response_sha256",
    "nonce_echo_count",
    "reject_code",
    "detail_code",
}
PRODUCER_PROJECTION_FIELDS = {
    "final_item_status",
    "dual_review_unanimous_pass",
    "q1_query",
    "q2_template",
    "proposal_sha256",
    "runtime_projection_gold_or_observation_fields_present",
}
FAILURE_FIELDS = {
    "schema_version",
    "dataset",
    "qid",
    "question_sha256",
    "status",
    "reject_code",
    "detail_code",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_NONCE_RE = re.compile(r"(?<![0-9a-z])n[0-9a-f]{31}(?![0-9a-z])")
_CREDENTIAL_RE = re.compile(
    r"(?i)(?:bearer\s+[a-z0-9._~-]{12,}|\bsk-[a-z0-9_-]{12,})"
)
_ENDPOINT_RE = re.compile(r"https?://api\.deepseek\.com", re.IGNORECASE)


class GenerationV2AuditError(ValueError):
    """Safe, stable fail-closed audit error carrying no row content."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _DuplicateKey(ValueError):
    pass


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise _DuplicateKey(key)
        value[key] = child
    return value


def _fail(code: str) -> None:
    raise GenerationV2AuditError(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        _fail(code)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(project_root: Path, value: Path | str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _load_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(constant)),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, _DuplicateKey):
        _fail(code)
    if not isinstance(value, dict):
        _fail(code)
    return value


def _load_jsonl(path: Path, code: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_pairs,
                    parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(constant)),
                )
                if not isinstance(value, dict):
                    _fail(code)
                rows.append(value)
    except GenerationV2AuditError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, _DuplicateKey):
        _fail(code)
    return rows


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _normalised_nonce_tokens(value: str) -> list[str]:
    expanded = re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda match: chr(int(match.group(1), 16)),
        value,
    )
    return _NONCE_RE.findall(unicodedata.normalize("NFKC", expanded).casefold())


def _stage_spec(protocol: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    if stage == "producer":
        value = protocol.get("producer")
    else:
        reviewers = protocol.get("reviewers")
        value = reviewers.get("reviewer_1" if stage == "reviewer_1_blind" else "reviewer_2") if isinstance(reviewers, Mapping) else None
    if not isinstance(value, Mapping):
        _fail("protocol_stage_spec_invalid")
    return value


def _bound_path(project_root: Path, item: Any, code: str) -> Path:
    if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
        _fail(code)
    path = _resolve(project_root, item["path"])
    _require(path.is_file(), code)
    _require(_is_sha256(item.get("sha256")), code)
    _require(_sha256_file(path) == item["sha256"], code)
    if "size_bytes" in item:
        _require(type(item["size_bytes"]) is int and path.stat().st_size == item["size_bytes"], code)
    return path


def _manifest_entries(value: Any, code: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list):
        _fail(code)
    result: dict[str, Mapping[str, Any]] = {}
    for item in value:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            _fail(code)
        name = str(item["path"])
        if name in result or Path(name).name != name:
            _fail(code)
        result[name] = item
    return result


def _audit_protocol(
    *, project_root: Path, protocol_path: Path, protocol_manifest_path: Path
) -> tuple[dict[str, Any], Path]:
    _require(protocol_path.name == "protocol.json", "protocol_filename_mismatch")
    _require(protocol_manifest_path.name == "manifest.json", "protocol_manifest_filename_mismatch")
    _require(protocol_path.parent.resolve() == protocol_manifest_path.parent.resolve(), "protocol_freeze_directory_mismatch")
    _require({path.name for path in protocol_path.parent.iterdir()} == {"implementation_lock.json", "protocol.json", "report.json", "manifest.json"}, "protocol_freeze_output_set_not_closed")
    protocol = _load_json(protocol_path, "protocol_json_invalid")
    _require(protocol.get("schema_version") == PROTOCOL_SCHEMA_VERSION, "protocol_schema_mismatch")
    _require(protocol.get("experiment_id") == EXPERIMENT_ID, "protocol_experiment_mismatch")
    _require(protocol.get("status") == PROTOCOL_STATUS, "protocol_status_mismatch")
    parent = protocol.get("parent_identity_freeze")
    _require(isinstance(parent, Mapping), "protocol_parent_identity_missing")
    _require(parent.get("rows") == FIXED_DENOMINATOR, "protocol_fixed30_mismatch")
    _require(_is_sha256(parent.get("identity_sha256")), "protocol_identity_hash_invalid")
    diagnostic = protocol.get("v1_to_v2_subject_binding_diagnostic")
    _require(isinstance(diagnostic, Mapping), "protocol_precall_diagnostic_missing")
    _require(diagnostic.get("denominator") == FIXED_DENOMINATOR, "protocol_precall_denominator_mismatch")
    _require(diagnostic.get("precall_constructible") == EXPECTED_PRECALL_CONSTRUCTIBLE, "protocol_precall_29_mismatch")
    _require(diagnostic.get("precall_rejected") == EXPECTED_PRECALL_REJECTED, "protocol_precall_1_mismatch")
    inherited = protocol.get("pilot_decision_gate_inherited")
    _require(isinstance(inherited, Mapping), "protocol_decision_gate_missing")
    _require(inherited.get("fixed_denominator") == FIXED_DENOMINATOR, "protocol_gate_fixed30_mismatch")
    _require(inherited.get("accepted_min") == ACCEPTED_MIN, "protocol_accepted_min_mismatch")
    api = protocol.get("api_execution")
    _require(isinstance(api, Mapping), "protocol_api_contract_missing")
    _require(tuple(api.get("semantic_request_stages_exact") or ()) == STAGES, "protocol_stage_contract_mismatch")
    _require(api.get("crash_safe_write_ahead_required") is True, "protocol_wal_not_required")
    _require(api.get("wal_success_terminal_event_exact") == "semantic_calls_and_in_memory_validation_completed", "protocol_wal_terminal_mismatch")
    future = protocol.get("future_runner_output_contract")
    _require(isinstance(future, Mapping), "protocol_output_contract_missing")
    _require(tuple(future.get("required_files") or ()) == OUTPUT_FILES, "protocol_output_files_mismatch")

    protocol_manifest = _load_json(protocol_manifest_path, "protocol_manifest_invalid")
    _require(protocol_manifest.get("schema_version") == PROTOCOL_MANIFEST_SCHEMA_VERSION, "protocol_manifest_schema_mismatch")
    _require(protocol_manifest.get("experiment_id") == EXPERIMENT_ID, "protocol_manifest_experiment_mismatch")
    _require(protocol_manifest.get("status") == PROTOCOL_STATUS, "protocol_manifest_status_mismatch")
    _require(protocol_manifest.get("api_calls") == 0 and protocol_manifest.get("training_started") is False, "protocol_manifest_boundary_mismatch")
    _require(tuple(protocol_manifest.get("protocol_freeze_output_set_exact") or ()) == ("implementation_lock.json", "protocol.json", "report.json", "manifest.json"), "protocol_manifest_declared_output_set_mismatch")
    entries = _manifest_entries(protocol_manifest.get("outputs"), "protocol_manifest_outputs_invalid")
    _require(set(entries) == {"implementation_lock.json", "protocol.json", "report.json"}, "protocol_manifest_outputs_not_closed")
    for name, item in entries.items():
        path = protocol_path.parent / name
        _require(path.is_file(), "protocol_manifest_output_missing")
        _require(_sha256_file(path) == item.get("sha256"), "protocol_manifest_output_hash_mismatch")
        _require(path.stat().st_size == item.get("size_bytes"), "protocol_manifest_output_size_mismatch")
    _require(_sha256_file(protocol_path) == entries["protocol.json"]["sha256"], "protocol_hash_not_closed")
    protocol_report = _load_json(protocol_path.parent / "report.json", "protocol_report_invalid")
    _require(
        protocol_report.get("schema_version") == "hotpot-controller-silver-execution-freeze-report-2"
        and protocol_report.get("experiment_id") == EXPERIMENT_ID
        and protocol_report.get("status") == PROTOCOL_STATUS
        and protocol_report.get("api_calls") == 0,
        "protocol_report_identity_or_boundary_mismatch",
    )

    lock_binding = protocol.get("implementation_lock")
    lock_path = _bound_path(project_root, lock_binding, "implementation_lock_binding_invalid")
    lock = _load_json(lock_path, "implementation_lock_json_invalid")
    _require(lock.get("schema_version") == "hotpot-controller-silver-implementation-lock-1", "implementation_lock_schema_mismatch")
    _require(lock.get("experiment_id") == EXPERIMENT_ID, "implementation_lock_experiment_mismatch")
    implementations = lock.get("implementations")
    _require(isinstance(implementations, list) and implementations, "implementation_lock_empty")
    for item in implementations:
        _bound_path(project_root, item, "implementation_source_hash_mismatch")
    return protocol, lock_path


def _read_identity_and_chains(
    *, identity_path: Path, raw_path: Path
) -> tuple[list[dict[str, str]], dict[str, silver.HotpotSupportChain], set[str]]:
    identities = _load_jsonl(identity_path, "identity_jsonl_invalid")
    _require(len(identities) == FIXED_DENOMINATOR, "identity_fixed30_mismatch")
    expected_fields = {"dataset", "qid", "question"}
    _require(all(set(row) == expected_fields for row in identities), "identity_schema_mismatch")
    _require(all(row.get("dataset") == silver.DATASET for row in identities), "identity_dataset_mismatch")
    qids = [row.get("qid") for row in identities]
    _require(all(isinstance(qid, str) and qid for qid in qids), "identity_qid_invalid")
    _require(len(set(qids)) == FIXED_DENOMINATOR, "identity_qid_not_unique")
    _require(all(isinstance(row.get("question"), str) and row["question"] for row in identities), "identity_question_invalid")

    wanted = set(qids)
    raw_by_qid: dict[str, dict[str, Any]] = {}
    try:
        with raw_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
                if not isinstance(row, dict):
                    _fail("raw_jsonl_invalid")
                qid = str(row.get("id") or row.get("qid") or "").strip()
                if qid in wanted:
                    _require(qid not in raw_by_qid, "raw_qid_duplicate")
                    raw_by_qid[qid] = row
    except GenerationV2AuditError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, _DuplicateKey):
        _fail("raw_jsonl_invalid")
    _require(set(raw_by_qid) == wanted, "identity_raw_qid_join_incomplete")

    chains: dict[str, silver.HotpotSupportChain] = {}
    rejected: set[str] = set()
    for identity in identities:
        qid = identity["qid"]
        raw = raw_by_qid[qid]
        _require(question_sha256(identity["question"]) == question_sha256(str(raw.get("question") or "")), "identity_raw_question_hash_mismatch")
        try:
            chain = silver.extract_hotpot_support_chain(raw)
        except silver.HotpotSilverReject:
            rejected.add(qid)
            continue
        _require(chain.qid == qid and chain.question == identity["question"], "identity_chain_join_mismatch")
        chains[qid] = chain
    _require(len(chains) == EXPECTED_PRECALL_CONSTRUCTIBLE, "recomputed_precall_29_mismatch")
    _require(len(rejected) == EXPECTED_PRECALL_REJECTED, "recomputed_precall_1_mismatch")
    return identities, chains, rejected


def _review_object_valid(value: Any, protocol: Mapping[str, Any]) -> bool:
    if not isinstance(value, Mapping):
        return False
    schema = (((protocol.get("schemas") or {}).get("review_output_shared") or {}).get("schema") or {})
    required = schema.get("required")
    if not isinstance(required, list) or set(value) != set(required):
        return False
    booleans = [field for field in required if field not in {"schema_version", "verdict", "reject_codes"}]
    if value.get("schema_version") != "hotpot-controller-query-review-v1":
        return False
    if any(type(value.get(field)) is not bool for field in booleans):
        return False
    verdict, codes = value.get("verdict"), value.get("reject_codes")
    if verdict not in {"pass", "reject"} or not isinstance(codes, list) or len(codes) != len(set(codes)):
        return False
    allowed = set((((schema.get("properties") or {}).get("reject_codes") or {}).get("items") or {}).get("enum") or [])
    if any(not isinstance(code, str) or code not in allowed for code in codes):
        return False
    all_true = all(value[field] is True for field in booleans)
    return (verdict == "pass") == (all_true and not codes) and (verdict != "reject" or bool(codes))


def _audit_sensitive_outputs(
    *, role_rows: Mapping[str, Sequence[Mapping[str, Any]]], chains: Mapping[str, silver.HotpotSupportChain]
) -> None:
    for rows in role_rows.values():
        for row in rows:
            content = row.get("raw_response_content")
            parsed = row.get("parsed_response")
            serialised = json.dumps({"content": content, "parsed": parsed}, ensure_ascii=False)
            _require(not _normalised_nonce_tokens(serialised), "nonce_token_serialized_in_role_output")
            _require(_CREDENTIAL_RE.search(serialised) is None, "credential_value_serialized_in_role_output")
            _require(_ENDPOINT_RE.search(serialised) is None, "endpoint_value_serialized_in_role_output")
            chain = chains.get(str(row.get("qid") or ""))
            if chain is not None:
                secrets = (chain.bridge_title, chain.intermediate, *chain.final_answers)
                _require(not any(silver._contains_secret(serialised, secret) for secret in secrets), "chain_secret_serialized_in_role_response")


def _audit_generation(
    *,
    project_root: Path,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    implementation_lock_path: Path,
    generation_dir: Path,
) -> dict[str, Any]:
    _require(generation_dir.is_dir(), "generation_directory_missing")
    _require({path.name for path in generation_dir.iterdir()} == set(OUTPUT_FILES), "generation_output_set_not_closed")
    manifest = _load_json(generation_dir / "manifest.json", "generation_manifest_invalid")
    report = _load_json(generation_dir / "report.json", "generation_report_invalid")
    _require(manifest.get("schema_version") == GENERATION_MANIFEST_SCHEMA_VERSION, "generation_manifest_schema_mismatch")
    _require(report.get("schema_version") == GENERATION_REPORT_SCHEMA_VERSION, "generation_report_schema_mismatch")
    for value, code in ((manifest, "generation_manifest_identity_mismatch"), (report, "generation_report_identity_mismatch")):
        _require(value.get("experiment_id") == EXPERIMENT_ID and value.get("status") == GENERATION_STATUS, code)

    output_entries = _manifest_entries(manifest.get("outputs"), "generation_manifest_outputs_invalid")
    _require(tuple(output_entries) == MANIFEST_BOUND_OUTPUTS, "generation_manifest_output_order_or_set_mismatch")
    for name, item in output_entries.items():
        path = generation_dir / name
        _require(path.is_file(), "generation_manifest_output_missing")
        _require(_sha256_file(path) == item.get("sha256"), "generation_output_hash_mismatch")
        _require(path.stat().st_size == item.get("size_bytes"), "generation_output_size_mismatch")

    inputs = manifest.get("inputs")
    _require(isinstance(inputs, Mapping), "generation_manifest_inputs_missing")
    _require(set(inputs) == {"execution_protocol", "implementation_lock", "protocol_freeze_report", "protocol_freeze_manifest", "v1_supersession_addendum", "parent_identity", "parent_metadata_addendum_v1_1", "raw_train"}, "generation_manifest_inputs_schema_mismatch")
    bound_protocol = _bound_path(project_root, inputs["execution_protocol"], "generation_protocol_input_hash_mismatch")
    bound_lock = _bound_path(project_root, inputs["implementation_lock"], "generation_lock_input_hash_mismatch")
    bound_protocol_report = _bound_path(project_root, inputs["protocol_freeze_report"], "generation_protocol_report_input_hash_mismatch")
    bound_protocol_manifest = _bound_path(project_root, inputs["protocol_freeze_manifest"], "generation_protocol_manifest_input_hash_mismatch")
    bound_supersession = _bound_path(project_root, inputs["v1_supersession_addendum"], "generation_supersession_input_hash_mismatch")
    identity_path = _bound_path(project_root, inputs["parent_identity"], "generation_identity_input_hash_mismatch")
    _bound_path(project_root, inputs["parent_metadata_addendum_v1_1"], "generation_addendum_input_hash_mismatch")
    raw_path = _bound_path(project_root, inputs["raw_train"], "generation_raw_input_hash_mismatch")
    _require(bound_protocol.resolve() == protocol_path.resolve(), "generation_protocol_path_mismatch")
    _require(bound_lock.resolve() == implementation_lock_path.resolve(), "generation_lock_path_mismatch")
    _require(bound_protocol_report.resolve() == (protocol_path.parent / "report.json").resolve(), "generation_protocol_report_path_mismatch")
    _require(bound_protocol_manifest.resolve() == (protocol_path.parent / "manifest.json").resolve(), "generation_protocol_manifest_path_mismatch")
    supersession = _load_json(bound_supersession, "generation_supersession_json_invalid")
    superseded_by = supersession.get("superseded_by")
    _require(
        supersession.get("schema_version") == "hotpot-controller-silver-v1-supersession-addendum-1"
        and supersession.get("v1_experiment_id")
        == "QUERY-CONTROLLER-HOTPOT-SILVER-PILOT30-GENERATION-DUAL-REVIEW-SEED20260904-V1"
        and supersession.get("v1_api_calls") == 0
        and supersession.get("v1_generation_output_existed_at_supersession") is False
        and supersession.get("status") == "SUPERSEDED_BEFORE_ANY_API_CALL"
        and isinstance(superseded_by, Mapping)
        and superseded_by.get("experiment_id") == EXPERIMENT_ID
        and _resolve(project_root, str(superseded_by.get("protocol_path") or "")).resolve()
        == protocol_path.resolve()
        and _resolve(project_root, str(superseded_by.get("manifest_path") or "")).resolve()
        == bound_protocol_manifest.resolve()
        and superseded_by.get("protocol_sha256") == _sha256_file(protocol_path)
        and superseded_by.get("manifest_sha256") == _sha256_file(bound_protocol_manifest),
        "generation_supersession_binding_mismatch",
    )
    protocol_manifest = _load_json(bound_protocol_manifest, "generation_bound_protocol_manifest_invalid")
    external = protocol_manifest.get("external_append_only_artifact")
    _require(
        isinstance(external, Mapping)
        and _resolve(project_root, str(external.get("path") or "")).resolve()
        == bound_supersession.resolve()
        and external.get("commit_order")
        == "after_complete_v2_manifest_directory_commit"
        and external.get("status_at_v2_manifest_commit")
        == "PENDING_APPEND_ONLY_COMMIT"
        and external.get("must_bind_v2_manifest_sha256") is True,
        "generation_protocol_manifest_supersession_declaration_mismatch",
    )
    _require(inputs["parent_identity"].get("sha256") == (protocol.get("parent_identity_freeze") or {}).get("identity_sha256"), "generation_identity_protocol_hash_mismatch")
    identities, chains, precall_rejected = _read_identity_and_chains(identity_path=identity_path, raw_path=raw_path)
    qids = [row["qid"] for row in identities]
    identity_by_qid = {row["qid"]: row for row in identities}

    role_rows = {
        stage: _load_jsonl(generation_dir / filename, f"{stage}_role_jsonl_invalid")
        for stage, filename in ROLE_FILES.items()
    }
    semantic_rows = _load_jsonl(generation_dir / "semantic_call_ledger.jsonl", "semantic_ledger_invalid")
    transport_rows = _load_jsonl(generation_dir / "api_transport_attempt_ledger.jsonl", "transport_ledger_invalid")
    wal_rows = _load_jsonl(generation_dir / "api_call_wal.jsonl", "wal_invalid")
    accepted_actions = _load_jsonl(generation_dir / "accepted_actions.jsonl", "accepted_actions_invalid")
    failures = _load_jsonl(generation_dir / "failures.jsonl", "failures_invalid")

    # Fixed identity/order and role-specific serialization.
    role_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for stage, rows in role_rows.items():
        _require(len(rows) == FIXED_DENOMINATOR, "role_fixed30_mismatch")
        _require([row.get("qid") for row in rows] == qids, "role_qid_order_mismatch")
        expected_fields = ROLE_BASE_FIELDS | (PRODUCER_PROJECTION_FIELDS if stage == "producer" else set())
        spec = _stage_spec(protocol, stage)
        model = spec.get("model")
        for identity, row in zip(identities, rows):
            _require(set(row) == expected_fields, "role_schema_or_isolation_mismatch")
            _require(row.get("schema_version") == GENERATION_SCHEMA_VERSION, "role_schema_version_mismatch")
            _require(row.get("dataset") == identity["dataset"] and row.get("qid") == identity["qid"] and row.get("question") == identity["question"], "role_identity_mismatch")
            _require(row.get("question_sha256") == question_sha256(identity["question"]), "role_question_hash_mismatch")
            _require(row.get("stage") == stage and row.get("requested_model") == model, "role_stage_or_model_mismatch")
            expected_request_id = _sha256_text(f"{EXPERIMENT_ID}\0{identity['qid']}\0{stage}")
            _require(row.get("semantic_request_id") == expected_request_id, "role_semantic_request_id_mismatch")
            _require((identity["qid"], stage) not in role_by_key, "role_key_duplicate")
            role_by_key[(identity["qid"], stage)] = row
            parsed = row.get("parsed_response")
            if parsed is not None:
                _require(row.get("parsed_response_sha256") == _canonical_sha256(parsed), "role_parsed_response_hash_mismatch")
            else:
                # A rejected producer may retain only the hash of a parsed but
                # mechanically invalid object; the unsafe object itself is not
                # released in the role file.
                _require(
                    row.get("parsed_response_sha256") is None
                    or (row.get("status") == "rejected" and _is_sha256(row.get("parsed_response_sha256"))),
                    "role_null_parsed_hash_mismatch",
                )
            if row.get("status") == "accepted":
                _require(row.get("response_model") == model and row.get("finish_reason") == "stop", "accepted_role_response_contract_mismatch")
                _require(isinstance(row.get("raw_response_content"), str) and _is_sha256(row.get("raw_response_sha256")), "accepted_role_response_capture_missing")
                _require(row.get("nonce_echo_count") == 0 and parsed is not None, "accepted_role_validation_state_mismatch")
                if stage == "producer":
                    try:
                        validated = silver.validate_query_proposal(parsed, chains[identity["qid"]])
                    except (KeyError, silver.HotpotSilverReject):
                        _fail("accepted_producer_proposal_invalid")
                    _require(validated.proposal_sha256 == row.get("parsed_response_sha256"), "accepted_producer_proposal_hash_mismatch")
                else:
                    _require(_review_object_valid(parsed, protocol), "accepted_review_object_invalid")

    _require(len(semantic_rows) == EXPECTED_SEMANTIC_SLOTS, "semantic_slot_count_not_90")
    expected_semantic_order = [(qid, stage) for qid in qids for stage in STAGES]
    _require([(row.get("qid"), row.get("stage")) for row in semantic_rows] == expected_semantic_order, "semantic_slot_order_mismatch")
    semantic_by_id: dict[str, Mapping[str, Any]] = {}
    for row in semantic_rows:
        _require(set(row) == SEMANTIC_FIELDS, "semantic_ledger_schema_mismatch")
        qid, stage = str(row.get("qid") or ""), str(row.get("stage") or "")
        identity = identity_by_qid.get(qid)
        _require(identity is not None and stage in STAGES, "semantic_identity_or_stage_mismatch")
        _require(row.get("experiment_id") == EXPERIMENT_ID and row.get("dataset") == silver.DATASET, "semantic_experiment_or_dataset_mismatch")
        _require(row.get("question_sha256") == question_sha256(identity["question"]), "semantic_question_hash_mismatch")
        expected_request_id = _sha256_text(f"{EXPERIMENT_ID}\0{qid}\0{stage}")
        _require(row.get("semantic_request_id") == expected_request_id, "semantic_request_id_mismatch")
        _require(expected_request_id not in semantic_by_id, "semantic_request_id_duplicate")
        semantic_by_id[expected_request_id] = row
        spec = _stage_spec(protocol, stage)
        _require(row.get("requested_model") == spec.get("model") == row.get("expected_response_model"), "semantic_model_binding_mismatch")
        _require(row.get("prompt_template_sha256") == (spec.get("prompt") or {}).get("sha256"), "semantic_prompt_hash_mismatch")
        _require(_is_sha256(row.get("model_visible_messages_sha256")) and _is_sha256(row.get("safe_payload_sha256")), "semantic_payload_hash_invalid")
        role = role_by_key[(qid, stage)]
        for semantic_key, role_key in (
            ("status", "status"), ("response_model", "response_model"),
            ("finish_reason", "finish_reason"), ("raw_response_sha256", "raw_response_sha256"),
            ("parsed_response_sha256", "parsed_response_sha256"),
            ("nonce_echo_count", "nonce_echo_count"), ("reject_code", "reject_code"),
            ("detail_code", "detail_code"),
        ):
            _require(row.get(semantic_key) == role.get(role_key), "semantic_role_projection_mismatch")

    # The 90 slots include explicitly skipped rows.  No skipped row can hide a
    # physical attempt or a captured response.
    skipped_count = 0
    for row in semantic_rows:
        skipped = row.get("status") == "not_executed_upstream_failure"
        if skipped:
            skipped_count += 1
            _require(row.get("semantic_attempt_count") == 0 and row.get("transport_attempt_count") == 0, "skipped_slot_has_attempt")
            _require(all(row.get(key) is None for key in ("finish_reason", "response_model", "raw_response_sha256", "parsed_response_sha256", "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens")), "skipped_slot_has_response_capture")
            _require(role_by_key[(row["qid"], row["stage"])].get("raw_response_content") is None, "skipped_role_has_response_content")
        else:
            _require(row.get("semantic_attempt_count") == 1, "executed_slot_semantic_attempt_not_one")
            _require(type(row.get("transport_attempt_count")) is int and row["transport_attempt_count"] >= 1, "executed_slot_transport_attempt_missing")
    _require(skipped_count >= 3, "expected_precall_skipped_slots_missing")
    for qid in qids:
        stage_statuses = {
            stage: semantic_by_id[
                _sha256_text(f"{EXPERIMENT_ID}\0{qid}\0{stage}")
            ]["status"]
            for stage in STAGES
        }
        if qid in precall_rejected:
            _require(
                all(value == "not_executed_upstream_failure" for value in stage_statuses.values()),
                "precall_rejected_stage_execution_detected",
            )
        else:
            _require(
                stage_statuses["producer"] != "not_executed_upstream_failure",
                "constructible_producer_was_skipped",
            )
            reviewer_statuses = [stage_statuses[stage] for stage in STAGES[1:]]
            if stage_statuses["producer"] == "accepted":
                _require(
                    all(value != "not_executed_upstream_failure" for value in reviewer_statuses),
                    "accepted_producer_reviewer_was_skipped",
                )
            else:
                _require(
                    all(value == "not_executed_upstream_failure" for value in reviewer_statuses),
                    "rejected_producer_reviewer_execution_detected",
                )

    # Transport ledger and semantic-response capture conservation.
    max_attempts = int((protocol.get("api_execution") or {}).get("max_physical_attempts_per_semantic_request", 0))
    _require(max_attempts >= 1, "protocol_max_attempts_invalid")
    transports_by_semantic: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_attempts: set[tuple[str, int]] = set()
    for row in transport_rows:
        _require(set(row) == TRANSPORT_FIELDS, "transport_ledger_schema_mismatch")
        request_id = row.get("semantic_request_id")
        _require(isinstance(request_id, str) and request_id in semantic_by_id, "transport_semantic_join_mismatch")
        semantic = semantic_by_id[request_id]
        _require(row.get("experiment_id") == EXPERIMENT_ID and row.get("stage") == semantic.get("stage"), "transport_stage_or_experiment_mismatch")
        attempt = row.get("physical_attempt_index")
        _require(type(attempt) is int and 1 <= attempt <= max_attempts, "transport_attempt_index_invalid")
        _require((request_id, attempt) not in seen_attempts, "transport_attempt_duplicate")
        seen_attempts.add((request_id, attempt))
        _require(_is_sha256(row.get("request_body_sha256")), "transport_request_hash_invalid")
        _require(row.get("request_bytes_identical_to_attempt_1") is True, "transport_retry_body_drift")
        _require(type(row.get("response_received")) is bool and type(row.get("transport_retryable")) is bool, "transport_boolean_invalid")
        transports_by_semantic[request_id].append(row)
    for request_id, semantic in semantic_by_id.items():
        rows = sorted(transports_by_semantic.get(request_id, []), key=lambda row: row["physical_attempt_index"])
        _require(len(rows) == semantic.get("transport_attempt_count"), "semantic_transport_count_mismatch")
        if not rows:
            _require(semantic.get("status") == "not_executed_upstream_failure", "zero_transport_non_skipped_slot")
            continue
        _require([row["physical_attempt_index"] for row in rows] == list(range(1, len(rows) + 1)), "transport_attempt_sequence_gap")
        _require(len({row["request_body_sha256"] for row in rows}) == 1, "transport_retry_request_hash_drift")
        received_positions = [index for index, row in enumerate(rows) if row["response_received"]]
        _require(len(received_positions) <= 1, "multiple_responses_for_semantic_slot")
        if received_positions:
            _require(received_positions == [len(rows) - 1], "response_not_terminal_attempt")
            _require(semantic.get("status") != "transport_exhausted", "received_response_marked_exhausted")
        else:
            _require(semantic.get("status") == "transport_exhausted", "unreceived_response_not_exhausted")
            _require(len(rows) == max_attempts or rows[-1].get("transport_retryable") is False, "transport_stopped_before_terminal_condition")
        for row in rows[:-1]:
            _require(row.get("response_received") is False and row.get("transport_retryable") is True, "transport_retry_predecessor_invalid")

    # WAL is a second, append-before/append-after account of every physical
    # attempt.  A provider return additionally requires a response_captured
    # event before the result event; no response body is stored in the WAL.
    _require(bool(wal_rows), "wal_empty")
    _require(all(row.get("schema_version") == WAL_SCHEMA_VERSION and row.get("experiment_id") == EXPERIMENT_ID for row in wal_rows), "wal_schema_or_experiment_mismatch")
    events = [row.get("event") for row in wal_rows]
    _require(events[0] == "run_started", "wal_run_started_missing_or_not_first")
    _require(events[-1] == "semantic_calls_and_in_memory_validation_completed", "wal_success_terminal_missing_or_not_last")
    _require(events.count("run_started") == 1 and events.count("semantic_calls_and_in_memory_validation_completed") == 1, "wal_run_boundary_not_unique")
    _require("run_aborted" not in events, "wal_abort_present")
    _require(set(events) <= {"run_started", "intent", "response_captured", "result", "semantic_calls_and_in_memory_validation_completed"}, "wal_unknown_event")
    started = wal_rows[0]
    _require(set(started) == {"schema_version", "experiment_id", "event", "at_utc", "fixed_denominator", "precall_constructible", "precall_rejected"}, "wal_run_started_schema_mismatch")
    _require(started.get("fixed_denominator") == FIXED_DENOMINATOR and started.get("precall_constructible") == EXPECTED_PRECALL_CONSTRUCTIBLE and started.get("precall_rejected") == EXPECTED_PRECALL_REJECTED, "wal_fixed30_or_29_1_mismatch")
    intent_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    capture_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    result_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    seen_intent_before_result: set[tuple[str, int]] = set()
    for row in wal_rows[1:-1]:
        event = row.get("event")
        key = (str(row.get("semantic_request_id") or ""), row.get("physical_attempt_index"))
        _require(key in seen_attempts, "wal_attempt_not_in_transport")
        if event == "intent":
            _require(set(row) == {"schema_version", "experiment_id", "event", "semantic_request_id", "stage", "physical_attempt_index", "request_body_sha256", "at_utc"}, "wal_intent_schema_mismatch")
            _require(key not in intent_by_key and key not in result_by_key, "wal_intent_duplicate_or_late")
            intent_by_key[key] = row
            seen_intent_before_result.add(key)
        elif event == "response_captured":
            capture_fields = {
                "schema_version", "experiment_id", "event", "semantic_request_id",
                "stage", "physical_attempt_index", "request_body_sha256", "at_utc",
                "capture_complete", "response_text_fields_sha256", "response_model",
                "response_model_sha256", "response_model_matches_requested",
                "choices_count", "finish_reasons", "finish_reasons_sha256",
            }
            _require(set(row) == capture_fields, "wal_response_capture_schema_mismatch")
            _require(key in seen_intent_before_result and key not in capture_by_key and key not in result_by_key, "wal_response_capture_order_or_duplicate")
            _require(row.get("capture_complete") is True, "wal_response_capture_incomplete")
            _require(_is_sha256(row.get("response_text_fields_sha256")) and _is_sha256(row.get("response_model_sha256")) and _is_sha256(row.get("finish_reasons_sha256")), "wal_response_capture_hash_invalid")
            _require(type(row.get("response_model_matches_requested")) is bool and type(row.get("choices_count")) is int and isinstance(row.get("finish_reasons"), list), "wal_response_capture_projection_invalid")
            capture_by_key[key] = row
        elif event == "result":
            _require(set(row) == {"schema_version", "experiment_id", "event", "semantic_request_id", "stage", "physical_attempt_index", "request_body_sha256", "at_utc", "response_received", "http_status", "transport_error_class", "transport_retryable"}, "wal_result_schema_mismatch")
            _require(key in seen_intent_before_result and key not in result_by_key, "wal_result_without_unique_prior_intent")
            result_by_key[key] = row
        else:
            _fail("wal_boundary_event_in_attempt_region")
    _require(set(intent_by_key) == seen_attempts == set(result_by_key), "wal_intent_result_transport_conservation_failed")
    transport_by_key = {(row["semantic_request_id"], row["physical_attempt_index"]): row for row in transport_rows}
    for key in seen_attempts:
        intent, result, transport = intent_by_key[key], result_by_key[key], transport_by_key[key]
        _require(intent.get("stage") == result.get("stage") == transport.get("stage"), "wal_stage_mismatch")
        _require(intent.get("request_body_sha256") == result.get("request_body_sha256") == transport.get("request_body_sha256"), "wal_request_hash_mismatch")
        _require(intent.get("at_utc") == transport.get("started_at_utc") and result.get("at_utc") == transport.get("ended_at_utc"), "wal_transport_time_binding_mismatch")
        for key_name in ("response_received", "http_status", "transport_error_class", "transport_retryable"):
            _require(result.get(key_name) == transport.get(key_name), "wal_result_transport_projection_mismatch")
        if transport.get("response_received") is True:
            _require(key in capture_by_key, "wal_received_response_capture_missing")
            capture = capture_by_key[key]
            _require(capture.get("stage") == transport.get("stage") and capture.get("request_body_sha256") == transport.get("request_body_sha256"), "wal_response_capture_binding_mismatch")
        else:
            _require(key not in capture_by_key, "wal_capture_without_received_response")
    _require(set(capture_by_key) == {key for key, row in transport_by_key.items() if row.get("response_received") is True}, "wal_response_capture_conservation_failed")
    terminal = wal_rows[-1]
    _require(set(terminal) == {"schema_version", "experiment_id", "event", "at_utc", "unmatched_intent_count", "physical_attempt_count", "accepted_identity_count"}, "wal_terminal_schema_mismatch")
    _require(terminal.get("unmatched_intent_count") == 0, "wal_terminal_unmatched_intent")
    _require(terminal.get("physical_attempt_count") == len(transport_rows), "wal_terminal_attempt_count_mismatch")

    # Report identities, statuses, and accepted/failure conservation.
    item_statuses = report.get("item_statuses")
    _require(isinstance(item_statuses, list) and len(item_statuses) == FIXED_DENOMINATOR, "report_item_status_fixed30_mismatch")
    _require([row.get("qid") for row in item_statuses] == qids, "report_item_status_order_mismatch")
    status_by_qid: dict[str, Mapping[str, Any]] = {}
    for identity, row in zip(identities, item_statuses):
        _require(set(row) == {"dataset", "qid", "question_sha256", "status", "reject_code"}, "report_item_status_schema_mismatch")
        _require(row.get("dataset") == silver.DATASET and row.get("qid") == identity["qid"] and row.get("question_sha256") == question_sha256(identity["question"]), "report_item_status_identity_mismatch")
        status_by_qid[identity["qid"]] = row
    accepted_qids = [qid for qid in qids if status_by_qid[qid]["status"] == "accepted_generation_and_dual_review"]
    failed_qids = [qid for qid in qids if qid not in set(accepted_qids)]
    _require(set(qid for qid in qids if status_by_qid[qid]["status"] == "precall_rejected") == precall_rejected, "precall_rejected_identity_mismatch")
    _require(len(accepted_qids) >= ACCEPTED_MIN, "accepted_below_24")
    _require(report.get("fixed_denominator") == FIXED_DENOMINATOR and report.get("precall_rejected") == EXPECTED_PRECALL_REJECTED, "report_fixed30_or_29_1_mismatch")
    _require(report.get("producer_accepted") == sum(role_by_key[(qid, "producer")]["status"] == "accepted" for qid in qids), "report_producer_count_mismatch")
    _require(report.get("dual_review_accepted") == len(accepted_qids), "report_accepted_count_mismatch")
    _require(report.get("failed_items") == len(failed_qids), "report_failure_count_mismatch")
    _require(report.get("accepted_action_rows") == 2 * len(accepted_qids), "report_action_count_mismatch")
    _require(report.get("semantic_call_rows") == EXPECTED_SEMANTIC_SLOTS, "report_semantic_count_mismatch")
    _require(report.get("transport_attempt_rows") == len(transport_rows), "report_transport_count_mismatch")
    _require(report.get("wal_event_rows") == len(wal_rows), "report_wal_count_mismatch")
    gates = report.get("gates")
    _require(isinstance(gates, Mapping) and gates.get("accepted_min") == ACCEPTED_MIN and gates.get("accepted_min_pass") is True and gates.get("all_model_response_text_nonce_like_free") is True and gates.get("wal_intent_result_balanced") is True and gates.get("wal_response_capture_balanced") is True and gates.get("generation_and_dual_review_gate_pass") is True, "report_generation_gate_not_passed")
    scientific = report.get("scientific_boundary")
    _require(isinstance(scientific, Mapping) and scientific.get("endpoint_allowlist_pass") is True and scientific.get("retrieval_or_reader_calls") == 0 and scientific.get("training_started") is False, "report_scientific_boundary_mismatch")
    _require(terminal.get("accepted_identity_count") == len(accepted_qids), "wal_terminal_accepted_count_mismatch")

    # Producer's answer-free runtime projection is released only for final
    # accepted rows; all other qids must expose null q1/q2/proposal fields.
    for qid in qids:
        producer = role_by_key[(qid, "producer")]
        final_accepted = qid in set(accepted_qids)
        _require(producer.get("final_item_status") == status_by_qid[qid]["status"], "producer_final_status_mismatch")
        _require(producer.get("dual_review_unanimous_pass") is final_accepted, "producer_dual_review_flag_mismatch")
        _require(producer.get("runtime_projection_gold_or_observation_fields_present") is False, "producer_projection_gold_or_observation_flag")
        if final_accepted:
            parsed = producer.get("parsed_response")
            _require(isinstance(parsed, Mapping), "accepted_producer_projection_missing")
            _require(producer.get("q1_query") == parsed.get("q1") and producer.get("q2_template") == parsed.get("q2_template"), "accepted_producer_projection_content_mismatch")
            _require(producer.get("proposal_sha256") == producer.get("parsed_response_sha256"), "accepted_producer_projection_hash_mismatch")
            _require(all(role_by_key[(qid, stage)].get("status") == "accepted" for stage in STAGES), "accepted_item_stage_not_all_accepted")
        else:
            _require(all(producer.get(field) is None for field in ("q1_query", "q2_template", "proposal_sha256")), "rejected_producer_projection_not_null")

    # Exactly two, ordered and fully revalidated action rows per accepted qid.
    _require(len(accepted_actions) == 2 * len(accepted_qids), "accepted_action_cardinality_mismatch")
    _require([str(row.get("qid") or "") for row in accepted_actions] == [qid for qid in accepted_qids for _ in range(2)], "accepted_action_qid_order_mismatch")
    for index, qid in enumerate(accepted_qids):
        pair = accepted_actions[2 * index : 2 * index + 2]
        try:
            validated_pair = silver.validate_hotpot_action_pair(pair, chain=chains[qid], expected_split="train")
        except (KeyError, silver.HotpotSilverReject):
            _fail("accepted_action_pair_revalidation_failed")
        _require([row.get("slot") for row in validated_pair] == ["q1", "q2_dynamic"], "accepted_action_pair_slot_order_mismatch")
        producer, r1, r2 = (role_by_key[(qid, stage)] for stage in STAGES)
        for action in validated_pair:
            source = action.get("source_provenance")
            external = source.get("external_provenance") if isinstance(source, Mapping) else None
            _require(isinstance(external, Mapping), "accepted_action_external_provenance_missing")
            _require(source.get("proposal_sha256") == producer.get("proposal_sha256"), "accepted_action_proposal_hash_mismatch")
            _require(external.get("generation_experiment_id") == EXPERIMENT_ID and external.get("dual_review_unanimous_pass") is True, "accepted_action_generation_binding_mismatch")
            _require(external.get("producer_response_sha256") == producer.get("raw_response_sha256") and external.get("blind_review_response_sha256") == r1.get("raw_response_sha256") and external.get("gold_aware_review_response_sha256") == r2.get("raw_response_sha256"), "accepted_action_response_hash_binding_mismatch")

    _require(len(failures) == len(failed_qids), "failure_row_count_mismatch")
    _require([row.get("qid") for row in failures] == failed_qids, "failure_qid_order_mismatch")
    for row in failures:
        _require(set(row) == FAILURE_FIELDS, "failure_schema_mismatch")
        qid = row.get("qid")
        status = status_by_qid.get(str(qid))
        _require(status is not None and row.get("dataset") == silver.DATASET and row.get("question_sha256") == status.get("question_sha256") and row.get("status") == status.get("status") and row.get("reject_code") == status.get("reject_code"), "failure_status_projection_mismatch")
    expected_reject_counts = dict(sorted(Counter(row["reject_code"] for row in failures).items()))
    _require(report.get("reject_code_counts") == expected_reject_counts, "report_reject_counts_mismatch")

    _audit_sensitive_outputs(role_rows=role_rows, chains=chains)

    _require(manifest.get("fixed_denominator") == FIXED_DENOMINATOR, "manifest_fixed30_mismatch")
    _require(manifest.get("api_physical_attempts") == len(transport_rows), "manifest_transport_count_mismatch")
    _require(manifest.get("implementation_lock_verified") is True and manifest.get("protocol_freeze_manifest_verified") is True, "manifest_protocol_or_implementation_verification_missing")
    _require(manifest.get("wal_balanced") is True and manifest.get("wal_response_capture_balanced") is True and manifest.get("credential_or_endpoint_values_serialized") is False and manifest.get("endpoint_allowlist_pass") is True, "manifest_security_gate_mismatch")
    _require(manifest.get("retrieval_calls") == 0 and manifest.get("training_started") is False, "manifest_scientific_boundary_mismatch")

    # Search only generation outputs, not the protocol allowlist itself.  Exact
    # role/ledger schemas above prevent a credential field from being smuggled
    # in, while these patterns catch accidental literal values.
    for name in OUTPUT_FILES:
        text = (generation_dir / name).read_text(encoding="utf-8")
        _require(not _normalised_nonce_tokens(text), "nonce_literal_in_generation_artifact")
        _require(_CREDENTIAL_RE.search(text) is None, "credential_literal_in_generation_artifact")
        _require(_ENDPOINT_RE.search(text) is None, "endpoint_literal_in_generation_artifact")

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_GENERATION_V2_ARTIFACT_AUDIT_RETRIEVAL_NOT_RUN_NOT_TRAINED",
        "fixed_denominator": FIXED_DENOMINATOR,
        "precall_constructible": EXPECTED_PRECALL_CONSTRUCTIBLE,
        "precall_rejected": EXPECTED_PRECALL_REJECTED,
        "semantic_slots_including_skipped": len(semantic_rows),
        "skipped_semantic_slots": skipped_count,
        "physical_attempts": len(transport_rows),
        "wal_intents": len(intent_by_key),
        "wal_results": len(result_by_key),
        "wal_response_captures": len(capture_by_key),
        "response_received_attempts": sum(bool(row["response_received"]) for row in transport_rows),
        "accepted_identities": len(accepted_qids),
        "accepted_action_rows": len(accepted_actions),
        "gates": {
            "protocol_and_implementation_hashes_closed": True,
            "generation_output_hashes_closed": True,
            "fixed30_and_precall_29_1": True,
            "semantic_slots_exactly_90_including_skipped": True,
            "wal_intent_result_transport_response_capture_conserved": True,
            "wal_abort_absent": True,
            "accepted_actions_exactly_two_per_accepted_identity": True,
            "qid_order_and_hash_join_exact": True,
            "nonce_credential_endpoint_and_chain_secret_response_leak_absent": True,
            "role_artifact_isolation_and_model_binding_exact": True,
            "accepted_at_least_24": True,
        },
        "scientific_boundary": {
            "api_calls_made_by_auditor": 0,
            "retrieval_or_reader_calls_made_by_auditor": 0,
            "gpu_loaded_by_auditor": False,
            "training_started": False,
            "retrieval_support_gate": "NOT_RUN",
            "formal_pilot_release": "NOT_YET_AVAILABLE_RETRIEVAL_GATE_PENDING",
        },
    }


def audit_generation_v2(
    *,
    project_root: Path = PROJECT_ROOT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    protocol_manifest_path: Path | None = None,
    generation_dir: Path = DEFAULT_GENERATION_DIR,
) -> dict[str, Any]:
    """Audit a completed artifact and return only after every gate passes."""

    root = project_root.resolve()
    protocol = _resolve(root, protocol_path)
    protocol_manifest = _resolve(root, protocol_manifest_path) if protocol_manifest_path is not None else protocol.parent / "manifest.json"
    generation = _resolve(root, generation_dir)
    frozen, lock_path = _audit_protocol(
        project_root=root,
        protocol_path=protocol,
        protocol_manifest_path=protocol_manifest,
    )
    return _audit_generation(
        project_root=root,
        protocol=frozen,
        protocol_path=protocol,
        implementation_lock_path=lock_path,
        generation_dir=generation,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol_manifest", type=Path)
    parser.add_argument("--generation_dir", type=Path, default=DEFAULT_GENERATION_DIR)
    parser.add_argument("--execute_audit", action="store_true")
    args = parser.parse_args()
    if not args.execute_audit:
        raise SystemExit("No audit performed. Supply --execute_audit after reviewer approval.")
    try:
        report = audit_generation_v2(
            protocol_path=args.protocol,
            protocol_manifest_path=args.protocol_manifest,
            generation_dir=args.generation_dir,
        )
    except GenerationV2AuditError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error_code": exc.code}, sort_keys=True))
        raise SystemExit(2) from None
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
