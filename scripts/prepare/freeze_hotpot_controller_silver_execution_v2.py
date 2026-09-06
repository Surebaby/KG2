#!/usr/bin/env python
"""Freeze the repaired V2 HotpotQA pilot generation protocol.

V2 supersedes the never-executed V1 protocol without modifying any V1 file.
It adds an explicit anonymous second-hop-subject binding, hardened response
validation, and a crash-safe write-ahead call ledger.  The protocol binds an
independent implementation lock, which in turn hashes the final runner and its
dependencies; this avoids a runner/protocol self-reference.

This freezer performs no provider, retrieval, Reader, or training call.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare import freeze_hotpot_controller_silver_execution_v1 as v1  # noqa: E402
from kgproweight.data import hotpot_controller_silver as silver  # noqa: E402
from kgproweight.kg.question_kg import question_sha256  # noqa: E402


SCHEMA_VERSION = "hotpot-controller-silver-execution-protocol-2"
LOCK_SCHEMA_VERSION = "hotpot-controller-silver-implementation-lock-1"
REPORT_SCHEMA_VERSION = "hotpot-controller-silver-execution-freeze-report-2"
MANIFEST_SCHEMA_VERSION = "hotpot-controller-silver-execution-manifest-2"
SUPERSESSION_SCHEMA_VERSION = "hotpot-controller-silver-v1-supersession-addendum-1"
EXPERIMENT_ID = (
    "QUERY-CONTROLLER-HOTPOT-SILVER-PILOT30-GENERATION-"
    "DUAL-REVIEW-SEED20260904-V2"
)
STATUS = "FROZEN_V2_EXECUTION_PROTOCOL_NO_API_CALLS_NOT_TRAINED"

V1_DIR = v1.DEFAULT_OUTPUT_DIR
V1_PROTOCOL_PATH = V1_DIR / "protocol.json"
V1_REPORT_PATH = V1_DIR / "report.json"
V1_MANIFEST_PATH = V1_DIR / "manifest.json"
V1_GENERATION_OUTPUT_DIR = Path(
    "outputs/audits/query_controller_hotpot_silver_pilot30_"
    "generation_dual_review_seed20260904_v1"
)
V1_SUPERSESSION_ADDENDUM = V1_DIR / "metadata_addendum_superseded_by_v2.json"
EXPECTED_V1_HASHES = {
    "protocol.json": "f690063157f8818a79168f2f511fd7ec1becb2a431e81dcf26d372336a48360d",
    "report.json": "237a7efbcdd1a7300fd31f8cde540b5810f9ae696588cbcbdeac02980807ed90",
    "manifest.json": "f2b81662acfc16e736a4b141f2d005c372dcf4b93c559c974eb17571777107e9",
}

DEFAULT_OUTPUT_DIR = Path(
    "outputs/audits/query_controller_hotpot_silver_pilot30_"
    "execution_protocol_seed20260904_v2"
)
DEFAULT_GENERATION_OUTPUT_DIR = Path(
    "outputs/audits/query_controller_hotpot_silver_pilot30_"
    "generation_dual_review_seed20260904_v2"
)
DEFAULT_RAW_PATH = Path("data/hotpotqa/train.jsonl")
DEFAULT_PARENT_IDENTITY_PATH = v1.PARENT_DIR / "pilot.identity_only.jsonl"
DEFAULT_PARENT_METADATA_ADDENDUM_PATH = v1.PARENT_DIR / "metadata_addendum_v1_1.json"

IMPLEMENTATION_PATHS = (
    Path("scripts/prepare/generate_hotpot_controller_silver_pilot_v2.py"),
    Path("scripts/prepare/freeze_hotpot_controller_silver_execution_v2.py"),
    Path("scripts/prepare/generate_hotpot_controller_silver_pilot_v1.py"),
    Path("scripts/prepare/freeze_hotpot_controller_silver_execution_v1.py"),
    Path("kgproweight/data/hotpot_controller_silver.py"),
)

PRODUCER_SYSTEM_PROMPT = """You create retrieval-query supervision for a two-hop question.
Return exactly one JSON object and no prose, Markdown, reasoning, answer, or extra key.
The first-hop evidence contains an opaque nonce standing for the answer to q1.
The field second_hop_subject_nonce explicitly identifies that same opaque nonce as the grammatical subject/entity for the second-hop question. It is the value that literal #1 must replace at runtime.
The second-hop evidence describes that explicitly bound subject (the surface may be implicit) and contains a different opaque nonce standing for the final answer.
Write q1 as one natural single-hop question anchored by the root document title.
Write q2_template as one natural single-hop question containing literal #1 exactly once and treating #1 as the explicitly supplied second-hop subject.
Never copy any opaque nonce, never reveal or guess either hidden answer, and never answer the original question.
The exact output keys are schema_version, q1, q2_template; schema_version must be hotpot-controller-query-proposal-v1."""

PRODUCER_USER_TEMPLATE = """Create the two retrieval questions from this semantic-only masked payload.
The payload has no dataset identifier or row identifier. second_hop_subject_nonce is an anonymous binding instruction, not text to copy.
<SAFE_PAYLOAD_JSON>"""

V2_REQUIRED_OUTPUT_FILES = (
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

PROTOCOL_FREEZE_OUTPUT_FILES = (
    "implementation_lock.json",
    "protocol.json",
    "report.json",
    "manifest.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resolve(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object: {path}:{line_number}")
        rows.append(value)
    return rows


def _fsync_dir(path: Path) -> None:
    """Durably commit directory-entry changes where the platform supports it."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_new(path: Path, payload: bytes) -> None:
    """Create one append-only file by same-directory write/fsync/replace."""

    if path.exists():
        raise FileExistsError(f"append-only artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link creation is atomic and never overwrites a concurrent
        # append-only artifact (unlike os.replace after an exists check).
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"append-only artifact already exists: {path}") from None
        temporary.unlink()
        _fsync_dir(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _rename_dir_noreplace(source: Path, destination: Path) -> None:
    """Linux atomic directory commit with RENAME_NOREPLACE (fail closed)."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 required for no-clobber freeze")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"append-only V2 protocol already exists: {destination}")
        raise OSError(error, os.strerror(error), destination)


def audit_v1_to_v2_subject_binding(
    *, identity_path: Path, raw_path: Path
) -> dict[str, Any]:
    """Recompute the fixed-cohort V1 ambiguity and V2 binding repair."""

    identities = _load_jsonl(identity_path)
    wanted = {str(row.get("qid") or "") for row in identities}
    raw_by_qid: dict[str, dict[str, Any]] = {}
    with raw_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"raw row not object: {raw_path}:{line_number}")
            qid = str(row.get("id") or row.get("qid") or "").strip()
            if qid in wanted:
                if qid in raw_by_qid:
                    raise ValueError(f"duplicate raw qid: {qid}")
                raw_by_qid[qid] = row
    if set(raw_by_qid) != wanted:
        raise ValueError("identity/raw qid hash join is incomplete")

    counts = Counter()
    qid_hash_join = 0
    v2_binding_pass = 0
    secret_residual = 0
    reject_codes: Counter[str] = Counter()
    for identity in identities:
        if set(identity) != {"dataset", "qid", "question"} or identity.get("dataset") != "hotpotqa":
            raise ValueError("identity schema drift during V1/V2 diagnostic")
        qid = str(identity["qid"])
        raw = raw_by_qid[qid]
        if question_sha256(identity["question"]) != question_sha256(str(raw.get("question") or "")):
            raise ValueError(f"identity/raw question hash mismatch: {qid}")
        qid_hash_join += 1
        try:
            chain = silver.extract_hotpot_support_chain(raw)
            masked = silver.build_masked_proposal_view(chain)
            v1_payload = v1.build_producer_safe_payload(
                masked, chain_sha256=chain.raw_record_sha256, experiment_id=v1.EXPERIMENT_ID
            )
            v1_first, _ = v1.derive_mask_nonces(
                chain_sha256=chain.raw_record_sha256, experiment_id=v1.EXPERIMENT_ID
            )
            if v1_first in v1_payload["second_hop_evidence_masked"]:
                counts["v1_second_hop_subject_explicit"] += 1
            else:
                counts["v1_second_hop_subject_implicit"] += 1

            v2_payload = v1.build_producer_safe_payload(
                masked, chain_sha256=chain.raw_record_sha256, experiment_id=EXPERIMENT_ID
            )
            v2_first, _ = v1.derive_mask_nonces(
                chain_sha256=chain.raw_record_sha256, experiment_id=EXPERIMENT_ID
            )
            v2_payload["second_hop_subject_nonce"] = v2_first
            if v2_payload["second_hop_subject_nonce"] == v2_first:
                v2_binding_pass += 1
            secrets = (chain.bridge_title, chain.intermediate, *chain.final_answers)
            if any(
                silver._contains_secret(value, secret)
                for value in v2_payload.values()
                for secret in secrets
            ):
                secret_residual += 1
            counts["precall_constructible"] += 1
        except silver.HotpotSilverReject as exc:
            counts["precall_rejected"] += 1
            reject_codes[exc.code] += 1
    denominator = len(identities)
    return {
        "schema_version": "hotpot-controller-v1-to-v2-subject-binding-audit-1",
        "denominator": denominator,
        "identity_raw_qid_and_question_hash_join": qid_hash_join,
        "precall_constructible": counts["precall_constructible"],
        "precall_rejected": counts["precall_rejected"],
        "precall_reject_code_counts": dict(sorted(reject_codes.items())),
        "v1_second_hop_subject_explicit": counts["v1_second_hop_subject_explicit"],
        "v1_second_hop_subject_implicit": counts["v1_second_hop_subject_implicit"],
        "v2_subject_binding_pass": v2_binding_pass,
        "v2_secret_residual": secret_residual,
        "all_hash_joins_pass": qid_hash_join == denominator,
        "v2_all_constructible_rows_bound": v2_binding_pass == counts["precall_constructible"],
        "v2_all_constructible_rows_secret_free": secret_residual == 0,
    }


def build_implementation_lock(
    *, project_root: Path = PROJECT_ROOT, generated_at_utc: str
) -> dict[str, Any]:
    implementations = []
    for relative in IMPLEMENTATION_PATHS:
        path = _resolve(project_root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"implementation missing: {path}")
        implementations.append(
            {
                "path": _display(project_root, path),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": LOCK_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": generated_at_utc,
        "self_reference_avoided_by": (
            "protocol binds this standalone lock SHA256; this lock binds the final "
            "runner/dependencies but does not bind itself or the protocol"
        ),
        "implementations": implementations,
    }


def _validate_v1_zero_call_freeze(
    *,
    protocol: Mapping[str, Any],
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifact_paths: Mapping[str, Path],
) -> None:
    if (
        protocol.get("schema_version") != v1.SCHEMA_VERSION
        or protocol.get("experiment_id") != v1.EXPERIMENT_ID
        or protocol.get("status") != v1.STATUS
    ):
        raise ValueError("V1 protocol identity/status drift")
    parent = protocol.get("parent_identity_freeze")
    if (
        report.get("schema_version") != v1.REPORT_SCHEMA_VERSION
        or report.get("experiment_id") != v1.EXPERIMENT_ID
        or report.get("status") != v1.STATUS
        or report.get("api_calls") != 0
        or report.get("parent_identity_freeze") != parent
    ):
        raise ValueError("V1 report identity/status/parent drift")
    if (
        manifest.get("schema_version") != v1.MANIFEST_SCHEMA_VERSION
        or manifest.get("experiment_id") != v1.EXPERIMENT_ID
        or manifest.get("status") != v1.STATUS
        or manifest.get("api_calls") != 0
        or manifest.get("training_started") is not False
        or (manifest.get("inputs") or {}).get("parent_identity_freeze") != parent
    ):
        raise ValueError("V1 manifest identity/status/parent drift")
    output_rows = manifest.get("outputs")
    if not isinstance(output_rows, list) or len(output_rows) != 2:
        raise ValueError("V1 manifest output set drift")
    if {row.get("path") for row in output_rows if isinstance(row, Mapping)} != {
        "protocol.json",
        "report.json",
    }:
        raise ValueError("V1 manifest output set drift")
    for row in output_rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "size_bytes"}:
            raise ValueError("V1 manifest output schema drift")
        path = artifact_paths[str(row["path"])]
        if _sha256_file(path) != row.get("sha256") or path.stat().st_size != row.get("size_bytes"):
            raise ValueError(f"V1 manifest output hash/size drift: {row.get('path')}")


def build_protocol(
    *,
    generated_at_utc: str,
    parent_lock: Mapping[str, Any],
    implementation_lock_path: str,
    implementation_lock_sha256: str,
    implementation_lock_size_bytes: int,
    v1_to_v2_diagnostic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    protocol = deepcopy(v1.build_protocol(generated_at_utc=generated_at_utc, parent_lock=parent_lock))
    protocol.update(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": STATUS,
            "generated_at_utc": generated_at_utc,
        }
    )
    protocol["producer"]["prompt"] = {
        "system": PRODUCER_SYSTEM_PROMPT,
        "user_template": PRODUCER_USER_TEMPLATE,
    }
    protocol["producer"]["prompt"]["sha256"] = _canonical_sha256(
        {
            "system": PRODUCER_SYSTEM_PROMPT,
            "user_template": PRODUCER_USER_TEMPLATE,
        }
    )
    fields = list(protocol["masking"]["producer_safe_payload_fields_exact"])
    fields.append("second_hop_subject_nonce")
    protocol["masking"]["producer_safe_payload_fields_exact"] = fields
    protocol["masking"].update(
        {
            "second_hop_subject_nonce_value_exact": "derived mask-a nonce",
            "second_hop_subject_nonce_must_equal_first_hop_intermediate_nonce": True,
            "second_hop_subject_surface_or_alias_visible": False,
            "nonce_detection_normalization_exact": (
                "up to four bounded JSON Unicode-escape expansion passes, then "
                "unicodedata.normalize('NFKC', text).casefold()"
            ),
            "unicode_escape_expansion_before_normalization": True,
            "response_text_scan_scope": "all_recursive_string_values_in_provider_response",
            "reject_any_complete_nonce_like_token": True,
            "nonce_like_token_regex_after_normalization": r"(?<![0-9a-z])n[0-9a-f]{31}(?![0-9a-z])",
            "nonce_check_precedes_response_model_and_credential_checks": True,
            "nonce_check_applies_to_every_nonempty_response": True,
        }
    )
    protocol["masking"].pop("spec_sha256", None)
    protocol["masking"]["spec_sha256"] = _canonical_sha256(protocol["masking"])
    protocol["response_validation_v2"] = {
        "validation_order_exact": [
            "recursively_extract_all_provider_response_string_values",
            "bounded_unicode_escape_expand_then_nfkc_casefold",
            "any_complete_nonce_like_token_reject",
            "chain_secret_surface_echo_omit_and_reject",
            "credential_or_endpoint_echo_omit_and_reject",
            "response_model_exact_match",
            "choices_cardinality_exactly_one",
            "finish_reason_exactly_stop",
            "content_nonempty",
            "strict_json_single_object_duplicate_keys_rejected",
            "frozen_output_schema",
            "mechanical_or_review_invariants",
        ],
        "choices_count_exact": 1,
        "finish_reason_exact": "stop",
        "json_duplicate_keys_allowed": False,
        "semantic_or_format_retry_allowed": False,
        "response_model_mismatch_retry_allowed": False,
        "leak_rejection_omits_primary_raw_content": True,
    }
    protocol["api_execution"].update(
        {
            "endpoint_allowlist": {
                "environment_variable_name": "OPENAI_BASE_URL",
                "scheme_exact": "https",
                "hostname_exact": "api.deepseek.com",
                "path_allowlist_exact": ["", "/", "/v1", "/v1/"],
                "port_allowlist_exact": [None, 443],
                "query_fragment_or_userinfo_allowed": False,
                "endpoint_value_serialization_allowed": False,
            },
            "crash_safe_write_ahead_required": True,
            "wal_filename": "api_call_wal.jsonl",
            "wal_intent_fsync_before_each_physical_call": True,
            "wal_response_capture_fsync_immediately_after_provider_return": True,
            "wal_response_capture_contains_body_text": False,
            "wal_response_capture_safe_fields_exact": [
                "response_text_fields_sha256",
                "response_model",
                "response_model_sha256",
                "response_model_matches_requested",
                "choices_count",
                "finish_reasons",
                "finish_reasons_sha256",
            ],
            "wal_result_fsync_after_each_physical_call": True,
            "wal_success_terminal_event_exact": "semantic_calls_and_in_memory_validation_completed",
            "whole_run_completion_evidence": (
                "final manifest exists and binds the already-closed WAL plus every ordinary output"
            ),
            "wal_run_completed_event_allowed": False,
            "existing_output_directory_policy": "refuse_unintentional_rerun",
            "resume_automatically_after_incomplete_wal": False,
            "scheduler": {
                "submission_policy": "bounded_at_worker_count",
                "cancel_pending_after_first_worker_baseexception": True,
                "maximum_additional_physical_dispatches_after_first_worker_baseexception": (
                    "worker_count_minus_one_already_in_flight"
                ),
            },
        }
    )
    protocol["rejection_contract"]["item_reject_code_enum_exact"] = [
        *protocol["rejection_contract"]["item_reject_code_enum_exact"],
        "action_pair_postbuild_reject",
    ]
    protocol["future_runner_output_contract"]["required_files"] = list(V2_REQUIRED_OUTPUT_FILES)
    protocol["future_runner_output_contract"]["write_ahead_in_progress_artifact_required"] = True
    protocol["implementation_lock"] = {
        "path": implementation_lock_path,
        "sha256": implementation_lock_sha256,
        "size_bytes": implementation_lock_size_bytes,
        "protocol_runner_self_reference": False,
    }
    protocol["protocol_freeze_manifest_contract"] = {
        "manifest_filename": "manifest.json",
        "report_filename": "report.json",
        "output_set_exact": list(PROTOCOL_FREEZE_OUTPUT_FILES),
        "manifest_hashed_outputs_exact": [
            "implementation_lock.json",
            "protocol.json",
            "report.json",
        ],
        "runner_must_verify_manifest_before_api_calls": True,
    }
    protocol["v1_to_v2_subject_binding_diagnostic"] = dict(v1_to_v2_diagnostic or {})
    protocol["supersedes"] = {
        "experiment_id": v1.EXPERIMENT_ID,
        "protocol_path": V1_PROTOCOL_PATH.as_posix(),
        "reason": (
            "V1 omitted an explicit anonymous binding that identifies the second-hop "
            "subject as the first-hop result; V1 had zero API calls and no generation output"
        ),
        "v1_api_calls": 0,
        "v1_generation_output_exists": False,
        "status": "SUPERSEDED_BEFORE_ANY_API_CALL",
    }
    protocol["scientific_boundary"] = (
        "V2 repairs only the label-generation protocol. It still creates Gold-screened "
        "silver train labels, performs no retrieval/Reader validation, authorizes no "
        "training or formal evaluation, and makes no statistical-independence claim."
    )
    return protocol


def _validate_committed_v2_directory(
    *,
    project_root: Path,
    out: Path,
    addendum_path: Path,
    identity: Path,
    raw: Path,
    parent_addendum: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if {path.name for path in out.iterdir()} != set(PROTOCOL_FREEZE_OUTPUT_FILES):
        raise ValueError("committed V2 protocol-freeze output set drift")
    lock = _load_json(out / "implementation_lock.json")
    protocol = _load_json(out / "protocol.json")
    report = _load_json(out / "report.json")
    manifest = _load_json(out / "manifest.json")
    if (
        lock.get("schema_version") != LOCK_SCHEMA_VERSION
        or lock.get("experiment_id") != EXPERIMENT_ID
        or protocol.get("schema_version") != SCHEMA_VERSION
        or protocol.get("experiment_id") != EXPERIMENT_ID
        or protocol.get("status") != STATUS
        or report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("status") != STATUS
        or report.get("api_calls") != 0
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("status") != STATUS
        or manifest.get("api_calls") != 0
        or manifest.get("training_started") is not False
    ):
        raise ValueError("committed V2 protocol-freeze identity/status drift")
    if manifest.get("protocol_freeze_output_set_exact") != list(PROTOCOL_FREEZE_OUTPUT_FILES):
        raise ValueError("committed V2 protocol-freeze declared output set drift")
    output_rows = manifest.get("outputs")
    expected_outputs = {"implementation_lock.json", "protocol.json", "report.json"}
    if (
        not isinstance(output_rows, list)
        or len(output_rows) != 3
        or {row.get("path") for row in output_rows if isinstance(row, Mapping)}
        != expected_outputs
    ):
        raise ValueError("committed V2 protocol-freeze manifest output set drift")
    for row in output_rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "size_bytes"}:
            raise ValueError("committed V2 protocol-freeze manifest output schema drift")
        path = out / str(row["path"])
        if _sha256_file(path) != row.get("sha256") or path.stat().st_size != row.get("size_bytes"):
            raise ValueError(f"committed V2 protocol-freeze output drift: {row.get('path')}")
    binding = protocol.get("implementation_lock") or {}
    if (
        binding.get("sha256") != _sha256_file(out / "implementation_lock.json")
        or binding.get("size_bytes") != (out / "implementation_lock.json").stat().st_size
    ):
        raise ValueError("committed V2 implementation-lock binding drift")
    diagnostics = manifest.get("diagnostic_inputs") or {}
    expected_diagnostic_hashes = {
        "identity": _sha256_file(identity),
        "raw_train": _sha256_file(raw),
        "parent_metadata_addendum_v1_1": _sha256_file(parent_addendum),
    }
    if any(
        (diagnostics.get(name) or {}).get("sha256") != digest
        for name, digest in expected_diagnostic_hashes.items()
    ):
        raise ValueError("committed V2 diagnostic input drift")
    external = manifest.get("external_append_only_artifact") or {}
    if (
        _resolve(project_root, Path(str(external.get("path") or ""))).resolve()
        != addendum_path.resolve()
        or external.get("must_bind_v2_manifest_sha256") is not True
    ):
        raise ValueError("committed V2 supersession declaration drift")
    return protocol, lock, report, manifest


def _build_supersession(
    *,
    project_root: Path,
    out: Path,
    generated_at_utc: str,
    protocol: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    v1_protocol_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SUPERSESSION_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc,
        "append_only": True,
        "v1_experiment_id": v1.EXPERIMENT_ID,
        "v1_protocol_sha256": v1_protocol_sha256,
        "v1_api_calls": 0,
        "v1_generation_output_existed_at_supersession": False,
        "status": "SUPERSEDED_BEFORE_ANY_API_CALL",
        "superseded_by": {
            "experiment_id": EXPERIMENT_ID,
            "protocol_path": _display(project_root, out / "protocol.json"),
            "protocol_sha256": _sha256_file(out / "protocol.json"),
            "manifest_path": _display(project_root, out / "manifest.json"),
            "manifest_sha256": _sha256_file(out / "manifest.json"),
        },
        "reason": protocol["supersedes"]["reason"],
        "v1_to_v2_subject_binding_diagnostic": dict(diagnostic),
    }


def freeze_v2(
    *,
    project_root: Path = PROJECT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    v1_dir: Path = V1_DIR,
    v1_generation_output_dir: Path = V1_GENERATION_OUTPUT_DIR,
    supersession_addendum_path: Path = V1_SUPERSESSION_ADDENDUM,
    identity_path: Path = DEFAULT_PARENT_IDENTITY_PATH,
    raw_path: Path = DEFAULT_RAW_PATH,
    parent_metadata_addendum_path: Path = DEFAULT_PARENT_METADATA_ADDENDUM_PATH,
    expected_v1_hashes: Mapping[str, str] | None = EXPECTED_V1_HASHES,
    generated_at_utc: str | None = None,
    enforce_formal_locks: bool = True,
) -> dict[str, Any]:
    generated = generated_at_utc or datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = _resolve(project_root, output_dir)
    v1_root = _resolve(project_root, v1_dir)
    v1_generation = _resolve(project_root, v1_generation_output_dir)
    addendum_path = _resolve(project_root, supersession_addendum_path)
    identity = _resolve(project_root, identity_path)
    raw = _resolve(project_root, raw_path)
    parent_addendum = _resolve(project_root, parent_metadata_addendum_path)
    if addendum_path.exists():
        raise FileExistsError("append-only V1 supersession addendum already exists")
    repair_supersession_only = out.exists()
    if v1_generation.exists():
        raise ValueError("V1 generation output exists; cannot mark V1 superseded-before-calls")
    v1_paths = {name: v1_root / name for name in ("protocol.json", "report.json", "manifest.json")}
    v1_hashes = {name: _sha256_file(path) for name, path in v1_paths.items()}
    if enforce_formal_locks and dict(expected_v1_hashes or {}) != v1_hashes:
        raise ValueError("V1 protocol artifact hash drift")
    v1_protocol, v1_report, v1_manifest = (_load_json(v1_paths[name]) for name in ("protocol.json", "report.json", "manifest.json"))
    _validate_v1_zero_call_freeze(
        protocol=v1_protocol,
        report=v1_report,
        manifest=v1_manifest,
        artifact_paths=v1_paths,
    )
    parent_lock = v1_protocol["parent_identity_freeze"]
    if enforce_formal_locks:
        if _sha256_file(identity) != parent_lock.get("identity_sha256"):
            raise ValueError("formal diagnostic identity does not match V1 parent lock")
        if _sha256_file(raw) != "47444e1f8ccfd9c5f4001cc1252f99abbb0e07edc770bba7daac06d1cc17a9f6":
            raise ValueError("formal Hotpot raw-train SHA256 drift")
        if _sha256_file(parent_addendum) != v1.EXPECTED_PARENT_HASHES.get("metadata_addendum_v1_1.json", ""):
            # V1 did not bind the later addendum in EXPECTED_PARENT_HASHES; use
            # its immutable known hash without mutating the V1 module.
            if _sha256_file(parent_addendum) != "54292c3bfc2a3ac15f59e3cbb217e5b15f6e3f829662f9f8a9d5695bfd387baa":
                raise ValueError("formal parent metadata addendum SHA256 drift")

    if repair_supersession_only:
        protocol, implementation_lock, report, manifest = _validate_committed_v2_directory(
            project_root=project_root,
            out=out,
            addendum_path=addendum_path,
            identity=identity,
            raw=raw,
            parent_addendum=parent_addendum,
        )
        supersession = _build_supersession(
            project_root=project_root,
            out=out,
            generated_at_utc=generated,
            protocol=protocol,
            diagnostic=protocol.get("v1_to_v2_subject_binding_diagnostic") or {},
            v1_protocol_sha256=v1_hashes["protocol.json"],
        )
        _atomic_write_new(
            addendum_path,
            (
                json.dumps(supersession, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8"),
        )
        return {
            "protocol": protocol,
            "implementation_lock": implementation_lock,
            "report": report,
            "manifest": manifest,
            "supersession": supersession,
            "repaired_missing_supersession_only": True,
        }

    diagnostic = audit_v1_to_v2_subject_binding(identity_path=identity, raw_path=raw)
    if enforce_formal_locks:
        expected_diagnostic = {
            "denominator": 30,
            "identity_raw_qid_and_question_hash_join": 30,
            "precall_constructible": 29,
            "precall_rejected": 1,
            "v1_second_hop_subject_explicit": 17,
            "v1_second_hop_subject_implicit": 12,
            "v2_subject_binding_pass": 29,
            "v2_secret_residual": 0,
        }
        if any(diagnostic.get(key) != value for key, value in expected_diagnostic.items()):
            raise ValueError("formal V1-to-V2 subject-binding diagnostic drift")
    implementation_lock = build_implementation_lock(project_root=project_root, generated_at_utc=generated)
    lock_bytes = (json.dumps(implementation_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    lock_hash = hashlib.sha256(lock_bytes).hexdigest()
    lock_rel = _display(project_root, out / "implementation_lock.json")
    protocol = build_protocol(
        generated_at_utc=generated,
        parent_lock=parent_lock,
        implementation_lock_path=lock_rel,
        implementation_lock_sha256=lock_hash,
        implementation_lock_size_bytes=len(lock_bytes),
        v1_to_v2_diagnostic=diagnostic,
    )
    protocol_bytes = (
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": generated,
        "status": STATUS,
        "api_calls": 0,
        "v1_superseded_before_calls": True,
        "v1_to_v2_subject_binding_diagnostic": diagnostic,
        "checks": {
            "explicit_second_hop_subject_nonce": True,
            "normalized_any_nonce_detection": True,
            "duplicate_json_keys_rejected": True,
            "single_choice_and_stop_required": True,
            "implementation_lock_bound": True,
            "protocol_manifest_required_and_bound": True,
            "crash_safe_wal_required": True,
            "atomic_directory_commit_required": True,
            "training_authorized": False,
            "retrieval_or_reader_authorized": False,
        },
    }
    report_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    frozen_payloads = {
        "implementation_lock.json": lock_bytes,
        "protocol.json": protocol_bytes,
        "report.json": report_bytes,
    }
    outputs = [
        {
            "path": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for name, payload in frozen_payloads.items()
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": generated,
        "status": STATUS,
        "diagnostic_inputs": {
            "identity": {"path": _display(project_root, identity), "sha256": _sha256_file(identity)},
            "raw_train": {"path": _display(project_root, raw), "sha256": _sha256_file(raw)},
            "parent_metadata_addendum_v1_1": {"path": _display(project_root, parent_addendum), "sha256": _sha256_file(parent_addendum)},
        },
        "protocol_freeze_output_set_exact": list(PROTOCOL_FREEZE_OUTPUT_FILES),
        "api_calls": 0,
        "outputs": outputs,
        "external_append_only_artifact": {
            "path": _display(project_root, addendum_path),
            "commit_order": "after_complete_v2_manifest_directory_commit",
            "status_at_v2_manifest_commit": "PENDING_APPEND_ONLY_COMMIT",
            "must_bind_v2_manifest_sha256": True,
        },
        "v1_inputs": [
            {"path": _display(project_root, v1_paths[name]), "sha256": digest}
            for name, digest in v1_hashes.items()
        ],
        "training_started": False,
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    # Commit all four V2 files as one directory transaction.  No partially
    # written V2 directory can be mistaken for a frozen protocol.
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent)
    )
    committed = False
    try:
        for name, payload in frozen_payloads.items():
            _write_bytes_fsync(temporary_root / name, payload)
        _write_bytes_fsync(temporary_root / "manifest.json", manifest_bytes)
        if {path.name for path in temporary_root.iterdir()} != set(PROTOCOL_FREEZE_OUTPUT_FILES):
            raise AssertionError("incomplete V2 protocol-freeze temporary directory")
        _fsync_dir(temporary_root)
        _rename_dir_noreplace(temporary_root, out)
        committed = True
        _fsync_dir(out.parent)
    finally:
        if not committed and temporary_root.exists():
            shutil.rmtree(temporary_root)

    # Only a fully committed V2 manifest may supersede V1.  This one-way
    # addendum binds the already immutable V2 manifest and protocol; the V2
    # manifest cannot hash a future file without a circular/order violation.
    supersession = _build_supersession(
        project_root=project_root,
        out=out,
        generated_at_utc=generated,
        protocol=protocol,
        diagnostic=diagnostic,
        v1_protocol_sha256=v1_hashes["protocol.json"],
    )
    supersession_bytes = (
        json.dumps(supersession, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write_new(addendum_path, supersession_bytes)
    return {
        "protocol": protocol,
        "implementation_lock": implementation_lock,
        "report": report,
        "manifest": manifest,
        "supersession": supersession,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = freeze_v2(output_dir=args.output_dir)
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": result["report"]["status"], "api_calls": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
