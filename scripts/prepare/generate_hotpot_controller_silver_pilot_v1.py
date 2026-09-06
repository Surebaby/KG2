#!/usr/bin/env python
"""Run the frozen HotpotQA pilot30 query-label generation protocol.

The default command is deliberately non-executing.  Real provider calls require
``--execute_api``.  Tests inject a small ``CompletionClient`` and therefore do
not read credentials or use the network.

This runner is intentionally narrower than a general annotation utility.  It
consumes the already frozen identity denominator and execution protocol, keeps
pre-call failures in that denominator, makes at most one *semantic* call for
each stage, and permits retries only for the frozen transport-error classes.
It does not retrieve passages, call the Reader, train a model, or evaluate an
outcome metric.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgproweight.data import hotpot_controller_silver as silver  # noqa: E402
from kgproweight.kg.question_kg import question_sha256  # noqa: E402
from scripts.prepare import freeze_hotpot_controller_silver_execution_v1 as freeze  # noqa: E402


SCHEMA_VERSION = "hotpot-controller-silver-generation-run-1"
REPORT_SCHEMA_VERSION = "hotpot-controller-silver-generation-report-1"
MANIFEST_SCHEMA_VERSION = "hotpot-controller-silver-generation-manifest-1"
STATUS_COMPLETE = "COMPLETE_GENERATION_DUAL_REVIEW_RETRIEVAL_NOT_RUN_NOT_TRAINED"
STATUS_FAIL = "FAIL_GENERATION_OR_REVIEW_GATE_RETRIEVAL_NOT_RUN_NOT_TRAINED"

DEFAULT_PROTOCOL = Path(
    "outputs/audits/query_controller_hotpot_silver_pilot30_"
    "execution_protocol_seed20260904_v1/protocol.json"
)
EXPECTED_PROTOCOL_SHA256 = "f690063157f8818a79168f2f511fd7ec1becb2a431e81dcf26d372336a48360d"
DEFAULT_PARENT_DIR = Path(
    "outputs/audits/query_controller_hotpot_silver_label_coverage_"
    "pilot30_seed20260904_v1"
)
DEFAULT_IDENTITY = DEFAULT_PARENT_DIR / "pilot.identity_only.jsonl"
DEFAULT_ADDENDUM = DEFAULT_PARENT_DIR / "metadata_addendum_v1_1.json"
EXPECTED_ADDENDUM_SHA256 = "54292c3bfc2a3ac15f59e3cbb217e5b15f6e3f829662f9f8a9d5695bfd387baa"
DEFAULT_RAW = Path("data/hotpotqa/train.jsonl")
EXPECTED_RAW_SHA256 = "47444e1f8ccfd9c5f4001cc1252f99abbb0e07edc770bba7daac06d1cc17a9f6"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/audits/query_controller_hotpot_silver_pilot30_"
    "generation_dual_review_seed20260904_v1"
)

ROLE_FILES = {
    "producer": "producer_proposals.jsonl",
    "reviewer_1_blind": "reviewer_1_reviews.jsonl",
    "reviewer_2_gold_aware": "reviewer_2_reviews.jsonl",
}
OTHER_OUTPUT_FILES = (
    "accepted_actions.jsonl",
    "failures.jsonl",
    "semantic_call_ledger.jsonl",
    "api_transport_attempt_ledger.jsonl",
    "report.json",
)
ALL_OUTPUT_FILES = (*ROLE_FILES.values(), *OTHER_OUTPUT_FILES, "manifest.json")
STAGES = ("producer", "reviewer_1_blind", "reviewer_2_gold_aware")
_NONCE_RE = re.compile(freeze.NONCE_PATTERN)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class CompletionClient(Protocol):
    """Minimal injectable boundary used by the runner and CPU-only tests."""

    def complete(self, request_body: Mapping[str, Any], *, timeout: float) -> Any:
        ...


class OpenAICompletionClient:
    """OpenAI-compatible DeepSeek client; constructed only after --execute_api."""

    def __init__(self) -> None:
        from openai import OpenAI

        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY or OPENAI_API_KEY is required")
        # Do not expose either value through a public attribute or any artifact.
        base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.deepseek.com"
        self.__client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, request_body: Mapping[str, Any], *, timeout: float) -> Any:
        return self.__client.chat.completions.create(**dict(request_body), timeout=timeout)


class SyntheticTransportError(RuntimeError):
    """Test/fake-client transport failure with an explicit safe classification."""

    def __init__(self, error_class: str, *, http_status: int | None = None) -> None:
        self.error_class = error_class
        self.http_status = http_status
        super().__init__(error_class)


@dataclass
class StageResult:
    stage: str
    semantic_request_id: str
    requested_model: str
    messages_sha256: str
    safe_payload_sha256: str
    prompt_template_sha256: str
    semantic_status: str
    finish_reason: str | None = None
    response_model: str | None = None
    raw_content: str | None = None
    raw_response_sha256: str | None = None
    parsed: dict[str, Any] | None = None
    parsed_response_sha256: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    wall_time_ms: int = 0
    nonce_echo_count: int = 0
    reject_code: str | None = None
    detail_code: str | None = None
    transport_rows: list[dict[str, Any]] | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object: {path}:{line_number}")
        rows.append(value)
    return rows


def _read_selected_raw(path: Path, selected_qids: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"raw row is not an object: {path}:{line_number}")
            qid = str(value.get("id") or value.get("qid") or "").strip()
            if qid in selected_qids:
                if qid in found:
                    raise ValueError(f"duplicate selected raw qid: {qid}")
                found[qid] = value
    return found


def _assert_protocol(protocol: Mapping[str, Any], *, expected_rows: int) -> None:
    if protocol.get("schema_version") != freeze.SCHEMA_VERSION:
        raise ValueError("execution protocol schema drift")
    if protocol.get("status") != freeze.STATUS:
        raise ValueError("execution protocol status drift")
    if protocol.get("experiment_id") != freeze.EXPERIMENT_ID:
        raise ValueError("execution protocol experiment drift")
    if (protocol.get("parent_identity_freeze") or {}).get("rows") != expected_rows:
        raise ValueError("execution protocol denominator drift")
    auth = protocol.get("authorization") or {}
    if not all(
        auth.get(key) is True
        for key in (
            "pilot_q1_q2_producer_calls",
            "pilot_blind_reviewer_calls",
            "pilot_gold_aware_reviewer_calls",
        )
    ):
        raise ValueError("execution protocol does not authorize all three stages")
    if any(
        auth.get(key) is not False
        for key in ("retrieval_or_reader_calls", "training", "formal_em_f1_ihr_evaluation")
    ):
        raise ValueError("execution protocol scope is broader than this runner")
    if tuple((protocol.get("api_execution") or {}).get("semantic_request_stages_exact", ())) != STAGES:
        raise ValueError("semantic stage contract drift")
    if set((protocol.get("future_runner_output_contract") or {}).get("required_files", ())) != set(ALL_OUTPUT_FILES):
        raise ValueError("runner output contract drift")
    if (protocol.get("producer") or {}).get("post_parse_validator") != (
        "kgproweight.data.hotpot_controller_silver.validate_query_proposal"
    ):
        raise ValueError("producer post-parse validator drift")


def _render_messages(system: str, user_template: str, marker: str, payload: Mapping[str, Any]) -> list[dict[str, str]]:
    if user_template.count(marker) != 1:
        raise ValueError("prompt marker cardinality drift")
    payload_json = _canonical_bytes(payload).decode("utf-8")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_template.replace(marker, payload_json)},
    ]


def _request_body(stage_spec: Mapping[str, Any], messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    request = dict(stage_spec["request"])
    if request.pop("provider", None) != "deepseek_openai_compatible_api":
        raise ValueError("provider drift")
    if request.pop("api_family", None) != "chat.completions":
        raise ValueError("api family drift")
    thinking = request.pop("thinking", None)
    body = {"model": stage_spec["model"], "messages": [dict(item) for item in messages], **request}
    if thinking is not None:
        body["extra_body"] = {"thinking": thinking}
    return body


def _transport_class(exc: BaseException) -> tuple[str, int | None, bool]:
    if isinstance(exc, SyntheticTransportError):
        kind, status = exc.error_class, exc.http_status
    else:
        status_value = getattr(exc, "status_code", None)
        status = int(status_value) if isinstance(status_value, int) else None
        name = type(exc).__name__.casefold()
        if status is not None:
            kind = f"http_{status}"
        elif isinstance(exc, TimeoutError) or "timeout" in name:
            kind = "timeout"
        elif "connection" in name:
            kind = "connection_error"
        else:
            kind = "nonretryable_transport_error"
    retryable = kind in {
        "connection_error", "timeout", "http_429", "http_500", "http_502", "http_503", "http_504"
    }
    return kind, status, retryable


def _field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_response(response: Any) -> dict[str, Any]:
    choices = _field(response, "choices", []) or []
    if not isinstance(choices, Sequence) or not choices:
        return {"missing": True}
    choice = choices[0]
    message = _field(choice, "message", None)
    content = _field(message, "content", None)
    usage = _field(response, "usage", None)
    completion_details = _field(usage, "completion_tokens_details", None)
    reasoning_tokens = _field(completion_details, "reasoning_tokens", None)
    if reasoning_tokens is None:
        reasoning_content = _field(message, "reasoning_content", None)
        reasoning_tokens = None if not reasoning_content else 0
    return {
        "missing": not isinstance(content, str) or not content.strip(),
        "content": content if isinstance(content, str) else None,
        "model": str(_field(response, "model", "") or ""),
        "finish_reason": _field(choice, "finish_reason", None),
        "prompt_tokens": _field(usage, "prompt_tokens", None),
        "completion_tokens": _field(usage, "completion_tokens", None),
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": _field(usage, "total_tokens", None),
    }


class _ApiExecutor:
    def __init__(self, client: CompletionClient, protocol: Mapping[str, Any]) -> None:
        api = protocol["api_execution"]
        self.client = client
        self.experiment_id = str(protocol["experiment_id"])
        self.timeout = float(api["timeout_seconds_per_physical_attempt"])
        self.max_attempts = int(api["max_physical_attempts_per_semantic_request"])
        self.delay = float(api["minimum_inter_request_delay_seconds_per_worker"])
        self._thread_state = threading.local()

    def _pace(self) -> None:
        last = getattr(self._thread_state, "last_request", None)
        if last is not None:
            remaining = self.delay - (time.monotonic() - last)
            if remaining > 0:
                time.sleep(remaining)
        self._thread_state.last_request = time.monotonic()

    def call(
        self,
        *,
        stage: str,
        qid: str,
        stage_spec: Mapping[str, Any],
        messages: list[dict[str, str]],
        safe_payload_sha256: str,
        prompt_template_sha256: str,
    ) -> StageResult:
        request_id = _sha256_text(f"{self.experiment_id}\0{qid}\0{stage}")
        body = _request_body(stage_spec, messages)
        request_bytes = _canonical_bytes(body)
        request_hash = hashlib.sha256(request_bytes).hexdigest()
        rows: list[dict[str, Any]] = []
        started_semantic = time.monotonic()
        response: Any = None
        last_kind: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            self._pace()
            started_at = _utc_now()
            started = time.monotonic()
            status: int | None = None
            kind: str | None = None
            retryable = False
            received = False
            try:
                # A fresh object avoids cross-call mutation while preserving the
                # exact canonical request bytes across transport retries.
                outbound = json.loads(request_bytes.decode("utf-8"))
                response = self.client.complete(outbound, timeout=self.timeout)
                received = True
                status_value = _field(response, "http_status", 200)
                status = int(status_value) if isinstance(status_value, int) else 200
            except Exception as exc:  # noqa: BLE001 - classified, never serialized
                kind, status, retryable = _transport_class(exc)
                last_kind = kind
            ended_at = _utc_now()
            rows.append(
                {
                    "experiment_id": self.experiment_id,
                    "semantic_request_id": request_id,
                    "stage": stage,
                    "physical_attempt_index": attempt,
                    "request_body_sha256": request_hash,
                    "request_bytes_identical_to_attempt_1": True,
                    "started_at_utc": started_at,
                    "ended_at_utc": ended_at,
                    "wall_time_ms": round((time.monotonic() - started) * 1000),
                    "http_status": status,
                    "transport_error_class": kind,
                    "transport_retryable": retryable,
                    "response_received": received,
                }
            )
            if received:
                break
            if not retryable:
                break
        result = StageResult(
            stage=stage,
            semantic_request_id=request_id,
            requested_model=str(stage_spec["model"]),
            messages_sha256=_canonical_sha256(messages),
            safe_payload_sha256=safe_payload_sha256,
            prompt_template_sha256=prompt_template_sha256,
            semantic_status="response_received" if response is not None else "transport_exhausted",
            wall_time_ms=round((time.monotonic() - started_semantic) * 1000),
            transport_rows=rows,
            reject_code=None if response is not None else f"{stage.split('_blind')[0].split('_gold')[0]}_transport_exhausted",
            detail_code=last_kind,
        )
        if response is None:
            # Stage names and frozen item codes do not have the same spelling.
            result.reject_code = {
                "producer": "producer_transport_exhausted",
                "reviewer_1_blind": "reviewer_1_transport_exhausted",
                "reviewer_2_gold_aware": "reviewer_2_transport_exhausted",
            }[stage]
            return result
        parsed = _extract_response(response)
        result.finish_reason = parsed.get("finish_reason")
        result.response_model = parsed.get("model")
        result.raw_content = parsed.get("content")
        result.prompt_tokens = parsed.get("prompt_tokens")
        result.completion_tokens = parsed.get("completion_tokens")
        result.reasoning_tokens = parsed.get("reasoning_tokens")
        result.total_tokens = parsed.get("total_tokens")
        if result.raw_content is not None:
            result.raw_response_sha256 = _sha256_text(result.raw_content)
        if parsed.get("missing"):
            result.semantic_status = "rejected"
            result.reject_code = {
                "producer": "producer_missing_response",
                "reviewer_1_blind": "reviewer_1_missing_response",
                "reviewer_2_gold_aware": "reviewer_2_missing_response",
            }[stage]
            result.detail_code = "empty_or_missing_content"
        elif result.response_model != result.requested_model:
            result.semantic_status = "rejected"
            result.reject_code = {
                "producer": "producer_response_model_mismatch",
                "reviewer_1_blind": "reviewer_1_response_model_mismatch",
                "reviewer_2_gold_aware": "reviewer_2_response_model_mismatch",
            }[stage]
            result.detail_code = "response_model_mismatch"
        return result


def _parse_exact_json(content: str) -> dict[str, Any]:
    value = json.loads(content)
    if not isinstance(value, dict):
        raise TypeError("top_level_not_object")
    return value


def _validate_producer_schema(value: Mapping[str, Any]) -> None:
    if set(value) != {"schema_version", "q1", "q2_template"}:
        raise ValueError("producer_keys")
    if value.get("schema_version") != silver.PROPOSAL_SCHEMA_VERSION:
        raise ValueError("producer_schema_version")
    for field in ("q1", "q2_template"):
        item = value.get(field)
        if not isinstance(item, str) or not 2 <= len(item) <= 320:
            raise ValueError(f"producer_{field}_type_or_length")


def _validate_review_schema(value: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    schema = protocol["schemas"]["review_output_shared"]["schema"]
    required = set(schema["required"])
    if set(value) != required:
        raise ValueError("review_keys")
    if value.get("schema_version") != freeze.REVIEW_SCHEMA_VERSION:
        raise ValueError("review_schema_version")
    booleans = tuple(freeze.REVIEW_BOOLEAN_FIELDS)
    if any(type(value.get(field)) is not bool for field in booleans):
        raise ValueError("review_boolean_type")
    verdict = value.get("verdict")
    codes = value.get("reject_codes")
    if verdict not in {"pass", "reject"} or not isinstance(codes, list):
        raise ValueError("review_verdict_or_codes")
    if any(not isinstance(code, str) or code not in freeze.REVIEW_REJECT_CODES for code in codes):
        raise ValueError("review_reject_code")
    if len(codes) != len(set(codes)):
        raise ValueError("review_duplicate_reject_code")
    expected = {
        freeze.REVIEW_REJECT_CODE_BY_FIELD[field]
        for field in booleans
        if value[field] is False
    }
    actual_non_unknown = set(codes) - {"unknown_or_insufficient_context"}
    if actual_non_unknown != expected:
        raise ValueError("review_boolean_code_invariant")
    all_true = all(value[field] for field in booleans)
    if (verdict == "pass") != (all_true and not codes):
        raise ValueError("review_pass_invariant")
    if verdict == "reject" and not codes:
        raise ValueError("review_reject_without_code")


def _response_contains_secret(content: str, chain: silver.HotpotSupportChain) -> bool:
    secrets = (chain.bridge_title, chain.intermediate, *chain.final_answers)
    return any(silver._contains_secret(content, secret) for secret in secrets)


def _finalize_producer(result: StageResult, chain: silver.HotpotSupportChain, nonces: tuple[str, str]) -> None:
    if result.semantic_status != "response_received" or result.raw_content is None:
        return
    result.nonce_echo_count = freeze.nonce_echo_count(result.raw_content, nonces=nonces)
    if result.nonce_echo_count:
        result.semantic_status = "rejected"
        result.reject_code = "producer_nonce_echo"
        result.detail_code = "opaque_nonce_echo"
        return
    try:
        parsed = _parse_exact_json(result.raw_content)
    except (json.JSONDecodeError, TypeError):
        result.semantic_status = "rejected"
        result.reject_code = "producer_json_parse_error"
        result.detail_code = "strict_json_object_required"
        return
    result.parsed_response_sha256 = _canonical_sha256(parsed)
    try:
        _validate_producer_schema(parsed)
    except ValueError as exc:
        result.semantic_status = "rejected"
        result.reject_code = "producer_output_schema_error"
        result.detail_code = str(exc)
        return
    try:
        validated = silver.validate_query_proposal(parsed, chain)
    except silver.HotpotSilverReject as exc:
        result.semantic_status = "rejected"
        result.reject_code = "producer_mechanical_reject"
        result.detail_code = exc.code
        return
    result.parsed = {
        "schema_version": silver.PROPOSAL_SCHEMA_VERSION,
        "q1": validated.q1_query,
        "q2_template": validated.q2_template,
    }
    result.parsed_response_sha256 = validated.proposal_sha256
    result.semantic_status = "accepted"


def _finalize_review(
    result: StageResult,
    *,
    protocol: Mapping[str, Any],
    chain: silver.HotpotSupportChain,
    nonces: tuple[str, str],
) -> None:
    if result.semantic_status != "response_received" or result.raw_content is None:
        return
    result.nonce_echo_count = freeze.nonce_echo_count(result.raw_content, nonces=nonces)
    prefix = "reviewer_1" if result.stage == "reviewer_1_blind" else "reviewer_2"
    if result.nonce_echo_count:
        result.semantic_status = "rejected"
        result.reject_code = f"{prefix}_nonce_echo"
        result.detail_code = "opaque_nonce_echo"
        return
    # All reviewers are forbidden to emit chain secrets.  A valid frozen review
    # contains only booleans/codes, so any such surface is necessarily invalid.
    if _response_contains_secret(result.raw_content, chain):
        result.semantic_status = "rejected"
        result.reject_code = f"{prefix}_output_schema_error"
        result.detail_code = "forbidden_chain_secret_echo"
        return
    try:
        parsed = _parse_exact_json(result.raw_content)
    except (json.JSONDecodeError, TypeError):
        result.semantic_status = "rejected"
        result.reject_code = f"{prefix}_json_parse_error"
        result.detail_code = "strict_json_object_required"
        return
    result.parsed_response_sha256 = _canonical_sha256(parsed)
    try:
        _validate_review_schema(parsed, protocol)
    except ValueError as exc:
        result.semantic_status = "rejected"
        result.reject_code = f"{prefix}_output_schema_error"
        result.detail_code = str(exc)
        return
    result.parsed = dict(parsed)
    if parsed["verdict"] == "pass":
        result.semantic_status = "accepted"
    else:
        result.semantic_status = "rejected"
        result.reject_code = f"{prefix}_reject"
        result.detail_code = ",".join(parsed["reject_codes"])


def _skipped_result(
    *,
    stage: str,
    experiment_id: str,
    qid: str,
    requested_model: str,
    prompt_template_sha256: str,
    detail: str,
) -> StageResult:
    return StageResult(
        stage=stage,
        semantic_request_id=_sha256_text(f"{experiment_id}\0{qid}\0{stage}"),
        requested_model=requested_model,
        messages_sha256=_canonical_sha256([]),
        safe_payload_sha256=_canonical_sha256({}),
        prompt_template_sha256=prompt_template_sha256,
        semantic_status="not_executed_upstream_failure",
        reject_code=None,
        detail_code=detail,
        transport_rows=[],
    )


def _reject_sensitive_serialization(result: StageResult) -> None:
    """Fail closed if provider content contains a credential/endpoint value.

    Exception strings are never retained.  This additional guard covers the
    unlikely case that provider content itself reflects a configured secret.
    The raw hash remains auditable, while the unsafe content is discarded.
    """

    content = result.raw_content
    if not content:
        return
    configured = {
        value
        for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_BASE_URL")
        if (value := os.environ.get(name)) and len(value) >= 6
    }
    if not any(value in content for value in configured):
        return
    prefix = "producer" if result.stage == "producer" else (
        "reviewer_1" if result.stage == "reviewer_1_blind" else "reviewer_2"
    )
    result.raw_content = None
    result.semantic_status = "rejected"
    result.reject_code = f"{prefix}_output_schema_error"
    result.detail_code = "credential_or_endpoint_echo_omitted"


def _ledger_row(
    result: StageResult, *, experiment_id: str, identity: Mapping[str, Any]
) -> dict[str, Any]:
    row = {
        "experiment_id": experiment_id,
        "dataset": identity["dataset"],
        "qid": identity["qid"],
        "question_sha256": question_sha256(identity["question"]),
        "semantic_request_id": result.semantic_request_id,
        "stage": result.stage,
        "requested_model": result.requested_model,
        "expected_response_model": result.requested_model,
        "model_visible_messages_sha256": result.messages_sha256,
        "safe_payload_sha256": result.safe_payload_sha256,
        "prompt_template_sha256": result.prompt_template_sha256,
        "semantic_attempt_count": 0 if result.semantic_status == "not_executed_upstream_failure" else 1,
        "transport_attempt_count": len(result.transport_rows or []),
        "status": result.semantic_status,
        "finish_reason": result.finish_reason,
        "response_model": result.response_model,
        "raw_response_sha256": result.raw_response_sha256,
        "parsed_response_sha256": result.parsed_response_sha256,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "total_tokens": result.total_tokens,
        "wall_time_ms": result.wall_time_ms,
        "nonce_echo_count": result.nonce_echo_count,
        "reject_code": result.reject_code,
        "detail_code": result.detail_code,
    }
    if tuple(row) != tuple(freeze.SEMANTIC_CALL_LEDGER_FIELDS):
        raise AssertionError("semantic ledger field drift")
    return row


def _role_row(identity: Mapping[str, Any], result: StageResult) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": identity["dataset"],
        "qid": identity["qid"],
        "question": identity["question"],
        "question_sha256": question_sha256(identity["question"]),
        "stage": result.stage,
        "status": result.semantic_status,
        "semantic_request_id": result.semantic_request_id,
        "requested_model": result.requested_model,
        "response_model": result.response_model,
        "finish_reason": result.finish_reason,
        "raw_response_content": result.raw_content,
        "raw_response_sha256": result.raw_response_sha256,
        "parsed_response": result.parsed,
        "parsed_response_sha256": result.parsed_response_sha256,
        "nonce_echo_count": result.nonce_echo_count,
        "reject_code": result.reject_code,
        "detail_code": result.detail_code,
    }


def _stage_spec(protocol: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    if stage == "producer":
        return protocol["producer"]
    key = "reviewer_1" if stage == "reviewer_1_blind" else "reviewer_2"
    return protocol["reviewers"][key]


def _messages_and_payload(
    *,
    stage: str,
    protocol: Mapping[str, Any],
    safe_payload: Mapping[str, Any],
    proposal: Mapping[str, Any] | None,
    chain: silver.HotpotSupportChain,
) -> tuple[list[dict[str, str]], dict[str, Any], str]:
    spec = _stage_spec(protocol, stage)
    prompt = spec["prompt"]
    if stage == "producer":
        payload = dict(safe_payload)
        marker = "<SAFE_PAYLOAD_JSON>"
    elif stage == "reviewer_1_blind":
        payload = {"semantic_payload": dict(safe_payload), "proposal": dict(proposal or {})}
        marker = "<BLIND_REVIEW_PAYLOAD_JSON>"
    else:
        if proposal is None:
            raise ValueError("gold-aware reviewer requires proposal")
        validated = silver.validate_query_proposal(proposal, chain)
        payload = {
            "original_question": chain.question,
            "root_document_title": chain.root_title,
            "bridge_document_title": chain.bridge_title,
            "first_hop_support": {
                "document_title": chain.first_hop.document_title,
                "sentence_index": chain.first_hop.sentence_index,
                "evidence_excerpt": chain.first_hop.evidence_excerpt,
            },
            "intermediate_answer": chain.intermediate,
            "second_hop_support": {
                "document_title": chain.second_hop.document_title,
                "sentence_index": chain.second_hop.sentence_index,
                "evidence_excerpt": chain.second_hop.evidence_excerpt,
            },
            "final_answers": list(chain.final_answers),
            "proposal": dict(proposal),
            "instantiated_q2": validated.q2_query,
        }
        marker = "<GOLD_AWARE_REVIEW_PAYLOAD_JSON>"
    messages = _render_messages(prompt["system"], prompt["user_template"], marker, payload)
    return messages, payload, str(prompt["sha256"])


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_generation_pilot(
    *,
    client: CompletionClient,
    project_root: Path = PROJECT_ROOT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    identity_path: Path = DEFAULT_IDENTITY,
    metadata_addendum_path: Path = DEFAULT_ADDENDUM,
    raw_path: Path = DEFAULT_RAW,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_rows: int = freeze.PILOT_ROWS,
    enforce_formal_locks: bool = True,
) -> dict[str, Any]:
    """Execute one frozen pilot and atomically materialise its nine artifacts."""

    paths = {
        "protocol": _resolve(project_root, protocol_path),
        "identity": _resolve(project_root, identity_path),
        "addendum": _resolve(project_root, metadata_addendum_path),
        "raw": _resolve(project_root, raw_path),
        "output": _resolve(project_root, output_dir),
    }
    if paths["output"].exists():
        raise FileExistsError(f"append-only output already exists: {paths['output']}")
    protocol_hash = _sha256_file(paths["protocol"])
    addendum_hash = _sha256_file(paths["addendum"])
    raw_hash = _sha256_file(paths["raw"])
    if enforce_formal_locks:
        if protocol_hash != EXPECTED_PROTOCOL_SHA256:
            raise ValueError("formal execution protocol SHA256 drift")
        if addendum_hash != EXPECTED_ADDENDUM_SHA256:
            raise ValueError("formal metadata addendum SHA256 drift")
        if raw_hash != EXPECTED_RAW_SHA256:
            raise ValueError("formal raw train SHA256 drift")
    protocol = _load_json(paths["protocol"])
    addendum = _load_json(paths["addendum"])
    _assert_protocol(protocol, expected_rows=expected_rows)
    if enforce_formal_locks:
        # Re-check the four parent locks at execution time, rather than trusting
        # that the identity-only directory stayed unchanged after freezing.
        artifacts = (protocol.get("parent_identity_freeze") or {}).get("artifacts") or {}
        if set(artifacts) != set(freeze.PARENT_FILENAMES):
            raise ValueError("formal parent artifact inventory drift")
        for filename, lock in artifacts.items():
            if not isinstance(lock, Mapping):
                raise ValueError("formal parent artifact lock malformed")
            parent_path = _resolve(project_root, Path(str(lock.get("path") or "")))
            if _sha256_file(parent_path) != lock.get("sha256"):
                raise ValueError(f"formal parent artifact SHA256 drift: {filename}")
    if addendum.get("selection_result_changed") is not False:
        raise ValueError("parent metadata addendum changed the frozen identity set")
    hardening = addendum.get("precall_hardening_audit") or {}
    if hardening.get("builder_version") != silver.BUILDER_VERSION:
        raise ValueError("metadata addendum builder version drift")
    if int(hardening.get("fixed_denominator", -1)) != expected_rows:
        raise ValueError("metadata addendum denominator drift")
    if str(hardening.get("builder_sha256")) != _sha256_file(
        _resolve(project_root, Path("kgproweight/data/hotpot_controller_silver.py"))
    ):
        raise ValueError("metadata addendum builder SHA256 drift")

    identities = _load_jsonl(paths["identity"])
    if len(identities) != expected_rows or len({row.get("qid") for row in identities}) != expected_rows:
        raise ValueError("fixed parent denominator or qid uniqueness drift")
    parent_identity_hash = (protocol.get("parent_identity_freeze") or {}).get("identity_sha256")
    if _sha256_file(paths["identity"]) != parent_identity_hash:
        raise ValueError("identity file does not match execution protocol")
    for row in identities:
        if set(row) != {"dataset", "qid", "question"} or row.get("dataset") != silver.DATASET:
            raise ValueError("identity schema drift")
    raw_by_qid = _read_selected_raw(paths["raw"], {str(row["qid"]) for row in identities})

    experiment_id = str(protocol["experiment_id"])
    executor = _ApiExecutor(client, protocol)
    prepared: dict[str, dict[str, Any]] = {}
    stages_by_qid: dict[str, dict[str, StageResult]] = {str(row["qid"]): {} for row in identities}
    failures: dict[str, dict[str, Any]] = {}

    # Pre-call integrity is intentionally exhaustive and API-free.  Every
    # rejected identity receives three skipped semantic ledger rows.
    for identity in identities:
        qid = str(identity["qid"])
        try:
            raw = raw_by_qid[qid]
            if question_sha256(str(identity["question"])) != question_sha256(str(raw.get("question") or "")):
                raise ValueError("identity_question_hash_mismatch")
            chain = silver.extract_hotpot_support_chain(raw)
            if chain.qid != qid or chain.question != identity["question"]:
                raise ValueError("identity_chain_join_mismatch")
            masked = silver.build_masked_proposal_view(chain)
            safe_payload = freeze.build_producer_safe_payload(
                masked, chain_sha256=chain.raw_record_sha256, experiment_id=experiment_id
            )
            nonces = freeze.derive_mask_nonces(
                chain_sha256=chain.raw_record_sha256, experiment_id=experiment_id
            )
            prepared[qid] = {"identity": identity, "chain": chain, "safe_payload": safe_payload, "nonces": nonces}
        except (KeyError, ValueError, silver.HotpotSilverReject) as exc:
            detail = exc.code if isinstance(exc, silver.HotpotSilverReject) else type(exc).__name__
            failures[qid] = {
                "schema_version": SCHEMA_VERSION,
                "dataset": identity["dataset"],
                "qid": qid,
                "question_sha256": question_sha256(identity["question"]),
                "status": "precall_rejected",
                "reject_code": "input_identity_or_chain_integrity_error",
                "detail_code": detail,
            }
            for stage in STAGES:
                spec = _stage_spec(protocol, stage)
                stages_by_qid[qid][stage] = _skipped_result(
                    stage=stage,
                    experiment_id=experiment_id,
                    qid=qid,
                    requested_model=str(spec["model"]),
                    prompt_template_sha256=str(spec["prompt"]["sha256"]),
                    detail="precall_identity_or_chain_failure",
                )
    expected_precall = int(hardening.get("precall_rejected", -1))
    if enforce_formal_locks and len(failures) != expected_precall:
        raise ValueError("observed pre-call failure count differs from bound addendum")

    workers = int(protocol["api_execution"]["worker_count"])

    def producer_job(qid: str) -> StageResult:
        item = prepared[qid]
        messages, payload, prompt_hash = _messages_and_payload(
            stage="producer", protocol=protocol, safe_payload=item["safe_payload"], proposal=None, chain=item["chain"]
        )
        result = executor.call(
            stage="producer", qid=qid, stage_spec=protocol["producer"], messages=messages,
            safe_payload_sha256=_canonical_sha256(payload), prompt_template_sha256=prompt_hash,
        )
        _reject_sensitive_serialization(result)
        _finalize_producer(result, item["chain"], item["nonces"])
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(producer_job, qid): qid for qid in prepared}
        for future in as_completed(futures):
            qid = futures[future]
            stages_by_qid[qid]["producer"] = future.result()

    # Fill skipped review ledgers for producer failures, then issue both review
    # roles as fresh, context-isolated calls for every valid proposal.
    review_jobs: list[tuple[str, str]] = []
    for qid, item in prepared.items():
        producer = stages_by_qid[qid]["producer"]
        if producer.semantic_status != "accepted":
            identity = item["identity"]
            failures[qid] = {
                "schema_version": SCHEMA_VERSION, "dataset": identity["dataset"], "qid": qid,
                "question_sha256": question_sha256(identity["question"]), "status": "producer_rejected",
                "reject_code": producer.reject_code, "detail_code": producer.detail_code,
            }
            for stage in STAGES[1:]:
                spec = _stage_spec(protocol, stage)
                stages_by_qid[qid][stage] = _skipped_result(
                    stage=stage, experiment_id=experiment_id, qid=qid,
                    requested_model=str(spec["model"]), detail="producer_upstream_failure",
                    prompt_template_sha256=str(spec["prompt"]["sha256"]),
                )
        else:
            review_jobs.extend((qid, stage) for stage in STAGES[1:])

    def review_job(qid: str, stage: str) -> StageResult:
        item = prepared[qid]
        proposal = stages_by_qid[qid]["producer"].parsed
        messages, payload, prompt_hash = _messages_and_payload(
            stage=stage, protocol=protocol, safe_payload=item["safe_payload"], proposal=proposal, chain=item["chain"]
        )
        spec = _stage_spec(protocol, stage)
        result = executor.call(
            stage=stage, qid=qid, stage_spec=spec, messages=messages,
            safe_payload_sha256=_canonical_sha256(payload), prompt_template_sha256=prompt_hash,
        )
        _reject_sensitive_serialization(result)
        _finalize_review(result, protocol=protocol, chain=item["chain"], nonces=item["nonces"])
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(review_job, qid, stage): (qid, stage) for qid, stage in review_jobs}
        for future in as_completed(futures):
            qid, stage = futures[future]
            stages_by_qid[qid][stage] = future.result()

    accepted_actions: list[dict[str, Any]] = []
    item_statuses: list[dict[str, Any]] = []
    for identity in identities:
        qid = str(identity["qid"])
        if qid in failures:
            item_statuses.append({**{key: failures[qid][key] for key in ("dataset", "qid", "question_sha256")}, "status": failures[qid]["status"], "reject_code": failures[qid]["reject_code"]})
            continue
        item = prepared[qid]
        r1, r2 = stages_by_qid[qid]["reviewer_1_blind"], stages_by_qid[qid]["reviewer_2_gold_aware"]
        if r1.semantic_status == "accepted" and r2.semantic_status == "accepted":
            proposal = stages_by_qid[qid]["producer"].parsed
            try:
                pair = silver.build_hotpot_action_pair(
                    item["chain"], proposal or {}, split="train",
                    extra_source_provenance={
                        "generation_experiment_id": experiment_id,
                        "producer_response_sha256": stages_by_qid[qid]["producer"].raw_response_sha256,
                        "blind_review_response_sha256": r1.raw_response_sha256,
                        "gold_aware_review_response_sha256": r2.raw_response_sha256,
                        "dual_review_unanimous_pass": True,
                    },
                )
                proposal_hash = silver.validate_query_proposal(proposal or {}, item["chain"]).proposal_sha256
                if any(row["source_provenance"]["proposal_sha256"] != proposal_hash for row in pair):
                    raise silver.HotpotSilverReject("accepted_action_proposal_hash_mismatch")
                accepted_actions.extend(pair)
                item_statuses.append({"dataset": identity["dataset"], "qid": qid, "question_sha256": question_sha256(identity["question"]), "status": "accepted_generation_and_dual_review", "reject_code": None})
            except silver.HotpotSilverReject as exc:
                failures[qid] = {
                    "schema_version": SCHEMA_VERSION, "dataset": identity["dataset"], "qid": qid,
                    "question_sha256": question_sha256(identity["question"]), "status": "action_pair_rejected",
                    "reject_code": "unanimous_review_failed", "detail_code": exc.code,
                }
                item_statuses.append({"dataset": identity["dataset"], "qid": qid, "question_sha256": question_sha256(identity["question"]), "status": "action_pair_rejected", "reject_code": "unanimous_review_failed"})
        else:
            if r1.semantic_status == "accepted" and r2.reject_code == "reviewer_2_reject":
                code = "reviewer_disagreement_or_unknown"
            elif r2.semantic_status == "accepted" and r1.reject_code == "reviewer_1_reject":
                code = "reviewer_disagreement_or_unknown"
            elif r1.reject_code == "reviewer_1_reject" and r2.reject_code == "reviewer_2_reject":
                code = "unanimous_review_failed"
            else:
                code = r1.reject_code or r2.reject_code or "unanimous_review_failed"
            detail = r1.detail_code if r1.semantic_status != "accepted" else r2.detail_code
            failures[qid] = {
                "schema_version": SCHEMA_VERSION, "dataset": identity["dataset"], "qid": qid,
                "question_sha256": question_sha256(identity["question"]), "status": "review_rejected",
                "reject_code": code, "detail_code": detail,
            }
            item_statuses.append({"dataset": identity["dataset"], "qid": qid, "question_sha256": question_sha256(identity["question"]), "status": "review_rejected", "reject_code": code})

    semantic_rows: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []
    role_rows: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGES}
    final_status_by_qid = {str(row["qid"]): row for row in item_statuses}
    for identity in identities:
        qid = str(identity["qid"])
        for stage in STAGES:
            result = stages_by_qid[qid][stage]
            semantic_rows.append(_ledger_row(result, experiment_id=experiment_id, identity=identity))
            transport_rows.extend(result.transport_rows or [])
            role_row = _role_row(identity, result)
            if stage == "producer":
                final_item = final_status_by_qid[qid]
                accepted = final_item["status"] == "accepted_generation_and_dual_review"
                proposal = result.parsed if accepted else None
                role_row.update(
                    {
                        "final_item_status": final_item["status"],
                        "dual_review_unanimous_pass": accepted,
                        "q1_query": proposal.get("q1") if proposal else None,
                        "q2_template": proposal.get("q2_template") if proposal else None,
                        "proposal_sha256": result.parsed_response_sha256 if accepted else None,
                        "runtime_projection_gold_or_observation_fields_present": False,
                    }
                )
            role_rows[stage].append(role_row)
    if len(semantic_rows) != expected_rows * 3 or len(item_statuses) != expected_rows:
        raise AssertionError("fixed-denominator conservation failed")
    counts_by_request = Counter(row["semantic_request_id"] for row in transport_rows)
    if any(row["transport_attempt_count"] != counts_by_request[row["semantic_request_id"]] for row in semantic_rows):
        raise AssertionError("semantic/transport ledger conservation failed")
    if any(tuple(row) != tuple(freeze.TRANSPORT_ATTEMPT_LEDGER_FIELDS) for row in transport_rows):
        raise AssertionError("transport ledger field drift")
    if any(row["nonce_echo_count"] != 0 for row in semantic_rows if row["status"] == "accepted"):
        raise AssertionError("accepted response echoed an opaque nonce")

    accepted_items = len(accepted_actions) // 2
    response_rows = [row for row in semantic_rows if row["raw_response_sha256"] is not None]
    all_response_nonces_clean = all(row["nonce_echo_count"] == 0 for row in response_rows)
    accepted_min_gate = accepted_items >= int(protocol["pilot_decision_gate_inherited"]["accepted_min"])
    generation_review_gate = accepted_min_gate and all_response_nonces_clean
    status = STATUS_COMPLETE if generation_review_gate else STATUS_FAIL
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at_utc": _utc_now(),
        "status": status,
        "fixed_denominator": expected_rows,
        "precall_rejected": sum(item["status"] == "precall_rejected" for item in item_statuses),
        "producer_accepted": sum(stages_by_qid[str(item["qid"])]["producer"].semantic_status == "accepted" for item in identities),
        "dual_review_accepted": accepted_items,
        "failed_items": len(failures),
        "accepted_action_rows": len(accepted_actions),
        "semantic_call_rows": len(semantic_rows),
        "transport_attempt_rows": len(transport_rows),
        "semantic_requests_executed": sum(row["semantic_attempt_count"] for row in semantic_rows),
        "physical_api_attempts": len(transport_rows),
        "reject_code_counts": dict(sorted(Counter(row["reject_code"] for row in failures.values()).items())),
        "item_statuses": item_statuses,
        "gates": {
            "generation_and_dual_review_accepted_min": int(protocol["pilot_decision_gate_inherited"]["accepted_min"]),
            "generation_and_dual_review_accepted_min_pass": accepted_min_gate,
            "generation_and_dual_review_gate_pass": generation_review_gate,
            "all_parent_identities_accounted_for": len(item_statuses) == expected_rows,
            "all_model_responses_nonce_echo_free": all_response_nonces_clean,
            "nonce_echo_response_count": sum(row["nonce_echo_count"] > 0 for row in response_rows),
            "retrieval_support_gate": "NOT_RUN",
            "full_pilot_release_decision": "NOT_YET_AVAILABLE_RETRIEVAL_GATE_PENDING",
        },
        "scientific_boundary": {
            "labels_are_gold_screened_silver": True,
            "reviewer_2_is_train_gold_aware": True,
            "reviewers_statistically_independent_claimed": False,
            "retrieval_or_reader_calls": 0,
            "training_started": False,
            "em_f1_ihr_evaluated": False,
        },
    }

    paths["output"].mkdir(parents=True, exist_ok=False)
    for stage, filename in ROLE_FILES.items():
        _write_jsonl(paths["output"] / filename, role_rows[stage])
    _write_jsonl(paths["output"] / "accepted_actions.jsonl", accepted_actions)
    _write_jsonl(paths["output"] / "failures.jsonl", [failures[str(row["qid"])] for row in identities if str(row["qid"]) in failures])
    _write_jsonl(paths["output"] / "semantic_call_ledger.jsonl", semantic_rows)
    _write_jsonl(paths["output"] / "api_transport_attempt_ledger.jsonl", transport_rows)
    _write_json(paths["output"] / "report.json", report)

    implementation_paths = (
        Path("scripts/prepare/generate_hotpot_controller_silver_pilot_v1.py"),
        Path("scripts/prepare/freeze_hotpot_controller_silver_execution_v1.py"),
        Path("kgproweight/data/hotpot_controller_silver.py"),
    )
    outputs = []
    for filename in ALL_OUTPUT_FILES[:-1]:
        output_path = paths["output"] / filename
        outputs.append({"path": filename, "sha256": _sha256_file(output_path), "size_bytes": output_path.stat().st_size})
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at_utc": _utc_now(),
        "status": status,
        "inputs": {
            "execution_protocol": {"path": _display(project_root, paths["protocol"]), "sha256": protocol_hash},
            "parent_identity": {"path": _display(project_root, paths["identity"]), "sha256": _sha256_file(paths["identity"])},
            "parent_metadata_addendum_v1_1": {"path": _display(project_root, paths["addendum"]), "sha256": addendum_hash},
            "raw_train": {"path": _display(project_root, paths["raw"]), "sha256": raw_hash},
        },
        "implementations": [
            {"path": _display(project_root, _resolve(project_root, path)), "sha256": _sha256_file(_resolve(project_root, path))}
            for path in implementation_paths
        ],
        "outputs": outputs,
        "fixed_denominator": expected_rows,
        "api_semantic_requests": sum(row["semantic_attempt_count"] for row in semantic_rows),
        "api_physical_attempts": len(transport_rows),
        "credential_or_endpoint_values_serialized": False,
        "environment_variable_names_only": ["OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_BASE_URL"],
        "retrieval_calls": 0,
        "training_started": False,
    }
    _write_json(paths["output"] / "manifest.json", manifest)
    if set(path.name for path in paths["output"].iterdir()) != set(ALL_OUTPUT_FILES):
        raise AssertionError("runner did not materialise exactly the nine frozen files")
    return {"report": report, "manifest": manifest, "output_dir": paths["output"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--metadata_addendum", type=Path, default=DEFAULT_ADDENDUM)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--execute_api", action="store_true",
        help="Required safety latch. Without it the command makes no call and writes nothing.",
    )
    args = parser.parse_args()
    if not args.execute_api:
        raise SystemExit("No API call made. Re-run with --execute_api after reviewing the frozen protocol.")
    result = run_generation_pilot(
        client=OpenAICompletionClient(), protocol_path=args.protocol, identity_path=args.identity,
        metadata_addendum_path=args.metadata_addendum, raw_path=args.raw, output_dir=args.output_dir,
    )
    print(json.dumps({"output_dir": str(result["output_dir"]), "status": result["report"]["status"], "dual_review_accepted": result["report"]["dual_review_accepted"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
