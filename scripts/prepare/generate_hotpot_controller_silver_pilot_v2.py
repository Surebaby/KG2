#!/usr/bin/env python
"""Execute the hardened V2 HotpotQA pilot30 silver-generation protocol.

The CLI is non-executing unless ``--execute_api`` is supplied.  Before the
first physical provider attempt, the runner creates an append-only output
directory and fsyncs an ``intent`` event to ``api_call_wal.jsonl``.  Every
return or classified exception receives a second fsynced ``result`` event.
Thus an interruption remains visible and the existing directory prevents an
unintentional repeat of paid semantic calls.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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
from typing import Any
import unicodedata
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgproweight.data import hotpot_controller_silver as silver  # noqa: E402
from kgproweight.kg.question_kg import question_sha256  # noqa: E402
from scripts.prepare import freeze_hotpot_controller_silver_execution_v1 as freeze_v1  # noqa: E402
from scripts.prepare import freeze_hotpot_controller_silver_execution_v2 as freeze_v2  # noqa: E402
from scripts.prepare import generate_hotpot_controller_silver_pilot_v1 as runner_v1  # noqa: E402


SCHEMA_VERSION = "hotpot-controller-silver-generation-run-2"
REPORT_SCHEMA_VERSION = "hotpot-controller-silver-generation-report-2"
MANIFEST_SCHEMA_VERSION = "hotpot-controller-silver-generation-manifest-2"
STATUS_COMPLETE = "COMPLETE_V2_GENERATION_DUAL_REVIEW_RETRIEVAL_NOT_RUN_NOT_TRAINED"
STATUS_FAIL = "FAIL_V2_GENERATION_OR_REVIEW_GATE_RETRIEVAL_NOT_RUN_NOT_TRAINED"

DEFAULT_PROTOCOL = freeze_v2.DEFAULT_OUTPUT_DIR / "protocol.json"
DEFAULT_IMPLEMENTATION_LOCK = freeze_v2.DEFAULT_OUTPUT_DIR / "implementation_lock.json"
DEFAULT_PROTOCOL_REPORT = freeze_v2.DEFAULT_OUTPUT_DIR / "report.json"
DEFAULT_PROTOCOL_MANIFEST = freeze_v2.DEFAULT_OUTPUT_DIR / "manifest.json"
DEFAULT_V1_SUPERSESSION_ADDENDUM = freeze_v2.V1_SUPERSESSION_ADDENDUM
DEFAULT_IDENTITY = runner_v1.DEFAULT_IDENTITY
DEFAULT_ADDENDUM = runner_v1.DEFAULT_ADDENDUM
DEFAULT_RAW = runner_v1.DEFAULT_RAW
DEFAULT_OUTPUT_DIR = freeze_v2.DEFAULT_GENERATION_OUTPUT_DIR

ROLE_FILES = dict(runner_v1.ROLE_FILES)
STAGES = runner_v1.STAGES
ALL_OUTPUT_FILES = freeze_v2.V2_REQUIRED_OUTPUT_FILES
_NORMALIZED_NONCE_RE = re.compile(r"(?<![0-9a-z])n[0-9a-f]{31}(?![0-9a-z])")
_UNICODE_ESCAPE_RE = re.compile(
    r"(?:\\u[dD][89aAbB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2}|"
    r"\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})"
)
_CREDENTIAL_LIKE_RE = re.compile(r"(?<![0-9a-z])sk-[0-9a-z_-]{8,}(?![0-9a-z])")


def validate_deepseek_endpoint(value: str) -> None:
    """Accept only the frozen HTTPS DeepSeek API origin (optionally /v1)."""

    parsed = urlparse(value)
    if (
        parsed.scheme.casefold() != "https"
        or (parsed.hostname or "").casefold() != "api.deepseek.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.path not in ("", "/", "/v1", "/v1/")
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("OPENAI_BASE_URL is outside the frozen DeepSeek HTTPS allowlist")


class OpenAICompletionClientV2:
    """Fail-closed OpenAI-compatible client with no serialisable endpoint field."""

    def __init__(self) -> None:
        from openai import OpenAI

        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY or OPENAI_API_KEY is required")
        base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.deepseek.com"
        validate_deepseek_endpoint(base_url)
        # Disable the SDK's hidden transport retries: every physical attempt
        # must be visible as a WAL intent/result pair in _V2ApiExecutor.
        self.__client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    def complete(self, request_body: Mapping[str, Any], *, timeout: float) -> Any:
        return self.__client.chat.completions.create(**dict(request_body), timeout=timeout)


class DuplicateJSONKeyError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return freeze_v2._sha256_file(path)


def _resolve(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display(project_root: Path, path: Path) -> str:
    return freeze_v2._display(project_root, path)


def build_producer_safe_payload_v2(
    masked_view: Mapping[str, Any], *, chain_sha256: str, experiment_id: str
) -> dict[str, str]:
    payload = freeze_v1.build_producer_safe_payload(
        masked_view, chain_sha256=chain_sha256, experiment_id=experiment_id
    )
    first_nonce, _ = freeze_v1.derive_mask_nonces(
        chain_sha256=chain_sha256, experiment_id=experiment_id
    )
    payload["second_hop_subject_nonce"] = first_nonce
    expected = (
        "original_question",
        "root_document_title",
        "first_hop_evidence_masked",
        "second_hop_evidence_masked",
        "second_hop_subject_nonce",
    )
    if tuple(payload) != expected:
        raise AssertionError("V2 producer safe-payload field/order drift")
    if payload["second_hop_subject_nonce"] != first_nonce:
        raise AssertionError("V2 second-hop-subject binding drift")
    # The evidence may use an implicit subject (e.g. a pronoun); the separate
    # field is precisely what removes that V1 ambiguity.
    return payload


def _decode_one_unicode_escape(token: str) -> str:
    values = re.findall(r"[0-9a-fA-F]{4,8}", token)
    if len(values) == 2:
        high, low = (int(value, 16) for value in values)
        if 0xD800 <= high <= 0xDBFF and 0xDC00 <= low <= 0xDFFF:
            return chr(0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))
    if len(values) == 1:
        codepoint = int(values[0], 16)
        if codepoint <= 0x10FFFF and not 0xD800 <= codepoint <= 0xDFFF:
            return chr(codepoint)
    # Preserve malformed/lone-surrogate material for a deterministic reject
    # path rather than introducing an unencodable surrogate into audit files.
    return "\N{REPLACEMENT CHARACTER}"


def security_normalize_text(value: str) -> str:
    """Boundedly expose Unicode escapes, then apply NFKC and case folding."""

    expanded = value
    for _ in range(4):
        replaced = _UNICODE_ESCAPE_RE.sub(
            lambda match: _decode_one_unicode_escape(match.group(0)), expanded
        )
        if replaced == expanded:
            break
        expanded = replaced
    return unicodedata.normalize("NFKC", expanded).casefold()


def normalized_nonce_like_tokens(value: object) -> list[str]:
    text = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=True, sort_keys=True
    )
    return _NORMALIZED_NONCE_RE.findall(security_normalize_text(text))


def strict_json_object(content: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateJSONKeyError(key)
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite_json_constant:{value}")

    value = json.loads(content, object_pairs_hook=reject_duplicates, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError("top_level_not_object")
    return value


def _field(obj: Any, key: str, default: Any = None) -> Any:
    return runner_v1._field(obj, key, default)


def _provider_response_tree(
    value: Any, *, _seen: set[int] | None = None, _depth: int = 0
) -> Any:
    """Convert an SDK response to a bounded JSON-like tree for full text scan."""

    if _depth > 32:
        raise ValueError("provider_response_nesting_too_deep")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    seen = _seen if _seen is not None else set()
    object_id = id(value)
    if object_id in seen:
        raise ValueError("provider_response_cycle")
    seen.add(object_id)
    try:
        if isinstance(value, Mapping):
            return {
                str(key): _provider_response_tree(item, _seen=seen, _depth=_depth + 1)
                for key, item in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [
                _provider_response_tree(item, _seen=seen, _depth=_depth + 1)
                for item in value
            ]
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return _provider_response_tree(
                model_dump(mode="python"), _seen=seen, _depth=_depth + 1
            )
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return _provider_response_tree(
                to_dict(), _seen=seen, _depth=_depth + 1
            )
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, Mapping):
            public = {
                str(key): item
                for key, item in attributes.items()
                if not str(key).startswith("_")
            }
            return _provider_response_tree(public, _seen=seen, _depth=_depth + 1)
    finally:
        seen.discard(object_id)
    raise TypeError(f"unsupported_provider_response_type:{type(value).__name__}")


def _collect_string_values(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, str):
        texts.append(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            texts.extend(_collect_string_values(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            texts.extend(_collect_string_values(item))
    return texts


def _response_parts(response: Any) -> dict[str, Any]:
    tree = _provider_response_tree(response)
    if not isinstance(tree, Mapping):
        raise TypeError("provider_response_top_level_not_object")
    choices_value = tree.get("choices")
    choices = list(choices_value) if isinstance(choices_value, Sequence) and not isinstance(choices_value, (str, bytes)) else []
    texts = [text for text in _collect_string_values(tree) if text]
    choice = choices[0] if len(choices) == 1 else None
    message = choice.get("message") if isinstance(choice, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    usage = tree.get("usage")
    completion_details = usage.get("completion_tokens_details") if isinstance(usage, Mapping) else None
    finish_reasons = [
        item.get("finish_reason") if isinstance(item, Mapping) else None for item in choices
    ]
    return {
        "choices_count": len(choices),
        "all_texts": texts,
        "content": content if isinstance(content, str) else None,
        "model": str(tree.get("model") or ""),
        "http_status": tree.get("http_status", 200),
        "finish_reason": finish_reasons[0] if len(finish_reasons) == 1 else None,
        "finish_reasons": finish_reasons,
        "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, Mapping) else None,
        "completion_tokens": usage.get("completion_tokens") if isinstance(usage, Mapping) else None,
        "reasoning_tokens": completion_details.get("reasoning_tokens") if isinstance(completion_details, Mapping) else None,
        "total_tokens": usage.get("total_tokens") if isinstance(usage, Mapping) else None,
    }


def _contains_forbidden_text(
    texts: Sequence[str], forbidden_chain_secrets: Sequence[str]
) -> tuple[str | None, list[str]]:
    normalized = [security_normalize_text(text) for text in texts]
    nonce_tokens = [
        token for text in normalized for token in _NORMALIZED_NONCE_RE.findall(text)
    ]
    if nonce_tokens:
        return "normalized_nonce_like_token_echo_omitted", nonce_tokens
    for secret in forbidden_chain_secrets:
        normalized_secret = security_normalize_text(str(secret)).strip()
        if normalized_secret and any(
            silver._contains_secret(text, normalized_secret)
            for text in normalized
        ):
            return "forbidden_chain_secret_echo_omitted", []
    configured = [
        value
        for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_BASE_URL")
        if (value := os.environ.get(name)) and len(value) >= 6
    ]
    configured.extend(("https://api.deepseek.com", "api.deepseek.com"))
    if any(
        security_normalize_text(secret) in text
        for secret in configured
        for text in normalized
    ) or any(_CREDENTIAL_LIKE_RE.search(text) for text in normalized):
        return "credential_or_endpoint_echo_omitted", []
    return None, []


def _safe_wal_metadata_text(value: Any) -> str | None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", value):
        return None
    normalized = security_normalize_text(value)
    if _NORMALIZED_NONCE_RE.search(normalized) or _CREDENTIAL_LIKE_RE.search(normalized):
        return None
    if "api.deepseek.com" in normalized:
        return None
    return value


def _safe_token_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _response_capture_summary(
    parts: Mapping[str, Any], *, requested_model: str
) -> dict[str, Any]:
    texts = list(parts.get("all_texts") or [])
    text_hash = hashlib.sha256(
        json.dumps(texts, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    actual_model = parts.get("model") if isinstance(parts.get("model"), str) else ""
    finish_reasons = list(parts.get("finish_reasons") or [])
    return {
        "response_text_fields_sha256": text_hash,
        # Provider-controlled strings are never copied unless they exactly
        # equal a frozen request value / protocol enum.  Hashes retain audit
        # identity without persisting a possible secret echoed in metadata.
        "response_model": requested_model if actual_model == requested_model else None,
        "response_model_sha256": _sha256_text(actual_model),
        "response_model_matches_requested": actual_model == requested_model,
        "choices_count": parts.get("choices_count"),
        "finish_reasons": [
            value if value in {"stop", "length", "content_filter", "tool_calls"} else None
            for value in finish_reasons
        ],
        "finish_reasons_sha256": hashlib.sha256(
            json.dumps(finish_reasons, ensure_ascii=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }


class _WriteAheadLedger:
    def __init__(self, path: Path, *, experiment_id: str) -> None:
        self.path = path
        self.experiment_id = experiment_id
        self._lock = threading.Lock()
        self._handle = path.open("x", encoding="utf-8")
        self._intents: Counter[tuple[str, int]] = Counter()
        self._capture_attempts: Counter[tuple[str, int]] = Counter()
        self._captures: Counter[tuple[str, int]] = Counter()
        self._results: Counter[tuple[str, int]] = Counter()

    def append(self, row: Mapping[str, Any]) -> None:
        payload = {"schema_version": "hotpot-controller-api-wal-event-1", "experiment_id": self.experiment_id, **dict(row)}
        with self._lock:
            self._handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())
            request_id = row.get("semantic_request_id")
            attempt = row.get("physical_attempt_index")
            if isinstance(request_id, str) and isinstance(attempt, int):
                key = (request_id, attempt)
                if row.get("event") == "intent":
                    self._intents[key] += 1
                elif row.get("event") == "response_captured":
                    self._capture_attempts[key] += 1
                    if row.get("capture_complete") is True:
                        self._captures[key] += 1
                elif row.get("event") == "result":
                    self._results[key] += 1

    def unmatched_intent_count(self) -> int:
        with self._lock:
            return sum((self._intents - self._results).values())

    def interruption_boundary(self) -> dict[str, Any]:
        with self._lock:
            unresolved = self._intents - self._results
            unknown_after_dispatch = sum(
                max(count - self._capture_attempts[key], 0)
                for key, count in unresolved.items()
            )
            capture_incomplete = sum(
                min(count, self._capture_attempts[key])
                - min(count, self._captures[key])
                for key, count in unresolved.items()
            )
            response_captured_locally = sum(
                min(count, self._captures[key]) for key, count in unresolved.items()
            )
            if unknown_after_dispatch:
                boundary = "UNKNOWN_AFTER_DISPATCH"
            elif capture_incomplete:
                boundary = "PROVIDER_RETURNED_CAPTURE_INCOMPLETE"
            elif response_captured_locally:
                boundary = "RESPONSE_CAPTURED_LOCALLY_NOT_RESULT_RECORDED"
            else:
                boundary = "NO_UNMATCHED_DISPATCH"
            return {
                "unmatched_intent_count": sum(unresolved.values()),
                "unknown_after_dispatch_count": unknown_after_dispatch,
                "provider_returned_capture_incomplete_count": capture_incomplete,
                "response_captured_without_result_count": response_captured_locally,
                "provider_outcome_boundary": boundary,
            }

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()
                os.fsync(self._handle.fileno())
                self._handle.close()


class _V2ApiExecutor:
    def __init__(
        self,
        client: runner_v1.CompletionClient,
        protocol: Mapping[str, Any],
        wal: _WriteAheadLedger,
        stop_event: threading.Event,
    ) -> None:
        api = protocol["api_execution"]
        self.client = client
        self.protocol = protocol
        self.experiment_id = str(protocol["experiment_id"])
        self.timeout = float(api["timeout_seconds_per_physical_attempt"])
        self.max_attempts = int(api["max_physical_attempts_per_semantic_request"])
        self.delay = float(api["minimum_inter_request_delay_seconds_per_worker"])
        self.wal = wal
        self.stop_event = stop_event
        self._thread_state = threading.local()

    def _pace(self) -> None:
        last = getattr(self._thread_state, "last_request", None)
        if last is not None:
            remaining = self.delay - (time.monotonic() - last)
            if remaining > 0:
                time.sleep(remaining)
        self._thread_state.last_request = time.monotonic()

    @staticmethod
    def _prefix(stage: str) -> str:
        return "producer" if stage == "producer" else ("reviewer_1" if stage == "reviewer_1_blind" else "reviewer_2")

    def call(
        self,
        *,
        stage: str,
        qid: str,
        stage_spec: Mapping[str, Any],
        messages: list[dict[str, str]],
        safe_payload_sha256: str,
        prompt_template_sha256: str,
        forbidden_chain_secrets: Sequence[str],
    ) -> runner_v1.StageResult:
        request_id = _sha256_text(f"{self.experiment_id}\0{qid}\0{stage}")
        body = runner_v1._request_body(stage_spec, messages)
        request_bytes = _canonical_bytes(body)
        request_hash = hashlib.sha256(request_bytes).hexdigest()
        transport_rows: list[dict[str, Any]] = []
        response: Any = None
        response_parts: dict[str, Any] | None = None
        last_kind: str | None = None
        semantic_started = time.monotonic()
        for attempt in range(1, self.max_attempts + 1):
            if self.stop_event.is_set():
                raise RuntimeError("peer_baseexception_cancelled_before_dispatch")
            self._pace()
            if self.stop_event.is_set():
                raise RuntimeError("peer_baseexception_cancelled_before_dispatch")
            started_at = _utc_now()
            self.wal.append(
                {
                    "event": "intent",
                    "semantic_request_id": request_id,
                    "stage": stage,
                    "physical_attempt_index": attempt,
                    "request_body_sha256": request_hash,
                    "at_utc": started_at,
                }
            )
            started = time.monotonic()
            status: int | None = None
            kind: str | None = None
            retryable = False
            received = False
            try:
                response = self.client.complete(
                    json.loads(request_bytes.decode("utf-8")), timeout=self.timeout
                )
            except Exception as exc:  # noqa: BLE001 - only safe class is retained
                kind, status, retryable = runner_v1._transport_class(exc)
                last_kind = kind
            else:
                received = True
                try:
                    response_parts = _response_parts(response)
                    capture = _response_capture_summary(
                        response_parts, requested_model=str(stage_spec["model"])
                    )
                except BaseException as exc:
                    # The provider returned, but safe extraction did not finish.
                    # Persist that exact boundary without any response body.
                    self.wal.append(
                        {
                            "event": "response_captured",
                            "semantic_request_id": request_id,
                            "stage": stage,
                            "physical_attempt_index": attempt,
                            "request_body_sha256": request_hash,
                            "at_utc": _utc_now(),
                            "capture_complete": False,
                            "capture_error_type": type(exc).__name__,
                            "response_text_fields_sha256": None,
                            "response_model": None,
                            "choices_count": None,
                            "finish_reasons": [],
                        }
                    )
                    raise
                self.wal.append(
                    {
                        "event": "response_captured",
                        "semantic_request_id": request_id,
                        "stage": stage,
                        "physical_attempt_index": attempt,
                        "request_body_sha256": request_hash,
                        "at_utc": _utc_now(),
                        "capture_complete": True,
                        **capture,
                    }
                )
                status_value = response_parts["http_status"]
                status = int(status_value) if isinstance(status_value, int) else 200
            ended_at = _utc_now()
            elapsed = round((time.monotonic() - started) * 1000)
            self.wal.append(
                {
                    "event": "result",
                    "semantic_request_id": request_id,
                    "stage": stage,
                    "physical_attempt_index": attempt,
                    "request_body_sha256": request_hash,
                    "at_utc": ended_at,
                    "response_received": received,
                    "http_status": status,
                    "transport_error_class": kind,
                    "transport_retryable": retryable,
                }
            )
            transport_rows.append(
                {
                    "experiment_id": self.experiment_id,
                    "semantic_request_id": request_id,
                    "stage": stage,
                    "physical_attempt_index": attempt,
                    "request_body_sha256": request_hash,
                    "request_bytes_identical_to_attempt_1": True,
                    "started_at_utc": started_at,
                    "ended_at_utc": ended_at,
                    "wall_time_ms": elapsed,
                    "http_status": status,
                    "transport_error_class": kind,
                    "transport_retryable": retryable,
                    "response_received": received,
                }
            )
            if received or not retryable:
                break
        result = runner_v1.StageResult(
            stage=stage,
            semantic_request_id=request_id,
            requested_model=str(stage_spec["model"]),
            messages_sha256=_canonical_sha256(messages),
            safe_payload_sha256=safe_payload_sha256,
            prompt_template_sha256=prompt_template_sha256,
            semantic_status="response_received" if response is not None else "transport_exhausted",
            wall_time_ms=round((time.monotonic() - semantic_started) * 1000),
            transport_rows=transport_rows,
        )
        prefix = self._prefix(stage)
        if response is None:
            result.reject_code = f"{prefix}_transport_exhausted"
            result.detail_code = last_kind
            return result

        if response_parts is None:
            raise AssertionError("provider response was not safely captured")
        parts = response_parts
        all_texts = parts["all_texts"]
        result.raw_response_sha256 = (
            hashlib.sha256(
                json.dumps(
                    {"response_text_fields": all_texts},
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if all_texts
            else None
        )

        # Frozen order: scan every textual response field before inspecting the
        # model id, credentials, cardinality, finish reason, or JSON content.
        leak_detail, tokens = _contains_forbidden_text(
            all_texts, forbidden_chain_secrets
        )
        result.nonce_echo_count = len(tokens)
        if leak_detail is not None:
            result.raw_content = None
            result.semantic_status = "rejected"
            result.reject_code = (
                f"{prefix}_nonce_echo"
                if leak_detail == "normalized_nonce_like_token_echo_omitted"
                else f"{prefix}_output_schema_error"
            )
            result.detail_code = leak_detail
            return result
        actual_model = parts["model"]
        actual_finish_reason = parts["finish_reason"]
        # Only security-cleared primary content and strictly projected metadata
        # may enter semantic/role ledgers.  Mismatching provider-controlled
        # model strings are represented by null plus a fixed detail code.
        result.raw_content = parts["content"]
        result.response_model = (
            result.requested_model if actual_model == result.requested_model else None
        )
        result.finish_reason = (
            actual_finish_reason
            if actual_finish_reason in {"stop", "length", "content_filter", "tool_calls"}
            else None
        )
        result.prompt_tokens = _safe_token_count(parts["prompt_tokens"])
        result.completion_tokens = _safe_token_count(parts["completion_tokens"])
        result.reasoning_tokens = _safe_token_count(parts["reasoning_tokens"])
        result.total_tokens = _safe_token_count(parts["total_tokens"])
        if actual_model != result.requested_model:
            result.raw_content = None
            result.semantic_status = "rejected"
            result.reject_code = f"{prefix}_response_model_mismatch"
            result.detail_code = "response_model_mismatch"
            return result
        if parts["choices_count"] != 1:
            result.raw_content = None
            result.semantic_status = "rejected"
            result.reject_code = f"{prefix}_output_schema_error"
            result.detail_code = "choices_cardinality_not_one"
            return result
        if actual_finish_reason != "stop":
            result.raw_content = None
            result.semantic_status = "rejected"
            result.reject_code = f"{prefix}_output_schema_error"
            result.detail_code = "finish_reason_not_stop"
            return result
        if not isinstance(result.raw_content, str) or not result.raw_content.strip():
            result.semantic_status = "rejected"
            result.reject_code = f"{prefix}_missing_response"
            result.detail_code = "empty_or_missing_content"
            return result
        result.semantic_status = "response_validated"
        return result


def _finalize_producer(result: runner_v1.StageResult, chain: silver.HotpotSupportChain) -> None:
    if result.semantic_status != "response_validated" or result.raw_content is None:
        return
    try:
        parsed = strict_json_object(result.raw_content)
    except DuplicateJSONKeyError:
        result.semantic_status = "rejected"
        result.reject_code = "producer_json_parse_error"
        result.detail_code = "duplicate_json_key"
        return
    except (json.JSONDecodeError, TypeError, ValueError):
        result.semantic_status = "rejected"
        result.reject_code = "producer_json_parse_error"
        result.detail_code = "strict_json_object_required"
        return
    result.parsed_response_sha256 = _canonical_sha256(parsed)
    try:
        runner_v1._validate_producer_schema(parsed)
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
    result.parsed = {"schema_version": silver.PROPOSAL_SCHEMA_VERSION, "q1": validated.q1_query, "q2_template": validated.q2_template}
    result.parsed_response_sha256 = validated.proposal_sha256
    result.semantic_status = "accepted"


def _finalize_review(result: runner_v1.StageResult, *, protocol: Mapping[str, Any], chain: silver.HotpotSupportChain) -> None:
    if result.semantic_status != "response_validated" or result.raw_content is None:
        return
    prefix = "reviewer_1" if result.stage == "reviewer_1_blind" else "reviewer_2"
    try:
        parsed = strict_json_object(result.raw_content)
    except DuplicateJSONKeyError:
        result.semantic_status = "rejected"
        result.reject_code = f"{prefix}_json_parse_error"
        result.detail_code = "duplicate_json_key"
        return
    except (json.JSONDecodeError, TypeError, ValueError):
        result.semantic_status = "rejected"
        result.reject_code = f"{prefix}_json_parse_error"
        result.detail_code = "strict_json_object_required"
        return
    result.parsed_response_sha256 = _canonical_sha256(parsed)
    try:
        runner_v1._validate_review_schema(parsed, protocol)
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


def _verify_protocol_and_lock(
    *,
    project_root: Path,
    protocol_path: Path,
    implementation_lock_path: Path,
    protocol_report_path: Path,
    protocol_manifest_path: Path,
    v1_supersession_addendum_path: Path,
    identity_path: Path,
    raw_path: Path,
    parent_metadata_addendum_path: Path,
    expected_rows: int,
    enforce_formal_locks: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = runner_v1._load_json(protocol_path)
    lock = runner_v1._load_json(implementation_lock_path)
    report = runner_v1._load_json(protocol_report_path)
    manifest = runner_v1._load_json(protocol_manifest_path)
    freeze_paths = {
        "implementation_lock.json": implementation_lock_path,
        "protocol.json": protocol_path,
        "report.json": protocol_report_path,
        "manifest.json": protocol_manifest_path,
    }
    if any(path.name != name for name, path in freeze_paths.items()):
        raise ValueError("V2 protocol-freeze filename drift")
    parents = {path.resolve().parent for path in freeze_paths.values()}
    if len(parents) != 1:
        raise ValueError("V2 protocol-freeze files are not in one atomic directory")
    freeze_dir = next(iter(parents))
    if {path.name for path in freeze_dir.iterdir()} != set(
        freeze_v2.PROTOCOL_FREEZE_OUTPUT_FILES
    ):
        raise ValueError("V2 protocol-freeze actual output set drift")
    if protocol.get("schema_version") != freeze_v2.SCHEMA_VERSION or protocol.get("experiment_id") != freeze_v2.EXPERIMENT_ID or protocol.get("status") != freeze_v2.STATUS:
        raise ValueError("V2 execution protocol identity/status drift")
    if (
        report.get("schema_version") != freeze_v2.REPORT_SCHEMA_VERSION
        or report.get("experiment_id") != freeze_v2.EXPERIMENT_ID
        or report.get("status") != freeze_v2.STATUS
        or report.get("api_calls") != 0
    ):
        raise ValueError("V2 protocol-freeze report identity/status drift")
    if (
        manifest.get("schema_version") != freeze_v2.MANIFEST_SCHEMA_VERSION
        or manifest.get("experiment_id") != freeze_v2.EXPERIMENT_ID
        or manifest.get("status") != freeze_v2.STATUS
        or manifest.get("api_calls") != 0
        or manifest.get("training_started") is not False
    ):
        raise ValueError("V2 protocol-freeze manifest identity/status drift")
    if manifest.get("protocol_freeze_output_set_exact") != list(
        freeze_v2.PROTOCOL_FREEZE_OUTPUT_FILES
    ):
        raise ValueError("V2 protocol-freeze declared output set drift")
    external = manifest.get("external_append_only_artifact") or {}
    if (
        _resolve(project_root, Path(str(external.get("path") or ""))).resolve()
        != v1_supersession_addendum_path.resolve()
        or external.get("commit_order")
        != "after_complete_v2_manifest_directory_commit"
        or external.get("status_at_v2_manifest_commit")
        != "PENDING_APPEND_ONLY_COMMIT"
        or external.get("must_bind_v2_manifest_sha256") is not True
    ):
        raise ValueError("V2 supersession-addendum declaration drift")
    output_rows = manifest.get("outputs")
    if not isinstance(output_rows, list):
        raise ValueError("V2 protocol-freeze manifest outputs missing")
    expected_hashed = {
        "implementation_lock.json", "protocol.json", "report.json"
    }
    if {
        row.get("path") for row in output_rows if isinstance(row, Mapping)
    } != expected_hashed or len(output_rows) != len(expected_hashed):
        raise ValueError("V2 protocol-freeze manifest hashed output set drift")
    for row in output_rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "size_bytes"}:
            raise ValueError("V2 protocol-freeze manifest output schema drift")
        path = freeze_paths[str(row["path"])]
        if _sha256_file(path) != row.get("sha256") or path.stat().st_size != row.get("size_bytes"):
            raise ValueError(f"V2 protocol-freeze output hash/size drift: {row.get('path')}")
    supersession = runner_v1._load_json(v1_supersession_addendum_path)
    superseded_by = supersession.get("superseded_by") or {}
    if (
        supersession.get("schema_version") != freeze_v2.SUPERSESSION_SCHEMA_VERSION
        or supersession.get("v1_experiment_id") != freeze_v1.EXPERIMENT_ID
        or supersession.get("status") != "SUPERSEDED_BEFORE_ANY_API_CALL"
        or supersession.get("v1_api_calls") != 0
        or supersession.get("v1_generation_output_existed_at_supersession") is not False
        or superseded_by.get("experiment_id") != freeze_v2.EXPERIMENT_ID
        or _resolve(project_root, Path(str(superseded_by.get("protocol_path") or ""))).resolve()
        != protocol_path.resolve()
        or _resolve(project_root, Path(str(superseded_by.get("manifest_path") or ""))).resolve()
        != protocol_manifest_path.resolve()
        or superseded_by.get("protocol_sha256") != _sha256_file(protocol_path)
        or superseded_by.get("manifest_sha256") != _sha256_file(protocol_manifest_path)
    ):
        raise ValueError("V1-to-V2 supersession addendum binding drift")
    if (protocol.get("parent_identity_freeze") or {}).get("rows") != expected_rows:
        raise ValueError("V2 parent denominator drift")
    identity_hash = _sha256_file(identity_path)
    if (protocol.get("parent_identity_freeze") or {}).get("identity_sha256") != identity_hash:
        raise ValueError("V2 protocol parent identity hash drift")
    diagnostic_inputs = manifest.get("diagnostic_inputs") or {}
    expected_diagnostic_inputs = {
        "identity": identity_path,
        "raw_train": raw_path,
        "parent_metadata_addendum_v1_1": parent_metadata_addendum_path,
    }
    for name, path in expected_diagnostic_inputs.items():
        row = diagnostic_inputs.get(name) or {}
        if (
            _resolve(project_root, Path(str(row.get("path") or ""))).resolve()
            != path.resolve()
            or row.get("sha256") != _sha256_file(path)
        ):
            raise ValueError(f"V2 manifest diagnostic input drift: {name}")
    binding = protocol.get("implementation_lock") or {}
    if (
        _sha256_file(implementation_lock_path) != binding.get("sha256")
        or implementation_lock_path.stat().st_size != binding.get("size_bytes")
        or _resolve(project_root, Path(str(binding.get("path") or ""))).resolve()
        != implementation_lock_path.resolve()
    ):
        raise ValueError("V2 implementation-lock SHA256 drift")
    if lock.get("schema_version") != freeze_v2.LOCK_SCHEMA_VERSION or lock.get("experiment_id") != freeze_v2.EXPERIMENT_ID:
        raise ValueError("V2 implementation-lock identity drift")
    implementations = lock.get("implementations")
    if not isinstance(implementations, list) or not implementations:
        raise ValueError("V2 implementation-lock entries missing")
    seen_implementations: set[str] = set()
    for item in implementations:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256", "size_bytes"}:
            raise ValueError("V2 implementation-lock entry schema drift")
        if str(item["path"]) in seen_implementations:
            raise ValueError("V2 implementation-lock duplicate path")
        seen_implementations.add(str(item["path"]))
        path = _resolve(project_root, Path(str(item.get("path") or "")))
        if _sha256_file(path) != item.get("sha256") or path.stat().st_size != item.get("size_bytes"):
            raise ValueError(f"V2 implementation SHA256 drift: {item.get('path')}")
    if enforce_formal_locks:
        expected_implementation_paths = {
            _display(project_root, _resolve(project_root, path))
            for path in freeze_v2.IMPLEMENTATION_PATHS
        }
        if seen_implementations != expected_implementation_paths:
            raise ValueError("formal V2 implementation-lock path set drift")
    expected_fields = (
        "original_question", "root_document_title", "first_hop_evidence_masked",
        "second_hop_evidence_masked", "second_hop_subject_nonce",
    )
    if tuple(protocol["masking"]["producer_safe_payload_fields_exact"]) != expected_fields:
        raise ValueError("V2 safe-payload schema drift")
    response = protocol.get("response_validation_v2") or {}
    if (
        response.get("choices_count_exact") != 1
        or response.get("finish_reason_exact") != "stop"
        or response.get("json_duplicate_keys_allowed") is not False
        or response.get("leak_rejection_omits_primary_raw_content") is not True
    ):
        raise ValueError("V2 response-validation contract drift")
    api = protocol.get("api_execution") or {}
    scheduler = api.get("scheduler") or {}
    if (
        not isinstance(api.get("worker_count"), int)
        or api["worker_count"] < 1
        or api.get("wal_response_capture_fsync_immediately_after_provider_return") is not True
        or api.get("wal_response_capture_contains_body_text") is not False
        or scheduler.get("submission_policy") != "bounded_at_worker_count"
        or scheduler.get("cancel_pending_after_first_worker_baseexception") is not True
    ):
        raise ValueError("V2 API/WAL/scheduler contract drift")
    if set(protocol["future_runner_output_contract"]["required_files"]) != set(ALL_OUTPUT_FILES):
        raise ValueError("V2 output contract drift")
    manifest_contract = protocol.get("protocol_freeze_manifest_contract") or {}
    if (
        manifest_contract.get("output_set_exact")
        != list(freeze_v2.PROTOCOL_FREEZE_OUTPUT_FILES)
        or manifest_contract.get("runner_must_verify_manifest_before_api_calls") is not True
    ):
        raise ValueError("V2 protocol-freeze manifest contract drift")
    if expected_rows == freeze_v1.PILOT_ROWS:
        expected_protocol = freeze_v2.build_protocol(
            generated_at_utc=str(protocol.get("generated_at_utc") or ""),
            parent_lock=protocol.get("parent_identity_freeze") or {},
            implementation_lock_path=str(binding.get("path") or ""),
            implementation_lock_sha256=str(binding.get("sha256") or ""),
            implementation_lock_size_bytes=int(binding.get("size_bytes") or -1),
            v1_to_v2_diagnostic=protocol.get("v1_to_v2_subject_binding_diagnostic") or {},
        )
        if protocol != expected_protocol:
            raise ValueError("V2 execution protocol differs from canonical freezer output")

    v1_inputs = manifest.get("v1_inputs")
    if not isinstance(v1_inputs, list) or len(v1_inputs) != 3:
        raise ValueError("V2 manifest V1 input closure missing")
    v1_hashes: dict[str, str] = {}
    for row in v1_inputs:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
            raise ValueError("V2 manifest V1 input schema drift")
        path = _resolve(project_root, Path(str(row["path"])))
        name = path.name
        if name in v1_hashes or name not in freeze_v2.EXPECTED_V1_HASHES:
            raise ValueError("V2 manifest V1 input path set drift")
        if _sha256_file(path) != row.get("sha256"):
            raise ValueError(f"V2 manifest V1 input hash drift: {name}")
        v1_hashes[name] = str(row["sha256"])
    if set(v1_hashes) != set(freeze_v2.EXPECTED_V1_HASHES):
        raise ValueError("V2 manifest V1 input path set drift")
    if enforce_formal_locks and v1_hashes != freeze_v2.EXPECTED_V1_HASHES:
        raise ValueError("formal V2 manifest V1 input hash drift")
    if supersession.get("append_only") is not True or supersession.get(
        "v1_protocol_sha256"
    ) != v1_hashes["protocol.json"]:
        raise ValueError("V1-to-V2 supersession V1 binding drift")
    return protocol, lock, report, manifest


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    freeze_v2._fsync_dir(path.parent)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    freeze_v2._fsync_dir(path.parent)


def _snapshot_files(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": path,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }


def _assert_snapshot_unchanged(snapshot: Mapping[str, Mapping[str, Any]]) -> None:
    for name, item in snapshot.items():
        path = item["path"]
        if (
            not isinstance(path, Path)
            or not path.is_file()
            or _sha256_file(path) != item["sha256"]
            or path.stat().st_size != item["size_bytes"]
        ):
            raise RuntimeError(f"input_snapshot_changed:{name}")


def _manifest_input(
    *, project_root: Path, item: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "path": _display(project_root, item["path"]),
        "sha256": item["sha256"],
        "size_bytes": item["size_bytes"],
    }


def _model_spec(protocol: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    return runner_v1._stage_spec(protocol, stage)


def _skipped(protocol: Mapping[str, Any], *, stage: str, qid: str, detail: str) -> runner_v1.StageResult:
    spec = _model_spec(protocol, stage)
    return runner_v1._skipped_result(
        stage=stage, experiment_id=str(protocol["experiment_id"]), qid=qid,
        requested_model=str(spec["model"]), prompt_template_sha256=str(spec["prompt"]["sha256"]), detail=detail,
    )


def _run_bounded_jobs(
    jobs: Sequence[Any],
    *,
    worker_count: int,
    stop_event: threading.Event,
    execute: Any,
    consume: Any,
) -> None:
    """Run with at most ``worker_count`` submitted futures at any time.

    Submission happens only after every future in the completed batch has
    returned successfully.  Therefore a worker BaseException can leave at
    most ``worker_count - 1`` already-dispatched calls; no queued backlog is
    allowed to start after the failure is observed.
    """

    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    iterator = iter(jobs)
    pool = ThreadPoolExecutor(max_workers=worker_count)
    in_flight: dict[Future[Any], Any] = {}

    def guarded_execute(job: Any) -> Any:
        if stop_event.is_set():
            raise RuntimeError("peer_baseexception_cancelled_before_job")
        try:
            return execute(job)
        except BaseException:
            stop_event.set()
            raise

    def fill() -> None:
        while len(in_flight) < worker_count:
            if stop_event.is_set():
                break
            try:
                job = next(iterator)
            except StopIteration:
                break
            in_flight[pool.submit(guarded_execute, job)] = job

    try:
        fill()
        while in_flight:
            done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
            completed: list[tuple[Any, Any]] = []
            # Resolve the entire completed batch before submitting replacement
            # work.  Any BaseException exits directly to cancellation below.
            for future in done:
                job = in_flight.pop(future)
                completed.append((job, future.result()))
            for job, result in completed:
                consume(job, result)
            fill()
    except BaseException:
        stop_event.set()
        for future in in_flight:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True, cancel_futures=True)


def _run_generation_pilot_v2_impl(
    *,
    client: runner_v1.CompletionClient,
    project_root: Path = PROJECT_ROOT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    implementation_lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK,
    protocol_report_path: Path = DEFAULT_PROTOCOL_REPORT,
    protocol_manifest_path: Path = DEFAULT_PROTOCOL_MANIFEST,
    v1_supersession_addendum_path: Path = DEFAULT_V1_SUPERSESSION_ADDENDUM,
    identity_path: Path = DEFAULT_IDENTITY,
    metadata_addendum_path: Path = DEFAULT_ADDENDUM,
    raw_path: Path = DEFAULT_RAW,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_rows: int = freeze_v1.PILOT_ROWS,
    enforce_formal_locks: bool = True,
) -> dict[str, Any]:
    paths = {
        "protocol": _resolve(project_root, protocol_path),
        "lock": _resolve(project_root, implementation_lock_path),
        "protocol_report": _resolve(project_root, protocol_report_path),
        "protocol_manifest": _resolve(project_root, protocol_manifest_path),
        "v1_supersession": _resolve(project_root, v1_supersession_addendum_path),
        "identity": _resolve(project_root, identity_path),
        "addendum": _resolve(project_root, metadata_addendum_path),
        "raw": _resolve(project_root, raw_path),
        "output": _resolve(project_root, output_dir),
    }
    if paths["output"].exists():
        raise FileExistsError(f"append-only output/in-progress directory already exists: {paths['output']}")
    protocol, implementation_lock, protocol_report, protocol_manifest = _verify_protocol_and_lock(
        project_root=project_root,
        protocol_path=paths["protocol"],
        implementation_lock_path=paths["lock"],
        protocol_report_path=paths["protocol_report"],
        protocol_manifest_path=paths["protocol_manifest"],
        v1_supersession_addendum_path=paths["v1_supersession"],
        identity_path=paths["identity"],
        raw_path=paths["raw"],
        parent_metadata_addendum_path=paths["addendum"],
        expected_rows=expected_rows,
        enforce_formal_locks=enforce_formal_locks,
    )
    endpoint_allowlist_pass = isinstance(client, OpenAICompletionClientV2)
    if enforce_formal_locks and not endpoint_allowlist_pass:
        raise ValueError("formal V2 run requires the endpoint-allowlisted client")
    addendum = runner_v1._load_json(paths["addendum"])
    if addendum.get("selection_result_changed") is not False or (addendum.get("precall_hardening_audit") or {}).get("builder_version") != silver.BUILDER_VERSION:
        raise ValueError("parent hardening addendum drift")
    if enforce_formal_locks:
        if _sha256_file(paths["addendum"]) != runner_v1.EXPECTED_ADDENDUM_SHA256 or _sha256_file(paths["raw"]) != runner_v1.EXPECTED_RAW_SHA256:
            raise ValueError("formal parent addendum/raw SHA256 drift")
        artifacts = protocol["parent_identity_freeze"]["artifacts"]
        for lock_item in artifacts.values():
            parent_path = _resolve(project_root, Path(lock_item["path"]))
            if _sha256_file(parent_path) != lock_item["sha256"]:
                raise ValueError("formal parent identity-freeze artifact drift")
    identities = runner_v1._load_jsonl(paths["identity"])
    if len(identities) != expected_rows or len({row.get("qid") for row in identities}) != expected_rows:
        raise ValueError("fixed denominator/qid uniqueness drift")
    if _sha256_file(paths["identity"]) != protocol["parent_identity_freeze"]["identity_sha256"]:
        raise ValueError("parent identity hash drift")
    raw_by_qid = runner_v1._read_selected_raw(paths["raw"], {str(row["qid"]) for row in identities})

    prepared: dict[str, dict[str, Any]] = {}
    stages: dict[str, dict[str, runner_v1.StageResult]] = {str(row["qid"]): {} for row in identities}
    failures: dict[str, dict[str, Any]] = {}
    experiment_id = str(protocol["experiment_id"])
    for identity in identities:
        qid = str(identity["qid"])
        try:
            if set(identity) != {"dataset", "qid", "question"} or identity["dataset"] != silver.DATASET:
                raise ValueError("identity_schema")
            raw = raw_by_qid[qid]
            if question_sha256(identity["question"]) != question_sha256(str(raw.get("question") or "")):
                raise ValueError("question_hash_join")
            chain = silver.extract_hotpot_support_chain(raw)
            if chain.qid != qid or chain.question != identity["question"]:
                raise ValueError("chain_identity_join")
            masked = silver.build_masked_proposal_view(chain)
            safe = build_producer_safe_payload_v2(masked, chain_sha256=chain.raw_record_sha256, experiment_id=experiment_id)
            prepared[qid] = {"identity": identity, "chain": chain, "safe_payload": safe}
        except (KeyError, ValueError, silver.HotpotSilverReject) as exc:
            detail = exc.code if isinstance(exc, silver.HotpotSilverReject) else type(exc).__name__
            failures[qid] = {"schema_version": SCHEMA_VERSION, "dataset": identity.get("dataset"), "qid": qid, "question_sha256": question_sha256(str(identity.get("question") or "")), "status": "precall_rejected", "reject_code": "input_identity_or_chain_integrity_error", "detail_code": detail}
            for stage in STAGES:
                stages[qid][stage] = _skipped(protocol, stage=stage, qid=qid, detail="precall_identity_or_chain_failure")
    expected_precall = int((addendum.get("precall_hardening_audit") or {}).get("precall_rejected", -1))
    if enforce_formal_locks and len(failures) != expected_precall:
        raise ValueError("precall count differs from bound parent addendum")

    # All validation above is API-free.  From this point onward the directory
    # itself is the non-reuse guard and WAL intents precede every paid attempt.
    paths["output"].mkdir(parents=True, exist_ok=False)
    wal = _WriteAheadLedger(paths["output"] / "api_call_wal.jsonl", experiment_id=experiment_id)
    wal.append(
        {
            "event": "run_started",
            "at_utc": _utc_now(),
            "fixed_denominator": expected_rows,
            "precall_constructible": len(prepared),
            "precall_rejected": len(failures),
        }
    )
    # Persist both the non-reuse directory and its WAL entry before the first
    # provider dispatch.  A power loss must not erase the paid-call guard.
    freeze_v2._fsync_dir(paths["output"])
    freeze_v2._fsync_dir(paths["output"].parent)
    stop_event = threading.Event()
    executor = _V2ApiExecutor(client, protocol, wal, stop_event)
    workers = int(protocol["api_execution"]["worker_count"])

    def producer_job(qid: str) -> runner_v1.StageResult:
        item = prepared[qid]
        messages, payload, prompt_hash = runner_v1._messages_and_payload(
            stage="producer", protocol=protocol, safe_payload=item["safe_payload"], proposal=None, chain=item["chain"]
        )
        result = executor.call(stage="producer", qid=qid, stage_spec=protocol["producer"], messages=messages, safe_payload_sha256=_canonical_sha256(payload), prompt_template_sha256=prompt_hash, forbidden_chain_secrets=(item["chain"].bridge_title, item["chain"].intermediate, *item["chain"].final_answers))
        _finalize_producer(result, item["chain"])
        return result

    try:
        producer_qids = list(prepared)
        _run_bounded_jobs(
            producer_qids,
            worker_count=workers,
            stop_event=stop_event,
            execute=producer_job,
            consume=lambda qid, result: stages[qid].__setitem__("producer", result),
        )

        review_jobs: list[tuple[str, str]] = []
        for qid, item in prepared.items():
            producer = stages[qid]["producer"]
            if producer.semantic_status != "accepted":
                identity = item["identity"]
                failures[qid] = {"schema_version": SCHEMA_VERSION, "dataset": identity["dataset"], "qid": qid, "question_sha256": question_sha256(identity["question"]), "status": "producer_rejected", "reject_code": producer.reject_code, "detail_code": producer.detail_code}
                for stage in STAGES[1:]:
                    stages[qid][stage] = _skipped(protocol, stage=stage, qid=qid, detail="producer_upstream_failure")
            else:
                review_jobs.extend((qid, stage) for stage in STAGES[1:])

        def review_job(job: tuple[str, str]) -> runner_v1.StageResult:
            qid, stage = job
            item = prepared[qid]
            proposal = stages[qid]["producer"].parsed
            messages, payload, prompt_hash = runner_v1._messages_and_payload(
                stage=stage, protocol=protocol, safe_payload=item["safe_payload"], proposal=proposal, chain=item["chain"]
            )
            spec = _model_spec(protocol, stage)
            result = executor.call(stage=stage, qid=qid, stage_spec=spec, messages=messages, safe_payload_sha256=_canonical_sha256(payload), prompt_template_sha256=prompt_hash, forbidden_chain_secrets=(item["chain"].bridge_title, item["chain"].intermediate, *item["chain"].final_answers))
            _finalize_review(result, protocol=protocol, chain=item["chain"])
            return result

        def consume_review(job: tuple[str, str], result: runner_v1.StageResult) -> None:
            qid, stage = job
            stages[qid][stage] = result

        _run_bounded_jobs(
            review_jobs,
            worker_count=workers,
            stop_event=stop_event,
            execute=review_job,
            consume=consume_review,
        )
    except BaseException as exc:
        boundary = wal.interruption_boundary()
        wal.append(
            {
                "event": "run_aborted",
                "at_utc": _utc_now(),
                "error_type": type(exc).__name__,
                **boundary,
            }
        )
        wal.close()
        # Deliberately retain the in-progress directory and WAL.  No final
        # manifest is written and a future invocation refuses this directory.
        raise

    accepted_actions: list[dict[str, Any]] = []
    item_statuses: list[dict[str, Any]] = []
    for identity in identities:
        qid = str(identity["qid"])
        if qid in failures:
            failure = failures[qid]
            item_statuses.append({"dataset": identity["dataset"], "qid": qid, "question_sha256": failure["question_sha256"], "status": failure["status"], "reject_code": failure["reject_code"]})
            continue
        item = prepared[qid]
        r1, r2 = stages[qid]["reviewer_1_blind"], stages[qid]["reviewer_2_gold_aware"]
        if r1.semantic_status == r2.semantic_status == "accepted":
            proposal = stages[qid]["producer"].parsed
            try:
                pair = silver.build_hotpot_action_pair(
                    item["chain"], proposal or {}, split="train",
                    extra_source_provenance={
                        "generation_experiment_id": experiment_id,
                        "producer_response_sha256": stages[qid]["producer"].raw_response_sha256,
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
                failures[qid] = {"schema_version": SCHEMA_VERSION, "dataset": identity["dataset"], "qid": qid, "question_sha256": question_sha256(identity["question"]), "status": "action_pair_rejected", "reject_code": "action_pair_postbuild_reject", "detail_code": exc.code}
                item_statuses.append({"dataset": identity["dataset"], "qid": qid, "question_sha256": question_sha256(identity["question"]), "status": "action_pair_rejected", "reject_code": "action_pair_postbuild_reject"})
        else:
            if (r1.semantic_status == "accepted" and r2.reject_code == "reviewer_2_reject") or (r2.semantic_status == "accepted" and r1.reject_code == "reviewer_1_reject"):
                code = "reviewer_disagreement_or_unknown"
            elif r1.reject_code == "reviewer_1_reject" and r2.reject_code == "reviewer_2_reject":
                code = "unanimous_review_failed"
            else:
                code = r1.reject_code or r2.reject_code or "unanimous_review_failed"
            detail = r1.detail_code if r1.semantic_status != "accepted" else r2.detail_code
            failures[qid] = {"schema_version": SCHEMA_VERSION, "dataset": identity["dataset"], "qid": qid, "question_sha256": question_sha256(identity["question"]), "status": "review_rejected", "reject_code": code, "detail_code": detail}
            item_statuses.append({"dataset": identity["dataset"], "qid": qid, "question_sha256": question_sha256(identity["question"]), "status": "review_rejected", "reject_code": code})

    final_by_qid = {row["qid"]: row for row in item_statuses}
    semantic_rows: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []
    role_rows: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGES}
    for identity in identities:
        qid = str(identity["qid"])
        for stage in STAGES:
            result = stages[qid][stage]
            semantic_rows.append(runner_v1._ledger_row(result, experiment_id=experiment_id, identity=identity))
            transport_rows.extend(result.transport_rows or [])
            role_row = runner_v1._role_row(identity, result)
            role_row["schema_version"] = SCHEMA_VERSION
            if stage == "producer":
                accepted = final_by_qid[qid]["status"] == "accepted_generation_and_dual_review"
                proposal = result.parsed if accepted else None
                role_row.update(
                    {
                        "final_item_status": final_by_qid[qid]["status"],
                        "dual_review_unanimous_pass": accepted,
                        "q1_query": proposal.get("q1") if proposal else None,
                        "q2_template": proposal.get("q2_template") if proposal else None,
                        "proposal_sha256": result.parsed_response_sha256 if accepted else None,
                        "runtime_projection_gold_or_observation_fields_present": False,
                    }
                )
            role_rows[stage].append(role_row)
    if len(semantic_rows) != expected_rows * 3 or len(item_statuses) != expected_rows:
        raise AssertionError("V2 fixed-denominator conservation failed")
    transport_counts = Counter(row["semantic_request_id"] for row in transport_rows)
    if any(row["transport_attempt_count"] != transport_counts[row["semantic_request_id"]] for row in semantic_rows):
        raise AssertionError("V2 semantic/transport conservation failed")
    wal_rows = runner_v1._load_jsonl(paths["output"] / "api_call_wal.jsonl")
    intent_keys = [(row["semantic_request_id"], row["physical_attempt_index"]) for row in wal_rows if row["event"] == "intent"]
    capture_keys = [
        (row["semantic_request_id"], row["physical_attempt_index"])
        for row in wal_rows
        if row["event"] == "response_captured" and row.get("capture_complete") is True
    ]
    result_keys = [(row["semantic_request_id"], row["physical_attempt_index"]) for row in wal_rows if row["event"] == "result"]
    wal_balanced = Counter(intent_keys) == Counter(result_keys) and len(intent_keys) == len(transport_rows)
    if not wal_balanced:
        raise AssertionError("completed V2 run has unmatched WAL events")
    received_keys = [
        (row["semantic_request_id"], row["physical_attempt_index"])
        for row in transport_rows
        if row["response_received"]
    ]
    response_capture_balanced = Counter(capture_keys) == Counter(received_keys)
    if not response_capture_balanced:
        raise AssertionError("completed V2 run has missing/duplicate response captures")

    accepted_items = len(accepted_actions) // 2
    response_rows = [row for row in semantic_rows if row["raw_response_sha256"] is not None]
    nonce_clean = all(row["nonce_echo_count"] == 0 for row in response_rows)
    accepted_min = int(protocol["pilot_decision_gate_inherited"]["accepted_min"])
    accepted_min_pass = accepted_items >= accepted_min
    generation_gate = (
        accepted_min_pass and nonce_clean and wal_balanced and response_capture_balanced
    )
    status = STATUS_COMPLETE if generation_gate else STATUS_FAIL
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at_utc": _utc_now(),
        "status": status,
        "fixed_denominator": expected_rows,
        "precall_rejected": sum(row["status"] == "precall_rejected" for row in item_statuses),
        "producer_accepted": sum(stages[str(row["qid"])]["producer"].semantic_status == "accepted" for row in identities),
        "dual_review_accepted": accepted_items,
        "failed_items": len(failures),
        "accepted_action_rows": len(accepted_actions),
        "semantic_call_rows": len(semantic_rows),
        "transport_attempt_rows": len(transport_rows),
        "wal_event_rows": len(wal_rows) + 1,
        "item_statuses": item_statuses,
        "reject_code_counts": dict(sorted(Counter(row["reject_code"] for row in failures.values()).items())),
        "gates": {
            "accepted_min": accepted_min,
            "accepted_min_pass": accepted_min_pass,
            "all_model_response_text_nonce_like_free": nonce_clean,
            "wal_intent_result_balanced": wal_balanced,
            "wal_response_capture_balanced": response_capture_balanced,
            "generation_and_dual_review_gate_pass": generation_gate,
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
            "endpoint_allowlist_pass": endpoint_allowlist_pass,
        },
    }

    wal.append(
        {
            "event": "semantic_calls_and_in_memory_validation_completed",
            "at_utc": _utc_now(),
            "unmatched_intent_count": 0,
            "physical_attempt_count": len(transport_rows),
            "accepted_identity_count": accepted_items,
        }
    )
    wal.close()
    # All ordinary final artifacts use write+fsync+atomic replace, so an
    # interrupted file is never mistaken for a complete JSON/JSONL artifact.
    for stage, filename in ROLE_FILES.items():
        _atomic_write_jsonl(paths["output"] / filename, role_rows[stage])
    _atomic_write_jsonl(paths["output"] / "accepted_actions.jsonl", accepted_actions)
    _atomic_write_jsonl(paths["output"] / "failures.jsonl", [failures[str(row["qid"])] for row in identities if str(row["qid"]) in failures])
    _atomic_write_jsonl(paths["output"] / "semantic_call_ledger.jsonl", semantic_rows)
    _atomic_write_jsonl(paths["output"] / "api_transport_attempt_ledger.jsonl", transport_rows)
    _atomic_write_json(paths["output"] / "report.json", report)
    if {path.name for path in paths["output"].iterdir()} != set(ALL_OUTPUT_FILES[:-1]):
        raise AssertionError("V2 pre-manifest output set is not exactly frozen")
    outputs = []
    for filename in ALL_OUTPUT_FILES[:-1]:
        path = paths["output"] / filename
        outputs.append({"path": filename, "sha256": _sha256_file(path), "size_bytes": path.stat().st_size})
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at_utc": _utc_now(),
        "status": status,
        "inputs": {
            "execution_protocol": {"path": _display(project_root, paths["protocol"]), "sha256": _sha256_file(paths["protocol"])},
            "implementation_lock": {"path": _display(project_root, paths["lock"]), "sha256": _sha256_file(paths["lock"])},
            "protocol_freeze_report": {"path": _display(project_root, paths["protocol_report"]), "sha256": _sha256_file(paths["protocol_report"])},
            "protocol_freeze_manifest": {"path": _display(project_root, paths["protocol_manifest"]), "sha256": _sha256_file(paths["protocol_manifest"])},
            "v1_supersession_addendum": {"path": _display(project_root, paths["v1_supersession"]), "sha256": _sha256_file(paths["v1_supersession"])},
            "parent_identity": {"path": _display(project_root, paths["identity"]), "sha256": _sha256_file(paths["identity"])},
            "parent_metadata_addendum_v1_1": {"path": _display(project_root, paths["addendum"]), "sha256": _sha256_file(paths["addendum"])},
            "raw_train": {"path": _display(project_root, paths["raw"]), "sha256": _sha256_file(paths["raw"])},
        },
        "implementation_lock_verified": True,
        "protocol_freeze_manifest_verified": True,
        "implementation_count": len(implementation_lock["implementations"]),
        "outputs": outputs,
        "fixed_denominator": expected_rows,
        "api_physical_attempts": len(transport_rows),
        "wal_balanced": wal_balanced,
        "wal_response_capture_balanced": response_capture_balanced,
        "credential_or_endpoint_values_serialized": False,
        "endpoint_allowlist_pass": endpoint_allowlist_pass,
        "endpoint_environment_variable_name": "OPENAI_BASE_URL",
        "retrieval_calls": 0,
        "training_started": False,
    }
    _atomic_write_json(paths["output"] / "manifest.json", manifest)
    freeze_v2._fsync_dir(paths["output"])
    if set(path.name for path in paths["output"].iterdir()) != set(ALL_OUTPUT_FILES):
        raise AssertionError("V2 did not materialise exactly the frozen output files")
    return {"report": report, "manifest": manifest, "output_dir": paths["output"]}


def _append_outer_abort_if_needed(output_dir: Path, exc: BaseException) -> None:
    """Record failures outside the dispatch block without leaking exception text."""

    wal_path = output_dir / "api_call_wal.jsonl"
    if not wal_path.is_file() or (output_dir / "manifest.json").exists():
        return
    try:
        rows = runner_v1._load_jsonl(wal_path)
    except Exception:  # a partial WAL is itself the durable crash marker
        return
    if any(row.get("event") == "run_aborted" for row in rows):
        return
    intents = Counter(
        (row.get("semantic_request_id"), row.get("physical_attempt_index"))
        for row in rows
        if row.get("event") == "intent"
    )
    results = Counter(
        (row.get("semantic_request_id"), row.get("physical_attempt_index"))
        for row in rows
        if row.get("event") == "result"
    )
    captures = Counter(
        (row.get("semantic_request_id"), row.get("physical_attempt_index"))
        for row in rows
        if row.get("event") == "response_captured" and row.get("capture_complete") is True
    )
    capture_attempts = Counter(
        (row.get("semantic_request_id"), row.get("physical_attempt_index"))
        for row in rows
        if row.get("event") == "response_captured"
    )
    unmatched = sum((intents - results).values())
    unresolved = intents - results
    unknown_after_dispatch = sum(
        max(count - capture_attempts[key], 0) for key, count in unresolved.items()
    )
    capture_incomplete = sum(
        min(count, capture_attempts[key]) - min(count, captures[key])
        for key, count in unresolved.items()
    )
    response_captured = sum(
        min(count, captures[key]) for key, count in unresolved.items()
    )
    if unknown_after_dispatch:
        boundary = "UNKNOWN_AFTER_DISPATCH"
    elif capture_incomplete:
        boundary = "PROVIDER_RETURNED_CAPTURE_INCOMPLETE"
    elif response_captured:
        boundary = "RESPONSE_CAPTURED_LOCALLY_NOT_RESULT_RECORDED"
    else:
        boundary = "NO_UNMATCHED_DISPATCH"
    event = {
        "schema_version": "hotpot-controller-api-wal-event-1",
        "experiment_id": freeze_v2.EXPERIMENT_ID,
        "event": "run_aborted",
        "at_utc": _utc_now(),
        "error_type": type(exc).__name__,
        "unmatched_intent_count": unmatched,
        "unknown_after_dispatch_count": unknown_after_dispatch,
        "provider_returned_capture_incomplete_count": capture_incomplete,
        "response_captured_without_result_count": response_captured,
        "provider_outcome_boundary": boundary,
    }
    with wal_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_generation_pilot_v2(
    *,
    client: runner_v1.CompletionClient,
    project_root: Path = PROJECT_ROOT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    implementation_lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK,
    protocol_report_path: Path = DEFAULT_PROTOCOL_REPORT,
    protocol_manifest_path: Path = DEFAULT_PROTOCOL_MANIFEST,
    v1_supersession_addendum_path: Path = DEFAULT_V1_SUPERSESSION_ADDENDUM,
    identity_path: Path = DEFAULT_IDENTITY,
    metadata_addendum_path: Path = DEFAULT_ADDENDUM,
    raw_path: Path = DEFAULT_RAW,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_rows: int = freeze_v1.PILOT_ROWS,
    enforce_formal_locks: bool = True,
) -> dict[str, Any]:
    """Public crash-audited wrapper around the V2 implementation."""

    resolved_output = _resolve(project_root, output_dir)
    try:
        return _run_generation_pilot_v2_impl(
            client=client,
            project_root=project_root,
            protocol_path=protocol_path,
            implementation_lock_path=implementation_lock_path,
            protocol_report_path=protocol_report_path,
            protocol_manifest_path=protocol_manifest_path,
            v1_supersession_addendum_path=v1_supersession_addendum_path,
            identity_path=identity_path,
            metadata_addendum_path=metadata_addendum_path,
            raw_path=raw_path,
            output_dir=output_dir,
            expected_rows=expected_rows,
            enforce_formal_locks=enforce_formal_locks,
        )
    except BaseException as exc:
        _append_outer_abort_if_needed(resolved_output, exc)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--implementation_lock", type=Path, default=DEFAULT_IMPLEMENTATION_LOCK)
    parser.add_argument("--protocol_report", type=Path, default=DEFAULT_PROTOCOL_REPORT)
    parser.add_argument("--protocol_manifest", type=Path, default=DEFAULT_PROTOCOL_MANIFEST)
    parser.add_argument(
        "--v1_supersession_addendum",
        type=Path,
        default=DEFAULT_V1_SUPERSESSION_ADDENDUM,
    )
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--metadata_addendum", type=Path, default=DEFAULT_ADDENDUM)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute_api", action="store_true")
    args = parser.parse_args()
    if not args.execute_api:
        raise SystemExit("No API call made. V2 requires explicit --execute_api after protocol review.")
    result = run_generation_pilot_v2(
        client=OpenAICompletionClientV2(), protocol_path=args.protocol,
        implementation_lock_path=args.implementation_lock,
        protocol_report_path=args.protocol_report,
        protocol_manifest_path=args.protocol_manifest,
        v1_supersession_addendum_path=args.v1_supersession_addendum,
        identity_path=args.identity,
        metadata_addendum_path=args.metadata_addendum, raw_path=args.raw, output_dir=args.output_dir,
    )
    print(json.dumps({"status": result["report"]["status"], "dual_review_accepted": result["report"]["dual_review_accepted"]}, sort_keys=True))


if __name__ == "__main__":
    main()
