"""Bounded, resumable DeepSeek transport for SFT data creation, never training.

No SDK retries; each paid attempt has a durable intent before transmission.
An interrupted attempt remains charged at its reserved upper estimate and is
never silently repeated. Credentials are read only in the live HTTP adapter.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import threading
from typing import Any
from urllib.parse import urlsplit


API_VERSION = "sft-v3-api-v1"
# USD / 1M tokens, peak, cache-miss: conservative for all times/cache states.
# Source verified 2026-09-06: https://api-docs.deepseek.com/quick_start/pricing/
RATES = {"deepseek-v4-flash": (0.44, 1.32), "deepseek-v4-pro": (1.32, 3.96)}
ALLOWED_RESPONSE_MODELS = {
    "deepseek-v4-flash": {"deepseek-v4-flash", "DeepSeek-V4-Flash-0731", "deepseek-v4-flash-0731"},
    "deepseek-v4-pro": {"deepseek-v4-pro", "DeepSeek-V4-Pro-0813", "deepseek-v4-pro-0813"},
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def upper_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if (not isinstance(model, str) or model not in RATES or type(prompt_tokens) is not int or type(completion_tokens) is not int
            or min(prompt_tokens, completion_tokens) < 0):
        raise ValueError("unsupported model or invalid token usage")
    input_rate, output_rate = RATES[model]
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000


def request_body(model: str, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
    if not isinstance(model, str) or model not in RATES or type(max_tokens) is not int or not 1 <= max_tokens <= 2400:
        raise ValueError("model/output budget outside the SFT v3 transport contract")
    if not isinstance(messages, list) or not messages or any(not isinstance(m, dict)
                           or set(m) != {"role", "content"} or m["role"] not in ("system", "user")
                           or not isinstance(m["content"], str) for m in messages):
        raise ValueError("only explicit system/user text messages may leave the project")
    return {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.2 if model == "deepseek-v4-flash" else 0.0,
            "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"},
            "stream": False}


class BudgetStop(RuntimeError):
    pass


class UnresolvedAttempt(RuntimeError):
    pass


def _safe_provider_payload(payload: Any) -> dict[str, Any] | None:
    """Whitelist nested fields without trusting the provider's JSON shape.

    Do not copy request/header echoes or arbitrary exception/error bodies.
    Invalid field shapes are represented by absent fields and make the
    response unusable. Content itself is the requested research artifact.
    """
    if not isinstance(payload, dict):
        return None
    result: dict[str, Any] = {}
    for key in ("id", "model", "system_fingerprint"):
        if isinstance(payload.get(key), str):
            result[key] = payload[key]
    for key in ("created", "http_status"):
        if type(payload.get(key)) is int:
            result[key] = payload[key]
    if isinstance(payload.get("usage"), dict):
        result["usage"] = {key: payload["usage"][key]
                           for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                                       "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
                           if type(payload["usage"].get(key)) is int}
    if isinstance(payload.get("choices"), list):
        choices = []
        for item in payload["choices"]:
            choice: dict[str, Any] = {}
            if isinstance(item, dict):
                if isinstance(item.get("finish_reason"), str):
                    choice["finish_reason"] = item["finish_reason"]
                if type(item.get("index")) is int:
                    choice["index"] = item["index"]
                if isinstance(item.get("message"), dict):
                    choice["message"] = {
                        key: item["message"][key]
                        for key in ("content", "reasoning_content", "role", "refusal")
                        if isinstance(item["message"].get(key), str)
                    }
                    # Unknown/structured reasoning must not turn into an
                    # apparently empty reasoning field after sanitization.
                    if (item["message"].get("reasoning_content") not in (None, "")
                            or item.get("has_reasoning_content") is True):
                        choice["has_reasoning_content"] = True
            choices.append(choice)
        result["choices"] = choices
    return result


def _read_wal_json(line: str) -> dict[str, Any]:
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate WAL JSON key")
            out[key] = value
        return out
    def invalid_constant(_):
        raise ValueError("non-finite WAL JSON value")
    result = json.loads(line, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise ValueError("WAL event must be an object")
    return result


def _finite_nonnegative(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


class DurableCalls:
    """Thread-safe single-process WAL. Caller holds an exclusive run lock.

    Files are append-only. A response is persisted *before* callers parse its
    content, so schema failures and wrong answers never disappear from logs.
    """
    def __init__(self, path: Path, *, budget_usd: float, max_calls: int):
        if (type(budget_usd) not in (int, float) or not math.isfinite(budget_usd)
                or not 0 < budget_usd <= 1000 or type(max_calls) is not int
                or not 0 < max_calls <= 100000):
            raise ValueError("explicit positive budget and call ceiling required")
        self.path, self.budget_usd, self.max_calls = path, budget_usd, max_calls
        self.lock = threading.Lock()
        self.intents: dict[str, dict[str, Any]] = {}
        self.results: dict[str, dict[str, Any]] = {}
        if path.exists():
            with path.open() as handle:
                for line in handle:
                    row = _read_wal_json(line)
                    cid = row.get("call_id")
                    if not isinstance(cid, str) or not cid.strip():
                        raise ValueError("WAL call_id must be nonempty text")
                    event = row.get("event")
                    if event not in ("intent", "result"):
                        raise ValueError("unknown WAL event")
                    target = self.intents if event == "intent" else self.results
                    if cid in target:
                        raise ValueError("duplicate or unknown WAL event")
                    if event == "intent":
                        request = row.get("request")
                        if not isinstance(request, dict) or sha_json(request) != row.get("request_sha256"):
                            raise ValueError("WAL intent request digest mismatch")
                        expected = request_body(request.get("model"), request.get("messages"), request.get("max_tokens"))
                        prompt_upper = len(canonical(request["messages"]).encode("utf-8")) + 512
                        reserve = upper_cost(request["model"], prompt_upper, request["max_tokens"])
                        if (request != expected or row.get("api_version") != API_VERSION
                                or row.get("prompt_tokens_upper") != prompt_upper
                                or not _finite_nonnegative(row.get("reserved_upper_usd"))
                                or abs(row["reserved_upper_usd"] - reserve) > 1e-12):
                            raise ValueError("WAL intent contract or reservation mismatch")
                    else:
                        if cid not in self.intents:
                            raise ValueError("result without prior intent")
                        if (row.get("request_sha256") != self.intents[cid]["request_sha256"]
                                or not _finite_nonnegative(row.get("charged_upper_usd"))
                                or type(row.get("usable")) is not bool):
                            raise ValueError("WAL result digest or accounting mismatch")
                        intent = self.intents[cid]
                        payload = _safe_provider_payload(row.get("payload"))
                        if payload != row.get("payload"):
                            raise ValueError("WAL response contains non-whitelisted fields")
                        usage = (payload or {}).get("usage") or {}
                        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
                        valid_usage = type(pt) is int and type(ct) is int and pt >= 0 and ct >= 0
                        expected_charge = (upper_cost(intent["request"]["model"], pt, ct)
                                           if valid_usage else intent["reserved_upper_usd"])
                        if abs(row["charged_upper_usd"] - expected_charge) > 1e-12:
                            raise ValueError("WAL recorded charge disagrees with provider usage/reservation")
                        over = valid_usage and (pt > intent["prompt_tokens_upper"]
                                                 or ct > intent["request"]["max_tokens"]
                                                 or expected_charge > intent["reserved_upper_usd"] + 1e-9)
                        if over != (row.get("error_class") == "ProviderUsageExceedsReservation"):
                            raise ValueError("WAL reservation violation marker mismatch")
                    target[cid] = row
        if any(row.get("error_class") == "ProviderUsageExceedsReservation" for row in self.results.values()):
            raise BudgetStop("existing WAL records a provider reservation violation; run remains stopped")
        if len(self.intents) > max_calls or self.committed_upper_usd() > budget_usd + 1e-9:
            raise BudgetStop("existing WAL exceeds requested run bounds")

    def committed_upper_usd(self) -> float:
        return sum(self.results.get(cid, {}).get("charged_upper_usd", row["reserved_upper_usd"])
                   for cid, row in self.intents.items())

    def _append(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Persist directory entries as well as file data before transmission.
        directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def complete(self, *, call_id: str, model: str, messages: list[dict[str, str]],
                 max_tokens: int, transport: Any) -> dict[str, Any]:
        body = request_body(model, messages, max_tokens)
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("call_id must be nonempty text")
        digest = sha_json(body)
        # UTF-8 bytes bound ordinary byte-BPE content tokens; 512 additionally
        # reserves message framing. Usage exceeding this assumption stops run.
        prompt_upper = len(canonical(messages).encode("utf-8")) + 512
        reserve = upper_cost(model, prompt_upper, max_tokens)
        with self.lock:
            if any(row.get("error_class") == "ProviderUsageExceedsReservation" for row in self.results.values()):
                raise BudgetStop("provider reservation violation previously stopped this run")
            if call_id in self.intents:
                if self.intents[call_id]["request_sha256"] != digest:
                    raise ValueError("attempt identity reused with different request")
                if call_id not in self.results:
                    raise UnresolvedAttempt("durable intent has no response; automatic retry forbidden")
                return self.results[call_id]
            if (len(self.intents) >= self.max_calls
                    or self.committed_upper_usd() + reserve > self.budget_usd):
                raise BudgetStop("next request would exceed frozen API budget/call ceiling")
            intent = {"event": "intent", "call_id": call_id, "utc": utc_now(),
                      "request_sha256": digest, "request": body,
                      "reserved_upper_usd": reserve, "prompt_tokens_upper": prompt_upper,
                      "api_version": API_VERSION}
            self._append(intent)
            self.intents[call_id] = intent
        try:
            payload = _safe_provider_payload(transport(body))
            error_class = None
        except Exception as exc:
            # Never serialize exception text: HTTP libraries can embed headers.
            payload, error_class = None, type(exc).__name__
        result: dict[str, Any] = {
            "event": "result", "call_id": call_id, "utc": utc_now(),
            "request_sha256": digest, "payload": payload, "error_class": error_class,
            "charged_upper_usd": reserve, "usable": False,
        }
        if isinstance(payload, dict):
            usage = payload.get("usage") or {}
            pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
            valid_usage = type(pt) is int and type(ct) is int and pt >= 0 and ct >= 0
            if valid_usage:
                cost = upper_cost(model, pt, ct)
                result["charged_upper_usd"] = cost
                if pt > prompt_upper or ct > max_tokens or cost > reserve + 1e-9:
                    result["error_class"] = "ProviderUsageExceedsReservation"
            choices = payload.get("choices")
            if (valid_usage and not result["error_class"]
                    and payload.get("model") in ALLOWED_RESPONSE_MODELS[model]
                    and isinstance(choices, list) and len(choices) == 1
                    and isinstance(choices[0], dict)
                    and choices[0].get("finish_reason") == "stop"
                    and isinstance(choices[0].get("message"), dict)
                    and isinstance(choices[0].get("message", {}).get("content"), str)
                    and choices[0]["message"]["content"].strip()
                    and not choices[0].get("has_reasoning_content")
                    and not choices[0]["message"].get("refusal")):
                result["usable"] = True
        with self.lock:
            self._append(result)
            self.results[call_id] = result
        if result["error_class"] == "ProviderUsageExceedsReservation":
            raise BudgetStop("provider usage violated reservation; response preserved, run stopped")
        return result


def live_transport(env_file: Path):
    """Load only four permitted env keys. Reject third-party endpoints."""
    names = {"DEEPSEEK_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_BASE_URL", "OPENAI_BASE_URL"}
    values = {key: os.environ[key] for key in names if os.environ.get(key)}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            stripped = line.strip().removeprefix("export ")
            key, sep, raw = stripped.partition("=")
            key = key.strip()
            if sep and key in names and key not in values:
                tokens = shlex.split(raw, comments=True, posix=True)
                if len(tokens) != 1:
                    raise ValueError("credential file contains ambiguous value syntax")
                values[key] = tokens[0]
    key = values.get("DEEPSEEK_API_KEY") or values.get("OPENAI_API_KEY")
    endpoint = values.get("DEEPSEEK_BASE_URL") or values.get("OPENAI_BASE_URL") or "https://api.deepseek.com"
    url = urlsplit(endpoint)
    if (url.scheme != "https" or url.hostname != "api.deepseek.com" or url.port not in {None, 443}
            or url.username or url.password or url.query or url.fragment
            or url.path.rstrip("/") not in {"", "/v1"}):
        raise ValueError("only the official credential-free DeepSeek endpoint is allowed")
    if not key:
        raise ValueError("no project DeepSeek credential is configured")
    import httpx
    client = httpx.Client(timeout=httpx.Timeout(120, connect=20), follow_redirects=False)

    def send(body: dict[str, Any]) -> dict[str, Any]:
        response = client.post(endpoint.rstrip("/") + "/chat/completions", json=body,
                               headers={"Authorization": "Bearer " + key})
        if response.status_code != 200:
            # Do not store arbitrary provider error bodies or request headers.
            return {"http_status": response.status_code}
        # The same shape-safe nested allowlist is applied again at the durable
        # boundary, including when tests/custom transports supply the payload.
        return _safe_provider_payload(response.json())

    return send
