import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from kgproweight.data.sft_v3_api import (
    BudgetStop, DurableCalls, UnresolvedAttempt, live_transport, request_body, _safe_provider_payload,
)


MSG = [{"role": "system", "content": "Return JSON."}, {"role": "user", "content": "Public question."}]


def reply(body):
    return {"model": body["model"], "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]}


def call(ledger, transport=reply, call_id="q1/producer"):
    return ledger.complete(call_id=call_id, model="deepseek-v4-flash", messages=MSG,
                           max_tokens=100, transport=transport)


def test_intent_exists_before_transport_and_response_before_return(tmp_path):
    path = tmp_path / "wal.jsonl"
    ledger = DurableCalls(path, budget_usd=1, max_calls=4)
    def provider(body):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["event"] for r in rows] == ["intent"]
        assert rows[0]["request"] == body
        return reply(body)
    assert call(ledger, provider)["usable"]
    assert [json.loads(line)["event"] for line in path.read_text().splitlines()] == ["intent", "result"]


def test_resume_reuses_exact_response_without_repaying(tmp_path):
    path = tmp_path / "wal"
    first = call(DurableCalls(path, budget_usd=1, max_calls=1))
    resumed = DurableCalls(path, budget_usd=1, max_calls=1)
    def forbidden(_):
        pytest.fail("provider must not be called again")
    assert call(resumed, forbidden) == first
    with pytest.raises(ValueError, match="different request"):
        resumed.complete(call_id="q1/producer", model="deepseek-v4-pro", messages=MSG,
                         max_tokens=100, transport=forbidden)


def test_crash_during_transport_never_resubmits_unknown_paid_call(tmp_path):
    path = tmp_path / "wal"
    ledger = DurableCalls(path, budget_usd=1, max_calls=2)
    def crash(_):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        call(ledger, crash)
    resumed = DurableCalls(path, budget_usd=1, max_calls=2)
    assert resumed.committed_upper_usd() > 0
    with pytest.raises(UnresolvedAttempt):
        call(resumed)


def test_error_message_never_serialized_and_reserved_cost_retained(tmp_path):
    path = tmp_path / "wal"
    ledger = DurableCalls(path, budget_usd=1, max_calls=1)
    def error(_):
        raise RuntimeError("Authorization Bearer SECRET-MUST-NOT-APPEAR")
    result = call(ledger, error)
    assert not result["usable"]
    assert result["error_class"] == "RuntimeError"
    assert result["charged_upper_usd"] == ledger.intents["q1/producer"]["reserved_upper_usd"]
    assert "SECRET" not in path.read_text()


def test_budget_and_concurrent_call_ceiling_checked_before_network(tmp_path):
    ledger = DurableCalls(tmp_path / "wal", budget_usd=1, max_calls=1)
    seen = []
    def provider(body):
        seen.append(1)
        return reply(body)
    def worker(i):
        try:
            call(ledger, provider, str(i))
            return True
        except BudgetStop:
            return False
    with ThreadPoolExecutor(8) as pool:
        outcomes = list(pool.map(worker, range(8)))
    assert sum(outcomes) == len(seen) == 1
    small = DurableCalls(tmp_path / "small", budget_usd=0.000001, max_calls=1)
    with pytest.raises(BudgetStop):
        call(small, provider)
    assert len(seen) == 1


@pytest.mark.parametrize("mutation", ["model", "length", "no_usage", "thinking", "empty"])
def test_noncontract_provider_responses_remain_logged_but_unusable(tmp_path, mutation):
    def provider(body):
        result = reply(body)
        if mutation == "model": result["model"] = "unfrozen-model"
        if mutation == "length": result["choices"][0]["finish_reason"] = "length"
        if mutation == "no_usage": result.pop("usage")
        if mutation == "thinking": result["choices"][0]["message"]["reasoning_content"] = "unexpected mode"
        if mutation == "empty": result["choices"][0]["message"]["content"] = ""
        return result
    ledger = DurableCalls(tmp_path / "wal", budget_usd=1, max_calls=2)
    assert not call(ledger, provider)["usable"]
    assert len(ledger.results) == 1


def test_response_exceeding_budget_assumption_is_preserved_and_stops(tmp_path):
    def provider(body):
        result = reply(body)
        result["usage"]["completion_tokens"] = 101
        return result
    ledger = DurableCalls(tmp_path / "wal", budget_usd=1, max_calls=2)
    with pytest.raises(BudgetStop, match="reservation"):
        call(ledger, provider)
    assert not ledger.results["q1/producer"]["usable"]


@pytest.mark.parametrize("endpoint", ["http://api.deepseek.com", "https://third.example/v1", "https://api.deepseek.com@third.example", "https://x:y@api.deepseek.com", "https://api.deepseek.com/v1?key=bad"])
def test_credentials_cannot_be_sent_to_redirected_or_nonofficial_endpoints(tmp_path, monkeypatch, endpoint):
    for key in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY='test-never-transmit'\nOPENAI_BASE_URL='" + endpoint + "'\n")
    with pytest.raises(ValueError, match="official"):
        live_transport(env)


def test_body_pins_non_thinking_json_and_disallows_hidden_metadata():
    body = request_body("deepseek-v4-flash", MSG, 1600)
    assert body["thinking"] == {"type": "disabled"}
    assert body["response_format"] == {"type": "json_object"}
    with pytest.raises(ValueError):
        request_body("deepseek-v4-flash", [dict(MSG[1], gold_answer="secret")], 1600)


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(usage=[]),
    lambda x: x.update(usage="bad"),
    lambda x: x.update(usage={"prompt_tokens": True, "completion_tokens": 10}),
    lambda x: x.update(usage={"prompt_tokens": float("nan"), "completion_tokens": 10}),
    lambda x: x.update(choices=None),
    lambda x: x.update(choices={}),
    lambda x: x.update(choices=[None]),
    lambda x: x.update(choices=[42]),
    lambda x: x.update(choices=[{}]),
    lambda x: x["choices"][0].update(message=None),
    lambda x: x["choices"][0].update(message=[]),
    lambda x: x["choices"][0]["message"].update(content=[{"text":"bad"}]),
    lambda x: x["choices"][0]["message"].update(reasoning_content={"structured":"unexpected"}),
    lambda x: x["choices"][0]["message"].update(refusal="No answer"),
    lambda x: x.update(model=[]),
])
def test_malformed_nested_provider_shapes_are_persisted_and_not_retried(tmp_path, mutation):
    path = tmp_path / "wal"
    ledger = DurableCalls(path, budget_usd=1, max_calls=2)
    def provider(body):
        payload = reply(body)
        mutation(payload)
        return payload
    result = call(ledger, provider)
    assert result["usable"] is False
    assert len(path.read_text().splitlines()) == 2
    resumed = DurableCalls(path, budget_usd=1, max_calls=2)
    assert call(resumed, lambda _: pytest.fail("must use persisted failure")) == result


@pytest.mark.parametrize("payload", [None, [], "unstructured error", 123])
def test_nonobject_provider_result_is_also_durable(tmp_path, payload):
    ledger = DurableCalls(tmp_path / "wal", budget_usd=1, max_calls=1)
    result = call(ledger, lambda _: payload)
    assert not result["usable"]
    assert result["charged_upper_usd"] == ledger.intents["q1/producer"]["reserved_upper_usd"]
    assert result["payload"] is None


def test_nested_unknown_fields_do_not_leak_and_sanitizer_is_idempotent(tmp_path):
    path = tmp_path / "wal"
    def provider(body):
        payload = reply(body)
        payload["Authorization"] = "SECRET_HEADER"
        payload["usage"]["request_headers"] = "SECRET_USAGE"
        payload["choices"][0]["request"] = "SECRET_CHOICE"
        payload["choices"][0]["message"]["headers"] = "SECRET_MESSAGE"
        return payload
    result = call(DurableCalls(path, budget_usd=1, max_calls=1), provider)
    assert result["usable"]
    assert "SECRET" not in path.read_text()
    sanitized = result["payload"]
    assert _safe_provider_payload(sanitized) == sanitized
    raw = reply({"model":"deepseek-v4-flash"})
    raw["choices"][0]["message"]["reasoning_content"] = ["nonempty reasoning"]
    once = _safe_provider_payload(raw)
    assert _safe_provider_payload(once) == once


def test_reservation_breach_persistently_stops_current_and_resumed_run(tmp_path):
    path = tmp_path / "wal"
    ledger = DurableCalls(path, budget_usd=1, max_calls=4)
    def excessive(body):
        value = reply(body)
        value["usage"]["completion_tokens"] = 101
        return value
    with pytest.raises(BudgetStop):
        call(ledger, excessive)
    with pytest.raises(BudgetStop):
        call(ledger, lambda _: pytest.fail("no new network after breach"), "q2/producer")
    with pytest.raises(BudgetStop):
        call(ledger)
    with pytest.raises(BudgetStop, match="run remains stopped"):
        DurableCalls(path, budget_usd=1, max_calls=4)


@pytest.mark.parametrize("mutation", [
    lambda rows: rows[0].update(reserved_upper_usd=-1),
    lambda rows: rows[0].update(request_sha256="changed"),
    lambda rows: rows[1].update(request_sha256="changed"),
    lambda rows: rows[1].update(charged_upper_usd=-1),
    lambda rows: rows[1].update(charged_upper_usd=0),
    lambda rows: rows[1].update(usable=1),
])
def test_corrupt_wal_accounting_does_not_reopen_budget(tmp_path, mutation):
    path = tmp_path / "wal"
    call(DurableCalls(path, budget_usd=1, max_calls=2))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(rows)
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError):
        DurableCalls(path, budget_usd=1, max_calls=2)


def test_duplicate_wal_json_keys_are_rejected(tmp_path):
    path = tmp_path / "wal"
    path.write_text('{"event":"intent","event":"result","call_id":"q1"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        DurableCalls(path, budget_usd=1, max_calls=2)
