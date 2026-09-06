"""CPU/fake-client tests for the hardened Hotpot silver runner V2."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from kgproweight.data import hotpot_controller_silver as silver
from scripts.prepare import freeze_hotpot_controller_silver_execution_v1 as freeze_v1
from scripts.prepare import freeze_hotpot_controller_silver_execution_v2 as freeze_v2
from scripts.prepare import generate_hotpot_controller_silver_pilot_v2 as runner


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _raw(index: int, *, implicit_subject: bool = False) -> dict:
    root, bridge, final = f"Root {index}", f"Bridge {index}", f"Final {index}"
    second = f"It has requested attribute {final}." if implicit_subject else f"{bridge} has requested attribute {final}."
    return {
        "id": f"qid-{index}",
        "question": f"What requested attribute belongs to the organization linked with {root}?",
        "golden_answers": [final],
        "metadata": {
            "type": "bridge", "level": "medium",
            "supporting_facts": {"title": [root, bridge], "sent_id": [0, 0]},
            "context": {"title": [root, bridge], "sentences": [[f"{root} links directly to {bridge}."], [second]]},
        },
    }


def _pass_review() -> dict:
    return {
        "schema_version": freeze_v1.REVIEW_SCHEMA_VERSION,
        **{field: True for field in freeze_v1.REVIEW_BOOLEAN_FIELDS},
        "verdict": "pass", "reject_codes": [],
    }


class FakeClient:
    def __init__(self, mode: str = "valid") -> None:
        self.mode = mode
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    def complete(self, request_body, *, timeout):
        with self._lock:
            self.requests.append(deepcopy(dict(request_body)))
        system = request_body["messages"][0]["content"]
        model = request_body["model"]
        if self.mode == "abort":
            raise KeyboardInterrupt()
        if "You create retrieval-query" in system:
            user = request_body["messages"][1]["content"]
            payload = json.loads(user[user.index("{") :])
            root = payload["root_document_title"]
            proposal = {
                "schema_version": silver.PROPOSAL_SCHEMA_VERSION,
                "q1": f"Which organization is {root} directly linked with?",
                "q2_template": "What requested attribute does #1 have?",
            }
            if self.mode == "duplicate":
                content = '{"schema_version":"hotpot-controller-query-proposal-v1","q1":"Which organization is Root 0 directly linked with?","q1":"Duplicate?","q2_template":"What requested attribute does #1 have?"}'
            elif self.mode == "nonce_and_model_mismatch":
                content = json.dumps({**proposal, "q1": "Which organization is Ｎ" + "Ａ" * 31 + "?"})
                model = "wrong-model"
            else:
                content = json.dumps(proposal)
        else:
            content = json.dumps(_pass_review())
        message = {"content": content}
        if self.mode == "reasoning_nonce" and "You create retrieval-query" in system:
            message["reasoning_content"] = "\\u004e" + "a" * 31
        if self.mode == "refusal_chain_secret" and "You create retrieval-query" in system:
            message["refusal"] = "\\u0042ridge 0"
        if self.mode == "tool_credential" and "You create retrieval-query" in system:
            message["tool_calls"] = [
                {"function": {"name": "x", "arguments": "sk-adversarial123"}}
            ]
        choices = [{"message": message, "finish_reason": "stop"}]
        if self.mode == "two_choices" and "You create retrieval-query" in system:
            choices.append(deepcopy(choices[0]))
        if self.mode == "length" and "You create retrieval-query" in system:
            choices[0]["finish_reason"] = "length"
        if self.mode == "second_choice_nonce" and "You create retrieval-query" in system:
            choices.append(
                {
                    "message": {"content": "ok", "reasoning_content": "N" + "b" * 31},
                    "finish_reason": "stop",
                }
            )
        if self.mode == "metadata_secret" and "You create retrieval-query" in system:
            model = "sk-adversarial123"
            choices[0]["finish_reason"] = "sk-adversarial123"
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if self.mode == "usage_secret" and "You create retrieval-query" in system:
            usage["total_tokens"] = "sk-adversarial123"
        return {
            "http_status": 200, "model": model, "choices": choices,
            "usage": usage,
        }


def _inputs(tmp_path: Path, *, rows: int = 2) -> dict[str, Path]:
    raw_rows = [_raw(index, implicit_subject=index == 1) for index in range(rows)]
    raw = tmp_path / "raw.jsonl"
    identity = tmp_path / "identity.jsonl"
    _write_jsonl(raw, raw_rows)
    _write_jsonl(identity, [{"dataset": "hotpotqa", "qid": row["id"], "question": row["question"]} for row in raw_rows])
    builder = runner.PROJECT_ROOT / "kgproweight/data/hotpot_controller_silver.py"
    addendum = tmp_path / "addendum.json"
    _write_json(
        addendum,
        {"selection_result_changed": False, "precall_hardening_audit": {"builder_version": silver.BUILDER_VERSION, "builder_sha256": _sha(builder), "fixed_denominator": rows, "precall_rejected": 0}},
    )
    freeze_dir = tmp_path / "protocol-freeze"
    freeze_dir.mkdir()
    lock_value = freeze_v2.build_implementation_lock(project_root=runner.PROJECT_ROOT, generated_at_utc="2026-09-05T00:00:00+00:00")
    lock = freeze_dir / "implementation_lock.json"
    _write_json(lock, lock_value)
    parent = {
        "rows": 30,
        "artifacts": {name: {} for name in freeze_v1.PARENT_FILENAMES},
    }
    protocol_value = freeze_v2.build_protocol(
        generated_at_utc="2026-09-05T00:00:00+00:00", parent_lock=parent,
        implementation_lock_path=str(lock), implementation_lock_sha256=_sha(lock),
        implementation_lock_size_bytes=lock.stat().st_size,
    )
    # Unit fixtures may use a smaller denominator; formal V2 is always 30.
    protocol_value["parent_identity_freeze"]["rows"] = rows
    protocol_value["parent_identity_freeze"]["identity_sha256"] = _sha(identity)
    protocol = freeze_dir / "protocol.json"
    _write_json(protocol, protocol_value)
    protocol_report = freeze_dir / "report.json"
    _write_json(
        protocol_report,
        {
            "schema_version": freeze_v2.REPORT_SCHEMA_VERSION,
            "experiment_id": freeze_v2.EXPERIMENT_ID,
            "status": freeze_v2.STATUS,
            "api_calls": 0,
        },
    )
    supersession = tmp_path / "v1-supersession.json"
    protocol_manifest = freeze_dir / "manifest.json"
    _write_json(
        protocol_manifest,
        {
            "schema_version": freeze_v2.MANIFEST_SCHEMA_VERSION,
            "experiment_id": freeze_v2.EXPERIMENT_ID,
            "status": freeze_v2.STATUS,
            "api_calls": 0,
            "training_started": False,
            "protocol_freeze_output_set_exact": list(
                freeze_v2.PROTOCOL_FREEZE_OUTPUT_FILES
            ),
            "diagnostic_inputs": {
                "identity": {"path": str(identity), "sha256": _sha(identity)},
                "raw_train": {"path": str(raw), "sha256": _sha(raw)},
                "parent_metadata_addendum_v1_1": {
                    "path": str(addendum),
                    "sha256": _sha(addendum),
                },
            },
            "external_append_only_artifact": {
                "path": str(supersession),
                "commit_order": "after_complete_v2_manifest_directory_commit",
                "status_at_v2_manifest_commit": "PENDING_APPEND_ONLY_COMMIT",
                "must_bind_v2_manifest_sha256": True,
            },
            "outputs": [
                {
                    "path": path.name,
                    "sha256": _sha(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in (lock, protocol, protocol_report)
            ],
            "v1_inputs": [
                {"path": str(runner.PROJECT_ROOT / freeze_v2.V1_DIR / name), "sha256": _sha(runner.PROJECT_ROOT / freeze_v2.V1_DIR / name)}
                for name in ("protocol.json", "report.json", "manifest.json")
            ],
        },
    )
    _write_json(
        supersession,
        {
            "schema_version": freeze_v2.SUPERSESSION_SCHEMA_VERSION,
            "append_only": True,
            "v1_experiment_id": freeze_v1.EXPERIMENT_ID,
            "v1_protocol_sha256": _sha(
                runner.PROJECT_ROOT / freeze_v2.V1_DIR / "protocol.json"
            ),
            "status": "SUPERSEDED_BEFORE_ANY_API_CALL",
            "v1_api_calls": 0,
            "v1_generation_output_existed_at_supersession": False,
            "superseded_by": {
                "experiment_id": freeze_v2.EXPERIMENT_ID,
                "protocol_path": str(protocol),
                "protocol_sha256": _sha(protocol),
                "manifest_path": str(protocol_manifest),
                "manifest_sha256": _sha(protocol_manifest),
            },
        },
    )
    return {
        "raw": raw,
        "identity": identity,
        "lock": lock,
        "protocol": protocol,
        "protocol_report": protocol_report,
        "protocol_manifest": protocol_manifest,
        "v1_supersession": supersession,
        "addendum": addendum,
    }


def _run(tmp_path: Path, client: FakeClient, *, rows: int = 2):
    inputs = _inputs(tmp_path, rows=rows)
    output = tmp_path / "output"
    result = runner.run_generation_pilot_v2(
        client=client, protocol_path=inputs["protocol"], implementation_lock_path=inputs["lock"],
        protocol_report_path=inputs["protocol_report"],
        protocol_manifest_path=inputs["protocol_manifest"],
        v1_supersession_addendum_path=inputs["v1_supersession"],
        identity_path=inputs["identity"], metadata_addendum_path=inputs["addendum"],
        raw_path=inputs["raw"], output_dir=output, expected_rows=rows, enforce_formal_locks=False,
    )
    return result, output


def test_safe_payload_explicitly_binds_implicit_second_hop_subject() -> None:
    chain = silver.extract_hotpot_support_chain(_raw(1, implicit_subject=True))
    masked = silver.build_masked_proposal_view(chain)
    payload = runner.build_producer_safe_payload_v2(masked, chain_sha256=chain.raw_record_sha256, experiment_id=freeze_v2.EXPERIMENT_ID)
    assert payload["second_hop_subject_nonce"] in payload["first_hop_evidence_masked"]
    assert payload["second_hop_subject_nonce"] not in payload["second_hop_evidence_masked"]
    assert "Bridge 1" not in json.dumps(payload)


def test_success_has_balanced_fsynced_wal_and_answer_free_projection(tmp_path: Path) -> None:
    result, output = _run(tmp_path, FakeClient())
    assert {path.name for path in output.iterdir()} == set(runner.ALL_OUTPUT_FILES)
    assert result["report"]["dual_review_accepted"] == 2
    wal = runner.runner_v1._load_jsonl(output / "api_call_wal.jsonl")
    assert wal[0]["event"] == "run_started"
    assert wal[-1]["event"] == "semantic_calls_and_in_memory_validation_completed"
    assert len([row for row in wal if row["event"] == "intent"]) == 6
    captures = [row for row in wal if row["event"] == "response_captured"]
    assert len(captures) == 6
    assert all(row["capture_complete"] is True for row in captures)
    assert all("content" not in row and "reasoning" not in row for row in captures)
    assert len([row for row in wal if row["event"] == "result"]) == 6
    projections = runner.runner_v1._load_jsonl(output / "producer_proposals.jsonl")
    assert all(row["dual_review_unanimous_pass"] for row in projections)
    encoded = json.dumps(projections)
    assert "intermediate_answer" not in encoded and "final_answers" not in encoded
    assert result["manifest"]["protocol_freeze_manifest_verified"] is True


@pytest.mark.parametrize(
    ("mode", "reject_code", "detail"),
    [
        ("duplicate", "producer_json_parse_error", "duplicate_json_key"),
        ("two_choices", "producer_output_schema_error", "choices_cardinality_not_one"),
        ("length", "producer_output_schema_error", "finish_reason_not_stop"),
    ],
)
def test_strict_response_contract_rejects_without_semantic_retry(tmp_path: Path, mode: str, reject_code: str, detail: str) -> None:
    client = FakeClient(mode)
    result, output = _run(tmp_path, client, rows=1)
    assert len(client.requests) == 1
    assert result["report"]["dual_review_accepted"] == 0
    producer = runner.runner_v1._load_jsonl(output / "semantic_call_ledger.jsonl")[0]
    assert producer["reject_code"] == reject_code
    assert producer["detail_code"] == detail


def test_normalized_any_nonce_check_precedes_model_mismatch(tmp_path: Path) -> None:
    client = FakeClient("nonce_and_model_mismatch")
    _, output = _run(tmp_path, client, rows=1)
    producer = runner.runner_v1._load_jsonl(output / "semantic_call_ledger.jsonl")[0]
    assert producer["reject_code"] == "producer_nonce_echo"
    assert producer["nonce_echo_count"] == 1
    role = runner.runner_v1._load_jsonl(output / "producer_proposals.jsonl")[0]
    assert role["raw_response_content"] is None


@pytest.mark.parametrize(
    ("mode", "reject_code", "detail"),
    [
        ("reasoning_nonce", "producer_nonce_echo", "normalized_nonce_like_token_echo_omitted"),
        ("second_choice_nonce", "producer_nonce_echo", "normalized_nonce_like_token_echo_omitted"),
        ("refusal_chain_secret", "producer_output_schema_error", "forbidden_chain_secret_echo_omitted"),
        ("tool_credential", "producer_output_schema_error", "credential_or_endpoint_echo_omitted"),
        ("metadata_secret", "producer_output_schema_error", "credential_or_endpoint_echo_omitted"),
        ("usage_secret", "producer_output_schema_error", "credential_or_endpoint_echo_omitted"),
    ],
)
def test_recursive_response_scan_omits_leaks_from_every_text_field(
    tmp_path: Path, mode: str, reject_code: str, detail: str
) -> None:
    _, output = _run(tmp_path, FakeClient(mode), rows=1)
    producer = runner.runner_v1._load_jsonl(output / "semantic_call_ledger.jsonl")[0]
    role = runner.runner_v1._load_jsonl(output / "producer_proposals.jsonl")[0]
    assert producer["reject_code"] == reject_code
    assert producer["detail_code"] == detail
    assert role["raw_response_content"] is None
    wal_text = (output / "api_call_wal.jsonl").read_text(encoding="utf-8")
    assert "sk-adversarial123" not in wal_text
    assert "Bridge 0" not in wal_text
    assert "\\u0042ridge 0" not in wal_text


def test_manifest_tamper_fails_before_client_or_output(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, rows=1)
    report = json.loads(inputs["protocol_report"].read_text(encoding="utf-8"))
    report["status"] = "TAMPERED"
    _write_json(inputs["protocol_report"], report)
    client = FakeClient()
    with pytest.raises(ValueError, match="report identity/status|hash/size"):
        runner.run_generation_pilot_v2(
            client=client,
            protocol_path=inputs["protocol"],
            implementation_lock_path=inputs["lock"],
            protocol_report_path=inputs["protocol_report"],
            protocol_manifest_path=inputs["protocol_manifest"],
            v1_supersession_addendum_path=inputs["v1_supersession"],
            identity_path=inputs["identity"],
            metadata_addendum_path=inputs["addendum"],
            raw_path=inputs["raw"],
            output_dir=tmp_path / "output",
            expected_rows=1,
            enforce_formal_locks=False,
        )
    assert client.requests == []
    assert not (tmp_path / "output").exists()


def test_abort_writes_terminal_audit_and_existing_dir_blocks_rerun(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, rows=1)
    output = tmp_path / "output"
    kwargs = dict(
        client=FakeClient("abort"), protocol_path=inputs["protocol"], implementation_lock_path=inputs["lock"],
        protocol_report_path=inputs["protocol_report"], protocol_manifest_path=inputs["protocol_manifest"],
        v1_supersession_addendum_path=inputs["v1_supersession"],
        identity_path=inputs["identity"], metadata_addendum_path=inputs["addendum"], raw_path=inputs["raw"],
        output_dir=output, expected_rows=1, enforce_formal_locks=False,
    )
    with pytest.raises(KeyboardInterrupt):
        runner.run_generation_pilot_v2(**kwargs)
    wal = runner.runner_v1._load_jsonl(output / "api_call_wal.jsonl")
    assert wal[0]["event"] == "run_started"
    assert wal[-1]["event"] == "run_aborted"
    assert wal[-1]["error_type"] == "KeyboardInterrupt"
    assert "error" not in wal[-1] or "message" not in wal[-1]
    assert not (output / "manifest.json").exists()
    with pytest.raises(FileExistsError, match="already exists"):
        runner.run_generation_pilot_v2(**kwargs)


def test_provider_return_then_capture_failure_has_precise_wal_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, rows=1)
    output = tmp_path / "output"

    def fail_capture(response):
        raise KeyboardInterrupt()

    monkeypatch.setattr(runner, "_response_parts", fail_capture)
    with pytest.raises(KeyboardInterrupt):
        runner.run_generation_pilot_v2(
            client=FakeClient(),
            protocol_path=inputs["protocol"],
            implementation_lock_path=inputs["lock"],
            protocol_report_path=inputs["protocol_report"],
            protocol_manifest_path=inputs["protocol_manifest"],
            v1_supersession_addendum_path=inputs["v1_supersession"],
            identity_path=inputs["identity"],
            metadata_addendum_path=inputs["addendum"],
            raw_path=inputs["raw"],
            output_dir=output,
            expected_rows=1,
            enforce_formal_locks=False,
        )
    wal = runner.runner_v1._load_jsonl(output / "api_call_wal.jsonl")
    capture = next(row for row in wal if row["event"] == "response_captured")
    assert capture["capture_complete"] is False
    assert wal[-1]["provider_outcome_boundary"] == "PROVIDER_RETURNED_CAPTURE_INCOMPLETE"
    assert wal[-1]["provider_returned_capture_incomplete_count"] == 1


def test_bounded_scheduler_stops_new_dispatches_after_worker_baseexception(
    tmp_path: Path,
) -> None:
    class InterruptFirstClient(FakeClient):
        physical_calls = 0

        def complete(self, request_body, *, timeout):
            with self._lock:
                self.physical_calls += 1
                index = self.physical_calls
            if index == 1:
                raise KeyboardInterrupt()
            time.sleep(0.02)
            return super().complete(request_body, timeout=timeout)

    inputs = _inputs(tmp_path, rows=8)
    client = InterruptFirstClient()
    with pytest.raises((KeyboardInterrupt, RuntimeError)):
        runner.run_generation_pilot_v2(
            client=client,
            protocol_path=inputs["protocol"],
            implementation_lock_path=inputs["lock"],
            protocol_report_path=inputs["protocol_report"],
            protocol_manifest_path=inputs["protocol_manifest"],
            v1_supersession_addendum_path=inputs["v1_supersession"],
            identity_path=inputs["identity"],
            metadata_addendum_path=inputs["addendum"],
            raw_path=inputs["raw"],
            output_dir=tmp_path / "output",
            expected_rows=8,
            enforce_formal_locks=False,
        )
    # The frozen worker_count is two; no queued backlog may dispatch.
    assert client.physical_calls <= 2


def test_post_dispatch_materialization_failure_is_also_audited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, rows=1)
    output = tmp_path / "output"

    def fail_write(*args, **kwargs):
        raise OSError("intentionally not serialized")

    monkeypatch.setattr(runner, "_atomic_write_jsonl", fail_write)
    with pytest.raises(OSError):
        runner.run_generation_pilot_v2(
            client=FakeClient(), protocol_path=inputs["protocol"],
            implementation_lock_path=inputs["lock"], identity_path=inputs["identity"],
            protocol_report_path=inputs["protocol_report"],
            protocol_manifest_path=inputs["protocol_manifest"],
            v1_supersession_addendum_path=inputs["v1_supersession"],
            metadata_addendum_path=inputs["addendum"], raw_path=inputs["raw"],
            output_dir=output, expected_rows=1, enforce_formal_locks=False,
        )
    wal = runner.runner_v1._load_jsonl(output / "api_call_wal.jsonl")
    assert any(row["event"] == "semantic_calls_and_in_memory_validation_completed" for row in wal)
    assert wal[-1]["event"] == "run_aborted"
    assert wal[-1]["error_type"] == "OSError"
    assert "intentionally" not in json.dumps(wal)
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize(
    "value",
    ["http://api.deepseek.com", "https://evil.example/v1", "https://api.deepseek.com/v2", "https://user@api.deepseek.com/v1", "https://api.deepseek.com/v1?x=1"],
)
def test_endpoint_allowlist_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(ValueError, match="allowlist"):
        runner.validate_deepseek_endpoint(value)


@pytest.mark.parametrize("value", ["https://api.deepseek.com", "https://api.deepseek.com/", "https://api.deepseek.com/v1", "https://api.deepseek.com:443/v1/"])
def test_endpoint_allowlist_accepts_only_frozen_origin(value: str) -> None:
    runner.validate_deepseek_endpoint(value)


def test_v2_sdk_client_disables_hidden_retries_and_does_not_expose_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-value")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    client = runner.OpenAICompletionClientV2()
    assert captured["max_retries"] == 0
    assert captured["base_url"] == "https://api.deepseek.com/v1"
    assert not hasattr(client, "base_url")
    assert not hasattr(client, "api_key")


def test_strict_json_rejects_nested_duplicate_keys() -> None:
    with pytest.raises(runner.DuplicateJSONKeyError):
        runner.strict_json_object('{"outer":{"x":1,"x":2}}')
