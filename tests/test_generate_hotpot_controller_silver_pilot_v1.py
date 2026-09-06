"""CPU-only tests for the frozen Hotpot pilot generation runner."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading

import pytest

from kgproweight.data import hotpot_controller_silver as silver
from scripts.prepare import freeze_hotpot_controller_silver_execution_v1 as freeze
from scripts.prepare import generate_hotpot_controller_silver_pilot_v1 as runner


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _raw(index: int, *, invalid: bool = False) -> dict:
    root = f"Root {index}"
    bridge = f"Bridge {index}"
    final = f"Final {index}"
    question = f"What requested attribute belongs to the organization linked with {root}?"
    if invalid:
        question = f"What requested attribute links {root} and {bridge}?"
    return {
        "id": f"qid-{index}",
        "question": question,
        "golden_answers": [final],
        "metadata": {
            "type": "bridge",
            "level": "medium",
            "supporting_facts": {"title": [root, bridge], "sent_id": [0, 0]},
            "context": {
                "title": [root, bridge],
                "sentences": [
                    [f"{root} is directly linked with {bridge}."],
                    [f"{bridge} has the requested attribute {final}."],
                ],
            },
        },
    }


def _all_pass_review() -> dict:
    return {
        "schema_version": freeze.REVIEW_SCHEMA_VERSION,
        **{field: True for field in freeze.REVIEW_BOOLEAN_FIELDS},
        "verdict": "pass",
        "reject_codes": [],
    }


class FakeClient:
    def __init__(self, *, producer_mode: str = "valid", fail_first_transport: bool = False) -> None:
        self.producer_mode = producer_mode
        self.fail_first_transport = fail_first_transport
        self.requests: list[dict] = []
        self.thread_ids: set[int] = set()
        self._lock = threading.Lock()
        self._failed = False

    def complete(self, request_body, *, timeout):
        with self._lock:
            self.requests.append(deepcopy(dict(request_body)))
            self.thread_ids.add(threading.get_ident())
            system = request_body["messages"][0]["content"]
            if self.fail_first_transport and "You create retrieval-query" in system and not self._failed:
                self._failed = True
                raise runner.SyntheticTransportError("timeout")
        model = request_body["model"]
        if "You create retrieval-query" in system:
            if self.producer_mode == "invalid_json":
                content = "not json"
            else:
                user = request_body["messages"][1]["content"]
                payload = json.loads(user[user.index("{") :])
                root = payload["root_document_title"]
                if self.producer_mode == "nonce_echo":
                    root = runner._NONCE_RE.search(user).group(0)
                content = json.dumps(
                    {
                        "schema_version": silver.PROPOSAL_SCHEMA_VERSION,
                        "q1": f"Which organization is {root} directly linked with?",
                        "q2_template": "What requested attribute does #1 have?",
                    }
                )
        else:
            content = json.dumps(_all_pass_review())
        return {
            "http_status": 200,
            "model": model,
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }


def _inputs(tmp_path: Path, *, invalid_second: bool = False) -> dict[str, Path]:
    rows = [_raw(0), _raw(1, invalid=invalid_second)]
    raw = tmp_path / "raw.jsonl"
    _write_jsonl(raw, rows)
    identity = tmp_path / "identity.jsonl"
    _write_jsonl(
        identity,
        [{"dataset": "hotpotqa", "qid": row["id"], "question": row["question"]} for row in rows],
    )
    parent_lock = {
        "rows": 2,
        "artifacts": {name: {} for name in freeze.PARENT_FILENAMES},
        "identity_sha256": _sha256(identity),
    }
    # build_protocol is intentionally pilot30-only; patch only the synthetic
    # parent row count after constructing the otherwise exact frozen contract.
    protocol = freeze.build_protocol(
        generated_at_utc="2026-09-04T00:00:00+00:00",
        parent_lock={"rows": 30, "artifacts": {name: {} for name in freeze.PARENT_FILENAMES}},
    )
    protocol["parent_identity_freeze"] = parent_lock
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, protocol)
    builder_path = Path(runner.PROJECT_ROOT) / "kgproweight/data/hotpot_controller_silver.py"
    precall = 1 if invalid_second else 0
    addendum = {
        "selection_result_changed": False,
        "precall_hardening_audit": {
            "builder_version": silver.BUILDER_VERSION,
            "builder_sha256": _sha256(builder_path),
            "fixed_denominator": 2,
            "precall_rejected": precall,
        },
    }
    addendum_path = tmp_path / "metadata_addendum.json"
    _write_json(addendum_path, addendum)
    return {"raw": raw, "identity": identity, "protocol": protocol_path, "addendum": addendum_path}


def _run(tmp_path: Path, client: FakeClient, *, invalid_second: bool = False):
    paths = _inputs(tmp_path, invalid_second=invalid_second)
    output = tmp_path / "output"
    result = runner.run_generation_pilot(
        client=client,
        protocol_path=paths["protocol"],
        identity_path=paths["identity"],
        metadata_addendum_path=paths["addendum"],
        raw_path=paths["raw"],
        output_dir=output,
        expected_rows=2,
        enforce_formal_locks=False,
    )
    return result, output


def test_happy_path_accounts_all_rows_isolates_contexts_and_binds_actions(tmp_path: Path) -> None:
    client = FakeClient()
    result, output = _run(tmp_path, client)
    assert {path.name for path in output.iterdir()} == set(runner.ALL_OUTPUT_FILES)
    assert result["report"]["fixed_denominator"] == 2
    assert result["report"]["dual_review_accepted"] == 2
    assert result["report"]["semantic_call_rows"] == 6
    assert result["report"]["transport_attempt_rows"] == 6
    assert result["report"]["scientific_boundary"]["retrieval_or_reader_calls"] == 0

    semantic = runner._load_jsonl(output / "semantic_call_ledger.jsonl")
    assert [row["stage"] for row in semantic] == list(runner.STAGES) * 2
    assert all(row["status"] == "accepted" for row in semantic)
    assert all(row["nonce_echo_count"] == 0 for row in semantic)
    actions = runner._load_jsonl(output / "accepted_actions.jsonl")
    assert len(actions) == 4
    for offset in range(0, 4, 2):
        assert actions[offset]["source_provenance"]["proposal_sha256"] == actions[offset + 1]["source_provenance"]["proposal_sha256"]
        assert actions[offset + 1]["target"]["dependencies"] == ["q1"]
    projections = runner._load_jsonl(output / "producer_proposals.jsonl")
    assert len(projections) == 2
    assert all(row["dual_review_unanimous_pass"] is True for row in projections)
    assert all(row["final_item_status"] == "accepted_generation_and_dual_review" for row in projections)
    assert all(row["question"] and row["q1_query"] and row["q2_template"] for row in projections)
    assert all(row["proposal_sha256"] == row["parsed_response_sha256"] for row in projections)
    assert all(row["runtime_projection_gold_or_observation_fields_present"] is False for row in projections)
    serialized_projections = json.dumps(projections, ensure_ascii=False)
    assert "intermediate_answer" not in serialized_projections
    assert "final_answers" not in serialized_projections
    assert "verified_observations" not in serialized_projections

    producer_requests = [item for item in client.requests if item["model"] == "deepseek-v4-flash" and "You create retrieval-query" in item["messages"][0]["content"]]
    blind_requests = [item for item in client.requests if "blind structural reviewer" in item["messages"][0]["content"]]
    gold_requests = [item for item in client.requests if "train-Gold-aware adjudicator" in item["messages"][0]["content"]]
    assert len(producer_requests) == len(blind_requests) == len(gold_requests) == 2
    for request in producer_requests + blind_requests:
        encoded = json.dumps(request, ensure_ascii=False)
        assert "qid-" not in encoded and '"dataset"' not in encoded
        assert "Bridge 0" not in encoded and "Final 0" not in encoded
    for request in gold_requests:
        encoded = json.dumps(request, ensure_ascii=False)
        assert "qid-" not in encoded and '"dataset"' not in encoded
        assert "intermediate_answer" in encoded and "final_answers" in encoded
        assert "reviewer_1" not in encoded


def test_precall_failure_is_retained_and_makes_zero_calls_for_that_identity(tmp_path: Path) -> None:
    client = FakeClient()
    result, output = _run(tmp_path, client, invalid_second=True)
    assert result["report"]["precall_rejected"] == 1
    assert result["report"]["fixed_denominator"] == 2
    assert len(client.requests) == 3
    semantic = runner._load_jsonl(output / "semantic_call_ledger.jsonl")
    skipped = [row for row in semantic if row["qid"] == "qid-1"]
    assert len(skipped) == 3
    assert all(row["status"] == "not_executed_upstream_failure" for row in skipped)
    assert all(row["transport_attempt_count"] == 0 for row in skipped)
    assert len(runner._load_jsonl(output / "failures.jsonl")) == 1


def test_retry_is_transport_only_and_request_body_is_identical(tmp_path: Path) -> None:
    client = FakeClient(fail_first_transport=True)
    result, output = _run(tmp_path, client)
    assert result["report"]["dual_review_accepted"] == 2
    transport = runner._load_jsonl(output / "api_transport_attempt_ledger.jsonl")
    producer_first = [row for row in transport if row["stage"] == "producer" and row["physical_attempt_index"] in {1, 2}]
    retried_request = next(row["semantic_request_id"] for row in producer_first if row["physical_attempt_index"] == 2)
    same_request = [row for row in transport if row["semantic_request_id"] == retried_request]
    assert len(same_request) == 2
    assert len({row["request_body_sha256"] for row in same_request}) == 1
    assert all(row["request_bytes_identical_to_attempt_1"] is True for row in same_request)


def test_invalid_content_is_not_retried_and_reviews_are_skipped(tmp_path: Path) -> None:
    client = FakeClient(producer_mode="invalid_json")
    result, output = _run(tmp_path, client)
    assert len(client.requests) == 2
    assert result["report"]["producer_accepted"] == 0
    assert result["report"]["dual_review_accepted"] == 0
    semantic = runner._load_jsonl(output / "semantic_call_ledger.jsonl")
    assert [row["reject_code"] for row in semantic if row["stage"] == "producer"] == [
        "producer_json_parse_error", "producer_json_parse_error"
    ]
    assert all(row["status"] == "not_executed_upstream_failure" for row in semantic if row["stage"] != "producer")


def test_nonce_echo_is_rejected_without_content_retry(tmp_path: Path) -> None:
    client = FakeClient(producer_mode="nonce_echo")
    result, output = _run(tmp_path, client)
    assert len(client.requests) == 2
    assert result["report"]["dual_review_accepted"] == 0
    assert result["report"]["gates"]["all_model_responses_nonce_echo_free"] is False
    semantic = runner._load_jsonl(output / "semantic_call_ledger.jsonl")
    producers = [row for row in semantic if row["stage"] == "producer"]
    assert all(row["reject_code"] == "producer_nonce_echo" for row in producers)
    assert all(row["nonce_echo_count"] == 1 for row in producers)


def test_append_only_and_default_cli_safety_latch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    _, output = _run(tmp_path, client)
    paths = _inputs(tmp_path / "second")
    with pytest.raises(FileExistsError, match="append-only"):
        runner.run_generation_pilot(
            client=client,
            protocol_path=paths["protocol"], identity_path=paths["identity"],
            metadata_addendum_path=paths["addendum"], raw_path=paths["raw"],
            output_dir=output, expected_rows=2, enforce_formal_locks=False,
        )
    monkeypatch.setattr("sys.argv", ["generate_hotpot_controller_silver_pilot_v1.py"])
    with pytest.raises(SystemExit, match="No API call made"):
        runner.main()


def test_review_schema_enforces_boolean_code_and_pass_invariants() -> None:
    protocol = freeze.build_protocol(
        generated_at_utc="2026-09-04T00:00:00+00:00",
        parent_lock={"rows": 30, "artifacts": {name: {} for name in freeze.PARENT_FILENAMES}},
    )
    runner._validate_review_schema(_all_pass_review(), protocol)
    bad = _all_pass_review()
    bad["q1_single_hop"] = False
    bad["verdict"] = "reject"
    with pytest.raises(ValueError, match="boolean_code_invariant"):
        runner._validate_review_schema(bad, protocol)
    bad["reject_codes"] = ["q1_not_single_hop"]
    runner._validate_review_schema(bad, protocol)
