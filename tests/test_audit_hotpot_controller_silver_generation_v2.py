"""CPU-only tests for the independent Hotpot generation-V2 artifact audit."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import threading

import pytest

from kgproweight.data import hotpot_controller_silver as silver
from scripts.diagnose import audit_hotpot_controller_silver_generation_v2 as audit
from scripts.prepare import freeze_hotpot_controller_silver_execution_v1 as freeze_v1
from scripts.prepare import freeze_hotpot_controller_silver_execution_v2 as freeze_v2
from scripts.prepare import generate_hotpot_controller_silver_pilot_v2 as runner


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _raw(index: int, *, valid: bool = True) -> dict:
    root, bridge, final = f"Root {index}", f"Bridge {index}", f"Final {index}"
    return {
        "id": f"qid-{index:02d}",
        "question": f"What requested attribute belongs to the organization linked with {root}?",
        "golden_answers": [final],
        "metadata": {
            "type": "bridge" if valid else "comparison",
            "level": "medium",
            "supporting_facts": {"title": [root, bridge], "sent_id": [0, 0]},
            "context": {
                "title": [root, bridge],
                "sentences": [
                    [f"{root} links directly to {bridge}."],
                    [[f"{bridge} has requested attribute {final}."][0]],
                ],
            },
        },
    }


def _pass_review() -> dict:
    return {
        "schema_version": freeze_v1.REVIEW_SCHEMA_VERSION,
        **{field: True for field in freeze_v1.REVIEW_BOOLEAN_FIELDS},
        "verdict": "pass",
        "reject_codes": [],
    }


class _FakeFormalClient(runner.OpenAICompletionClientV2):
    """No-network client that still exercises the formal-client type gate."""

    def __init__(self, *, reject_producers: int = 0) -> None:
        self.reject_producers = reject_producers
        self.producer_calls = 0
        self._lock = threading.Lock()

    def complete(self, request_body, *, timeout):
        del timeout
        system = request_body["messages"][0]["content"]
        model = request_body["model"]
        if "You create retrieval-query" in system:
            with self._lock:
                self.producer_calls += 1
                call_number = self.producer_calls
            user = request_body["messages"][1]["content"]
            payload = json.loads(user[user.index("{") :])
            root = payload["root_document_title"]
            if call_number <= self.reject_producers:
                content = json.dumps(
                    {
                        "schema_version": silver.PROPOSAL_SCHEMA_VERSION,
                        "q1": "This is deliberately not a question",
                        "q2_template": "What requested attribute does #1 have?",
                    }
                )
            else:
                content = json.dumps(
                    {
                        "schema_version": silver.PROPOSAL_SCHEMA_VERSION,
                        "q1": f"Which organization is {root} directly linked with?",
                        "q2_template": "What requested attribute does #1 have?",
                    }
                )
        else:
            content = json.dumps(_pass_review())
        return {
            "http_status": 200,
            "model": model,
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }


def _build_bundle(root: Path, *, reject_producers: int = 0) -> dict[str, Path]:
    protocol_dir = root / "protocol"
    protocol_dir.mkdir(parents=True)
    raw_rows = [_raw(index, valid=index != 29) for index in range(30)]
    raw_path = root / "raw.jsonl"
    identity_path = root / "identity.jsonl"
    addendum_path = root / "addendum.json"
    _write_jsonl(raw_path, raw_rows)
    _write_jsonl(
        identity_path,
        [
            {"dataset": "hotpotqa", "qid": row["id"], "question": row["question"]}
            for row in raw_rows
        ],
    )
    builder_path = runner.PROJECT_ROOT / "kgproweight/data/hotpot_controller_silver.py"
    _write_json(
        addendum_path,
        {
            "selection_result_changed": False,
            "precall_hardening_audit": {
                "builder_version": silver.BUILDER_VERSION,
                "builder_sha256": _sha(builder_path),
                "fixed_denominator": 30,
                "precall_rejected": 1,
            },
        },
    )
    implementation_fixture = root / "locked_implementation_fixture.py"
    implementation_fixture.write_text("# immutable test implementation\n", encoding="utf-8")
    lock = {
        "schema_version": freeze_v2.LOCK_SCHEMA_VERSION,
        "experiment_id": freeze_v2.EXPERIMENT_ID,
        "implementations": [
            {
                "path": str(implementation_fixture),
                "sha256": _sha(implementation_fixture),
                "size_bytes": implementation_fixture.stat().st_size,
            }
        ],
    }
    lock_path = protocol_dir / "implementation_lock.json"
    _write_json(lock_path, lock)
    parent = {
        "rows": 30,
        "artifacts": {name: {} for name in freeze_v1.PARENT_FILENAMES},
        "identity_sha256": _sha(identity_path),
    }
    diagnostic = {
        "denominator": 30,
        "precall_constructible": 29,
        "precall_rejected": 1,
    }
    protocol = freeze_v2.build_protocol(
        generated_at_utc="2026-09-05T00:00:00+00:00",
        parent_lock=parent,
        implementation_lock_path=str(lock_path),
        implementation_lock_sha256=_sha(lock_path),
        implementation_lock_size_bytes=lock_path.stat().st_size,
        v1_to_v2_diagnostic=diagnostic,
    )
    protocol_path = protocol_dir / "protocol.json"
    _write_json(protocol_path, protocol)
    freeze_report_path = protocol_dir / "report.json"
    _write_json(
        freeze_report_path,
        {
            "schema_version": freeze_v2.REPORT_SCHEMA_VERSION,
            "experiment_id": freeze_v2.EXPERIMENT_ID,
            "status": freeze_v2.STATUS,
            "api_calls": 0,
        },
    )
    protocol_outputs = []
    for name in ("implementation_lock.json", "protocol.json", "report.json"):
        path = protocol_dir / name
        protocol_outputs.append(
            {"path": name, "sha256": _sha(path), "size_bytes": path.stat().st_size}
        )
    supersession_path = root / "supersession.json"
    protocol_manifest_path = protocol_dir / "manifest.json"
    _write_json(
        protocol_manifest_path,
        {
            "schema_version": freeze_v2.MANIFEST_SCHEMA_VERSION,
            "experiment_id": freeze_v2.EXPERIMENT_ID,
            "status": freeze_v2.STATUS,
            "diagnostic_inputs": {
                "identity": {"path": str(identity_path), "sha256": _sha(identity_path)},
                "raw_train": {"path": str(raw_path), "sha256": _sha(raw_path)},
                "parent_metadata_addendum_v1_1": {
                    "path": str(addendum_path),
                    "sha256": _sha(addendum_path),
                },
            },
            "protocol_freeze_output_set_exact": list(
                freeze_v2.PROTOCOL_FREEZE_OUTPUT_FILES
            ),
            "api_calls": 0,
            "outputs": protocol_outputs,
            "v1_inputs": [
                {
                    "path": str(runner.PROJECT_ROOT / freeze_v2.V1_DIR / name),
                    "sha256": _sha(runner.PROJECT_ROOT / freeze_v2.V1_DIR / name),
                }
                for name in ("protocol.json", "report.json", "manifest.json")
            ],
            "external_append_only_artifact": {
                "path": str(supersession_path),
                "commit_order": "after_complete_v2_manifest_directory_commit",
                "status_at_v2_manifest_commit": "PENDING_APPEND_ONLY_COMMIT",
                "must_bind_v2_manifest_sha256": True,
            },
            "training_started": False,
        },
    )
    _write_json(
        supersession_path,
        {
            "schema_version": freeze_v2.SUPERSESSION_SCHEMA_VERSION,
            "append_only": True,
            "v1_experiment_id": freeze_v1.EXPERIMENT_ID,
            "v1_protocol_sha256": _sha(
                runner.PROJECT_ROOT / freeze_v2.V1_DIR / "protocol.json"
            ),
            "v1_api_calls": 0,
            "v1_generation_output_existed_at_supersession": False,
            "status": "SUPERSEDED_BEFORE_ANY_API_CALL",
            "superseded_by": {
                "experiment_id": freeze_v2.EXPERIMENT_ID,
                "protocol_path": str(protocol_path),
                "manifest_path": str(protocol_manifest_path),
                "protocol_sha256": _sha(protocol_path),
                "manifest_sha256": _sha(protocol_manifest_path),
            },
        },
    )
    generation_dir = root / "generation"
    runner.run_generation_pilot_v2(
        client=_FakeFormalClient(reject_producers=reject_producers),
        project_root=runner.PROJECT_ROOT,
        protocol_path=protocol_path,
        implementation_lock_path=lock_path,
        protocol_report_path=freeze_report_path,
        protocol_manifest_path=protocol_dir / "manifest.json",
        v1_supersession_addendum_path=supersession_path,
        identity_path=identity_path,
        metadata_addendum_path=addendum_path,
        raw_path=raw_path,
        output_dir=generation_dir,
        expected_rows=30,
        enforce_formal_locks=False,
    )
    return {
        "protocol": protocol_path,
        "protocol_manifest": protocol_dir / "manifest.json",
        "generation": generation_dir,
    }


@pytest.fixture(scope="module")
def baseline_bundle(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return _build_bundle(tmp_path_factory.mktemp("hotpot-v2-audit-baseline"))


def _copy_generation(tmp_path: Path, bundle: dict[str, Path]) -> Path:
    destination = tmp_path / "generation"
    shutil.copytree(bundle["generation"], destination)
    return destination


def _rehash(generation: Path, *filenames: str) -> None:
    manifest = _load_json(generation / "manifest.json")
    entries = {item["path"]: item for item in manifest["outputs"]}
    for filename in filenames:
        path = generation / filename
        entries[filename]["sha256"] = _sha(path)
        entries[filename]["size_bytes"] = path.stat().st_size
    _write_json(generation / "manifest.json", manifest)


def _audit(bundle: dict[str, Path], generation: Path | None = None) -> dict:
    return audit.audit_generation_v2(
        project_root=runner.PROJECT_ROOT,
        protocol_path=bundle["protocol"],
        protocol_manifest_path=bundle["protocol_manifest"],
        generation_dir=generation or bundle["generation"],
    )


def test_complete_fake_artifact_passes_every_gate(baseline_bundle: dict[str, Path]) -> None:
    report = _audit(baseline_bundle)
    assert report["fixed_denominator"] == 30
    assert report["precall_constructible"] == 29
    assert report["precall_rejected"] == 1
    assert report["semantic_slots_including_skipped"] == 90
    assert report["accepted_identities"] == 29
    assert all(report["gates"].values())


def test_semantic_slot_loss_fails_closed(tmp_path: Path, baseline_bundle: dict[str, Path]) -> None:
    generation = _copy_generation(tmp_path, baseline_bundle)
    path = generation / "semantic_call_ledger.jsonl"
    _write_jsonl(path, _load_jsonl(path)[:-1])
    _rehash(generation, path.name)
    with pytest.raises(audit.GenerationV2AuditError, match="semantic_slot_count_not_90"):
        _audit(baseline_bundle, generation)


def test_wal_result_without_pair_fails_closed(tmp_path: Path, baseline_bundle: dict[str, Path]) -> None:
    generation = _copy_generation(tmp_path, baseline_bundle)
    path = generation / "api_call_wal.jsonl"
    rows = _load_jsonl(path)
    removed = next(index for index, row in enumerate(rows) if row["event"] == "result")
    rows.pop(removed)
    _write_jsonl(path, rows)
    report_path = generation / "report.json"
    report = _load_json(report_path)
    report["wal_event_rows"] = len(rows)
    _write_json(report_path, report)
    _rehash(generation, path.name, report_path.name)
    with pytest.raises(audit.GenerationV2AuditError, match="wal_intent_result_transport_conservation_failed"):
        _audit(baseline_bundle, generation)


def test_wal_received_response_without_capture_fails_closed(
    tmp_path: Path, baseline_bundle: dict[str, Path]
) -> None:
    generation = _copy_generation(tmp_path, baseline_bundle)
    path = generation / "api_call_wal.jsonl"
    rows = _load_jsonl(path)
    removed = next(
        index for index, row in enumerate(rows) if row["event"] == "response_captured"
    )
    rows.pop(removed)
    _write_jsonl(path, rows)
    report_path = generation / "report.json"
    report = _load_json(report_path)
    report["wal_event_rows"] = len(rows)
    _write_json(report_path, report)
    _rehash(generation, path.name, report_path.name)
    with pytest.raises(
        audit.GenerationV2AuditError,
        match="wal_received_response_capture_missing|wal_response_capture_conservation_failed",
    ):
        _audit(baseline_bundle, generation)


def test_wal_abort_is_never_accepted(tmp_path: Path, baseline_bundle: dict[str, Path]) -> None:
    generation = _copy_generation(tmp_path, baseline_bundle)
    path = generation / "api_call_wal.jsonl"
    rows = _load_jsonl(path)
    rows.insert(
        -1,
        {
            "schema_version": audit.WAL_SCHEMA_VERSION,
            "experiment_id": audit.EXPERIMENT_ID,
            "event": "run_aborted",
            "at_utc": "2026-09-05T00:00:00Z",
            "error_type": "Synthetic",
            "unmatched_intent_count": 0,
            "provider_outcome_boundary": "NO_UNMATCHED_DISPATCH",
        },
    )
    _write_jsonl(path, rows)
    report_path = generation / "report.json"
    report = _load_json(report_path)
    report["wal_event_rows"] = len(rows)
    _write_json(report_path, report)
    _rehash(generation, path.name, report_path.name)
    with pytest.raises(audit.GenerationV2AuditError, match="wal_abort_present"):
        _audit(baseline_bundle, generation)


def test_response_capture_mismatch_fails_closed(tmp_path: Path, baseline_bundle: dict[str, Path]) -> None:
    generation = _copy_generation(tmp_path, baseline_bundle)
    transport_path = generation / "api_transport_attempt_ledger.jsonl"
    transport = _load_jsonl(transport_path)
    target = transport[0]
    target["response_received"] = False
    _write_jsonl(transport_path, transport)
    wal_path = generation / "api_call_wal.jsonl"
    wal = _load_jsonl(wal_path)
    for row in wal:
        if (
            row.get("event") == "result"
            and row.get("semantic_request_id") == target["semantic_request_id"]
            and row.get("physical_attempt_index") == target["physical_attempt_index"]
        ):
            row["response_received"] = False
    _write_jsonl(wal_path, wal)
    _rehash(generation, transport_path.name, wal_path.name)
    with pytest.raises(audit.GenerationV2AuditError, match="unreceived_response_not_exhausted"):
        _audit(baseline_bundle, generation)


def test_accepted_action_must_be_strict_pair(tmp_path: Path, baseline_bundle: dict[str, Path]) -> None:
    generation = _copy_generation(tmp_path, baseline_bundle)
    path = generation / "accepted_actions.jsonl"
    _write_jsonl(path, _load_jsonl(path)[:-1])
    _rehash(generation, path.name)
    with pytest.raises(audit.GenerationV2AuditError, match="accepted_action_cardinality_mismatch"):
        _audit(baseline_bundle, generation)


def test_qid_order_drift_fails_closed(tmp_path: Path, baseline_bundle: dict[str, Path]) -> None:
    generation = _copy_generation(tmp_path, baseline_bundle)
    path = generation / "reviewer_1_reviews.jsonl"
    rows = _load_jsonl(path)
    rows[0], rows[1] = rows[1], rows[0]
    _write_jsonl(path, rows)
    _rehash(generation, path.name)
    with pytest.raises(audit.GenerationV2AuditError, match="role_qid_order_mismatch"):
        _audit(baseline_bundle, generation)


def test_question_hash_drift_fails_closed(tmp_path: Path, baseline_bundle: dict[str, Path]) -> None:
    generation = _copy_generation(tmp_path, baseline_bundle)
    path = generation / "reviewer_2_reviews.jsonl"
    rows = _load_jsonl(path)
    rows[0]["question_sha256"] = "0" * 64
    _write_jsonl(path, rows)
    _rehash(generation, path.name)
    with pytest.raises(audit.GenerationV2AuditError, match="role_question_hash_mismatch"):
        _audit(baseline_bundle, generation)


def test_role_cross_field_isolation_fails_closed(tmp_path: Path, baseline_bundle: dict[str, Path]) -> None:
    generation = _copy_generation(tmp_path, baseline_bundle)
    path = generation / "reviewer_1_reviews.jsonl"
    rows = _load_jsonl(path)
    rows[0]["reviewer_2_response"] = {"verdict": "pass"}
    _write_jsonl(path, rows)
    _rehash(generation, path.name)
    with pytest.raises(audit.GenerationV2AuditError, match="role_schema_or_isolation_mismatch"):
        _audit(baseline_bundle, generation)


def test_normalized_nonce_leak_fails_closed(tmp_path: Path, baseline_bundle: dict[str, Path]) -> None:
    generation = _copy_generation(tmp_path, baseline_bundle)
    path = generation / "producer_proposals.jsonl"
    rows = _load_jsonl(path)
    rows[0]["raw_response_content"] += " Ｎ" + "Ａ" * 31
    _write_jsonl(path, rows)
    _rehash(generation, path.name)
    with pytest.raises(audit.GenerationV2AuditError, match="nonce_token_serialized_in_role_output"):
        _audit(baseline_bundle, generation)


def test_chain_secret_response_leak_fails_closed(tmp_path: Path, baseline_bundle: dict[str, Path]) -> None:
    generation = _copy_generation(tmp_path, baseline_bundle)
    path = generation / "reviewer_2_reviews.jsonl"
    rows = _load_jsonl(path)
    rows[0]["raw_response_content"] += " Bridge 0"
    _write_jsonl(path, rows)
    _rehash(generation, path.name)
    with pytest.raises(audit.GenerationV2AuditError, match="chain_secret_serialized_in_role_response"):
        _audit(baseline_bundle, generation)


def test_unbound_file_change_fails_hash_closure(tmp_path: Path, baseline_bundle: dict[str, Path]) -> None:
    generation = _copy_generation(tmp_path, baseline_bundle)
    path = generation / "failures.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(audit.GenerationV2AuditError, match="generation_output_hash_mismatch"):
        _audit(baseline_bundle, generation)


def test_manifest_fixed_denominator_drift_fails_closed(tmp_path: Path, baseline_bundle: dict[str, Path]) -> None:
    generation = _copy_generation(tmp_path, baseline_bundle)
    manifest_path = generation / "manifest.json"
    manifest = _load_json(manifest_path)
    manifest["fixed_denominator"] = 29
    _write_json(manifest_path, manifest)
    with pytest.raises(audit.GenerationV2AuditError, match="manifest_fixed30_mismatch"):
        _audit(baseline_bundle, generation)


def test_accepted_below_24_run_is_rejected(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path / "below24", reject_producers=6)
    with pytest.raises(
        audit.GenerationV2AuditError,
        match="generation_(manifest|report)_identity_mismatch|accepted_below_24",
    ):
        _audit(bundle)
