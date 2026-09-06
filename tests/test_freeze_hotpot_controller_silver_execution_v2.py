"""CPU-only tests for the repaired Hotpot silver execution protocol V2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare import freeze_hotpot_controller_silver_execution_v1 as v1
from scripts.prepare import freeze_hotpot_controller_silver_execution_v2 as v2


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _raw(index: int, *, implicit: bool = False, invalid: bool = False) -> dict:
    root, bridge, final = f"Root {index}", f"Bridge {index}", f"Final {index}"
    question = f"What attribute belongs to the organization linked with {root}?"
    if invalid:
        question = f"What attribute links {root} and {bridge}?"
    second = f"It has the requested attribute {final}." if implicit else f"{bridge} has the requested attribute {final}."
    return {
        "id": f"qid-{index}", "question": question, "golden_answers": [final],
        "metadata": {
            "type": "bridge", "level": "medium",
            "supporting_facts": {"title": [root, bridge], "sent_id": [0, 0]},
            "context": {"title": [root, bridge], "sentences": [[f"{root} links to {bridge}."], [second]]},
        },
    }


def test_v1_to_v2_diagnostic_recomputes_explicit_implicit_reject_and_hash_join(tmp_path: Path) -> None:
    rows = [_raw(0), _raw(1, implicit=True), _raw(2, invalid=True)]
    raw = tmp_path / "raw.jsonl"
    identity = tmp_path / "identity.jsonl"
    _write_jsonl(raw, rows)
    _write_jsonl(identity, [{"dataset": "hotpotqa", "qid": row["id"], "question": row["question"]} for row in rows])
    audit = v2.audit_v1_to_v2_subject_binding(identity_path=identity, raw_path=raw)
    assert audit["denominator"] == 3
    assert audit["identity_raw_qid_and_question_hash_join"] == 3
    assert audit["precall_constructible"] == 2
    assert audit["precall_rejected"] == 1
    assert audit["v1_second_hop_subject_explicit"] == 1
    assert audit["v1_second_hop_subject_implicit"] == 1
    assert audit["v2_subject_binding_pass"] == 2
    assert audit["v2_secret_residual"] == 0


def test_protocol_freezes_v2_binding_validation_wal_endpoint_and_postbuild_code() -> None:
    parent = {"rows": 30, "artifacts": {name: {} for name in v1.PARENT_FILENAMES}}
    protocol = v2.build_protocol(
        generated_at_utc="2026-09-05T00:00:00+00:00",
        parent_lock=parent,
        implementation_lock_path="lock.json",
        implementation_lock_sha256="a" * 64,
        implementation_lock_size_bytes=123,
        v1_to_v2_diagnostic={"denominator": 30},
    )
    fields = protocol["masking"]["producer_safe_payload_fields_exact"]
    assert fields[-1] == "second_hop_subject_nonce"
    assert protocol["masking"]["second_hop_subject_nonce_must_equal_first_hop_intermediate_nonce"] is True
    assert protocol["masking"]["response_text_scan_scope"] == "all_recursive_string_values_in_provider_response"
    assert "surface may be implicit" in protocol["producer"]["prompt"]["system"]
    validation = protocol["response_validation_v2"]
    assert validation["choices_count_exact"] == 1
    assert validation["finish_reason_exact"] == "stop"
    assert validation["json_duplicate_keys_allowed"] is False
    assert protocol["api_execution"]["wal_intent_fsync_before_each_physical_call"] is True
    assert protocol["api_execution"]["wal_response_capture_fsync_immediately_after_provider_return"] is True
    assert protocol["api_execution"]["scheduler"]["submission_policy"] == "bounded_at_worker_count"
    assert protocol["api_execution"]["wal_success_terminal_event_exact"] == "semantic_calls_and_in_memory_validation_completed"
    assert protocol["api_execution"]["wal_run_completed_event_allowed"] is False
    endpoint = protocol["api_execution"]["endpoint_allowlist"]
    assert endpoint["scheme_exact"] == "https"
    assert endpoint["hostname_exact"] == "api.deepseek.com"
    assert "action_pair_postbuild_reject" in protocol["rejection_contract"]["item_reject_code_enum_exact"]
    assert protocol["implementation_lock"]["protocol_runner_self_reference"] is False
    assert protocol["v1_to_v2_subject_binding_diagnostic"] == {"denominator": 30}


def test_implementation_lock_binds_runner_without_binding_itself_or_protocol() -> None:
    lock = v2.build_implementation_lock(
        project_root=v2.PROJECT_ROOT, generated_at_utc="2026-09-05T00:00:00+00:00"
    )
    paths = {item["path"] for item in lock["implementations"]}
    assert "scripts/prepare/generate_hotpot_controller_silver_pilot_v2.py" in paths
    assert "scripts/prepare/freeze_hotpot_controller_silver_execution_v2.py" in paths
    assert not any(path.endswith("protocol.json") or path.endswith("implementation_lock.json") for path in paths)
    assert "does not bind itself" in lock["self_reference_avoided_by"]


def test_freezer_writes_new_protocol_and_append_only_v1_supersession_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    v1_dir = tmp_path / "v1"
    v1_dir.mkdir()
    parent = {"rows": 30, "artifacts": {name: {} for name in v1.PARENT_FILENAMES}}
    v1_protocol = v1.build_protocol(
        generated_at_utc="2026-09-04T00:00:00+00:00", parent_lock=parent
    )
    v1_protocol_path = v1_dir / "protocol.json"
    v1_report_path = v1_dir / "report.json"
    v1_protocol_path.write_text(json.dumps(v1_protocol), encoding="utf-8")
    v1_report_path.write_text(
        json.dumps(
            {
                "schema_version": v1.REPORT_SCHEMA_VERSION,
                "experiment_id": v1.EXPERIMENT_ID,
                "status": v1.STATUS,
                "api_calls": 0,
                "parent_identity_freeze": parent,
            }
        ),
        encoding="utf-8",
    )
    (v1_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": v1.MANIFEST_SCHEMA_VERSION,
                "experiment_id": v1.EXPERIMENT_ID,
                "status": v1.STATUS,
                "api_calls": 0,
                "training_started": False,
                "inputs": {"parent_identity_freeze": parent},
                "outputs": [
                    {
                        "path": path.name,
                        "sha256": v2._sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in (v1_protocol_path, v1_report_path)
                ],
            }
        ),
        encoding="utf-8",
    )
    raw_rows = [_raw(0), _raw(1, implicit=True), _raw(2, invalid=True)]
    raw = tmp_path / "raw.jsonl"
    identity = tmp_path / "identity.jsonl"
    _write_jsonl(raw, raw_rows)
    _write_jsonl(identity, [{"dataset": "hotpotqa", "qid": row["id"], "question": row["question"]} for row in raw_rows])
    parent_addendum = tmp_path / "parent_addendum.json"
    parent_addendum.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "v2"
    supersession = v1_dir / "supersession.json"
    original_atomic_write_new = v2._atomic_write_new
    addendum_commit_observations: list[bool] = []

    def observed_addendum_commit(path: Path, payload: bytes) -> None:
        if path == supersession:
            addendum_commit_observations.append(
                output.is_dir()
                and {item.name for item in output.iterdir()}
                == set(v2.PROTOCOL_FREEZE_OUTPUT_FILES)
                and (output / "manifest.json").is_file()
            )
        original_atomic_write_new(path, payload)

    monkeypatch.setattr(v2, "_atomic_write_new", observed_addendum_commit)
    result = v2.freeze_v2(
        project_root=v2.PROJECT_ROOT,
        output_dir=output,
        v1_dir=v1_dir,
        v1_generation_output_dir=tmp_path / "v1-generation-absent",
        supersession_addendum_path=supersession,
        identity_path=identity,
        raw_path=raw,
        parent_metadata_addendum_path=parent_addendum,
        expected_v1_hashes=None,
        generated_at_utc="2026-09-05T00:00:00+00:00",
        enforce_formal_locks=False,
    )
    assert result["report"]["api_calls"] == 0
    assert addendum_commit_observations == [True]
    assert result["report"]["v1_to_v2_subject_binding_diagnostic"]["v1_second_hop_subject_implicit"] == 1
    assert supersession.is_file()
    assert json.loads(supersession.read_text())["status"] == "SUPERSEDED_BEFORE_ANY_API_CALL"
    supersession_row = json.loads(supersession.read_text())
    assert supersession_row["superseded_by"]["manifest_sha256"] == v2._sha256_file(
        output / "manifest.json"
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["external_append_only_artifact"]["status_at_v2_manifest_commit"] == "PENDING_APPEND_ONLY_COMMIT"
    assert {path.name for path in output.iterdir()} == {
        "implementation_lock.json", "protocol.json", "report.json", "manifest.json"
    }
    frozen_hashes = {
        path.name: v2._sha256_file(path) for path in output.iterdir()
    }
    # Simulate a crash after the complete V2 directory commit but before the
    # external V1 addendum commit.  A rerun may only repair that missing link.
    supersession.unlink()
    repaired = v2.freeze_v2(
        project_root=v2.PROJECT_ROOT,
        output_dir=output,
        v1_dir=v1_dir,
        v1_generation_output_dir=tmp_path / "v1-generation-absent",
        supersession_addendum_path=supersession,
        identity_path=identity,
        raw_path=raw,
        parent_metadata_addendum_path=parent_addendum,
        expected_v1_hashes=None,
        generated_at_utc="2026-09-05T00:00:01+00:00",
        enforce_formal_locks=False,
    )
    assert repaired["repaired_missing_supersession_only"] is True
    assert addendum_commit_observations == [True, True]
    assert supersession.is_file()
    assert {
        path.name: v2._sha256_file(path) for path in output.iterdir()
    } == frozen_hashes


def test_atomic_append_only_primitives_never_replace_existing_targets(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        v2._atomic_write_new(target, b"replacement")
    assert target.read_bytes() == b"original"

    source_dir = tmp_path / "source"
    destination_dir = tmp_path / "destination"
    source_dir.mkdir()
    destination_dir.mkdir()
    (source_dir / "source-only").write_text("source", encoding="utf-8")
    (destination_dir / "destination-only").write_text("destination", encoding="utf-8")
    with pytest.raises(FileExistsError):
        v2._rename_dir_noreplace(source_dir, destination_dir)
    assert (source_dir / "source-only").is_file()
    assert (destination_dir / "destination-only").read_text(encoding="utf-8") == "destination"


def test_v1_chain_validator_rejects_manifest_output_hash_drift(tmp_path: Path) -> None:
    parent = {"rows": 1}
    protocol_path = tmp_path / "protocol.json"
    report_path = tmp_path / "report.json"
    protocol = {
        "schema_version": v1.SCHEMA_VERSION,
        "experiment_id": v1.EXPERIMENT_ID,
        "status": v1.STATUS,
        "parent_identity_freeze": parent,
    }
    report = {
        "schema_version": v1.REPORT_SCHEMA_VERSION,
        "experiment_id": v1.EXPERIMENT_ID,
        "status": v1.STATUS,
        "api_calls": 0,
        "parent_identity_freeze": parent,
    }
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest = {
        "schema_version": v1.MANIFEST_SCHEMA_VERSION,
        "experiment_id": v1.EXPERIMENT_ID,
        "status": v1.STATUS,
        "api_calls": 0,
        "training_started": False,
        "inputs": {"parent_identity_freeze": parent},
        "outputs": [
            {
                "path": protocol_path.name,
                "sha256": "0" * 64,
                "size_bytes": protocol_path.stat().st_size,
            },
            {
                "path": report_path.name,
                "sha256": v2._sha256_file(report_path),
                "size_bytes": report_path.stat().st_size,
            },
        ],
    }
    with pytest.raises(ValueError, match="hash/size drift"):
        v2._validate_v1_zero_call_freeze(
            protocol=protocol,
            report=report,
            manifest=manifest,
            artifact_paths={
                "protocol.json": protocol_path,
                "report.json": report_path,
                "manifest.json": tmp_path / "manifest.json",
            },
        )
