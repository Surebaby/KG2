"""Tests for the append-only Hotpot silver generation/review protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pytest

from kgproweight.data import hotpot_controller_silver as hotpot_silver
from scripts.prepare import freeze_hotpot_controller_silver_execution_v1 as freeze


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _synthetic_parent(
    project: Path, *, rows: int = freeze.PILOT_ROWS
) -> tuple[Path, dict[str, str]]:
    parent = project / "parent"
    parent.mkdir(parents=True)
    identity_path = parent / "pilot.identity_only.jsonl"
    identity_path.write_text(
        "".join(
            json.dumps(
                {
                    "dataset": "hotpotqa",
                    "qid": f"qid-{index}",
                    "question": f"What is the property of Root {index}?",
                },
                separators=(",", ":"),
            )
            + "\n"
            for index in range(rows)
        ),
        encoding="utf-8",
    )
    experiment_id = "TEST-PARENT-HOTPOT-PILOT"
    protocol = {
        "experiment_id": experiment_id,
        "status": freeze.PARENT_PROTOCOL_STATUS,
        "authorization": {"q1_q2_generation": False, "training": False},
    }
    report = {
        "experiment_id": experiment_id,
        "status": "COMPLETE_TEST_SIZED_IDENTITY_ONLY_SILVER_PILOT_NOT_FORMAL",
        "checks": {
            "all_freeze_gates_pass": True,
            "model_calls": 0,
            "q1_q2_records_generated": 0,
        },
    }
    _write_json(parent / "protocol.json", protocol)
    _write_json(parent / "report.json", report)
    manifest = {
        "experiment_id": experiment_id,
        "status": report["status"],
        "outputs": [
            {"path": name, "sha256": _sha256(parent / name)}
            for name in freeze.PARENT_FILENAMES[:3]
        ],
    }
    _write_json(parent / "manifest.json", manifest)
    hashes = {name: _sha256(parent / name) for name in freeze.PARENT_FILENAMES}
    return parent, hashes


def test_mask_nonces_are_deterministic_distinct_fixed_length_and_bound() -> None:
    chain = "a" * 64
    first = freeze.derive_mask_nonces(chain_sha256=chain)
    second = freeze.derive_mask_nonces(chain_sha256=chain)
    changed_chain = freeze.derive_mask_nonces(chain_sha256="b" * 64)
    changed_experiment = freeze.derive_mask_nonces(
        chain_sha256=chain, experiment_id="OTHER-EXPERIMENT"
    )
    assert first == second
    assert first[0] != first[1]
    assert all(len(value) == freeze.NONCE_LENGTH for value in first)
    assert all(re.fullmatch(freeze.NONCE_PATTERN, value) for value in first)
    assert first != changed_chain
    assert first != changed_experiment


def test_producer_safe_payload_drops_identifiers_and_named_masks() -> None:
    masked_view = {
        "schema_version": hotpot_silver.PROPOSAL_VIEW_SCHEMA_VERSION,
        "dataset": "hotpotqa",
        "qid": "secret-row-id",
        "original_question": "What property does Root have?",
        "root_document_title": "Root",
        "first_hop_evidence_masked": (
            f"Root is linked to {hotpot_silver.INTERMEDIATE_MASK}."
        ),
        "second_hop_evidence_masked": (
            f"{hotpot_silver.INTERMEDIATE_MASK} has property "
            f"{hotpot_silver.FINAL_MASK}."
        ),
        "required_output": {"unused": True},
    }
    chain_hash = "c" * 64
    payload = freeze.build_producer_safe_payload(
        masked_view, chain_sha256=chain_hash
    )
    nonces = freeze.derive_mask_nonces(chain_sha256=chain_hash)
    assert tuple(payload) == (
        "original_question",
        "root_document_title",
        "first_hop_evidence_masked",
        "second_hop_evidence_masked",
    )
    assert not {"dataset", "qid", "id", "question_key"} & set(payload)
    serialized = json.dumps(payload)
    assert "secret-row-id" not in serialized
    assert hotpot_silver.INTERMEDIATE_MASK not in serialized
    assert hotpot_silver.FINAL_MASK not in serialized
    assert nonces[0] in payload["first_hop_evidence_masked"]
    assert nonces[0] in payload["second_hop_evidence_masked"]
    assert nonces[1] in payload["second_hop_evidence_masked"]
    assert freeze.nonce_echo_count("safe response", nonces=nonces) == 0
    assert freeze.nonce_echo_count(
        {"q1": f"bad {nonces[0]}", "q2": nonces[1]}, nonces=nonces
    ) == 2


def test_protocol_freezes_models_prompts_schemas_contexts_and_no_claim() -> None:
    parent = {
        "rows": freeze.PILOT_ROWS,
        "artifacts": {name: {} for name in freeze.PARENT_FILENAMES},
    }
    protocol = freeze.build_protocol(
        generated_at_utc="2026-09-04T00:00:00+00:00", parent_lock=parent
    )
    assert protocol["producer"]["model"] == "deepseek-v4-flash"
    assert protocol["producer"]["candidates_per_identity_exact"] == 1
    assert protocol["producer"]["request"]["temperature"] == 0.0
    assert protocol["producer"]["format_or_semantic_retry"] is False
    reviewers = protocol["reviewers"]
    assert reviewers["reviewer_1"]["role"] == "blind_structural"
    assert reviewers["reviewer_2"]["role"] == "train_gold_aware_adjudicator"
    assert reviewers["reviewer_1"]["model"] != reviewers["reviewer_2"]["model"]
    assert "final_answers" in reviewers["reviewer_1"]["does_not_see"]
    assert "final_answers" in reviewers["reviewer_2"]["sees"]
    assert reviewers["fresh_message_array_and_fresh_context_per_physical_call"] is True
    assert reviewers["statistically_independent_claim_allowed"] is False
    assert protocol["masking"]["producer_response_nonce_echo_count_required"] == 0
    assert protocol["schemas"]["producer_output"]["sha256"] == freeze._canonical_sha256(
        freeze.PRODUCER_OUTPUT_SCHEMA
    )
    for role in ("producer", "reviewer_1_blind", "reviewer_2_gold_aware"):
        prompt = freeze._prompt_locks()[role]
        assert prompt["sha256"] == freeze._canonical_sha256(
            {"system": prompt["system"], "user_template": prompt["user_template"]}
        )
    assert protocol["api_execution"]["transport_retry_request_body_must_be_byte_identical"] is True
    assert protocol["api_execution"]["invalid_or_rejected_content_retry_allowed"] is False
    assert protocol["authorization"]["training"] is False
    assert protocol["authorization"]["formal_em_f1_ihr_evaluation"] is False


def test_freezer_binds_four_parent_files_and_makes_no_calls(tmp_path: Path) -> None:
    project = tmp_path / "project"
    parent, hashes = _synthetic_parent(project)
    output = project / "execution-protocol"
    result = freeze.freeze_execution_protocol(
        project_root=project,
        parent_dir=parent,
        output_dir=output,
        expected_parent_hashes=hashes,
        expected_rows=freeze.PILOT_ROWS,
        experiment_id="TEST-EXECUTION-PROTOCOL",
        enforce_formal_locks=False,
        generated_at_utc="2026-09-04T00:00:00+00:00",
    )
    assert set(path.name for path in output.iterdir()) == {
        "protocol.json",
        "report.json",
        "manifest.json",
    }
    lock = result["protocol"]["parent_identity_freeze"]
    assert set(lock["artifacts"]) == set(freeze.PARENT_FILENAMES)
    assert {name: value["sha256"] for name, value in lock["artifacts"].items()} == hashes
    checks = result["report"]["checks"]
    assert checks["all_freeze_gates_pass"] is True
    assert checks["api_calls"] == 0
    assert checks["dotenv_or_environment_read"] is False
    assert result["manifest"]["api_calls"] == 0
    assert result["manifest"]["training_started"] is False
    serialized = "".join(path.read_text(encoding="utf-8") for path in output.iterdir())
    assert "sk-test-secret" not in serialized
    assert "api.deepseek.com" not in serialized
    assert "OPENAI_API_KEY" in serialized  # variable name, never its value


def test_parent_hash_drift_fails_before_output(tmp_path: Path) -> None:
    project = tmp_path / "project"
    parent, hashes = _synthetic_parent(project)
    hashes["report.json"] = "0" * 64
    output = project / "must-not-exist"
    with pytest.raises(ValueError, match="parent artifact SHA256 drift"):
        freeze.freeze_execution_protocol(
            project_root=project,
            parent_dir=parent,
            output_dir=output,
            expected_parent_hashes=hashes,
            expected_rows=freeze.PILOT_ROWS,
            experiment_id="TEST-EXECUTION-PROTOCOL",
            enforce_formal_locks=False,
        )
    assert not output.exists()


def test_append_only_refuses_second_invocation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    parent, hashes = _synthetic_parent(project)
    output = project / "execution-protocol"
    kwargs = dict(
        project_root=project,
        parent_dir=parent,
        output_dir=output,
        expected_parent_hashes=hashes,
        expected_rows=freeze.PILOT_ROWS,
        experiment_id="TEST-EXECUTION-PROTOCOL",
        enforce_formal_locks=False,
    )
    freeze.freeze_execution_protocol(**kwargs)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze.freeze_execution_protocol(**kwargs)


def test_formal_mode_cannot_disable_parent_locks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="immutable"):
        freeze.freeze_execution_protocol(
            project_root=tmp_path,
            expected_parent_hashes=None,
        )
    assert not (tmp_path / freeze.DEFAULT_OUTPUT_DIR).exists()
