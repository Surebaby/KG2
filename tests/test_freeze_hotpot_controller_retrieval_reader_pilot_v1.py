"""CPU-only tests for the Hotpot retrieval/Reader protocol freezer."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from kgproweight.kg.question_kg import question_sha256
from scripts.prepare import freeze_hotpot_controller_retrieval_reader_pilot_v1 as freeze


GENERATION_ID = "TEST-HOTPOT-GENERATION-V2"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source_row(index: int, *, accepted: bool) -> dict:
    question = f"Where was the organization linked to Root {index} founded?"
    q1 = f"Which organization is Root {index} linked to?"
    q2 = "Where was #1 founded?"
    proposal = {
        "schema_version": "hotpot-controller-query-proposal-v1",
        "q1": q1,
        "q2_template": q2,
    }
    proposal_hash = freeze._canonical_sha256(proposal)
    return {
        "schema_version": "hotpot-controller-silver-generation-run-1",
        "dataset": "hotpotqa",
        "qid": f"qid-{index}",
        "question": question,
        "question_sha256": question_sha256(question),
        "stage": "producer",
        "status": "accepted" if accepted else "rejected",
        "semantic_request_id": f"request-{index}",
        "requested_model": "model-a",
        "response_model": "model-a" if accepted else None,
        "finish_reason": "stop" if accepted else None,
        "raw_response_content": json.dumps(proposal) if accepted else None,
        "raw_response_sha256": "a" * 64 if accepted else None,
        "parsed_response": proposal if accepted else None,
        "parsed_response_sha256": proposal_hash if accepted else None,
        "nonce_echo_count": 0,
        "reject_code": None if accepted else "producer_reject",
        "detail_code": None,
        "final_item_status": freeze.ACCEPTED_STATUS if accepted else "producer_rejected",
        "dual_review_unanimous_pass": accepted,
        "q1_query": q1 if accepted else None,
        "q2_template": q2 if accepted else None,
        "proposal_sha256": proposal_hash if accepted else None,
        "runtime_projection_gold_or_observation_fields_present": False,
        # Successor generation protocols may append answer-free audit fields;
        # the freezer requires its safety subset and then projects these away.
        "api_call_wal_commit_sha256": "b" * 64,
    }


def _parent_v8(tmp_path: Path) -> tuple[Path, Path]:
    digest = lambda ch: ch * 64
    models = {
        "base_model": {"tree_sha256": digest("1")},
        "strong_sft": {"tree_sha256": digest("2")},
        "retrieval_encoder": {"tree_sha256": digest("3")},
        "cross_encoder": {"tree_sha256": digest("4")},
    }
    wiki = {
        "corpus": {"sha256": digest("5")},
        "dense_index": {"sha256": digest("6")},
        "bm25_index": {"tree_sha256": digest("7")},
    }
    protocol = {
        "status": "AUTHORIZED_SMOKE_THEN_CONDITIONAL_GOLD_FREE_DEVELOPMENT90",
        "gold_access": False,
        "content_reverification": {
            "full_hash_verification_performed_by_this_command": True,
            "content": {"models": models, "wiki18": wiki},
        },
        "generation_role_identity": {
            "same_base_tree_sha256": digest("1"),
            "same_adapter_tree_sha256": digest("2"),
        },
        "tokenizer_and_chat_template": {"chat_template_utf8_sha256": digest("8")},
    }
    protocol_path = tmp_path / "parent-v8" / "protocol.json"
    manifest_path = protocol_path.with_name("manifest.json")
    _write_json(protocol_path, protocol)
    protocol_lock = freeze._file_lock(protocol_path)
    _write_json(
        manifest_path,
        {
            "status": "AUTHORIZED_SMOKE_THEN_CONDITIONAL_GOLD_FREE_DEVELOPMENT90",
            "protocol": protocol_lock,
        },
    )
    return protocol_path, manifest_path


def _generation(tmp_path: Path, *, accepted: int = 2, total: int = 3) -> Path:
    directory = tmp_path / "generation-v2"
    rows = [_source_row(index, accepted=index < accepted) for index in range(total)]
    producer = directory / "producer_proposals.jsonl"
    report = directory / "report.json"
    _write_jsonl(producer, rows)
    _write_json(
        report,
        {
            "experiment_id": GENERATION_ID,
            "status": freeze.GENERATION_STATUS,
            "fixed_denominator": total,
            "dual_review_accepted": accepted,
            "scientific_boundary": {
                "retrieval_or_reader_calls": 0,
                "training_started": False,
            },
        },
    )
    manifest = {
        "experiment_id": GENERATION_ID,
        "status": freeze.GENERATION_STATUS,
        "fixed_denominator": total,
        "retrieval_calls": 0,
        "training_started": False,
        "outputs": [
            {
                "path": "producer_proposals.jsonl",
                "sha256": freeze._sha256_file(producer),
                "size_bytes": producer.stat().st_size,
            },
            {
                "path": "report.json",
                "sha256": freeze._sha256_file(report),
                "size_bytes": report.stat().st_size,
            },
        ],
    }
    _write_json(directory / "manifest.json", manifest)
    # These deliberately malformed files prove the freezer does not need to
    # deserialize the annotation-bearing action/reviewer artifacts.
    (directory / "accepted_actions.jsonl").write_text("NOT JSON\n", encoding="utf-8")
    (directory / "reviewer_2_reviews.jsonl").write_text("NOT JSON\n", encoding="utf-8")
    return directory


def test_freezer_projects_only_accepted_answer_free_rows_and_binds_assets(tmp_path: Path) -> None:
    generation = _generation(tmp_path)
    parent_protocol, parent_manifest = _parent_v8(tmp_path)
    output = tmp_path / "frozen"
    result = freeze.freeze_protocol(
        project_root=freeze.PROJECT_ROOT,
        generation_dir=generation,
        generation_experiment_id=GENERATION_ID,
        parent_v8_protocol_path=parent_protocol,
        parent_v8_manifest_path=parent_manifest,
        output_dir=output,
        expected_rows=3,
        accepted_min=2,
        generated_at_utc="2026-09-05T00:00:00+00:00",
    )
    runtime_rows = freeze._load_jsonl(output / "runtime_inputs.answer_free.jsonl")
    assert len(runtime_rows) == 2
    assert all(set(row) == set(freeze.runner._INPUT_FIELDS) for row in runtime_rows)
    encoded = json.dumps(runtime_rows, ensure_ascii=False)
    assert "observation" not in encoded and "gold" not in encoded.casefold()
    assert result["report"]["source_fixed_denominator"] == 3
    assert result["report"]["runtime_input_rows"] == 2
    assert result["protocol"]["source_generation_experiment_id"] == GENERATION_ID
    assert result["protocol"]["model_asset_identity"]["adapter_tree_sha256"] == "2" * 64
    assert result["protocol"]["authorization"]["training"] is False
    assert result["report"]["source_files_forbidden_and_not_read"] == list(
        freeze.SOURCE_FILES_FORBIDDEN
    )


def test_freezer_fails_closed_on_nonanswer_free_flag_or_hash_drift(tmp_path: Path) -> None:
    generation = _generation(tmp_path)
    rows = freeze._load_jsonl(generation / "producer_proposals.jsonl")
    rows[0]["runtime_projection_gold_or_observation_fields_present"] = True
    _write_jsonl(generation / "producer_proposals.jsonl", rows)
    # Rebind manifest so the failure is specifically the content boundary.
    manifest = freeze._load_json(generation / "manifest.json")
    lock = next(item for item in manifest["outputs"] if item["path"] == "producer_proposals.jsonl")
    lock["sha256"] = freeze._sha256_file(generation / "producer_proposals.jsonl")
    lock["size_bytes"] = (generation / "producer_proposals.jsonl").stat().st_size
    _write_json(generation / "manifest.json", manifest)
    parent_protocol, parent_manifest = _parent_v8(tmp_path)
    with pytest.raises(freeze.HotpotRuntimeFreezeError, match="boundary violation"):
        freeze.freeze_protocol(
            project_root=freeze.PROJECT_ROOT,
            generation_dir=generation,
            generation_experiment_id=GENERATION_ID,
            parent_v8_protocol_path=parent_protocol,
            parent_v8_manifest_path=parent_manifest,
            output_dir=tmp_path / "frozen",
            expected_rows=3,
            accepted_min=2,
        )


def test_freezer_requires_materialized_successor_and_is_append_only(tmp_path: Path) -> None:
    parent_protocol, parent_manifest = _parent_v8(tmp_path)
    with pytest.raises(FileNotFoundError, match="not all materialised"):
        freeze.freeze_protocol(
            project_root=freeze.PROJECT_ROOT,
            generation_dir=tmp_path / "missing-v2",
            generation_experiment_id=GENERATION_ID,
            parent_v8_protocol_path=parent_protocol,
            parent_v8_manifest_path=parent_manifest,
            output_dir=tmp_path / "frozen",
            expected_rows=3,
            accepted_min=2,
        )
    generation = _generation(tmp_path)
    output = tmp_path / "frozen"
    freeze.freeze_protocol(
        project_root=freeze.PROJECT_ROOT,
        generation_dir=generation,
        generation_experiment_id=GENERATION_ID,
        parent_v8_protocol_path=parent_protocol,
        parent_v8_manifest_path=parent_manifest,
        output_dir=output,
        expected_rows=3,
        accepted_min=2,
    )
    with pytest.raises(FileExistsError, match="append-only"):
        freeze.freeze_protocol(
            project_root=freeze.PROJECT_ROOT,
            generation_dir=generation,
            generation_experiment_id=GENERATION_ID,
            parent_v8_protocol_path=parent_protocol,
            parent_v8_manifest_path=parent_manifest,
            output_dir=output,
            expected_rows=3,
            accepted_min=2,
        )


def test_cli_requires_explicit_successor_generation_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["freeze_hotpot_controller_retrieval_reader_pilot_v1.py"])
    with pytest.raises(SystemExit):
        freeze.main()
