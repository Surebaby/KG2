"""CPU-only tests for the v8 implementation/runtime freezer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare import freeze_dynamic_decomposition_v8_implementation as freeze


def _runtime_contract() -> dict:
    return {
        "schema_version": "dynamic-decomposition-v8-production-runtime-contract-1",
        "runtime_version": "dynamic-decomposition-v8-production-runtime-1",
        "gold_access": False,
        "prospective_unlocked": False,
        "seed": 42,
        "production_staged": True,
        "staged_retrieval_contract": {
            "stable_deduplicate_cache_misses": True,
            "backend_batch_stages": ["root_all", "q1_all", "q2_BC_all"],
            "engineering_smoke_logical_retrieval_requests": 84,
            "maximum_full_index_passes_per_attempt": 3,
        },
        "shared_hf_runtime": {
            "one_physical_model_instance_for_roles": list(
                freeze.RUNTIME_MODEL_ROLES
            ),
            "base_model_path": "models/llama3-8b",
            "strong_sft_adapter_path": (
                "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_"
                "no_text_head/final"
            ),
            "tokenizer_path": "models/llama3-8b",
            "tokenizer_source": "base_model_tokenizer_matching_legacy_SFT_evaluation",
            "pad_token_policy": "set_to_eos_when_missing",
            "torch_dtype": "bfloat16",
            "device": "cuda:0",
            "local_files_only": True,
            "peft_is_trainable": False,
            "chat_template_source": "base_tokenizer.chat_template",
            "chat_template_add_generation_prompt": True,
            "model_input_truncation": False,
            "max_input_tokens_fail_closed": 6144,
            "role_max_new_tokens": {
                "controller": 96,
                "subanswer_reader": 96,
                "final_reader": 512,
            },
            "decoding": {
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "seed": 42,
            },
        },
        "canonical_retrieval": {
            "dense_top_k": 100,
            "bm25_top_k": 100,
            "rrf_k": 60,
            "rrf_output_k": 100,
            "bge_top_k": 10,
            "query_max_tokens": 128,
            "rerank_text_chars": 1200,
            "model_visible_passage_chars": 1200,
            "expected_documents": freeze.EXPECTED_WIKI18_DOCUMENTS,
            "corpus_path": "indexes_wiki18/corpus_flashrag.jsonl",
            "dense_index_path": "indexes_wiki18/e5_fp16.dat",
            "bm25_index_path": "indexes_wiki18/bm25",
            "e5_model_path": "models/e5-base-v2",
            "bge_model_path": "models/bge-reranker-v2-m3",
            "silent_fallback_allowed": False,
        },
        "logical_budget_by_arm": {
            "A_canonical_one_shot": {
                "retrieval": 1,
                "controller": 0,
                "subanswer_reader": 0,
                "final_reader": 1,
            },
            "B_observation_blind": {
                "retrieval": 3,
                "controller": 2,
                "subanswer_reader": 1,
                "final_reader": 1,
            },
            "C_answer_conditioned": {
                "retrieval": 3,
                "controller": 2,
                "subanswer_reader": 1,
                "final_reader": 1,
            },
        },
        "cache_contract": {
            "scope": "in_memory_for_one_locked_materialization_attempt",
            "persistent_cache_in_this_runner": False,
            "outer_append_only_resume_required": True,
            "logical_requests_equal_cache_hits_plus_cache_misses": True,
            "physical_executions_equal_cache_misses": True,
            "generation_key_binds": ["role"],
            "retrieval_key_binds": ["query_utf8"],
            "key_forbidden_fields": ["arm_label", "outcome", "gold"],
        },
        "first_attempts": {
            "engineering_smoke": {
                "experiment_id": freeze.SMOKE_EXPERIMENT_ID,
                "output_dir": freeze.SMOKE_ATTEMPT001.as_posix(),
                "cohort_path": freeze.SMOKE_COHORT_RELATIVE.as_posix(),
                "cohort_sha256": freeze.EXPECTED_SMOKE_COHORT_SHA256,
                "scope": "CONSUMED_IDENTITY_ONLY_4X3_NOT_FRESH_DEVELOPMENT",
            },
            "development": {
                "experiment_id": freeze.DEVELOPMENT_EXPERIMENT_ID,
                "output_dir": freeze.DEVELOPMENT_ATTEMPT001.as_posix(),
                "scope": "FROZEN_DEVELOPMENT90_ONLY",
            },
        },
    }


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_runtime_contract_locks_rrf100_shared_model_and_two_call_budget() -> None:
    contract = _runtime_contract()
    assert freeze.validate_runtime_contract(contract) == contract

    bad_rrf = json.loads(json.dumps(contract))
    bad_rrf["canonical_retrieval"]["rrf_output_k"] = 50
    with pytest.raises(freeze.V8FreezeError, match="rrf_output_k"):
        freeze.validate_runtime_contract(bad_rrf)

    bad_roles = json.loads(json.dumps(contract))
    bad_roles["shared_hf_runtime"]["one_physical_model_instance_for_roles"] = [
        "controller"
    ]
    with pytest.raises(
        freeze.V8FreezeError, match="one_physical_model_instance_for_roles"
    ):
        freeze.validate_runtime_contract(bad_roles)

    bad_calls = json.loads(json.dumps(contract))
    bad_calls["logical_budget_by_arm"]["C_answer_conditioned"]["controller"] = 3
    with pytest.raises(freeze.V8FreezeError, match="logical per-arm budget"):
        freeze.validate_runtime_contract(bad_calls)

    bad_staging = json.loads(json.dumps(contract))
    bad_staging["production_staged"] = False
    with pytest.raises(freeze.V8FreezeError, match="production_staged"):
        freeze.validate_runtime_contract(bad_staging)

    bad_passes = json.loads(json.dumps(contract))
    bad_passes["staged_retrieval_contract"][
        "maximum_full_index_passes_per_attempt"
    ] = 4
    with pytest.raises(freeze.V8FreezeError, match="maximum_full_index_passes"):
        freeze.validate_runtime_contract(bad_passes)


def test_file_and_tree_content_locks_are_deterministic_and_detect_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    _write(root / "b.txt", b"bravo")
    _write(root / "nested" / "a.txt", b"alpha")
    first = freeze.tree_lock(root)
    second = freeze.tree_lock(root)
    assert first == second
    assert first["file_count"] == 2
    assert freeze._verify_prior_tree_lock(first, label="synthetic") == (
        freeze.concise_tree_lock(first)
    )

    (root / "nested" / "a.txt").write_bytes(b"changed")
    with pytest.raises(freeze.V8FreezeError, match="full-content drift"):
        freeze._verify_prior_tree_lock(first, label="synthetic")


def test_self_commitment_detects_payload_mutation() -> None:
    committed = freeze._self_committed_payload(
        {"status": "FROZEN", "gold_access": False}, field="body_sha256"
    )
    assert freeze.verify_self_commitment(committed, field="body_sha256")
    committed["status"] = "CHANGED"
    assert not freeze.verify_self_commitment(committed, field="body_sha256")


def test_attempt_lifecycle_is_append_only_hash_validated_and_terminal(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "run" / "attempt001"
    protocol_lock = {"path": "protocol.json", "sha256": "a" * 64}
    cohort_lock = {"sha256": "b" * 64, "row_count": 12}
    freeze.reserve_attempt_directory(
        attempt_dir=attempt,
        experiment_id=freeze.SMOKE_EXPERIMENT_ID,
        attempt_id="attempt001",
        implementation_protocol=protocol_lock,
        cohort_lock=cohort_lock,
        created_at_utc="2026-09-04T00:00:00+00:00",
    )
    with pytest.raises(FileExistsError):
        freeze.reserve_attempt_directory(
            attempt_dir=attempt,
            experiment_id=freeze.SMOKE_EXPERIMENT_ID,
            attempt_id="attempt001",
            implementation_protocol=protocol_lock,
            cohort_lock=cohort_lock,
            created_at_utc="2026-09-04T00:00:00+00:00",
        )

    _write(attempt / "materialization.jsonl", b'{"row":1}\n')
    descriptor = freeze.commit_stage_boundary(
        attempt_dir=attempt,
        stage_name="gold_free_materialization",
        artifact_paths=["materialization.jsonl"],
        row_count=1,
        stage_config_sha256="c" * 64,
        completed_at_utc="2026-09-04T00:01:00+00:00",
    )
    assert descriptor["status"] == "COMPLETE_HASH_VALIDATED_STAGE_BOUNDARY"
    assert freeze.validate_reusable_stage(attempt, "gold_free_materialization")[
        "stage"
    ] == descriptor

    terminal = freeze.finalize_attempt(
        attempt_dir=attempt,
        success=True,
        reason="all Gold-free smoke gates passed",
        completed_at_utc="2026-09-04T00:02:00+00:00",
        required_complete_stages=["gold_free_materialization"],
    )
    assert terminal["status"] == "COMPLETE"
    assert (attempt / freeze.RUNNING_MANIFEST).is_file()
    assert (attempt / freeze.COMPLETE_MANIFEST).is_file()
    assert not (attempt / freeze.FAILED_MANIFEST).exists()
    with pytest.raises(freeze.V8FreezeError, match="terminal"):
        freeze.commit_stage_boundary(
            attempt_dir=attempt,
            stage_name="late",
            artifact_paths=["materialization.jsonl"],
            row_count=1,
            stage_config_sha256="d" * 64,
            completed_at_utc="2026-09-04T00:03:00+00:00",
        )


def test_partial_stage_cannot_be_reused_and_retry_needs_new_attempt(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    first = run_root / "attempt001"
    protocol_lock = {"path": "protocol.json", "sha256": "a" * 64}
    cohort_lock = {"sha256": "b" * 64, "row_count": 12}
    freeze.reserve_attempt_directory(
        attempt_dir=first,
        experiment_id=freeze.SMOKE_EXPERIMENT_ID,
        attempt_id="attempt001",
        implementation_protocol=protocol_lock,
        cohort_lock=cohort_lock,
        created_at_utc="2026-09-04T00:00:00+00:00",
    )
    _write(first / "partial.jsonl", b'{"row":1}\n')
    with pytest.raises(freeze.V8FreezeError, match="partial output is not reusable"):
        freeze.validate_reusable_stage(first, "materialization")
    failed = freeze.finalize_attempt(
        attempt_dir=first,
        success=False,
        reason="interrupted; partial rows retained but not reusable",
        completed_at_utc="2026-09-04T00:01:00+00:00",
    )
    failed_lock = freeze.file_lock(first / freeze.FAILED_MANIFEST)

    second = run_root / "attempt002"
    running = freeze.reserve_attempt_directory(
        attempt_dir=second,
        experiment_id=freeze.SMOKE_EXPERIMENT_ID,
        attempt_id="attempt002",
        implementation_protocol=protocol_lock,
        cohort_lock=cohort_lock,
        created_at_utc="2026-09-04T00:02:00+00:00",
        retry_of=failed_lock,
    )
    assert failed["status"] == "FAILED_RETAINED_APPEND_ONLY"
    assert running["retry_of"]["sha256"] == failed_lock["sha256"]
    assert not (second / "partial.jsonl").exists()


def test_committed_stage_revalidation_rejects_later_artifact_mutation(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt001"
    freeze.reserve_attempt_directory(
        attempt_dir=attempt,
        experiment_id="EXPERIMENT",
        attempt_id="attempt001",
        implementation_protocol={"sha256": "a" * 64},
        cohort_lock={"sha256": "b" * 64},
        created_at_utc="2026-09-04T00:00:00+00:00",
    )
    artifact = _write(attempt / "rows.jsonl", b'{"row":1}\n')
    freeze.commit_stage_boundary(
        attempt_dir=attempt,
        stage_name="rows",
        artifact_paths=["rows.jsonl"],
        row_count=1,
        stage_config_sha256="c" * 64,
        completed_at_utc="2026-09-04T00:01:00+00:00",
    )
    artifact.write_bytes(b'{"row":2}\n')
    with pytest.raises(freeze.V8FreezeError, match="stage artifact drift"):
        freeze.validate_reusable_stage(attempt, "rows")


def test_actual_parent_design_smoke_and_development_locks_validate_without_gold() -> None:
    parents = freeze._validate_design_and_smoke(freeze.PROJECT_ROOT)
    cohort = freeze._validate_development_cohort()
    assert parents["design_protocol"]["sha256"] == (
        freeze.EXPECTED_DESIGN_PROTOCOL_SHA256
    )
    assert parents["smoke"]["cohort"]["sha256"] == (
        freeze.EXPECTED_SMOKE_COHORT_SHA256
    )
    assert cohort["row_count"] == 90
    assert cohort["prospective_opened_or_hashed_by_this_command"] is False
    retry_parent = freeze._validate_v1_failed_smoke_retry_parent(freeze.PROJECT_ROOT)
    assert retry_parent["attempt001_failed"]["sha256"] == (
        freeze.EXPECTED_SMOKE_ATTEMPT001_FAILED_SHA256
    )
    assert freeze.DEFAULT_OUTPUT_DIR != freeze.PARENT_IMPLEMENTATION_V1_RELATIVE
    assert freeze.DEFAULT_OUTPUT_DIR.name.endswith("_v2")


def test_writer_is_append_only_and_manifest_is_self_committed(tmp_path: Path) -> None:
    protocol = {
        "experiment_id": freeze.EXPERIMENT_ID,
        "status": freeze.STATUS,
        "created_at_utc": "2026-09-04T00:00:00+00:00",
        "authorization": {
            "prospective_open_or_hash": False,
            "gold_attachment": False,
        },
    }
    output = tmp_path / "freeze"
    freeze.write_implementation_freeze(protocol, output)
    manifest = freeze.read_json(output / "manifest.json")
    assert freeze.verify_self_commitment(
        manifest, field="manifest_body_canonical_sha256"
    )
    with pytest.raises(FileExistsError):
        freeze.write_implementation_freeze(protocol, output)


def test_public_cli_exposes_no_skip_gold_prospective_or_alternate_cohort_flag() -> None:
    source = Path(freeze.__file__).read_text(encoding="utf-8")
    cli_section = source[source.index("def parse_args") :]
    for forbidden in (
        "--skip",
        "--gold",
        "--prospective",
        "--cohort",
        "--verify_large_content",
    ):
        assert forbidden not in cli_section
