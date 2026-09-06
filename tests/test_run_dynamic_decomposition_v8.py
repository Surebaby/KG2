"""CPU-only tests for the append-only v8 production driver."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.pilot import run_dynamic_decomposition_v8 as driver
from scripts.prepare import freeze_dynamic_decomposition_v8_implementation as freeze


def _retrieval_event(key: str, *, cache_hit: bool) -> dict:
    return {
        "content_cache_key_sha256": key,
        "content_cache_key_mode": "asset_bound_query_and_retrieval_stack",
        "cache_hit": cache_hit,
    }


def _model_event(*, cache_hit: bool) -> dict:
    return {
        "content_cache_key_mode": "asset_bound_model_visible_token_ids",
        "cache_hit": cache_hit,
        "runtime_telemetry": {"shared_model_object_id": 7731},
    }


def _passages(prefix: str) -> list[dict[str, str]]:
    return [
        {
            "doc_id": f"{prefix}-{index}",
            "title": f"{prefix} title {index}",
            "text": f"{prefix} evidence {index}",
        }
        for index in range(10)
    ]


def _row(dataset: str, index: int, contract: dict) -> dict:
    question = f"Original question {dataset} {index}?"
    q1 = f"First subquestion {dataset} {index}?"
    q2 = f"Second subquestion {dataset} {index}?"
    b_passages = _passages(f"bc-{dataset}-{index}")
    arms = {
        "A_canonical_one_shot": {
            "retrieval": _retrieval_event(f"root-{dataset}-{index}", cache_hit=False),
            "final_passages": _passages(f"a-{dataset}-{index}"),
            "final": {"reader_event": _model_event(cache_hit=False)},
        },
        "B_observation_blind": {
            "q2_state_mode": "q2_no_verified_subanswer",
            "q2_action": {
                "proposal_valid": True,
                "selected_query": q2,
                "selection_source": "q2_static",
                "used_fallback": False,
            },
            "q2_controller": _model_event(cache_hit=False),
            "q2_retrieval": _retrieval_event(f"q2-{dataset}-{index}", cache_hit=False),
            "final_passages": deepcopy(b_passages),
            "final": {"reader_event": _model_event(cache_hit=False)},
        },
        "C_answer_conditioned": {
            "dynamic_eligible": False,
            "q2_state_mode": "q2_no_verified_subanswer",
            "q2_action": {
                "proposal_valid": True,
                "selected_query": q2,
                "selection_source": "q2_static",
                "used_fallback": False,
            },
            "q2_controller": _model_event(cache_hit=True),
            "q2_retrieval": _retrieval_event(f"q2-{dataset}-{index}", cache_hit=True),
            "final_passages": deepcopy(b_passages),
            "final": {"reader_event": _model_event(cache_hit=True)},
        },
    }
    return {
        "gold_access": False,
        "identity": {"dataset": dataset, "qid": f"q{index}", "question": question},
        "shared": {
            "q1_action": {
                "proposal_valid": True,
                "selected_query": q1,
                "selection_source": "q1",
            },
            "subanswer_binding": {"verified": False},
            "root_retrieval": _retrieval_event(
                f"root-{dataset}-{index}", cache_hit=True
            ),
            "q1_retrieval": _retrieval_event(f"q1-{dataset}-{index}", cache_hit=False),
            "q1_controller": _model_event(cache_hit=False),
            "subanswer_reader": _model_event(cache_hit=False),
        },
        "arms": arms,
        "counterfactual_identity": {
            "ineligible_c": True,
            "b_c_q2_prompt_byte_identical": True,
            "b_c_q2_response_byte_identical": True,
            "b_c_q2_query_byte_identical": True,
            "b_c_final_passages_byte_identical": True,
            "b_c_final_prompt_byte_identical": True,
            "b_c_prediction_byte_identical": True,
        },
        "budget": {
            "logical_by_arm": deepcopy(contract["logical_budget_by_arm"]),
            "joint_cache_accounting": {
                "controller": {
                    "logical_requests": 4,
                    "cache_hits": 1,
                    "cache_misses": 3,
                    "physical_executions": 3,
                },
                "subanswer_reader": {
                    "logical_requests": 2,
                    "cache_hits": 1,
                    "cache_misses": 1,
                    "physical_executions": 1,
                },
                "final_reader": {
                    "logical_requests": 3,
                    "cache_hits": 1,
                    "cache_misses": 2,
                    "physical_executions": 2,
                },
                "retrieval": {
                    "logical_requests": 7,
                    "cache_hits": 4,
                    "cache_misses": 3,
                    "physical_executions": 3,
                },
            },
        },
    }


def _contract() -> dict:
    return freeze.validate_runtime_contract(
        freeze.literal_constant(
            freeze.PROJECT_ROOT
            / "scripts/pilot/materialize_dynamic_decomposition_v8.py",
            freeze.RUNTIME_CONTRACT_CONSTANT,
        )
    )


def _protocol(tmp_path: Path, contract: dict) -> dict:
    return {
        "runtime_contract": contract,
        "run_registry": {
            "smoke": {
                "experiment_id": freeze.SMOKE_EXPERIMENT_ID,
                "cohort_n": 12,
                "cohort_sha256": freeze.EXPECTED_SMOKE_COHORT_SHA256,
                "first_attempt_dir": str(tmp_path / "smoke_attempt001"),
                "next_authorized_attempt_id": "attempt002",
                "next_authorized_experiment_id": freeze.SMOKE_EXPERIMENT_ID_ATTEMPT002,
                "next_authorized_attempt_dir": str(tmp_path / "smoke_attempt002"),
            },
            "development": {
                "experiment_id": freeze.DEVELOPMENT_EXPERIMENT_ID,
                "cohort_n": 90,
                "cohort_sha256": freeze.EXPECTED_DEVELOPMENT_SHA256,
                "first_attempt_dir": str(tmp_path / "development_attempt001"),
            },
        },
        "gates": {
            "smoke": {
                "row_count": 12,
                "logical_retrieval_requests": 84,
                "full_index_passes_max": 3,
                "runtime_error_count": 0,
            },
            "development_gold_free": {
                "itt_cardinality": 90,
                "logical_retrieval_requests": 630,
                "full_index_passes_max": 3,
                "q1_schema_valid_rate_min_each_dataset": 0.95,
                "B_q2_static_valid_rate_min": 0.90,
                "a1_admissible_rate_min_each_dataset": 0.40,
                "C_dynamic_transition_rate_all_itt_min_each_dataset": 0.32,
                "empty_repeat_padding_query_rate_max": 0.05,
            },
        },
    }


def _smoke_result(contract: dict) -> dict:
    rows = [
        _row(dataset, index, contract)
        for dataset in freeze.DATASETS
        for index in range(4)
    ]
    return {
        "experiment_id": freeze.SMOKE_EXPERIMENT_ID,
        "gold_access": False,
        "prospective_unlocked": False,
        "joint_cache_accounting": {
            "retrieval": {
                "logical_requests": 84,
                "cache_hits": 48,
                "cache_misses": 36,
                "physical_executions": 36,
            }
        },
        "retrieval_batch_telemetry": {
            "backend_batch_invocations": 3,
            "full_index_passes": 3,
            "unique_query_count_by_batch": [12, 12, 12],
            "stage_batches": [
                {
                    "stage": "root_all",
                    "logical_request_groups": 24,
                    "unique_miss_query_count": 12,
                    "backend_invoked": True,
                },
                {
                    "stage": "q1_all",
                    "logical_request_groups": 12,
                    "unique_miss_query_count": 12,
                    "backend_invoked": True,
                },
                {
                    "stage": "q2_BC_all",
                    "logical_request_groups": 24,
                    "unique_miss_query_count": 12,
                    "backend_invoked": True,
                },
            ],
        },
        "rows": rows,
    }


def test_gold_free_smoke_report_passes_complete_fake_runtime(tmp_path: Path) -> None:
    contract = _contract()
    protocol = _protocol(tmp_path, contract)
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("placeholder\n", encoding="utf-8")
    report = driver.build_gold_free_mechanism_report(
        result=_smoke_result(contract),
        protocol=protocol,
        scope="smoke",
        rows_lock=freeze.file_lock(rows_path),
        created_at_utc="2026-09-04T00:00:00+00:00",
    )
    assert report["status"] == "PASS"
    assert report["all_pass"] is True
    assert report["metrics"]["shared_model_object_id_count"] == 1
    assert report["metrics"]["asset_bound_content_cache_event_rate"] == 1.0
    assert report["metrics"]["a1_ineligible_full_content_identity_rate"] == 1.0
    assert report["metrics"]["logical_retrieval_requests"] == 84
    assert report["metrics"]["retrieval_full_index_passes"] == 3
    assert report["metrics"]["retrieval_batch_telemetry_integrity"] is True


def test_smoke_report_rejects_more_than_three_full_index_passes(
    tmp_path: Path,
) -> None:
    contract = _contract()
    protocol = _protocol(tmp_path, contract)
    result = _smoke_result(contract)
    result["retrieval_batch_telemetry"]["full_index_passes"] = 4
    result["retrieval_batch_telemetry"]["backend_batch_invocations"] = 4
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("placeholder\n", encoding="utf-8")
    report = driver.build_gold_free_mechanism_report(
        result=result,
        protocol=protocol,
        scope="smoke",
        rows_lock=freeze.file_lock(rows_path),
        created_at_utc="2026-09-04T00:00:00+00:00",
    )
    assert report["all_pass"] is False
    assert report["gate_results"]["full_index_passes_within_frozen_max"] is False


def test_execute_smoke_fake_is_append_only_and_writes_complete_stages(
    tmp_path: Path,
) -> None:
    contract = _contract()
    protocol = _protocol(tmp_path, contract)
    result = _smoke_result(contract)
    fake_runner = SimpleNamespace(
        runtime_contract=lambda: deepcopy(contract),
        materialize_locked_consumed_smoke4x3_production=(
            lambda **_kwargs: deepcopy(result)
        ),
    )
    verified = {
        "protocol": protocol,
        "protocol_lock": {"path": "protocol.json", "sha256": "a" * 64},
        "model_asset_identity": {
            "base_model_tree_sha256": "1" * 64,
            "adapter_tree_sha256": "2" * 64,
            "tokenizer_tree_sha256": "1" * 64,
        },
        "retrieval_asset_identity": {
            "corpus_sha256": "3" * 64,
            "dense_index_sha256": "4" * 64,
            "bm25_tree_sha256": "5" * 64,
            "e5_tree_sha256": "6" * 64,
            "bge_tree_sha256": "7" * 64,
        },
    }

    completed = driver.execute_scope(
        scope="smoke",
        attempt_number=1,
        project_root=tmp_path,
        verified=verified,
        runner_module=fake_runner,
        hf_factory=lambda **_kwargs: object(),
        retriever_factory=lambda **_kwargs: object(),
    )
    attempt = tmp_path / "smoke_attempt001"
    assert completed["status"] == "COMPLETE"
    assert (attempt / "rows.jsonl").is_file()
    assert (attempt / "report.json").is_file()
    assert (attempt / freeze.RUNNING_MANIFEST).is_file()
    assert (attempt / freeze.COMPLETE_MANIFEST).is_file()
    assert not (attempt / freeze.FAILED_MANIFEST).exists()
    assert freeze.validate_reusable_stage(attempt, "gold_free_rows")
    assert freeze.validate_reusable_stage(attempt, "gold_free_report")

    with pytest.raises(FileExistsError):
        driver.execute_scope(
            scope="smoke",
            attempt_number=1,
            project_root=tmp_path,
            verified=verified,
            runner_module=fake_runner,
            hf_factory=lambda **_kwargs: object(),
            retriever_factory=lambda **_kwargs: object(),
        )


def test_execute_failure_writes_failed_manifest_and_retains_attempt(tmp_path: Path) -> None:
    contract = _contract()
    protocol = _protocol(tmp_path, contract)
    fake_runner = SimpleNamespace(
        runtime_contract=lambda: deepcopy(contract),
        materialize_locked_consumed_smoke4x3_production=(
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("fake crash"))
        ),
    )
    verified = {
        "protocol": protocol,
        "protocol_lock": {"path": "protocol.json", "sha256": "a" * 64},
        "model_asset_identity": {},
        "retrieval_asset_identity": {},
    }
    with pytest.raises(RuntimeError, match="fake crash"):
        driver.execute_scope(
            scope="smoke",
            project_root=tmp_path,
            verified=verified,
            runner_module=fake_runner,
            hf_factory=lambda **_kwargs: object(),
            retriever_factory=lambda **_kwargs: object(),
        )
    attempt = tmp_path / "smoke_attempt001"
    assert (attempt / freeze.FAILED_MANIFEST).is_file()
    assert (attempt / freeze.RUNNING_MANIFEST).is_file()
    assert not (attempt / freeze.COMPLETE_MANIFEST).exists()


def test_attempt002_binds_attempt001_failed_and_rewrites_internal_identity(
    tmp_path: Path,
) -> None:
    contract = _contract()
    protocol_lock = {"path": "protocol-v2.json", "sha256": "a" * 64}
    first = tmp_path / "smoke_attempt001"
    freeze.reserve_attempt_directory(
        attempt_dir=first,
        experiment_id=freeze.SMOKE_EXPERIMENT_ID,
        attempt_id="attempt001",
        implementation_protocol={"path": "protocol-v1.json", "sha256": "b" * 64},
        cohort_lock={"sha256": freeze.EXPECTED_SMOKE_COHORT_SHA256, "row_count": 12},
        created_at_utc="2026-09-04T00:00:00+00:00",
    )
    freeze.finalize_attempt(
        attempt_dir=first,
        success=False,
        reason="retained sequential latency failure",
        completed_at_utc="2026-09-04T00:01:00+00:00",
    )
    failed_lock = freeze.file_lock(first / freeze.FAILED_MANIFEST)
    protocol = _protocol(tmp_path, contract)
    protocol["run_registry"]["smoke"]["attempt002_retry_parent"] = failed_lock
    result = _smoke_result(contract)
    fake_runner = SimpleNamespace(
        runtime_contract=lambda: deepcopy(contract),
        materialize_locked_consumed_smoke4x3_production=(
            lambda **_kwargs: deepcopy(result)
        ),
    )
    verified = {
        "protocol": protocol,
        "protocol_lock": protocol_lock,
        "model_asset_identity": {},
        "retrieval_asset_identity": {},
    }
    completed = driver.execute_scope(
        scope="smoke",
        attempt_number=2,
        project_root=tmp_path,
        verified=verified,
        runner_module=fake_runner,
        hf_factory=lambda **_kwargs: object(),
        retriever_factory=lambda **_kwargs: object(),
    )
    second = tmp_path / "smoke_attempt002"
    running = freeze.read_json(second / freeze.RUNNING_MANIFEST)
    report = freeze.read_json(second / "report.json")
    terminal = freeze.read_json(second / freeze.COMPLETE_MANIFEST)
    assert completed["experiment_id"] == freeze.SMOKE_EXPERIMENT_ID_ATTEMPT002
    assert running["retry_of"] == failed_lock
    assert report["experiment_id"] == freeze.SMOKE_EXPERIMENT_ID_ATTEMPT002
    assert report["intended_output_dir"] == str(second.resolve())
    assert report["retry_of_attempt001_contract"] is True
    assert report["original_first_attempt_contract"]["experiment_id"] == (
        freeze.SMOKE_EXPERIMENT_ID
    )
    assert terminal["experiment_id"] == freeze.SMOKE_EXPERIMENT_ID_ATTEMPT002


def test_development_prerequisite_rejects_tampered_complete_terminal(
    tmp_path: Path,
) -> None:
    contract = _contract()
    protocol = _protocol(tmp_path, contract)
    result = _smoke_result(contract)
    fake_runner = SimpleNamespace(
        runtime_contract=lambda: deepcopy(contract),
        materialize_locked_consumed_smoke4x3_production=(
            lambda **_kwargs: deepcopy(result)
        ),
    )
    verified = {
        "protocol": protocol,
        "protocol_lock": {"path": "protocol.json", "sha256": "a" * 64},
        "model_asset_identity": {},
        "retrieval_asset_identity": {},
    }
    driver.execute_scope(
        scope="smoke",
        project_root=tmp_path,
        verified=verified,
        runner_module=fake_runner,
        hf_factory=lambda **_kwargs: object(),
        retriever_factory=lambda **_kwargs: object(),
    )
    assert driver._validate_prior_smoke_pass(protocol)

    terminal_path = tmp_path / "smoke_attempt001" / freeze.COMPLETE_MANIFEST
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["status"] = "COMPLETE_BUT_TAMPERED"
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    with pytest.raises(
        driver.V8ProductionDriverError,
        match="terminal manifest is invalid",
    ):
        driver._validate_prior_smoke_pass(protocol)


def test_public_cli_has_only_scope_and_attempt_controls() -> None:
    source = Path(driver.__file__).read_text(encoding="utf-8")
    cli = source[source.index("def parse_args") :]
    assert '"--scope"' in cli
    assert '"--attempt"' in cli
    for forbidden in (
        "--cohort",
        "--output",
        "--model",
        "--retriever",
        "--gold",
        "--prospective",
        "--resume",
        "--implementation",
    ):
        assert forbidden not in cli
