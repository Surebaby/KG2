#!/usr/bin/env python
"""Freeze the approved two-controller-call design for v8 development.

This artifact supersedes the contradictory q2 timing/fallback prose in the
earlier narrative plan.  It is a design freeze, not an implementation lock:
runtime code/model/index hashes must be bound separately after the runner and
its engineering smoke pass.  The command loads only the already locked
development cohort.  It neither opens nor hashes the sealed prospective rows.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgproweight.retrieval.dynamic_decomposition_v8_cohort import (  # noqa: E402
    COHORT_LOADER_VERSION,
    EXPECTED_DEVELOPMENT_SHA256,
    EXPECTED_MANIFEST_SHA256,
    SEALED_PROSPECTIVE_SHA256,
    load_frozen_v8_cohort,
)


SCHEMA_VERSION = "dynamic-decomposition-v8-two-call-design-protocol-1"
MANIFEST_SCHEMA_VERSION = "dynamic-decomposition-v8-two-call-design-manifest-1"
EXPERIMENT_ID = (
    "SUBQUESTION-DECOMPOSITION-V8-TWO-CALL-DESIGN-"
    "DEV90-SEED42-V1"
)
STATUS = "FROZEN_APPROVED_BEFORE_V8_RUNNER_EXECUTION"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/audits/"
    "subquestion_decomposition_v8_two_call_design_dev90_seed20260904_v1"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_protocol(
    *,
    generated_at_utc: str,
    cohort_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete approved design payload without runtime side effects."""

    if int(cohort_lock.get("row_count", -1)) != 90:
        raise ValueError("two-call design requires the frozen development90 cohort")
    if cohort_lock.get("development_sha256") != EXPECTED_DEVELOPMENT_SHA256:
        raise ValueError("development cohort SHA mismatch")
    if cohort_lock.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("cohort manifest SHA mismatch")

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": STATUS,
        "created_at_utc": generated_at_utc,
        "researcher_authorization": {
            "decision": "APPROVED_IN_THREAD",
            "date": "2026-09-04",
            "authorized_scope": "ZERO_TRAINING_LOCAL_DEVELOPMENT90_ONLY",
            "evaluation_protocol_change_approved": True,
            "prospective_validation_authorized": False,
            "gold_attachment_authorized": False,
            "training_authorized": False,
            "reward_or_loss_change_authorized": False,
            "rrf_output_top100_approved": True,
            "single_strong_sft_model_for_all_generation_roles_approved": True,
        },
        "supersession": {
            "is_authoritative": True,
            "supersedes_narrative_rules": [
                "B q2_static must be generated before q1 retrieval",
                "all C failures reuse a pre-generated B q2_static action",
                "B and C joint physical call counts must be identical",
            ],
            "reason": (
                "Those three rules cannot coexist with exactly two logical "
                "controller calls per standalone arm."
            ),
        },
        "scope": {
            "role": "development",
            "datasets": ["hotpotqa", "2wikimultihopqa", "musique"],
            "n_per_dataset": 30,
            "itt_n": 90,
            "selection_seed": 20260904,
            "generation_seed": 42,
            "cohort_loader_version": COHORT_LOADER_VERSION,
            "cohort_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "development_sha256": EXPECTED_DEVELOPMENT_SHA256,
            "prospective_sha256_declared_not_opened": SEALED_PROSPECTIVE_SHA256,
            "prospective_unlocked": False,
        },
        "estimands": {
            "primary": (
                "C-B: answer-and-bound-evidence-conditioned q2 policy versus "
                "observation-blind q2 policy"
            ),
            "secondary": "C-A: complete dynamic system versus canonical one-shot",
            "post_treatment_only": ["a1_admissible", "dynamic_q2_valid"],
            "all_rows_retained_in_itt": True,
        },
        "arm_contract": {
            "A": {
                "name": "canonical_one_shot",
                "logical_calls": {
                    "retrieval": 1,
                    "controller": 0,
                    "q1_reader": 0,
                    "final_reader": 1,
                },
            },
            "B": {
                "name": "observation_blind_q2_control",
                "logical_calls": {
                    "retrieval": 3,
                    "controller": 2,
                    "q1_reader": 1,
                    "final_reader": 1,
                },
                "slot2_state_fields_exact": [
                    "state_version",
                    "mode",
                    "gold_access",
                    "original_question",
                    "q1_query",
                    "verified_subanswer",
                ],
                "slot2_verified_subanswer_literal": "NO_VERIFIED_SUBANSWER",
                "slot2_forbidden_information": [
                    "root_observation",
                    "q1_documents",
                    "retrieval_scores",
                    "reader_raw_output",
                    "verified_a1",
                    "binding_telemetry",
                    "arm_label",
                    "gold",
                ],
            },
            "C": {
                "name": "answer_conditioned_dynamic_q2",
                "logical_calls": {
                    "retrieval": 3,
                    "controller": 2,
                    "q1_reader": 1,
                    "final_reader": 1,
                },
                "eligible_slot2_state_fields_exact": [
                    "state_version",
                    "mode",
                    "gold_access",
                    "original_question",
                    "q1_query",
                    "verified_subanswer",
                    "bound_evidence",
                ],
                "ineligible_slot2": "BYTE_IDENTICAL_TO_B_STATIC_PROMPT_AND_CONTENT_KEY",
                "eligible_dynamic_invalid": "DETERMINISTIC_ORIGINAL_QUESTION_FALLBACK",
                "third_controller_call_allowed": False,
            },
        },
        "shared_state": {
            "root_query_and_top10_A_B_C_byte_identical": True,
            "q1_controller_prompt_output_query_B_C_byte_identical": True,
            "q1_top10_reader_output_binding_B_C_byte_identical": True,
            "q1_binding_document_priority_applies_to_B_and_C": True,
        },
        "fallback_matrix": [
            {
                "condition": "q1_controller_output_invalid",
                "selected_query": "original_question",
                "ordinary_fallback": True,
            },
            {
                "condition": "B_q2_static_output_invalid",
                "selected_query": "original_question",
                "ordinary_fallback": True,
            },
            {
                "condition": "C_a1_inadmissible",
                "slot2": "same_static_sentinel_prompt_and_content_key_as_B",
                "selected_query": "same_as_B",
                "byte_identity_required_through_prediction": True,
            },
            {
                "condition": "C_a1_admissible_and_dynamic_output_invalid",
                "slot2": "dynamic_call_already_consumed",
                "selected_query": "original_question",
                "read_B_static_artifact": False,
                "third_controller_call": False,
                "retain_in_itt": True,
            },
            {
                "condition": "retriever_or_model_runtime_error",
                "ordinary_fallback": False,
                "retain_failure_row": True,
                "runtime_gate_fails": True,
            },
        ],
        "admissibility": {
            "source": "frozen_unique_document_surface_locality_binder",
            "claim_boundary": "mechanical_lexical_locality_not_semantic_entailment",
            "dynamic_transition_valid_definition": (
                "a1_admissible AND generated_q2_valid AND selection_source=q2_dynamic"
            ),
            "dynamic_transition_denominator": "all_90_ITT_rows",
        },
        "passage_budget": {
            "retriever_stack": (
                "E5_top100 + BM25_top100 + RRF_k60_top100 + BGE_rerank_top10"
            ),
            "dense_candidate_top_k": 100,
            "sparse_candidate_top_k": 100,
            "rrf_k": 60,
            "rrf_output_candidate_top_k": 100,
            "cross_encoder_top_k": 10,
            "logical_retrieval_slots_B_C": ["root", "q1", "q2"],
            "final_merge": "root6+q1_novel2+q2_novel2+root_rank7_10_backfill",
            "final_passages": 10,
            "unique_documents_required": 10,
            "fallback_retrieval_is_still_a_logical_request": True,
        },
        "generation_assets_and_settings": {
            "base_model": "models/llama3-8b",
            "adapter": (
                "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_"
                "no_text_head/final"
            ),
            "roles_share_one_loaded_model": [
                "q1_controller",
                "q2_controller",
                "q1_subanswer_reader",
                "final_reader",
            ],
            "load_dtype": "bfloat16",
            "device": "cuda:0",
            "local_files_only": True,
            "decoding": "greedy",
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "seed": 42,
            "max_new_tokens": {
                "controller": 96,
                "q1_subanswer_reader": 96,
                "final_reader": 512,
            },
        },
        "call_and_cache_accounting": {
            "logical_requests_equal_cache_hits_plus_cache_misses": True,
            "logical_budget_B_C_identical": True,
            "joint_physical_budget_B_C_identity_required": False,
            "physical_execution_definition": "unique_validated_content_key_cache_miss",
            "report": [
                "per_arm_logical_requests",
                "joint_physical_executions",
                "cache_hits",
                "cache_misses",
                "standalone_equivalent_cost",
                "prompt_tokens",
                "generated_tokens",
                "wall_time",
            ],
            "controller_cache_key_fields_minimum": [
                "role",
                "base_model_hash",
                "adapter_hash",
                "tokenizer_hash",
                "chat_template_hash",
                "model_visible_prompt_bytes",
                "model_visible_token_ids",
                "decoding",
                "seed",
                "max_new_tokens",
            ],
            "retrieval_cache_key_fields_minimum": [
                "query_bytes",
                "E5_hash",
                "BM25_hash",
                "RRF_config_hash",
                "reranker_hash",
                "index_hash",
                "retrieval_config_hash",
            ],
            "cache_key_forbidden_fields": ["arm", "outcome", "gold"],
        },
        "gold_and_seal_boundary": {
            "materializer_gold_access": False,
            "materializer_accepts_arbitrary_cohort_path": False,
            "prospective_rejected_before_open_or_hash": True,
            "predictions_frozen_before_independent_gold_join": True,
            "score_job_may_regenerate_or_retry": False,
            "official_support_source": "UNKNOWN",
            "support_gate_until_source_versioned": "NOT_AVAILABLE",
        },
        "gold_free_mechanism_gates": {
            "itt_cardinality": {"total": 90, "per_dataset": 30},
            "q1_schema_valid_rate_min_each_dataset": 0.95,
            "B_q2_static_valid_rate_min": 0.90,
            "a1_admissible_rate_min_each_dataset": 0.40,
            "C_dynamic_transition_rate_all_itt_min_each_dataset": 0.32,
            "empty_repeat_padding_query_rate_max": 0.05,
            "logical_ledger_exact_rate": 1.0,
            "logical_B_C_budget_identity_rate": 1.0,
            "cache_accounting_conservation_rate": 1.0,
            "B_static_allowlist_rate": 1.0,
            "C_dynamic_state_binding_integrity_rate": 1.0,
            "a1_ineligible_full_content_identity_rate": 1.0,
            "eligible_dynamic_invalid_original_Q_no_third_call_rate": 1.0,
            "root_and_q1_shared_byte_identity_rate": 1.0,
            "final_10_unique_rate": 1.0,
            "runtime_error_count": 0,
            "gold_or_forbidden_recursive_field_access_count": 0,
        },
        "development_outcome_gates_after_gold_free_pass": {
            "primary_C_minus_B_pooled_EM_min": 0.05,
            "primary_C_minus_B_pooled_F1_strictly_positive": True,
            "gained_minus_lost_min_each_dataset": -1,
            "a1_ineligible_prediction_byte_identity_rate": 1.0,
            "prospective_unlock_requires_new_researcher_approval": True,
        },
        "implementation_freeze_required_before_execution": [
            "runner_and_prompt_builder_hashes",
            "cohort_loader_and_core_hashes",
            "model_base_adapter_tokenizer_chat_template_tree_hashes",
            "retrieval_index_and_config_hashes",
            "generation_and_context_truncation_settings",
            "append_only_output_and_resume_semantics",
            "engineering_smoke_report_and_test_inventory",
        ],
        "cohort_runtime_lock": dict(cohort_lock),
    }


def freeze_protocol(
    *,
    project_root: Path,
    output_dir: Path,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    output_path = output_dir if output_dir.is_absolute() else project_root / output_dir
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite append-only protocol: {output_path}")

    cohort = load_frozen_v8_cohort(role="development")
    cohort_lock = {
        "loader_version": cohort["loader_version"],
        "manifest_sha256": cohort["manifest_sha256"],
        "development_sha256": cohort["cohort_sha256"],
        "row_count": len(cohort["rows"]),
        "prospective_opened_or_hashed_by_this_command": False,
    }
    timestamp = generated_at_utc or datetime.now(timezone.utc).isoformat()
    protocol = build_protocol(generated_at_utc=timestamp, cohort_lock=cohort_lock)

    output_path.mkdir(parents=True, exist_ok=False)
    protocol_path = output_path / "protocol.json"
    _write_json(protocol_path, protocol)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": STATUS,
        "created_at_utc": timestamp,
        "gold_access": False,
        "prospective_opened_or_hashed": False,
        "outputs": [
            {
                "path": "protocol.json",
                "sha256": _sha256_file(protocol_path),
                "size_bytes": protocol_path.stat().st_size,
            }
        ],
        "implementation_inventory": [
            {
                "path": Path(__file__).resolve().relative_to(project_root).as_posix(),
                "sha256": _sha256_file(Path(__file__).resolve()),
                "size_bytes": Path(__file__).resolve().stat().st_size,
            }
        ],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    _write_json(output_path / "manifest.json", manifest)
    return {"protocol": protocol, "manifest": manifest, "output_dir": output_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = freeze_protocol(project_root=PROJECT_ROOT, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": STATUS,
                "output_dir": str(result["output_dir"]),
                "protocol_sha256": result["manifest"]["outputs"][0]["sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
