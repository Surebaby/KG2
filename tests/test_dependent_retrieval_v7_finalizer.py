from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare import finalize_paired_dependent_retrieval_v7 as finalizer
from scripts.prepare import freeze_dependent_retrieval_v7 as freeze
from kgproweight.utils.logging import artifact_identity


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return path


def _dummy(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")
    return path


def _passage(document_id: str) -> dict[str, str]:
    return {
        "id": document_id,
        "title": f"Title {document_id}",
        "contents": f"Title {document_id}\nBody for {document_id}.",
    }


def _budget(
    *,
    key: str,
    dataset: str,
    qid: str,
    question: str,
    slot: str,
    depth: int,
    root: bool,
) -> dict:
    suffix = "root" if root else "dependent"
    query_b = f"root query {qid}" if root else question + "\nB bridge relation"
    query_c = query_b if root else question + "\nC bridge relation"
    physical_b = f"root-shared::{qid}" if root else f"B::{key}::{slot}::{suffix}"
    physical_c = physical_b if root else f"C::{key}::{slot}::{suffix}"
    hop_hash = hashlib.sha256(f"{key}:{slot}:{depth}".encode()).hexdigest()
    logical = f"{key}::{slot}::{suffix}"
    return {
        "schema_version": finalizer.EXPECTED_BUDGET_SCHEMA,
        "question_key": key,
        "dataset": dataset,
        "qid": qid,
        "logical_hop_id": slot,
        "logical_hop_sha256": hop_hash,
        "dependency_depth": depth,
        "is_root": root,
        "paired_active": True,
        "paired_skip_reason": None,
        "B": {
            "logical_query_count": 1,
            "physical_slot_count": 1,
            "logical_slot_id": f"B::{logical}",
            "physical_slot_id": physical_b,
            "query": query_b,
            "query_sha256": hashlib.sha256(query_b.encode()).hexdigest(),
        },
        "C": {
            "logical_query_count": 1,
            "physical_slot_count": 1,
            "logical_slot_id": f"C::{logical}",
            "physical_slot_id": physical_c,
            "query": query_c,
            "query_sha256": hashlib.sha256(query_c.encode()).hexdigest(),
        },
        "actual_shared_physical_search_count": 1 if root else 0,
        "actual_independent_physical_search_count": 0 if root else 2,
        "cross_arm_query_strings_identical": root,
        "budget_equal": True,
        "gold_access": False,
    }


def _materialization_rows(
    *, n_per_dataset: int = 20
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    arms = {"A": [], "B": [], "C": []}
    details: list[dict] = []
    budgets: list[dict] = []
    for dataset in finalizer.DATASETS:
        for index in range(n_per_dataset):
            qid = f"{dataset}-{index}"
            key = f"{dataset}::{qid}"
            question = f"Which place is linked to synthetic item {qid}?"
            qhash = hashlib.sha256(question.encode()).hexdigest()
            family = hashlib.sha256(f"family:{key}".encode()).hexdigest()
            original = [_passage(f"{qid}-a-{rank}") for rank in range(10)]
            passages = {
                "A": original,
                "B": [*original[:9], _passage(f"{qid}-b-new")],
                "C": [*original[:9], _passage(f"{qid}-c-new")],
            }
            common = {
                "schema_version": finalizer.EXPECTED_ARM_SCHEMA,
                "row_id": f"dependent-retrieval-v7::{key}",
                "question_key": key,
                "dataset": dataset,
                "qid": qid,
                "question": question,
                "question_sha256": qhash,
                "family_sha256": family,
                "role": "development_consumed",
                "kg_subgraph": [],
                "gold_access": False,
            }
            arms["A"].append(
                {
                    **common,
                    "arm": finalizer.ARMS[0],
                    "retrieved_passages": passages["A"],
                    "passages_sha256": finalizer._passages_sha256(passages["A"]),
                }
            )
            ce_trace: dict[str, list[dict]] = {}
            merge: dict[str, dict] = {}
            safety: dict[str, dict] = {}
            for arm in ("B", "C"):
                new_key = finalizer.passage_score_key(passages[arm][-1])
                old_key = finalizer.passage_score_key(original[-1])
                ce_trace[arm] = [
                    {
                        "arm": arm,
                        "question": question,
                        "question_sha256": qhash,
                        "document_key": new_key,
                        "document_id": passages[arm][-1]["id"],
                        "document_text_sha256": hashlib.sha256(
                            passages[arm][-1]["contents"].encode()
                        ).hexdigest(),
                        "score": 2.0,
                        "uses_exact_original_question": True,
                    }
                ]
                merge[arm] = {
                    "selected_new": [{"document_key": new_key}],
                    "evicted_originals": [
                        {
                            "document_key": old_key,
                            "original_rank": 10,
                            "score": 1.0,
                            "replacement_score": 2.0,
                        }
                    ],
                }
                safety[arm] = {
                    "output_count": 10,
                    "prefix8_exact": True,
                    "unauthorized_original_displacements": 0,
                    "root_passages_injected": 0,
                    "duplicate_output_documents": 0,
                    "fallback_exact": False,
                }
                arms[arm].append(
                    {
                        **common,
                        "arm": finalizer.ARMS[1 if arm == "B" else 2],
                        "retrieved_passages": passages[arm],
                        "passages_sha256": finalizer._passages_sha256(passages[arm]),
                        "fallback_to_a": False,
                        "retrieval_trace": {
                            "successful_paired_dependent_hops": 1,
                            "dependent_query_count": 1,
                            "fallback_reason": None,
                            "merge": merge[arm],
                            "safety": safety[arm],
                            "final_ce_trace": ce_trace[arm],
                            "gold_access": False,
                        },
                    }
                )

            root_budget = _budget(
                key=key,
                dataset=dataset,
                qid=qid,
                question=question,
                slot="hop_1",
                depth=1,
                root=True,
            )
            dep_budget = _budget(
                key=key,
                dataset=dataset,
                qid=qid,
                question=question,
                slot="hop_2",
                depth=2,
                root=False,
            )
            budgets.extend([root_budget, dep_budget])
            producer_hash = hashlib.sha256(f"producer:{key}".encode()).hexdigest()
            hop_root = {
                "logical_hop_id": "hop_1",
                "logical_hop_sha256": root_budget["logical_hop_sha256"],
                "step_index": 1,
                "dependency_depth": 1,
                "dependencies": [],
                "root_query_shared": True,
                "gold_access": False,
            }
            hop_dep = {
                "logical_hop_id": "hop_2",
                "logical_hop_sha256": dep_budget["logical_hop_sha256"],
                "step_index": 2,
                "dependency_depth": 2,
                "dependencies": ["hop_1"],
                "paired_active": True,
                "paired_skip_reason": None,
                "B": {
                    "query": dep_budget["B"]["query"],
                    "query_sha256": dep_budget["B"]["query_sha256"],
                    "rerank": {
                        "ce_pairs": [
                            {
                                "document_key": finalizer.passage_score_key(
                                    passages["B"][-1]
                                ),
                                "selected_rank": 1,
                            }
                        ]
                    },
                },
                "C": {
                    "query": dep_budget["C"]["query"],
                    "query_sha256": dep_budget["C"]["query_sha256"],
                    "rerank": {
                        "ce_pairs": [
                            {
                                "document_key": finalizer.passage_score_key(
                                    passages["C"][-1]
                                ),
                                "selected_rank": 1,
                            }
                        ]
                    },
                },
                "gold_access": False,
            }
            details.append(
                {
                    "schema_version": finalizer.EXPECTED_REPORT_SCHEMA,
                    "runner_version": "synthetic-v7-runner-1",
                    "question_key": key,
                    "dataset": dataset,
                    "qid": qid,
                    "question": question,
                    "question_sha256": qhash,
                    "family_sha256": family,
                    "target_type": "relation_graph"
                    if dataset == "hotpotqa"
                    else "subquery_graph",
                    "plan_sha256": hashlib.sha256(f"plan:{key}".encode()).hexdigest(),
                    "plan_executable": True,
                    "plan_validation_errors": [],
                    "has_dependent_step": True,
                    "execution_status": "dependent_retrieval_complete",
                    "fallback_reason": None,
                    "successful_paired_dependent_hops": 1,
                    "hop_telemetry": [hop_root, hop_dep],
                    "subanswer_telemetry": [
                        {
                            "task_id": hashlib.sha256(f"task:{key}".encode()).hexdigest(),
                            "producer_slot": "hop_1",
                            "step_sha256": hashlib.sha256(f"step:{key}".encode()).hexdigest(),
                            "producer_passages_sha256": producer_hash,
                            "verified": True,
                            "promoted_value": "Synthetic Bridge",
                            "telemetry": {
                                "strict_parse": {"valid": True, "error_code": None},
                                "prompt_passages_sha256": producer_hash,
                                "verifier_passages_sha256": producer_hash,
                                "same_passage_bytes_for_prompt_and_verifier": True,
                                "verification": {
                                    "verification_scope": "surface_locality_not_semantic_entailment",
                                    "verified": True,
                                    "verified_answer": "Synthetic Bridge",
                                },
                                "gold_access": False,
                            },
                            "gold_access": False,
                        }
                    ],
                    "budget_ledger": [root_budget, dep_budget],
                    "merge": merge,
                    "safety": safety,
                    "arm_a_passages_sha256": finalizer._passages_sha256(passages["A"]),
                    "arm_b_passages_sha256": finalizer._passages_sha256(passages["B"]),
                    "arm_c_passages_sha256": finalizer._passages_sha256(passages["C"]),
                    "all_dependent_queries_start_with_exact_original_question": True,
                    "all_final_ce_pairs_use_exact_original_question": True,
                    "gold_access": False,
                }
            )
    return arms["A"], arms["B"], arms["C"], details, budgets


def _full_synthetic_chain(tmp_path: Path) -> dict[str, Path]:
    a, b, c, details, budgets = _materialization_rows()
    base = tmp_path / "models" / "base"
    adapter = tmp_path / "models" / "strong_sft"
    _dummy(base / "config.json", "{}")
    _dummy(base / "model.safetensors.index.json", "{}")
    _dummy(adapter / "adapter_config.json", "{}")
    _dummy(adapter / "adapter_model.safetensors", "synthetic-weights")
    base_tree = freeze.tree_lock(base)
    adapter_tree = freeze.tree_lock(adapter)
    model_artifact = {
        "base_model": {
            "identity": artifact_identity(base),
            "config": finalizer._file_lock(base / "config.json"),
            "weight_index": finalizer._file_lock(
                base / "model.safetensors.index.json"
            ),
        },
        "strong_sft_adapter": {
            "identity": artifact_identity(adapter),
            "config": finalizer._file_lock(adapter / "adapter_config.json"),
            "weights": finalizer._file_lock(adapter / "adapter_model.safetensors"),
        },
        "load_contract": {
            "torch_dtype": "bfloat16",
            "device": "cuda:0",
            "local_files_only": True,
            "separate_process_required": True,
        },
    }
    for detail in details:
        for attempt in detail["subanswer_telemetry"]:
            attempt["telemetry"]["model_artifact"] = deepcopy(model_artifact)
    material = tmp_path / "material"
    paths = {
        "arm_a": _write_jsonl(material / "arm_a.jsonl", a),
        "arm_b": _write_jsonl(material / "arm_b.jsonl", b),
        "arm_c": _write_jsonl(material / "arm_c.jsonl", c),
        "execution_details": _write_jsonl(material / "execution_details.jsonl", details),
        "budget_ledger": _write_jsonl(material / "budget_ledger.jsonl", budgets),
    }
    order = [row["question_key"] for row in a]
    prereg = {
        "schema_version": freeze.SCHEMA_VERSION,
        "status": freeze.STATUS,
        "scope": freeze.SCOPE,
        "population": {
            "n": 40,
            "by_dataset": {"hotpotqa": 20, "musique": 20},
            "globally_fresh": False,
            "independent_confirmation": False,
            "new_role": "development_consumed",
            "question_key_order_sha256": hashlib.sha256(
                "\n".join(order).encode()
            ).hexdigest(),
        },
        "decision_gates": {
            "materialization": freeze.MATERIALIZATION_GATES,
            "gold_free_mechanism": freeze.MECHANISM_GATES,
            "development_utility": freeze.UTILITY_GATES,
        },
        "models": {
            "subanswer_and_final_strong_sft": artifact_identity(adapter),
            "base_model": artifact_identity(base),
            "inherited_content_locks": {
                "strong_sft": adapter_tree,
                "base_model": base_tree,
            },
        },
    }
    prereg_path = _write_json(tmp_path / "prereg" / "protocol.json", prereg)
    prereg_lock = finalizer._file_lock(prereg_path)
    prereg_manifest_path = _write_json(
        tmp_path / "prereg" / "manifest.json",
        {
            "status": freeze.STATUS,
            "gold_access": False,
            "artifacts": {"protocol": prereg_lock},
        },
    )
    prereg_manifest_lock = finalizer._file_lock(prereg_manifest_path)
    addendum = {
        "schema_version": "subquestion-dependent-retrieval-v7-effective-addendum-1",
        "status": "FROZEN_APPEND_ONLY_ADDENDUM_BEFORE_V7_GPU_OR_RETRIEVAL",
        "parents": {"parent_preregistration": prereg_lock},
        "effective_invariants": {
            "producer_passages_max": 10,
            "producer_text_unicode_chars_max_each": 1200,
            "reader_and_verifier_projection_hash_equal": True,
        },
    }
    addendum_path = _write_json(tmp_path / "addendum" / "protocol.json", addendum)
    addendum_lock = finalizer._file_lock(addendum_path)
    addendum_manifest_path = _write_json(
        tmp_path / "addendum" / "manifest.json",
        {
            "status": "FROZEN_APPEND_ONLY_ADDENDUM_BEFORE_V7_GPU_OR_RETRIEVAL",
            "gold_access": False,
            "protocol": addendum_lock,
        },
    )
    addendum_manifest_lock = finalizer._file_lock(addendum_manifest_path)

    design_status = "RULES_AND_SELECTION_ALGORITHM_FROZEN_BEFORE_V7_GPU_OR_RETRIEVAL"
    design_path = _write_json(
        tmp_path / "design" / "protocol.json",
        {
            "schema_version": "subquestion-dependent-retrieval-v7-design-freeze-1",
            "status": design_status,
        },
    )
    design_lock = finalizer._file_lock(design_path)
    design_manifest_path = _write_json(
        tmp_path / "design" / "manifest.json",
        {"status": design_status, "gold_access": False, "protocol": design_lock},
    )
    design_manifest_lock = finalizer._file_lock(design_manifest_path)
    design_trajectory_path = _write_json(
        tmp_path / "design" / "trajectory.json",
        {
            "schema_version": (
                "subquestion-dependent-retrieval-v7-recursive-trajectory-design-addendum-1"
            ),
            "status": finalizer.EXPECTED_TRAJECTORY_ADDENDUM_STATUS,
            "gold_access": False,
        },
    )
    design_trajectory_lock = finalizer._file_lock(design_trajectory_path)
    design_trajectory_manifest_path = _write_json(
        tmp_path / "design" / "trajectory.manifest.json",
        {
            "status": finalizer.EXPECTED_TRAJECTORY_ADDENDUM_STATUS,
            "gold_access": False,
            "addendum": design_trajectory_lock,
        },
    )
    design_trajectory_manifest_lock = finalizer._file_lock(
        design_trajectory_manifest_path
    )
    trajectory = {
        "schema_version": finalizer.EXPECTED_TRAJECTORY_ADDENDUM_SCHEMA,
        "status": finalizer.EXPECTED_TRAJECTORY_ADDENDUM_STATUS,
        "scope": finalizer.EXPECTED_TRAJECTORY_ADDENDUM_SCOPE,
        "parents": {
            "design_protocol": design_lock,
            "design_manifest": design_manifest_lock,
            "parent_preregistration": prereg_lock,
            "parent_preregistration_manifest": prereg_manifest_lock,
            "producer_truncation_addendum": addendum_lock,
            "producer_truncation_addendum_manifest": addendum_manifest_lock,
            "design_trajectory_addendum": design_trajectory_lock,
            "design_trajectory_addendum_manifest": design_trajectory_manifest_lock,
        },
        "effective_invariants": finalizer.EXPECTED_TRAJECTORY_INVARIANTS,
        "gold_access": False,
    }
    trajectory_path = _write_json(tmp_path / "trajectory" / "protocol.json", trajectory)
    trajectory_lock = finalizer._file_lock(trajectory_path)
    trajectory_manifest_path = _write_json(
        tmp_path / "trajectory" / "manifest.json",
        {
            "status": finalizer.EXPECTED_TRAJECTORY_ADDENDUM_STATUS,
            "gold_access": False,
            "protocol": trajectory_lock,
        },
    )
    trajectory_manifest_lock = finalizer._file_lock(trajectory_manifest_path)

    runtime_paths = {
        "retrieval_runner": _dummy(tmp_path / "code" / "retrieval_runner.py", "runner"),
        "subanswer_generator": _dummy(
            tmp_path / "code" / "subanswer_generator.py", "generator"
        ),
        "gold_finalizer": Path(finalizer.__file__),
        "evaluator": Path(
            "scripts/eval/evaluate_paired_dependent_retrieval_v7.py"
        ),
    }
    runtime_code = {
        name: finalizer._file_lock(path) for name, path in runtime_paths.items()
    }
    implementation = {
        "schema_version": finalizer.EXPECTED_IMPLEMENTATION_SCHEMA,
        "status": finalizer.EXPECTED_IMPLEMENTATION_STATUS,
        "experiment_id": freeze.FUTURE_EXPERIMENT_IDS["implementation_lock"],
        "lock_issuer": finalizer._file_lock(
            Path("scripts/prepare/freeze_dependent_retrieval_v7_implementation.py")
        ),
        "parents": {
            "preregistration": prereg_lock,
            "truncation_addendum": addendum_lock,
            "trajectory_semantics_addendum": trajectory_lock,
            "trajectory_semantics_addendum_manifest": trajectory_manifest_lock,
        },
        "runtime_code": runtime_code,
        "actual_local_import_closure": {
            "synthetic_dependency.py": finalizer._file_lock(
                _dummy(tmp_path / "code" / "synthetic_dependency.py", "dependency")
            )
        },
        "content_reverification": {"full_hash_verification_performed": True},
        "authorization": {
            "planner": True,
            "gold_free_materialization": False,
            "gold_attachment": False,
            "answer_evaluation": False,
            "training": False,
        },
        "gold_access": False,
    }
    implementation_path = _write_json(
        tmp_path / "implementation" / "protocol.json", implementation
    )
    implementation_lock = finalizer._file_lock(implementation_path)
    implementation_manifest = finalizer._file_lock(
        _dummy(tmp_path / "implementation" / "manifest.json", "manifest")
    )
    plan_inputs = {
        name: finalizer._file_lock(_dummy(tmp_path / "plan_inputs" / name, name))
        for name in {
            "development",
            "planner_cohort",
            "canonical_A_contexts",
            "planner_predictions",
            "planner_report",
            "planner_manifest",
        }
    }
    plan = {
        "schema_version": finalizer.EXPECTED_PLAN_LOCK_SCHEMA,
        "status": finalizer.EXPECTED_PLAN_LOCK_STATUS,
        "experiment_id": finalizer.EXPECTED_PLAN_LOCK_EXPERIMENT_ID,
        "scope": freeze.SCOPE,
        "lock_issuer": finalizer._file_lock(
            Path("scripts/prepare/freeze_dependent_retrieval_v7_plans.py")
        ),
        "parents": {
            "preregistration": prereg_lock,
            "truncation_addendum": addendum_lock,
            "trajectory_semantics_addendum": trajectory_lock,
            "trajectory_semantics_addendum_manifest": trajectory_manifest_lock,
            "implementation_lock": implementation_lock,
            "implementation_manifest": implementation_manifest,
        },
        "inputs": plan_inputs,
        "runtime_code": runtime_code,
        "population": {
            "n": 40,
            "by_dataset": {"hotpotqa": 20, "musique": 20},
            "plan_executable_gate_pass": True,
            "question_key_order_sha256": prereg["population"][
                "question_key_order_sha256"
            ],
        },
        "materialization_contract": {
            "experiment_id": freeze.FUTURE_EXPERIMENT_IDS["materialization"],
            "runner_version": "synthetic-v7-runner-1",
            "n": 40,
            "by_dataset": {"hotpotqa": 20, "musique": 20},
            "gold_access": False,
            "network_access": False,
            "max_plan_steps": 4,
        },
        "authorization": {
            "planner_complete": True,
            "gold_free_materialization": True,
            "gold_attachment": False,
            "answer_evaluation": False,
            "training": False,
        },
        "gold_access": False,
    }
    plan_path = _write_json(tmp_path / "plan_lock" / "protocol.json", plan)
    plan_lock = finalizer._file_lock(plan_path)

    observed = finalizer.audit_materialization(a, b, c, details, budgets)
    safety_summary = {
        name: observed[name] for name in freeze.MATERIALIZATION_GATES
    }
    report = {
        "schema_version": finalizer.EXPECTED_REPORT_SCHEMA,
        "status": finalizer.EXPECTED_REPORT_STATUS,
        "runner_version": "synthetic-v7-runner-1",
        "experiment_id": freeze.FUTURE_EXPERIMENT_IDS["materialization"],
        "development_only": True,
        "gold_access": False,
        "preregistration": prereg_lock,
        "truncation_addendum": addendum_lock,
        "trajectory_semantics_addendum": trajectory_lock,
        "implementation_lock": implementation_lock,
        "plan_lock": plan_lock,
        "outputs": {name: finalizer._file_lock(path) for name, path in paths.items()},
        "safety_summary": safety_summary,
        "materialization_gate": {"passed": True, "observed": safety_summary},
        "gold_free_mechanism_gate": {"passed": True, "by_dataset": {}},
        "gate_decision": "PASS_READY_FOR_SEPARATE_GOLD_FINALIZER",
        "by_dataset": observed["by_dataset"],
    }
    report_path = _write_json(material / "report.json", report)
    return {
        "preregistration": prereg_path,
        "addendum": addendum_path,
        "trajectory": trajectory_path,
        "implementation": implementation_path,
        "plan_lock": plan_path,
        "report": report_path,
    }


def test_v7_finalizer_recomputes_budget_subanswer_and_mechanism_gates() -> None:
    a, b, c, details, budgets = _materialization_rows(n_per_dataset=2)
    observed = finalizer.audit_materialization(
        a, b, c, details, budgets, expected_per_dataset=2
    )
    assert observed["B_C_query_budget_equal_every_question_depth_and_hop"] is True
    assert observed["unverified_subanswers_used"] == 0
    assert observed["by_dataset"]["hotpotqa"][
        "strict_subanswer_json_parse_rate"
    ] == 1.0
    assert observed["by_dataset"]["musique"][
        "mechanically_verified_subanswer_rate"
    ] == 1.0
    assert observed["by_dataset"]["hotpotqa"][
        "retained_new_dependent_document_question_rate_C"
    ] == 1.0


def test_v7_finalizer_rejects_unequal_hop_budget() -> None:
    a, b, c, details, budgets = _materialization_rows(n_per_dataset=1)
    details[0]["budget_ledger"][1]["C"]["logical_query_count"] = 0
    budgets[1]["C"]["logical_query_count"] = 0
    with pytest.raises(finalizer.V7FinalizationError, match="logical query budgets"):
        finalizer.audit_materialization(
            a, b, c, details, budgets, expected_per_dataset=1
        )


def test_v7_finalizer_rejects_active_query_without_verified_dependency() -> None:
    a, b, c, details, budgets = _materialization_rows(n_per_dataset=1)
    details[0]["subanswer_telemetry"][0]["verified"] = False
    details[0]["subanswer_telemetry"][0]["promoted_value"] = None
    details[0]["subanswer_telemetry"][0]["telemetry"]["verification"][
        "verified"
    ] = False
    details[0]["subanswer_telemetry"][0]["telemetry"]["verification"][
        "verified_answer"
    ] = None
    with pytest.raises(finalizer.V7FinalizationError, match="unverified dependency"):
        finalizer.audit_materialization(
            a, b, c, details, budgets, expected_per_dataset=1
        )


def test_v7_finalizer_rejects_non_original_final_ce_question() -> None:
    a, b, c, details, budgets = _materialization_rows(n_per_dataset=1)
    b[0]["retrieval_trace"]["final_ce_trace"][0]["question"] = "short query"
    with pytest.raises(finalizer.V7FinalizationError, match="exact original question"):
        finalizer.audit_materialization(
            a, b, c, details, budgets, expected_per_dataset=1
        )


def test_v7_valid_synthetic_chain_passes_without_gold(tmp_path: Path) -> None:
    paths = _full_synthetic_chain(tmp_path)
    bundle = finalizer.validate_gold_free_materialization(
        report_path=paths["report"],
        preregistration_path=paths["preregistration"],
        truncation_addendum_path=paths["addendum"],
        trajectory_semantics_addendum_path=paths["trajectory"],
        implementation_lock_path=paths["implementation"],
        plan_lock_path=paths["plan_lock"],
        enforce_canonical_trajectory_hash=False,
    )
    assert bundle["observed"]["n"] == 40
    assert bundle["observed"]["gold_access"] is False


def test_v7_failed_mechanism_gate_never_opens_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _full_synthetic_chain(tmp_path)
    monkeypatch.setattr(
        finalizer,
        "EXPECTED_TRAJECTORY_ADDENDUM_SHA256",
        finalizer._file_lock(paths["trajectory"])["sha256"],
    )
    monkeypatch.setattr(
        finalizer,
        "EXPECTED_TRAJECTORY_ADDENDUM_MANIFEST_SHA256",
        finalizer._file_lock(paths["trajectory"].with_name("manifest.json"))["sha256"],
    )
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    report["by_dataset"]["hotpotqa"]["strict_subanswer_json_parse_rate"] = 0.49
    _write_json(paths["report"], report)
    opened = False

    def forbidden_gold_open(_paths: object) -> object:
        nonlocal opened
        opened = True
        raise AssertionError("Gold reader must not be called")

    monkeypatch.setattr(finalizer, "_index_raw_gold", forbidden_gold_open)
    output = tmp_path / "finalizer-output"
    missing_gold = tmp_path / "must-not-be-opened" / "dev.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "finalize_paired_dependent_retrieval_v7.py",
            "--retrieval_report",
            str(paths["report"]),
            "--preregistration",
            str(paths["preregistration"]),
            "--truncation_addendum",
            str(paths["addendum"]),
            "--implementation_lock",
            str(paths["implementation"]),
            "--trajectory_semantics_addendum",
            str(paths["trajectory"]),
            "--plan_lock",
            str(paths["plan_lock"]),
            "--hotpot_dev",
            str(missing_gold),
            "--musique_dev",
            str(missing_gold),
            "--output_dir",
            str(output),
            "--experiment_id",
            freeze.FUTURE_EXPERIMENT_IDS["gold_attachment"],
            "--evaluation_experiment_id",
            freeze.FUTURE_EXPERIMENT_IDS["evaluation"],
        ],
    )
    with pytest.raises(finalizer.V7FinalizationError, match="runner/recomputed metric"):
        finalizer.main()
    assert opened is False
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"].startswith("FAILED_RUNTIME")
    assert not (output / "arm_a.scored.jsonl").exists()


def test_attach_scorer_gold_preserves_strict_three_arm_pairing() -> None:
    a, b, c, _, _ = _materialization_rows(n_per_dataset=1)
    index = {
        row["question_key"]: {
            "question": row["question"],
            "gold_answers": ["Synthetic Place"],
        }
        for row in a
    }
    scored = finalizer.attach_scorer_gold({"A": a, "B": b, "C": c}, index)
    for rows in scored.values():
        assert all(row["gold_answers"] == ["Synthetic Place"] for row in rows)
        assert all(
            row["gold_attachment"] == "SCORER_ONLY_AFTER_ALL_GOLD_FREE_GATES"
            for row in rows
        )


def test_plan_lock_is_mandatory_in_gold_free_chain(tmp_path: Path) -> None:
    paths = _full_synthetic_chain(tmp_path)
    plan = json.loads(paths["plan_lock"].read_text(encoding="utf-8"))
    plan["authorization"]["gold_free_materialization"] = False
    _write_json(paths["plan_lock"], plan)
    with pytest.raises(finalizer.V7FinalizationError, match="post-plan authorization"):
        finalizer.validate_gold_free_materialization(
            report_path=paths["report"],
            preregistration_path=paths["preregistration"],
            truncation_addendum_path=paths["addendum"],
            trajectory_semantics_addendum_path=paths["trajectory"],
            implementation_lock_path=paths["implementation"],
            plan_lock_path=paths["plan_lock"],
            enforce_canonical_trajectory_hash=False,
        )


def test_finalizer_rejects_tampered_actual_c_subanswer_model_identity(
    tmp_path: Path,
) -> None:
    paths = _full_synthetic_chain(tmp_path)
    details_path = paths["report"].parent / "execution_details.jsonl"
    details = finalizer._read_jsonl(details_path)
    details[0]["subanswer_telemetry"][0]["telemetry"]["model_artifact"][
        "strong_sft_adapter"
    ]["identity"]["inventory_sha256"] = "0" * 64
    _write_jsonl(details_path, details)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    report["outputs"]["execution_details"] = finalizer._file_lock(details_path)
    _write_json(paths["report"], report)
    with pytest.raises(finalizer.V7FinalizationError, match="identity differs"):
        finalizer.validate_gold_free_materialization(
            report_path=paths["report"],
            preregistration_path=paths["preregistration"],
            truncation_addendum_path=paths["addendum"],
            trajectory_semantics_addendum_path=paths["trajectory"],
            implementation_lock_path=paths["implementation"],
            plan_lock_path=paths["plan_lock"],
            enforce_canonical_trajectory_hash=False,
        )


def test_finalizer_rejects_recursive_trajectory_parent_tamper(tmp_path: Path) -> None:
    paths = _full_synthetic_chain(tmp_path)
    trajectory = json.loads(paths["trajectory"].read_text(encoding="utf-8"))
    trajectory["parents"]["parent_preregistration"]["sha256"] = "f" * 64
    _write_json(paths["trajectory"], trajectory)
    prereg_lock = finalizer._file_lock(paths["preregistration"])
    addendum_lock = finalizer._file_lock(paths["addendum"])
    with pytest.raises(finalizer.V7FinalizationError, match="parent"):
        finalizer._validate_trajectory_addendum(
            paths["trajectory"],
            preregistration_lock=prereg_lock,
            addendum_lock=addendum_lock,
            enforce_canonical_hash=False,
        )
