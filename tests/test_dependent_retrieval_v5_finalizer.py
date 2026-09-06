"""CPU-only safety tests for the isolated v5 scorer-input finalizer."""

from copy import deepcopy
import hashlib
import json

import pytest

from scripts.prepare.finalize_dependent_retrieval_v5 import (
    ANSWER_UTILITY_GATES,
    EVALUATOR_DECISION_GATES,
    _assert_gold_free,
    _audit_materialization_safety,
    _enforce_report_safety,
    _json_sha256,
    _sha256_file,
    _validate_experiment_ids,
    _validate_preregistration,
    _validate_v4_controls,
    _validate_v4_gate_identity,
    _validate_v4_scorer_gold_locks,
)
from kgproweight.retrieval.dependent_merge_v5 import POLICY_VERSION
from kgproweight.retrieval.dependent_v5 import SELECTOR_VERSION
from scripts.prepare import freeze_dependent_retrieval_v5 as v5_freeze


def _passage(name):
    return {"id": name, "contents": f"{name}\nbody"}


def _population():
    arm_a, arm_b, details = [], [], []
    for dataset in ("hotpotqa", "musique"):
        for index in range(30):
            qid = f"q{index}"
            key = f"{dataset}::{qid}"
            common = {
                "row_id": f"v5::{key}",
                "question_key": key,
                "dataset": dataset,
                "qid": qid,
                "question": f"Question {dataset} {index}?",
                "question_sha256": f"sha-{dataset}-{index}",
                "split": "pilot",
                "gold_access": False,
                "kg_subgraph": [["h", "r", "t"]],
                "legacy_kg_sha256": "kg",
            }
            original = [_passage(f"{dataset}-{index}-o{rank}") for rank in range(1, 11)]
            changed = index % 2 == 0
            if changed:
                replacement = _passage(f"{dataset}-{index}-new")
                output = [*deepcopy(original[:9]), replacement]
                selected = [{
                    "document_key": f"id:{replacement['id']}",
                    "score": 0.9,
                    "hop_id": "hop_2",
                }]
                evicted = [{
                    "document_key": f"id:{original[9]['id']}",
                    "original_rank": 10,
                    "score": 0.1,
                    "replaced_by": f"id:{replacement['id']}",
                    "replacement_score": 0.9,
                }]
            else:
                output = deepcopy(original)
                selected, evicted = [], []
            arm_a.append({
                **common,
                "arm": "A_question_only",
                "retrieved_passages": original,
                "passages_sha256": _json_sha256(original),
            })
            arm_b.append({
                **common,
                "arm": "B_dependent",
                "retrieved_passages": output,
                "passages_sha256": _json_sha256(output),
                "fallback_to_a": not changed,
                "retrieval_trace": {
                    "plan_executable": True,
                    "has_dependent_step": True,
                    "dependent_query_count": 1,
                    "second_hop_query_count": 1,
                    "new_dependent_candidate_count": int(changed),
                    "fallback_reason": None if changed else "no_candidate_strictly_better",
                },
            })
            details.append({
                "dataset": dataset,
                "qid": qid,
                "gold_access": False,
                "execution_status": "executed",
                "plan_executable": True,
                "has_dependent_step": True,
                "second_hop_query_count": 1,
                "hops": [
                    {"hop_id": "hop_1", "dependencies": []},
                    {"hop_id": "hop_2", "dependencies": ["hop_1"]},
                ],
                "merge": {
                    "protected_originals": 8,
                    "total_budget": 10,
                    "fallback_exact": not changed,
                    "selected_new": selected,
                    "evicted_originals": evicted,
                },
            })
    return arm_a, arm_b, details


def _report():
    return {
        "schema_version": "plan-once-dependent-retrieval-v5-report-1",
        "status": "COMPLETE_INPUTS_NOT_ANSWER_EVALUATED",
        "gold_access": False,
        "development_only": True,
        "canonical_pipeline_modified": False,
        "runner_version": "synthetic-v5-runner",
        "selector_version": SELECTOR_VERSION,
        "merge_policy_version": POLICY_VERSION,
        "settings": {
            "protected_originals": 8,
            "total_passages": 10,
            "root_hop_injection": False,
        },
        "by_dataset": {
            dataset: {
                "n": 30,
                "runtime_errors": 0,
                "fallback_execution_error": 0,
                "unauthorized_original_displacements": 0,
                "root_passages_injected": 0,
                "all_top10": True,
                "prefix8_exact": True,
                "fallback_exact": True,
                "plan_executable_rate": 1.0,
                "dependent_hop_query_nonempty_rate": 1.0,
                "retained_new_dependent_document_question_rate": 0.50,
            }
            for dataset in ("hotpotqa", "musique")
        },
        "safety_summary": {
            "all_top10": True,
            "prefix8_exact": True,
            "unauthorized_original_displacements": 0,
            "root_passages_injected": 0,
            "fallback_exact": True,
            "runtime_errors": 0,
        },
    }


def test_valid_same60_recomputes_top10_prefix8_and_strict_displacement():
    arm_a, arm_b, details = _population()
    result = _audit_materialization_safety(arm_a, arm_b, details)
    assert result["n"] == 60
    assert result["all_top10"] is True
    assert result["prefix8_exact"] is True
    assert result["unauthorized_original_displacements"] == 0
    assert result["root_passages_injected"] == 0
    assert result["changed_questions"] == 30
    assert result["selected_new_documents"] == 30
    _enforce_report_safety(_report(), result)


def test_premerge_bridge_abstention_is_allowed_only_as_exact_fallback():
    arm_a, arm_b, details = _population()
    # Row 1 is already an exact fallback.  Model the selector abstaining before
    # the merge policy is called, as the formal runner does.
    details[1]["execution_status"] = "fallback_bridge_abstain"
    details[1]["merge"] = None
    details[1]["safety"] = {
        "output_count": 10,
        "prefix8_exact": True,
        "unauthorized_original_displacements": 0,
        "root_passages_injected": 0,
        "fallback_exact": True,
    }
    _audit_materialization_safety(arm_a, arm_b, details)
    arm_b[1]["retrieved_passages"][-1] = _passage("changed-without-merge")
    arm_b[1]["passages_sha256"] = _json_sha256(arm_b[1]["retrieved_passages"])
    arm_b[1]["fallback_to_a"] = False
    with pytest.raises(ValueError, match="missing v5 merge telemetry"):
        _audit_materialization_safety(arm_a, arm_b, details)


def test_prefix8_change_fails_closed():
    arm_a, arm_b, details = _population()
    arm_b[0]["retrieved_passages"][0] = _passage("unauthorized")
    arm_b[0]["passages_sha256"] = _json_sha256(arm_b[0]["retrieved_passages"])
    with pytest.raises(ValueError, match="prefix8 exact"):
        _audit_materialization_safety(arm_a, arm_b, details)


def test_non_improving_or_root_hop_displacement_fails_closed():
    arm_a, arm_b, details = _population()
    details[0]["merge"]["evicted_originals"][0]["replacement_score"] = 0.05
    with pytest.raises(ValueError, match="non-improving"):
        _audit_materialization_safety(arm_a, arm_b, details)

    arm_a, arm_b, details = _population()
    details[0]["merge"]["selected_new"][0]["hop_id"] = "hop_1"
    with pytest.raises(ValueError, match="root or unknown-hop"):
        _audit_materialization_safety(arm_a, arm_b, details)


def test_report_must_attest_every_v5_materialization_safety_gate():
    arm_a, arm_b, details = _population()
    computed = _audit_materialization_safety(arm_a, arm_b, details)
    report = _report()
    report["safety_summary"]["unauthorized_original_displacements"] = 1
    with pytest.raises(ValueError, match="unauthorized_original_displacements"):
        _enforce_report_safety(report, computed)


def test_report_must_lock_fixed_budget_prefix_and_no_root_injection():
    arm_a, arm_b, details = _population()
    computed = _audit_materialization_safety(arm_a, arm_b, details)
    for field, invalid in (
        ("protected_originals", 7),
        ("total_passages", 11),
        ("root_hop_injection", True),
    ):
        report = _report()
        report["settings"][field] = invalid
        with pytest.raises(ValueError, match="materialization setting differs"):
            _enforce_report_safety(report, computed)


def test_report_must_pass_gold_free_mechanism_gates_before_gold_join():
    arm_a, arm_b, details = _population()
    computed = _audit_materialization_safety(arm_a, arm_b, details)
    report = _report()
    report["by_dataset"]["musique"][
        "retained_new_dependent_document_question_rate"
    ] = 0.49
    computed["by_dataset"]["musique"][
        "retained_new_dependent_document_question_rate"
    ] = 0.49
    with pytest.raises(ValueError, match="Gold-free mechanism gate failed"):
        _enforce_report_safety(report, computed)


def test_upstream_gold_or_support_fields_are_rejected_recursively():
    _assert_gold_free({"retrieval_trace": {"score": 1.0}}, where="row")
    with pytest.raises(ValueError, match="forbidden fields"):
        _assert_gold_free({"metadata": {"gold_answers": ["hidden"]}}, where="row")
    with pytest.raises(ValueError, match="forbidden fields"):
        _assert_gold_free({"supporting_facts": []}, where="row")


def test_preregistration_scope_and_three_experiment_ids_are_strict(tmp_path):
    prereg = tmp_path / "protocol.json"
    counts = {"corpus": 21015324, "dense": 21015324, "bm25": 21015324}
    asset_paths = {
        "corpus": str(tmp_path / "corpus"),
        "dense": str(tmp_path / "dense"),
        "bm25": str(tmp_path / "bm25"),
    }
    protocol = {
        "schema_version": v5_freeze.SCHEMA_VERSION,
        "status": v5_freeze.STATUS,
        "scope": v5_freeze.SCOPE,
        "experiment_ids": dict(v5_freeze.EXPERIMENT_IDS),
        "inputs": {},
        "code": {},
        "models": {},
        "settings": {},
        "retrieval_assets": {
            "expected_documents": 21015324,
            "counts": counts,
            "corpus": {"path": asset_paths["corpus"]},
            "dense_index": {"path": asset_paths["dense"]},
            "bm25_index": {"path": asset_paths["bm25"]},
        },
    }
    prereg.write_text(json.dumps(protocol), encoding="utf-8")
    prereg_lock = {
        "path": str(prereg.resolve()),
        "size_bytes": prereg.stat().st_size,
        "sha256": hashlib.sha256(prereg.read_bytes()).hexdigest(),
    }
    report = {
        "experiment_id": v5_freeze.EXPERIMENT_IDS["materialization"],
        "preregistration": prereg_lock,
        "runtime_locks": {
            "preregistration": prereg_lock,
            "inputs": {},
            "code": {},
            "models": {},
            "settings": {},
        },
        "settings": {},
        "retrieval_assets": {
            "expected_docs": 21015324,
            "counts": counts,
            "paths": asset_paths,
        },
    }
    lock = _validate_preregistration(prereg, report)
    assert lock["path"] == str(prereg)
    _validate_experiment_ids(*v5_freeze.EXPERIMENT_IDS.values())
    with pytest.raises(ValueError, match="three non-empty distinct"):
        _validate_experiment_ids("V5", "V5", "V5-EVAL")


def test_v4_utility_gate_identity_cannot_be_relaxed(tmp_path):
    protocol = tmp_path / "v4.json"
    protocol.write_text(json.dumps({"decision_gates": EVALUATOR_DECISION_GATES}), encoding="utf-8")
    _, lock = _validate_v4_gate_identity(protocol)
    assert lock["path"] == str(protocol)
    weakened = dict(EVALUATOR_DECISION_GATES, pooled_net_correct_gain_min=2)
    protocol.write_text(json.dumps({"decision_gates": weakened}), encoding="utf-8")
    with pytest.raises(ValueError, match="differ from the required unchanged"):
        _validate_v4_gate_identity(protocol)
    assert ANSWER_UTILITY_GATES == {
        "pooled_net_correct_gain_min": 3,
        "pooled_delta_f1_gt": 0.0,
        "max_net_correct_loss_per_dataset": 1,
        "parse_count_delta_min": 0,
    }


def test_v5_population_arm_model_and_generation_are_locked_to_v4():
    adapter = {"path": "/adapter", "inventory_sha256": "a"}
    base = {"path": "/base", "inventory_sha256": "b"}
    generation = {
        "seed": 42, "decode": "greedy", "do_sample": False,
        "temperature": None, "top_p": None,
        "max_new_tokens": 512, "top_k_passages": 10,
    }
    v4 = {
        "qid_order_sha256": "qids",
        "question_key_order_sha256": "keys",
        "inputs": {"retrieval_arm_a_no_gold": {"sha256": "arm-a"}},
        "models": {"strong_sft": adapter},
        "base_model": base,
        "generation": generation,
    }
    _validate_v4_controls(
        v4,
        arm_a_sha256="arm-a",
        qid_order_sha256="qids",
        question_key_order_sha256="keys",
        adapter_identity=adapter,
        base_identity=base,
        generation=generation,
    )
    with pytest.raises(ValueError, match="Arm A bytes differ"):
        _validate_v4_controls(
            v4,
            arm_a_sha256="different",
            qid_order_sha256="qids",
            question_key_order_sha256="keys",
            adapter_identity=adapter,
            base_identity=base,
            generation=generation,
        )


def test_scorer_gold_paths_and_hashes_are_locked_to_v4_before_join(tmp_path):
    paths = []
    locks = {}
    for dataset in ("hotpotqa", "musique"):
        root = tmp_path / dataset
        root.mkdir()
        path = root / "dev.jsonl"
        path.write_text('{"golden_answers":["hidden"]}\n', encoding="utf-8")
        paths.append(path)
        locks[dataset] = {"path": str(path.resolve()), "sha256": _sha256_file(path)}
    protocol = {"inputs": {"scorer_gold": locks}}
    assert _validate_v4_scorer_gold_locks(protocol, paths) == locks

    alternate = tmp_path / "alternate.jsonl"
    alternate.write_bytes(paths[0].read_bytes())
    with pytest.raises(ValueError, match="path differs from v4"):
        _validate_v4_scorer_gold_locks(protocol, [alternate, paths[1]])

    paths[0].write_text('{"golden_answers":["changed"]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 differs from v4"):
        _validate_v4_scorer_gold_locks(protocol, paths)
