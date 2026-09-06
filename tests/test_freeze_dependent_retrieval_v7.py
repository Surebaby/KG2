"""CPU-only tests for the v7 question-only cohort/rules freezer."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from kgproweight.kg.question_kg import question_sha256
from kgproweight.utils.logging import artifact_identity
from scripts.prepare import freeze_dependent_retrieval_v7 as freeze
from scripts.prepare import freeze_dependent_retrieval_v7_truncation_addendum as truncation


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _json(path: Path, value: object) -> Path:
    return _write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _jsonl(path: Path, rows: list[dict]) -> Path:
    return _write(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def _source_rows() -> list[dict]:
    rows = []
    for dataset in ("2wikimultihopqa", "hotpotqa", "musique"):
        for index in range(100):
            question = f"Question {dataset} {index}?"
            rows.append(
                {
                    "schema_version": "saeg-eval-cohort-v1",
                    "question_key": f"{dataset}::dev_{index}",
                    "dataset": dataset,
                    "qid": f"dev_{index}",
                    "question": question,
                    "question_sha256": question_sha256(question),
                    "family_sha256": freeze.sha256_text(f"family\0{dataset}\0{index}"),
                    "role": "confirmation",
                    "gold_access": False,
                    "passages_sha256": freeze.sha256_text(f"passages\0{dataset}\0{index}"),
                    "passage_graph_sha256": freeze.sha256_text(f"graph\0{dataset}\0{index}"),
                    "passage_graph_nonempty": bool(index % 2),
                }
            )
    return rows


def _design() -> dict:
    return {
        "status": "RULES_AND_SELECTION_ALGORITHM_FROZEN_BEFORE_V7_GPU_OR_RETRIEVAL",
        "scope": "DEVELOPMENT_ONLY_FEASIBILITY_COMPARISON_ON_GLOBALLY_CONSUMED_ROWS",
        "population": {
            "datasets_in_order": list(freeze.DATASETS),
            "selected_per_dataset": freeze.SELECTED_PER_DATASET,
            "reclassification": {
                "v7_role": "development_consumed",
                "globally_fresh": False,
                "independent_confirmation": False,
            },
        },
        "arms": {
            "A_canonical_one_shot": {},
            "B_entity_hint_top1": {},
            "C_verified_subanswer": {},
        },
        "planner": {},
        "paired_execution": {
            "budget_invariant": (
                "For every question/depth/hop B equals C; each is either 0 or 1."
            )
        },
        "retrieval_and_merge": {},
        "generation": {
            "subanswer": {"max_new_tokens": 96},
            "final_answer": {"max_new_tokens": 512},
        },
        "fallback": {},
        "gold_policy": {},
        "decision_gates": {
            "materialization": dict(freeze.MATERIALIZATION_GATES),
            "gold_free_mechanism": dict(freeze.MECHANISM_GATES),
            "development_utility": dict(freeze.UTILITY_GATES),
        },
        "anti_p_hacking": [],
        "required_telemetry": [],
        "scientific_boundary": "development only",
    }


def _synthetic_args(tmp_path: Path) -> dict[str, object]:
    source = _jsonl(tmp_path / "source.question_only.jsonl", _source_rows())
    contexts = _write(tmp_path / "contexts.question_only.jsonl", "answer-free contexts\n")
    design = _json(tmp_path / "design.json", _design())
    design_manifest = _json(
        tmp_path / "design_manifest.json", {"status": _design()["status"]}
    )
    exposure = _json(
        tmp_path / "exposure.json",
        {
            "status": "ADVISORY_IDENTITY_SCAN_NOT_A_FRESHNESS_PROOF",
            "matching": {
                "known_scored_identity_overlap": {"hotpotqa": 0, "musique": 0}
            },
        },
    )
    parent = _json(
        tmp_path / "parent.json",
        {
            "status": "FROZEN_ANSWER_FREE_BEFORE_SAEG_DEVELOPMENT",
            "integrity": {
                "confirmation_opened": False,
                "gold_in_protocol_or_cohorts": False,
            },
            "outputs": {"confirmation": {"sha256": freeze.sha256_file(source)}},
            "inputs": {"fresh_contexts": {"sha256": freeze.sha256_file(contexts)}},
        },
    )

    model_roots = {}
    models = {}
    content_locks = {}
    for name in ("retrieval_encoder", "cross_encoder", "strong_sft", "base_model"):
        root = tmp_path / "models" / name
        _write(root / "config.json", name)
        model_roots[name] = root
        models[name] = artifact_identity(root)
        content_locks[name] = {"tree_sha256": freeze.sha256_text(name)}
    v6 = _json(
        tmp_path / "v6.json",
        {
            "status": "FROZEN_BEFORE_RETRIEVAL",
            "retrieval_assets": {
                "expected_documents": freeze.EXPECTED_WIKI18_DOCUMENTS
            },
            "retrieval_asset_content_locks": {},
            "settings": {
                "network_access": False,
                "rrf_candidate_k": 100,
                "retrieval_query_max_length": 128,
                "step_rerank_topk": 10,
                "protected_originals": 8,
                "total_passages": 10,
                "ce_max_chars": 1200,
                "root_hop_injection": False,
            },
            "models": models,
            "model_content_locks": content_locks,
        },
    )
    planner_adapter = tmp_path / "planner"
    _write(planner_adapter / "adapter_config.json", "planner")
    planner_config = _write(tmp_path / "planner.yaml", "model: planner\n")
    code_paths = {
        name: _write(tmp_path / "code" / f"{name}.py", name)
        for name in (
            "planner_generator",
            "subanswer_module",
            "dependent_helpers",
            "query_renderer",
            "merge_helper",
            "prompt_factory",
            "answer_parser",
        )
    }
    return {
        "source_path": source,
        "parent_protocol_path": parent,
        "contexts_path": contexts,
        "design_path": design,
        "design_manifest_path": design_manifest,
        "exposure_audit_path": exposure,
        "v6_protocol_path": v6,
        "planner_adapter_path": planner_adapter,
        "planner_config_path": planner_config,
        "code_paths": code_paths,
        "output_dir": tmp_path / "out",
        "expected_source_sha256": freeze.sha256_file(source),
        "expected_parent_protocol_sha256": freeze.sha256_file(parent),
        "expected_design_sha256": freeze.sha256_file(design),
        "expected_design_manifest_sha256": freeze.sha256_file(design_manifest),
        "expected_exposure_audit_sha256": freeze.sha256_file(exposure),
        "expected_v6_protocol_sha256": freeze.sha256_file(v6),
        "expected_planner_config_sha256": freeze.sha256_file(planner_config),
        "verify_local_artifact_identity": True,
    }


def test_selection_is_order_invariant_balanced_and_reclassified() -> None:
    rows = _source_rows()
    freeze.validate_source_rows(rows)
    forward = freeze.select_development_rows(rows)
    reverse = freeze.select_development_rows(list(reversed(rows)))

    assert [row["question_key"] for row in forward] == [
        row["question_key"] for row in reverse
    ]
    assert len(forward) == 40
    assert {row["dataset"] for row in forward} == {"hotpotqa", "musique"}
    assert all(row["role"] == "development_consumed" for row in forward)
    assert all(row["source_role"] == "confirmation" for row in forward)
    assert all(row["globally_fresh"] is False for row in forward)
    assert all(row["independent_confirmation"] is False for row in forward)
    assert Counter(row["dataset"] for row in forward) == {
        "hotpotqa": 20,
        "musique": 20,
    }


def test_bundle_locks_three_arms_budget_gates_and_unmaterialized_remainder(
    tmp_path: Path,
) -> None:
    args = _synthetic_args(tmp_path)
    bundle = freeze.build_freeze_bundle(**args)
    protocol = bundle["protocol"]

    assert protocol["status"] == freeze.STATUS
    assert protocol["execution_authorization"].startswith("BLOCKED")
    assert protocol["population"]["n"] == 40
    assert protocol["population"]["globally_fresh"] is False
    assert set(protocol["arms"]) == {
        "A_canonical_one_shot",
        "B_entity_hint_top1",
        "C_verified_subanswer",
    }
    assert protocol["decision_gates"]["materialization"] == freeze.MATERIALIZATION_GATES
    assert protocol["decision_gates"]["gold_free_mechanism"] == freeze.MECHANISM_GATES
    assert protocol["decision_gates"]["development_utility"] == freeze.UTILITY_GATES
    assert protocol["generation"]["subanswer"]["max_new_tokens"] == 96
    assert protocol["generation"]["final_answer"]["max_new_tokens"] == 512
    assert protocol["gold_access"] is False
    assert protocol["gpu_calls"] == protocol["retrieval_calls"] == 0
    assert protocol["unselected"]["counts"] == {
        "2wikimultihopqa": 100,
        "hotpotqa": 80,
        "musique": 80,
    }
    assert protocol["unselected"]["unselected_rows_materialized_here"] is False
    assert "question" not in protocol["unselected"]
    assert "passages" not in protocol["unselected"]
    assert all(
        row["target_type"] == freeze.TARGET_TYPES[row["dataset"]]
        for row in bundle["planners"]
    )


def test_recursive_gold_field_in_question_only_source_is_rejected() -> None:
    rows = _source_rows()
    rows[0]["metadata"] = {"supporting_facts": ["hidden"]}
    with pytest.raises(ValueError, match="forbidden Gold/answer fields"):
        freeze.validate_source_rows(rows)


def test_source_role_or_gold_access_drift_is_rejected() -> None:
    rows = _source_rows()
    rows[101]["role"] = "development"
    with pytest.raises(ValueError, match="not answer-free confirmation"):
        freeze.validate_source_rows(rows)


def test_design_budget_or_gate_drift_is_rejected(tmp_path: Path) -> None:
    args = _synthetic_args(tmp_path)
    design_path = args["design_path"]
    design = json.loads(design_path.read_text(encoding="utf-8"))
    design["generation"]["subanswer"]["max_new_tokens"] = 97
    _json(design_path, design)
    args["expected_design_sha256"] = freeze.sha256_file(design_path)
    with pytest.raises(ValueError, match="generation budget drift"):
        freeze.build_freeze_bundle(**args)


def test_writer_is_append_only_and_does_not_modify_parent(tmp_path: Path) -> None:
    args = _synthetic_args(tmp_path)
    source_before = freeze.sha256_file(args["source_path"])
    bundle = freeze.build_freeze_bundle(**args)
    result = freeze.write_freeze_bundle(bundle, args["output_dir"])

    assert result["protocol"]["sha256"]
    assert result["manifest"]["sha256"]
    assert freeze.sha256_file(args["source_path"]) == source_before
    assert sum(1 for _ in (args["output_dir"] / "development.question_only.jsonl").open()) == 40
    assert sum(
        1
        for _ in (args["output_dir"] / "reclassification_ledger.question_only.jsonl").open()
    ) == 40
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze.write_freeze_bundle(bundle, args["output_dir"])


def _synthetic_truncation_args(tmp_path: Path) -> dict[str, object]:
    parent_protocol = _json(
        tmp_path / "parent_protocol.json",
        {
            "status": freeze.STATUS,
            "execution_authorization": "BLOCKED_UNTIL_SEPARATE_IMPLEMENTATION_HASH_LOCK",
            "gold_access": False,
            "gpu_calls": 0,
            "retrieval_calls": 0,
        },
    )
    artifacts = {}
    for name in (
        "development",
        "planner",
        "reclassification_ledger",
        "unselected_commitment",
    ):
        path = _write(tmp_path / f"{name}.json", name)
        artifacts[name] = freeze.file_lock(path)
    artifacts["protocol"] = freeze.file_lock(parent_protocol)
    parent_manifest = _json(
        tmp_path / "parent_manifest.json",
        {"status": freeze.STATUS, "artifacts": artifacts},
    )
    addendum_value = {
        "status": truncation.STATUS,
        "effective_override": {
            "maximum_passages": 10,
            "maximum_unicode_characters_per_passage": 1200,
            "same_projection_for_prompt_and_verifier": True,
            "projected_fields": ["doc_id", "title", "text"],
            "truncation": "Python Unicode slicing text[:1200]",
            "projection_hash": "canonical JSON ensure_ascii=false sort_keys=true",
        },
        "telemetry_additions": ["projection hash"],
        "unchanged": ["cohort"],
        "gold_access": False,
        "scientific_boundary": "pre-execution only",
    }
    addendum = _json(tmp_path / "addendum.json", addendum_value)
    addendum_manifest = _json(
        tmp_path / "addendum_manifest.json",
        {
            "status": truncation.STATUS,
            "addendum": {"sha256": freeze.sha256_file(addendum)},
        },
    )
    return {
        "parent_protocol_path": parent_protocol,
        "parent_manifest_path": parent_manifest,
        "addendum_path": addendum,
        "addendum_manifest_path": addendum_manifest,
        "expected_parent_protocol_sha256": freeze.sha256_file(parent_protocol),
        "expected_parent_manifest_sha256": freeze.sha256_file(parent_manifest),
        "expected_addendum_sha256": freeze.sha256_file(addendum),
        "expected_addendum_manifest_sha256": freeze.sha256_file(addendum_manifest),
    }


def test_truncation_addendum_binds_same_prompt_and_verifier_projection(
    tmp_path: Path,
) -> None:
    protocol = truncation.build_protocol(**_synthetic_truncation_args(tmp_path))
    invariants = protocol["effective_invariants"]
    assert protocol["status"] == truncation.STATUS
    assert protocol["execution_authorization"].startswith("BLOCKED")
    assert invariants["producer_passages_max"] == 10
    assert invariants["producer_text_unicode_chars_max_each"] == 1200
    assert invariants["python_slice"] == "text[:1200]"
    assert invariants["reader_and_verifier_projection_hash_equal"] is True
    assert invariants["answer_in_unseen_suffix_never_verified"] is True
    assert protocol["gold_access"] is False


def test_truncation_addendum_rejects_character_limit_drift(tmp_path: Path) -> None:
    args = _synthetic_truncation_args(tmp_path)
    path = args["addendum_path"]
    value = json.loads(path.read_text(encoding="utf-8"))
    value["effective_override"]["maximum_unicode_characters_per_passage"] = 1201
    _json(path, value)
    args["expected_addendum_sha256"] = freeze.sha256_file(path)
    manifest_path = args["addendum_manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["addendum"]["sha256"] = args["expected_addendum_sha256"]
    _json(manifest_path, manifest)
    args["expected_addendum_manifest_sha256"] = freeze.sha256_file(manifest_path)
    with pytest.raises(ValueError, match="character"):
        truncation.build_protocol(**args)
