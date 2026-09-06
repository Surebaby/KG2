from __future__ import annotations

import json
from pathlib import Path

import pytest

from kgproweight.utils.logging import artifact_identity
from scripts.prepare import freeze_dependent_retrieval_v6 as freeze


def _file(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _json(path: Path, value: object) -> Path:
    return _file(path, json.dumps(value, ensure_ascii=False) + "\n")


def synthetic_build_args(tmp_path: Path) -> dict[str, object]:
    inputs = {name: _file(tmp_path / "inputs" / name, name) for name in freeze.DEFAULT_INPUTS}
    code = {name: _file(tmp_path / "code" / name, name) for name in freeze.DEFAULT_CODE}
    corpus = _file(tmp_path / "wiki" / "corpus", "corpus")
    dense = _file(tmp_path / "wiki" / "dense", "dense")
    bm25 = tmp_path / "wiki" / "bm25"
    _file(bm25 / "index", "bm25")
    e5, ce, sft, base = tmp_path / "e5", tmp_path / "ce", tmp_path / "sft", tmp_path / "base"
    for path, marker in ((e5, "e5"), (ce, "ce"), (sft, "sft"), (base, "base")):
        _file(path / "config.json", marker)

    design_materialization = {
        name: value for name, value in freeze.MATERIALIZATION_GATES.items()
        if name != "duplicate_dependent_queries"
    }
    design = _json(tmp_path / "design.json", {
        "status": "RULES_FROZEN_BEFORE_IMPLEMENTATION_AND_RETRIEVAL",
        "scope": "ADAPTIVE_DEVELOPMENT_COMBINATION_EXPERIMENT",
        "population": {"datasets": dict(freeze.DATASETS), "confirmation_opened": False},
        "decision_gates": {
            "materialization": design_materialization,
            "mechanism_unchanged_from_v5": dict(freeze.MECHANISM_GATES),
            "answer_utility_unchanged_from_v4_v5": dict(freeze.ANSWER_UTILITY_GATES),
        },
        "required_telemetry": ["queries"],
        "anti_p_hacking": ["one fixed run"],
    })
    v4_inputs = _json(tmp_path / "v4_inputs.json", {
        "inputs": {
            name: {"path": str(path), "sha256": freeze.sha256_file(path)}
            for name, path in inputs.items()
        }
    })
    v4_eval = _json(tmp_path / "v4_eval.json", {
        "status": "FROZEN_AFTER_RETRIEVAL_BEFORE_ANSWER_GENERATION",
        "n": 60,
        "decision_gates": {
            "pooled_net_correct_gain_min": 3,
            "max_net_correct_loss_per_dataset": 1,
            "parse_count_delta_min": 0,
            "plan_executable_rate_min_each_dataset": 0.8,
            "second_hop_query_nonempty_rate_min_each_dataset": 0.8,
            "new_dependent_candidate_question_rate_min_each_dataset": 0.5,
        },
        "qid_order_sha256": "qid-order",
        "question_key_order_sha256": "key-order",
        "models": {"strong_sft": artifact_identity(sft)},
        "base_model": artifact_identity(base),
    })
    preflight = _json(tmp_path / "preflight.json", {
        "status": "PASS", "expected_docs": freeze.EXPECTED_DOCUMENTS,
        "counts": {name: freeze.EXPECTED_DOCUMENTS for name in ("corpus", "dense", "bm25")},
        "paths": {"corpus": str(corpus.resolve()), "dense": str(dense.resolve()), "bm25": str(bm25.resolve())},
    })
    return {
        "design_path": design, "v4_eval_path": v4_eval, "v4_input_path": v4_inputs,
        "wiki18_preflight_path": preflight, "input_paths": inputs, "code_paths": code,
        "corpus_path": corpus, "dense_path": dense, "bm25_path": bm25,
        "retrieval_encoder_path": e5, "cross_encoder_path": ce,
        "strong_sft_path": sft, "base_model_path": base,
        "expected_design_sha256": freeze.sha256_file(design),
        "expected_v4_eval_sha256": freeze.sha256_file(v4_eval),
        "expected_v4_input_sha256": freeze.sha256_file(v4_inputs),
    }


def test_v6_freeze_locks_every_asset_and_exact_settings(tmp_path: Path) -> None:
    protocol = freeze.build_protocol(**synthetic_build_args(tmp_path))
    assert protocol["status"] == freeze.STATUS
    assert protocol["scope"] == freeze.SCOPE
    assert set(protocol["inputs"]) == set(freeze.DEFAULT_INPUTS)
    assert set(protocol["code"]) == set(freeze.DEFAULT_CODE) | {"preregistration_freezer"}
    assert protocol["settings"]["datasets_in_order"] == ["hotpotqa", "musique"]
    assert protocol["settings"]["question_anchor_template"] == "{original_question}\n{subquery}"
    assert protocol["settings"]["max_query_variants"] == 2
    assert protocol["settings"]["retrieval_query_max_length"] == 128
    assert protocol["settings"]["candidates_per_query_variant"] == 2
    assert protocol["settings"]["protected_originals"] == 8
    assert protocol["settings"]["total_passages"] == 10
    assert protocol["settings"]["ce_max_chars"] == 1200
    assert protocol["retrieval_assets"]["expected_documents"] == 21_015_324
    assert protocol["retrieval_asset_content_locks"]["bm25_index"]["files"][0]["sha256"]
    assert protocol["model_content_locks"]["cross_encoder"]["tree_sha256"]
    assert protocol["model_content_locks"]["retrieval_encoder"]["tree_sha256"]
    assert protocol["decision_gates"]["materialization"]["duplicate_output_documents"] == 0
    assert protocol["decision_gates"]["answer_utility"] == freeze.ANSWER_UTILITY_GATES


def test_v6_freeze_rejects_same60_input_drift(tmp_path: Path) -> None:
    args = synthetic_build_args(tmp_path)
    args["input_paths"]["hotpot_plans"].write_text("drift", encoding="utf-8")
    with pytest.raises(ValueError, match="same60 input drifted"):
        freeze.build_protocol(**args)


def test_v6_freeze_rejects_design_gate_drift(tmp_path: Path) -> None:
    args = synthetic_build_args(tmp_path)
    path = args["design_path"]
    value = json.loads(path.read_text(encoding="utf-8"))
    value["decision_gates"]["mechanism_unchanged_from_v5"]["plan_executable_rate_min_each_dataset"] = 0.7
    _json(path, value)
    args["expected_design_sha256"] = freeze.sha256_file(path)
    with pytest.raises(ValueError, match="mechanism gates drifted"):
        freeze.build_protocol(**args)
