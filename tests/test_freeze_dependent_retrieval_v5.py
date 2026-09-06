from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kgproweight.utils.logging import artifact_identity
from scripts.prepare import freeze_dependent_retrieval_v5 as freeze


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _file(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> dict[str, object]:
    inputs = {
        name: _file(tmp_path / "inputs" / f"{name}.jsonl", f'{{"name":"{name}"}}\n')
        for name in freeze.DEFAULT_INPUTS
    }
    code = {
        name: _file(tmp_path / "code" / f"{name}.py", f"# {name}\n")
        for name in freeze.DEFAULT_CODE
    }
    corpus = _file(tmp_path / "wiki18" / "corpus.jsonl", "{}\n")
    dense = _file(tmp_path / "wiki18" / "dense.dat", "dense")
    bm25 = tmp_path / "wiki18" / "bm25"
    _file(bm25 / "params.index.json", "{}\n")

    ce = tmp_path / "models" / "ce"
    adapter = tmp_path / "models" / "adapter"
    base = tmp_path / "models" / "base"
    for directory, marker in ((ce, "ce"), (adapter, "adapter"), (base, "base")):
        _file(directory / "config.json", json.dumps({"marker": marker}) + "\n")

    design = _write_json(tmp_path / "design.json", {
        "status": "RULES_FROZEN_BEFORE_IMPLEMENTATION_AND_RETRIEVAL",
        "scope": "ADAPTIVE_DEVELOPMENT_COMBINATION_EXPERIMENT",
        "population": {
            "datasets": dict(freeze.DATASETS),
            "confirmation_opened": False,
        },
        "decision_gates": {
            "answer_utility_unchanged_from_v4": dict(freeze.ANSWER_UTILITY_GATES),
        },
        "anti_p_hacking": ["one fixed parameterization"],
    })
    v4_eval = _write_json(tmp_path / "v4_eval.json", {
        "status": "FROZEN_AFTER_RETRIEVAL_BEFORE_ANSWER_GENERATION",
        "n": 60,
        "qid_order_sha256": "qid-order",
        "question_key_order_sha256": "key-order",
        "decision_gates": {
            **freeze.V4_EVAL_PROTOCOL_GATE_SUBSET,
            **freeze.MECHANISM_GATES,
        },
        "models": {"strong_sft": artifact_identity(adapter)},
        "base_model": artifact_identity(base),
    })
    v4_input = _write_json(tmp_path / "v4_input.json", {
        "inputs": {
            name: {"path": str(path), "sha256": _sha(path)}
            for name, path in inputs.items()
        },
    })
    wiki18_preflight = _write_json(tmp_path / "wiki18_preflight.json", {
        "status": "PASS",
        "expected_docs": freeze.EXPECTED_DOCUMENTS,
        "counts": {
            "corpus": freeze.EXPECTED_DOCUMENTS,
            "dense": freeze.EXPECTED_DOCUMENTS,
            "bm25": freeze.EXPECTED_DOCUMENTS,
        },
        "paths": {
            "corpus": str(corpus.resolve()),
            "dense": str(dense.resolve()),
            "bm25": str(bm25.resolve()),
        },
    })
    return {
        "design_protocol_path": design,
        "v4_eval_protocol_path": v4_eval,
        "v4_input_protocol_path": v4_input,
        "wiki18_preflight_path": wiki18_preflight,
        "input_paths": inputs,
        "code_paths": code,
        "corpus_path": corpus,
        "dense_index_path": dense,
        "bm25_index_path": bm25,
        "cross_encoder_path": ce,
        "adapter_path": adapter,
        "base_model_path": base,
        "expected_design_sha256": _sha(design),
        "expected_v4_eval_sha256": _sha(v4_eval),
        "expected_v4_input_protocol_sha256": _sha(v4_input),
    }


def test_build_protocol_locks_same60_code_assets_and_settings(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    protocol = freeze.build_protocol(**args)

    assert protocol["status"] == freeze.STATUS
    assert protocol["scope"] == freeze.SCOPE
    assert protocol["population"]["datasets"] == {"hotpotqa": 30, "musique": 30}
    assert protocol["population"]["n"] == 60
    assert set(protocol["inputs"]) == set(freeze.DEFAULT_INPUTS)
    assert all(protocol["inputs"][name]["sha256"] == _sha(path)
               for name, path in args["input_paths"].items())
    assert set(protocol["code"]) == set(freeze.DEFAULT_CODE) | {"preregistration_freezer"}
    assert protocol["settings"]["rrf_candidate_k"] == 100
    assert protocol["settings"]["step_rerank_topk"] == 10
    assert protocol["settings"]["bridge_max_docs"] == 10
    assert protocol["settings"]["bridge_max_candidates"] == 2
    assert protocol["settings"]["bridge_max_body_chars"] == 1200
    assert protocol["settings"]["protected_originals"] == 8
    assert protocol["settings"]["candidates_per_dependent_hop"] == 2
    assert protocol["settings"]["total_passages"] == 10
    assert protocol["settings"]["ce_max_chars"] == 1200
    assert protocol["retrieval_assets"]["expected_documents"] == 21_015_324
    assert protocol["decision_gates"]["answer_utility_unchanged_from_v4"] == (
        freeze.ANSWER_UTILITY_GATES
    )
    assert len(set(protocol["experiment_ids"].values())) == 3


def test_build_protocol_rejects_same60_input_drift(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    input_paths = args["input_paths"]
    input_paths["musique_plans"].write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="same-60 input musique_plans drifted"):
        freeze.build_protocol(**args)


def test_build_protocol_rejects_v4_gate_drift(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    path = args["v4_eval_protocol_path"]
    value = json.loads(path.read_text(encoding="utf-8"))
    value["decision_gates"]["pooled_net_correct_gain_min"] = 2
    _write_json(path, value)
    args["expected_v4_eval_sha256"] = _sha(path)

    with pytest.raises(ValueError, match="v4 frozen evaluation gate drifted"):
        freeze.build_protocol(**args)


def test_build_protocol_rejects_design_hash_drift(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    path = args["design_protocol_path"]
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="v5 design freeze SHA256 drift"):
        freeze.build_protocol(**args)

