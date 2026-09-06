from __future__ import annotations

import hashlib

import pytest

from scripts.prepare.materialize_2wiki_official_raw_planner_inputs_v1 import (
    EXPECTED_CANDIDATE_FIELDS,
    _assert_answer_free,
    _balanced_preflight,
    _derive_runtime_rows,
)


def _row(index: int, question_type: str = "comparison") -> dict:
    question = f"Which entity is associated with item {index}?"
    return {
        "schema_version": "2wiki-proofkg-official-raw-question-only-v2",
        "dataset": "2wikimultihopqa",
        "qid": f"q{index}",
        "question": question,
        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "family_version": "answer-free-lexical-family-v1",
        "family_sha256": hashlib.sha256(f"family-{index}".encode()).hexdigest(),
        "question_type": question_type,
        "target_type": "relation_graph",
        "gold_access": False,
    }


def test_derives_only_runner_identity_fields() -> None:
    source = _row(1)
    assert set(source) == EXPECTED_CANDIDATE_FIELDS
    result = _derive_runtime_rows([source])[0]
    assert result["row_id"] == "OFFICIAL-RAW-V2-0001"
    assert result["question_key"] == "2wikimultihopqa::q1"
    assert result["question"] == source["question"]
    assert "answer" not in result
    assert "target" not in result


def test_rejects_extra_gold_field() -> None:
    row = _row(1)
    row["answer"] = "gold"
    with pytest.raises(ValueError, match="candidate field mismatch"):
        _derive_runtime_rows([row])


def test_recursive_prohibited_field_guard() -> None:
    with pytest.raises(ValueError, match="prohibited fields"):
        _assert_answer_free(
            {"safe": [{"supporting_facts": ["hidden"]}]}, location="row"
        )


def test_rejects_question_hash_mismatch() -> None:
    row = _row(1)
    row["question_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="question hash mismatch"):
        _derive_runtime_rows([row])


def test_preflight_balanced_by_question_type() -> None:
    question_types = (
        "bridge_comparison",
        "comparison",
        "compositional",
        "inference",
    )
    rows = []
    for type_index, question_type in enumerate(question_types):
        rows.extend(
            _derive_runtime_rows(
                [_row(type_index * 10 + offset, question_type) for offset in range(3)]
            )
        )
    selected = _balanced_preflight(rows, per_type=2)
    assert len(selected) == 8
    assert {qtype: sum(r["question_type"] == qtype for r in selected) for qtype in question_types} == {
        qtype: 2 for qtype in question_types
    }


def test_rejects_non_relation_graph() -> None:
    row = _row(1)
    row["target_type"] = "subquery_graph"
    with pytest.raises(ValueError, match="non relation_graph"):
        _derive_runtime_rows([row])
