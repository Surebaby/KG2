from __future__ import annotations

import copy

import pytest

from kgproweight.kg.question_kg import question_sha256
from scripts.prepare.merge_2wiki_proofkg_extension_plans_v1 import merge_plan_rows


def _cohort(qid: str) -> dict:
    question = f"Where does unique marker {qid} lead?"
    return {
        "question_key": f"2wikimultihopqa::{qid}",
        "dataset": "2wikimultihopqa",
        "qid": qid,
        "question": question,
        "question_sha256": question_sha256(question),
        "gold_access": False,
    }


def _prediction(row: dict, *, valid: bool = True) -> dict:
    return {
        **row,
        "generated_text": "{}",
        "predicted_target": {} if valid else None,
        "schema_valid": valid,
        "validation_errors": [] if valid else ["invalid"],
    }


def test_merge_is_exact_combined_order_and_preserves_invalid_rows():
    v1b_cohort = [_cohort(f"v-{index:03d}") for index in range(300)]
    reserve_cohort = [_cohort(f"r-{index:03d}") for index in range(50)]
    combined = [
        row
        for pair in zip(v1b_cohort[:50], reserve_cohort)
        for row in pair
    ] + v1b_cohort[50:]
    v1b_predictions = [_prediction(row) for row in reversed(v1b_cohort)]
    reserve_predictions = [
        _prediction(row, valid=index != 7)
        for index, row in enumerate(reversed(reserve_cohort))
    ]
    merged, gates = merge_plan_rows(
        combined_rows=combined,
        v1b_cohort_rows=v1b_cohort,
        reserve_cohort_rows=reserve_cohort,
        v1b_prediction_rows=v1b_predictions,
        reserve_prediction_rows=reserve_predictions,
    )
    assert all(gates.values())
    assert [row["question_key"] for row in merged] == [
        row["question_key"] for row in combined
    ]
    assert sum(not row["schema_valid"] for row in merged) == 1


def test_merge_rejects_prediction_identity_drift_or_gold_field():
    v1b_cohort = [_cohort(f"v-{index:03d}") for index in range(300)]
    reserve_cohort = [_cohort(f"r-{index:03d}") for index in range(50)]
    combined = [*v1b_cohort, *reserve_cohort]
    v1b_predictions = [_prediction(row) for row in v1b_cohort]
    reserve_predictions = [_prediction(row) for row in reserve_cohort]
    drifted = copy.deepcopy(reserve_predictions)
    drifted[0]["answer"] = "forbidden"
    with pytest.raises(ValueError, match="forbidden fields"):
        merge_plan_rows(
            combined_rows=combined,
            v1b_cohort_rows=v1b_cohort,
            reserve_cohort_rows=reserve_cohort,
            v1b_prediction_rows=v1b_predictions,
            reserve_prediction_rows=drifted,
        )
