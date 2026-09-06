"""CPU-only tests for the canonical Query Controller v1 action contract."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from kgproweight.eval.query_controller_v1 import (
    ActionValidationError,
    SCHEMA_VERSION,
    STATE_VERSION,
    evaluate_action_records,
    parse_target_response,
    validate_action_record,
)
from kgproweight.kg.question_kg import question_sha256
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256
from scripts.eval.evaluate_query_controller_actions import score_prediction_rows


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(slot: str = "q1") -> dict:
    dataset = "musique"
    qid = "train_123"
    question = "When was the university that owned The Example founded?"
    q1_target = {
        "action": "retrieve",
        "query": "The Example owned by university",
        "anchor": "The Example",
        "relation_intent": "owned by",
        "pid": None,
        "dependencies": [],
        "output_slot": "q1",
        "source_action": "text",
    }
    if slot == "q1":
        state = {
            "state_version": STATE_VERSION,
            "original_question": question,
            "previous_actions": [],
            "verified_observations": [],
        }
        target = q1_target
        intermediate_used = False
        turn = 1
    else:
        answer = "Example University"
        excerpt = "The Example is owned by Example University."
        state = {
            "state_version": STATE_VERSION,
            "original_question": question,
            "previous_actions": [
                {
                    "slot": "q1",
                    "action": "retrieve",
                    "query": q1_target["query"],
                    "output_slot": "q1",
                }
            ],
            "verified_observations": [
                {
                    "answer": answer,
                    "answer_sha256": _sha(answer),
                    "evidence_excerpt": excerpt,
                    "evidence_excerpt_sha256": _sha(excerpt),
                    "document_id": "doc-7",
                    "document_title": "The Example",
                    "sentence_index": 0,
                    "provenance": {
                        "source": "train_annotation_support",
                        "annotation_path": (
                            "metadata.metadata.question_decomposition[0].answer"
                        ),
                        "binding_method": (
                            "decomposition_step_support_answer_surface"
                        ),
                    },
                }
            ],
        }
        target = {
            "action": "retrieve",
            "query": "Example University founding year",
            "anchor": answer,
            "relation_intent": "founded",
            "pid": "P571",
            "dependencies": ["q1"],
            "output_slot": "q2",
            "source_action": "text",
        }
        intermediate_used = True
        turn = 2
    return {
        "schema_version": SCHEMA_VERSION,
        "example_id": f"{dataset}::{qid}::{slot}",
        "dataset": dataset,
        "qid": qid,
        "question_key": f"{dataset}::{qid}",
        "question_sha256": question_sha256(question),
        "family_sha256": family_sha256(question),
        "split": "train",
        "slot": slot,
        "turn_index": turn,
        "state": state,
        "target": target,
        "source_provenance": {"source_split": "train", "annotation": "decomposition"},
        "gold_boundary": {
            "train_intermediate_annotation_used": intermediate_used,
            "gold_final_answer_visible": False,
            "evaluation_gold_access": False,
        },
    }


def test_q1_and_q2_records_validate_and_nullable_pid_is_allowed() -> None:
    q1 = _record("q1")
    q2 = _record("q2_dynamic")
    assert validate_action_record(q1) == q1
    assert validate_action_record(q2) == q2


def test_confirmation_is_a_valid_action_release_split() -> None:
    row = _record("q2_dynamic")
    row["split"] = "confirmation"
    assert validate_action_record(row, expected_split="confirmation") == row
    with pytest.raises(ActionValidationError) as caught:
        validate_action_record(row, expected_split="dev")
    assert "split" in caught.value.codes


def test_q2_requires_closed_dependency_and_verified_answer_state_use() -> None:
    row = _record("q2_dynamic")
    row["target"]["dependencies"] = []
    row["target"]["query"] = "unrelated founding year"
    with pytest.raises(ActionValidationError) as caught:
        validate_action_record(row)
    assert "dependency_not_closed" in caught.value.codes
    assert "state_not_used" in caught.value.codes


def test_q2_state_use_requires_whole_normalized_surface_boundary() -> None:
    row = _record("q2_dynamic")
    row["state"]["verified_observations"][0]["answer"] = "Film B"
    row["state"]["verified_observations"][0]["answer_sha256"] = _sha("Film B")
    row["target"]["query"] = "Film BLAH director"
    with pytest.raises(ActionValidationError) as caught:
        validate_action_record(row)
    assert "state_not_used" in caught.value.codes

    row["target"]["query"] = "Film B director"
    assert validate_action_record(row) == row


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda row: row["target"].update(query="#1 founding year"), "unresolved_placeholder"),
        (lambda row: row["target"].update(query=row["state"]["original_question"]), "query_repeat"),
        (lambda row: row["target"].update(source_action="graph"), "source_action_not_text"),
        (lambda row: row["gold_boundary"].update(gold_final_answer_visible=True), "gold_boundary"),
        (lambda row: row["target"].update(pid="occupation"), "pid"),
    ],
)
def test_mechanical_contract_rejects_invalid_actions(mutation, code: str) -> None:
    row = _record("q2_dynamic")
    mutation(row)
    with pytest.raises(ActionValidationError) as caught:
        validate_action_record(row)
    assert code in caught.value.codes


@pytest.mark.parametrize(
    ("container", "code"),
    [
        ("previous_action", "previous_action_schema"),
        ("observation", "observation_schema"),
        ("observation_provenance", "observation_provenance_schema"),
    ],
)
def test_model_visible_state_rejects_malicious_extra_gold_key(
    container: str, code: str
) -> None:
    row = _record("q2_dynamic")
    if container == "previous_action":
        row["state"]["previous_actions"][0]["gold_answer"] = "SECRET"
    elif container == "observation":
        row["state"]["verified_observations"][0]["gold_answer"] = "SECRET"
    else:
        row["state"]["verified_observations"][0]["provenance"][
            "gold_answer"
        ] = "SECRET"
    with pytest.raises(ActionValidationError) as caught:
        validate_action_record(row)
    assert code in caught.value.codes


def test_json_array_fields_reject_non_list_python_sequences() -> None:
    row = _record("q2_dynamic")
    row["state"]["previous_actions"] = tuple(row["state"]["previous_actions"])
    row["state"]["verified_observations"] = tuple(
        row["state"]["verified_observations"]
    )
    row["target"]["dependencies"] = ("q1",)
    with pytest.raises(ActionValidationError) as caught:
        validate_action_record(row)
    assert "previous_actions_type" in caught.value.codes
    assert "verified_observations_type" in caught.value.codes
    assert "dependencies_type" in caught.value.codes


def test_exact_json_prediction_parser_reuses_reference_state() -> None:
    row = _record("q2_dynamic")
    response = json.dumps(row["target"], ensure_ascii=False)
    assert parse_target_response(response, reference_record=row) == row["target"]
    bad = deepcopy(row["target"])
    bad["query"] = "When was it founded?"
    with pytest.raises(ActionValidationError, match="state_not_used"):
        parse_target_response(json.dumps(bad), reference_record=row)


def test_release_audit_reports_independent_rates_and_duplicates() -> None:
    q1, q2 = _record("q1"), _record("q2_dynamic")
    report = evaluate_action_records([q1, q2])
    assert report["all_valid"] is True
    assert report["metrics"]["schema_valid_rate"] == 1.0
    assert report["metrics"]["state_use_valid_rate"] == 1.0
    duplicate = evaluate_action_records([q1, q1])
    assert duplicate["all_valid"] is False
    assert duplicate["duplicate_example_id_count"] == 1


def test_prediction_scorer_does_not_credit_reference_on_parse_failure() -> None:
    row = _record("q1")
    good = {
        "example_id": row["example_id"],
        "response_text": json.dumps(row["target"], ensure_ascii=False),
    }
    report, details = score_prediction_rows([row], [good])
    assert report["metrics"]["parsed_rate"] == 1.0
    assert report["metrics"]["schema_valid_rate"] == 1.0
    assert details[0]["target_exact"] is True

    bad = {"example_id": row["example_id"], "response_text": "not JSON"}
    report, details = score_prediction_rows([row], [bad])
    assert report["metrics"]["parsed_rate"] == 0.0
    assert report["metrics"]["schema_valid_rate"] == 0.0
    assert details[0]["target_exact"] is False
