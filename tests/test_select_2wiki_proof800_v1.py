from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json

import pytest

from kgproweight.kg.question_kg import make_question_kg_record, question_sha256
from kgproweight.reward.trajectory_source_gate import make_source_gate_record
from scripts.prepare.freeze_mixed_ppo_three_dataset_v4_proof800 import (
    normalise_proof_candidates,
)
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.prepare.materialize_2wiki_proofkg_unified_v2 import (
    CANDIDATE_SCHEMA_VERSION,
)
from scripts.prepare.select_2wiki_proof800_v1 import (
    DATASET,
    HISTORICAL_CUTOFF,
    QTYPES,
    TARGET_BY_TYPE,
    _canonical_sha256,
    _passages_sha256,
    assess_candidate,
    choose_exact_proof800,
)


def _joined_candidate(qid: str = "row-1", qtype: str = "inference"):
    question = f"Which result follows the unique marker {qid}?"
    triple = ["Alpha", "parent", "Beta"]
    plan = {
        "recognized": True,
        "anchors": ["Alpha"],
        "hops": [
            {
                "subject": "Alpha",
                "pids": ["P22"],
                "output_slot": "hop_1",
                "relation_role": "answer_operand",
            }
        ],
    }
    record = make_question_kg_record(
        dataset=DATASET,
        qid=qid,
        question=question,
        triples=[triple],
        query_plan=plan,
        provenance={
            "builder_version": "clean-closure-test-v1",
            "gold_access": False,
            "complete_plan_execution": True,
            "historical_cutoff": HISTORICAL_CUTOFF,
            "planner_predictions_sha256": "c" * 64,
        },
    )
    record.update(
        {
            "planner_schema_valid": True,
            "runtime_error": None,
            "execution": {
                "anchor_entities": {
                    "Alpha": {
                        "surface": "Alpha",
                        "qid": "Q1",
                        "abstained": False,
                    }
                },
                "hops": [
                    {
                        "hop_index": 1,
                        "subject": "Alpha",
                        "pids": ["P22"],
                        "input_entities": [
                            {"surface": "Alpha", "qid": "Q1", "abstained": False}
                        ],
                        "matches": [triple],
                        "output_entities": [],
                    }
                ],
                "n_triples": 1,
                "complete_plan_execution": True,
            },
        }
    )
    passages = [
        {"id": f"doc-{index}", "source": "canonical", "contents": f"Text {index}."}
        for index in range(10)
    ]
    passage_hash = _passages_sha256(passages)
    cohort = {
        "schema_version": "2wiki-proofkg-official-raw-question-only-v2",
        "dataset": DATASET,
        "qid": qid,
        "question": question,
        "question_sha256": question_sha256(question),
        "family_version": FAMILY_VERSION,
        "family_sha256": family_sha256(question),
        "question_type": qtype,
        "gold_access": False,
    }
    silver = {
        "dataset": DATASET,
        "qid": qid,
        "question": question,
        "question_sha256": question_sha256(question),
        "answer": "Beta",
        "steps": [],
        "teacher_output": "",
        "retrieved_passages": passages,
        "metadata": {
            "gold_answer": "Beta",
            "question_type": qtype,
            "retrieved_passages_sha256": passage_hash,
        },
    }
    gate = make_source_gate_record(
        record,
        dataset=DATASET,
        qid=qid,
        question=question,
        text_evidence_available=True,
        historical_cutoff=HISTORICAL_CUTOFF,
    )
    wrapper = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "question_key": f"{DATASET}::{qid}",
        "dataset": DATASET,
        "qid": qid,
        "question": question,
        "question_sha256": question_sha256(question),
        "family_version": FAMILY_VERSION,
        "family_sha256": family_sha256(question),
        "question_type": qtype,
        "proof_passages_sha256": passage_hash,
        "question_kg_record": record,
        "gold_access": False,
        "evaluation_eligible": False,
    }
    telemetry = {
        "question_key": f"{DATASET}::{qid}",
        "dataset": DATASET,
        "qid": qid,
        "question_sha256": question_sha256(question),
        "family_sha256": family_sha256(question),
        "question_type": qtype,
        "gold_access_false": True,
        "runtime_error_zero": True,
        "all_root_anchors_resolved": True,
        "all_hops_complete": True,
        "graph_nonempty": True,
        "retained_edges_traceable": True,
        "m_graph": 1,
        "kg_sha256": _canonical_sha256(record["kg_subgraph"]),
        "execution_sha256": _canonical_sha256(record["execution"]),
        "runtime_record_sha256": _canonical_sha256(record),
    }
    return cohort, wrapper, silver, record, gate, telemetry


def _assess(values):
    cohort, wrapper, silver, record, gate, telemetry = values
    return assess_candidate(
        cohort=cohort,
        wrapper=wrapper,
        silver=silver,
        record=record,
        gate=gate,
        closure_telemetry=telemetry,
        historical_cutoff=HISTORICAL_CUTOFF,
        planner_predictions_sha256="c" * 64,
    )


def test_strict_candidate_passes_all_mechanical_checks_and_downstream_wrapper():
    values = _joined_candidate()
    eligible, checks = _assess(values)
    assert eligible is True
    assert checks and all(checks.values())
    wrapper = values[1]
    rows, reasons = normalise_proof_candidates(
        [wrapper], historical_cutoff=HISTORICAL_CUTOFF
    )
    assert len(rows) == 1
    assert reasons == {"eligible": 1}


@pytest.mark.parametrize(
    ("mutate", "failed_check"),
    [
        (
            lambda values: values[3]["execution"]["anchor_entities"]["Alpha"].update(
                qid="", abstained=True
            ),
            "all_root_anchors_resolved",
        ),
        (
            lambda values: values[3]["execution"]["hops"][0].update(matches=[]),
            "all_hops_complete_and_traceable",
        ),
        (
            lambda values: values[3]["provenance"].update(historical_cutoff="2099"),
            "provenance_complete",
        ),
        (
            lambda values: values[2]["retrieved_passages"].pop(),
            "passages_complete_and_hash_bound",
        ),
        (
            lambda values: values[5].update(kg_sha256="0" * 64),
            "closure_hash_attestation_exact",
        ),
    ],
)
def test_strict_candidate_fails_closed_on_each_required_contract(mutate, failed_check):
    values = list(copy.deepcopy(_joined_candidate()))
    mutate(values)
    eligible, checks = _assess(values)
    assert eligible is False
    assert checks[failed_check] is False


def _selection_row(qtype: str, index: int, *, family_mod: int = 250) -> dict:
    question = f"Question {qtype} marker {index}?"
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "question_key": f"{DATASET}::{qtype}-{index}",
        "dataset": DATASET,
        "qid": f"{qtype}-{index}",
        "question": question,
        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "family_sha256": f"{qtype}-family-{index % family_mod}",
        "family_version": FAMILY_VERSION,
        "question_type": qtype,
        "gold_access": False,
    }


def test_exact_selector_hits_four_200_quotas_and_prefers_family_diversity():
    candidates = [
        _selection_row(qtype, index, family_mod=(150 if qtype == "inference" else 250))
        for qtype in QTYPES
        for index in range(230)
    ]
    selected, stats = choose_exact_proof800(candidates)
    assert len(selected) == 800
    assert Counter(row["question_type"] for row in selected) == Counter(TARGET_BY_TYPE)
    assert len({row["qid"] for row in selected}) == 800
    assert stats["by_question_type"]["inference"]["selected_unique_families"] == 150
    assert stats["by_question_type"]["inference"]["selected_repeated_family_rows"] == 50
    assert stats["by_question_type"]["comparison"]["selected_repeated_family_rows"] == 0


def test_exact_selector_fails_instead_of_lowering_one_type_quota():
    candidates = [
        _selection_row(qtype, index)
        for qtype in QTYPES
        for index in range(199 if qtype == "inference" else 205)
    ]
    with pytest.raises(RuntimeError, match=r"Proof800/inference: only 199/200"):
        choose_exact_proof800(candidates)


def test_selection_is_deterministic_and_does_not_consume_answer_values():
    candidates = [
        _selection_row(qtype, index, family_mod=50)
        for qtype in QTYPES
        for index in range(210)
    ]
    first, _ = choose_exact_proof800(candidates)
    second, _ = choose_exact_proof800(list(reversed(candidates)))
    assert [row["question_key"] for row in first] == [row["question_key"] for row in second]
    assert all("answer" not in row and "gold_answer" not in row for row in first)
