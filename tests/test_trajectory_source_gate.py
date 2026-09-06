from __future__ import annotations

import copy

import pytest

from kgproweight.kg.question_kg import make_question_kg_record
from kgproweight.reward.trajectory_source_gate import evaluate_graph_gate


def _record(dataset: str = "2wikimultihopqa"):
    question = "What does Alpha ultimately link to?"
    record = make_question_kg_record(
        dataset=dataset,
        qid="q1",
        question=question,
        triples=[("Alpha", "links to", "Beta"), ("Beta", "links to", "Gamma")],
        query_plan={
            "recognized": True,
            "hops": [
                {"subject": "Alpha", "pids": ["P1"], "output_slot": "hop_1"},
                {"subject": "$hop_1", "pids": ["P2"], "output_slot": "hop_2"},
            ],
        },
        provenance={
            "builder_version": "synthetic-test-builder",
            "gold_access": False,
            "complete_plan_execution": True,
        },
    )
    record["runtime_error"] = None
    record["execution"] = {
        "complete_plan_execution": True,
        "hops": [
            {
                "hop_index": 1,
                "input_entities": [{"qid": "Q1"}],
                "matches": [["Alpha", "links to", "Beta"]],
            },
            {
                "hop_index": 2,
                "input_entities": [{"qid": "Q2"}],
                "matches": [["Beta", "links to", "Gamma"]],
            },
        ],
    }
    return question, record


@pytest.mark.parametrize("dataset", ["2wikimultihopqa", "hotpotqa", "musique"])
def test_gate_is_dataset_independent_for_an_equally_valid_record(dataset):
    question, record = _record(dataset)
    decision = evaluate_graph_gate(
        record,
        dataset=dataset,
        qid="q1",
        question=question,
        historical_cutoff="2020-12-09T23:59:59Z",
    )
    assert decision.m_graph == 1
    assert decision.graph_eligible is True


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (lambda row: row["provenance"].update(gold_access=True), "gold_access_false"),
        (lambda row: row.update(runtime_error="boom"), "runtime_error_zero"),
        (lambda row: row["execution"].update(complete_plan_execution=False), "complete_declared"),
        (lambda row: row["execution"]["hops"][1].update(matches=[]), "all_hops_executed_with_qid_pid_tail"),
    ],
)
def test_gate_fails_closed_on_contract_violation(mutation, failed_check):
    question, original = _record()
    record = copy.deepcopy(original)
    mutation(record)
    decision = evaluate_graph_gate(
        record,
        dataset="2wikimultihopqa",
        qid="q1",
        question=question,
        historical_cutoff="2020-12-09T23:59:59Z",
    )
    assert decision.m_graph == 0
    assert decision.checks[failed_check] is False


def test_empty_graph_routes_to_text_without_dataset_fallback():
    question, record = _record("hotpotqa")
    record["kg_subgraph"] = []
    record["execution"] = {}
    record["query_plan"] = {}
    record["provenance"] = {
        "builder_version": "outcome-only",
        "gold_access": False,
        "complete_plan_execution": False,
    }
    decision = evaluate_graph_gate(
        record,
        dataset="hotpotqa",
        qid="q1",
        question=question,
        historical_cutoff="2020-12-09T23:59:59Z",
    )
    assert decision.m_graph == 0
    assert decision.routing_reason == "no_trusted_graph"

