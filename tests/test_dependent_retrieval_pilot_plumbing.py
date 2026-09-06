"""Small regression tests for the dependent-retrieval pilot boundaries."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from kgproweight.kg.question_kg import question_sha256
import scripts.pilot.audit_plan_once_dependent_retrieval as runner
from scripts.prepare.finalize_dependent_retrieval_pilot import (
    _enforce_retrieval_completion_gates,
    _validate_and_enrich,
)
from scripts.eval.evaluate_dependent_retrieval_pilot import _decision, _mechanism


def _args():
    return SimpleNamespace(
        max_hops=4,
        max_query_variants=2,
        step_rerank_topk=10,
        cross_encoder_model="unused",
        bridge_source_docs=5,
        max_bridge_candidates=2,
        original_quota=6,
        per_hop_quota=2,
        total_passages=10,
    )


class _Retriever:
    def __init__(self):
        self.search_calls = []
        self.batch_calls = []

    def search(self, query):
        self.search_calls.append(query)
        return self._result(query)

    def batch_search(self, queries):
        self.batch_calls.append(list(queries))
        return [self._result(query) for query in queries]

    @staticmethod
    def _result(query):
        if query == "Root author":
            return [
                {"id": "h1", "title": "Bridge Alpha", "contents": "Bridge Alpha\nbody"},
                {"id": "h2", "title": "Bridge Beta", "contents": "Bridge Beta\nbody"},
            ]
        if query in {"Bridge Alpha birthplace", "Bridge Beta birthplace"}:
            return [{"id": query, "title": "City", "contents": "City\nbody"}]
        return []


class _LayerRetriever:
    """Deterministic fake exposing equivalent scalar and batch interfaces."""

    def __init__(self):
        self.search_calls = []
        self.batch_calls = []

    @staticmethod
    def _result(query):
        if query in {"Root A author", "Root B author"}:
            owner = query.split()[1]
            return [
                {
                    "id": f"{owner}-bridge-1",
                    "title": f"Bridge {owner} Alpha",
                    "contents": f"Bridge {owner} Alpha\nbody",
                },
                {
                    "id": f"{owner}-bridge-2",
                    "title": f"Bridge {owner} Beta",
                    "contents": f"Bridge {owner} Beta\nbody",
                },
            ]
        if query.endswith("birthplace"):
            return [{
                "id": f"answer::{query}",
                "title": f"City for {query}",
                "contents": f"City for {query}\nbody",
            }]
        return []

    def search(self, query):
        self.search_calls.append(query)
        return self._result(query)

    def batch_search(self, queries):
        self.batch_calls.append(list(queries))
        return [self._result(query) for query in queries]


def _two_hop_row(label):
    return {
        "dataset": "hotpotqa",
        "qid": f"synthetic-{label}",
        "question_sha256": f"synthetic-{label}",
        "arm_a_passages": [
            {"id": f"{label}-o{index}", "contents": f"Original {label} {index}\nbody"}
            for index in range(10)
        ],
        "plan": {
            "steps": [
                {
                    "subject": f"Root {label}",
                    "relation_label": "author",
                    "output_slot": "hop_1",
                    "dependencies": [],
                },
                {
                    "subject": "$hop_1",
                    "relation_label": "birthplace",
                    "output_slot": "hop_2",
                    "dependencies": ["hop_1"],
                },
            ]
        },
    }


def test_layer_batched_execution_is_equivalent_to_scalar_query_order(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_rerank_step_results",
        lambda query, results, **kwargs: [dict(row) for row in results[: kwargs["topk"]]],
    )
    monkeypatch.setattr(
        runner,
        "rerank_passages",
        lambda queries, results, **kwargs: [
            [dict(row) for row in current[: kwargs["topk"]]] for current in results
        ],
    )
    rows = [_two_hop_row("A"), _two_hop_row("B")]

    scalar_retriever = _LayerRetriever()
    scalar = [runner._execute_row(row, scalar_retriever, _args()) for row in rows]

    batch_retriever = _LayerRetriever()
    batched = runner._execute_rows_batched(rows, batch_retriever, _args())

    assert batched == scalar
    assert batch_retriever.batch_calls == [
        ["Root A author", "Root B author"],
        [
            "Bridge A Alpha birthplace",
            "Bridge A Beta birthplace",
            "Bridge B Alpha birthplace",
            "Bridge B Beta birthplace",
        ],
    ]
    assert [query for layer in batch_retriever.batch_calls for query in layer] == (
        ["Root A author", "Root B author"]
        + [
            "Bridge A Alpha birthplace",
            "Bridge A Beta birthplace",
            "Bridge B Alpha birthplace",
            "Bridge B Beta birthplace",
        ]
    )
    # The scalar executor is retained only as a test oracle; the formal path
    # must use no one-query full-index scans.
    assert batch_retriever.search_calls == []


def test_layer_batched_full_retriever_error_is_not_silently_fallback(monkeypatch):
    class BrokenBatchRetriever:
        def batch_search(self, queries):
            raise RuntimeError("synthetic full-index failure")

    with pytest.raises(RuntimeError, match="synthetic full-index failure"):
        runner._execute_rows_batched(
            [_two_hop_row("A")], BrokenBatchRetriever(), _args()
        )


def test_runner_executes_two_query_variants_as_one_logical_hop(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_rerank_step_results",
        lambda query, results, **kwargs: [dict(row) for row in results[: kwargs["topk"]]],
    )
    row = {
        "dataset": "hotpotqa",
        "qid": "synthetic",
        "question_sha256": "synthetic",
        "arm_a_passages": [
            {"id": f"o{index}", "contents": f"Original {index}\nbody"}
            for index in range(10)
        ],
        "plan": {
            "steps": [
                {
                    "subject": "Root",
                    "relation_label": "author",
                    "output_slot": "hop_1",
                    "dependencies": [],
                },
                {
                    "subject": "$hop_1",
                    "relation_label": "birthplace",
                    "output_slot": "hop_2",
                    "dependencies": ["hop_1"],
                },
            ]
        },
    }
    passages, detail = runner._execute_row(row, _Retriever(), _args())
    assert detail["execution_status"] == "executed"
    assert detail["dependent_query_count"] == 2
    assert detail["second_hop_query_count"] == 2
    assert len(detail["hops"]) == 2
    assert len(passages) == 10


def test_runner_execution_error_falls_back_exactly(monkeypatch):
    class BrokenRetriever:
        def search(self, query):
            raise RuntimeError("synthetic retrieval failure")

    original = [{"id": str(index), "contents": str(index)} for index in range(10)]
    row = {
        "dataset": "hotpotqa",
        "qid": "synthetic",
        "question_sha256": "synthetic",
        "arm_a_passages": original,
        "plan": {
            "steps": [{
                "subject": "Root", "relation_label": "author",
                "output_slot": "hop_1", "dependencies": [],
            }]
        },
    }
    passages, detail = runner._execute_row(row, BrokenRetriever(), _args())
    assert passages == original
    assert passages is not original
    assert detail["execution_status"] == "fallback_execution_error"
    assert detail["fallback_exact"] is True
    assert detail["fallback_reason"] == "execution_error"


def test_finalizer_joins_gold_only_after_strict_pair_validation():
    question = "Synthetic question?"
    common = {
        "row_id": "r1",
        "question_key": "hotpotqa::q1",
        "dataset": "hotpotqa",
        "qid": "q1",
        "question": question,
        "question_sha256": question_sha256(question),
        "split": "pilot",
        "gold_access": False,
        "kg_subgraph": [["h", "r", "t"]],
        "legacy_kg_sha256": "kg",
    }
    arm_a = [{**common, "arm": "A_question_only", "retrieved_passages": []}]
    arm_b = [{**common, "arm": "B_dependent", "retrieved_passages": []}]
    gold = {
        "hotpotqa::q1": {
            "question": question,
            "question_sha256": question_sha256(question),
            "gold_answers": ["answer"],
        }
    }
    final_a, final_b = _validate_and_enrich(arm_a, arm_b, gold)
    assert final_a[0]["gold_answers"] == final_b[0]["gold_answers"] == ["answer"]
    with pytest.raises(ValueError, match="already contains Gold"):
        _validate_and_enrich([{**arm_a[0], "gold_answers": ["leak"]}], arm_b, gold)
    mismatched = deepcopy(arm_b)
    mismatched[0]["qid"] = "other"
    with pytest.raises(ValueError, match="common fields differ"):
        _validate_and_enrich(arm_a, mismatched, gold)


def test_finalizer_requires_zero_retrieval_runtime_errors():
    report = {
        "by_dataset": {
            dataset: {
                "n": 30,
                "runtime_errors": 0,
                "fallback_execution_error": 0,
                "fallback_exact": True,
            }
            for dataset in ("hotpotqa", "musique")
        }
    }
    _enforce_retrieval_completion_gates(report)
    report["by_dataset"]["musique"]["runtime_errors"] = 1
    with pytest.raises(ValueError, match="runtime_errors"):
        _enforce_retrieval_completion_gates(report)


def test_second_hop_rate_uses_dependent_step_eligible_subset_only():
    rows = []
    for index in range(30):
        eligible = index < 23
        rows.append({
            "retrieval_trace": {
                "plan_executable": True,
                "has_dependent_step": eligible,
                "dependent_query_count": 1 if eligible else 0,
                "second_hop_query_count": 1 if eligible else 0,
                "new_dependent_candidate_count": 1 if eligible else 0,
            }
        })
    result = _mechanism(rows)
    assert result["has_dependent_step_observed_n"] == 30
    assert result["dependent_step_eligible_n"] == 23
    assert result["second_hop_query_observed_n"] == 23
    assert result["second_hop_query_nonempty_rate"] == 1.0


def test_second_hop_rate_is_na_when_no_plan_has_a_dependency():
    rows = [{
        "retrieval_trace": {
            "plan_executable": True,
            "has_dependent_step": False,
            "dependent_query_count": 0,
            "second_hop_query_count": 0,
            "new_dependent_candidate_count": 0,
        }
    }]
    result = _mechanism(rows)
    assert result["dependent_step_eligible_n"] == 0
    assert result["second_hop_query_observed_n"] == 0
    assert result["second_hop_query_nonempty_rate"] is None
    protocol = {
        "decision_gates": {
            "pooled_net_correct_gain_min": 0,
            "max_net_correct_loss_per_dataset": 1,
            "parse_count_delta_min": 0,
            "second_hop_query_nonempty_rate_min_each_dataset": 0.8,
        }
    }
    outcome = _decision(
        protocol,
        {"net_correct": 0, "delta_f1": 0.1, "parse_count_delta": 0},
        {"hotpotqa": {"net_correct": 0}},
        {"hotpotqa": result},
        True,
    )
    assert outcome["checks"]["second_hop_query_nonempty_rate"] is False
    assert outcome["all_pass"] is False
