"""Fake-only regression tests for the isolated v6 materialisation runner."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

import scripts.pilot.audit_plan_once_dependent_retrieval_v6 as runner


QUESTION = "Where was the author of Root Work born?"


def _args():
    return SimpleNamespace(max_hops=4, max_query_variants=2, step_rerank_topk=10)


def _doc(doc_id: str, text: str | None = None):
    return {
        "id": doc_id,
        "title": doc_id,
        "contents": text or f"{doc_id}\nbody for {doc_id}",
    }


def _arm_a():
    return [_doc(f"original-{index}") for index in range(1, 11)]


def _row(*, dependent: bool = True):
    steps = [{
        "subject": "Root Work",
        "relation_label": "author",
        "output_slot": "hop_1",
        "dependencies": [],
    }]
    if dependent:
        steps.append({
            "subject": "$hop_1",
            "relation_label": "place of birth",
            "output_slot": "hop_2",
            "dependencies": ["hop_1"],
        })
    return {
        "dataset": "hotpotqa",
        "qid": "synthetic",
        "question": QUESTION,
        "question_sha256": "synthetic-question-sha",
        "arm_a_passages": _arm_a(),
        "plan": {"steps": steps},
    }


def _hint(surface: str, source: str = "v5_accepted"):
    return {
        "surface": surface,
        "normalized_surface": surface.casefold(),
        "score": 10,
        "provenance": [],
        "admission": {"source": source},
    }


def _hint_selector(*, empty=False):
    def select(**kwargs):
        hints = [] if empty else [_hint("Bridge A"), _hint("Bridge B", "raw_rank_fill")]
        return hints, {
            "raw_candidate_count": 2,
            "hard_rejected_candidates": [],
            "hints": [value["admission"] for value in hints],
        }

    return select


class _Retriever:
    def __init__(self):
        self.batch_calls = []

    def batch_search(self, queries):
        self.batch_calls.append(list(queries))
        result = []
        for query in queries:
            if query == "Root Work author":
                result.append([_doc("root", "Root Work\nRoot context")])
            elif query == f"{QUESTION}\nBridge A place of birth":
                result.append([
                    _doc("a1", "A1\nweak branch A"),
                    _doc("a2", "A2\nweak branch A"),
                    _doc("a3", "A3\nnot eligible"),
                ])
            elif query == f"{QUESTION}\nBridge B place of birth":
                result.append([
                    _doc("b1", "B1\nstrong unique second branch"),
                    _doc("b2", "B2\nstrong second branch"),
                    _doc("b3", "B3\nnot eligible"),
                ])
            elif query == f"{QUESTION}\nplace of birth":
                result.append([_doc("fallback", "Fallback\nstrong fallback result")])
            else:
                result.append([])
        return result


class _CrossEncoder:
    def __init__(self, *, low=False):
        self.predict_calls = []
        self.low = low

    def predict(self, pairs, show_progress_bar=False):
        assert show_progress_bar is False
        self.predict_calls.append(list(pairs))
        scores = []
        for _, text in pairs:
            if self.low and any(value in text for value in ("B1", "B2", "Fallback")):
                scores.append(-10.0)
            elif "B1" in text or "Fallback" in text:
                scores.append(9.0)
            elif "B2" in text:
                scores.append(8.0)
            elif "A1" in text:
                scores.append(0.2)
            elif "A2" in text:
                scores.append(0.1)
            elif "original-9" in text or "original-10" in text:
                scores.append(1.0)
            else:
                scores.append(0.0)
        return scores


def test_no_dependent_step_is_exact_and_does_not_search():
    row = _row(dependent=False)
    retriever = _Retriever()
    ce = _CrossEncoder()
    passages, detail = runner._execute_rows_batched_v6(
        [row], retriever, _args(), cross_encoder=ce
    )[0]
    assert passages == row["arm_a_passages"]
    assert passages is not row["arm_a_passages"]
    assert detail["execution_status"] == "fallback_no_dependent_step"
    assert retriever.batch_calls == []
    assert ce.predict_calls == []


def test_second_query_variant_has_independent_candidates_and_can_win(monkeypatch):
    monkeypatch.setattr(runner, "select_bridge_query_hints_v6", _hint_selector())
    row = _row()
    original = deepcopy(row["arm_a_passages"])
    retriever = _Retriever()
    ce = _CrossEncoder()
    passages, detail = runner._execute_rows_batched_v6(
        [row], retriever, _args(), cross_encoder=ce
    )[0]

    assert retriever.batch_calls == [
        ["Root Work author"],
        [
            f"{QUESTION}\nBridge A place of birth",
            f"{QUESTION}\nBridge B place of birth",
        ],
    ]
    assert passages[:8] == original[:8]
    assert [value["id"] for value in passages[-2:]] == ["b1", "b2"]
    assert detail["execution_status"] == "executed_changed"
    assert detail["dependent_query_count"] == 2
    assert detail["all_dependent_queries_start_with_exact_question"] is True
    assert detail["max_query_variants_per_logical_hop"] == 2
    assert detail["query_hint_summary"]["v5_admitted_hints"] == 1
    assert detail["query_hint_summary"]["exploratory_hints"] == 1
    assert detail["merge"]["candidate_occurrences_considered"] == 4
    assert {row["query_variant_id"] for row in detail["merge"]["selected_new"]} == {
        "hop_2::q2"
    }
    assert detail["safety"]["prefix8_exact"] is True
    assert detail["safety"]["root_passages_injected"] == 0
    assert detail["safety"]["duplicate_output_documents"] == 0
    final_pairs = ce.predict_calls[-1]
    assert len(final_pairs) == 6  # A tail2 + top2 from each of two branches.
    assert {question for question, _ in final_pairs} == {QUESTION}
    assert all("a3" not in text.casefold() and "b3" not in text.casefold() for _, text in final_pairs)


def test_no_hint_uses_question_anchored_relation_fallback(monkeypatch):
    monkeypatch.setattr(runner, "select_bridge_query_hints_v6", _hint_selector(empty=True))
    row = _row()
    retriever = _Retriever()
    passages, detail = runner._execute_rows_batched_v6(
        [row], retriever, _args(), cross_encoder=_CrossEncoder()
    )[0]
    assert retriever.batch_calls == [
        ["Root Work author"],
        [f"{QUESTION}\nplace of birth"],
    ]
    assert detail["hops"][1]["query_renderer"]["mode"] == "no_hint_fallback"
    assert detail["dependent_query_count"] == 1
    assert detail["execution_status"] == "executed_changed"
    assert passages[-1]["id"] == "fallback"


def test_low_candidates_leave_arm_a_byte_equal(monkeypatch):
    monkeypatch.setattr(runner, "select_bridge_query_hints_v6", _hint_selector())
    row = _row()
    passages, detail = runner._execute_rows_batched_v6(
        [row], _Retriever(), _args(), cross_encoder=_CrossEncoder(low=True)
    )[0]
    assert passages == row["arm_a_passages"]
    assert detail["execution_status"] == "fallback_no_candidate_strictly_better"
    assert detail["fallback_exact"] is True


def test_global_retriever_failure_propagates(monkeypatch):
    monkeypatch.setattr(runner, "select_bridge_query_hints_v6", _hint_selector())

    class Broken:
        def batch_search(self, queries):
            raise RuntimeError("synthetic full-index failure")

    with pytest.raises(RuntimeError, match="synthetic full-index failure"):
        runner._execute_rows_batched_v6(
            [_row()], Broken(), _args(), cross_encoder=_CrossEncoder()
        )


def test_final_cross_encoder_failure_propagates(monkeypatch):
    monkeypatch.setattr(runner, "select_bridge_query_hints_v6", _hint_selector())

    class BrokenFinal(_CrossEncoder):
        def predict(self, pairs, show_progress_bar=False):
            if pairs and all(question == QUESTION for question, _ in pairs):
                raise RuntimeError("synthetic final CE failure")
            return super().predict(pairs, show_progress_bar=show_progress_bar)

    with pytest.raises(RuntimeError, match="synthetic final CE failure"):
        runner._execute_rows_batched_v6(
            [_row()], _Retriever(), _args(), cross_encoder=BrokenFinal()
        )
