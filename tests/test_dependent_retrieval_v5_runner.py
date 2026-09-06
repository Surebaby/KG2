"""Fake-only regression tests for the isolated v5 materialisation runner."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

import scripts.pilot.audit_plan_once_dependent_retrieval_v5 as runner


def _args():
    return SimpleNamespace(max_hops=4, max_query_variants=2, step_rerank_topk=10)


def _formal_args(tmp_path):
    inputs = {}
    for name in ("cohort", "retrieval_contexts", "musique_plans", "hotpot_plans"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        inputs[name] = path
    return SimpleNamespace(
        **inputs,
        datasets=["hotpotqa", "musique"],
        n_per_dataset=30,
        rrf_candidate_k=100,
        step_rerank_topk=10,
        cross_encoder_model=str(tmp_path / "ce"),
        max_hops=4,
        max_query_variants=2,
        experiment_id=runner.v5_freeze.EXPERIMENT_IDS["materialization"],
        preregistration=tmp_path / "protocol.json",
    )


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
        "question": "Where was the author of Root Work born?",
        "question_sha256": "synthetic-question-sha",
        "arm_a_passages": _arm_a(),
        "plan": {"steps": steps},
    }


class _Retriever:
    def __init__(self):
        self.batch_calls = []

    def batch_search(self, queries):
        self.batch_calls.append(list(queries))
        output = []
        for query in queries:
            if query == "Root Work author":
                output.append([
                    _doc("root-only", "Root Work\nRoot context"),
                    _doc("bridge-source", "Bridge Person\nBridge Person is an author."),
                ])
            elif query == "Bridge Person place of birth":
                output.append([
                    _doc("dependent-high", "Dependent High\nUseful answer passage"),
                    _doc("dependent-mid", "Dependent Mid\nUseful supporting passage"),
                    _doc("dependent-rank3", "Dependent Rank3\nNot eligible for merge"),
                ])
            else:
                output.append([])
        return output


class _CrossEncoder:
    def __init__(self):
        self.predict_calls = []

    def predict(self, pairs, show_progress_bar=False):
        assert show_progress_bar is False
        self.predict_calls.append(list(pairs))
        scores = []
        for _, text in pairs:
            if "dependent-high" in text.casefold() or "Dependent High" in text:
                scores.append(9.0)
            elif "dependent-mid" in text.casefold() or "Dependent Mid" in text:
                scores.append(8.0)
            elif "original-9" in text or "original-10" in text:
                scores.append(-2.0)
            else:
                scores.append(1.0)
        return scores


def _accept_bridge(**kwargs):
    assert kwargs["max_docs"] == 10
    assert kwargs["max_candidates"] == 2
    assert kwargs["max_body_chars"] == 1200
    assert len(kwargs["consumers"]) == 1
    assert kwargs["consumers"][0]["output_slot"] == "hop_2"
    return ([{"surface": "Bridge Person", "score": 10}], {
        "raw_candidate_count": 2,
        "accepted_count": 1,
        "candidate_decisions": [
            {"surface": "Bridge Person", "decision": "accept"},
            {"surface": "Wrong Type", "decision": "reject"},
        ],
        "fallback_recommended": False,
    })


def test_no_dependent_step_returns_arm_a_exact_without_search_or_ce():
    row = _row(dependent=False)
    retriever = _Retriever()
    ce = _CrossEncoder()
    passages, detail = runner._execute_rows_batched_v5(
        [row], retriever, _args(), cross_encoder=ce
    )[0]
    assert passages == row["arm_a_passages"]
    assert passages is not row["arm_a_passages"]
    assert detail["execution_status"] == "fallback_no_dependent_step"
    assert detail["fallback_exact"] is True
    assert detail["fallback_reason"] == "no_dependent_step"
    assert retriever.batch_calls == []
    assert ce.predict_calls == []


def test_required_bridge_abstention_stops_next_layer_and_falls_back_exact(monkeypatch):
    monkeypatch.setattr(
        runner,
        "select_bridge_candidates_v5",
        lambda **kwargs: ([], {
            "raw_candidate_count": 2,
            "accepted_count": 0,
            "candidate_decisions": [
                {"surface": "Root Work", "decision": "reject"},
                {"surface": "Wrong Type", "decision": "reject"},
            ],
            "fallback_recommended": True,
        }),
    )
    row = _row()
    retriever = _Retriever()
    ce = _CrossEncoder()
    passages, detail = runner._execute_rows_batched_v5(
        [row], retriever, _args(), cross_encoder=ce
    )[0]
    assert passages == row["arm_a_passages"]
    assert detail["execution_status"] == "fallback_bridge_abstain"
    assert detail["fallback_reason"] == "bridge_abstain"
    assert detail["selector_summary"] == {
        "required_producers": 1,
        "accepted_producers": 0,
        "raw_candidates": 2,
        "accepted_candidates": 0,
        "rejected_candidates": 2,
    }
    assert retriever.batch_calls == [["Root Work author"]]
    # Only the root-hop rerank ran; there was no final union score call.
    assert len(ce.predict_calls) == 1


def test_v5_executes_dependency_layers_and_protects_original_prefix(monkeypatch):
    monkeypatch.setattr(runner, "select_bridge_candidates_v5", _accept_bridge)
    row = _row()
    original_before = deepcopy(row["arm_a_passages"])
    retriever = _Retriever()
    ce = _CrossEncoder()
    passages, detail = runner._execute_rows_batched_v5(
        [row], retriever, _args(), cross_encoder=ce
    )[0]

    assert retriever.batch_calls == [
        ["Root Work author"],
        ["Bridge Person place of birth"],
    ]
    assert [hop["dependency_depth"] for hop in detail["hops"]] == [1, 2]
    assert [hop["dependencies"] for hop in detail["hops"]] == [[], ["slot_1"]]
    assert detail["execution_status"] == "executed_changed"
    assert detail["fallback_exact"] is False
    assert detail["dependent_query_count"] == 1
    assert detail["new_dependent_candidate_count"] == 2
    assert passages[:8] == original_before[:8]
    assert [value["id"] for value in passages[-2:]] == [
        "dependent-high", "dependent-mid"
    ]
    assert "root-only" not in {value["id"] for value in passages}
    assert detail["safety"]["prefix8_exact"] is True
    assert detail["safety"]["unauthorized_original_displacements"] == 0
    assert detail["safety"]["root_passages_injected"] == 0
    assert row["arm_a_passages"] == original_before

    # Root rerank, dependent rerank, then exactly one full-question union call.
    assert len(ce.predict_calls) == 3
    final_pairs = ce.predict_calls[-1]
    assert len(final_pairs) == 4  # A tail2 + dependent top2; root is absent.
    assert {question for question, _ in final_pairs} == {row["question"]}
    assert all("root-only" not in text for _, text in final_pairs)
    assert all("dependent-rank3" not in text for _, text in final_pairs)


def test_lower_scoring_dependent_documents_leave_arm_a_exact(monkeypatch):
    monkeypatch.setattr(runner, "select_bridge_candidates_v5", _accept_bridge)

    class LowCandidateCE(_CrossEncoder):
        def predict(self, pairs, show_progress_bar=False):
            self.predict_calls.append(list(pairs))
            return [
                -10.0 if "Dependent" in text else 1.0
                for _, text in pairs
            ]

    row = _row()
    passages, detail = runner._execute_rows_batched_v5(
        [row], _Retriever(), _args(), cross_encoder=LowCandidateCE()
    )[0]
    assert passages == row["arm_a_passages"]
    assert detail["execution_status"] == "fallback_no_candidate_strictly_better"
    assert detail["fallback_exact"] is True
    assert detail["new_dependent_candidate_count"] == 0


def test_global_batch_retrieval_failure_propagates(monkeypatch):
    monkeypatch.setattr(runner, "select_bridge_candidates_v5", _accept_bridge)

    class BrokenRetriever:
        def batch_search(self, queries):
            raise RuntimeError("synthetic full-index failure")

    with pytest.raises(RuntimeError, match="synthetic full-index failure"):
        runner._execute_rows_batched_v5(
            [_row()], BrokenRetriever(), _args(), cross_encoder=_CrossEncoder()
        )


def test_global_final_cross_encoder_failure_propagates(monkeypatch):
    monkeypatch.setattr(runner, "select_bridge_candidates_v5", _accept_bridge)
    row = _row()

    class FinalFailureCE(_CrossEncoder):
        def predict(self, pairs, show_progress_bar=False):
            # Detect the final call from its scientific contract: every pair is
            # scored against the original full question.  Its cardinality can
            # legitimately change after candidate dedup or a scoring-scope
            # optimisation, whereas fail-fast CE handling must not.
            if pairs and all(question == row["question"] for question, _ in pairs):
                raise RuntimeError("synthetic final CE failure")
            return super().predict(pairs, show_progress_bar=show_progress_bar)

    with pytest.raises(RuntimeError, match="synthetic final CE failure"):
        runner._execute_rows_batched_v5(
            [row], _Retriever(), _args(), cross_encoder=FinalFailureCE()
        )


def test_topological_schedule_batches_parallel_roots_before_consumer():
    steps = [
        {
            "subject": "Root A", "relation_label": "author",
            "output_slot": "hop_1", "dependencies": [],
        },
        {
            "subject": "Root B", "relation_label": "director",
            "output_slot": "hop_2", "dependencies": [],
        },
        {
            "subject": "$hop_1", "relation_label": "place of birth",
            "output_slot": "hop_3", "dependencies": ["hop_1"],
        },
        {
            "subject": "$hop_3", "relation_label": "country",
            "output_slot": "hop_4", "dependencies": ["hop_3"],
        },
    ]
    schedule = runner._step_schedule(steps)
    assert [value["dependency_depth"] for value in schedule] == [1, 1, 2, 3]
    assert schedule[0]["consumers"] == [steps[2]]
    assert schedule[1]["consumers"] == []
    assert schedule[2]["consumers"] == [steps[3]]


def test_dataset_summary_exposes_fail_closed_safety_fields():
    row = _row(dependent=False)
    passages, detail = runner._execute_rows_batched_v5(
        [row], _Retriever(), _args(), cross_encoder=_CrossEncoder()
    )[0]
    common = {"dataset": "hotpotqa", "qid": "synthetic"}
    arm_a = [{**common, "passages_sha256": runner._sha256_json(row["arm_a_passages"])}]
    arm_b = [{**common, "passages_sha256": runner._sha256_json(passages)}]
    summary = runner._aggregate_dataset("hotpotqa", [detail], arm_a, arm_b)
    assert summary["fallback_no_dependent_step"] == 1
    assert summary["fallback_exact"] is True
    assert summary["prefix8_exact"] is True
    assert summary["unauthorized_original_displacements"] == 0
    assert summary["root_passages_injected"] == 0
    assert summary["runtime_errors"] == 0


def test_formal_runtime_rechecks_preregistered_inputs_code_models_and_settings(
    monkeypatch, tmp_path
):
    args = _formal_args(tmp_path)
    code_file = tmp_path / "code.py"
    code_file.write_text("# frozen\n", encoding="utf-8")
    monkeypatch.setattr(
        runner.v5_freeze,
        "DEFAULT_CODE",
        {name: code_file for name in runner.v5_freeze.DEFAULT_CODE},
    )
    fake_identity = lambda path: {
        "path": str(path), "exists": True, "kind": "directory", "files": []
    }
    monkeypatch.setattr(runner, "artifact_identity", fake_identity)
    inputs = {
        name: runner._file_lock(getattr(args, name))
        for name in ("cohort", "retrieval_contexts", "musique_plans", "hotpot_plans")
    }
    code = {
        name: runner._file_lock(path)
        for name, path in runner.v5_freeze.DEFAULT_CODE.items()
    }
    code["preregistration_freezer"] = runner._file_lock(code_file)
    models = {
        name: fake_identity(tmp_path / name)
        for name in ("cross_encoder", "strong_sft", "base_model")
    }
    protocol = {
        "schema_version": runner.v5_freeze.SCHEMA_VERSION,
        "status": runner.v5_freeze.STATUS,
        "scope": runner.v5_freeze.SCOPE,
        "experiment_ids": dict(runner.v5_freeze.EXPERIMENT_IDS),
        "inputs": inputs,
        "code": code,
        "models": models,
        "settings": runner._runtime_settings(args),
    }
    args.preregistration.write_text(json.dumps(protocol), encoding="utf-8")
    observed, runtime = runner._validate_preregistration_runtime(args)
    assert observed == protocol
    assert runtime["inputs"] == inputs
    assert runtime["code"] == code

    args.rrf_candidate_k = 99
    with pytest.raises(ValueError, match="settings differ"):
        runner._validate_preregistration_runtime(args)
