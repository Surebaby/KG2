"""CPU-only state-machine tests for the approved v8 two-call runner."""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from kgproweight.retrieval.dynamic_decomposition_v8 import (
    NO_RELEVANT_ANSWER,
    NO_VERIFIED_SUBANSWER,
    build_dynamic_q2_state,
    parse_and_bind_subanswer,
)
import scripts.pilot.materialize_dynamic_decomposition_v8 as runner


QUESTION = "Where was the director of Film Alpha born?"
Q1 = "Who directed Film Alpha?"
Q2_BLIND = "In which city was Film Alpha's director born?"
Q2_DYNAMIC = "In which city was Jane Smith born?"
SECRET = "FORBIDDEN-GOLD-SECRET-91f8"


def _documents(prefix: str, *, answer_doc: bool = False):
    rows = []
    for rank in range(1, 11):
        body = f"Ordinary evidence for {prefix} document {rank}."
        if answer_doc and rank == 3:
            body = "Film Alpha was directed by Jane Smith in 1998."
        rows.append(
            {
                "id": f"{prefix}-{rank}",
                "title": f"{prefix} title {rank}",
                "contents": f"{prefix} title {rank}\n{body}",
                "rerank_score": 1.0 - rank / 100,
                # Deliberately poison non-allowlisted retriever metadata.  It
                # must never enter a model prompt or materialized row.
                "gold_answer": SECRET,
                "supporting_facts": [SECRET],
            }
        )
    return rows


class FakeController:
    def __init__(self, *, dynamic_response: str = Q2_DYNAMIC):
        self.dynamic_response = dynamic_response
        self.messages = []

    def __call__(self, messages):
        self.messages.append(deepcopy(messages))
        payload = json.loads(messages[1]["content"])
        if payload["task"] == "generate_q1":
            return Q1
        state = payload["state"]
        if state["verified_subanswer"] == NO_VERIFIED_SUBANSWER:
            return Q2_BLIND
        assert state["verified_subanswer"] == "Jane Smith"
        return self.dynamic_response


class FakeReader:
    def __init__(self, response: str):
        self.response = response
        self.messages = []

    def __call__(self, messages):
        self.messages.append(deepcopy(messages))
        return self.response


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def batch_search(self, queries):
        assert len(queries) == 1
        query = queries[0]
        self.queries.append(query)
        if query == QUESTION:
            return [_documents("root")]
        if query == Q1:
            return [_documents("q1", answer_doc=True)]
        if query == Q2_BLIND:
            return [_documents("blind")]
        if query == Q2_DYNAMIC:
            return [_documents("dynamic")]
        raise AssertionError(f"unexpected fake retrieval query: {query!r}")


def _run_one(*, reader_response="Jane Smith", dynamic_response=Q2_DYNAMIC):
    controller_backend = FakeController(dynamic_response=dynamic_response)
    reader_backend = FakeReader(reader_response)
    retrieval_backend = FakeRetriever()
    controller = runner._CachedTextInvoker(controller_backend, label="controller")
    reader = runner._CachedTextInvoker(reader_backend, label="subanswer_reader")
    retriever = runner._CachedRetriever(retrieval_backend)
    output = runner._run_identity_row(
        {"dataset": "hotpotqa", "qid": "dev-one", "question": QUESTION},
        controller=controller,
        subanswer_reader=reader,
        retriever=retriever,
    )
    return output, controller_backend, reader_backend, retrieval_backend


def _payload(messages):
    return json.loads(messages[1]["content"])


def test_eligible_runner_uses_blind_B_and_bound_C_under_exact_two_call_budget():
    output, controller, reader, retriever = _run_one()
    arms = output["arms"]

    assert output["budget"]["logical_by_arm"] == {
        runner.ARM_B: {
            "controller_calls": 2,
            "subanswer_reader_calls": 1,
            "retrieval_calls": 3,
        },
        runner.ARM_C: {
            "controller_calls": 2,
            "subanswer_reader_calls": 1,
            "retrieval_calls": 3,
        },
    }
    assert output["budget"]["physical_for_row"] == {
        "controller_calls": 3,
        "subanswer_reader_calls": 1,
        "retrieval_calls": 4,
    }
    assert output["budget"]["joint_cache_accounting"] == {
        "controller": {"logical_requests": 4, "cache_hits": 1, "cache_misses": 3},
        "subanswer_reader": {
            "logical_requests": 2,
            "cache_hits": 1,
            "cache_misses": 1,
        },
        "retrieval": {"logical_requests": 6, "cache_hits": 2, "cache_misses": 4},
    }
    assert len(controller.messages) == 3
    blind_state = _payload(controller.messages[1])["state"]
    dynamic_state = _payload(controller.messages[2])["state"]
    assert blind_state == {
        "gold_access": False,
        "mode": "q2_no_verified_subanswer",
        "original_question": QUESTION,
        "q1_query": Q1,
        "state_version": blind_state["state_version"],
        "verified_subanswer": NO_VERIFIED_SUBANSWER,
    }
    assert set(blind_state) == {
        "state_version",
        "mode",
        "gold_access",
        "original_question",
        "q1_query",
        "verified_subanswer",
    }
    assert dynamic_state["verified_subanswer"] == "Jane Smith"
    assert dynamic_state["bound_evidence"]["supporting_doc_id"] == "q1-3"
    assert arms[runner.ARM_B]["q2_action"]["selected_query"] == Q2_BLIND
    assert arms[runner.ARM_C]["q2_action"]["selected_query"] == Q2_DYNAMIC
    assert len(arms[runner.ARM_B]["final_passages"]) == 10
    assert len(arms[runner.ARM_C]["final_passages"]) == 10
    assert len(reader.messages) == 1
    assert retriever.queries == [QUESTION, Q1, Q2_BLIND, Q2_DYNAMIC]


def test_ineligible_C_reuses_B_prompt_response_query_and_passages_byte_exactly():
    output, controller, _reader, retriever = _run_one(
        reader_response=NO_RELEVANT_ANSWER
    )
    cf = output["counterfactual_identity"]
    assert cf == {
        "ineligible_c": True,
        "b_c_q2_prompt_byte_identical": True,
        "b_c_q2_response_byte_identical": True,
        "b_c_q2_query_byte_identical": True,
        "b_c_q2_top10_byte_identical": True,
        "b_c_final_passages_byte_identical": True,
    }
    assert output["arms"][runner.ARM_B]["q2_prompt_sha256"] == output["arms"][
        runner.ARM_C
    ]["q2_prompt_sha256"]
    assert output["arms"][runner.ARM_C]["q2_controller"]["cache_hit"] is True
    assert output["budget"]["physical_for_row"] == {
        "controller_calls": 2,
        "subanswer_reader_calls": 1,
        "retrieval_calls": 3,
    }
    accounting = output["budget"]["joint_cache_accounting"]
    assert accounting["controller"] == {
        "logical_requests": 4,
        "cache_hits": 2,
        "cache_misses": 2,
    }
    assert all(
        values["logical_requests"] == values["cache_hits"] + values["cache_misses"]
        for values in accounting.values()
    )
    assert len(controller.messages) == 2
    assert retriever.queries == [QUESTION, Q1, Q2_BLIND]


def test_eligible_invalid_dynamic_query_falls_to_Q_without_third_controller_call():
    output, controller, _reader, retriever = _run_one(
        dynamic_response="$hop_1 birthplace"
    )
    action = output["arms"][runner.ARM_C]["q2_action"]
    assert action["proposal_valid"] is False
    assert action["parse_error"] == "unresolved_placeholder"
    assert action["selected_query"] == QUESTION
    assert action["selection_source"] == "original_question"
    assert action["fallback_reason"] == "invalid_q2_dynamic:unresolved_placeholder"
    assert output["budget"]["logical_by_arm"][runner.ARM_C]["controller_calls"] == 2
    assert output["budget"]["physical_for_row"]["controller_calls"] == 3
    assert len(controller.messages) == 3
    # Q was already retrieved for root, so the deterministic q2 fallback uses
    # the retrieval cache; importantly, there is no extra controller proposal.
    assert retriever.queries == [QUESTION, Q1, Q2_BLIND]


def test_gold_like_retriever_metadata_is_never_consumed_or_emitted():
    output, controller, reader, _retriever = _run_one()
    serialized = json.dumps(
        {
            "output": output,
            "controller_prompts": controller.messages,
            "reader_prompts": reader.messages,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    assert SECRET not in serialized
    assert '"gold_answer"' not in serialized
    assert '"supporting_facts"' not in serialized
    assert output["gold_access"] is False


def test_q2_prompt_builder_rejects_forged_gold_or_mismatched_bound_state():
    q1_documents = _documents("q1", answer_doc=True)
    binding = parse_and_bind_subanswer(
        "Jane Smith", q1_query=Q1, q1_passages=q1_documents
    )
    state = build_dynamic_q2_state(
        original_question=QUESTION,
        q1_query=Q1,
        binding=binding,
    )
    state["bound_evidence"]["gold_answer"] = SECRET
    with pytest.raises(runner.V8RunnerError, match="exact allowlist"):
        runner.build_q2_controller_messages(state)

    state = build_dynamic_q2_state(
        original_question=QUESTION,
        q1_query=Q1,
        binding=binding,
    )
    state["verified_subanswer"] = "Somebody Else"
    with pytest.raises(runner.V8RunnerError, match="binding mismatch"):
        runner.build_q2_controller_messages(state)


def test_identity_runner_rejects_any_extra_gold_field():
    controller = runner._CachedTextInvoker(FakeController(), label="controller")
    reader = runner._CachedTextInvoker(FakeReader("Jane Smith"), label="reader")
    retriever = runner._CachedRetriever(FakeRetriever())
    with pytest.raises(runner.V8RunnerError, match="exactly dataset/qid/question"):
        runner._run_identity_row(
            {
                "dataset": "hotpotqa",
                "qid": "dev-one",
                "question": QUESTION,
                "gold_answer": "forbidden",
            },
            controller=controller,
            subanswer_reader=reader,
            retriever=retriever,
        )


def test_public_materializer_has_no_prospective_path_and_requests_development(monkeypatch):
    observed = []

    def sealed_loader(*, role):
        observed.append(role)
        raise RuntimeError("stop after role check")

    monkeypatch.setattr(runner, "load_frozen_v8_cohort", sealed_loader)
    with pytest.raises(RuntimeError, match="role check"):
        runner.materialize_frozen_development(
            controller_backend=FakeController(),
            subanswer_reader_backend=FakeReader(NO_RELEVANT_ANSWER),
            retriever_backend=FakeRetriever(),
        )
    assert observed == [runner.DEVELOPMENT_ROLE]


def test_engineering_fake_model_smoke_is_deterministic_and_json_serializable():
    first, *_ = _run_one()
    second, *_ = _run_one()
    first_bytes = json.dumps(first, ensure_ascii=False, sort_keys=True).encode("utf-8")
    second_bytes = json.dumps(second, ensure_ascii=False, sort_keys=True).encode("utf-8")
    assert first_bytes == second_bytes
