"""CPU-only tests for the Hotpot Controller retrieval/Reader pilot."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import pytest

from kgproweight.kg.question_kg import question_sha256
from scripts.pilot import run_hotpot_controller_retrieval_reader_pilot_v1 as runner


def _runtime_row(index: int = 0) -> dict:
    question = f"Where was the organization linked to Root {index} founded?"
    q1 = f"Which organization is Root {index} linked to?"
    q2 = "Where was #1 founded?"
    proposal = {
        "schema_version": "hotpot-controller-query-proposal-v1",
        "q1": q1,
        "q2_template": q2,
    }
    return {
        "schema_version": runner.INPUT_SCHEMA_VERSION,
        "dataset": "hotpotqa",
        "qid": f"qid-{index}",
        "question": question,
        "question_sha256": question_sha256(question),
        "q1_query": q1,
        "q1_query_sha256": runner._sha256_text(q1),
        "q2_template": q2,
        "q2_template_sha256": runner._sha256_text(q2),
        "proposal_sha256": runner._sha256_value(proposal),
        "source_projection_row_sha256": hashlib.sha256(
            f"source-{index}".encode()
        ).hexdigest(),
    }


def _doc(doc_id: str, text: str) -> dict:
    return {"id": doc_id, "title": f"Title {doc_id}", "contents": text}


def _q1_docs(index: int = 0, *, has_surface: bool = True) -> list[dict]:
    bridge = f"Bridge Predicted {index}"
    first = (
        f"Root evidence names {bridge} as the linked organization."
        if has_surface
        else "Root evidence has no matching organization surface."
    )
    return [_doc(f"q1-{index}-{rank}", first if rank == 0 else f"q1 filler {rank}") for rank in range(10)]


def _q2_docs(index: int = 0, *, overlap: int = 1) -> list[dict]:
    q1 = _q1_docs(index)
    docs = [deepcopy(q1[rank]) for rank in range(overlap)]
    docs.extend(
        _doc(f"q2-{index}-{rank}", f"q2 evidence {rank}")
        for rank in range(10 - overlap)
    )
    return docs


class FakeRetriever:
    def __init__(self, *, q1_surface_by_index: dict[int, bool] | None = None, overlap: int = 1):
        self.q1_surface_by_index = q1_surface_by_index or {}
        self.overlap = overlap
        self.batches: list[list[str]] = []

    def batch_search(self, queries):
        queries = list(queries)
        self.batches.append(queries)
        outputs = []
        for query in queries:
            if query.startswith("Which organization is Root"):
                index = int(query.split("Root ", 1)[1].split(" ", 1)[0])
                outputs.append(
                    _q1_docs(
                        index,
                        has_surface=self.q1_surface_by_index.get(index, True),
                    )
                )
            elif query.startswith("Where was Bridge Predicted"):
                index = int(query.split("Predicted ", 1)[1].split(" ", 1)[0])
                outputs.append(_q2_docs(index, overlap=self.overlap))
            else:  # pragma: no cover - makes unexpected query use conspicuous
                raise AssertionError(f"unexpected retrieval query: {query}")
        return outputs


class FakeHF:
    def __init__(self):
        self.roles: list[str] = []
        self.calls: list[list[dict[str, str]]] = []

    def bind_role(self, role: str):
        self.roles.append(role)

        def generate(messages):
            messages = deepcopy(list(messages))
            self.calls.append(messages)
            user = messages[-1]["content"]
            match = re.search(r"Root (\d+)", user)
            if match is None:
                raise AssertionError("fake could not identify prompt")
            index = int(match.group(1))
            if "Which organization" in user:
                return (
                    "[Step 1]\nReasoning: locate organization\n"
                    "Knowledge Used: []\nConclusion: found\n\n"
                    f"[Final Answer]\nBridge Predicted {index}"
                )
            return (
                "[Step 1]\nReasoning: combine evidence\n"
                "Knowledge Used: []\nConclusion: done\n\n"
                f"[Final Answer]\nFinal Predicted {index}"
            )

        return generate


def test_predicted_observation_drives_q2_and_same_reader_finishes() -> None:
    retriever = FakeRetriever(overlap=1)
    hf = FakeHF()
    result = runner.materialize_runtime([_runtime_row(0)], hf_runtime=hf, retriever_runtime=retriever)
    row = result["rows"][0]
    assert row["status"] == "complete_gold_free_runtime"
    assert retriever.batches == [
        ["Which organization is Root 0 linked to?"],
        ["Where was Bridge Predicted 0 founded?"],
    ]
    assert row["q2_query"] == "Where was Bridge Predicted 0 founded?"
    assert row["q1_reader"]["parsed"]["binding"]["verified_answer"] == "Bridge Predicted 0"
    assert row["final_reader"]["final_answer"] == "Final Predicted 0"
    assert hf.roles == ["final_reader"]
    assert len(hf.calls) == 2
    assert result["q1_reader_requests"] == result["final_reader_requests"] == 1


def test_q1_binding_failure_makes_no_q2_retrieval_or_final_call() -> None:
    retriever = FakeRetriever(q1_surface_by_index={0: False})
    hf = FakeHF()
    result = runner.materialize_runtime([_runtime_row(0)], hf_runtime=hf, retriever_runtime=retriever)
    row = result["rows"][0]
    assert row["status"] == "q1_binding_failed_no_q2_no_final"
    assert row["failure_reason"] == "answer_surface_not_found"
    assert retriever.batches == [["Which organization is Root 0 linked to?"]]
    assert row["q2_query"] is None
    assert row["q2_retrieval"] is None
    assert row["final_reader"] is None
    assert len(hf.calls) == 1
    assert result["q2_retrieval_requests"] == 0
    assert result["final_reader_requests"] == 0
    report = runner.build_report(
        result, source_fixed_denominator=1, runtime_candidate_min=1
    )
    assert report["gates"]["q1_binding_failure_has_no_q2_or_final"] is True
    assert report["gates"]["runtime_pass_candidates_min"] is False
    assert report["gates"]["final_passage_budget_and_bound_first"] is False
    assert report["all_pass"] is False


def test_merge_is_bound_first_q2_novel_then_q1_backfill() -> None:
    q1 = _q1_docs(0)
    generation = (
        "[Step 1]\nReasoning: locate\nKnowledge Used: []\nConclusion: x\n\n"
        "[Final Answer]\nBridge Predicted 0"
    )
    from kgproweight.retrieval.canonical_subqa_v9_1 import parse_and_bind_canonical_subanswer

    binding = parse_and_bind_canonical_subanswer(
        generation,
        subquestion="Which organization is Root 0 linked to?",
        retrieved_passages=q1,
    )["binding"]
    merged, telemetry = runner.merge_bound_q1_with_q2(
        q1_passages=q1,
        q2_passages=_q2_docs(0, overlap=8),
        binding=binding,
    )
    assert len({row["doc_id"] for row in merged}) == 10
    assert merged[0]["doc_id"] == "q1-0-0"
    assert telemetry["q2_novel_selected"] == 2
    assert telemetry["q1_backfill_selected"] == 7
    assert [row["doc_id"] for row in merged[1:3]] == ["q2-0-0", "q2-0-1"]


def test_runtime_gate_uses_fixed_30_denominator_and_allows_failed_rows_if_24_pass() -> None:
    rows = [_runtime_row(index) for index in range(25)]
    retriever = FakeRetriever(q1_surface_by_index={24: False})
    result = runner.materialize_runtime(rows, hf_runtime=FakeHF(), retriever_runtime=retriever)
    report = runner.build_report(
        result, source_fixed_denominator=30, runtime_candidate_min=24
    )
    assert report["runtime_pass_candidates"] == 24
    assert report["runtime_pass_candidate_rate_on_fixed_denominator"] == 0.8
    assert report["gates"]["runtime_pass_candidates_min"] is True
    assert report["all_pass"] is True


def test_input_is_exact_answer_free_projection_and_hash_bound() -> None:
    row = _runtime_row(0)
    assert runner._validate_runtime_input(row) == row
    for forbidden in ("observation", "answer", "accepted_actions"):
        mutated = {**row, forbidden: "secret"}
        with pytest.raises(runner.HotpotRuntimeError, match="field/order"):
            runner._validate_runtime_input(mutated)
    bad = deepcopy(row)
    bad["q1_query"] = "Which place is Root 0 linked to?"
    with pytest.raises(runner.HotpotRuntimeError, match="q1 hash mismatch"):
        runner._validate_runtime_input(bad)
    repeated_slot = deepcopy(row)
    repeated_slot["q2_template"] = "Did #1 and #1 share a founding place?"
    repeated_slot["q2_template_sha256"] = runner._sha256_text(
        repeated_slot["q2_template"]
    )
    repeated_slot["proposal_sha256"] = runner._sha256_value(
        {
            "schema_version": "hotpot-controller-query-proposal-v1",
            "q1": repeated_slot["q1_query"],
            "q2_template": repeated_slot["q2_template"],
        }
    )
    with pytest.raises(runner.HotpotRuntimeError, match="literal #1 exactly once"):
        runner._validate_runtime_input(repeated_slot)


def test_cli_has_nonexecuting_safety_latch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_hotpot_controller_retrieval_reader_pilot_v1.py"])
    with pytest.raises(SystemExit, match="No runtime call made"):
        runner.main()


def test_failed_manifest_error_fields_do_not_serialize_secret_or_query() -> None:
    fields = runner._safe_error_fields(
        RuntimeError("GOLD_SECRET Where was the hidden answer? /private/path")
    )
    encoded = json.dumps(fields)
    assert set(fields) == {"error_type", "error_message_sha256"}
    assert "GOLD_SECRET" not in encoded
    assert "hidden answer" not in encoded
    assert "/private/path" not in encoded
