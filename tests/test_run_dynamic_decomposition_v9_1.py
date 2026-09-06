"""Focused CPU-only tests for the v9.1 append-only runner adapters."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from kgproweight.retrieval.canonical_subqa_v9_1 import (
    PROVENANCE_BINDER_VERSION,
    bind_subanswer_rank_first,
)
from scripts.pilot import materialize_dynamic_decomposition_v8 as v8_engine
from scripts.pilot import run_dynamic_decomposition_v9_1 as runner


SUBQUESTION = "Who directed Film Alpha?"
ORIGINAL_QUESTION = "Which city did the director of Film Alpha die in?"


def _passages(prefix: str, *, answer_in: tuple[int, ...] = ()) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank in range(1, 11):
        text = f"{prefix} ordinary evidence at rank {rank}."
        if rank in answer_in:
            text = f"Film Alpha was directed by Jane Smith; evidence rank {rank}."
        rows.append(
            {
                "doc_id": f"{prefix}-{rank}",
                "title": f"{prefix} title {rank}",
                "text": text,
                "rerank_score": float(20 - rank),
            }
        )
    return rows


def _binding(q1_passages: list[dict[str, object]] | None = None) -> dict:
    passages = q1_passages or _passages("q1", answer_in=(8, 9))
    return bind_subanswer_rank_first(
        "Jane Smith",
        subquestion=SUBQUESTION,
        retrieved_passages=passages,
    )


def test_engine_patch_restores_every_hook_after_exception(monkeypatch) -> None:
    sentinel_parse = object()
    monkeypatch.setattr(v8_engine, "parse_and_bind_subanswer", sentinel_parse)
    before = {
        name: getattr(v8_engine, name)
        for name in (
            "build_subanswer_reader_messages",
            "parse_and_bind_subanswer",
            "build_dynamic_q2_state",
            "build_dynamic_q2_action",
            "merge_fixed_budget_passages",
        )
    }

    with pytest.raises(RuntimeError, match="synthetic failure"):
        with runner._patched_v91_engine():
            assert v8_engine.build_subanswer_reader_messages is runner._build_subanswer_messages_v91
            assert v8_engine.parse_and_bind_subanswer is runner._parse_and_bind_v91
            assert v8_engine.build_dynamic_q2_state is runner._build_dynamic_state_v91
            assert v8_engine.build_dynamic_q2_action is runner._build_dynamic_action_v91
            assert v8_engine.merge_fixed_budget_passages is runner._merge_v91
            raise RuntimeError("synthetic failure")

    assert {name: getattr(v8_engine, name) for name in before} == before
    assert v8_engine.parse_and_bind_subanswer is sentinel_parse


def test_canonical_subquestion_prompt_excludes_original_question() -> None:
    messages = runner._build_subanswer_messages_v91(
        original_question=ORIGINAL_QUESTION,
        q1_query=SUBQUESTION,
        q1_passages=_passages("q1", answer_in=(3,)),
    )
    rendered = json.dumps(messages, ensure_ascii=False)
    assert SUBQUESTION in rendered
    assert ORIGINAL_QUESTION not in rendered
    assert "[Final Answer]" in rendered
    assert "[Knowledge Graph Context]\\n  (empty)" in rendered


def test_v91_binding_flows_through_dynamic_state_and_action_without_mutation() -> None:
    binding = _binding()
    original = deepcopy(binding)
    state = runner._build_dynamic_state_v91(
        original_question=ORIGINAL_QUESTION,
        q1_query=SUBQUESTION,
        binding=binding,
    )
    assert state["mode"] == "q2_dynamic"
    assert state["gold_access"] is False
    assert state["verified_subanswer"] == "Jane Smith"
    assert set(state["bound_evidence"]) == v8_engine.BOUND_EVIDENCE_FIELDS
    assert state["bound_evidence"]["supporting_doc_rank"] == 8

    action = runner._build_dynamic_action_v91(
        "Where did Jane Smith die?",
        original_question=ORIGINAL_QUESTION,
        q1_query=SUBQUESTION,
        binding=binding,
    )
    assert action["dynamic_eligible"] is True
    assert action["proposal_valid"] is True
    assert action["selected_query"] == "Where did Jane Smith die?"
    assert action["selection_source"] == "q2_dynamic"
    assert binding == original
    assert binding["binder_version"] == PROVENANCE_BINDER_VERSION


def test_v91_dynamic_action_keeps_v8_fail_closed_fallback() -> None:
    action = runner._build_dynamic_action_v91(
        ORIGINAL_QUESTION,
        original_question=ORIGINAL_QUESTION,
        q1_query=SUBQUESTION,
        binding=_binding(),
    )
    assert action["proposal_valid"] is False
    assert action["selected_query"] == ORIGINAL_QUESTION
    assert action["selection_source"] == "original_question"
    assert action["used_fallback"] is True


def test_v91_merge_prioritizes_selected_rank_first_document() -> None:
    root = _passages("root")
    q1 = _passages("q1", answer_in=(8, 9))
    q2 = _passages("q2")
    binding = _binding(q1)
    original = deepcopy(binding)

    merged, telemetry = runner._merge_v91(
        root,
        q1,
        q2,
        root_query=ORIGINAL_QUESTION,
        q1_query=SUBQUESTION,
        q2_query="Where did Jane Smith die?",
        q1_binding=binding,
    )
    assert len(merged) == 10
    assert len({row["document_key"] for row in merged}) == 10
    assert telemetry["q1_binding_document_key"] == "id:q1-8"
    assert telemetry["q1_binding_prioritized_into_novel_slot"] is True
    assert telemetry["selected_by_slot"] == {
        "root_prefix": 6,
        "q1_novel": 2,
        "q2_novel": 2,
        "root_backfill": 0,
    }
    assert merged[6]["document_key"] == "id:q1-8"
    assert binding == original


def test_v91_merge_rejects_foreign_binder_version() -> None:
    binding = _binding()
    binding["binder_version"] = "forged"
    with pytest.raises(runner.V91RunnerError, match="non-v9.1 binding"):
        runner._merge_v91(
            _passages("root"),
            _passages("q1", answer_in=(8, 9)),
            _passages("q2"),
            root_query=ORIGINAL_QUESTION,
            q1_query=SUBQUESTION,
            q2_query="Where did Jane Smith die?",
            q1_binding=binding,
        )


def _report_rows() -> list[dict]:
    rows: list[dict] = []
    generation = """[Step 1]
Reasoning: The passage identifies the director.
Knowledge Used: []
Conclusion: Jane Smith directed Film Alpha.

[Final Answer]
Jane Smith"""
    for dataset in runner.DATASETS:
        for index in range(4):
            binding = runner._parse_and_bind_v91(
                generation,
                q1_query=SUBQUESTION,
                q1_passages=_passages(
                    f"report-{dataset}-{index}", answer_in=(2, 5)
                ),
            )
            rows.append(
                {
                    "gold_access": False,
                    "identity": {
                        "dataset": dataset,
                        "qid": f"{dataset}-{index}",
                        "question": f"Question {index}?",
                    },
                    "shared": {
                        "subanswer_binding": binding,
                    },
                }
            )
    return rows


@pytest.mark.parametrize("parent_gate", [True, False])
def test_report_extends_parent_report_without_erasing_parent_gates(
    monkeypatch,
    parent_gate: bool,
) -> None:
    calls: list[dict] = []

    def fake_parent_report(**kwargs):
        calls.append(kwargs)
        return {
            "schema_version": "synthetic-v8-report",
            "parent_marker": "preserve-me",
            "gate_results": {"synthetic_parent_gate": parent_gate},
            "all_pass": parent_gate,
        }

    monkeypatch.setattr(
        runner.v8_driver,
        "build_gold_free_mechanism_report",
        fake_parent_report,
    )
    result = {"rows": _report_rows()}
    report = runner.build_report(
        result,
        protocol={"gold_access": False},
        scope="smoke",
        experiment_id="SYNTHETIC-V91-SMOKE",
        rows_lock={"path": "rows.jsonl", "sha256": "a" * 64},
    )
    assert len(calls) == 1
    assert calls[0]["result"] is result
    assert calls[0]["scope"] == "smoke"
    assert report["parent_marker"] == "preserve-me"
    assert report["gate_results"]["synthetic_parent_gate"] is parent_gate
    assert report["gate_results"]["binder_version_rate_1"] is True
    assert report["all_pass"] is parent_gate
    assert report["status"] == ("PASS" if parent_gate else "FAIL_STOP_GOLD_FREE_GATES")


def test_write_helpers_are_append_only_and_preserve_first_bytes(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    rows_path = tmp_path / "rows.jsonl"
    runner._write_json_new(report_path, {"status": "FIRST"})
    runner._write_rows_new(rows_path, [{"row": 1}, {"row": 2}])
    report_bytes = report_path.read_bytes()
    rows_bytes = rows_path.read_bytes()

    with pytest.raises(FileExistsError):
        runner._write_json_new(report_path, {"status": "OVERWRITE"})
    with pytest.raises(FileExistsError, match="append-only"):
        runner._write_rows_new(rows_path, [{"row": 999}])
    assert report_path.read_bytes() == report_bytes
    assert rows_path.read_bytes() == rows_bytes
    assert rows_bytes.count(b"\n") == 2


def test_smoke_prerequisite_rehashes_report_and_binds_protocol(
    tmp_path: Path,
    monkeypatch,
) -> None:
    smoke_dir = tmp_path / "smoke"
    smoke_dir.mkdir()
    protocol_lock = {"path": "protocol.json", "sha256": "b" * 64}
    running_path = smoke_dir / "manifest.running.json"
    runner._write_json_new(
        running_path,
        {
            "schema_version": "dynamic-decomposition-v9.1-running-manifest-1",
            "experiment_id": runner.SMOKE_EXPERIMENT_ID,
            "status": "RUNNING_NEW_APPEND_ONLY_ATTEMPT",
            "protocol": protocol_lock,
            "gold_access": False,
            "prospective_opened_or_hashed": False,
        },
    )
    rows_path = smoke_dir / "rows.jsonl"
    runner._write_rows_new(rows_path, [{"gold_access": False}])
    rows_lock = runner._file_lock(rows_path)
    report_path = smoke_dir / "report.json"
    runner._write_json_new(
        report_path,
        {
            "schema_version": "dynamic-decomposition-v9.1-gold-free-report-1",
            "experiment_id": runner.SMOKE_EXPERIMENT_ID,
            "status": "PASS",
            "all_pass": True,
            "scope": "smoke",
            "gold_access": False,
            "prospective_opened_or_hashed": False,
            "rows": rows_lock,
            "preflight": {"new_protocol": protocol_lock},
        },
    )
    report_lock = runner._file_lock(report_path)
    runner._write_json_new(
        smoke_dir / "manifest.complete.json",
        {
            "schema_version": "dynamic-decomposition-v9.1-terminal-manifest-1",
            "experiment_id": runner.SMOKE_EXPERIMENT_ID,
            "status": "PASS",
            "protocol_sha256": "b" * 64,
            "gold_access": False,
            "answer_scoring_performed": False,
            "prospective_opened_or_hashed": False,
            "report": report_lock,
            "rows": rows_lock,
            "running": runner._file_lock(running_path),
        },
    )
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "SMOKE_RUN_DIR", Path("smoke"))

    runner._require_smoke_pass(protocol_lock)
    with pytest.raises(runner.V91RunnerError, match="running manifest is invalid"):
        runner._require_smoke_pass({"path": "protocol.json", "sha256": "c" * 64})

    # Mutation after manifest creation must fail the content-lock check.
    report_path.write_text('{"all_pass":false,"scope":"smoke"}\n', encoding="utf-8")
    with pytest.raises(runner.V91RunnerError, match="content drift"):
        runner._require_smoke_pass(protocol_lock)
