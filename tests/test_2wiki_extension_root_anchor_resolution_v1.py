from __future__ import annotations

import hashlib
import json

import pytest

from kgproweight.kg.entity_linker import LinkCandidate, LinkResult
from scripts.prepare.freeze_2wiki_extension_root_anchor_resolution_v1 import (
    WORKLIST_FIELDS,
    build_worklist,
)
from scripts.prepare.materialize_2wiki_root_anchor_resolution_v1 import (
    RateLimiter,
    abstain_cross_context_cache_conflicts,
    resolve_request,
    write_canonical_resolution_caches,
)


def _qhash(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()


def _identity(qid: str, question: str) -> dict:
    return {
        "dataset": "2wikimultihopqa",
        "qid": qid,
        "question_key": f"2wikimultihopqa::{qid}",
        "question": question,
        "question_sha256": _qhash(question),
    }


def _cohort(qid: str, question: str, qtype: str = "inference") -> dict:
    return {**_identity(qid, question), "question_type": qtype}


def _plan(qid: str, question: str, anchors: list[str]) -> dict:
    return {
        **_identity(qid, question),
        "predicted_target": {"anchors": anchors, "steps": []},
        "gold_access": False,
    }


def _runtime(
    qid: str,
    question: str,
    anchors: list[str],
    qids: list[str | None],
) -> dict:
    entities = {}
    for surface, resolved in zip(anchors, qids):
        entities[surface] = {
            "qid": resolved,
            "abstained": resolved is None,
            "abstain_reason": "no candidates, no cache hit" if resolved is None else "",
        }
    return {
        **_identity(qid, question),
        "query_plan": {"anchors": anchors},
        "provenance": {"gold_access": False},
        "execution": {"anchor_entities": entities},
        "runtime_error": None,
    }


def test_worklist_projects_only_unresolved_root_and_question_context():
    question = "Was Alpha released before Beta (film)?"
    cohort = [_cohort("qid-1", question, "comparison")]
    plans = [_plan("qid-1", question, ["Alpha", "Beta"])]
    runtime = [_runtime("qid-1", question, ["Alpha", "Beta"], ["Q1", None])]

    rows, counts = build_worklist(
        cohort_rows=cohort, plan_rows=plans, runtime_rows=runtime
    )

    assert len(rows) == 1
    assert set(rows[0]) == WORKLIST_FIELDS
    assert rows[0]["root_anchor_surface"] == "Beta"
    assert rows[0]["completed_root_anchor_surface"] == "Beta (film)"
    assert rows[0]["question"] == question
    assert "Q1" not in json.dumps(rows[0])
    assert counts["resolved_anchor_occurrences"] == 1
    assert counts["unresolved_anchor_occurrences"] == 1
    assert counts["partially_resolved_questions"] == 1
    assert counts["all_roots_resolved_questions"] == 0


def test_worklist_preserves_same_surface_in_different_question_contexts():
    q1 = "Where was Mercury born?"
    q2 = "When was Mercury released?"
    rows, counts = build_worklist(
        cohort_rows=[_cohort("a", q1), _cohort("b", q2)],
        plan_rows=[_plan("a", q1, ["Mercury"]), _plan("b", q2, ["Mercury"])],
        runtime_rows=[
            _runtime("a", q1, ["Mercury"], [None]),
            _runtime("b", q2, ["Mercury"], [None]),
        ],
    )
    assert len(rows) == 2
    assert len({row["request_id"] for row in rows}) == 2
    assert counts["unique_unresolved_anchor_surfaces"] == 1


def test_worklist_rejects_runtime_anchor_drift():
    question = "Who founded Alpha?"
    with pytest.raises(ValueError, match="planner/runtime root-anchor mismatch"):
        build_worklist(
            cohort_rows=[_cohort("a", question)],
            plan_rows=[_plan("a", question, ["Alpha"])],
            runtime_rows=[_runtime("a", question, ["Beta"], [None])],
        )


class _Title:
    def __init__(self, result: LinkResult):
        self.result = result

    def resolve(self, surface: str) -> LinkResult:
        return self.result


class _Candidates:
    def __init__(self, candidates: list[LinkCandidate]):
        self.candidates = candidates
        self.search_calls = 0

    def _search_candidates(self, mention: str) -> list[LinkCandidate]:
        self.search_calls += 1
        return self.candidates

    def _score_candidates(self, mention, candidates, question, **kwargs):
        return sorted(candidates, key=lambda value: value.score, reverse=True)


def _request() -> dict:
    question = "Where is Alpha located?"
    return {
        "request_id": "r1",
        "question_key": "2wikimultihopqa::a",
        "dataset": "2wikimultihopqa",
        "qid": "a",
        "question": question,
        "question_sha256": _qhash(question),
        "root_anchor_surface": "Alpha",
        "completed_root_anchor_surface": "Alpha",
        "gold_access": False,
    }


def test_exact_title_resolution_short_circuits_context_search():
    candidate_linker = _Candidates([])
    result = resolve_request(
        _request(),
        title_resolver=_Title(
            LinkResult(
                mention="Alpha",
                selected_qid="Q123",
                selected_label="Alpha",
                score=1.0,
                margin=1.0,
            )
        ),
        candidate_linker=candidate_linker,
        limiter=RateLimiter(0),
        fallback_min_score=0.25,
        fallback_min_margin=0.10,
    )
    assert result["outcome"] == "positive"
    assert result["resolution_method"] == "exact_wikipedia_title"
    assert candidate_linker.search_calls == 0


def test_context_fallback_requires_both_score_and_margin():
    title = _Title(
        LinkResult(mention="Alpha", abstained=True, abstain_reason="not exact")
    )
    accepted = resolve_request(
        _request(),
        title_resolver=title,
        candidate_linker=_Candidates(
            [
                LinkCandidate("Q10", "Alpha", score=0.40),
                LinkCandidate("Q11", "Alpha other", score=0.20),
            ]
        ),
        limiter=RateLimiter(0),
        fallback_min_score=0.25,
        fallback_min_margin=0.10,
    )
    tied = resolve_request(
        _request(),
        title_resolver=title,
        candidate_linker=_Candidates(
            [
                LinkCandidate("Q10", "Alpha", score=0.40),
                LinkCandidate("Q11", "Alpha other", score=0.35),
            ]
        ),
        limiter=RateLimiter(0),
        fallback_min_score=0.25,
        fallback_min_margin=0.10,
    )
    assert accepted["outcome"] == "positive"
    assert accepted["resolution_method"] == "wikidata_question_context"
    assert tied["outcome"] == "abstain"
    assert tied["resolved_qid"] == ""
    assert tied["abstain_reason"] == "fallback_margin_below_0.10"


def test_resolution_caches_are_byte_deterministic_and_deduplicated(tmp_path):
    rows = [
        {
            "outcome": "positive",
            "resolution_method": "wikidata_question_context",
            "root_anchor_surface": "Beta",
            "completed_root_anchor_surface": "Beta",
            "resolved_qid": "Q2",
        },
        {
            "outcome": "positive",
            "resolution_method": "exact_wikipedia_title",
            "root_anchor_surface": "Alpha",
            "completed_root_anchor_surface": "Alpha (film)",
            "resolved_qid": "Q1",
        },
        {
            "outcome": "positive",
            "resolution_method": "wikidata_question_context",
            "root_anchor_surface": "Beta",
            "completed_root_anchor_surface": "Beta",
            "resolved_qid": "Q2",
        },
    ]
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    write_canonical_resolution_caches(
        title_cache_path=first / "title.jsonl",
        entity_cache_path=first / "entity.jsonl",
        results=rows,
    )
    write_canonical_resolution_caches(
        title_cache_path=second / "title.jsonl",
        entity_cache_path=second / "entity.jsonl",
        results=list(reversed(rows)),
    )
    assert (first / "title.jsonl").read_bytes() == (second / "title.jsonl").read_bytes()
    assert (first / "entity.jsonl").read_bytes() == (second / "entity.jsonl").read_bytes()
    assert len((first / "entity.jsonl").read_text().splitlines()) == 1


def test_cross_question_cache_qid_conflict_abstains_fail_closed():
    common = {
        "outcome": "positive",
        "resolution_method": "wikidata_question_context",
        "root_anchor_surface": "Mercury",
        "completed_root_anchor_surface": "Mercury",
        "resolved_label": "Mercury",
    }
    rows, conflicts = abstain_cross_context_cache_conflicts(
        [{**common, "resolved_qid": "Q1"}, {**common, "resolved_qid": "Q2"}]
    )
    assert conflicts == 1
    assert all(row["outcome"] == "abstain" for row in rows)
    assert all(row["resolved_qid"] == "" for row in rows)
    assert all(
        row["abstain_reason"] == "cross_context_cache_key_qid_conflict"
        for row in rows
    )
