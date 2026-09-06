"""Regression tests for the isolated precision-first v5 merge policy."""

from copy import deepcopy

import pytest

from kgproweight.retrieval.dependent_merge_v5 import (
    DependentMergeV5Error,
    merge_dependent_passages_v5,
    passage_score_key,
)


def _doc(doc_id: str):
    return {"id": doc_id, "contents": f"Title {doc_id}\nbody for {doc_id}"}


def _original():
    return [_doc(f"o{index}") for index in range(1, 11)]


def _scores(*passages, default=0.0):
    return {passage_score_key(passage): default for passage in passages}


def _dependent(hop_id, passages, *, depth=2, query="Bridge relation"):
    return {
        "hop_id": hop_id,
        "query": query,
        "dependencies": ["hop_1"],
        "dependency_depth": depth,
        "passages": passages,
    }


def test_no_dependent_hop_returns_arm_a_exact_without_requiring_scores():
    original = _original()
    before = deepcopy(original)
    merged, telemetry = merge_dependent_passages_v5(
        original,
        [{"hop_id": "hop_1", "dependencies": [], "passages": [_doc("root")]}],
        {},
    )
    assert merged == before
    assert original == before
    assert merged is not original
    assert telemetry["fallback_exact"] is True
    assert telemetry["fallback_reason"] == "no_dependent_hop"
    assert telemetry["selected_new"] == []


def test_high_score_replaces_only_weakest_unprotected_original():
    original = _original()
    candidate = _doc("new-high")
    scores = _scores(*original, candidate)
    scores[passage_score_key(original[8])] = 0.40
    scores[passage_score_key(original[9])] = 0.20
    scores[passage_score_key(candidate)] = 0.30
    merged, telemetry = merge_dependent_passages_v5(
        original, [_dependent("hop_2", [candidate])], scores
    )
    assert [row["id"] for row in merged] == [
        "o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9", "new-high"
    ]
    assert telemetry["changed"] is True
    assert telemetry["evicted_originals"][0]["original_rank"] == 10
    assert telemetry["selected_new"][0]["hop_id"] == "hop_2"
    assert telemetry["selected_new"][0]["provenance"][0]["source"] == "dependent_query"


@pytest.mark.parametrize(
    ("candidate_score", "reason"),
    [(0.19, "candidate_score_lower"), (0.20, "score_tie_original_wins")],
)
def test_low_or_equal_score_cannot_displace_original(candidate_score, reason):
    original = _original()
    candidate = _doc("new")
    scores = _scores(*original, candidate)
    scores[passage_score_key(original[8])] = 0.40
    scores[passage_score_key(original[9])] = 0.20
    scores[passage_score_key(candidate)] = candidate_score
    merged, telemetry = merge_dependent_passages_v5(
        original, [_dependent("hop_2", [candidate])], scores
    )
    assert merged == original
    assert telemetry["fallback_exact"] is True
    assert telemetry["rejected_not_strictly_better"][0]["reason"] == reason


def test_root_is_excluded_and_deeper_dependent_hop_has_priority():
    original = _original()
    root = _doc("root-high")
    shallow = _doc("shallow")
    deep = _doc("deep")
    scores = _scores(*original, root, shallow, deep)
    scores[passage_score_key(original[8])] = 0.10
    scores[passage_score_key(original[9])] = 0.05
    scores[passage_score_key(root)] = 100.0
    scores[passage_score_key(shallow)] = 0.80
    scores[passage_score_key(deep)] = 0.80
    hops = [
        {"hop_id": "hop_1", "dependencies": [], "passages": [root]},
        _dependent("hop_2", [shallow], depth=2),
        _dependent("hop_3", [deep], depth=3),
    ]
    merged, telemetry = merge_dependent_passages_v5(original, hops, scores)
    ids = [row["id"] for row in merged]
    assert "root-high" not in ids
    assert ids[-2:] == ["deep", "shallow"]
    assert [row["hop_id"] for row in telemetry["selected_new"]] == ["hop_3", "hop_2"]
    assert ids[:8] == [f"o{index}" for index in range(1, 9)]


def test_full_question_score_precedes_dependency_depth():
    original = _original()
    shallow_high = _doc("shallow-high")
    deep_low = _doc("deep-low")
    scores = _scores(*original, shallow_high, deep_low)
    scores[passage_score_key(original[8])] = 0.30
    scores[passage_score_key(original[9])] = 0.20
    scores[passage_score_key(shallow_high)] = 0.90
    scores[passage_score_key(deep_low)] = 0.25
    merged, telemetry = merge_dependent_passages_v5(
        original,
        [
            _dependent("hop_2", [shallow_high], depth=2),
            _dependent("hop_3", [deep_low], depth=3),
        ],
        scores,
        protected_originals=9,
    )
    assert [row["id"] for row in merged][-1] == "shallow-high"
    assert [row["document_key"] for row in telemetry["selected_new"]] == [
        "id:shallow-high"
    ]


def test_document_id_breaks_cross_hop_tie_before_hop_order():
    original = _original()
    later_alpha = _doc("alpha")
    earlier_zulu = _doc("zulu")
    scores = _scores(*original, later_alpha, earlier_zulu)
    scores[passage_score_key(original[8])] = 0.30
    scores[passage_score_key(original[9])] = 0.20
    scores[passage_score_key(later_alpha)] = 0.90
    scores[passage_score_key(earlier_zulu)] = 0.90
    _, telemetry = merge_dependent_passages_v5(
        original,
        [
            _dependent("hop_2", [earlier_zulu], depth=2),
            _dependent("hop_3", [later_alpha], depth=2),
        ],
        scores,
        protected_originals=9,
    )
    assert [row["document_key"] for row in telemetry["selected_new"]] == [
        "id:alpha"
    ]


def test_top2_is_a_hard_candidate_cutoff_and_does_not_scan_to_fill():
    original = _original()
    duplicate = deepcopy(original[0])
    low = _doc("low")
    rank3_high = _doc("rank3-high")
    scores = _scores(*original, low, rank3_high)
    scores[passage_score_key(original[8])] = 0.60
    scores[passage_score_key(original[9])] = 0.50
    scores[passage_score_key(low)] = 0.10
    scores[passage_score_key(rank3_high)] = 100.0
    merged, telemetry = merge_dependent_passages_v5(
        original,
        [_dependent("hop_2", [duplicate, low, rank3_high])],
        scores,
    )
    assert merged == original
    assert telemetry["candidate_occurrences_considered"] == 2
    assert telemetry["duplicates_with_original"] == 1
    assert telemetry["duplicate_original_observations"] == [{
        "document_key": "id:o1",
        "original_ranks": [1],
        "provenance": {
            "source": "dependent_query",
            "hop_id": "hop_2",
            "dependency_depth": 2,
            "rank": 1,
            "query": "Bridge relation",
            "query_sha256": "77c9723bd44f59a94c9604438515e43052c1e552c30546277a37a033961937b4",
        },
    }]
    assert "id:rank3-high" not in telemetry["output_document_keys"]


def test_candidate_dedup_aggregates_provenance_and_output_is_deterministic():
    original = _original()
    repeated = _doc("repeated")
    other = _doc("other")
    scores = _scores(*original, repeated, other)
    scores[passage_score_key(original[8])] = 0.20
    scores[passage_score_key(original[9])] = 0.10
    scores[passage_score_key(repeated)] = 0.90
    scores[passage_score_key(other)] = 0.80
    hops = [
        _dependent("hop_2", [repeated, other], depth=2, query="q2"),
        _dependent("hop_3", [repeated], depth=3, query="q3"),
    ]
    first = merge_dependent_passages_v5(original, hops, scores)
    second = merge_dependent_passages_v5(original, hops, scores)
    assert first == second
    merged, telemetry = first
    assert len(merged) == 10
    assert len({passage_score_key(row) for row in merged}) == 10
    assert telemetry["duplicates_across_dependent_hops"] == 1
    selected = next(row for row in telemetry["selected_new"] if row["document_key"] == "id:repeated")
    assert selected["dependency_depth"] == 3
    assert [event["hop_id"] for event in selected["provenance"]] == ["hop_2", "hop_3"]
    assert {row["decision"] for row in telemetry["candidate_inventory"]} == {"selected"}


def test_missing_score_fails_closed_without_mutating_inputs():
    original = _original()
    candidate = _doc("new")
    before = deepcopy(original)
    scores = _scores(*original)
    with pytest.raises(DependentMergeV5Error, match="missing deterministic score"):
        merge_dependent_passages_v5(
            original, [_dependent("hop_2", [candidate])], scores
        )
    assert original == before
