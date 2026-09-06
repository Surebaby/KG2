"""Regression tests for the v6 logical-hop/query-variant merge adapter."""

from copy import deepcopy

import pytest

from kgproweight.retrieval.dependent_merge_v6 import (
    POLICY_VERSION,
    DependentMergeV6Error,
    merge_dependent_passages_v6,
    passage_score_key,
)


def _doc(doc_id: str, *, provenance=None):
    row = {"id": doc_id, "contents": f"Title {doc_id}\nbody for {doc_id}"}
    if provenance is not None:
        row["retrieval_provenance"] = provenance
    return row


def _original():
    return [_doc(f"o{index}") for index in range(1, 11)]


def _scores(*passages, default=0.0):
    return {passage_score_key(passage): default for passage in passages}


def _logical(*variants, logical_hop_id="hop_2", depth=2, dependent=True):
    return {
        "logical_hop_id": logical_hop_id,
        "dependencies": ["hop_1"] if dependent else [],
        "is_dependent": dependent,
        "dependency_depth": depth,
        "query_variants": list(variants),
    }


def _variant(variant_id, query, hint, passages):
    return {
        "query_variant_id": variant_id,
        "query": query,
        "hint": hint,
        "passages": passages,
    }


def test_each_variant_has_its_own_top2_cutoff_and_final_replacements_stay_at_two():
    original = _original()
    v1a, v1b, v1_rank3 = _doc("v1a"), _doc("v1b"), _doc("v1-rank3")
    v2a, v2b, v2_rank3 = _doc("v2a"), _doc("v2b"), _doc("v2-rank3")
    scores = _scores(*original, v1a, v1b, v1_rank3, v2a, v2b, v2_rank3)
    scores.update({
        "id:o9": 0.20,
        "id:o10": 0.10,
        "id:v1a": 0.90,
        "id:v1b": 0.80,
        "id:v1-rank3": 100.0,
        "id:v2a": 0.70,
        "id:v2b": 0.60,
        "id:v2-rank3": 100.0,
    })
    logical = _logical(
        _variant("bridge_a", "full question bridge A relation", "Bridge A", [v1a, v1b, v1_rank3]),
        _variant("bridge_b", "full question bridge B relation", "Bridge B", [v2a, v2b, v2_rank3]),
    )

    merged, telemetry = merge_dependent_passages_v6(original, [logical], scores)

    assert [row["id"] for row in merged[:8]] == [f"o{i}" for i in range(1, 9)]
    assert [row["id"] for row in merged[-2:]] == ["v1a", "v1b"]
    assert len(telemetry["selected_new"]) == 2
    assert telemetry["candidate_occurrences_considered"] == 4
    assert telemetry["unique_new_candidates"] == 4
    inventory_keys = {row["document_key"] for row in telemetry["candidate_inventory"]}
    assert "id:v1-rank3" not in inventory_keys
    assert "id:v2-rank3" not in inventory_keys
    assert telemetry["policy_version"] == POLICY_VERSION
    assert telemetry["query_variant_count"] == 2
    assert telemetry["candidates_per_query_variant"] == 2


def test_cross_variant_dedup_preserves_all_variant_provenance_and_existing_events():
    original = _original()
    upstream = {"source": "upstream", "rank": 7}
    repeated_a = _doc("repeated", provenance=[upstream])
    repeated_b = _doc("repeated")
    other = _doc("other")
    scores = _scores(*original, repeated_a, other)
    scores.update({"id:o9": 0.20, "id:o10": 0.10, "id:repeated": 0.90, "id:other": 0.80})
    logical = _logical(
        _variant("bridge_a", "Question plus bridge A", {"surface": "A"}, [repeated_a]),
        _variant("bridge_b", "Question plus bridge B", {"surface": "B"}, [repeated_b, other]),
    )

    merged, telemetry = merge_dependent_passages_v6(original, [logical], scores)

    repeated = next(row for row in merged if row["id"] == "repeated")
    provenance = repeated["retrieval_provenance"]
    assert upstream in provenance
    v6_events = [row for row in provenance if row.get("logical_hop_id") == "hop_2"]
    assert [row["query_variant_id"] for row in v6_events] == ["bridge_a", "bridge_b"]
    assert [row["query"] for row in v6_events] == [
        "Question plus bridge A", "Question plus bridge B"
    ]
    assert [row["hint"] for row in v6_events] == [
        {"surface": "A"}, {"surface": "B"}
    ]
    assert all(row["hop_id"] == "hop_2" for row in v6_events)
    selected = next(
        row for row in telemetry["selected_new"]
        if row["document_key"] == "id:repeated"
    )
    selected_variant_ids = {
        row["query_variant_id"]
        for row in selected["provenance"]
        if row.get("query_variant_id")
    }
    assert selected_variant_ids == {
        "bridge_a", "bridge_b"
    }
    assert telemetry["duplicates_across_dependent_hops"] == 1
    assert len({passage_score_key(row) for row in merged}) == 10


def test_full_question_tie_keeps_arm_a_and_inputs_are_not_mutated():
    original = _original()
    candidate = _doc("candidate")
    logical = _logical(
        _variant("bridge", "full question candidate relation", "candidate", [candidate])
    )
    scores = _scores(*original, candidate)
    scores.update({"id:o9": 0.40, "id:o10": 0.20, "id:candidate": 0.20})
    before_original = deepcopy(original)
    before_logical = deepcopy(logical)

    merged, telemetry = merge_dependent_passages_v6(original, [logical], scores)

    assert merged == before_original
    assert original == before_original
    assert logical == before_logical
    assert telemetry["fallback_exact"] is True
    assert telemetry["rejected_not_strictly_better"][0]["reason"] == "score_tie_original_wins"
    rejected = telemetry["rejected_not_strictly_better"][0]
    assert rejected["logical_hop_id"] == "hop_2"
    assert rejected["query_variant_id"] == "bridge"
    assert rejected["hint"] == "candidate"


def test_root_query_variants_cannot_enter_the_final_context():
    original = _original()
    root = _doc("root")
    logical = _logical(
        _variant("root_query", "original question", None, [root]),
        logical_hop_id="hop_1",
        depth=1,
        dependent=False,
    )
    merged, telemetry = merge_dependent_passages_v6(original, [logical], {})
    assert merged == original
    assert telemetry["fallback_reason"] == "no_dependent_hop"
    assert telemetry["logical_hop_count"] == 1
    assert telemetry["dependent_logical_hop_count"] == 0


def test_duplicate_with_arm_a_keeps_variant_provenance_in_telemetry():
    original = _original()
    logical = _logical(
        _variant("bridge", "full question duplicate", {"surface": "dup"}, [deepcopy(original[0])])
    )
    scores = _scores(*original)
    merged, telemetry = merge_dependent_passages_v6(original, [logical], scores)
    assert merged == original
    observation = telemetry["duplicate_original_observations"][0]["provenance"]
    assert observation["logical_hop_id"] == "hop_2"
    assert observation["query_variant_id"] == "bridge"
    assert observation["query"] == "full question duplicate"
    assert observation["hint"] == {"surface": "dup"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"candidates_per_query_variant": 3}, "must be 1 or 2"),
        ({"protected_originals": 7}, "at most two final passage replacements"),
    ],
)
def test_frozen_caps_fail_closed(kwargs, message):
    with pytest.raises(DependentMergeV6Error, match=message):
        merge_dependent_passages_v6(_original(), [], {}, **kwargs)


def test_duplicate_variant_ids_within_a_logical_hop_fail_closed():
    logical = _logical(
        _variant("same", "q1", None, []),
        _variant("same", "q2", None, []),
    )
    with pytest.raises(DependentMergeV6Error, match="duplicate query_variant_id"):
        merge_dependent_passages_v6(_original(), [logical], {})
