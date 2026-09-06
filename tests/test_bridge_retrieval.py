from kgproweight.retrieval.bridge import (
    additive_bridge_candidates,
    bridge_v2_rejection_reason,
    extract_bridge_queries,
    filter_bridge_queries_v2,
    reciprocal_rank_fuse,
)


def test_bridge_queries_are_deterministic_and_exclude_question_entities():
    question = "Who directed Film Alpha?"
    docs = [
        {
            "id": "1",
            "contents": "Film Alpha\nFilm Alpha was directed by Jane Smith. Jane Smith was born in Paris.",
        },
        {
            "id": "2",
            "contents": "Jane Smith\nJane Smith is a French director.",
        },
    ]
    first = extract_bridge_queries(question, docs, max_bridges=2)
    second = extract_bridge_queries(question, docs, max_bridges=2)
    assert first == second
    assert first[0] == "Jane Smith"
    assert all("Film Alpha" != value for value in first)


def test_equal_weight_rrf_promotes_documents_seen_by_multiple_queries():
    original = [{"id": "a"}, {"id": "b"}]
    bridge = [{"id": "c"}, {"id": "b"}]
    fused = reciprocal_rank_fuse([original, bridge], topk=3, rrf_k=60)
    assert fused[0]["id"] == "b"
    assert fused[0]["bridge_query_sources"] == 2


def test_bridge_v2_abstains_from_generic_singletons_but_keeps_named_entities():
    queries = ["The", "June", "English", "Mozart", "Bextor", "English Channel"]
    assert filter_bridge_queries_v2(queries) == ["Mozart", "Bextor", "English Channel"]
    assert bridge_v2_rejection_reason("The") == "weak_singleton"


def test_bridge_v2_rejects_repeated_extraction_fragments_without_refilling():
    queries = [
        "Tetrisphere Tetrisphere",
        "Tamra Davis Tamra Davis",
        "Kevin Tapani Kevin Ray Tapani",
        "West Side Story",
    ]
    assert filter_bridge_queries_v2(queries) == ["West Side Story"]
    assert all(
        bridge_v2_rejection_reason(query) == "repeated_fragment"
        for query in queries[:3]
    )


def test_bridge_v2_can_abstain_from_all_v1_queries():
    assert filter_bridge_queries_v2(["She", "January"]) == []


def test_additive_bridge_fusion_preserves_originals_and_adds_only_new_docs():
    original = [{"id": "a"}, {"id": "b"}]
    bridge_one = [{"id": "b"}, {"id": "c"}, {"id": "d"}]
    bridge_two = [{"id": "c"}, {"id": "e"}]
    result = additive_bridge_candidates(
        original,
        [bridge_one, bridge_two],
        max_bridge_only=2,
    )
    assert [doc["id"] for doc in result[:2]] == ["a", "b"]
    assert [doc["id"] for doc in result[2:]] == ["c", "e"]
    assert len(result) == 4


def test_additive_bridge_fusion_can_disable_additions_without_mutating_input():
    original = [{"id": "a"}]
    result = additive_bridge_candidates(original, [[{"id": "b"}]], max_bridge_only=0)
    assert result == original
    assert result is not original
