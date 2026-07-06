"""Disk-backed cache contracts for entity / subgraph storage."""

from __future__ import annotations

from kgproweight.kg.cache import EntityCache, SubgraphCache


def test_entity_cache_round_trip(tmp_path):
    path = tmp_path / "entity_cache.jsonl"
    cache = EntityCache(path)
    cache.set("Barack Obama", "Q76")
    cache.set("Michelle Obama", "Q13133")
    assert cache.get("barack obama") == "Q76"
    assert cache.get("MICHELLE OBAMA") == "Q13133"
    assert "barack obama" in cache
    assert len(cache) == 2

    # Reload from disk and check persistence.
    reloaded = EntityCache(path)
    assert len(reloaded) == 2
    assert reloaded.get("Michelle Obama") == "Q13133"


def test_entity_cache_skips_empty(tmp_path):
    path = tmp_path / "entity_cache.jsonl"
    cache = EntityCache(path)
    cache.set("", "Q1")  # empty label is ignored
    cache.set("x", "")  # empty qid is ignored
    assert len(cache) == 0


def test_subgraph_cache_round_trip(tmp_path):
    path = tmp_path / "subgraph_cache.jsonl"
    cache = SubgraphCache(path)
    triples = [("Barack Obama", "spouse", "Michelle Obama"), ("Michelle Obama", "occupation", "Lawyer")]
    cache.set("Q76_2", triples)
    assert cache.get("Q76_2") == triples
    assert "Q76_2" in cache

    # Reload from disk.
    reloaded = SubgraphCache(path)
    assert reloaded.get("Q76_2") == triples


def test_subgraph_cache_handles_malformed(tmp_path):
    path = tmp_path / "subgraph_cache.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"key": "good_1", "triples": [["a", "b", "c"]]}',
                "not-json",
                '{"key": "good_2", "triples": [["d", "e", "f"], ["short"]]}',
            ]
        )
    )
    cache = SubgraphCache(path)
    assert cache.get("good_1") == [("a", "b", "c")]
    assert cache.get("good_2") == [("d", "e", "f")]
