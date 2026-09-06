import pytest

from scripts.prepare.materialize_saeg_v1_evaluation_inputs import (
    passage_items,
    standard_wikidata_triples,
)


def test_qpeg_edge_is_serialized_as_passage_evidence_not_triple():
    graph = {"question_key": "d::q", "edges": [{
        "head_surface": "Ada",
        "relation_surface": "evidence sentence",
        "tail_surface": "Ada wrote notes.",
        "passage_id": "7",
        "passage_rank": 0,
        "sentence_index": 1,
        "sentence_sha256": "x",
        "relevance_score": 0.9,
    }]}
    assert passage_items(graph)[0]["passage_id"] == "P1"
    assert "triple" not in passage_items(graph)[0]


def test_wikidata_branch_accepts_only_standard_triples():
    assert standard_wikidata_triples({"kg_subgraph": [["Ed Wood", "occupation", "director"]]}) == [
        ["Ed Wood", "occupation", "director"]
    ]
    with pytest.raises(ValueError, match="pseudo-triple"):
        standard_wikidata_triples({"kg_subgraph": [["Ed Wood", "evidence sentence", "text"]]})
