import copy

import pytest

from kgproweight.kg.source_adaptive_evidence_graph import (
    fuse_edges,
    make_passage_edge,
    make_record,
    make_wikidata_edge,
    passages_sha256,
    validate_record,
)


PASSAGES = [{"id": "x", "title": "Ada", "contents": "Ada\nAda was born in London."}]


def _p(index=1):
    return make_passage_edge(
        edge_index=index,
        triple=["Ada", "evidence sentence", "Ada was born in London."],
        passage_id="x",
        passage_rank=0,
        sentence_index=0,
        construction_gold_access=True,
    )


def _w(index=1, triple=None):
    return make_wikidata_edge(
        edge_index=index,
        triple=triple or ["Ada Lovelace", "place of birth", "London"],
        hop_index=index,
        input_qids=["Q7259"],
        pid="P19",
        tail_qid="Q84",
        cutoff="2020-12-09T23:59:59Z",
        builder_version="test",
    )


def test_passage_hash_ignores_nonsemantic_id_but_preserves_order():
    other = [{"id": "different", "title": "Ada", "contents": "Ada\nAda was born in London."}]
    assert passages_sha256(PASSAGES) == passages_sha256(other)


def test_fusion_is_wikidata_first_capped_and_deterministic():
    edges, telemetry = fuse_edges(
        [_w(1), _w(2, ["London", "country", "United Kingdom"])],
        [_p()],
        cap=2,
    )
    assert [edge["source_type"] for edge in edges] == ["wikidata", "wikidata"]
    assert telemetry["passage_retained"] == 0
    assert telemetry["truncated"] == 1


def test_record_hash_detects_mutation():
    record = make_record(
        dataset="2wikimultihopqa",
        qid="q1",
        question="Where was Ada born?",
        passages=PASSAGES,
        routing_mode="P_ONLY",
        edges=[_p()],
        routing={"cap": 8},
    )
    broken = copy.deepcopy(record)
    broken["edges"][0]["triple"][2] = "Paris"
    with pytest.raises(ValueError, match="sentence hash|graph hash"):
        validate_record(broken)


def test_no_graph_record_is_explicit_and_evaluation_ineligible():
    record = make_record(
        dataset="hotpotqa",
        qid="q2",
        question="Question?",
        passages=PASSAGES,
        routing_mode="N_REPLAY",
        edges=[],
        routing={"fallback_reason": "training_replay"},
    )
    assert record["edges"] == []
    assert record["construction_gold_access"] is False
    assert record["evaluation_eligible"] is False


def test_source_mode_mismatch_fails():
    with pytest.raises(ValueError, match="routing mode"):
        make_record(
            dataset="2wikimultihopqa",
            qid="q3",
            question="Question?",
            passages=PASSAGES,
            routing_mode="P_ONLY",
            edges=[_w()],
            routing={},
        )
