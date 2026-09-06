from scripts.prepare.audit_saeg_v1_cross_source_alignment import align_record


def _edge(edge_id, source, triple, hop=None):
    edge = {"edge_id": edge_id, "source_type": source, "triple": triple, "provenance": {}}
    if hop is not None:
        edge["provenance"]["hop_index"] = hop
    return edge


def test_alignment_prefers_tail_evidence_and_is_one_to_one():
    record = {
        "record_id": "2wikimultihopqa::q::P_W_FUSED",
        "dataset": "2wikimultihopqa",
        "qid": "q",
        "question_sha256": "h",
        "edges": [
            _edge("W1", "wikidata", ["Ada", "mother", "Anne"], 1),
            _edge("W2", "wikidata", ["Anne", "mother", "Mary"], 2),
            _edge("P1", "passage", ["Ada", "evidence sentence", "Ada's mother was Anne."]),
            _edge("P2", "passage", ["Anne", "evidence sentence", "Anne's mother was Mary."]),
        ],
    }
    result = align_record(record)
    assert result["all_hops_one_to_one_aligned"] is True
    assert [row["passage_edge_id"] for row in result["alignments"]] == ["P1", "P2"]
    assert result["sft_target_route"] == "P_W_JOINT"


def test_unmatched_hop_fails_closed_to_w_only_target():
    record = {
        "record_id": "2wikimultihopqa::q::P_W_FUSED",
        "dataset": "2wikimultihopqa",
        "qid": "q",
        "question_sha256": "h",
        "edges": [
            _edge("W1", "wikidata", ["Ada", "mother", "Anne"], 1),
            _edge("P1", "passage", ["Unrelated", "evidence sentence", "No matching entity."]),
        ],
    }
    result = align_record(record)
    assert result["all_hops_one_to_one_aligned"] is False
    assert result["sft_target_route"] == "W_ONLY_FAIL_CLOSED"
