from scripts.prepare.materialize_saeg_v1_sft_dataset import (
    p_or_n_steps,
    passage_evidence,
    validate_trajectory,
)


def _record():
    return {
        "record_id": "hotpotqa::q1::P_ONLY",
        "edges": [{
            "edge_id": "P1",
            "source_type": "passage",
            "triple": ["Ada", "evidence sentence", "Ada was born in London."],
            "construction_gold_access": True,
            "provenance": {
                "passage_id": "ctx-1",
                "passage_rank": 0,
                "sentence_index": 0,
                "sentence_sha256": "hash",
            },
        }],
    }


def test_passage_asset_becomes_evidence_object_not_kg_triple():
    evidence = passage_evidence(_record())
    assert evidence == [{
        "passage_id": "P1",
        "title": "Ada",
        "sentence": "Ada was born in London.",
        "source_passage_id": "ctx-1",
        "passage_rank": 0,
        "sentence_index": 0,
        "sentence_sha256": "hash",
        "construction_gold_access": True,
    }]


def test_p_only_rewrite_uses_passage_field_and_empty_knowledge():
    source = {
        "qid": "q1::qpeg",
        "steps": [{
            "index": 1,
            "text": (
                "Reasoning: Evidence.\n"
                "Knowledge Used: [(Ada, evidence sentence, Ada was born in London.)]\n"
                "Conclusion: London"
            ),
            "label": 1,
            "cited_triples": [["Ada", "evidence sentence", "Ada was born in London."]],
        }],
    }
    steps = p_or_n_steps(source, passage_evidence(_record()), cite_passages=True)
    assert "Knowledge Used: []" in steps[0]["text"]
    assert "Passage Used: [P1]" in steps[0]["text"]
    assert steps[0]["cited_triples"] == []
    assert steps[0]["cited_passage_ids"] == ["P1"]


def test_validator_rejects_passage_pseudo_triple_in_kg_subgraph():
    row = {
        "qid": "bad",
        "answer": "London",
        "kg_subgraph": [["Ada", "evidence sentence", "Ada was born in London."]],
        "passage_evidence": [],
        "steps": [],
        "teacher_output": "",
    }
    try:
        validate_trajectory(row)
    except ValueError as exc:
        assert "non-standard passage pseudo-triple" in str(exc)
    else:
        raise AssertionError("expected passage pseudo-triple to fail")
