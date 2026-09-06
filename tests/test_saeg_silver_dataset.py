from kgproweight.data.silver_dataset import SilverTrajectory


def test_saeg_sidecars_round_trip_without_turning_passages_into_kg():
    row = {
        "qid": "q1",
        "question": "Question?",
        "answer": "Answer",
        "dataset": "2wikimultihopqa",
        "evidence_mode": "P_W_FUSED",
        "kg_subgraph": [["A", "parent", "B"]],
        "passage_evidence": [{"passage_id": "P1", "title": "A", "sentence": "A is B's parent."}],
        "steps": [{
            "index": 1,
            "text": "Reasoning: X\nKnowledge Used: [(A, parent, B)]\nPassage Used: [P1]\nConclusion: X",
            "label": 1.0,
            "cited_triples": [["A", "parent", "B"]],
            "cited_edge_ids": ["W1"],
            "cited_passage_ids": ["P1"],
        }],
    }
    output = SilverTrajectory.from_dict(row).to_dict()
    assert output["kg_subgraph"] == [["A", "parent", "B"]]
    assert output["passage_evidence"][0]["passage_id"] == "P1"
    assert output["steps"][0]["cited_edge_ids"] == ["W1"]
    assert output["steps"][0]["cited_passage_ids"] == ["P1"]


def test_legacy_silver_without_saeg_fields_remains_loadable():
    row = {
        "qid": "legacy",
        "question": "Q",
        "answer": "A",
        "dataset": "hotpotqa",
        "steps": [],
        "kg_subgraph": [],
    }
    output = SilverTrajectory.from_dict(row).to_dict()
    assert "passage_evidence" not in output
    assert "evidence_mode" not in output
