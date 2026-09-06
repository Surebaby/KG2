from scripts.pilot.build_ai_plan_proof_kg_pilot import execute_ai_plan


class FakeRetriever:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def fetch_edges(self, qid, pids):
        self.calls.append((qid, tuple(pids)))
        return list(self.values.get((qid, pids[0]), []))


def _edge(head_qid, head, pid, relation, tail, tail_qid=None):
    return {
        "head_qid": head_qid,
        "head_label": head,
        "pid": pid,
        "relation": relation,
        "tail_qid": tail_qid,
        "tail_value": tail,
    }


def _decision(anchors, steps):
    return {
        "row_id": "T-001",
        "anchors": anchors,
        "steps": steps,
        "operation": "compose_relation",
        "should_abstain": "NO",
    }


def test_execute_ai_plan_propagates_tail_qid_to_second_hop():
    decision = _decision(
        [{"surface": "Ada", "title": "Ada", "qid": "Q1"}],
        [
            {"subject_ref": "anchor_1", "pid": "P26", "output_slot": "hop_1", "dependencies": []},
            {"subject_ref": "$hop_1", "pid": "P27", "output_slot": "hop_2", "dependencies": ["hop_1"]},
        ],
    )
    retriever = FakeRetriever(
        {
            ("Q1", "P26"): [_edge("Q1", "Ada", "P26", "spouse", "Alex", "Q2")],
            ("Q2", "P27"): [_edge("Q2", "Alex", "P27", "country of citizenship", "United States", "Q30")],
        }
    )

    triples, execution = execute_ai_plan(decision, retriever)

    assert retriever.calls == [("Q1", ("P26",)), ("Q2", ("P27",))]
    assert triples[-1] == ("Alex", "country of citizenship", "United States")
    assert execution["complete_plan_execution"] is True


def test_execute_ai_plan_keeps_literal_final_value():
    decision = _decision(
        [{"surface": "Child", "title": "Child", "qid": "Q10"}],
        [
            {"subject_ref": "anchor_1", "pid": "P22", "output_slot": "hop_1", "dependencies": []},
            {"subject_ref": "$hop_1", "pid": "P570", "output_slot": "hop_2", "dependencies": ["hop_1"]},
        ],
    )
    retriever = FakeRetriever(
        {
            ("Q10", "P22"): [_edge("Q10", "Child", "P22", "father", "Father", "Q11")],
            ("Q11", "P570"): [_edge("Q11", "Father", "P570", "date of death", "1955-01-02")],
        }
    )

    triples, execution = execute_ai_plan(decision, retriever)

    assert triples[-1][2] == "1955-01-02"
    assert execution["hops"][-1]["output_entities"] == []
    assert execution["complete_plan_execution"] is True


def test_execute_ai_plan_marks_missing_intermediate_incomplete():
    decision = _decision(
        [{"surface": "Child", "title": "Child", "qid": "Q10"}],
        [
            {"subject_ref": "anchor_1", "pid": "P22", "output_slot": "hop_1", "dependencies": []},
            {"subject_ref": "$hop_1", "pid": "P570", "output_slot": "hop_2", "dependencies": ["hop_1"]},
        ],
    )
    retriever = FakeRetriever({})

    triples, execution = execute_ai_plan(decision, retriever)

    assert triples == []
    assert retriever.calls == [("Q10", ("P22",))]
    assert execution["complete_plan_execution"] is False


def test_execute_ai_plan_keeps_comparison_branches_separate():
    decision = _decision(
        [
            {"surface": "Film A", "title": "Film A", "qid": "Q101"},
            {"surface": "Film B", "title": "Film B", "qid": "Q102"},
        ],
        [
            {"subject_ref": "anchor_1", "pid": "P162", "output_slot": "hop_1", "dependencies": []},
            {"subject_ref": "anchor_2", "pid": "P162", "output_slot": "hop_2", "dependencies": []},
            {"subject_ref": "$hop_1", "pid": "P27", "output_slot": "hop_3", "dependencies": ["hop_1"]},
            {"subject_ref": "$hop_2", "pid": "P27", "output_slot": "hop_4", "dependencies": ["hop_2"]},
        ],
    )
    retriever = FakeRetriever(
        {
            ("Q101", "P162"): [_edge("Q101", "Film A", "P162", "producer", "Producer A", "Q201")],
            ("Q102", "P162"): [_edge("Q102", "Film B", "P162", "producer", "Producer B", "Q202")],
            ("Q201", "P27"): [_edge("Q201", "Producer A", "P27", "country of citizenship", "Canada", "Q16")],
            ("Q202", "P27"): [_edge("Q202", "Producer B", "P27", "country of citizenship", "France", "Q142")],
        }
    )

    triples, execution = execute_ai_plan(decision, retriever)

    assert retriever.calls == [
        ("Q101", ("P162",)), ("Q102", ("P162",)), ("Q201", ("P27",)), ("Q202", ("P27",))
    ]
    assert {triple[2] for triple in triples[-2:]} == {"Canada", "France"}
    assert execution["complete_plan_execution"] is True
