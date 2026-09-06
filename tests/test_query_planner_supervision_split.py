from scripts.prepare.split_query_planner_supervision import (
    assign_family,
    compute_assignments,
    family_signature,
    summarize,
)


def _record(qid, question, anchor):
    return {
        "question_key": f"2wikimultihopqa::{qid}",
        "dataset": "2wikimultihopqa",
        "qid": qid,
        "question": question,
        "target_type": "relation_graph",
        "target": {
            "anchors": [anchor],
            "steps": [{"pid": "P57", "dependencies": []}],
        },
    }


def test_family_signature_replaces_question_anchor():
    first = _record("q1", "Who directed Film A?", "Film A")
    second = _record("q2", "Who directed Film B?", "Film B")
    assert family_signature(first) == family_signature(second)


def test_same_family_never_crosses_splits_and_seen_is_removed():
    records = [
        _record("q1", "Who directed Film A?", "Film A"),
        _record("q2", "Who directed Film B?", "Film B"),
        _record("q3", "Where was Person C born?", "Person C"),
    ]
    assignments, hashes = compute_assignments(
        records,
        seen_keys={"2wikimultihopqa::q3"},
        seed=7,
        dev_bp=2000,
        confirmation_bp=2000,
    )
    assert assignments[records[0]["question_key"]] == assignments[records[1]["question_key"]]
    assert assignments[records[2]["question_key"]] == "seen_diagnostics"
    summary = summarize(records, assignments, hashes)
    assert not any(summary["family_overlap"].values())

    expanded, _ = compute_assignments(
        records,
        seen_keys={"2wikimultihopqa::q1"},
        seed=7,
        dev_bp=2000,
        confirmation_bp=2000,
    )
    assert expanded["2wikimultihopqa::q1"] == "seen_diagnostics"
    assert expanded["2wikimultihopqa::q2"] == "train"


def test_family_assignment_is_deterministic():
    assert assign_family("family", seed=42, dev_bp=200, confirmation_bp=200) == assign_family(
        "family", seed=42, dev_bp=200, confirmation_bp=200
    )


def test_musique_family_keeps_operator_but_removes_anchor():
    def record(qid, anchor):
        return {
            "question_key": f"musique::{qid}", "dataset": "musique", "qid": qid,
            "question": "q", "target_type": "subquery_graph",
            "target": {"steps": [{
                "subquery_template": f"{anchor} >> director",
                "dependencies": [],
            }]},
        }
    assert family_signature(record("m1", "Film A")) == family_signature(record("m2", "Film B"))
