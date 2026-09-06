"""HotpotQA zero-shot relation-graph executor (v2).

HotpotQA now routes to the relation-graph conversion (anchors + subject + pid),
with a generic syntax normalisation that strips a stray ``>>`` operator leaking
into the subject (the dev_171 failure).  MuSiQue keeps the subquery branch.
"""

from __future__ import annotations

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.training.query_planner import planner_messages
from scripts.pilot.build_automatic_proofkg_from_plans import convert_predicted_target


def _rel_steps(*steps):
    return {"anchors": [], "steps": list(steps)}


def _hop(i, subject, pid, deps=None):
    return {"step": i, "subject": subject, "relation_label": "r", "pid": pid,
            "output_slot": f"hop_{i}", "dependencies": deps or []}


def test_hotpotqa_relation_graph_two_hop_converts():
    predicted = {"anchors": ["The Big Lebowski"], "steps": [
        _hop(1, "The Big Lebowski", "P57"),
        _hop(2, "$hop_1", "P569", ["hop_1"]),
    ]}
    plan, diagnostic = convert_predicted_target("hotpotqa", "q", predicted)
    assert plan.recognized
    assert plan.operation == "execute_zero_shot_subquery_graph"
    assert plan.anchors == ["The Big Lebowski"]
    assert [list(h.pids) for h in plan.hops] == [["P57"], ["P569"]]
    assert plan.hops[0].relation_role == "bridge"


def test_hotpotqa_subject_cleans_stray_arrow():
    # dev_171 regression: a ">> relation" leaked into the subject field.
    predicted = {"anchors": ["State Senate of California"], "steps": [
        _hop(1, "State Senate of California", "P136"),
        {"step": 2, "subject": "$hop_1 >> place of birth", "relation_label": "place of birth",
         "pid": "P19", "output_slot": "hop_2", "dependencies": ["hop_1"]},
    ]}
    plan, diagnostic = convert_predicted_target("hotpotqa", "q", predicted)
    assert plan.recognized
    assert plan.hops[1].subject == "$hop_1"  # stray ">> ..." stripped


def test_hotpotqa_gold_fields_not_consumed():
    predicted = {
        "anchors": ["The Big Lebowski"],
        "steps": [_hop(1, "The Big Lebowski", "P57")],
        "answer": "Joel Coen",
        "golden_answers": ["Joel Coen"],
        "supporting_facts": [{"title": "The Big Lebowski"}],
    }
    plan, diagnostic = convert_predicted_target("hotpotqa", "q", predicted)
    assert plan.recognized
    assert plan.anchors == ["The Big Lebowski"]
    assert [list(h.pids) for h in plan.hops] == [["P57"]]


def test_hotpotqa_invalid_subject_pid_abstains():
    predicted = {"anchors": [], "steps": [_hop(1, "", "P57")]}
    plan, diagnostic = convert_predicted_target("hotpotqa", "q", predicted)
    assert not plan.recognized


def test_planner_messages_hotpot_relation_graph_hint():
    row = {"dataset": "hotpotqa", "question": "q", "target_type": "relation_graph"}
    content = planner_messages(row, include_target=False)[1]["content"]
    assert "hop_N output" in content


def test_question_key_and_hash_consistent():
    assert question_key("hotpotqa", "dev_1") == "hotpotqa::dev_1"
    q = "Who directed The Big Lebowski?"
    plan, _ = convert_predicted_target("hotpotqa", q,
        {"anchors": ["The Big Lebowski"], "steps": [_hop(1, "The Big Lebowski", "P57")]})
    assert plan.question_sha256 == question_sha256(q)
