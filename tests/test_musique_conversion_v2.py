"""MuSiQue executor/conversion v2 (frozen deterministic NL template dictionary)."""

from __future__ import annotations

from scripts.pilot.build_automatic_proofkg_from_plans import convert_predicted_target


def _steps(*templates):
    return {"steps": [
        {"subquery_template": t, "dependencies": d, "output_slot": f"step_{i}"}
        for i, (t, d) in enumerate(templates, start=1)
    ]}


def test_nl_founded_by_converts():
    plan, diagnostic = convert_predicted_target(
        "musique", "q", _steps(("who founded the mormon church", []))
    )
    assert plan.recognized
    assert plan.anchors == ["the mormon church"]
    assert list(plan.hops[0].pids) == ["P112"]


def test_nl_mother_of_with_dependency():
    predicted = {"steps": [
        {"subquery_template": "Some Entity >> occupation", "dependencies": [], "output_slot": "step_1"},
        {"subquery_template": "Who is the mother of #1?", "dependencies": ["step_1"], "output_slot": "step_2"},
    ]}
    plan, diagnostic = convert_predicted_target("musique", "q", predicted)
    assert plan.recognized
    assert any(h.subject == "$step_1" and list(h.pids) == ["P25"] for h in plan.hops)


def test_nl_played_in_converts():
    plan, diagnostic = convert_predicted_target(
        "musique", "q", _steps(("who played copper in fox and the hound", []))
    )
    assert plan.recognized
    assert plan.anchors == ["fox and the hound"]
    assert list(plan.hops[0].pids) == ["P161"]


def test_nl_part_of_series_converts():
    plan, diagnostic = convert_predicted_target(
        "musique", "q", _steps(("What series is Steven the Sword Fighter a part of?", []))
    )
    assert plan.recognized
    assert plan.anchors == ["Steven the Sword Fighter"]
    assert list(plan.hops[0].pids) == ["P179"]


def test_nl_comparison_abstains():
    plan, diagnostic = convert_predicted_target(
        "musique", "q",
        _steps(("Who had the most Champions League wins between 1992 and 2013?", []))
    )
    assert not plan.recognized


def test_nl_count_abstains():
    plan, diagnostic = convert_predicted_target(
        "musique", "q", _steps(("how many games are in a #1 season", []))
    )
    assert not plan.recognized


def test_unknown_relation_still_abstains():
    plan, diagnostic = convert_predicted_target(
        "musique", "q", _steps(("Some Entity >> co-writer", []))
    )
    assert any(row.get("reason") == "unknown_relation_label" for row in diagnostic)
    assert not plan.recognized


def test_provenance_recorded_for_nl_conversion():
    plan, diagnostic = convert_predicted_target(
        "musique", "q", _steps(("who founded the liberal party in australia", []))
    )
    exec_row = [row for row in diagnostic if row["status"] == "EXECUTABLE"][0]
    prov = exec_row["provenance"]
    assert prov["rule"] == "nl_founded_by"
    assert prov["original_template"] == "who founded the liberal party in australia"
    assert prov["pid"] == "P112"
    assert prov["direction"] == "forward"


def test_canonical_entity_relation_unchanged():
    # the existing entity >> relation path must keep working (provenance added).
    plan, diagnostic = convert_predicted_target(
        "musique", "q", _steps(("Example College >> country", []))
    )
    assert plan.recognized
    assert list(plan.hops[0].pids) == ["P17"]
