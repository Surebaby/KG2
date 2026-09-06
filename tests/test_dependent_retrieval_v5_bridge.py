import json

from kgproweight.retrieval.dependent_v5 import (
    infer_expected_bridge_profile,
    select_bridge_candidates_v5,
)


def _doc(doc_id: str, title: str, body: str) -> dict:
    return {"id": doc_id, "title": title, "contents": f'"{title}"\n{body}'}


def _select(step, consumers, passages, *, target_type="subquery_graph", query=None):
    if query is None:
        query = str(step.get("subquery_template") or step.get("subject") or "")
    return select_bridge_candidates_v5(
        step=step,
        consumers=consumers,
        target_type=target_type,
        query=query,
        question="A synthetic Gold-free multi-hop question",
        passages=passages,
    )


def test_profile_intersects_producer_range_and_consumer_domain():
    step = {
        "subquery_template": "Big Jim McLain >> producer",
        "dependencies": [],
        "output_slot": "step_1",
    }
    consumer = {
        "subquery_template": "who did #1 play in true grit",
        "dependencies": ["step_1"],
        "output_slot": "step_2",
    }
    profile = infer_expected_bridge_profile(step, [consumer], "subquery_graph")
    assert profile["producer_types"] == ["organization", "person"]
    assert profile["consumer_types"] == ["person"]
    assert profile["expected_types"] == ["person"]
    assert profile["expected_type_source"] == "producer_consumer_intersection"
    assert not profile["profile_conflict"]


def test_relation_label_controls_type_and_pid_disagreement_is_telemetry_only():
    step = {
        "subject": "Ada",
        "relation_label": "place of birth",
        "pid": "P569",  # P569 is date, while the textual relation maps to P19.
        "dependencies": [],
        "output_slot": "hop_1",
    }
    profile = infer_expected_bridge_profile(step, [], "relation_graph")
    assert profile["expected_types"] == ["location"]
    assert profile["relation_label_expected_pid"] == "P19"
    assert profile["pid_label_conflict"]
    assert not profile["pid_used_for_type_inference"]


def test_natural_wh_character_profile_is_not_replaced_by_work_title_frequency():
    step = {
        "subquery_template": "What character did Walt Disney create in 1928?",
        "dependencies": [],
        "output_slot": "step_1",
    }
    consumer = {
        "subquery_template": "What old show was named after #1?",
        "dependencies": ["step_1"],
        "output_slot": "step_2",
    }
    passages = [
        _doc("p1", "Plane Crazy", "Plane Crazy is a 1928 animated short film created by Walt Disney."),
        _doc("p2", "Mickey Mouse", "Mickey Mouse is an American cartoon character created by Walt Disney in 1928."),
    ]
    accepted, telemetry = _select(step, [consumer], passages)
    assert [row["surface"] for row in accepted] == ["Mickey Mouse"]
    plane = next(row for row in telemetry["candidate_decisions"] if row["surface"] == "Plane Crazy")
    assert plane["decision"] == "reject"
    assert "high_confidence_type_conflict" in plane["reasons"]


def test_relation_local_person_survives_when_unrelated_organization_is_rejected():
    step = {
        "subquery_template": "The Secret World of Og >> author",
        "dependencies": [],
        "output_slot": "step_1",
    }
    consumer = {
        "subquery_template": "#1 >> educated at",
        "dependencies": ["step_1"],
        "output_slot": "step_2",
    }
    passages = [
        _doc(
            "p1",
            "The Secret World of Og",
            "The Secret World of Og is a children's novel written by Canadian author Pierre Berton.",
        ),
        _doc(
            "p2",
            "Sarasota Opera",
            "Sarasota Opera is a professional opera company in Florida. It performed The Secret World of Og.",
        ),
    ]
    accepted, telemetry = _select(step, [consumer], passages)
    assert [row["surface"] for row in accepted] == ["Pierre Berton"]
    decision = next(row for row in telemetry["candidate_decisions"] if row["surface"] == "Pierre Berton")
    assert decision["relation_local_support"]["strength"] == 2
    assert decision["admission_basis"] == "relation_local_type_unknown"
    opera = next(row for row in telemetry["candidate_decisions"] if row["surface"] == "Sarasota Opera")
    assert "high_confidence_type_conflict" in opera["reasons"]


def test_explicit_lead_alias_and_weak_month_are_rejected_but_show_is_admitted():
    step = {
        "subject": "Onika Tanya Maraj",
        "relation_label": "judge of",
        "pid": "P166",
        "dependencies": [],
        "output_slot": "hop_1",
    }
    consumer = {
        "subject": "$hop_1",
        "relation_label": "host",
        "pid": "P162",
        "dependencies": ["hop_1"],
        "output_slot": "hop_2",
    }
    passages = [
        _doc(
            "p1",
            "Nicki Minaj",
            "Onika Tanya Maraj-Petty (born December 8, 1982), known professionally as Nicki Minaj, is a rapper. She was a judge on American Idol.",
        ),
        _doc(
            "p2",
            "American Idol",
            "American Idol is an American singing television show hosted by Ryan Seacrest.",
        ),
    ]
    accepted, telemetry = _select(
        step,
        [consumer],
        passages,
        target_type="relation_graph",
        query="Onika Tanya Maraj judge of",
    )
    assert [row["surface"] for row in accepted] == ["American Idol"]
    decisions = {row["surface"]: row for row in telemetry["candidate_decisions"]}
    assert "explicit_subject_alias" in decisions["Nicki Minaj"]["reasons"]
    assert "weak_singleton" in decisions["December"]["reasons"]
    assert telemetry["raw_candidate_inspect_cap"] == 12


def test_candidate_already_present_as_complete_question_phrase_is_hard_rejected():
    step = {
        "subquery_template": "Example Work >> author",
        "dependencies": [],
        "output_slot": "step_1",
    }
    consumer = {
        "subquery_template": "#1 >> educated at",
        "dependencies": ["step_1"],
        "output_slot": "step_2",
    }
    passages = [
        _doc("p1", "Example Work", "Example Work was written by Ada Lovelace."),
        _doc("p2", "Ada Lovelace", "Ada Lovelace (born 1815) was an English writer."),
    ]
    accepted, telemetry = select_bridge_candidates_v5(
        step=step,
        consumers=[consumer],
        target_type="subquery_graph",
        query="Example Work author",
        question="Which university did Ada Lovelace attend?",
        passages=passages,
    )
    assert accepted == []
    ada = next(row for row in telemetry["candidate_decisions"] if row["surface"] == "Ada Lovelace")
    assert ada["original_question_phrase"]
    assert "original_question_phrase" in ada["reasons"]
    assert telemetry["fallback_recommended"]


def test_question_phrase_rejection_ignores_candidate_disambiguator():
    step = {
        "subquery_template": "Example Work >> author",
        "dependencies": [],
        "output_slot": "step_1",
    }
    consumer = {
        "subquery_template": "#1 >> educated at",
        "dependencies": ["step_1"],
        "output_slot": "step_2",
    }
    passages = [
        _doc(
            "p1",
            "Ada Lovelace (writer)",
            "Example Work was written by Ada Lovelace, an English writer.",
        )
    ]
    accepted, telemetry = select_bridge_candidates_v5(
        step=step,
        consumers=[consumer],
        target_type="subquery_graph",
        query="Example Work author",
        question="Which university did Ada Lovelace attend?",
        passages=passages,
    )
    assert accepted == []
    candidate = next(
        row for row in telemetry["candidate_decisions"]
        if row["surface"] == "Ada Lovelace (writer)"
    )
    assert candidate["original_question_phrase"] is True
    assert "original_question_phrase" in candidate["reasons"]


def test_explicit_alias_is_only_a_hard_reject_for_non_alias_relations():
    step = {
        "subject": "Onika Tanya Maraj",
        "relation_label": "also known as",
        "dependencies": [],
        "output_slot": "hop_1",
    }
    passages = [
        _doc(
            "p1",
            "Nicki Minaj",
            "Onika Tanya Maraj (born 1982), known professionally as Nicki Minaj, is a rapper.",
        )
    ]
    _, telemetry = select_bridge_candidates_v5(
        step=step,
        consumers=[],
        target_type="relation_graph",
        query="Onika Tanya Maraj also known as",
        question="What is the stage name of Onika Tanya Maraj?",
        passages=passages,
    )
    nicki = next(row for row in telemetry["candidate_decisions"] if row["surface"] == "Nicki Minaj")
    assert nicki["alias_echo"]
    assert nicki["alias_relation_exempt"]
    assert "explicit_subject_alias" not in nicki["reasons"]


def test_preexisting_weak_and_repeated_rules_are_reused_without_filling_quota():
    step = {
        "subject": "Example Work",
        "relation_label": "author",
        "dependencies": [],
        "output_slot": "hop_1",
    }
    consumer = {
        "subject": "$hop_1",
        "relation_label": "date of birth",
        "dependencies": ["hop_1"],
        "output_slot": "hop_2",
    }
    passages = [
        _doc("p1", "June", "June was mentioned as the author."),
        _doc("p2", "Ada Ada", "Ada Ada was mentioned as the author."),
        _doc("p3", "Ada Lovelace", "Example Work was written by Ada Lovelace."),
        _doc("p4", "Florida", "Florida is a state in the United States."),
    ]
    accepted, telemetry = _select(
        step,
        [consumer],
        passages,
        target_type="relation_graph",
        query="Example Work author",
    )
    assert [row["surface"] for row in accepted] == ["Ada Lovelace"]
    assert telemetry["accepted_count"] == 1
    assert telemetry["max_candidates"] == 2
    assert not telemetry["low_confidence_fill_allowed"]
    reasons = {
        row["surface"]: row["reasons"] for row in telemetry["candidate_decisions"]
    }
    assert "weak_singleton" in reasons["June"]
    assert "repeated_fragment" in reasons["Ada Ada"]


def test_strict_subject_echo_ignores_wikipedia_disambiguator():
    step = {
        "subject": "Alpha",
        "relation_label": "director",
        "dependencies": [],
        "output_slot": "hop_1",
    }
    consumer = {
        "subject": "$hop_1",
        "relation_label": "date of birth",
        "dependencies": ["hop_1"],
        "output_slot": "hop_2",
    }
    passages = [
        _doc("p1", "Alpha (film)", "Alpha is a film directed by Jane Smith."),
        _doc("p2", "Jane Smith", "Jane Smith (born 1970) is an American director."),
    ]
    accepted, telemetry = _select(
        step,
        [consumer],
        passages,
        target_type="relation_graph",
        query="Alpha director",
    )
    assert [row["surface"] for row in accepted] == ["Jane Smith"]
    alpha = next(row for row in telemetry["candidate_decisions"] if row["surface"] == "Alpha (film)")
    assert alpha["strict_subject_echo"]
    assert "strict_subject_echo" in alpha["reasons"]


def test_full_name_outranks_short_fragment_at_the_same_admission_tier():
    step = {
        "subquery_template": "Permission to Fly >> co-writer",
        "dependencies": [],
        "output_slot": "step_1",
    }
    consumer = {
        "subquery_template": "#1 >> record label",
        "dependencies": ["step_1"],
        "output_slot": "step_2",
    }
    passages = [
        _doc(
            "p1",
            "Permission to Fly",
            "Permission to Fly was co-written by Pruitt, credited under her full name Jordan Pruitt.",
        )
    ]
    accepted, _ = _select(step, [consumer], passages)
    assert [row["surface"] for row in accepted] == ["Jordan Pruitt", "Pruitt"]


def test_high_confidence_type_match_outranks_relation_local_unknown():
    step = {
        "subquery_template": "Example Work >> author",
        "dependencies": [],
        "output_slot": "step_1",
    }
    consumer = {
        "subquery_template": "#1 >> educated at",
        "dependencies": ["step_1"],
        "output_slot": "step_2",
    }
    passages = [
        _doc("p1", "Example Work", "Example Work was written by Jane Smith."),
        _doc("p2", "Mary Jones", "Mary Jones (born 1970) is an English historian."),
    ]
    accepted, telemetry = select_bridge_candidates_v5(
        step=step,
        consumers=[consumer],
        target_type="subquery_graph",
        query="Example Work author",
        question="Who wrote Example Work?",
        passages=passages,
        max_candidates=1,
    )
    assert [row["surface"] for row in accepted] == ["Mary Jones"]
    decisions = {row["surface"]: row for row in telemetry["candidate_decisions"]}
    assert decisions["Mary Jones"]["admission_tier"] == 2
    assert decisions["Jane Smith"]["admission_tier"] == 1


def test_source_document_budget_is_top10_and_never_scans_rank11():
    step = {
        "subquery_template": "Example Work >> author",
        "dependencies": [],
        "output_slot": "step_1",
    }
    consumer = {
        "subquery_template": "#1 >> educated at",
        "dependencies": ["step_1"],
        "output_slot": "step_2",
    }
    passages = [
        _doc(f"p{index}", f"Noise {index}", "No relevant relation here.")
        for index in range(1, 10)
    ] + [
        _doc("p10", "Rank Ten Person", "Example Work was written by Rank Ten Person."),
        _doc("p11", "Rank Eleven Person", "Example Work was written by Rank Eleven Person."),
    ]
    accepted, telemetry = select_bridge_candidates_v5(
        step=step,
        consumers=[consumer],
        target_type="subquery_graph",
        query="Example Work author",
        question="A different multi-hop question",
        passages=passages,
        max_candidates=2,
    )
    surfaces = [row["surface"] for row in accepted]
    assert "Rank Ten Person" in surfaces
    assert "Rank Eleven Person" not in {
        row["surface"] for row in telemetry["candidate_decisions"]
    }
    assert telemetry["source_document_limit"] == 10
    assert telemetry["source_documents_inspected"] == 10


def test_profile_conflict_rejects_candidates_and_recommends_exact_fallback():
    step = {
        "subquery_template": "Example Novel >> author",
        "dependencies": [],
        "output_slot": "step_1",
    }
    # A host relation consumes a show/event, which conflicts with author -> person.
    consumer = {
        "subquery_template": "#1 >> host",
        "dependencies": ["step_1"],
        "output_slot": "step_2",
    }
    passages = [
        _doc("p1", "Example Novel", "Example Novel was written by Jane Smith."),
        _doc("p2", "Jane Smith", "Jane Smith (born 1970) is a writer."),
    ]
    accepted, telemetry = _select(step, [consumer], passages)
    assert accepted == []
    assert telemetry["profile"]["profile_conflict"]
    assert telemetry["all_rejected"]
    assert telemetry["fallback_recommended"]
    assert telemetry["fallback_reason"] == "all_candidates_rejected"


def test_unknown_relation_does_not_guess_from_a_frequent_title():
    step = {
        "subquery_template": "Alpha >> bespoke association",
        "dependencies": [],
        "output_slot": "step_1",
    }
    consumer = {
        "subquery_template": "what happened after #1",
        "dependencies": ["step_1"],
        "output_slot": "step_2",
    }
    passages = [
        _doc("p1", "Popular Person", "Popular Person (born 1970) is a writer."),
        _doc("p2", "Popular Person", "Popular Person appears again without a relation cue."),
    ]
    accepted, telemetry = _select(step, [consumer], passages)
    assert accepted == []
    assert telemetry["profile"]["expected_types"] == []
    assert telemetry["fallback_recommended"]


def test_empty_passages_returns_explicit_empty_fallback_telemetry():
    step = {
        "subquery_template": "Alpha >> author",
        "dependencies": [],
        "output_slot": "step_1",
    }
    accepted, telemetry = _select(step, [], [])
    assert accepted == []
    assert telemetry["fallback_reason"] == "no_passages"
    assert telemetry["fallback_recommended"]


def test_selector_is_deterministic_and_telemetry_is_json_serialisable():
    step = {
        "subquery_template": "Example Work >> author",
        "dependencies": [],
        "output_slot": "step_1",
    }
    consumer = {
        "subquery_template": "#1 >> educated at",
        "dependencies": ["step_1"],
        "output_slot": "step_2",
    }
    passages = [
        _doc("p1", "Example Work", "Example Work was written by Ada Lovelace."),
        _doc("p2", "Ada Lovelace", "Ada Lovelace (born 1815) was an English writer."),
    ]
    first = _select(step, [consumer], passages)
    second = _select(step, [consumer], passages)
    assert first == second
    json.dumps(first, sort_keys=True)
