"""Behavioral checks for the conservative, Gold-free v2.3 scorer contract."""
from __future__ import annotations

import pytest

from kgproweight.reward.proofkg_process_v2_2 import score_proofkg_v2_2
from kgproweight.reward.proofkg_process_v2_3 import (
    SCORER_VERSION,
    build_execution_trace_v2_3,
    score_proofkg_v2_3,
)


def _case(answer="Country C", *, tail="Country C", mode="sound", extra_tails=()):
    triples = [
        ("Film Z", "director", "Person A"),
        ("Person A", "father", "Person B"),
        ("Person B", "nationality", tail),
    ]
    plan = {"hops": [
        {"output_slot": f"hop_{i + 1}", "subject": t[0] if i == 0 else f"$hop_{i}",
         "relation_role": "answer_operand" if i == 2 else "bridge", "pids": ["P1"]}
        for i, t in enumerate(triples)
    ]}
    execution = {"hops": [
        {"hop_index": i + 1, "matches": [t]}
        for i, t in enumerate(triples)
    ]}
    execution["hops"][-1]["matches"].extend(
        ("Person B", "nationality", value) for value in extra_tails
    )
    blocks = []
    for i, (head, relation, value) in enumerate(triples, 1):
        reasoning = f"The frozen graph states that {head} has {relation} {value}."
        conclusion = f"{head} has {relation} {value}."
        if mode == "negated_conclusion":
            conclusion = f"It is false that {head} has {relation} {value}."
        elif mode == "unrelated_reasoning":
            reasoning = "Bananas float through outer space and the square root of blue is seven."
        elif mode == "paraphrase":
            reasoning = f"We can read the {relation} of {head} directly from the provided edge."
            conclusion = f"The value of this relation is {value}."
        blocks.append(
            f"[Step {i}]\nReasoning: {reasoning}\n"
            f"Knowledge Used: [({head}, {relation}, {value})]\nConclusion: {conclusion}\n"
        )
    return {
        "question": "What nationality is the father of the director of Film Z?",
        "generation": "\n".join(blocks) + f"\n[Final Answer]\n{answer}",
        "kg_triples": [t for hop in execution["hops"] for t in hop["matches"]],
        "execution_trace": build_execution_trace_v2_3(plan, execution),
        "planned_hops": 3,
    }


def test_historical_scorer_behavior_is_preserved_but_not_claimed_semantic_in_v23():
    case = _case(mode="negated_conclusion")
    legacy = score_proofkg_v2_2(**case)
    repaired = score_proofkg_v2_3(**case)
    assert legacy["score"] == pytest.approx(1.0)
    assert legacy["components"]["G_conclusion_grounding"] == 1.0
    assert repaired["scorer_version"] == SCORER_VERSION
    assert repaired["score"] == pytest.approx(0.85)
    assert repaired["semantic_verified_coverage"] == 0.0
    assert "G_conclusion_grounding" not in repaired["components"]


@pytest.mark.parametrize("mode", ["sound", "negated_conclusion", "unrelated_reasoning", "paraphrase"])
def test_free_text_semantics_always_abstain_without_paraphrase_penalty(mode):
    result = score_proofkg_v2_3(**_case(mode=mode))
    assert result["trajectory_valid"]
    assert result["score"] == pytest.approx(0.85)
    assert result["semantic_status"] == "unverified_no_semantic_verifier"
    assert result["components"]["G_semantic_verified"] == 0.0
    assert result["components"]["m_G_semantic_verified"] == 0.0
    assert result["telemetry"]["semantic_component"] == 0.0


@pytest.mark.parametrize("answer", [
    "not Country C", "Country C or Country D", "Country", "C", "Country C and Country D",
    "Country C / Country D", "Country C [or Country D]", "Country C\nCountry D",
    "Country C\n[Final Answer]\nCountry D", "The answer is not Country C.",
])
def test_partial_negated_and_multiple_answers_receive_no_consistency_credit(answer):
    result = score_proofkg_v2_3(**_case(answer))
    assert result["score"] == pytest.approx(0.40)
    assert result["components"]["A_answer_consistency"] == 0.0
    assert result["components"]["m_A_deterministic"] == 1.0


@pytest.mark.parametrize(("answer", "tail"), [
    ("  COUNTRY   C. ", "Country C"),
    ('"Country C"', "Country C"),
    ("US", "US"), ("X", "X"), ("no", "no"),
    ("Not Here", "Not Here"), ("Either Or", "Either Or"),
    ("中华人民共和国", "中华人民共和国"),
    ("12 January 2003", "January 12, 2003"),
    ("2003-01-12", "12 January 2003"),
    ("2003", "2003"),
])
def test_whole_short_unicode_and_date_answers_are_not_blanket_rejected(answer, tail):
    result = score_proofkg_v2_3(**_case(answer, tail=tail))
    assert result["components"]["A_answer_consistency"] == 1.0
    assert result["score"] == pytest.approx(0.85)


@pytest.mark.parametrize(("answer", "tail"), [
    ("2003", "12 January 2003"),
    ("12 January 2003", "2003"),
    ("13 January 2003", "12 January 2003"),
    ("January 12 nonsense", "12 January 2003"),
    ("Country C is not the answer", "Country C"),
    ("中华", "中华人民共和国"),
])
def test_incompatible_dates_or_shortened_surfaces_abstain(answer, tail):
    result = score_proofkg_v2_3(**_case(answer, tail=tail))
    assert result["components"]["A_answer_consistency"] == 0.0
    assert result["score"] == pytest.approx(0.40)


@pytest.mark.parametrize("extra_tails", [("Country D",), ("Q999",)])
def test_source_derivation_abstention_cannot_renormalize_score_upward(extra_tails):
    result = score_proofkg_v2_3(**_case(extra_tails=extra_tails))
    assert result["components"]["m_A_deterministic"] == 0.0
    assert result["score"] == pytest.approx(0.40)
    assert result["telemetry"]["fixed_denominator"] == 1.0


def test_unknown_free_text_answer_paraphrase_abstains_without_negative_penalty():
    result = score_proofkg_v2_3(**_case("The nationality is Country C."))
    assert result["trajectory_valid"]
    assert result["components"]["answer_match_status"] == "surface_not_verified_against_derived_answer"
    assert result["score"] == pytest.approx(result["telemetry"]["structural_component"])


def test_invalid_format_retains_negative_score():
    case = _case()
    case["generation"] = "[Final Answer]\nCountry C"
    result = score_proofkg_v2_3(**case)
    assert not result["trajectory_valid"]
    assert result["score"] == -1.0
