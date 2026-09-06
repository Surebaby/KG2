"""Parsing of Teacher trajectories into ``ParsedStep`` objects."""

from __future__ import annotations

from kgproweight.data.parsers import (
    extract_final_answer,
    parse_steps,
    parse_teacher_output,
)


_SAMPLE = """\
[Step 1]
Reasoning: We must find Barack Obama's spouse.
Knowledge Used: [(Barack Obama, spouse, Michelle Obama)]
Conclusion: His spouse is Michelle Obama.

[Step 2]
Reasoning: She is a lawyer and author.
Knowledge Used: [(Michelle Obama, occupation, Lawyer)]
Conclusion: Michelle Obama is a lawyer.

[Final Answer] Michelle Obama
"""


def test_parse_steps_count_and_indices():
    steps = parse_steps(_SAMPLE)
    assert len(steps) == 2
    assert [s.index for s in steps] == [1, 2]


def test_parse_steps_triples():
    steps = parse_steps(_SAMPLE)
    assert steps[0].cited_triples and steps[0].cited_triples[0] == (
        "Barack Obama",
        "spouse",
        "Michelle Obama",
    )
    assert steps[1].cited_triples[0] == ("Michelle Obama", "occupation", "Lawyer")


def test_parse_steps_conclusions_extracted():
    steps = parse_steps(_SAMPLE)
    assert "Michelle Obama" in (steps[0].intermediate_conclusion or "")
    assert "lawyer" in (steps[1].intermediate_conclusion or "").lower()


def test_extract_final_answer():
    assert extract_final_answer(_SAMPLE) == "Michelle Obama"


def test_legacy_alias_parse_teacher_output():
    assert parse_teacher_output is parse_steps


def test_step_header_variants():
    raw = """### Step 1
Reasoning: …
Conclusion: A.

Step 2: Reasoning: …
Conclusion: B.

[Final Answer] B
"""
    steps = parse_steps(raw)
    assert len(steps) == 2
    assert [s.index for s in steps] == [1, 2]


def test_parenthesised_prose_outside_knowledge_used_is_not_a_citation():
    raw = """[Step 1]
Reasoning: These refer to other contexts (e.g., a composer, a language).
Knowledge Used: [(Young, language of work or name, English)]
Conclusion: The KG is not useful.
[Final Answer] unknown
"""
    steps = parse_steps(raw)
    assert steps[0].cited_triples == [("Young", "language of work or name", "English")]


def test_known_kg_exact_matching_handles_commas_inside_values():
    kg = [("University of California, Santa Cruz", "located in", "Santa Cruz, California")]
    raw = """[Step 1]
Reasoning: The university is in Santa Cruz.
Knowledge Used: [(University of California, Santa Cruz, located in, Santa Cruz, California)]
Conclusion: It is in Santa Cruz.
[Final Answer] Santa Cruz
"""
    step = parse_steps(raw, known_kg=kg)[0]
    assert step.cited_triples == kg
    assert step.knowledge_used_valid


def test_unknown_and_multiple_knowledge_used_fields_are_contract_errors():
    kg = [("Young", "language of work or name", "English")]
    raw = """[Step 1]
Reasoning: Example.
Knowledge Used: [(Absent, relation, triple)]
Knowledge Used: [(Young, language of work or name, English)]
Conclusion: Example.
[Final Answer] Example
"""
    step = parse_steps(raw, known_kg=kg)[0]
    assert not step.knowledge_used_valid
    assert "knowledge_used_field_count=2" in step.citation_contract_errors
    assert "unknown_or_malformed_knowledge_used_content" in step.citation_contract_errors


def test_raw_unknown_citation_is_visible_to_telemetry_but_not_reward():
    kg = [("Young", "language of work or name", "English")]
    raw = """[Step 1]
Reasoning: Example.
Knowledge Used: [(Young, language of work or name, English), (Ghost, relation, Thing)]
Conclusion: Example.
[Final Answer] Example
"""

    step = parse_steps(raw, known_kg=kg)[0]

    # Only the prompt-KG-matched triple remains reward-visible.
    assert step.cited_triples == kg
    assert step.unknown_citation_surfaces == ["(Ghost, relation, Thing)"]
    assert not step.knowledge_used_malformed_content
    assert not step.knowledge_used_valid


def test_malformed_citation_content_is_distinct_from_unknown_surface():
    kg = [("Young", "language of work or name", "English")]
    raw = """[Step 1]
Reasoning: Example.
Knowledge Used: [(Young, language of work or name, English), broken]
Conclusion: Example.
[Final Answer] Example
"""

    step = parse_steps(raw, known_kg=kg)[0]

    assert step.cited_triples == kg
    assert step.unknown_citation_surfaces == []
    assert step.knowledge_used_malformed_content
