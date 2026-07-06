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
Knowledge Used: (Barack Obama, spouse, Michelle Obama)
Conclusion: His spouse is Michelle Obama.

[Step 2]
Reasoning: She is a lawyer and author.
Knowledge Used: (Michelle Obama, occupation, Lawyer)
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
