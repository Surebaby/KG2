from scripts.pilot.audit_proofkg_process_rankability import score_process_candidate


KG = [
    ["Song A", "performer", "Singer B"],
    ["Singer B", "country of citizenship", "Country C"],
]


def _trace(answer: str, *, unknown: bool = False) -> str:
    first = "(Invented, relation, Value)" if unknown else "(Song A, performer, Singer B)"
    return f"""[Step 1]
Reasoning: The supplied evidence identifies the performer of Song A as Singer B.
Knowledge Used: [{first}]
Conclusion: The performer is Singer B.

[Step 2]
Reasoning: The supplied evidence identifies Singer B's country of citizenship.
Knowledge Used: [(Singer B, country of citizenship, Country C)]
Conclusion: Singer B has citizenship in Country C.

[Step 3]
Reasoning: The requested country is therefore the terminal value in the evidence path.
Knowledge Used: [(Singer B, country of citizenship, Country C)]
Conclusion: The requested country is Country C.

[Final Answer]
{answer}"""


def test_grounded_terminal_answer_outranks_unsupported_answer():
    correct = score_process_candidate(
        question="What country is the performer of Song A from?",
        kg=KG,
        generation=_trace("Country C"),
    )
    wrong = score_process_candidate(
        question="What country is the performer of Song A from?",
        kg=KG,
        generation=_trace("Country Z"),
    )
    assert correct["score"] > wrong["score"]
    assert correct["components"]["reachable_edge_coverage"] == 1.0


def test_unknown_citation_is_penalized():
    grounded = score_process_candidate(
        question="What country is the performer of Song A from?",
        kg=KG,
        generation=_trace("Country C"),
    )
    unknown = score_process_candidate(
        question="What country is the performer of Song A from?",
        kg=KG,
        generation=_trace("Country C", unknown=True),
    )
    assert grounded["score"] > unknown["score"]
    assert unknown["unknown_citations"] == 1


def test_invalid_short_trace_receives_fixed_penalty():
    invalid = score_process_candidate(
        question="What country is the performer of Song A from?",
        kg=KG,
        generation="[Step 1]\nReasoning: Too short.\nKnowledge Used: []\nConclusion: Unknown.\n[Final Answer]\nCountry C",
    )
    assert invalid["trajectory_valid"] is False
    assert invalid["score"] == -1.0
