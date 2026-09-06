from scripts.pilot.score_proofkg_dynamic_validity_confirmation import (
    required_steps,
    score_process_candidate,
)


KG = [
    ["Song A", "performer", "Singer B"],
    ["Singer B", "country of citizenship", "Country C"],
]
RECORD = {"kg_subgraph": KG, "query_plan": {"hops": [{}, {}]}}


def _trace(answer: str) -> str:
    return f"""[Step 1]
Reasoning: The supplied path identifies Singer B as the performer of Song A.
Knowledge Used: [(Song A, performer, Singer B)]
Conclusion: The performer is Singer B.

[Step 2]
Reasoning: The supplied path identifies Country C as Singer B's citizenship.
Knowledge Used: [(Singer B, country of citizenship, Country C)]
Conclusion: Singer B has citizenship in Country C.

[Final Answer]
{answer}"""


def test_two_hop_proof_uses_two_step_input_conditioned_gate():
    assert required_steps(RECORD) == 2
    scored = score_process_candidate(
        question="What country is the performer of Song A from?",
        record=RECORD,
        generation=_trace("Country C"),
    )
    assert scored["trajectory_valid"] is True
    assert scored["components"]["reachable_edge_coverage"] == 1.0


def test_answer_path_alignment_discriminates_without_gold_argument():
    correct = score_process_candidate(
        question="What country is the performer of Song A from?",
        record=RECORD,
        generation=_trace("Country C"),
    )
    wrong = score_process_candidate(
        question="What country is the performer of Song A from?",
        record=RECORD,
        generation=_trace("Country Z"),
    )
    assert correct["score"] > wrong["score"]


def test_empty_proof_retains_legacy_three_step_gate():
    empty = {"kg_subgraph": [], "query_plan": {"hops": [{}, {}]}}
    scored = score_process_candidate(
        question="What country is the performer of Song A from?",
        record=empty,
        generation=_trace("Country C"),
    )
    assert required_steps(empty) == 3
    assert scored["eligible_proofkg"] is False
    assert scored["trajectory_valid"] is False
