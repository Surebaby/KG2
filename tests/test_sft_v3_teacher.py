"""Teacher/reviewer boundaries do not invent semantic certification."""

import json

import pytest

from kgproweight.data.sft_v3_contract import build_sft_v3_messages
from kgproweight.data.sft_v3_teacher import (
    REVIEW_SCHEMA_VERSION, TEACHER_SCHEMA_VERSION,
    build_sft_v3_review_messages, build_sft_v3_teacher_messages,
    validate_sft_v3_review_response, validate_sft_v3_teacher_response,
)


def passages():
    return [{"contents": f"Entity {i} has a documented fact."} for i in range(1, 11)]


def teacher():
    trace = "\n\n".join(
        f"[Step {i}]\nReasoning: Entity {i} has a documented fact in the evidence.\n"
        f"Knowledge Used: []\nConclusion: Entity {i} is documented."
        for i in (1, 2)
    ) + "\n\n[Final Answer]\nExample"
    return {
        "schema_version": TEACHER_SCHEMA_VERSION, "status": "supported", "teacher_output": trace,
        "evidence": {"schema_version": "sft-v3-evidence-v1", "steps": [
            {"step_index": i, "supports": [{"passage_index": i, "quote": f"Entity {i} has a documented fact."}],
             "derivation_from_steps": []} for i in (1, 2)
        ]},
    }


def review():
    return {
        "schema_version": REVIEW_SCHEMA_VERSION, "verdict": "accept",
        "steps": [{"step_index": i, "verdict": "accept", "reason": "Exact claim is supported."} for i in (1, 2)],
        "chain_complete": True, "final_supported": True, "concise_answer": True,
        "reason": "All proposed claims follow from visible evidence.",
    }


def test_producer_user_evidence_exactly_matches_student():
    generated = build_sft_v3_teacher_messages(question="Q", retrieved_passages=passages())
    student = build_sft_v3_messages(question="Q", retrieved_passages=passages())
    assert generated[1] == student[1]
    assert len(generated) == 2
    assert "insufficient_evidence" in generated[0]["content"]


def test_supported_candidate_requires_future_semantic_and_answer_gates():
    result = validate_sft_v3_teacher_response(json.dumps(teacher()), retrieved_passages=passages())
    assert result["valid"] and result["candidate_supported"]
    assert result["semantic_grounding_verified"] is False


def test_insufficient_evidence_is_valid_response_but_not_accepted_candidate():
    proposal = {"schema_version": TEACHER_SCHEMA_VERSION, "status": "insufficient_evidence",
                "teacher_output": "", "evidence": {"schema_version": "sft-v3-evidence-v1", "steps": []}}
    result = validate_sft_v3_teacher_response(json.dumps(proposal), retrieved_passages=passages())
    assert result["valid"] and not result["candidate_supported"]
    proposal["teacher_output"] = "guess"
    assert not validate_sft_v3_teacher_response(json.dumps(proposal), retrieved_passages=passages())["valid"]


@pytest.mark.parametrize("raw", [
    '{"schema_version":"first","schema_version":"sft-v3-teacher-v1"}',
    '{"nested":{"key":1,"key":2}}',
    'NaN', '{"value":NaN}', '[]', 'null',
    '```json\n{}\n```', '{} {}', 'a plausible answer',
])
def test_duplicate_keys_non_json_and_non_finite_are_rejected(raw):
    assert not validate_sft_v3_teacher_response(raw, retrieved_passages=passages())["valid"]
    assert not validate_sft_v3_review_response(raw, step_count=2)["valid"]


@pytest.mark.parametrize("mutate", [
    lambda x: x.update(answer="unknown key"),
    lambda x: x.update(status="accepted"),
    lambda x: x.update(schema_version="old"),
    lambda x: x.update(teacher_output="[Final Answer]\nExample"),
    lambda x: x.update(evidence=None),
    lambda x: x["evidence"]["steps"][0]["supports"][0].update(quote="made up quote"),
])
def test_producer_schema_and_evidence_fail_closed(mutate):
    proposal = teacher()
    mutate(proposal)
    result = validate_sft_v3_teacher_response(json.dumps(proposal), retrieved_passages=passages())
    assert not result["valid"] and not result["candidate_supported"]


def test_reviewer_gets_same_evidence_and_candidate_without_gold_or_other_review():
    proposal = teacher()
    messages = build_sft_v3_review_messages(
        question="Q", retrieved_passages=passages(),
        teacher_output=proposal["teacher_output"], evidence=proposal["evidence"],
    )
    original = build_sft_v3_messages(question="Q", retrieved_passages=passages())
    assert messages[1]["content"].startswith(original[1]["content"] + "\n\n")
    payload = json.loads(messages[1]["content"].split("Candidate for review (JSON):\n", 1)[1])
    assert set(payload) == {"teacher_output", "evidence"}
    assert payload["teacher_output"] == proposal["teacher_output"]
    assert len(messages) == 2


def test_reviewer_prompt_rejects_invalid_candidate_instead_of_repairing():
    proposal = teacher()
    with pytest.raises(ValueError, match="mechanical contract"):
        build_sft_v3_review_messages(question="Q", retrieved_passages=passages(),
                                     teacher_output="Example", evidence=proposal["evidence"])


def test_unanimous_complete_review_is_only_model_judgment():
    result = validate_sft_v3_review_response(json.dumps(review()), step_count=2)
    assert result["valid"] and result["accepted"]
    assert result["judgment_source"] == "model_review_not_human_ground_truth"


@pytest.mark.parametrize("verdict", ["reject", "uncertain"])
def test_reject_or_uncertainty_is_valid_but_not_admitted(verdict):
    proposed = review()
    proposed["verdict"] = verdict
    proposed["steps"][0]["verdict"] = verdict
    proposed["chain_complete"] = False
    result = validate_sft_v3_review_response(json.dumps(proposed), step_count=2)
    assert result["valid"] and not result["accepted"]


@pytest.mark.parametrize("mutate", [
    lambda x: x.update(unknown="extra"),
    lambda x: x.update(verdict="pass"),
    lambda x: x.update(chain_complete=1),
    lambda x: x.update(reason=" "),
    lambda x: x["steps"].pop(),
    lambda x: x["steps"].reverse(),
    lambda x: x["steps"][0].update(step_index=True),
    lambda x: x["steps"][0].update(verdict="uncertain"),
    lambda x: x["steps"][0].update(reason=""),
    lambda x: x["steps"][0].update(extra="bad"),
    lambda x: x.update(final_supported=False),
])
def test_overall_accept_cannot_override_bad_or_missing_steps(mutate):
    proposed = review()
    mutate(proposed)
    result = validate_sft_v3_review_response(json.dumps(proposed), step_count=2)
    assert not result["valid"] and not result["accepted"]
