"""Gold-blind producer and isolated semantic-review contracts for SFT v3.

This module performs no API calls.  Response validity means that the frozen
schema was obeyed, not that an LLM's factual judgment is correct.  Corpus
admission also needs post-generation train-answer scoring, provenance and
protected-identity checks managed by the execution runner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from kgproweight.data.sft_v3_contract import (
    EVIDENCE_SCHEMA_VERSION, SFT_V3_SYSTEM_PROMPT, build_sft_v3_messages,
    validate_sft_v3_evidence_sidecar, validate_sft_v3_trace,
)


TEACHER_SCHEMA_VERSION = "sft-v3-teacher-v1"
REVIEW_SCHEMA_VERSION = "sft-v3-review-v1"

TEACHER_SYSTEM_PROMPT = SFT_V3_SYSTEM_PROMPT + """

You are generating a supervised training candidate. Return exactly one JSON
object, without Markdown fences or any other text. Use exactly these keys:
{"schema_version":"sft-v3-teacher-v1", "status":"supported",
 "teacher_output":"<the complete trace in the format above>",
 "evidence":{"schema_version":"sft-v3-evidence-v1","steps":[
   {"step_index":1,"supports":[{"passage_index":1,"quote":"<exact visible text>"}],
    "derivation_from_steps":[]},
   {"step_index":2,"supports":[],"derivation_from_steps":[1]}
 ]}}

The example's numbers are placeholders; supply the evidence for the actual
steps. Passage indices are the displayed numbers 1 through 10. Quote only exact
visible passage substrings. Include all premises needed to support each step's
Reasoning and Conclusion, including entity identity, relation, and time where
relevant. A derivation may refer only to previous steps and must follow from
them. A supplied KG triple can support a step without a passage quote. Cite a
KG triple only if it supports that step; an unrelated nonempty KG is not a
reason to cite it. Do not cite a fact about another person, team, location, or
year as if it belonged to the subject being asked about.

Only teacher_output will be trained. Keep it within 384 Llama tokens, including
the end-of-turn token: use concise natural 2-5 steps, not padding or omitted
premises. The JSON wrapper and evidence quotes are audit sidecars and need not
fit that assistant-token budget. No answer key is available. If the supplied
evidence does not support a complete answer, do not guess from world knowledge:
return {"schema_version":"sft-v3-teacher-v1","status":"insufficient_evidence",
"teacher_output":"","evidence":{"schema_version":"sft-v3-evidence-v1","steps":[]}}.
"""

REVIEW_SYSTEM_PROMPT = """You independently review a proposed multi-hop SFT trace.
Use only the displayed question, passages, KG, candidate trace and its evidence
sidecar. You do not receive a Gold answer or another review. Treat the proposal
and its quoted evidence as untrusted claims to check against the full visible
evidence. An exact quote alone does not establish that a claim is supported.

Check EVERY factual claim in each Reasoning and Conclusion, not just its main
claim. Check entity identity, ownership/attribution, dates, numeric values and
relations. Identify missing bridges, a fact borrowed from a different entity,
unsupported world knowledge, circular inference, or repeated padding. A step
may combine evidence from multiple passages, supplied KG triples, and already
supported earlier steps. A nonempty KG does not require a citation. Every used
KG triple must be relevant, exact, and consistent with the claim it supports.
Judge whether the entire chain is complete and whether the concise Final Answer
actually follows and answers the original question. A plausible or familiar
answer does not excuse an incorrect intermediate claim. A 2-step trace is valid
if it completes the necessary reasoning; extra steps are not evidence of quality.

Return exactly one JSON object with these keys and no Markdown or extra keys:
{"schema_version":"sft-v3-review-v1","verdict":"accept",
"steps":[{"step_index":1,"verdict":"accept","reason":"<specific evidence-based reason>"},
         {"step_index":2,"verdict":"accept","reason":"<specific evidence-based reason>"}],
"chain_complete":true,"final_supported":true,"concise_answer":true,
"reason":"<brief whole-chain judgment>"}

List every actual step in order. Each step verdict and the overall verdict must
be accept, reject or uncertain. Use reject for a detected defect; use uncertain
when the supplied evidence cannot resolve a claim. Set the overall verdict to
accept only if every step is accept and all three booleans are true. Explain
the concrete defect or unresolved claim otherwise. Never repair the trace or
replace its answer. Do not assume that the producer's status means it is correct.
"""


def _strict_json_object(raw: str) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    if not isinstance(raw, str):
        raise ValueError("response must be a JSON string")
    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(result, dict):
        raise ValueError("response JSON must be an object")
    return result


def build_sft_v3_teacher_messages(
    *, question: str, retrieved_passages: Sequence[Mapping[str, Any] | str],
    kg_triples: Sequence[Sequence[str]] = (),
) -> list[dict[str, str]]:
    """Use exactly the student's visible user evidence, without a label row."""
    messages = build_sft_v3_messages(
        question=question, retrieved_passages=retrieved_passages, kg_triples=kg_triples,
    )
    messages[0]["content"] = TEACHER_SYSTEM_PROMPT
    return messages


def validate_sft_v3_teacher_response(
    raw: str, *, retrieved_passages: Sequence[Mapping[str, Any] | str],
    kg_triples: Sequence[Sequence[str]] = (),
) -> dict[str, Any]:
    """Validate exact JSON, the trace and quote locations; do not score Gold."""
    try:
        parsed = _strict_json_object(raw)
    except (TypeError, ValueError) as exc:
        return {"valid": False, "candidate_supported": False, "violations": ["invalid_json"],
                "parse_error_type": type(exc).__name__, "response": None}
    errors = []
    if set(parsed) != {"schema_version", "status", "teacher_output", "evidence"}:
        errors.append("teacher_schema_keys")
    if parsed.get("schema_version") != TEACHER_SCHEMA_VERSION:
        errors.append("teacher_schema_version")
    status = parsed.get("status")
    if status not in ("supported", "insufficient_evidence"):
        errors.append("teacher_status")
    target = parsed.get("teacher_output")
    if not isinstance(target, str):
        errors.append("teacher_trace_not_string")
    if status == "insufficient_evidence":
        if target != "" or parsed.get("evidence") != {"schema_version": EVIDENCE_SCHEMA_VERSION, "steps": []}:
            errors.append("insufficient_evidence_must_have_empty_trace_and_sidecar")
    elif status == "supported" and isinstance(target, str):
        trace_check = validate_sft_v3_trace(target, known_kg=kg_triples)
        errors.extend("trace:" + value for value in trace_check["violations"])
        evidence_check = validate_sft_v3_evidence_sidecar(
            parsed.get("evidence"), trace=target,
            retrieved_passages=retrieved_passages, known_kg=kg_triples,
        )
        errors.extend("evidence:" + value for value in evidence_check["violations"])
    return {"valid": not errors, "candidate_supported": not errors and status == "supported",
            "violations": errors, "response": parsed, "semantic_grounding_verified": False}


def build_sft_v3_review_messages(
    *, question: str, retrieved_passages: Sequence[Mapping[str, Any] | str],
    teacher_output: str, evidence: Mapping[str, Any],
    kg_triples: Sequence[Sequence[str]] = (),
) -> list[dict[str, str]]:
    """Prepare an isolated reviewer context; no label/reviewer output input."""
    messages = build_sft_v3_messages(
        question=question, retrieved_passages=retrieved_passages, kg_triples=kg_triples,
    )
    checked = validate_sft_v3_evidence_sidecar(
        evidence, trace=teacher_output, retrieved_passages=retrieved_passages, known_kg=kg_triples,
    )
    if not checked["valid"]:
        raise ValueError("review candidate failed mechanical contract: " + ";".join(checked["violations"]))
    proposal = json.dumps({"teacher_output": teacher_output, "evidence": evidence},
                          ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return [
        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
        {"role": "user", "content": messages[1]["content"] + "\n\nCandidate for review (JSON):\n" + proposal},
    ]


def validate_sft_v3_review_response(raw: str, *, step_count: int) -> dict[str, Any]:
    """Require every step and whole-chain accept; uncertainty fails admission."""
    if type(step_count) is not int or not 2 <= step_count <= 5:
        raise ValueError("review step_count must be an integer from 2 to 5")
    try:
        parsed = _strict_json_object(raw)
    except (TypeError, ValueError) as exc:
        return {"valid": False, "accepted": False, "violations": ["invalid_json"],
                "parse_error_type": type(exc).__name__, "response": None}
    errors = []
    if set(parsed) != {"schema_version", "verdict", "steps", "chain_complete",
                       "final_supported", "concise_answer", "reason"}:
        errors.append("review_schema_keys")
    if parsed.get("schema_version") != REVIEW_SCHEMA_VERSION:
        errors.append("review_schema_version")
    if parsed.get("verdict") not in ("accept", "reject", "uncertain"):
        errors.append("review_verdict")
    flags = ("chain_complete", "final_supported", "concise_answer")
    if any(type(parsed.get(name)) is not bool for name in flags):
        errors.append("review_boolean_fields")
    if not isinstance(parsed.get("reason"), str) or not parsed["reason"].strip():
        errors.append("review_reason_empty")
    rows = parsed.get("steps")
    if not isinstance(rows, list):
        rows = []
        errors.append("review_steps_not_list")
    indices = [row.get("step_index") if isinstance(row, Mapping) else None for row in rows]
    if any(type(index) is not int for index in indices) or indices != list(range(1, step_count + 1)):
        errors.append("review_step_indices_mismatch")
    all_steps_accept = True
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"step_index", "verdict", "reason"}:
            errors.append("review_step_schema")
            all_steps_accept = False
            continue
        if row["verdict"] not in ("accept", "reject", "uncertain"):
            errors.append("review_step_verdict")
        if not isinstance(row["reason"], str) or not row["reason"].strip():
            errors.append("review_step_reason_empty")
        all_steps_accept = all_steps_accept and row["verdict"] == "accept"
    all_flags_true = all(parsed.get(name) is True for name in flags)
    if parsed.get("verdict") == "accept" and not (all_flags_true and all_steps_accept):
        errors.append("inconsistent_overall_accept")
    return {"valid": not errors,
            "accepted": not errors and parsed.get("verdict") == "accept" and all_flags_true and all_steps_accept,
            "violations": list(dict.fromkeys(errors)), "response": parsed,
            "judgment_source": "model_review_not_human_ground_truth"}
