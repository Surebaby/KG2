"""Conservative structural/answer-consistency ProofKG scorer v2.3.

This successor does not verify the semantics of free-form Reasoning or
Conclusion.  v2.2's tail-string grounding score is retained only as lexical
telemetry, and its 0.15 semantic bonus is removed for *every* trajectory.
Correct paraphrases, negations, and unrelated reasoning can consequently still
tie: their semantics are unverified, not labelled correct or incorrect.

The finite answer contract accepts only the complete normalized derived answer
surface or a date denoting exactly the same precision interval.  It does not
infer aliases, accept substrings, strip answer-list items, or infer entailment
from a negation blacklist.  Short legitimate labels and names containing words
such as "not" or "or" remain admissible when the entire label agrees.

The original citation/order weights (0.05/0.35) and determinate graph-answer
weight (0.45) remain, with a fixed denominator of one.  Thus semantic abstention
cannot inflate a score: the maximum is 0.85, or 0.40 without a verified answer.
This is a combined repair experiment, not evidence of PPO benefit.  Historical
v2.1/v2.2 implementations are imported without modification.  No gold is read.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Sequence, Tuple

from kgproweight.data.parsers import extract_final_answer, parse_steps
from kgproweight.reward.proofkg_process_v2_2 import (
    _answer_consistency_v2_2,
    _conclusion_grounding,
    _dependency_order,
    _dynamic_min_steps,
    _parse_date_interval,
    _precise_citation,
    build_execution_trace_v2_2,
)


SCORER_VERSION = "proofkg-structural-answer-v2-3-frozen-1"
SEMANTIC_CONTRACT_VERSION = "free-text-semantics-abstain-v1"
ANSWER_CONTRACT_VERSION = "whole-derived-surface-or-equal-date-interval-v1"

# Accept the schema's bracketed heading anywhere, and colon variants only at a
# line start.  Consuming the complete remainder avoids the historical parser's
# first-line/next-bracket truncation hiding alternative or contradictory text.
_FINAL_MARKER = re.compile(
    r"\[\s*Final Answer\s*\]|^[ \t]*\*{0,3}Final Answer\*{0,3}[ \t]*[:：]",
    re.IGNORECASE | re.MULTILINE,
)


def build_execution_trace_v2_3(plan, execution):
    """The immutable v2.2 provenance/derivation trace contract is unchanged."""
    return build_execution_trace_v2_2(plan, execution)


def _normalize_answer_surface(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    # Preserve all lexical tokens, including articles and negation.  Unicode
    # word tokens avoid the old ASCII-only collapse of non-English labels.
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _whole_final_answer(generation: str) -> Tuple[str, str]:
    markers = list(_FINAL_MARKER.finditer(generation))
    if len(markers) != 1:
        return "", "final_field_count_not_one"
    answer = generation[markers[0].end():].strip()
    if not answer:
        return "", "empty_final_field"
    # A multi-line explanation may be factually right, but it lies outside the
    # finite single-answer-surface contract.  Do not retain just its first line.
    if "\n" in answer or "\r" in answer:
        return answer, "multiline_final_field_unverified"
    return answer, "single_final_field"


def _strict_answer_match(answer: str, derived: str) -> Tuple[float, str]:
    left, right = _normalize_answer_surface(answer), _normalize_answer_surface(derived)
    if left and right and left == right:
        return 1.0, "whole_normalized_surface_equal"
    try:
        left_date, right_date = _parse_date_interval(answer), _parse_date_interval(derived)
    except (TypeError, ValueError, OverflowError):
        # Unsupported prose must abstain rather than crashing date conversion.
        left_date = right_date = None
    if left_date is not None and left_date == right_date:
        return 1.0, "equal_date_precision_interval"
    return 0.0, "surface_not_verified_against_derived_answer"


def score_proofkg_v2_3(
    *,
    question: str,
    generation: str,
    kg_triples: Sequence[Sequence[str]],
    execution_trace: List[Dict[str, Any]],
    planned_hops: int,
) -> Dict[str, Any]:
    """Return a structural/answer score and explicit zero semantic coverage.

    Source answer derivation and format validity are inherited from v2.2; the
    answer-equivalence and reward aggregation contracts are superseded here.
    This function neither changes the production parser nor evaluation rules.
    """
    triples = {
        tuple(str(value).strip() for value in edge)
        for edge in kg_triples if len(edge) == 3
    }
    steps = parse_steps(generation, known_kg=list(triples))
    answer, extraction_status = _whole_final_answer(generation)
    # Preserve v2.2's format-only gate.  A strict final-surface mismatch changes
    # answer credit, not the shared production trajectory validity/outcome rule.
    valid = (
        bool(steps)
        and len(steps) >= _dynamic_min_steps(planned_hops)
        and extract_final_answer(generation) is not None
    )
    if valid:
        for expected, step in enumerate(steps, 1):
            match = re.search(
                r"(?is)\breasoning\s*:\s*(.*?)(?:knowledge used\s*:|conclusion\s*:|$)",
                step.raw_text.strip(),
            )
            if step.index != expected or not match or len(match.group(1).strip()) < 20:
                valid = False
                break
    result = {
        "scorer_version": SCORER_VERSION,
        "score_kind": "structural_and_graph_answer_consistency",
        "semantic_contract_version": SEMANTIC_CONTRACT_VERSION,
        "answer_contract_version": ANSWER_CONTRACT_VERSION,
        "semantic_status": "unverified_no_semantic_verifier",
        "semantic_verified_coverage": 0.0,
        "trajectory_valid": bool(valid),
        "prediction": answer,
        "n_steps": len(steps),
        "score": -1.0,
        "components": {},
    }
    if not valid:
        return result

    required = {triple for hop in execution_trace for triple in hop["matched_triples"]}
    precision, n_known, n_unknown = _precise_citation(steps, required)
    order = _dependency_order(execution_trace, steps)
    cited = {triple for step in steps for triple in step.cited_triples}
    coverage = sum(
        bool(set(hop["matched_triples"]) & cited) for hop in execution_trace
    ) / max(1, len(execution_trace))
    # Pass an empty answer to the historical *source derivation* helper, so no
    # legacy substring answer equivalence is evaluated or credited in v2.3.
    _, answer_mask, operator, derived, derivation_status = _answer_consistency_v2_2(
        question, execution_trace, ""
    )
    components = {
        "P_precise_citation": precision,
        "H_hop_coverage": coverage,
        "O_dependency_order": order,
        "C": coverage * order,
        "L_conclusion_tail_mention_telemetry": _conclusion_grounding(execution_trace, steps),
        "G_semantic_verified": 0.0,
        "m_G_semantic_verified": 0.0,
        "m_A_deterministic": float(answer_mask),
        "operator": operator,
        "derived_answer": derived,
        "derivation_status": derivation_status,
    }
    result["components"] = components
    answer_score, match_status = 0.0, "source_answer_derivation_abstained"
    if components["m_A_deterministic"]:
        if extraction_status == "single_final_field":
            answer_score, match_status = _strict_answer_match(answer, components["derived_answer"])
        else:
            match_status = extraction_status
    components["A_answer_consistency"] = answer_score
    components["answer_match_status"] = match_status
    structural = 0.05 * components["P_precise_citation"] + 0.35 * components["C"]
    answer_component = 0.45 * components["m_A_deterministic"] * answer_score
    result["score"] = float(structural + answer_component)
    result["telemetry"] = {
        "n_known_citations": n_known,
        "n_unknown_citations": n_unknown,
        "structural_component": float(structural),
        "answer_component": float(answer_component),
        "semantic_component": 0.0,
        "fixed_denominator": 1.0,
        "maximum_score": 0.85,
        "answer_extraction_status": extraction_status,
        "answer_surface_verified": bool(answer_score),
    }
    return result
