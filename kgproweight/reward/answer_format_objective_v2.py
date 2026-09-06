"""Opt-in answer/format objective; no evaluation or process-validity changes.

The caller computes the frozen canonical EM/F1 against train labels.  This
module never receives those labels and never puts answer text in telemetry.
Valid trajectories keep the existing answer and process terms.  Only a complete
two-step response whose sole old violation is a three-step minimum can retain
its answer term; all its process credit is removed and a fixed penalty applies.
The broader unique-Final-only salvage proposal was rejected before execution.
This narrow exception does not establish factual correctness of its reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Sequence

from kgproweight.data.parsers import extract_final_answer


OBJECTIVE_VERSION = "answer-format-objective-v2"
ANSWER_WEIGHT = 4.0
F1_WEIGHT = 0.1
FORMAT_PENALTY = 1.0
SEVERE_INVALID_REWARD = -4.0
SOURCE_PROCESS_ABS_BOUND = 0.3

# Deliberately the same marker recognition as source-gate format-v2.  This is
# an answer-eligibility guard for invalid trajectories, not a new validator.
_FINAL_FIELD = re.compile(
    r"\[\s*Final Answer\s*\]|^[ \t]*(?:\*\*)?Final Answer(?:\*\*)?[ \t]*[:：]",
    re.IGNORECASE | re.MULTILINE,
)
# Cover all Step headings understood by parse_steps, including empty blocks.
_STEP_FIELD = re.compile(
    r"\[\s*step\s+\d+\s*\]|^\s*#{1,3}\s*Step\s+\d+\s*$"
    r"|^\s*Step\s+\d+\s*[:.)]",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class FinalAnswerContractV2:
    """Gold-free eligibility.  Use telemetry(), not asdict(), for logging."""

    eligible: bool
    reason: str
    literal_final_count: int
    answer: str = field(default="", repr=False)

    def telemetry(self) -> dict[str, Any]:
        return {
            "answer_contract_eligible": self.eligible,
            "answer_contract_reason": self.reason,
            "literal_final_count": self.literal_final_count,
        }


def _firstline(response: str) -> str:
    return (extract_final_answer(response) or "").split("\n", 1)[0].strip()


def inspect_final_answer_v2(response: str) -> FinalAnswerContractV2:
    """Inspect a unique, populated terminal Final without consulting Gold.

    Whitespace after the heading is treated exactly as the historical parser
    treats it.  A punctuation-only first answer line is not rescued by text on
    later lines.  Parsing the full response must agree with parsing the literal
    Final suffix, preventing an earlier loose ``Final Answer`` mention from
    silently becoming the rewarded answer.
    """
    if not isinstance(response, str):
        raise TypeError("response must be a string")
    matches = list(_FINAL_FIELD.finditer(response))
    count = len(matches)
    if count != 1:
        return FinalAnswerContractV2(False, "literal_final_count_not_one", count)
    final = matches[0]
    suffix = response[final.end():]
    literal_firstline = suffix.strip().split("\n", 1)[0].strip()
    if not any(char.isalnum() for char in literal_firstline):
        return FinalAnswerContractV2(False, "empty_or_decoration_only_firstline", count)
    if _STEP_FIELD.search(suffix):
        return FinalAnswerContractV2(False, "step_after_final", count)
    if re.search(r"\bFinal Answer\b", suffix, flags=re.IGNORECASE):
        return FinalAnswerContractV2(False, "additional_final_text_after_field", count)
    answer = _firstline(response[final.start():])
    if not any(char.isalnum() for char in answer):
        return FinalAnswerContractV2(False, "legacy_firstline_empty", count)
    if _firstline(response) != answer:
        return FinalAnswerContractV2(False, "legacy_extraction_ambiguous", count)
    return FinalAnswerContractV2(True, "unique_nonempty_terminal_final", count, answer)


@dataclass(frozen=True)
class ShortfallSalvageContractV2:
    """Eligibility of the one permitted format exception, with no label data."""

    eligible: bool
    reason: str
    literal_final_count: int
    parsed_step_count: int
    required_steps: int
    answer: str = field(default="", repr=False)

    def telemetry(self) -> dict[str, Any]:
        return {
            "shortfall_salvage_eligible": self.eligible,
            "shortfall_salvage_reason": self.reason,
            "literal_final_count": self.literal_final_count,
            "parsed_step_count": self.parsed_step_count,
            "required_steps": self.required_steps,
        }


def _explicit_passage_ids(response: str) -> set[int]:
    """Recognize explicit numeric citations, without inferring entailment.

    Includes Passage 3, Passages 2 and 3, [P3], and [P2, P3].  A bare [3]
    outside a Passage phrase is not treated as evidence because it can be a
    date/reference in ordinary prose.  This is a syntax guard, not a verifier.
    """
    ids: set[int] = set()
    patterns = (
        r"\bpassages?\s*[:#]?\s*\[?\s*\d+"
        r"(?:(?:\s*,\s*|\s+and\s+|\s*&\s*|\s*[-–]\s*)\d+)*\s*\]?",
        r"\[\s*[Pp]\s*\d+(?:\s*,\s*(?:[Pp]\s*)?\d+)*\s*\]",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, response, flags=re.IGNORECASE):
            ids.update(int(value) for value in re.findall(r"\d+", match.group(0)))
    return ids


def inspect_shortfall_salvage_v2(
    response: str,
    *,
    steps: Sequence[Any],
    required_steps: int,
    violations: Sequence[str],
    known_passage_ids: Sequence[int],
) -> ShortfallSalvageContractV2:
    """Permit exactly two complete steps only when the old minimum is three.

    Pass the unchanged source-gate format-v2 validation's steps, required_steps,
    and violations.  ``steps`` must have been parsed against the actual prompt
    KG, and known_passage_ids must identify only the rendered passages.  Neither
    Gold nor output correctness is inspected here.  Do not apply this guard to
    change the treatment of a response that the old validator accepted.
    """
    final = inspect_final_answer_v2(response)

    def verdict(eligible: bool, reason: str) -> ShortfallSalvageContractV2:
        return ShortfallSalvageContractV2(
            eligible, reason, final.literal_final_count, len(steps), required_steps,
            final.answer if eligible else "",
        )

    if not final.eligible:
        return verdict(False, "final:" + final.reason)
    if required_steps != 3:
        return verdict(False, "required_steps_not_three")
    if list(violations) != ["invalid_step_sequence_content_or_minimum"]:
        return verdict(False, "not_sole_minimum_or_content_violation")
    raw_headers = list(_STEP_FIELD.finditer(response))
    raw_indices = [int(re.search(r"\d+", match.group(0)).group()) for match in raw_headers]
    # Count even empty blocks that parse_steps drops.  Also reject malformed
    # bracket headings so a trailing '[Step 3' cannot hide from both parsers.
    bracket_starts = list(re.finditer(r"\[\s*step\s+\d+\b", response, re.IGNORECASE))
    full_brackets = [match for match in raw_headers if match.group(0).lstrip().startswith("[")]
    if raw_indices != [1, 2] or len(bracket_starts) != len(full_brackets):
        return verdict(False, "raw_step_headers_not_exactly_one_two")
    if len(steps) != 2 or [step.index for step in steps] != [1, 2]:
        return verdict(False, "parsed_steps_not_exactly_one_two")
    reasonings: list[str] = []
    for step in steps:
        body = step.raw_text.strip()
        fields = {}
        for label in ("Reasoning", "Knowledge Used", "Conclusion"):
            matches = list(re.finditer(
                rf"^[ \t]*{re.escape(label)}[ \t]*:", body,
                flags=re.IGNORECASE | re.MULTILINE,
            ))
            if len(matches) != 1:
                return verdict(False, "step_field_count_not_one")
            fields[label] = matches[0]
        # Reproduce the old per-step Reasoning test as well as the field-count
        # check: minimum-step failure can otherwise hide empty/short content.
        if "Reasoning:" not in body:
            return verdict(False, "reasoning_missing_or_short")
        reasoning = re.split(
            r"Knowledge Used:|Conclusion:|Final Answer:", body.split("Reasoning:", 1)[1]
        )[0].strip()
        if len(reasoning) < 20:
            return verdict(False, "reasoning_missing_or_short")
        # In this exception, a lower-case next label must not let its content
        # inflate an otherwise short Reasoning.  The old valid path is intact.
        reasoning_content = re.split(
            r"(?im)^[ \t]*(?:Reasoning|Knowledge Used|Conclusion)[ \t]*:",
            body[fields["Reasoning"].end():],
        )[0].strip()
        if len(reasoning_content) < 20:
            return verdict(False, "reasoning_missing_or_short")
        reasonings.append(" ".join(reasoning_content.casefold().split()))
        conclusion_suffix = body[fields["Conclusion"].end():]
        conclusion = re.split(
            r"(?im)^[ \t]*(?:Reasoning|Knowledge Used|Conclusion)[ \t]*:", conclusion_suffix,
        )[0].strip()
        if not any(char.isalnum() for char in conclusion):
            return verdict(False, "conclusion_empty_or_decoration_only")
        if (not getattr(step, "knowledge_used_valid", False)
                or getattr(step, "unknown_citation_surfaces", ())
                or getattr(step, "knowledge_used_malformed_content", False)):
            return verdict(False, "unknown_or_malformed_kg_citation")
    if reasonings[0] == reasonings[1]:
        return verdict(False, "duplicate_normalized_reasoning")
    known = {int(value) for value in known_passage_ids}
    if _explicit_passage_ids(response) - known:
        return verdict(False, "unknown_explicit_passage_id")
    return verdict(True, "complete_two_steps_only_minimum_shortfall")


@dataclass(frozen=True)
class AnswerFormatObjectiveV2:
    """A numeric-only decomposition suitable for JSON/TensorBoard telemetry."""

    case: str
    answer_component: float
    format_component: float
    text_component: float
    graph_component: float
    answer_signal_applied: bool
    process_allowed: bool
    version: str = OBJECTIVE_VERSION

    @property
    def outcome_component(self) -> float:
        return self.answer_component + self.format_component

    @property
    def trajectory_reward(self) -> float:
        return self.outcome_component + self.text_component + self.graph_component

    def telemetry(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "case": self.case,
            "answer_component": self.answer_component,
            "format_component": self.format_component,
            "outcome_component": self.outcome_component,
            "text_component": self.text_component,
            "graph_component": self.graph_component,
            "trajectory_reward": self.trajectory_reward,
            "answer_signal_applied": self.answer_signal_applied,
            "process_allowed": self.process_allowed,
        }


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def compose_answer_format_objective_v2(
    *,
    trajectory_valid: bool,
    salvage_contract: ShortfallSalvageContractV2,
    outcome_em: float,
    outcome_f1: float,
    text_component: float = 0.0,
    graph_component: float = 0.0,
) -> AnswerFormatObjectiveV2:
    """Compose the fixed experimental objective from already-scored terms.

    ``trajectory_valid`` must be the unchanged source-gate format-v2 result.
    The strict shortfall inspection is used only for invalid-answer salvage. It
    cannot make a valid trajectory invalid or enable Graph/Text computation.
    ``text_component`` and ``graph_component`` are already weighted by the
    frozen source gate; .3(1-alpha)T + .2 alpha G has absolute bound .3.
    """
    if not isinstance(trajectory_valid, bool):
        raise TypeError("trajectory_valid must be a bool")
    if not isinstance(salvage_contract, ShortfallSalvageContractV2):
        raise TypeError("salvage_contract must be a ShortfallSalvageContractV2")
    em = _finite(outcome_em, "outcome_em")
    f1 = _finite(outcome_f1, "outcome_f1")
    if em not in {0.0, 1.0} or not 0.0 <= f1 <= 1.0:
        raise ValueError("canonical outcome EM must be 0/1 and F1 must be in [0,1]")
    if trajectory_valid:
        text = _finite(text_component, "text_component")
        graph = _finite(graph_component, "graph_component")
        if (abs(text) > 0.3 + 1e-12 or abs(graph) > 0.2 + 1e-12
                or abs(text + graph) > SOURCE_PROCESS_ABS_BOUND + 1e-12):
            raise ValueError("source-gated process exceeds frozen magnitude bound")
        return AnswerFormatObjectiveV2(
            case="valid_legacy_preserved",
            answer_component=ANSWER_WEIGHT * (em + F1_WEIGHT * f1),
            format_component=0.0, text_component=text, graph_component=graph,
            answer_signal_applied=True, process_allowed=True,
        )
    if salvage_contract.eligible:
        return AnswerFormatObjectiveV2(
            case="format_invalid_answer_retained",
            answer_component=ANSWER_WEIGHT * (em + F1_WEIGHT * f1),
            format_component=-FORMAT_PENALTY, text_component=0.0,
            graph_component=0.0, answer_signal_applied=True, process_allowed=False,
        )
    return AnswerFormatObjectiveV2(
        case="invalid_answer_unavailable", answer_component=0.0,
        format_component=SEVERE_INVALID_REWARD, text_component=0.0,
        graph_component=0.0, answer_signal_applied=False, process_allowed=False,
    )
