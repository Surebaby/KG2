"""ProofKG process reward v2.2.

This is a versioned successor to :mod:`proofkg_process_v2`.  The v2.1
implementation is intentionally left untouched because completed experiments
and their manifests identify ``proofkg-process-v2-1-frozen-1``.

v2.2 keeps the frozen v2.1 reward weights and validity/citation components, but
repairs deterministic answer derivation for comparison questions:

* temporal tails are parsed as values instead of being indexed as triples;
* full dates and bare years are compared with explicit precision intervals;
* multiple values are handled conservatively (overlap => abstain);
* dependent operands are mapped back to the root entity requested by the
  question; and
* same/different questions compare per-operand value sets rather than the first
  and last flattened tails.

No gold answer enters this module.  The generated answer is used only to test
consistency with the answer deterministically derived from the supplied proof.
"""

from __future__ import annotations

import calendar
from datetime import date
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from kgproweight.data.parsers import extract_final_answer, parse_steps


SCORER_VERSION = "proofkg-process-v2-2-frozen-1"

_QUESTION_SAME_DIFF = re.compile(
    r"\b(same|different|common|share|both)\b", re.IGNORECASE
)
_COMPARATIVE = re.compile(
    r"\b(first|earlier|later|older|younger|more|less|higher|lower|longer|"
    r"shorter|before|after|oldest|youngest)\b",
    re.IGNORECASE,
)
_MINIMUM = re.compile(
    r"\b(first|earlier|older|less|lower|shorter|before|oldest)\b",
    re.IGNORECASE,
)
_MONTHS = {name.casefold(): index for index, name in enumerate(calendar.month_name) if name}
_MONTHS.update(
    {name.casefold(): index for index, name in enumerate(calendar.month_abbr) if name}
)
_QID = re.compile(r"^q\d+$", re.IGNORECASE)


def _norm(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _phrase_in(phrase: object, text: object) -> bool:
    needle = _norm(phrase)
    haystack = _norm(text)
    return bool(needle and re.search(rf"(?:^| ){re.escape(needle)}(?: |$)", haystack))


def _dynamic_min_steps(planned_hops: int) -> int:
    return max(2, min(3, int(planned_hops)))


def _hop_number(output_slot: str) -> int:
    match = re.search(r"(\d+)$", str(output_slot))
    return int(match.group(1)) if match else 0


def build_execution_trace_v2_2(
    plan: Mapping[str, Any], execution: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """Build the v2.2 trace while retaining root/provenance information."""

    plan_hops = list(plan.get("hops") or [])
    exec_hops = list(execution.get("hops") or [])
    executed_by_number = {
        int(hop.get("hop_index", -1)): hop
        for hop in exec_hops
        if isinstance(hop, Mapping)
    }
    trace: List[Dict[str, Any]] = []
    for planned in plan_hops:
        slot = str(planned.get("output_slot") or "")
        subject = str(planned.get("subject") or "")
        dependencies = [subject[1:]] if subject.startswith("$") else []
        executed = executed_by_number.get(_hop_number(slot), {})
        trace.append(
            {
                "output_slot": slot,
                "dependencies": dependencies,
                "subject": subject,
                "relation_role": str(
                    planned.get("relation_role") or "answer_operand"
                ),
                "pids": list(planned.get("pids") or []),
                "matched_triples": [
                    tuple(item)
                    for item in (executed.get("matches") or [])
                    if isinstance(item, (list, tuple)) and len(item) == 3
                ],
                "input_entities": list(executed.get("input_entities") or []),
                "output_entities": list(executed.get("output_entities") or []),
            }
        )
    return trace


def _precise_citation(steps: Sequence[Any], required: set) -> Tuple[float, int, int]:
    known = {triple for step in steps for triple in step.cited_triples}
    unknown = {
        str(surface).strip()
        for step in steps
        for surface in step.unknown_citation_surfaces
        if str(surface).strip()
    }
    denominator = len(known) + len(unknown)
    return (
        len(known & required) / denominator if denominator else 0.0,
        len(known),
        len(unknown),
    )


def _dependency_order(trace: List[Dict[str, Any]], steps: Sequence[Any]) -> float:
    first_citation: Dict[str, int] = {}
    for hop in trace:
        hop_triples = set(hop["matched_triples"])
        for index, step in enumerate(steps):
            if hop_triples & set(step.cited_triples):
                first_citation[hop["output_slot"]] = index
                break
    ordered = total = 0
    for hop in trace:
        if not hop["dependencies"]:
            continue
        total += 1
        child = first_citation.get(hop["output_slot"])
        parent = first_citation.get(hop["dependencies"][0])
        if child is not None and parent is not None and parent < child:
            ordered += 1
    return ordered / total if total else 1.0


def _conclusion_grounding(trace: List[Dict[str, Any]], steps: Sequence[Any]) -> float:
    grounded = 0
    for hop in trace:
        hop_triples = set(hop["matched_triples"])
        if not hop_triples:
            continue
        outputs = [
            str(entity.get("label") or entity.get("qid") or "")
            for entity in hop.get("output_entities") or []
            if isinstance(entity, Mapping)
        ]
        outputs = [value for value in outputs if value]
        if not outputs:
            outputs = [str(triple[2]) for triple in hop_triples if triple[2]]
        for step in steps:
            if hop_triples & set(step.cited_triples):
                conclusion = step.intermediate_conclusion or ""
                if any(_phrase_in(output, conclusion) for output in outputs):
                    grounded += 1
                break
    return grounded / max(1, len(trace))


def _question_operator(question: str) -> str:
    if _QUESTION_SAME_DIFF.search(question):
        return "same_different"
    if _COMPARATIVE.search(question):
        return "temporal"
    return "terminal_bridge"


def _parse_date_interval(value: object) -> Optional[Tuple[int, int]]:
    """Parse a supported date into inclusive ordinal bounds.

    A bare year deliberately remains imprecise: its interval spans the whole
    year.  This prevents a coarse ``1904`` value from being cherry-picked over
    a conflicting full date.  Unsupported or invalid strings fail closed.
    """

    raw = str(value or "").strip()
    if not raw:
        return None
    raw = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", raw, flags=re.IGNORECASE)
    raw = raw.replace(",", " ")
    raw = " ".join(raw.split())

    year_match = re.fullmatch(r"([+-]?\d{3,4})", raw)
    if year_match:
        year = int(year_match.group(1))
        if 1 <= year <= 9999:
            return date(year, 1, 1).toordinal(), date(year, 12, 31).toordinal()
        return None

    iso_match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", raw)
    if iso_match:
        try:
            parsed = date(*(int(part) for part in iso_match.groups()))
        except ValueError:
            return None
        ordinal = parsed.toordinal()
        return ordinal, ordinal

    parts = raw.split()
    day: Optional[int] = None
    month: Optional[int] = None
    year: Optional[int] = None
    if len(parts) == 2 and parts[0].casefold() in _MONTHS and parts[1].isdigit():
        month, year = _MONTHS[parts[0].casefold()], int(parts[1])
        try:
            first = date(year, month, 1)
            last = date(year, month, calendar.monthrange(year, month)[1])
        except ValueError:
            return None
        return first.toordinal(), last.toordinal()
    if len(parts) == 3:
        if parts[0].isdigit() and parts[1].casefold() in _MONTHS:
            day, month, year = int(parts[0]), _MONTHS[parts[1].casefold()], int(parts[2])
        elif parts[0].casefold() in _MONTHS and parts[1].isdigit():
            month, day, year = _MONTHS[parts[0].casefold()], int(parts[1]), int(parts[2])
    if day is None or month is None or year is None:
        return None
    try:
        parsed = date(year, month, day)
    except ValueError:
        return None
    ordinal = parsed.toordinal()
    return ordinal, ordinal


def _root_surface(
    operand: Mapping[str, Any], trace_by_slot: Mapping[str, Mapping[str, Any]]
) -> str:
    """Map an answer operand back through ``$hop_N`` to its root subject."""

    current: Mapping[str, Any] = operand
    visited = set()
    while True:
        slot = str(current.get("output_slot") or "")
        if slot in visited:
            return ""
        visited.add(slot)
        subject = str(current.get("subject") or "").strip()
        if subject.startswith("$"):
            parent = trace_by_slot.get(subject[1:])
            if parent is None:
                return ""
            current = parent
            continue
        if subject:
            return subject
        entities = current.get("input_entities") or []
        for entity in entities:
            if not isinstance(entity, Mapping):
                continue
            surface = str(
                entity.get("surface")
                or entity.get("resolved_surface")
                or entity.get("label")
                or ""
            ).strip()
            if surface:
                return surface
        triples = current.get("matched_triples") or []
        return str(triples[0][0]).strip() if triples else ""


def _operand_tail_sets(
    operands: Sequence[Mapping[str, Any]],
) -> List[set]:
    return [
        {
            _norm(triple[2])
            for triple in operand.get("matched_triples") or []
            if len(triple) == 3 and _norm(triple[2])
        }
        for operand in operands
    ]


def _same_different_derivation(
    question: str, operands: Sequence[Mapping[str, Any]]
) -> Tuple[str, int, str]:
    groups = _operand_tail_sets(operands)
    if len(groups) != 2 or any(not group for group in groups):
        return "", 0, "insufficient_operands"

    # These records carry neither preferred-rank/qualifier information nor a
    # complete value-identity map.  Treating a multi-valued nationality/country
    # as an existential set intersection disagrees with the benchmark's
    # canonical-answer semantics for some rows.  Repeated equivalent strings
    # collapse above; multiple *different* values therefore abstain.
    if any(len(group) != 1 for group in groups):
        return "", 0, "ambiguous_multivalue_operand"

    # Exact QID comparisons are meaningful; QID-vs-label comparisons are not.
    qid_modes = [all(_QID.fullmatch(value) for value in group) for group in groups]
    if any(qid_modes) and not all(qid_modes):
        return "", 0, "qid_label_mismatch"

    equal = next(iter(groups[0])) == next(iter(groups[1]))
    asks_different = bool(re.search(r"\bdifferent\b", question, re.IGNORECASE))
    proposition_true = not equal if asks_different else equal
    return ("yes" if proposition_true else "no"), 1, "singleton_equality"


def _temporal_derivation(
    question: str, operands: Sequence[Mapping[str, Any]], trace: Sequence[Mapping[str, Any]]
) -> Tuple[str, int, str]:
    if len(operands) < 2:
        return "", 0, "insufficient_operands"

    intervals: List[Tuple[int, int]] = []
    for operand in operands:
        parsed = [
            interval
            for triple in operand.get("matched_triples") or []
            if len(triple) == 3
            for interval in [_parse_date_interval(triple[2])]
            if interval is not None
        ]
        if not parsed:
            return "", 0, "unparseable_operand"
        # Multiple claims define an uncertainty envelope.  We choose an answer
        # only when the envelopes are strictly ordered.
        intervals.append(
            (min(interval[0] for interval in parsed), max(interval[1] for interval in parsed))
        )

    choose_minimum = bool(_MINIMUM.search(question))
    winners: List[int] = []
    for index, (lower, upper) in enumerate(intervals):
        others = intervals[:index] + intervals[index + 1 :]
        if choose_minimum:
            wins = all(upper < other_lower for other_lower, _ in others)
        else:
            wins = all(lower > other_upper for _, other_upper in others)
        if wins:
            winners.append(index)
    if len(winners) != 1:
        return "", 0, "overlapping_or_tied_intervals"

    by_slot = {str(hop.get("output_slot") or ""): hop for hop in trace}
    root = _root_surface(operands[winners[0]], by_slot)
    if not root:
        return "", 0, "root_unresolved"
    return root, 1, "strict_interval_order"


def _answer_consistency_v2_2(
    question: str,
    trace: List[Dict[str, Any]],
    answer: str,
) -> Tuple[float, int, str, str, str]:
    """Return ``(A, m_A, operator, derived_answer, derivation_status)``."""

    operator = _question_operator(question)
    operands = [
        hop for hop in trace if hop.get("relation_role") == "answer_operand"
    ]
    tails = [
        str(triple[2])
        for hop in operands
        for triple in hop.get("matched_triples") or []
        if len(triple) == 3 and str(triple[2])
    ]

    if operator == "terminal_bridge":
        derived, mask, status = _terminal_derivation(tails)
        consistent = float(
            bool(answer)
            and bool(mask)
            and _answers_equivalent(answer, derived)
        )
        return consistent, mask, operator, derived, status

    if operator == "same_different":
        derived, mask, status = _same_different_derivation(question, operands)
    else:
        derived, mask, status = _temporal_derivation(question, operands, trace)

    consistent = float(
        bool(mask)
        and bool(answer)
        and (_phrase_in(answer, derived) or _phrase_in(derived, answer))
    )
    return consistent, mask, operator, derived, status


def _answers_equivalent(left: object, right: object) -> bool:
    if _phrase_in(left, right) or _phrase_in(right, left):
        return True
    left_date = _parse_date_interval(left)
    right_date = _parse_date_interval(right)
    return bool(left_date is not None and left_date == right_date)


def _terminal_derivation(tails: Sequence[str]) -> Tuple[str, int, str]:
    unique = sorted(
        {str(tail).strip() for tail in tails if str(tail).strip()},
        key=lambda value: (_norm(value), value),
    )
    if not unique:
        return "", 0, "missing_terminal_tail"
    has_qid = any(_QID.fullmatch(value) for value in unique)
    readable = [value for value in unique if not _QID.fullmatch(value)]
    if has_qid:
        # Without a frozen QID-to-label identity map, a mixed QID/label list may
        # denote either aliases or genuinely different values.
        return "", 0, "opaque_qid_terminal_tail"
    if len(readable) == 1:
        return readable[0], 1, "terminal_singleton"

    intervals = [_parse_date_interval(value) for value in readable]
    if all(interval is not None for interval in intervals):
        parsed = [interval for interval in intervals if interval is not None]
        if max(lower for lower, _ in parsed) <= min(upper for _, upper in parsed):
            # Prefer the most precise surface when coarse and fine dates agree.
            derived = min(
                zip(readable, parsed),
                key=lambda item: (item[1][1] - item[1][0], _norm(item[0]), item[0]),
            )[0]
            return derived, 1, "compatible_date_multivalue"
        return "", 0, "conflicting_date_multivalue"
    return "", 0, "ambiguous_terminal_multivalue"


def score_proofkg_v2_2(
    *,
    question: str,
    generation: str,
    kg_triples: Sequence[Sequence[str]],
    execution_trace: List[Dict[str, Any]],
    planned_hops: int,
) -> Dict[str, Any]:
    triples = {
        tuple(str(value).strip() for value in edge)
        for edge in kg_triples
        if len(edge) == 3
    }
    steps = parse_steps(generation, known_kg=list(triples))
    answer = (extract_final_answer(generation) or "").split("\n", 1)[0].strip()

    min_steps = _dynamic_min_steps(planned_hops)
    valid = bool(steps) and len(steps) >= min_steps and extract_final_answer(generation) is not None
    if valid:
        expected = 1
        for step in steps:
            if step.index != expected or not step.raw_text.strip():
                valid = False
                break
            match = re.search(
                r"(?is)\breasoning\s*:\s*(.*?)(?:knowledge used\s*:|conclusion\s*:|$)",
                step.raw_text.strip(),
            )
            if not match or len(match.group(1).strip()) < 20:
                valid = False
                break
            expected += 1

    if not valid:
        return {
            "scorer_version": SCORER_VERSION,
            "score": -1.0,
            "trajectory_valid": False,
            "prediction": answer,
            "n_steps": len(steps),
            "components": {},
        }

    required = {
        triple for hop in execution_trace for triple in hop["matched_triples"]
    }
    n_hops = max(1, len(execution_trace))
    precision, n_known, n_unknown = _precise_citation(steps, required)
    order = _dependency_order(execution_trace, steps)
    grounding = _conclusion_grounding(execution_trace, steps)
    cited = {triple for step in steps for triple in step.cited_triples}
    covered = [
        bool(set(hop["matched_triples"]) & cited) for hop in execution_trace
    ]
    hop_coverage = sum(covered) / n_hops
    answer_score, answer_mask, operator, derived, derivation_status = (
        _answer_consistency_v2_2(question, execution_trace, answer)
    )

    ordered_coverage = hop_coverage * order
    score = (
        0.05 * precision
        + 0.15 * grounding
        + 0.35 * ordered_coverage
        + 0.45 * answer_mask * answer_score
    ) / (0.55 + 0.45 * answer_mask)
    return {
        "scorer_version": SCORER_VERSION,
        "score": float(score),
        "trajectory_valid": True,
        "prediction": answer,
        "n_steps": len(steps),
        "components": {
            "P_precise_citation": precision,
            "H_hop_coverage": hop_coverage,
            "O_dependency_order": order,
            "G_conclusion_grounding": grounding,
            "A_answer_consistency": answer_score,
            "m_A_deterministic": float(answer_mask),
            "C": ordered_coverage,
            "operator": operator,
            "derived_answer": derived,
            "derivation_status": derivation_status,
        },
        "telemetry": {
            "n_known_citations": n_known,
            "n_unknown_citations": n_unknown,
        },
    }
