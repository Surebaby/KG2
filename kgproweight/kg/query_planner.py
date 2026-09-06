"""Conservative, gold-free query plans for per-question proof KG retrieval.

This first planner recognises common multi-hop QA templates and abstains on
unknown wording.  A plan specifies relation PIDs and operations; it never reads
answers, supporting facts or dataset decomposition annotations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re
from typing import Dict, List, Sequence

from kgproweight.kg.entity_linker import extract_mentions


PLANNER_VERSION = "rule-query-plan-2"


@dataclass(frozen=True)
class QueryHop:
    subject: str
    pids: Sequence[str]
    output_slot: str
    relation_role: str


@dataclass
class QueryPlan:
    question_sha256: str
    planner_version: str = PLANNER_VERSION
    recognized: bool = False
    operation: str = "abstain"
    anchors: List[str] = field(default_factory=list)
    hops: List[QueryHop] = field(default_factory=list)
    confidence: str = "none"
    abstain_reason: str = "unrecognized_question_template"

    def to_dict(self) -> Dict:
        value = asdict(self)
        value["hops"] = [asdict(hop) for hop in self.hops]
        return value


_GENERIC = {
    "all", "both", "film", "films", "movie", "movies", "country", "nationality",
    "first", "same", "earlier", "later", "director", "directors", "which", "what",
}


def _sha(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()


def _clean_anchor(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip(" ,.?\"'"))
    value = re.sub(r"^both\s+", "", value, flags=re.I)
    value = re.sub(
        r"^(?:both\s+)?(?:(?:the\s+)?(?:films?|movies?)\s*:?\s*|director\s+of\s+(?:the\s+)?(?:film\s+)?)",
        "", value, flags=re.I,
    )
    value = re.sub(r"\s+films?$", "", value, flags=re.I)
    return value.strip()


def _fallback_anchors(question: str, max_n: int = 4) -> List[str]:
    anchors: List[str] = []
    for mention in extract_mentions(question, max_n=max_n + 3):
        clean = _clean_anchor(mention)
        if clean and clean.lower() not in _GENERIC and clean not in anchors:
            anchors.append(clean)
        if len(anchors) >= max_n:
            break
    return anchors


def _two_entities(question: str) -> List[str]:
    patterns = (
        r"(?:born\s+(?:first|earlier|later)|(?:released|published)\s+earlier|came\s+out\s+earlier|who\s+is\s+(?:older|younger)|director\s+(?:born\s+)?(?:earlier|later)|director\s+who\s+is\s+(?:older|younger)),?\s+(.+?)\s+or\s+(.+?)(?:\?|$)",
        r"(?:films?|movies?)\s+(.+?)\s+(?:and|or)\s+(.+?)(?:\?|$)",
        r"between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"(?:earlier|later|older|younger),?\s+(.+?)\s+or\s+(.+?)(?:\?|$)",
        r"(?:out of|are|were|do)\s+(.+?)\s+(?:and|or)\s+(.+?)(?:\?|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.I)
        if not match:
            continue
        values = [_clean_anchor(match.group(1)), _clean_anchor(match.group(2))]
        # Trim trailing comparison clauses from the second entity.
        values[1] = _clean_anchor(re.split(
            r"\s+(?:born|from|located|share|have|both|of the same|in the same)\b",
            values[1], maxsplit=1, flags=re.I,
        )[0])
        values[0] = _clean_anchor(values[0])
        if all(value and value.lower() not in _GENERIC for value in values):
            return values
    return _fallback_anchors(question, max_n=2)


def _plan_repeated(anchors: Sequence[str], first_pid: str | None, final_pid: str, operation: str) -> List[QueryHop]:
    hops: List[QueryHop] = []
    for index, anchor in enumerate(anchors, start=1):
        subject = anchor
        if first_pid:
            bridge = f"entity_{index}"
            hops.append(QueryHop(subject, [first_pid], bridge, "bridge"))
            subject = f"${bridge}"
        hops.append(QueryHop(subject, [final_pid], f"value_{index}", "answer_operand"))
    return hops


def _composition_plan(plan: QueryPlan, anchor: str, pids: Sequence[str]) -> QueryPlan:
    clean_anchor = _clean_anchor(anchor)
    if not clean_anchor:
        return plan
    hops: List[QueryHop] = []
    subject = clean_anchor
    for index, pid in enumerate(pids, start=1):
        slot = f"hop_{index}"
        hops.append(QueryHop(subject, [pid], slot, "bridge" if index < len(pids) else "answer"))
        subject = f"${slot}"
    plan.recognized, plan.operation, plan.anchors = True, "compose_relation", [clean_anchor]
    plan.hops, plan.confidence, plan.abstain_reason = hops, "high", ""
    return plan


def plan_question(question: str) -> QueryPlan:
    q = str(question).strip()
    lower = q.lower()
    plan = QueryPlan(question_sha256=_sha(q))

    director_context = bool(re.search(
        r"directors?\s+of|director\s+of|has the director|have the directors|films? have (?:the )?directors",
        lower,
    ))
    temporal_compare = bool(re.search(r"born (?:first|earlier|later)|\bolder\b|\byounger\b", lower))
    release_compare = bool(re.search(r"released earlier|came out earlier|published earlier", lower))
    same_birth_place = bool(re.search(r"born (?:in )?the same place|same place of birth", lower))
    same_nationality = bool(re.search(r"same (?:nationality|country)|share the same nationality|both from the same country", lower))
    started_compare = bool(re.search(r"started first|founded first|formed first|established first", lower))

    if director_context and temporal_compare:
        anchors = _two_entities(q)
        if len(anchors) == 2:
            plan.recognized, plan.operation, plan.anchors = True, "argmin_or_argmax_date", anchors
            plan.hops = _plan_repeated(anchors, "P57", "P569", plan.operation)
            plan.confidence, plan.abstain_reason = "high", ""
            return plan
    if director_context and same_nationality:
        anchors = _two_entities(q)
        if len(anchors) == 2:
            plan.recognized, plan.operation, plan.anchors = True, "compare_equality", anchors
            plan.hops = _plan_repeated(anchors, "P57", "P27", plan.operation)
            plan.confidence, plan.abstain_reason = "high", ""
            return plan
    if release_compare:
        anchors = _two_entities(q)
        if len(anchors) == 2:
            plan.recognized, plan.operation, plan.anchors = True, "argmin_publication_date", anchors
            plan.hops = _plan_repeated(anchors, None, "P577", plan.operation)
            plan.confidence, plan.abstain_reason = "high", ""
            return plan
    if same_birth_place:
        anchors = _two_entities(q)
        if len(anchors) == 2:
            plan.recognized, plan.operation, plan.anchors = True, "compare_equality", anchors
            plan.hops = _plan_repeated(anchors, None, "P19", plan.operation)
            plan.confidence, plan.abstain_reason = "high", ""
            return plan
    if temporal_compare:
        anchors = _two_entities(q)
        if len(anchors) == 2:
            plan.recognized, plan.operation, plan.anchors = True, "argmin_or_argmax_date", anchors
            plan.hops = _plan_repeated(anchors, None, "P569", plan.operation)
            plan.confidence, plan.abstain_reason = "high", ""
            return plan
    if started_compare:
        anchors = _two_entities(q)
        if len(anchors) == 2:
            plan.recognized, plan.operation, plan.anchors = True, "argmin_inception", anchors
            plan.hops = _plan_repeated(anchors, None, "P571", plan.operation)
            plan.confidence, plan.abstain_reason = "high", ""
            return plan
    if same_nationality:
        anchors = _two_entities(q)
        if len(anchors) == 2:
            pid = "P17" if re.search(r"located|cities|places|schools?|colleges?", lower) else "P27"
            plan.recognized, plan.operation, plan.anchors = True, "compare_equality", anchors
            plan.hops = _plan_repeated(anchors, None, pid, plan.operation)
            plan.confidence, plan.abstain_reason = "medium", ""
            return plan

    patterns = (
        (r"nationality is the director of (?:film )?(.+?)(?:\?|$)", ("P57", "P27")),
        (r"performer of (?:song )?(.+?)\s*['’]s birthday", ("P175", "P569")),
        (r"date of death of the director of (?:film )?(.+?)(?:\?|$)", ("P57", "P570")),
        (r"date of death of (.+?)['’]s father", ("P22", "P570")),
        (r"father of the director of (?:film )?(.+?)(?:\?|$)", ("P57", "P22")),
        (r"child of the director of (?:film )?(.+?)(?:\?|$)", ("P57", "P40")),
        (r"director of (?:film )?(.+?) graduate from", ("P57", "P69")),
        (r"director of (?:film )?(.+?) born", ("P57", "P569")),
        (r"date of birth of the director of (?:film )?(.+?)(?:\?|$)", ("P57", "P569")),
        (r"date of birth of the founder of (?:magazine )?(.+?)(?:\?|$)", ("P112", "P569")),
        (r"performer of (?:song )?(.+?) born", ("P175", "P19")),
        (r"child-in-law of (.+?)(?:\?|$)", ("P40", "P26")),
        (r"(?:who is )?(.+?)['’]s uncle", ("P22", "P3373")),
        (r"maternal grandfather of (.+?)(?:\?|$)", ("P25", "P22")),
        (r"(?:who is )?(.+?)['’]s maternal grandfather", ("P25", "P22")),
        (r"paternal grandfather of (.+?)(?:\?|$)", ("P22", "P22")),
        (r"(?:who is )?(.+?)['’]s paternal grandfather", ("P22", "P22")),
        (r"paternal grandmother of (.+?)(?:\?|$)", ("P22", "P25"), "compose_relation"),
        (r"(?:who is )?(.+?)['’]s paternal grandmother", ("P22", "P25")),
        (r"mother of the director of (?:film )?(.+?)(?:\?|$)", ("P57", "P25"), "compose_relation"),
        (r"spouse of the (.+?) performer", ("P175", "P26"), "compose_relation"),
        (r"when was the (?:institute|university|company) that owned (.+?) founded", ("P127", "P571"), "compose_relation"),
    )
    for entry in patterns:
        pattern, pids = entry[:2]
        match = re.search(pattern, q, flags=re.I)
        if not match:
            continue
        return _composition_plan(plan, match.group(1), pids)

    return plan
