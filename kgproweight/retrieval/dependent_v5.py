"""Gold-free bridge admission for dependency-aware retrieval experiments.

This module is deliberately separate from :mod:`kgproweight.retrieval.dependent`.
It is a development candidate for a future, versioned retrieval protocol; merely
importing it cannot change the canonical evaluator or the v4 pilot.

The selector uses only a predicted query-plan step, the steps which consume its
output, the rendered query, the original question, and retrieved passages.  It
never reads answers, supporting facts, dataset decompositions, or dataset/QID
specific rules.  Its policy is precision first:

* infer the expected bridge type from the producing relation and consuming
  relation domains;
* reject pre-existing weak/repeated bridge surfaces, subject echoes and aliases,
  and high-confidence type conflicts;
* admit a candidate only when it has relation-local passage support or a
  high-confidence compatible encyclopaedic type;
* return fewer than the requested maximum rather than fill with low-confidence
  candidates.  An empty result explicitly recommends exact caller fallback.

All decisions are returned as JSON-serialisable telemetry so a later experiment
can audit the admission gate without consulting Gold labels.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from kgproweight.kg.kg_filter import _RELATION_LABEL_TO_PID
from kgproweight.kg.qpeg import passage_sentences
from kgproweight.kg.entity_linker import passage_title
from kgproweight.retrieval.bridge import bridge_v2_rejection_reason
from kgproweight.retrieval.dependent import (
    DependentRetrievalError,
    dependency_refs,
    extract_deterministic_bridge_candidates,
    normalize_dependency_ref,
)


SELECTOR_VERSION = "dependent-bridge-admission-v5-development-1"
TARGET_TYPES = frozenset({"relation_graph", "subquery_graph"})

PERSON = "person"
ORGANIZATION = "organization"
LOCATION = "location"
WORK = "creative_work"
EVENT = "event"
CHARACTER = "character"
TAXON = "taxon"
TEMPORAL = "temporal"

_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)\s*")


def _clean(value: object) -> str:
    return _SPACE_RE.sub(" ", str(value or "").strip())


def _norm(value: object) -> str:
    return " ".join(_WORD_RE.findall(_clean(value).casefold()))


def _sha256_text(value: object) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _canonical_entity_surface(value: object) -> str:
    """A strict surface canonicalisation used only for self-echo rejection.

    It intentionally does *not* use fuzzy matching.  Leading ``the``, a common
    Wikipedia disambiguator, and an exact duplicated phrase are safe to remove;
    broad substring rules such as ``Seminole`` ~= ``Black Seminoles`` are not.
    """

    text = _PARENTHETICAL_RE.sub(" ", _clean(value).casefold())
    tokens = _WORD_RE.findall(text)
    if tokens[:1] == ["the"]:
        tokens = tokens[1:]
    if len(tokens) >= 2 and len(tokens) % 2 == 0:
        half = len(tokens) // 2
        if tokens[:half] == tokens[half:]:
            tokens = tokens[:half]
    return " ".join(tokens)


def _relation_from_step(step: Mapping[str, Any], target_type: str) -> Tuple[str, str]:
    """Return the textual relation/query semantics and its source."""

    if target_type not in TARGET_TYPES:
        raise DependentRetrievalError(f"unsupported target_type={target_type!r}")
    if target_type == "relation_graph":
        relation = _clean(step.get("relation_label") or step.get("relation"))
        subject = _clean(step.get("subject"))
        if not relation and ">>" in subject:
            relation = _clean(subject.partition(">>")[2])
        if relation:
            return relation, "relation_label"
        return "", "missing"

    template = _clean(step.get("subquery_template"))
    if ">>" in template:
        relation = _clean(template.rpartition(">>")[2])
        return relation, "subquery_relation" if relation else "missing"
    return template, "natural_subquery" if template else "missing"


def _subject_surfaces(step: Mapping[str, Any], target_type: str, query: str) -> List[str]:
    """Return explicit producer subjects; placeholders are not entity surfaces."""

    result: List[str] = []
    if target_type == "relation_graph":
        subject = _clean(step.get("subject"))
        if ">>" in subject:
            subject = _clean(subject.partition(">>")[0])
        if subject and not dependency_refs(subject):
            result.append(subject)
    else:
        template = _clean(step.get("subquery_template"))
        if ">>" in template:
            subject = _clean(template.rpartition(">>")[0])
            if subject and not dependency_refs(subject):
                result.append(subject)

    # A dependent step has a placeholder in its frozen schema but a concrete
    # subject in the rendered query.  Remove a literal trailing relation only;
    # do not guess where an arbitrary natural-language query's subject ends.
    relation, _ = _relation_from_step(step, target_type)
    clean_query = _clean(query)
    if relation and clean_query.casefold().endswith(relation.casefold()):
        concrete = _clean(clean_query[: -len(relation)])
        if concrete and not dependency_refs(concrete) and concrete not in result:
            result.append(concrete)
    return result


# The rules below are relation-family semantics, not dataset or question IDs.
# Sets are intentionally broad where a predicate legitimately has multiple
# ranges (for example a producer may be a person or an organisation).
_RANGE_RULES: Tuple[Tuple[re.Pattern[str], frozenset[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), frozenset(types), name)
    for pattern, types, name in (
        (r"\b(?:place of birth|place of death|birthplace|death place|location|located in|administrative territorial entity|country|citizenship)\b", (LOCATION,), "range_location"),
        (r"\b(?:date of birth|date of death|inception|publication date|release date|founded|formed|established|century|year)\b", (TEMPORAL,), "range_temporal"),
        (r"\b(?:judge of|part of (?:the )?film series|present in work|original title)\b", (WORK, EVENT), "range_work_event"),
        (r"\b(?:member of sports team|sports team|record label|educated at|employer|manufacturer|league)\b", (ORGANIZATION,), "range_organization"),
        (r"\b(?:genus|species|taxon)\b", (TAXON,), "range_taxon"),
        (r"\b(?:sibling|spouse|child|father|mother|captain|commander|author|co writer|writer|screenwriter|performer|cast member|director)\b", (PERSON,), "range_person"),
        (r"\b(?:producer|creator|founder|founded by|developed by)\b", (PERSON, ORGANIZATION), "range_agent"),
        (r"\bwho\s+(?:played|portrayed|voiced)\b", (PERSON,), "natural_performer"),
        (r"\bwho\s+(?:issued|conceived|founded|created|wrote|produced)\b", (PERSON, ORGANIZATION), "natural_agent"),
    )
)

_DOMAIN_RULES: Tuple[Tuple[re.Pattern[str], frozenset[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), frozenset(types), name)
    for pattern, types, name in (
        (r"\b(?:date of birth|date of death|place of birth|place of death|middle name|occupation|position held|sibling|spouse|child|father|mother|educated at|award received|member of sports team)\b", (PERSON,), "domain_person"),
        (r"\b(?:host|original title|cast member|director|author|screenwriter|performer)\b", (WORK, EVENT), "domain_work_event"),
        (r"\b(?:league)\b", (ORGANIZATION,), "domain_sports_organization"),
        (r"\b(?:vice president|vice-president)\s+of\b", (ORGANIZATION,), "natural_domain_organization"),
        (r"\bwho\s+did\s+(?:#\d+|\$(?:hop|step)_\d+|(?:hop|step)_\d+)\s+(?:play|portray|voice)\b", (PERSON,), "natural_domain_performer"),
        (r"\bevent caused\s+(?:#\d+|\$(?:hop|step)_\d+|(?:hop|step)_\d+)\s+to become\b", (PERSON,), "natural_domain_person"),
    )
)

_WH_RULES: Tuple[Tuple[re.Pattern[str], frozenset[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), frozenset(types), name)
    for pattern, types, name in (
        (r"\bwhat\s+(?:fictional\s+|cartoon\s+)?character\b", (CHARACTER,), "wh_character"),
        (r"\b(?:what|which)\s+(?:film|movie|show|series|novel|book|song|album|episode|work)\b", (WORK,), "wh_work"),
        (r"\b(?:what|which)\s+(?:team|club|company|organisation|organization|university|school|party|label|league)\b", (ORGANIZATION,), "wh_organization"),
        (r"\b(?:which|what)\s+(?:city|country|place|island|district|state)\b|^\s*where\b", (LOCATION,), "wh_location"),
        (r"\b(?:what year|which year|what date)\b|^\s*when\b", (TEMPORAL,), "wh_temporal"),
        (r"\b(?:which\s+)?(?:writer|actor|actress|singer|director|player|person)\b", (PERSON,), "wh_person_noun"),
        (r"^\s*(?:who|whom|whose)\b", (PERSON, ORGANIZATION), "wh_agent"),
    )
)


def _first_matching_types(
    value: str,
    rules: Sequence[Tuple[re.Pattern[str], frozenset[str], str]],
) -> Tuple[set[str], List[str]]:
    for pattern, types, name in rules:
        if pattern.search(value):
            return set(types), [name]
    return set(), []


def _consumer_text(consumer: Mapping[str, Any], target_type: str) -> str:
    relation, source = _relation_from_step(consumer, target_type)
    if source == "natural_subquery":
        return _clean(consumer.get("subquery_template"))
    return relation


def _relevant_consumers(
    step: Mapping[str, Any],
    consumers: Sequence[Mapping[str, Any]],
    target_type: str,
) -> List[Mapping[str, Any]]:
    slot = normalize_dependency_ref(step.get("output_slot"))
    if slot is None:
        return list(consumers)
    relevant: List[Mapping[str, Any]] = []
    for consumer in consumers:
        source = (
            _clean(consumer.get("subject"))
            if target_type == "relation_graph"
            else _clean(consumer.get("subquery_template"))
        )
        declared = {
            normalize_dependency_ref(value)
            for value in (consumer.get("dependencies") or [])
        }
        refs = set(dependency_refs(source)) | {value for value in declared if value}
        if slot in refs:
            relevant.append(consumer)
    # The API contract calls these "consumers".  If a caller has already
    # filtered them but uses an unrecognised slot spelling, retain them rather
    # than silently discarding all downstream evidence.
    return relevant or list(consumers)


def infer_expected_bridge_profile(
    step: Mapping[str, Any],
    consumers: Sequence[Mapping[str, Any]],
    target_type: str,
) -> Dict[str, Any]:
    """Infer a conservative bridge range/domain profile from plan text only."""

    relation, relation_source = _relation_from_step(step, target_type)
    relation_types, relation_rules = _first_matching_types(relation, _RANGE_RULES)
    step_text = (
        _clean(step.get("subquery_template"))
        if target_type == "subquery_graph"
        else relation
    )
    wh_types, wh_rules = _first_matching_types(step_text, _WH_RULES)

    if relation_types:
        producer_types = set(relation_types)
        producer_rules = list(relation_rules)
        if wh_types:
            overlap = producer_types & wh_types
            if overlap:
                producer_types = overlap
                producer_rules.extend(wh_rules)
    else:
        producer_types = set(wh_types)
        producer_rules = list(wh_rules)

    relevant = _relevant_consumers(step, consumers, target_type)
    consumer_type_sets: List[set[str]] = []
    consumer_evidence: List[Dict[str, Any]] = []
    for index, consumer in enumerate(relevant, start=1):
        text = _consumer_text(consumer, target_type)
        types, rules = _first_matching_types(text, _DOMAIN_RULES)
        if not types and target_type == "subquery_graph":
            # Some natural consumers describe the placeholder's semantic role
            # rather than using a canonical relation suffix.
            types, rules = _first_matching_types(
                _clean(consumer.get("subquery_template")), _DOMAIN_RULES
            )
        if types:
            consumer_type_sets.append(types)
        consumer_evidence.append({
            "consumer_index": index,
            "text": text,
            "types": sorted(types),
            "rules": rules,
        })

    consumer_types: set[str] = set()
    consumer_conflict = False
    if consumer_type_sets:
        consumer_types = set(consumer_type_sets[0])
        for types in consumer_type_sets[1:]:
            intersection = consumer_types & types
            if not intersection:
                consumer_conflict = True
                consumer_types.clear()
                break
            consumer_types = intersection

    profile_conflict = consumer_conflict
    if producer_types and consumer_types:
        expected = producer_types & consumer_types
        if not expected:
            profile_conflict = True
            expected = set()
        source = "producer_consumer_intersection"
    elif producer_types:
        expected = set(producer_types)
        source = "producer_range"
    elif consumer_types:
        expected = set(consumer_types)
        source = "consumer_domain"
    else:
        expected = set()
        source = "unknown"

    relation_key = _clean(relation).casefold()
    expected_pid = _RELATION_LABEL_TO_PID.get(relation_key)
    observed_pid = _clean(step.get("pid")).upper()
    pid_label_conflict = bool(
        expected_pid and observed_pid and expected_pid.upper() != observed_pid
    )

    return {
        "relation_text": relation,
        "relation_source": relation_source,
        "producer_types": sorted(producer_types),
        "producer_rules": producer_rules,
        "consumer_types": sorted(consumer_types),
        "consumer_evidence": consumer_evidence,
        "expected_types": sorted(expected),
        "expected_type_source": source,
        "profile_conflict": profile_conflict,
        "consumer_conflict": consumer_conflict,
        "observed_pid": observed_pid or None,
        "relation_label_expected_pid": expected_pid,
        "pid_label_conflict": pid_label_conflict,
        "pid_used_for_type_inference": False,
    }


_TITLE_DISAMBIGUATOR_RULES: Tuple[Tuple[re.Pattern[str], str, str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), kind, name)
    for pattern, kind, name in (
        (r"\((?:[^)]*\b)?(?:film|television series|tv series|novel|book|album|song|video game|play)\b[^)]*\)", WORK, "title_work_disambiguator"),
        (r"\((?:fictional|cartoon|comics?) character\)", CHARACTER, "title_character_disambiguator"),
        (r"\((?:writer|actor|actress|singer|politician|footballer|athlete|director|producer)\)", PERSON, "title_person_disambiguator"),
    )
)

_LEAD_TYPE_RULES: Tuple[Tuple[re.Pattern[str], str, str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), kind, name)
    for pattern, kind, name in (
        (r"\b(?:is|was)\s+(?:an?|the)\s+(?:[a-z-]+\s+){0,6}(?:fictional|cartoon|comic|animated)\s+character\b", CHARACTER, "lead_character"),
        (r"\b(?:is|was)\s+(?:an?|the)\s+(?:[a-z0-9-]+\s+){0,6}(?:film|movie|television series|tv series|novel|book|album|song|video game|stage play|television show)\b", WORK, "lead_work"),
        (r"\b(?:is|was|are|were)\s+(?:an?|the)\s+(?:[a-z-]+\s+){0,6}(?:organization|organisation|company|corporation|agency|institution|university|school|college|political party|sports team|football club|hockey team|band|record label|opera company)\b", ORGANIZATION, "lead_organization"),
        (r"\b(?:is|was)\s+(?:an?|the)\s+(?:[a-z-]+\s+){0,6}(?:city|town|village|country|island|district|municipality|province|state|region|palace|castle)\b", LOCATION, "lead_location"),
        (r"\b(?:is|was)\s+(?:an?|the)\s+(?:[a-z0-9-]+\s+){0,6}(?:battle|election|tournament|championship|festival|war|ceremony|competition)\b", EVENT, "lead_event"),
        (r"\b(?:is|was)\s+(?:an?|the)\s+(?:[a-z-]+\s+){0,7}(?:species|genus|taxon)\b", TAXON, "lead_taxon"),
        (r"\(born\b|\b(?:is|was)\s+(?:an?|the)\s+(?:[a-z-]+\s+){0,7}(?:actor|actress|writer|author|singer|musician|politician|footballer|athlete|director|producer|journalist|scientist|physicist|artist|composer|presenter|historian|rapper)\b", PERSON, "lead_person"),
    )
)


def classify_candidate_from_title_lead(
    surface: str,
    passages: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Classify only from an exact candidate-title article and its lead."""

    canonical = _canonical_entity_surface(surface)
    types: set[str] = set()
    evidence: List[Dict[str, Any]] = []
    for rank, passage in enumerate(passages, start=1):
        title = passage_title(dict(passage))
        if not title or _canonical_entity_surface(title) != canonical:
            continue
        contents = str(passage.get("contents") or passage.get("text") or "")
        body_lines = contents.splitlines()
        lead = _clean(" ".join(body_lines[1:]) if len(body_lines) > 1 else contents)[:900]
        for pattern, kind, rule in _TITLE_DISAMBIGUATOR_RULES:
            if pattern.search(title):
                types.add(kind)
                evidence.append({"rule": rule, "passage_rank": rank})
        for pattern, kind, rule in _LEAD_TYPE_RULES:
            if pattern.search(lead):
                types.add(kind)
                evidence.append({"rule": rule, "passage_rank": rank})
    return {
        "types": sorted(types),
        "confidence": "high" if types else "unknown",
        "type_evidence_events": evidence,
    }


_ALIAS_CUE_RE = re.compile(
    r"\b(?:also known as|known as|known professionally as|professionally known as|"
    r"stage name|pen name|birth name|born as)\b",
    re.IGNORECASE,
)
_ALIAS_RELATION_RE = re.compile(
    r"\b(?:alias|also known as|known as|nickname|stage name|pen name|birth name)\b",
    re.IGNORECASE,
)


def _contains_norm_phrase(container: str, phrase: str) -> bool:
    if not container or not phrase:
        return False
    return f" {phrase} " in f" {container} "


def _is_subject_alias(
    candidate: str,
    subjects: Sequence[str],
    passages: Sequence[Mapping[str, Any]],
) -> Tuple[bool, List[Dict[str, Any]]]:
    candidate_norm = _norm(candidate)
    evidence: List[Dict[str, Any]] = []
    if not candidate_norm or not subjects:
        return False, evidence
    for rank, passage in enumerate(passages, start=1):
        title = passage_title(dict(passage))
        if _canonical_entity_surface(title) != _canonical_entity_surface(candidate):
            continue
        contents = str(passage.get("contents") or passage.get("text") or "")
        lead = _clean(contents)[:900]
        lead_norm = _norm(lead)
        if not _ALIAS_CUE_RE.search(lead):
            continue
        for subject in subjects:
            subject_norm = _norm(subject)
            if (
                _contains_norm_phrase(lead_norm, subject_norm)
                and _contains_norm_phrase(lead_norm, candidate_norm)
            ):
                evidence.append({
                    "rule": "lead_explicit_alias_cue",
                    "passage_rank": rank,
                    "passage_title": title,
                })
    return bool(evidence), evidence


_CUE_RULES: Tuple[Tuple[re.Pattern[str], Tuple[re.Pattern[str], ...], str], ...] = tuple(
    (
        re.compile(relation_pattern, re.IGNORECASE),
        tuple(re.compile(cue, re.IGNORECASE) for cue in cues),
        name,
    )
    for relation_pattern, cues, name in (
        (r"\b(?:performer|cast member|who .*play|who .*portray|who .*voice)\b", (r"\bplay(?:ed|s|ing)?\b", r"\bportray(?:ed|s|ing)?\b", r"\bvoic(?:e|ed|es|ing)\b", r"\bstarr(?:ed|ing)?\b", r"\bcast\b"), "cue_performer"),
        (r"\b(?:author|writer|co writer|screenwriter)\b", (r"\bauthor(?:ed|s)?\b", r"\bwrit(?:e|es|ten|ing|er)\b", r"\bco[- ]?writ"), "cue_writer"),
        (r"\bproducer\b", (r"\bproduc(?:e|ed|er|ers|ing|tion)\b",), "cue_producer"),
        (r"\b(?:creator|created|who .*create|conceived)\b", (r"\bcreat(?:e|ed|es|or|ing)\b", r"\bconceiv(?:e|ed|es|ing)\b"), "cue_creator"),
        (r"\b(?:founder|founded by|who .*found)\b", (r"\bfound(?:ed|er|ers|ing)?\b",), "cue_founder"),
        (r"\b(?:educated at|study|studied)\b", (r"\beducat(?:e|ed|ion)\b", r"\battend(?:ed|s|ing)?\b", r"\bgraduat(?:e|ed|ion)\b", r"\bstud(?:y|ied)\b"), "cue_education"),
        (r"\b(?:member of sports team|member of|league)\b", (r"\bmember\b", r"\bplay(?:ed|s|ing)?\s+for\b", r"\bsign(?:ed|s|ing)?\s+(?:with|for)\b", r"\bteam\b", r"\bleague\b"), "cue_membership"),
        (r"\b(?:place of death|death place)\b", (r"\bdied\b", r"\bdeath\b"), "cue_death_place"),
        (r"\b(?:place of birth|birthplace)\b", (r"\bborn\b", r"\bbirthplace\b"), "cue_birth_place"),
        (r"\b(?:location|located in|administrative territorial entity|country)\b", (r"\blocat(?:e|ed|ion)\b", r"\bbased\b", r"\bfounded\s+in\b", r"\bcountry\b", r"\bcity\b"), "cue_location"),
        (r"\b(?:sibling|brother|sister)\b", (r"\bsibling\b", r"\bbrother\b", r"\bsister\b"), "cue_sibling"),
        (r"\b(?:father|mother|child|spouse)\b", (r"\bfather\b", r"\bmother\b", r"\bson\b", r"\bdaughter\b", r"\bchild\b", r"\bmarried\b", r"\bspouse\b"), "cue_family"),
        (r"\b(?:judge of|judge on)\b", (r"\bjudg(?:e|ed|es|ing)\b",), "cue_judge"),
        (r"\b(?:captain|commander)\b", (r"\bcaptain(?:ed|s|cy)?\b", r"\bcommand(?:ed|er|ers|ing)?\b"), "cue_command"),
        (r"\b(?:manufacturer|developed by)\b", (r"\bmanufactur(?:e|ed|er|ers|ing)\b", r"\bdevelop(?:ed|er|ers|ing)\b", r"\bbuilt by\b"), "cue_manufacturer"),
        (r"\b(?:issued)\b", (r"\bissu(?:e|ed|es|ing)\b",), "cue_issued"),
        (r"\b(?:named after)\b", (r"\bnamed after\b",), "cue_named_after"),
        (r"\b(?:part of|series)\b", (r"\bpart of\b", r"\bseries\b"), "cue_part_of"),
        (r"\b(?:record label)\b", (r"\brecord label\b", r"\bsigned (?:to|with)\b",), "cue_record_label"),
    )
)


def _relation_cues(relation_text: str) -> Tuple[Tuple[re.Pattern[str], ...], str | None]:
    for relation_pattern, cues, name in _CUE_RULES:
        if relation_pattern.search(relation_text):
            return cues, name
    return (), None


def relation_local_support(
    candidate: str,
    relation_text: str,
    subjects: Sequence[str],
    passages: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Find candidate + relation-cue evidence, preferring same-sentence subject."""

    cues, cue_family = _relation_cues(relation_text)
    if not cues:
        return {"strength": 0, "cue_family": None, "support_events": []}
    candidate_norm = _norm(candidate)
    subject_norms = [_norm(value) for value in subjects if _norm(value)]
    evidence: List[Dict[str, Any]] = []
    best = 0
    for rank, passage in enumerate(passages, start=1):
        title = passage_title(dict(passage))
        title_norm = _norm(title)
        document_text_norm = _norm(passage.get("contents") or passage.get("text"))
        document_has_subject = any(
            _contains_norm_phrase(document_text_norm, subject)
            or _canonical_entity_surface(title) == _canonical_entity_surface(subject)
            for subject in subjects
            if _norm(subject)
        )
        for sentence_index, sentence in enumerate(passage_sentences(passage)):
            sentence_norm = _norm(sentence)
            if not _contains_norm_phrase(sentence_norm, candidate_norm):
                continue
            matched_cue = next((cue.pattern for cue in cues if cue.search(sentence)), None)
            if not matched_cue:
                continue
            same_sentence_subject = any(
                _contains_norm_phrase(sentence_norm, subject)
                for subject in subject_norms
            )
            # Natural subqueries do not always expose a clean subject surface.
            # Candidate+cue is still useful, but receives the lower tier.
            strength = 2 if same_sentence_subject else 1
            if document_has_subject and strength < 2:
                strength = 1
            best = max(best, strength)
            evidence.append({
                "passage_rank": rank,
                "passage_id": str(passage.get("id") or f"rank-{rank}"),
                "passage_title": title,
                "sentence_index": sentence_index,
                "sentence_sha256": _sha256_text(sentence),
                "cue": matched_cue,
                "same_sentence_subject": same_sentence_subject,
                "document_has_subject": document_has_subject,
            })
    return {"strength": best, "cue_family": cue_family, "support_events": evidence}


def _specificity(surface: str, all_surfaces: Sequence[str]) -> int:
    """Prefer a full name only when another candidate is its token subset."""

    tokens = _norm(surface).split()
    token_set = set(tokens)
    has_shorter_variant = any(
        other != surface
        and set(_norm(other).split()) < token_set
        for other in all_surfaces
        if _norm(other)
    )
    return len(tokens) + (2 if has_shorter_variant else 0)


def select_bridge_candidates_v5(
    *,
    step: Mapping[str, Any],
    consumers: Sequence[Mapping[str, Any]],
    target_type: str,
    query: str,
    question: str,
    passages: Sequence[Mapping[str, Any]],
    max_candidates: int = 2,
    max_docs: int = 10,
    max_body_chars: int = 1200,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return admitted bridges and complete Gold-free decision telemetry.

    ``max_candidates`` is a cap, never a quota.  The function returns an empty
    list when every candidate is unsafe or weak, and reports
    ``fallback_recommended=True`` for the caller to preserve Arm A exactly.
    """

    if target_type not in TARGET_TYPES:
        raise DependentRetrievalError(f"unsupported target_type={target_type!r}")
    if (
        not 1 <= max_candidates <= 2
        or not 1 <= max_docs <= 10
        or max_body_chars <= 0
    ):
        raise DependentRetrievalError("selector limits must be positive")
    if not isinstance(consumers, Sequence) or isinstance(consumers, (str, bytes)):
        raise DependentRetrievalError("consumers must be a sequence of plan steps")
    if not passages:
        telemetry = {
            "selector_version": SELECTOR_VERSION,
            "gold_access": False,
            "query": _clean(query),
            "query_sha256": _sha256_text(_clean(query)),
            "question_sha256": _sha256_text(_clean(question)),
            "profile": infer_expected_bridge_profile(step, consumers, target_type),
            "subjects": _subject_surfaces(step, target_type, query),
            "raw_candidate_count": 0,
            "raw_candidate_inspect_cap": 12,
            "source_document_limit": max_docs,
            "source_documents_inspected": 0,
            "max_body_chars": max_body_chars,
            "accepted_count": 0,
            "accepted_surfaces": [],
            "candidate_decisions": [],
            "all_rejected": True,
            "fallback_recommended": True,
            "fallback_reason": "no_passages",
            "max_candidates": max_candidates,
            "low_confidence_fill_allowed": False,
        }
        return [], telemetry

    profile = infer_expected_bridge_profile(step, consumers, target_type)
    relation_text = str(profile["relation_text"])
    subjects = _subject_surfaces(step, target_type, query)
    # Frozen inspection budget: admission sees at most the v4 extractor's first
    # twelve candidates.  This is deliberately not tuned per dataset or row.
    provisional_cap = 12
    raw_candidates = extract_deterministic_bridge_candidates(
        _clean(query),
        passages,
        exclude_surfaces=subjects,
        max_docs=max_docs,
        max_candidates=provisional_cap,
        max_body_chars=max_body_chars,
    )
    raw_surfaces = [str(row.get("surface") or "") for row in raw_candidates]
    expected_types = set(profile.get("expected_types") or [])

    decisions: List[Dict[str, Any]] = []
    admitted: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
    for raw_index, raw in enumerate(raw_candidates):
        surface = _clean(raw.get("surface"))
        reasons: List[str] = []
        weak_reason = bridge_v2_rejection_reason(surface)
        if weak_reason:
            reasons.append(weak_reason)

        canonical = _canonical_entity_surface(surface)
        strict_subject_echo = bool(
            canonical
            and any(canonical == _canonical_entity_surface(subject) for subject in subjects)
        )
        if strict_subject_echo:
            reasons.append("strict_subject_echo")

        question_norm = _norm(question)
        candidate_norm = _canonical_entity_surface(surface)
        original_question_phrase = bool(
            candidate_norm and _contains_norm_phrase(question_norm, candidate_norm)
        )
        if original_question_phrase:
            reasons.append("original_question_phrase")

        alias_echo, alias_evidence = _is_subject_alias(surface, subjects, passages)
        alias_relation_exempt = bool(_ALIAS_RELATION_RE.search(relation_text))
        if alias_echo and not alias_relation_exempt:
            reasons.append("explicit_subject_alias")

        candidate_type = classify_candidate_from_title_lead(surface, passages)
        observed_types = set(candidate_type["types"])
        type_match = bool(expected_types and observed_types & expected_types)
        type_conflict = bool(expected_types and observed_types and not type_match)
        if type_conflict:
            reasons.append("high_confidence_type_conflict")

        local = relation_local_support(surface, relation_text, subjects, passages)
        if profile.get("profile_conflict"):
            reasons.append("producer_consumer_type_conflict")

        hard_reject = bool(reasons)
        if not hard_reject:
            if int(local["strength"]) >= 1 and type_match:
                tier = 3
                basis = "relation_local_type_compatible"
            elif type_match:
                tier = 2
                basis = "high_confidence_type_match"
            elif int(local["strength"]) >= 1 and not type_conflict:
                tier = 1
                basis = "relation_local_type_unknown"
            else:
                tier = 0
                basis = "insufficient_gold_free_support"
                reasons.append(basis)
        else:
            tier = 0
            basis = "rejected"

        decision = {
            "raw_rank": raw_index + 1,
            "surface": surface,
            "normalized_surface": _norm(surface),
            "base_score": int(raw.get("score") or 0),
            "base_provenance": deepcopy(list(raw.get("provenance") or [])),
            "title_body_corroboration_count": len({
                str(event.get("location") or "")
                for event in (raw.get("provenance") or [])
                if str(event.get("location") or "") in {"title", "body"}
            }),
            "candidate_type": candidate_type,
            "expected_types": sorted(expected_types),
            "type_match": type_match,
            "type_conflict": type_conflict,
            "relation_local_support": local,
            "strict_subject_echo": strict_subject_echo,
            "original_question_phrase": original_question_phrase,
            "alias_echo": alias_echo,
            "alias_relation_exempt": alias_relation_exempt,
            "alias_evidence": alias_evidence,
            "admission_tier": tier,
            "admission_basis": basis,
            "decision": "accept" if tier > 0 else "reject",
            "reasons": list(dict.fromkeys(reasons)),
        }
        decisions.append(decision)
        if tier > 0:
            candidate = deepcopy(dict(raw))
            candidate["admission"] = {
                "selector_version": SELECTOR_VERSION,
                "tier": tier,
                "basis": basis,
                "expected_types": sorted(expected_types),
                "candidate_types": sorted(observed_types),
                "relation_support_strength": int(local["strength"]),
            }
            sort_key = (
                -tier,
                -int(local["strength"]),
                -int(decision["title_body_corroboration_count"]),
                -_specificity(surface, raw_surfaces),
                -int(raw.get("score") or 0),
                raw_index,
                _norm(surface),
            )
            admitted.append((sort_key, candidate))

    admitted.sort(key=lambda item: item[0])
    accepted = [candidate for _, candidate in admitted[:max_candidates]]
    accepted_surfaces = [str(row.get("surface") or "") for row in accepted]
    telemetry = {
        "selector_version": SELECTOR_VERSION,
        "gold_access": False,
        "target_type": target_type,
        "query": _clean(query),
        "query_sha256": _sha256_text(_clean(query)),
        "question_sha256": _sha256_text(_clean(question)),
        "profile": profile,
        "subjects": subjects,
        "raw_candidate_count": len(raw_candidates),
        "raw_candidate_inspect_cap": provisional_cap,
        "source_document_limit": max_docs,
        "source_documents_inspected": min(len(passages), max_docs),
        "max_body_chars": max_body_chars,
        "accepted_count": len(accepted),
        "accepted_surfaces": accepted_surfaces,
        "candidate_decisions": decisions,
        "all_rejected": not accepted,
        "fallback_recommended": not accepted,
        "fallback_reason": "all_candidates_rejected" if not accepted else None,
        "max_candidates": max_candidates,
        "low_confidence_fill_allowed": False,
    }
    return accepted, telemetry


__all__ = [
    "SELECTOR_VERSION",
    "classify_candidate_from_title_lead",
    "infer_expected_bridge_profile",
    "relation_local_support",
    "select_bridge_candidates_v5",
]
