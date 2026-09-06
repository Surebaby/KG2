"""Pure helpers for plan-once, dependency-aware passage retrieval.

This module deliberately does not instantiate models or retrievers.  It turns an
already-frozen, answer-free query plan into search queries, extracts conservative
bridge candidates from first-hop passages, and merges passage lists under a
fixed budget while retaining retrieval provenance.

The helpers are fail-closed: malformed plans, unresolved dependency slots and
residual placeholders raise :class:`DependentRetrievalError`.  Callers can then
fall back to the canonical full-question retrieval result byte-for-byte.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from kgproweight.data.entity_filter import clean_entities
from kgproweight.data.parsers import ENTITY_RE
from kgproweight.kg.entity_linker import passage_title


TARGET_TYPES = frozenset({"relation_graph", "subquery_graph"})

# These fields are never needed to execute an answer-free retrieval plan.  Their
# presence usually means a caller accidentally passed a source/gold record rather
# than the frozen predicted plan.
PROHIBITED_PLAN_KEYS = frozenset(
    {
        "answer",
        "answers",
        "gold_answer",
        "gold_answers",
        "golden_answers",
        "gold_target",
        "supporting_facts",
        "question_decomposition",
        "evidences",
        "paragraph_text",
    }
)

_SLOT_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:\$(?:hop|step)_([1-9]\d*)|#([1-9]\d*)|(?:hop|step)_([1-9]\d*))(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)
_SLOT_FULL_RE = re.compile(
    r"^(?:\$(?:hop|step)_([1-9]\d*)|#([1-9]\d*)|(?:hop|step)_([1-9]\d*))$",
    flags=re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")
_WEAK_BRIDGE_SINGLETONS = frozenset(
    "a an the this that these those he him his she her hers they them their "
    "it its person people man men woman women".split()
)


class DependentRetrievalError(ValueError):
    """The plan cannot be executed without guessing or using hidden data."""


def _clean_text(value: object) -> str:
    return _WS_RE.sub(" ", str(value or "").strip())


def _normalise(value: object) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).split())


def normalize_dependency_ref(value: object) -> str | None:
    """Map ``$hop_N``, ``#N`` and ``step_N`` spellings to ``slot_N``.

    Bare ``hop_N`` is accepted too because relation-graph output slots use that
    spelling even though subjects refer to them as ``$hop_N``.
    """

    match = _SLOT_FULL_RE.fullmatch(_clean_text(value))
    if not match:
        return None
    number = next(group for group in match.groups() if group is not None)
    return f"slot_{int(number)}"


def dependency_refs(value: object) -> List[str]:
    """Return unique canonical dependency slots in textual occurrence order."""

    refs: List[str] = []
    for match in _SLOT_TOKEN_RE.finditer(str(value or "")):
        number = next(group for group in match.groups() if group is not None)
        ref = f"slot_{int(number)}"
        if ref not in refs:
            refs.append(ref)
    return refs


def _canonical_slot_values(slot_values: Mapping[str, Sequence[str] | str]) -> Dict[str, List[str]]:
    canonical: Dict[str, List[str]] = {}
    for raw_slot, raw_values in slot_values.items():
        slot = normalize_dependency_ref(raw_slot)
        if slot is None and re.fullmatch(r"slot_[1-9]\d*", str(raw_slot)):
            slot = str(raw_slot)
        if slot is None:
            raise DependentRetrievalError(f"invalid dependency slot key: {raw_slot!r}")
        values = [raw_values] if isinstance(raw_values, str) else list(raw_values)
        cleaned: List[str] = []
        for value in values:
            text = _clean_text(value)
            if text and text not in cleaned:
                cleaned.append(text)
        if not cleaned:
            raise DependentRetrievalError(f"dependency {raw_slot!r} has no usable values")
        canonical[slot] = cleaned
    return canonical


def replace_dependency_refs(
    template: str,
    slot_values: Mapping[str, Sequence[str] | str],
    *,
    max_variants: int = 2,
) -> List[str]:
    """Substitute all dependency tokens and return deterministic query variants."""

    if max_variants <= 0:
        raise DependentRetrievalError("max_variants must be positive")
    refs = dependency_refs(template)
    if not refs:
        raise DependentRetrievalError("dependent query contains no dependency reference")
    values = _canonical_slot_values(slot_values)
    missing = [ref for ref in refs if ref not in values]
    if missing:
        raise DependentRetrievalError(f"unresolved dependencies: {missing}")

    variants: List[str] = []
    for choices in itertools.product(*(values[ref] for ref in refs)):
        replacements = dict(zip(refs, choices))

        def substitute(match: re.Match[str]) -> str:
            number = next(group for group in match.groups() if group is not None)
            return replacements[f"slot_{int(number)}"]

        rendered = _clean_text(_SLOT_TOKEN_RE.sub(substitute, template))
        if dependency_refs(rendered):
            raise DependentRetrievalError(f"residual dependency reference: {rendered!r}")
        if rendered and rendered not in variants:
            variants.append(rendered)
        if len(variants) >= max_variants:
            break
    if not variants:
        raise DependentRetrievalError("dependency substitution produced no query")
    return variants


def _relation_text(step: Mapping[str, Any]) -> str:
    # A raw P-id is deliberately not used as a textual search query.  It is
    # meaningful to Wikidata execution, but generally meaningless to Wiki18
    # dense/BM25 retrieval.
    relation = _clean_text(step.get("relation_label") or step.get("relation"))
    # Generic Hotpot zero-shot compatibility: some relation-graph generations
    # leak the subquery spelling ``subject >> relation`` into ``subject`` while
    # leaving relation_label empty.  Preserve an explicit relation when one is
    # present; otherwise the right operand is the only model-generated textual
    # relation available.  Never infer a label from the PID.
    raw_subject = _clean_text(step.get("subject"))
    if not relation and ">>" in raw_subject:
        relation = _clean_text(raw_subject.partition(">>")[2])
    if not relation:
        raise DependentRetrievalError("relation-graph step has no textual relation label")
    return relation


def _relation_subject(step: Mapping[str, Any]) -> str:
    """Normalise a relation subject, including generic leaked ``>>`` syntax.

    Hotpot zero-shot plans occasionally put a subquery suffix in the relation
    subject (for example ``$hop_1 >> place of birth``).  The already-frozen KG
    executor handles this generically by keeping the left operand; passage
    retrieval must use the same non-question-specific rule.
    """

    subject = _clean_text(step.get("subject"))
    if ">>" in subject:
        subject = _clean_text(subject.partition(">>")[0])
    if not subject:
        raise DependentRetrievalError("relation-graph step has no subject")
    return subject


def _subquery_parts(step: Mapping[str, Any]) -> Tuple[str, str | None]:
    template = _clean_text(step.get("subquery_template"))
    if not template:
        raise DependentRetrievalError("subquery-graph step has no subquery_template")
    if ">>" not in template:
        return template, None
    left, marker, relation = template.rpartition(">>")
    left, relation = _clean_text(left), _clean_text(relation)
    if not marker or not left or not relation:
        raise DependentRetrievalError(f"invalid canonical subquery: {template!r}")
    return left, relation


def render_root_query(step: Mapping[str, Any], target_type: str) -> str:
    """Render an independent/root plan step as one textual search query."""

    if target_type not in TARGET_TYPES:
        raise DependentRetrievalError(f"unsupported target_type={target_type!r}")
    if target_type == "relation_graph":
        subject = _relation_subject(step)
        query = f"{subject} {_relation_text(step)}"
    else:
        left, relation = _subquery_parts(step)
        query = left if relation is None else f"{left} {relation}"
    if dependency_refs(query) or step.get("dependencies"):
        raise DependentRetrievalError("root step contains a dependency")
    return _clean_text(query)


def instantiate_dependent_queries(
    step: Mapping[str, Any],
    target_type: str,
    slot_values: Mapping[str, Sequence[str] | str],
    *,
    max_variants: int = 2,
) -> List[str]:
    """Render a dependent step after substituting observed bridge entities."""

    if target_type not in TARGET_TYPES:
        raise DependentRetrievalError(f"unsupported target_type={target_type!r}")
    if target_type == "relation_graph":
        subject = _relation_subject(step)
        templates = replace_dependency_refs(subject, slot_values, max_variants=max_variants)
        relation = _relation_text(step)
        return [_clean_text(f"{value} {relation}") for value in templates]

    left, relation = _subquery_parts(step)
    if relation is None:
        return replace_dependency_refs(left, slot_values, max_variants=max_variants)
    values = replace_dependency_refs(left, slot_values, max_variants=max_variants)
    return [_clean_text(f"{value} {relation}") for value in values]


def _find_prohibited_keys(value: Any, *, location: str = "plan") -> List[str]:
    found: List[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            if name in PROHIBITED_PLAN_KEYS:
                found.append(f"{location}.{name}")
            found.extend(_find_prohibited_keys(child, location=f"{location}.{name}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_prohibited_keys(child, location=f"{location}[{index}]"))
    return found


def validate_plan_for_dependent_retrieval(
    plan: Mapping[str, Any],
    target_type: str,
    *,
    max_steps: int = 6,
) -> List[str]:
    """Return structural/leakage errors; an empty list means safe to execute."""

    errors = [f"prohibited_field:{path}" for path in _find_prohibited_keys(plan)]
    if target_type not in TARGET_TYPES:
        return [*errors, f"unsupported_target_type:{target_type}"]
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return [*errors, "missing_steps"]
    if len(steps) > max_steps:
        errors.append(f"too_many_steps:{len(steps)}>{max_steps}")

    produced: set[str] = set()
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping):
            errors.append(f"step_{index}:not_object")
            continue
        slot = normalize_dependency_ref(step.get("output_slot"))
        if slot is None:
            errors.append(f"step_{index}:invalid_output_slot")
        elif slot in produced:
            errors.append(f"step_{index}:duplicate_output_slot:{slot}")

        raw_dependencies = step.get("dependencies") or []
        if not isinstance(raw_dependencies, list):
            errors.append(f"step_{index}:dependencies_not_list")
            raw_dependencies = []
        declared: List[str] = []
        for raw in raw_dependencies:
            dependency = normalize_dependency_ref(raw)
            if dependency is None:
                errors.append(f"step_{index}:invalid_dependency:{raw}")
            elif dependency not in declared:
                declared.append(dependency)

        if target_type == "relation_graph":
            try:
                source = _relation_subject(step)
            except DependentRetrievalError as exc:
                source = ""
                errors.append(f"step_{index}:{exc}")
            try:
                _relation_text(step)
            except DependentRetrievalError as exc:
                errors.append(f"step_{index}:{exc}")
        else:
            source = _clean_text(step.get("subquery_template"))
            try:
                _subquery_parts(step)
            except DependentRetrievalError as exc:
                errors.append(f"step_{index}:{exc}")
        refs = dependency_refs(source)
        for ref in [*declared, *refs]:
            if ref not in produced:
                errors.append(f"step_{index}:unresolved_dependency:{ref}")
        if set(refs) != set(declared):
            errors.append(
                f"step_{index}:dependency_declaration_mismatch:"
                f"declared={sorted(declared)},refs={sorted(refs)}"
            )
        if slot is not None:
            produced.add(slot)
    return list(dict.fromkeys(errors))


def _document_key(passage: Mapping[str, Any] | str) -> str:
    if isinstance(passage, Mapping) and passage.get("id") is not None:
        return f"id:{passage['id']}"
    if isinstance(passage, Mapping):
        title = passage_title(dict(passage))
        text = str(passage.get("contents") or passage.get("text") or "")
    else:
        title, text = "", str(passage)
    blob = f"{_normalise(title)}\n{_normalise(text)}"
    return "text:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def extract_deterministic_bridge_candidates(
    step_query: str,
    passages: Sequence[Mapping[str, Any]],
    *,
    exclude_surfaces: Sequence[str] = (),
    max_docs: int = 5,
    max_candidates: int = 2,
    max_body_chars: int = 1200,
) -> List[Dict[str, Any]]:
    """Extract ranked title/entity candidates with exact passage provenance.

    A passage title contributes three votes and a title-cased body mention one
    vote per document.  Query anchors and caller-supplied surfaces are excluded.
    Ties follow first occurrence and then normalised surface, so output is stable.
    """

    if max_docs <= 0 or max_candidates <= 0 or max_body_chars <= 0:
        raise DependentRetrievalError("bridge extraction limits must be positive")
    excluded = {_normalise(value) for value in exclude_surfaces if _normalise(value)}
    query_key = _normalise(step_query)
    scores: Counter[str] = Counter()
    surfaces: Dict[str, str] = {}
    first_seen: Dict[str, int] = {}
    provenance: Dict[str, List[Dict[str, Any]]] = {}
    occurrence = 0

    def add(surface: object, weight: int, doc: Mapping[str, Any], rank: int, source: str) -> None:
        nonlocal occurrence
        text = _clean_text(surface)
        key = _normalise(text)
        if not key or len(key) < 3 or len(text) > 100:
            return
        if key in _WEAK_BRIDGE_SINGLETONS:
            return
        if key in excluded or re.search(rf"\b{re.escape(key)}\b", query_key):
            return
        occurrence += 1
        scores[key] += weight
        surfaces.setdefault(key, text)
        first_seen.setdefault(key, occurrence)
        event = {
            "document_key": _document_key(doc),
            "rank": rank,
            "location": source,
        }
        if event not in provenance.setdefault(key, []):
            provenance[key].append(event)

    for rank, doc in enumerate(list(passages)[:max_docs], start=1):
        title = passage_title(dict(doc))
        if title:
            add(title, 3, doc, rank, "title")
        contents = str(doc.get("contents") or doc.get("text") or "")
        lines = contents.splitlines()
        body = "\n".join(lines[1:]) if title and lines else contents
        mentions = clean_entities(list(dict.fromkeys(ENTITY_RE.findall(body[:max_body_chars]))))
        for mention in mentions:
            add(mention, 1, doc, rank, "body")

    ordered = sorted(scores, key=lambda key: (-scores[key], first_seen[key], key))
    return [
        {
            "surface": surfaces[key],
            "normalized_surface": key,
            "score": scores[key],
            "provenance": provenance[key],
        }
        for key in ordered[:max_candidates]
    ]


def _as_passage(value: Mapping[str, Any] | str) -> Dict[str, Any]:
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {"contents": str(value)}


def _query_provenance(*, source: str, rank: int, query: str = "", hop_id: str = "") -> Dict[str, Any]:
    value: Dict[str, Any] = {"source": source, "rank": rank}
    if hop_id:
        value["hop_id"] = hop_id
    if query:
        value["query"] = query
        value["query_sha256"] = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return value


def merge_passages_with_provenance(
    original_passages: Sequence[Mapping[str, Any] | str],
    hop_results: Sequence[Mapping[str, Any]],
    *,
    original_quota: int = 6,
    per_hop_quota: int = 2,
    total: int = 10,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Merge original and per-hop results deterministically under fixed quotas.

    Original passages are selected first, followed by each hop in caller order.
    Remaining slots are backfilled from the original tail.  Duplicate documents
    are emitted once while all retrieval paths are retained in the document's
    ``retrieval_provenance`` list.
    """

    if total <= 0 or original_quota < 0 or per_hop_quota < 0:
        raise DependentRetrievalError("invalid passage quotas")
    if original_quota > total:
        raise DependentRetrievalError("original_quota cannot exceed total")

    selected: List[Dict[str, Any]] = []
    index_by_key: Dict[str, int] = {}
    duplicate_count = 0
    source_selected: Counter[str] = Counter()

    def add(raw: Mapping[str, Any] | str, event: Dict[str, Any], *, count_source: str) -> bool:
        nonlocal duplicate_count
        key = _document_key(raw)
        if key in index_by_key:
            duplicate_count += 1
            existing = selected[index_by_key[key]].setdefault("retrieval_provenance", [])
            if event not in existing:
                existing.append(event)
            return False
        if len(selected) >= total:
            return False
        passage = _as_passage(raw)
        passage["retrieval_provenance"] = [event]
        index_by_key[key] = len(selected)
        selected.append(passage)
        source_selected[count_source] += 1
        return True

    original_cursor = 0
    original_added = 0
    while original_cursor < len(original_passages) and original_added < original_quota:
        rank = original_cursor + 1
        if add(
            original_passages[original_cursor],
            _query_provenance(source="original_question", rank=rank),
            count_source="original_prefix",
        ):
            original_added += 1
        original_cursor += 1

    hop_summaries: List[Dict[str, Any]] = []
    for hop_index, result in enumerate(hop_results, start=1):
        hop_id = _clean_text(result.get("hop_id")) or f"hop_{hop_index}"
        raw_query = result.get("query")
        query = _clean_text(raw_query) if isinstance(raw_query, str) else ""
        passages = result.get("passages") or []
        if not query or not isinstance(passages, Sequence) or isinstance(passages, (str, bytes)):
            raise DependentRetrievalError(f"invalid hop result for {hop_id}")
        added = 0
        scanned = 0
        for rank, passage in enumerate(passages, start=1):
            if added >= per_hop_quota or len(selected) >= total:
                break
            scanned += 1
            if add(
                passage,
                _query_provenance(
                    source="dependent_query", rank=rank, query=query, hop_id=hop_id
                ),
                count_source=hop_id,
            ):
                added += 1
        hop_summaries.append(
            {"hop_id": hop_id, "query": query, "selected": added, "scanned": scanned}
        )

    while original_cursor < len(original_passages) and len(selected) < total:
        rank = original_cursor + 1
        add(
            original_passages[original_cursor],
            _query_provenance(source="original_question", rank=rank),
            count_source="original_backfill",
        )
        original_cursor += 1

    telemetry = {
        "total_selected": len(selected),
        "original_quota": original_quota,
        "per_hop_quota": per_hop_quota,
        "total_budget": total,
        "duplicate_paths_merged": duplicate_count,
        "selected_by_source": dict(sorted(source_selected.items())),
        "hop_summaries": hop_summaries,
    }
    return selected, telemetry


__all__ = [
    "DependentRetrievalError",
    "dependency_refs",
    "extract_deterministic_bridge_candidates",
    "instantiate_dependent_queries",
    "merge_passages_with_provenance",
    "normalize_dependency_ref",
    "render_root_query",
    "replace_dependency_refs",
    "validate_plan_for_dependent_retrieval",
]
