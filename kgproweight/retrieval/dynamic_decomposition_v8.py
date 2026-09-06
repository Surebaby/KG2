"""Gold-free pure helpers for dynamic subquestion decomposition v8.

This module deliberately contains no model, retriever, dataset, or scorer
access.  It implements only the contracts that can be checked mechanically:

* one-line natural-language retrieval-query parsing;
* one-line concise subanswer/sentinel parsing;
* deterministic, fail-closed lexical binding of a subanswer to one of the
  already retrieved q1 documents;
* static/dynamic q2 action construction and deterministic fallback selection;
* the frozen ``root6 + q1-novel2 + q2-novel2`` passage allocation, with
  backfill from root ranks 7--10.

The lexical binder is intentionally *not* a semantic support verifier.  Its
output states this limitation explicitly.  Public helpers consume only their
documented string and retrieved-passage projections; extra answer-label,
supporting-fact, decomposition, or other Gold-derived fields are neither
consumed nor emitted.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


QUERY_CONTRACT_VERSION = "dynamic-decomposition-v8-query-contract-1"
SUBANSWER_CONTRACT_VERSION = "dynamic-decomposition-v8-subanswer-contract-1"
PROVENANCE_BINDER_VERSION = "dynamic-decomposition-v8-unique-doc-surface-binder-1"
Q2_ACTION_POLICY_VERSION = "dynamic-decomposition-v8-q2-action-policy-1"
PASSAGE_MERGE_POLICY_VERSION = "dynamic-decomposition-v8-root6-q1n2-q2n2-1"

NO_RELEVANT_ANSWER = "NO_RELEVANT_ANSWER"
NO_VERIFIED_SUBANSWER = "NO_VERIFIED_SUBANSWER"
MIN_QUERY_CHARS = 3
MAX_QUERY_CHARS = 256
MAX_SUBANSWER_CHARS = 256
EXPECTED_TOP_K = 10
FINAL_PASSAGE_BUDGET = 10
BOUND_EXCERPT_MAX_CHARS = 512
PASSAGE_TEXT_MAX_CHARS = 1200

_SPACE_RE = re.compile(r" +")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
_UNRESOLVED_PLACEHOLDER_RE = re.compile(
    r"(?:"
    r"(?<![A-Za-z0-9_])(?:\$(?:hop|step)_?[1-9]\d*|#[1-9]\d*|"
    r"(?:hop|step)_[1-9]\d*)(?![A-Za-z0-9_])"
    r"|\{\{?\s*(?:answer|entity|subject|object|hop|step)(?:_[1-9]\d*)?\s*\}?\}"
    r"|<\s*(?:answer|entity|subject|object|hop|step)(?:_[1-9]\d*)?\s*>"
    r")",
    flags=re.IGNORECASE,
)
_FORBIDDEN_GOLD_MARKER_RE = re.compile(
    r"\b(?:gold(?:en)?[_ -]?(?:answer|answers|target)|supporting[_ -]?facts?|"
    r"question[_ -]?decomposition)\b",
    flags=re.IGNORECASE,
)
_NULL_LIKE_SURFACES = frozenset(
    {
        "unknown",
        "n a",
        "na",
        "none",
        "null",
        "not known",
        "not available",
        "not enough information",
        "cannot determine",
        "insufficient information",
    }
)
_BOOLEAN_LIKE_SURFACES = frozenset({"yes", "no", "true", "false", "both", "neither"})
_SAFE_PASSAGE_COPY_FIELDS = ("source", "corpus_path", "is_multimodal")
_SCORE_FIELDS = ("rerank_score", "score", "retrieval_score")
_STATIC_ACTION_FIELDS = frozenset(
    {
        "policy_version",
        "gold_access",
        "slot",
        "controller_state_sha256",
        "response_sha256",
        "proposal_valid",
        "proposal_query",
        "parse_error",
        "selected_query",
        "selection_source",
        "used_fallback",
        "fallback_reason",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class DynamicDecompositionV8Error(ValueError):
    """Invalid caller-owned input or an internally inconsistent v8 action."""


class QueryParseError(DynamicDecompositionV8Error):
    """A controller response violates the one-line query contract."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class SubanswerParseError(DynamicDecompositionV8Error):
    """A reader response violates the concise subanswer contract."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(payload)


def _has_forbidden_control(value: str) -> bool:
    return any(unicodedata.category(char) in {"Cc", "Cf", "Zl", "Zp"} for char in value)


def _clean_one_line(value: str, *, error_type: type[ValueError]) -> str:
    if "\n" in value or "\r" in value:
        raise error_type("multiline", "response must contain exactly one line")
    if _has_forbidden_control(value):
        raise error_type("control_character", "response contains a control character")
    normalized = unicodedata.normalize("NFKC", value).strip()
    return _SPACE_RE.sub(" ", normalized)


def _word_surface(value: object) -> str:
    clean = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", clean, flags=re.UNICODE).split())


def _validate_context_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicDecompositionV8Error(f"{field} must be a non-empty string")
    if "\n" in value or "\r" in value or _has_forbidden_control(value):
        raise DynamicDecompositionV8Error(f"{field} must contain exactly one safe line")
    return value


def _reject_structured_wrapper(clean: str, *, error_type: type[ValueError]) -> None:
    if clean.startswith("```") or clean.endswith("```"):
        raise error_type("markdown_wrapper", "Markdown wrappers are not allowed")
    if len(clean) >= 2 and (clean[0], clean[-1]) in {("{", "}"), ("[", "]")}:
        try:
            parsed = json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            return
        if isinstance(parsed, (dict, list)):
            raise error_type("structured_wrapper", "JSON/list wrappers are not allowed")


def parse_query_response(
    response_text: str,
    *,
    previous_queries: Sequence[str] = (),
) -> dict[str, Any]:
    """Parse one controller response under the frozen mechanical contract.

    The function does not claim to determine whether the query is semantically
    single-hop.  It checks only the one-line surface contract, unresolved
    placeholders, forbidden Gold-field markers, length, and normalized repeats.
    """

    if not isinstance(response_text, str) or not response_text.strip():
        raise QueryParseError("empty_response", "query response must be non-empty text")
    try:
        query = _clean_one_line(response_text, error_type=QueryParseError)
    except TypeError as exc:  # defensive: custom error construction must remain explicit
        raise QueryParseError("invalid_response", str(exc)) from exc
    if not query:
        raise QueryParseError("empty_response", "query response must be non-empty text")
    _reject_structured_wrapper(query, error_type=QueryParseError)
    if not MIN_QUERY_CHARS <= len(query) <= MAX_QUERY_CHARS:
        raise QueryParseError(
            "query_length",
            f"query must contain {MIN_QUERY_CHARS}--{MAX_QUERY_CHARS} characters",
        )
    if _UNRESOLVED_PLACEHOLDER_RE.search(query):
        raise QueryParseError("unresolved_placeholder", "query contains an unresolved placeholder")
    if _FORBIDDEN_GOLD_MARKER_RE.search(query):
        raise QueryParseError("forbidden_gold_marker", "query contains a forbidden Gold-field marker")
    normalized_query = _word_surface(query)
    if not normalized_query:
        raise QueryParseError("no_lexical_content", "query has no lexical content")

    if isinstance(previous_queries, (str, bytes)) or not isinstance(previous_queries, Sequence):
        raise DynamicDecompositionV8Error("previous_queries must be a sequence of strings")
    normalized_previous: list[str] = []
    for index, previous in enumerate(previous_queries):
        previous = _validate_context_text(previous, field=f"previous_queries[{index}]")
        normalized_previous.append(_word_surface(previous))
    if normalized_query in normalized_previous:
        raise QueryParseError("repeated_query", "query repeats a previous query after normalization")

    return {
        "contract_version": QUERY_CONTRACT_VERSION,
        "gold_access": False,
        "validation_scope": "surface_contract_not_single_hop_semantics",
        "query": query,
        "normalized_query": normalized_query,
        "query_sha256": _sha256_text(query),
        "previous_query_count": len(normalized_previous),
    }


def parse_subanswer_response(response_text: str) -> dict[str, Any]:
    """Parse a one-line concise answer or the exact abstention sentinel.

    There are no model-reported type, citation, or abstention fields.  Harmless
    horizontal padding is normalized, while line breaks, wrappers, and case- or
    spelling-coerced sentinels are rejected.
    """

    if not isinstance(response_text, str) or not response_text.strip():
        raise SubanswerParseError("empty_response", "subanswer response must be non-empty text")
    answer = _clean_one_line(response_text, error_type=SubanswerParseError)
    if not answer:
        raise SubanswerParseError("empty_response", "subanswer response must be non-empty text")
    _reject_structured_wrapper(answer, error_type=SubanswerParseError)
    if len(answer) > MAX_SUBANSWER_CHARS:
        raise SubanswerParseError(
            "answer_too_long", f"subanswer exceeds {MAX_SUBANSWER_CHARS} characters"
        )
    if answer.casefold() == NO_RELEVANT_ANSWER.casefold() and answer != NO_RELEVANT_ANSWER:
        raise SubanswerParseError(
            "sentinel_not_exact", f"abstention sentinel must be exactly {NO_RELEVANT_ANSWER}"
        )
    if answer == NO_RELEVANT_ANSWER:
        return {
            "contract_version": SUBANSWER_CONTRACT_VERSION,
            "gold_access": False,
            "abstained": True,
            "answer": None,
            "response_sha256": _sha256_text(answer),
        }
    if _FORBIDDEN_GOLD_MARKER_RE.search(answer):
        raise SubanswerParseError(
            "forbidden_gold_marker", "subanswer contains a forbidden Gold-field marker"
        )
    if not _word_surface(answer):
        raise SubanswerParseError("no_lexical_content", "subanswer has no lexical content")
    return {
        "contract_version": SUBANSWER_CONTRACT_VERSION,
        "gold_access": False,
        "abstained": False,
        "answer": answer,
        "response_sha256": _sha256_text(answer),
    }


def _document_identity(passage: Mapping[str, Any], rank: int) -> tuple[str, str]:
    observed: list[str] = []
    for key in ("id", "doc_id", "document_id"):
        if key not in passage or passage[key] is None:
            continue
        raw = passage[key]
        if isinstance(raw, bool) or not isinstance(raw, (str, int)):
            raise DynamicDecompositionV8Error(
                f"passage {rank} {key} must be a string or integer"
            )
        value = str(raw)
        if not value or value != value.strip() or len(value) > 256:
            raise DynamicDecompositionV8Error(f"passage {rank} has an invalid {key}")
        if _has_forbidden_control(value):
            raise DynamicDecompositionV8Error(
                f"passage {rank} {key} contains a control character"
            )
        observed.append(value)
    if not observed:
        raise DynamicDecompositionV8Error(f"passage {rank} has no stable document id")
    if len(set(observed)) != 1:
        raise DynamicDecompositionV8Error(f"passage {rank} has conflicting document ids")
    return f"id:{observed[0]}", observed[0]


def _document_text(passage: Mapping[str, Any], rank: int) -> str:
    contents = passage.get("contents")
    text = passage.get("text")
    if isinstance(contents, str) and contents.strip():
        if isinstance(text, str) and text.strip() and text != contents:
            raise DynamicDecompositionV8Error(
                f"passage {rank} has conflicting contents/text fields"
            )
        return contents
    if isinstance(text, str) and text.strip():
        return text
    raise DynamicDecompositionV8Error(f"passage {rank} has no non-empty text")


def _document_title(passage: Mapping[str, Any], text: str, rank: int) -> str:
    explicit = passage.get("title")
    if explicit is not None and not isinstance(explicit, str):
        raise DynamicDecompositionV8Error(f"passage {rank} title must be a string")
    if isinstance(explicit, str) and explicit.strip():
        return unicodedata.normalize("NFKC", explicit).strip().strip('"')
    first_line = text.splitlines()[0] if text else ""
    return unicodedata.normalize("NFKC", first_line).strip().strip('"')


def _document_score(passage: Mapping[str, Any], rank: int) -> float | None:
    for key in _SCORE_FIELDS:
        if key not in passage or passage[key] is None:
            continue
        raw = passage[key]
        if isinstance(raw, bool):
            raise DynamicDecompositionV8Error(f"passage {rank} {key} must be finite numeric")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise DynamicDecompositionV8Error(
                f"passage {rank} {key} must be finite numeric"
            ) from exc
        if not math.isfinite(value):
            raise DynamicDecompositionV8Error(f"passage {rank} {key} must be finite numeric")
        return value
    return None


def _prepare_passages(
    passages: Sequence[Mapping[str, Any]],
    *,
    role: str,
    require_unique: bool,
) -> list[dict[str, Any]]:
    if isinstance(passages, (str, bytes)) or not isinstance(passages, Sequence):
        raise DynamicDecompositionV8Error(f"{role} passages must be a sequence")
    if len(passages) != EXPECTED_TOP_K:
        raise DynamicDecompositionV8Error(
            f"{role} passages must contain exactly {EXPECTED_TOP_K} documents"
        )
    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, raw in enumerate(passages, start=1):
        if not isinstance(raw, Mapping):
            raise DynamicDecompositionV8Error(f"{role} passage {rank} must be an object")
        key, document_id = _document_identity(raw, rank)
        if require_unique and key in seen:
            raise DynamicDecompositionV8Error(f"duplicate document identity in {role}: {key}")
        seen.add(key)
        original_text = _document_text(raw, rank)
        title = _document_title(raw, original_text, rank)
        text = original_text[:PASSAGE_TEXT_MAX_CHARS]
        projection: dict[str, Any] = {"id": document_id, "contents": text}
        if title:
            projection["title"] = title
        for field in _SAFE_PASSAGE_COPY_FIELDS:
            if field in raw and isinstance(raw[field], (str, bool, int, float)):
                projection[field] = deepcopy(raw[field])
        prompt_visible = {key: projection[key] for key in ("id", "title", "contents") if key in projection}
        prepared.append(
            {
                "key": key,
                "document_id": document_id,
                "rank": rank,
                "title": title,
                "text": text,
                "score": _document_score(raw, rank),
                "projection": projection,
                "prompt_sha256": _canonical_json_sha256(prompt_visible),
            }
        )
    return prepared


def project_top10_passages_for_prompt(
    passages: Sequence[Mapping[str, Any]],
    *,
    role: str,
) -> list[dict[str, str]]:
    """Project one retrieved top-10 to the only document fields models may see."""

    if not isinstance(role, str) or not role.strip():
        raise DynamicDecompositionV8Error("passage projection role must be non-empty")
    prepared = _prepare_passages(passages, role=role, require_unique=True)
    return [
        {
            "doc_id": str(document["document_id"]),
            "title": str(document["title"]),
            "text": str(document["text"]),
        }
        for document in prepared
    ]


def _surface_pattern(answer: str) -> re.Pattern[str]:
    normalized = unicodedata.normalize("NFKC", answer).casefold()
    if not normalized:
        raise DynamicDecompositionV8Error("cannot locate an empty answer")
    return re.compile(
        rf"(?<!\w){re.escape(normalized)}(?!\w)",
        flags=re.UNICODE,
    )


def _numeric_compound_subspan(
    text: str,
    match: re.Match[str],
    normalized_answer: str,
) -> bool:
    """Reject a numeric answer embedded in a decimal, ratio, or date token.

    Ordinary sentence punctuation remains valid: ``1990.`` and ``42,`` match,
    while ``38`` in ``38.5`` and either component of ``12/38`` do not.  This is
    a lexical boundary rule, not model-reported answer-type inference.
    """

    if not normalized_answer or not (
        normalized_answer[0].isdigit() or normalized_answer[-1].isdigit()
    ):
        return False
    start, end = match.span()
    if normalized_answer[0].isdigit() and start >= 1:
        # A bare ``38`` is not the same lexical surface as ``-38``, ``+38`` or
        # the leading-decimal form ``.38``.  Sentence punctuation remains
        # accepted because it follows rather than precedes the number.
        if text[start - 1] in "+-−.":
            return True
    if normalized_answer[0].isdigit() and start >= 2:
        if text[start - 1] in ".,/-" and text[start - 2].isdigit():
            return True
    if normalized_answer[-1].isdigit() and end + 1 < len(text):
        if text[end] in ".,/-" and text[end + 1].isdigit():
            return True
    return False


def _casefold_with_offset_map(text: str) -> tuple[str, str, list[int], list[int]]:
    """Return NFKC text, folded text, and folded-character origin spans.

    Unicode casefolding may expand one source character (for example ``ß``
    to ``ss`` or ``İ`` to ``i\N{COMBINING DOT ABOVE}``).  Regex spans in the
    folded string therefore cannot be used directly to slice the evidence
    unit.  ``starts[i]``/``ends[i]`` map each folded character back to the
    half-open span of its NFKC source character.
    """

    normalized = unicodedata.normalize("NFKC", text)
    folded_parts: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for source_index, character in enumerate(normalized):
        folded_character = character.casefold()
        if not folded_character:
            continue
        folded_parts.append(folded_character)
        starts.extend([source_index] * len(folded_character))
        ends.extend([source_index + 1] * len(folded_character))
    folded = "".join(folded_parts)
    if len(folded) != len(starts) or len(folded) != len(ends):
        raise DynamicDecompositionV8Error("internal casefold offset-map length mismatch")
    return normalized, folded, starts, ends


def _map_folded_match_to_source(
    *,
    normalized_text: str,
    folded_text: str,
    starts: Sequence[int],
    ends: Sequence[int],
    match: re.Match[str],
    normalized_answer: str,
) -> tuple[int, int] | None:
    """Map one folded regex match back to a verified NFKC source span."""

    folded_start, folded_end = match.span()
    if (
        folded_start < 0
        or folded_end <= folded_start
        or folded_end > len(folded_text)
        or folded_end > len(starts)
        or folded_end > len(ends)
    ):
        return None
    source_start = int(starts[folded_start])
    source_end = int(ends[folded_end - 1])
    if not 0 <= source_start < source_end <= len(normalized_text):
        return None
    source_surface = normalized_text[source_start:source_end]
    if source_surface.casefold() != normalized_answer:
        # This also rejects a regex match that begins/ends inside the expansion
        # of one source character rather than covering an exact lexical span.
        return None
    if folded_text[folded_start:folded_end] != normalized_answer:
        return None
    return source_start, source_end


def _evidence_units(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    seen: set[str] = set()
    unit_index = 0
    for line in str(document["text"]).splitlines() or [str(document["text"])]:
        clean_line = _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", line).strip())
        if not clean_line:
            continue
        for raw_sentence in _SENTENCE_SPLIT_RE.split(clean_line):
            sentence = raw_sentence.strip()
            if not sentence or sentence in seen:
                continue
            seen.add(sentence)
            unit_index += 1
            units.append({"location": "text", "unit_index": unit_index, "text": sentence})
    title = str(document.get("title") or "").strip()
    if title and title not in seen:
        unit_index += 1
        units.append({"location": "title", "unit_index": unit_index, "text": title})
    return units


def _bounded_excerpt(text: str, start: int, end: int) -> tuple[str, int, int]:
    if len(text) <= BOUND_EXCERPT_MAX_CHARS:
        return text, start, end
    half = (BOUND_EXCERPT_MAX_CHARS - (end - start)) // 2
    left = max(0, start - max(0, half))
    right = min(len(text), left + BOUND_EXCERPT_MAX_CHARS)
    left = max(0, right - BOUND_EXCERPT_MAX_CHARS)
    return text[left:right], start - left, end - left


def _empty_binding(*, reason: str, parsed: bool, abstained: bool = False) -> dict[str, Any]:
    return {
        "binder_version": PROVENANCE_BINDER_VERSION,
        "gold_access": False,
        "verification_scope": "unique_document_surface_locality_not_semantic_entailment",
        "parsed": parsed,
        "abstained": abstained,
        "verified": False,
        "reason": reason,
        "verified_answer": None,
        "matching_document_count": 0,
        "matching_unit_count": 0,
        "supporting_document_key": None,
        "supporting_doc_id": None,
        "supporting_doc_rank": None,
        "supporting_sentence": None,
        "supporting_sentence_sha256": None,
        "support_location": None,
        "support_unit_index": None,
        "surface_match_mode": None,
        "surface_start": None,
        "surface_end": None,
        "bound_evidence_excerpt": None,
        "bound_evidence_excerpt_sha256": None,
        "supporting_document_prompt_sha256": None,
    }


def bind_subanswer_provenance(
    parsed_subanswer: Mapping[str, Any],
    *,
    q1_query: str,
    q1_passages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind an answer surface to exactly one q1 top-10 document.

    Multiple mentions inside the same document are resolved deterministically;
    occurrence in more than one document is rejected because the model supplied
    no citation.  This is lexical locality only, never semantic entailment.
    """

    q1_query = _validate_context_text(q1_query, field="q1_query")
    documents = _prepare_passages(q1_passages, role="q1", require_unique=True)
    if not isinstance(parsed_subanswer, Mapping):
        raise DynamicDecompositionV8Error("parsed_subanswer must be an object")
    if parsed_subanswer.get("contract_version") != SUBANSWER_CONTRACT_VERSION:
        raise DynamicDecompositionV8Error("parsed_subanswer has an unexpected contract version")
    if parsed_subanswer.get("gold_access") is not False:
        raise DynamicDecompositionV8Error("parsed_subanswer gold_access must be false")
    if parsed_subanswer.get("abstained") is True:
        return _empty_binding(reason="reader_sentinel", parsed=True, abstained=True)
    answer = parsed_subanswer.get("answer")
    if not isinstance(answer, str) or not answer:
        raise DynamicDecompositionV8Error("non-abstaining parsed_subanswer lacks answer text")

    answer_key = _word_surface(answer)
    if answer_key in _NULL_LIKE_SURFACES:
        return _empty_binding(reason="null_like_answer", parsed=True)
    if answer_key in _BOOLEAN_LIKE_SURFACES:
        return _empty_binding(reason="non_extractive_boolean_answer", parsed=True)
    query_key = _word_surface(q1_query)
    if answer_key and re.search(rf"(?<!\w){re.escape(answer_key)}(?!\w)", query_key):
        return _empty_binding(reason="q1_surface_echo", parsed=True)

    pattern = _surface_pattern(answer)
    normalized_answer = unicodedata.normalize("NFKC", answer).casefold()
    matches_by_document: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    total_matches = 0
    for document in documents:
        document_matches: list[dict[str, Any]] = []
        for unit in _evidence_units(document):
            normalized_unit, folded_unit, offset_starts, offset_ends = (
                _casefold_with_offset_map(str(unit["text"]))
            )
            for match in pattern.finditer(folded_unit):
                if _numeric_compound_subspan(folded_unit, match, normalized_answer):
                    continue
                source_span = _map_folded_match_to_source(
                    normalized_text=normalized_unit,
                    folded_text=folded_unit,
                    starts=offset_starts,
                    ends=offset_ends,
                    match=match,
                    normalized_answer=normalized_answer,
                )
                if source_span is None:
                    continue
                source_start, source_end = source_span
                excerpt, excerpt_start, excerpt_end = _bounded_excerpt(
                    normalized_unit, source_start, source_end
                )
                excerpt_surface = excerpt[excerpt_start:excerpt_end]
                if excerpt_surface.casefold() != normalized_answer:
                    # Verified output must never carry an excerpt/span which no
                    # longer contains the matched answer after normalization.
                    continue
                document_matches.append(
                    {
                        **unit,
                        "text": normalized_unit,
                        "surface_start": source_start,
                        "surface_end": source_end,
                        "excerpt": excerpt,
                        "excerpt_start": excerpt_start,
                        "excerpt_end": excerpt_end,
                    }
                )
        if document_matches:
            matches_by_document.append((document, document_matches))
            total_matches += len(document_matches)

    if not matches_by_document:
        result = _empty_binding(reason="answer_surface_not_found", parsed=True)
        result["matching_unit_count"] = 0
        return result
    if len(matches_by_document) != 1:
        result = _empty_binding(reason="answer_surface_ambiguous_across_documents", parsed=True)
        result["matching_document_count"] = len(matches_by_document)
        result["matching_unit_count"] = total_matches
        return result

    document, matches = matches_by_document[0]
    chosen = min(
        matches,
        key=lambda row: (
            row["location"] == "title",
            _word_surface(row["text"]) == answer_key,
            int(row["unit_index"]),
            int(row["surface_start"]),
        ),
    )
    sentence = str(chosen["text"])
    excerpt = str(chosen["excerpt"])
    return {
        "binder_version": PROVENANCE_BINDER_VERSION,
        "gold_access": False,
        "verification_scope": "unique_document_surface_locality_not_semantic_entailment",
        "parsed": True,
        "abstained": False,
        "verified": True,
        "reason": "verified_unique_document_surface",
        "verified_answer": answer,
        "matching_document_count": 1,
        "matching_unit_count": len(matches),
        "supporting_document_key": document["key"],
        "supporting_doc_id": document["document_id"],
        "supporting_doc_rank": document["rank"],
        "supporting_sentence": sentence,
        "supporting_sentence_sha256": _sha256_text(sentence),
        "support_location": chosen["location"],
        "support_unit_index": chosen["unit_index"],
        "surface_match_mode": "nfkc_casefold_exact_with_boundaries",
        "surface_start": chosen["surface_start"],
        "surface_end": chosen["surface_end"],
        "bound_evidence_excerpt": excerpt,
        "bound_evidence_excerpt_sha256": _sha256_text(excerpt),
        "bound_excerpt_surface_start": chosen["excerpt_start"],
        "bound_excerpt_surface_end": chosen["excerpt_end"],
        "supporting_document_prompt_sha256": document["prompt_sha256"],
    }


def parse_and_bind_subanswer(
    response_text: str,
    *,
    q1_query: str,
    q1_passages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate caller inputs, then parse and bind one model response."""

    # Validate caller-owned inputs before classifying a model response.  A
    # broken top-k input must not be hidden as an ordinary reader parse failure.
    _validate_context_text(q1_query, field="q1_query")
    _prepare_passages(q1_passages, role="q1", require_unique=True)
    try:
        parsed = parse_subanswer_response(response_text)
    except SubanswerParseError as exc:
        result = _empty_binding(reason=f"parse_error:{exc.code}", parsed=False)
        result["response_sha256"] = _sha256_text(response_text) if isinstance(response_text, str) else None
        return result
    result = bind_subanswer_provenance(
        parsed,
        q1_query=q1_query,
        q1_passages=q1_passages,
    )
    result["response_sha256"] = parsed["response_sha256"]
    return result


def build_static_q2_state(*, original_question: str, q1_query: str) -> dict[str, Any]:
    """Return the fixed observation-blind state used by B and ineligible C."""

    question = _validate_context_text(original_question, field="original_question")
    q1 = _validate_context_text(q1_query, field="q1_query")
    return {
        "state_version": Q2_ACTION_POLICY_VERSION,
        "mode": "q2_no_verified_subanswer",
        "gold_access": False,
        "original_question": question,
        "q1_query": q1,
        "verified_subanswer": NO_VERIFIED_SUBANSWER,
    }


def _safe_bound_evidence(binding: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "verified_answer",
        "supporting_document_key",
        "supporting_doc_id",
        "supporting_doc_rank",
        "supporting_sentence_sha256",
        "support_location",
        "support_unit_index",
        "bound_evidence_excerpt",
        "bound_evidence_excerpt_sha256",
        "supporting_document_prompt_sha256",
    }
    missing = sorted(key for key in required if binding.get(key) is None)
    if missing:
        raise DynamicDecompositionV8Error(f"verified binding lacks provenance fields: {missing}")
    return {key: deepcopy(binding[key]) for key in sorted(required)}


def build_dynamic_q2_state(
    *,
    original_question: str,
    q1_query: str,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the allowlisted answer/observation-conditioned q2 state."""

    question = _validate_context_text(original_question, field="original_question")
    q1 = _validate_context_text(q1_query, field="q1_query")
    if not isinstance(binding, Mapping):
        raise DynamicDecompositionV8Error("binding must be an object")
    if binding.get("binder_version") != PROVENANCE_BINDER_VERSION:
        raise DynamicDecompositionV8Error("binding has an unexpected binder version")
    if binding.get("gold_access") is not False or binding.get("verified") is not True:
        raise DynamicDecompositionV8Error("dynamic q2 state requires a verified Gold-free binding")
    safe_evidence = _safe_bound_evidence(binding)
    return {
        "state_version": Q2_ACTION_POLICY_VERSION,
        "mode": "q2_dynamic",
        "gold_access": False,
        "original_question": question,
        "q1_query": q1,
        "verified_subanswer": safe_evidence["verified_answer"],
        "bound_evidence": safe_evidence,
    }


def build_static_q2_action(
    response_text: str,
    *,
    original_question: str,
    q1_query: str,
) -> dict[str, Any]:
    """Construct static q2 action, falling back to the original question."""

    state = build_static_q2_state(original_question=original_question, q1_query=q1_query)
    response_sha256 = _sha256_text(response_text) if isinstance(response_text, str) else None
    try:
        proposal = parse_query_response(
            response_text,
            previous_queries=(original_question, q1_query),
        )
    except QueryParseError as exc:
        action = {
            "policy_version": Q2_ACTION_POLICY_VERSION,
            "gold_access": False,
            "slot": "q2_static",
            "controller_state_sha256": _canonical_json_sha256(state),
            "response_sha256": response_sha256,
            "proposal_valid": False,
            "proposal_query": None,
            "parse_error": exc.code,
            "selected_query": original_question,
            "selection_source": "original_question",
            "used_fallback": True,
            "fallback_reason": f"invalid_q2_static:{exc.code}",
        }
        _validate_static_action(action, original_question, q1_query)
        return action
    action = {
        "policy_version": Q2_ACTION_POLICY_VERSION,
        "gold_access": False,
        "slot": "q2_static",
        "controller_state_sha256": _canonical_json_sha256(state),
        "response_sha256": response_sha256,
        "proposal_valid": True,
        "proposal_query": proposal["query"],
        "parse_error": None,
        "selected_query": proposal["query"],
        "selection_source": "q2_static",
        "used_fallback": False,
        "fallback_reason": None,
    }
    _validate_static_action(action, original_question, q1_query)
    return action


def _validate_static_action(
    action: Mapping[str, Any],
    original_question: str,
    q1_query: str,
) -> None:
    if not isinstance(action, Mapping):
        raise DynamicDecompositionV8Error("static_action must be an object")
    if set(action) != _STATIC_ACTION_FIELDS:
        missing = sorted(_STATIC_ACTION_FIELDS - set(action))
        extra = sorted(set(action) - _STATIC_ACTION_FIELDS)
        raise DynamicDecompositionV8Error(
            f"static_action field set mismatch; missing={missing}, extra={extra}"
        )
    if action.get("policy_version") != Q2_ACTION_POLICY_VERSION or action.get("slot") != "q2_static":
        raise DynamicDecompositionV8Error("static_action has an unexpected policy or slot")
    if action.get("gold_access") is not False:
        raise DynamicDecompositionV8Error("static_action gold_access must be false")
    proposal_valid = action.get("proposal_valid")
    used_fallback = action.get("used_fallback")
    if type(proposal_valid) is not bool:
        raise DynamicDecompositionV8Error("static_action proposal_valid must be boolean")
    if type(used_fallback) is not bool:
        raise DynamicDecompositionV8Error("static_action used_fallback must be boolean")

    expected_state = build_static_q2_state(
        original_question=original_question,
        q1_query=q1_query,
    )
    expected_state_sha256 = _canonical_json_sha256(expected_state)
    if action.get("controller_state_sha256") != expected_state_sha256:
        raise DynamicDecompositionV8Error("static_action controller state hash mismatch")
    response_sha256 = action.get("response_sha256")
    if response_sha256 is not None and (
        not isinstance(response_sha256, str) or _SHA256_RE.fullmatch(response_sha256) is None
    ):
        raise DynamicDecompositionV8Error("static_action response SHA must be null or SHA256")

    selected = action.get("selected_query")
    if not isinstance(selected, str) or not selected:
        raise DynamicDecompositionV8Error("static_action lacks a selected query")
    proposal_query = action.get("proposal_query")
    parse_error = action.get("parse_error")
    selection_source = action.get("selection_source")
    fallback_reason = action.get("fallback_reason")
    if proposal_valid:
        if response_sha256 is None:
            raise DynamicDecompositionV8Error("valid static action lacks response SHA")
        if used_fallback is not False:
            raise DynamicDecompositionV8Error("valid static action cannot use fallback")
        if not isinstance(proposal_query, str) or selected != proposal_query:
            raise DynamicDecompositionV8Error(
                "valid static action proposal and selected query must match"
            )
        if selection_source != "q2_static":
            raise DynamicDecompositionV8Error("valid static action has wrong selection source")
        if parse_error is not None or fallback_reason is not None:
            raise DynamicDecompositionV8Error("valid static action contains fallback telemetry")
        try:
            reparsed = parse_query_response(
                selected,
                previous_queries=(original_question, q1_query),
            )
        except QueryParseError as exc:
            raise DynamicDecompositionV8Error(
                f"static_action selected query fails contract reparse: {exc.code}"
            ) from exc
        if reparsed["query"] != selected:
            raise DynamicDecompositionV8Error(
                "static_action selected query changed during contract reparse"
            )
    else:
        if used_fallback is not True:
            raise DynamicDecompositionV8Error("invalid static action must use fallback")
        if proposal_query is not None:
            raise DynamicDecompositionV8Error("invalid static action cannot retain a proposal query")
        if selected != original_question:
            raise DynamicDecompositionV8Error("invalid static action did not fall back byte-for-byte")
        if selection_source != "original_question":
            raise DynamicDecompositionV8Error("invalid static action has wrong selection source")
        if not isinstance(parse_error, str) or not parse_error:
            raise DynamicDecompositionV8Error("invalid static action lacks parse error telemetry")
        if fallback_reason != f"invalid_q2_static:{parse_error}":
            raise DynamicDecompositionV8Error("invalid static action fallback reason mismatch")
        # The original dataset question is an environment fallback, not a
        # controller-generated query.  It therefore receives only transport
        # safety validation: legitimate question text may itself contain a
        # literal token such as ``#1``.
        _validate_context_text(selected, field="static_action fallback selected_query")


def build_dynamic_q2_action(
    response_text: str | None,
    *,
    original_question: str,
    q1_query: str,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Build eligible C's q2 action; invalid output falls back directly to Q.

    Ineligible C must not call this helper.  It uses the same observation-blind
    state and response path as B.  This keeps each arm at one slot-2 controller
    action without obtaining an unbudgeted static proposal after dynamic failure.
    """

    original_question = _validate_context_text(original_question, field="original_question")
    q1_query = _validate_context_text(q1_query, field="q1_query")
    if not isinstance(binding, Mapping):
        raise DynamicDecompositionV8Error("binding must be an object")
    if binding.get("binder_version") != PROVENANCE_BINDER_VERSION:
        raise DynamicDecompositionV8Error("binding has an unexpected binder version")
    state = build_dynamic_q2_state(
        original_question=original_question,
        q1_query=q1_query,
        binding=binding,
    )
    response_hash = _sha256_text(response_text) if isinstance(response_text, str) else None
    if not isinstance(response_text, str):
        parse_error = "missing_dynamic_response"
        proposal = None
    else:
        try:
            proposal = parse_query_response(
                response_text,
                previous_queries=(original_question, q1_query),
            )
            parse_error = None
        except QueryParseError as exc:
            proposal = None
            parse_error = exc.code
    if proposal is None:
        return {
            "policy_version": Q2_ACTION_POLICY_VERSION,
            "gold_access": False,
            "slot": "q2_dynamic",
            "dynamic_eligible": True,
            "controller_state_sha256": _canonical_json_sha256(state),
            "response_sha256": response_hash,
            "proposal_evaluated": True,
            "proposal_valid": False,
            "proposal_query": None,
            "parse_error": parse_error,
            "selected_query": original_question,
            "selection_source": "original_question",
            "used_fallback": True,
            "fallback_reason": f"invalid_q2_dynamic:{parse_error}",
        }
    return {
        "policy_version": Q2_ACTION_POLICY_VERSION,
        "gold_access": False,
        "slot": "q2_dynamic",
        "dynamic_eligible": True,
        "controller_state_sha256": _canonical_json_sha256(state),
        "response_sha256": response_hash,
        "proposal_evaluated": True,
        "proposal_valid": True,
        "proposal_query": proposal["query"],
        "parse_error": None,
        "selected_query": proposal["query"],
        "selection_source": "q2_dynamic",
        "used_fallback": False,
        "fallback_reason": None,
    }


def _retrieval_event(*, source: str, query: str, rank: int, score: float | None) -> dict[str, Any]:
    return {
        "source": source,
        "query": query,
        "query_sha256": _sha256_text(query),
        "rank": rank,
        "score": score,
    }


def merge_fixed_budget_passages(
    root_passages: Sequence[Mapping[str, Any]],
    q1_passages: Sequence[Mapping[str, Any]],
    q2_passages: Sequence[Mapping[str, Any]],
    *,
    root_query: str,
    q1_query: str,
    q2_query: str,
    q1_binding: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge three top-10 lists under the fixed v8 evidence budget.

    Selection order is root ranks 1--6, two q1 documents novel to the current
    set, two q2 documents novel to the current set, then root ranks 7--10 until
    exactly ten documents are present.  A verified q1 support document is first
    in the q1 slot if it is not already in the root prefix.
    """

    root_query = _validate_context_text(root_query, field="root_query")
    q1_query = _validate_context_text(q1_query, field="q1_query")
    q2_query = _validate_context_text(q2_query, field="q2_query")
    root = _prepare_passages(root_passages, role="root", require_unique=True)
    q1 = _prepare_passages(q1_passages, role="q1", require_unique=False)
    q2 = _prepare_passages(q2_passages, role="q2", require_unique=False)

    selected: list[dict[str, Any]] = []
    index_by_key: dict[str, int] = {}
    prompt_hash_by_key: dict[str, str] = {}
    duplicate_paths = 0
    selected_by_slot = {"root_prefix": 0, "q1_novel": 0, "q2_novel": 0, "root_backfill": 0}

    def add(document: Mapping[str, Any], event: Mapping[str, Any], *, slot: str) -> bool:
        nonlocal duplicate_paths
        key = str(document["key"])
        if key in index_by_key:
            if prompt_hash_by_key[key] != document["prompt_sha256"]:
                raise DynamicDecompositionV8Error(
                    f"document identity {key} maps to different prompt-visible bytes"
                )
            duplicate_paths += 1
            provenance = selected[index_by_key[key]]["retrieval_provenance"]
            if dict(event) not in provenance:
                provenance.append(dict(event))
            return False
        passage = deepcopy(document["projection"])
        passage["document_key"] = key
        passage["retrieval_provenance"] = [dict(event)]
        index_by_key[key] = len(selected)
        prompt_hash_by_key[key] = str(document["prompt_sha256"])
        selected.append(passage)
        selected_by_slot[slot] += 1
        return True

    for document in root[:6]:
        add(
            document,
            _retrieval_event(
                source="root", query=root_query, rank=document["rank"], score=document["score"]
            ),
            slot="root_prefix",
        )

    binding_key: str | None = None
    binding_prioritized = False
    if q1_binding is not None:
        if not isinstance(q1_binding, Mapping):
            raise DynamicDecompositionV8Error("q1_binding must be an object")
        if q1_binding.get("binder_version") != PROVENANCE_BINDER_VERSION:
            raise DynamicDecompositionV8Error("q1_binding has an unexpected binder version")
        if q1_binding.get("gold_access") is not False:
            raise DynamicDecompositionV8Error("q1_binding gold_access must be false")
        if q1_binding.get("verified") is True:
            binding_key = str(q1_binding.get("supporting_document_key") or "")
            candidates = [document for document in q1 if document["key"] == binding_key]
            if len(candidates) != 1:
                raise DynamicDecompositionV8Error(
                    "verified q1 binding does not identify exactly one q1 top-10 document"
                )
            bound = candidates[0]
            if bound["prompt_sha256"] != q1_binding.get("supporting_document_prompt_sha256"):
                raise DynamicDecompositionV8Error("verified q1 binding document bytes changed")
            add(
                bound,
                _retrieval_event(
                    source="q1", query=q1_query, rank=bound["rank"], score=bound["score"]
                ),
                slot="q1_novel",
            )
            binding_prioritized = binding_key not in {document["key"] for document in root[:6]}

    q1_added = selected_by_slot["q1_novel"]
    for document in q1:
        if q1_added >= 2:
            break
        if binding_key is not None and document["key"] == binding_key:
            continue
        if add(
            document,
            _retrieval_event(
                source="q1", query=q1_query, rank=document["rank"], score=document["score"]
            ),
            slot="q1_novel",
        ):
            q1_added += 1

    q2_added = 0
    for document in q2:
        if q2_added >= 2:
            break
        if add(
            document,
            _retrieval_event(
                source="q2", query=q2_query, rank=document["rank"], score=document["score"]
            ),
            slot="q2_novel",
        ):
            q2_added += 1

    for document in root[6:10]:
        if len(selected) >= FINAL_PASSAGE_BUDGET:
            break
        add(
            document,
            _retrieval_event(
                source="root_backfill",
                query=root_query,
                rank=document["rank"],
                score=document["score"],
            ),
            slot="root_backfill",
        )
    if len(selected) != FINAL_PASSAGE_BUDGET:
        raise DynamicDecompositionV8Error(
            f"fixed passage allocation produced {len(selected)} rather than {FINAL_PASSAGE_BUDGET}"
        )

    telemetry = {
        "policy_version": PASSAGE_MERGE_POLICY_VERSION,
        "gold_access": False,
        "allocation": "root6+q1_novel2+q2_novel2+root7_10_backfill",
        "total_selected": len(selected),
        "selected_by_slot": selected_by_slot,
        "duplicate_paths_merged": duplicate_paths,
        "q1_binding_document_key": binding_key,
        "q1_binding_prioritized_into_novel_slot": binding_prioritized,
        "output_document_keys": [passage["document_key"] for passage in selected],
        "output_unique_document_count": len({passage["document_key"] for passage in selected}),
        "root_query_sha256": _sha256_text(root_query),
        "q1_query_sha256": _sha256_text(q1_query),
        "q2_query_sha256": _sha256_text(q2_query),
    }
    return selected, telemetry


__all__ = [
    "BOUND_EXCERPT_MAX_CHARS",
    "DynamicDecompositionV8Error",
    "EXPECTED_TOP_K",
    "FINAL_PASSAGE_BUDGET",
    "MAX_QUERY_CHARS",
    "MAX_SUBANSWER_CHARS",
    "MIN_QUERY_CHARS",
    "NO_RELEVANT_ANSWER",
    "NO_VERIFIED_SUBANSWER",
    "PASSAGE_TEXT_MAX_CHARS",
    "PASSAGE_MERGE_POLICY_VERSION",
    "PROVENANCE_BINDER_VERSION",
    "Q2_ACTION_POLICY_VERSION",
    "QUERY_CONTRACT_VERSION",
    "QueryParseError",
    "SUBANSWER_CONTRACT_VERSION",
    "SubanswerParseError",
    "bind_subanswer_provenance",
    "build_dynamic_q2_action",
    "build_dynamic_q2_state",
    "build_static_q2_action",
    "build_static_q2_state",
    "merge_fixed_budget_passages",
    "parse_and_bind_subanswer",
    "parse_query_response",
    "parse_subanswer_response",
    "project_top10_passages_for_prompt",
]
