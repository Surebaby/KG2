"""Pure HotpotQA silver helpers for an observation-conditioned Controller.

This module deliberately performs no file I/O, retrieval, model inference, or
API calls.  It has four responsibilities:

* extract a conservative two-document bridge chain from one raw HotpotQA train
  row;
* expose only masked evidence to an external query proposer;
* validate a proposed natural-language ``q1`` and dependent ``q2`` template;
* materialise a pair with the Controller-v1 *target/state shape*.

The emitted records use a companion schema version.  The frozen v4.4 central
validator has an exact two-dataset provenance allowlist and is implementation-
hash locked, so this module must neither modify it nor pretend that HotpotQA has
2Wiki/MuSiQue annotation paths.  ``validate_hotpot_action_pair`` reuses a
temporary, non-serialised projection only to check the unchanged v1 structural
contract; the real emitted provenance remains Hotpot-specific.  A future
three-dataset successor protocol can admit this companion provenance explicitly.

Gold answers are used only as a strict train-side eligibility/leakage screen and
are never emitted.  Passing these pure checks does not prove retrieval or reader
utility.  A release freezer must separately require q1/q2 Wiki18 support recall
and a passage-bound Reader answer for q1.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from kgproweight.eval.query_controller_v1 import (
    SCHEMA_VERSION as CENTRAL_V1_SCHEMA_VERSION,
    STATE_VERSION,
    family_sha256,
    validate_action_record,
)
from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.retrieval.dependent import dependency_refs, replace_dependency_refs
from kgproweight.retrieval.dynamic_decomposition_v8 import parse_query_response


SCHEMA_VERSION = "query-controller-action-hotpot-companion-v1"
BUILDER_VERSION = "hotpot-controller-silver-pure-v1_1-alias-and-proposal-binding"
PROPOSAL_VIEW_SCHEMA_VERSION = "hotpot-controller-proposal-view-v1"
PROPOSAL_SCHEMA_VERSION = "hotpot-controller-query-proposal-v1"
DATASET = "hotpotqa"
SOURCE_ACTION = "text"
INTERMEDIATE_MASK = "[INTERMEDIATE]"
FINAL_MASK = "[FINAL]"
HOTPOT_BINDING_METHOD = "hotpot_support_title_surface_in_root_support_sentence"

_SPACE_RE = re.compile(r"\s+")
_TRAILING_PARENTHETICAL_RE = re.compile(r"\s*\([^()]+\)\s*$")
_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", flags=re.IGNORECASE)
_Q2_SLOT_RE = re.compile(r"(?<![A-Za-z0-9_])#1(?![A-Za-z0-9_])", re.IGNORECASE)
_ANY_PLACEHOLDER_RE = re.compile(
    r"(?:"
    r"(?<![A-Za-z0-9_])(?:\$(?:hop|step)_?[1-9]\d*|#[1-9]\d*|"
    r"(?:hop|step)_[1-9]\d*)(?![A-Za-z0-9_])"
    r"|\{\{?\s*(?:answer|entity|subject|object|hop|step|intermediate|final)"
    r"(?:_[1-9]\d*)?\s*\}?\}"
    r"|<\s*(?:answer|entity|subject|object|hop|step|intermediate|final)"
    r"(?:_[1-9]\d*)?\s*>"
    r"|\[(?:INTERMEDIATE|FINAL)\]"
    r")",
    flags=re.IGNORECASE,
)
_QUESTION_WORD_RE = re.compile(
    r"\b(?:who|whom|whose|what|which|when|where|why|how|"
    r"is|are|was|were|do|does|did|can|could|would|will|has|have|had)\b",
    flags=re.IGNORECASE,
)
_GENERIC_Q2_WORDS = frozenset(
    "a an the this that these those who whom whose what which when where why how "
    "is are was were be been being do does did can could would will has have had "
    "of for to in on at by from with and or it its they them their entity result "
    "prior".split()
)
_UNSAFE_FINALS = frozenset({"yes", "no", "unknown", "none"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_HOTPOT_ANNOTATION_PATH_RE = re.compile(
    r"metadata\.supporting_facts\.title\[([0-9]+)\]"
)


class HotpotSilverReject(ValueError):
    """A raw row or proposal cannot safely yield a Hotpot companion pair."""

    def __init__(
        self,
        code: str,
        message: str = "",
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = str(code)
        self.details = dict(details or {})
        super().__init__(message or self.code)


@dataclass(frozen=True)
class SupportSentence:
    support_fact_position: int
    context_position: int
    document_title: str
    sentence_index: int
    evidence_excerpt: str

    @property
    def document_id_suffix(self) -> str:
        return f"context::{self.context_position}"


@dataclass(frozen=True)
class HotpotSupportChain:
    qid: str
    question: str
    raw_record_sha256: str
    root_title: str
    bridge_title: str
    intermediate: str
    bridge_support_title_position: int
    first_hop: SupportSentence
    second_hop: SupportSentence
    final_answers: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedQueryProposal:
    q1_query: str
    q2_template: str
    q2_query: str
    q1_relation_intent: str
    q2_relation_intent: str
    proposal_sha256: str


def _clean(value: object) -> str:
    return _SPACE_RE.sub(" ", str(value or "").strip())


def _surface(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).split())


def _folded_surface(value: object) -> str:
    """Surface form used only for conservative secret/alias screening."""

    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return _surface(without_marks)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_line(value: object) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if "\n" in value or "\r" in value:
        return False
    return not any(
        unicodedata.category(char) in {"Cc", "Cf", "Zl", "Zp"} for char in value
    )


def _normalizable_line(value: object) -> bool:
    """Allow benign horizontal padding but reject multiline/control content."""

    return (
        isinstance(value, str)
        and bool(_clean(value))
        and "\n" not in value
        and "\r" not in value
        and not any(
            unicodedata.category(char) in {"Cc", "Cf", "Zl", "Zp"}
            for char in value
        )
    )


def _title_aliases(value: object) -> tuple[str, ...]:
    title = _clean(value)
    stripped = _clean(_TRAILING_PARENTHETICAL_RE.sub("", title))
    article_stripped = _clean(_LEADING_ARTICLE_RE.sub("", stripped))
    aliases: list[str] = []
    for candidate in (title, stripped, article_stripped):
        if candidate and _folded_surface(candidate) not in {
            _folded_surface(item) for item in aliases
        }:
            aliases.append(candidate)
    return tuple(aliases)


def _contains_surface(container: object, needle: object) -> bool:
    haystack = _surface(container)
    item = _surface(needle)
    return bool(item and f" {item} " in f" {haystack} ")


def _surface_count(container: object, needle: object) -> int:
    haystack = _surface(container)
    item = _surface(needle)
    if not haystack or not item:
        return 0
    return len(re.findall(rf"(?<!\w){re.escape(item)}(?!\w)", haystack))


def _contains_title(container: object, title: object) -> bool:
    return any(_contains_surface(container, alias) for alias in _title_aliases(title))


def _ordered_subsequence(container: object, needle: object, *, max_gap: int = 2) -> bool:
    """Conservatively catch aliases such as ``US Highway 60`` for ``US 60``."""

    def collapse_initials(tokens: list[str]) -> list[str]:
        collapsed: list[str] = []
        initials: list[str] = []
        for token in (*tokens, ""):
            if len(token) == 1 and token.isascii() and token.isalpha():
                initials.append(token)
                continue
            if initials:
                collapsed.append("".join(initials))
                initials = []
            if token:
                collapsed.append(token)
        return collapsed

    haystack = collapse_initials(_folded_surface(container).split())
    target = collapse_initials(_folded_surface(needle).split())
    if not target or len(haystack) < len(target):
        return False
    if len(target) == 1:
        return target[0] in haystack
    for start, token in enumerate(haystack):
        if token != target[0]:
            continue
        cursor = start
        matched = True
        for wanted in target[1:]:
            next_position = next(
                (
                    position
                    for position in range(cursor + 1, min(len(haystack), cursor + max_gap + 2))
                    if haystack[position] == wanted
                ),
                None,
            )
            if next_position is None:
                matched = False
                break
            cursor = next_position
        if matched:
            return True
    return False


def _contains_secret(container: object, secret: object) -> bool:
    haystack = _folded_surface(container)
    item = _folded_surface(secret)
    return bool(
        item
        and (
            f" {item} " in f" {haystack} "
            or _ordered_subsequence(container, secret)
        )
    )


def _alias_equivalent(left: object, right: object) -> bool:
    def folded_tokens(value: object) -> list[str]:
        text = _TRAILING_PARENTHETICAL_RE.sub("", _clean(value))
        decomposed = unicodedata.normalize("NFKD", text)
        text = "".join(
            character
            for character in decomposed
            if not unicodedata.combining(character)
        )
        return _surface(text).split()

    left_tokens = folded_tokens(left)
    right_tokens = folded_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    if left_tokens == right_tokens:
        return True
    shorter, longer = sorted((left_tokens, right_tokens), key=len)
    width = len(shorter)
    if any(
        longer[offset : offset + width] == shorter
        for offset in range(len(longer) - width + 1)
    ):
        return True
    cursor = 0
    for token in longer:
        if cursor < len(shorter) and token == shorter[cursor]:
            cursor += 1
    return cursor == len(shorter)


def _mask_pattern(value: object) -> re.Pattern[str] | None:
    tokens = re.findall(
        r"\w+", unicodedata.normalize("NFKC", _clean(value)), flags=re.UNICODE
    )
    if not tokens:
        return None
    return re.compile(
        r"(?<!\w)" + r"[\W_]+".join(re.escape(token) for token in tokens) + r"(?!\w)",
        flags=re.IGNORECASE | re.UNICODE,
    )


def _ordered_mask_pattern(value: object, *, max_gap: int = 2) -> re.Pattern[str] | None:
    """Mask expanded aliases such as ``David Robert Coleman`` for ``David Coleman``."""

    tokens = re.findall(
        r"\w+", unicodedata.normalize("NFKC", _clean(value)), flags=re.UNICODE
    )
    if len(tokens) < 2:
        return None
    separator = rf"(?:[\W_]+(?:\w+[\W_]+){{0,{max_gap}}})"
    return re.compile(
        r"(?<!\w)" + separator.join(re.escape(token) for token in tokens) + r"(?!\w)",
        flags=re.IGNORECASE | re.UNICODE,
    )


def _mask_values(text: str, values: Sequence[str], marker: str) -> str:
    result = str(text)
    candidates = sorted(
        {alias for value in values for alias in _title_aliases(value)},
        key=lambda item: (len(_surface(item).split()), len(item)),
        reverse=True,
    )
    for value in candidates:
        pattern = _mask_pattern(value)
        if pattern is not None:
            result = pattern.sub(marker, result)
        # Some Hotpot titles omit middle names while the bound support sentence
        # spells them out.  The secret scanner already treats that as the same
        # entity, so masking must use the identical conservative contract.
        ordered_pattern = _ordered_mask_pattern(value)
        if ordered_pattern is not None:
            result = ordered_pattern.sub(marker, result)
    return _clean(result)


def _unique_titles(values: Sequence[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        title = _clean(value)
        key = _surface(title)
        if title and key not in seen:
            seen.add(key)
            result.append(title)
    return result


def _secret_aliases(values: Sequence[object]) -> tuple[str, ...]:
    aliases: list[str] = []
    seen: set[str] = set()
    for value in values:
        for alias in _title_aliases(value):
            key = _surface(alias)
            if key and key not in seen:
                seen.add(key)
                aliases.append(alias)
    return tuple(aliases)


def _bound_title_surface(title: str, excerpt: str) -> str | None:
    for alias in sorted(_title_aliases(title), key=lambda item: len(_surface(item)), reverse=True):
        if _surface_count(excerpt, alias) == 1:
            return alias
    return None


def extract_hotpot_support_chain(raw_row: Mapping[str, Any]) -> HotpotSupportChain:
    """Extract one strict, directionally unambiguous Hotpot bridge chain.

    The root document is the sole supporting title explicitly present in the
    question.  The other supporting title must occur exactly once in one root
    support sentence and becomes the train-side intermediate observation.
    Final-answer annotations are used only to reject leakage/degenerate chains
    and to select one second-hop support sentence; they are not returned by any
    serialisation helper.
    """

    if not isinstance(raw_row, Mapping):
        raise HotpotSilverReject("raw_not_object")
    raw_ids = [
        raw_row.get(field)
        for field in ("id", "qid")
        if raw_row.get(field) not in (None, "")
    ]
    if (
        not raw_ids
        or any(not _normalizable_line(value) for value in raw_ids)
        or len({_clean(value) for value in raw_ids}) != 1
    ):
        raise HotpotSilverReject("raw_identity_invalid")
    raw_question = raw_row.get("question")
    if not _normalizable_line(raw_question):
        raise HotpotSilverReject("raw_identity_invalid")
    qid = _clean(raw_ids[0])
    question = _clean(raw_question)
    metadata = raw_row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise HotpotSilverReject("metadata_missing")
    if _clean(metadata.get("type")).casefold() != "bridge":
        raise HotpotSilverReject("not_bridge_type")

    support = metadata.get("supporting_facts")
    context = metadata.get("context")
    if not isinstance(support, Mapping) or not isinstance(context, Mapping):
        raise HotpotSilverReject("support_or_context_missing")
    support_titles_value = support.get("title")
    support_indices_value = support.get("sent_id")
    context_titles_value = context.get("title")
    context_sentences_value = context.get("sentences")
    if not isinstance(support_titles_value, list) or not isinstance(
        support_indices_value, list
    ):
        raise HotpotSilverReject("support_arrays_not_lists")
    if not isinstance(context_titles_value, list) or not isinstance(
        context_sentences_value, list
    ):
        raise HotpotSilverReject("context_arrays_not_lists")
    support_titles = list(support_titles_value)
    support_indices = list(support_indices_value)
    context_titles = list(context_titles_value)
    context_sentences = list(context_sentences_value)
    if not support_titles or len(support_titles) != len(support_indices):
        raise HotpotSilverReject("support_arrays_misaligned")
    if len(support_titles) != 2:
        raise HotpotSilverReject(
            "support_fact_count_not_two", details={"count": len(support_titles)}
        )
    if not context_titles or len(context_titles) != len(context_sentences):
        raise HotpotSilverReject("context_arrays_misaligned")
    if any(not _normalizable_line(title) for title in support_titles):
        raise HotpotSilverReject("support_title_invalid")
    if any(not _normalizable_line(title) for title in context_titles):
        raise HotpotSilverReject("context_title_invalid")

    titles = _unique_titles(support_titles)
    if len(titles) != 2:
        raise HotpotSilverReject(
            "support_title_count_not_two", details={"count": len(titles)}
        )
    if _alias_equivalent(titles[0], titles[1]) or any(
        _contains_title(left, right) for left, right in ((titles[0], titles[1]), (titles[1], titles[0]))
    ):
        raise HotpotSilverReject("support_titles_overlap")

    question_title_hits = [title for title in titles if _contains_title(question, title)]
    if len(question_title_hits) != 1:
        raise HotpotSilverReject(
            "question_root_title_not_unique", details={"count": len(question_title_hits)}
        )
    root_title = question_title_hits[0]
    bridge_title = next(title for title in titles if _surface(title) != _surface(root_title))
    if _contains_title(question, bridge_title) or _contains_secret(
        question, bridge_title
    ):
        raise HotpotSilverReject("bridge_title_already_in_question")

    context_positions: dict[str, int] = {}
    for title in titles:
        positions = [
            index
            for index, context_title in enumerate(context_titles)
            if _surface(context_title) == _surface(title)
        ]
        if len(positions) != 1:
            raise HotpotSilverReject(
                "support_title_context_join_not_unique",
                details={"title_sha256": _sha256_text(title), "count": len(positions)},
            )
        context_positions[_surface(title)] = positions[0]

    grouped: dict[str, list[SupportSentence]] = {_surface(title): [] for title in titles}
    seen_pointers: set[tuple[str, int]] = set()
    for support_position, (title_value, sentence_index) in enumerate(
        zip(support_titles, support_indices)
    ):
        title = _clean(title_value)
        title_key = _surface(title)
        if title_key not in grouped:
            raise HotpotSilverReject("support_title_identity_drift")
        if type(sentence_index) is not int or sentence_index < 0:
            raise HotpotSilverReject("support_sentence_index_invalid")
        pointer = (title_key, sentence_index)
        if pointer in seen_pointers:
            raise HotpotSilverReject("duplicate_support_pointer")
        seen_pointers.add(pointer)
        context_position = context_positions[title_key]
        sentences = context_sentences[context_position]
        if not isinstance(sentences, list) or sentence_index >= len(sentences):
            raise HotpotSilverReject("support_sentence_pointer_out_of_range")
        raw_excerpt = sentences[sentence_index]
        if not _normalizable_line(raw_excerpt):
            raise HotpotSilverReject("support_sentence_invalid")
        excerpt = _clean(raw_excerpt)
        grouped[title_key].append(
            SupportSentence(
                support_fact_position=support_position,
                context_position=context_position,
                document_title=title,
                sentence_index=sentence_index,
                evidence_excerpt=excerpt,
            )
        )
    root_support = grouped[_surface(root_title)]
    bridge_support = grouped[_surface(bridge_title)]
    if not root_support or not bridge_support:
        raise HotpotSilverReject("support_document_has_no_sentence")

    first_hop_candidates: list[tuple[SupportSentence, str]] = []
    for sentence in root_support:
        intermediate = _bound_title_surface(bridge_title, sentence.evidence_excerpt)
        if intermediate is not None:
            first_hop_candidates.append((sentence, intermediate))
    if len(first_hop_candidates) != 1:
        raise HotpotSilverReject(
            "bridge_surface_in_root_support_not_unique",
            details={"count": len(first_hop_candidates)},
        )
    first_hop, intermediate = first_hop_candidates[0]
    if any(_contains_title(sentence.evidence_excerpt, root_title) for sentence in bridge_support):
        raise HotpotSilverReject("bidirectional_support_title_link")

    raw_answers = raw_row.get("golden_answers")
    if not isinstance(raw_answers, list):
        raise HotpotSilverReject("final_answers_missing")
    final_answers: list[str] = []
    seen_answers: set[str] = set()
    for value in raw_answers:
        if not _normalizable_line(value):
            raise HotpotSilverReject("final_answer_invalid")
        answer = _clean(value)
        key = _surface(answer)
        if answer and key not in seen_answers:
            seen_answers.add(key)
            final_answers.append(answer)
    if not final_answers:
        raise HotpotSilverReject("final_answers_missing")
    if any(
        _surface(answer) in _UNSAFE_FINALS
        or len(_surface(answer).replace(" ", "")) <= 1
        for answer in final_answers
    ):
        raise HotpotSilverReject("unsafe_short_or_boolean_final_alias")

    for answer in final_answers:
        if _contains_secret(question, answer):
            raise HotpotSilverReject("final_alias_in_question")
        if any(
            _alias_equivalent(answer, value)
            for value in (root_title, bridge_title, intermediate)
        ):
            raise HotpotSilverReject("final_alias_equals_chain_entity")
        if _contains_secret(root_title, answer) or _contains_secret(bridge_title, answer):
            raise HotpotSilverReject("final_alias_in_support_title")
        if any(_contains_secret(sentence.evidence_excerpt, answer) for sentence in root_support):
            raise HotpotSilverReject("final_alias_in_first_hop_support")

    second_hop_candidates = [
        sentence
        for sentence in bridge_support
        if any(_contains_surface(sentence.evidence_excerpt, answer) for answer in final_answers)
    ]
    if len(second_hop_candidates) != 1:
        raise HotpotSilverReject(
            "final_surface_in_second_hop_support_not_unique",
            details={"count": len(second_hop_candidates)},
        )
    second_hop = second_hop_candidates[0]
    second_hop_surface = _surface(second_hop.evidence_excerpt)
    if any(
        (answer_surface := _surface(answer))
        and (
            second_hop_surface == answer_surface
            or second_hop_surface.startswith(f"{answer_surface} ")
        )
        for answer in final_answers
    ):
        raise HotpotSilverReject("final_alias_leads_second_hop_support")
    bridge_title_positions = [
        index
        for index, title in enumerate(support_titles)
        if _surface(title) == _surface(bridge_title)
    ]
    if not bridge_title_positions:
        raise HotpotSilverReject("bridge_support_title_position_missing")

    try:
        raw_record_sha256 = _canonical_sha256(dict(raw_row))
    except (TypeError, ValueError) as exc:
        raise HotpotSilverReject("raw_not_json_safe", str(exc)) from exc
    return HotpotSupportChain(
        qid=qid,
        question=question,
        raw_record_sha256=raw_record_sha256,
        root_title=root_title,
        bridge_title=bridge_title,
        intermediate=intermediate,
        bridge_support_title_position=bridge_title_positions[0],
        first_hop=first_hop,
        second_hop=second_hop,
        final_answers=tuple(final_answers),
    )


def _assert_no_secret(
    fields: Mapping[str, str],
    secrets: Sequence[str],
    *,
    code: str,
) -> None:
    for field, text in fields.items():
        for secret in secrets:
            if _contains_secret(text, secret):
                raise HotpotSilverReject(
                    code,
                    details={
                        "field": field,
                        "secret_sha256": _sha256_text(secret),
                    },
                )


def build_masked_proposal_view(chain: HotpotSupportChain) -> dict[str, Any]:
    """Build the only semantic view an external q1/q2 proposer should receive."""

    if not isinstance(chain, HotpotSupportChain):
        raise TypeError("chain must be HotpotSupportChain")
    bridge_values = _secret_aliases((chain.bridge_title, chain.intermediate))
    final_values = _secret_aliases(chain.final_answers)
    first = _mask_values(chain.first_hop.evidence_excerpt, bridge_values, INTERMEDIATE_MASK)
    first = _mask_values(first, final_values, FINAL_MASK)
    second = _mask_values(chain.second_hop.evidence_excerpt, bridge_values, INTERMEDIATE_MASK)
    second = _mask_values(second, final_values, FINAL_MASK)
    if INTERMEDIATE_MASK not in first:
        raise HotpotSilverReject("masked_first_hop_lacks_intermediate_marker")
    if FINAL_MASK not in second:
        raise HotpotSilverReject("masked_second_hop_lacks_final_marker")
    semantic = {
        "original_question": chain.question,
        "root_document_title": chain.root_title,
        "first_hop_evidence_masked": first,
        "second_hop_evidence_masked": second,
    }
    _assert_no_secret(
        semantic,
        (*bridge_values, *final_values),
        code="masked_proposal_view_secret_residual",
    )
    return {
        "schema_version": PROPOSAL_VIEW_SCHEMA_VERSION,
        "dataset": DATASET,
        "qid": chain.qid,
        **semantic,
        "required_output": {
            "schema_version": PROPOSAL_SCHEMA_VERSION,
            "q1": (
                "one natural single-hop question answerable by the first-hop evidence; "
                "ask for [INTERMEDIATE] without copying the marker"
            ),
            "q2_template": (
                "one natural single-hop question for [FINAL], using literal #1 exactly once"
            ),
        },
    }


def _validate_natural_question(value: object, *, field: str) -> str:
    if not _normalizable_line(value):
        raise HotpotSilverReject(f"{field}_not_safe_line")
    text = _clean(value)
    if not text.endswith("?") or _QUESTION_WORD_RE.search(text) is None:
        raise HotpotSilverReject(f"{field}_not_natural_question")
    return text


def validate_query_proposal(
    proposal: Mapping[str, Any],
    chain: HotpotSupportChain,
) -> ValidatedQueryProposal:
    """Validate one answer-free q1 plus one exactly dependent q2 template."""

    if not isinstance(proposal, Mapping):
        raise HotpotSilverReject("proposal_not_object")
    if set(proposal) != {"schema_version", "q1", "q2_template"}:
        raise HotpotSilverReject("proposal_schema")
    if proposal.get("schema_version") != PROPOSAL_SCHEMA_VERSION:
        raise HotpotSilverReject("proposal_schema_version")
    q1 = _validate_natural_question(proposal.get("q1"), field="q1")
    q2_template = _validate_natural_question(
        proposal.get("q2_template"), field="q2_template"
    )
    if dependency_refs(q1) or _ANY_PLACEHOLDER_RE.search(q1):
        raise HotpotSilverReject("q1_contains_placeholder")
    if len(_Q2_SLOT_RE.findall(q2_template)) != 1:
        raise HotpotSilverReject("q2_template_dependency_count")
    without_slot = _Q2_SLOT_RE.sub(" ", q2_template)
    if _ANY_PLACEHOLDER_RE.search(without_slot) or dependency_refs(without_slot):
        raise HotpotSilverReject("q2_template_contains_other_placeholder")
    if dependency_refs(q2_template) != ["slot_1"]:
        raise HotpotSilverReject("q2_template_dependency_invalid")
    q1_forbidden = (
        *_secret_aliases((chain.bridge_title, chain.intermediate)),
        *_secret_aliases(chain.final_answers),
    )
    _assert_no_secret({"q1": q1}, q1_forbidden, code="q1_secret_leak")
    _assert_no_secret(
        {"q2_template": q2_template},
        q1_forbidden,
        code="q2_template_secret_leak",
    )
    if not _contains_title(q1, chain.root_title):
        raise HotpotSilverReject("q1_missing_root_anchor")
    if _surface(q1) == _surface(chain.question):
        raise HotpotSilverReject("q1_repeats_original_question")

    relation_tokens = [
        token
        for token in _surface(without_slot).split()
        if token not in _GENERIC_Q2_WORDS
    ]
    if not relation_tokens:
        raise HotpotSilverReject("q2_template_no_relation_content")
    try:
        q2 = replace_dependency_refs(
            q2_template, {"step_1": chain.intermediate}, max_variants=1
        )[0]
        parse_query_response(q1, previous_queries=(chain.question,))
        parse_query_response(q2, previous_queries=(chain.question, q1))
    except ValueError as exc:
        raise HotpotSilverReject("proposal_query_contract", str(exc)) from exc
    if not _contains_surface(q2, chain.intermediate):
        raise HotpotSilverReject("q2_does_not_use_intermediate")
    _assert_no_secret(
        {"q2": q2}, _secret_aliases(chain.final_answers), code="q2_final_answer_leak"
    )

    q1_intent = _clean(q1[:-1])
    q2_intent = _clean(_Q2_SLOT_RE.sub("prior result", q2_template)[:-1])
    canonical = {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "q1": q1,
        "q2_template": q2_template,
    }
    return ValidatedQueryProposal(
        q1_query=q1,
        q2_template=q2_template,
        q2_query=q2,
        q1_relation_intent=q1_intent,
        q2_relation_intent=q2_intent,
        proposal_sha256=_canonical_sha256(canonical),
    )


def _observation(chain: HotpotSupportChain) -> dict[str, Any]:
    excerpt = chain.first_hop.evidence_excerpt
    return {
        "answer": chain.intermediate,
        "answer_sha256": _sha256_text(chain.intermediate),
        "evidence_excerpt": excerpt,
        "evidence_excerpt_sha256": _sha256_text(excerpt),
        "document_id": f"{DATASET}::{chain.qid}::{chain.first_hop.document_id_suffix}",
        "document_title": chain.root_title,
        "sentence_index": chain.first_hop.sentence_index,
        "provenance": {
            "source": "train_annotation_support",
            "annotation_path": (
                "metadata.supporting_facts.title"
                f"[{chain.bridge_support_title_position}]"
            ),
            "binding_method": HOTPOT_BINDING_METHOD,
        },
    }


def _safe_extra_provenance(
    value: Mapping[str, Any] | None,
    *,
    secrets: Sequence[str],
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise HotpotSilverReject("extra_provenance_not_object")
    result = deepcopy(dict(value))

    def visit(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                if not isinstance(key, str) or not _safe_line(key):
                    raise HotpotSilverReject(
                        "extra_provenance_key_invalid", details={"field": path}
                    )
                key_text = key
                key_surface = _surface(re.sub(r"[_-]+", " ", key_text))
                if key_surface in {
                    "answer",
                    "answers",
                    "gold answer",
                    "gold answers",
                    "golden answer",
                    "golden answers",
                    "final answer",
                    "final answers",
                    "bridge",
                    "intermediate",
                }:
                    raise HotpotSilverReject(
                        "extra_provenance_forbidden_key", details={"field": f"{path}.{key_text}"}
                    )
                visit(child, f"{path}.{key_text}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{path}[{index}]")
        elif isinstance(node, str):
            for secret in secrets:
                if _contains_secret(node, secret):
                    raise HotpotSilverReject(
                        "extra_provenance_secret_leak", details={"field": path}
                    )
        elif node is not None and not isinstance(node, (bool, int, float)):
            raise HotpotSilverReject(
                "extra_provenance_not_json_safe", details={"field": path}
            )

    visit(result, "source_provenance.extra")
    return result


def build_hotpot_action_pair(
    chain: HotpotSupportChain,
    proposal: Mapping[str, Any] | ValidatedQueryProposal,
    *,
    split: str,
    extra_source_provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialise and validate one companion q1/q2_dynamic action pair."""

    if split not in {"train", "dev", "confirmation"}:
        raise HotpotSilverReject("invalid_split")
    if isinstance(proposal, ValidatedQueryProposal):
        revalidated = validate_query_proposal(
            {
                "schema_version": PROPOSAL_SCHEMA_VERSION,
                "q1": proposal.q1_query,
                "q2_template": proposal.q2_template,
            },
            chain,
        )
        if revalidated != proposal:
            raise HotpotSilverReject("validated_proposal_integrity")
        validated = revalidated
    else:
        validated = validate_query_proposal(proposal, chain)
    extra = _safe_extra_provenance(
        extra_source_provenance,
        secrets=(
            *_secret_aliases((chain.bridge_title, chain.intermediate)),
            *_secret_aliases(chain.final_answers),
        ),
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET,
        "qid": chain.qid,
        "question_key": question_key(DATASET, chain.qid),
        "question_sha256": question_sha256(chain.question),
        "family_sha256": family_sha256(chain.question),
        "split": split,
    }
    common_provenance: dict[str, Any] = {
        "builder_version": BUILDER_VERSION,
        "companion_schema_version": SCHEMA_VERSION,
        "raw_record_sha256": chain.raw_record_sha256,
        "proposal_sha256": validated.proposal_sha256,
        "chain_extraction_method": "unique_question_root_and_support_title_link",
        "source_action_policy": "text_only_pid_null",
        "central_v1_direct_validation": "PENDING_THREE_DATASET_SUCCESSOR",
        "retrieval_replay": "NOT_RUN_BY_PURE_BUILDER",
        "supporting_fact_annotation_used_for_label_construction": True,
    }
    if extra:
        common_provenance["external_provenance"] = extra
    q1 = {
        **identity,
        "example_id": f"{DATASET}::{chain.qid}::q1",
        "slot": "q1",
        "turn_index": 1,
        "state": {
            "state_version": STATE_VERSION,
            "original_question": chain.question,
            "previous_actions": [],
            "verified_observations": [],
        },
        "target": {
            "action": "retrieve",
            "query": validated.q1_query,
            "anchor": chain.root_title,
            "relation_intent": validated.q1_relation_intent,
            "pid": None,
            "dependencies": [],
            "output_slot": "q1",
            "source_action": SOURCE_ACTION,
        },
        "source_provenance": {
            **deepcopy(common_provenance),
            "train_intermediate_annotation_used": False,
            "intermediate_value_visible_to_controller_input": False,
        },
        "gold_boundary": {
            "train_intermediate_annotation_used": False,
            "gold_final_answer_visible": False,
            "evaluation_gold_access": False,
        },
    }
    observation = _observation(chain)
    q2 = {
        **identity,
        "example_id": f"{DATASET}::{chain.qid}::q2_dynamic",
        "slot": "q2_dynamic",
        "turn_index": 2,
        "state": {
            "state_version": STATE_VERSION,
            "original_question": chain.question,
            "previous_actions": [
                {
                    "slot": "q1",
                    "action": "retrieve",
                    "query": validated.q1_query,
                    "output_slot": "q1",
                }
            ],
            "verified_observations": [observation],
        },
        "target": {
            "action": "retrieve",
            "query": validated.q2_query,
            "anchor": chain.intermediate,
            "relation_intent": validated.q2_relation_intent,
            "pid": None,
            "dependencies": ["q1"],
            "output_slot": "q2",
            "source_action": SOURCE_ACTION,
        },
        "source_provenance": {
            **deepcopy(common_provenance),
            "proposal_schema_version": PROPOSAL_SCHEMA_VERSION,
            "q2_template": validated.q2_template,
            "raw_annotation_path": observation["provenance"]["annotation_path"],
            "root_context_position": chain.first_hop.context_position,
            "bridge_context_position": chain.second_hop.context_position,
            "support_document_id": observation["document_id"],
            "support_document_title_sha256": _sha256_text(chain.root_title),
            "support_sentence_index": chain.first_hop.sentence_index,
            "support_excerpt_sha256": observation["evidence_excerpt_sha256"],
            "binding_method": HOTPOT_BINDING_METHOD,
            "train_intermediate_annotation_used": True,
            "intermediate_value_visible_to_controller_input": True,
        },
        "gold_boundary": {
            "train_intermediate_annotation_used": True,
            "gold_final_answer_visible": False,
            "evaluation_gold_access": False,
        },
    }
    return validate_hotpot_action_pair((q1, q2), chain=chain, expected_split=split)


def _central_v1_structural_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project only for in-memory reuse of the frozen v1 structural checks."""

    projected = deepcopy(dict(record))
    projected["schema_version"] = CENTRAL_V1_SCHEMA_VERSION
    if projected.get("slot") == "q2_dynamic":
        state = projected.get("state")
        observations = (
            state.get("verified_observations") if isinstance(state, Mapping) else None
        )
        if (
            isinstance(observations, list)
            and len(observations) == 1
            and isinstance(observations[0], Mapping)
        ):
            observations[0]["provenance"] = {
                "source": "train_annotation_support",
                "annotation_path": "metadata.metadata.question_decomposition[0].answer",
                "binding_method": "decomposition_step_support_answer_surface",
            }
    return projected


def _validate_source_provenance_pair(
    q1: Mapping[str, Any],
    q2: Mapping[str, Any],
    *,
    chain: HotpotSupportChain,
) -> None:
    q1_source = q1.get("source_provenance")
    q2_source = q2.get("source_provenance")
    if not isinstance(q1_source, Mapping) or not isinstance(q2_source, Mapping):
        raise HotpotSilverReject("source_provenance_schema")

    fixed = {
        "builder_version": BUILDER_VERSION,
        "companion_schema_version": SCHEMA_VERSION,
        "raw_record_sha256": chain.raw_record_sha256,
        "chain_extraction_method": "unique_question_root_and_support_title_link",
        "source_action_policy": "text_only_pid_null",
        "central_v1_direct_validation": "PENDING_THREE_DATASET_SUCCESSOR",
        "retrieval_replay": "NOT_RUN_BY_PURE_BUILDER",
        "supporting_fact_annotation_used_for_label_construction": True,
    }
    for source in (q1_source, q2_source):
        if any(source.get(key) != value for key, value in fixed.items()):
            raise HotpotSilverReject("source_provenance_binding")
        proposal_sha256 = source.get("proposal_sha256")
        if not isinstance(proposal_sha256, str) or _SHA256_RE.fullmatch(
            proposal_sha256
        ) is None:
            raise HotpotSilverReject("source_provenance_proposal_sha256")
    if q1_source.get("proposal_sha256") != q2_source.get("proposal_sha256"):
        raise HotpotSilverReject("source_provenance_proposal_mismatch")

    common_keys = {
        *fixed,
        "proposal_sha256",
        "train_intermediate_annotation_used",
        "intermediate_value_visible_to_controller_input",
    }
    if "external_provenance" in q1_source or "external_provenance" in q2_source:
        common_keys.add("external_provenance")
        if q1_source.get("external_provenance") != q2_source.get(
            "external_provenance"
        ):
            raise HotpotSilverReject("external_provenance_pair_mismatch")
        _safe_extra_provenance(
            q1_source.get("external_provenance"),
            secrets=(
                *_secret_aliases((chain.bridge_title, chain.intermediate)),
                *_secret_aliases(chain.final_answers),
            ),
        )
    if set(q1_source) != common_keys:
        raise HotpotSilverReject("q1_source_provenance_schema")
    q2_only = {
        "proposal_schema_version",
        "q2_template",
        "raw_annotation_path",
        "root_context_position",
        "bridge_context_position",
        "support_document_id",
        "support_document_title_sha256",
        "support_sentence_index",
        "support_excerpt_sha256",
        "binding_method",
    }
    if set(q2_source) != common_keys | q2_only:
        raise HotpotSilverReject("q2_source_provenance_schema")
    if q1_source.get("train_intermediate_annotation_used") is not False:
        raise HotpotSilverReject("q1_source_provenance_gold_boundary")
    if q1_source.get("intermediate_value_visible_to_controller_input") is not False:
        raise HotpotSilverReject("q1_source_provenance_visibility")
    expected_q2 = {
        "raw_annotation_path": (
            f"metadata.supporting_facts.title[{chain.bridge_support_title_position}]"
        ),
        "root_context_position": chain.first_hop.context_position,
        "bridge_context_position": chain.second_hop.context_position,
        "support_document_id": (
            f"{DATASET}::{chain.qid}::{chain.first_hop.document_id_suffix}"
        ),
        "support_document_title_sha256": _sha256_text(chain.root_title),
        "support_sentence_index": chain.first_hop.sentence_index,
        "support_excerpt_sha256": _sha256_text(chain.first_hop.evidence_excerpt),
        "binding_method": HOTPOT_BINDING_METHOD,
        "train_intermediate_annotation_used": True,
        "intermediate_value_visible_to_controller_input": True,
    }
    if any(q2_source.get(key) != value for key, value in expected_q2.items()):
        raise HotpotSilverReject("q2_source_provenance_binding")

    proposal_contract = {
        "schema_version": q2_source.get("proposal_schema_version"),
        "q1": q1.get("target", {}).get("query"),
        "q2_template": q2_source.get("q2_template"),
    }
    try:
        rebound = validate_query_proposal(proposal_contract, chain)
    except HotpotSilverReject as exc:
        raise HotpotSilverReject("source_provenance_proposal_contract", str(exc)) from exc
    if (
        rebound.proposal_sha256 != q1_source.get("proposal_sha256")
        or rebound.proposal_sha256 != q2_source.get("proposal_sha256")
        or rebound.q2_query != q2.get("target", {}).get("query")
        or rebound.q1_relation_intent != q1.get("target", {}).get("relation_intent")
        or rebound.q2_relation_intent != q2.get("target", {}).get("relation_intent")
    ):
        raise HotpotSilverReject("source_provenance_proposal_binding")


def validate_hotpot_action_pair(
    pair: Sequence[Mapping[str, Any]],
    *,
    chain: HotpotSupportChain,
    expected_split: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Strictly validate companion provenance plus unchanged Controller shape."""

    if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
        raise HotpotSilverReject("action_pair_cardinality")
    rows = [deepcopy(dict(row)) for row in pair if isinstance(row, Mapping)]
    if len(rows) != 2 or [row.get("slot") for row in rows] != ["q1", "q2_dynamic"]:
        raise HotpotSilverReject("action_pair_slot_order")
    if any(row.get("schema_version") != SCHEMA_VERSION for row in rows):
        raise HotpotSilverReject("action_pair_schema_version")
    if len({(row.get("dataset"), row.get("qid")) for row in rows}) != 1:
        raise HotpotSilverReject("action_pair_identity")
    if any(row.get("dataset") != DATASET or row.get("qid") != chain.qid for row in rows):
        raise HotpotSilverReject("action_pair_chain_identity")
    if expected_split is not None and any(row.get("split") != expected_split for row in rows):
        raise HotpotSilverReject("action_pair_split")

    q1, q2 = rows
    if any(
        not isinstance(row.get("state"), Mapping)
        or row["state"].get("original_question") != chain.question
        for row in rows
    ):
        raise HotpotSilverReject("action_pair_question_binding")
    if any(
        not isinstance(row.get("target"), Mapping)
        or row["target"].get("pid") is not None
        or row["target"].get("source_action") != SOURCE_ACTION
        for row in rows
    ):
        raise HotpotSilverReject("hotpot_target_pid_or_source_action")
    _validate_source_provenance_pair(q1, q2, chain=chain)

    try:
        for row in rows:
            validate_action_record(
                _central_v1_structural_projection(row),
                expected_split=expected_split,
            )
    except ValueError as exc:
        raise HotpotSilverReject("central_v1_structural_projection_failed", str(exc)) from exc

    if q1["target"].get("anchor") != chain.root_title:
        raise HotpotSilverReject("q1_anchor_chain_mismatch")
    if q2["target"].get("anchor") != chain.intermediate:
        raise HotpotSilverReject("q2_anchor_chain_mismatch")
    observations = q2["state"].get("verified_observations") or []
    if len(observations) != 1 or not isinstance(observations[0], Mapping):
        raise HotpotSilverReject("hotpot_observation_cardinality")
    observation = observations[0]
    provenance = observation.get("provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "source", "annotation_path", "binding_method"
    }:
        raise HotpotSilverReject("hotpot_observation_provenance_schema")
    annotation_match = _HOTPOT_ANNOTATION_PATH_RE.fullmatch(
        str(provenance.get("annotation_path") or "")
    )
    if (
        provenance.get("source") != "train_annotation_support"
        or provenance.get("binding_method") != HOTPOT_BINDING_METHOD
        or annotation_match is None
        or int(annotation_match.group(1)) != chain.bridge_support_title_position
    ):
        raise HotpotSilverReject("hotpot_observation_provenance_binding")
    expected_observation = _observation(chain)
    if dict(observation) != expected_observation:
        raise HotpotSilverReject("hotpot_observation_chain_mismatch")
    if not _contains_surface(observation["evidence_excerpt"], observation["answer"]):
        raise HotpotSilverReject("hotpot_observation_answer_not_in_excerpt")
    expected_previous = {
        "slot": "q1",
        "action": "retrieve",
        "query": q1["target"]["query"],
        "output_slot": "q1",
    }
    if q2["state"].get("previous_actions") != [expected_previous]:
        raise HotpotSilverReject("q2_previous_action_pair_mismatch")

    q1_visible = {
        "query": str(q1["target"].get("query") or ""),
        "anchor": str(q1["target"].get("anchor") or ""),
        "relation_intent": str(q1["target"].get("relation_intent") or ""),
    }
    _assert_no_secret(
        q1_visible,
        (
            *_secret_aliases((chain.bridge_title, chain.intermediate)),
            *_secret_aliases(chain.final_answers),
        ),
        code="q1_visible_secret_leak",
    )
    q2_visible = {
        "query": str(q2["target"].get("query") or ""),
        "relation_intent": str(q2["target"].get("relation_intent") or ""),
        "evidence_excerpt": str(observation.get("evidence_excerpt") or ""),
        "document_title": str(observation.get("document_title") or ""),
    }
    _assert_no_secret(
        q2_visible,
        _secret_aliases(chain.final_answers),
        code="q2_visible_final_answer_leak",
    )
    return q1, q2


def companion_compatibility_report(
    pair: Sequence[Mapping[str, Any]],
    *,
    chain: HotpotSupportChain,
) -> dict[str, Any]:
    """Declare, rather than conceal, the pending central-validator integration."""

    validate_hotpot_action_pair(pair, chain=chain)
    projections = [_central_v1_structural_projection(row) for row in pair]
    return {
        "schema_version": SCHEMA_VERSION,
        "companion_pair_valid": True,
        "central_v1_direct_validation_supported": False,
        "central_v1_structural_projection_valid": all(
            validate_action_record(row) == row for row in projections
        ),
        "pending_successor_change": (
            "admit exact Hotpot annotation-path pattern and binding method in a new "
            "three-dataset validator/protocol; do not modify frozen v4.4 in place"
        ),
        "retrieval_or_reader_validation_performed": False,
    }


__all__ = [
    "BUILDER_VERSION",
    "DATASET",
    "FINAL_MASK",
    "HOTPOT_BINDING_METHOD",
    "HotpotSilverReject",
    "HotpotSupportChain",
    "INTERMEDIATE_MASK",
    "PROPOSAL_SCHEMA_VERSION",
    "PROPOSAL_VIEW_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SupportSentence",
    "ValidatedQueryProposal",
    "build_hotpot_action_pair",
    "build_masked_proposal_view",
    "companion_compatibility_report",
    "extract_hotpot_support_chain",
    "validate_hotpot_action_pair",
    "validate_query_proposal",
]
