#!/usr/bin/env python
"""Read-only audit of recoverable HotpotQA bridge supervision.

The audit asks a deliberately narrower question than whether HotpotQA supplies
gold subquestions.  It measures when the train annotations determine a
two-document structural chain with exact sentence provenance:

    question names A -> an A support sentence mentions B -> a B support
    sentence contains the final answer surface

No q1/q2 text is generated.  The input is never modified and this command has
no output-file option; its only output is a JSON report on stdout.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data/hotpotqa/train.jsonl"
SCHEMA_VERSION = "hotpot-controller-bridge-capacity-audit-v1"
MATCH_VERSION = "qpeg-ascii-whole-surface-v1"
FAMILY_VERSION = "answer-free-lexical-family-v1"

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_TRAILING_PARENTHETICAL_RE = re.compile(r"\s*\([^()]+\)\s*$")
_ENTITY_SPAN_RE = re.compile(
    r"\b(?:[A-Z][\w'’-]*)(?:\s+(?:[A-Z][\w'’-]*|of|the|and|&)){0,5}\b"
)
_QUOTED_RE = re.compile(r"([\"“][^\"”]+[\"”]|'[^']{2,}')")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
_QUESTION_OPENERS = {
    "who", "what", "when", "where", "which", "why", "how", "were", "was",
    "are", "is", "did", "do", "does", "name",
}
_BOOLEAN_ANSWERS = {"yes", "no"}


def _surface(value: object) -> str:
    """QPEG-compatible conservative ASCII surface normalisation."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(_WORD_RE.findall(text))


def _contains_surface(container: object, needle: object) -> bool:
    haystack = _surface(container)
    item = _surface(needle)
    return bool(item and f" {item} " in f" {haystack} ")


def _folded_surface(value: object) -> str:
    """Accent-folded surface used only for conservative identity screening."""
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(_WORD_RE.findall(without_marks.casefold()))


def _ordered_identity_alias(left: object, right: object) -> bool:
    """Detect exact/accent-folded ordered expansions without fuzzy matching."""
    left_text = _TRAILING_PARENTHETICAL_RE.sub("", str(left or ""))
    right_text = _TRAILING_PARENTHETICAL_RE.sub("", str(right or ""))
    left_tokens = _folded_surface(left_text).split()
    right_tokens = _folded_surface(right_text).split()
    if not left_tokens or not right_tokens:
        return False
    if left_tokens == right_tokens:
        return True
    shorter, longer = sorted((left_tokens, right_tokens), key=len)
    if len(shorter) == 1 and len(shorter[0]) < 4:
        return False
    cursor = 0
    for token in longer:
        if cursor < len(shorter) and token == shorter[cursor]:
            cursor += 1
    return cursor == len(shorter)


def _answer_free_family_signature(question: str) -> str:
    """Mirror the frozen Controller/QPEG lexical-family partition."""
    text = unicodedata.normalize("NFKC", str(question or "")).strip()
    text = _QUOTED_RE.sub(" <entity> ", text)
    text = _NUMBER_RE.sub(" <num> ", text)

    def replace_entity(match: re.Match[str]) -> str:
        value = match.group(0)
        return value if value.casefold() in _QUESTION_OPENERS else " <entity> "

    text = _ENTITY_SPAN_RE.sub(replace_entity, text).casefold()
    return " ".join(re.sub(r"[^a-z0-9<>]+", " ", text).split())


def _family_sha256(question: str) -> str:
    return hashlib.sha256(
        _answer_free_family_signature(question).encode("utf-8")
    ).hexdigest()


def _answers(row: Mapping[str, Any]) -> list[str]:
    values = row.get("golden_answers") or []
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _answer_in(text: object, answers: Sequence[str]) -> bool:
    return any(_contains_surface(text, answer) for answer in answers)


def _answer_equals(text: object, answers: Sequence[str]) -> bool:
    value = _surface(text)
    return any(value and value == _surface(answer) for answer in answers)


def _answer_leads(text: object, answers: Sequence[str]) -> bool:
    value = _folded_surface(text)
    return any(
        normalized and (value == normalized or value.startswith(f"{normalized} "))
        for normalized in (_folded_surface(answer) for answer in answers)
    )


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _direction_label(forward: bool, reverse: bool) -> str:
    if forward and reverse:
        return "bidirectional"
    if forward:
        return "title0_to_title1_only"
    if reverse:
        return "title1_to_title0_only"
    return "none"


def _append_example(
    examples: dict[str, list[dict[str, Any]]],
    category: str,
    payload: Mapping[str, Any],
    *,
    limit: int,
) -> None:
    if len(examples[category]) < limit:
        examples[category].append(dict(payload))


def audit_records(
    rows: Iterable[Mapping[str, Any]], *, example_limit: int = 5
) -> dict[str, Any]:
    """Audit rows without mutating them and return a deterministic report."""
    if example_limit < 0:
        raise ValueError("example_limit must be non-negative")

    totals: Counter[str] = Counter()
    integrity: Counter[str] = Counter()
    support_directions: Counter[str] = Counter()
    article_directions: Counter[str] = Counter()
    binding: Counter[str] = Counter()
    question_orientation: Counter[str] = Counter()
    orientation_crosscheck: Counter[str] = Counter()
    leakage: Counter[str] = Counter()
    funnel: Counter[str] = Counter()
    pilot: Counter[str] = Counter()
    pilot_levels: Counter[str] = Counter()
    pilot_families: set[str] = set()
    pilot_families_by_level: dict[str, set[str]] = defaultdict(set)
    invalid_qids: set[str] = set()
    invalid_references: list[dict[str, Any]] = []
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row_index, row in enumerate(rows, start=1):
        totals["rows"] += 1
        metadata = row.get("metadata") or {}
        if metadata.get("type") != "bridge":
            totals["non_bridge"] += 1
            continue
        totals["bridge"] += 1
        qid = str(row.get("id") or row.get("qid") or "")
        question = str(row.get("question") or "")
        level = str(metadata.get("level") or "unknown")
        answers = _answers(row)
        if not answers:
            totals["missing_answer"] += 1
        is_boolean = any(_surface(answer) in _BOOLEAN_ANSWERS for answer in answers)
        if is_boolean:
            totals["boolean_bridge"] += 1
        else:
            totals["nonboolean_bridge"] += 1

        support = metadata.get("supporting_facts") or {}
        support_titles = list(support.get("title") or [])
        support_indices = list(support.get("sent_id") or [])
        context = metadata.get("context") or {}
        context_titles = list(context.get("title") or [])
        context_sentences = list(context.get("sentences") or [])

        if len(support_titles) != len(support_indices):
            integrity["support_title_sent_id_length_mismatch_qids"] += 1
            invalid_qids.add(qid)
            continue
        unique_titles: list[str] = []
        for title in support_titles:
            title = str(title)
            if title not in unique_titles:
                unique_titles.append(title)
        if len(unique_titles) != 2:
            integrity["not_exactly_two_unique_support_titles_qids"] += 1
            invalid_qids.add(qid)
            continue
        integrity["exactly_two_unique_support_titles_qids"] += 1
        if len(context_titles) != len(context_sentences):
            integrity["context_title_sentence_length_mismatch_qids"] += 1
            invalid_qids.add(qid)
            continue

        title_positions = {
            title: [index for index, value in enumerate(context_titles) if value == title]
            for title in unique_titles
        }
        bad_title = False
        for title, positions in title_positions.items():
            if len(positions) != 1:
                integrity[
                    "missing_context_title_references"
                    if not positions
                    else "duplicate_context_title_references"
                ] += 1
                invalid_qids.add(qid)
                invalid_references.append({
                    "qid": qid,
                    "row_index": row_index,
                    "reason": "support_title_context_resolution_not_unique",
                    "title": title,
                    "context_positions": positions,
                })
                bad_title = True
        if bad_title:
            continue

        support_by_title: dict[str, list[tuple[int, str]]] = {
            title: [] for title in unique_titles
        }
        bad_sentence = False
        for title, sentence_index in zip(
            support_titles, support_indices, strict=True
        ):
            position = title_positions[str(title)][0]
            sentences = (
                context_sentences[position]
                if position < len(context_sentences)
                else None
            )
            if (
                not isinstance(sentence_index, int)
                or isinstance(sentence_index, bool)
                or sentence_index < 0
            ):
                reason = "invalid_sent_id_type_or_negative"
            elif not isinstance(sentences, list):
                reason = "context_sentences_not_list"
            elif sentence_index >= len(sentences):
                reason = "sent_id_out_of_range"
            else:
                support_by_title[str(title)].append(
                    (sentence_index, str(sentences[sentence_index]))
                )
                continue
            integrity[f"{reason}_references"] += 1
            invalid_qids.add(qid)
            invalid_references.append({
                "qid": qid,
                "row_index": row_index,
                "reason": reason,
                "title": str(title),
                "sent_id": sentence_index,
                "available_sentence_count": (
                    len(sentences) if isinstance(sentences, list) else None
                ),
            })
            bad_sentence = True
        if bad_sentence:
            continue
        integrity["exact_support_references_valid_qids"] += 1

        title0, title1 = unique_titles
        support_hits_01 = [
            item for item in support_by_title[title0]
            if _contains_surface(item[1], title1)
        ]
        support_hits_10 = [
            item for item in support_by_title[title1]
            if _contains_surface(item[1], title0)
        ]
        support_label = _direction_label(bool(support_hits_01), bool(support_hits_10))
        support_directions[support_label] += 1

        article0 = context_sentences[title_positions[title0][0]]
        article1 = context_sentences[title_positions[title1][0]]
        article_hits_01 = [
            (index, str(sentence)) for index, sentence in enumerate(article0)
            if _contains_surface(sentence, title1)
        ]
        article_hits_10 = [
            (index, str(sentence)) for index, sentence in enumerate(article1)
            if _contains_surface(sentence, title0)
        ]
        article_directions[
            _direction_label(bool(article_hits_01), bool(article_hits_10))
        ] += 1

        if support_label not in {
            "title0_to_title1_only", "title1_to_title0_only"
        }:
            if support_label == "bidirectional":
                _append_example(
                    examples,
                    "bidirectional_cross_title",
                    {"qid": qid, "question": question, "titles": unique_titles},
                    limit=example_limit,
                )
            continue
        binding["unique_cross_title_direction_qids"] += 1
        if support_label == "title0_to_title1_only":
            source_title, target_title = title0, title1
            bridge_hits = support_hits_01
        else:
            source_title, target_title = title1, title0
            bridge_hits = support_hits_10
        if len(bridge_hits) == 1:
            binding["unique_bridge_support_sentence_qids"] += 1
        else:
            binding["multiple_bridge_support_sentence_qids"] += 1

        source_support = support_by_title[source_title]
        target_support = support_by_title[target_title]
        source_answer_hits = [
            item for item in source_support if _answer_in(item[1], answers)
        ]
        target_answer_hits = [
            item for item in target_support if _answer_in(item[1], answers)
        ]
        question_has_source = _contains_surface(question, source_title)
        question_has_target = _contains_surface(question, target_title)
        source_has_answer = bool(source_answer_hits)
        target_has_answer = bool(target_answer_hits)

        if question_has_source and not question_has_target:
            question_label = "source_only"
        elif question_has_target and not question_has_source:
            question_label = "target_only_inverse"
        elif question_has_source and question_has_target:
            question_label = "both"
        else:
            question_label = "neither"
        question_orientation[question_label] += 1

        question_one = question_has_source ^ question_has_target
        answer_one = source_has_answer ^ target_has_answer
        if question_one and answer_one:
            question_root_is_source = question_has_source
            answer_doc_is_target = target_has_answer
            if question_root_is_source == answer_doc_is_target:
                orientation_crosscheck["opposite_document_agreement"] += 1
                if question_root_is_source:
                    orientation_crosscheck["forward_agreement"] += 1
                else:
                    orientation_crosscheck["inverse_agreement"] += 1
                    _append_example(
                        examples,
                        "inverse_chain_despite_mention_arrow",
                        {
                            "qid": qid,
                            "question": question,
                            "mention_source_title": source_title,
                            "mention_target_title": target_title,
                            "final_answers": answers,
                        },
                        limit=example_limit,
                    )
            else:
                orientation_crosscheck["same_document_conflict"] += 1
                _append_example(
                    examples,
                    "question_and_answer_same_support_document",
                    {
                        "qid": qid,
                        "question": question,
                        "mention_source_title": source_title,
                        "mention_target_title": target_title,
                        "final_answers": answers,
                    },
                    limit=example_limit,
                )
        elif question_one:
            orientation_crosscheck["question_only_signal"] += 1
        elif answer_one:
            orientation_crosscheck["answer_document_only_signal"] += 1
        else:
            orientation_crosscheck["neither_or_ambiguous_signals"] += 1

        if not (question_has_source and not question_has_target):
            continue
        funnel["question_names_mention_source_only"] += 1
        if not target_answer_hits:
            continue
        funnel["final_surface_in_target_support"] += 1
        leakage["future_support_would_leak_if_exposed_before_q2"] += 1

        final_equals_target = _answer_equals(target_title, answers)
        if final_equals_target:
            leakage["final_answer_equals_intermediate_title"] += 1
            _append_example(
                examples,
                "final_answer_equals_intermediate",
                {
                    "qid": qid,
                    "question": question,
                    "intermediate_title": target_title,
                    "final_answers": answers,
                },
                limit=example_limit,
            )
            continue
        funnel["final_not_exact_intermediate_title"] += 1
        if len(bridge_hits) != 1:
            continue
        funnel["unique_first_hop_support_binding"] += 1
        if len(target_answer_hits) != 1:
            continue
        funnel["unique_second_hop_answer_binding"] += 1

        bridge_excerpt = bridge_hits[0][1]
        if source_answer_hits:
            leakage["final_surface_in_any_first_hop_support"] += 1
            if _answer_in(bridge_excerpt, answers):
                leakage["final_surface_in_bound_first_hop_excerpt"] += 1
            _append_example(
                examples,
                "future_answer_in_first_hop_support",
                {
                    "qid": qid,
                    "question": question,
                    "source_title": source_title,
                    "target_title": target_title,
                    "final_answers": answers,
                    "bound_first_hop_excerpt": bridge_excerpt,
                },
                limit=example_limit,
            )
            continue
        funnel["future_surface_absent_first_hop_support"] += 1
        if _answer_in(question, answers):
            leakage["final_surface_already_in_original_question"] += 1
            continue
        funnel["future_surface_absent_original_question"] += 1
        if _answer_in(source_title, answers) or _answer_in(target_title, answers):
            leakage["final_surface_in_visible_source_or_intermediate_title"] += 1
            continue
        funnel["future_surface_absent_visible_titles"] += 1
        if is_boolean:
            continue
        funnel["nonboolean"] += 1
        if any(len(_surface(answer).replace(" ", "")) < 2 for answer in answers):
            leakage["one_character_final_answer"] += 1
            continue
        funnel["answer_length_at_least_two"] += 1

        # Precision-first pilot: exactly one annotated sentence per article.
        if len(support_titles) != 2:
            continue
        pilot["precision_pool_before_identity_hardening"] += 1
        if any(_ordered_identity_alias(target_title, answer) for answer in answers):
            pilot["conservative_intermediate_final_alias_exclusions"] += 1
            _append_example(
                examples,
                "intermediate_final_identity_alias",
                {
                    "qid": qid,
                    "question": question,
                    "intermediate_title": target_title,
                    "final_answers": answers,
                },
                limit=example_limit,
            )
            continue
        pilot["after_conservative_alias_screen"] += 1
        target_excerpt = target_answer_hits[0][1]
        if _answer_leads(target_excerpt, answers):
            pilot["answer_leading_second_hop_identity_risk_exclusions"] += 1
            _append_example(
                examples,
                "answer_leads_second_hop_sentence",
                {
                    "qid": qid,
                    "question": question,
                    "intermediate_title": target_title,
                    "final_answers": answers,
                    "second_hop_excerpt": target_excerpt,
                },
                limit=example_limit,
            )
            continue
        pilot["identity_hardened_eligible_qids"] += 1
        pilot_levels[level] += 1
        family = _family_sha256(question)
        pilot_families.add(family)
        pilot_families_by_level[level].add(family)
        _append_example(
            examples,
            "identity_hardened_pilot_eligible",
            {
                "qid": qid,
                "level": level,
                "question": question,
                "root_title": source_title,
                "intermediate_title": target_title,
                "first_hop_excerpt": bridge_excerpt,
                "final_answers": answers,
                "second_hop_excerpt": target_excerpt,
            },
            limit=example_limit,
        )

    bridge_count = totals["bridge"]
    valid_support_count = integrity["exact_support_references_valid_qids"]
    unique_direction_count = binding["unique_cross_title_direction_qids"]
    unique_sentence_count = binding["unique_bridge_support_sentence_qids"]
    orientation_denominator = (
        orientation_crosscheck["opposite_document_agreement"]
        + orientation_crosscheck["same_document_conflict"]
    )
    inverse_denominator = (
        orientation_crosscheck["forward_agreement"]
        + orientation_crosscheck["inverse_agreement"]
    )
    eligible = pilot["identity_hardened_eligible_qids"]
    family_count = len(pilot_families)
    by_level_families = {
        level: len(values)
        for level, values in sorted(pilot_families_by_level.items())
    }
    balanced_pilot_feasible = all(
        len(pilot_families_by_level.get(level, set())) >= 10
        for level in ("easy", "medium", "hard")
    )
    ordered_funnel = [
        {"stage": "all_bridge", "count": bridge_count},
        {
            "stage": "exact_support_references_valid",
            "count": valid_support_count,
        },
        {
            "stage": "unique_cross_title_direction",
            "count": unique_direction_count,
        },
        {
            "stage": "question_names_mention_source_only",
            "count": funnel["question_names_mention_source_only"],
        },
        {
            "stage": "final_surface_in_target_support",
            "count": funnel["final_surface_in_target_support"],
        },
        {
            "stage": "final_not_exact_intermediate_title",
            "count": funnel["final_not_exact_intermediate_title"],
        },
        {
            "stage": "unique_first_hop_support_binding",
            "count": funnel["unique_first_hop_support_binding"],
        },
        {
            "stage": "unique_second_hop_answer_binding",
            "count": funnel["unique_second_hop_answer_binding"],
        },
        {
            "stage": "future_surface_absent_first_hop_support",
            "count": funnel["future_surface_absent_first_hop_support"],
        },
        {
            "stage": "future_surface_absent_original_question",
            "count": funnel["future_surface_absent_original_question"],
        },
        {
            "stage": "answer_length_at_least_two",
            "count": funnel["answer_length_at_least_two"],
        },
        {
            "stage": "exactly_one_support_sentence_per_title",
            "count": pilot["precision_pool_before_identity_hardening"],
        },
        {
            "stage": "conservative_intermediate_final_alias_absent",
            "count": pilot["after_conservative_alias_screen"],
        },
        {
            "stage": "answer_leading_identity_risk_absent",
            "count": eligible,
        },
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_READ_ONLY_CAPACITY_AUDIT_NOT_ACTION_SUPERVISION",
        "scope": {
            "dataset": "hotpotqa",
            "split": "train",
            "row_types_audited": ["bridge"],
            "input_mutated": False,
            "gold_or_action_rows_generated": False,
        },
        "definitions": {
            "strict_title_match": (
                "case-folded ASCII alphanumeric whole-surface match, preserving token order; "
                "the same lexical contract used by QPEG-v1"
            ),
            "unique_cross_title_direction": (
                "exactly one supporting-title direction is mentioned in annotated support "
                "sentences; this alone is not treated as hop order"
            ),
            "structural_chain": (
                "question names mention-source A and not B; one A support sentence mentions "
                "B; one B support sentence contains the canonical final-answer surface"
            ),
            "future_tail_proxy": (
                "HotpotQA supplies no explicit per-hop tails, so golden_answers are scanned "
                "as the future/final tail; literal leakage counts are lower bounds"
            ),
            "recoverable_label_boundary": (
                "The audit can recover A, intermediate title B, and sentence provenance. "
                "HotpotQA still does not determine canonical natural-language q1/q2 wording "
                "or a relation-intent label."
            ),
        },
        "totals": dict(sorted(totals.items())),
        "support_integrity": {
            **dict(sorted(integrity.items())),
            "invalid_qid_count": len(invalid_qids),
            "invalid_references": invalid_references,
        },
        "cross_title_mentions": {
            "match_version": MATCH_VERSION,
            "annotated_support_sentences": {
                **dict(sorted(support_directions.items())),
                "unique_direction_count": unique_direction_count,
                "unique_direction_rate_all_bridge": _rate(
                    unique_direction_count, bridge_count
                ),
                "unique_direction_rate_valid_support": _rate(
                    unique_direction_count, valid_support_count
                ),
            },
            "two_supporting_articles_all_sentences": {
                **dict(sorted(article_directions.items())),
                "unique_direction_count": (
                    article_directions["title0_to_title1_only"]
                    + article_directions["title1_to_title0_only"]
                ),
            },
            "support_sentence_binding": {
                **dict(sorted(binding.items())),
                "unique_sentence_given_unique_direction_rate": _rate(
                    unique_sentence_count, unique_direction_count
                ),
            },
        },
        "answer_type": {
            "nonboolean_bridge_count": totals["nonboolean_bridge"],
            "boolean_bridge_count": totals["boolean_bridge"],
            "nonboolean_rate": _rate(totals["nonboolean_bridge"], bridge_count),
        },
        "orientation": {
            "question_title_signal_given_unique_direction": dict(
                sorted(question_orientation.items())
            ),
            "question_answer_document_crosscheck": {
                **dict(sorted(orientation_crosscheck.items())),
                "crosscheck_denominator": orientation_denominator,
                "opposite_document_agreement_rate": _rate(
                    orientation_crosscheck["opposite_document_agreement"],
                    orientation_denominator,
                ),
                "inverse_share_when_opposite_documents": _rate(
                    orientation_crosscheck["inverse_agreement"], inverse_denominator
                ),
            },
        },
        "strict_forward_funnel": {
            **dict(sorted(funnel.items())),
            "ordered_stages": ordered_funnel,
        },
        "literal_future_leakage": {
            **dict(sorted(leakage.items())),
            "scan_is_lower_bound": True,
            "target_future_support_must_not_be_serialized_before_q2": True,
        },
        "pilot_capacity": {
            **dict(sorted(pilot.items())),
            "eligible_qids_by_level": dict(sorted(pilot_levels.items())),
            "unique_answer_free_families": family_count,
            "unique_answer_free_families_by_level": by_level_families,
            "balanced_10_easy_10_medium_10_hard_feasible": balanced_pilot_feasible,
            "remaining_unique_families_after_balanced_pilot30": max(
                0, family_count - 30
            ),
            "train600_dev60_confirmation30_family_disjoint_feasible": (
                family_count >= 690
            ),
            "family_capacity_margin_after_690": max(0, family_count - 690),
            "selection_rule": (
                "select one qid per answer-free lexical family; stratify 10/10/10 by "
                "easy/medium/hard; order within each stratum by a frozen SHA-256 seed; "
                "then manually verify q1/q2 semantics and reject identity-only or ambiguous "
                "relation rewrites"
            ),
        },
        "diagnostic_examples": dict(sorted(examples.items())),
        "scientific_boundary": (
            "This is a train-label availability and leakage audit. It does not establish "
            "that exact q1/q2 text can be recovered, that a Reader will predict the bridge, "
            "or that Controller training improves retrieval or QA."
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_file(path: Path, *, example_limit: int = 5) -> dict[str, Any]:
    def rows() -> Iterable[dict[str, Any]]:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"row is not an object at {path}:{line_number}")
                yield row

    report = audit_records(rows(), example_limit=example_limit)
    report["input"] = {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--example_limit", type=int, default=5)
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"input does not exist: {args.input}")
    report = audit_file(args.input, example_limit=args.example_limit)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
