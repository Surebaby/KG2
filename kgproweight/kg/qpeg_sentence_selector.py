"""Train-only full-sentence selector for QPEG-v3 evidence graphs."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from kgproweight.kg.qpeg import (
    QPEG_SCHEMA_VERSION,
    compute_passages_sha256,
    passage_sentences,
    passage_title,
    validate_qpeg_record,
)
from kgproweight.kg.question_kg import question_key, question_sha256


QPEG_SENTENCE_FEATURE_VERSION = "qpeg-sentence-features-v1"
QPEG_SENTENCE_EXTRACTOR_VERSION = "qpeg-v3-trainonly-sentence-selector-v1"
_WORD_RE = re.compile(r"[a-z0-9]+")
_YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_CAPITALIZED_RE = re.compile(r"(?:^|\s)[A-Z][\w'’.-]+")
_QUESTION_CUES = {
    "who", "what", "when", "where", "which", "same", "both", "earlier",
    "later", "older", "younger", "first", "country", "city", "year",
    "date", "director", "writer", "spouse", "mother", "father", "born",
}


def _tokens(value: object) -> set[str]:
    return set(_WORD_RE.findall(str(value or "").casefold()))


def _norm(value: object) -> str:
    return " ".join(_WORD_RE.findall(str(value or "").casefold()))


def _coverage(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(right))


def sentence_features(
    *,
    dataset: str,
    question: str,
    title: str,
    sentence: str,
    passage_rank: int,
    sentence_index: int,
    all_titles: Sequence[str],
) -> dict[str, Any]:
    """Answer-free lexical/structural sentence features."""
    q_tokens = _tokens(question)
    title_tokens = _tokens(title)
    sentence_tokens = _tokens(sentence)
    other_titles = [
        _tokens(value) for value in all_titles if value and value.casefold() != title.casefold()
    ]
    other_title_mentions = sum(
        bool(tokens) and tokens.issubset(sentence_tokens) for tokens in other_titles
    )
    return {
        "dataset": str(dataset).strip().lower(),
        "passage_rank": float(passage_rank),
        "sentence_index": float(sentence_index),
        "question_sentence_coverage": _coverage(q_tokens, sentence_tokens),
        "sentence_question_coverage": _coverage(sentence_tokens, q_tokens),
        "question_title_coverage": _coverage(q_tokens, title_tokens),
        "title_question_coverage": _coverage(title_tokens, q_tokens),
        "sentence_token_count": float(len(sentence_tokens)),
        "title_token_count": float(len(title_tokens)),
        "question_cue_overlap": float(len(q_tokens & sentence_tokens & _QUESTION_CUES)),
        "title_mentioned_in_question": float(bool(title_tokens) and title_tokens.issubset(q_tokens)),
        "sentence_mentions_other_title_count": float(other_title_mentions),
        "sentence_has_year": float(bool(_YEAR_RE.search(sentence))),
        "sentence_has_number": float(bool(_NUMBER_RE.search(sentence))),
        "sentence_capitalized_token_count": float(len(_CAPITALIZED_RE.findall(sentence))),
        "is_first_sentence": float(sentence_index == 0),
    }


def sentence_candidates(
    *, dataset: str, question: str, passages: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    titles = [passage_title(passage) for passage in passages]
    candidates: list[dict[str, Any]] = []
    for passage_rank, passage in enumerate(passages):
        title = titles[passage_rank]
        passage_id = str(passage.get("id") or f"rank-{passage_rank}")
        if not title:
            continue
        for sentence_index, sentence in enumerate(passage_sentences(passage)):
            if not sentence:
                continue
            candidates.append({
                "head_surface": title,
                "relation_surface": "evidence sentence",
                "tail_surface": sentence,
                "passage_id": passage_id,
                "passage_rank": passage_rank,
                "sentence_index": sentence_index,
                "sentence_sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest(),
                "extraction_rule": "learned_full_sentence_selector",
                "relevance_score": 0.0,
                "features": sentence_features(
                    dataset=dataset,
                    question=question,
                    title=title,
                    sentence=sentence,
                    passage_rank=passage_rank,
                    sentence_index=sentence_index,
                    all_titles=titles,
                ),
            })
    return candidates


def select_sentence_edges(
    *,
    candidates: Sequence[Mapping[str, Any]],
    vectorizer: Any,
    classifier: Any,
    threshold: float,
    max_edges: int = 4,
) -> tuple[list[dict[str, Any]], list[float]]:
    if not 1 <= max_edges <= 12:
        raise ValueError("max_edges must be in [1, 12]")
    if not candidates:
        return [], []
    probabilities = classifier.predict_proba(
        vectorizer.transform([row["features"] for row in candidates])
    )[:, 1]
    ranked = sorted(
        zip(candidates, (float(value) for value in probabilities)),
        key=lambda value: (
            -value[1],
            int(value[0]["passage_rank"]),
            int(value[0]["sentence_index"]),
            str(value[0]["passage_id"]),
        ),
    )
    selected: list[tuple[dict[str, Any], float]] = []
    seen: set[tuple[str, str, str]] = set()
    for source, score in ranked:
        if score < threshold:
            continue
        key = (
            _norm(source.get("head_surface")),
            _norm(source.get("relation_surface")),
            _norm(source.get("tail_surface")),
        )
        if not all(key) or key in seen:
            continue
        seen.add(key)
        selected.append((dict(source), score))
        if len(selected) == max_edges:
            break
    edges: list[dict[str, Any]] = []
    for row, score in selected:
        row.pop("features", None)
        row["relevance_score"] = round(score, 6)
        edges.append(row)
    return edges, [score for _, score in selected]


def build_selected_sentence_record(
    *,
    dataset: str,
    qid: str,
    question: str,
    passages: Sequence[Mapping[str, Any]],
    vectorizer: Any,
    classifier: Any,
    threshold: float,
    max_edges: int = 4,
    passages_sha256: str | None = None,
) -> dict[str, Any]:
    raw_question = str(question).strip()
    computed_passages_sha256 = compute_passages_sha256(passages)
    if passages_sha256 is not None and passages_sha256 != computed_passages_sha256:
        raise ValueError("passages_sha256 mismatch")
    candidates = sentence_candidates(dataset=dataset, question=raw_question, passages=passages)
    edges, scores = select_sentence_edges(
        candidates=candidates,
        vectorizer=vectorizer,
        classifier=classifier,
        threshold=threshold,
        max_edges=max_edges,
    )
    record = {
        "schema_version": QPEG_SCHEMA_VERSION,
        "extractor_version": QPEG_SENTENCE_EXTRACTOR_VERSION,
        "question_key": question_key(dataset, qid),
        "dataset": str(dataset).strip().lower(),
        "qid": str(qid).strip(),
        "question": raw_question,
        "question_sha256": question_sha256(raw_question),
        "passages_sha256": computed_passages_sha256,
        "gold_access": False,
        "max_edges": max_edges,
        "edges": edges,
        "kg_subgraph": [
            [edge["head_surface"], edge["relation_surface"], edge["tail_surface"]]
            for edge in edges
        ],
        "provenance_complete": bool(edges),
        "build_status": "nonempty" if edges else "empty",
        "candidate_count": len(candidates),
        "selector": {
            "feature_version": QPEG_SENTENCE_FEATURE_VERSION,
            "threshold": threshold,
            "max_selected_edges": max_edges,
            "selected_scores": scores,
        },
    }
    payload = {
        "question_key": record["question_key"],
        "question_sha256": record["question_sha256"],
        "passages_sha256": record["passages_sha256"],
        "edges": edges,
    }
    record["qpeg_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    validate_qpeg_record(record, passages=passages)
    return record
