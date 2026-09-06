"""Deterministic question-conditioned passage evidence graphs (QPEG-v1).

QPEG is a same-resource transformation: it may read only the question and the
already retrieved passages.  Every emitted edge points back to one passage
sentence.  No answer, support annotation, decomposition, Wikidata lookup, or
model-generated relation is used here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_key, question_sha256


QPEG_SCHEMA_VERSION = "qpeg-question-record-v1"
QPEG_EXTRACTOR_VERSION = "qpeg-deterministic-passage-v1.1-precision"
QPEG_COMPATIBLE_EXTRACTOR_VERSIONS = {
    QPEG_EXTRACTOR_VERSION,
    "qpeg-v2-trainonly-selector-v1",
    "qpeg-v3-trainonly-sentence-selector-v1",
}
QPEG_MAX_EDGES = 12

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'A-Z0-9])")
_SPACE_RE = re.compile(r"\s+")
_TAIL_STOP_RE = re.compile(r"\s+(?:who|which|that|where|when)\s+|[;]", re.IGNORECASE)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of", "on",
    "or", "the", "to", "was", "were", "what", "when", "where", "which", "who",
    "whom", "whose", "why", "with",
}

# Ordered, precision-first surface patterns.  Labels are literal normalisations
# of the matched phrase, never invented question-specific relations.
_RELATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in (
        ("born in", r"\b(?:is|was|were)?\s*born in\s+(?P<tail>[^.;]+)"),
        ("born on", r"\b(?:is|was|were)?\s*born on\s+(?P<tail>[^.;]+)"),
        ("died in", r"\b(?:died|death was) in\s+(?P<tail>[^.;]+)"),
        ("died on", r"\bdied on\s+(?P<tail>[^.;]+)"),
        ("located in", r"\b(?:is|was|are|were)?\s*located in\s+(?P<tail>[^.;]+)"),
        ("based in", r"\b(?:is|was|are|were)?\s*based in\s+(?P<tail>[^.;]+)"),
        ("part of", r"\b(?:is|was|are|were)?\s*(?:a )?part of\s+(?P<tail>[^.;]+)"),
        ("member of", r"\b(?:is|was|are|were)?\s*(?:a )?member of\s+(?P<tail>[^.;]+)"),
        ("capital of", r"\b(?:is|was)?\s*(?:the )?capital of\s+(?P<tail>[^.;]+)"),
        ("directed by", r"\b(?:is|was|were)?\s*directed by\s+(?P<tail>[^.;]+)"),
        ("written by", r"\b(?:is|was|were)?\s*written by\s+(?P<tail>[^.;]+)"),
        ("created by", r"\b(?:is|was|were)?\s*created by\s+(?P<tail>[^.;]+)"),
        ("founded by", r"\b(?:is|was|were)?\s*founded by\s+(?P<tail>[^.;]+)"),
        ("produced by", r"\b(?:is|was|were)?\s*produced by\s+(?P<tail>[^.;]+)"),
        ("published by", r"\b(?:is|was|were)?\s*published by\s+(?P<tail>[^.;]+)"),
        ("married to", r"\b(?:is|was|were)?\s*married to\s+(?P<tail>[^.;]+)"),
        ("known for", r"\b(?:is|was|were)?\s*known for\s+(?P<tail>[^.;]+)"),
        ("played by", r"\b(?:is|was|were)?\s*played by\s+(?P<tail>[^.;]+)"),
        ("stars", r"\b(?:stars|starring)\s+(?P<tail>[^.;]+)"),
        ("released", r"\b(?:released|was released)\s+(?P<tail>[^.;]+)"),
        ("won", r"\b(?:won|has won|had won)\s+(?P<tail>[^.;]+)"),
        ("joined", r"\b(?:joined|has joined|had joined)\s+(?P<tail>[^.;]+)"),
        ("served as", r"\b(?:served|serves) as\s+(?P<tail>[^.;]+)"),
        ("became", r"\bbecame\s+(?P<tail>[^.;]+)"),
    )
)


def _sha_json(value: Any) -> str:
    # Match the historical canonical retrieval-context identity function from
    # freeze_inference_proofkg_preregistration.py. Changing JSON separators
    # would make byte-identical passage objects appear to have a new identity.
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compute_passages_sha256(passages: Sequence[Mapping[str, Any]]) -> str:
    """Canonical identity used by both frozen contexts and online QPEG joins."""
    return _sha_json(list(passages))


def _clean(value: object) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", str(value or ""))).strip()


def _norm(value: object) -> str:
    return " ".join(_WORD_RE.findall(_clean(value).casefold()))


def _tokens(value: object) -> set[str]:
    return {token for token in _WORD_RE.findall(_clean(value).casefold()) if len(token) > 2 and token not in _STOPWORDS}


def passage_title(passage: Mapping[str, Any]) -> str:
    explicit = _clean(passage.get("title"))
    if explicit:
        return explicit.strip('"')
    contents = str(passage.get("contents") or passage.get("text") or "")
    first = contents.splitlines()[0] if contents else ""
    return _clean(first).strip('"')


def passage_sentences(passage: Mapping[str, Any]) -> list[str]:
    # Train-only selector fitting may retain the dataset's original sentence
    # boundaries explicitly. Runtime retrieval passages do not carry this key
    # and continue through the historical text splitter below.
    explicit = passage.get("_sentences")
    if isinstance(explicit, list):
        return [_clean(sentence) for sentence in explicit if _clean(sentence)]
    contents = str(passage.get("contents") or passage.get("text") or "")
    lines = contents.splitlines()
    body = " ".join(lines[1:]) if len(lines) > 1 else contents
    body = _clean(body)
    if not body:
        return []
    return [_clean(sentence) for sentence in _SENTENCE_SPLIT_RE.split(body) if _clean(sentence)]


def _trim_tail(value: str) -> str:
    tail = _TAIL_STOP_RE.split(_clean(value), maxsplit=1)[0]
    tail = tail.strip(" ,:-\"'")
    if len(tail) > 180:
        tail = tail[:180].rsplit(" ", 1)[0].rstrip(" ,:-")
    return tail


@dataclass(frozen=True)
class _Candidate:
    head: str
    relation: str
    tail: str
    passage_id: str
    passage_rank: int
    sentence_index: int
    sentence_sha256: str
    extraction_rule: str
    relevance_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "head_surface": self.head,
            "relation_surface": self.relation,
            "tail_surface": self.tail,
            "passage_id": self.passage_id,
            "passage_rank": self.passage_rank,
            "sentence_index": self.sentence_index,
            "sentence_sha256": self.sentence_sha256,
            "extraction_rule": self.extraction_rule,
            "relevance_score": round(self.relevance_score, 6),
        }


def _candidate_score(question: str, title: str, sentence: str, *, base: float) -> float:
    question_tokens = _tokens(question)
    sentence_tokens = _tokens(sentence)
    title_tokens = _tokens(title)
    overlap = len(question_tokens & sentence_tokens) / max(1, len(question_tokens))
    title_overlap = len(question_tokens & title_tokens) / max(1, len(title_tokens))
    return base + 2.0 * overlap + 0.75 * title_overlap


def _predicate_candidates(
    question: str,
    passage: Mapping[str, Any],
    passage_rank: int,
) -> list[_Candidate]:
    title = passage_title(passage)
    passage_id = str(passage.get("id") or f"rank-{passage_rank}")
    if not title:
        return []
    result: list[_Candidate] = []
    sentences = passage_sentences(passage)
    for sentence_index, sentence in enumerate(sentences):
        sentence_hash = hashlib.sha256(sentence.encode("utf-8")).hexdigest()
        matched = False
        for relation, pattern in _RELATION_PATTERNS:
            found = pattern.search(sentence)
            if not found:
                continue
            tail = _trim_tail(found.group("tail"))
            if len(tail) < 2 or _norm(tail) == _norm(title):
                continue
            matched = True
            result.append(_Candidate(
                head=title,
                relation=relation,
                tail=tail,
                passage_id=passage_id,
                passage_rank=passage_rank,
                sentence_index=sentence_index,
                sentence_sha256=sentence_hash,
                extraction_rule=f"surface_pattern:{relation}",
                relevance_score=_candidate_score(question, title, sentence, base=2.0),
            ))
        # A first-sentence copula is a stable encyclopaedic description edge.
        if sentence_index == 0 and not matched:
            copula = re.search(r"\b(?:is|was|are|were)\b\s+(?P<tail>[^.;]+)", sentence, re.IGNORECASE)
            if copula:
                tail = _trim_tail(copula.group("tail"))
                if len(tail) >= 2 and _norm(tail) != _norm(title):
                    result.append(_Candidate(
                        head=title,
                        relation="is",
                        tail=tail,
                        passage_id=passage_id,
                        passage_rank=passage_rank,
                        sentence_index=sentence_index,
                        sentence_sha256=sentence_hash,
                        extraction_rule="first_sentence_copula",
                        relevance_score=_candidate_score(question, title, sentence, base=1.5),
                    ))
    # No semantic relation means no edge. The frozen protocol requires an
    # explicit empty graph; duplicating a sentence as a pseudo-triple is noise.
    return result


def _bridge_candidates(
    question: str,
    passages: Sequence[Mapping[str, Any]],
) -> list[_Candidate]:
    titles = [passage_title(passage) for passage in passages]
    result: list[_Candidate] = []
    for passage_rank, passage in enumerate(passages):
        source_title = titles[passage_rank]
        passage_id = str(passage.get("id") or f"rank-{passage_rank}")
        if not source_title:
            continue
        for sentence_index, sentence in enumerate(passage_sentences(passage)):
            sentence_norm = _norm(sentence)
            for target_rank, target_title in enumerate(titles):
                if target_rank == passage_rank or not target_title:
                    continue
                target_norm = _norm(target_title)
                if len(target_norm) < 4 or f" {target_norm} " not in f" {sentence_norm} ":
                    continue
                result.append(_Candidate(
                    head=source_title,
                    relation="mentions",
                    tail=target_title,
                    passage_id=passage_id,
                    passage_rank=passage_rank,
                    sentence_index=sentence_index,
                    sentence_sha256=hashlib.sha256(sentence.encode("utf-8")).hexdigest(),
                    extraction_rule="cross_passage_title_mention",
                    relevance_score=_candidate_score(question, source_title, sentence, base=2.5),
                ))
    return result


def _select(candidates: Iterable[_Candidate], max_edges: int) -> list[dict[str, Any]]:
    if not 1 <= max_edges <= QPEG_MAX_EDGES:
        raise ValueError(f"max_edges must be in [1, {QPEG_MAX_EDGES}]")
    ordered = sorted(
        candidates,
        key=lambda edge: (
            -edge.relevance_score,
            edge.passage_rank,
            edge.sentence_index,
            _norm(edge.head),
            _norm(edge.relation),
            _norm(edge.tail),
        ),
    )
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in ordered:
        key = (_norm(edge.head), _norm(edge.relation), _norm(edge.tail))
        if not all(key) or key[0] == key[2] or key in seen:
            continue
        seen.add(key)
        selected.append(edge.as_dict())
        if len(selected) == max_edges:
            break
    return selected


def build_qpeg_record(
    *,
    dataset: str,
    qid: str,
    question: str,
    passages: Sequence[Mapping[str, Any]],
    passages_sha256: str | None = None,
    max_edges: int = QPEG_MAX_EDGES,
) -> dict[str, Any]:
    raw_question = str(question).strip()
    clean_question = _clean(raw_question)
    if not clean_question:
        raise ValueError("question must be non-empty")
    if not passages:
        raise ValueError("passages must be non-empty")
    computed_passages_hash = compute_passages_sha256(passages)
    if passages_sha256 is not None and passages_sha256 != computed_passages_hash:
        raise ValueError("passages_sha256 mismatch")

    candidates: list[_Candidate] = []
    for rank, passage in enumerate(passages):
        candidates.extend(_predicate_candidates(clean_question, passage, rank))
    candidates.extend(_bridge_candidates(clean_question, passages))
    edges = _select(candidates, max_edges=max_edges)
    triples = [
        [edge["head_surface"], edge["relation_surface"], edge["tail_surface"]]
        for edge in edges
    ]
    record = {
        "schema_version": QPEG_SCHEMA_VERSION,
        "extractor_version": QPEG_EXTRACTOR_VERSION,
        "question_key": question_key(dataset, qid),
        "dataset": str(dataset).strip().lower(),
        "qid": str(qid).strip(),
        # Preserve the exact stripped dataset question for identity joins.
        # NFKC/whitespace normalization is used only by the extractor.
        "question": raw_question,
        "question_sha256": question_sha256(raw_question),
        "passages_sha256": computed_passages_hash,
        "gold_access": False,
        "max_edges": max_edges,
        "edges": edges,
        "kg_subgraph": triples,
        "provenance_complete": bool(edges),
        "build_status": "nonempty" if edges else "empty",
        "candidate_count": len(candidates),
    }
    record["qpeg_sha256"] = _sha_json({
        "question_key": record["question_key"],
        "question_sha256": record["question_sha256"],
        "passages_sha256": record["passages_sha256"],
        "edges": record["edges"],
    })
    validate_qpeg_record(record, passages=passages)
    return record


def validate_qpeg_record(
    record: Mapping[str, Any],
    *,
    passages: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    if record.get("schema_version") != QPEG_SCHEMA_VERSION:
        raise ValueError("unexpected QPEG schema_version")
    if record.get("extractor_version") not in QPEG_COMPATIBLE_EXTRACTOR_VERSIONS:
        raise ValueError("unexpected QPEG extractor_version")
    expected_key = question_key(str(record.get("dataset") or ""), str(record.get("qid") or ""))
    if record.get("question_key") != expected_key:
        raise ValueError("QPEG question_key mismatch")
    question = str(record.get("question") or "").strip()
    if record.get("question_sha256") != question_sha256(question):
        raise ValueError("QPEG question hash mismatch")
    if record.get("gold_access") is not False:
        raise ValueError("QPEG gold_access must be false")
    max_edges = int(record.get("max_edges") or 0)
    edges = list(record.get("edges") or [])
    triples = list(record.get("kg_subgraph") or [])
    if not 1 <= max_edges <= QPEG_MAX_EDGES or len(edges) > max_edges:
        raise ValueError("QPEG edge budget violation")
    if len(edges) != len(triples):
        raise ValueError("QPEG edge/triple length mismatch")
    required = {
        "head_surface", "relation_surface", "tail_surface", "passage_id",
        "passage_rank", "sentence_index", "sentence_sha256", "extraction_rule",
        "relevance_score",
    }
    seen: set[tuple[str, str, str]] = set()
    sentence_lookup: dict[tuple[int, int], tuple[str, str, str]] = {}
    if passages is not None:
        if record.get("passages_sha256") != compute_passages_sha256(passages):
            raise ValueError("QPEG record/passages hash mismatch")
        for passage_rank, passage in enumerate(passages):
            pid = str(passage.get("id") or f"rank-{passage_rank}")
            for sentence_index, sentence in enumerate(passage_sentences(passage)):
                sentence_lookup[(passage_rank, sentence_index)] = (
                    pid, hashlib.sha256(sentence.encode("utf-8")).hexdigest(), sentence
                )
    for index, (edge, triple) in enumerate(zip(edges, triples)):
        if not isinstance(edge, Mapping) or not required.issubset(edge):
            raise ValueError(f"QPEG edge {index} lacks required provenance")
        expected_triple = [edge["head_surface"], edge["relation_surface"], edge["tail_surface"]]
        if list(triple) != expected_triple:
            raise ValueError(f"QPEG edge {index} does not match kg_subgraph")
        key = tuple(_norm(part) for part in expected_triple)
        if not all(key) or key in seen:
            raise ValueError(f"QPEG edge {index} is empty or duplicate")
        seen.add(key)
        if passages is not None:
            location = (int(edge["passage_rank"]), int(edge["sentence_index"]))
            expected = sentence_lookup.get(location)
            if expected is None:
                raise ValueError(f"QPEG edge {index} points outside passages")
            if (str(edge["passage_id"]), str(edge["sentence_sha256"])) != expected[:2]:
                raise ValueError(f"QPEG edge {index} provenance hash mismatch")
            if _norm(edge["tail_surface"]) not in _norm(expected[2]):
                raise ValueError(f"QPEG edge {index} tail is not present in provenance sentence")
    if bool(edges) != bool(record.get("provenance_complete")):
        raise ValueError("QPEG provenance_complete mismatch")
    expected_status = "nonempty" if edges else "empty"
    if record.get("build_status") != expected_status:
        raise ValueError("QPEG build_status mismatch")
    expected_qpeg_hash = _sha_json({
        "question_key": record["question_key"],
        "question_sha256": record["question_sha256"],
        "passages_sha256": record["passages_sha256"],
        "edges": edges,
    })
    if record.get("qpeg_sha256") != expected_qpeg_hash:
        raise ValueError("QPEG content hash mismatch")
