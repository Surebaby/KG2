"""Versioned identity/schema helpers for per-question KG artifacts.

The legacy consumers key by raw question text.  New query-aware pilots use the
stable ``dataset::qid`` identity and retain a question hash as a fail-fast
guard against joining the right id to the wrong text.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


QUESTION_KG_SCHEMA_VERSION = "question-kg-by-dataset-qid-1"
Triple = Tuple[str, str, str]


def question_sha256(question: str) -> str:
    return hashlib.sha256(str(question).strip().encode("utf-8")).hexdigest()


def question_key(dataset: str, qid: str) -> str:
    clean_dataset = str(dataset).strip().lower()
    clean_qid = str(qid).strip()
    if not clean_dataset or not clean_qid or "::" in clean_dataset:
        raise ValueError("dataset and qid must be non-empty; dataset cannot contain '::'")
    return f"{clean_dataset}::{clean_qid}"


def _triples(values: Iterable[Sequence[object]]) -> List[List[str]]:
    result: List[List[str]] = []
    seen: set[Triple] = set()
    for value in values:
        if len(value) != 3:
            raise ValueError(f"invalid KG triple: {value!r}")
        triple = tuple(str(part).strip() for part in value)
        if not all(triple):
            raise ValueError(f"KG triple components must be non-empty: {value!r}")
        if triple not in seen:
            seen.add(triple)
            result.append(list(triple))
    return result


def make_question_kg_record(
    *,
    dataset: str,
    qid: str,
    question: str,
    triples: Iterable[Sequence[object]],
    query_plan: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    clean_question = str(question).strip()
    if not clean_question:
        raise ValueError("question must be non-empty")
    return {
        "schema_version": QUESTION_KG_SCHEMA_VERSION,
        "question_key": question_key(dataset, qid),
        "dataset": str(dataset).strip().lower(),
        "qid": str(qid).strip(),
        "question": clean_question,
        "question_sha256": question_sha256(clean_question),
        "kg_subgraph": _triples(triples),
        "query_plan": dict(query_plan or {}),
        "provenance": dict(provenance or {}),
    }


def validate_question_kg_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != QUESTION_KG_SCHEMA_VERSION:
        raise ValueError(f"unexpected question KG schema: {record.get('schema_version')!r}")
    expected_key = question_key(str(record.get("dataset") or ""), str(record.get("qid") or ""))
    if record.get("question_key") != expected_key:
        raise ValueError("question_key does not match dataset/qid")
    question = str(record.get("question") or "").strip()
    if record.get("question_sha256") != question_sha256(question):
        raise ValueError("question hash mismatch")
    _triples(record.get("kg_subgraph") or [])


def load_question_kg_index(records: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for value in records:
        validate_question_kg_record(value)
        key = str(value["question_key"])
        if key in index:
            raise ValueError(f"duplicate question KG key: {key}")
        index[key] = dict(value)
    return index
