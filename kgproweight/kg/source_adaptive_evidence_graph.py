"""Versioned schema for source-adaptive passage/Wikidata evidence graphs."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_key, question_sha256


SAEG_SCHEMA_VERSION = "saeg-question-record-v1"
SAEG_EDGE_SCHEMA_VERSION = "saeg-edge-v1"
SOURCE_TYPES = {"passage", "wikidata"}
EDGE_TYPES = {"evidence_sentence", "relational_fact"}
ROUTING_MODES = {"P_ONLY", "W_ONLY", "P_W_FUSED", "N_REPLAY"}
_SPACE = re.compile(r"\s+")


def _clean(value: object) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", str(value or ""))).strip()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def canonical_triple(value: Sequence[object]) -> list[str]:
    if len(value) != 3:
        raise ValueError(f"evidence triple must have three components: {value!r}")
    triple = [_clean(part) for part in value]
    if not all(triple):
        raise ValueError(f"evidence triple components must be non-empty: {value!r}")
    return triple


def passages_sha256(passages: Sequence[Mapping[str, Any]]) -> str:
    """Hash semantic passage content while ignoring source-specific row ids."""
    payload = []
    for rank, passage in enumerate(passages):
        payload.append({
            "rank": rank,
            "title": _clean(passage.get("title")),
            "contents": _clean(passage.get("contents")),
        })
    return sha256_json(payload)


def make_passage_edge(
    *,
    edge_index: int,
    triple: Sequence[object],
    passage_id: str,
    passage_rank: int,
    sentence_index: int,
    construction_gold_access: bool,
) -> dict[str, Any]:
    canonical = canonical_triple(triple)
    if passage_rank < 0 or sentence_index < 0:
        raise ValueError("passage_rank and sentence_index must be non-negative")
    edge = {
        "schema_version": SAEG_EDGE_SCHEMA_VERSION,
        "edge_id": f"P{edge_index}",
        "source_type": "passage",
        "edge_type": "evidence_sentence",
        "triple": canonical,
        "valid": True,
        "construction_gold_access": bool(construction_gold_access),
        "provenance": {
            "passage_id": _clean(passage_id),
            "passage_rank": int(passage_rank),
            "sentence_index": int(sentence_index),
            "sentence_sha256": hashlib.sha256(canonical[2].encode("utf-8")).hexdigest(),
        },
    }
    validate_edge(edge)
    return edge


def make_wikidata_edge(
    *,
    edge_index: int,
    triple: Sequence[object],
    hop_index: int,
    input_qids: Sequence[str],
    pid: str,
    tail_qid: str | None,
    cutoff: str,
    builder_version: str,
) -> dict[str, Any]:
    canonical = canonical_triple(triple)
    qids = [_clean(value) for value in input_qids if _clean(value)]
    if hop_index < 1 or not qids:
        raise ValueError("Wikidata edge requires hop_index>=1 and at least one input QID")
    if not re.fullmatch(r"P\d+", _clean(pid)):
        raise ValueError(f"invalid Wikidata PID: {pid!r}")
    if any(not re.fullmatch(r"Q\d+", value) for value in qids):
        raise ValueError(f"invalid input QID list: {qids!r}")
    clean_tail_qid = _clean(tail_qid)
    if clean_tail_qid and not re.fullmatch(r"Q\d+", clean_tail_qid):
        raise ValueError(f"invalid tail QID: {tail_qid!r}")
    edge = {
        "schema_version": SAEG_EDGE_SCHEMA_VERSION,
        "edge_id": f"W{edge_index}",
        "source_type": "wikidata",
        "edge_type": "relational_fact",
        "triple": canonical,
        "valid": True,
        "construction_gold_access": False,
        "provenance": {
            "hop_index": int(hop_index),
            "input_qids": qids,
            "pid": _clean(pid),
            "tail_value": canonical[2],
            "tail_qid": clean_tail_qid or None,
            "historical_cutoff": _clean(cutoff),
            "builder_version": _clean(builder_version),
        },
    }
    validate_edge(edge)
    return edge


def validate_edge(edge: Mapping[str, Any]) -> None:
    if edge.get("schema_version") != SAEG_EDGE_SCHEMA_VERSION:
        raise ValueError("unexpected SAEG edge schema")
    source = str(edge.get("source_type") or "")
    edge_type = str(edge.get("edge_type") or "")
    if source not in SOURCE_TYPES or edge_type not in EDGE_TYPES:
        raise ValueError("invalid SAEG edge source/type")
    if (source, edge_type) not in {
        ("passage", "evidence_sentence"),
        ("wikidata", "relational_fact"),
    }:
        raise ValueError("source_type and edge_type disagree")
    if edge.get("valid") is not True:
        raise ValueError("materialized SAEG edges must be valid")
    canonical_triple(edge.get("triple") or [])
    provenance = edge.get("provenance") or {}
    if source == "passage":
        required = {"passage_id", "passage_rank", "sentence_index", "sentence_sha256"}
        if not required <= set(provenance):
            raise ValueError("passage edge provenance incomplete")
        expected = hashlib.sha256(canonical_triple(edge["triple"])[2].encode("utf-8")).hexdigest()
        if provenance.get("sentence_sha256") != expected:
            raise ValueError("passage sentence hash mismatch")
    else:
        required = {
            "hop_index", "input_qids", "pid", "tail_value", "historical_cutoff", "builder_version"
        }
        if not required <= set(provenance):
            raise ValueError("Wikidata edge provenance incomplete")
        if not re.fullmatch(r"P\d+", str(provenance.get("pid") or "")):
            raise ValueError("Wikidata edge PID invalid")
        if not provenance.get("input_qids"):
            raise ValueError("Wikidata edge input_qids empty")


def fuse_edges(
    wikidata_edges: Sequence[Mapping[str, Any]],
    passage_edges: Sequence[Mapping[str, Any]],
    *,
    cap: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep W required-hop edges first, then fill remaining slots from P."""
    if cap < 1:
        raise ValueError("edge cap must be positive")
    retained: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    duplicate_removed = 0
    for raw in list(wikidata_edges) + list(passage_edges):
        validate_edge(raw)
        key = tuple(part.casefold() for part in canonical_triple(raw["triple"]))
        if key in seen:
            duplicate_removed += 1
            continue
        seen.add(key)
        if len(retained) < cap:
            retained.append(dict(raw))
    # IDs are local to the final prompt graph and must be deterministic after
    # truncation, while source is kept explicit.
    counters = {"passage": 0, "wikidata": 0}
    for edge in retained:
        source = str(edge["source_type"])
        counters[source] += 1
        edge["edge_id"] = f"{'P' if source == 'passage' else 'W'}{counters[source]}"
    telemetry = {
        "cap": cap,
        "wikidata_input": len(wikidata_edges),
        "passage_input": len(passage_edges),
        "wikidata_retained": counters["wikidata"],
        "passage_retained": counters["passage"],
        "duplicate_removed": duplicate_removed,
        "truncated": max(0, len(wikidata_edges) + len(passage_edges) - duplicate_removed - cap),
    }
    return retained, telemetry


def make_record(
    *,
    dataset: str,
    qid: str,
    question: str,
    passages: Sequence[Mapping[str, Any]],
    routing_mode: str,
    edges: Sequence[Mapping[str, Any]],
    routing: Mapping[str, Any],
) -> dict[str, Any]:
    if routing_mode not in ROUTING_MODES:
        raise ValueError(f"invalid routing mode: {routing_mode!r}")
    clean_edges = [dict(edge) for edge in edges]
    for edge in clean_edges:
        validate_edge(edge)
    if routing_mode == "N_REPLAY" and clean_edges:
        raise ValueError("N_REPLAY must have an empty graph")
    if routing_mode != "N_REPLAY" and not clean_edges:
        raise ValueError(f"{routing_mode} must have at least one edge")
    expected_sources = {
        "P_ONLY": {"passage"},
        "W_ONLY": {"wikidata"},
        "P_W_FUSED": {"passage", "wikidata"},
        "N_REPLAY": set(),
    }[routing_mode]
    actual_sources = {str(edge["source_type"]) for edge in clean_edges}
    if routing_mode != "P_W_FUSED" and actual_sources != expected_sources:
        raise ValueError("routing mode and edge sources disagree")
    if routing_mode == "P_W_FUSED" and not actual_sources <= expected_sources:
        raise ValueError("fused record contains an unknown edge source")
    record = {
        "schema_version": SAEG_SCHEMA_VERSION,
        "record_id": f"{question_key(dataset, qid)}::{routing_mode}",
        "question_key": question_key(dataset, qid),
        "dataset": str(dataset).strip().lower(),
        "qid": str(qid).strip(),
        "question": str(question).strip(),
        "question_sha256": question_sha256(question),
        "passages_sha256": passages_sha256(passages),
        "routing": {"mode": routing_mode, **dict(routing)},
        "construction_gold_access": any(bool(edge.get("construction_gold_access")) for edge in clean_edges),
        "evaluation_eligible": False,
        "edges": clean_edges,
    }
    record["graph_sha256"] = sha256_json(clean_edges)
    validate_record(record)
    return record


def validate_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != SAEG_SCHEMA_VERSION:
        raise ValueError("unexpected SAEG record schema")
    expected_key = question_key(str(record.get("dataset") or ""), str(record.get("qid") or ""))
    if record.get("question_key") != expected_key:
        raise ValueError("SAEG question_key mismatch")
    mode = str((record.get("routing") or {}).get("mode") or "")
    if mode not in ROUTING_MODES:
        raise ValueError("SAEG routing mode invalid")
    question = str(record.get("question") or "").strip()
    if record.get("question_sha256") != question_sha256(question):
        raise ValueError("SAEG question hash mismatch")
    edges = record.get("edges") or []
    for edge in edges:
        validate_edge(edge)
    if record.get("graph_sha256") != sha256_json(edges):
        raise ValueError("SAEG graph hash mismatch")
    if record.get("evaluation_eligible") is not False:
        raise ValueError("train-only SAEG assets must be evaluation-ineligible")
    if mode == "N_REPLAY" and edges:
        raise ValueError("N_REPLAY graph must be empty")

