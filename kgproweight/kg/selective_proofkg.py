"""Selective ProofKG augmentation: eval-side integrity + deterministic merge.

The heavy per-edge semantic validation happens OFFLINE (see
scripts/prepare/build_selective_proofkg_records.py).  Eval only does identity /
structure integrity checks and a deterministic merge, so a corrupt or mismatched
record file FAILS FAST instead of silently falling back to legacy.

Two failure classes are deliberately distinct:

  * a LEGAL record with no trusted edges (or, for arm C, not complete) -> exact
    legacy fallback;
  * a MISSING file / hash mismatch / duplicate key / identity mismatch / bad
    schema -> fail-fast (never a silent legacy fallback).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

RECORD_SCHEMA = "selective-proofkg-record-v1"
VALIDATOR_VERSION = "selective-proofkg-validator-1"
MAX_KG_TRIPLES = 12
_QID = re.compile(r"^Q[1-9][0-9]*$")
_PID = re.compile(r"^P[1-9][0-9]*$")


def _norm(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def validate_selective_proofkg_record(record: Dict[str, Any], *, dataset: str, qid: str) -> None:
    """Fail-fast on a corrupt/mismatched record.  Raises ValueError on any problem."""
    if not isinstance(record, dict):
        raise ValueError("record is not a dict")
    if record.get("schema_version") != RECORD_SCHEMA:
        raise ValueError(f"bad schema_version: {record.get('schema_version')!r}")
    if record.get("dataset") != dataset or str(record.get("qid")) != str(qid):
        raise ValueError(f"identity mismatch: {record.get('dataset')}:{record.get('qid')} != {dataset}:{qid}")
    if not record.get("question_sha256"):
        raise ValueError("missing question_sha256")
    if record.get("validator_version") != VALIDATOR_VERSION:
        raise ValueError(f"bad validator_version: {record.get('validator_version')!r}")
    if "partial_eligible" not in record or "complete_eligible" not in record:
        raise ValueError("missing eligibility fields")
    if "routing_reasons" not in record:
        raise ValueError("missing routing_reasons")
    edges = record.get("trusted_edges")
    if not isinstance(edges, list):
        raise ValueError("trusted_edges is not a list")
    seen = set()
    for e in edges:
        if not isinstance(e, dict):
            raise ValueError("edge is not a dict")
        for f in ("head", "relation", "tail", "plan_step_index"):
            if f not in e:
                raise ValueError(f"edge missing field {f}")
        if not _QID.match(str(e.get("head_qid") or "")):
            raise ValueError(f"edge missing head_qid: {e.get('head_qid')!r}")
        if not _PID.match(str(e.get("pid") or "")):
            raise ValueError(f"edge missing pid: {e.get('pid')!r}")
        if not isinstance(e.get("plan_step_index"), int) or e["plan_step_index"] < 1:
            raise ValueError(f"bad plan_step_index: {e.get('plan_step_index')!r}")
        key = (_norm(e["head"]), _norm(e["relation"]), _norm(e["tail"]))
        if key in seen:
            raise ValueError(f"duplicate edge: {key}")
        seen.add(key)


def select_selective_proof_edges(record: Dict[str, Any], *, arm: str) -> List[Tuple[str, str, str]]:
    """Return the trusted Proof edges as (head, relation, tail) triples for the arm.

    Arm 'partial' (B): edges if partial_eligible, else [] (legacy fallback).
    Arm 'complete' (C): edges if complete_eligible, else [].
    """
    eligible = record.get("complete_eligible" if arm == "complete" else "partial_eligible")
    if not eligible:
        return []
    return [(e["head"], e["relation"], e["tail"]) for e in record.get("trusted_edges") or []]


def merge_legacy_and_proof_edges(
    legacy: Sequence[Sequence[str]], proof_edges: Sequence[Sequence[str]], *, cap: int = MAX_KG_TRIPLES
) -> Tuple[List[Tuple[str, str, str]], Dict[str, int]]:
    """Deterministic merge: proof-first (plan-step order), legacy fills the rest.

    Returns (merged triples, counters).  Dedup by normalized (head, relation, tail).
    """
    merged: List[Tuple[str, str, str]] = []
    seen = set()
    proof_retained = legacy_retained = legacy_displaced = duplicate_removed = 0

    for edge in proof_edges:
        t = tuple(str(x).strip() for x in edge)
        if len(t) != 3:
            continue
        key = (_norm(t[0]), _norm(t[1]), _norm(t[2]))
        if key in seen:
            duplicate_removed += 1
            continue
        seen.add(key)
        merged.append(t)
        proof_retained += 1
        if len(merged) >= cap:
            break

    if len(merged) < cap:
        for edge in legacy:
            t = tuple(str(x).strip() for x in edge)
            if len(t) != 3:
                continue
            key = (_norm(t[0]), _norm(t[1]), _norm(t[2]))
            if key in seen:
                legacy_displaced += 1
                continue
            seen.add(key)
            merged.append(t)
            legacy_retained += 1
            if len(merged) >= cap:
                break

    counters = {
        "proof_retained": proof_retained,
        "legacy_retained": legacy_retained,
        "legacy_displaced": legacy_displaced,
        "duplicate_removed": duplicate_removed,
    }
    return merged, counters
