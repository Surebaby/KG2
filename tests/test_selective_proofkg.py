"""Selective ProofKG augmentation: eval-side integrity + deterministic merge."""

from __future__ import annotations

import pytest

from kgproweight.kg.selective_proofkg import (
    merge_legacy_and_proof_edges,
    select_selective_proof_edges,
    validate_selective_proofkg_record,
)


def _record(**kw):
    base = {
        "schema_version": "selective-proofkg-record-v1",
        "dataset": "hotpotqa", "qid": "q1", "question_sha256": "a" * 64,
        "validator_version": "selective-proofkg-validator-1",
        "historical_cutoff": "2020-12-09T23:59:59Z",
        "partial_eligible": True, "complete_eligible": False,
        "routing_reasons": [], "trusted_edges": [],
    }
    base.update(kw)
    return base


def _edge(head="X", rel="occupation", tail="Q1", qid="Q9", pid="P106", step=1):
    return {
        "head": head, "head_qid": qid, "relation": rel, "pid": pid,
        "tail": tail, "tail_qid": tail if tail.startswith("Q") else None,
        "plan_step_index": step, "provenance": "store",
    }


# --- selection (B/C routing) -------------------------------------------------
def test_partial_eligible_merges():
    rec = _record(partial_eligible=True, trusted_edges=[_edge()])
    assert select_selective_proof_edges(rec, arm="partial") == [("X", "occupation", "Q1")]


def test_partial_no_edge_returns_empty():
    rec = _record(partial_eligible=False, trusted_edges=[])
    assert select_selective_proof_edges(rec, arm="partial") == []


def test_complete_eligible_merges():
    rec = _record(complete_eligible=True, trusted_edges=[_edge()])
    assert select_selective_proof_edges(rec, arm="complete") == [("X", "occupation", "Q1")]


def test_complete_ineligible_returns_empty():
    rec = _record(complete_eligible=False, trusted_edges=[_edge()])
    assert select_selective_proof_edges(rec, arm="complete") == []


# --- merge / truncate --------------------------------------------------------
def test_merge_proof_first_legacy_second():
    legacy = [("A", "r", "B")]
    proof = [("P", "r", "Q")]
    merged, counters = merge_legacy_and_proof_edges(legacy, proof)
    assert merged == [("P", "r", "Q"), ("A", "r", "B")]
    assert counters["proof_retained"] == 1 and counters["legacy_retained"] == 1


def test_merge_dedup_by_canonical_triple():
    legacy = [("A", "r", "B")]
    proof = [("a", "R", "b")]  # same canonical triple
    merged, counters = merge_legacy_and_proof_edges(legacy, proof)
    assert merged == [("a", "R", "b")]
    assert counters["legacy_displaced"] == 1


def test_merge_cap_12():
    legacy = [("L", "r", str(i)) for i in range(20)]
    proof = [("P", "r", str(i)) for i in range(5)]
    merged, _ = merge_legacy_and_proof_edges(legacy, proof, cap=12)
    assert len(merged) == 12


def test_merge_displaced_count_correct():
    legacy = [("A", "r", "B"), ("C", "r", "D")]
    proof = [("A", "r", "B")]  # dup of legacy[0]
    merged, counters = merge_legacy_and_proof_edges(legacy, proof)
    assert counters["proof_retained"] == 1
    assert counters["legacy_displaced"] == 1
    assert counters["legacy_retained"] == 1


# --- integrity fail-fast -----------------------------------------------------
def test_validate_bad_schema_fails():
    with pytest.raises(ValueError):
        validate_selective_proofkg_record(_record(schema_version="wrong"), dataset="hotpotqa", qid="q1")


def test_validate_identity_mismatch_fails():
    with pytest.raises(ValueError):
        validate_selective_proofkg_record(_record(qid="q2"), dataset="hotpotqa", qid="q1")


def test_validate_bad_validator_fails():
    with pytest.raises(ValueError):
        validate_selective_proofkg_record(_record(validator_version="old"), dataset="hotpotqa", qid="q1")


def test_validate_edge_missing_qid_fails():
    rec = _record(trusted_edges=[{"head": "X", "relation": "r", "tail": "t", "plan_step_index": 1}])
    with pytest.raises(ValueError):
        validate_selective_proofkg_record(rec, dataset="hotpotqa", qid="q1")


def test_validate_duplicate_edge_fails():
    rec = _record(trusted_edges=[_edge(), _edge()])
    with pytest.raises(ValueError):
        validate_selective_proofkg_record(rec, dataset="hotpotqa", qid="q1")


def test_validate_good_record_passes():
    validate_selective_proofkg_record(_record(trusted_edges=[_edge()]), dataset="hotpotqa", qid="q1")
