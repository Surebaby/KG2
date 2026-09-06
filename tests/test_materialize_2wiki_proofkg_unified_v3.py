from __future__ import annotations

import json

import pytest

from kgproweight.kg.question_kg import make_question_kg_record, question_sha256
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.prepare.materialize_2wiki_official_raw_canonical_retrieval_v1 import (
    STATUS as OFFICIAL_RETRIEVAL_STATUS,
)
from scripts.prepare.materialize_2wiki_proofkg_unified_v2 import (
    SCHEMA_VERSION as LEGACY_V2_SCHEMA,
    STATUS as LEGACY_V2_STATUS,
    _json_sha256,
)
from scripts.prepare.materialize_2wiki_proofkg_unified_v3 import (
    CANDIDATE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SOURCE_RELEASE,
    build_official_raw_supply,
)
from scripts.prepare.audit_auto1500_v4_clean_reproducibility import canonical_sha256


def _fixture(qid: str = "q1", qtype: str = "inference"):
    question = "Who is Alice's mother?"
    plan = {
        "recognized": True,
        "anchors": ["Alice"],
        "hops": [
            {
                "subject": "Alice",
                "pids": ["P25"],
                "output_slot": "hop_1",
            }
        ],
    }
    triple = ["Alice", "mother", "Carol"]
    runtime = make_question_kg_record(
        dataset="2wikimultihopqa",
        qid=qid,
        question=question,
        triples=[triple],
        query_plan=plan,
        provenance={
            "builder_version": "closure-v3-test",
            "planner_predictions_sha256": "f" * 64,
            "gold_access": False,
            "complete_plan_execution": True,
        },
    )
    runtime.update(
        {
            "planner_schema_valid": True,
            "execution": {
                "complete_plan_execution": True,
                "anchor_entities": {"Alice": {"qid": "Q1"}},
                "hops": [
                    {
                        "hop_index": 1,
                        "input_entities": [{"qid": "Q1"}],
                        "matches": [triple],
                    }
                ],
            },
            "runtime_error": None,
        }
    )
    key = f"2wikimultihopqa::{qid}"
    request = {
        "question_key": key,
        "dataset": "2wikimultihopqa",
        "qid": qid,
        "question": question,
        "question_sha256": question_sha256(question),
        "family_version": FAMILY_VERSION,
        "family_sha256": family_sha256(question),
        "question_type": qtype,
        "role": "official_raw_proofkg_rollout_retrieval",
        "gold_access": False,
        "evaluation_eligible": False,
        "closure_runtime_record_sha256": canonical_sha256(runtime),
        "closure_kg_sha256": canonical_sha256(runtime["kg_subgraph"]),
        "closure_execution_sha256": canonical_sha256(runtime["execution"]),
    }
    passages = [
        {"id": f"p{i}", "source": "canonical", "contents": f"Evidence {i}."}
        for i in range(10)
    ]
    retrieval = {
        **{field: request[field] for field in (
            "question_key", "dataset", "qid", "question", "question_sha256",
            "family_sha256", "role", "gold_access",
        )},
        "retrieval_source": (
            "E5@100+BM25@100->RRF60@50->bge-reranker-v2-m3@10->pack3860"
        ),
        "passages": passages,
        "passages_sha256": _json_sha256(passages),
    }
    raw = {"id": qid, "question": question, "golden_answers": ["Carol"]}
    return request, runtime, raw, retrieval


def test_official_raw_v3_has_unambiguous_source_and_exact_hash_join():
    request, runtime, raw, retrieval = _fixture()
    silver, records, gates, wrappers, _stats = build_official_raw_supply(
        requests=[request],
        runtime_rows=[runtime],
        raw_rows=[raw],
        retrieval_rows=[retrieval],
        blocked_qids=set(),
        blocked_hashes=set(),
        blocked_families=set(),
        cutoff="2020-12-09T23:59:59Z",
    )
    assert len(silver) == len(records) == len(gates) == len(wrappers) == 1
    assert silver[0]["steps"] == []
    assert silver[0]["metadata"]["proof_source"] == SOURCE_RELEASE
    assert silver[0]["metadata"]["unified_supply_version"] == SCHEMA_VERSION
    assert len(silver[0]["retrieved_passages"]) == 10
    assert records[0]["provenance"]["unified_source_release"] == SOURCE_RELEASE
    assert records[0]["provenance"]["unified_supply_version"] == SCHEMA_VERSION
    assert records[0]["provenance"]["source_gold_steps_copied"] == 0
    assert records[0]["provenance"]["canonical_retrieval_release"] == (
        OFFICIAL_RETRIEVAL_STATUS
    )
    assert gates[0]["m_graph"] == 1
    assert wrappers[0]["schema_version"] == CANDIDATE_SCHEMA_VERSION
    assert wrappers[0]["source_release"] == SOURCE_RELEASE
    assert "answer" not in wrappers[0]


def test_official_raw_v3_rejects_runtime_hash_drift():
    request, runtime, raw, retrieval = _fixture()
    request["closure_execution_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="closure hash mismatch"):
        build_official_raw_supply(
            requests=[request],
            runtime_rows=[runtime],
            raw_rows=[raw],
            retrieval_rows=[retrieval],
            blocked_qids=set(),
            blocked_hashes=set(),
            blocked_families=set(),
            cutoff="2020-12-09T23:59:59Z",
        )


def test_official_raw_v3_fails_closed_on_protected_family():
    request, runtime, raw, retrieval = _fixture()
    with pytest.raises(RuntimeError, match="strict scope lost rows"):
        build_official_raw_supply(
            requests=[request],
            runtime_rows=[runtime],
            raw_rows=[raw],
            retrieval_rows=[retrieval],
            blocked_qids=set(),
            blocked_hashes=set(),
            blocked_families={request["family_sha256"]},
            cutoff="2020-12-09T23:59:59Z",
        )


def test_v2_contract_remains_distinct_and_unchanged():
    # v3 is additive; historical reserve/extension releases remain consumable
    # only under their own v2 schema/status.
    assert LEGACY_V2_STATUS == "COMPLETE_STRICT_CANDIDATE_SUPPLY_NOT_SELECTED_NOT_TRAINED"
    assert LEGACY_V2_SCHEMA == "2wiki-unified-proofkg-candidate-supply-v2"
    assert LEGACY_V2_SCHEMA != SCHEMA_VERSION
    assert SOURCE_RELEASE != "automatic_proofkg_extension_reserve_v1"
