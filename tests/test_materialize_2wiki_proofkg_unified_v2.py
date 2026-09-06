from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kgproweight.kg.question_kg import make_question_kg_record, question_sha256
from scripts.prepare.freeze_mixed_ppo_three_dataset_v4_proof800 import (
    normalise_proof_candidates,
)
from scripts.prepare.materialize_2wiki_proofkg_unified_v2 import (
    CANONICAL_RETRIEVAL_STACK,
    CLEAN_REEXECUTION_SCHEMA_VERSION,
    COMPLETE_PROTECTED_LEDGER_VERSION,
    _fallback_source_from_raw_retrieval,
    _validate_old_attestation_release,
    build_candidate_wrappers,
    build_supply,
)
from scripts.prepare.audit_auto1500_v4_clean_reproducibility import (
    QUESTION_EDGE_SCHEMA_VERSION,
    QUESTION_ROOT_SCHEMA_VERSION,
    canonical_sha256,
    SCHEMA_VERSION as ATTESTATION_RELEASE_SCHEMA,
    STATUS as ATTESTATION_RELEASE_STATUS,
)


def _rows(qid: str, question: str, qtype: str = "inference"):
    plan = {
        "recognized": True,
        "hops": [{"subject": "Alice", "pids": ["P25"], "output_slot": "hop_1"}],
    }
    triple = ["Alice", "mother", "Carol"]
    base = make_question_kg_record(
        dataset="2wikimultihopqa", qid=qid, question=question,
        triples=[triple], query_plan=plan,
        provenance={"builder_version": "test-builder", "gold_access": False, "complete_plan_execution": True},
    )
    runtime = dict(base)
    runtime.update({
        "execution": {
            "complete_plan_execution": True,
            "hops": [{"hop_index": 1, "input_entities": [{"qid": "Q1"}], "matches": [triple]}],
        },
        "runtime_error": None,
    })
    silver = {
        "qid": qid, "dataset": "2wikimultihopqa", "question": question,
        "answer": "Carol", "kg_subgraph": [["GOLD", "derived", "must disappear"]],
        "steps": [{"reasoning": "must disappear"}],
        "retrieved_passages": [{"id": str(i), "source": "wiki", "contents": "text"} for i in range(10)],
        "metadata": {"gold_answer": "Carol", "question_type": qtype},
    }
    cohort = {
        "question_key": f"2wikimultihopqa::{qid}", "dataset": "2wikimultihopqa",
        "qid": qid, "question": question, "question_sha256": question_sha256(question),
        "question_type": qtype,
    }
    return silver, base, runtime, cohort


def _attestations(runtime: dict, *, edge_ok: bool = True, root_ok: bool = True):
    common = {
        "question_key": runtime["question_key"],
        "dataset": runtime["dataset"],
        "qid": runtime["qid"],
        "question_sha256": runtime["question_sha256"],
        "old_runtime_sha256": canonical_sha256(runtime),
        "old_plan_sha256": canonical_sha256(runtime["query_plan"]),
        "old_execution_sha256": canonical_sha256(runtime["execution"]),
        "historical_cutoff": "2020-12-09T23:59:59Z",
        "gold_access": False,
    }
    edge = {
        **common,
        "schema_version": QUESTION_EDGE_SCHEMA_VERSION,
        "kg_sha256": canonical_sha256(runtime["kg_subgraph"]),
        "executed_edge_count": 1,
        "independently_reproduced_edge_count": 1 if edge_ok else 0,
        "all_executed_edges_independently_reproduced": edge_ok,
        "edge_details_sha256": "a" * 64,
    }
    root = {
        **common,
        "schema_version": QUESTION_ROOT_SCHEMA_VERSION,
        "root_anchor_count": 1,
        "independently_attested_root_count": 1 if root_ok else 0,
        "all_root_anchors_independently_attested": root_ok,
        "root_details_sha256": "b" * 64,
    }
    return edge, root


def test_unified_supply_strips_gold_process_fields_and_passes_strict_gate():
    old_silver, base, old_runtime, _cohort = _rows("old", "Who is Alice's mother?")
    edge, root = _attestations(old_runtime)
    new_silver, _new_base, new_runtime, new_cohort = _rows("new", "Who is Bob's mother?")
    silver, records, gates, stats = build_supply(
        old_silver=[old_silver], old_records=[base], old_runtime=[old_runtime],
        new_source=[new_silver], new_cohort=[new_cohort], new_runtime=[new_runtime],
        blocked_qids=set(), blocked_hashes=set(), blocked_families=set(),
        cutoff="2020-12-09T23:59:59Z",
        old_edge_attestations=[edge], old_root_attestations=[root],
    )
    assert len(silver) == len(records) == len(gates) == 2
    assert all(row["steps"] == [] for row in silver)
    assert all(row["kg_subgraph"] == [["Alice", "mother", "Carol"]] for row in silver)
    assert all(row["teacher_output"] == "" for row in silver)
    assert all(row["m_graph"] == 1 for row in gates)
    assert stats["eligible_by_source"] == {
        "automatic_proofkg_extension_v1": 1,
        "automatic_proofkg_train_k4_v1": 1,
    }
    old_record = next(
        row
        for row in records
        if row["qid"] == "old"
    )
    assert old_record["provenance"]["old_trace_admission"]["mode"] == (
        "independent_edge_plus_root_attestation"
    )

    wrappers = build_candidate_wrappers(silver, records, gates)
    assert [row["question_type"] for row in wrappers] == ["inference", "inference"]
    assert all("answer" not in row and "steps" not in row for row in wrappers)
    downstream, reasons = normalise_proof_candidates(
        wrappers, historical_cutoff="2020-12-09T23:59:59Z"
    )
    assert len(downstream) == 2
    assert reasons == {"eligible": 2}


def test_unified_supply_excludes_current_family_before_graph_selection():
    source, base, runtime, _cohort = _rows("old", "Who is Alice's mother?")
    from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256

    silver, records, gates, stats = build_supply(
        old_silver=[source], old_records=[base], old_runtime=[runtime],
        new_source=[], new_cohort=[], new_runtime=[],
        blocked_qids=set(), blocked_hashes=set(),
        blocked_families={family_sha256(source["question"])},
        cutoff="2020-12-09T23:59:59Z",
    )
    assert silver == records == gates == []
    assert stats["excluded"] == {"family_sha256": 1}


@pytest.mark.parametrize(
    ("edge_ok", "root_ok", "reason"),
    [
        (False, True, "edge_attestation_not_complete"),
        (True, False, "root_attestation_not_complete"),
    ],
)
def test_old_runtime_attestation_contract_fails_closed(edge_ok, root_ok, reason):
    source, base, runtime, _cohort = _rows("old", "Who is Alice's mother?")
    edge, root = _attestations(runtime, edge_ok=edge_ok, root_ok=root_ok)
    silver, records, gates, stats = build_supply(
        old_silver=[source], old_records=[base], old_runtime=[runtime],
        old_edge_attestations=[edge], old_root_attestations=[root],
        new_source=[], new_cohort=[], new_runtime=[],
        blocked_qids=set(), blocked_hashes=set(), blocked_families=set(),
        cutoff="2020-12-09T23:59:59Z",
    )
    assert silver == records == gates == []
    assert stats["excluded"] == {f"old_admission:{reason}": 1}


def test_old_runtime_missing_or_hash_drifted_attestation_is_not_reused():
    source, base, runtime, _cohort = _rows("old", "Who is Alice's mother?")
    edge, root = _attestations(runtime)
    edge["old_execution_sha256"] = "0" * 64
    silver, records, gates, stats = build_supply(
        old_silver=[source], old_records=[base], old_runtime=[runtime],
        old_edge_attestations=[edge], old_root_attestations=[root],
        new_source=[], new_cohort=[], new_runtime=[],
        blocked_qids=set(), blocked_hashes=set(), blocked_families=set(),
        cutoff="2020-12-09T23:59:59Z",
    )
    assert silver == records == gates == []
    assert stats["excluded"] == {
        "old_admission:edge_attestation_old_execution_sha256_mismatch": 1
    }


def _clean_runtime(runtime: dict, *, valid_contract: bool = True) -> dict:
    clean = json.loads(json.dumps(runtime))
    clean["kg_subgraph"] = [["Alice", "mother", "Dora"]]
    clean["execution"]["hops"][0]["matches"] = [["Alice", "mother", "Dora"]]
    clean["provenance"] = {
        **clean["provenance"],
        "gold_access": False,
        "clean_reexecution_contract": {
            "schema_version": CLEAN_REEXECUTION_SCHEMA_VERSION,
            "root_resolver_input": "question_and_root_anchor_surfaces_only",
            "expected_old_qids_used_as_resolver_targets": False,
            "old_v2_store_used": False,
            "gold_access": False,
            "historical_cutoff": "2020-12-09T23:59:59Z",
            "clean_store_manifest_sha256": "1" * 64,
            "historical_cache_sha256": "2" * 64,
            "resolver_code_sha256": "3" * 64,
            "executor_code_sha256": "4" * 64,
        },
    }
    if not valid_contract:
        clean["provenance"]["clean_reexecution_contract"].pop(
            "resolver_code_sha256"
        )
    return clean


def test_new_clean_reexecution_can_replace_old_graph_without_old_qid_target():
    source, base, runtime, _cohort = _rows("old", "Who is Alice's mother?")
    clean = _clean_runtime(runtime)
    silver, records, gates, stats = build_supply(
        old_silver=[source], old_records=[base], old_runtime=[runtime],
        old_edge_attestations=[], old_root_attestations=[],
        old_clean_runtime=[clean],
        new_source=[], new_cohort=[], new_runtime=[],
        blocked_qids=set(), blocked_hashes=set(), blocked_families=set(),
        cutoff="2020-12-09T23:59:59Z",
    )
    assert len(silver) == len(records) == len(gates) == 1
    assert records[0]["kg_subgraph"] == [["Alice", "mother", "Dora"]]
    assert records[0]["provenance"]["old_trace_admission"]["mode"] == (
        "new_clean_reexecuted_runtime"
    )
    assert stats["old_trace_admission_modes"] == {
        "new_clean_reexecuted_runtime": 1
    }


def test_unbound_clean_reexecution_boolean_is_insufficient_and_fails_closed():
    source, base, runtime, _cohort = _rows("old", "Who is Alice's mother?")
    clean = _clean_runtime(runtime, valid_contract=False)
    silver, records, gates, stats = build_supply(
        old_silver=[source], old_records=[base], old_runtime=[runtime],
        old_edge_attestations=[], old_root_attestations=[],
        old_clean_runtime=[clean],
        new_source=[], new_cohort=[], new_runtime=[],
        blocked_qids=set(), blocked_hashes=set(), blocked_families=set(),
        cutoff="2020-12-09T23:59:59Z",
    )
    assert silver == records == gates == []
    assert stats["excluded"] == {
        "old_admission:clean_runtime_contract_resolver_code_sha256_missing": 1
    }


def test_attestation_release_requires_bound_report_manifest_and_hashes(tmp_path: Path):
    release = tmp_path / "attestation"
    release.mkdir()
    edge = release / "question_edge_attestations.jsonl"
    root = release / "question_root_attestations.jsonl"
    edge.write_text(
        json.dumps(
            {
                "schema_version": QUESTION_EDGE_SCHEMA_VERSION,
                "question_key": "2wikimultihopqa::q1",
                "all_executed_edges_independently_reproduced": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    root.write_text(
        json.dumps(
            {
                "schema_version": QUESTION_ROOT_SCHEMA_VERSION,
                "question_key": "2wikimultihopqa::q1",
                "all_root_anchors_independently_attested": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    protected = {
        name: {"sha256": hashlib.sha256(name.encode()).hexdigest()}
        for name in ("ledger", "report", "manifest")
    }
    report = {
        "schema_version": ATTESTATION_RELEASE_SCHEMA,
        "status": ATTESTATION_RELEASE_STATUS,
        "checks": {"all_edges": True, "root_partition": True},
        "protected_ledger": {
            "version": COMPLETE_PROTECTED_LEDGER_VERSION,
            "complete": True,
            "current_family_recomputed": True,
            **protected,
        },
        "counts": {
            "protected_safe_strict_questions": 1,
            "executed_edges": 1,
            "independently_reproduced_edges": 1,
            "all_edges_attested_questions": 1,
            "all_roots_attested_questions": 1,
            "reresolution_worklist_questions": 0,
        },
        "outputs": {
            "question_edge_attestations": {"sha256": digest(edge)},
            "question_root_attestations": {"sha256": digest(root)},
        },
    }
    (release / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (release / "manifest.json").write_text(
        json.dumps(
            {
                "status": ATTESTATION_RELEASE_STATUS,
                "run": {
                    "report": {"sha256": digest(release / "report.json")},
                    "training_started": False,
                },
            }
        ),
        encoding="utf-8",
    )
    actual_edge, actual_root, metadata = _validate_old_attestation_release(
        release, protected_ledger_binding=protected
    )
    assert (actual_edge, actual_root) == (edge, root)
    assert metadata == [release / "report.json", release / "manifest.json"]

    report["outputs"]["question_root_attestations"]["sha256"] = "0" * 64
    (release / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        _validate_old_attestation_release(
            release, protected_ledger_binding=protected
        )


def test_attestation_release_rejects_missing_complete_ledger_binding(tmp_path: Path):
    release = tmp_path / "attestation"
    release.mkdir()
    for name in ("question_edge_attestations.jsonl", "question_root_attestations.jsonl"):
        (release / name).write_text("{}\n", encoding="utf-8")
    (release / "report.json").write_text(
        json.dumps(
            {
                "schema_version": ATTESTATION_RELEASE_SCHEMA,
                "status": ATTESTATION_RELEASE_STATUS,
                "checks": {"legacy_interim": True},
                "counts": {},
                "outputs": {},
            }
        ),
        encoding="utf-8",
    )
    (release / "manifest.json").write_text(
        json.dumps({"status": ATTESTATION_RELEASE_STATUS}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="complete protected ledger"):
        _validate_old_attestation_release(
            release,
            protected_ledger_binding={
                name: {"sha256": "0" * 64}
                for name in ("ledger", "report", "manifest")
            },
        )


def _retrieval(cohort: dict) -> dict:
    passages = [
        {
            "id": f"doc-{index}",
            "source": "canonical",
            "contents": f"Canonical passage {index}.",
        }
        for index in range(10)
    ]
    return {
        "question_key": cohort["question_key"],
        "dataset": cohort["dataset"],
        "qid": cohort["qid"],
        "question": cohort["question"],
        "question_sha256": cohort["question_sha256"],
        "gold_access": False,
        "passages": passages,
        "passages_sha256": hashlib.sha256(
            json.dumps(
                passages, sort_keys=True, ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest(),
        "retrieval_source": CANONICAL_RETRIEVAL_STACK,
    }


def test_reserve_route_always_uses_exact_raw_and_canonical_retrieval():
    stale_source, _base, runtime, cohort = _rows(
        "reserve", "Who is Reserve's mother?", "comparison"
    )
    cohort.update(
        {
            "schema_version": (
                "automatic-proofkg-extension-reserve-question-only-v1"
            ),
            "source_role": "automatic_proofkg_extension_reserve_candidate",
            "gold_access": False,
        }
    )
    stale_source["answer"] = "WRONG CURRICULUM ANSWER"
    raw = {
        "id": cohort["qid"],
        "question": cohort["question"],
        "golden_answers": ["Carol"],
        "metadata": {"supporting_facts": ["must not copy"]},
    }
    retrieval = _retrieval(cohort)
    silver, records, gates, stats = build_supply(
        old_silver=[],
        old_records=[],
        old_runtime=[],
        # Even if a stale curriculum happens to contain this qid, the frozen
        # reserve role must not silently take its passages/outcome.
        new_source=[stale_source],
        new_cohort=[cohort],
        new_runtime=[runtime],
        new_raw=[raw],
        new_retrieval=[retrieval],
        blocked_qids=set(),
        blocked_hashes=set(),
        blocked_families=set(),
        cutoff="2020-12-09T23:59:59Z",
    )
    assert len(silver) == len(records) == len(gates) == 1
    assert silver[0]["answer"] == "Carol"
    assert silver[0]["retrieved_passages"] == retrieval["passages"]
    assert silver[0]["steps"] == []
    assert "supporting_facts" not in silver[0]
    assert stats["eligible_by_source"] == {
        "automatic_proofkg_extension_reserve_v1": 1
    }
    wrapper = build_candidate_wrappers(silver, records, gates)[0]
    assert wrapper["proof_passages_sha256"] == retrieval["passages_sha256"]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda row: row.update(question_sha256="0" * 64), "question_sha256"),
        (lambda row: row.update(retrieval_source="bm25-fallback"), "backend drift"),
        (lambda row: row.update(supporting_facts=[]), "forbidden fields"),
        (lambda row: row.update(passages_sha256="0" * 64), "ten-passage/hash"),
    ],
)
def test_raw_retrieval_fallback_rejects_drift_or_gold_fields(mutation, error):
    _source, _base, _runtime, cohort = _rows("reserve", "A reserve question?")
    cohort["gold_access"] = False
    raw = {
        "id": cohort["qid"],
        "question": cohort["question"],
        "golden_answers": ["Carol"],
    }
    retrieval = _retrieval(cohort)
    mutation(retrieval)
    with pytest.raises(ValueError, match=error):
        _fallback_source_from_raw_retrieval(
            cohort=cohort, raw=raw, retrieval=retrieval
        )


def test_v1b_role_never_silently_falls_back_when_curriculum_source_missing():
    _source, _base, runtime, cohort = _rows("v1b", "A v1b question?")
    cohort.update(
        {
            "schema_version": "automatic-proofkg-extension-question-only-v1",
            "source_role": "automatic_proofkg_extension_candidate",
            "gold_access": False,
        }
    )
    with pytest.raises(ValueError, match="v1b row is missing"):
        build_supply(
            old_silver=[], old_records=[], old_runtime=[], new_source=[],
            new_cohort=[cohort], new_runtime=[runtime], new_raw=[],
            new_retrieval=[], blocked_qids=set(), blocked_hashes=set(),
            blocked_families=set(), cutoff="2020-12-09T23:59:59Z",
        )
