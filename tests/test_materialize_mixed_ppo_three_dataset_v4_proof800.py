from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from kgproweight.kg.question_kg import make_question_kg_record, question_sha256
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.prepare.materialize_mixed_ppo_three_dataset_v4_proof800 import (
    DEFAULT_PROOF_SUPPLY,
    EXPECTED_EXPANSION_STATUS,
    EXPECTED_PARENT_EXPERIMENT,
    EXPECTED_PARENT_MANIFEST_PHASE,
    EXPECTED_PARENT_SCHEMA,
    EXPECTED_PROOF_SUPPLY_SCHEMA,
    EXPECTED_PROOF_SUPPLY_STATUS,
    EXPECTED_PROTOCOL_EXPERIMENT,
    EXPECTED_PROTOCOL_SCHEMA,
    EXPECTED_PROTOCOL_STATUS,
    EXPECTED_REPLAY_EXPERIMENT,
    EXPECTED_REPLAY_MANIFEST_PHASE,
    EXPECTED_REPLAY_SCHEMA,
    EXPECTED_REPLAY_SELECTION_SCHEMA,
    EXPECTED_RETRIEVAL_STACK,
    _canonical_sha256,
    _identity as file_identity,
    _load_ordinary_context_release,
    _load_protocol,
    _passages_sha256,
    _validate_expansion_requirement_join,
    _validate_expansion_release,
    _validate_hm_reconciled_contexts,
    _validate_parent_release,
    _validate_proof_supply_release,
    _validate_protocol_protected_ledger,
    _validate_replay_release,
    assemble_materialized_rows,
    identity_overlap_counts,
    validate_schedule_assets,
)
from scripts.prepare.audit_auto1500_v4_clean_reproducibility import (
    PROTECTED_LEDGER_SCHEMA_VERSION,
)
from scripts.prepare.freeze_mixed_ppo_three_dataset_v4_proof800 import (
    _protocol_manifest_extra,
)


def _identity(dataset: str, qid: str, *, eligible: bool, source_role: str) -> dict:
    question = f"What is the answer for {dataset} {qid}?"
    row = {
        "question_key": f"{dataset}::{qid}",
        "dataset": dataset,
        "qid": qid,
        "question": question,
        "question_sha256": question_sha256(question),
        "family_version": FAMILY_VERSION,
        "family_sha256": family_sha256(question),
        "route": "2wiki_proof_inference" if eligible else f"{dataset}_outcome",
        "source_role": source_role,
        "proof_source": "synthetic-proof" if eligible else "none",
        "question_type": "inference" if eligible else "unknown",
        "process_reward_eligible": eligible,
        "gold_access": False,
    }
    if eligible:
        row["proof_passages_sha256"] = _passages_sha256(_passages(qid))
    return row


def _passages(prefix: str) -> list[dict]:
    return [
        {"id": f"{prefix}-{index}", "source": "e5", "contents": f"Evidence {index}."}
        for index in range(10)
    ]


def _proof_record(identity: dict) -> dict:
    record = make_question_kg_record(
        dataset=identity["dataset"],
        qid=identity["qid"],
        question=identity["question"],
        triples=[["Alpha", "links to", "Beta"]],
        query_plan={
            "recognized": True,
            "hops": [
                {"subject": "Alpha", "pids": ["P1"], "output_slot": "hop_1"}
            ],
        },
        provenance={
            "builder_version": "synthetic-proof",
            "gold_access": False,
            "complete_plan_execution": True,
            "historical_cutoff": "2020-12-09T23:59:59Z",
        },
    )
    record["execution"] = {
        "complete_plan_execution": True,
        "hops": [
            {
                "hop_index": 1,
                "input_entities": [{"qid": "Q1"}],
                "matches": [["Alpha", "links to", "Beta"]],
            }
        ],
    }
    record["runtime_error"] = None
    record["process_reward_eligible"] = True
    identity["proof_record_sha256"] = _canonical_sha256(record)
    return record


def test_assembly_keeps_gold_outcome_only_and_recomputes_fail_closed_gate():
    hotpot = _identity("hotpotqa", "h1", eligible=False, source_role="retained_parent")
    musique = _identity("musique", "m1", eligible=False, source_role="new_retrieval")
    ordinary = _identity(
        "2wikimultihopqa", "o1", eligible=False, source_role="retained_parent"
    )
    proof = _identity(
        "2wikimultihopqa", "p1", eligible=True, source_role="selected_unified_proof"
    )
    record = _proof_record(proof)
    population = [hotpot, ordinary, proof, musique]
    raw = {
        row["question_key"]: {
            "id": row["qid"],
            "question": row["question"],
            "golden_answers": [f"gold-{row['qid']}", f"alias-{row['qid']}"],
            "metadata": {},
        }
        for row in population
    }
    parent = {
        row["question_key"]: {
            "dataset": row["dataset"],
            "qid": row["qid"],
            "question": row["question"],
            "retrieved_passages": _passages(row["qid"]),
        }
        for row in (hotpot, ordinary)
    }
    expansion = {
        musique["question_key"]: {
            **{key: musique[key] for key in ("dataset", "qid", "question", "question_sha256")},
            "passages": _passages("m1"),
        }
    }
    proof_silver = {
        proof["question_key"]: {
            "dataset": proof["dataset"],
            "qid": proof["qid"],
            "question": proof["question"],
            "kg_subgraph": copy.deepcopy(record["kg_subgraph"]),
            "retrieved_passages": _passages("p1"),
        }
    }
    silver, records, gates, sources = assemble_materialized_rows(
        population=population,
        raw_by_key=raw,
        parent_silver_by_key=parent,
        ordinary_context_by_key={},
        expansion_retrieval_by_key=expansion,
        proof_silver_by_key=proof_silver,
        proof_record_by_key={proof["question_key"]: record},
        cutoff="2020-12-09T23:59:59Z",
    )

    assert len(silver) == len(records) == len(gates) == 4
    assert [row["m_graph"] for row in gates] == [0, 0, 1, 0]
    assert all(row["steps"] == [] and len(row["retrieved_passages"]) == 10 for row in silver)
    assert all(row["metadata"]["gold_use"] == "outcome_reward_label_only" for row in silver)
    assert all(
        row["metadata"]["failed_qpeg_or_saeg_p_edges_included"] is False
        for row in silver
    )
    assert all(row["answer"].startswith("gold-") for row in silver)
    assert all("gold_answer" not in record for record in records)
    assert records[2]["provenance"]["historical_cutoff"] == "2020-12-09T23:59:59Z"
    assert sources == {
        "canonical_expansion_wiki18_rrf_reranked_top10": 1,
        "retained_v2_frozen_context": 2,
        "unified_2wiki_strict_proof_supply": 1,
    }


@pytest.mark.parametrize("failure", ["record_hash", "cutoff", "trace"])
def test_selected_proof_fails_closed_on_supply_drift(failure: str):
    proof = _identity(
        "2wikimultihopqa", "p1", eligible=True, source_role="selected_unified_proof"
    )
    record = _proof_record(proof)
    if failure == "record_hash":
        proof["proof_record_sha256"] = "0" * 64
    elif failure == "cutoff":
        record["provenance"]["historical_cutoff"] = "2021-01-01T00:00:00Z"
        proof["proof_record_sha256"] = _canonical_sha256(record)
    else:
        record["execution"]["hops"][0]["matches"] = []
        proof["proof_record_sha256"] = _canonical_sha256(record)
    raw = {
        proof["question_key"]: {
            "id": proof["qid"], "question": proof["question"], "golden_answers": ["x"]
        }
    }
    source = {
        proof["question_key"]: {
            "dataset": proof["dataset"], "qid": proof["qid"],
            "question": proof["question"], "kg_subgraph": record["kg_subgraph"],
            "retrieved_passages": _passages("p"),
        }
    }
    with pytest.raises(ValueError):
        assemble_materialized_rows(
            population=[proof], raw_by_key=raw, parent_silver_by_key={},
            ordinary_context_by_key={},
            expansion_retrieval_by_key={}, proof_silver_by_key=source,
            proof_record_by_key={proof["question_key"]: record},
            cutoff="2020-12-09T23:59:59Z",
        )


def test_replay_overlap_uses_dataset_scoped_qid_exact_hash_and_current_family():
    left = [_identity("hotpotqa", "h1", eligible=False, source_role="retained_parent")]
    exact = copy.deepcopy(left[0])
    same_family = copy.deepcopy(left[0])
    same_family["qid"] = "h2"
    same_family["question_key"] = "hotpotqa::h2"
    cross_dataset = copy.deepcopy(left[0])
    cross_dataset["dataset"] = "musique"
    cross_dataset["question_key"] = "musique::h1"
    assert identity_overlap_counts(left, [exact]) == {
        "qid": 1, "question_sha256": 1, "family_sha256": 1
    }
    assert identity_overlap_counts(left, [same_family]) == {
        "qid": 0, "question_sha256": 1, "family_sha256": 1
    }
    assert identity_overlap_counts(left, [cross_dataset]) == {
        "qid": 0, "question_sha256": 0, "family_sha256": 0
    }


def test_schedule_validation_checks_k4_identity_and_weight_join():
    population = [
        _identity("hotpotqa", "h1", eligible=False, source_role="retained_parent"),
        _identity("musique", "m1", eligible=False, source_role="retained_parent"),
    ]
    groups = [{**row, "prompt_group_index": index} for index, row in enumerate(population, 1)]
    weights = [
        {
            "dataset": row["dataset"], "qid": row["qid"],
            "question_sha256": row["question_sha256"],
            "process_reward_eligible": row["process_reward_eligible"],
            "sampling_probability": 0.5,
        }
        for row in population
    ]
    schedule = []
    for group_index, row in enumerate(groups, 1):
        schedule.extend(
            {
                "prompt_group_index": group_index,
                "within_group_rollout": within,
                "dataset": row["dataset"], "qid": row["qid"],
                "question_sha256": row["question_sha256"],
                "process_reward_eligible": row["process_reward_eligible"],
            }
            for within in range(1, 5)
        )
    assert all(validate_schedule_assets(population, weights, groups, schedule).values())
    schedule[4]["qid"] = "drift"
    assert validate_schedule_assets(population, weights, groups, schedule)[
        "schedule_k4_identity_exact"
    ] is False


def test_expansion_release_requires_no_fallback_attestation(tmp_path: Path):
    contexts = tmp_path / "retrieval_contexts.jsonl"
    contexts.write_text("{}\n", encoding="utf-8")
    report = {
        "status": EXPECTED_EXPANSION_STATUS,
        "retrieval": EXPECTED_RETRIEVAL_STACK,
        "gates": {"all_ok": True},
        "backend_attestation": {
            "mode": "cross_encoder",
            "requested_backend": "bge-reranker-v2-m3",
            "load_succeeded": True,
            "backend_fallback": False,
            "config": {"sha256": "a"},
            "weights": {"sha256": "b"},
            "tokenizer": {"sha256": "c"},
        },
        "outputs": {
            "combined": {"sha256": hashlib.sha256(contexts.read_bytes()).hexdigest()}
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest = {
        "status": EXPECTED_EXPANSION_STATUS,
        "run": {
            "outputs": {
                "combined": {
                    "sha256": hashlib.sha256(contexts.read_bytes()).hexdigest()
                },
                "report": {
                    "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()
                },
            },
            "training_started": False,
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert set(_validate_expansion_release(tmp_path, contexts)) == {"report.json", "manifest.json"}
    reconciliation_protocol = tmp_path / "hm_reconciliation_protocol.json"
    reconciliation_protocol.write_text("{}", encoding="utf-8")
    report["inputs"] = {
        "hm_reconciliation_protocol": file_identity(reconciliation_protocol)
    }
    report["reconciliation"] = {
        "reused_contexts": 812,
        "newly_retrieved_contexts": 11,
        "retired_contexts": 6,
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest["run"]["outputs"]["report"]["sha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert set(
        _validate_expansion_release(
            tmp_path,
            contexts,
            reconciliation_protocol_path=reconciliation_protocol,
        )
    ) == {"report.json", "manifest.json"}
    report["reconciliation"]["newly_retrieved_contexts"] = 10
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="812/11/6"):
        _validate_expansion_release(
            tmp_path,
            contexts,
            reconciliation_protocol_path=reconciliation_protocol,
        )
    report["reconciliation"]["newly_retrieved_contexts"] = 11
    report["backend_attestation"]["backend_fallback"] = True
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="fallback"):
        _validate_expansion_release(tmp_path, contexts)


def test_expansion_join_requires_all_823_reconciled_contexts_exactly():
    population = []
    requests = []
    expansion = {}
    for dataset, count in (("hotpotqa", 417), ("musique", 406)):
        for index in range(count):
            row = _identity(
                dataset, f"{dataset}-{index}", eligible=False, source_role="new_retrieval"
            )
            population.append(row)
            request = {
                key: row[key]
                for key in (
                    "question_key",
                    "dataset",
                    "qid",
                    "question",
                    "question_sha256",
                    "family_sha256",
                    "gold_access",
                )
            }
            requests.append(request)
            expansion[row["question_key"]] = {
                "dataset": row["dataset"],
                "qid": row["qid"],
                "question": row["question"],
                "question_sha256": row["question_sha256"],
            }
    indexed, counts = _validate_expansion_requirement_join(
        population, requests, expansion
    )
    assert len(indexed) == 823
    assert counts == {"hotpotqa": 417, "musique": 406}
    expansion.pop("musique::musique-405")
    with pytest.raises(ValueError, match="exactly cover all"):
        _validate_expansion_requirement_join(population, requests, expansion)


def test_reused_hm_context_must_match_frozen_passage_hash():
    row = _identity(
        "hotpotqa", "reuse-1", eligible=False, source_role="new_retrieval"
    )
    passages = _passages("reuse")
    context = {
        "dataset": row["dataset"],
        "qid": row["qid"],
        "question": row["question"],
        "passages": passages,
        "passages_sha256": _passages_sha256(passages),
    }
    key = row["question_key"]
    _validate_hm_reconciled_contexts(
        {key: context},
        requirements={key: row},
        reused={key: {**row, "passages_sha256": _passages_sha256(passages)}},
        new={},
    )
    drifted = copy.deepcopy(context)
    drifted["passages"][0]["contents"] = "silently drifted"
    with pytest.raises(ValueError, match="passage hash mismatch"):
        _validate_hm_reconciled_contexts(
            {key: drifted},
            requirements={key: row},
            reused={key: {**row, "passages_sha256": _passages_sha256(passages)}},
            new={},
        )


def test_ordinary200_source_release_joins_replacements_by_frozen_line_and_hash(
    tmp_path: Path,
):
    parent_path = tmp_path / "parent.jsonl"
    replacement_path = tmp_path / "replacement.jsonl"
    ordinary_path = tmp_path / "ordinary200.identity_provenance.jsonl"
    protocol_path = tmp_path / "protocol.json"
    manifest_path = tmp_path / "manifest.json"
    population = []
    parent_source = []
    replacement_source = []
    frozen = []
    for index in range(200):
        role = "replacement_ordinary" if index == 199 else "retained_parent_ordinary"
        origin = (
            "proofkg_curriculum_mix_v1"
            if role == "replacement_ordinary"
            else "mixed_ppo_v2_materialized"
        )
        identity = _identity(
            "2wikimultihopqa", f"ordinary-{index}", eligible=False, source_role=role
        )
        identity["route"] = "2wiki_ordinary_outcome"
        identity["source_origin"] = origin
        identity["source_line_number"] = 1 if index == 199 else index + 1
        source = {
            "dataset": identity["dataset"],
            "qid": identity["qid"],
            "question": identity["question"],
            "accepted": True,
            "answer": "outcome-must-not-be-copied",
            "steps": [{"forbidden": "source trace"}],
            "kg_subgraph": [["must", "not", "copy"]],
            "retrieved_passages": _passages(identity["qid"]),
        }
        identity["source_record_sha256"] = _canonical_sha256(source)
        identity["source_passages_sha256"] = _canonical_sha256(
            source["retrieved_passages"]
        )
        population.append(copy.deepcopy(identity))
        frozen.append(
            {
                key: identity[key]
                for key in (
                    "dataset",
                    "qid",
                    "question",
                    "question_sha256",
                    "family_version",
                    "family_sha256",
                    "question_type",
                    "route",
                    "source_role",
                    "source_origin",
                    "source_line_number",
                    "source_record_sha256",
                    "source_passages_sha256",
                    "process_reward_eligible",
                    "gold_access",
                )
            }
        )
        (replacement_source if index == 199 else parent_source).append(source)
    parent_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in parent_source),
        encoding="utf-8",
    )
    replacement_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in replacement_source),
        encoding="utf-8",
    )
    ordinary_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in frozen),
        encoding="utf-8",
    )
    ordinary_protocol = {
        "schema_version": "2wiki-ordinary200-full-ledger-protocol-v2",
        "status": "FROZEN_ANSWER_FREE_OUTCOME_SOURCE_BOUND_NOT_TRAINED",
        "selection": {"target_n": 200, "counts": {"gates": {"all": True}}},
        "inputs": {
            "parent_materialized_outcome_passages": file_identity(parent_path),
            "replacement_curriculum_outcome_passages": file_identity(replacement_path),
        },
        "outputs": {"ordinary200": file_identity(ordinary_path)},
    }
    protocol_path.write_text(json.dumps(ordinary_protocol), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "status": "FROZEN_ANSWER_FREE_OUTCOME_SOURCE_BOUND_NOT_TRAINED",
                "run": {
                    "protocol_sha256": hashlib.sha256(
                        protocol_path.read_bytes()
                    ).hexdigest(),
                    "ordinary200": file_identity(ordinary_path),
                    "training_started": False,
                },
            }
        ),
        encoding="utf-8",
    )
    main_protocol = {
        "inputs": {"ordinary200_successor_protocol": file_identity(protocol_path)}
    }
    contexts, release_files, counts = _load_ordinary_context_release(
        main_protocol,
        population=population,
        parent_silver_path=parent_path,
    )
    assert len(contexts) == 200
    assert counts == {
        "mixed_ppo_v2_materialized": 199,
        "proofkg_curriculum_mix_v1": 1,
    }
    replacement = contexts["2wikimultihopqa::ordinary-199"]
    assert replacement["source_role"] == "replacement_ordinary"
    assert "steps" not in replacement and "kg_subgraph" not in replacement
    assert set(release_files) == {
        "protocol",
        "manifest",
        "identities",
        "parent_materialized_outcome_passages",
        "replacement_curriculum_outcome_passages",
    }


def test_ordinary200_source_release_fails_on_population_provenance_drift(
    tmp_path: Path,
):
    # Exercise the earliest fail-closed boundary without constructing source files.
    missing = tmp_path / "missing-protocol.json"
    main_protocol = {
        "inputs": {
            "ordinary200_successor_protocol": {
                "path": str(missing),
                "sha256": "0" * 64,
            }
        }
    }
    with pytest.raises(FileNotFoundError, match="ordinary200 successor"):
        _load_ordinary_context_release(
            main_protocol, population=[], parent_silver_path=tmp_path / "parent"
        )


def test_protocol_requires_exact_live_protected_ledger_hashes():
    live = {
        name: {"sha256": hashlib.sha256(name.encode()).hexdigest()}
        for name in ("ledger", "report", "manifest")
    }
    protocol = {
        "protected_ledger": {
            "version": PROTECTED_LEDGER_SCHEMA_VERSION,
            "complete": True,
            "current_family_recomputed": True,
            **copy.deepcopy(live),
        }
    }
    _validate_protocol_protected_ledger(protocol, live_binding=live)
    protocol["protected_ledger"]["ledger"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        _validate_protocol_protected_ledger(protocol, live_binding=live)


def test_proof_supply_release_binds_outputs_manifest_and_complete_ledger(tmp_path: Path):
    silver = tmp_path / "silver_train.jsonl"
    records = tmp_path / "question_kg_records.jsonl"
    gates = tmp_path / "source_gate_records.jsonl"
    candidates = tmp_path / "proof_candidates.jsonl"
    silver.write_text("{}\n", encoding="utf-8")
    records.write_text("{}\n", encoding="utf-8")
    gates.write_text("{}\n", encoding="utf-8")
    candidates.write_text("{}\n", encoding="utf-8")
    proof_files = {
        "silver_train": silver,
        "question_kg_records": records,
        "source_gate_records": gates,
        "proof_candidates": candidates,
    }
    protected_dir = tmp_path / "protected"
    protected_dir.mkdir()
    protected_paths = {
        "ledger": protected_dir / "protected_identities.question_only.jsonl",
        "report": protected_dir / "report.json",
        "manifest": protected_dir / "manifest.json",
    }
    for name, path in protected_paths.items():
        path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
    protected = {name: file_identity(path) for name, path in protected_paths.items()}
    report = {
        "schema_version": EXPECTED_PROOF_SUPPLY_SCHEMA,
        "experiment_id": "TEST-OFFICIAL-V3",
        "status": EXPECTED_PROOF_SUPPLY_STATUS,
        "training_started": False,
        "checks": {"strict": True},
        "protected_ledger": {
            "version": PROTECTED_LEDGER_SCHEMA_VERSION,
            "complete": True,
            "current_family_recomputed": True,
            **copy.deepcopy(protected),
        },
        "outputs": {name: file_identity(path) for name, path in proof_files.items()},
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest = {
        "status": EXPECTED_PROOF_SUPPLY_STATUS,
        "run": {
            "phase": "unified_2wiki_proofkg_official_raw_v3_candidate_supply",
            "experiment_id": "TEST-OFFICIAL-V3",
            "report": file_identity(report_path),
            "training_started": False,
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    metadata = _validate_proof_supply_release(
        tmp_path,
        proof_files=proof_files,
        protected_ledger_binding=protected,
    )
    assert set(metadata) == {"report.json", "manifest.json"}

    report["protected_ledger"]["complete"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="complete protected ledger"):
        _validate_proof_supply_release(
            tmp_path,
            proof_files=proof_files,
            protected_ledger_binding=protected,
        )

    report["protected_ledger"]["complete"] = True
    report["checks"] = {}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest["run"]["report"] = file_identity(report_path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="status/schema/checks"):
        _validate_proof_supply_release(
            tmp_path,
            proof_files=proof_files,
            protected_ledger_binding=protected,
        )

    report["checks"] = {"strict": True}
    report["outputs"]["silver_train"]["sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest["run"]["report"] = file_identity(report_path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch|output hash mismatch"):
        _validate_proof_supply_release(
            tmp_path,
            proof_files=proof_files,
            protected_ledger_binding=protected,
        )

    report["outputs"]["silver_train"] = file_identity(silver)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest["status"] = "FAILED"
    manifest["run"]["report"] = file_identity(report_path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="status/schema/checks"):
        _validate_proof_supply_release(
            tmp_path,
            proof_files=proof_files,
            protected_ledger_binding=protected,
        )


def test_final_v4_accepts_only_official_raw_unified_v3_contract(tmp_path: Path):
    assert EXPECTED_PROOF_SUPPLY_SCHEMA == (
        "2wiki-unified-proofkg-official-raw-candidate-supply-v3"
    )
    assert EXPECTED_PROOF_SUPPLY_STATUS == (
        "COMPLETE_STRICT_OFFICIAL_RAW_CANDIDATE_SUPPLY_NOT_SELECTED_NOT_TRAINED"
    )
    assert DEFAULT_PROOF_SUPPLY == Path(
        "data/derived/2wiki_unified_proofkg_official_raw_v3"
    )

    silver = tmp_path / "silver_train.jsonl"
    records = tmp_path / "question_kg_records.jsonl"
    gates = tmp_path / "source_gate_records.jsonl"
    candidates = tmp_path / "proof_candidates.jsonl"
    silver.write_text("{}\n", encoding="utf-8")
    records.write_text("{}\n", encoding="utf-8")
    gates.write_text("{}\n", encoding="utf-8")
    candidates.write_text("{}\n", encoding="utf-8")
    proof_files = {
        "silver_train": silver,
        "question_kg_records": records,
        "source_gate_records": gates,
        "proof_candidates": candidates,
    }
    protected = {
        name: {"sha256": hashlib.sha256(name.encode()).hexdigest()}
        for name in ("ledger", "report", "manifest")
    }
    report = {
        # A legacy-v2 release must fail closed even if every other field and
        # file hash is internally consistent.
        "schema_version": "2wiki-unified-proofkg-candidate-supply-v2",
        "experiment_id": "TEST-LEGACY-V2",
        "status": "COMPLETE_STRICT_CANDIDATE_SUPPLY_NOT_SELECTED_NOT_TRAINED",
        "training_started": False,
        "checks": {"strict": True},
        "protected_ledger": {
            "version": PROTECTED_LEDGER_SCHEMA_VERSION,
            "complete": True,
            "current_family_recomputed": True,
            **copy.deepcopy(protected),
        },
        "outputs": {
            name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in proof_files.items()
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": report["status"],
                "run": {
                    "phase": "unified_2wiki_proofkg_candidate_supply",
                    "experiment_id": "TEST-LEGACY-V2",
                    "report": {
                        "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()
                    },
                    "training_started": False,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="status/schema/checks"):
        _validate_proof_supply_release(
            tmp_path,
            proof_files=proof_files,
            protected_ledger_binding=protected,
        )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_v4_protocol_requires_nonempty_gates_and_bound_manifest(tmp_path: Path):
    output_names = (
        "population",
        "sampling_weights",
        "prompt_groups",
        "fixed_rollout_schedule",
        "retrieval_requests",
    )
    outputs = {}
    for name in output_names:
        path = tmp_path / f"{name}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        outputs[name] = file_identity(path)
    protocol_path = tmp_path / "protocol.json"
    protocol = {
        "schema_version": EXPECTED_PROTOCOL_SCHEMA,
        "status": EXPECTED_PROTOCOL_STATUS,
        "experiment_id": EXPECTED_PROTOCOL_EXPERIMENT,
        "population": {
            "unique_total": 3000,
            "unique_by_dataset": {
                "2wikimultihopqa": 1000,
                "hotpotqa": 1000,
                "musique": 1000,
            },
        },
        "schedule": {
            "prompt_groups": 3000,
            "rollouts_per_prompt": 4,
            "trajectories": 12000,
            "proof_groups": 800,
        },
        "gates": {"all": True},
        "outputs": outputs,
    }
    _write_json(protocol_path, protocol)
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "status": EXPECTED_PROTOCOL_STATUS,
        "run": _protocol_manifest_extra(protocol_path),
    }
    _write_json(manifest_path, manifest)
    loaded, paths, loaded_manifest = _load_protocol(protocol_path)
    assert loaded == protocol
    assert set(paths) == set(output_names)
    assert loaded_manifest == manifest_path.resolve()

    protocol["gates"] = {"all": False}
    _write_json(protocol_path, protocol)
    manifest["run"]["protocol_sha256"] = file_identity(protocol_path)["sha256"]
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="failed frozen gates"):
        _load_protocol(protocol_path)

    protocol["gates"] = {"all": True}
    _write_json(protocol_path, protocol)
    manifest["run"]["protocol_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="manifest/status/hash"):
        _load_protocol(protocol_path)


def _make_parent_release(tmp_path: Path) -> tuple[Path, dict]:
    directory = tmp_path / "parent"
    directory.mkdir()
    parent_protocol = tmp_path / "parent_protocol.json"
    _write_json(parent_protocol, {"frozen": True})
    silver = directory / "silver_train.jsonl"
    silver.write_text(
        "".join(
            json.dumps(
                {
                    "dataset": "hotpotqa",
                    "qid": f"parent-{index}",
                    "question": f"Parent question {index}?",
                    "retrieved_passages": _passages(f"parent-{index}"),
                },
                sort_keys=True,
            )
            + "\n"
            for index in range(1799)
        ),
        encoding="utf-8",
    )
    report = {
        "schema_version": EXPECTED_PARENT_SCHEMA,
        "experiment_id": EXPECTED_PARENT_EXPERIMENT,
        "status": "COMPLETE_DATA_NOT_TRAINED",
        "counts": {"unique_population": 1799},
        "gates": {"all": True},
        "inputs": {"protocol": file_identity(parent_protocol)},
        "outputs": {"silver_train": file_identity(silver)},
    }
    report_path = directory / "report.json"
    _write_json(report_path, report)
    _write_json(
        directory / "manifest.json",
        {
            "status": "COMPLETE_DATA_NOT_TRAINED",
            "run": {
                "phase": EXPECTED_PARENT_MANIFEST_PHASE,
                "experiment_id": EXPECTED_PARENT_EXPERIMENT,
                "report_sha256": file_identity(report_path)["sha256"],
            },
        },
    )
    return directory, {"inputs": {"parent_protocol": file_identity(parent_protocol)}}


@pytest.mark.parametrize("failure", ["silver", "manifest", "parent_protocol"])
def test_parent_release_chain_fails_closed(tmp_path: Path, failure: str):
    directory, final_protocol = _make_parent_release(tmp_path)
    files, rows = _validate_parent_release(
        directory, final_protocol=final_protocol
    )
    assert len(rows) == 1799 and set(files) == {
        "silver_train.jsonl",
        "report.json",
        "manifest.json",
    }
    if failure == "silver":
        with (directory / "silver_train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
    elif failure == "manifest":
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["run"]["report_sha256"] = "0" * 64
        _write_json(directory / "manifest.json", manifest)
    else:
        final_protocol["inputs"]["parent_protocol"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        _validate_parent_release(directory, final_protocol=final_protocol)


def _make_replay_release(
    tmp_path: Path,
) -> tuple[Path, dict[str, dict]]:
    directory = tmp_path / "replay"
    protected_dir = tmp_path / "protected"
    directory.mkdir()
    protected_dir.mkdir()
    protected = {}
    for name in ("ledger", "report", "manifest"):
        path = protected_dir / name
        path.write_text(f"{name}\n", encoding="utf-8")
        protected[name] = file_identity(path)
    silver_path = directory / "silver_train.jsonl"
    selection_path = directory / "selection_records.jsonl"
    silver_lines = []
    selections = []
    for index in range(2000):
        value = index
        letters = []
        while True:
            letters.append(chr(ord("a") + value % 26))
            value = value // 26 - 1
            if value < 0:
                break
        unique_word = "".join(reversed(letters))
        question = f"Which replay entity named token{unique_word} is selected?"
        row = {
            "dataset": "hotpotqa",
            "qid": f"replay-{index}",
            "question": question,
            "answer": f"answer-{index}",
            "accepted": True,
            "steps": [
                {
                    "index": step,
                    "text": (
                        f"Reasoning: reason {step}\nKnowledge Used: []\n"
                        f"Conclusion: conclusion {step}"
                    ),
                    "label": 1.0,
                    "cited_triples": [],
                }
                for step in range(1, 4)
            ],
            "kg_subgraph": [],
            "retrieved_passages": [],
            "metadata": {"gold_answer": f"answer-{index}"},
        }
        raw = json.dumps(row, sort_keys=True).encode("utf-8")
        silver_lines.append(raw.decode("utf-8") + "\n")
        selections.append(
            {
                "schema_version": EXPECTED_REPLAY_SELECTION_SCHEMA,
                "dataset": "hotpotqa",
                "qid": row["qid"],
                "question": question,
                "question_sha256": question_sha256(question),
                "family_version": FAMILY_VERSION,
                "family_sha256": family_sha256(question),
                "rendered_steps": 3,
                "source_row_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    silver_path.write_text("".join(silver_lines), encoding="utf-8")
    selection_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selections),
        encoding="utf-8",
    )
    report = {
        "schema_version": EXPECTED_REPLAY_SCHEMA,
        "experiment_id": EXPECTED_REPLAY_EXPERIMENT,
        "status": "COMPLETE_DATA_NOT_TRAINED",
        "selection": {"n_samples": 2000, "dataset_counts": {"hotpotqa": 2000}},
        "protected_ledger": {
            "protected_identities.question_only.jsonl": protected["ledger"],
            "report.json": protected["report"],
            "manifest.json": protected["manifest"],
        },
        "outputs": {
            "silver_train": file_identity(silver_path),
            "selection_records": file_identity(selection_path),
        },
        "gates": {"all": True},
    }
    report_path = directory / "report.json"
    _write_json(report_path, report)
    _write_json(
        directory / "manifest.json",
        {
            "status": "COMPLETE_DATA_NOT_TRAINED",
            "run": {
                "phase": EXPECTED_REPLAY_MANIFEST_PHASE,
                "experiment_id": EXPECTED_REPLAY_EXPERIMENT,
                "report_sha256": file_identity(report_path)["sha256"],
                "silver_train_sha256": file_identity(silver_path)["sha256"],
                "selection_records_sha256": file_identity(selection_path)["sha256"],
                "protected_ledger_sha256": protected["ledger"]["sha256"],
                "training_started": False,
            },
        },
    )
    return directory, protected


@pytest.mark.parametrize("failure", ["silver", "selection", "report_gate", "manifest"])
def test_replay_release_chain_fails_closed(tmp_path: Path, failure: str):
    directory, protected = _make_replay_release(tmp_path)
    files, rows = _validate_replay_release(
        directory, protected_ledger_binding=protected
    )
    assert len(rows) == 2000 and len(files) == 4
    if failure == "silver":
        with (directory / "silver_train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
    elif failure == "selection":
        with (directory / "selection_records.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
    elif failure == "report_gate":
        report = json.loads((directory / "report.json").read_text())
        report["gates"]["all"] = False
        _write_json(directory / "report.json", report)
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["run"]["report_sha256"] = file_identity(
            directory / "report.json"
        )["sha256"]
        _write_json(directory / "manifest.json", manifest)
    else:
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["run"]["report_sha256"] = "0" * 64
        _write_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError):
        _validate_replay_release(directory, protected_ledger_binding=protected)
