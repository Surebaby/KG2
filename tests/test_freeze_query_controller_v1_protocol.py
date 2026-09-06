"""Tests for strict-eligible Controller-v1 identity freezing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kgproweight.eval.query_controller_v1 import SCHEMA_VERSION, STATE_VERSION
from kgproweight.kg.question_kg import question_sha256
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256
from scripts.prepare.freeze_query_controller_v1_protocol import (
    IDENTITY_FIELDS,
    _load_action_candidates,
    _protocol_body,
    run_freeze,
)
from scripts.prepare.build_query_controller_action_supervision_v1 import (
    FORMAL_FUTURE_GOLD_FREE_MECHANISM_GATE,
    FORMAL_OUTCOME_UNLOCK_RULE,
    FORMAL_PROBE_EVALUATION_CONTRACT,
    FORMAL_RUNTIME_CONFIG,
    FORMAL_TRAINING_PROBE_GATES,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _action_pair(dataset: str, qid: str, question: str, split: str) -> list[dict]:
    bridge = f"bridge-{qid}"
    q1_query = f"{qid} associated organization"
    excerpt = f"The associated organization is {bridge}."
    identity = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "qid": qid,
        "question_key": f"{dataset}::{qid}",
        "question_sha256": question_sha256(question),
        "family_sha256": family_sha256(question),
        "split": split,
    }
    q1 = {
        **identity,
        "example_id": f"{dataset}::{qid}::q1",
        "slot": "q1",
        "turn_index": 1,
        "state": {
            "state_version": STATE_VERSION,
            "original_question": question,
            "previous_actions": [],
            "verified_observations": [],
        },
        "target": {
            "action": "retrieve",
            "query": q1_query,
            "anchor": qid,
            "relation_intent": "associated organization",
            "pid": None,
            "dependencies": [],
            "output_slot": "q1",
            "source_action": "text",
        },
        "source_provenance": {"candidate": True},
        "gold_boundary": {
            "train_intermediate_annotation_used": False,
            "gold_final_answer_visible": False,
            "evaluation_gold_access": False,
        },
    }
    q2 = {
        **identity,
        "example_id": f"{dataset}::{qid}::q2_dynamic",
        "slot": "q2_dynamic",
        "turn_index": 2,
        "state": {
            "state_version": STATE_VERSION,
            "original_question": question,
            "previous_actions": [
                {
                    "slot": "q1",
                    "action": "retrieve",
                    "query": q1_query,
                    "output_slot": "q1",
                }
            ],
            "verified_observations": [
                {
                    "answer": bridge,
                    "answer_sha256": _sha(bridge),
                    "evidence_excerpt": excerpt,
                    "evidence_excerpt_sha256": _sha(excerpt),
                    "document_id": f"doc-{qid}",
                    "document_title": f"title-{qid}",
                    "sentence_index": 0,
                    "provenance": {
                        "source": "train_annotation_support",
                        "annotation_path": "metadata.evidences.entity[0]",
                        "binding_method": "fact_title_and_answer_surface",
                    },
                }
            ],
        },
        "target": {
            "action": "retrieve",
            "query": f"{bridge} location",
            "anchor": bridge,
            "relation_intent": "location",
            "pid": None,
            "dependencies": ["q1"],
            "output_slot": "q2",
            "source_action": "text",
        },
        "source_provenance": {
            "candidate": True,
            "train_intermediate_annotation_used": True,
        },
        "gold_boundary": {
            "train_intermediate_annotation_used": True,
            "gold_final_answer_visible": False,
            "evaluation_gold_access": False,
        },
    }
    return [q1, q2]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _synthetic_inputs(root: Path) -> tuple[dict[str, Path], tuple[Path, ...], Path, Path, Path]:
    candidates = root / "candidates"
    questions = [
        "Who founded lowercasealpha academy?",
        "Where was lowercasebeta engine created?",
        "When did lowercasegamma factory close?",
        "Which country contains lowercasedelta village?",
    ]
    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    for dataset in ("2wikimultihopqa", "musique"):
        for index, question in enumerate(questions[:3]):
            train_rows.extend(_action_pair(dataset, f"train-{dataset}-{index}", question, "train"))
        dev_rows.extend(_action_pair(dataset, f"dev-{dataset}-0", questions[3], "dev"))
    _write_jsonl(candidates / "train.jsonl", train_rows)
    _write_jsonl(candidates / "dev.jsonl", dev_rows)

    consumed = root / "consumed.jsonl"
    _write_jsonl(
        consumed,
        [{"dataset": "hotpotqa", "qid": "old", "question": "Why did olditem vanish?"}],
    )

    phase0_source = root / "phase0.identity.jsonl"
    phase0_tasks = root / "phase0.tasks.jsonl"
    identities: list[dict] = []
    tasks: list[dict] = []
    for index in range(37):
        dataset = "hotpotqa" if index < 17 else "musique"
        qid = f"phase0-{index}"
        question = f"How is lowercasephaseword{chr(97 + index % 26)} item number {index} described?"
        identity = {
            "dataset": dataset,
            "qid": qid,
            "question": question,
            "question_sha256": question_sha256(question),
            "family_sha256": family_sha256(question),
        }
        identities.append(identity)
        tasks.append(
            {
                "dataset": dataset,
                "qid": qid,
                "question_sha256": identity["question_sha256"],
            }
        )
    tasks.extend(tasks[:4])
    _write_jsonl(phase0_source, identities)
    _write_jsonl(phase0_tasks, tasks)

    sealed_parent = root / "sealed-parent"
    _write_json(
        sealed_parent / "report.json",
        {
            "status": "COMPLETE_FROZEN_SCOPE_A_DEV30_PROSPECTIVE300_NO_RESERVE",
            "checks": {"raw_train_qid_overlap": 0, "raw_train_family_overlap": 0},
            "prospective_seal": {"status": "FROZEN_UNOPENED_FOR_METHOD_DEVELOPMENT"},
        },
    )
    _write_json(
        sealed_parent / "manifest.json",
        {"outputs": [{"path": "prospective.identity_only.jsonl", "sha256": "declared-seal"}]},
    )
    # Deliberately do not create prospective.identity_only.jsonl. A successful
    # freeze proves the content file was not required, opened, or hashed.
    return (
        {"train": candidates / "train.jsonl", "dev": candidates / "dev.jsonl"},
        (consumed.relative_to(root),),
        phase0_tasks.relative_to(root),
        phase0_source.relative_to(root),
        sealed_parent.relative_to(root),
    )


def test_freezer_selects_only_valid_pairs_and_preserves_isolation(tmp_path: Path) -> None:
    paths, consumed, phase0_tasks, phase0_source, parent = _synthetic_inputs(tmp_path)
    output = tmp_path / "frozen"
    result = run_freeze(
        project_root=tmp_path,
        candidate_paths=paths,
        expected_candidate_hashes=None,
        output_dir=output,
        split_sizes={"train": 2, "dev": 1, "confirmation": 1},
        consumed_identity_paths=consumed,
        expected_consumed_hashes=None,
        phase0_task_path=phase0_tasks,
        phase0_identity_source_path=phase0_source,
        expected_phase0_hashes=None,
        sealed_parent_dir=parent,
        expected_parent_hashes=None,
        expected_sealed_declared_sha256="declared-seal",
        require_v4_3_failure_lineage=False,
    )
    rows_by_role = {
        role: [json.loads(line) for line in (output / f"{role}.identity_only.jsonl").read_text().splitlines()]
        for role in ("train", "dev", "confirmation")
    }
    assert {role: len(rows) for role, rows in rows_by_role.items()} == {
        "train": 4,
        "dev": 2,
        "confirmation": 2,
    }
    assert all(tuple(row) == IDENTITY_FIELDS for rows in rows_by_role.values() for row in rows)
    assert all(
        len(row["action_pair_sha256"]) == 64
        for rows in rows_by_role.values()
        for row in rows
    )
    qids = [
        (row["dataset"], row["qid"])
        for rows in rows_by_role.values()
        for row in rows
    ]
    families = [
        (row["dataset"], row["family_sha256"])
        for rows in rows_by_role.values()
        for row in rows
    ]
    assert len(qids) == len(set(qids))
    assert len(families) == len(set(families))
    protocol = result["protocol"]
    assert protocol["cohort"]["source_split_for_role"]["confirmation"] == "train"
    assert protocol["dataset_readiness"]["hotpotqa"]["status"].endswith("UNKNOWN")
    assert "annotation-derived_but_passage-bound" in protocol["gold_boundary"][
        "training_q2_model_visible_intermediate"
    ]
    assert protocol["gold_boundary"]["runtime_q2_model_visible_intermediate"].startswith(
        "Reader-predicted"
    )
    assert protocol["gold_boundary"]["freezer_direct_gold_final_answer_accessed"] is False
    assert protocol["gold_boundary"]["candidate_eligibility_is_gold_screened"] is True
    assert (
        protocol["gold_boundary"]["upstream_candidate_builder_gold_final_answer_use"]
        == "leakage_exclusion_only"
    )
    assert result["report"]["checks"]["phase0_unique_qids_registered"] == 37
    assert protocol["action_contract"][
        "identity_lock_binds_canonical_q1_q2_action_pair_sha256"
    ] is True
    assert set(protocol["implementation_locks"]) == {
        "central_action_validator",
        "protocol_freezer",
        "action_builder",
        "controller_trainer",
        "controller_train_cli",
        "controller_greedy_runner",
        "controller_generate_cli",
        "controller_mechanism_scorer",
    }
    assert all(
        len(lock["sha256"]) == 64
        for lock in protocol["implementation_locks"].values()
    )
    assert not (tmp_path / parent / "prospective.identity_only.jsonl").exists()


def test_freezer_is_append_only(tmp_path: Path) -> None:
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_freeze(project_root=tmp_path, output_dir=output)


def test_candidate_pool_rejects_incomplete_pair(tmp_path: Path) -> None:
    pair = _action_pair(
        "musique", "train-one", "Who founded lowercaseepsilon academy?", "train"
    )
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, pair[:1])
    with pytest.raises(ValueError, match="exact q1/q2 pair"):
        _load_action_candidates(path, source_split="train")


def test_candidate_pool_rejects_pair_hash_drift(tmp_path: Path) -> None:
    dataset = "musique"
    qid = "train-one"
    pair = _action_pair(
        dataset, qid, "Who founded lowercaseepsilon academy?", "train"
    )
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, pair)
    with pytest.raises(ValueError, match="action-pair content hash mismatch"):
        _load_action_candidates(
            path,
            source_split="train",
            expected_pair_hashes={
                ("train", dataset, qid): {
                    "family_sha256": pair[0]["family_sha256"],
                    "action_pair_canonical_sha256": "0" * 64,
                }
            },
        )


def test_freezer_runtime_and_probe_contract_match_formal_release_validator() -> None:
    body = _protocol_body(
        sealed_lock={},
        cohort_locks={},
        candidate_inventory=[],
        candidate_v4_1_evidence=None,
        consumed_inventory=[],
        implementation_locks={},
        predecessor_failure_lineage={},
        identity_continuity={},
    )
    assert body["training_contract"]["runtime_config"] == FORMAL_RUNTIME_CONFIG
    assert body["training_contract"]["probe_gates"] == FORMAL_TRAINING_PROBE_GATES
    assert body["probe_evaluation_contract"] == FORMAL_PROBE_EVALUATION_CONTRACT
    assert (
        body["future_gold_free_mechanism_gate"]
        == FORMAL_FUTURE_GOLD_FREE_MECHANISM_GATE
    )
    assert body["outcome_unlock_rule"] == FORMAL_OUTCOME_UNLOCK_RULE
    assert body["training_contract"]["runtime_config"]["experiment_id"] == (
        "QUERY-CONTROLLER-ACTION-V1-PROBE20-SEED42-V4-4"
    )
