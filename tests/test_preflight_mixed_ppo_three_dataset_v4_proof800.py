from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from kgproweight.kg.question_kg import make_question_kg_record, question_sha256
from kgproweight.reward.trajectory_source_gate import make_source_gate_record
from scripts.prepare.freeze_mixed_ppo_three_dataset_v4_proof800 import expand_k4
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.prepare.materialize_mixed_ppo_three_dataset_v4_proof800 import (
    EXPECTED_PROOF_SUPPLY_SCHEMA,
    EXPECTED_PROOF_SUPPLY_STATUS,
    EXPECTED_PROTOCOL_EXPERIMENT,
    EXPECTED_PROTOCOL_SCHEMA,
    EXPECTED_PROTOCOL_STATUS,
    REPORT_SCHEMA,
    STATUS as DATA_STATUS,
    _identity,
)
from scripts.prepare.preflight_mixed_ppo_three_dataset_v4_proof800 import (
    _validate_report_and_manifest,
    audit_core_assets,
)


@pytest.fixture(scope="module")
def exact_v4_rows() -> dict[str, list[dict]]:
    silver = []
    records = []
    gates = []
    groups = []
    for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
        for index in range(1000):
            qid = f"{dataset}-{index}"
            question = f"Unique question for {dataset} item {index}?"
            eligible = dataset == "2wikimultihopqa" and index < 800
            triples = [[f"Head {index}", "linked to", f"Tail {index}"]] if eligible else []
            plan = {
                "recognized": eligible,
                "hops": (
                    [{"subject": f"Head {index}", "pids": ["P1"], "output_slot": "hop_1"}]
                    if eligible
                    else []
                ),
            }
            provenance = {
                "builder_version": "synthetic-official-v3" if eligible else "outcome-only",
                "gold_access": False,
                "complete_plan_execution": eligible,
                "historical_cutoff": "2020-12-09T23:59:59Z",
            }
            record = make_question_kg_record(
                dataset=dataset,
                qid=qid,
                question=question,
                triples=triples,
                query_plan=plan,
                provenance=provenance,
            )
            record["execution"] = {
                "complete_plan_execution": eligible,
                "hops": (
                    [
                        {
                            "hop_index": 1,
                            "input_entities": [{"qid": f"Q{index + 1}"}],
                            "matches": copy.deepcopy(triples),
                        }
                    ]
                    if eligible
                    else []
                ),
            }
            record["runtime_error"] = None
            record["process_reward_eligible"] = eligible
            gate = make_source_gate_record(
                record,
                dataset=dataset,
                qid=qid,
                question=question,
                text_evidence_available=True,
                historical_cutoff="2020-12-09T23:59:59Z",
            )
            family = family_sha256(question)
            passages = [
                {
                    "id": f"{qid}-{passage_index}",
                    "source": "canonical",
                    "contents": f"Evidence {passage_index} for {qid}.",
                }
                for passage_index in range(10)
            ]
            silver.append(
                {
                    "accepted": True,
                    "dataset": dataset,
                    "qid": qid,
                    "question": question,
                    "answer": f"gold-{qid}",
                    "kg_subgraph": copy.deepcopy(triples),
                    "retrieved_passages": passages,
                    "steps": [],
                    "teacher_output": "",
                    "passage_evidence": None,
                    "evidence_mode": None,
                    "metadata": {
                        "gold_answer": f"gold-{qid}",
                        "gold_use": "outcome_reward_label_only",
                        "family_version": FAMILY_VERSION,
                        "family_sha256": family,
                        "process_reward_eligible": eligible,
                        "source_gold_trace_removed": True,
                        "failed_qpeg_or_saeg_p_edges_included": False,
                    },
                }
            )
            records.append(record)
            gates.append(gate)
            groups.append(
                {
                    "schema_version": "mixed-ppo-prompt-group-v4-proof800",
                    "prompt_group_index": len(groups) + 1,
                    "question_key": f"{dataset}::{qid}",
                    "dataset": dataset,
                    "qid": qid,
                    "question": question,
                    "question_sha256": question_sha256(question),
                    "family_version": FAMILY_VERSION,
                    "family_sha256": family,
                    "route": "2wiki_proof" if eligible else f"{dataset}_outcome",
                    "process_reward_eligible": eligible,
                    "gold_access": False,
                }
            )
    weights = [
        {
            "dataset": row["dataset"],
            "qid": row["qid"],
            "question_sha256": row["question_sha256"],
            "process_reward_eligible": row["process_reward_eligible"],
            "sampling_probability": 1.0 / 3000,
        }
        for row in groups
    ]
    replay = [
        {
            "dataset": "hotpotqa",
            "qid": f"replay-{index}",
            "question": f"Replay-only unique question {index}?",
        }
        for index in range(2000)
    ]
    protected = [
        {
            "dataset": "musique",
            "qid": f"protected-{index}",
            "question": f"Protected-only unique question {index}?",
        }
        for index in range(3)
    ]
    return {
        "silver": silver,
        "question_kg": records,
        "source_gates": gates,
        "weights": weights,
        "groups": groups,
        "schedule": expand_k4(groups),
        "replay": replay,
        "protected": protected,
    }


def test_exact_v4_core_hard_gates_pass(exact_v4_rows: dict[str, list[dict]]):
    checks = audit_core_assets(**exact_v4_rows)
    assert checks
    assert all(checks.values()), {name: value for name, value in checks.items() if not value}


def test_preflight_detects_graph_gold_schedule_and_overlap_drift(
    exact_v4_rows: dict[str, list[dict]],
):
    values = dict(exact_v4_rows)
    values["silver"] = list(values["silver"])
    values["silver"][0] = copy.deepcopy(values["silver"][0])
    values["silver"][0]["retrieved_passages"][0]["answer"] = "forbidden"
    values["source_gates"] = list(values["source_gates"])
    values["source_gates"][1000] = copy.deepcopy(values["source_gates"][1000])
    values["source_gates"][1000]["eligibility_checks"]["traceable"] = False
    values["schedule"] = list(values["schedule"])
    values["schedule"][-1] = copy.deepcopy(values["schedule"][-1])
    values["schedule"][-1]["qid"] = "schedule-drift"
    values["replay"] = list(values["replay"])
    values["replay"][0] = {
        "dataset": values["silver"][0]["dataset"],
        "qid": values["silver"][0]["qid"],
        "question": values["silver"][0]["question"],
    }

    checks = audit_core_assets(**values)
    assert checks["gold_forbidden_from_prompt_evidence_and_traces"] is False
    assert checks["source_gate_strict_and_fail_closed"] is False
    assert checks["fixed_rollout_schedule_k4_12000"] is False
    assert checks[
        "population_replay_qid_hash_current_family_overlap_zero"
    ] is False


def test_preflight_recomputes_source_gate_from_question_kg(
    exact_v4_rows: dict[str, list[dict]],
):
    values = dict(exact_v4_rows)
    values["question_kg"] = list(values["question_kg"])
    values["question_kg"][1000] = copy.deepcopy(values["question_kg"][1000])
    # KG and execution hashes remain unchanged, so a preflight which trusts
    # the stored gate would miss this eligibility-critical plan mutation.
    values["question_kg"][1000]["query_plan"]["recognized"] = False
    checks = audit_core_assets(**values)
    assert checks["source_gate_payload_hashes_exact"] is True
    assert checks["source_gate_schema_version_and_recomputation_exact"] is False


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_release_manifest_binds_official_v3_and_rejects_legacy_tag(tmp_path: Path):
    data_dir = tmp_path / "data"
    replay_dir = tmp_path / "replay"
    proof_dir = tmp_path / "proof"
    protected_dir = tmp_path / "protected"
    data_dir.mkdir()
    replay_dir.mkdir()
    proof_dir.mkdir()
    protected_dir.mkdir()
    paths = {}
    for name in (
        "silver_train",
        "question_kg_records",
        "source_gate_records",
        "sampling_weights",
        "prompt_groups",
        "fixed_rollout_schedule",
    ):
        path = data_dir / f"{name}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        paths[name] = path
    replay_paths = {}
    for name in ("silver_train.jsonl", "selection_records.jsonl", "report.json", "manifest.json"):
        path = replay_dir / name
        path.write_text("{}\n", encoding="utf-8")
        replay_paths[name] = _identity(path)
    proof_payloads = {}
    for name in (
        "silver_train",
        "question_kg_records",
        "source_gate_records",
        "proof_candidates",
    ):
        path = proof_dir / f"{name}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        proof_payloads[name] = _identity(path)
    proof_report = proof_dir / "report.json"
    _write(
        proof_report,
        {
            "schema_version": EXPECTED_PROOF_SUPPLY_SCHEMA,
            "experiment_id": "TEST-OFFICIAL-V3",
            "status": EXPECTED_PROOF_SUPPLY_STATUS,
            "checks": {"all": True},
            "outputs": copy.deepcopy(proof_payloads),
            "training_started": False,
        },
    )
    proof_manifest = proof_dir / "manifest.json"
    _write(
        proof_manifest,
        {
            "status": EXPECTED_PROOF_SUPPLY_STATUS,
            "run": {
                "phase": "unified_2wiki_proofkg_official_raw_v3_candidate_supply",
                "experiment_id": "TEST-OFFICIAL-V3",
                "report": _identity(proof_report),
                "training_started": False,
            },
        },
    )
    proof_release = {
        "report.json": _identity(proof_report),
        "manifest.json": _identity(proof_manifest),
    }
    protocol_path = tmp_path / "protocol.json"
    protocol = {
        "schema_version": EXPECTED_PROTOCOL_SCHEMA,
        "status": EXPECTED_PROTOCOL_STATUS,
        "experiment_id": EXPECTED_PROTOCOL_EXPERIMENT,
        "gates": {"all": True},
        "outputs": {
            name: _identity(paths[name])
            for name in ("sampling_weights", "prompt_groups", "fixed_rollout_schedule")
        },
    }
    _write(protocol_path, protocol)
    protocol_manifest_path = tmp_path / "manifest.json"
    _write(
        protocol_manifest_path,
        {
            "status": EXPECTED_PROTOCOL_STATUS,
            "run": {
                "phase": "mixed_ppo_v4_answer_free_protocol_freeze",
                "experiment_id": EXPECTED_PROTOCOL_EXPERIMENT,
                "protocol_sha256": _identity(protocol_path)["sha256"],
                "training_started": False,
            },
        },
    )
    protected = {}
    for index, name in enumerate(("ledger", "report", "manifest"), start=1):
        path = protected_dir / name
        path.write_text(f"protected-{index}\n", encoding="utf-8")
        protected[name] = _identity(path)
    report = {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": "TEST-FINAL-V4",
        "status": DATA_STATUS,
        "training_started": False,
        "gates": {"all": True},
        "inputs": {
            "protocol": _identity(protocol_path),
            "protocol_manifest": _identity(protocol_manifest_path),
            "proof_supply": copy.deepcopy(proof_payloads),
            "replay": copy.deepcopy(replay_paths),
            "release_metadata": {
                "proof_supply": copy.deepcopy(proof_release),
                "protected_ledger": copy.deepcopy(protected),
            },
        },
        "outputs": {name: _identity(path) for name, path in paths.items()},
    }
    report_path = data_dir / "report.json"
    _write(report_path, report)
    manifest = {
        "status": DATA_STATUS,
        "run": {
            "phase": "mixed_ppo_v4_proof800_data_materialization",
            "experiment_id": "TEST-FINAL-V4",
            "training_started": False,
            "protocol_manifest": _identity(protocol_manifest_path),
            "report": _identity(report_path),
            "outputs": copy.deepcopy(report["outputs"]),
            "replay": copy.deepcopy(replay_paths),
            "proof_supply": {
                "schema_version": EXPECTED_PROOF_SUPPLY_SCHEMA,
                "status": EXPECTED_PROOF_SUPPLY_STATUS,
                "payloads": copy.deepcopy(proof_payloads),
                "release_metadata": copy.deepcopy(proof_release),
            },
        },
    }
    checks = _validate_report_and_manifest(
        data_dir=data_dir,
        replay_dir=replay_dir,
        report=report,
        manifest=manifest,
        paths=paths,
        protected_binding=protected,
    )
    assert all(checks.values()), checks

    manifest["run"]["proof_supply"]["schema_version"] = (
        "2wiki-unified-proofkg-candidate-supply-v2"
    )
    checks = _validate_report_and_manifest(
        data_dir=data_dir,
        replay_dir=replay_dir,
        report=report,
        manifest=manifest,
        paths=paths,
        protected_binding=protected,
    )
    assert checks["official_unified_v3_payload_and_release_hashes_bound"] is False
