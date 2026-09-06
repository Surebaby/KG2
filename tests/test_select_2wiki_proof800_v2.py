from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

import scripts.prepare.select_2wiki_proof800_v2 as selector_v2
from kgproweight.kg.question_kg import question_sha256
from scripts.prepare.select_2wiki_proof800_v2 import (
    CLOSURE_PASS_STATUS,
    CLOSURE_REPORT_SCHEMA,
    DATASET,
    QTYPES,
    TARGET_BY_TYPE,
    TOTAL_TARGET,
    UNIFIED_CONTRACT_SCHEMA,
    UNIFIED_CONTRACT_STATUS,
    UNIFIED_RELEASE_SCHEMA,
    UNIFIED_RELEASE_STATUS,
    UNIFIED_REQUIRED_OUTPUTS,
    UNIFIED_WRAPPER_SCHEMA,
    _canonical_sha256,
    _identity,
    _silver_identity_exact,
    _validate_protocol,
    choose_exact_proof800,
    freeze_protocol_after_code,
    validate_closure_v3_release,
    validate_unified_contract,
)


TYPE_COUNTS = {
    "bridge_comparison": 390,
    "comparison": 390,
    "compositional": 389,
    "inference": 331,
}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _closure_release(tmp_path: Path) -> Path:
    directory = tmp_path / "closure"
    directory.mkdir()
    runtime: list[dict] = []
    telemetry: list[dict] = []
    index = 0
    for qtype, count in TYPE_COUNTS.items():
        for _ in range(count):
            qid = f"qid-{index}"
            key = f"{DATASET}::{qid}"
            trace = {
                "dataset": DATASET,
                "qid": qid,
                "question_key": key,
                "kg_subgraph": [],
                "execution": {},
            }
            runtime.append(trace)
            telemetry.append(
                {
                    "schema_version": (
                        "2wiki-official-raw-strict-eligibility-telemetry-v1"
                    ),
                    "dataset": DATASET,
                    "qid": qid,
                    "question_key": key,
                    "question_type": qtype,
                    "gold_access_false": True,
                    "runtime_record_sha256": _canonical_sha256(trace),
                    "kg_sha256": _canonical_sha256([]),
                    "execution_sha256": _canonical_sha256({}),
                }
            )
            index += 1
    runtime_report = directory / "runtime_report.json"
    runtime_details = directory / "runtime_details.jsonl"
    telemetry_path = directory / "strict_eligibility_telemetry.jsonl"
    execution_lock = directory / "execution_lock.json"
    closure_report = directory / "closure_report.json"
    _write_json(runtime_report, {"n": 1500})
    _write_jsonl(runtime_details, runtime)
    _write_jsonl(telemetry_path, telemetry)
    _write_json(execution_lock, {"locked": True})
    _write_json(closure_report, {"stop_reason": "no_new_requests"})
    report = {
        "schema_version": CLOSURE_REPORT_SCHEMA,
        "status": CLOSURE_PASS_STATUS,
        "all_pass": True,
        "decision": "CONTINUE_TO_PROOF800_SELECTION",
        "gold_access": False,
        "training_started": False,
        "gates": {"all_frozen_closure_gates": True},
        "inputs": {
            "execution_lock": _identity(execution_lock),
            "closure_report": _identity(closure_report),
        },
        "outputs": {
            "runtime_report": _identity(runtime_report),
            "runtime_details": _identity(runtime_details),
            "strict_eligibility_telemetry": {
                **_identity(telemetry_path),
                "rows": 1500,
            },
        },
        "scientific_boundary": {
            "structural_and_source_eligibility_only": True,
            "passages_or_answers_read": False,
            "proof800_selected": False,
            "training_started": False,
        },
    }
    report_path = directory / "report.json"
    _write_json(report_path, report)
    manifest = {
        "status": CLOSURE_PASS_STATUS,
        "run": {
            "report": _identity(report_path),
            "runtime_details": _identity(runtime_details),
            "strict_eligibility_telemetry": _identity(telemetry_path),
            "training_started": False,
        },
    }
    _write_json(directory / "manifest.json", manifest)
    return directory


def _rewrite_release_report(directory: Path, **changes: object) -> None:
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(changes)
    _write_json(report_path, report)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run"]["report"] = _identity(report_path)
    if "status" in changes:
        manifest["status"] = changes["status"]
    _write_json(manifest_path, manifest)


def test_authoritative_closure_v3_contract_is_accepted(tmp_path: Path):
    directory = _closure_release(tmp_path)
    telemetry, binding = validate_closure_v3_release(directory)
    assert len(telemetry) == 1500
    assert set(binding) == {
        "report",
        "manifest",
        "runtime_report",
        "runtime_details",
        "strict_eligibility_telemetry",
        "execution_lock",
        "closure_report",
    }


def test_old_closure_v2_schema_is_rejected_without_relabeling(tmp_path: Path):
    directory = _closure_release(tmp_path)
    _rewrite_release_report(
        directory, schema_version="2wiki-official-raw-clean-closure-v2"
    )
    with pytest.raises(ValueError, match="closure-v3"):
        validate_closure_v3_release(directory)


@pytest.mark.parametrize(
    "status",
    [
        "FAIL_DIAGNOSTIC_CLEAN_CLOSURE_RETAINED_NOT_SELECTED_NOT_TRAINED",
        "SUPERSEDED_NOT_CONSUMABLE",
    ],
)
def test_failed_or_superseded_closure_status_is_rejected(
    tmp_path: Path, status: str
):
    directory = _closure_release(tmp_path)
    _rewrite_release_report(directory, status=status)
    with pytest.raises(ValueError):
        validate_closure_v3_release(directory)


def test_append_only_supersession_marker_is_rejected(tmp_path: Path):
    directory = _closure_release(tmp_path)
    _write_json(
        directory / "metadata_addendum.json",
        {"status": "SUPERSEDED_BEFORE_SELECTION_NOT_CONSUMABLE"},
    )
    with pytest.raises(ValueError, match="superseded"):
        validate_closure_v3_release(directory)


def test_final_unified_contract_binds_materializer_code(tmp_path: Path):
    implementation = tmp_path / "materializer.py"
    implementation.write_text("SCHEMA_VERSION = 'v3'\n", encoding="utf-8")
    contract_path = tmp_path / "unified_contract.json"
    contract = {
        "schema_version": UNIFIED_CONTRACT_SCHEMA,
        "status": UNIFIED_CONTRACT_STATUS,
        "release_schema_version": UNIFIED_RELEASE_SCHEMA,
        "release_status": UNIFIED_RELEASE_STATUS,
        "candidate_wrapper_schema_version": UNIFIED_WRAPPER_SCHEMA,
        "required_outputs": list(UNIFIED_REQUIRED_OUTPUTS),
        "implementation": _identity(implementation),
        "training_started": False,
    }
    _write_json(contract_path, contract)
    loaded, binding = validate_unified_contract(contract_path)
    assert loaded == contract
    assert binding["materializer"]["sha256"] == _identity(implementation)["sha256"]
    implementation.write_text("SCHEMA_VERSION = 'drift'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 drift"):
        validate_unified_contract(contract_path)


def _selection_row(qtype: str, index: int) -> dict:
    identity = f"{qtype}-{index}"
    return {
        "dataset": DATASET,
        "qid": identity,
        "question_sha256": hashlib.sha256(identity.encode()).hexdigest(),
        "family_sha256": f"{qtype}-family-{index}",
        "question_type": qtype,
    }


def test_four_question_type_hard_gates_remain_exactly_200_each():
    assert TARGET_BY_TYPE == {qtype: 200 for qtype in QTYPES}
    assert TOTAL_TARGET == 800
    rows = [_selection_row(qtype, index) for qtype in QTYPES for index in range(200)]
    selected, stats = choose_exact_proof800(rows)
    assert len(selected) == 800
    assert Counter(row["question_type"] for row in selected) == Counter(
        TARGET_BY_TYPE
    )
    assert stats["selected_total"] == 800


def test_four_question_type_hard_gates_fail_closed_instead_of_lowering_quota():
    rows = [
        _selection_row(qtype, index)
        for qtype in QTYPES
        for index in range(199 if qtype == "inference" else 200)
    ]
    with pytest.raises(RuntimeError, match="Proof800/inference: only 199/200"):
        choose_exact_proof800(rows)


def test_silver_identity_accepts_absent_redundant_hash_after_live_recompute():
    question = "Which entity is linked by the second hop?"
    assert _silver_identity_exact(
        {
            "dataset": DATASET,
            "qid": "qid-live-hash",
            "question": question,
        },
        dataset=DATASET,
        qid="qid-live-hash",
        question=question,
        question_hash=question_sha256(question),
    )


def test_silver_identity_rejects_question_whose_live_hash_does_not_match():
    question = "Which entity is linked by the second hop?"
    assert not _silver_identity_exact(
        {
            "dataset": DATASET,
            "qid": "qid-live-hash",
            "question": "Which different entity is linked by the second hop?",
        },
        dataset=DATASET,
        qid="qid-live-hash",
        question=question,
        question_hash=question_sha256(question),
    )


def test_silver_identity_rejects_inconsistent_explicit_hash():
    question = "Which entity is linked by the second hop?"
    assert not _silver_identity_exact(
        {
            "dataset": DATASET,
            "qid": "qid-live-hash",
            "question": question,
            "question_sha256": "0" * 64,
        },
        dataset=DATASET,
        qid="qid-live-hash",
        question=question,
        question_hash=question_sha256(question),
    )


def test_refrozen_protocol_uses_unique_id_preserves_old_and_detects_code_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    old_dir = tmp_path / "old_preregistration"
    old_dir.mkdir()
    sentinel = old_dir / "sentinel.txt"
    sentinel.write_text("immutable-old-protocol\n", encoding="utf-8")

    cohort_rows = []
    for qtype, count in TYPE_COUNTS.items():
        for index in range(count):
            cohort_rows.append(
                {
                    "qid": f"{qtype}-{index}",
                    "question_sha256": hashlib.sha256(
                        f"question-{qtype}-{index}".encode()
                    ).hexdigest(),
                    "family_sha256": hashlib.sha256(
                        f"family-{qtype}-{index}".encode()
                    ).hexdigest(),
                    "question_type": qtype,
                }
            )
    cohort_path = tmp_path / "cohort.jsonl"
    _write_jsonl(cohort_path, cohort_rows)
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("", encoding="utf-8")
    bound_path = tmp_path / "bound.json"
    _write_json(bound_path, {"bound": True})
    contract_path = tmp_path / "unified_contract.json"
    _write_json(contract_path, {"contract": True})
    contract = {
        "schema_version": UNIFIED_CONTRACT_SCHEMA,
        "status": UNIFIED_CONTRACT_STATUS,
        "release_schema_version": UNIFIED_RELEASE_SCHEMA,
        "release_status": UNIFIED_RELEASE_STATUS,
        "candidate_wrapper_schema_version": UNIFIED_WRAPPER_SCHEMA,
        "required_outputs": list(UNIFIED_REQUIRED_OUTPUTS),
    }
    contract_binding = {
        "contract": _identity(contract_path),
        "materializer": _identity(bound_path),
    }
    monkeypatch.setattr(
        selector_v2,
        "_cohort_rows",
        lambda _path: (cohort_rows, {"cohort": _identity(cohort_path)}),
    )
    monkeypatch.setattr(
        selector_v2,
        "_validate_planner_postflight",
        lambda _path: {"predictions": _identity(bound_path)},
    )
    monkeypatch.setattr(
        selector_v2,
        "validate_protected_ledger_release",
        lambda _path: (empty_path, {"ledger": _identity(empty_path)}),
    )
    monkeypatch.setattr(
        selector_v2,
        "_validate_replay_release",
        lambda _path: (empty_path, {"selection_records": _identity(empty_path)}),
    )
    monkeypatch.setattr(
        selector_v2,
        "_validate_ordinary_release",
        lambda _path: (empty_path, {"ordinary200": _identity(empty_path)}),
    )
    monkeypatch.setattr(
        selector_v2,
        "validate_closure_v3_release",
        lambda _path: ([], {"report": _identity(bound_path)}),
    )
    monkeypatch.setattr(
        selector_v2,
        "validate_unified_contract",
        lambda _path: (contract, contract_binding),
    )

    new_dir = tmp_path / "p0refresh1_preregistration"
    experiment_id = "2WIKI-PROOF800-TEST-PROTOCOL-P0REFRESH1"
    freeze_protocol_after_code(
        cohort_release=cohort_path,
        planner_postflight=bound_path,
        protected_ledger_dir=tmp_path,
        replay_dir=tmp_path,
        ordinary_protocol=bound_path,
        closure_dir=tmp_path,
        unified_contract_path=contract_path,
        output_dir=new_dir,
        experiment_id=experiment_id,
    )
    protocol = json.loads((new_dir / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((new_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((new_dir / "manifest.json").read_text(encoding="utf-8"))
    assert protocol["experiment_id"] == experiment_id
    assert report["experiment_id"] == experiment_id
    assert manifest["run"]["experiment_id"] == experiment_id
    assert sentinel.read_text(encoding="utf-8") == "immutable-old-protocol\n"
    loaded, loaded_cohort = _validate_protocol(new_dir)
    assert loaded["experiment_id"] == experiment_id
    assert len(loaded_cohort) == 1500

    protocol["code"]["v4_freezer"]["sha256"] = "0" * 64
    _write_json(new_dir / "protocol.json", protocol)
    report["protocol"] = _identity(new_dir / "protocol.json")
    _write_json(new_dir / "report.json", report)
    manifest["run"]["protocol"] = _identity(new_dir / "protocol.json")
    manifest["run"]["report"] = _identity(new_dir / "report.json")
    _write_json(new_dir / "manifest.json", manifest)
    with pytest.raises(ValueError, match="SHA256 drift"):
        _validate_protocol(new_dir)
