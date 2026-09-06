from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.pilot.build_automatic_proofkg_from_plans import _ExactEntityCacheLinker
from scripts.prepare.freeze_2wiki_extension_clean_closure_v2 import (
    CUTOFF,
    EXPERIMENT_ID,
    file_lock,
    freeze,
    md5_file,
    sha256_file,
    validate_canonical_cache,
)
from scripts.prepare.run_2wiki_extension_clean_closure_v2_locked import validate_lock
from scripts.prepare.run_inference_proofkg_closure import _run_executor


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _qhash(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()


def _build_fixture(tmp_path: Path):
    plans_path = tmp_path / "plans.jsonl"
    plans = []
    for index in range(300):
        qid = f"q{index}"
        question = f"Question {index}?"
        plans.append(
            {
                "dataset": "2wikimultihopqa",
                "qid": qid,
                "question_key": f"2wikimultihopqa::{qid}",
                "question": question,
                "question_sha256": _qhash(question),
                "predicted_target": {
                    "anchors": [f"Entity {index}"],
                    "steps": [
                        {
                            "subject": f"Entity {index}",
                            "pid": "P17",
                            "output_slot": "hop_1",
                            "dependencies": [],
                        }
                    ],
                },
                "gold_access": False,
            }
        )
    _write_jsonl(plans_path, plans)
    planner_protocol_path = tmp_path / "planner_protocol.json"
    _write_json(planner_protocol_path, {"status": "FROZEN"})

    resolver_dir = tmp_path / "resolver"
    resolver_dir.mkdir()
    resolver_protocol_path = tmp_path / "resolver_protocol.json"
    _write_json(
        resolver_protocol_path,
        {
            "status": "FROZEN_BEFORE_NETWORK_NO_GOLD",
            "outputs": {"planned_resolution_output_dir": str(resolver_dir.resolve())},
        },
    )
    title_cache = resolver_dir / "title_cache.jsonl"
    entity_cache = resolver_dir / "entity_cache.jsonl"
    results_path = resolver_dir / "resolution_results.jsonl"
    _write_jsonl(title_cache, [{"label": "Entity 0", "qid": "Q1"}])
    _write_jsonl(entity_cache, [])
    _write_jsonl(
        results_path,
        [
            {
                "request_id": "r1",
                "gold_access": False,
                "outcome": "positive",
                "resolution_method": "exact_wikipedia_title",
                "completed_root_anchor_surface": "Entity 0",
                "root_anchor_surface": "Entity 0",
                "resolved_qid": "Q1",
            }
        ],
    )
    resolver_report = {
        "schema_version": "2wiki-root-anchor-resolution-result-v1",
        "status": "PASS_ROOT_ANCHOR_CONTINUE_GATE",
        "continue_gate_before_clean_closure_v2": {
            "all_pass": True,
            "decision": "CONTINUE_TO_CLEAN_CLOSURE_V2",
        },
        "checks": {
            "request_log_join_rate": 1.0,
            "runtime_errors": 0,
            "gold_access_false": True,
            "old_cache_fallback": False,
        },
        "counts": {"results": 1},
        "inputs": {"protocol": file_lock(resolver_protocol_path)},
        "outputs": {
            "resolution_results": file_lock(results_path),
            "title_cache": file_lock(title_cache),
            "entity_cache": file_lock(entity_cache),
        },
        "gold_access": False,
        "training_started": False,
    }
    _write_json(resolver_dir / "report.json", resolver_report)
    _write_json(resolver_dir / "manifest.json", {"status": resolver_report["status"]})

    combined_path = tmp_path / "2wiki_proofkg_extension_combined_v1_n350.jsonl"
    _write_jsonl(combined_path, plans)
    store = tmp_path / "versioned_store_v5"
    store.mkdir()
    aliases = store / "aliases.jsonl"
    edges = store / "edges.jsonl"
    _write_jsonl(aliases, [])
    _write_jsonl(edges, [])
    _write_json(
        store / "store_manifest.json",
        {
            "schema_version": "versioned-2wiki-evidence-store-1",
            "status": "COMPLETE_NOT_EVALUATED",
            "experiment_id": "VERSIONED-2WIKI-EVIDENCE-STORE-V5-MIXED3-V4-SEED42",
            "inputs": {
                "excluded_cohorts": [
                    {
                        "path": str(combined_path.resolve()),
                        "md5": md5_file(combined_path),
                    }
                ]
            },
            "outputs": {
                "aliases": {"path": str(aliases.resolve()), "md5": md5_file(aliases)},
                "edges": {"path": str(edges.resolve()), "md5": md5_file(edges)},
            },
        },
    )

    historical = tmp_path / "historical.jsonl"
    _write_jsonl(
        historical,
        [
            {
                "schema_version": "wikidata-historical-entity-revision-1",
                "key": f"wikidata-historical-entity-revision-1::{CUTOFF}::Q1",
                "qid": "Q1",
                "cutoff": CUTOFF,
                "entity": {"id": "Q1", "claims": {}},
            }
        ],
    )
    closure_v1_report = tmp_path / "closure_v1_report.json"
    _write_json(
        closure_v1_report,
        {
            "schema_version": "inference-proofkg-closure-v3b-1",
            "stop_reason": "no_new_requests",
            "cutoff": CUTOFF,
            "closure_cache_sha256": sha256_file(historical),
        },
    )
    return {
        "plans": plans_path,
        "planner_protocol": planner_protocol_path,
        "resolver_protocol": resolver_protocol_path,
        "resolver_dir": resolver_dir,
        "store": store,
        "historical": historical,
        "closure_v1_report": closure_v1_report,
        "lock_dir": tmp_path / "lock",
        "run_dir": tmp_path / "run",
        "attestation_dir": tmp_path / "attestation",
    }


def _freeze_fixture(paths):
    return freeze(
        plans_path=paths["plans"],
        planner_protocol_path=paths["planner_protocol"],
        resolver_protocol_path=paths["resolver_protocol"],
        resolver_dir=paths["resolver_dir"],
        clean_store_dir=paths["store"],
        historical_cache_path=paths["historical"],
        closure_v1_report_path=paths["closure_v1_report"],
        output_dir=paths["lock_dir"],
        planned_run_dir=paths["run_dir"],
        planned_attestation_dir=paths["attestation_dir"],
        experiment_id=EXPERIMENT_ID,
    )


def test_exact_entity_cache_linker_bypasses_fuzzy_and_known_fixes(tmp_path):
    cache = tmp_path / "entity.jsonl"
    _write_jsonl(cache, [{"label": "Alpha Entity", "qid": "Q123"}])
    linker = _ExactEntityCacheLinker(cache)
    assert linker.link_single("alpha entity").selected_qid == "Q123"
    assert linker.link_single("Alpha Entit").abstained  # would be a fuzzy near-match
    assert linker.link_single("Ed Wood").abstained  # regular EntityLinker has a hard-coded fix


def test_closure_propagates_exact_mode_and_experiment_id(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs

    monkeypatch.setattr("scripts.prepare.run_inference_proofkg_closure.subprocess.run", fake_run)
    args = SimpleNamespace(
        versioned_alias_store=str(tmp_path / "store"),
        plans=tmp_path / "plans.jsonl",
        protocol=tmp_path / "protocol.json",
        entity_index=tmp_path / "ABSENT.json",
        entity_cache=tmp_path / "entity.jsonl",
        title_cache=tmp_path / "title.jsonl",
        dataset="2wikimultihopqa",
        experiment_id="CLEAN-V2",
        exact_entity_cache_only=True,
    )
    _run_executor(tmp_path / "round_0", tmp_path / "historical.jsonl", args)
    command = captured["command"]
    assert "--exact_entity_cache_only" in command
    assert command[command.index("--experiment_id") + 1] == "CLEAN-V2-ROUND_0"


def test_canonical_cache_rejects_unsorted_and_conflicting_rows(tmp_path):
    cache = tmp_path / "cache.jsonl"
    _write_jsonl(cache, [{"label": "Beta", "qid": "Q2"}, {"label": "Alpha", "qid": "Q1"}])
    with pytest.raises(ValueError, match="deterministically sorted"):
        validate_canonical_cache(cache)
    _write_jsonl(tmp_path / "conflict.jsonl", [{"label": "Alpha", "qid": "Q1"}, {"label": "alpha", "qid": "Q2"}])
    with pytest.raises(ValueError, match="multiple QIDs"):
        validate_canonical_cache(tmp_path / "conflict.jsonl")


def test_freeze_and_locked_preflight_bind_every_input_and_exact_mode(tmp_path):
    paths = _build_fixture(tmp_path)
    report = _freeze_fixture(paths)
    assert report["status"] == "FROZEN_DIAGNOSTIC_CANDIDATE_BUILD_BEFORE_NETWORK"
    protocol_path = paths["lock_dir"] / "protocol.json"
    protocol, command = validate_lock(protocol_path)
    assert "--exact_entity_cache_only" in command
    assert command[command.index("--entity_cache") + 1] == str(
        (paths["resolver_dir"] / "entity_cache.jsonl").resolve()
    )
    assert command[command.index("--title_cache") + 1] == str(
        (paths["resolver_dir"] / "title_cache.jsonl").resolve()
    )
    assert protocol["scientific_boundary"]["final_training_eligibility"] is False
    assert protocol["scientific_boundary"]["v5_not_complete_ledger_attested"] is True


def test_locked_preflight_fails_hard_on_cache_hash_drift(tmp_path):
    paths = _build_fixture(tmp_path)
    _freeze_fixture(paths)
    with (paths["resolver_dir"] / "entity_cache.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"label": "Injected", "qid": "Q999"}) + "\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_lock(paths["lock_dir"] / "protocol.json")


def test_freeze_refuses_failed_root_resolution_gate(tmp_path):
    paths = _build_fixture(tmp_path)
    report_path = paths["resolver_dir"] / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "FAIL_ROOT_ANCHOR_CONTINUE_GATE"
    _write_json(report_path, report)
    _write_json(paths["resolver_dir"] / "manifest.json", {"status": report["status"]})
    with pytest.raises(ValueError, match="did not pass"):
        _freeze_fixture(paths)


def test_freeze_refuses_existing_run_directory(tmp_path):
    paths = _build_fixture(tmp_path)
    paths["run_dir"].mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        _freeze_fixture(paths)
