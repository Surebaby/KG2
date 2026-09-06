from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare.freeze_2wiki_official_raw_n1500_clean_closure_v1 import (
    DEFAULT_POLICY_DIR,
    exact_consumer_resolution,
    file_identity,
    validate_policy,
    validate_root_resolution,
)
from scripts.prepare.run_2wiki_official_raw_n1500_clean_closure_v1_locked import (
    compare_runtime_roots_to_dry_run,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _root_fixture(tmp_path: Path, *, dry_qid: str = "Q1"):
    store = tmp_path / "v6"
    aliases = store / "aliases.jsonl"
    _write_jsonl(
        aliases,
        [
            {
                "normalized_alias": "alpha (film)",
                "candidates": [{"qid": "Q2", "label": "Alpha", "evidence_count": 1}],
            }
        ],
    )
    manifest = store / "store_manifest.json"
    _write_json(manifest, {"schema_version": "versioned-2wiki-evidence-store-1"})
    v6 = {
        "path": str(store.resolve()),
        "aliases": file_identity(aliases),
        "store_manifest": file_identity(manifest),
    }

    resolver_impl = tmp_path / "resolver.py"
    resolver_impl.write_text("# frozen resolver\n", encoding="utf-8")
    occurrence = {
        "request_id": "r1",
        "question_key": "2wikimultihopqa::q1",
        "dataset": "2wikimultihopqa",
        "qid": "q1",
        "question_sha256": "a" * 64,
        "root_position": 1,
        "root_anchor_surface": "Alpha",
        "completed_root_anchor_surface": "Alpha (film)",
    }
    worklist = tmp_path / "root_worklist.jsonl"
    _write_jsonl(worklist, [{**occurrence, "gold_access": False}])
    protocol_path = tmp_path / "root_protocol.json"
    root = tmp_path / "root"
    _write_json(
        protocol_path,
        {
            "schema_version": "2wiki-full-root-anchor-resolution-protocol-v2",
            "status": "FROZEN_ALL_ROOTS_BEFORE_NETWORK_NO_GOLD_NOT_TRAINED",
            "inputs": {
                "resolver_implementation": file_identity(resolver_impl),
                "v6_store_manifest": file_identity(manifest),
                "v6_aliases": file_identity(aliases),
            },
            "outputs": {
                "worklist": file_identity(worklist),
                "planned_materialized_output_dir": str(root.resolve()),
            },
        },
    )
    title = root / "title_cache.jsonl"
    entity = root / "entity_cache.jsonl"
    results = root / "resolution_results.jsonl"
    dry = root / "consumer_dry_run.jsonl"
    _write_jsonl(title, [{"label": "Alpha (film)", "qid": "Q1"}])
    _write_jsonl(entity, [{"label": "Alpha", "qid": "Q3"}])
    _write_jsonl(
        results,
        [
            {
                **occurrence,
                "outcome": "positive",
                "resolved_qid": "Q1",
                "gold_access": False,
            }
        ],
    )
    _write_jsonl(
        dry,
        [
            {
                **occurrence,
                "projected_qid": "Q1",
                "dry_run_qid": dry_qid,
                "dry_run_source": "new_exact_title_cache",
                "gold_access": False,
            }
        ],
    )
    status = "PASS_ROOT_ANCHOR_CONTINUE_GATE_EXACT_CONSUMER"
    gates = {
        "question_identity_join_eq_1": True,
        "request_result_join_eq_1": True,
        "recognized_plan_rate_ge_0_97": True,
        "runtime_errors_zero": True,
        "gold_access_false": True,
        "v6_binding_exact": True,
        "worklist_all_recognized_roots_exact": True,
        "projection_equals_dry_run_every_occurrence": True,
        "all_roots_resolved_question_rate_ge_0_80": True,
        "anchor_occurrence_resolution_rate_ge_0_80": True,
        "all_pass": True,
        "decision": "CONTINUE_TO_CLEAN_CLOSURE",
    }
    report = {
        "schema_version": "2wiki-full-root-anchor-resolution-result-v2",
        "status": status,
        "counts": {
            "questions_total": 1500,
            "fail": 0,
            "root_anchor_occurrences": 1,
            "projection_dry_run_occurrence_matches": 1,
            "projection_dry_run_occurrence_mismatches": 0,
        },
        "rates": {
            "projection_dry_run_occurrence_match_rate": 1.0,
            "dry_run_all_roots_resolved_question_rate_all_questions": 0.9,
            "dry_run_anchor_occurrence_resolution_rate": 0.9,
        },
        "gates": gates,
        "inputs": {
            "v6_store_manifest": file_identity(manifest),
            "protocol": file_identity(protocol_path),
            "worklist": file_identity(worklist),
        },
        "outputs": {
            "resolution_results": file_identity(results),
            "consumer_dry_run": file_identity(dry),
            "title_cache": file_identity(title),
            "entity_cache": file_identity(entity),
        },
    }
    _write_json(root / "report.json", report)
    _write_json(root / "manifest.json", {"status": status})
    return protocol_path, root, v6


def test_exact_consumer_uses_title_then_unique_v6_alias_then_entity_then_abstain():
    common = {
        "title_cache": {"alpha (film)": "Q1"},
        "v6_aliases": {"alpha (film)": {"Q2"}, "beta": {"Q4"}, "amb": {"Q5", "Q6"}},
        "entity_cache": {"alpha": "Q3", "gamma": "Q7", "amb": "Q8"},
    }
    assert exact_consumer_resolution(surface="Alpha", completed_surface="Alpha (film)", **common) == (
        "new_exact_title_cache",
        "Q1",
    )
    assert exact_consumer_resolution(surface="Beta", completed_surface="Beta", **common) == (
        "clean_v6_exact_alias",
        "Q4",
    )
    assert exact_consumer_resolution(surface="Gamma", completed_surface="Gamma", **common) == (
        "new_exact_entity_cache",
        "Q7",
    )
    assert exact_consumer_resolution(surface="Amb", completed_surface="Amb", **common) == (
        "new_exact_entity_cache",
        "Q8",
    )
    assert exact_consumer_resolution(surface="Missing", completed_surface="Missing", **common)[1] is None


def test_root_validator_independently_replays_final_exact_consumer(tmp_path):
    protocol, root, v6 = _root_fixture(tmp_path)
    report, locks, dry = validate_root_resolution(
        root_protocol_path=protocol, root_dir=root, v6_store=v6
    )
    assert report["status"] == "PASS_ROOT_ANCHOR_CONTINUE_GATE_EXACT_CONSUMER"
    assert locks["consumer_dry_run"]["sha256"]
    assert dry["r1"]["dry_run_qid"] == "Q1"


def test_root_validator_rejects_projection_dry_run_that_differs_from_consumer(tmp_path):
    protocol, root, v6 = _root_fixture(tmp_path, dry_qid="Q9")
    with pytest.raises(ValueError, match="does not match exact consumer"):
        validate_root_resolution(root_protocol_path=protocol, root_dir=root, v6_store=v6)


def test_runtime_root_comparison_detects_qid_drift():
    dry = {
        ("2wikimultihopqa::q1", "Alpha"): {
            "dry_run_qid": "Q1",
        }
    }
    good = compare_runtime_roots_to_dry_run(
        question_key_value="2wikimultihopqa::q1",
        anchors=["Alpha"],
        anchor_entities={"Alpha": {"qid": "Q1", "abstained": False}},
        dry_roots=dry,
    )
    assert good["matches"] == 1 and good["mismatches"] == 0
    bad = compare_runtime_roots_to_dry_run(
        question_key_value="2wikimultihopqa::q1",
        anchors=["Alpha"],
        anchor_entities={"Alpha": {"qid": "Q2", "abstained": False}},
        dry_roots=dry,
    )
    assert bad["matches"] == 0 and bad["mismatches"] == 1


def test_real_method_policy_is_hash_bound_and_still_waiting_for_root():
    path = DEFAULT_POLICY_DIR / "protocol.json"
    if not path.is_file():
        pytest.skip("append-only n1500 method policy not materialized")
    policy = validate_policy(path)
    assert policy["status"] == "FROZEN_POLICY_WAITING_FOR_ROOT_RESOLUTION"
    assert policy["scientific_boundary"]["root_resolution_complete"] is False
    assert policy["scientific_boundary"]["training_started"] is False
    assert policy["closure_policy"]["max_rounds"] == 4
    assert policy["postflight_gates"]["strict_graph_eligible_per_question_type"] == {
        "op": ">=",
        "value": 200,
    }
