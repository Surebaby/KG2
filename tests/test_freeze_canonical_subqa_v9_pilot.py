from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare import freeze_canonical_subqa_v9_pilot as freeze


def _candidate(dataset: str, qid: str, family: str, qhash: str | None = None):
    return freeze.Candidate(
        dataset=dataset,
        qid=qid,
        question=f"question {qid}",
        question_sha256=qhash or f"hash-{qid}",
        family_sha256=family,
    )


def test_selection_excludes_qid_and_family_and_is_deterministic():
    dataset = "hotpotqa"
    candidates = {
        "a": _candidate(dataset, "a", "family-a"),
        "b": _candidate(dataset, "b", "family-b"),
        "c": _candidate(dataset, "c", "family-c"),
        "d": _candidate(dataset, "d", "family-d"),
        "e": _candidate(dataset, "e", "family-e"),
    }
    excluded_qids = {(dataset, "a")}
    excluded_families = {(dataset, "family-b")}
    first, stats = freeze.select_fresh_train_pilot(
        dataset=dataset,
        candidates=candidates,
        excluded_qids=excluded_qids,
        excluded_families=excluded_families,
        salt="fixed-salt",
        n=2,
    )
    second, _ = freeze.select_fresh_train_pilot(
        dataset=dataset,
        candidates=dict(reversed(tuple(candidates.items()))),
        excluded_qids=excluded_qids,
        excluded_families=excluded_families,
        salt="fixed-salt",
        n=2,
    )
    assert first == second
    assert {row.qid for row in first}.isdisjoint({"a", "b"})
    assert stats["eligible_unique_dataset_scoped_families"] == 3
    assert stats["remaining_one_per_family_capacity_after_selection"] == 1


def test_selection_keeps_one_qid_per_family():
    dataset = "2wikimultihopqa"
    candidates = {
        "a": _candidate(dataset, "a", "same", "02"),
        "b": _candidate(dataset, "b", "same", "01"),
        "c": _candidate(dataset, "c", "different", "03"),
    }
    selected, stats = freeze.select_fresh_train_pilot(
        dataset=dataset,
        candidates=candidates,
        excluded_qids=set(),
        excluded_families=set(),
        salt="fixed-salt",
        n=2,
    )
    assert len(selected) == 2
    assert len({row.family_sha256 for row in selected}) == 2
    assert {row.qid for row in selected} == {"b", "c"}
    assert stats["eligible_unique_qids"] == 3


def test_selection_fails_closed_when_family_capacity_is_too_small():
    dataset = "musique"
    with pytest.raises(ValueError, match="eligible families"):
        freeze.select_fresh_train_pilot(
            dataset=dataset,
            candidates={"a": _candidate(dataset, "a", "one")},
            excluded_qids=set(),
            excluded_families=set(),
            salt="fixed-salt",
            n=2,
        )


def test_identity_projection_does_not_access_gold_fields():
    class GuardedRow(dict):
        def get(self, key, default=None):
            if key in {"answer", "golden_answers", "supporting_facts", "decomposition"}:
                raise AssertionError(f"forbidden field accessed: {key}")
            return super().get(key, default)

    row = GuardedRow(
        {
            "id": "safe-id",
            "question": "Who is linked to the safe entity?",
            "answer": "SECRET",
        }
    )
    candidate = freeze._candidate_from_identity_fields(row, dataset="hotpotqa")
    assert candidate is not None
    assert candidate.qid == "safe-id"


def test_parent_metadata_validation_never_needs_prospective_file(tmp_path, monkeypatch):
    project = tmp_path / "project"
    parent = project / freeze.V8_COHORT_FREEZE_DIR
    parent.mkdir(parents=True)
    report = {
        "status": "COMPLETE_FROZEN_SCOPE_A_DEV30_PROSPECTIVE300_NO_RESERVE",
        "checks": {
            "all_freeze_gates_pass": True,
            "raw_train_qid_overlap": 0,
            "raw_train_family_overlap": 0,
        },
        "prospective_seal": {"status": "FROZEN_UNOPENED_FOR_METHOD_DEVELOPMENT"},
    }
    report_path = parent / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    report_sha = freeze._sha256_file(report_path)
    manifest = {
        "status": report["status"],
        "outputs": [
            {"path": "report.json", "sha256": report_sha},
            {"path": "prospective.identity_only.jsonl", "sha256": "f" * 64},
        ],
    }
    manifest_path = parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = freeze._sha256_file(manifest_path)

    # No prospective file is created.  Success therefore proves the validator
    # uses only the report/manifest lock and never opens or hashes its content.
    result = freeze.validate_v8_parent_metadata(
        project_root=project,
        expected_report_sha256=report_sha,
        expected_manifest_sha256=manifest_sha,
    )
    assert result["prospective_content_opened"] is False
    assert result["prospective_content_hashed"] is False


def test_public_rows_have_exact_identity_allowlist():
    rows = freeze._public_rows([_candidate("hotpotqa", "qid-1", "family-1")])
    assert tuple(rows[0]) == freeze.OUTPUT_ROW_FIELDS
    assert set(rows[0]) == {"dataset", "qid", "question"}
