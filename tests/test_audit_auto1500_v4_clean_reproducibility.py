from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from kgproweight.kg.question_kg import make_question_kg_record, question_sha256
from scripts.prepare.audit_auto1500_v4_clean_reproducibility import (
    DATASET,
    audit_candidate,
    canonical_edge_identity,
    canonical_sha256,
    make_reresolution_rows,
    PROTECTED_IDENTITY_SCHEMA_VERSION,
    PROTECTED_LEDGER_SCHEMA_VERSION,
    PROTECTED_LEDGER_STATUS,
    recover_expected_edge,
    validate_clean_store_protected_ledger,
    validate_protected_ledger_release,
)
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256


class _Store:
    def __init__(self, edges=(), aliases=None):
        self.edges = list(edges)
        self.aliases = aliases or {}

    def fetch_edges(self, qid, pids):
        return [
            dict(row)
            for row in self.edges
            if row["head_qid"] == qid and row["pid"] in pids
        ]

    def resolve(self, surface):
        qid = self.aliases.get(surface)
        return SimpleNamespace(
            abstained=qid is None,
            selected_qid=qid,
        )


def _edge(*, tail_qid, tail_value, head_label="Alice", raw=None):
    return {
        "head_qid": "Q1",
        "head_label": head_label,
        "pid": "P27",
        "relation": "country of citizenship",
        "tail_qid": tail_qid,
        "tail_value": tail_value,
        "tail_raw_value": raw,
    }


def _candidate():
    plan = {
        "recognized": True,
        "planner_version": "test-planner",
        "anchors": ["Original Alice"],
        "hops": [
            {
                "subject": "Original Alice",
                "pids": ["P27"],
                "output_slot": "hop_1",
            }
        ],
    }
    record = make_question_kg_record(
        dataset=DATASET,
        qid="question-1",
        question="What country is Original Alice from?",
        triples=[["Alice", "country of citizenship", "British"]],
        query_plan=plan,
        provenance={
            "builder_version": "test-builder",
            "gold_access": False,
            "complete_plan_execution": True,
        },
    )
    record.update(
        {
            "execution": {
                "anchor_entities": {
                    "Original Alice": {
                        "surface": "Original Alice",
                        # These are old resolver outputs and must not self-attest.
                        "resolved_surface": "Self-derived Alice",
                        "label": "Self-derived Alice",
                        "qid": "Q1",
                        "abstained": False,
                    }
                },
                "hops": [
                    {
                        "hop_index": 1,
                        "pids": ["P27"],
                        "input_entities": [{"qid": "Q1"}],
                        "matches": [
                            ["Alice", "country of citizenship", "British"]
                        ],
                        "output_entities": [],
                    }
                ],
                "complete_plan_execution": True,
            },
            "runtime_error": None,
        }
    )
    question = record["question"]
    return {
        "question_key": record["question_key"],
        "dataset": DATASET,
        "qid": record["qid"],
        "question": question,
        "question_sha256": question_sha256(question),
        "family_sha256": family_sha256(question),
        "question_type": "inference",
        "runtime": record,
    }


def test_entity_identity_uses_qid_not_same_display_label():
    first = _edge(tail_qid="Q145", tail_value="British")
    second = _edge(tail_qid="Q174193", tail_value="British")
    assert canonical_edge_identity(first) != canonical_edge_identity(second)


def test_legacy_store_only_disambiguates_then_independent_source_must_contain_edge():
    historical = [
        _edge(tail_qid="Q145", tail_value="British"),
        _edge(tail_qid="Q174193", tail_value="British"),
    ]
    legacy = [
        _edge(
            tail_qid="Q145",
            tail_value="British",
            head_label="Alice (researcher)",
        )
    ]
    identity, method = recover_expected_edge(
        match=["Alice", "country of citizenship", "British"],
        pairs=[("Q1", "P27")],
        historical_edges=historical,
        legacy_identity_edges=legacy,
        output_entities=[],
    )
    assert identity == ("Q1", "P27", "entity", "Q145")
    assert method == "legacy_identity_tail_disambiguation"
    assert identity in {canonical_edge_identity(row) for row in historical}


def test_old_runtime_resolved_label_cannot_self_attest_root():
    candidate = _candidate()
    independent_edge = _edge(tail_qid="Q145", tail_value="British")
    result = audit_candidate(
        candidate,
        old_identity_store=_Store([independent_edge]),
        clean_store=_Store([independent_edge]),
        historical=_Store([independent_edge]),
        historical_entities={
            "Q1": {
                "entity": {
                    "id": "Q1",
                    "labels": {"en": {"value": "Self-derived Alice"}},
                    "aliases": {},
                }
            }
        },
        cutoff="2020-12-09T23:59:59Z",
    )
    assert result["question_edge"][
        "all_executed_edges_independently_reproduced"
    ] is True
    assert result["question_root"][
        "all_root_anchors_independently_attested"
    ] is False
    assert result["roots"][0]["surface_variants"] == ["Original Alice"]


def test_reresolution_worklist_has_surfaces_but_expected_qids_are_separate():
    candidate = _candidate()
    runtime = candidate["runtime"]
    root_summary = {
        "question_key": candidate["question_key"],
        "old_plan_sha256": canonical_sha256(runtime["query_plan"]),
        "old_execution_sha256": canonical_sha256(runtime["execution"]),
        "all_root_anchors_independently_attested": False,
    }
    root = {
        "question_key": candidate["question_key"],
        "root_anchor_surface": "Original Alice",
        "expected_old_qid_audit_only": "Q1",
    }
    worklist, audit_only = make_reresolution_rows(
        [candidate], [root_summary], [root]
    )
    assert worklist[0]["root_anchor_surfaces"] == ["Original Alice"]
    assert "Q1" not in json.dumps(worklist)
    assert audit_only[0]["expected_old_root_qids"][0]["expected_old_qid"] == "Q1"
    assert audit_only[0]["must_not_be_model_or_resolver_input"] is True


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protected_release(tmp_path: Path) -> Path:
    release = tmp_path / "ledger"
    release.mkdir()
    question = "Who is Alice's mother?"
    row = {
        "schema_version": PROTECTED_IDENTITY_SCHEMA_VERSION,
        "dataset": DATASET,
        "qid": "heldout-1",
        "question": question,
        "question_sha256": question_sha256(question),
        "family_version": "answer-free-lexical-family-v1",
        "family_sha256": family_sha256(question),
        "gold_access": False,
        "source_paths": [],
        "source_roles": [],
    }
    ledger = release / "protected_identities.question_only.jsonl"
    ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")
    # The release validator also checks that every frozen source still exists.
    source = release / "source.question_only.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = {
        "schema_version": PROTECTED_LEDGER_SCHEMA_VERSION,
        "status": PROTECTED_LEDGER_STATUS,
        "complete": True,
        "identity_scope": "dataset-scoped",
        "current_family_recomputed": True,
        "source_files": 1,
        "source_inventory": [{"path": str(source), "sha256": _sha(source)}],
        "unique": {
            "dataset_qids": 1,
            "dataset_question_sha256": 1,
            "dataset_current_families": 1,
            "by_dataset": {
                DATASET: {"qids": 1, "question_sha256": 1, "current_families": 1}
            },
        },
        "output": {"rows": 1, "sha256": _sha(ledger)},
        "scientific_boundary": {
            "gold_or_outcome_values_used_for_identity_selection": False,
            "gold_fields_emitted": False,
            "data_raw_modified": False,
            "training_started": False,
        },
    }
    report_path = release / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (release / "manifest.json").write_text(
        json.dumps(
            {
                "status": PROTECTED_LEDGER_STATUS,
                "run": {
                    "report_sha256": _sha(report_path),
                    "protected_identities_sha256": _sha(ledger),
                },
            }
        ),
        encoding="utf-8",
    )
    return release


def test_complete_protected_ledger_recomputes_family_and_binds_all_hashes(tmp_path):
    release = _protected_release(tmp_path)
    ledger, report, manifest, parsed = validate_protected_ledger_release(release)
    assert ledger.name == "protected_identities.question_only.jsonl"
    assert (report.name, manifest.name) == ("report.json", "manifest.json")
    assert parsed["complete"] is True

    row = json.loads(ledger.read_text(encoding="utf-8"))
    row["family_sha256"] = "0" * 64
    ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch|malformed protected ledger row"):
        validate_protected_ledger_release(release)


def test_clean_store_must_bind_same_complete_ledger(tmp_path):
    release = _protected_release(tmp_path)
    ledger, report, manifest, _parsed = validate_protected_ledger_release(release)
    binding = {
        "ledger": {"sha256": _sha(ledger)},
        "report": {"sha256": _sha(report)},
        "manifest": {"sha256": _sha(manifest)},
    }
    store = tmp_path / "store"
    store.mkdir()
    store_manifest = {
        "protected_ledger": {
            "schema_version": PROTECTED_LEDGER_SCHEMA_VERSION,
            "complete": True,
            "current_family_recomputed": True,
            "report_sha256": _sha(report),
            "protected_identities_sha256": _sha(ledger),
            "manifest_sha256": _sha(manifest),
        },
        "scientific_boundary": {
            "selected_question_keys_excluded_exactly": True,
            "selected_current_lexical_families_recomputed_and_excluded": True,
        },
    }
    (store / "store_manifest.json").write_text(
        json.dumps(store_manifest), encoding="utf-8"
    )
    assert validate_clean_store_protected_ledger(
        store, protected_release=binding
    ) == store_manifest

    store_manifest["protected_ledger"]["protected_identities_sha256"] = "0" * 64
    (store / "store_manifest.json").write_text(
        json.dumps(store_manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="newly built store is required"):
        validate_clean_store_protected_ledger(store, protected_release=binding)
