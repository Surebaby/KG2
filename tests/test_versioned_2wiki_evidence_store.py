import json

from kgproweight.kg.versioned_evidence_store import (
    STORE_SCHEMA_VERSION,
    VersionedEvidenceStore,
    normalize_alias,
)
from scripts.prepare.build_versioned_2wiki_evidence_store import build_store
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256


def test_store_builder_excludes_confirmation_family_and_keeps_literal() -> None:
    rows = [
        {
            "_id": "keep",
            "evidences": [["Alice", "date of birth", "1 January 1900"]],
            "evidences_id": [["Q1", "date of birth", "1 January 1900"]],
        },
        {
            "_id": "drop",
            "evidences": [["Bob", "spouse", "Carol"]],
            "evidences_id": [["Q2", "spouse", "Q3"]],
        },
    ]
    assignments = {
        "2wikimultihopqa::keep": {"split": "train", "family_sha256": "keep-family"},
        "2wikimultihopqa::drop": {"split": "train", "family_sha256": "held-family"},
    }
    aliases, edges, counts = build_store(
        official_rows=rows,
        official_alias_rows=[{"Q_id": "Q1", "aliases": ["A. Person"], "demonyms": []}],
        assignments=assignments,
        excluded_families={"held-family"},
    )
    assert ("Q1", "Alice") in aliases[normalize_alias("Alice")]
    assert ("Q1", "A. Person") in aliases[normalize_alias("A. Person")]
    assert "Q1::P569" in edges
    assert "Q2::P26" not in edges
    assert counts["excluded_confirmation_family_rows"] == 1


def test_store_builder_excludes_exact_question_even_if_family_version_differs() -> None:
    rows = [
        {
            "_id": "selected",
            "question": "Who is Alice's mother?",
            "evidences": [["Alice", "mother", "Carol"]],
            "evidences_id": [["Q1", "mother", "Q3"]],
        }
    ]
    assignments = {
        "2wikimultihopqa::selected": {
            "split": "train",
            "family_sha256": "obsolete-family-version",
        }
    }
    _aliases, edges, counts = build_store(
        official_rows=rows,
        official_alias_rows=[],
        assignments=assignments,
        excluded_families={"current-family-that-does-not-match"},
        excluded_question_keys={"2wikimultihopqa::selected"},
    )
    assert not edges
    assert counts["excluded_selected_question_rows"] == 1


def test_store_builder_recomputes_current_lexical_family() -> None:
    question = "Who is Alice's mother?"
    rows = [
        {
            "_id": "same-template",
            "question": question,
            "evidences": [["Alice", "mother", "Carol"]],
            "evidences_id": [["Q1", "mother", "Q3"]],
        }
    ]
    assignments = {
        "2wikimultihopqa::same-template": {
            "split": "train",
            "family_sha256": "obsolete-family-version",
        }
    }
    _aliases, edges, counts = build_store(
        official_rows=rows,
        official_alias_rows=[],
        assignments=assignments,
        excluded_families=set(),
        excluded_current_families={family_sha256(question)},
    )
    assert not edges
    assert counts["excluded_current_lexical_family_rows"] == 1


def test_read_only_store_resolves_unique_alias_and_fetches_edges(tmp_path) -> None:
    (tmp_path / "store_manifest.json").write_text(
        json.dumps({"schema_version": STORE_SCHEMA_VERSION}), encoding="utf-8"
    )
    (tmp_path / "aliases.jsonl").write_text(
        json.dumps({
            "normalized_alias": normalize_alias("Alice"),
            "candidates": [{"qid": "Q1", "label": "Alice"}],
        }) + "\n",
        encoding="utf-8",
    )
    edge = {
        "head_qid": "Q1", "head_label": "Alice", "pid": "P569",
        "relation": "date of birth", "tail_qid": None, "tail_value": "1900-01-01",
    }
    (tmp_path / "edges.jsonl").write_text(
        json.dumps({"key": "Q1::P569", "edges": [edge]}) + "\n",
        encoding="utf-8",
    )
    store = VersionedEvidenceStore(tmp_path)
    resolved = store.resolve("ALICE")
    assert resolved.selected_qid == "Q1"
    assert not resolved.abstained
    assert store.fetch_edges("q1", ["p569"]) == [edge]


def test_read_only_store_abstains_on_ambiguous_alias(tmp_path) -> None:
    (tmp_path / "store_manifest.json").write_text(
        json.dumps({"schema_version": STORE_SCHEMA_VERSION}), encoding="utf-8"
    )
    (tmp_path / "aliases.jsonl").write_text(
        json.dumps({
            "normalized_alias": "alex",
            "candidates": [
                {"qid": "Q1", "label": "Alex"},
                {"qid": "Q2", "label": "Alex"},
            ],
        }) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "edges.jsonl").write_text("", encoding="utf-8")
    result = VersionedEvidenceStore(tmp_path).resolve("Alex")
    assert result.abstained
    assert "ambiguous" in result.abstain_reason
