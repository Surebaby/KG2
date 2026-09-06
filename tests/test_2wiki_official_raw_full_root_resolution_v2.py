from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.prepare.freeze_2wiki_official_raw_full_root_resolution_v2 import (
    WORKLIST_FIELDS,
    build_all_root_worklist,
)
from scripts.prepare.materialize_2wiki_full_root_resolution_v2 import (
    DRY_RUN_FIELDS,
    exact_consumer_dry_run,
)


def _hash(question: str) -> str:
    return hashlib.sha256(question.strip().encode()).hexdigest()


def _identity(qid: str, question: str) -> dict:
    return {
        "dataset": "2wikimultihopqa",
        "qid": qid,
        "question_key": f"2wikimultihopqa::{qid}",
        "question": question,
        "question_sha256": _hash(question),
        "gold_access": False,
    }


def _cohort(qid: str, question: str, qtype: str = "comparison") -> dict:
    return {**_identity(qid, question), "question_type": qtype}


def _plan(qid: str, question: str, anchors: list[str]) -> dict:
    return {
        **_identity(qid, question),
        "predicted_target": {
            "anchors": anchors,
            "steps": [
                {
                    "step": index,
                    "subject": anchor,
                    "relation_label": "publication date",
                    "pid": "P577",
                    "output_slot": f"hop_{index}",
                    "dependencies": [],
                }
                for index, anchor in enumerate(anchors, start=1)
            ],
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _store(path: Path) -> None:
    path.mkdir()
    (path / "store_manifest.json").write_text(
        json.dumps({"schema_version": "versioned-2wiki-evidence-store-1"}),
        encoding="utf-8",
    )
    _write_jsonl(
        path / "aliases.jsonl",
        [
            {
                "normalized_alias": "beta",
                "candidates": [{"qid": "Q2", "label": "Beta", "evidence_count": 1}],
            }
        ],
    )
    (path / "edges.jsonl").write_text("", encoding="utf-8")


def test_freezer_projects_every_root_and_never_reads_old_resolution_state():
    q1 = "Were Alpha and Beta released in the same year?"
    q2 = "Who directed Gamma?"
    worklist, counts = build_all_root_worklist(
        [_cohort("a", q1), _cohort("b", q2, "inference")],
        [_plan("a", q1, ["Alpha", "Beta"]), _plan("b", q2, ["Gamma"])],
    )
    assert len(worklist) == 3
    assert counts["questions_total"] == 2
    assert counts["questions_recognized"] == 2
    assert counts["root_anchor_occurrences"] == 3
    assert all(set(row) == WORKLIST_FIELDS for row in worklist)
    assert [row["root_position"] for row in worklist if row["qid"] == "a"] == [1, 2]
    assert "resolved_qid" not in json.dumps(worklist)


def test_freezer_retains_unrecognized_question_in_denominator_without_inventing_root():
    q1 = "Who directed Alpha?"
    q2 = "Broken planner output"
    good = _plan("a", q1, ["Alpha"])
    bad = {**_identity("b", q2), "predicted_target": None}
    worklist, counts = build_all_root_worklist(
        [_cohort("a", q1), _cohort("b", q2)], [good, bad]
    )
    assert len(worklist) == 1
    assert counts["questions_total"] == 2
    assert counts["questions_recognized"] == 1
    assert counts["questions_unrecognized"] == 1


def test_exact_consumer_dry_run_uses_title_then_v6_alias_then_entity(tmp_path: Path):
    store = tmp_path / "store"
    _store(store)
    title_cache = tmp_path / "title.jsonl"
    entity_cache = tmp_path / "entity.jsonl"
    _write_jsonl(title_cache, [{"label": "Alpha", "qid": "Q1"}])
    _write_jsonl(entity_cache, [{"label": "Gamma", "qid": "Q3"}])
    roots = []
    results = []
    for position, (surface, qid) in enumerate(
        [("Alpha", "Q1"), ("Beta", "Q2"), ("Gamma", "Q3")], start=1
    ):
        request_id = f"r{position}"
        roots.append(
            {
                "request_id": request_id,
                "question_key": "2wikimultihopqa::x",
                "dataset": "2wikimultihopqa",
                "qid": "x",
                "question": "Compare Alpha, Beta, and Gamma.",
                "question_sha256": _hash("Compare Alpha, Beta, and Gamma."),
                "root_position": position,
                "root_anchor_surface": surface,
                "completed_root_anchor_surface": surface,
            }
        )
        results.append({"request_id": request_id, "outcome": "positive", "resolved_qid": qid})
    rows, stats = exact_consumer_dry_run(
        worklist=roots,
        results=results,
        title_cache_path=title_cache,
        entity_cache_path=entity_cache,
        v6_store_dir=store,
    )
    assert [row["resolution_source"] for row in rows] == [
        "new_exact_title_cache",
        "clean_v6_exact_alias",
        "new_exact_entity_cache",
    ]
    assert all(set(row) == DRY_RUN_FIELDS for row in rows)
    assert stats["resolved_anchor_occurrences"] == 3
    assert stats["all_roots_resolved_questions"] == 1
    assert stats["occurrence_mismatches"] == 0


def test_exact_consumer_dry_run_exposes_projection_mismatch(tmp_path: Path):
    store = tmp_path / "store"
    _store(store)
    title_cache = tmp_path / "title.jsonl"
    entity_cache = tmp_path / "entity.jsonl"
    title_cache.write_text("", encoding="utf-8")
    entity_cache.write_text("", encoding="utf-8")
    question = "What is Beta?"
    request = {
        "request_id": "r1",
        "question_key": "2wikimultihopqa::x",
        "dataset": "2wikimultihopqa",
        "qid": "x",
        "question": question,
        "question_sha256": _hash(question),
        "root_position": 1,
        "root_anchor_surface": "Beta",
        "completed_root_anchor_surface": "Beta",
    }
    rows, stats = exact_consumer_dry_run(
        worklist=[request],
        results=[{"request_id": "r1", "outcome": "positive", "resolved_qid": "Q999"}],
        title_cache_path=title_cache,
        entity_cache_path=entity_cache,
        v6_store_dir=store,
    )
    assert rows[0]["dry_run_qid"] == "Q2"
    assert rows[0]["projected_qid"] == "Q999"
    assert rows[0]["matched"] is False
    assert stats["occurrence_mismatches"] == 1
