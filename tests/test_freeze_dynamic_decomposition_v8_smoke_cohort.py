from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare import freeze_dynamic_decomposition_v8_smoke_cohort as freeze


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_selector_projects_only_identity_and_takes_four_per_dataset(tmp_path):
    rows = []
    for dataset in freeze.DATASETS:
        for index in range(6):
            rows.append(
                {
                    "dataset": dataset,
                    "qid": f"{dataset}-{index}",
                    "question": f"Question {dataset} {index}?",
                    "gold_answer": "MUST_NOT_APPEAR",
                    "supporting_facts": ["MUST_NOT_APPEAR"],
                }
            )
    source = tmp_path / "source.jsonl"
    _write(source, rows)
    selected = freeze.select_consumed_smoke_rows(source)
    assert len(selected) == 12
    assert all(tuple(row) == freeze.FIELDS for row in selected)
    assert {dataset: sum(row["dataset"] == dataset for row in selected) for dataset in freeze.DATASETS} == {
        dataset: 4 for dataset in freeze.DATASETS
    }
    assert "MUST_NOT_APPEAR" not in json.dumps(selected)


def test_selector_fails_closed_on_insufficient_dataset(tmp_path):
    source = tmp_path / "source.jsonl"
    _write(
        source,
        [
            {"dataset": "hotpotqa", "qid": f"h-{index}", "question": "Question?"}
            for index in range(4)
        ],
    )
    with pytest.raises(ValueError, match="count mismatch"):
        freeze.select_consumed_smoke_rows(source)
