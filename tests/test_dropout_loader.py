"""D_dropout loader — bug-fix #5 regression tests."""

from __future__ import annotations

import json
from pathlib import Path

from kgproweight.data.d_dropout_loader import DropoutDataset, DropoutItem, load_dropout_dataset


def _item_dict():
    return {
        "qid": "hp_001",
        "id": "hp_001",
        "question": "Who is the spouse of Barack Obama?",
        "answer": "Michelle Obama",
        "golden_answers": ["Michelle Obama"],
        "metadata": {
            "dropout": {
                "original_kg": [
                    ["Barack Obama", "spouse", "Michelle Obama"],
                    ["Michelle Obama", "occupation", "Lawyer"],
                ],
                "modified_kg": [
                    ["Barack Obama", "spouse", "Joe Biden"],
                    ["Michelle Obama", "occupation", "Lawyer"],
                ],
                "severed_triple": ["Barack Obama", "spouse", "Michelle Obama"],
                "replacement": ["Barack Obama", "spouse", "Joe Biden"],
            }
        },
    }


def test_effective_kg_prefers_modified():
    item = DropoutItem.from_dict(_item_dict())
    assert item.modified_kg, "modified_kg must be populated"
    # bug #5 regression: pipeline should see the *severed* triple, not the original.
    assert ("Barack Obama", "spouse", "Joe Biden") in item.effective_kg
    assert ("Barack Obama", "spouse", "Michelle Obama") not in item.effective_kg


def test_effective_kg_falls_back_when_modified_missing():
    payload = _item_dict()
    payload["metadata"]["dropout"]["modified_kg"] = []
    item = DropoutItem.from_dict(payload)
    # Falls back to the original 2-hop subgraph.
    assert ("Barack Obama", "spouse", "Michelle Obama") in item.effective_kg


def test_dataset_round_trip(tmp_path: Path):
    p = tmp_path / "d_dropout.jsonl"
    p.write_text(json.dumps(_item_dict()) + "\n", encoding="utf-8")
    ds = load_dropout_dataset(p)
    assert isinstance(ds, DropoutDataset)
    assert len(ds) == 1
    assert ds[0].qid == "hp_001"


def test_to_flashrag_dataset_preserves_dropout_block(tmp_path: Path):
    p = tmp_path / "d_dropout.jsonl"
    p.write_text(json.dumps(_item_dict()) + "\n", encoding="utf-8")
    ds = load_dropout_dataset(p)
    rows = ds.to_flashrag_dataset()
    assert rows[0]["id"] == "hp_001"
    assert rows[0]["question"].startswith("Who is")
    assert rows[0]["golden_answers"] == ["Michelle Obama"]
    dropout = rows[0]["metadata"]["dropout"]
    assert dropout["modified_kg"][0] == ["Barack Obama", "spouse", "Joe Biden"]
    assert dropout["original_kg"][0] == ["Barack Obama", "spouse", "Michelle Obama"]
