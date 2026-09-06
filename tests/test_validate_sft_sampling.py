from types import SimpleNamespace

import json

import pytest

from scripts.eval.validate_sft import (
    apply_input_overrides,
    sample_trajectories,
    select_fixed_trajectories,
    validate_split_protocol,
)


def _items(n=50):
    return [SimpleNamespace(qid=f"q{i:03d}", question=f"question {i}") for i in range(n)]


def test_sft_validation_sampling_is_seeded_and_order_independent():
    items = _items()
    a = [x.qid for x in sample_trajectories(items, 12, seed=42)]
    b = [x.qid for x in sample_trajectories(list(reversed(items)), 12, seed=42)]
    c = [x.qid for x in sample_trajectories(items, 12, seed=7)]
    assert a == b
    assert a != c
    assert len(set(a)) == 12


def test_fixed_selection_preserves_requested_order(tmp_path):
    items = _items(4)
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        "\n".join(json.dumps({"qid": qid}) for qid in ["q003", "q001"]) + "\n",
        encoding="utf-8",
    )
    chosen = select_fixed_trajectories(items, str(selection))
    assert [item.qid for item in chosen] == ["q003", "q001"]


def test_input_override_changes_only_explicit_fields(tmp_path):
    item = SimpleNamespace(
        qid="q001",
        question="original question",
        retrieved_passages=[{"id": "old"}],
        kg_subgraph=[("h", "r", "t")],
    )
    overrides = tmp_path / "overrides.jsonl"
    overrides.write_text(
        json.dumps({"qid": "q001", "retrieved_passages": [{"id": "new"}]}) + "\n",
        encoding="utf-8",
    )
    apply_input_overrides([item], str(overrides))
    assert item.retrieved_passages == [{"id": "new"}]
    assert item.kg_subgraph == [("h", "r", "t")]
    assert item.question == "original question"

    missing = tmp_path / "missing.jsonl"
    missing.write_text(json.dumps({"qid": "q999", "retrieved_passages": []}) + "\n")
    with pytest.raises(ValueError, match="missing selected qids"):
        apply_input_overrides([item], str(missing))


def test_train_diagnostic_requires_explicit_flag_and_fixed_selection():
    with pytest.raises(ValueError, match="allow_train_diagnostic"):
        validate_split_protocol(
            "train", allow_train_diagnostic=False, selection_jsonl="cohort.jsonl"
        )
    with pytest.raises(ValueError, match="selection_jsonl"):
        validate_split_protocol(
            "train", allow_train_diagnostic=True, selection_jsonl=None
        )
    validate_split_protocol(
        "train", allow_train_diagnostic=True, selection_jsonl="cohort.jsonl"
    )
    with pytest.raises(ValueError, match="only with --split train"):
        validate_split_protocol(
            "val", allow_train_diagnostic=True, selection_jsonl="cohort.jsonl"
        )
