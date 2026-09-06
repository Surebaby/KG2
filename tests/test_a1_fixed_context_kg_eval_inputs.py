import pytest

from scripts.prepare.build_a1_fixed_context_kg_eval_inputs import (
    context_to_passages,
    non_kg_projection,
)


def test_context_to_passages_preserves_dataset_order():
    metadata = {
        "context": {
            "title": ["First", "Second"],
            "content": [["A.", "B."], ["C."]],
        }
    }

    passages = context_to_passages(metadata)

    assert [row["id"] for row in passages] == ["frozen_context_1", "frozen_context_2"]
    assert passages[0]["contents"] == "First\nA. B."
    assert passages[1]["contents"] == "Second\nC."


def test_context_to_passages_rejects_misaligned_context():
    with pytest.raises(ValueError, match="lengths differ"):
        context_to_passages({"context": {"title": ["Only"], "content": []}})


def test_non_kg_projection_supports_strict_paired_arm_check():
    common = {"qid": "q1", "question": "Q?", "retrieved_passages": [{"contents": "P"}]}
    left = {**common, "kg_subgraph": [["a", "r", "b"]]}
    right = {**common, "kg_subgraph": [["x", "r", "y"]]}

    assert non_kg_projection(left) == non_kg_projection(right)
