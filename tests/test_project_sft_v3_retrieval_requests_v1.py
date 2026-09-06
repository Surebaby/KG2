import pytest

from scripts.prepare import project_sft_v3_retrieval_requests_v1 as projection
from scripts.prepare import freeze_sft_v3_protected_ledger_v1 as ledger


def row():
    identity = ledger.identity({"dataset": "hotpotqa", "qid": "a", "question": "who wrote Silver Lake?"})
    return {**identity, "question_key": "hotpotqa::a", "role": "sft_v3_train_candidate_reserve",
            "gold_access": False, "split": "train", "selection_rank": "ORIGINAL_HASH_RANK",
            "within_split_dataset_rank": 2, "schema_version": "parent-v1",
            "source": {"path": "MUST_STAY_IN_PARENT"}, "teacher_acceptance_pending": True}


def test_strict_projection_keeps_identity_but_uses_ordinal_not_hash_rank():
    parent = row()
    result = projection.project(parent)
    assert set(result) == projection.FIELDS
    assert len(result) == 12
    assert result["selection_rank"] == 2
    assert parent["selection_rank"] == "ORIGINAL_HASH_RANK"
    assert "source" not in result
    assert "teacher_acceptance_pending" not in result


@pytest.mark.parametrize("rank", [True, 0, -1, "2"])
def test_projection_refuses_ambiguous_ordinal_rank(rank):
    parent = row()
    parent["within_split_dataset_rank"] = rank
    with pytest.raises(ValueError):
        projection.project(parent)
