import pytest

from scripts.prepare.freeze_mixed_ppo_three_dataset_v4_proof800 import _identity
from scripts.prepare.freeze_mixed_ppo_v4_hm_reconciliation_v2 import (
    reconcile_retrieval_contexts,
)


def _population(dataset: str, qid: str, serial: int) -> dict:
    return _identity(
        {"id": qid, "question": f"Which marker belongs to Item {serial}?"},
        dataset=dataset,
        stratum="2hop" if dataset == "musique" else "bridge/easy",
        question_type="2hop" if dataset == "musique" else "bridge",
        source_role="new_retrieval",
    )


def _context(row: dict, *, qid: str | None = None) -> dict:
    return {
        "dataset": row["dataset"],
        "qid": qid or row["qid"],
        "question": row["question"],
        "question_sha256": row["question_sha256"],
        "family_sha256": row["family_sha256"],
        "passages_sha256": "a" * 64,
        "passages": [{"id": "1", "contents": "not copied by reconciliation"}],
        "gold_access": False,
    }


def test_reconciliation_emits_only_missing_requests_and_binds_reuse():
    first = _population("hotpotqa", "h1", 1)
    second = _population("musique", "m1", 2)
    retired = _population("musique", "m-old", 3)
    reused, new, removed = reconcile_retrieval_contexts(
        [first, second], [_context(first), _context(retired)]
    )

    assert [row["qid"] for row in reused] == ["h1"]
    assert [row["qid"] for row in new] == ["m1"]
    assert [row["qid"] for row in removed] == ["m-old"]
    assert reused[0]["passages_sha256"] == "a" * 64
    assert "passages" not in reused[0]
    assert new[0]["gold_access"] is False


def test_reconciliation_rejects_context_identity_drift():
    row = _population("hotpotqa", "h1", 1)
    context = _context(row)
    context["question_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="identity drift"):
        reconcile_retrieval_contexts([row], [context])


def test_reconciliation_rejects_duplicate_context_identity():
    row = _population("hotpotqa", "h1", 1)
    with pytest.raises(ValueError, match="invalid/duplicate identity"):
        reconcile_retrieval_contexts([row], [_context(row), _context(row)])
