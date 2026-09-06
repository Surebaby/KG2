import pytest

from scripts.prepare import freeze_mixed_ppo_three_dataset_v4_proof800 as v4
from scripts.prepare.freeze_2wiki_ordinary200_full_ledger_v2 import (
    OUTPUT_FIELDS,
    select_ordinary200,
)


def _full(qid: str, question: str, qtype: str = "comparison") -> dict:
    return {
        "dataset": "2wikimultihopqa",
        "qid": qid,
        "question": question,
        "answer": "source outcome must not be copied",
        "retrieved_passages": [{"id": "p1", "contents": "source passage"}],
        "accepted": True,
        "metadata": {
            "question_type": qtype,
            "train_only": True,
            "source_split": "train",
        },
    }


def _identity(row: dict) -> dict:
    return v4._identity(row, dataset="2wikimultihopqa")


def test_retains_safe_parent_and_replaces_blocked_parent_with_traceable_source():
    parent_safe = _full("parent-safe", "When was Example One founded?")
    parent_blocked = _full("parent-blocked", "Who is the mother of Alice?")
    replacement = _full("replacement", "Where was Example Two born?")
    selected, stats = select_ordinary200(
        parent_rows=[_identity(parent_safe), _identity(parent_blocked)],
        parent_source_rows=[(10, parent_safe), (11, parent_blocked)],
        replacement_source_rows=[(20, replacement)],
        protected_rows=[
            _identity(_full("protected-alias", "Who is the mother of Bob?"))
        ],
        replay_rows=[],
        proof_identity_rows=[],
        n=2,
    )

    assert {row["qid"] for row in selected} == {"parent-safe", "replacement"}
    assert stats["retained_parent"] == 1
    assert stats["removed_parent"] == 1
    assert stats["replacements_selected"] == 1
    assert all(set(row) == OUTPUT_FIELDS for row in selected)
    assert all("answer" not in row and "retrieved_passages" not in row for row in selected)
    replacement_row = next(row for row in selected if row["qid"] == "replacement")
    assert replacement_row["source_origin"] == "proofkg_curriculum_mix_v1"
    assert replacement_row["source_line_number"] == 20
    assert len(replacement_row["source_record_sha256"]) == 64
    assert len(replacement_row["source_passages_sha256"]) == 64


def test_proof_candidate_family_is_blocked_for_parent_and_replacement():
    parent = _full("parent", "Who is the mother of Alice?")
    same_proof_family = _full("candidate-blocked", "Who is the mother of Bob?")
    safe = _full("candidate-safe", "When was Example Three founded?")
    proof = _identity(_full("proof", "Who is the mother of Carol?"))

    selected, stats = select_ordinary200(
        parent_rows=[_identity(parent)],
        parent_source_rows=[(1, parent)],
        replacement_source_rows=[(2, same_proof_family), (3, safe)],
        protected_rows=[],
        replay_rows=[],
        proof_identity_rows=[proof],
        n=1,
    )

    assert [row["qid"] for row in selected] == ["candidate-safe"]
    assert stats["removed_parent"] == 1
    assert stats["replacement_exclusion_counts_first_match"][
        "blocked_family_sha256"
    ] == 1


def test_refill_is_deterministic_under_source_order():
    blocked_parent = _full("parent", "Who is the mother of Alice?")
    candidates = [
        _full("c1", "When was Example One founded?"),
        _full("c2", "Where was Example Two born?"),
        _full("c3", "Who directed Example Three?"),
    ]
    kwargs = dict(
        parent_rows=[_identity(blocked_parent)],
        parent_source_rows=[(1, blocked_parent)],
        protected_rows=[_identity(blocked_parent)],
        replay_rows=[],
        proof_identity_rows=[],
        n=1,
    )
    first, _ = select_ordinary200(
        replacement_source_rows=list(enumerate(candidates, start=2)), **kwargs
    )
    second, _ = select_ordinary200(
        replacement_source_rows=list(reversed(list(enumerate(candidates, start=2)))),
        **kwargs,
    )
    assert first == second


def test_missing_materialized_parent_source_fails_closed():
    parent = _full("parent", "When was Example One founded?")
    with pytest.raises(ValueError, match="source join miss"):
        select_ordinary200(
            parent_rows=[_identity(parent)],
            parent_source_rows=[],
            replacement_source_rows=[],
            protected_rows=[],
            replay_rows=[],
            proof_identity_rows=[],
            n=1,
        )


def test_replacement_without_outcome_or_passages_fails_closed():
    parent = _full("parent", "Who is the mother of Alice?")
    bad = _full("bad", "When was Example Four founded?")
    bad["answer"] = ""
    with pytest.raises(ValueError, match="lacks outcome"):
        select_ordinary200(
            parent_rows=[_identity(parent)],
            parent_source_rows=[(1, parent)],
            replacement_source_rows=[(2, bad)],
            protected_rows=[_identity(parent)],
            replay_rows=[],
            proof_identity_rows=[],
            n=1,
        )
