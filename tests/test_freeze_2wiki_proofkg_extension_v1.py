import pytest

from kgproweight.kg.question_kg import question_key, question_sha256
from scripts.prepare.freeze_2wiki_proofkg_extension_v1 import (
    DATASET,
    FORBIDDEN_OUTPUT_FIELDS,
    build_identity_registry,
    select_extension_candidates,
)
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256


def _source(qid: str, question: str, qtype: str = "inference") -> dict:
    return {
        "dataset": DATASET,
        "qid": qid,
        "question": question,
        "answer": "must not leak",
        "steps": [{"text": "must not leak"}],
        "kg_subgraph": [["must", "not", "leak"]],
        "metadata": {"question_type": qtype, "train_only": True},
    }


def _assignment(row: dict, *, split: str = "train") -> dict:
    return {
        "question_key": question_key(DATASET, row["qid"]),
        "dataset": DATASET,
        "qid": row["qid"],
        "split": split,
        "family_sha256": "legacy-assignment-family-" + row["qid"],
    }


def _registry(rows, label="test"):
    return build_identity_registry(rows, label=label)


def test_auto1500_excludes_qid_and_hash_but_allows_current_family_reuse():
    auto = {
        "dataset": DATASET,
        "qid": "old",
        "question": "Who is the mother of Alice?",
        "question_sha256": question_sha256("Who is the mother of Alice?"),
    }
    same_family = _source("new", "Who is the mother of Bob?")
    assert family_sha256(auto["question"]) == family_sha256(same_family["question"])

    selected, telemetry = select_extension_candidates(
        [same_family],
        [_assignment(same_family)],
        auto_train_registry=_registry([auto], "auto"),
        protected_registry=_registry([], "protected"),
        quotas={"inference": 1},
        seed=42,
    )
    assert [row["qid"] for row in selected] == ["new"]
    assert telemetry["selected_rows_reusing_auto1500_current_family"] == 1


def test_protected_family_excludes_different_qid_and_exact_question():
    protected = {
        "dataset": DATASET,
        "qid": "eval-qid",
        "question": "Who is the mother of Alice?",
    }
    same_family = _source("candidate-family", "Who is the mother of Bob?")
    allowed = _source("candidate-safe", "When was the River Example founded?")
    assignments = [_assignment(same_family), _assignment(allowed)]

    selected, telemetry = select_extension_candidates(
        [same_family, allowed],
        assignments,
        auto_train_registry=_registry([], "auto"),
        protected_registry=_registry([protected], "protected"),
        quotas={"inference": 1},
        seed=42,
    )
    assert [row["qid"] for row in selected] == ["candidate-safe"]
    assert telemetry["exclusion_counts_first_match"]["protected_family"] == 1


def test_projection_is_question_only_and_deterministic():
    source = [
        _source("q1", "When was Example One founded?"),
        _source("q2", "Where was Example Two born?"),
        _source("q3", "Who directed Example Three?"),
    ]
    assignments = [_assignment(row) for row in source]
    kwargs = {
        "auto_train_registry": _registry([], "auto"),
        "protected_registry": _registry([], "protected"),
        "quotas": {"inference": 2},
        "seed": 7,
    }
    first, first_stats = select_extension_candidates(source, assignments, **kwargs)
    second, second_stats = select_extension_candidates(
        list(reversed(source)), list(reversed(assignments)), **kwargs
    )
    assert first == second
    assert first_stats == second_stats
    assert all(not FORBIDDEN_OUTPUT_FIELDS.intersection(row) for row in first)
    assert all(row["gold_access"] is False for row in first)


def test_non_train_assignment_fails_closed():
    source = _source("q1", "When was Example One founded?")
    with pytest.raises(ValueError, match="invalid/non-train planner assignment"):
        select_extension_candidates(
            [source],
            [_assignment(source, split="dev")],
            auto_train_registry=_registry([], "auto"),
            protected_registry=_registry([], "protected"),
            quotas={"inference": 1},
        )


def test_insufficient_quota_fails_closed():
    source = _source("q1", "When was Example One founded?")
    with pytest.raises(ValueError, match="need 2"):
        select_extension_candidates(
            [source],
            [_assignment(source)],
            auto_train_registry=_registry([], "auto"),
            protected_registry=_registry([], "protected"),
            quotas={"inference": 2},
        )
