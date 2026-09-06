from pathlib import Path

import pytest

from kgproweight.kg.question_kg import question_sha256
from scripts.prepare.freeze_mixed_ppo_three_dataset_v4_proof800 import (
    DEFAULT_PROTECTED,
)
from scripts.prepare.freeze_mixed_ppo_v4_protected_ledger import (
    DEFAULT_OUT,
    SOURCE_SPECS,
    _identity,
)
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256


def test_v4_freezer_defaults_to_single_versioned_complete_ledger() -> None:
    assert DEFAULT_PROTECTED == (
        DEFAULT_OUT / "protected_identities.question_only.jsonl",
    )


def test_ledger_recomputes_non_authoritative_historical_family() -> None:
    row = {
        "dataset": "2wikimultihopqa",
        "qid": "q1",
        "question": "Who is Ada's mother?",
        "question_sha256": question_sha256("Who is Ada's mother?"),
        "family_sha256": "stale-historical-family",
    }
    result = _identity(row)
    assert result["family_sha256"] == family_sha256(row["question"])
    assert result["supplied_family_sha256"] == "stale-historical-family"


def test_ledger_rejects_question_hash_drift() -> None:
    with pytest.raises(ValueError, match="question hash mismatch"):
        _identity({
            "dataset": "hotpotqa",
            "qid": "train_1",
            "question": "A question?",
            "question_sha256": "bad",
        })


def test_every_declared_source_exists_and_has_a_role() -> None:
    assert len(SOURCE_SPECS) == len({path for path, _, _ in SOURCE_SPECS})
    for path, role, _ in SOURCE_SPECS:
        assert Path(path).is_file()
        assert role
