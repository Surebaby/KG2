from kgproweight.kg.question_kg import question_key
from scripts.prepare.freeze_automatic_proofkg_training_cohort import (
    QUESTION_TYPES,
    select_training_rows,
)


def _row(index, qtype):
    return {
        "dataset": "2wikimultihopqa",
        "qid": f"q{index}",
        "question": f"Question {index}?",
        "metadata": {"question_type": qtype, "train_only": True},
        # These fields deliberately exist in the source and must not be copied.
        "steps": [{"text": "gold trace"}],
        "answer": "gold",
    }


def test_selector_is_balanced_question_only_and_family_disjoint():
    rows = []
    assignments = {}
    index = 0
    for qtype in QUESTION_TYPES:
        for _ in range(3):
            row = _row(index, qtype)
            rows.append(row)
            key = question_key("2wikimultihopqa", row["qid"])
            assignments[key] = {
                "question_key": key, "split": "train", "family_sha256": f"f{index}"
            }
            index += 1
    excluded_key = question_key("2wikimultihopqa", "q0")
    selected = select_training_rows(
        rows, assignments,
        excluded_keys={excluded_key}, excluded_families={"f1"},
        per_type=1, seed=42,
    )
    assert len(selected) == 4
    assert {row["question_type"] for row in selected} == set(QUESTION_TYPES)
    assert all("steps" not in row and "answer" not in row for row in selected)
    assert all(row["question_key"] != excluded_key for row in selected)
    assert all(row["family_sha256"] != "f1" for row in selected)
