import pytest

from scripts.prepare.freeze_saeg_v1_ppo_pool import interleave_datasets, select_family_unique


def test_dataset_interleave_has_balanced_prefix():
    rows = {
        dataset: [{"dataset": dataset, "qid": f"{dataset}-{i}"} for i in range(3)]
        for dataset in ("hotpotqa", "2wikimultihopqa", "musique")
    }
    output = interleave_datasets(rows)
    assert [row["dataset"] for row in output[:3]] == ["hotpotqa", "2wikimultihopqa", "musique"]
    assert [row["schedule_index"] for row in output] == list(range(9))


def test_dataset_interleave_rejects_imbalance():
    with pytest.raises(ValueError, match="not balanced"):
        interleave_datasets({
            "hotpotqa": [{"qid": "1"}],
            "2wikimultihopqa": [],
            "musique": [{"qid": "1"}],
        })


def test_family_unique_selection_respects_and_updates_blocklist():
    rows = [
        {"dataset": "hotpotqa", "qid": "a", "family_sha256": "blocked"},
        {"dataset": "hotpotqa", "qid": "b", "family_sha256": "f1"},
        {"dataset": "hotpotqa", "qid": "c", "family_sha256": "f1"},
        {"dataset": "hotpotqa", "qid": "d", "family_sha256": "f2"},
    ]
    blocked = {"blocked"}
    selected = select_family_unique(rows, 2, label="test", blocked_families=blocked)
    assert {row["family_sha256"] for row in selected} == {"f1", "f2"}
    assert blocked == {"blocked", "f1", "f2"}
