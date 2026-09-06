import json

from kgproweight.training.phase1_distill import StratifiedSilverFilter
from scripts.utils.repair_legacy_silver_v2 import (
    _finalize_per_dataset,
    _sample_selected,
)


def _row(dataset: str, qid: str, bucket: str, quality_pass: bool = True):
    return {
        "qid": qid,
        "dataset": dataset,
        "accepted": False,
        "metadata": {
            "quality_pass": quality_pass,
            "quality_reject_reason": "" if quality_pass else "bad",
            "kg_bucket": bucket,
            "selection_pass": False,
        },
    }


def test_finalize_legacy_quotas_are_applied_per_dataset(tmp_path):
    candidate = tmp_path / "candidate.jsonl"
    output = tmp_path / "selected.jsonl"
    rows = []
    for dataset in ("a", "b"):
        for bucket in ("kg_rich", "kg_medium", "kg_sparse"):
            rows.extend(
                _row(dataset, f"{dataset}-{bucket}-{idx}", bucket)
                for idx in range(4)
            )
        rows.append(_row(dataset, f"{dataset}-rejected", "kg_sparse", False))
    candidate.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = _finalize_per_dataset(
        candidate,
        output,
        StratifiedSilverFilter(),
        seed=42,
        selection_metadata={"selection_protocol_version": "test"},
    )

    assert report["total"] == 26
    assert report["accepted"] == 18
    assert report["per_dataset"]["a"]["accepted_bucket_counts"] == {
        "kg_rich": 4,
        "kg_medium": 3,
        "kg_sparse": 2,
    }
    assert report["per_dataset"]["b"]["accepted_bucket_counts"] == {
        "kg_rich": 4,
        "kg_medium": 3,
        "kg_sparse": 2,
    }
    selected = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(
        row["metadata"]["selection_protocol_version"] == "test"
        for row in selected
    )
    for dataset in ("a", "b"):
        accepted = [row for row in selected if row["dataset"] == dataset and row["accepted"]]
        assert len(accepted) == 9
        assert sum(row["metadata"]["kg_bucket"] == "kg_rich" for row in accepted) == 4
        assert sum(row["metadata"]["kg_bucket"] == "kg_medium" for row in accepted) == 3
        assert sum(row["metadata"]["kg_bucket"] == "kg_sparse" for row in accepted) == 2


def test_legacy_hash_sample_is_deterministic_and_partitions():
    qid = "train_123"
    hits = [_sample_selected(qid, 25, remainder) for remainder in range(25)]
    assert sum(hits) == 1
    assert hits == [_sample_selected(qid, 25, remainder) for remainder in range(25)]
    assert _sample_selected(qid, 0, 0)
