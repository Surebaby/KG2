from collections import Counter

from scripts.prepare.freeze_saeg_v1_sft_epoch import (
    build_epoch,
    coverage_first_sample,
    exact_group_quotas,
)


def _rows():
    groups = {
        ("hotpotqa", "P_ONLY"): 3,
        ("hotpotqa", "N_REPLAY"): 2,
        ("2wikimultihopqa", "P_ONLY"): 2,
        ("2wikimultihopqa", "W_ONLY"): 2,
        ("2wikimultihopqa", "P_W_FUSED"): 2,
        ("2wikimultihopqa", "N_REPLAY"): 2,
        ("musique", "P_ONLY"): 3,
        ("musique", "N_REPLAY"): 2,
    }
    return [
        {
            "qid": f"{dataset}::{mode}::{index}",
            "source_qid": f"q-{index}",
            "dataset": dataset,
            "evidence_mode": mode,
            "accepted": True,
            "answer": "unchanged",
            "metadata": {},
        }
        for (dataset, mode), size in groups.items()
        for index in range(size)
    ]


def test_4860_quotas_are_exact_and_dataset_balanced():
    quotas = exact_group_quotas(4860)
    assert sum(quotas.values()) == 4860
    assert quotas[("hotpotqa", "P_ONLY")] == 1458
    assert quotas[("hotpotqa", "N_REPLAY")] == 162
    assert quotas[("2wikimultihopqa", "P_ONLY")] == 324
    assert quotas[("2wikimultihopqa", "W_ONLY")] == 486
    assert quotas[("2wikimultihopqa", "P_W_FUSED")] == 648
    assert quotas[("2wikimultihopqa", "N_REPLAY")] == 162
    assert quotas[("musique", "P_ONLY")] == 1458
    assert quotas[("musique", "N_REPLAY")] == 162


def test_coverage_first_sampling_does_not_repeat_before_full_coverage():
    rows = [{"qid": f"q{index}"} for index in range(4)]
    sampled = coverage_first_sample(
        rows, 6, seed=42, key=("hotpotqa", "P_ONLY")
    )
    assert len({row["qid"] for row in sampled[:4]}) == 4
    assert len(sampled) == 6


def test_epoch_is_deterministic_and_preserves_targets():
    rows = _rows()
    first, quotas = build_epoch(rows, epoch_size=30, seed=42)
    second, _ = build_epoch(rows, epoch_size=30, seed=42)
    assert [row["qid"] for row in first] == [row["qid"] for row in second]
    realised = Counter((row["dataset"], row["evidence_mode"]) for row in first)
    assert dict(realised) == quotas
    assert all(row["answer"] == "unchanged" for row in first)
    assert [row["metadata"]["sft_epoch_sample_index"] for row in first] == list(range(30))
