from collections import defaultdict

from scripts.prepare.prepare_saeg_v1_sampling_weights import assign_sampling_probabilities


def test_sampling_probabilities_balance_dataset_then_source():
    candidates = []
    groups = {
        "hotpotqa": {"P_ONLY": 3, "N_REPLAY": 1},
        "2wikimultihopqa": {"P_ONLY": 2, "W_ONLY": 3, "P_W_FUSED": 4, "N_REPLAY": 1},
        "musique": {"P_ONLY": 3, "N_REPLAY": 1},
    }
    for dataset, modes in groups.items():
        for mode, count in modes.items():
            for index in range(count):
                candidates.append({
                    "candidate_id": f"{dataset}::{mode}::{index}",
                    "dataset": dataset,
                    "source_mode": mode,
                })
    rows = assign_sampling_probabilities(candidates)
    by_dataset = defaultdict(float)
    by_group = defaultdict(float)
    for row in rows:
        by_dataset[row["dataset"]] += row["sampling_probability"]
        by_group[(row["dataset"], row["source_mode"])] += row["sampling_probability"]
    assert all(abs(value - 1 / 3) < 1e-9 for value in by_dataset.values())
    assert abs(by_group[("hotpotqa", "P_ONLY")] - 0.30) < 1e-9
    assert abs(by_group[("2wikimultihopqa", "P_W_FUSED")] - (0.40 / 3)) < 1e-9
    assert abs(sum(row["sampling_probability"] for row in rows) - 1.0) < 1e-9
