from scripts.pilot.freeze_query_aware_2wiki_confirmation import select_rows


def test_confirmation_selection_is_deterministic_stratified_and_disjoint():
    quotas = {"comparison": 2, "inference": 1}
    rows = []
    for stratum, count in (("comparison", 5), ("inference", 4)):
        for index in range(count):
            rows.append(
                {
                    "id": f"{stratum}_{index}",
                    "question": f"q {stratum} {index}",
                    "golden_answers": ["gold"],
                    "metadata": {"type": stratum},
                }
            )
    first = select_rows(rows, excluded={"comparison_0"}, seed=42, quotas=quotas)
    second = select_rows(reversed(rows), excluded={"comparison_0"}, seed=42, quotas=quotas)
    assert first == second
    assert len(first) == 3
    assert {row["source_id"] for row in first}.isdisjoint({"comparison_0"})
    assert sum(row["stratum"] == "comparison" for row in first) == 2
