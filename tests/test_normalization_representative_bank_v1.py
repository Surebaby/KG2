"""Identity-only selection tests; no models, rewards or outcome labels."""
from copy import deepcopy
from collections import Counter

from scripts.prepare import normalization_representative_bank_v1 as supplement


def population():
    groups, old, assignments = [], [], []
    for dataset, graph, total, existing in (("hotpotqa", 0, 55, 7), ("musique", 0, 55, 4),
                                           ("2wikimultihopqa", 1, 50, 40), ("2wikimultihopqa", 0, 20, 4)):
        for index in range(total):
            word = chr(97 + index // 26) + chr(97 + index % 26)
            item = {"dataset": dataset, "qid": f"{dataset}-{graph}-{index}",
                    "question": f"Where does synthetic {dataset} graph{graph} token{word} lead?"}
            item.update(supplement.bank.row_identity(item))
            item.update(question_key=supplement.bank.key(item), process_reward_eligible=bool(graph))
            groups.append(item)
            if index < existing:
                old.append(deepcopy(item))
                assignments.append({**item, "split": "train"})
    return groups, old, assignments


def test_exact_quotas_reuse_first_and_hash_order_independent_of_quality_fields():
    groups, old, assignments = population()
    selected, summary = supplement.select(groups, old, assignments, [])
    assert Counter((row["dataset"], row["process_reward_eligible"]) for row in selected) == {
        ("hotpotqa", False): 40, ("musique", False): 40,
        ("2wikimultihopqa", True): 32, ("2wikimultihopqa", False): 8}
    assert summary["reused_questions"] == 47 and summary["new_questions"] == 73
    changed = deepcopy(groups)
    for index, row in enumerate(changed):
        row.update(em=index % 2, trajectory_valid=bool(index % 3), reward=1000 - index)
    new, _ = supplement.select(list(reversed(changed)), list(reversed(old)), list(reversed(assignments)), [])
    assert [(row["question_key"], row["reuse"]) for row in new] == [(row["question_key"], row["reuse"]) for row in selected]


def test_consumed_families_and_protected_identities_are_excluded():
    groups, old, assignments = population()
    consumed = next(row for row in groups if row["dataset"] == "hotpotqa")
    assignments = [item for item in assignments if item["qid"] != consumed["qid"]]
    assignments.append({**consumed, "split": "confirmation"})
    protected = [next(row for row in groups if row["dataset"] == "musique")]
    selected, summary = supplement.select(groups, old, assignments, protected)
    assert consumed["family_sha256"] not in {row["family_sha256"] for row in selected}
    assert protected[0]["qid"] not in {row["qid"] for row in selected}
    assert len({row["family_sha256"] for row in selected}) == len(selected) == 120
    assert summary["consumed_family_count_excluded"] == 1
