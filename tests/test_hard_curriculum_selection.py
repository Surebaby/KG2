from collections import defaultdict

from scripts.prepare.freeze_2wiki_hard_curriculum_v1 import select_candidate_strata


def select_stratum(rows):
    by_qid = defaultdict(list)
    for row in rows:
        by_qid[row["qid"]].append(row)
    return select_candidate_strata(by_qid)


def test_requires_contrastive_sampled_outcomes():
    rows = [{"qid": "q1", "candidate_type": "greedy", "em": 0}]
    rows += [{"qid": "q1", "candidate_type": "sampled", "em": 0} for _ in range(4)]
    assert select_stratum(rows) == {}


def test_recovery_and_stability_are_determined_only_by_greedy():
    sampled = [
        {"candidate_type": "sampled", "em": value}
        for value in (0, 1, 0, 1)
    ]
    rows = []
    for qid, greedy in (("recover", 0), ("retain", 1)):
        rows.append({"qid": qid, "candidate_type": "greedy", "em": greedy})
        rows.extend({**row, "qid": qid} for row in sampled)
    assert select_stratum(rows) == {"recover": "recovery", "retain": "stability"}


def test_module_compiles_and_boolean_literal_is_python_value():
    # Regression for the first freeze attempt, which used JSON's lowercase
    # ``false`` inside a Python dict and failed before writing the protocol.
    import scripts.prepare.freeze_2wiki_hard_curriculum_v1 as module

    assert module.__name__.endswith("freeze_2wiki_hard_curriculum_v1")
