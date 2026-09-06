from __future__ import annotations

from collections import Counter

import pytest

from kgproweight.kg.question_kg import validate_question_kg_record
from scripts.prepare.freeze_mixed_ppo_three_dataset_v1 import (
    build_prompt_group_schedule,
    build_sampling_weights,
    expand_k4_schedule,
    select_ordinary_2wiki,
)
from scripts.prepare.materialize_mixed_ppo_three_dataset_v1 import (
    build_silver_row,
    make_outcome_only_kg_record,
)


def _row(dataset: str, qid: str, route: str, eligible: bool = False):
    return {
        "question_key": f"{dataset}::{qid}",
        "dataset": dataset,
        "qid": qid,
        "question": f"Question {qid}?",
        "question_sha256": f"hash-{qid}",
        "family_sha256": f"family-{qid}",
        "route": route,
        "process_reward_eligible": eligible,
    }


def _population():
    return [
        *(_row("hotpotqa", f"h{i}", "hotpotqa_outcome") for i in range(600)),
        *(_row("musique", f"m{i}", "musique_outcome") for i in range(599)),
        *(_row("2wikimultihopqa", f"o{i}", "2wiki_ordinary_outcome") for i in range(392)),
        *(_row("2wikimultihopqa", f"r{i}", "2wiki_hard_recovery", True) for i in range(25)),
        *(_row("2wikimultihopqa", f"s{i}", "2wiki_hard_stability", True) for i in range(183)),
    ]


def test_fixed_schedule_is_balanced_k4_and_has_declared_exposures():
    groups = build_prompt_group_schedule(_population())
    schedule = expand_k4_schedule(groups)
    assert len(groups) == 1800
    assert len(schedule) == 7200
    assert Counter(row["dataset"] for row in groups) == Counter({
        "hotpotqa": 600, "2wikimultihopqa": 600, "musique": 600,
    })
    assert Counter(row["route"] for row in groups if row["dataset"] == "2wikimultihopqa") == Counter({
        "2wiki_ordinary_outcome": 300,
        "2wiki_hard_recovery": 150,
        "2wiki_hard_stability": 150,
    })
    assert Counter(row["dataset"] for row in groups[:300]) == Counter({
        "hotpotqa": 100, "2wikimultihopqa": 100, "musique": 100,
    })
    assert all(
        len({(row["dataset"], row["qid"]) for row in schedule[start:start + 4]}) == 1
        for start in range(0, len(schedule), 4)
    )
    assert sum(row["process_reward_eligible"] for row in schedule) == 1200


def test_sampling_weights_sum_to_one_and_match_target_stratum_mass():
    weights = build_sampling_weights(_population())
    assert len(weights) == 1799
    assert abs(sum(row["sampling_probability"] for row in weights) - 1.0) < 1e-12
    mass = Counter()
    for row in weights:
        mass[row["stratum"]] += row["sampling_probability"]
    assert mass["hotpotqa_outcome"] == pytest.approx(1.0 / 3.0)
    assert mass["musique_outcome"] == pytest.approx(1.0 / 3.0)
    assert mass["2wiki_ordinary_outcome"] == pytest.approx(1.0 / 6.0)
    assert mass["2wiki_hard_recovery"] == pytest.approx(1.0 / 12.0)
    assert mass["2wiki_hard_stability"] == pytest.approx(1.0 / 12.0)


def test_ordinary_selection_excludes_hard_and_consumed_qid_or_family():
    candidates = [
        {
            "dataset": "2wikimultihopqa", "qid": f"q{i}",
            "question": f"Question {i}?", "question_sha256": f"hash-{i}",
            "family_sha256": f"family-{i}",
        }
        for i in range(8)
    ]
    selected = select_ordinary_2wiki(
        candidates,
        n=4,
        blocked_qids={("2wikimultihopqa", "q0")},
        blocked_families={"family-1", "family-2"},
    )
    assert len(selected) == 4
    assert all(row["qid"] != "q0" for row in selected)
    assert all(row["family_sha256"] not in {"family-1", "family-2"} for row in selected)


def test_outcome_only_record_and_silver_row_are_explicitly_graph_free():
    identity = {
        "dataset": "hotpotqa",
        "qid": "h0",
        "question": "Who is the answer?",
        "question_sha256": "be775a06196a68c7bee707fe6fd193a30d16438cb55f51b02e2fbc7aedc0d98c",
        "question_key": "hotpotqa::h0",
        "family_sha256": "family-h0",
        "route": "hotpotqa_outcome",
        "process_reward_eligible": False,
    }
    record = make_outcome_only_kg_record(identity)
    validate_question_kg_record(record)
    assert record["kg_subgraph"] == []
    assert record["provenance"]["gold_access"] is False
    assert record["provenance"]["complete_plan_execution"] is False
    assert record["process_reward_eligible"] is False

    raw = {
        "id": "h0", "question": identity["question"],
        "golden_answers": ["Ada", "Ada Lovelace"],
        "metadata": {"type": "bridge"},
    }
    passages = [
        {"id": str(i), "contents": f"Passage {i}", "source": "test"}
        for i in range(10)
    ]
    silver = build_silver_row(identity, raw=raw, retrieved_passages=passages, kg_subgraph=[])
    assert silver["steps"] == []
    assert silver["kg_subgraph"] == []
    assert silver["metadata"]["gold_answer"] == "Ada"
    assert silver["metadata"]["failed_qpeg_or_saeg_p_edges_included"] is False
    assert all("golden_answers" not in passage for passage in silver["retrieved_passages"])
