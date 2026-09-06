from __future__ import annotations

from collections import Counter

from kgproweight.kg.question_kg import question_key, question_sha256
from scripts.prepare.freeze_2wiki_proofkg_extension_reserve_v1 import (
    build_reserve_and_combined,
    project_official_train_identities,
)
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256


DATASET = "2wikimultihopqa"


def _row(qid: str, qtype: str, marker: str) -> dict:
    question = f"Where does unique marker {marker} lead?"
    return {
        "dataset": DATASET,
        "qid": qid,
        "question": question,
        "question_sha256": question_sha256(question),
        "question_type": qtype,
        "family_version": FAMILY_VERSION,
        "family_sha256": family_sha256(question),
        "target_type": "relation_graph",
        "gold_access": False,
        "evaluation_eligible": False,
        "metadata": {"question_type": qtype, "train_only": True},
    }


def _assignment(row: dict) -> dict:
    return {
        "question_key": question_key(DATASET, row["qid"]),
        "dataset": DATASET,
        "qid": row["qid"],
        "split": "train",
        "family_sha256": "assignment-" + row["qid"],
    }


def test_reserve_is_exact_disjoint_question_only_and_combined_is_deterministic():
    parent = [_row(f"parent-{index}", "inference", f"parent{index}") for index in range(300)]
    auto = [_row("old", "inference", "old")]
    protected = [_row("protected", "inference", "protected")]
    sources = [
        *[_row(f"i-{index}", "inference", f"infer{index}") for index in range(5)],
        *[_row(f"c-{index}", "compositional", f"compose{index}") for index in range(4)],
        *[_row(f"x-{index}", "comparison", f"compare{index}") for index in range(3)],
        auto[0],
        protected[0],
        parent[0],
    ]
    assignments = [_assignment(row) for row in sources]
    kwargs = {
        "auto1500_rows": auto,
        "parent_rows": parent,
        "protected_rows": protected,
        "quotas": {"inference": 3, "compositional": 2, "comparison": 1},
        "seed": 42,
    }
    reserve, combined, telemetry = build_reserve_and_combined(
        source_rows=sources, assignment_rows=assignments, **kwargs
    )
    reserve_reversed, combined_reversed, telemetry_reversed = build_reserve_and_combined(
        source_rows=list(reversed(sources)),
        assignment_rows=list(reversed(assignments)),
        **kwargs,
    )
    assert reserve == reserve_reversed
    assert combined == combined_reversed
    assert telemetry == telemetry_reversed
    assert Counter(row["question_type"] for row in reserve) == Counter(
        {"inference": 3, "compositional": 2, "comparison": 1}
    )
    assert len(combined) == 306
    assert not ({row["qid"] for row in reserve} & {"old", "protected", "parent-0"})
    assert all(row["gold_access"] is False for row in combined)
    assert all("answer" not in row and "steps" not in row and "kg_subgraph" not in row for row in combined)
    assert all(telemetry["gates"].values())


def test_official_projection_drops_gold_support_and_non_train_rows():
    raw_train = {
        "id": "q-train",
        "question": "Which item is linked?",
        "golden_answers": ["must not copy"],
        "metadata": {"type": "inference", "supporting_facts": ["must not copy"]},
    }
    raw_dev = {
        "id": "q-dev",
        "question": "Which other item is linked?",
        "golden_answers": ["must not copy"],
        "metadata": {"type": "comparison"},
    }
    raw_duplicate = {
        "id": "z-duplicate",
        "question": "Which item is linked?",
        "golden_answers": ["must not copy"],
        "metadata": {"type": "inference"},
    }
    assignments = [
        {
            "question_key": f"{DATASET}::q-train",
            "dataset": DATASET,
            "qid": "q-train",
            "split": "train",
        },
        {
            "question_key": f"{DATASET}::q-dev",
            "dataset": DATASET,
            "qid": "q-dev",
            "split": "dev",
        },
        {
            "question_key": f"{DATASET}::z-duplicate",
            "dataset": DATASET,
            "qid": "z-duplicate",
            "split": "train",
        },
    ]
    projected, counts = project_official_train_identities(
        [raw_duplicate, raw_train, raw_dev], assignments
    )
    assert projected == [
        {
            "dataset": DATASET,
            "qid": "q-train",
            "question": "Which item is linked?",
            "metadata": {"question_type": "inference", "train_only": True},
        }
    ]
    assert counts == {
        "duplicate_question_hash": 1,
        "non_train_assignment": 1,
        "projected_train": 1,
    }
    assert "golden_answers" not in projected[0]
