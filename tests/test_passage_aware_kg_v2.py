from __future__ import annotations

from scripts.pilot.build_passage_aware_kg_overrides_v2 import select_additive_kg
from scripts.prepare.augment_entity_desc_index_from_local_subgraphs import infer_root_label
from scripts.pilot.score_kg_v2_zero_train_eval import evaluate_advancement_gate
from scripts.pilot.build_passage_aware_kg_overrides_v3 import (
    direct_answer_intent_pids,
    select_precision_delta,
)
from scripts.pilot.score_kg_v3_zero_train_eval import evaluate_v3_advancement_gate


def test_v2_falls_back_to_old_kg_instead_of_replacing_it_with_empty():
    old = [("Alpha", "instance of", "film")]
    selected, diagnostics = select_additive_kg(
        old,
        [],
        question="Unrelated question with no usable entity anchors?",
        question_mentions=[],
        passage_titles=[],
        max_keep=12,
        min_keep=5,
    )
    assert selected == old
    assert diagnostics["fallback_to_old"] is True
    assert diagnostics["n_selected_old"] == 1


def test_v2_deduplicates_union_and_records_both_sources():
    shared = ("Alpha", "directed by", "Director One")
    selected, diagnostics = select_additive_kg(
        [shared],
        [shared, ("Director One", "place of birth", "City Two")],
        question="Where was the director of Alpha born?",
        question_mentions=["Alpha"],
        passage_titles=["Alpha", "Director One"],
        max_keep=12,
        min_keep=0,
    )
    assert len(selected) == len(set(selected))
    shared_row = next(row for row in diagnostics["provenance"] if row["triple"] == list(shared))
    assert shared_row["sources"] == ["old_question_kg", "passage_kg"]


def test_v2_chain_policy_keeps_question_and_passage_anchors_within_budget():
    triples = [
        ("Film Alpha", "directed by", "Director One"),
        ("Director One", "place of birth", "City Two"),
        ("City Two", "country", "Country Three"),
        ("Noise A", "instance of", "Noise B"),
    ]
    selected, diagnostics = select_additive_kg(
        [],
        triples,
        question="Where was the director of Film Alpha born?",
        question_mentions=["Film Alpha"],
        passage_titles=["Director One", "City Two"],
        max_keep=3,
        min_keep=0,
    )
    assert len(selected) <= 3
    assert ("Film Alpha", "directed by", "Director One") in selected
    assert diagnostics["n_selected_passage"] >= 1


def test_local_subgraph_root_label_uses_dominant_head():
    label, share = infer_root_label(
        [
            ("Kelly Preston", "place of birth", "Honolulu"),
            ("Kelly Preston", "spouse", "John Travolta"),
            ("Honolulu", "country", "United States"),
        ]
    )
    assert label == "Kelly Preston"
    assert share == 2 / 3


def test_exploratory_advancement_gate_requires_gain_without_large_arm_loss():
    pairs = {
        "a": {"mcnemar": {"net": 1}},
        "b": {"mcnemar": {"net": 1}},
        "c": {"mcnemar": {"net": 0}},
        "d": {"mcnemar": {"net": 0}},
    }
    assert evaluate_advancement_gate(pairs, 1)["status"] == "PASS_EXPLORATORY_ADVANCE"
    pairs["d"]["mcnemar"]["net"] = -2
    assert evaluate_advancement_gate(pairs, 1)["status"] == "FAIL_STOP_KG_ROUTE"


def test_v3_preserves_stored_prefix_and_adds_only_direct_intent_delta():
    old = [("Morgan Llywelyn", "place of birth", "New York City")]
    v2 = [
        *old,
        ("Anna Seghers", "place of birth", "Mainz"),
        ("Anna Seghers", "spouse", "Johann Lorenz Schmidt"),
    ]
    linked = [
        {
            "mention": "Anna Seghers", "label": "Anna Seghers", "qid": "Q1",
            "score": 0.76, "margin": 0.76, "abstained": False,
        }
    ]
    final, diag = select_precision_delta(
        old,
        v2,
        question="Who was born earlier, Anna Seghers or Morgan Llywelyn?",
        linked_entities=linked,
        passage_titles=["Anna Seghers"],
    )
    assert final[0] == old[0]
    assert ("Anna Seghers", "place of birth", "Mainz") in final
    assert ("Anna Seghers", "spouse", "Johann Lorenz Schmidt") not in final
    assert diag["n_delta_selected"] == 1


def test_v3_rejects_generic_or_low_confidence_wrong_entity_links():
    v2 = [
        ("Bill Stewart", "occupation", "rugby union player"),
        ("English", "country", "United Kingdom"),
    ]
    linked = [
        {"mention": "Bill Stewart", "label": "Bill Stewart", "qid": "Q2",
         "score": 0.61, "margin": 0.61, "abstained": False},
        {"mention": "English", "label": "English", "qid": "Q3",
         "score": 0.70, "margin": 0.30, "abstained": False},
    ]
    final, diag = select_precision_delta(
        [],
        v2,
        question="Bill Stewart starred in a film based on what?",
        linked_entities=linked,
        passage_titles=["Bill Stewart (actor)"],
    )
    assert final == []
    assert diag["n_delta_selected"] == 0


def test_v3_final_clause_intent_overrides_bridge_keywords():
    assert direct_answer_intent_pids(
        "Get Bruce starred the actress and writer who started her career in what capacity?"
    ) == {"P106", "P39"}
    assert direct_answer_intent_pids(
        "Who directed the 2007 horror starring the actor who played John Kramer?"
    ) == {"P57"}


def test_v3_rejects_title_only_answer_relation_distractor():
    final, diag = select_precision_delta(
        [],
        [("Dead Silence", "director", "James Wan")],
        question="Who directed the 2007 horror starring Tobin Bell and Danielle Savre?",
        linked_entities=[
            {"mention": "Dead Silence", "label": "Dead Silence", "qid": "Q4",
             "score": 0.77, "margin": 0.77, "abstained": False},
        ],
        passage_titles=["Dead Silence"],
    )
    assert final == []
    assert diag["rejected"]["anchor_not_explicit_in_question"] == 1


def test_v3_gate_requires_no_regression_and_retains_verified_gains():
    pairs = {
        name: {"mcnemar": {"net": net}}
        for name, net in zip(("a", "b", "c", "d"), (0, 0, 2, 2))
    }
    metrics = {
        name: {"parse_rate": 1.0}
        for name in (
            "hidden_sft_old", "hidden_sft_v3", "hidden_ppo_old", "hidden_ppo_v3",
            "hard_sft_old", "hard_sft_v3", "hard_ppo_old", "hard_ppo_v3",
        )
    }
    retained = {
        "sft": ["train_11904", "train_14764"],
        "ppo": ["train_11904", "train_14764"],
    }
    assert evaluate_v3_advancement_gate(pairs, metrics, retained)["status"] == "PASS_TO_INDEPENDENT_VAL200"
    pairs["d"]["mcnemar"]["net"] = -1
    assert evaluate_v3_advancement_gate(pairs, metrics, retained)["status"] == "FAIL_STOP_KG_V3_ROUTE"
