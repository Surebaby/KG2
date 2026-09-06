from __future__ import annotations

from types import SimpleNamespace

import pytest

from kgproweight.config.schemas import PPOConfig
from kgproweight.reward.proofkg_process_v2 import (
    SCORER_VERSION as V21_SCORER_VERSION,
    build_execution_trace,
    score_proofkg_v2,
)
from kgproweight.reward.proofkg_process_v2_2 import (
    SCORER_VERSION,
    _parse_date_interval,
    build_execution_trace_v2_2,
    score_proofkg_v2_2,
)
from kgproweight.training.phase3_ppo import Phase3PPOConfig, _validate_mixed_reward_config
from kgproweight.training.reward_function import KGProWeightRewardFunction, RewardSpec


class _Tokenizer:
    def __call__(self, text, **_kwargs):
        return {"input_ids": list(text.encode("utf-8"))}

    def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
        return bytes(int(value) for value in ids).decode("utf-8")


def _response(triples, answer: str) -> str:
    blocks = []
    for index, triple in enumerate(triples, start=1):
        head, relation, tail = triple
        blocks.append(
            f"[Step {index}]\n"
            "Reasoning: The supplied proof edge directly supports this required hop.\n"
            f"Knowledge Used: [({head}, {relation}, {tail})]\n"
            f"Conclusion: {head} has {relation} {tail}.\n"
        )
    return "".join(blocks) + f"[Final Answer] {answer}"


def _score(question, plan, execution, answer):
    triples = [tuple(edge) for hop in execution["hops"] for edge in hop["matches"]]
    # The format gate needs at least two steps.  If a test plan has fewer proof
    # hops, repeat the final cited edge as a reasoning-only second step.
    response_triples = triples if len(triples) >= 2 else triples * 2
    return score_proofkg_v2_2(
        question=question,
        generation=_response(response_triples, answer),
        kg_triples=triples,
        execution_trace=build_execution_trace_v2_2(plan, execution),
        planned_hops=len(plan["hops"]),
    )


def _direct_temporal():
    plan = {
        "hops": [
            {
                "subject": "Euphemia Vale Blake",
                "pids": ["P570"],
                "output_slot": "hop_1",
                "relation_role": "answer_operand",
            },
            {
                "subject": "Camilo Torres Restrepo",
                "pids": ["P570"],
                "output_slot": "hop_2",
                "relation_role": "answer_operand",
            },
        ]
    }
    execution = {
        "hops": [
            {
                "hop_index": 1,
                "matches": [
                    ["Euphemia Vale Blake", "date of death", "28 October 1904"],
                    ["Euphemia Vale Blake", "date of death", "1904"],
                ],
                "input_entities": [{"surface": "Euphemia Vale Blake"}],
                "output_entities": [],
            },
            {
                "hop_index": 2,
                "matches": [
                    ["Camilo Torres", "date of death", "15 February 1966"]
                ],
                "input_entities": [{"surface": "Camilo Torres Restrepo"}],
                "output_entities": [],
            },
        ]
    }
    return plan, execution


def test_v22_temporal_parses_full_dates_and_bare_years():
    plan, execution = _direct_temporal()
    result = _score(
        "Who died later, Euphemia Vale Blake or Camilo Torres Restrepo?",
        plan,
        execution,
        "Camilo Torres Restrepo",
    )

    assert result["scorer_version"] == SCORER_VERSION
    assert result["components"]["operator"] == "temporal"
    assert result["components"]["m_A_deterministic"] == 1.0
    assert result["components"]["A_answer_consistency"] == 1.0
    assert result["components"]["derived_answer"] == "Camilo Torres Restrepo"


def test_v22_date_parser_supports_reordered_full_date_iso_and_rejects_invalid():
    assert _parse_date_interval("28 October 1904") == _parse_date_interval("October 28, 1904")
    assert _parse_date_interval("1904-10-28") == _parse_date_interval("28 October 1904")
    year = _parse_date_interval("1904")
    day = _parse_date_interval("28 October 1904")
    assert year is not None and day is not None and year[0] < day[0] < year[1]
    assert _parse_date_interval("31 February 1904") is None


def test_v22_terminal_date_accepts_equivalent_surface_order():
    plan = {
        "hops": [
            {"subject": "Director A", "pids": ["P569"], "output_slot": "hop_1", "relation_role": "answer_operand"}
        ]
    }
    execution = {
        "hops": [
            {"hop_index": 1, "matches": [["Director A", "date of birth", "28 February 1940"]]}
        ]
    }
    result = _score("When was Director A born?", plan, execution, "February 28, 1940")
    assert result["components"]["m_A_deterministic"] == 1.0
    assert result["components"]["A_answer_consistency"] == 1.0


def test_v22_terminal_opaque_qid_abstains_without_label_mapping():
    plan = {
        "hops": [
            {"subject": "Person A", "pids": ["P22"], "output_slot": "hop_1", "relation_role": "answer_operand"}
        ]
    }
    execution = {"hops": [{"hop_index": 1, "matches": [["Person A", "father", "Q123"]]}]}
    result = _score("Who is Person A's father?", plan, execution, "A Human Name")
    assert result["components"]["m_A_deterministic"] == 0.0
    assert result["components"]["derivation_status"] == "opaque_qid_terminal_tail"


def test_v22_terminal_compatible_coarse_and_full_dates_choose_precise_surface():
    plan = {
        "hops": [
            {"subject": "Composer A", "pids": ["P570"], "output_slot": "hop_1", "relation_role": "answer_operand"}
        ]
    }
    execution = {
        "hops": [
            {"hop_index": 1, "matches": [["Composer A", "date of death", "January 2003"], ["Composer A", "date of death", "12 January 2003"]]}
        ]
    }
    result = _score("When did Composer A die?", plan, execution, "January 12, 2003")
    assert result["components"]["m_A_deterministic"] == 1.0
    assert result["components"]["A_answer_consistency"] == 1.0
    assert result["components"]["derived_answer"] == "12 January 2003"


def test_v22_bridge_comparison_maps_winner_back_to_root_film():
    plan = {
        "hops": [
            {"subject": "In Old Cheyenne (1941 film)", "pids": ["P57"], "output_slot": "hop_1", "relation_role": "bridge"},
            {"subject": "Man by the Wayside", "pids": ["P57"], "output_slot": "hop_2", "relation_role": "bridge"},
            {"subject": "$hop_1", "pids": ["P569"], "output_slot": "hop_3", "relation_role": "answer_operand"},
            {"subject": "$hop_2", "pids": ["P569"], "output_slot": "hop_4", "relation_role": "answer_operand"},
        ]
    }
    triples = [
        ["In Old Cheyenne", "director", "Joseph Kane"],
        ["Man by the Wayside", "director", "William Dieterle"],
        ["Joseph Kane", "date of birth", "19 March 1894"],
        ["William Dieterle", "date of birth", "15 July 1893"],
    ]
    execution = {
        "hops": [
            {"hop_index": index, "matches": [triple], "input_entities": [], "output_entities": []}
            for index, triple in enumerate(triples, start=1)
        ]
    }
    result = _score(
        "Which film has the director born first, In Old Cheyenne or Man By The Wayside?",
        plan,
        execution,
        "Man By The Wayside",
    )

    assert result["components"]["m_A_deterministic"] == 1.0
    assert result["components"]["A_answer_consistency"] == 1.0
    assert result["components"]["derived_answer"] == "Man by the Wayside"


def test_v22_temporal_conflicting_multivalues_abstain():
    plan = {
        "hops": [
            {"subject": "Film A", "pids": ["P569"], "output_slot": "hop_1", "relation_role": "answer_operand"},
            {"subject": "Film B", "pids": ["P569"], "output_slot": "hop_2", "relation_role": "answer_operand"},
        ]
    }
    execution = {
        "hops": [
            {"hop_index": 1, "matches": [["Director A", "date of birth", "1893"], ["Director A", "date of birth", "1899"]]},
            {"hop_index": 2, "matches": [["Director B", "date of birth", "1898"], ["Director B", "date of birth", "1908"]]},
        ]
    }
    result = _score("Which film has the director born first, Film A or Film B?", plan, execution, "Film A")

    assert result["components"]["m_A_deterministic"] == 0.0
    assert result["components"]["A_answer_consistency"] == 0.0
    assert result["components"]["derivation_status"] == "overlapping_or_tied_intervals"


def test_v22_temporal_multivalue_result_is_independent_of_claim_order():
    plan, execution = _direct_temporal()
    left = _score(
        "Who died later, Euphemia Vale Blake or Camilo Torres Restrepo?",
        plan,
        execution,
        "Camilo Torres Restrepo",
    )
    reversed_execution = {
        "hops": [
            {**hop, "matches": list(reversed(hop["matches"]))}
            for hop in reversed(execution["hops"])
        ]
    }
    right = _score(
        "Who died later, Euphemia Vale Blake or Camilo Torres Restrepo?",
        plan,
        reversed_execution,
        "Camilo Torres Restrepo",
    )
    assert right["components"]["derived_answer"] == left["components"]["derived_answer"]
    assert right["components"]["A_answer_consistency"] == left["components"]["A_answer_consistency"]


@pytest.mark.parametrize(("right_value", "answer"), [("American", "yes"), ("German", "no")])
def test_v22_same_country_uses_singleton_operand_values(right_value, answer):
    plan = {
        "hops": [
            {"subject": "Film A", "pids": ["P495"], "output_slot": "hop_1", "relation_role": "answer_operand"},
            {"subject": "Film B", "pids": ["P495"], "output_slot": "hop_2", "relation_role": "answer_operand"},
        ]
    }
    execution = {
        "hops": [
            {"hop_index": 1, "matches": [["Film A", "country of origin", "American"]]},
            {"hop_index": 2, "matches": [["Film B", "country of origin", right_value]]},
        ]
    }
    result = _score("Are both Film A and Film B from the same country?", plan, execution, answer)

    assert result["components"]["m_A_deterministic"] == 1.0
    assert result["components"]["A_answer_consistency"] == 1.0
    assert result["components"]["derived_answer"] == answer


def test_v22_same_country_multivalue_abstains_instead_of_using_list_order():
    plan = {
        "hops": [
            {"subject": "Film A", "pids": ["P495"], "output_slot": "hop_1", "relation_role": "answer_operand"},
            {"subject": "Film B", "pids": ["P495"], "output_slot": "hop_2", "relation_role": "answer_operand"},
        ]
    }
    execution = {
        "hops": [
            {"hop_index": 1, "matches": [["Film A", "country of origin", "Irish"], ["Film A", "country of origin", "American"]]},
            {"hop_index": 2, "matches": [["Film B", "country of origin", "American"]]},
        ]
    }
    result = _score("Are both Film A and Film B from the same country?", plan, execution, "no")

    assert result["components"]["m_A_deterministic"] == 0.0
    assert result["components"]["derivation_status"] == "ambiguous_multivalue_operand"


def test_frozen_v21_temporal_behavior_is_not_rewritten():
    plan, execution = _direct_temporal()
    triples = [tuple(edge) for hop in execution["hops"] for edge in hop["matches"]]
    old = score_proofkg_v2(
        question="Who died later, Euphemia Vale Blake or Camilo Torres Restrepo?",
        generation=_response(triples, "Camilo Torres Restrepo"),
        kg_triples=triples,
        execution_trace=build_execution_trace(plan, execution),
        planned_hops=2,
    )

    assert old["scorer_version"] == V21_SCORER_VERSION
    assert old["components"]["m_A_deterministic"] == 0.0


def test_reward_runtime_dispatches_v22_without_changing_v21_config():
    plan, execution = _direct_temporal()
    triples = [tuple(edge) for hop in execution["hops"] for edge in hop["matches"]]
    runtime = {
        "question_key": "2wikimultihopqa::comparison-1",
        "query_plan": plan,
        "execution": execution,
        "provenance": {"gold_access": False, "complete_plan_execution": True},
    }
    reward = KGProWeightRewardFunction(
        alpha_gate=SimpleNamespace(),
        prm_annotator=SimpleNamespace(),
        text_reward_model=SimpleNamespace(),
        tokenizer=_Tokenizer(),
        outcome_weight=4.0,
        min_valid_steps=3,
        min_reasoning_chars=20,
        proofkg_process_reward=True,
        proofkg_process_version="v2_2",
        proofkg_process_weight=0.2,
        proofkg_f1_weight=0.1,
        proofkg_dynamic_validity=True,
        mixed_outcome_reward=True,
    )
    spec = RewardSpec(
        query="Who died later, Euphemia Vale Blake or Camilo Torres Restrepo?",
        gold_answer="Camilo Torres Restrepo",
        kg_subgraph=triples,
        metadata={
            "dataset": "2wikimultihopqa",
            "qid": "comparison-1",
            "question_kg_runtime": runtime,
        },
    )
    result = reward(prompt="", response=_response(triples, "Camilo Torres Restrepo"), spec=spec)

    assert PPOConfig(proofkg_process_version="v2_2").proofkg_process_version == "v2_2"
    assert result["proofkg_process"]["scorer_version"] == SCORER_VERSION
    assert result["proofkg_process"]["process_applied"] is True


def test_mixed_config_accepts_v22_but_legacy_v21_remains_valid():
    common = dict(
        silver_path="s",
        output_dir="o",
        mixed_outcome_reward=True,
        outcome_weight=4.0,
        proofkg_f1_weight=0.1,
        proofkg_dynamic_validity=True,
        proofkg_process_reward=True,
        proofkg_process_weight=0.2,
        question_kg_records_path="records.jsonl",
    )
    _validate_mixed_reward_config(
        Phase3PPOConfig(**common, proofkg_process_version="v2_1")
    )
    _validate_mixed_reward_config(
        Phase3PPOConfig(**common, proofkg_process_version="v2_2")
    )
