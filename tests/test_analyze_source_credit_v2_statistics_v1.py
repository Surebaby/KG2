"""Independent synthetic statistical checks; no fresh inputs or Gold are read."""
from copy import deepcopy

import pytest

from scripts.pilot import analyze_source_credit_v2_fresh_confirmation_v1 as analysis


def estimate(point, low, high, families=96):
    return {"point": point, "ci95": [low, high], "families": families}


def decision_summary():
    return {
        "all132_ITT": {"metrics": {"sampled_valid": estimate(.8, .75, .85)}},
        "three_dataset_macro": {"sampled_valid": estimate(.8, .75, .85)},
        "graph96_ITT": {"metrics": {
            "oracle_minus_greedy_em": estimate(.1, .04, .2),
            "features_v2_A_minus_greedy_em": estimate(.02, -.02, .08),
            "features_v2_A_minus_F_em": estimate(0., 0., 0.),
        }},
        "source_PASS79": {"pairwise": {"features_v2": {"A": {
            **estimate(.7, .6, .8), "mixed_outcome_families": 25,
        }}}},
    }


def test_k4_crosspairs_average_within_question_before_family_mean():
    candidates = [
        {"candidate_id": str(i), "variants": {"features_v2": {"A": {"process": score}}}}
        for i, score in enumerate((4., 2., 3., 1.))
    ]
    outcomes = {str(i): {"em": float(i < 2)} for i in range(4)}
    pair = analysis.pairwise(candidates, outcomes, "features_v2", "A")
    assert pair == {"family_accuracy": .75, "correct_incorrect_pairs": 4, "wins": 3, "ties": 0}

    # One family contributes 4 pairs; another contributes only 1. The primary
    # estimand gives each family one vote rather than treating 5 pairs as iid.
    rows = []
    for i, detail in enumerate((pair, {"family_accuracy": 0., "correct_incorrect_pairs": 1, "wins": 0, "ties": 0})):
        rows.append({"family_sha256": str(i), "dataset": "2wikimultihopqa",
                     "bootstrap_stratum": "graph:comparison",
                     "pairwise": {variant: {arm: deepcopy(detail) for arm in analysis.ARMS}
                                  for variant in analysis.VARIANTS}})
    result = analysis.cohort_summary(rows)["pairwise"]["features_v2"]["A"]
    assert result["point"] == .375
    assert result["micro_pair_accuracy_diagnostic"] == .6
    assert result["mixed_outcome_families"] == result["families"] == 2
    assert result["correct_incorrect_pairs"] == 5


def test_fixed_stratification_preserves_micro_and_three_domain_macro_weights():
    rows = []
    for i, (dataset, stratum, value) in enumerate((
            ("2wikimultihopqa", "graph:comparison", 0.),
            ("2wikimultihopqa", "graph:comparison", 0.),
            ("hotpotqa", "ordinary:hotpotqa", 1.),
            ("musique", "ordinary:musique", 1.))):
        rows.append({"family_sha256": str(i), "dataset": dataset,
                     "bootstrap_stratum": stratum, "valid": value})
    micro = analysis.bootstrap_estimates(rows, ["valid"], replicates=1000)["valid"]
    macro = analysis.bootstrap_estimates(rows, ["valid"], replicates=1000, macro_dataset=True)["valid"]
    assert micro["point"] == .5 and micro["ci95"] == [.5, .5]
    assert macro["point"] == pytest.approx(2/3)
    assert macro["ci95"] == pytest.approx([2/3, 2/3])
    with pytest.raises(ValueError, match="all three"):
        analysis.bootstrap_estimates(rows[:2], ["valid"], macro_dataset=True)


def test_paired_identical_arms_have_zero_delta_without_independent_bootstraps():
    rows = [{"family_sha256": str(i), "dataset": "2wikimultihopqa",
             "bootstrap_stratum": "graph:comparison", "A": float(i % 2), "F": float(i % 2), "A_minus_F": 0.}
            for i in range(10)]
    result = analysis.bootstrap_estimates(rows, ["A", "F", "A_minus_F"], replicates=1000)
    assert result["A"] == result["F"]
    assert result["A_minus_F"] == {"point": 0., "ci95": [0., 0.], "families": 10}
    with pytest.raises(ValueError, match="unique family"):
        analysis.bootstrap_estimates(rows + [rows[0]], ["A"])


def test_health_failure_is_separate_from_independent_utility_pass():
    result = analysis.decide(decision_summary())
    assert result["health_status"] == "FAIL"
    assert result["independent_utility_status"] == "PASS"
    assert result["overall_status"] == "FAIL"
    assert result["engineering_probe_eligibility"] is True
    assert result["matched600_investment_clearance"] is False
    assert result["full_ppo_auto_launch"] is False


def test_fewer_than_25_mixed_families_cannot_mechanically_fail_utility():
    summary = decision_summary()
    summary["source_PASS79"]["pairwise"]["features_v2"]["A"].update(
        point=.1, ci95=[0., .2], mixed_outcome_families=24)
    result = analysis.decide(summary)
    assert result["independent_utility_status"] == "INCONCLUSIVE"
    assert all(value["status"] == "INCONCLUSIVE" for value in result["utility"].values())
    assert result["utility"]["source_pass_A_pairwise"]["point_ci_status_diagnostic"] == "FAIL"
    assert result["engineering_probe_eligibility"] is False
    # Overall can separately fail its health target, without calling the
    # information-limited process utility a failure.
    assert result["health_status"] == result["overall_status"] == "FAIL"


def test_oracle_information_shortfall_stays_inconclusive_even_with_narrow_ci():
    summary = decision_summary()
    summary["all132_ITT"]["metrics"]["sampled_valid"] = estimate(.95, .91, .98)
    summary["three_dataset_macro"]["sampled_valid"] = estimate(.95, .91, .98)
    summary["graph96_ITT"]["metrics"]["oracle_minus_greedy_em"] = estimate(.01, -.01, .02)
    result = analysis.decide(summary)
    assert result["information"]["graph_valid_oracle_minus_raw_greedy_em"]["status"] == "INCONCLUSIVE"
    assert result["health_status"] == "PASS"
    assert result["independent_utility_status"] == result["overall_status"] == "INCONCLUSIVE"


def test_point_threshold_pass_is_not_a_significance_or_equivalence_claim():
    assert analysis.target_decision(estimate(0., -.2, .2), 0.)["status"] == "PASS"
    assert analysis.target_decision(estimate(-.01, -.2, .2), 0.)["status"] == "INCONCLUSIVE"
    assert analysis.target_decision(estimate(-.1, -.2, -.01), 0.)["status"] == "FAIL"
    assert analysis.decide(decision_summary())["A_equals_F_can_pass_but_does_not_establish_equivalence"] is True


def test_all_invalid_sampled_question_remains_itt_zero_with_raw_greedy_retained():
    key = "hotpotqa::synthetic-all-invalid"
    rows = []
    for index in range(5):
        rows.append({"candidate_id": f"{key}::k{index}", "candidate_index": index,
                     "generation_kind": "sampled" if index < 4 else "greedy",
                     "generation": "[Final Answer]\nSynthetic Answer", "trajectory_valid": False,
                     "n_response_tokens": 12, "reached_max_new_tokens": False,
                     "process_row_sha256": f"synthetic-{index}",
                     "variants": {variant: {arm: {"alpha_effective": 0., "text_component": 0.,
                                                "graph_component": 0., "process": 0.}
                                           for arm in analysis.ARMS} for variant in analysis.VARIANTS}})
    context = {
        "cohort": {key: {"question_key": key, "dataset": "hotpotqa", "family_sha256": "synthetic-family",
                         "question_type": "ordinary", "proposal_role": "ordinary"}},
        "checks": {}, "grouped": {key: rows},
        "rankings": {key: {"rankings": {variant: {arm: {"selected_candidate_id": None} for arm in analysis.ARMS}
                                            for variant in analysis.VARIANTS}}},
    }
    questions, candidates = analysis.question_metrics(context, {key: ["Synthetic Answer"]})
    assert len(questions) == 1 and len(candidates) == 5
    question = questions[0]
    assert question["all_sampled_invalid"] == 1. and question["sampled_valid"] == 0.
    assert question["raw_greedy_em"] == question["raw_sampled_mean_em"] == 1.
    assert question["format_gated_greedy_em"] == question["valid_oracle_em"] == 0.
    assert question["source_status"] == "ORDINARY" and question["bootstrap_stratum"] == "ordinary:hotpotqa"
    for variant in analysis.VARIANTS:
        for arm in analysis.ARMS:
            assert question[f"{variant}_{arm}_em"] == question[f"{variant}_{arm}_f1"] == 0.
            assert question["pairwise"][variant][arm]["family_accuracy"] is None
            assert question["pairwise"][variant][arm]["correct_incorrect_pairs"] == 0
