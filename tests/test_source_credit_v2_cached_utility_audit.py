from copy import deepcopy
import math

import pytest

from scripts.pilot.audit_source_credit_v2_cached_and_utility_v1 import (
    independent_terms, pair_summary, top1_summary,
)


def fixture(dimension=6):
    names = [f"f{i}" for i in range(dimension)]
    row = {"features": {"m_graph": 1, "values": dict.fromkeys(names, .5)},
           "trajectory_valid": True, "raw_text": [-1., 0., 1.], "raw_graph": .75}
    artifact = {"feature_names": names, "weights": [0.] * (dimension - 1) + [2.], "bias": -.25,
                "feature_standardization": {"mean": dict.fromkeys(names, .25), "scale": dict.fromkeys(names, .5)},
                "normalization": {"fixed_alpha": .4, "graph_center": .25, "graph_scale": .25,
                                  "text_center": -.5, "text_scale": .1,
                                  "text_v2": {"text_center": 0., "text_scale": .5}}}
    return row, artifact


@pytest.mark.parametrize("dimension", [4, 6])
def test_independent_logistic_softsign_and_exact_budget(dimension):
    row, artifact = fixture(dimension)
    before = deepcopy((row, artifact))
    terms = independent_terms(row, artifact, "A")
    expected_alpha = 1. / (1. + math.exp(-.75))
    assert terms["alpha"] == pytest.approx(expected_alpha)
    assert terms["z"] == [-2., 0., 2.]
    assert terms["bounded"] == [-2/3, 0., 2/3]
    assert terms["graph"] == pytest.approx(.2 * expected_alpha)
    assert terms["text"] == 0.
    assert sum(terms["steps"]) == terms["text"]
    assert (row, artifact) == before


@pytest.mark.parametrize("arm", ["A", "F", "T"])
def test_masked_graph_zero_text_preserved(arm):
    row, artifact = fixture()
    row["features"]["m_graph"] = 0
    row["raw_text"] = [1.]
    result = independent_terms(row, artifact, arm)
    assert result["alpha"] == result["graph"] == 0.
    assert result["text"] == pytest.approx(.2)


@pytest.mark.parametrize("arm", ["A", "F", "T"])
def test_invalid_process_zero(arm):
    row, artifact = fixture()
    row["trajectory_valid"] = False
    row["raw_text"] = []
    result = independent_terms(row, artifact, arm)
    assert result["alpha"] == result["graph"] == result["text"] == result["process"] == 0.
    assert result["steps"] == []


def test_old_normalization_is_explicit_hard_clip():
    row, artifact = fixture(4)
    del artifact["normalization"]["text_v2"]
    assert independent_terms(row, artifact, "T")["bounded"] == [-1., 1., 1.]


def diagnostics():
    return [{"candidate_id": cid, "question_key": "q", "valid": True,
             "credit_eligible": True, "em": float(cid == "b"), "f1": float(cid == "b"),
             "versions": {"v": {arm: {"process": 0.} for arm in ("A", "F", "T")}}}
            for cid in ("b", "a")]


def test_top1_ties_never_use_gold_or_input_order():
    data = diagnostics()
    result = top1_summary(data, "v")
    assert result["selected_ids"]["A"] == [["q", "a"]]
    assert result["arms"]["A"]["em"] == 0.
    assert result["oracle_at2"]["em"] == 1.
    for r in data:
        r["em"] = 1 - r["em"]
    assert top1_summary(list(reversed(data)), "v")["selected_ids"] == result["selected_ids"]


def test_top1_invalid_candidates_excluded_even_when_score_higher():
    data = diagnostics()
    data[0]["valid"] = False
    data[0]["versions"]["v"]["A"]["process"] = 9.
    result = top1_summary(data, "v")
    assert result["selected_ids"]["A"] == [["q", "a"]]
    assert result["oracle_at2"]["em"] == 1. and result["format_valid_oracle_at2"]["em"] == 0.
    data[1]["valid"] = False
    result = top1_summary(data, "v")
    assert result["selected_ids"]["A"] == [["q", None]]
    assert result["arms"]["A"]["em"] == 0. and result["all_invalid_questions"] == 1


def test_pair_tie_adjustment_and_fixed_population():
    data = diagnostics()
    data[0]["versions"]["v"]["A"]["process"] = .1
    data[0]["versions"]["v"]["T"]["process"] = -.1
    result = pair_summary(data, "v", graph_only=True)
    assert result["n_pairs"] == 1
    assert result["rankings"]["A"]["tie_adjusted_accuracy"] == 1.
    assert result["rankings"]["F"]["tie_adjusted_accuracy"] == .5
    assert result["rankings"]["T"]["tie_adjusted_accuracy"] == 0.
    assert result["A_minus_F"]["improved"] == 1
    data[0]["credit_eligible"] = False
    assert pair_summary(data, "v", graph_only=True)["n_pairs"] == 0
    assert pair_summary(data, "v")["n_pairs"] == 1


def test_pair_and_top1_reject_incomplete_k2():
    for fn in (pair_summary, top1_summary):
        with pytest.raises(ValueError, match="K2"):
            fn(diagnostics()[:1], "v")
