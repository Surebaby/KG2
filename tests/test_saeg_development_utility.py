from kgproweight.data.saeg_dataset import route_eval_arm
from scripts.eval.evaluate_saeg_v1_development_utility import aggregate, paired


def _row(key, arm, em, f1=None, well=True):
    return {
        "question_key": key,
        "arm": arm,
        "em": em,
        "f1": em if f1 is None else f1,
        "well_formed": well,
        "citation_contract_valid": well,
        "known_wikidata_citations": 0,
        "known_passage_citations": 0,
        "prompt_tokens": 100,
    }


def test_not_evaluable_arm_is_explicit():
    assert aggregate([]) == {"n": 0, "status": "NOT_EVALUABLE"}


def test_paired_reports_direction_and_net_correct():
    rows = [
        _row("q1", "A_no_graph", 0),
        _row("q1", "D_fused", 1),
        _row("q2", "A_no_graph", 1),
        _row("q2", "D_fused", 0),
        _row("q3", "A_no_graph", 0),
        _row("q3", "D_fused", 1),
    ]
    result = paired(rows, "A_no_graph", "D_fused")
    assert result["n"] == 3
    assert result["gained_correct"] == 2
    assert result["lost_correct"] == 1
    assert result["net_correct"] == 1
    assert result["delta_em"] == 1 / 3


def test_empty_passage_is_a_paired_fallback_not_an_ineligible_qid():
    row = {
        "question": "Q?",
        "passages": [{"contents": "context"}],
        "passage_evidence": [],
        "wikidata_kg": [],
        "source_status": {
            "passage": "empty_fail_closed",
            "wikidata": "not_eligible_frozen_structural_failure",
        },
        "arms": {
            "A_no_graph": {"eligible": True},
            "B_passage": {"eligible": False},
            "C_wikidata": {"eligible": False},
            "D_fused": {"eligible": False},
        },
    }
    assert route_eval_arm(row, "A_no_graph") == route_eval_arm(row, "B_passage")
    assert route_eval_arm(row, "A_no_graph") == route_eval_arm(row, "D_fused")
