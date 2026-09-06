import pytest

from scripts.pilot.probe_dense_rank_repair_v1 import rank_diagnostic


def test_actual_inversion_detected_without_mutating_source_order():
    docs=[{"id":"b"},{"id":"a"},{"id":"c"}]
    scores=[.2,.8,.1]
    result=rank_diagnostic(docs,scores)
    assert not result["returned_descending"]
    assert result["adjacent_inversion_after_ranks"]==[1]
    assert result["stable_score_desc_docid_tie_order"]==[1,0,2]
    assert not result["returned_top1_is_best_score"]
    assert scores==[.2,.8,.1]
    assert [d["id"] for d in docs]==["b","a","c"]


def test_monotone_scores_do_not_claim_rank_bug():
    result=rank_diagnostic([{"id":"a"},{"id":"b"}],[.8,.2])
    assert result["returned_descending"]
    assert not result["stable_sort_changes_order"]


def test_exact_tie_order_change_is_distinct_from_inversion():
    result=rank_diagnostic([{"id":"b"},{"id":"a"}],[.8,.8])
    assert result["returned_descending"]
    assert result["adjacent_exact_score_ties"]==1
    assert result["stable_sort_changes_order"]


@pytest.mark.parametrize("docs,scores",[([],[]),([{"id":"a"}],[float("nan")]),([{"id":"a"}],[]),([{"id":"a"},{"id":"a"}],[.8,.2])])
def test_bad_scores_or_duplicate_identity_fail(docs,scores):
    with pytest.raises(ValueError):
        rank_diagnostic(docs,scores)
