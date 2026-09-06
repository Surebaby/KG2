import pytest

from kgproweight.data.saeg_dataset import assert_role_allowed, route_eval_arm


def _row(role="development", p=True, w=True):
    return {
        "question": "Q?",
        "passages": [{"contents": "context"}],
        "passage_evidence": [{"passage_id": "P1"}] if p else [],
        "wikidata_kg": [["A", "relation", "B"]] if w else [],
        "source_status": {
            "passage": "nonempty" if p else "empty_fail_closed",
            "wikidata": "nonempty" if w else "not_eligible_frozen_structural_failure",
        },
        "role": role,
    }


def test_no_graph_and_passage_arms_change_only_evidence_sources():
    no_graph = route_eval_arm(_row(), "A_no_graph")
    passage = route_eval_arm(_row(), "B_passage")
    assert no_graph["retrieved_passages"] == passage["retrieved_passages"]
    assert no_graph["kg_triples"] == no_graph["passage_evidence"] == []
    assert passage["kg_triples"] == []
    assert passage["passage_evidence"] == [{"passage_id": "P1"}]


def test_empty_passage_arm_fails_closed_but_stays_evaluable():
    routed = route_eval_arm(_row(p=False, w=False), "B_passage")
    assert routed["fallback_no_graph"]


def test_ineligible_wikidata_arm_raises_instead_of_silent_mixing():
    with pytest.raises(ValueError, match="not structurally eligible"):
        route_eval_arm(_row(w=False), "C_wikidata")


def test_confirmation_and_reporting_roles_are_guarded():
    with pytest.raises(PermissionError, match="sealed"):
        assert_role_allowed(_row(role="confirmation"))
    with pytest.raises(PermissionError, match="cannot be used"):
        assert_role_allowed(_row(role="reporting_only_nonconfirmatory"))
    assert_role_allowed(_row(role="confirmation"), allow_confirmation=True)
