from scripts.pilot.adjudicate_query_aware_confirmation import adjudicate


def test_adjudication_fails_if_any_frozen_gate_fails():
    manifest = {
        "preregistered_gates": {
            "plan_recognized_rate_min": 0.9,
            "reference_relation_recall_min": 0.9,
            "expected_explicit_anchor_in_plan_recall_min": 0.9,
            "expected_explicit_anchor_linked_from_plan_recall_min": 0.85,
            "complete_plan_execution_rate_min": 0.8,
            "full_relation_value_chain_rate_evaluable_min": 0.7,
            "per_stratum_plan_recognized_rate_min": 0.8,
        }
    }
    report = {
        "overall": {
            "rates": {
                "plan_recognized": 0.95,
                "reference_relation_recall": 0.95,
                "expected_explicit_anchor_in_plan_recall": 0.95,
                "expected_explicit_anchor_linked_from_plan_recall": 0.9,
                "complete_plan_execution": 0.79,
                "full_relation_value_chain_rate_evaluable": 0.8,
            }
        }
    }
    details = [
        {"stratum": "a", "query_plan": {"recognized": True}},
        {"stratum": "b", "query_plan": {"recognized": True}},
    ]
    result = adjudicate(manifest, report, details)
    assert result["decision"] == "FAIL_STOP_STRUCTURAL_CONFIRMATION"
    assert not result["checks"]["complete_plan_execution_rate_min"]["passed"]
