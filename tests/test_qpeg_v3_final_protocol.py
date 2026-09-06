from __future__ import annotations

import pytest

from scripts.prepare.build_qpeg_v3_final_ab_inputs import _validate_prerequisites
from scripts.eval.evaluate_qpeg_v3_final_ab import _decision_gates


def _valid():
    return (
        {"status": "PASS_TRAIN_HOLDOUT_ADVANCE_CORRECTED_RUNTIME_CAP"},
        {
            "status": "COMPLETE_NOT_EVALUATED",
            "n": 900,
            "integrity": {
                "identity_unique": True,
                "gold_access_false": True,
                "provenance_validated": True,
                "max_edges": 4,
            },
        },
        {"status": "PASS_REUSE_EXACT_PROMPTS", "gates": {"all_pass": True}},
    )


def test_qpeg_v3_final_prerequisites_pass_only_when_all_inputs_are_frozen():
    gate, materialization, reuse = _valid()
    _validate_prerequisites(
        gate_addendum=gate, materialization=materialization, reuse=reuse
    )


@pytest.mark.parametrize("field", ["identity_unique", "gold_access_false", "provenance_validated"])
def test_qpeg_v3_final_prerequisites_reject_integrity_failure(field):
    gate, materialization, reuse = _valid()
    materialization["integrity"][field] = False
    with pytest.raises(ValueError):
        _validate_prerequisites(
            gate_addendum=gate, materialization=materialization, reuse=reuse
        )


def test_qpeg_v3_final_prerequisites_reject_failed_prompt_reuse():
    gate, materialization, reuse = _valid()
    reuse["gates"]["all_pass"] = False
    with pytest.raises(ValueError):
        _validate_prerequisites(
            gate_addendum=gate, materialization=materialization, reuse=reuse
        )


def test_qpeg_v3_final_decision_gates_use_frozen_macro_and_loss_rules():
    protocol = {
        "decision_gates": {
            "macro_delta_em_gt": 0.0,
            "macro_delta_f1_gt": 0.0,
            "max_net_correct_loss_per_dataset": 6,
            "max_parse_rate_drop": 0.01,
        }
    }
    values = {}
    for dataset, delta in zip(("hotpotqa", "2wikimultihopqa", "musique"), (0.01, 0.02, -0.01)):
        values[dataset] = {"paired": {
            "delta_em": delta,
            "delta_f1": 0.01,
            "net_correct": -3 if delta < 0 else 1,
            "no_graph_parse_rate": 1.0,
            "qpeg_v3_parse_rate": 0.99,
        }}
    assert all(_decision_gates(values, protocol).values())
    values["musique"]["paired"]["net_correct"] = -7
    assert _decision_gates(values, protocol)["no_dataset_net_loss_gt_6"] is False
