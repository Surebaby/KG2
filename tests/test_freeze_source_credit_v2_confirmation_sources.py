from copy import deepcopy

import pytest

from kgproweight.reward.source_quality_gate_v1 import canonical_sha256
from scripts.prepare.freeze_source_credit_v2_confirmation_sources import (
    WRAPPER_ALLOWED, make_wrapper, require_gold_free, select_graph_inputs,
)


def population():
    cohort, inputs = [], []
    for index in range(132):
        graph = index < 96
        record = {"qid": str(index), "dataset": "synthetic", "kg_subgraph": []}
        identity = {"question_key": f"synthetic::{index}", "dataset": "synthetic", "qid": str(index),
                    "question": f"Question {index}?", "question_sha256": f"qhash{index}", "family_sha256": f"family{index}"}
        cohort.append({**identity, "proposal_role": "graph" if graph else "ordinary",
                       "question_type": ("bridge_comparison", "comparison", "compositional")[index % 3]})
        item = {**identity, "m_graph": int(graph), "fullsource_record": record,
                "source_record_sha256": canonical_sha256(record)}
        item["input_sha256"] = canonical_sha256(item)
        inputs.append(item)
    return cohort, inputs


def test_exact_identity_selection_keeps_all_graph_rows_without_label_filter():
    cohort, inputs = population()
    before = deepcopy((cohort, inputs))
    selected = select_graph_inputs(cohort, inputs)
    assert len(selected) == 96
    assert selected == inputs[:96]
    assert (cohort, inputs) == before


@pytest.mark.parametrize("change", ["drop", "identity", "record", "input_hash", "m_graph"])
def test_population_or_source_tampering_rejected(change):
    cohort, inputs = population()
    if change == "drop":
        inputs.pop()
    elif change == "identity":
        inputs[0]["family_sha256"] = "different"
    elif change == "record":
        inputs[0]["fullsource_record"]["qid"] = "different"
    elif change == "input_hash":
        inputs[0]["input_sha256"] = "changed"
    else:
        inputs[0]["m_graph"] = 0
    with pytest.raises(ValueError):
        select_graph_inputs(cohort, inputs)


@pytest.mark.parametrize("key", ["gold", "gold_answer", "gold_answers", "golden_answers", "gold_answer_aliases", "gold_target"])
def test_gold_rejected_recursively(key):
    with pytest.raises(ValueError, match="Gold"):
        require_gold_free({"outer": [{key: "never inspect this value"}]})


def test_wrapper_changes_only_explicit_provenance_and_denies_clearance():
    parent = {"experiment_id": "parent", "source_credit_scope": "reward_credit_only_input_unchanged",
              "weights": [1., 2., 3., 4., 5., 6.], "bias": 7.,
              "feature_standardization": {"mean": {"a": .1}, "scale": {"a": .2}},
              "normalization": {"text_v2": {"text_center": .3}, "fixed_alpha": .4, "graph_scale": .5},
              "source_credit_mask": {"path": "old"}, "training_clearance": False,
              "independent_confirmation_clearance": False, "ppo_launch_clearance": False}
    parent["payload_sha256"] = canonical_sha256(parent)
    before = deepcopy(parent)
    wrapper = make_wrapper(parent, {"path": "new"}, {"parent_gate": {"path": "parent"}}, "features_v2")
    assert parent == before
    assert {key: value for key, value in wrapper.items() if key not in WRAPPER_ALLOWED} == {
        key: value for key, value in parent.items() if key not in WRAPPER_ALLOWED}
    for field in ("training_clearance", "independent_confirmation_clearance", "ppo_launch_clearance"):
        assert wrapper[field] is False
    digest = wrapper.pop("payload_sha256")
    assert canonical_sha256(wrapper) == digest
