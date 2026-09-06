from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace

import pytest

from kgproweight.data.parsers import ParsedStep, parse_steps
from kgproweight.kg.question_kg import make_question_kg_record
from kgproweight.reward.proofkg_process_v2_3 import SCORER_VERSION
from kgproweight.reward.source_quality_gate_v1 import (
    FEATURE_NAMES as V1_NAMES, compute_gate_features,
)
from kgproweight.reward.source_trajectory_features_v2 import (
    FEATURE_NAMES, FEATURE_VERSION, compute_gate_features_v2,
)


def _fixture():
    triples = [("Alpha", "links", "Beta"), ("Beta", "links", "Gamma")]
    plan = {"recognized": True, "hops": [
        {"subject": "Alpha", "output_slot": "hop_1", "pids": ["P1"], "relation_role": "bridge"},
        {"subject": "$hop_1", "output_slot": "hop_2", "pids": ["P2"], "relation_role": "answer_operand"},
    ]}
    record = make_question_kg_record(
        dataset="2wikimultihopqa", qid="synthetic-features-v2", question="Where does Alpha lead?",
        triples=triples, query_plan=plan,
        provenance={"builder_version": "synthetic-test", "gold_access": False,
                    "complete_plan_execution": True, "historical_cutoff": "2020-12-09T23:59:59Z"},
    )
    record["execution"] = {"complete_plan_execution": True, "hops": [
        {"hop_index": index, "input_entities": [{"qid": f"Q{index}", "score": .8}], "matches": [triple]}
        for index, triple in enumerate(triples, 1)
    ]}
    spec = SimpleNamespace(query=record["question"], kg_subgraph=triples,
                           metadata={"dataset": record["dataset"], "qid": record["qid"],
                                     "source_quality_record": record})
    steps = [ParsedStep(index=i, raw_text="Unused text", cited_triples=[triple],
                        knowledge_used_field_count=1, knowledge_used_valid=True)
             for i, triple in enumerate(triples, 1)]
    return spec, steps, {"scorer_version": SCORER_VERSION}


def test_registered_six_dimensions_keep_legacy_values_and_hard_gate():
    spec, steps, proof = _fixture()
    old = compute_gate_features(spec, steps, proof)
    result = compute_gate_features_v2(spec, steps, proof)
    assert FEATURE_NAMES == V1_NAMES + ("source_edge_coverage", "min_step_citation_precision")
    assert result["feature_version"] == FEATURE_VERSION
    assert tuple(result["values"]) == FEATURE_NAMES
    assert {name: result["values"][name] for name in V1_NAMES} == old["values"]
    assert result["hard_gate"] == old["hard_gate"]
    assert result["m_graph"] == old["m_graph"] == 1
    assert result["values"]["source_edge_coverage"] == 1
    assert result["values"]["min_step_citation_precision"] == 1


@pytest.mark.parametrize("mutation", ["repeat_citation", "repeat_step", "repeat_all_steps", "reorder_steps"])
def test_valid_repetition_and_reordering_do_not_change_any_feature(mutation):
    spec, steps, proof = _fixture()
    original = compute_gate_features_v2(spec, steps, proof)["values"]
    if mutation == "repeat_citation":
        steps[0].cited_triples *= 19
    elif mutation == "repeat_step":
        steps.append(deepcopy(steps[0]))
    elif mutation == "repeat_all_steps":
        steps *= 4
    else:
        steps.reverse()
    assert compute_gate_features_v2(spec, steps, proof)["values"] == original


def test_missing_visible_edge_is_distinguished_from_repeated_subset_with_same_v1():
    spec, steps, proof = _fixture()
    full = compute_gate_features_v2(spec, steps, proof)
    steps[1].cited_triples = list(steps[0].cited_triples)
    subset = compute_gate_features_v2(spec, steps, proof)
    assert {k: full["values"][k] for k in V1_NAMES} == {k: subset["values"][k] for k in V1_NAMES}
    assert subset["values"]["source_edge_coverage"] == .5
    assert subset["values"]["min_step_citation_precision"] == 1


def test_unsupported_step_is_distinguished_despite_same_global_citation_set():
    spec, steps, proof = _fixture()
    full = compute_gate_features_v2(spec, steps, proof)
    steps[0].cited_triples.extend(steps[1].cited_triples)
    steps[1].cited_triples.clear()
    empty = compute_gate_features_v2(spec, steps, proof)
    assert {k: full["values"][k] for k in V1_NAMES} == {k: empty["values"][k] for k in V1_NAMES}
    assert empty["values"]["source_edge_coverage"] == 1
    assert empty["values"]["min_step_citation_precision"] == 0


def test_unique_unknown_and_malformed_observations_lower_weakest_step_precision():
    spec, steps, proof = _fixture()
    steps[0].unknown_citation_surfaces = ["(Other, links, Nope)"] * 12
    steps[0].knowledge_used_malformed_content = True
    result = compute_gate_features_v2(spec, steps, proof)
    assert result["values"]["source_edge_coverage"] == 1
    assert result["values"]["min_step_citation_precision"] == pytest.approx(1 / 3)


def test_malformed_step_repetition_keeps_new_features_but_preserves_legacy_counting():
    spec, steps, proof = _fixture()
    steps[0].knowledge_used_malformed_content = True
    original = compute_gate_features_v2(spec, steps, proof)["values"]
    steps.append(deepcopy(steps[0]))
    repeated = compute_gate_features_v2(spec, steps, proof)["values"]
    assert repeated["source_edge_coverage"] == original["source_edge_coverage"]
    assert repeated["min_step_citation_precision"] == original["min_step_citation_precision"]
    assert repeated["cite_match"] != original["cite_match"]
    assert repeated["cite_match"] == compute_gate_features(spec, steps, proof)["values"]["cite_match"]


def test_cited_tuple_outside_visible_graph_never_counts_as_supported():
    spec, steps, proof = _fixture()
    steps[0].cited_triples.append(("Other", "links", "Nope"))
    result = compute_gate_features_v2(spec, steps, proof)
    assert result["values"]["source_edge_coverage"] == 1
    assert result["values"]["min_step_citation_precision"] == .5


@pytest.mark.parametrize("empty", ["steps", "graph", "citations"])
def test_empty_support_never_gets_perfect_new_features(empty):
    spec, steps, proof = _fixture()
    if empty == "steps":
        steps = []
    elif empty == "graph":
        spec.kg_subgraph = []
    else:
        for step in steps:
            step.cited_triples = []
    result = compute_gate_features_v2(spec, steps, proof)
    assert result["values"]["source_edge_coverage"] == 0
    assert result["values"]["min_step_citation_precision"] == 0
    if empty == "graph":
        assert result["m_graph"] == 0


@pytest.mark.parametrize("mutation", ["identity", "visible_graph", "missing_record"])
def test_new_features_do_not_repair_a_failed_hard_gate(mutation):
    spec, steps, proof = _fixture()
    if mutation == "identity":
        spec.metadata["qid"] = "different-request"
    elif mutation == "visible_graph":
        spec.kg_subgraph = spec.kg_subgraph[:1]
    else:
        spec.metadata.pop("source_quality_record")
    result = compute_gate_features_v2(spec, steps, proof)
    assert result["m_graph"] == 0
    assert result["hard_gate"] == compute_gate_features(spec, steps, proof)["hard_gate"]


class _VersionOnlyProof(Mapping):
    def __getitem__(self, key):
        if key != "scorer_version":
            raise AssertionError(f"forbidden proof field read: {key}")
        return SCORER_VERSION

    def __iter__(self):
        raise AssertionError("proof result must never be copied or enumerated")

    def __len__(self):
        return 1


class _CitationOnlyStep(SimpleNamespace):
    def __getattribute__(self, name):
        if name in {"raw_text", "intermediate_conclusion", "mentioned_entities", "prediction", "gold_answer"}:
            raise AssertionError(f"forbidden free-text/answer field read: {name}")
        return super().__getattribute__(name)


def test_result_and_free_text_are_not_read_even_when_access_would_raise():
    spec, steps, proof = _fixture()
    expected = compute_gate_features_v2(spec, steps, proof)
    citation_only = [_CitationOnlyStep(cited_triples=s.cited_triples,
                                      unknown_citation_surfaces=s.unknown_citation_surfaces,
                                      knowledge_used_malformed_content=s.knowledge_used_malformed_content)
                     for s in steps]
    assert compute_gate_features_v2(spec, citation_only, _VersionOnlyProof()) == expected


def test_final_answer_negation_reasoning_and_passage_changes_are_explicitly_unresolved():
    spec, _, proof = _fixture()
    template = ("[Step 1]\nReasoning: {reason}\nKnowledge Used: [(Alpha, links, Beta)]\nConclusion: {conclusion}\n"
                "[Step 2]\nReasoning: {reason}\nKnowledge Used: [(Beta, links, Gamma)]\nConclusion: {conclusion}\n"
                "[Final Answer]\n{answer}")
    correct = template.format(reason="These edges give the requested chain of relations.", conclusion="Gamma follows.", answer="Gamma")
    contradicted = template.format(reason="These edges do not imply the conclusion and I ignore them.", conclusion="Gamma does not follow.", answer="Wrong")
    first = compute_gate_features_v2(spec, parse_steps(correct, known_kg=spec.kg_subgraph), proof)
    spec.retrieved_passages = [{"text": "An unrelated or repeated passage."}] * 10
    second = compute_gate_features_v2(spec, parse_steps(contradicted, known_kg=spec.kg_subgraph), proof)
    assert first == second
    assert first["telemetry"]["free_text_semantics_verified"] is False


def test_empty_proof_preserves_contract_and_wrong_explicit_version_is_rejected():
    spec, steps, proof = _fixture()
    assert compute_gate_features_v2(spec, steps, {}) == compute_gate_features_v2(spec, steps, proof)
    with pytest.raises(ValueError, match="v2.3"):
        compute_gate_features_v2(spec, steps, {"scorer_version": "wrong"})


def test_inputs_are_unchanged():
    spec, steps, proof = _fixture()
    snapshot = deepcopy((spec, steps, proof))
    compute_gate_features_v2(spec, steps, proof)
    assert (spec, steps, proof) == snapshot
