"""Frozen counterexamples for the opt-in answer/format objective."""

from __future__ import annotations

import json
import math

import pytest

from kgproweight.data.parsers import extract_final_answer
from kgproweight.reward.answer_format_objective_v2 import (
    FORMAT_PENALTY,
    SOURCE_PROCESS_ABS_BOUND,
    compose_answer_format_objective_v2,
    inspect_final_answer_v2,
    inspect_shortfall_salvage_v2,
)
from kgproweight.reward.proofkg_process import canonical_exact_match, canonical_token_f1
from kgproweight.training.reward_function import RewardSpec, validate_source_gate_trajectory_v2


def _two_steps(answer="Gamma"):
    return (
        "[Step 1]\nReasoning: The first passage connects Alpha with the bridge entity Beta.\n"
        "Knowledge Used: []\nConclusion: Alpha connects to Beta.\n"
        "[Step 2]\nReasoning: The second passage supplies the next link from Beta to Gamma.\n"
        "Knowledge Used: []\nConclusion: Beta connects to Gamma.\n"
        f"[Final Answer]\n{answer}"
    )


def _salvage(response, *, known_passage_ids=range(1, 11)):
    validation = validate_source_gate_trajectory_v2(
        RewardSpec(query="Where does Alpha connect?", gold_answer="", kg_subgraph=[]), response,
    )
    contract = inspect_shortfall_salvage_v2(
        response, steps=validation["steps"], required_steps=validation["required_steps"],
        violations=validation["violations"], known_passage_ids=known_passage_ids,
    )
    return contract, validation


def _compose(response, gold="Gamma", *, valid=False, **kwargs):
    contract, _ = _salvage(response)
    # Production scores valid responses with the existing parser.  The new
    # guard is used for invalid responses only; labels never enter that guard.
    answer = (
        (extract_final_answer(response) or "").split("\n", 1)[0].strip()
        if valid else contract.answer
    )
    return compose_answer_format_objective_v2(
        trajectory_valid=valid, salvage_contract=contract,
        outcome_em=canonical_exact_match(answer, gold),
        outcome_f1=canonical_token_f1(answer, gold), **kwargs,
    )


@pytest.mark.parametrize("response", [
    "[Final Answer]\nGamma", "[Final Answer] Gamma", "Final Answer: Gamma",
    "**Final Answer**: Gamma", "Final Answer：Gamma", "[Final Answer]\n\n Gamma ",
    "[Final Answer]\n东京", "[Final Answer]\n0", "[Final Answer]\nno",
])
def test_supported_unique_final_uses_existing_firstline(response):
    inspected = inspect_final_answer_v2(response)
    assert inspected.eligible
    assert inspected.answer == extract_final_answer(response).split("\n", 1)[0].strip()


@pytest.mark.parametrize("response,reason", [
    ("Gamma", "literal_final_count_not_one"),
    ("[Final Answer]", "empty_or_decoration_only_firstline"),
    ("[Final Answer]\n **...** ", "empty_or_decoration_only_firstline"),
    ("[Final Answer]\n...\nGamma", "empty_or_decoration_only_firstline"),
    ("[Final Answer] Gamma\n[Final Answer] Gamma", "literal_final_count_not_one"),
    ("[Final Answer] Gamma\n**Final Answer**: Other", "literal_final_count_not_one"),
    ("[Final Answer] Gamma\nFinal Answer Other", "additional_final_text_after_field"),
    ("[Final Answer]\nGamma\n[Step 4]", "step_after_final"),
    ("[Final Answer]\nGamma\n[ step 4 ]", "step_after_final"),
    ("[Final Answer]\nGamma\n### Step 4", "step_after_final"),
    ("[Final Answer]\nGamma\nStep 4:", "step_after_final"),
    ("[Final Answer]\nGamma\nStep 4.", "step_after_final"),
    ("[Final Answer]\nGamma\nStep 4)", "step_after_final"),
    ("Reasoning: Final Answer Not Gamma.\n[ Final Answer ] Gamma", "legacy_extraction_ambiguous"),
])
def test_answer_guard_rejects_empty_ambiguous_or_nonterminal_final(response, reason):
    inspected = inspect_final_answer_v2(response)
    assert not inspected.eligible
    assert inspected.reason == reason
    assert inspected.answer == ""
    result = _compose(response)
    assert result.trajectory_reward == -4.0
    assert result.answer_signal_applied is result.process_allowed is False


def test_answer_guard_does_not_search_later_lines_for_gold():
    response = "[Final Answer]\nWrong\nGamma"
    inspected = inspect_final_answer_v2(response)
    assert inspected.eligible and inspected.answer == "Wrong"
    assert _compose(_two_steps("Wrong\nGamma")).trajectory_reward == -1.0


@pytest.mark.parametrize("answer,gold,em,f1,expected", [
    ("Gamma", "Gamma", 1.0, 1.0, 3.4),
    ("New York", "New York City", 0.0, 0.8, -0.68),
    ("Unrelated", "Gamma", 0.0, 0.0, -1.0),
    ("not yes", "yes", 0.0, 0.0, -1.0),
])
def test_invalid_answer_signal_uses_frozen_canonical_outcome(answer, gold, em, f1, expected):
    assert canonical_exact_match(answer, gold) == em
    assert canonical_token_f1(answer, gold) == pytest.approx(f1)
    result = _compose(_two_steps(answer), gold)
    assert result.trajectory_reward == pytest.approx(expected)
    assert result.format_component == -1.0
    assert result.case == "format_invalid_answer_retained"


def test_invalid_never_retains_any_process_even_if_caller_passes_process_scores():
    result = _compose(_two_steps(), text_component=999.0, graph_component=999.0)
    assert result.trajectory_reward == pytest.approx(3.4)
    assert result.text_component == result.graph_component == 0.0
    assert result.process_allowed is False


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("text,graph", [(-1.0, -1.0), (-1.0, 1.0), (0.0, 0.0), (1.0, -1.0), (1.0, 1.0)])
@pytest.mark.parametrize("answer,gold", [("Gamma", "Gamma"), ("Wrong", "Gamma"), ("New York", "New York City")])
def test_same_answer_valid_dominates_invalid_for_all_process_extremes(alpha, text, graph, answer, gold):
    weighted_text, weighted_graph = 0.3 * (1.0 - alpha) * text, 0.2 * alpha * graph
    response = _two_steps(answer)
    valid = _compose(response, gold, valid=True, text_component=weighted_text, graph_component=weighted_graph)
    invalid = _compose(response, gold)
    legacy = 4.0 * (canonical_exact_match(answer, gold) + 0.1 * canonical_token_f1(answer, gold))
    # This is scalar preservation, not an assertion that Final-only is valid.
    assert valid.trajectory_reward == legacy + weighted_text + weighted_graph
    assert valid.trajectory_reward - invalid.trajectory_reward >= FORMAT_PENALTY - SOURCE_PROCESS_ABS_BOUND - 1e-12


def test_valid_branch_is_not_reclassified_by_new_answer_guard():
    # The old format validator accepts some loose/space variants.  Whichever
    # response the caller deems valid must keep its old outcome and process.
    contract, _ = _salvage("not a Final heading")
    result = compose_answer_format_objective_v2(
        trajectory_valid=True, salvage_contract=contract, outcome_em=1.0,
        outcome_f1=1.0, text_component=-0.3, graph_component=0.0,
    )
    assert result.trajectory_reward == pytest.approx(4.1)
    assert result.case == "valid_legacy_preserved"
    assert result.process_allowed


def test_correct_final_only_retains_severe_penalty_after_broad_proposal_was_rejected():
    result = _compose("[Final Answer]\nGamma")
    assert result.trajectory_reward == -4.0
    assert not result.answer_signal_applied and not result.process_allowed


def test_telemetry_and_repr_do_not_expose_answer_or_gold():
    response = "[Final Answer]\nanswer-secret-canary"
    contract = inspect_final_answer_v2(response)
    result = _compose(response, "label-secret-canary")
    rendered = json.dumps({**contract.telemetry(), **result.telemetry()}) + repr(contract) + repr(result)
    assert "answer-secret-canary" not in rendered
    assert "label-secret-canary" not in rendered
    assert "alias" not in rendered and "gold" not in rendered


@pytest.mark.parametrize("em,f1", [(math.nan, 0.0), (0.0, math.inf), (-1.0, 0.0), (0.5, 0.0), (0.0, 1.1)])
def test_noncanonical_or_nonfinite_metrics_fail(em, f1):
    with pytest.raises(ValueError):
        compose_answer_format_objective_v2(
            trajectory_valid=False, salvage_contract=_salvage(_two_steps())[0],
            outcome_em=em, outcome_f1=f1,
        )


@pytest.mark.parametrize("text,graph", [(math.nan, 0.0), (0.0, math.inf), (0.31, 0.0), (0.0, 0.21), (0.3, 0.2)])
def test_valid_process_bound_or_nonfinite_failure(text, graph):
    with pytest.raises(ValueError):
        _compose("[Final Answer] Gamma", valid=True, text_component=text, graph_component=graph)


def test_complete_two_steps_are_only_a_minimum_failure_under_frozen_validator():
    contract, validation = _salvage(_two_steps())
    assert validation["valid"] is False
    assert validation["required_steps"] == 3
    assert validation["violations"] == ["invalid_step_sequence_content_or_minimum"]
    assert contract.eligible
    assert contract.reason == "complete_two_steps_only_minimum_shortfall"
    assert _compose(_two_steps()).trajectory_reward == pytest.approx(3.4)


@pytest.mark.parametrize("mutation,reason", [
    (lambda text: text.replace("[Step 2]", "[Step 3]"), "raw_step_headers_not_exactly_one_two"),
    (lambda text: text.replace("[Final Answer]", "[Step 3]\n[Final Answer]"), "raw_step_headers_not_exactly_one_two"),
    (lambda text: text.replace("[Final Answer]", "[Step 3\n[Final Answer]"), "raw_step_headers_not_exactly_one_two"),
    (lambda text: text.replace("[Final Answer]", "### Step 3\n[Final Answer]"), "raw_step_headers_not_exactly_one_two"),
    (lambda text: text.replace("[Final Answer]", "Step 3:\n[Final Answer]"), "raw_step_headers_not_exactly_one_two"),
    (lambda text: text.replace("The first passage connects Alpha with the bridge entity Beta.", ""), "reasoning_missing_or_short"),
    (lambda text: text.replace("The first passage connects Alpha with the bridge entity Beta.", "Tiny."), "reasoning_missing_or_short"),
    (lambda text: text.replace("The first passage connects Alpha with the bridge entity Beta.", "Tiny.").replace("Knowledge Used:", "knowledge used:"), "reasoning_missing_or_short"),
    (lambda text: text.replace("Reasoning:", "reasoning:", 1), "reasoning_missing_or_short"),
    (lambda text: text.replace("Conclusion: Alpha connects to Beta.", "Conclusion: ..."), "conclusion_empty_or_decoration_only"),
    (lambda text: text.replace("Knowledge Used: []", "Knowledge Used: [(Fake, invented, Triple)]", 1), "unknown_or_malformed_kg_citation"),
    (lambda text: text.replace("Knowledge Used: []", "Knowledge Used: [Fake prose]", 1), "unknown_or_malformed_kg_citation"),
    (lambda text: text.replace("The second passage supplies the next link from Beta to Gamma.", " THE FIRST PASSAGE  CONNECTS ALPHA WITH THE BRIDGE ENTITY BETA. "), "duplicate_normalized_reasoning"),
    (lambda text: text.replace("The first passage", "Passage 11"), "unknown_explicit_passage_id"),
    (lambda text: text.replace("The first passage", "Passages 2 and 11"), "unknown_explicit_passage_id"),
    (lambda text: text.replace("The first passage", "Passages 2, 11"), "unknown_explicit_passage_id"),
    (lambda text: text.replace("The first passage", "[P11]"), "unknown_explicit_passage_id"),
    (lambda text: text.replace("The first passage", "[P2, P11]"), "unknown_explicit_passage_id"),
])
def test_shortfall_salvage_rejects_nonminimal_structure_and_citation_bypasses(mutation, reason):
    response = mutation(_two_steps())
    contract, _ = _salvage(response)
    assert not contract.eligible
    assert contract.reason == reason
    assert _compose(response).trajectory_reward == -4.0


@pytest.mark.parametrize("label", ["Reasoning", "Knowledge Used", "Conclusion"])
@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_or_duplicate_fields_are_not_salvaged(label, duplicate):
    lines = _two_steps().splitlines()
    index = next(index for index, line in enumerate(lines) if line.startswith(label + ":"))
    if duplicate:
        lines.insert(index, lines[index])
    else:
        del lines[index]
    response = "\n".join(lines)
    contract, _ = _salvage(response)
    assert not contract.eligible
    assert _compose(response).trajectory_reward == -4.0


def test_shortfall_rule_never_salvages_another_required_step_count():
    response = _two_steps()
    _, validation = _salvage(response)
    for required in (1, 2, 4, 5):
        contract = inspect_shortfall_salvage_v2(
            response, steps=validation["steps"], required_steps=required,
            violations=["invalid_step_sequence_content_or_minimum"], known_passage_ids=range(1, 11),
        )
        assert not contract.eligible and contract.reason == "required_steps_not_three"


def test_known_explicit_passage_ids_are_allowed_but_never_prove_support():
    response = _two_steps().replace("The first passage", "Passages 2 and 10")
    assert _salvage(response)[0].eligible
    assert not _salvage(response, known_passage_ids=range(1, 4))[0].eligible


def test_shortfall_contract_telemetry_does_not_expose_answer():
    contract, _ = _salvage(_two_steps("salvage-answer-canary"))
    assert contract.eligible
    assert "salvage-answer-canary" not in json.dumps(contract.telemetry()) + repr(contract)
