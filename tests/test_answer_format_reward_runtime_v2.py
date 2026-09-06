"""Independent integration checks for the opt-in shortfall-only objective.

All inputs, source masks and scorer outputs are synthetic CPU fixtures.  These
tests preserve the strict process validator and exercise the real token reward
path, including its terminal EOS token; they never load a research checkpoint.
"""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

import kgproweight.training.phase3_ppo as ppo
import kgproweight.training.reward_function as reward_module
from kgproweight.training.reward_function import KGProWeightRewardFunction
from scripts.prepare.resolve_phase3_ppo_runtime_config import resolve_phase3_ppo_runtime_config
from tests.test_ppo_emf1_reward_contract_v1 import _ForbiddenComponent, _Tokenizer
from tests.test_source_credit_ppo_runtime_v2 import _gate
from tests.test_source_gated_ppo_reward_v1 import _Text, _score


MODES = ("learned", "fixed", "text")


def _response(answer="Gamma", steps=3, cited=False):
    reasoning = (
        "The first document establishes the starting entity and its connection.",
        "The second document identifies the terminal entity from the previous connection.",
        "The retrieved connection resolves the question and supports the final answer.",
        "An additional document supplies a fourth independent descriptive statement.",
        "The fifth source supports a distinct observation about the same question.",
        "This sixth observation exceeds the configured maximum reasoning length.",
    )
    blocks = []
    for index in range(steps):
        knowledge = (
            "[(Alpha, links to, Beta)]" if cited and index == 0 else
            "[(Beta, links to, Gamma)]" if cited and index == 1 else "[]"
        )
        blocks.append(
            f"[Step {index + 1}]\nReasoning: {reasoning[index]}\n"
            f"Knowledge Used: {knowledge}\nConclusion: Observation {index + 1} is established.\n"
        )
    return "\n".join(blocks) + f"\n[Final Answer]\n{answer}"


def _reward(gate, *, mode="learned", version="v2", text=None):
    return KGProWeightRewardFunction(
        alpha_gate=_ForbiddenComponent("legacy alpha"),
        prm_annotator=_ForbiddenComponent("legacy PRM"),
        text_reward_model=text or _Text(), tokenizer=_Tokenizer(),
        outcome_weight=4.0, text_reward_scale=0.3, max_steps=5,
        proofkg_process_reward=True, proofkg_process_version="v2_3",
        proofkg_process_weight=0.2, proofkg_f1_weight=0.1,
        proofkg_dynamic_validity=True, mixed_outcome_reward=True,
        mixed_text_reward=True, runtime_contract_version="v2",
        source_gated_reward_version="v1", source_gate_format_version="v2",
        source_gate_credit_version="v2", source_gate_mode=mode,
        source_quality_gate=gate, center_text_reward=False,
        answer_format_reward_version=version,
    )


def _ordinary(spec):
    ordinary = deepcopy(spec)
    ordinary.kg_subgraph = []
    ordinary.metadata = {"dataset": "musique", "qid": "synthetic-ordinary"}
    return ordinary


def _forbid_process(monkeypatch, gate, text):
    monkeypatch.setattr(reward_module, "score_proofkg_v2_3", _ForbiddenComponent("invalid Graph"))
    monkeypatch.setattr(gate, "predict", _ForbiddenComponent("invalid alpha"))
    monkeypatch.setattr(text, "score_steps", _ForbiddenComponent("invalid Text"))


def _assert_terminal_only(result, expected):
    assert result["trajectory_valid"] is False
    assert result["trajectory_reward"] == pytest.approx(expected)
    assert result["mixed_reward"]["outcome"] == pytest.approx(expected)
    assert result["mixed_reward"]["text"] == result["mixed_reward"]["process"] == 0.0
    assert result["source_gate"]["alpha_effective"] == 0.0
    assert result["source_gate"]["invalid_not_scored"] is True
    assert torch.count_nonzero(result["token_rewards"]) == 1
    assert result["token_rewards"][-1] == pytest.approx(expected)
    assert result["token_rewards"].sum().item() == pytest.approx(expected, abs=1e-6)
    assert sum(result["per_step_rewards"]) == pytest.approx(expected)
    assert result["text_step_spans"] == []


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("status", ["PASS", "UNVERIFIED", "FAIL"])
def test_valid_reward_preserves_source_credit_alpha_and_every_existing_numeric_field(tmp_path, mode, status):
    gate, spec, _ = _gate(tmp_path, status)
    response = _response(cited=True)
    old = _score(_reward(gate, mode=mode, version="legacy"), spec, response)
    new = _score(_reward(gate, mode=mode), spec, response)
    assert new["trajectory_valid"] is True
    for name in ("trajectory_reward", "per_step_rewards", "source_gate", "proofkg_process",
                 "step_spans", "text_step_spans", "format_contract_violations"):
        assert new[name] == old[name]
    for name, value in old["mixed_reward"].items():
        assert new["mixed_reward"][name] == value
    assert torch.equal(new["token_rewards"], old["token_rewards"])
    assert new["answer_format_reward"]["format_component"] == 0.0
    assert new["answer_format_reward"]["process_allowed"] is True


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("answer", "expected_em", "expected_f1", "expected_reward"), [
    ("Gamma", 1.0, 1.0, 3.4), ("Delta", 0.0, 0.0, -1.0),
    ("Gamma Delta", 0.0, 2 / 3, -1.0 + 0.4 * 2 / 3),
])
def test_two_complete_steps_retain_answer_only_without_changing_validity_or_calling_process(
    tmp_path, monkeypatch, mode, answer, expected_em, expected_f1, expected_reward,
):
    gate, spec, _ = _gate(tmp_path)
    spec = _ordinary(spec)
    text = _Text()
    _forbid_process(monkeypatch, gate, text)
    response = _response(answer, steps=2)
    old = _score(_reward(gate, mode=mode, version="legacy", text=text), spec, response)
    new = _score(_reward(gate, mode=mode, text=text), spec, response)
    _assert_terminal_only(old, -4.0)
    _assert_terminal_only(new, expected_reward)
    assert new["format_contract_violations"] == old["format_contract_violations"]
    assert new["proofkg_process"]["required_steps"] == 3
    assert new["proofkg_process"]["outcome_em"] == expected_em
    assert new["proofkg_process"]["outcome_f1"] == pytest.approx(expected_f1)
    details = new["answer_format_reward"]
    assert details["answer_signal_applied"] is True
    assert details["format_component"] == -1.0
    assert details["process_allowed"] is False
    assert details["answer_component"] == pytest.approx(4 * (expected_em + 0.1 * expected_f1))


def test_shortfall_salvage_uses_the_frozen_alias_scoring(tmp_path):
    gate, spec, _ = _gate(tmp_path)
    spec = _ordinary(spec)
    spec.gold_answer = "Canonical terminal entity"
    spec.gold_answer_aliases = ["Gamma"]
    result = _score(_reward(gate), spec, _response(steps=2))
    _assert_terminal_only(result, 3.4)
    assert result["proofkg_process"]["outcome_em_matched_alias"] == "Gamma"
    assert result["proofkg_process"]["outcome_f1_matched_alias"] == "Gamma"


def test_correct_final_only_is_visible_in_canonical_telemetry_but_cannot_collect_reward(tmp_path, monkeypatch):
    gate, spec, _ = _gate(tmp_path)
    text = _Text()
    _forbid_process(monkeypatch, gate, text)
    result = _score(_reward(gate, text=text), _ordinary(spec), "[Final Answer]\nGamma")
    _assert_terminal_only(result, -4.0)
    assert result["proofkg_process"]["outcome_em"] == 0.0
    assert result["answer_format_reward"]["canonical_em"] == 1.0
    assert result["answer_format_reward"]["canonical_f1"] == 1.0
    assert result["answer_format_reward"]["answer_component"] == 0.0
    assert result["answer_format_reward"]["format_component"] == -4.0
    assert result["answer_format_reward"]["answer_signal_applied"] is False


def _severe_examples():
    short = _response(steps=2)
    duplicate_reasoning = short.replace(
        "The second document identifies the terminal entity from the previous connection.",
        "The first document establishes the starting entity and its connection.",
    )
    return {
        "final_only": "[Final Answer]\nGamma",
        "one_step": _response(steps=1),
        "missing_reasoning": short.replace("Reasoning:", "Missing:", 1),
        "empty_reasoning": short.replace(
            "The first document establishes the starting entity and its connection.", "", 1),
        "missing_knowledge": short.replace("Knowledge Used: []\n", "", 1),
        "missing_conclusion": short.replace("Conclusion: Observation 1 is established.\n", "", 1),
        "unknown_citation": short.replace("Knowledge Used: []", "Knowledge Used: [(Fake, links, Other)]", 1),
        "duplicate_reasoning": duplicate_reasoning,
        "duplicate_final": short + "\n[Final Answer]\nGamma",
        "empty_final": _response(answer="", steps=2),
        "too_many_steps": _response(steps=6),
        "empty_late_step": short.replace("[Final Answer]", "[Step 3]\n[Final Answer]"),
        "step_after_final": short + "\n[Step 3]",
        "repeated_step_index": short.replace("[Step 2]", "[Step 1]"),
    }


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case,response", list(_severe_examples().items()))
def test_severe_or_ambiguous_outputs_remain_minus_four_with_no_process(
    tmp_path, monkeypatch, mode, case, response,
):
    gate, spec, _ = _gate(tmp_path)
    text = _Text()
    _forbid_process(monkeypatch, gate, text)
    result = _score(_reward(gate, mode=mode, text=text), _ordinary(spec), response)
    _assert_terminal_only(result, -4.0)
    assert result["answer_format_reward"]["answer_signal_applied"] is False
    assert result["answer_format_reward"]["process_allowed"] is False


@pytest.mark.parametrize("status", ["PASS", "UNVERIFIED", "FAIL"])
def test_original_two_hop_graph_remains_valid_without_receiving_shortfall_penalty(tmp_path, status):
    gate, spec, _ = _gate(tmp_path, status)
    result = _score(_reward(gate, mode="fixed"), spec, _response(steps=2, cited=True))
    assert result["trajectory_valid"] is True
    assert result["proofkg_process"]["required_steps"] == 2
    assert result["answer_format_reward"]["format_component"] == 0.0
    assert result["source_gate"]["source_credit_mask"]["status"] == status


def test_opt_in_reaches_exact_runtime_cli_and_legacy_config_defaults(tmp_path):
    base = Path("configs/training/phase3_ppo_mixed4_source_credit_v2_features_a_probe_seed42.yaml").resolve()
    assert resolve_phase3_ppo_runtime_config(base)["answer_format_reward_version"] == "legacy"
    path = tmp_path / "objective.yaml"
    path.write_text(yaml.safe_dump({"includes": [str(base)], "training": {"ppo": {
        "answer_format_reward_version": "v2",
    }}}))
    resolved = resolve_phase3_ppo_runtime_config(path)
    assert resolved["answer_format_reward_version"] == "v2"
    ppo._validate_mixed_reward_config(ppo.Phase3PPOConfig(**resolved))


def test_unknown_objective_version_is_rejected_before_model_allocation(tmp_path, monkeypatch):
    base = Path("configs/training/phase3_ppo_mixed4_source_credit_v2_features_a_probe_seed42.yaml")
    cfg = ppo.Phase3PPOConfig(**resolve_phase3_ppo_runtime_config(base))
    cfg.answer_format_reward_version = "unknown"
    monkeypatch.setattr(ppo, "_build_models", _ForbiddenComponent("model allocation"))
    with pytest.raises(ValueError, match="answer_format_reward_version"):
        ppo.run_phase3_ppo(cfg)
