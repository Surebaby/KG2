"""CPU-only synthetic contracts; no research model is loaded or updated."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import kgproweight.training.reward_function as reward_module
import kgproweight.training.phase3_ppo as ppo
from kgproweight.data.parsers import parse_steps
from kgproweight.kg.training_question_kg import apply_training_question_kg
from kgproweight.reward import source_quality_gate_v1 as gate_module
from kgproweight.training.reward_function import KGProWeightRewardFunction, RewardSpec
from scripts.prepare.resolve_phase3_ppo_runtime_config import resolve_phase3_ppo_runtime_config
from tests.test_ppo_emf1_reward_contract_v1 import _Tokenizer, _ForbiddenComponent, _response
from tests.test_trajectory_source_gate import _record


def _gate():
    data = {
        "schema_version": gate_module.ARTIFACT_SCHEMA,
        "gate_version": gate_module.GATE_VERSION,
        "feature_version": gate_module.FEATURE_VERSION,
        "feature_names": list(gate_module.FEATURE_NAMES),
        "target_version": gate_module.TARGET_VERSION,
        "bank_source": "synthetic_unit_fixture",
        "training_clearance": False,
        "weights": [0., 0., 0., 0.], "bias": 0.,
        "feature_standardization": {
            "mean": dict.fromkeys(gate_module.FEATURE_NAMES, 0.),
            "scale": dict.fromkeys(gate_module.FEATURE_NAMES, 1.),
        },
        "normalization": {
            "graph_center": .2, "graph_scale": .4,
            "text_center": .2, "text_scale": .2, "fixed_alpha": .25,
            "input_contract": "raw_v23_graph_score_and_mean_raw_rearag_step_scores",
            "text_application_scope": "step_normalize_then_clip_then_mean_v1",
            "graph_application_scope": "trajectory_normalize_then_clip_v1",
            "application_clip": [-1, 1],
        },
    }
    data["payload_sha256"] = gate_module.canonical_sha256(data)
    return gate_module.SourceQualityGateV1(data, allow_synthetic=True, allow_unvalidated=True)


class _Text:
    name = "rearag"
    is_dummy = False

    def __init__(self, scores=(.8, .3, -.1), limit=4096):
        self.backend = SimpleNamespace(tokenizer=_Tokenizer(), max_length=limit)
        self.scores = scores
        self.calls = []

    def score_steps(self, prompts, texts):
        self.calls.append((list(prompts), list(texts)))
        return self.scores[:len(texts)]


def _spec(eligible=True):
    query, record = _record()
    record["provenance"]["historical_cutoff"] = "2020-12-09T23:59:59Z"
    if not eligible:
        record["provenance"]["gold_access"] = True
    return RewardSpec(
        query=query, gold_answer="Gamma", gold_answer_aliases=["Gamma"],
        kg_subgraph=[tuple(row) for row in record["kg_subgraph"]],
        retrieved_passages=[{"title": "Document", "text": "TEXT_ONLY_SENTINEL"}],
        metadata={"dataset": "2wikimultihopqa", "qid": "q1",
                  "source_quality_record": record, "question_kg_runtime": {}},
    )


def _reward(mode="learned", text=None, gate=None):
    return KGProWeightRewardFunction(
        alpha_gate=_ForbiddenComponent("legacy alpha"),
        prm_annotator=_ForbiddenComponent("PRM"),
        text_reward_model=text or _Text(), tokenizer=_Tokenizer(),
        outcome_weight=4., text_reward_scale=.3, max_steps=5,
        proofkg_process_reward=True, proofkg_process_version="v2_3",
        proofkg_process_weight=.2, proofkg_f1_weight=.1,
        proofkg_dynamic_validity=True, mixed_outcome_reward=True,
        mixed_text_reward=True, runtime_contract_version="v2",
        source_gated_reward_version="v1", source_gate_mode=mode,
        source_quality_gate=gate or _gate(), center_text_reward=False,
    )


def _score(fn, spec=None, response=None):
    response = response or _response()
    ids = _Tokenizer()(response)["input_ids"] + [0]
    return fn("", response, spec or _spec(), response_ids=ids)


@pytest.mark.parametrize(("mode", "alpha"), [("text", 0.), ("fixed", .25), ("learned", .5)])
def test_tfa_share_raw_signals_but_apply_frozen_source_weights(mode, alpha):
    fn = _reward(mode)
    result = _score(fn)
    details = result["source_gate"]
    assert details["m_graph"] == 1
    assert details["alpha_effective"] == alpha
    assert details["graph_normalized"] == pytest.approx(min(1., (details["graph_raw"] - .2) / .4))
    # [.8,.3,-.1] -> [3,.5,-1.5] -> [1,.5,-1], preserving each step.
    assert details["text_normalized"] == pytest.approx(1 / 6)
    assert result["mixed_reward"]["text"] == pytest.approx(.3 * (1-alpha) / 6)
    assert result["mixed_reward"]["process"] == pytest.approx(.2 * alpha * details["graph_normalized"])
    assert result["mixed_reward"]["outcome"] == pytest.approx(4.4)
    assert result["trajectory_reward"] == pytest.approx(sum(result["per_step_rewards"]))
    assert result["trajectory_reward"] == pytest.approx(float(result["token_rewards"].sum()), abs=1e-6)
    assert result["token_rewards"][-1] == pytest.approx(4.4 + result["mixed_reward"]["process"])
    for (_, end), value in zip(result["text_step_spans"], result["mixed_reward"]["text_weighted_step_rewards"]):
        assert result["token_rewards"][end-1] == pytest.approx(value)
    assert fn.composite.text_baseline_n_obs == 0


@pytest.mark.parametrize("mode", ["text", "fixed", "learned"])
def test_hardgate_zero_blocks_graph_scorer_in_every_arm(monkeypatch, mode):
    monkeypatch.setattr(reward_module, "score_proofkg_v2_3", _ForbiddenComponent("Graph"))
    result = _score(_reward(mode), _spec(eligible=False))
    assert result["source_gate"]["m_graph"] == 0
    assert result["source_gate"]["alpha_effective"] == 0
    assert result["mixed_reward"]["process"] == 0
    assert result["mixed_reward"]["text"] == pytest.approx(.05)


@pytest.mark.parametrize("mode", ["text", "fixed", "learned"])
def test_invalid_is_minus_four_without_any_process_scoring(monkeypatch, mode):
    monkeypatch.setattr(reward_module, "score_proofkg_v2_3", _ForbiddenComponent("Graph"))
    gate = _gate()
    monkeypatch.setattr(gate, "predict", _ForbiddenComponent("source alpha"))
    text = _Text()
    monkeypatch.setattr(text, "score_steps", _ForbiddenComponent("Text"))
    result = _score(_reward(mode, text, gate), response=_response().replace("Reasoning:", "Missing:", 1))
    assert result["trajectory_valid"] is False
    assert result["trajectory_reward"] == -4
    assert torch.count_nonzero(result["token_rewards"]) == 1
    assert result["token_rewards"][-1] == -4
    assert result["mixed_reward"]["text"] == result["mixed_reward"]["process"] == 0


def test_passage_only_causal_prefix_and_preflight_budget():
    text = _Text()
    result = _score(_reward(text=text))
    prompts, texts = text.calls[0]
    assert "TEXT_ONLY_SENTINEL" in prompts[0]
    assert "(Alpha, links to, Beta)" not in prompts[0]
    assert texts[0] not in prompts[0] and texts[0] in prompts[1]
    assert texts[1] not in prompts[1] and texts[1] in prompts[2]
    assert "[Final Answer]" not in "".join(texts)
    assert result["source_gate"]["token_budget"]["truncated_tokens"] == 0
    assert len(result["source_gate"]["token_budget"]["step_lengths"]) == 3
    too_short = _Text(limit=10)
    with pytest.raises(RuntimeError, match="implicit truncation forbidden"):
        _score(_reward(text=too_short))
    assert too_short.calls == []


@pytest.mark.parametrize("scores", [(float("nan"), 0., 0.), (2., 0., 0.), (.1,)])
def test_text_score_count_finiteness_and_range_fail_hard(scores):
    with pytest.raises(RuntimeError, match="one finite"):
        _score(_reward(text=_Text(scores)))


def test_source_diagnostics_use_scaled_residuals_and_preserve_features():
    result = _score(_reward())
    diagnostics = ppo._mixed_reward_dataset_diagnostics([result])["2wikimultihopqa"]
    assert diagnostics["text_clip_frac"] == pytest.approx(2 / 3)
    assert diagnostics["text_centered_unclipped_step_mean"] == pytest.approx(2 / 3)
    batch = ppo._source_gate_batch_diagnostics([result])
    assert batch["source_gate_alpha_effective_mean"] == .5
    assert batch["source_gate_records"][0]["features"]["telemetry"]["policy_entropy_used"] is False


@pytest.mark.parametrize("eligible", [True, False])
@pytest.mark.parametrize("n_steps", [2, 3, 6])
def test_bank_format_helper_exactly_matches_runtime(eligible, n_steps):
    spec = _spec(eligible)
    response = _response(steps=n_steps)
    result = _score(_reward(text=_Text((.2,) * 6)), spec, response)
    shared = reward_module.validate_source_gate_trajectory_v1(spec, response)
    assert shared["valid"] == result["trajectory_valid"]
    assert shared["violations"] == result["format_contract_violations"]
    assert shared["required_steps"] == (2 if eligible else 3)


def test_runtime_requires_actual_frozen_rearag_backend():
    from kgproweight.reward.text_reward_model import RearagPromptScorer, TextRewardModel
    cfg = ppo.Phase3PPOConfig(silver_path="unused", output_dir="unused", source_gated_reward_version="v1", mixed_text_reward=True)
    with pytest.raises(RuntimeError, match="actual RearagPromptScorer"):
        ppo._assert_mixed_text_backend(cfg, _Text())
    model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
    backend = RearagPromptScorer(model, _Tokenizer(), device="cpu")
    wrapped = TextRewardModel(backend, name="rearag")
    ppo._assert_mixed_text_backend(cfg, wrapped)
    model.train()
    with pytest.raises(RuntimeError, match="frozen eval-mode"):
        ppo._assert_mixed_text_backend(cfg, wrapped)
    model.eval().requires_grad_(True)
    with pytest.raises(RuntimeError, match="frozen eval-mode"):
        ppo._assert_mixed_text_backend(cfg, wrapped)


def test_original_record_is_preserved_without_synthesizing_legacy_identity():
    spec = _spec()
    original = spec.metadata["source_quality_record"]
    traj = SimpleNamespace(dataset="2wikimultihopqa", qid="q1", question=spec.query,
                           kg_subgraph=[], metadata={})
    apply_training_question_kg([traj], {"2wikimultihopqa::q1": original})
    assert traj.metadata["source_quality_record"] == original
    assert "schema_version" not in traj.metadata["question_kg_runtime"]
    traj.metadata["source_quality_record"]["provenance"]["gold_access"] = True
    assert original["provenance"]["gold_access"] is False


def test_all_nine_fullmethod_configs_forward_new_contract():
    paths = sorted(Path("configs/training").glob("phase3_ppo_mixed4_sourcegate_v1_*_seed42.yaml"))
    assert len(paths) == 9
    for path in paths:
        cfg = ppo.Phase3PPOConfig(**resolve_phase3_ppo_runtime_config(path))
        ppo._validate_mixed_reward_config(cfg)
        assert cfg.source_gated_reward_version == "v1"
        assert cfg.source_gate_mode == {"a": "learned", "f": "fixed", "t": "text"}[path.name.split("_v1_")[1][0]]
        assert cfg.source_gate_calibration_path.endswith("source_quality_gate_v1_seed42/gate.json")
        assert cfg.alpha_gate_path is None and cfg.alpha_override is None
        assert cfg.runtime_contract_version == "v2" and cfg.gamma == 1.
        assert cfg.proofkg_process_reward and cfg.proofkg_process_version == "v2_3"
        assert cfg.mixed_outcome_reward and cfg.mixed_text_reward
        assert cfg.center_text_reward is False


def test_missing_artifact_fails_before_any_model_allocation(monkeypatch, tmp_path):
    cfg = ppo.Phase3PPOConfig(**resolve_phase3_ppo_runtime_config(
        "configs/training/phase3_ppo_mixed4_sourcegate_v1_a_probe_seed42.yaml"))
    cfg.source_gate_calibration_path = str(tmp_path / "missing.json")
    monkeypatch.setattr(ppo, "_build_models", _ForbiddenComponent("policy load"))
    with pytest.raises(FileNotFoundError):
        ppo.run_phase3_ppo(cfg)


@pytest.mark.parametrize("mutation", [
    {"source_gate_calibration_path": None}, {"center_text_reward": True},
    {"runtime_contract_version": "legacy"}, {"mixed_text_reward": False},
    {"proofkg_process_reward": False}, {"proofkg_process_version": "v2_2"},
    {"source_gate_mode": "old_alpha"}, {"text_reward_backend": "dummy"},
])
def test_incompatible_source_config_is_rejected(mutation):
    cfg = ppo.Phase3PPOConfig(**resolve_phase3_ppo_runtime_config(
        "configs/training/phase3_ppo_mixed4_sourcegate_v1_a_probe_seed42.yaml"))
    with pytest.raises(ValueError):
        ppo._validate_mixed_reward_config(replace(cfg, **mutation))
