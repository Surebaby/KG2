"""Read real CPU TensorBoard events; no model loading or PPO updates."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from kgproweight.training.ppo_tensorboard import (
    log_ppo_batch,
    log_ppo_stats,
    log_run_metadata,
)


def _events(path):
    return EventAccumulator(str(path), size_guidance={"scalars": 0, "histograms": 0}).Reload()


def _row(dataset="2wiki", mask=1, valid=True, alpha=.8, em=1.0, f1=1.0):
    return {
        "trajectory_valid": valid,
        "proofkg_process": {"outcome_em": em, "outcome_f1": f1},
        "mixed_reward": {
            "dataset": dataset, "outcome": 4.4 if valid else -4,
            "text": .03 if valid else 0, "process": .1 if mask and valid else 0,
            "total": 4.53 if valid else -4,
            "text_raw_step_scores": [.2, .3] if valid else [],
            "text_centered_clipped_step_scores": [-1., .5] if valid else [],
        },
        "source_gate": {
            "m_graph": mask, "alpha_effective": alpha if mask and valid else 0,
            "alpha_predicted": alpha if valid else 0, "invalid_not_scored": not valid,
            "graph_raw": .6 if valid else 0, "graph_normalized": 1 if valid else 0,
            "graph_normalized_unclipped": 1.5 if valid else 0,
            "text_normalized_unclipped_steps": [-1.5, .5] if valid else [],
            "features": {"values": {"density": .4, "link_confidence": .9,
                                      "cite_any": 1., "cite_match": .8}},
        },
    }


def test_real_event_has_official_stats_and_safe_raw_distributions(tmp_path):
    stats = {"ppo/loss/policy": torch.tensor(.125, requires_grad=True),
             "objective/kl": np.float64(2.),
             "ppo/policy/advantages": np.array([[-1., 2., np.nan, np.inf]]),
             "broken": float("nan"), "empty": [], "label": "not numeric",
             "nested": {"value": torch.tensor(1.5, dtype=torch.bfloat16)}}
    with SummaryWriter(str(tmp_path)) as writer:
        log_ppo_stats(writer, stats, step=4, histograms=True)
    events = _events(tmp_path)
    assert events.Scalars("ppo/loss/policy")[0].value == pytest.approx(.125)
    assert events.Scalars("objective/kl")[0].step == 4
    assert events.Scalars("ppo/policy/advantages/raw_mean")[0].value == .5
    assert events.Histograms("ppo/policy/advantages/raw_histogram")[0].histogram_value.num == 2
    assert events.Scalars("telemetry/nonfinite/ppo/policy/advantages")[0].value == 2
    assert events.Scalars("telemetry/nonfinite/broken")[0].value == 1
    assert events.Scalars("nested/value")[0].value == 1.5
    assert "broken" not in events.Tags()["scalars"]
    assert "empty" not in events.Tags()["scalars"]
    assert "label" not in events.Tags()["scalars"]


def test_gate_and_dataset_curves_preserve_denominators(tmp_path):
    rows = [_row(), _row(valid=False, em=0, f1=.5),
            _row(dataset="hotpotqa", mask=0, em=0, f1=.25)]
    with SummaryWriter(str(tmp_path)) as writer:
        log_ppo_batch(writer, step=12, update_index=1, stats={}, reward_infos=rows)
    events = _events(tmp_path)
    assert events.Scalars("reward/all/valid_rate")[0].value == pytest.approx(2/3)
    assert events.Scalars("reward/all/em_mean")[0].value == pytest.approx(1/3)
    assert events.Scalars("reward/dataset/2wiki/em_mean")[0].value == .5
    assert events.Scalars("reward/dataset/2wiki/m_graph/1/f1_mean")[0].value == .75
    assert events.Scalars("gate/all/alpha_effective_mean")[0].value == pytest.approx(.8/3)
    assert events.Scalars("gate/eligible/alpha_effective_mean")[0].value == pytest.approx(.4)
    assert events.Scalars("gate/eligible_valid/alpha_predicted_mean")[0].value == pytest.approx(.8)
    assert events.Scalars("reward/m_graph/1/graph_clip_frac")[0].value == 1
    assert events.Scalars("reward/all/text_clip_frac")[0].value == .5
    assert "reward/m_graph/0/graph_raw_mean" not in events.Tags()["scalars"]
    assert events.Histograms("gate/eligible_valid/alpha_predicted_distribution")[0].histogram_value.num == 1


def test_no_eligible_rows_do_not_fabricate_gate_means(tmp_path):
    with SummaryWriter(str(tmp_path)) as writer:
        log_ppo_batch(writer, step=16, update_index=4, stats={},
                      reward_infos=[_row(mask=0)], histogram_every=10)
    events = _events(tmp_path)
    assert events.Scalars("gate/all/eligible_count")[0].value == 0
    assert events.Scalars("gate/all/alpha_effective_mean")[0].value == 0
    assert not any(tag.startswith("gate/eligible/") for tag in events.Tags()["scalars"])
    assert not events.Tags()["histograms"]


def test_all_invalid_graph_cohort_has_no_scored_predictions(tmp_path):
    with SummaryWriter(str(tmp_path)) as writer:
        log_ppo_batch(writer, step=4, update_index=1, stats={},
                      reward_infos=[_row(valid=False)])
    events = _events(tmp_path)
    assert events.Scalars("gate/eligible/alpha_effective_mean")[0].value == 0
    assert events.Scalars("gate/all/eligible_valid_count")[0].value == 0
    assert not any(tag.startswith("gate/eligible_valid/") for tag in events.Tags()["scalars"])
    assert "reward/all/graph_raw_mean" not in events.Tags()["scalars"]
    assert "reward/all/text_raw_step_mean" not in events.Tags()["scalars"]


def test_metadata_records_parameters_and_redacts_explicit_secret_fields(tmp_path):
    @dataclass
    class Config:
        learning_rate: float = 1e-6
        max_new_tokens: int = 384
        output_dir: Path = Path("checkpoints/test")

    with SummaryWriter(str(tmp_path)) as writer:
        log_run_metadata(writer, Config(), {"experiment_id": "TEST_ONLY", "auth": {
            "password": "do-not-log", "api_key": "also-secret"}})
    events = _events(tmp_path)
    assert events.Scalars("config/max_new_tokens")[0].value == 384
    metadata = events.Tensors("run/metadata/text_summary")[0].tensor_proto.string_val[0].decode()
    assert "TEST_ONLY" in metadata and "[REDACTED]" in metadata
    assert "do-not-log" not in metadata and "also-secret" not in metadata
    assert not list(tmp_path.glob("*/events.out.tfevents.*"))


def test_negative_histogram_interval_rejected():
    with pytest.raises(ValueError, match="histogram_every"):
        log_ppo_batch(None, step=0, stats={}, reward_infos=[], histogram_every=-1)


def test_production_reward_info_reaches_real_events(tmp_path):
    # Exercise the real RewardFunction output contract with CPU-only synthetic
    # scorer fixtures; this creates neither model updates nor research metrics.
    from tests.test_source_gated_ppo_reward_v1 import _reward, _score, _spec
    from tests.test_ppo_emf1_reward_contract_v1 import _response

    rows = [
        _score(_reward()),
        _score(_reward(), _spec(eligible=False), _response("Gamma Ray")),
        _score(_reward(), response=_response().replace("Reasoning:", "Missing:", 1)),
    ]
    token_rewards_before = [row["token_rewards"].clone() for row in rows]
    with SummaryWriter(str(tmp_path)) as writer:
        log_ppo_batch(writer, step=12, update_index=1, stats={"objective/kl": .01},
                      reward_infos=rows)
    events = _events(tmp_path)
    assert events.Scalars("reward/all/em_mean")[0].value == pytest.approx(1 / 3)
    assert events.Scalars("reward/all/f1_mean")[0].value == pytest.approx(5 / 9)
    assert events.Scalars("reward/all/valid_rate")[0].value == pytest.approx(2 / 3)
    assert events.Scalars("reward/dataset/2wikimultihopqa/m_graph/0/f1_mean")[0].value == pytest.approx(2 / 3)
    assert events.Scalars("gate/all/alpha_effective_mean")[0].value == pytest.approx(1 / 6)
    assert events.Scalars("gate/eligible/alpha_effective_mean")[0].value == pytest.approx(.25)
    assert events.Scalars("gate/eligible_valid/alpha_predicted_mean")[0].value == pytest.approx(.5)
    assert events.Scalars("reward/all/text_clip_frac")[0].value == pytest.approx(2 / 3)
    assert events.Histograms("reward/all/text_raw_step_distribution")[0].histogram_value.num == 6
    assert events.Histograms("reward/m_graph/1/graph_raw_distribution")[0].histogram_value.num == 1
    assert events.Scalars("reward/all/graph_component_mean")[0].value == pytest.approx(
        rows[0]["mixed_reward"]["process"] / 3
    )
    assert all(torch.equal(row["token_rewards"], previous)
               for row, previous in zip(rows, token_rewards_before))
