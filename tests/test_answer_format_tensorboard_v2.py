"""Read objective decomposition from real events, preserving group denominators."""
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from kgproweight.training.ppo_tensorboard import log_ppo_batch


def test_shortfall_answer_and_severe_penalty_are_separate_curves(tmp_path):
    cases = [(True, "valid_legacy_preserved", 4.4, 0., True),
             (False, "format_invalid_answer_retained", 4.4, -1., True),
             (False, "invalid_answer_unavailable", 0., -4., False)]
    rows = [{"trajectory_valid": valid,
             "mixed_reward": {"dataset": "musique", "outcome": answer+penalty, "text": 0., "process": 0., "total": answer+penalty},
             "answer_format_reward": {"version": "v2", "case": case, "answer_component": answer,
                 "format_component": penalty, "answer_signal_applied": eligible,
                 "canonical_em": 1., "canonical_f1": 1.},
             "source_gate": {"m_graph": 0, "alpha_effective": 0., "alpha_predicted": 0.}}
            for valid, case, answer, penalty, eligible in cases]
    with SummaryWriter(str(tmp_path)) as writer:
        log_ppo_batch(writer, step=12, update_index=1, stats={}, reward_infos=rows)
    events = EventAccumulator(str(tmp_path), size_guidance={"scalars": 0}).Reload()
    for group in ("reward/all", "reward/dataset/musique", "reward/m_graph/0", "reward/dataset/musique/m_graph/0"):
        scalar = lambda name: events.Scalars(group+"/"+name)[0].value
        assert scalar("valid_rate") == pytest.approx(1/3)
        assert scalar("shortfall_salvage_rate") == pytest.approx(1/3)
        assert scalar("severe_invalid_rate") == pytest.approx(1/3)
        assert scalar("answer_signal_applied_rate") == pytest.approx(2/3)
        assert scalar("answer_component_mean") == pytest.approx(8.8/3)
        assert scalar("format_component_mean") == pytest.approx(-5/3)
        assert scalar("outcome_mean") == pytest.approx(3.8/3)
        assert scalar("canonical_em_mean") == 1.
        assert scalar("canonical_f1_mean") == 1.
