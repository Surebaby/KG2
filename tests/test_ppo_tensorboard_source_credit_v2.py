"""Read actual TensorBoard events for softsign and all six gate features."""
import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from kgproweight.training.ppo_tensorboard import log_ppo_batch
from tests.test_ppo_tensorboard import _events, _row
from tests.test_source_credit_ppo_runtime_v2 import _gate, _reward
from tests.test_source_gated_ppo_reward_v1 import _score


def test_real_v2_reward_events_do_not_label_softsign_tails_as_clipping(tmp_path):
    gate,spec,_old=_gate(tmp_path/'mask')
    row=_score(_reward(gate),spec)
    original=row['token_rewards'].clone()
    row['source_gate']['features']['values'].update(source_edge_coverage=.75,min_step_citation_precision=.5)
    dest=tmp_path/'events'
    with SummaryWriter(str(dest)) as writer:
        log_ppo_batch(writer,step=12,update_index=1,stats={},reward_infos=[row])
    events=_events(dest)
    for prefix in ['reward/all','reward/m_graph/1','reward/dataset/2wikimultihopqa']:
        assert events.Scalars(prefix+'/text_clip_frac')[0].value==0
        assert events.Scalars(prefix+'/text_raw_z_outside_unit_frac')[0].value==pytest.approx(2/3)
        assert events.Scalars(prefix+'/text_softsign_saturation_frac')[0].value==0
    assert events.Scalars('gate/eligible_valid/feature_source_edge_coverage_mean')[0].value==.75
    assert events.Scalars('gate/eligible_valid/feature_min_step_citation_precision_mean')[0].value==.5
    assert torch.equal(row['token_rewards'],original)


def test_mixed_versions_keep_step_denominators_and_legacy_clip_values(tmp_path):
    old=_row()
    new=_row()
    new['source_gate']['text_normalization_v2']={
        'hard_clip_frac':0.,'soft_saturation_frac':0.,'raw_z_outside_unit_frac':.5}
    with SummaryWriter(str(tmp_path)) as writer:
        log_ppo_batch(writer,step=4,update_index=1,stats={},reward_infos=[old,new])
    events=_events(tmp_path)
    assert events.Scalars('reward/all/text_clip_frac')[0].value==.25
    assert events.Scalars('reward/all/text_raw_z_outside_unit_frac')[0].value==.5


def test_probe_middle_graph_batch_writes_six_feature_histograms_then_keeps_sparse_cadence(tmp_path):
    """The real fixed probe order is H4, W4, M4; graph first appears at step8."""
    graph = _row(dataset="2wikimultihopqa")
    graph["source_gate"]["features"]["values"].update(source_edge_coverage=.75, min_step_citation_precision=.5)
    rows = [_row(dataset="hotpotqa", mask=0), graph, _row(dataset="musique", mask=0), graph]
    with SummaryWriter(str(tmp_path)) as writer:
        for index, row in enumerate(rows, 1):
            log_ppo_batch(writer, step=4*index, update_index=index, stats={}, reward_infos=[row])
        log_ppo_batch(writer, step=40, update_index=10, stats={}, reward_infos=[graph])
    events = _events(tmp_path)
    features = ("density", "link_confidence", "cite_any", "cite_match", "source_edge_coverage", "min_step_citation_precision")
    for tag in ["gate/eligible_valid/alpha_predicted_distribution"] + [
            f"gate/eligible_valid/feature_{feature}_distribution" for feature in features]:
        assert [point.step for point in events.Histograms(tag)] == [8, 40]
    assert [point.step for point in events.Scalars("gate/eligible_valid/alpha_predicted_mean")] == [8, 16, 40]


def test_explicit_histogram_disable_also_disables_initial_probe_histograms(tmp_path):
    with SummaryWriter(str(tmp_path)) as writer:
        for index in range(1, 4):
            log_ppo_batch(writer, step=4*index, update_index=index, stats={}, reward_infos=[_row()], histogram_every=0)
    assert not _events(tmp_path).Tags()["histograms"]
