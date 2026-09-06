"""Independent CPU attacks on the manually authorized A-smoke600 boundary."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kgproweight.reward import source_gate_bounded_dispatch_v1 as dispatch
from kgproweight.reward import source_gate_smoke_scope_v1 as smoke
from kgproweight.reward.source_credit_gate_v1 import FrozenSourceCreditMask
from kgproweight.reward.source_credit_gate_v2 import ARTIFACT_SCHEMA, SourceCreditGateV2
from kgproweight.training import phase3_ppo as training
from scripts.prepare.resolve_phase3_ppo_runtime_config import resolve_phase3_ppo_runtime_config
from tests.test_source_credit_gate_v1 import _artifact, _signed, _write_release
from tests.test_source_credit_gate_v2 import _v2_artifact
from tests.test_source_gate_smoke_scope_v1 import probe_release, smoke_release


@pytest.mark.parametrize('changes', [
    {'learning_rate':2e-6}, {'sft_replay_ratio':.2}, {'sft_anchor_weight':.2},
    {'gamma':.9}, {'seed':43}, {'max_new_tokens':512},
])
def test_resigned_runtime_cannot_change_frozen_scientific_fields(smoke_release, changes):
    r=smoke_release
    cfg=replace(r.cfg,**changes)
    actual=smoke.config_identity(cfg,root=r.root)
    r.scope['runtime_config']=actual
    r.scope['runtime_config_sha256']=smoke.digest(actual)
    r.rescope()
    with pytest.raises(ValueError,match='scientific runtime configuration'):
        smoke.validate_smoke_scope(r.child,cfg,cfg.source_gate_calibration_path,r.mask,root=r.root)


@pytest.mark.parametrize('field', ['question_kg_records_path','rollout_sampling_weights_path','sft_replay_silver_path'])
def test_resigned_inputs_must_match_the_accepted_probe_source_bytes(smoke_release, field):
    r=smoke_release
    r.refs[field]=r.put(getattr(r.cfg,field),{'synthetic':'substituted at the same path'})
    r.rescope()
    with pytest.raises(ValueError,match='data changed'):
        r.validate()


@pytest.mark.parametrize('key', ['probe_training_manifest','probe_independent_lineage','probe_independent_parameters_events'])
def test_replacing_pass_label_is_insufficient_without_authorized_probe_evidence(smoke_release,key):
    r=smoke_release
    ref=r.refs[key]
    value=json.loads((r.root/ref['path']).read_text())
    value['synthetic_replacement']=True
    r.refs[key]=r.put(ref['path'],value)
    r.rescope()
    with pytest.raises(ValueError,match='probe evidence'):
        r.validate()


@pytest.mark.parametrize('field,value', [('full_ppo_clearance',True),('matched600_clearance',True),('manual_A_smoke600_clearance',False)])
def test_manual_A_permission_does_not_expand_to_matched_controls_or_full(smoke_release,field,value):
    r=smoke_release;r.scope[field]=value;r.rescope()
    with pytest.raises(ValueError):r.validate()


def test_authorized_schedule_cannot_be_shortened_and_resigned(smoke_release):
    r=smoke_release;path=r.root/r.cfg.fixed_rollout_schedule_path
    path.write_text('\n'.join(path.read_text().splitlines()[:200]))
    r.refs['schedule']={'path':r.cfg.fixed_rollout_schedule_path,'sha256':smoke.file_sha(path),'bytes':path.stat().st_size}
    r.rescope()
    with pytest.raises(ValueError,match='schedule differs'):r.validate()


def test_complete_smoke_child_routes_through_smoke_not_probe(smoke_release):
    r=smoke_release
    assert dispatch.module_for_scope(r.child) is smoke
    result=dispatch.validate_bounded_scope(r.child,r.cfg,r.cfg.source_gate_calibration_path,r.mask)
    assert result['trajectory_limit']==600 and result['health_status']=='FAIL'
    assert result['manual_A_smoke600_clearance'] is True and result['matched600_clearance'] is False


def test_ambiguous_probe_schema_and_smoke_scope_pair_cannot_dispatch(smoke_release):
    r=smoke_release;r.scope['schema_version']=dispatch.probe.SCHEMA;r.rescope()
    with pytest.raises(ValueError,match='unsupported bounded'):dispatch.module_for_scope(r.child)


def test_supervisor_cannot_be_removed_from_bound_execution_code(smoke_release):
    r=smoke_release;r.scope['code_bindings'].pop('scripts/train/supervise_scoped_smoke600_v1.py');r.rescope()
    with pytest.raises(ValueError,match='code closure'):r.validate()


def test_actual_reward_model_path_still_bound_for_smoke(smoke_release,monkeypatch):
    r=smoke_release
    monkeypatch.setattr(smoke,'model_path',lambda role:str(r.root/'models/foreign'))
    with pytest.raises(ValueError,match='ReaRAG environment'):
        dispatch.validate_bounded_execution_paths(SimpleNamespace(artifact=r.child),r.cfg)


@pytest.mark.parametrize('diagnostic',[False,True])
def test_strip_smoke_scope_and_resign_never_confers_clearance(tmp_path,diagnostic):
    path,_=_write_release(tmp_path/'mask');mask=FrozenSourceCreditMask.load(path)
    artifact=_v2_artifact(mask)
    artifact.update(bank_source='real_frozen_policy_rollouts',training_clearance=True,independent_confirmation_clearance=True,
                    training_clearance_scope='complete_A_smoke600_only')
    with pytest.raises(ValueError,match='registered bounded execution scope'):
        SourceCreditGateV2(_signed(artifact),mask=mask,allow_unvalidated=diagnostic)


def test_historical_v1_general_clearance_description_does_not_select_v2(tmp_path,monkeypatch):
    path,_=_write_release(tmp_path/'mask');mask=FrozenSourceCreditMask.load(path)
    artifact=_artifact(mask)
    artifact['training_clearance_scope']='heuristic train-only diagnostics; no independent confirmation'
    gate_path=tmp_path/'historical-v1.json';gate_path.write_text(json.dumps(_signed(artifact)))
    def forbidden(*args,**kwargs):pytest.fail('historical v1 must use its existing loader')
    monkeypatch.setattr(SourceCreditGateV2,'load',forbidden)
    assert dispatch.load_referenced_bounded_before_dispatch(SimpleNamespace(source_gate_calibration_path=str(gate_path))) is None


def test_stripped_v2_scope_claim_is_checked_even_after_disabling_reward_dispatch(tmp_path,monkeypatch):
    """A v2 true-clearance claim alone remains recognized before any CUDA."""
    cfg=training.Phase3PPOConfig(**resolve_phase3_ppo_runtime_config(
        Path(__file__).resolve().parents[1]/'configs/training/phase3_ppo_mixed4_answer_format_v2_a_smoke_seed42.yaml'))
    path=tmp_path/'stripped-v2-claim.json'
    path.write_text(json.dumps({'schema_version':ARTIFACT_SCHEMA,'training_clearance':True}))
    cfg=replace(cfg,source_gate_calibration_path=str(path),source_gated_reward_version='disabled',
                source_gate_credit_version='disabled',answer_format_reward_version='legacy',center_text_reward=True)
    training._validate_mixed_reward_config(cfg)
    class SeenBeforeCUDA(RuntimeError):pass
    observed=[]
    def reject(path,*,runtime_config,**kwargs):
        observed.append(runtime_config);raise SeenBeforeCUDA
    def forbidden(*args,**kwargs):pytest.fail('CUDA/model access preceded bounded validation')
    monkeypatch.setattr(SourceCreditGateV2,'load',reject)
    monkeypatch.setattr(training,'set_seed',lambda seed:None)
    monkeypatch.setattr(training,'_build_models',forbidden)
    monkeypatch.setattr(training.torch.cuda,'is_available',forbidden)
    with pytest.raises(SeenBeforeCUDA):training.run_phase3_ppo(cfg)
    assert observed==[cfg]
