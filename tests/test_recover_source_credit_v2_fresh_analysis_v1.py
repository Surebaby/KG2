"""CPU recovery tests using artificial releases; never open fresh Gold."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts.pilot import recover_source_credit_v2_fresh_analysis_v1 as r
from scripts.pilot import analyze_source_credit_v2_fresh_confirmation_v1 as original
from tests.test_analyze_source_credit_v2_integrity_v1 import population


def artificial_parent(tmp_path):
    parent=tmp_path/'parent'; parent.mkdir()
    scoring=tmp_path/'scoring'; scoring.mkdir()
    failed=tmp_path/'analysis'; failed.mkdir()
    p={'status':'FROZEN','seed':42,'experiment_id':'SYNTHETIC-RECOVERY-CPU',
       'code_bindings':{r.PARENT_ANALYZER:r.bind(r.ROOT/r.PARENT_ANALYZER)},
       'analysis':{'decision_rules':original.decision_rules(),'gold_sources':{'do-not-open':'NONEXISTENT'}}}
    r.write_new(parent/'protocol.json',p)
    r.write_new(failed/'started.json',{'gold_values_opened':False,'automatic_reanalysis_allowed':False})
    r.write_new(failed/'failed.json',{'status':'FAIL','error_type':'ValueError','gold_boundary_entered':False})
    for name in ('processes.jsonl','rankings.jsonl','prepared.json'):
        (scoring/name).write_text('{}\n')
    r.write_new(scoring/'report.json',{'n_questions':132,'n_candidates':660})
    r.write_new(scoring/'manifest.json',{'status':'COMPLETE_GOLD_FREE_SCORED_NOT_ANALYZED',
         'protocol_sha256':r.sha(parent/'protocol.json'),'gold_access':False,'model_updates':0,
         'outputs':{name:r.bind(scoring/name) for name in r.OUTPUTS}})
    return parent/'protocol.json',scoring,failed


@pytest.fixture
def frozen(tmp_path):
    parent,scoring,failed=artificial_parent(tmp_path)
    out=tmp_path/'recovery'
    result=r.freeze(parent_protocol=parent,scoring=scoring,failed_analysis=failed,out=out)
    return out/'protocol.json',parent,scoring,failed,result


def test_exact_single_patch_and_all_other_functions_byte_identical():
    source=(r.ROOT/r.PARENT_ANALYZER).read_bytes()
    corrected,diff,functions=r.fixed_patch(source)
    assert hashlib.sha256(corrected).hexdigest()==r.PATCHED_SOURCE_SHA256
    assert corrected==source.replace(r.OLD.encode(),r.NEW.encode(),1)
    assert 'decision_rules' in functions and 'bootstrap_estimates' in functions and 'load_gold_after_seal' in functions
    assert len([x for x in diff.splitlines() if x.startswith('+') and not x.startswith('+++')])==1
    with pytest.raises(ValueError): r.fixed_patch(source+b'\n')


def test_freeze_preserves_original_failed_release_and_does_not_open_gold(frozen):
    recovery,parent,scoring,failed,result=frozen
    assert result['gold_values_opened'] is False
    assert list(sorted(x.name for x in failed.iterdir()))==['failed.json','started.json']
    assert r.sha(r.ROOT/r.PARENT_ANALYZER)==r.PARENT_SOURCE_SHA256
    p=json.loads(recovery.read_text())
    assert p['parent_protocol_sha256']==r.sha(parent)
    assert p['parent_decision_rules']==original.decision_rules()
    assert r.checked(p['bindings']['scoring_manifest'])==scoring/'manifest.json'
    with pytest.raises(ValueError,match='new recovery'):
        r.freeze(parent_protocol=parent,scoring=scoring,failed_analysis=failed,out=recovery.parent)


def test_corrected_copy_truthful_file_and_explicit_root_restores_import_path(frozen):
    recovery,*_=frozen
    before=list(sys.path)
    p,module,refs=r.load_recovery(recovery)
    assert sys.path==before
    assert Path(module.__file__)==r.checked(refs['corrected_analyzer'])
    assert Path(module.__file__)!=r.ROOT/r.PARENT_ANALYZER
    assert module.ROOT==r.ROOT and module.decision_rules()==original.decision_rules()
    assert p['runtime_path_resolution']['executed_module_file']==module.__file__


@pytest.mark.parametrize('asset',['corrected_analyzer','patch_record','patch_diff','parent_failed','scoring:rankings.jsonl','parent_protocol'])
def test_bound_recovery_or_parent_mutation_fails_before_gold(frozen,asset):
    recovery,*_=frozen
    p=json.loads(recovery.read_text())
    path=Path(p['bindings'][asset]['path'])
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError): r.load_recovery(recovery)


@pytest.mark.parametrize('fault',['gold_entered','before_gold','report'])
def test_freeze_refuses_already_consumed_parent_attempt(tmp_path,fault):
    parent,scoring,failed=artificial_parent(tmp_path)
    if fault=='gold_entered':
        (failed/'failed.json').write_text(json.dumps({'status':'FAIL','error_type':'ValueError','gold_boundary_entered':True}))
    else: (failed/(fault+'.json')).write_text('{}')
    with pytest.raises(ValueError):
        r.freeze(parent_protocol=parent,scoring=scoring,failed_analysis=failed,out=tmp_path/'recovery')
    assert not (tmp_path/'recovery').exists()


def test_exact_cpu_replay_accepts_real_invalid_minus_one_semantics_and_ranks_none(frozen):
    recovery,*_=frozen
    _,module,_=r.load_recovery(recovery)
    rows,predictions,inputs,checks,backend,gates=population(steps=0)
    assert all(not row['trajectory_valid'] and row['raw_graph']==-1 for row in rows)
    with pytest.raises(ValueError,match='out-of-range'):
        original.verify_process_rows(rows,predictions,inputs,{},checks,protocol_sha256='synthetic-protocol',gates=gates,rearag_tokenizer=backend.tokenizer)
    grouped=module.verify_process_rows(rows,predictions,inputs,{},checks,protocol_sha256='synthetic-protocol',gates=gates,rearag_tokenizer=backend.tokenizer)
    assert len(grouped)==132 and not backend.calls
    from scripts.prepare.score_source_credit_v2_fresh_confirmation_v1 import rank_questions
    ranks=rank_questions(rows)
    verified=module.verify_rankings(ranks,grouped)
    assert all(rank['all_sampled_invalid'] for rank in verified.values())
    assert all(arm['selected_candidate_id'] is None for rank in verified.values() for variant in rank['rankings'].values() for arm in variant.values())


@pytest.mark.parametrize('fault',['too_low','fake_diagnostic','nonzero_process','valid_negative'])
def test_recovery_does_not_relax_proof_exactness_invalid_rewards_or_valid_range(frozen,fault):
    recovery,*_=frozen
    _,module,_=r.load_recovery(recovery)
    rows,predictions,inputs,checks,backend,gates=population(steps=2 if fault=='valid_negative' else 0)
    row=rows[0]
    if fault=='too_low': row['raw_graph']=-1.1
    elif fault=='fake_diagnostic': row['raw_graph']=-.5
    elif fault=='nonzero_process': row['variants']['features_v2']['A']['process']=.01
    else: row['raw_graph']=-1.
    row['process_row_sha256']=module.digest({k:v for k,v in row.items() if k!='process_row_sha256'})
    with pytest.raises(ValueError):
        module.verify_process_rows(rows,predictions,inputs,{},checks,protocol_sha256='synthetic-protocol',gates=gates,rearag_tokenizer=backend.tokenizer)


def test_run_failure_before_verified_seal_never_opens_gold_and_preserves_parent(frozen,tmp_path,monkeypatch):
    recovery,_,_,failed,_=frozen
    old={p.name:r.sha(p) for p in failed.iterdir()}
    monkeypatch.setattr(r,'verify',lambda **kwargs: (_ for _ in ()).throw(ValueError('synthetic pre-Gold failure')))
    out=tmp_path/'analysis_v1_1'
    with pytest.raises(ValueError): r.run(recovery_protocol=recovery,out=out)
    assert json.loads((out/'failed.json').read_text())['gold_boundary_entered'] is False
    assert not (out/'before_gold.json').exists()
    assert old=={p.name:r.sha(p) for p in failed.iterdir()}
    with pytest.raises(ValueError): r.run(recovery_protocol=recovery,out=out)


def test_recovery_publisher_binds_actual_copy_before_synthetic_gold(frozen,tmp_path,monkeypatch):
    recovery,*_=frozen
    p,module,refs=r.load_recovery(recovery)
    context={'protocol':{'experiment_id':'SYNTHETIC-RECOVERY-CPU','analysis':{'gold_sources':{}}},
             'protocol_sha256':p['parent_protocol_sha256'],'frozen_files':refs,
             'seal':{'status':'PASS_BEFORE_GOLD','gold_values_opened':False,'frozen_files':refs}}
    events=[]; out=tmp_path/'analysis_v1_1'
    def read_gold(ctx):
        assert (out/'before_gold.json').is_file()
        events.append('synthetic_labels_after_seal'); return {'synthetic':['NeverEmitted']}
    decision={'status':'PASS','overall_status':'PASS','health_status':'PASS','independent_utility_status':'PASS',
              'engineering_probe_eligibility':True,'matched600_investment_clearance':True}
    module.load_gold_after_seal=read_gold
    module.analyze=lambda ctx,labels:([],[],{'synthetic':True},decision)
    monkeypatch.setattr(r,'verify',lambda **kwargs:(p,module,refs,context))
    result=r.run(recovery_protocol=recovery,out=out)
    assert result['status']=='PASS' and events==['synthetic_labels_after_seal']
    report=json.loads((out/'report.json').read_text())
    assert report['protocol_sha256']==p['parent_protocol_sha256']
    assert report['recovery']['executed_analyzer']['sha256']==r.PATCHED_SOURCE_SHA256
    assert report['recovery']['original_analyzer_executed_unmodified'] is False
    assert all('NeverEmitted' not in path.read_text() for path in out.iterdir())
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['recovery_protocol']['sha256']==r.sha(recovery)
    for ref in manifest['outputs'].values(): r.checked(ref)
