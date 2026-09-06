#!/usr/bin/env python
"""Describe the preserved300-trajectory guard stop; no protocol changes."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,math

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'outputs/audits/ppo_a_smoke600_gpu_supervision_terminal300_20260906_v1/ppo_mixed4_answer_format_v2_a_smoke600_scoped_seed42_20260906_v1'
PREFIX=ROOT/'outputs/audits/ppo_a_smoke600_gpu_supervision_20260906_v1/intermediate200_snapshot'
MID=ROOT/'outputs/audits/ppo_a_smoke600_first200_independent_health_20260906_v1'
OUT=ROOT/'outputs/audits/ppo_a_smoke600_stopped300_independent_health_20260906_v1'


def sha(raw):return hashlib.sha256(raw).hexdigest()
def bind(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}


def aggregate(rows):
    domains={};batches=[]
    for row in rows:
        assert len(row['mixed_reward_by_dataset'])==1
        domain=next(iter(row['mixed_reward_by_dataset']));g=row['mixed_reward_by_dataset'][domain]
        n=g['count'];bad=n-row['n_valid'];penalty=n*(4*(row['mixed_outcome_em_mean']+.1*row['mixed_outcome_f1_mean'])-g['outcome_mean'])
        severe=(penalty-bad)/3
        assert abs(severe-round(severe))<1e-7 and -.0001<=severe<=bad+.0001
        severe=round(severe)
        records=row['source_gate_records'];assert len(records)==n
        assert sum(not r.get('invalid_not_scored',False) for r in records)==row['n_valid']
        v={'rollouts':n,'prompt_groups':1,'strict_valid':row['n_valid'],'invalid':bad,'severe_invalid':severe,'two_step_shortfall':bad-severe,
           'answer_signal_eligible':n-severe,'training_EM_sum':n*row['mixed_outcome_em_mean'],'training_F1_sum':n*row['mixed_outcome_f1_mean'],
           'length_capped':row['length_capped_count'],'text_scored_steps':row['mixed_text_step_count'],'structural_graph_eligible':row['proofkg_eligible_count'],
           'valid_graph_scorer_applied':row['proofkg_process_applied_count'],'source_mask_m1':sum(r['m_graph'] for r in records),
           'alpha_graph_active':sum(r['alpha_effective']>0 for r in records),'training_reward_sum':n*row['mean_reward']}
        domains.setdefault(domain,Counter()).update(v);batches.append({'step':row['step'],'dataset':domain,'qid':row['rollout_qids'][0],**v})
    total=Counter()
    for v in domains.values():total.update(v)
    for v in [total,*domains.values()]:
        n=v['rollouts'];v['strict_valid_rate']=v['strict_valid']/n;v['training_EM']=v['training_EM_sum']/n;v['training_F1']=v['training_F1_sum']/n;v['length_capped_rate']=v['length_capped']/n
    return {'overall':dict(total),'by_dataset':{k:dict(v) for k,v in domains.items()},'batches':batches}


def main():
    if OUT.exists():raise FileExistsError('new terminal report required; no overwrite')
    manifest=json.loads((RUN/'manifest.json').read_text());raw=(RUN/'history.jsonl').read_bytes();lines=raw.splitlines();h=[json.loads(x) for x in lines]
    assert manifest['status']=='FAILED' and manifest['run']['failed_at_step']==300 and manifest['run']['failure_type']=='pre_registered_smoke_health_guard'
    assert len(h)==75 and [r['step'] for r in h]==list(range(4,301,4))
    assert [json.loads(x) for x in (PREFIX/'history.jsonl').read_bytes().splitlines()]==h[:50]
    total=aggregate(h);last15=aggregate(h[-15:]);after200=aggregate(h[50:])
    cfg=manifest['run']['config'];window=cfg['health_guard_window'];assert window==15 and cfg['health_guard_after_steps']==200
    scans=[]
    for i,row in enumerate(h):
        if row['step']<cfg['health_guard_after_steps']:continue
        tail=h[i-window+1:i+1]
        avg={k:sum(r[k] for r in tail)/window for k in ('valid_rate','length_capped_frac','ppo_mean_kl')}
        reasons=[]
        if avg['valid_rate']<cfg['health_guard_min_valid_rate']:reasons.append('valid_rate')
        if avg['length_capped_frac']>cfg['health_guard_max_length_capped_frac']:reasons.append('length_capped_frac')
        if avg['ppo_mean_kl']>cfg['health_guard_max_mean_kl']:reasons.append('ppo_mean_kl')
        scans.append({'step':row['step'],'means':avg,'violations':reasons})
    assert scans[-1]['violations']==['valid_rate'] and all(not x['violations'] for x in scans[:-1])
    finite_checks=[]
    def inspect(v):
        if isinstance(v,float):finite_checks.append(math.isfinite(v))
        elif isinstance(v,dict):
            for x in v.values():inspect(x)
        elif isinstance(v,list):
            for x in v:inspect(x)
    inspect(h);assert all(finite_checks)
    alpha=[r['alpha_effective'] for x in h for r in x['source_gate_records'] if r['alpha_effective']>0]
    for row in h:
        for rec in row['source_gate_records']:
            if rec['alpha_effective']>0:assert rec['features']['source_credit_mask']['status']=='PASS' and not rec.get('invalid_not_scored',False)
            if rec.get('invalid_not_scored',False):assert rec['alpha_effective']==rec['text_component']==rec['graph_component']==0
    assert cfg['total_steps']==600 and not (RUN/'final').exists()
    report={'schema_version':'source-credit-v2-smoke-stopped300-independent-health-v1','experiment_id':'SOURCE-CREDIT-V2-A-SMOKE600-STOPPED300-READONLY-20260906-V1',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'PRE_REGISTERED_FORMAT_GUARD_STOP_CONFIRMED','training_completed600':False,
        'termination':{'planned_trajectories':600,'completed_trajectories':300,'PPO_batches':75,'unique_scheduled_prompts':75,'failure_type':manifest['run']['failure_type'],'failure_reason':manifest['run']['failure_reason'],
            'first_independently_reproduced_guard_failure':scans[-1],'all_prior_guard_checks_from200_passed':True,'checkpoint':manifest['run']['failure_checkpoint']},
        'all300':total,'last15_batches_60_trajectories':last15,'after200_new100':after200,'guard_scans':scans,
        'runtime_health':{'all_recorded_floats_finite':True,'recorded_float_checks':len(finite_checks),'replay_items_seen':h[-1]['sft_replay_items_seen'],'replay_actual_ratio':h[-1]['sft_replay_actual_ratio'],
            'positive_alpha_records':len(alpha),'positive_alpha_mean':sum(alpha)/len(alpha),'positive_alpha_min':min(alpha),'positive_alpha_max':max(alpha),'fresh_confirmation_health_status':'FAIL'},
        'classification_method':{'exact_aggregate_equations':['P = 4*sum(training_EM+0.1*training_F1) - sum(outcome_component)','N_severe=(P-N_invalid)/3','N_shortfall=N_invalid-N_severe'],
            'integer_and_bounds_validation':'75/75 PASS','pertrajectory_semantic_annotations':False,'available_full_text_crosscheck':'Only trailing20 at step200; independently crosschecked in the bound first200 report. No full step300 sample file was saved by the200-trajectory sample cadence.'},
        'interpretation':['The stop is the unchanged pre-registered format-validity cost guard, not NaN/Inf, a memory failure or a new decision threshold.','Severe invalid versus complete two-step shortfall counts follow exactly from stored reward components; individual causes outside the existing step200 sample cannot be reconstructed.','The answer objective retains answer signal for eligible shortfalls but removes all process credit. Therefore poor strict format and nonzero training answer scores can coexist.','All300 and the terminal60 window must be distinguished; the guard uses the latter, while the earlier200 snapshot remains a separate passed checkpoint.','This unique-question training schedule cannot establish whether PPO caused changes in format or answer quality. No baseline or held-out improvement has been measured by this report.'],
        'limits':{'training_restart_authorized_by_this_report':False,'guard_or_reward_changes':False,'new_gold_evaluation':False,'matched_controls_or_full_authorized':False,'original_fresh_health_FAIL_preserved':True},
        'gold_values_opened':False,'training_or_protocol_modified':False}
    OUT.mkdir(parents=True,exist_ok=False)
    with (OUT/'history_stopped300.jsonl').open('xb') as f:f.write(raw)
    report['bindings']={'terminal_history':bind(OUT/'history_stopped300.jsonl'),'terminal_manifest':bind(RUN/'manifest.json'),'first200_immutable_history':bind(PREFIX/'history.jsonl'),
        'first200_report':bind(MID/'report.json'),'first200_snapshot_binding':bind(MID/'snapshot_binding.json'),'analysis_code':bind(Path(__file__)),
        'frozen_training_code':bind(ROOT/'kgproweight/training/phase3_ppo.py'),'frozen_objective_code':bind(ROOT/'kgproweight/reward/answer_format_objective_v2.py')}
    with (OUT/'report.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    with (OUT/'manifest.json').open('x') as f:json.dump({'schema_version':report['schema_version'],'status':report['status'],'report':bind(OUT/'report.json'),'bindings':report['bindings']},f,ensure_ascii=False,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'status':report['status'],'all300':total['overall'],'last60':last15['overall'],'last60_domains':last15['by_dataset'],'report':bind(OUT/'report.json')},ensure_ascii=False))


if __name__=='__main__':main()
