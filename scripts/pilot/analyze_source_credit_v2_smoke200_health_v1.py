#!/usr/bin/env python
"""Read-only first200 explanation from existing telemetry; no new evaluation."""
from pathlib import Path
import hashlib,json,math,re,sys
from collections import Counter
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kgproweight.training.reward_function import RewardSpec,validate_source_gate_trajectory
from kgproweight.reward.answer_format_objective_v2 import inspect_shortfall_salvage_v2

RUN=ROOT/'outputs/audits/ppo_a_smoke600_gpu_supervision_20260906_v1/ppo_mixed4_answer_format_v2_a_smoke600_scoped_seed42_20260906_v1'
OUT=ROOT/'outputs/audits/ppo_a_smoke600_first200_independent_health_20260906_v1'


def digest(raw):return hashlib.sha256(raw).hexdigest()
def bind(path):return {'path':str(path.relative_to(ROOT)),'sha256':digest(path.read_bytes()),'bytes':path.stat().st_size}


def main():
    if OUT.exists():raise FileExistsError('append-only new intermediate report required')
    all_lines=(RUN/'history.jsonl').read_bytes().splitlines()
    lines=[line for line in all_lines if json.loads(line)['step']<=200]
    h=[json.loads(line) for line in lines]
    assert len(h)==50 and [r['step'] for r in h]==list(range(4,201,4))
    domains={};batches=[]
    for row in h:
        assert len(row['mixed_reward_by_dataset'])==1
        dataset=next(iter(row['mixed_reward_by_dataset']));group=row['mixed_reward_by_dataset'][dataset]
        n=group['count'];valid=row['n_valid'];invalid=n-valid
        # This is the recorded training answer signal, not unconditional baseline EM/F1.
        answer_total=n*4*(row['mixed_outcome_em_mean']+.1*row['mixed_outcome_f1_mean'])
        penalty=answer_total-n*group['outcome_mean']
        severe=(penalty-invalid)/3
        assert abs(severe-round(severe))<1e-7 and -.0001<=severe<=invalid+.0001
        severe=round(severe);shortfall=invalid-severe
        records=row['source_gate_records']
        assert len(records)==n and sum(not r.get('invalid_not_scored',False) for r in records)==valid
        for rec in records:
            if rec.get('invalid_not_scored'):assert rec['text_component']==rec['graph_component']==rec['alpha_effective']==0
            if rec['alpha_effective']>0:assert rec['features']['source_credit_mask']['status']=='PASS'
        values={'rollouts':n,'prompt_groups':1,'strict_valid':valid,'invalid':invalid,'severe_invalid':severe,'two_step_shortfall':shortfall,
            'answer_signal_eligible':n-severe,'training_EM_sum':n*row['mixed_outcome_em_mean'],'training_F1_sum':n*row['mixed_outcome_f1_mean'],
            'length_capped':row['length_capped_count'],'text_scored_steps':row['mixed_text_step_count'],
            'structural_graph_eligible':row['proofkg_eligible_count'],'valid_graph_scorer_applied':row['proofkg_process_applied_count'],
            'source_mask_m1':sum(r['m_graph'] for r in records),'alpha_graph_active':sum(r['alpha_effective']>0 for r in records),
            'training_reward_sum':n*row['mean_reward']}
        domains.setdefault(dataset,Counter()).update(values)
        batches.append({'step':row['step'],'dataset':dataset,'qid':row['rollout_qids'][0],**values})
    totals=Counter()
    for values in domains.values():totals.update(values)
    for values in [totals,*domains.values()]:
        n=values['rollouts'];values.update({'strict_valid_rate':values['strict_valid']/n,'training_EM':values['training_EM_sum']/n,
            'training_F1':values['training_F1_sum']/n,'length_capped_rate':values['length_capped']/n,'answer_signal_eligible_rate':values['answer_signal_eligible']/n})
    # Existing final20 textual samples cross-check aggregate reward-case inversion.
    sample_path=RUN/'samples/step_00200.txt';sample_raw=sample_path.read_text()
    headers=list(re.finditer(r'^--- Sample (\d+) qid=(\S+) response_tokens=(\d+) length_capped=(true|false) ---\n',sample_raw,re.M))
    assert len(headers)==20
    rows_path=ROOT/'data/silver_data/mixed_ppo_three_dataset_v4_proof800_n3000_k4_seed42/question_kg_records.jsonl'
    wanted={m[2] for m in headers};question_rows={}
    for line in rows_path.open():
        rec=json.loads(line)
        if rec['qid'] in wanted:
            assert rec['qid'] not in question_rows
            question_rows[rec['qid']]=rec
    old_records=[(row,rec) for row in h[-5:] for rec in row['source_gate_records']]
    sample_checks=[];case_counts=Counter()
    for i,mark in enumerate(headers):
        text=sample_raw[mark.end():headers[i+1].start() if i+1<len(headers) else len(sample_raw)].strip()
        rec=question_rows[mark[2]]
        spec=RewardSpec(query=rec['question'],gold_answer='',kg_subgraph=[tuple(e) for e in rec['kg_subgraph']],metadata={'dataset':rec['dataset'],'qid':rec['qid'],'source_quality_record':rec})
        validity=validate_source_gate_trajectory(spec,text,max_steps=5,min_valid_steps=3,min_reasoning_chars=20,format_version='v2')
        salvage=inspect_shortfall_salvage_v2(text,steps=validity['steps'],required_steps=validity['required_steps'],violations=validity['violations'],known_passage_ids=list(range(1,11)))
        assert validity['valid']==(not old_records[i][1].get('invalid_not_scored',False))
        assert rec['qid']==old_records[i][0]['rollout_qids'][i%4]
        case='valid' if validity['valid'] else 'two_step_shortfall' if salvage.eligible else 'severe_invalid'
        case_counts[case]+=1
        sample_checks.append({'sample_index':i+1,'dataset':rec['dataset'],'qid':rec['qid'],'case':case,'steps':len(validity['steps']),
            'required_steps':validity['required_steps'],'violations':validity['violations'],'salvage_reason':salvage.reason,'response_tokens':int(mark[3]),'length_capped':mark[4]=='true'})
    assert case_counts['two_step_shortfall']==sum(b['two_step_shortfall'] for b in batches[-5:])
    assert case_counts['severe_invalid']==sum(b['severe_invalid'] for b in batches[-5:])
    alpha=[r['alpha_effective'] for row in h for r in row['source_gate_records'] if r['alpha_effective']>0]
    window={key:sum(row[key] for row in h[-15:])/15 for key in ('valid_rate','length_capped_frac','ppo_mean_kl')}
    guard=window['valid_rate']>=.7 and window['length_capped_frac']<=.2 and window['ppo_mean_kl']<=10
    numeric=[]
    def finite(value):
        if isinstance(value,float):numeric.append(math.isfinite(value))
        elif isinstance(value,dict):
            for v in value.values():finite(v)
        elif isinstance(value,list):
            for v in value:finite(v)
    finite(h);assert all(numeric)
    report={'schema_version':'source-credit-v2-smoke-first200-independent-health-v1','experiment_id':'SOURCE-CREDIT-V2-A-SMOKE600-FIRST200-READONLY-20260906-V1',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'INTERMEDIATE_DESCRIPTIVE_AUDIT_COMPLETE_NOT_FINAL_TRAINING_RESULT',
        'history_scope':{'through_trajectory':200,'PPO_batches':50,'unique_scheduled_prompts':50,'strictly_prefix':True},'overall':dict(totals),'by_dataset':{k:dict(v) for k,v in domains.items()},
        'classification_method':{'type':'exact_aggregate_reconstruction_from_frozen_reward_components','equations':['penalty_total = 4 * sum(training_EM + 0.1*training_F1) - sum(recorded_outcome_component)','severe_count = (penalty_total - invalid_count)/3','two_step_shortfall_count = invalid_count - severe_count'],
            'integer_and_bounds_checks':'50/50 PASS','trajectory_details_available_only_for_last20':True,'last20_shared_frozen_validator_crosscheck':'20/20 PASS','known_passage_ids_contract':'frozen10-passages, IDs1..10; no retrieval/model/Gold reread'},
        'alpha_graph':{'positive_credit_records':len(alpha),'positive_alpha_mean':sum(alpha)/len(alpha),'positive_alpha_min':min(alpha),'positive_alpha_max':max(alpha),
            'positive_graph_components':sum(r['graph_component']>0 for row in h for r in row['source_gate_records']),'negative_graph_components':sum(r['graph_component']<0 for row in h for r in row['source_gate_records']),
            'gate_parameters_frozen':True,'ordinary_alpha_zero':all(r['alpha_effective']==0 for row in h if next(iter(row['mixed_reward_by_dataset']))!='2wikimultihopqa' for r in row['source_gate_records'])},
        'guard_at200':{'last15_batch_means':window,'pass':guard,'after_trajectories':200,'window_batches':15,'thresholds':{'min_valid_rate':.7,'max_length_capped_frac':.2,'max_mean_KL':10},'fresh_confirmation_health_status':'FAIL','not_the90percent_confirmation_health_gate':True},
        'runtime_checks':{'recorded_float_count':len(numeric),'all_recorded_floats_finite':True,'initial_reference_KL':h[0]['ppo_mean_kl'],'current_batch_KL':h[-1]['ppo_mean_kl'],'replay_items_seen':h[-1]['sft_replay_items_seen'],'replay_ratio':h[-1]['sft_replay_actual_ratio']},
        'interpretation':['Strict format weakness is concentrated in HotpotQA, largely complete two-step outputs rejected by the unchanged ordinary three-step minimum.','The v2 answer exception retains answer reward for those shortfalls with a fixed minus1 penalty; all their Text/Graph credit remains zero.','Only3/200 responses hit the length cap; length truncation alone does not explain57 invalid outputs.','Positive alpha/Graph on47 source-PASS valid records and440 scored text steps show process channels remain active; Graph scorer application and actual masked credit use different denominators.','The three domains have different questions and only50 prompt groups; these training answer/reward statistics cannot establish held-out improvement, alpha superiority or a causal learning trend.'],
        'limitations':['Aggregate severe-invalid count is exactly identifiable, but reasons for the first180 outputs were not individually logged in the available samples.','The last20 are a trailing diagnostic sample, not random/manual semantic annotation. One severe example has missing Final and required Step fields; this cannot classify all16 severe outputs.','No new Gold evaluation or model scoring was run, and no training/config/reward/gate/protocol was changed.','This is a200-prefix explanation; final600 and the original fresh health FAIL remain separate.'],
        'sample_crosschecks':sample_checks,'batches':batches,'gold_values_opened':False,'training_or_protocol_modified':False}
    OUT.mkdir(parents=True,exist_ok=False)
    snapshot=OUT/'history_first200.jsonl'
    with snapshot.open('xb') as f:f.write(b'\n'.join(lines)+b'\n')
    report['bindings']={'history_first200':bind(snapshot),'sample_text':bind(sample_path),'questionKG':bind(rows_path),'analysis_code':bind(Path(__file__)),
        'frozen_reward_code':bind(ROOT/'kgproweight/reward/answer_format_objective_v2.py'),'frozen_validator_code':bind(ROOT/'kgproweight/training/reward_function.py')}
    with (OUT/'report.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    with (OUT/'manifest.json').open('x') as f:json.dump({'schema_version':report['schema_version'],'status':report['status'],'report':bind(OUT/'report.json'),'bindings':report['bindings']},f,ensure_ascii=False,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'status':report['status'],'overall':report['overall'],'guard':report['guard_at200'],'report':bind(OUT/'report.json')},ensure_ascii=False))


if __name__=='__main__':main()
