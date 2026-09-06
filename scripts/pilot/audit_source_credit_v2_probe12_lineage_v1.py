#!/usr/bin/env python
"""Independent read-only probe lineage/numeric audit; never opens Gold labels.

Inputs are frozen metadata, Gold-free question-KG/source records, already
written training telemetry and checkpoint bytes. Tensor and event reviews are
separate audits. No production artifact, training decision or gate is changed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgproweight.reward.source_credit_gate_v1 import FrozenSourceCreditMask
from kgproweight.reward.proofkg_process import is_identity_safe_automatic_proofkg

RUN_NAME = "ppo_mixed4_answer_format_v2_a_probe12_scoped_seed42_20260906_v1"
INPUT = ROOT / "outputs/audits/ppo_a_probe12_gpu_supervision_20260906_v1"
SCOPE_DIR = ROOT / "outputs/calibration/source_credit_gate_v2_probe12_scoped_20260906_v1"
OUTPUT = ROOT / "outputs/audits/ppo_a_probe12_independent_lineage_audit_20260906_v1"


def sha(path, algorithm="sha256"):
    h=hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def binding(path):
    path=Path(path)
    return {"path":str(path.relative_to(ROOT)),"sha256":sha(path),"bytes":path.stat().st_size}


def audit(out=OUTPUT):
    out=Path(out)
    if out.exists():raise FileExistsError("independent audit requires a new output directory")
    run=INPUT/RUN_NAME
    manifest=read(run/"manifest.json"); history=[json.loads(l) for l in (run/"history.jsonl").read_text().splitlines()]
    scope=read(SCOPE_DIR/"scope.json"); gate=read(SCOPE_DIR/"gate.json"); cfg=manifest["run"]["config"]
    parent=read(ROOT/scope["bindings"]["parent_gate"]["path"])
    launch=read(INPUT/"ppo_a_probe12_scoped_20260906_v1_supervision/launch.json")
    status=read(INPUT/"ppo_a_probe12_scoped_20260906_v1_supervision/status.json")
    log=(INPUT/"ppo_a_probe12_scoped_20260906_v1.log").read_text()
    remote=read(ROOT/"outputs/audits/ppo_remote_prelaunch_20260906_v1/remote_assets.json")
    remote_scope=read(ROOT/"outputs/audits/ppo_remote_prelaunch_20260906_v1/remote_probe_scope.json")
    checks=[]
    def check(name,ok,detail=None):
        checks.append({"name":name,"pass":bool(ok),**({"detail":detail} if detail is not None else {})})
    def close(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
    check("normal_exit_and_complete",status["exit_code"]==0 and manifest["status"]=="COMPLETE" and "Phase 3b PPO done." in log)
    check("full_config_matches_frozen_scope",cfg==scope["runtime_config"] and canonical(cfg)==scope["runtime_config_sha256"])
    check("launch_binds_scope_and_config",launch["scope_sha256"]==sha(SCOPE_DIR/"scope.json") and launch["config_sha256"]==scope["bindings"]["config"]["sha256"])
    check("launch_limits_unchanged",launch["trajectory_limit"]==12 and launch["automatic_restart"] is False and launch["automatic_smoke_or_full"] is False)
    check("experiment_ids_connected",manifest["run"]["experiment_id"]==RUN_NAME and Path(cfg["output_dir"]).name==RUN_NAME)
    check("history_complete_exact_order",[h["step"] for h in history]==[4,8,12] and manifest["run"]["history_tail"]==history)
    check("history_manifest_bytes",sha(run/"history.jsonl","md5")==manifest["run"]["output_artifacts"]["history"]["md5"])
    for name,ref in scope["bindings"].items():
        p=ROOT/ref["path"]
        check("frozen_binding:"+name,p.is_file() and p.stat().st_size==ref["bytes"] and sha(p)==ref["sha256"])
    for name,ref in scope["code_bindings"].items():check("frozen_code:"+name,sha(ROOT/name)==ref["sha256"])
    check("prelaunch_remote_scope_positive_and_negative",all(remote_scope["checks"].values()))
    check("prelaunch_remote_sha_verification",remote["status"]=="PASS" and all(v["match"] for v in remote["files"].values()))
    for field in ("silver_path","question_kg_records_path","sft_replay_silver_path","rollout_sampling_weights_path"):
        ref=scope["bindings"][field]
        check("remote_sha_matches_scope:"+field,remote["files"][ref["path"]]["sha256"]==ref["sha256"])
    for field,key in (("silver_path","silver"),("question_kg_records_path","question_kg_records"),("fixed_rollout_schedule_path","fixed_rollout_schedule"),("source_gate_calibration_path","source_quality_gate")):
        check("runtime_manifest_input:"+field,sha(ROOT/cfg[field],"md5")==manifest["run"]["input_artifacts"][key]["md5"])
    changes={"payload_sha256","experiment_id","training_clearance","independent_confirmation_clearance","ppo_launch_clearance","training_clearance_scope","execution_scope","source_credit_mask"}
    check("scientific_parent_gate_unchanged",{k:v for k,v in parent.items() if k not in changes}=={k:v for k,v in gate.items() if k not in changes})
    check("parent_training_flags_remain_false",all(parent[k] is False for k in ("training_clearance","independent_confirmation_clearance","ppo_launch_clearance")))
    check("health_fail_and_no_expansion_retained",scope["confirmation_status"]=={"independent_utility":"PASS","health":"FAIL","overall":"FAIL"} and scope["matched600_clearance"] is False and scope["full_ppo_clearance"] is False)
    mask=FrozenSourceCreditMask.load(ROOT/scope["bindings"]["parent_mask"]["path"])
    mask_counts=Counter((x["original_m_graph"],x["status"]) for x in mask._entries.values())
    check("original_mask800_671",mask_counts=={(1,"PASS"):671,(1,"UNVERIFIED"):100,(1,"FAIL"):29,(0,"UNVERIFIED"):30})
    schedule=[json.loads(l) for l in (ROOT/cfg["fixed_rollout_schedule_path"]).read_text().splitlines()]
    check("schedule12_K4",len(schedule)==12 and [r["rollout_index"] for r in schedule]==list(range(1,13)))
    eligibility=Counter();structural_exclusions=[];eligible_missing_execution=[];source_pass_ineligible=[]
    for line in (ROOT/cfg["question_kg_records_path"]).open():
        row=json.loads(line);kg=row["kg_subgraph"];hops=(row.get("query_plan")or{}).get("hops")or[];prov=row.get("provenance")or{}
        expected_key=row["dataset"]+"::"+row["qid"]
        # Independent replication of the existing conservative predicate.
        complete=prov.get("complete_plan_execution")
        structural=bool(kg) and bool(hops) and prov.get("gold_access") is False and len(kg)>=len(hops) and (complete is None or complete is True)
        independent=structural and row["question_key"]==expected_key
        actual=is_identity_safe_automatic_proofkg(row,kg,dataset=row["dataset"],qid=row["qid"])
        if independent!=actual:raise ValueError("independent eligibility predicate differs")
        eligibility[(bool(kg),actual)]+=1
        entry=mask._entries.get(expected_key,{})
        if entry.get("status")=="PASS" and not actual:source_pass_ineligible.append(expected_key)
        if kg and not actual:structural_exclusions.append({"question_key":expected_key,"kg_edges":len(kg),"planned_hops":len(hops),"complete_plan_execution":complete,"gold_access":prov.get("gold_access"),"source_mask_status":entry.get("status"),"reason":"frozen_unique_edge_count_less_than_planned_hop_count"})
        if actual:
            executed={int(x.get("hop_index",-1)):x for x in (row.get("execution")or{}).get("hops",[]) if isinstance(x,dict)}
            if not all(i in executed and executed[i].get("matches") for i in range(1,len(hops)+1)):eligible_missing_execution.append(expected_key)
    check("execution_eligibility798_independently_reproduced",eligibility=={(False,False):2200,(True,True):798,(True,False):2} and not eligible_missing_execution and not source_pass_ineligible)
    check("execution798_matches_runtime_log","'eligible_rows': 798, 'missing_execution_rows': 0" in log and "Automatic ProofKG reward eligibility: 798/3000" in log)
    check("exact_reference_mode_initial_KL",manifest["run"]["reference_mode"]=="explicit_frozen_sft_snapshot" and manifest["run"]["initial_reference_kl"]==0 and history[0]["ppo_mean_kl"]==0 and all(h["uses_explicit_sft_reference"] for h in history) and "Pre-update explicit-SFT-reference KL: 0.000000" in log)
    numeric_count=0;nonfinite=[]
    def inspect(value,path="history"):
        nonlocal numeric_count
        if isinstance(value,float):
            numeric_count+=1
            if not math.isfinite(value):nonfinite.append(path)
        elif isinstance(value,dict):
            for k,v in value.items():inspect(v,path+"/"+k)
        elif isinstance(value,list):
            for i,v in enumerate(value):inspect(v,path+"/"+str(i))
    inspect(history);check("all_recorded_floats_finite",not nonfinite,{"floats":numeric_count,"nonfinite_paths":nonfinite})
    norm=gate["normalization"];batch_summary=[];record_count=0;positive_graph_records=0
    for bi,h in enumerate(history):
        sched=schedule[bi*4:(bi+1)*4];records=h["source_gate_records"]
        check(f"batch{bi}_schedule_identity",h["rollout_qids"]==[x["qid"] for x in sched] and h["rollout_strata"]==[x["stratum"] for x in sched])
        check(f"batch{bi}_four_records",len(records)==4)
        valid_records=[]
        for ri,(record,schedrow) in enumerate(zip(records,sched)):
            record_count+=1;features=record["features"];bound=features["source_credit_mask"];qkey=schedrow["dataset"]+"::"+schedrow["qid"]
            check(f"record{record_count}_identity_and_mask",bound["question_key"]==qkey and bound["mask_payload_sha256"]==mask.payload_sha256 and record["artifact_payload_sha256"]==gate["payload_sha256"])
            invalid=record.get("invalid_not_scored") is True
            if invalid:
                check(f"record{record_count}_invalid_process_zero",all(record[k]==0 for k in ("alpha_predicted","alpha_effective","text_component","graph_component")))
                continue
            valid_records.append(record)
            m=features["m_graph"]
            if m:
                check(f"record{record_count}_source_PASS_required",bound["status"]=="PASS" and mask._entries[qkey]["status"]=="PASS")
                logit=gate["bias"]+sum(w*(features["values"][name]-gate["feature_standardization"]["mean"][name])/gate["feature_standardization"]["scale"][name] for name,w in zip(gate["feature_names"],gate["weights"]))
                pred=1/(1+math.exp(-max(-60,min(60,logit))))
            else:pred=0.0
            alpha=m*pred
            graph_z=max(-1,min(1,(record["graph_raw"]-norm["graph_center"])/norm["graph_scale"])) if m else 0.0
            zs=record["text_normalized_unclipped_steps"]
            expected_text=sum(.3*(1-alpha)*(z/(1+abs(z)))/len(zs) for z in zs)
            expected_graph=.2*alpha*graph_z
            check(f"record{record_count}_alpha_and_components",close(pred,record["alpha_predicted"]) and close(alpha,record["alpha_effective"]) and close(expected_text,record["text_component"]) and close(expected_graph,record["graph_component"]))
            if alpha>0 and record["graph_component"]!=0:positive_graph_records+=1
        check(f"batch{bi}_valid_counts",len(valid_records)==h["n_valid"] and h["valid_rate"]==len(valid_records)/4)
        check(f"batch{bi}_text_graph_means",close(sum(r["text_component"] for r in records)/4,h["source_gate_text_component_mean"]) and close(sum(r["graph_component"] for r in records)/4,h["source_gate_graph_component_mean"]))
        groups=list(h["mixed_reward_by_dataset"].values())
        check(f"batch{bi}_reward_decomposition",len(groups)==1 and close(groups[0]["outcome_mean"]+h["source_gate_text_component_mean"]+h["source_gate_graph_component_mean"],h["mean_reward"]))
        batch_summary.append({"step":h["step"],"dataset":sched[0]["dataset"],"valid":h["n_valid"],"rollouts":4,"training_outcome_EM":h["mixed_outcome_em_mean"],"training_outcome_F1":h["mixed_outcome_f1_mean"],"reward":h["mean_reward"],"KL":h["ppo_mean_kl"],"text_steps":h["mixed_text_step_count"],"source_PASS_valid_alpha":[r["alpha_effective"] for r in valid_records if r["m_graph"]],"length_capped":h["length_capped_count"],"replay_items":h["sft_replay_items"]})
    check("graph_alpha_positive_path_covered",positive_graph_records==3)
    check("fractional_replay_exact",[h["sft_replay_items"] for h in history]==[0,0,1] and history[-1]["sft_replay_items_seen"]==1 and close(history[-1]["sft_replay_actual_ratio"],1/12) and "Supervised replay step=12 items=1 cumulative=1/12" in log)
    check("three_real_optimizer_batch_calls",log.count("TIMING upd=")==3)
    cp=run/"final";checkpoint_inventory=manifest["run"]["output_artifacts"]["final_checkpoint"]["files"]
    check("checkpoint_file_inventory",sorted(p.name for p in cp.iterdir() if p.is_file())==sorted(x["name"] for x in checkpoint_inventory))
    for item in checkpoint_inventory:
        file=cp/item["name"]
        check("checkpoint_file:"+item["name"],file.is_file() and file.stat().st_size==item["size_bytes"] and ("md5" not in item or sha(file,"md5")==item["md5"]))
    original_cfg=read(ROOT/cfg["sft_checkpoint"]/"adapter_config.json");final_cfg=read(cp/"adapter_config.json")
    raw_config_diff={k:[original_cfg.get(k),final_cfg.get(k)] for k in set(original_cfg)|set(final_cfg) if original_cfg.get(k)!=final_cfg.get(k)}
    a=deepcopy(original_cfg);b=deepcopy(final_cfg)
    a["target_modules"]=sorted(a["target_modules"]);b["target_modules"]=sorted(b["target_modules"])
    check("adapter_configuration_semantics_unchanged",a==b)
    check("adapter_changed_from_original_sft",sha(cp/"adapter_model.safetensors")!=scope["bindings"]["sft:adapter_model.safetensors"]["sha256"])
    check("original_sft_still_exact",sha(ROOT/cfg["sft_checkpoint"]/"adapter_model.safetensors")==scope["bindings"]["sft:adapter_model.safetensors"]["sha256"])
    tb=read(run/"tensorboard_run.json")
    check("tensorboard_lineage_pointer",tb["experiment_id"]==RUN_NAME and tb["step_unit"]=="completed_rollout_trajectories" and tb["histogram_initial_ppo_batches"]==3 and tb["log_dir"] in log)
    inputs={name:binding(path) for name,path in {"audit_code":Path(__file__),"training_manifest":run/"manifest.json","history":run/"history.jsonl","log":INPUT/"ppo_a_probe12_scoped_20260906_v1.log","launch":INPUT/"ppo_a_probe12_scoped_20260906_v1_supervision/launch.json","status":INPUT/"ppo_a_probe12_scoped_20260906_v1_supervision/status.json","scope":SCOPE_DIR/"scope.json","child_gate":SCOPE_DIR/"gate.json","checkpoint_adapter":cp/"adapter_model.safetensors","checkpoint_value_head":cp/"pytorch_model.bin","eligibility_code":ROOT/"kgproweight/reward/proofkg_process.py","source_mask_code":ROOT/"kgproweight/reward/source_credit_gate_v1.py","remote_assets":ROOT/"outputs/audits/ppo_remote_prelaunch_20260906_v1/remote_assets.json","remote_scope":ROOT/"outputs/audits/ppo_remote_prelaunch_20260906_v1/remote_probe_scope.json"}.items()}
    passed=all(c["pass"] for c in checks)
    report={"schema_version":"source-credit-v2-probe12-independent-lineage-audit-v1","experiment_id":"SOURCE-CREDIT-V2-PROBE12-INDEPENDENT-LINEAGE-AUDIT-20260906-V1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS_ENGINEERING_LINEAGE_ONLY" if passed else "FAIL",
        "checks":checks,"checks_passed":sum(c["pass"] for c in checks),"checks_total":len(checks),"inputs":inputs,"training_experiment_id":RUN_NAME,"configuration_sha256":canonical(cfg),"batch_summary":batch_summary,
        "scope":{"trajectories":12,"unique_questions":3,"PPO_batch_calls":3,"replay_items":1,"nonzero_alpha_graph_records":positive_graph_records,"text_steps":sum(h["mixed_text_step_count"] for h in history),"valid_trajectories":sum(h["n_valid"] for h in history),"fresh_health_status":"FAIL","matched600_clearance":False,"full_ppo_clearance":False},
        "graph_population":{"nonempty_graph":800,"frozen_structural_execution_eligible":798,"source_PASS":671,"structural_exclusions":structural_exclusions,"source_PASS_structurally_ineligible":source_pass_ineligible,"explanation":"800 is input graph coverage; 798 passes the unchanged unique-edge-count >= hop-count execution predicate; 671 passes the independent original source mask. The two exclusions are UNVERIFIED and never had graph credit; no rows or thresholds changed."},
        "checkpoint":{"adapter_sha256":sha(cp/"adapter_model.safetensors"),"original_adapter_sha256":scope["bindings"]["sft:adapter_model.safetensors"]["sha256"],"adapter_config_raw_diff":raw_config_diff,"config_diff_semantic":"target_modules order only; same membership","tensor_finiteness_and_parameter_delta":"separate independent tensor/event auditor, not recomputed here"},
        "limitations":["Three training questions with K4 are an engineering probe; training EM/F1 do not establish held-out performance or learning gain.","Numerical checks cover recorded telemetry and byte/metadata lineage; raw ReaRAG/model forwards and Gold answer metrics were not rerun.","The source scorer verifies structure/derived-answer consistency, not free-form reasoning semantics.","Fresh confirmation health FAIL remains. Rolling guard starts at trajectory200 and does not certify health in this12-trajectory run.","TensorBoard event content and tensor-level checkpoint differences are audited separately; this report verifies their lineage pointer/checkpoint bytes.","Paths into the older remote release are resolved symlink targets; prelaunch SHA and bound input bytes, not directory names, identify reused assets."],"gold_values_opened":False,"training_rerun":False,"gate_or_code_modified":False}
    out.mkdir(parents=True,exist_ok=False)
    with (out/"report.json").open("x") as f:json.dump(report,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    with (out/"manifest.json").open("x") as f:json.dump({"schema_version":report["schema_version"],"status":report["status"],"experiment_id":report["experiment_id"],"report":binding(out/"report.json"),"inputs":inputs},f,ensure_ascii=False,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps({"status":report["status"],"checks_passed":report["checks_passed"],"checks_total":len(checks),"failed":[c for c in checks if not c["pass"]],"report":binding(out/"report.json")},ensure_ascii=False))
    return 0 if passed else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path,default=OUTPUT)
    raise SystemExit(audit(parser.parse_args().out))
