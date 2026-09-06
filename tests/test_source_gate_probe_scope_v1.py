"""CPU scope fixtures deliberately use synthetic authority; no Gold/model reads."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kgproweight.reward import source_gate_probe_scope_v1 as s
from kgproweight.training.phase3_ppo import Phase3PPOConfig


@pytest.fixture
def release(tmp_path, monkeypatch):
    root = tmp_path
    monkeypatch.chdir(root)
    monkeypatch.setattr(s, "__file__", str(root / "kgproweight/reward/source_gate_probe_scope_v1.py"))
    monkeypatch.setattr(s, "project_root", lambda: root)
    def put(name, value):
        p = root / name; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, sort_keys=True))
        return {"path":name,"sha256":s.file_sha(p),"bytes":p.stat().st_size}
    cfg = Phase3PPOConfig(silver_path="data/silver.jsonl",output_dir="outputs/probe",sft_checkpoint="models/policy",
        total_steps=12,batch_size=4,rollouts_per_prompt=4,source_gate_mode="learned",source_gate_credit_version="v2",
        source_gate_format_version="v2",source_gated_reward_version="v1",answer_format_reward_version="v2",runtime_contract_version="v2",
        mixed_outcome_reward=True,mixed_text_reward=True,proofkg_process_reward=True,proofkg_process_version="v2_3",log_with="tensorboard",
        source_gate_calibration_path="release/gate.json",question_kg_records_path="data/kg.jsonl",
        rollout_sampling_weights_path="data/sampling.jsonl",sft_replay_silver_path="data/replay.jsonl",fixed_rollout_schedule_path="data/schedule.jsonl")
    actual=s.config_identity(cfg,root=root)
    monkeypatch.setattr(s,"PARENT_CONFIG_SCIENTIFIC_SHA256",s.digest({k:v for k,v in actual.items() if k not in ("output_dir","source_gate_calibration_path")}))
    refs = {"parent_mask":put("parent/mask.json",{"synthetic":True}),"recovery_protocol":put("parent/recovery.json",{}),
            "config":put("configs/child.yaml",{}),"parent_config":put("configs/parent.yaml",{})}
    parent={"weights":[.1,.2],"normalization":{"alpha_cap":.5},"training_clearance":False,
        "independent_confirmation_clearance":False,"ppo_launch_clearance":False,
        "source_credit_mask":{**refs["parent_mask"],"payload_sha256":"maskpayload"}}
    refs["parent_gate"]=put("parent/gate.json",parent)
    report={"independent_utility_status":"PASS","engineering_probe_eligibility":True,"health_status":"FAIL","overall_status":"FAIL",
            "decision":{"matched600_investment_clearance":False,"full_ppo_auto_launch":False},"integrity":{"status":"PASS"},
            "recovery":{"protocol":refs["recovery_protocol"]}}
    refs["utility_report"]=put("parent/report.json",report)
    refs["utility_manifest"]=put("parent/manifest.json",{"outputs":{"report.json":refs["utility_report"]}})
    for field in ("silver_path","question_kg_records_path","rollout_sampling_weights_path","sft_replay_silver_path"):
        refs[field]=put(actual[field],{})
    models={}
    for role,name in (("base_model","base"),("rearag_model","rearag"),("policy_tokenizer","policy")):
        ref=put(f"models/{name}/tokenizer.json",{"role":role})
        models[role]={"path":f"models/{name}","files":{"tokenizer.json":ref}}
    refs["sft:adapter_model.safetensors"]=put("models/policy/adapter_model.safetensors",{})
    refs["sft:adapter_config.json"]=put("models/policy/adapter_config.json",{"base_model_name_or_path":"models/base"})
    refs["model_authority"]=put("parent/models.json",{"models":models,"source_bindings":{"policy":refs["sft:adapter_model.safetensors"],"policy_config":refs["sft:adapter_config.json"]}})
    for attr,key in (("PARENT_GATE_SHA256","parent_gate"),("PARENT_MASK_SHA256","parent_mask"),("UTILITY_REPORT_SHA256","utility_report"),
                     ("RECOVERY_PROTOCOL_SHA256","recovery_protocol"),("MODEL_AUTHORITY_SHA256","model_authority")):
        monkeypatch.setattr(s,attr,refs[key]["sha256"])
    schedule=[{"rollout_index":i+1,"dataset":("hotpotqa","2wikimultihopqa","musique")[i//4],"qid":str(i//4)} for i in range(12)]
    p=root/actual["fixed_rollout_schedule_path"];p.write_text("\n".join(json.dumps(x) for x in schedule))
    refs["schedule"]={"path":actual["fixed_rollout_schedule_path"],"sha256":s.file_sha(p),"bytes":p.stat().st_size}
    names={"kgproweight/reward/source_gate_probe_scope_v1.py","kgproweight/reward/source_credit_gate_v2.py","kgproweight/training/phase3_ppo.py",
        "scripts/train/phase3_ppo.py","scripts/train/_split_args.py","scripts/prepare/resolve_phase3_ppo_runtime_config.py",
        "scripts/prepare/freeze_source_credit_v2_probe12_scope_v1.py","scripts/sourcegate_python.sh","configs/child.yaml","configs/parent.yaml"}
    code={name:put(name,{}) for name in names}
    scope={"schema_version":s.SCHEMA,"scope":"complete_A_probe12_only","child_gate_path":cfg.source_gate_calibration_path,
        "runtime_config":actual,"runtime_config_sha256":s.digest(actual),"bindings":refs,"code_bindings":code,"models":models,
        "confirmation_status":{"independent_utility":"PASS","health":"FAIL","overall":"FAIL"},
        "limits":{"trajectories":12,"prompt_groups":3,"rollouts_per_prompt":4,"ppo_batches":3,"automatic_resume":False,"automatic_smoke_or_full":False},
        "matched600_clearance":False,"full_ppo_clearance":False}
    child=deepcopy(parent); child.update(training_clearance=True,independent_confirmation_clearance=True,training_clearance_scope="complete_A_probe12_only")
    child["execution_scope"]=put("release/scope.json",scope)
    mask=SimpleNamespace(manifest_sha256=s.PARENT_MASK_SHA256,payload_sha256="maskpayload",_entries={})
    for n,(graph,status,count) in enumerate(((1,"PASS",671),(1,"UNVERIFIED",100),(1,"FAIL",29),(0,"UNVERIFIED",30))):
        for i in range(count):mask._entries[f"{n}/{i}"]={"original_m_graph":graph,"status":status}
    def validate(): return s.validate_probe_scope(child,cfg,cfg.source_gate_calibration_path,mask,root=root)
    def rescope():child["execution_scope"]=put("release/scope.json",scope)
    return SimpleNamespace(root=root,cfg=cfg,scope=scope,child=child,mask=mask,refs=refs,put=put,validate=validate,rescope=rescope)


def test_exact_synthetic_scope_passes(release):
    assert release.validate()["trajectory_limit"] == 12
    assert release.validate()["health_status"] == "FAIL"


@pytest.mark.parametrize("changes",[{"total_steps":600},{"total_steps":12000},{"batch_size":8},{"rollouts_per_prompt":1},{"learning_rate":2e-6},
    {"source_gated_reward_version":"disabled"},{"source_gate_mode":"fixed"},{"mixed_text_reward":False},{"output_dir":"outputs/unregistered"}])
def test_actual_cli_override_cannot_expand_or_change_probe(release,changes):
    cfg=replace(release.cfg,**changes)
    with pytest.raises(ValueError):s.validate_probe_scope(release.child,cfg,cfg.source_gate_calibration_path,release.mask,root=release.root)


def test_resigning_cfg_and_scope_cannot_change_science(release):
    cfg=replace(release.cfg,learning_rate=9e-6)
    release.scope["runtime_config"]=s.config_identity(cfg,root=release.root)
    release.scope["runtime_config_sha256"]=s.digest(release.scope["runtime_config"]);release.rescope()
    with pytest.raises(ValueError,match="scientific"):s.validate_probe_scope(release.child,cfg,cfg.source_gate_calibration_path,release.mask,root=release.root)


@pytest.mark.parametrize("kind",["alpha","mask","health","limit","code_missing","code_drift","model_drift","data_drift","data_ref","authority"])
def test_bound_release_mutations_failclosed(release,kind):
    if kind=="alpha":release.child["weights"][0]=.3
    elif kind=="mask":release.mask._entries.pop(next(iter(release.mask._entries)))
    elif kind=="health":release.scope["confirmation_status"]["health"]="PASS"
    elif kind=="limit":release.scope["limits"]["trajectories"]=600
    elif kind=="code_missing":release.scope["code_bindings"].pop("scripts/train/_split_args.py")
    elif kind=="code_drift":(release.root/"kgproweight/training/phase3_ppo.py").write_text("changed")
    elif kind=="model_drift":(release.root/"models/rearag/tokenizer.json").write_text("changed")
    elif kind=="data_drift":(release.root/release.cfg.silver_path).write_text("changed")
    elif kind=="data_ref":release.scope["bindings"]["silver_path"]=release.put("data/substitute.jsonl",{})
    elif kind=="authority":release.scope["bindings"]["model_authority"]=release.put("parent/forgedmodels.json",{})
    release.rescope()
    with pytest.raises(ValueError):release.validate()


def test_exact_model_loader_paths_before_cuda(release,monkeypatch):
    monkeypatch.setattr(s,"model_path",lambda name:release.root/"models/rearag")
    gate=SimpleNamespace(artifact=release.child)
    s.validate_probe_execution_paths(gate,release.cfg)
    monkeypatch.setattr(s,"model_path",lambda name:release.root/"models/foreign")
    with pytest.raises(ValueError,match="ReaRAG"):s.validate_probe_execution_paths(gate,release.cfg)


def test_cwd_and_execution_root_cannot_be_substituted(release,monkeypatch,tmp_path):
    foreign=release.root/"foreign";foreign.mkdir();monkeypatch.chdir(foreign)
    with pytest.raises(ValueError,match="project root"):release.validate()


@pytest.mark.parametrize("value",[None,{},SimpleNamespace(total_steps=12)])
def test_actual_dataclass_required(release,value):
    with pytest.raises(ValueError):s.validate_probe_scope(release.child,value,release.cfg.source_gate_calibration_path,release.mask,root=release.root)


def test_binding_paths_are_portable_and_content_verified(release):
    ref=release.put("portable/file.json",{})
    assert s.bound_path(ref,root=release.root).is_file()
    for bad in ("../file",str(release.root/"portable/file.json")):
        with pytest.raises(ValueError):s.bound_path({**ref,"path":bad},root=release.root)
