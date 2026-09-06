"""Synthetic CPU smoke authority fixture plus essential budget/guard checks."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from kgproweight.reward import source_gate_probe_scope_v1 as probe
from kgproweight.reward import source_gate_smoke_scope_v1 as smoke
from tests.test_source_gate_probe_scope_v1 import release as probe_release


@pytest.fixture
def smoke_release(probe_release,monkeypatch):
    r=probe_release
    for name in ("PARENT_GATE_SHA256","UTILITY_REPORT_SHA256","PARENT_MASK_SHA256","RECOVERY_PROTOCOL_SHA256","MODEL_AUTHORITY_SHA256"):
        monkeypatch.setattr(smoke,name,getattr(probe,name))
    monkeypatch.setattr(smoke,"__file__",str(r.root/"kgproweight/reward/source_gate_smoke_scope_v1.py"))
    monkeypatch.setattr(smoke,"project_root",lambda:r.root)
    monkeypatch.setattr(smoke,"model_path",lambda name:str(r.root/"models/rearag"))
    parent_scope=r.put("prior/probe_scope.json",deepcopy(r.scope))
    r.cfg=replace(r.cfg,total_steps=600,save_every_steps=200,health_guard_after_steps=200,health_guard_window=15,
        health_guard_min_valid_rate=.7,health_guard_max_length_capped_frac=.2,health_guard_max_mean_kl=10.0)
    actual=smoke.config_identity(r.cfg,root=r.root)
    monkeypatch.setattr(smoke,"PARENT_CONFIG_SCIENTIFIC_SHA256",smoke.digest({k:v for k,v in actual.items() if k not in ("output_dir","source_gate_calibration_path")}))
    r.scope.update(schema_version=smoke.SCHEMA,scope="complete_A_smoke600_only",manual_A_smoke600_clearance=True,
        runtime_config=actual,runtime_config_sha256=smoke.digest(actual),
        limits={"trajectories":600,"prompt_groups":150,"rollouts_per_prompt":4,"ppo_batches":150,"automatic_resume":False,"automatic_restart_or_expansion":False})
    rows=[{"rollout_index":i+1,"dataset":("hotpotqa","2wikimultihopqa","musique")[(i//4)%3],"qid":str(i//4)} for i in range(600)]
    path=r.root/r.cfg.fixed_rollout_schedule_path;path.write_text("\n".join(json.dumps(x) for x in rows))
    r.refs["schedule"]={"path":r.cfg.fixed_rollout_schedule_path,"sha256":smoke.file_sha(path),"bytes":path.stat().st_size}
    monkeypatch.setattr(smoke,"SMOKE_SCHEDULE_SHA256",r.refs["schedule"]["sha256"])
    evidence={"parent_smoke_config":r.refs["parent_config"],
        "probe_training_manifest":r.put("prior/training_manifest.json",{"status":"COMPLETE"}),
        "probe_independent_lineage":r.put("prior/lineage.json",{"status":"PASS_ENGINEERING_LINEAGE_ONLY","inputs":{"scope":parent_scope}}),
        "probe_independent_parameters_events":r.put("prior/tensors_events.json",{"status":"PASS_ENGINEERING_PROBE12_NOT_PERFORMANCE_EVIDENCE"})}
    auth={"schema_version":"ppo-a-smoke600-human-authorization-v1","authorized_complete_A_smoke600":True,
        "trajectory_limit":600,"start_from_original_strong_sft":True,"use_probe_checkpoint_as_initialization":False,
        "automatic_matched600_clearance":False,"full12000_clearance":False,"automatic_restart_or_expansion":False,
        "fresh_confirmation_health_status":"FAIL","guard":{"after_trajectories":200,"window_ppo_batches":15,"min_valid_rate":.7,
        "max_length_capped_frac":.2,"max_mean_kl":10.0,"nonfinite_immediate":True},"evidence":evidence}
    r.refs.update(evidence);r.refs["probe_scope"]=parent_scope
    r.refs["manual_authorization"]=r.put("authorization.json",auth)
    monkeypatch.setattr(smoke,"MANUAL_AUTHORIZATION_SHA256",r.refs["manual_authorization"]["sha256"])
    r.child["training_clearance_scope"]="complete_A_smoke600_only"
    for name in ("kgproweight/reward/source_gate_smoke_scope_v1.py","kgproweight/reward/source_gate_bounded_dispatch_v1.py",
                 "scripts/prepare/freeze_source_credit_v2_smoke600_scope_v1.py","scripts/train/supervise_scoped_smoke600_v1.py"):
        r.scope["code_bindings"][name]=r.put(name,{})
    r.rescope()
    r.validate=lambda:smoke.validate_smoke_scope(r.child,r.cfg,r.cfg.source_gate_calibration_path,r.mask,root=r.root)
    return r


def test_exact_smoke_and_original_models_pass(smoke_release):
    r=smoke_release
    result=r.validate()
    assert result["trajectory_limit"]==600 and result["health_status"]=="FAIL" and result["manual_A_smoke600_clearance"]
    from types import SimpleNamespace
    smoke.validate_smoke_execution_paths(SimpleNamespace(artifact=r.child),r.cfg)


@pytest.mark.parametrize("changes",[{"total_steps":12000},{"total_steps":604},{"total_steps":12},{"mini_batch_size":2},
    {"sft_checkpoint":"outputs/probe/final"},{"health_guard_after_steps":0},{"health_guard_after_steps":201},
    {"health_guard_window":16},{"health_guard_min_valid_rate":.6},{"health_guard_max_mean_kl":20.0},
    {"health_guard_max_length_capped_frac":.3},{"save_every_steps":600}])
def test_budget_model_and_original_guard_cannot_change(smoke_release,changes):
    r=smoke_release;cfg=replace(r.cfg,**changes)
    with pytest.raises(ValueError):smoke.validate_smoke_scope(r.child,cfg,cfg.source_gate_calibration_path,r.mask,root=r.root)


def test_resigned_data_replacement_breaks_accepted_probe_lineage(smoke_release):
    r=smoke_release
    r.refs["silver_path"]=r.put(r.cfg.silver_path,{"changed":"same logical input path"});r.rescope()
    with pytest.raises(ValueError,match="data changed"):r.validate()


def test_resigned_reordered600_schedule_rejected(smoke_release):
    r=smoke_release;path=r.root/r.cfg.fixed_rollout_schedule_path
    lines=path.read_text().splitlines();lines[0],lines[4]=lines[4],lines[0];path.write_text("\n".join(lines))
    r.refs["schedule"]={"path":r.cfg.fixed_rollout_schedule_path,"sha256":smoke.file_sha(path),"bytes":path.stat().st_size};r.rescope()
    with pytest.raises(ValueError,match="schedule differs"):r.validate()


def test_manual_authority_tamper_even_resigned_fails(smoke_release):
    r=smoke_release;path=r.root/r.refs["manual_authorization"]["path"];auth=json.loads(path.read_text());auth["trajectory_limit"]=12000
    r.refs["manual_authorization"]=r.put("authorization.json",auth);r.rescope()
    with pytest.raises(ValueError,match="authority"):r.validate()


def test_probe_constructor_and_smoke_constructor_not_interchangeable(smoke_release):
    r=smoke_release
    with pytest.raises(ValueError,match="unsupported"):probe.validate_probe_scope(r.child,r.cfg,r.cfg.source_gate_calibration_path,r.mask,root=r.root)
