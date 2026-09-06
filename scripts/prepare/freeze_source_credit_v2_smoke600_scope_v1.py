#!/usr/bin/env python
"""Freeze a portable limited child gate/config after real utility confirmation.

Only labels' opaque file hashes are read; no model, optimizer, training, or
confirmation analysis is run. Existing parent flags and mask bytes are preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from kgproweight.reward import source_gate_smoke_scope_v1 as scope
from kgproweight.reward.source_credit_gate_v2 import SourceCreditGateV2
from kgproweight.reward.source_quality_gate_v1 import canonical_sha256
from kgproweight.training.phase3_ppo import Phase3PPOConfig
from scripts.prepare.resolve_phase3_ppo_runtime_config import resolve_phase3_ppo_runtime_config

PARENT_GATE = "outputs/calibration/source_credit_gate_v2_representative_local_seed42_20260906_v1/features_v2/gate.json"
PARENT_CONFIG = "configs/training/phase3_ppo_mixed4_answer_format_v2_a_smoke_seed42.yaml"
CONFIG = "configs/training/phase3_ppo_mixed4_answer_format_v2_a_smoke600_scoped_seed42.yaml"
OUTPUT = "outputs/calibration/source_credit_gate_v2_smoke600_scoped_20260906_v1"
ANALYSIS = "outputs/audits/source_credit_v2_fresh_confirmation_local4090_20260906_v1/analysis_recovery_v1"
MODEL_AUTHORITY = "outputs/audits/source_credit_v2_fresh_confirmation_inputs_20260906_v1/manifest.scope_v2.json"
RECOVERY_PROTOCOL = "outputs/audits/source_credit_v2_fresh_confirmation_analysis_recovery_20260906_v1/protocol.json"

AUTHORIZATION = "outputs/audits/ppo_a_smoke600_authorization_20260906_v1/authorization.json"


def bind(path):
    path = Path(path)
    if not path.is_absolute(): path = ROOT / path
    return {"path": scope.portable_path(path, root=ROOT), "sha256": scope.file_sha(path), "bytes": path.stat().st_size}


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def config_files(path):
    import yaml
    found = set()
    def visit(source):
        source = source.resolve()
        name = scope.portable_path(source, root=ROOT)
        if name in found: return
        found.add(name)
        data = yaml.safe_load(source.read_text()) or {}
        for child in data.get("includes") or []: visit(source.parent / child)
    visit(Path(path))
    return found


def freeze(*, out=OUTPUT, config=CONFIG):
    out = Path(out)
    if not out.is_absolute(): out = ROOT / out
    config = Path(config)
    if not config.is_absolute(): config = ROOT / config
    scope.require(not out.exists(), "new scoped-gate output required; parents are never overwritten")
    cfg = Phase3PPOConfig(**resolve_phase3_ppo_runtime_config(config))
    runtime = scope.config_identity(cfg, root=ROOT)
    parent_runtime = resolve_phase3_ppo_runtime_config(ROOT / PARENT_CONFIG)
    changed = {k for k in runtime if runtime[k] != parent_runtime[k]}
    scope.require(changed == {"source_gate_calibration_path", "output_dir"}, "only gate/output config paths may change")
    child_path = out / "gate.json"
    scope.require(runtime["source_gate_calibration_path"] == scope.portable_path(child_path, root=ROOT), "config/out child path mismatch")
    scope.require(scope.digest({k:v for k,v in runtime.items() if k not in changed}) == scope.PARENT_CONFIG_SCIENTIFIC_SHA256,
                  "parent runtime scientific settings changed")
    parent_gate_path = ROOT / PARENT_GATE
    scope.require(scope.file_sha(parent_gate_path) == scope.PARENT_GATE_SHA256, "wrong parent training gate")
    parent = SourceCreditGateV2.load(parent_gate_path, allow_unvalidated=True)
    mask = parent.mask
    scope.require(mask.manifest_sha256 == scope.PARENT_MASK_SHA256, "wrong parent training mask")
    counts = Counter((v["original_m_graph"], v["status"]) for v in mask._entries.values())
    scope.require(counts == {(1,"PASS"):671,(1,"UNVERIFIED"):100,(1,"FAIL"):29,(0,"UNVERIFIED"):30}, "mask800/671 changed")
    refs = {"parent_gate": bind(parent_gate_path), "parent_mask": bind(mask.manifest_path),
            "utility_report": bind(ROOT / ANALYSIS / "report.json"), "utility_manifest": bind(ROOT / ANALYSIS / "manifest.json"),
            "recovery_protocol": bind(ROOT / RECOVERY_PROTOCOL), "schedule": bind(ROOT / runtime["fixed_rollout_schedule_path"]),
            "config": bind(config), "parent_config": bind(ROOT / PARENT_CONFIG), "model_authority": bind(ROOT / MODEL_AUTHORITY)}
    authorization = json.loads((ROOT / AUTHORIZATION).read_text())
    scope.require(scope.file_sha(ROOT / AUTHORIZATION) == scope.MANUAL_AUTHORIZATION_SHA256, "manual A600 authorization changed")
    refs["manual_authorization"] = bind(ROOT / AUTHORIZATION)
    for key, frozen in authorization["evidence"].items():
        refs[key] = bind(ROOT / frozen["path"])
        scope.require(refs[key] == frozen, "authorized probe evidence changed")
    probe_lineage = json.loads((ROOT / refs["probe_independent_lineage"]["path"]).read_text())
    refs["probe_scope"] = bind(ROOT / probe_lineage["inputs"]["scope"]["path"])
    scope.require(refs["probe_scope"] == probe_lineage["inputs"]["scope"], "probe lineage scope changed")
    for field in ("silver_path", "question_kg_records_path", "rollout_sampling_weights_path", "sft_replay_silver_path"):
        refs[field] = bind(ROOT / runtime[field])
    for name in ("adapter_model.safetensors", "adapter_config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"):
        path = ROOT / runtime["sft_checkpoint"] / name
        if path.is_file(): refs["sft:" + name] = bind(path)
    require_report = json.loads((ROOT / ANALYSIS / "report.json").read_text())
    scope.require(refs["utility_report"]["sha256"] == scope.UTILITY_REPORT_SHA256
                  and refs["recovery_protocol"]["sha256"] == scope.RECOVERY_PROTOCOL_SHA256
                  and require_report["independent_utility_status"] == "PASS" and require_report["health_status"] == "FAIL"
                  and require_report["engineering_probe_eligibility"] is True
                  and require_report["decision"]["matched600_investment_clearance"] is False, "real limited probe eligibility missing")
    # Bind current training code, including small enforcement changes. Historical
    # confirmation continues to identify its original executed code snapshots.
    code_names = {scope.portable_path(p, root=ROOT) for p in (ROOT / "kgproweight").rglob("*.py")}
    code_names |= {"scripts/train/phase3_ppo.py", "scripts/train/_split_args.py", "scripts/prepare/resolve_phase3_ppo_runtime_config.py",
                   "scripts/prepare/freeze_source_credit_v2_smoke600_scope_v1.py", "scripts/sourcegate_python.sh",
                   "scripts/train/supervise_scoped_smoke600_v1.py"}
    code_names |= config_files(config)
    code = {name: bind(ROOT / name) for name in sorted(code_names)}
    bound_scope = {"schema_version": scope.SCHEMA, "scope": "complete_A_smoke600_only",
        "experiment_id": "SOURCE-CREDIT-V2-MANUAL-A-SMOKE600-SEED42-20260906-V1",
        "child_gate_path": scope.portable_path(child_path, root=ROOT), "runtime_config": runtime,
        "runtime_config_sha256": scope.digest(runtime), "bindings": refs, "code_bindings": code,
        "models": json.loads((ROOT / MODEL_AUTHORITY).read_text())["models"],
        "confirmation_status": {"independent_utility": "PASS", "health": "FAIL", "overall": "FAIL"},
        "limits": {"trajectories":600,"prompt_groups":150,"rollouts_per_prompt":4,"ppo_batches":150,
                   "automatic_resume":False,"automatic_restart_or_expansion":False},
        "matched600_clearance": False, "full_ppo_clearance": False, "manual_A_smoke600_clearance": True,
        "parent_mask_population": {"graph_inputs":800,"source_PASS":671,"source_UNVERIFIED":100,"source_FAIL":29,"ordinary":30},
        "scope_does_not_establish": ["alpha superiority", "PPO performance improvement", "health PASS", "matched controls/full clearance"],
        "new_smoke_optimizer_updates_at_freeze": 0, "gold_values_opened": False}
    out.mkdir(parents=True, exist_ok=False)
    write(out / "scope.json", bound_scope)
    artifact = deepcopy(parent.artifact)
    artifact.pop("payload_sha256")
    artifact.update(experiment_id=bound_scope["experiment_id"], training_clearance=True, independent_confirmation_clearance=True,
                    ppo_launch_clearance=False, training_clearance_scope="complete_A_smoke600_only",
                    execution_scope=bind(out / "scope.json"))
    artifact["source_credit_mask"]["path"] = refs["parent_mask"]["path"]
    artifact["payload_sha256"] = canonical_sha256(artifact)
    write(child_path, artifact)
    # Real production loader must reject default loading, then pass only the
    # exact CLI-forwarded limited config; neither path allocates CUDA models.
    default_rejected = False
    try: SourceCreditGateV2.load(child_path)
    except ValueError: default_rejected = True
    scope.require(default_rejected, "default loader unexpectedly accepts scoped child")
    loaded = SourceCreditGateV2.load(child_path, runtime_config=cfg)
    report = {"schema_version": scope.SCHEMA, "status": "CPU_READY_BOUNDED_A_SMOKE600_NOT_STARTED",
        "default_loader_rejected": True, "exact_runtime_loader_pass": True,
        "actual_remote_base_and_rearag_paths_require_prelaunch_check": True,
        "scope_validation": loaded.execution_scope_validation, "parent_flags_unchanged": True,
        "parent_mask_sha256_unchanged": True, "alpha_and_normalization_unchanged": True,
        "config_changed_fields": sorted(changed), "training_started": False, "optimizer_updates":0, "manual_A_smoke600_clearance":True,
        "gold_values_opened":False, "matched600_clearance":False, "health_status":"FAIL",
        "files": {"gate": bind(child_path), "scope": bind(out / "scope.json"), "config": bind(config)}}
    write(out / "report.json", report)
    write(out / "manifest.json", {"schema_version": scope.SCHEMA, "status": report["status"],
        "experiment_id": bound_scope["experiment_id"], "outputs": {n:bind(out/n) for n in ("scope.json","gate.json","report.json")}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(OUTPUT))
    parser.add_argument("--config", type=Path, default=Path(CONFIG))
    args = parser.parse_args()
    print(json.dumps(freeze(**vars(args)), ensure_ascii=False))


if __name__ == "__main__": main()
