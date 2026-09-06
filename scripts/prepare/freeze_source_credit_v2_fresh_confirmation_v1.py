#!/usr/bin/env python
"""Freeze the single fresh132 confirmation before generation or label analysis.

Only existing answer-free inputs/manifests are decoded. Gold sources are hashed
as opaque bytes; their values are opened by the separate post-ranking analyzer.
No source release, gate, training clearance or evaluation baseline is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "source-credit-v2-fresh-confirmation-protocol-v1"
INPUT_DIR = ROOT / "outputs/audits/source_credit_v2_fresh_confirmation_inputs_20260906_v1"
SOURCE_DIR = ROOT / "outputs/audits/source_credit_v2_fresh_confirmation_source_20260906_v1"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def checked(info):
    path = Path(info.get("origin_path") or info["path"])
    if not path.is_absolute():
        path = ROOT / path
    if sha(path) != info["sha256"]:
        raise ValueError(f"frozen binding changed: {path}")
    return path


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def generation_contract():
    return {"candidates_per_question": 4, "greedy_per_question": 1,
            "greedy_candidate_index": 4, "batch_size": 1, "dtype": "bfloat16",
            "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
            "max_new_tokens": 384, "max_input_tokens": 6144,
            "eos_token_ids": [128001, 128009], "device": "cuda:0"}


def environment_identity():
    packages = {}
    for name in ("torch", "transformers", "peft", "numpy", "accelerate", "tokenizers", "safetensors", "trl"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    return {"python": sys.version, "executable": sys.executable, "packages": packages,
            "network": "local model files only; no teacher API",
            "model_weight_validation": "full SHA before freeze and before each applicable GPU stage"}


def freeze(out, experiment_id):
    from scripts.prepare.source_quality_candidate_bank_v1 import assert_gold_free, input_hash
    from scripts.prepare.generate_source_credit_v2_fresh_confirmation_v1 import GENERATION_CODE_FILES
    from scripts.prepare.score_source_credit_v2_fresh_confirmation_v1 import SCORING_CODE_FILES
    from scripts.pilot.analyze_source_credit_v2_fresh_confirmation_v1 import ANALYSIS_CODE_FILES, decision_rules

    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("fresh protocol output already exists; never overwrite")
    manifest_path = INPUT_DIR / "manifest.scope_v2.json"
    inputs_manifest = json.loads(manifest_path.read_text())
    source_manifest = json.loads((SOURCE_DIR / "manifest.json").read_text())
    inputs_path = checked(inputs_manifest["outputs"]["inputs.jsonl"])
    inputs = [json.loads(line) for line in inputs_path.read_text().splitlines()]
    if (len(inputs) != 132 or len({r["question_key"] for r in inputs}) != 132
            or len({r["family_sha256"] for r in inputs}) != 132):
        raise ValueError("fresh132 identity count changed")
    if Counter(r["dataset"] for r in inputs) != {
            "2wikimultihopqa": 108, "hotpotqa": 12, "musique": 12}:
        raise ValueError("fresh132 dataset population changed")
    for row in inputs:
        assert_gold_free(row)
        if input_hash(row) != row["input_sha256"] or len(row["retrieved_passages"]) != 10:
            raise ValueError("frozen prompt/evidence input changed")
    for info in inputs_manifest["source_bindings"].values():
        checked(info)
    for model in inputs_manifest["models"].values():
        model_dir = Path(model["path"])
        if not model_dir.is_absolute():
            model_dir = ROOT / model_dir
        for name, info in model["files"].items():
            if Path(name).name != name or sha(model_dir / name) != info["sha256"]:
                raise ValueError(f"frozen model/tokenizer changed: {model_dir / name}")
    for info in source_manifest["source_bindings"].values():
        checked(info)
    for info in source_manifest["outputs"].values():
        checked(info)
    source_checks = [json.loads(line) for line in checked(
        source_manifest["outputs"]["question_checks.jsonl"]).read_text().splitlines()]
    graph_keys = {r["question_key"] for r in inputs if r["confirmation_identity_proposal_role"] == "graph"}
    if (len(source_checks) != 96 or {r["question_key"] for r in source_checks} != graph_keys
            or Counter(r["status"] for r in source_checks) != {"PASS": 79, "UNVERIFIED": 11, "FAIL": 6}):
        raise ValueError("fresh source-credit population changed")
    gate_bindings = {v: bind(SOURCE_DIR / v / "gate.json") for v in ("norm_only", "features_v2")}
    for info in gate_bindings.values():
        gate = json.loads(checked(info).read_text())
        if any(gate.get(k) is True for k in (
                "training_clearance", "independent_confirmation_clearance", "ppo_launch_clearance")):
            raise ValueError("unconsumed confirmation cannot already authorize training")

    # These existing manifests bind label-bearing files without decoding labels.
    graph_report = json.loads(checked(inputs_manifest["source_bindings"]["unified_report"]).read_text())
    gold_sources = {
        "graph": bind(checked(graph_report["outputs"]["silver_train"])),
        "ordinary": bind(checked(inputs_manifest["source_bindings"]["ordinary_evidence_projection_source"]))}
    code_files = sorted(set(GENERATION_CODE_FILES) | set(SCORING_CODE_FILES) | set(ANALYSIS_CODE_FILES) | {
        "scripts/prepare/freeze_source_credit_v2_fresh_confirmation_v1.py",
        "scripts/pilot/analyze_source_credit_v2_fresh_confirmation_v1.py",
        "scripts/pilot/run_source_credit_v2_fresh_confirmation_v1.py",
        "kgproweight/eval/pred_processing.py"})
    code_bindings = {name: bind(ROOT / name) for name in code_files}
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT)
    config_names = [f"configs/training/phase3_ppo_mixed4_answer_format_v2_{arm}_{stage}_seed42.yaml"
                    for arm in ("a", "f", "t") for stage in ("probe", "smoke")]
    protocol = {
        "schema_version": SCHEMA, "status": "FROZEN", "experiment_id": experiment_id,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "seed": 42,
        "bindings": {"inputs_manifest": bind(manifest_path), "inputs": bind(inputs_path),
                     "source_manifest": bind(SOURCE_DIR / "manifest.json"),
                     "source_checks": bind(checked(source_manifest["outputs"]["question_checks.jsonl"])),
                     "cohort": bind(checked(inputs_manifest["source_bindings"]["proposal_cohort"]))},
        "code_bindings": code_bindings,
        "runtime_environment": environment_identity(),
        "code_version": {"git_head": git_head, "worktree_dirty": bool(git_status),
                         "status_sha256": hashlib.sha256(git_status).hexdigest(),
                         "executed_code_snapshot": "code_snapshot; exact file SHA takes precedence over dirty HEAD"},
        "generation": generation_contract(),
        "scoring": {"format_version": "v2", "max_steps": 5, "ordinary_min_steps": 3,
                    "min_reasoning_chars": 20, "text_backend": "rearag", "dtype": "bf16",
                    "max_text_length": 4096, "process_weights": {"text": .3, "graph": .2},
                    "rank_samples": 4, "greedy_in_ranking": False,
                    "rank_tie_break": "candidate_index_ascending", "gates": gate_bindings,
                    "rearag_model": inputs_manifest["models"]["rearag_model"]},
        "analysis": {"primary_variant": "features_v2", "secondary_variant": "norm_only",
                     "gold_sources": gold_sources, "bootstrap_replicates": 20000,
                     "bootstrap_seed": 42, "confidence_level": .95,
                     "resampling_unit": "family; pairwise averaged within question first",
                     "bootstrap_strata": "graph_question_type_or_ordinary_dataset",
                     "greedy_comparator": "raw canonical EM/F1, invalid greedy retained; valid-only also reported",
                     "decision_rules": decision_rules(),
                     "gold_boundary": "open values only after exact 660 process rows and 132 rankings are sealed",
                     "process_ranking_excludes_answer_reward": True,
                     "all_invalid_sample_question": "retain question; no selected candidate; top1 EM/F1 zero",
                     "variants_never_selected_or_refit_using_confirmation": True,
                     "no_adaptive_resampling_or_identity_replacement": True},
        "population": {"questions": 132, "sampled": 528, "greedy": 132, "graph": 96,
                       "source_pass": 79, "source_unverified": 11, "source_fail": 6,
                       "ordinary": 36, "graph_types": {"bridge_comparison": 32,
                           "comparison": 32, "compositional": 32}, "inference_coverage": 0,
                       "scope": inputs_manifest["confirmation_scope"],
                       "not_balanced_three_dataset_performance_estimate": True},
        "preserved_training": {"policy_path": inputs_manifest["policy_path"],
                               "source_credit_parent_gates": {v: bind(ROOT /
                                   "outputs/calibration/source_credit_gate_v2_representative_local_seed42_20260906_v1" / v / "gate.json")
                                   for v in ("norm_only", "features_v2")},
                               "production_mask_rule": "derive from original training gate/mask (800 inputs, 671 credit PASS); confirmation-only96 mask must never replace the training mask",
                               "formula_valid": "4*(EM+0.1*F1)+0.30*(1-alpha)*TN+0.20*alpha*GN",
                               "answer_format_reward_version": "v2",
                               "configs": {name: bind(ROOT / name) for name in config_names},
                               "new_sft": "PAUSED", "full_ppo_auto_launch": False},
        "clearance": {"training_clearance": False, "independent_confirmation_clearance": False,
                      "ppo_launch_clearance": False,
                      "release_rule": "new derived gate only after verified integrity PASS and independent_utility_status PASS; never edit parent gate flags",
                      "probe_scope": "separately bound maximum12-trajectory engineering execution; health alone does not veto it",
                      "matched600_auto_investment": "overall_status PASS, then successful engineering probe; otherwise explicit bounded follow-up decision",
                      "full_ppo_auto_launch": False},
        "optimizer_updates": 0, "gold_values_opened": False,
        "boundary": "Frozen fresh-family pre-PPO diagnostic, not trained PPO benefit or proof of alpha superiority"}
    out.mkdir(parents=True, exist_ok=False)
    for name, info in code_bindings.items():
        target = out / "code_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checked(info), target)
        if sha(target) != info["sha256"]:
            raise ValueError("code snapshot mismatch")
    write_new(out / "protocol.json", protocol)
    write_new(out / "manifest.json", {"schema_version": SCHEMA, "experiment_id": experiment_id,
                                     "status": "FROZEN_NOT_GENERATED_NOT_TRAINED",
                                     "protocol": bind(out / "protocol.json")})
    print(json.dumps({"status": "FROZEN", "protocol": bind(out / "protocol.json"),
                      "questions": 132, "candidates": 660, "gold_values_opened": False}), flush=True)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    freeze(args.out, args.experiment_id)


if __name__ == "__main__":
    main()
