"""Freeze experimental step normalization and citation features on train only.

The old bank's calibration/confirmation results have already been consumed.
They may be described as development reanalysis after a fixed fit, but are
never represented as fresh confirmation or used for model selection here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from kgproweight.data.parsers import parse_steps
from kgproweight.reward.source_credit_gate_v1 import FrozenSourceCreditMask
from kgproweight.reward.source_credit_gate_v2 import (
    ARTIFACT_SCHEMA, GATE_VERSION, NORMALIZATION_CONTRACT, SourceCreditGateV2,
)
from kgproweight.reward.source_quality_gate_v1 import (
    FEATURE_NAMES as OLD_NAMES, FEATURE_VERSION as OLD_VERSION,
    assign_family_splits, canonical_sha256, heuristic_ratio_target,
)
from kgproweight.reward.source_reward_normalization_v2 import (
    fit_text_normalization_v2, normalize_text_steps_v2,
)
from kgproweight.reward.source_trajectory_features_v2 import (
    FEATURE_NAMES, FEATURE_VERSION, compute_gate_features_v2,
)
from kgproweight.training.reward_function import validate_source_gate_trajectory
from scripts.train.calibrate_source_credit_gate_v1 import _spec, write_json, write_rows
from scripts.train.calibrate_source_quality_gate_v1 import _metrics


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SCHEMA = "source-credit-v2-repair-protocol-v1"
MANIFEST_SCHEMA = "source-credit-v2-development-calibration-manifest-v1"
CODE_FILES = (
    "kgproweight/reward/source_credit_gate_v2.py",
    "kgproweight/reward/source_reward_normalization_v2.py",
    "kgproweight/reward/source_trajectory_features_v2.py",
    "scripts/train/calibrate_source_credit_gate_v2.py",
    "kgproweight/data/parsers.py",
    "kgproweight/reward/source_quality_gate_v1.py",
    "kgproweight/reward/source_credit_gate_v1.py",
    "kgproweight/reward/source_integrity_v1.py",
    "kgproweight/reward/trajectory_source_gate.py",
    "kgproweight/reward/proofkg_process_v2_3.py",
    "kgproweight/reward/proofkg_process_v2_2.py",
    "kgproweight/reward/proofkg_process.py",
    "kgproweight/kg/question_kg.py",
    "kgproweight/training/reward_function.py",
    "scripts/train/calibrate_source_quality_gate_v1.py",
    "scripts/train/calibrate_source_credit_gate_v1.py",
)


def identity(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check_binding(binding: dict) -> Path:
    path = Path(binding["path"])
    if not path.is_absolute(): path = ROOT / path
    if identity(path)["sha256"] != binding["sha256"]:
        raise ValueError(f"bound bytes changed: {path}")
    return path


def load_parent(parent_dir: Path):
    """Verify original outputs and the exact historical source snapshots."""
    manifest = json.loads((parent_dir / "manifest.json").read_text())
    if manifest.get("schema_version") != "source-credit-gate-calibration-manifest-v1":
        raise ValueError("v2 requires the frozen real source-credit-v1 parent")
    if manifest.get("bank_source") != "real_frozen_policy_rollouts":
        raise ValueError("real source-credit parent required")
    for binding in manifest["outputs"].values(): check_binding(binding)
    for name, binding in manifest["code_bindings"].items():
        archived = parent_dir / "runtime_code" / name
        if identity(archived)["sha256"] != binding["sha256"]:
            raise ValueError(f"historical source snapshot mismatch: {name}")
    rows = [json.loads(line) for line in (parent_dir / "candidates.credit_masked.jsonl").read_text().splitlines()]
    assignments = [json.loads(line) for line in (parent_dir / "assignments.jsonl").read_text().splitlines()]
    splits = assign_family_splits(row["family_sha256"] for row in rows)
    if len(rows) != len(assignments) or len({r["candidate_id"] for r in rows}) != len(rows):
        raise ValueError("parent rows/assignments must be unique and complete")
    mask = FrozenSourceCreditMask.load(check_binding(manifest["source_credit_mask"]))
    for row, assignment in zip(rows, assignments):
        if row["candidate_id"] != assignment["candidate_id"] or assignment["split"] != splits[row["family_sha256"]]:
            raise ValueError("parent candidate/family split mismatch")
        if row.get("gold_access_for_gate_target") is not False:
            raise ValueError("Gold-free gate rows required")
        if any(key in row for key in ("gold", "gold_answer", "gold_target", "answer_aliases", "target")):
            raise ValueError("gate calibration cannot consume Gold labels")
        mask.validate_masked_features(row["features"])
        quality = heuristic_ratio_target(row["raw_graph"], row["raw_text"],
            m_graph=row["features"]["m_graph"], trajectory_valid=row["trajectory_valid"])
        if quality != row["quality"]:
            raise ValueError("parent ratio labels do not reproduce")
    return rows, splits, mask, json.loads((parent_dir / "gate.json").read_text())


def derive_feature_rows(rows, mask):
    result = []
    for parent in rows:
        row = deepcopy(parent)
        spec = _spec(row)
        # The full response is checked for format violations before the shared
        # runtime view caps scoring/telemetry at max_steps=5.  Invalid six-step
        # outputs remain invalid; this only aligns their diagnostic features.
        validity = validate_source_gate_trajectory(spec, row["generation"], format_version="v2")
        if validity["valid"] != row["trajectory_valid"]:
            raise ValueError("derived feature view changed frozen format validity")
        steps = validity["steps"]
        features = compute_gate_features_v2(spec, steps, row["proof_result"])
        row["features"] = mask.mask_features(spec, features)
        if row["features"]["m_graph"] != parent["features"]["m_graph"]:
            raise ValueError("feature revision must not change source eligibility")
        row["schema_version"] = "source-credit-candidate-row-v2"
        row["parent_credit_row_sha256"] = canonical_sha256(parent)
        result.append(row)
    return result


def fit_logistic(train, names, *, seed=42, epochs=800):
    """Same optimizer and budget as v1; no target or selection-rule changes."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if len(train) < 2 or any(row.get(key, "train") != "train" for row in train for key in ("split", "family_split")):
        raise ValueError("at least two train-only observations required")
    x = np.asarray([[row["features"]["values"][key] for key in names] for row in train], dtype=np.float64)
    y = np.asarray([row["quality"]["target"] for row in train], dtype=np.float64)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any((y < 0) | (y > 1)):
        raise ValueError("finite features and soft targets in [0,1] required")
    means, scales = x.mean(axis=0), np.maximum(x.std(axis=0), .1)
    standardized = (x - means) / scales
    weights, bias = np.random.default_rng(seed).normal(0, .01, len(names)), 0.0
    for _ in range(epochs):
        prediction = 1 / (1 + np.exp(-np.clip(standardized @ weights + bias, -60, 60)))
        error = prediction - y
        weights -= .05 * (standardized.T @ error / len(train) + .001 * weights)
        bias -= .05 * float(error.mean())
    return {"weights": weights.tolist(), "bias": bias,
        "feature_standardization": {"mean": dict(zip(names, means.tolist())),
            "scale": dict(zip(names, scales.tolist())), "fit_split": "train_eligible_nonabstain", "std_floor": .1}}


def reanalysis_metrics(rows, splits, gate):
    eligible = [r for r in rows if r["quality"]["target"] is not None]
    train_y = [r["quality"]["target"] for r in eligible if splits[r["family_sha256"]] == "train"]
    result = {}
    for split in ("train", "calibration", "confirmation"):
        selected = [r for r in eligible if splits[r["family_sha256"]] == split]
        result[split] = _metrics(np.asarray([r["quality"]["target"] for r in selected]),
            np.asarray([gate.predict(r["features"]) for r in selected]), float(np.mean(train_y)))
        result[split].update(families=len({r["family_sha256"] for r in selected}),
            interpretation="train_fit" if split == "train" else "already_consumed_split_development_reanalysis")
    return result


def feature_ties(rows, splits):
    grouped = {}
    for row in rows:
        if splits[row["family_sha256"]] == "train" and row["trajectory_valid"] and row["features"]["m_graph"]:
            grouped.setdefault((row["dataset"], row["qid"]), []).append(row)
    pairs = [value for value in grouped.values() if len(value) == 2]
    ties = sum(left["features"]["values"] == right["features"]["values"] for left, right in pairs)
    return {"scope":"all_train_both_valid_source_credit_question_pairs_no_EM_filter",
        "pairs":len(pairs), "feature_identical_pairs":ties,
        "fraction":ties / len(pairs) if pairs else None}


def calibrate(parent_dir: Path, protocol_path: Path, output_dir: Path):
    if output_dir.exists(): raise FileExistsError("refusing to overwrite v2 calibration")
    protocol_binding = identity(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("explicit frozen v2 protocol required")
    if protocol.get("feature_names") != list(FEATURE_NAMES) or protocol.get("feature_version") != FEATURE_VERSION:
        raise ValueError("feature design differs from frozen protocol")
    if protocol.get("epochs") != 800 or protocol.get("seed") != 42 or protocol.get("variants") != ["norm_only", "features_v2"]:
        raise ValueError("unexpected experimental variants or fit budget")
    check_binding(protocol["parent_manifest"])
    if Path(protocol["parent_manifest"]["path"]).resolve() != (parent_dir / "manifest.json").resolve():
        raise ValueError("protocol parent differs from requested parent")
    code_bindings = protocol.get("code_bindings")
    if not isinstance(code_bindings, dict) or not set(CODE_FILES).issubset(code_bindings):
        raise ValueError("protocol must bind the complete calibration code dependency set")
    for name, binding in code_bindings.items():
        if Path(binding["path"]).resolve() != (ROOT / name).resolve():
            raise ValueError("protocol code binding must identify the actual live source path")
        check_binding(binding)
    output_dir.mkdir(parents=True)
    write_json(output_dir / "started.json", {"experiment_id":protocol["experiment_id"], "policy_optimizer_updates":0})
    try:
        rows, splits, mask, parent_gate = load_parent(parent_dir)
        train_text = [{**r, "split":"train"} for r in rows if splits[r["family_sha256"]] == "train"]
        text_stats = fit_text_normalization_v2(train_text)
        feature_rows = derive_feature_rows(rows, mask)
        reports, bindings = {}, {}
        for variant, selected, version, names in (
            ("norm_only", rows, OLD_VERSION, OLD_NAMES),
            ("features_v2", feature_rows, FEATURE_VERSION, FEATURE_NAMES),
        ):
            dest = output_dir / variant
            dest.mkdir()
            artifact = deepcopy(parent_gate)
            artifact.pop("payload_sha256")
            artifact.update(schema_version=ARTIFACT_SCHEMA, gate_version=GATE_VERSION,
                experiment_id=protocol["experiment_id"] + "-" + variant.upper(),
                feature_version=version, feature_names=list(names), training_clearance=False,
                independent_confirmation_clearance=False, ppo_launch_clearance=False,
                training_clearance_scope="requires_fresh_confirmation_after_v2_design",
                diagnostic_fit=True, already_consumed_internal_confirmation=True,
                repair_protocol=protocol_binding, parent_artifact=identity(parent_dir / "gate.json"),
                scientific_boundary="Fixed train-only fit; old held-out splits are development reanalysis. No independent reliability or PPO-gain claim.")
            norm = artifact["normalization"]
            norm.pop("application_clip", None)
            norm.update(input_contract=NORMALIZATION_CONTRACT, text_v2=text_stats,
                text_center=text_stats["text_center"], text_scale=text_stats["text_scale"],
                text_application_scope=text_stats["application_contract"],
                text_fit_population="valid_train_equal_question_then_candidate_then_step",
                text_application_clip=None, graph_application_clip=[-1.0, 1.0])
            train = [{**r,"split":"train"} for r in selected if splits[r["family_sha256"]] == "train" and r["quality"]["target"] is not None]
            if variant == "features_v2":
                artifact.update(fit_logistic(train, names))
                artifact["fit"]["model_selection"] = "fixed_final_no_reanalysis_selection"
            artifact["payload_sha256"] = canonical_sha256(artifact)
            gate = SourceCreditGateV2(artifact, mask=mask, allow_unvalidated=True)
            if variant == "features_v2":
                valid_graph_train = [r for r in selected if splits[r["family_sha256"]] == "train" and r["trajectory_valid"] and r["features"]["m_graph"]]
                artifact["normalization"]["fixed_alpha"] = float(np.mean([gate.predict(r["features"]) for r in valid_graph_train]))
                artifact.pop("payload_sha256")
                artifact["payload_sha256"] = canonical_sha256(artifact)
                gate = SourceCreditGateV2(artifact, mask=mask, allow_unvalidated=True)
            assignments = [{"candidate_id":r["candidate_id"], "dataset":r["dataset"], "qid":r["qid"],
                "family_sha256":r["family_sha256"], "split":splits[r["family_sha256"]],
                "m_graph":r["features"]["m_graph"], "alpha":gate.predict(r["features"]), **r["quality"]} for r in selected]
            report = {"experiment_id":artifact["experiment_id"], "variant":variant,
                "status":"DEVELOPMENT_REANALYSIS_NOT_INDEPENDENT_CONFIRMATION",
                "training_clearance":False, "ppo_launch_clearance":False,
                "metrics":reanalysis_metrics(selected,splits,gate), "train_feature_ties":feature_ties(selected,splits),
                "normalization":artifact["normalization"], "candidate_count":len(selected),
                "valid_candidates":sum(r["trajectory_valid"] for r in selected),
                "graph_credit_questions":len({(r["dataset"],r["qid"]) for r in selected if r["features"]["m_graph"]}),
                "raw_scores_inputs_and_generations_unchanged":True,
                "gold_access_for_fit":False, "policy_optimizer_updates":0}
            write_json(dest / "gate.json", artifact)
            write_json(dest / "report.json", report)
            write_rows(dest / "candidates.jsonl", selected)
            write_rows(dest / "assignments.jsonl", assignments)
            bindings[variant] = {name:identity(dest / name) for name in ("gate.json","report.json","candidates.jsonl","assignments.jsonl")}
            reports[variant] = report
        # The control changes only text normalization: exact old predictor stays.
        old_assignments = {r["candidate_id"]:r for r in map(json.loads,(parent_dir / "assignments.jsonl").read_text().splitlines())}
        for assignment in map(json.loads,(output_dir / "norm_only/assignments.jsonl").read_text().splitlines()):
            if assignment["alpha"] != old_assignments[assignment["candidate_id"]]["alpha"]:
                raise ValueError("normalization-only control changed an alpha prediction")
        for name,binding in protocol["code_bindings"].items():
            source=check_binding(binding)
            dest=output_dir / "runtime_code" / name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,dest)
        if identity(protocol_path) != protocol_binding:
            raise ValueError("frozen repair protocol changed during calibration")
        check_binding(protocol["parent_manifest"])
        manifest={"schema_version":MANIFEST_SCHEMA,"experiment_id":protocol["experiment_id"],
            "created_at_utc":datetime.now(timezone.utc).isoformat(),
            "status":"V2_ENGINEERING_REPAIR_FITTED_NOT_INDEPENDENT_CONFIRMATION",
            "protocol":protocol_binding,"parent_manifest":identity(parent_dir / "manifest.json"),
            "source_credit_mask":parent_gate["source_credit_mask"],"code_bindings":protocol["code_bindings"],
            "outputs":bindings,"training_clearance":False,"ppo_launch_clearance":False,"policy_optimizer_updates":0,
            "normalization_only_alpha_exactly_unchanged":True,
            "scientific_boundary":"No Gold used for fitting. Old confirmation was consumed before this repair; it cannot certify the revised gate."}
        write_json(output_dir / "manifest.json",manifest)
        return manifest
    except Exception as exc:
        write_json(output_dir / "FAILED.json",{"error_type":type(exc).__name__,"error":str(exc),"policy_optimizer_updates":0})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(calibrate(args.parent_dir.resolve(),args.protocol.resolve(),args.output_dir.resolve()),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
