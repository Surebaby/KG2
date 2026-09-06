#!/usr/bin/env python3
"""Fit a small, versioned source-credit gate from a frozen train-only bank.

This is CPU logistic regression against a heuristic score ratio.  Its Brier
metrics measure heuristic fidelity, not independent source reliability.  Real
PPO launch additionally requires independent process rankability and approval.
Synthetic fitting is explicit, diagnostic-only, and never production-loadable.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from kgproweight.kg.question_kg import question_sha256
from kgproweight.reward.proofkg_process_v2_3 import SCORER_VERSION, build_execution_trace_v2_3, score_proofkg_v2_3
from kgproweight.reward.source_quality_gate_v1 import (
    ARTIFACT_SCHEMA, DENOMINATOR_EPSILON, FEATURE_NAMES, FEATURE_VERSION,
    GATE_VERSION, QUALITY_ABSTAIN_THRESHOLD, SOURCE_SCALE_FLOOR, SPLIT_VERSION,
    TARGET_VERSION, SourceQualityGateV1, assign_family_splits, canonical_sha256,
    compute_gate_features, heuristic_ratio_target,
)
from kgproweight.training.reward_function import (
    source_gate_format_contract_version, validate_source_gate_trajectory,
)
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256


ROOT = Path(__file__).resolve().parents[2]
TEXT_CONTRACT = "rearag-passage-only-raw-tanh-nll-v1"
BANK_SCHEMA = "source-quality-candidate-bank-v1"
ISOLATION_SCHEMA = "source-quality-bank-isolation-v1"
ROW_SCHEMA = "source-quality-candidate-row-v1"
SPLIT_SEED = 42


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def _optional_git_head() -> str | None:
    """Portable releases may contain the bound sources without a .git tree."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _bound_path(binding: Mapping[str, Any], base: Path) -> Path:
    if not isinstance(binding, Mapping) or not binding.get("path") or len(str(binding.get("sha256") or "")) != 64:
        raise ValueError("file identity with path and SHA256 required")
    value = Path(str(binding["path"]))
    candidates = [value] if value.is_absolute() else [base / value, ROOT / value]
    for path in candidates:
        if path.is_file() and _identity(path)["sha256"] == binding["sha256"]:
            return path.resolve()
    raise ValueError(f"bound file missing or hash mismatch: {binding['path']}")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_bank(bank_manifest: Path, isolation_proof: Path, *, synthetic_test_only: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(bank_manifest.read_text(encoding="utf-8"))
    format_contract = manifest.get("format_contract_version", source_gate_format_contract_version("v1"))
    format_versions = {source_gate_format_contract_version(version): version for version in ("v1", "v2")}
    if format_contract not in format_versions:
        raise ValueError("candidate bank format contract version is unsupported")
    format_version = format_versions[format_contract]
    source_integrity = {}
    if format_version == "v2":
        if (not isinstance(manifest.get("source_integrity_clearance"), bool)
                or not isinstance(manifest.get("source_integrity_status"), str)
                or not manifest["source_integrity_status"].strip()):
            raise ValueError("v2 bank requires explicit source_integrity_clearance and source_integrity_status")
        source_integrity = {key: manifest[key] for key in (
            "source_integrity_clearance", "source_integrity_status")}
    if manifest.get("schema_version") != BANK_SCHEMA or manifest.get("feature_version") != FEATURE_VERSION or manifest.get("graph_scorer_version") != SCORER_VERSION or manifest.get("text_score_contract") != TEXT_CONTRACT:
        raise ValueError("candidate bank schema/scorer/feature/text contract mismatch")
    source = manifest.get("bank_source")
    expected_status = "SYNTHETIC_TEST_ONLY" if source == "synthetic" else "TRAIN_ONLY_CANDIDATES_FROZEN"
    if source not in {"synthetic", "real_frozen_policy_rollouts"} or manifest.get("status") != expected_status:
        raise ValueError("candidate bank source/status is not an explicit frozen release")
    if (source == "synthetic") != synthetic_test_only:
        raise ValueError("synthetic data require the explicit synthetic-test-only mode")
    if manifest.get("gold_access_for_gate_target") is not False:
        raise ValueError("bank must explicitly forbid gold access for gate targets")
    bank_path = _bound_path(manifest["bank"], bank_manifest.parent)
    bound_isolation = _bound_path(manifest["isolation_proof"], bank_manifest.parent)
    if bound_isolation != isolation_proof.resolve():
        raise ValueError("CLI isolation proof does not match the bank's frozen binding")
    isolation = json.loads(isolation_proof.read_text(encoding="utf-8"))
    if isolation.get("schema_version") != ISOLATION_SCHEMA or isolation.get("status") != "PASS" or isolation.get("bank_sha256") != _identity(bank_path)["sha256"] or isolation.get("family_version") != FAMILY_VERSION:
        raise ValueError("bank isolation proof is missing, stale, or failed")
    if isolation.get("overlap_counts") != {"qid": 0, "question_sha256": 0, "family_sha256": 0}:
        raise ValueError("bank does not declare triple-zero protected-set isolation")
    bindings = manifest.get("source_bindings") or {}
    if not isinstance(bindings, dict) or len(bindings) < 3:
        raise ValueError("bank must bind original evidence, source rows and policy artifacts")
    resolved_sources = {name: {"path": str(_bound_path(binding, bank_manifest.parent)), "sha256": str(binding["sha256"])} for name, binding in bindings.items()}
    ledger_path = _bound_path(isolation["protected_ledger_binding"], isolation_proof.parent)
    ledger = _jsonl(ledger_path)
    protected_qids = {str(row["qid"]) for row in ledger}
    protected_questions = {question_sha256(str(row["question"])) for row in ledger}
    protected_families = {family_sha256(str(row["question"])) for row in ledger}
    rows = _jsonl(bank_path)
    if not rows:
        raise ValueError("empty candidate bank")
    seen: set[str] = set()
    identities: dict[tuple[str, str], tuple[str, str]] = {}
    validated = []
    for row in rows:
        if row.get("schema_version") != ROW_SCHEMA or not isinstance(row.get("source_bindings"), dict) or not row["source_bindings"]:
            raise ValueError("candidate row lacks its versioned evidence/source contract")
        # Do not consume a stored target even if someone adds one later.
        if any(key in row for key in ("gold", "gold_answer", "gold_target", "target", "answer_aliases")):
            raise ValueError("gate calibration row must not carry gold/target labels")
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id in seen:
            raise ValueError("candidate_id must be nonempty and unique")
        seen.add(candidate_id)
        dataset, qid, question = str(row["dataset"]), str(row["qid"]), str(row["question"])
        qhash, family = question_sha256(question), family_sha256(question)
        if row.get("question_sha256") != qhash or row.get("family_sha256") != family or row.get("family_version") != FAMILY_VERSION:
            raise ValueError("candidate current-family/question identity mismatch")
        if qid in protected_qids or qhash in protected_questions or family in protected_families:
            raise ValueError("candidate overlaps the actual protected identity ledger")
        key = (dataset, qid)
        if key in identities and identities[key] != (qhash, family):
            raise ValueError("same question identity has conflicting family/question values")
        identities[key] = (qhash, family)
        record = row["source_quality_record"]
        spec = SimpleNamespace(query=question, kg_subgraph=record.get("kg_subgraph") or [], retrieved_passages=row["retrieved_passages"], metadata={"dataset": dataset, "qid": qid, "source_quality_record": record})
        generation = str(row["generation"])
        format_result = validate_source_gate_trajectory(spec, generation, format_version=format_version)
        format_fields = {key: format_result[key] for key in ("valid", "violations", "all_step_count", "required_steps", "contract_version")}
        if row.get("trajectory_valid") is not format_result["valid"] or canonical_sha256(row.get("format_validation")) != canonical_sha256(format_fields):
            raise ValueError("stored trajectory validity does not reproduce the shared PPO format contract")
        steps = format_result["steps"]
        stored_proof = row["proof_result"]
        features = compute_gate_features(spec, steps, stored_proof)
        if canonical_sha256(features) != canonical_sha256(row["features"]):
            raise ValueError("stored features do not reproduce from the frozen evidence/trajectory")
        proof = stored_proof
        if features["m_graph"]:
            plan, execution = record.get("query_plan") or {}, record.get("execution") or {}
            proof = score_proofkg_v2_3(question=question, generation=generation, kg_triples=spec.kg_subgraph, execution_trace=build_execution_trace_v2_3(plan, execution), planned_hops=len(plan.get("hops") or []))
            if canonical_sha256(proof) != canonical_sha256(stored_proof) or float(row["raw_graph"]) != proof["score"]:
                raise ValueError("stored graph reward does not reproduce with frozen v2.3")
        raw_text = row["raw_text"]
        if not isinstance(raw_text, list) or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v) or not -1 <= v <= 1 for v in raw_text):
            raise ValueError("raw_text must contain finite original ReaRAG scores in [-1,1]")
        # Format validity belongs to the shared PPO contract, not the graph
        # scorer.  A passage-only control can be valid with no scored graph.
        trajectory_valid = format_result["valid"]
        if trajectory_valid and len(raw_text) != len(steps):
            raise ValueError("valid trajectory must bind one ReaRAG score per parsed step")
        quality = heuristic_ratio_target(row["raw_graph"], raw_text, m_graph=features["m_graph"], trajectory_valid=trajectory_valid)
        validated.append({**row, "features": features, "quality": quality, "trajectory_valid": trajectory_valid})
    return validated, {
        "bank_source": source, "bank_manifest": _identity(bank_manifest), "bank": _identity(bank_path),
        "format_contract_version": format_contract,
        **source_integrity,
        "isolation_proof": _identity(isolation_proof), "protected_ledger": _identity(ledger_path),
        "source_bindings": resolved_sources, "input_manifest": manifest,
        "recomputed_overlap_counts": {"qid": 0, "question_sha256": 0, "family_sha256": 0},
    }


def _distribution(values: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {"n": 0}
    return {"n": len(values), "mean": float(np.mean(values)), "std": float(np.std(values)), "min": float(np.min(values)), "p05": float(np.quantile(values, .05)), "median": float(np.median(values)), "p95": float(np.quantile(values, .95)), "max": float(np.max(values))}


def _metrics(target: np.ndarray, prediction: np.ndarray, constant: float) -> dict[str, Any]:
    if not len(target):
        return {"n": 0, "brier": None, "constant_brier": None, "r2_vs_train_constant": None, "prediction": {"n": 0}}
    brier = float(np.mean((prediction - target) ** 2))
    constant_brier = float(np.mean((constant - target) ** 2))
    counts = Counter(np.round(prediction, 6).tolist())
    denominator = len(target) * (len(target) - 1) / 2
    ties = sum(count * (count - 1) / 2 for count in counts.values())
    return {
        "n": len(target), "brier": brier, "constant_brier": constant_brier,
        "r2_vs_train_constant": 1 - brier / constant_brier if constant_brier > 1e-12 else None,
        "bce": float(np.mean(-target * np.log(np.clip(prediction, 1e-12, 1)) - (1 - target) * np.log(np.clip(1 - prediction, 1e-12, 1)))),
        "prediction_tie_rate_rounded_1e6": ties / denominator if denominator else 0.0,
        "prediction": _distribution(prediction), "target": _distribution(target),
        "target_mass": {"zero": float(np.mean(target == 0)), "fractional": float(np.mean((target > 0) & (target < 1))), "one": float(np.mean(target == 1))},
    }


def fit_gate(rows: Sequence[Mapping[str, Any]], binding: Mapping[str, Any], *, experiment_id: str, seed: int = 42, epochs: int = 800) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    source_integrity = {key: binding[key] for key in (
        "source_integrity_clearance", "source_integrity_status") if key in binding}
    if binding.get("format_contract_version") == source_gate_format_contract_version("v2"):
        if (not isinstance(source_integrity.get("source_integrity_clearance"), bool)
                or not isinstance(source_integrity.get("source_integrity_status"), str)
                or not source_integrity["source_integrity_status"].strip()):
            raise ValueError("v2 fit requires explicit source integrity binding")
    splits = assign_family_splits((str(row["family_sha256"]) for row in rows), seed=SPLIT_SEED)
    fit_rows = [row for row in rows if row["quality"]["target"] is not None]
    train = [row for row in fit_rows if splits[row["family_sha256"]] == "train"]
    if len(train) < 2:
        raise ValueError("at least two non-abstaining eligible train candidates required")
    vector = lambda row: [row["features"]["values"][key] for key in FEATURE_NAMES]
    train_x = np.asarray([vector(row) for row in train], dtype=np.float64)
    train_y = np.asarray([row["quality"]["target"] for row in train], dtype=np.float64)
    means, scales = train_x.mean(axis=0), np.maximum(train_x.std(axis=0), .1)
    standardized = (train_x - means) / scales
    rng = np.random.default_rng(seed)
    weights, bias = rng.normal(0, .01, len(FEATURE_NAMES)), 0.0
    rate, l2 = .05, .001
    for _ in range(epochs):
        prediction = 1 / (1 + np.exp(-np.clip(standardized @ weights + bias, -60, 60)))
        error = prediction - train_y
        weights -= rate * (standardized.T @ error / len(train) + l2 * weights)
        bias -= rate * float(error.mean())
    predict = lambda subset: 1 / (1 + np.exp(-np.clip((np.asarray([vector(row) for row in subset], dtype=np.float64).reshape(-1, len(FEATURE_NAMES)) - means) / scales @ weights + bias, -60, 60)))
    metrics = {}
    for split in ("train", "calibration", "confirmation"):
        selected = [row for row in fit_rows if splits[row["family_sha256"]] == split]
        metrics[split] = _metrics(np.asarray([row["quality"]["target"] for row in selected]), predict(selected), float(train_y.mean()))
        metrics[split]["families"] = len({row["family_sha256"] for row in selected})
    valid_train = [row for row in rows if splits[row["family_sha256"]] == "train" and row["trajectory_valid"] and row["raw_text"]]
    graph_train = [row for row in valid_train if row["features"]["m_graph"]]
    if not graph_train or not valid_train:
        raise ValueError("source normalization requires valid train-only source observations")
    graphs = np.asarray([row["raw_graph"] for row in graph_train], dtype=np.float64)
    texts = np.asarray([np.mean(row["raw_text"]) for row in valid_train], dtype=np.float64)
    fixed_alpha = float(predict(graph_train).mean())
    gates = {"train_family_count_at_least_20": metrics["train"]["families"] >= 20}
    for split in ("calibration", "confirmation"):
        score = metrics[split]
        gates[f"{split}_families_at_least_10"] = score["families"] >= 10
        gates[f"{split}_brier_beats_constant"] = score["brier"] is not None and score["brier"] + 1e-6 < score["constant_brier"]
        gates[f"{split}_r2_positive"] = score["r2_vs_train_constant"] is not None and score["r2_vs_train_constant"] > 0
        gates[f"{split}_prediction_std_at_least_0_01"] = score["prediction"].get("std", 0) >= .01
    clearance = binding["bank_source"] == "real_frozen_policy_rollouts" and all(gates.values())
    artifact = {
        "schema_version": ARTIFACT_SCHEMA, "gate_version": GATE_VERSION, "feature_version": FEATURE_VERSION,
        "format_contract_version": binding.get("format_contract_version", source_gate_format_contract_version("v1")),
        **source_integrity,
        "feature_names": list(FEATURE_NAMES), "target_version": TARGET_VERSION,
        "target_semantics": "heuristic score ratio; not independent source reliability",
        "experiment_id": experiment_id, "bank_source": binding["bank_source"], "training_clearance": clearance,
        "training_clearance_scope": "heuristic_gate_calibration_only_not_ppo_launch_or_process_rankability",
        "weights": weights.tolist(), "bias": float(bias),
        "feature_standardization": {"mean": dict(zip(FEATURE_NAMES, means.tolist())), "scale": dict(zip(FEATURE_NAMES, scales.tolist())), "fit_split": "train_eligible_nonabstain", "std_floor": .1},
        "normalization": {
            "graph_center": float(graphs.mean()), "graph_scale": float(max(graphs.std(), SOURCE_SCALE_FLOOR)),
            "text_center": float(texts.mean()), "text_scale": float(max(texts.std(), SOURCE_SCALE_FLOOR)),
            "fixed_alpha": fixed_alpha, "input_contract": "raw_v23_graph_score_and_mean_raw_rearag_step_scores",
            "text_application_scope": "step_normalize_then_clip_then_mean_v1",
            "graph_application_scope": "trajectory_normalize_then_clip_v1",
            "application_clip": [-1.0, 1.0],
            "graph_fit_population": "valid_eligible_train_including_target_abstentions", "text_fit_population": "valid_train_including_passage_controls",
            "fixed_alpha_population": "valid_eligible_train_candidate_mean_including_target_abstentions",
            "std_floor": SOURCE_SCALE_FLOOR, "ema_used": False,
        },
        "fit": {"optimizer": "numpy_full_batch_logistic_soft_bce", "seed": seed, "epochs": epochs, "learning_rate": rate, "l2": l2, "model_selection": "fixed_final_no_confirmation_selection"},
        "split": {"version": SPLIT_VERSION, "seed": SPLIT_SEED, "fractions": [.6, .2, .2], "unit": "current_lexical_family"},
        "target_abstention": {"both_quality_at_most": QUALITY_ABSTAIN_THRESHOLD, "denominator_at_most": DENOMINATOR_EPSILON},
        "input_bindings": {key: value for key, value in binding.items() if key != "input_manifest"},
        "heuristic_fidelity_gates": gates,
    }
    artifact["payload_sha256"] = canonical_sha256(artifact)
    # Validate the serializable artifact through its real production reader.
    gate = SourceQualityGateV1(artifact, allow_synthetic=True, allow_unvalidated=True)
    assignments = [{"candidate_id": row["candidate_id"], "dataset": row["dataset"], "qid": row["qid"], "family_sha256": row["family_sha256"], "split": splits[row["family_sha256"]], "m_graph": row["features"]["m_graph"], **row["quality"], "alpha": gate.predict(row["features"])} for row in rows]
    report = {
        "experiment_id": experiment_id, "status": "REAL_BANK_HEURISTIC_GATE_CALIBRATED_NOT_PPO_CLEARANCE" if clearance else "SYNTHETIC_FIT_NOT_TRAINING_CLEARANCE" if binding["bank_source"] == "synthetic" else "FAIL_HEURISTIC_GATE_CALIBRATION_NOT_TRAINING_CLEARANCE",
        "format_contract_version": artifact["format_contract_version"],
        **source_integrity,
        "training_clearance": clearance, "metrics": metrics, "gates": gates,
        "family_counts": dict(Counter(splits.values())), "candidate_counts": dict(Counter(row["split"] for row in assignments)),
        "abstention_counts": dict(Counter(row["quality"]["abstain_reason"] or "used_for_fit" for row in rows)),
        "passage_only_fail_closed": all(row["alpha"] == 0 for row in assignments if not row["m_graph"]),
        "normalization": artifact["normalization"],
        "scientific_boundary": "Brier/R2/tie metrics are fidelity to a constructed score ratio, not independent evidence of source reliability, process quality, or PPO utility. No candidate-generation model was trained. Synthetic artifacts cannot authorize real PPO; process-only zero-update rankability remains a separate gate.",
    }
    return artifact, report, assignments


def calibrate(bank_manifest: Path, isolation_proof: Path, output_dir: Path, *, experiment_id: str, seed: int = 42, epochs: int = 800, synthetic_test_only: bool = False) -> dict[str, Any]:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("refusing to overwrite an existing gate calibration artifact")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    def write_json(name, value):
        with (output_dir / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    try:
        rows, bindings = validate_bank(bank_manifest, isolation_proof, synthetic_test_only=synthetic_test_only)
        artifact, report, assignments = fit_gate(rows, bindings, experiment_id=experiment_id, seed=seed, epochs=epochs)
        write_json("gate.json", artifact)
        write_json("report.json", report)
        with (output_dir / "assignments.jsonl").open("x", encoding="utf-8") as handle:
            for row in assignments:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        manifest = {
            "schema_version": "source-quality-gate-calibration-manifest-v1", "experiment_id": experiment_id,
            "status": report["status"], "created_at_utc": started, "seed": seed,
            "training_clearance": artifact["training_clearance"], "bank_source": bindings["bank_source"],
            "format_contract_version": artifact["format_contract_version"],
            **{key: artifact[key] for key in ("source_integrity_clearance", "source_integrity_status") if key in artifact},
            "inputs": {key: value for key, value in bindings.items() if key != "input_manifest"},
            "outputs": {name: _identity(output_dir / name) for name in ("gate.json", "report.json", "assignments.jsonl")},
            "git_head": _optional_git_head(),
            "code_identity_contract": "actual_source_sha256_bindings_authoritative; git_head_optional_for_portable_releases",
            "code_bindings": {str(path): _identity(path) for path in (Path(__file__), ROOT / "kgproweight/reward/source_quality_gate_v1.py", ROOT / "kgproweight/reward/proofkg_process_v2_3.py", ROOT / "kgproweight/training/reward_function.py", ROOT / "scripts/prepare/freeze_qpeg_v1_protocol.py")},
            "policy_optimizer_updates": 0, "light_gate_fit_epochs": epochs, "evaluation_protocol": TARGET_VERSION,
        }
        write_json("manifest.json", manifest)
        return manifest
    except Exception as exc:
        write_json("FAILED_CALIBRATION.json", {"experiment_id": experiment_id, "status": "FAIL_CALIBRATION_NOT_TRAINING_CLEARANCE", "created_at_utc": started, "error_type": type(exc).__name__, "error": str(exc), "policy_optimizer_updates": 0})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--isolation-proof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--synthetic-test-only", action="store_true")
    args = parser.parse_args()
    result = calibrate(args.bank_manifest, args.isolation_proof, args.output_dir, experiment_id=args.experiment_id, seed=args.seed, epochs=args.epochs, synthetic_test_only=args.synthetic_test_only)
    print(json.dumps({"status": result["status"], "training_clearance": result["training_clearance"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
