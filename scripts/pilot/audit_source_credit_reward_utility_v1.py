"""Independent train-only EM/ranking diagnostic after source-credit gate fitting.

Outcome labels enter this audit only. Its rows are not calibration inputs,
its pairwise metrics are not independent confirmation, and it never authorizes
PPO. Frozen Reader generations and every training artifact remain untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from scripts.prepare import source_quality_candidate_bank_v1 as banklib
from kgproweight.eval.pred_processing import extract_kg_proweight_answer
from kgproweight.reward.proofkg_process import canonical_exact_match, canonical_token_f1
from kgproweight.reward.source_credit_gate_v1 import SourceCreditGateV1
from kgproweight.reward.source_quality_gate_v1 import FEATURE_NAMES, canonical_sha256
from kgproweight.training.reward_function import _canonical_gold_surfaces
from scripts.train import calibrate_source_credit_gate_v1 as creditlib
from scripts.train import calibrate_source_quality_gate_v1 as fitlib


SCHEMA = "source-credit-reward-utility-diagnostic-v1"
ROW_SCHEMA = "source-credit-reward-utility-diagnostic-row-v1"
RANK_FIELDS = (
    "graph_full_raw_diagnostic", "graph_structure_raw_diagnostic",
    "rearag_raw_mean_diagnostic", "text_T_process", "learned_A_process", "fixed_F_process",
)
BOUNDARY = (
    "Descriptive train-only K2 frozen-SFT candidate diagnostic after fitting. Canonical EM hit/non-hit "
    "is not human semantic correctness. Outcome labels are used only here, never in source masking "
    "or gate fitting. Process-only rankings exclude outcome reward and invalid trajectories. "
    "A-F ranking differences are not PPO gains or independent process-utility confirmation. "
    "Graph raw diagnostic scores remain visible even when their actual reward credit is zero."
)


def distribution(values):
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"n": 0}
    if not np.all(np.isfinite(array)):
        raise ValueError("nonfinite diagnostic values")
    return {"n": len(array), "mean": float(array.mean()), "std": float(array.std()),
            **{key: float(np.quantile(array, q)) for key, q in (
                ("min", 0), ("p05", .05), ("p50", .5), ("p95", .95), ("max", 1))}}


def process_components(row, gate):
    """Reproduce the source-gated process terms without adding outcome reward."""
    features, norm = row["features"], gate.normalization
    # The gate must validate the marker even for masked-out candidates.
    alpha_a = gate.predict(features)
    if not math.isfinite(alpha_a) or not 0 <= alpha_a <= 1:
        raise ValueError("invalid learned alpha")
    alpha_f = float(norm["fixed_alpha"]) if features["m_graph"] else 0.0
    if not features["m_graph"] and (alpha_a != 0 or alpha_f != 0):
        raise ValueError("masked graph acquired nonzero alpha")
    scores = row["raw_text"]
    if any(not math.isfinite(value) or not -1 <= value <= 1 for value in scores):
        raise ValueError("invalid raw text reward")
    graph = float(row["raw_graph"])
    if not math.isfinite(graph):
        raise ValueError("nonfinite graph reward")
    valid = row["trajectory_valid"]
    if valid and not scores:
        raise ValueError("valid candidate lacks step scores")
    text_unclipped = [(value - norm["text_center"]) / norm["text_scale"] for value in scores] if valid else []
    text_clipped = [max(-1.0, min(1.0, value)) for value in text_unclipped]
    tn = sum(text_clipped) / len(text_clipped) if text_clipped else 0.0
    gn = max(-1.0, min(1.0, (graph - norm["graph_center"]) / norm["graph_scale"])) if valid else 0.0
    graph_a = 0.2 * alpha_a * gn if valid else 0.0
    graph_f = 0.2 * alpha_f * gn if valid else 0.0
    result = {
        "alpha_A": alpha_a if valid else None, "alpha_F": alpha_f if valid else None,
        "text_normalized_step_mean": tn, "graph_normalized": gn,
        "text_step_count": len(text_unclipped),
        "text_clipped_step_count": sum(abs(value) > 1.0 for value in text_unclipped),
        "text_T_process": 0.3 * tn,
        "learned_A_graph_component": graph_a, "fixed_F_graph_component": graph_f,
        "learned_A_text_component": 0.3 * (1 - alpha_a) * tn if valid else 0.0,
        "fixed_F_text_component": 0.3 * (1 - alpha_f) * tn if valid else 0.0,
        "graph_full_raw_diagnostic": graph,
        "graph_structure_raw_diagnostic": float((row["proof_result"].get("telemetry") or {}).get("structural_component", 0.0)),
        "rearag_raw_mean_diagnostic": sum(scores) / len(scores) if scores else None,
    }
    result["learned_A_process"] = result["learned_A_text_component"] + graph_a
    result["fixed_F_process"] = result["fixed_F_text_component"] + graph_f
    return result


def _rank_value(delta):
    return 1.0 if delta > 1e-10 else 0.0 if delta < -1e-10 else 0.5


def paired_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["question_key"]].append(row)
    if any(len(pair) != 2 for pair in grouped.values()):
        raise ValueError("process utility requires exact K2 groups")
    selected = []
    for pair in grouped.values():
        if all(row["valid"] for row in pair) and pair[0]["em"] != pair[1]["em"]:
            selected.append(sorted(pair, key=lambda row: row["em"], reverse=True))
    rankings = {}
    for field in RANK_FIELDS:
        counts = Counter()
        for hit, nonhit in selected:
            value = _rank_value(hit[field] - nonhit[field])
            counts[{1.0: "win", .5: "tie", 0.0: "loss"}[value]] += 1
        rankings[field] = {key: counts[key] for key in ("win", "tie", "loss")}
        rankings[field]["tie_adjusted_accuracy"] = (counts["win"] + .5 * counts["tie"]) / len(selected) if selected else None
    rank_deltas, margin_deltas = [], []
    transition = Counter()
    for hit, nonhit in selected:
        a_margin = hit["learned_A_process"] - nonhit["learned_A_process"]
        f_margin = hit["fixed_F_process"] - nonhit["fixed_F_process"]
        rank_deltas.append(_rank_value(a_margin) - _rank_value(f_margin))
        margin_deltas.append(a_margin - f_margin)
        transition[f"A{_rank_value(a_margin):g}_F{_rank_value(f_margin):g}"] += 1
    return {
        "question_pairs": len(grouped), "valid_em_hit_nonhit_pairs": len(selected),
        "both_valid_pairs": sum(all(row["valid"] for row in pair) for pair in grouped.values()),
        "both_EM_hit_pairs": sum(all(row["em"] == 1 for row in pair) for pair in grouped.values()),
        "both_EM_nonhit_pairs": sum(all(row["em"] == 0 for row in pair) for pair in grouped.values()),
        "credit_eligible_mixed_pairs_identical_four_features": sum(
            all(row["credit_eligible"] for row in pair) and pair[0]["feature_vector"] == pair[1]["feature_vector"]
            for pair in selected),
        "rankings": rankings,
        "A_minus_F": {
            "paired_ranking_delta": distribution(rank_deltas), "paired_margin_delta": distribution(margin_deltas),
            "improved_pairs": sum(value > 0 for value in rank_deltas),
            "unchanged_pairs": sum(value == 0 for value in rank_deltas),
            "worse_pairs": sum(value < 0 for value in rank_deltas),
            "rank_transitions": dict(transition),
            "independent_confirmation": False, "ppo_gain_claim": False,
        },
    }


def _summary(rows):
    valid_credit = [row for row in rows if row["valid"] and row["credit_eligible"]]
    masked_valid = [row for row in rows if row["valid"] and not row["credit_eligible"]]
    steps = sum(row["text_step_count"] for row in rows)
    clipped = sum(row["text_clipped_step_count"] for row in rows)
    return {
        "n_candidates": len(rows), "valid_candidates": sum(row["valid"] for row in rows),
        "answer_em": float(np.mean([row["em"] for row in rows])) if rows else None,
        "answer_f1": float(np.mean([row["f1"] for row in rows])) if rows else None,
        "format_gated_em": float(np.mean([row["em"] if row["valid"] else 0 for row in rows])) if rows else None,
        "format_gated_f1": float(np.mean([row["f1"] if row["valid"] else 0 for row in rows])) if rows else None,
        "alpha_A_valid_credit_eligible": distribution([row["alpha_A"] for row in valid_credit]),
        "text_step_clipping": {"steps": steps, "clipped_steps": clipped, "fraction": clipped / steps if steps else None},
        "masked_valid_candidates": len(masked_valid),
        "masked_graph_components_exactly_zero": all(row["learned_A_graph_component"] == row["fixed_F_graph_component"] == 0 for row in masked_valid),
        "source_status_counts": dict(Counter(row["source_credit_status"] for row in rows)),
    }


def audit(calibration_dir: Path, output_dir: Path, experiment_id: str):
    calibration_dir = calibration_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        manifest_path = calibration_dir / "manifest.json"
        manifest_identity = fitlib._identity(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != creditlib.MANIFEST_SCHEMA or (calibration_dir / "FAILED_CALIBRATION.json").exists():
            raise ValueError("source-credit calibration is missing, failed, or wrong schema")
        for name, bound in manifest["outputs"].items():
            path = calibration_dir / name
            if fitlib._identity(path)["sha256"] != bound["sha256"]:
                raise ValueError(f"calibration output hash mismatch: {name}")
        parent_view = calibration_dir / "parent_view/manifest.json"
        expected_view = manifest["parent_validation"]["validated_parent_view_manifest"]
        if fitlib._identity(parent_view)["sha256"] != expected_view["sha256"]:
            raise ValueError("validated parent view hash mismatch")
        parent = json.loads(parent_view.read_text())
        label_path = fitlib._bound_path(parent["source_bindings"]["silver"], parent_view.parent)
        labels = banklib.read_rows(label_path)
        index = {banklib.key(row): row for row in labels}
        if len(index) != len(labels):
            raise ValueError("duplicate frozen outcome source identities")
        gate = SourceCreditGateV1.load(calibration_dir / "gate.json", allow_unvalidated=True)
        candidates = banklib.read_rows(calibration_dir / "candidates.credit_masked.jsonl")
        assignments = banklib.read_rows(calibration_dir / "assignments.jsonl")
        split_index = {row["candidate_id"]: row["split"] for row in assignments}
        candidate_ids = {row["candidate_id"] for row in candidates}
        if (len(split_index) != len(assignments) or len(candidate_ids) != len(candidates)
                or candidate_ids != set(split_index)):
            raise ValueError("candidate/assignment population mismatch")
        diagnostics = []
        for row in candidates:
            if row.get("schema_version") != creditlib.ROW_SCHEMA:
                raise ValueError("audit requires explicit masked candidate row schema")
            key = banklib.key(row)
            label = index.get(key)
            if label is None or banklib.row_identity(row) != banklib.row_identity(label) or label["metadata"].get("source_split") != "train":
                raise ValueError("train outcome source identity mismatch")
            # Compute process terms before introducing outcome labels.
            terms = process_components(row, gate)
            surfaces = _canonical_gold_surfaces(label["metadata"]["gold_answer"], label["metadata"].get("gold_answer_aliases"))
            if not surfaces:
                raise ValueError("missing frozen train outcome label")
            answer = extract_kg_proweight_answer(extract_kg_proweight_answer(row["generation"]))
            diagnostics.append({
                "schema_version": ROW_SCHEMA, "candidate_id": row["candidate_id"], "question_key": key,
                "dataset": row["dataset"], "question_type": label["metadata"].get("question_type", "unknown"),
                "family_sha256": row["family_sha256"], "family_split": split_index[row["candidate_id"]],
                "valid": row["trajectory_valid"], "credit_eligible": bool(row["features"]["m_graph"]),
                "source_credit_status": row["source_credit"]["status"],
                "feature_vector": [row["features"]["values"][name] for name in FEATURE_NAMES],
                "em": max(canonical_exact_match(answer, surface) for surface in surfaces),
                "f1": max(canonical_token_f1(answer, surface) for surface in surfaces),
                "generation_sha256": canonical_sha256(row["generation"]),
                "calibration_input_eligible": False, "gold_used_only_for_independent_diagnostic": True, **terms,
            })
        groupby = lambda field: {str(value): _summary([row for row in diagnostics if row[field] == value])
                                  for value in sorted({row[field] for row in diagnostics})}
        paired_by_split = {split: paired_summary([row for row in diagnostics if row["family_split"] == split])
                           for split in ("train", "calibration", "confirmation")}
        report = {
            "schema_version": SCHEMA, "experiment_id": experiment_id,
            "status": "POST_FIT_TRAIN_ONLY_PROCESS_UTILITY_DIAGNOSTIC_NOT_PPO_CLEARANCE",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "overall": _summary(diagnostics),
            "by_dataset": groupby("dataset"), "by_question_type": groupby("question_type"),
            "by_family_split": groupby("family_split"), "by_credit_eligibility": groupby("credit_eligible"),
            "paired_all": paired_summary(diagnostics),
            "paired_credit_eligible": paired_summary([row for row in diagnostics if row["credit_eligible"]]),
            "paired_by_family_split": paired_by_split,
            "formula": {"TN": "mean(clip((raw_text_step-text_center)/text_scale,-1,1))",
                "GN": "clip((raw_graph-graph_center)/graph_scale,-1,1)",
                "T": "0.3*TN", "A": "0.3*(1-alpha_A)*TN+0.2*alpha_A*GN",
                "F": "0.3*(1-alpha_F)*TN+0.2*alpha_F*GN",
                "alpha_A": "new_gate.predict(masked_features)", "alpha_F": "mask_m_graph*fixed_alpha",
                "invalid": "excluded from process ranking", "outcome_in_process_comparison": False},
            "gate_training_clearance": gate.artifact["training_clearance"],
            "source_credit_clearance": gate.artifact["source_credit_clearance"],
            "source_integrity_clearance": gate.artifact["source_integrity_clearance"],
            "source_credit_scope": gate.artifact["source_credit_scope"],
            "normalization": gate.normalization,
            "calibration_input_eligible": False, "ppo_launch_clearance": False,
            "policy_optimizer_updates": 0, "gold_used_after_fit_only": True,
            "scientific_boundary": BOUNDARY,
            "bindings": {"calibration_manifest": manifest_identity,
                         "gate": fitlib._identity(calibration_dir / "gate.json"),
                         "masked_candidates": fitlib._identity(calibration_dir / "candidates.credit_masked.jsonl"),
                         "outcome_source": fitlib._identity(label_path), "audit_code": fitlib._identity(Path(__file__))},
        }
        if fitlib._identity(manifest_path) != manifest_identity:
            raise ValueError("calibration manifest changed during audit")
        creditlib.write_rows(output_dir / "candidate_diagnostics.jsonl", diagnostics)
        creditlib.write_json(output_dir / "report.json", report)
        creditlib.write_json(output_dir / "manifest.json", {"schema_version": SCHEMA,
            "experiment_id": experiment_id, "status": report["status"], "calibration_input_eligible": False,
            "ppo_launch_clearance": False, "bindings": report["bindings"],
            "outputs": {name: fitlib._identity(output_dir / name) for name in ("report.json", "candidate_diagnostics.jsonl")}})
        return report
    except BaseException as exc:
        creditlib.write_json(output_dir / "FAILED_AUDIT.json", {"experiment_id": experiment_id,
            "error_type": type(exc).__name__, "error": str(exc), "partial_outputs_retained": True})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    result = audit(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("status", "overall", "paired_credit_eligible")}, indent=2))
