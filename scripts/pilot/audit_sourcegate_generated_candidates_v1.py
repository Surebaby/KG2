"""Read-only CPU diagnostics of frozen SFT candidates; never a gate fit input.

Reuses existing answer metrics, format validation, graph scorer and family split.
Train outcome labels enter only this separate diagnostic, never source scoring
or calibration targets. No baseline/evaluation protocol or reward is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from scripts.prepare import source_quality_candidate_bank_v1 as banklib
from kgproweight.data.parsers import extract_final_answer
from kgproweight.eval.pred_processing import extract_kg_proweight_answer
from kgproweight.reward.proofkg_process import canonical_answer_normalize, canonical_exact_match, canonical_token_f1
from kgproweight.reward.source_quality_gate_v1 import assign_family_splits, FEATURE_NAMES
from kgproweight.training.reward_function import _canonical_gold_surfaces


def distribution(values):
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"n": 0}
    return {"n": len(array), "mean": float(array.mean()), "std": float(array.std()),
            **{name: float(np.quantile(array, q)) for name, q in
               (("min", 0), ("p50", .5), ("p90", .9), ("p95", .95), ("max", 1))}}


def summarize(rows):
    return {"n": len(rows), "questions": len({r["question_key"] for r in rows}),
            "families": len({r["family_sha256"] for r in rows}),
            "valid": sum(r["valid"] for r in rows), "length_capped": sum(r["length_capped"] for r in rows),
            "answer_em": float(np.mean([r["em"] for r in rows])) if rows else None,
            "answer_f1": float(np.mean([r["f1"] for r in rows])) if rows else None,
            "ppo_format_gated_em": float(np.mean([r["ppo_em"] for r in rows])) if rows else None,
            "ppo_format_gated_f1": float(np.mean([r["ppo_f1"] for r in rows])) if rows else None,
            "response_tokens": distribution([r["response_tokens"] for r in rows]),
            "step_counts": dict(Counter(r["steps"] for r in rows)),
            "violations": dict(Counter(v for r in rows for v in r["violations"])),
            "eligible_valid_graph_score": distribution([r["raw_graph"] for r in rows if r["valid"] and r["m_graph"]])}


def pair_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["question_key"]].append(row)
    pairs = list(groups.values())
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("diagnostic requires exactly K2 for each question")
    mixed = [pair for pair in pairs if pair[0]["em"] != pair[1]["em"]]
    rankable = [pair for pair in mixed if all(r["valid"] and r["m_graph"] for r in pair)]
    ranking = {}
    for field in ("raw_graph", "structural_component", "answer_component"):
        counts = Counter()
        for pair in rankable:
            good, bad = sorted(pair, key=lambda r: r["em"], reverse=True)
            delta = good[field] - bad[field]
            counts["win" if delta > 1e-10 else "loss" if delta < -1e-10 else "tie"] += 1
        ranking[field] = {k: counts[k] for k in ("win", "tie", "loss")}
        ranking[field]["tie_adjusted_accuracy"] = (counts["win"] + .5 * counts["tie"]) / len(rankable) if rankable else None
    return {"questions": len(pairs), "both_correct": sum(all(r["em"] == 1 for r in p) for p in pairs),
            "both_wrong": sum(all(r["em"] == 0 for r in p) for p in pairs), "mixed_correct_wrong": len(mixed),
            "both_valid": sum(all(r["valid"] for r in p) for p in pairs),
            "identical_trace": sum(p[0]["generation_sha256"] == p[1]["generation_sha256"] for p in pairs),
            "same_normalized_answer": sum(canonical_answer_normalize(p[0]["answer"]) == canonical_answer_normalize(p[1]["answer"]) for p in pairs),
            "oracle_answer_em_at_2": float(np.mean([max(r["em"] for r in p) for p in pairs])),
            "eligible_both_valid_mixed_pairs": len(rankable),
            "mixed_pairs_identical_gate_features": sum(p[0]["features"] == p[1]["features"] for p in rankable),
            "graph_ranking_on_eligible_both_valid_mixed_pairs": ranking}


def audit(bank_dir, generation_dir, output_dir, experiment_id):
    if output_dir.exists():
        raise ValueError("refusing to overwrite an audit")
    bank = banklib.load_release(bank_dir, banklib.PREPARE_VERSION)
    generated = banklib.load_release(generation_dir, banklib.GENERATION_VERSION)
    banklib.validate_code(bank, banklib.ROOT)
    inputs = banklib.validate_inputs(bank_dir, bank)
    predictions = banklib.read_rows(generation_dir / "generations.jsonl")
    bank_sha = banklib.file_sha(bank_dir / "manifest.json")
    if generated["bank_manifest_sha256"] != bank_sha or len(predictions) != len(inputs) * 2 or len(predictions) != generated["n_candidates"]:
        raise ValueError("candidate release identity/count mismatch")
    label_path = banklib.resolve(bank["source_bindings"]["silver"], bank_dir)
    labels = banklib.read_rows(label_path)
    label_index = {banklib.key(r): r for r in labels}
    if len(label_index) != len(labels):
        raise ValueError("duplicate source label identities")
    overlaps = banklib.isolation(inputs, banklib.read_rows(banklib.resolve(bank["source_bindings"]["protected_ledger"], bank_dir)))
    splits = assign_family_splits([r["family_sha256"] for r in inputs], seed=42)
    diagnostics = []
    for i, pred in enumerate(predictions):
        row, index = inputs[i // 2], i % 2
        expected = {"candidate_id": f"{row['question_key']}::k{index}", "dataset": row["dataset"], "qid": row["qid"],
                    "candidate_index": index, "seed": banklib.candidate_seed(bank["seed"], row["question_key"], index),
                    "input_sha256": row["input_sha256"], "bank_manifest_sha256": bank_sha,
                    "generation_contract_sha256": banklib.digest(bank["generation"]),
                    "policy_sha256": bank["source_bindings"]["policy"]["sha256"],
                    "base_model_identity_sha256": banklib.digest(bank["base_model"])}
        if any(pred.get(k) != v for k, v in expected.items()):
            raise ValueError("prediction identity/seed/model/order mismatch")
        ids = pred["response_token_ids"]
        eos = bank["generation"]["eos_token_ids"]
        capped = len(ids) >= 384 and not any(t in eos for t in ids)
        if (len(ids) != pred["n_response_tokens"] or len(ids) > 384 or
                capped != pred["reached_max_new_tokens"] or pred["effective_eos_token_ids"] != eos or
                ids != pred["raw_response_token_ids"] or any(t in eos for t in ids[:-1])):
            raise ValueError("response token/EOS contract mismatch")
        spec = SimpleNamespace(**row["spec"])
        validity = banklib.validate_source_gate_trajectory_v1(spec, pred["generation"])
        record = row["source_quality_record"]
        plan = record.get("query_plan") or {}
        proof = banklib.score_proofkg_v2_3(question=row["question"], generation=pred["generation"], kg_triples=row["kg_subgraph"],
                    execution_trace=banklib.build_execution_trace_v2_3(plan, record.get("execution") or {}), planned_hops=len(plan.get("hops") or []))
        features = banklib.compute_gate_features(spec, validity["steps"], proof)
        # Gold is joined only after gold-free validation, feature extraction and graph scoring.
        label = label_index[row["question_key"]]
        if banklib.row_identity(label) != banklib.row_identity(row) or label["metadata"]["source_split"] != "train":
            raise ValueError("train-only diagnostic label identity mismatch")
        aliases = _canonical_gold_surfaces(label["metadata"]["gold_answer"], label["metadata"].get("gold_answer_aliases"))
        if not aliases:
            raise ValueError("missing frozen outcome label")
        answer = extract_kg_proweight_answer(extract_kg_proweight_answer(pred["generation"]))
        ppo_answer = (extract_final_answer(pred["generation"]) or "").split("\n", 1)[0].strip()
        score = lambda metric, text: max(metric(text, a) for a in aliases)
        telemetry = proof.get("telemetry", {})
        diagnostics.append({"candidate_id": pred["candidate_id"], "question_key": row["question_key"],
            "dataset": row["dataset"], "question_type": label["metadata"].get("question_type", "unknown"),
            "family_sha256": row["family_sha256"], "family_split": splits[row["family_sha256"]], "m_graph": features["m_graph"],
            "valid": validity["valid"], "violations": validity["violations"], "steps": validity["all_step_count"],
            "required_steps": validity["required_steps"], "length_capped": capped, "response_tokens": len(ids),
            "answer": answer, "em": score(canonical_exact_match, answer), "f1": score(canonical_token_f1, answer),
            "ppo_em": score(canonical_exact_match, ppo_answer) if validity["valid"] else 0.0,
            "ppo_f1": score(canonical_token_f1, ppo_answer) if validity["valid"] else 0.0,
            "raw_graph": proof["score"], "structural_component": telemetry.get("structural_component", 0.0),
            "answer_component": telemetry.get("answer_component", 0.0), "proof_valid": proof["trajectory_valid"],
            "proof_components": proof["components"], "features": features["values"],
            "generation_sha256": banklib.digest(pred["generation"]),
            "diagnostic_only_gold_used_after_source_scoring": True, "calibration_input_eligible": False})
    group_by = lambda field: {value: summarize([r for r in diagnostics if r[field] == value]) for value in sorted({r[field] for r in diagnostics})}
    eligible = [r for r in diagnostics if r["valid"] and r["m_graph"]]
    vectors = [tuple(r["features"][f] for f in FEATURE_NAMES) for r in eligible]
    report = {"schema_version": "sourcegate-generated-candidates-diagnostic-v1", "experiment_id": experiment_id,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "GENERATION_AUDITED_TEXT_SCORING_AND_ALPHA_PENDING",
              "integrity": {"release_hashes": "PASS", "input_hashes_gold_free_order": "PASS", "candidate_identity_seed_model_contract": "PASS",
                            "response_length_eos": "PASS", "protected_overlap": overlaps},
              "overall": summarize(diagnostics), "by_dataset": group_by("dataset"), "by_graph_eligibility": group_by("m_graph"),
              "by_question_type": group_by("question_type"), "by_family_split": group_by("family_split"),
              "paired": pair_summary(diagnostics), "paired_graph_eligible": pair_summary([r for r in diagnostics if r["m_graph"]]),
              "eligible_valid_feature_distributions": {f: distribution([r["features"][f] for r in eligible]) for f in FEATURE_NAMES},
              "eligible_valid_unique_feature_vectors": len(set(vectors)),
              "eligible_valid_feature_matrix_rank": int(np.linalg.matrix_rank(np.asarray(vectors) - np.asarray(vectors).mean(axis=0))),
              "eligible_valid_graph_component_distributions": {f: distribution([r[f] for r in eligible]) for f in ("raw_graph", "structural_component", "answer_component")},
              "eligible_valid_graph_max_score_wrong": sum(r["raw_graph"] >= .85 - 1e-10 and r["em"] == 0 for r in eligible),
              "training_started": False, "text_scoring_performed": False, "alpha_calibrated": False,
              "scientific_boundary": "Descriptive train-only K2 stochastic SFT candidate diagnostic, graph-heavy bank. Not canonical evaluation, not PPO performance, not independent confirmation of process reward utility. Gold is used in this separate audit only; never feed audit rows to gate calibration. Graph score includes a derived-answer consistency component; structure-only ranking is reported separately.",
              "source_bindings": {"input_manifest": banklib.identity(bank_dir / "manifest.json"), "generation_manifest": banklib.identity(generation_dir / "manifest.json"),
                                  "generations": banklib.identity(generation_dir / "generations.jsonl"), "outcome_label_source": banklib.identity(label_path),
                                  "audit_code": banklib.identity(Path(__file__)), "frozen_scoring_code": {k:v for k,v in bank["source_bindings"].items() if k.startswith("code:")}},
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=banklib.ROOT, text=True).strip()}
    output_dir.mkdir(parents=True, exist_ok=False)
    banklib.write_rows(output_dir / "candidate_diagnostics.jsonl", diagnostics)
    banklib.finish(output_dir, report, ["candidate_diagnostics.jsonl"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    report = audit(**vars(args))
    print(json.dumps({k: report[k] for k in ("status", "integrity", "overall", "by_dataset", "by_question_type", "paired", "eligible_valid_feature_distributions", "eligible_valid_unique_feature_vectors")}, ensure_ascii=False, indent=2))
