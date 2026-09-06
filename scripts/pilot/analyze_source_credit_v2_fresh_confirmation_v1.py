#!/usr/bin/env python
"""Analyze the single, sealed fresh132 confirmation; never fit or select a gate.

The complete process-only ordering is checked before any Gold JSON is decoded.
Outputs are new files, contain metrics but no Gold values, and cannot update a
parent gate or launch PPO. PASS is a predeclared investment decision, not a
significance claim or evidence of trained PPO benefit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "source-credit-v2-fresh-confirmation-analysis-v1"
VARIANTS = ("features_v2", "norm_only")
ARMS = ("A", "F", "T")
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
TYPES = ("bridge_comparison", "comparison", "compositional")
ANALYSIS_CODE_FILES = [
    "scripts/pilot/analyze_source_credit_v2_fresh_confirmation_v1.py",
    "kgproweight/eval/pred_processing.py",
    "kgproweight/reward/proofkg_process.py",
    "kgproweight/training/reward_function.py",
    "kgproweight/reward/source_credit_gate_v2.py",
    "kgproweight/reward/proofkg_process_v2_3.py",
    "scripts/prepare/score_source_credit_v2_fresh_confirmation_v1.py",
]
DECISION_RULES = {
    "version": "fresh132-five-targets-family25-three-state-v1",
    "primary_variant": "features_v2", "secondary_variant": "norm_only_diagnostic_never_selected",
    "sampled_micro_valid_min": .90, "sampled_three_dataset_macro_valid_min": .90,
    "source_pass_mixed_outcome_families_min": 25,
    "graph_valid_oracle_minus_raw_greedy_em_min": .03,
    "source_pass_A_family_macro_pairwise_min": .65,
    "graph_A_minus_raw_greedy_em_min": 0., "graph_A_minus_F_em_min": 0.,
    "point_met": "PASS; CI need not exclude the target",
    "point_missed_ci_upper_below_target": "FAIL",
    "point_missed_ci_upper_at_or_above_target": "INCONCLUSIVE",
    "insufficient_mixed_families_or_oracle": "INCONCLUSIVE; do not use utility gates to fail",
    "priority": "integrity FAIL; utility status excludes health; overall includes unchanged health targets",
    "health_scope": "90-percent observation target; smoke investment risk, never redefine as utility failure",
    "pairwise": "valid sampled correct-vs-incorrect pairs; tie=.5; mean within question then equal-family mean",
    "bootstrap": {"replicates": 20000, "seed": 42, "confidence_level": .95,
                  "unit": "family", "strata": "graph_question_type_or_ordinary_dataset",
                  "method": "stratified nonparametric percentile; paired arms use identical resamples",
                  "pairwise_population": "conditional mixed-outcome source-PASS families, stratified by graph type"},
    "F_minus_T": "report_only", "length_cap_rate": "report_only_no_5_percent_gate",
    "no_adaptive_cohort_expansion": True, "significant_A_over_F_required": False,
    "ordinary_scope": "mask/text diagnostic before PPO; not held-out from formal PPO3000",
    "engineering_probe12": "independent utility PASS permits separately derived production gate; health alone does not veto bounded wiring probe",
    "matched600": "automatic investment clearance only if overall PASS; full PPO never auto-launched",
}


def decision_rules():
    return deepcopy(DECISION_RULES)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def checked(ref):
    if not isinstance(ref, dict) or not {"path", "sha256"} <= set(ref):
        raise ValueError("complete immutable file binding required")
    path = Path(ref["path"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file() or ("bytes" in ref and path.stat().st_size != ref["bytes"]) or sha(path) != ref["sha256"]:
        raise ValueError("immutable file binding mismatch")
    return path.resolve()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(canonical(value) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(actual, expected):
    return finite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)


def verify_analysis_protocol(p):
    a = p.get("analysis") or {}
    require(a.get("decision_rules") == decision_rules(), "analysis decision rules changed")
    for name, expected in {"primary_variant": "features_v2", "secondary_variant": "norm_only",
                           "bootstrap_replicates": 20000, "bootstrap_seed": 42,
                           "confidence_level": .95,
                           "bootstrap_strata": "graph_question_type_or_ordinary_dataset",
                           "process_ranking_excludes_answer_reward": True,
                           "variants_never_selected_or_refit_using_confirmation": True,
                           "no_adaptive_resampling_or_identity_replacement": True}.items():
        require(a.get(name) == expected, f"analysis contract mismatch: {name}")
    require(set(a.get("gold_sources") or {}) == {"graph", "ordinary"}, "two bound Gold sources required")
    require(set(ANALYSIS_CODE_FILES) <= set(p.get("code_bindings") or {}), "analysis code dependency binding missing")


def verify_population(cohort, inputs, source_checks):
    require(len(cohort) == len(inputs) == 132, "complete132 cohort required")
    bykey = {r["question_key"]: r for r in cohort}
    require(len(bykey) == 132 and len({r["family_sha256"] for r in cohort}) == 132,
            "duplicate question or global family")
    require(Counter(r["dataset"] for r in cohort) == {"2wikimultihopqa": 108, "hotpotqa": 12, "musique": 12},
            "fixed dataset composition differs")
    checks = {r["question_key"]: r for r in source_checks}
    require(len(source_checks) == len(checks) == 96, "complete96 source checks required")
    graph = [r for r in cohort if r["proposal_role"] == "graph"]
    ordinary = [r for r in cohort if r["proposal_role"] == "ordinary"]
    require(len(graph) == 96 and len(ordinary) == 36 and set(checks) == {r["question_key"] for r in graph},
            "graph/ordinary/source-check join differs")
    require(Counter(r["question_type"] for r in graph) == dict.fromkeys(TYPES, 32), "fixed three graph types differ")
    require(Counter(r["dataset"] for r in ordinary) == dict.fromkeys(DATASETS, 12), "ordinary36 composition differs")
    require(Counter(r["status"] for r in source_checks) == {"PASS": 79, "UNVERIFIED": 11, "FAIL": 6},
            "fixed source PASS79/UNVERIFIED11/FAIL6 composition differs")
    require(Counter(bykey[k]["question_type"] for k, r in checks.items() if r["status"] == "PASS") == {
        "bridge_comparison": 18, "comparison": 30, "compositional": 31}, "source PASS type composition differs")
    for row in inputs:
        key = row["question_key"]
        require(key in bykey, "unregistered input identity")
        c = bykey[key]
        for name in ("dataset", "qid", "question", "question_sha256", "family_sha256", "family_version"):
            require(row[name] == c[name], "cohort/input identity mismatch")
        require(c.get("gold_access") is False and row["m_graph"] == int(c["proposal_role"] == "graph"),
                "cohort graph scope mismatch")
        if key in checks:
            check = checks[key]
            require(check["input_sha256"] == row["input_sha256"] and check.get("gold_used") is False
                    and check.get("original_m_graph") == 1 and check.get("clearance") is (check["status"] == "PASS"),
                    "source status/input binding mismatch")
    return bykey, checks


def verify_cpu_components(row, original, gates=None, rearag_tokenizer=None):
    """Replay exact format/ProofKG and token budgets, without a model forward."""
    from kgproweight.training.reward_function import (validate_source_gate_trajectory,
        source_gate_text_inputs_v1, source_gate_text_budget_v1)
    from kgproweight.reward.proofkg_process_v2_3 import score_proofkg_v2_3, build_execution_trace_v2_3
    spec = SimpleNamespace(**original["spec"])
    validity = validate_source_gate_trajectory(spec, row["generation"], format_version="v2")
    expected_format = {k: validity[k] for k in ("valid", "violations", "all_step_count", "required_steps", "contract_version")}
    require(row["trajectory_valid"] is validity["valid"] and row["format_validation"] == expected_format,
            "independent format validation disagrees")
    require(len(row["raw_text"]) == (len(validity["steps"]) if validity["valid"] else 0),
            "Text scores must cover every and only valid step")
    require(row["raw_text_step_mean"] is None if not validity["valid"] else
            close(row["raw_text_step_mean"], sum(row["raw_text"])/len(row["raw_text"])), "raw Text mean differs")
    record = original["source_quality_record"]
    plan = record.get("query_plan") or {}
    proof = score_proofkg_v2_3(question=original["question"], generation=row["generation"],
        kg_triples=original["kg_subgraph"], execution_trace=build_execution_trace_v2_3(plan, record.get("execution") or {}),
        planned_hops=len(plan.get("hops") or []))
    require(proof == row["proof_result"] and row["raw_graph"] == float(proof["score"]),
            "independent source-backed ProofKG replay differs")
    require(row["raw_graph_invalid_is_diagnostic_only"] is (not validity["valid"]), "invalid Graph diagnostic flag differs")
    if validity["valid"]:
        prompts, texts = source_gate_text_inputs_v1(spec, validity["steps"])
        require(rearag_tokenizer is not None, "ReaRAG tokenizer required for complete token-budget replay")
        budget = source_gate_text_budget_v1(SimpleNamespace(backend=SimpleNamespace(tokenizer=rearag_tokenizer, max_length=4096)), prompts, texts)
        require(row["text_token_budget"] == budget, "independent ReaRAG token budget differs")
    else:
        require(row["text_token_budget"] is None, "invalid trajectory cannot have Text budget")
    return validity


def policy_tokenizer_with_frozen_pad(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def verify_process_rows(processes, generations, inputs, cohort, source_checks, *, protocol_sha256, gates=None, rearag_tokenizer=None):
    """Independent validation of membership, process algebra and ranking inputs."""
    require(len(processes) == len(generations) == 660, "complete660 process/generation rows required")
    source = {r["question_key"]: r for r in inputs}
    predictions = {r["candidate_id"]: r for r in generations}
    require(len(predictions) == 660, "duplicate generated candidate")
    seen, grouped = set(), defaultdict(list)
    for row in processes:
        cid, key, index = row["candidate_id"], row["question_key"], row["candidate_index"]
        require(cid not in seen and key in source and type(index) is int and 0 <= index <= 4
                and cid == f"{key}::k{index}" and cid in predictions, "process membership/index mismatch")
        seen.add(cid)
        require(row.get("schema_version") == "source-credit-v2-fresh-process-row-v1"
                and row.get("protocol_sha256") == protocol_sha256
                and row.get("process_row_sha256") == digest({k: v for k, v in row.items() if k != "process_row_sha256"}),
                "process payload/protocol hash mismatch")
        pred, original = predictions[cid], source[key]
        require(row["generation_sha256"] == digest(pred), "process bound generation differs")
        for name in ("question_key", "dataset", "qid", "question", "question_sha256", "family_sha256", "family_version", "input_sha256", "source_record_sha256"):
            require(row.get(name) == original.get(name), "process/input identity binding mismatch")
        for name in ("generation", "seed", "n_response_tokens", "reached_max_new_tokens", "generation_kind", "candidate_index"):
            require(row[name] == pred[name], "process/generation content mismatch")
        require(row["generation_kind"] == ("sampled" if index < 4 else "greedy"), "greedy/sample kind mismatch")
        valid = row.get("trajectory_valid")
        require(type(valid) is bool and row.get("rank_eligible") is valid
                and row.get("gold_access") is False and row.get("outcome_in_process") is False
                and row.get("model_updates") == 0, "invalid process eligibility/Gold/update flags")
        require(type(row["n_response_tokens"]) is int and 0 < row["n_response_tokens"] <= 384
                and type(row["reached_max_new_tokens"]) is bool, "invalid token health metadata")
        require(finite(row["raw_graph"]) and 0 <= row["raw_graph"] <= .85 + 1e-12
                and isinstance(row["raw_text"], list)
                and all(finite(x) and -1 <= x <= 1 for x in row["raw_text"]), "nonfinite/out-of-range component")
        require(bool(row["raw_text"]) is valid, "invalid trajectory Text must be empty")
        validity = None
        if gates is not None:
            validity = verify_cpu_components(row, original, gates, rearag_tokenizer)
        for variant in VARIANTS:
            terms = row["variants"][variant]
            features = row["features"][variant]["masked"]
            allowed = int(key in source_checks and source_checks[key]["status"] == "PASS")
            require(features["m_graph"] == allowed, "source credit population differs")
            if gates is not None:
                gate = gates[variant]
                raw_features = gate.compute_features(SimpleNamespace(**original["spec"]), validity["steps"], row["proof_result"] if valid else {})
                masked = gate.mask_features(SimpleNamespace(**original["spec"]), raw_features)
                require(row["features"][variant] == {"original": raw_features, "masked": masked}, "frozen features/mask recomputation differs")
            for arm in ARMS:
                item = terms[arm]
                require(item["rank_eligible"] is valid and all(finite(item[k]) for k in (
                    "alpha_effective", "text_component", "graph_component", "process")), "nonfinite process term")
                alpha = item["alpha_effective"]
                require(0 <= alpha <= 1 and (allowed or alpha == 0) and (arm != "T" or alpha == 0), "source mask/T alpha violation")
                if not valid:
                    require(all(item[k] == 0 for k in ("alpha_effective", "text_component", "graph_component", "process"))
                            and item["text_step_components"] == [], "invalid candidate cannot earn/rank process reward")
                    continue
                tn, gn = item["text_normalized_mean"], item["graph_normalized"]
                require(finite(tn) and finite(gn) and -1 <= tn <= 1 and -1 <= gn <= 1,
                        "invalid normalized process scale")
                require(close(item["text_component"], .3 * (1-alpha) * tn)
                        and close(item["graph_component"], .2 * alpha * gn)
                        and close(item["process"], item["text_component"] + item["graph_component"]),
                        "independent process-only algebra mismatch")
                require(len(item["text_normalized_steps"]) == len(row["raw_text"])
                        and close(tn, math.fsum(item["text_normalized_steps"]) / len(row["raw_text"])), "step-mean mismatch")
                expected_steps = [.3 * (1-alpha) * x / len(row["raw_text"]) for x in item["text_normalized_steps"]]
                require(len(item["text_step_components"]) == len(expected_steps)
                        and all(close(x, y) for x, y in zip(item["text_step_components"], expected_steps)), "step allocation mismatch")
                if gates is not None:
                    norm = gates[variant].normalization
                    z = [(x - norm["text_center"]) / norm["text_scale"] for x in row["raw_text"]]
                    expected = [x / (1 + abs(x)) for x in z]
                    gz = (row["raw_graph"] - norm["graph_center"]) / norm["graph_scale"] if original["m_graph"] else 0.
                    ea = (gates[variant].predict(features) if arm == "A" else norm["fixed_alpha"] if arm == "F" else 0.) * allowed
                    require(all(close(x, y) for x, y in zip(item["text_normalized_steps"], expected))
                            and close(gn, max(-1., min(1., gz))) and close(item["graph_normalized_unclipped"], gz)
                            and close(alpha, ea), "frozen normalization/alpha recomputation differs")
                    exact_steps = [.3 * (1 - ea) * value / len(expected) for value in expected]
                    exact_text = sum(exact_steps)
                    exact_graph = .2 * ea * max(-1., min(1., gz))
                    require(item["process"] == exact_text + exact_graph,
                            "exact process/tie value differs from frozen float reduction")
        grouped[key].append(row)
    require(set(grouped) == set(source), "process question population differs")
    for rows in grouped.values():
        require(len(rows) == 5 and {r["candidate_index"] for r in rows} == set(range(5)), "K4+greedy incomplete")
    return grouped


def verify_rankings(rankings, grouped):
    """Reconstruct order from sealed process scores without calling scorer ranker."""
    require(len(rankings) == len(grouped) == 132 and len({r["question_key"] for r in rankings}) == 132,
            "complete132 unique rankings required")
    result = {}
    for rank in rankings:
        key = rank["question_key"]
        require(key in grouped and rank.get("gold_access") is False and rank.get("greedy_in_ranking") is False
                and rank.get("rank_tie_break") == "candidate_index_ascending", "rank boundary violation")
        rows = sorted(grouped[key], key=lambda r: r["candidate_index"])
        sampled, greedy = rows[:4], rows[4]
        valid = [r for r in sampled if r["trajectory_valid"]]
        require(rank["sampled_candidate_ids"] == [r["candidate_id"] for r in sampled]
                and rank["greedy_candidate_id"] == greedy["candidate_id"]
                and rank["invalid_sampled_candidate_ids"] == [r["candidate_id"] for r in sampled if not r["trajectory_valid"]]
                and rank["all_sampled_invalid"] is (not valid), "rank sampled/invalid/greedy membership differs")
        for variant in VARIANTS:
            for arm in ARMS:
                ordered = sorted(valid, key=lambda r: (-r["variants"][variant][arm]["process"], r["candidate_index"]))
                expected = {"selected_candidate_id": ordered[0]["candidate_id"] if ordered else None,
                            "ordered_eligible_candidate_ids": [r["candidate_id"] for r in ordered],
                            "ordered_process_scores": [r["variants"][variant][arm]["process"] for r in ordered]}
                require(rank["rankings"][variant][arm] == expected, "sealed rank differs from independent process-only order")
        result[key] = rank
    return result


def verify_before_gold(protocol_path, scoring):
    """Return a fully checked answer-free context; this function never opens Gold."""
    from scripts.prepare import generate_source_credit_v2_fresh_confirmation_v1 as producer
    from kgproweight.reward.source_credit_gate_v2 import SourceCreditGateV2
    protocol_path, scoring = Path(protocol_path).resolve(), Path(scoring).resolve()
    context = producer.verify_protocol(protocol_path, verify_models=False)
    p, phash, inputs = context["protocol"], context["protocol_sha256"], context["inputs"]
    verify_analysis_protocol(p)
    from scripts.prepare.score_source_credit_v2_fresh_confirmation_v1 import validate_scoring_protocol
    validate_scoring_protocol(p)
    manifest_path = scoring / "manifest.json"
    release = json.loads(manifest_path.read_text())
    require(release.get("schema_version") == "source-credit-v2-fresh-process-scoring-v1"
            and release.get("status") == "COMPLETE_GOLD_FREE_SCORED_NOT_ANALYZED"
            and release.get("protocol_sha256") == phash and release.get("gold_access") is False
            and release.get("gate_fitting") is False and release.get("model_updates") == 0
            and release.get("ppo_launch_clearance") is False, "scoring release is not a complete immutable Gold-free seal")
    names = {"processes.jsonl", "rankings.jsonl", "report.json", "prepared.json"}
    require(set(release["outputs"]) == names, "incomplete scoring outputs")
    paths = {name: checked(ref) for name, ref in release["outputs"].items()}
    require(all(path == scoring / name for name, path in paths.items()), "scoring seal points outside its release")
    report, prepared = [json.loads(paths[n].read_text()) for n in ("report.json", "prepared.json")]
    require(report["protocol_sha256"] == phash and report["n_questions"] == 132 and report["n_candidates"] == 660
            and report["gold_access"] is False and report["status"] == release["status"]
            and checked(report["prepared"]) == paths["prepared.json"], "scoring report binding mismatch")
    require(checked(prepared["protocol"]) == protocol_path and prepared["scoring_config_sha256"] == digest(p["scoring"])
            and prepared["n_inputs"] == 132 and prepared["n_candidates"] == 660 and prepared["gold_access"] is False,
            "scoring prepared protocol/config differs")
    require(checked(prepared["scoring_code"]) == (ROOT / "scripts/prepare/score_source_credit_v2_fresh_confirmation_v1.py"),
            "scoring code binding differs")
    gm_path = checked(prepared["generation_manifest"])
    gm = json.loads(gm_path.read_text())
    require(gm.get("schema_version") == "source-credit-v2-fresh-confirmation-generations-v1"
            and gm.get("status") == "COMPLETE_GENERATED_NOT_SCORED" and gm.get("protocol_sha256") == phash,
            "generation release incomplete or another protocol")
    gen_path = checked(prepared["generations"])
    require(checked(gm["outputs"]["generations.jsonl"]) == gen_path, "generation seal/prepared mismatch")
    generations = read_rows(gen_path)
    from transformers import AutoTokenizer
    tokenizer = policy_tokenizer_with_frozen_pad(AutoTokenizer.from_pretrained(context["policy_path"], local_files_only=True))
    producer.verify_generation_rows(generations, context, tokenizer=tokenizer)
    cohort_path, checks_path = checked(p["bindings"]["cohort"]), checked(p["bindings"]["source_checks"])
    source_manifest = json.loads(context["source_manifest_path"].read_text())
    require(checked(context["input_manifest"]["source_bindings"]["proposal_cohort"]) == cohort_path, "original cohort stratum authority mismatch")
    require(checked(source_manifest["outputs"]["question_checks.jsonl"]) == checks_path, "source checks authority mismatch")
    cohort, checks = verify_population(read_rows(cohort_path), inputs, read_rows(checks_path))
    gates = {v: SourceCreditGateV2.load(checked(p["scoring"]["gates"][v]), allow_unvalidated=True) for v in VARIANTS}
    require(all(g.artifact.get("training_clearance") is False and g.artifact.get("independent_confirmation_clearance") is False
                for g in gates.values()), "fresh confirmation gate already cleared")
    require(gates[VARIANTS[0]].mask.payload_sha256 == gates[VARIANTS[1]].mask.payload_sha256, "variant masks differ")
    for variant, version in (("norm_only", "source-quality-trajectory-features-v1"), ("features_v2", "source-quality-trajectory-features-v2")):
        require(gates[variant].artifact["feature_version"] == version, "gate feature variant differs")
    model_info = p["scoring"]["rearag_model"]
    require(model_info == context["input_manifest"]["models"]["rearag_model"], "ReaRAG model identity differs")
    rearag_path = Path(model_info["path"])
    if not rearag_path.is_absolute(): rearag_path = ROOT / rearag_path
    tokenizer_bindings = {}
    for name, info in model_info["files"].items():
        if not name.endswith((".bin", ".safetensors")):
            path = checked({**info, "path": str(rearag_path / name)})
            tokenizer_bindings["rearag_tokenizer:" + name] = binding(path)
    require({f.name for f in rearag_path.glob("*.py")} <= set(model_info["files"]), "unbound ReaRAG tokenizer Python source")
    rearag_tokenizer = AutoTokenizer.from_pretrained(rearag_path, local_files_only=True, trust_remote_code=True)
    processes, ranks = read_rows(paths["processes.jsonl"]), read_rows(paths["rankings.jsonl"])
    grouped = verify_process_rows(processes, generations, inputs, cohort, checks, protocol_sha256=phash, gates=gates, rearag_tokenizer=rearag_tokenizer)
    rankings = verify_rankings(ranks, grouped)
    require(report["valid_candidates"] == sum(r["trajectory_valid"] for r in processes)
            and report["scored_text_steps"] == sum(len(r["raw_text"]) for r in processes), "scoring report count mismatch")
    # Capture immutability before the label boundary. A second check after analysis
    # prevents changing scores/protocol while labels are being read.
    frozen_files = {"protocol": binding(protocol_path), "scoring_manifest": binding(manifest_path),
                    "generation_manifest": binding(gm_path), "generations": binding(gen_path),
                    "cohort": binding(cohort_path), "source_checks": binding(checks_path), **release["outputs"], **tokenizer_bindings}
    for name, ref in p["code_bindings"].items():
        frozen_files["code:" + name] = binding(checked(ref))
    for name, ref in p["bindings"].items():
        frozen_files["protocol_binding:" + name] = binding(checked(ref))
    for name, ref in p["scoring"]["gates"].items():
        frozen_files["gate:" + name] = binding(checked(ref))
    for name, ref in source_manifest["outputs"].items():
        frozen_files["source_output:" + name] = binding(checked(ref))
    # ReaRAG forward is not repeated. The exact cached call inputs/values are
    # independently reconstructed and sealed alongside each complete process row.
    from kgproweight.training.reward_function import source_gate_text_inputs_v1, validate_source_gate_trajectory
    input_by_key = {r["question_key"]: r for r in inputs}
    prediction_order = {r["candidate_id"]: i for i, r in enumerate(generations)}
    for row in processes:
        candidate_path = scoring / "candidate_rows" / f"{prediction_order[row['candidate_id']]:04d}.json"
        require(json.loads(candidate_path.read_text()) == row, "candidate checkpoint differs from process seal")
        frozen_files["candidate:" + row["candidate_id"]] = binding(candidate_path)
        if not row["trajectory_valid"]:
            continue
        original = input_by_key[row["question_key"]]
        spec = SimpleNamespace(**original["spec"])
        valid = validate_source_gate_trajectory(spec, row["generation"], format_version="v2")
        prompts, texts = source_gate_text_inputs_v1(spec, valid["steps"])
        for position, (prompt, text, raw_score) in enumerate(zip(prompts, texts, row["raw_text"])):
            path = scoring / "raw_steps" / (digest([row["candidate_id"], position]) + ".json")
            expected = {"candidate_id": row["candidate_id"], "generation_sha256": row["generation_sha256"],
                "scoring_binding_sha256": digest(prepared), "step_position": position,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "raw_score": raw_score}
            require(json.loads(path.read_text()) == expected, "raw ReaRAG call cache differs from exact step/input/value")
            frozen_files[f"raw_step:{row['candidate_id']}:{position}"] = binding(path)
    return {"protocol": p, "protocol_sha256": phash, "cohort": cohort, "checks": checks,
            "grouped": grouped, "rankings": rankings, "frozen_files": frozen_files,
            "seal": {"schema_version": SCHEMA, "status": "PASS_BEFORE_GOLD", "gold_values_opened": False,
                     "checked_candidates": 660, "checked_question_rankings": 132, "checked_variant_arm_rankings": 792,
                     "frozen_files": frozen_files, "no_gold_used_for_ordering": True}}


def load_gold_after_seal(context):
    """The sole label-value reader: exact identity/question join, no emitted Gold."""
    from kgproweight.training.reward_function import _canonical_gold_surfaces
    require(context.get("seal", {}).get("status") == "PASS_BEFORE_GOLD", "Gold access requires complete rank seal")
    for ref in context["frozen_files"].values():
        checked(ref)
    result = {}
    for role in ("graph", "ordinary"):
        expected = {k: r for k, r in context["cohort"].items() if r["proposal_role"] == role}
        path = checked(context["protocol"]["analysis"]["gold_sources"][role])
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = f"{row['dataset']}::{row['qid']}"
                if key not in expected:
                    continue
                require(key not in result, "duplicate Gold identity; fail closed")
                want = expected[key]
                require(row["question"] == want["question"]
                        and hashlib.sha256(row["question"].strip().encode()).hexdigest() == want["question_sha256"],
                        "Gold exact question/hash join differs")
                meta = row.get("metadata") or {}
                require(role != "ordinary" or meta.get("source_split") == "train", "ordinary Gold source must be frozen train")
                surfaces = _canonical_gold_surfaces(meta.get("gold_answer"), meta.get("gold_answer_aliases"))
                require(bool(surfaces), "missing canonical frozen label surfaces")
                result[key] = surfaces
        require(set(expected) <= set(result), "Gold source missing requested identities")
    require(set(result) == set(context["cohort"]), "Gold join incomplete")
    return result


def answer_scores(generation, surfaces):
    from kgproweight.eval.pred_processing import extract_kg_proweight_answer
    from kgproweight.reward.proofkg_process import canonical_exact_match, canonical_token_f1
    answer = extract_kg_proweight_answer(extract_kg_proweight_answer(generation))
    if not answer.strip():
        return {"em": 0., "f1": 0.}
    require(bool(surfaces), "nonempty Gold surfaces required")
    return {"em": float(max(canonical_exact_match(answer, s) for s in surfaces)),
            "f1": float(max(canonical_token_f1(answer, s) for s in surfaces))}


def pairwise(valid_rows, scores, variant, arm):
    hits = [r for r in valid_rows if scores[r["candidate_id"]]["em"] == 1]
    misses = [r for r in valid_rows if scores[r["candidate_id"]]["em"] == 0]
    win = tie = 0
    for hit in hits:
        for miss in misses:
            a = hit["variants"][variant][arm]["process"]
            b = miss["variants"][variant][arm]["process"]
            win += a > b
            tie += a == b
    count = len(hits) * len(misses)
    return {"family_accuracy": (win + .5 * tie) / count if count else None,
            "correct_incorrect_pairs": count, "wins": win, "ties": tie}


def question_metrics(context, labels):
    questions, candidates = [], []
    for key in sorted(context["cohort"]):
        identity = context["cohort"][key]
        rows = sorted(context["grouped"][key], key=lambda r: r["candidate_index"])
        sampled, greedy = rows[:4], rows[4]
        valid = [r for r in sampled if r["trajectory_valid"]]
        scores = {r["candidate_id"]: answer_scores(r["generation"], labels[key]) for r in rows}
        for row in rows:
            candidates.append({"candidate_id": row["candidate_id"], "question_key": key,
                               "process_row_sha256": row["process_row_sha256"], "generation_kind": row["generation_kind"],
                               "trajectory_valid": row["trajectory_valid"], **scores[row["candidate_id"]]})
        role = identity["proposal_role"]
        q = {k: identity[k] for k in ("question_key", "dataset", "family_sha256", "question_type")}
        q.update({"role": role, "source_status": context["checks"][key]["status"] if role == "graph" else "ORDINARY",
                  "bootstrap_stratum": "graph:" + identity["question_type"] if role == "graph" else "ordinary:" + identity["dataset"],
                  "sampled_valid": len(valid)/4, "greedy_valid": float(greedy["trajectory_valid"]),
                  "sampled_length_cap": sum(r["reached_max_new_tokens"] for r in sampled)/4,
                  "greedy_length_cap": float(greedy["reached_max_new_tokens"]),
                  "sampled_response_tokens": sum(r["n_response_tokens"] for r in sampled)/4,
                  "greedy_response_tokens": float(greedy["n_response_tokens"]),
                  "all_sampled_invalid": float(not valid), "pairwise": {}, "selected_candidate_ids": {}})
        for metric in ("em", "f1"):
            q[f"raw_greedy_{metric}"] = scores[greedy["candidate_id"]][metric]
            q[f"format_gated_greedy_{metric}"] = q[f"raw_greedy_{metric}"] * greedy["trajectory_valid"]
            q[f"raw_sampled_mean_{metric}"] = math.fsum(scores[r["candidate_id"]][metric] for r in sampled)/4
            q[f"valid_oracle_{metric}"] = max((scores[r["candidate_id"]][metric] for r in valid), default=0.)
            q[f"oracle_minus_greedy_{metric}"] = q[f"valid_oracle_{metric}"] - q[f"raw_greedy_{metric}"]
        for variant in VARIANTS:
            q["pairwise"][variant], q["selected_candidate_ids"][variant] = {}, {}
            for arm in ARMS:
                selected = context["rankings"][key]["rankings"][variant][arm]["selected_candidate_id"]
                q["selected_candidate_ids"][variant][arm] = selected
                q["pairwise"][variant][arm] = pairwise(valid, scores, variant, arm)
                for metric in ("em", "f1"):
                    q[f"{variant}_{arm}_{metric}"] = scores[selected][metric] if selected else 0.
                for stat in ("alpha_effective", "text_component", "graph_component", "process"):
                    q[f"{variant}_{arm}_sampled_{stat}"] = math.fsum(r["variants"][variant][arm][stat] for r in sampled)/4
            for metric in ("em", "f1"):
                for a, b in (("A", "F"), ("F", "T")):
                    q[f"{variant}_{a}_minus_{b}_{metric}"] = q[f"{variant}_{a}_{metric}"] - q[f"{variant}_{b}_{metric}"]
                q[f"{variant}_A_minus_greedy_{metric}"] = q[f"{variant}_A_{metric}"] - q[f"raw_greedy_{metric}"]
            q[f"{variant}_A_F_selection_disagreement"] = float(q["selected_candidate_ids"][variant]["A"] != q["selected_candidate_ids"][variant]["F"])
        questions.append(q)
    return questions, candidates


def bootstrap_estimates(rows, keys, *, replicates=20000, seed=42, macro_dataset=False):
    """Stratified family bootstrap; each question here is one unique family."""
    if not rows:
        return {k: {"point": None, "ci95": [None, None], "families": 0} for k in keys}
    require(len({r["family_sha256"] for r in rows}) == len(rows), "bootstrap unit must be unique family")
    matrix = np.asarray([[r[k] for k in keys] for r in rows], dtype=float)
    require(bool(np.isfinite(matrix).all()), "nonfinite bootstrap estimand")
    strata = defaultdict(list)
    for i, row in enumerate(rows):
        strata[row["bootstrap_stratum"]].append(i)
    domains = Counter(r["dataset"] for r in rows)
    if macro_dataset:
        require(set(domains) == set(DATASETS), "three-dataset macro requires all three domains")
        weights = np.array([1/(3*domains[r["dataset"]]) for r in rows])
    else:
        weights = np.full(len(rows), 1/len(rows))
    point = weights @ matrix
    boot = np.zeros((replicates, len(keys)))
    rng = np.random.default_rng(seed)
    for stratum in sorted(strata):
        positions = strata[stratum]
        # Within a stratum weights are constant even for dataset macro. A
        # multinomial count vector is exactly an n-family bootstrap resample.
        counts = rng.multinomial(len(positions), np.full(len(positions), 1/len(positions)), size=replicates)
        boot += counts @ (matrix[positions] * weights[positions, None])
    intervals = np.quantile(boot, [.025, .975], axis=0)
    return {k: {"point": float(point[i]), "ci95": [float(intervals[0,i]), float(intervals[1,i])], "families": len(rows)}
            for i, k in enumerate(keys)}


def cohort_summary(rows):
    keys = sorted(k for k, v in rows[0].items() if type(v) in (int, float)) if rows else []
    result = {"questions": len(rows), "families": len(rows), "metrics": bootstrap_estimates(rows, keys), "pairwise": {}}
    for variant in VARIANTS:
        result["pairwise"][variant] = {}
        for arm in ARMS:
            mixed = [{"family_sha256": r["family_sha256"], "bootstrap_stratum": r["bootstrap_stratum"],
                      "dataset": r["dataset"], "accuracy": r["pairwise"][variant][arm]["family_accuracy"]}
                     for r in rows if r["pairwise"][variant][arm]["correct_incorrect_pairs"]]
            stats = bootstrap_estimates(mixed, ["accuracy"])["accuracy"]
            details = [r["pairwise"][variant][arm] for r in rows]
            pairs, wins, ties = (sum(r[k] for r in details) for k in ("correct_incorrect_pairs", "wins", "ties"))
            result["pairwise"][variant][arm] = {**stats, "mixed_outcome_families": len(mixed),
                "correct_incorrect_pairs": pairs, "wins": wins, "ties": ties,
                "micro_pair_accuracy_diagnostic": (wins + .5*ties)/pairs if pairs else None}
    return result


def target_decision(estimate, target, *, information=False):
    point, interval = estimate["point"], estimate["ci95"]
    if point is None:
        status = "INCONCLUSIVE"
    elif point >= target:
        status = "PASS"
    elif information:
        status = "INCONCLUSIVE"
    elif interval[1] is not None and interval[1] < target:
        status = "FAIL"
    else:
        status = "INCONCLUSIVE"
    return {"status": status, "target_min": target, **estimate}


def decide(summaries):
    overall, graph, passed = summaries["all132_ITT"], summaries["graph96_ITT"], summaries["source_PASS79"]
    health = {"sampled_micro_valid": target_decision(overall["metrics"]["sampled_valid"], .9),
              "sampled_three_dataset_macro_valid": target_decision(summaries["three_dataset_macro"]["sampled_valid"], .9)}
    pair = passed["pairwise"]["features_v2"]["A"]
    information = {"source_pass_mixed_families": {"status": "PASS" if pair["mixed_outcome_families"] >= 25 else "INCONCLUSIVE",
                                                  "observed": pair["mixed_outcome_families"], "target_min": 25},
                   "graph_valid_oracle_minus_raw_greedy_em": target_decision(graph["metrics"]["oracle_minus_greedy_em"], .03, information=True)}
    info_pass = all(v["status"] == "PASS" for v in information.values())
    utility = {"source_pass_A_pairwise": target_decision(pair, .65),
               "graph_A_minus_raw_greedy_em": target_decision(graph["metrics"]["features_v2_A_minus_greedy_em"], 0.),
               "graph_A_minus_F_em": target_decision(graph["metrics"]["features_v2_A_minus_F_em"], 0.)}
    if not info_pass:
        for value in utility.values():
            value["point_ci_status_diagnostic"] = value["status"]
            value["status"] = "INCONCLUSIVE"
            value["reason"] = "fixed information gate insufficient; no small-family utility rejection"
    health_status = ("FAIL" if any(v["status"] == "FAIL" for v in health.values()) else
                     "PASS" if all(v["status"] == "PASS" for v in health.values()) else "INCONCLUSIVE")
    if not info_pass:
        utility_status = "INCONCLUSIVE"
    elif any(v["status"] == "FAIL" for v in utility.values()):
        utility_status = "FAIL"
    elif all(v["status"] == "PASS" for v in utility.values()):
        utility_status = "PASS"
    else:
        utility_status = "INCONCLUSIVE"
    status = ("FAIL" if "FAIL" in (health_status, utility_status) else
              "PASS" if health_status == utility_status == "PASS" else "INCONCLUSIVE")
    return {"status": status, "overall_status": status, "health_status": health_status,
            "independent_utility_status": utility_status,
            "engineering_probe_eligibility": utility_status == "PASS",
            "health": health, "information": information, "utility": utility,
            "matched600_investment_clearance": status == "PASS", "full_ppo_auto_launch": False,
            "primary_variant": "features_v2", "secondary_cannot_rescue_primary": True,
            "A_equals_F_can_pass_but_does_not_establish_equivalence": True}



def analyze(context, labels):
    questions, candidates = question_metrics(context, labels)
    populations = {"all132_ITT": questions, "graph96_ITT": [r for r in questions if r["role"] == "graph"],
                   "source_PASS79": [r for r in questions if r["source_status"] == "PASS"],
                   "ordinary36_diagnostic": [r for r in questions if r["role"] == "ordinary"]}
    for status in ("UNVERIFIED", "FAIL"):
        populations[f"graph_source_{status}"] = [r for r in questions if r["source_status"] == status]
    for kind in TYPES:
        populations[f"graph_type_{kind}_ITT"] = [r for r in questions if r["role"] == "graph" and r["question_type"] == kind]
        populations[f"graph_type_{kind}_source_PASS"] = [r for r in populations[f"graph_type_{kind}_ITT"] if r["source_status"] == "PASS"]
    for dataset in DATASETS:
        populations[f"dataset_{dataset}"] = [r for r in questions if r["dataset"] == dataset]
        populations[f"ordinary_{dataset}"] = [r for r in populations["ordinary36_diagnostic"] if r["dataset"] == dataset]
    summaries = {name: cohort_summary(rows) for name, rows in populations.items()}
    macro_keys = ["sampled_valid", "raw_greedy_em", "raw_greedy_f1"] + [f"{v}_{a}_{m}" for v in VARIANTS for a in ARMS for m in ("em", "f1")]
    summaries["three_dataset_macro"] = bootstrap_estimates(questions, macro_keys, macro_dataset=True)
    return questions, candidates, summaries, decide(summaries)


def run(*, protocol, scoring, out):
    out = Path(out).resolve()
    require(not out.exists(), "analysis output exists; never overwrite a consumed confirmation")
    out.mkdir(parents=True, exist_ok=False)
    gold_opened = False
    write_new(out / "started.json", {"schema_version": SCHEMA, "status": "STARTED_NOT_ANALYZED",
                                    "gold_values_opened": False, "automatic_reanalysis_allowed": False})
    try:
        context = verify_before_gold(protocol, scoring)
        write_new(out / "before_gold.json", context["seal"])
        gold_opened = True
        labels = load_gold_after_seal(context)
        questions, candidates, summaries, decision = analyze(context, labels)
        del labels
        for ref in context["frozen_files"].values():
            checked(ref)
        for ref in context["protocol"]["analysis"]["gold_sources"].values():
            checked(ref)
        for name, rows in (("questions.jsonl", questions), ("candidate_metrics.jsonl", candidates)):
            with (out / name).open("x", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(canonical(row) + "\n")
        report = {"schema_version": SCHEMA, "status": decision["status"], "experiment_id": context["protocol"]["experiment_id"],
                  "protocol_sha256": context["protocol_sha256"],
                  "overall_status": decision["overall_status"], "health_status": decision["health_status"],
                  "independent_utility_status": decision["independent_utility_status"],
                  "engineering_probe_eligibility": decision["engineering_probe_eligibility"], "decision": decision, "decision_rules": decision_rules(),
                  "population_summaries": summaries, "integrity": {"status": "PASS", "before_gold": binding(out / "before_gold.json")},
                  "gold_values_opened_after_rank_seal": True, "gold_values_emitted": False, "model_updates": 0,
                  "gate_fitting": False, "parent_gate_modified": False, "ppo_launched": False,
                  "statistical_runtime": {"numpy": np.__version__, "python": sys.version},
                  "limitations": ["Fixed fresh132 is graph-heavy, not a balanced three-domain baseline.",
                    "Ordinary36 is diagnostic and overlaps future formal PPO training; graph96 does not.",
                    "Inference type absent; source-PASS bridge stratum is small.",
                    "Family bootstrap CIs are descriptive, not multiplicity-adjusted simultaneous confidence.",
                    "All-invalid sampled families remain ITT with zero process-top1 scores.",
                    "A-F equality or a degenerate zero-disagreement CI does not establish equivalence or alpha superiority.",
                    "F-T changes both graph weighting and the text coefficient; it is report-only.",
                    "No outcome-conditioned resampling, gate choice, refit or automatic full PPO launch."]}
        write_new(out / "report.json", report)
        write_new(out / "manifest.json", {"schema_version": SCHEMA, "status": report["status"],
            "protocol_sha256": context["protocol_sha256"], "outputs": {name: binding(out / name) for name in (
                "started.json", "before_gold.json", "questions.jsonl", "candidate_metrics.jsonl", "report.json")},
            "independent_utility_status": decision["independent_utility_status"],
            "engineering_probe_eligibility": decision["engineering_probe_eligibility"],
            "matched600_investment_clearance": decision["matched600_investment_clearance"], "full_ppo_auto_launch": False})
        return {"status": report["status"], "report": binding(out / "report.json"), "matched600_investment_clearance": decision["matched600_investment_clearance"]}
    except Exception as exc:
        write_new(out / "failed.json", {"schema_version": SCHEMA, "status": "FAIL", "category": "INTEGRITY_OR_ANALYSIS_FAILURE",
            "error_type": type(exc).__name__, "gold_boundary_entered": gold_opened,
            "matched600_investment_clearance": False, "full_ppo_auto_launch": False,
            "note": "No exception text emitted because a parser exception may contain label values."})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scoring", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = vars(parser.parse_args())
    try:
        print(canonical(run(**args)))
    except Exception as exc:
        print(canonical({"status": "FAIL", "error_type": type(exc).__name__, "details": "see new analysis directory failed.json"}), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
