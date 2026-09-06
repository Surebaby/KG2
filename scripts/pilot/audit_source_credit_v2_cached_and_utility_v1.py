"""CPU cached-real-ReaRAG reward integration and post-fit process utility.

No model weights, optimizer, GPU inference, gate fitting or selection. The
32 candidate identities are inherited unchanged from the completed v1 audit.
Gold is read only after every process score is computed, for outcome checks
and explicitly descriptive frozen-candidate utility; it is never emitted.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path

from scripts.pilot.check_source_credit_runtime_cached_v1 import (
    ROOT, CachedRealReaRAG, Forbidden, bound, identity, rows, sha, token_oracle,
)

ARMS = {"A": "learned", "F": "fixed", "T": "text"}
VARIANTS = ("norm_only", "features_v2")
BOUNDARY = (
    "CPU cached-real-ReaRAG routing/arithmetic/token allocation only, not a GPU A-probe. "
    "Utility is post-fit train-bank K2 descriptive reanalysis, not fresh confirmation, "
    "PPO gains or input-source repair. All variants are reported without selection or tuning. "
    "Top1 uses process scores only; invalid candidates are ineligible and all-invalid questions "
    "receive zero. Oracle EM/F1@2 are Gold-assisted upper bounds, not deployable selections."
)


def independent_terms(row, artifact, arm):
    """Scalar oracle: no production gate/normalizer/reward helper is called."""
    if arm not in ARMS:
        raise ValueError("unknown arm")
    features, norm = row["features"], artifact["normalization"]
    names = artifact["feature_names"]
    if len(names) != len(artifact["weights"]):
        raise ValueError("feature/weight dimension mismatch")
    values = [features["values"][name] for name in names]
    standard = artifact["feature_standardization"]
    logit = artifact["bias"] + sum(
        weight * (value - standard["mean"][name]) / standard["scale"][name]
        for name, value, weight in zip(names, values, artifact["weights"]))
    learned = 1.0 / (1.0 + math.exp(-max(-60., min(60., logit))))
    valid = row["trajectory_valid"]
    alpha = features["m_graph"] * ({"A": learned, "F": norm["fixed_alpha"], "T": 0.}[arm]) if valid else 0.
    raw = row["raw_text"] if valid else []
    if valid and not raw:
        raise ValueError("valid candidate missing actual cached step scores")
    text_norm = norm.get("text_v2", norm)
    z = [(value - text_norm["text_center"]) / text_norm["text_scale"] for value in raw]
    bounded = ([value / (1. + abs(value)) for value in z] if "text_v2" in norm
               else [max(-1., min(1., value)) for value in z])
    steps = [.3 * (1. - alpha) * value / len(bounded) for value in bounded]
    graph_z = (row["raw_graph"] - norm["graph_center"]) / norm["graph_scale"] if valid else 0.
    graph = .2 * alpha * max(-1., min(1., graph_z))
    text = sum(steps)
    numeric = [alpha, learned, logit, graph, text, *steps, *z, *bounded]
    if not all(math.isfinite(v) for v in numeric) or not 0 <= alpha <= 1:
        raise ValueError("nonfinite/out-of-range oracle")
    return {"alpha": alpha, "learned_unmasked": learned, "logit": logit,
            "z": z, "bounded": bounded, "steps": steps, "text": text,
            "graph": graph, "process": text + graph,
            "mean_bounded": math.fsum(bounded) / len(bounded) if bounded else 0.}


def rank_value(delta):
    return 1. if delta > 1e-10 else 0. if delta < -1e-10 else .5


def pair_summary(diagnostics, version, graph_only=False):
    grouped = defaultdict(list)
    for row in diagnostics:
        grouped[row["question_key"]].append(row)
    if any(len(pair) != 2 for pair in grouped.values()):
        raise ValueError("utility population is not exact K2")
    selected = [sorted(pair, key=lambda r: r["em"], reverse=True) for pair in grouped.values()
                if all(r["valid"] for r in pair) and pair[0]["em"] != pair[1]["em"]
                and (not graph_only or all(r["credit_eligible"] for r in pair))]
    rankings, deltas, ids = {}, [], []
    for arm in ARMS:
        counts = Counter(rank_value(hit["versions"][version][arm]["process"] -
                                    miss["versions"][version][arm]["process"]) for hit, miss in selected)
        rankings[arm] = {"win": counts[1.], "tie": counts[.5], "loss": counts[0.],
                         "tie_adjusted_accuracy": (counts[1.] + .5 * counts[.5]) / len(selected) if selected else None}
    for hit, miss in selected:
        a, f = (rank_value(hit["versions"][version][arm]["process"] -
                          miss["versions"][version][arm]["process"]) for arm in ("A", "F"))
        deltas.append(a - f)
        ids.append([hit["candidate_id"], miss["candidate_id"]])
    return {"n_pairs": len(selected), "pair_ids": sorted(ids), "rankings": rankings,
            "A_minus_F": {"improved": sum(d > 0 for d in deltas), "worse": sum(d < 0 for d in deltas),
                          "unchanged": sum(d == 0 for d in deltas),
                          "mean_rank_delta": sum(deltas) / len(deltas) if deltas else None}}


def top1_summary(diagnostics, version):
    """Selection never reads EM/F1; labels are used only after choosing an ID."""
    grouped = defaultdict(list)
    for row in diagnostics:
        grouped[row["question_key"]].append(row)
    if any(len(pair) != 2 for pair in grouped.values()):
        raise ValueError("top1 population is not exact K2")
    selected = {arm: [] for arm in ARMS}
    oracle_em, oracle_f1, valid_oracle_em, valid_oracle_f1 = [], [], [], []
    no_valid = 0
    for key, pair in sorted(grouped.items()):
        eligible = [row for row in pair if row["valid"]]
        no_valid += not eligible
        for arm in ARMS:
            # Fixed exact-score tie rule; no Gold tie breaking or variant selection.
            choice = min(eligible, key=lambda row: (-row["versions"][version][arm]["process"],
                                                   row["candidate_id"])) if eligible else None
            selected[arm].append({"question_key": key, "candidate_id": choice["candidate_id"] if choice else None,
                                  "em": choice["em"] if choice else 0., "f1": choice["f1"] if choice else 0.})
        oracle_em.append(max(r["em"] for r in pair))
        oracle_f1.append(max(r["f1"] for r in pair))
        valid_oracle_em.append(max((r["em"] for r in eligible), default=0.))
        valid_oracle_f1.append(max((r["f1"] for r in eligible), default=0.))
    n = len(grouped)
    mean = lambda values: sum(values) / n if n else None
    return {"questions": n, "all_invalid_questions": no_valid,
            "selection": "valid candidates only; highest process score; exact ties smallest candidate_id; no valid -> zero",
            "arms": {arm: {"em": mean([r["em"] for r in choices]), "f1": mean([r["f1"] for r in choices])}
                     for arm, choices in selected.items()},
            "oracle_at2": {"em": mean(oracle_em), "f1": mean(oracle_f1)},
            "format_valid_oracle_at2": {"em": mean(valid_oracle_em), "f1": mean(valid_oracle_f1)},
            "selected_ids": {arm: [[r["question_key"], r["candidate_id"]] for r in choices]
                             for arm, choices in selected.items()}}


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_rows(path, values):
    with path.open("x") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def write_tensorboard_diagnostic(output_dir, variant, reward_infos, artifact):
    """Read back actual production-writer events from 32 real cached A rows."""
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from kgproweight.training.ppo_tensorboard import log_ppo_batch
    if len(reward_infos) != 32:
        raise ValueError("TensorBoard diagnostic requires the fixed 32 real candidate rows")
    directory = output_dir / "tensorboard_zero_update_diagnostic" / (variant + "_cached_SFT_A")
    token_copies = [row["token_rewards"].clone() for row in reward_infos]
    with SummaryWriter(str(directory)) as writer:
        writer.add_text("diagnostic/contract", json.dumps({
            "backend": "cached-real-ReaRAG", "run_kind": "zero_update_SFT_candidate_diagnostic",
            "optimizer_updates": 0, "ppo_training_run": False,
            "normalization": artifact["normalization"], "gate_payload_sha256": artifact["payload_sha256"]}, sort_keys=True), 0)
        log_ppo_batch(writer, step=32, update_index=1, stats={"diagnostic/optimizer_updates": 0},
                      reward_infos=reward_infos, histogram_every=1)
    assert all(torch.equal(row["token_rewards"], before) for row, before in zip(reward_infos, token_copies))
    events = EventAccumulator(str(directory), size_guidance={"scalars": 0, "histograms": 0}).Reload()
    valid = [row for row in reward_infos if row["trajectory_valid"]]
    eligible = [row for row in valid if row["source_gate"]["m_graph"] == 1]
    norm = artifact["normalization"]["text_v2"]
    raw = [value for row in valid for value in row["mixed_reward"]["text_raw_step_scores"]]
    z = [(value - norm["text_center"]) / norm["text_scale"] for value in raw]
    expected = {"diagnostic/optimizer_updates": 0., "reward/all/count": 32.,
        "reward/all/valid_rate": len(valid) / 32., "reward/all/text_clip_frac": 0.,
        "reward/all/text_raw_z_outside_unit_frac": sum(abs(value) > 1 for value in z) / len(z),
        "reward/all/text_softsign_saturation_frac": sum(abs(value / (1 + abs(value))) >= .95 for value in z) / len(z),
        "gate/all/alpha_effective_mean": float(np.mean([row["source_gate"]["alpha_effective"] for row in reward_infos]))}
    for name in artifact["feature_names"]:
        expected["gate/eligible_valid/feature_" + name + "_mean"] = float(np.mean([
            row["source_gate"]["features"]["values"][name] for row in eligible]))
    readback = {}
    for tag, value in expected.items():
        entries = events.Scalars(tag)
        assert len(entries) == 1 and entries[0].step == 32
        assert abs(entries[0].value - value) < 1e-6, (tag, entries[0].value, value)
        readback[tag] = {"expected": value, "event_value": entries[0].value, "pass": True}
    event_files = {str(path.relative_to(output_dir)): identity(path) for path in directory.glob("events.out.tfevents.*")}
    assert event_files and events.Tags()["histograms"]
    return {"run_kind": "zero_update_SFT_candidate_diagnostic", "variant": variant,
            "candidate_count": 32, "arm": "A", "optimizer_updates": 0, "ppo_training_run": False,
            "event_files": event_files, "checked_scalar_tags": readback,
            "all_scalar_tag_count": len(events.Tags()["scalars"]), "all_histogram_tag_count": len(events.Tags()["histograms"]),
            "token_rewards_unchanged_after_logging": True}


def run(calibration_dir, output_dir, experiment_id, tensorboard_diagnostic=False):
    output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["HF_MODULES_CACHE"] = str(output_dir / "tokenizer_code_cache")
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from kgproweight.data.parsers import extract_final_answer
    from kgproweight.eval.pred_processing import extract_kg_proweight_answer
    from kgproweight.reward.proofkg_process import canonical_exact_match, canonical_token_f1
    from kgproweight.reward.source_credit_gate_v1 import SourceCreditGateV1
    from kgproweight.reward.source_credit_gate_v2 import SourceCreditGateV2
    from kgproweight.reward.source_quality_gate_v1 import canonical_sha256, heuristic_ratio_target
    from kgproweight.training.reward_function import (
        KGProWeightRewardFunction, RewardSpec, _canonical_gold_surfaces, source_gate_text_inputs_v1,
        source_gate_text_budget_v1, validate_source_gate_trajectory,
    )
    assert not torch.cuda.is_initialized() and os.environ["CUDA_VISIBLE_DEVICES"] == ""
    bindings = {}
    def track(name, path):
        bindings[name] = identity(path)
        return Path(path)
    def checked(name, value, base=ROOT):
        return track(name, bound(value, base))
    code_paths = [Path(__file__), ROOT / "scripts/pilot/check_source_credit_runtime_cached_v1.py",
                  ROOT / "kgproweight/training/reward_function.py", ROOT / "kgproweight/data/parsers.py",
                  ROOT / "kgproweight/eval/pred_processing.py", ROOT / "kgproweight/reward/proofkg_process_v2_3.py",
                  ROOT / "kgproweight/reward/source_credit_gate_v1.py", ROOT / "kgproweight/reward/source_credit_gate_v2.py",
                  ROOT / "kgproweight/reward/source_reward_normalization_v2.py", ROOT / "kgproweight/reward/source_trajectory_features_v2.py",
                  ROOT / "kgproweight/training/ppo_tensorboard.py"]
    code_bindings = {str(p.relative_to(ROOT)): identity(p) for p in code_paths}
    manifest_path = track("calibration_manifest", calibration_dir / "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] in {"source-credit-v2-development-calibration-manifest-v1",
                                          "source-credit-v2-representative-text-population-rebind-v1"}
    assert manifest["training_clearance"] is False and manifest["ppo_launch_clearance"] is False
    base_manifest = manifest
    if manifest["schema_version"] == "source-credit-v2-representative-text-population-rebind-v1":
        base_manifest = json.loads(checked("base_v2_manifest", manifest["parent_manifest"]).read_text())
        assert base_manifest["schema_version"] == "source-credit-v2-development-calibration-manifest-v1"
        assert base_manifest["training_clearance"] is False and base_manifest["ppo_launch_clearance"] is False
        checked("representative_normalization_bank", manifest["normalization_bank"])
        for name, binding in manifest["source_bindings"].items():
            checked("representative_source:" + name, binding)
        for variant in VARIANTS:
            for name, binding in base_manifest["outputs"][variant].items():
                checked(f"base_v2:{variant}:{name}", binding)
            for name in ("candidates.jsonl", "assignments.jsonl"):
                assert base_manifest["outputs"][variant][name]["sha256"] == manifest["outputs"][variant][name]["sha256"]
        assert manifest["source_credit_mask"] == base_manifest["source_credit_mask"]
    checked("protocol", base_manifest["protocol"])
    for variant in VARIANTS:
        for name, binding in manifest["outputs"][variant].items():
            checked(f"{variant}:{name}", binding)
    parent_manifest_path = checked("v1_manifest", base_manifest["parent_manifest"])
    parent_manifest = json.loads(parent_manifest_path.read_text())
    parent_paths = {name: checked("v1:" + name, binding)
                    for name, binding in parent_manifest["outputs"].items()}
    v1_gate = SourceCreditGateV1.load(parent_paths["gate.json"], allow_unvalidated=True)
    gates, candidate_views, assignments = {"v1": v1_gate}, {}, {}
    candidate_views["v1"] = {r["candidate_id"]: r for r in rows(parent_paths["candidates.credit_masked.jsonl"])}
    assignments["v1"] = {r["candidate_id"]: r for r in rows(parent_paths["assignments.jsonl"])}
    for variant in VARIANTS:
        gate_path = Path(bindings[f"{variant}:gate.json"]["path"])
        gates[variant] = SourceCreditGateV2.load(gate_path, allow_unvalidated=True)
        # Reuse the verified mask while checking the production constructor guard.
        try:
            SourceCreditGateV2(gates[variant].artifact, mask=gates[variant].mask)
        except ValueError as exc:
            assert "fresh confirmation" in str(exc)
        else:
            raise AssertionError("unconfirmed gate was accepted for production")
        assert gates[variant].artifact["source_credit_mask"] == manifest["source_credit_mask"]
        candidate_views[variant] = {r["candidate_id"]: r for r in rows(bindings[f"{variant}:candidates.jsonl"]["path"])}
        assignments[variant] = {r["candidate_id"]: r for r in rows(bindings[f"{variant}:assignments.jsonl"]["path"])}
    assert all(len(view) == 1660 and set(view) == set(candidate_views["v1"]) for view in candidate_views.values())
    assert all(set(view) == set(candidate_views["v1"]) for view in assignments.values())
    input_dir = ROOT / "outputs/audits/source_quality_candidate_bank_v1_inputs_seed42_tensorboard_v1"
    input_manifest = json.loads(track("input_manifest", input_dir / "manifest.json").read_text())
    input_index = {r["question_key"]: r for r in rows(checked("input_rows", input_manifest["outputs"]["inputs.jsonl"], input_dir))}
    generation_dir = ROOT / "outputs/audits/source_quality_candidate_bank_v1_generated_seed42_local4090_v1"
    generation_manifest = json.loads(track("generation_manifest", generation_dir / "manifest.json").read_text())
    assert generation_manifest["bank_manifest_sha256"] == sha(input_dir / "manifest.json")
    generations = {r["candidate_id"]: r for r in rows(checked("generations", generation_manifest["outputs"]["generations.jsonl"], generation_dir))}
    scored_manifest_path = checked("scored_manifest", parent_manifest["parent_validation"]["original_bank_manifest"])
    scored_manifest = json.loads(scored_manifest_path.read_text())
    scored = {r["candidate_id"]: r for r in rows(checked("scored_rows", parent_manifest["parent_validation"]["parent_bank"]))}
    prior_dir = ROOT / "outputs/audits/source_credit_runtime_cached_v1_local_seed42"
    prior_checks = rows(track("v1_cached_checks", prior_dir / "checks.jsonl"))
    prior_report = json.loads(track("v1_cached_report", prior_dir / "report.json").read_text())
    assert prior_report["all_checks_passed"] is True and len(prior_checks) == 96
    selected_ids = list(dict.fromkeys(r["candidate_id"] for r in prior_checks))
    reasons = {r["candidate_id"]: r["selection_reason"] for r in prior_checks}
    assert len(selected_ids) == 32
    track("mask_manifest", v1_gate.mask.manifest_path)
    # All score arithmetic is completed without opening Gold labels.
    process_by_id = {}
    for cid, old_row in candidate_views["v1"].items():
        key = old_row["dataset"] + "::" + old_row["qid"]
        frozen, generation, original = input_index[key], generations[cid], scored[cid]
        assert old_row["generation"] == generation["generation"] == original["generation"]
        assert old_row["raw_text"] == original["raw_text"] and old_row["raw_graph"] == original["raw_graph"]
        assert old_row["input_sha256"] == generation["input_sha256"] == frozen["input_sha256"]
        validated = {**original, "quality": heuristic_ratio_target(original["raw_graph"], original["raw_text"],
                     m_graph=original["features"]["m_graph"], trajectory_valid=original["trajectory_valid"])}
        assert old_row["parent_validated_row_sha256"] == canonical_sha256(validated)
        process_by_id[cid] = {}
        feature_spec = RewardSpec(**deepcopy(frozen["spec"]), gold_answer="")
        feature_validity = validate_source_gate_trajectory(feature_spec, old_row["generation"], format_version="v2")
        assert feature_validity["valid"] is old_row["trajectory_valid"]
        for version, view in candidate_views.items():
            row, gate = view[cid], gates[version]
            for field in ("generation", "raw_text", "raw_graph", "trajectory_valid", "input_sha256", "source_credit"):
                assert row[field] == old_row[field], (version, cid, field)
            if version == "features_v2":
                assert row["parent_credit_row_sha256"] == canonical_sha256(old_row)
            assert assignments[version][cid]["split"] == assignments["v1"][cid]["split"]
            gate.mask.validate_masked_features(row["features"])
            if version in VARIANTS:
                runtime_features = gate.mask_features(feature_spec, gate.compute_features(
                    feature_spec, feature_validity["steps"], row["proof_result"] if feature_validity["valid"] else {}))
                assert runtime_features["values"] == row["features"]["values"], ("all_bank_runtime_feature_mismatch", version, cid)
                assert runtime_features["m_graph"] == row["features"]["m_graph"]
            terms = {arm: independent_terms(row, gate.artifact, arm) for arm in ARMS}
            predicted = gate.predict(row["features"])
            expected_alpha = terms["A"]["learned_unmasked"] * row["features"]["m_graph"]
            assert abs(predicted - expected_alpha) < 1e-12
            assert abs(assignments[version][cid]["alpha"] - predicted) < 1e-12
            process_by_id[cid][version] = terms
        assert process_by_id[cid]["v1"]["A"]["alpha"] == process_by_id[cid]["norm_only"]["A"]["alpha"]
    process_digest = canonical_sha256(process_by_id)
    print(json.dumps({"stage": "all_process_terms_computed_before_gold_read", "candidates": len(process_by_id)}), flush=True)
    silver_path = checked("train_gold_source", input_manifest["source_bindings"]["silver"])
    labels_list = rows(silver_path)
    labels = {r["dataset"] + "::" + r["qid"]: r for r in labels_list}
    assert len(labels) == len(labels_list)
    diagnostics = []
    for cid, row in candidate_views["v1"].items():
        key = row["dataset"] + "::" + row["qid"]
        label = labels[key]
        assert label["question"] == row["question"] and label["metadata"]["source_split"] == "train"
        surfaces = _canonical_gold_surfaces(label["metadata"]["gold_answer"], label["metadata"].get("gold_answer_aliases"))
        answer = extract_kg_proweight_answer(extract_kg_proweight_answer(row["generation"]))
        diagnostics.append({"candidate_id": cid, "question_key": key, "dataset": row["dataset"],
            "valid": row["trajectory_valid"], "credit_eligible": bool(row["features"]["m_graph"]),
            "source_status": row["source_credit"]["status"], "split": assignments["v1"][cid]["split"],
            "em": max(canonical_exact_match(answer, surface) for surface in surfaces),
            "f1": max(canonical_token_f1(answer, surface) for surface in surfaces),
            "versions": process_by_id[cid], "calibration_input_eligible": False,
            "gold_access": "post_fit_diagnostic_only"})
    assert canonical_sha256(process_by_id) == process_digest
    prior_utility_dir = ROOT / "outputs/audits/source_credit_reward_utility_v1_local_seed42"
    prior_utility_manifest = json.loads(track("v1_utility_manifest", prior_utility_dir / "manifest.json").read_text())
    prior_utility_rows = {r["candidate_id"]: r for r in rows(checked("v1_utility_rows", prior_utility_manifest["outputs"]["candidate_diagnostics.jsonl"]))}
    for row in diagnostics:
        old = prior_utility_rows[row["candidate_id"]]
        assert (row["valid"], row["em"], row["f1"], row["credit_eligible"]) == (old["valid"], old["em"], old["f1"], old["credit_eligible"])
        for arm, field in (("A", "learned_A_process"), ("F", "fixed_F_process"), ("T", "text_T_process")):
            assert abs(row["versions"]["v1"][arm]["process"] - old[field]) < 1e-12
    utility = {version: {"paired_all": pair_summary(diagnostics, version),
                         "paired_credit_eligible": pair_summary(diagnostics, version, True),
                         "top1_all": top1_summary(diagnostics, version),
                         "top1_credit_eligible": top1_summary([r for r in diagnostics if r["credit_eligible"]], version)}
               for version in gates}
    for version, result in utility.items():
        assert result["paired_all"]["n_pairs"] == 143 and result["paired_credit_eligible"]["n_pairs"] == 114
        for population in ("paired_all", "paired_credit_eligible"):
            assert result[population]["pair_ids"] == utility["v1"][population]["pair_ids"]
            result[population]["pair_ids_sha256"] = canonical_sha256(result[population]["pair_ids"])
    write_rows(output_dir / "candidate_diagnostics.jsonl", diagnostics)
    write_json(output_dir / "utility.json", utility)
    # Only tokenizer weights/code are needed to replay production reward routing.
    policy_dir, rearag_dir = (ROOT / input_manifest[name]["path"] for name in ("policy_tokenizer", "rearag_model"))
    for kind, folder in (("policy_tokenizer", policy_dir), ("rearag_model", rearag_dir)):
        for name, binding in input_manifest[kind]["files"].items():
            if "tokenizer" in name or name == "special_tokens_map.json" or name == "tokenization_chatglm.py":
                checked(f"actual_tokenizer:{kind}:{name}", binding, folder)
    policy_tokenizer = AutoTokenizer.from_pretrained(str(policy_dir), local_files_only=True)
    rearag_tokenizer = AutoTokenizer.from_pretrained(str(rearag_dir), trust_remote_code=True, local_files_only=True)
    archived_reward = scored_manifest_path.parent / "runtime_code/kgproweight/training/reward_function.py"
    checked("archived_scoring_reward", scored_manifest["source_bindings"]["code:kgproweight/training/reward_function.py"] | {"path": str(archived_reward)})
    old_code = archived_reward.read_text()
    old_helper = next(ast.get_source_segment(old_code, node) for node in ast.parse(old_code).body
                      if isinstance(node, ast.FunctionDef) and node.name == "source_gate_text_inputs_v1")
    assert old_helper.strip() == inspect.getsource(source_gate_text_inputs_v1).strip()
    checks, cache_calls = [], 0
    tensorboard_rows = {variant: [] for variant in VARIANTS}
    for cid in selected_ids:
        old_row = candidate_views["v1"][cid]
        key = old_row["dataset"] + "::" + old_row["qid"]
        frozen, generation, metadata = input_index[key], generations[cid], labels[key]["metadata"]
        spec = RewardSpec(**deepcopy(frozen["spec"]), gold_answer=metadata["gold_answer"],
                          gold_answer_aliases=metadata.get("gold_answer_aliases") or [])
        spec_hash = canonical_sha256(spec.__dict__)
        validity = validate_source_gate_trajectory(spec, old_row["generation"], format_version="v2")
        assert validity["valid"] is old_row["trajectory_valid"]
        prompts, texts = source_gate_text_inputs_v1(spec, validity["steps"])
        ids = generation["response_token_ids"]
        assert policy_tokenizer.decode(ids, skip_special_tokens=True) == old_row["generation"]
        surfaces = _canonical_gold_surfaces(spec.gold_answer, spec.gold_answer_aliases)
        answer = extract_final_answer(old_row["generation"]) or ""
        outcome = (4. * (max(canonical_exact_match(answer, surface) for surface in surfaces) +
                            .1 * max(canonical_token_f1(answer, surface) for surface in surfaces)) if validity["valid"] else -4.)
        for variant in VARIANTS:
            row, gate = candidate_views[variant][cid], gates[variant]
            for arm, mode in ARMS.items():
                oracle = process_by_id[cid][variant][arm]
                cache = CachedRealReaRAG(rearag_tokenizer, prompts, texts, row["raw_text"])
                reward = KGProWeightRewardFunction(
                    alpha_gate=Forbidden(), prm_annotator=Forbidden(), text_reward_model=cache,
                    tokenizer=policy_tokenizer, outcome_weight=4., text_reward_scale=.3, max_steps=5,
                    proofkg_process_reward=True, proofkg_process_version="v2_3", proofkg_process_weight=.2,
                    proofkg_f1_weight=.1, proofkg_dynamic_validity=True, mixed_outcome_reward=True,
                    mixed_text_reward=True, runtime_contract_version="v2", source_gated_reward_version="v1",
                    source_gate_format_version="v2", source_gate_credit_version="v2", source_gate_mode=mode,
                    source_quality_gate=gate, center_text_reward=False)
                actual = reward(frozen["prompt"], row["generation"], spec, response_ids=ids)
                if tensorboard_diagnostic and arm == "A":
                    tensorboard_rows[variant].append(actual)
                total = outcome + oracle["process"]
                expected_tokens = token_oracle(ids, policy_tokenizer, oracle["steps"], outcome + oracle["graph"])
                tokens = actual["token_rewards"].cpu().double().numpy()
                token_error = float(np.max(np.abs(tokens - expected_tokens)))
                assert np.isfinite(tokens).all() and token_error < 1e-6 and abs(tokens.sum() - total) < 1e-6
                assert actual["trajectory_valid"] is validity["valid"]
                assert actual["source_gate"]["m_graph"] == row["features"]["m_graph"]
                assert actual["proofkg_process"]["required_steps"] == validity["required_steps"]
                assert abs(actual["source_gate"]["alpha_effective"] - oracle["alpha"]) < 1e-12
                for field, expected in (("outcome", outcome), ("text", oracle["text"]), ("process", oracle["graph"]), ("total", total)):
                    assert abs(actual["mixed_reward"][field] - expected) < 1e-10, (variant, cid, arm, field)
                assert abs(sum(actual["per_step_rewards"]) - total) < 1e-10
                assert cache.calls == int(validity["valid"])
                if validity["valid"]:
                    assert canonical_sha256(source_gate_text_budget_v1(cache, prompts, texts)) == canonical_sha256(row["text_token_budget"])
                    assert actual["mixed_reward"]["text_raw_step_scores"] == row["raw_text"]
                    telemetry = actual["source_gate"]["text_normalization_v2"]
                    assert telemetry["hard_clip_frac"] == 0.
                    assert telemetry["normalized_unclipped_step_scores"] == oracle["z"]
                    assert telemetry["bounded_step_scores"] == oracle["bounded"]
                    assert abs(telemetry["mean_bounded"] - oracle["mean_bounded"]) < 1e-12
                    assert actual["source_gate"]["features"]["values"] == row["features"]["values"]
                    if row["source_credit"]["parent_m_graph"]:
                        assert abs(actual["proofkg_process"]["process_score"] - row["raw_graph"]) < 1e-12
                if not row["features"]["m_graph"]:
                    assert oracle["alpha"] == oracle["graph"] == 0.
                assert canonical_sha256(spec.__dict__) == spec_hash
                checks.append({"candidate_id": cid, "variant": variant, "arm": arm,
                    "selection_reason": reasons[cid], "valid": validity["valid"], "source_status": row["source_credit"]["status"],
                    "parent_m_graph": row["source_credit"]["parent_m_graph"], "m_graph": row["features"]["m_graph"],
                    "feature_dimension": len(gate.artifact["feature_names"]), "alpha": oracle["alpha"],
                    "outcome": outcome, "text": oracle["text"], "graph": oracle["graph"], "total": total,
                    "token_sum": float(tokens.sum()), "token_max_abs_error": token_error,
                    "text_inputs_sha256": canonical_sha256([prompts, texts]),
                    "cached_real_scores_sha256": canonical_sha256(row["raw_text"]), "pass": True})
                cache_calls += cache.calls
        print(json.dumps({"candidate_id": cid, "reward_checks": len(checks), "status": "PASS"}), flush=True)
    assert len(checks) == 192 and not torch.cuda.is_initialized()
    assert canonical_sha256(process_by_id) == process_digest
    tensorboard_report = {variant: write_tensorboard_diagnostic(output_dir, variant, tensorboard_rows[variant], gates[variant].artifact)
                          for variant in VARIANTS} if tensorboard_diagnostic else None
    if tensorboard_report is not None:
        write_json(output_dir / "tensorboard_diagnostic.json", tensorboard_report)
    for name, binding in bindings.items():
        assert sha(binding["path"]) == binding["sha256"], ("input_changed_during_audit", name)
    for name, binding in code_bindings.items():
        assert sha(binding["path"]) == binding["sha256"], ("code_changed_during_audit", name)
    report = {"schema_version": "source-credit-v2-cached-and-utility-audit-v1", "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": 42,
        "status": "PASS_CPU_CACHED_REAL_REARAG_ZERO_UPDATE_AND_POST_FIT_DIAGNOSTIC_NOT_PPO_CLEARANCE",
        "selected_candidates": 32, "valid_selected_candidates": sum(candidate_views["v1"][cid]["trajectory_valid"] for cid in selected_ids),
        "reward_checks": len(checks), "all_checks_passed": True, "variants": list(VARIANTS), "arms": list(ARMS),
        "selection_ids_sha256": canonical_sha256(selected_ids), "cached_real_backend_calls": cache_calls,
        "max_token_abs_error": max(r["token_max_abs_error"] for r in checks),
        "normal_production_constructor_rejected_both_unconfirmed_gates": True,
        "gold_access": "bound train Gold opened only after all 1660 x 3 x A/F/T process terms were computed; outcome check and post-fit diagnostic only",
        "process_terms_sha256_before_and_after_gold": process_digest, "gold_written_to_targets_or_outputs": False,
        "utility_candidate_population": 1660, "utility_all_pairs": 143, "utility_graph_pairs": 114,
        "all_bank_runtime_feature_reconciliation": {"variants": list(VARIANTS), "candidates_per_variant": 1660,
            "valid_candidates_per_variant": 1471, "invalid_candidates_per_variant": 189,
            "feature_value_mismatches": 0, "m_graph_mismatches": 0, "gold_access": False},
        "backend": "cached-real-ReaRAG; exact original real raw_text; exact prompt/step/token budget checked",
        "gpu_used": False, "model_weights_loaded": False, "optimizer_updates": 0,
        "gpu_a_probe_complete": False, "ppo_training_started": False, "ppo_launch_clearance": False,
        "independent_confirmation": False, "source_integrity_clearance": False, "reader_inputs_repaired": False,
        "all_bound_inputs_unchanged_after_audit": True, "scientific_boundary": BOUNDARY,
        "code_unchanged_during_audit": True, "tensorboard_zero_update_diagnostic": tensorboard_report is not None,
        "inputs": bindings, "code_bindings": code_bindings,
        "model_identities_inherited_not_weights_rehashed": {name: input_manifest[name] for name in ("base_model", "rearag_model", "policy_tokenizer")},
        "formula": {"text_v2": "z=(raw_step-center)/scale; bounded=z/(1+abs(z)); text=.3*(1-alpha)*mean(bounded)",
                    "graph": ".2*alpha*clip((raw_graph-center)/scale,-1,1)",
                    "valid_outcome": "4*(EM+.1*F1)", "invalid": "-4 terminal only; no text scorer call"}}
    write_rows(output_dir / "checks.jsonl", checks)
    write_json(output_dir / "report.json", report)
    with (output_dir / "audit_script.executed.py").open("xb") as handle:
        handle.write(Path(__file__).read_bytes())
    write_json(output_dir / "manifest.json", {"schema_version": report["schema_version"], "experiment_id": experiment_id,
        "status": report["status"], "inputs": bindings, "code_bindings": code_bindings,
        "outputs": {name: identity(output_dir / name) for name in ("checks.jsonl", "report.json", "utility.json", "candidate_diagnostics.jsonl", "audit_script.executed.py",
                    *(("tensorboard_diagnostic.json",) if tensorboard_report is not None else ()))},
        "calibration_input_eligible": False, "ppo_launch_clearance": False})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--tensorboard-diagnostic", action="store_true",
                        help="write/read back actual 32-candidate A events in an explicit zero-update diagnostic run")
    args = parser.parse_args()
    existed = args.output_dir.exists()
    try:
        result = run(args.calibration_dir.resolve(), args.output_dir.resolve(), args.experiment_id, args.tensorboard_diagnostic)
    except Exception as exc:
        if not existed and args.output_dir.is_dir() and not (args.output_dir / "FAILED.json").exists():
            write_json(args.output_dir / "FAILED.json", {"status": "FAILED", "type": type(exc).__name__,
                "message": str(exc), "optimizer_updates": 0, "gpu_used": False})
        raise
    print(json.dumps({name: result[name] for name in ("status", "reward_checks", "max_token_abs_error")}))
