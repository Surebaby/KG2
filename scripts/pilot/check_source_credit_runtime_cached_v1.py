"""CPU production-reward audit using frozen *real* ReaRAG scores, no inference.

This is a zero-update integration check, not GPU A-probe or evidence of PPO
utility. Gold is read only for the outcome calculation and is not emitted.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[2]
EMPTY_ID = "2wikimultihopqa::4827b81c0baf11ebab90acde48001122::k1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bound(value, base=ROOT):
    path = Path(value["path"])
    for candidate in ([path] if path.is_absolute() else [base / path, ROOT / path]):
        if candidate.is_file() and sha(candidate) == value["sha256"]:
            return candidate.resolve()
    raise ValueError(f"Missing or changed bound file: {path}")


def identity(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def choose(candidates, generations, n=32):
    ordered = sorted(candidates, key=lambda r: hashlib.sha256(
        ("source-credit-cpu-v1-seed42\0" + r["candidate_id"]).encode()).hexdigest())
    selected, reasons = {}, {}
    def take(label, predicate, count):
        pool = [r for r in ordered if predicate(r)]
        added = 0
        for row in pool:
            key = row["candidate_id"]
            if key not in selected and added < count:
                selected[key] = row
                reasons[key] = label
                added += 1
        return len(pool)
    counts = {}
    counts["known_empty_final"] = take("known_empty_final", lambda r: r["candidate_id"] == EMPTY_ID, 1)
    assert counts["known_empty_final"] == 1
    for status in ("PASS", "FAIL", "UNVERIFIED"):
        for valid in (True, False):
            label = f"graph_{status}_{'valid' if valid else 'invalid'}"
            counts[label] = take(label, lambda r, s=status, v=valid: (
                r["source_credit"]["parent_m_graph"] == 1 and r["source_credit"]["status"] == s
                and r["trajectory_valid"] is v), 2)
    for length in (2, 4):
        label = f"pass_valid_steps_{length}{'_or_more' if length == 4 else ''}"
        counts[label] = take(label, lambda r, length=length: r["source_credit"]["status"] == "PASS"
                             and r["trajectory_valid"] and (
                                 len(r["raw_text"]) == 2 if length == 2 else len(r["raw_text"]) >= 4), 2)
    for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
        label = f"ordinary_{dataset}"
        counts[label] = take(label, lambda r, d=dataset: r["dataset"] == d
                             and r["source_credit"]["parent_m_graph"] == 0 and r["trajectory_valid"], 2)
    counts["valid_length_capped"] = take("valid_length_capped", lambda r: r["trajectory_valid"]
                                         and generations[r["candidate_id"]]["reached_max_new_tokens"], 2)
    take("deterministic_fill", lambda _r: True, max(0, n - len(selected)))
    assert 24 <= len(selected) <= 40
    return list(selected.values()), reasons, counts


class Forbidden:
    def __getattr__(self, name):
        raise AssertionError(f"unused legacy/model component accessed: {name}")
    def __call__(self, *args, **kwargs):
        raise AssertionError("unused legacy/model component called")


class CachedRealReaRAG:
    name = "cached-real-ReaRAG"
    is_dummy = False
    def __init__(self, tokenizer, prompts, texts, values):
        self.backend = SimpleNamespace(tokenizer=tokenizer, max_length=4096)
        self.prompts, self.texts, self.values = prompts, texts, list(values)
        self.calls = 0
    def score_steps(self, prompts, texts):
        assert list(prompts) == self.prompts and list(texts) == self.texts
        assert len(texts) == len(self.values)
        assert self.calls == 0
        self.calls += 1
        return list(self.values)


def token_oracle(ids, tokenizer, step_weights, terminal):
    """Independent linear prefix scan (production uses binary search)."""
    import numpy as np
    output = np.zeros(len(ids), dtype=np.float64)
    if step_weights:
        text = tokenizer.decode(ids, skip_special_tokens=False)
        heads = [m.start() for m in re.finditer(r"\[Step\s+\d+\]", text)]
        finals = [m.start() for m in re.finditer(
            r"\[Final Answer\]|(?:^|\n)\s*\*{0,3}Final Answer\*{0,3}\s*[:：]?", text, re.I)]
        assert len(heads) == len(step_weights) and finals
        end_chars = heads[1:] + [next(position for position in finals if position > heads[-1])]
        lengths = [len(tokenizer.decode(ids[:i], skip_special_tokens=False)) for i in range(len(ids) + 1)]
        for boundary, weight in zip(end_chars, step_weights):
            end = next(i for i, size in enumerate(lengths) if size >= boundary)
            output[max(0, min(end - 1, len(ids) - 1))] += weight
    output[-1] += terminal
    return output


def run(output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["HF_MODULES_CACHE"] = str(output_dir / "tokenizer_code_cache")
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from kgproweight.data.parsers import extract_final_answer
    from kgproweight.reward.proofkg_process import canonical_exact_match, canonical_token_f1
    from kgproweight.reward.source_credit_gate_v1 import SourceCreditGateV1
    from kgproweight.reward.source_quality_gate_v1 import FEATURE_NAMES, canonical_sha256, heuristic_ratio_target
    from kgproweight.training.reward_function import (
        KGProWeightRewardFunction, RewardSpec, _canonical_gold_surfaces,
        source_gate_text_inputs_v1, source_gate_text_budget_v1, validate_source_gate_trajectory,
    )
    assert not torch.cuda.is_initialized()
    calibration = ROOT / "outputs/calibration/source_credit_gate_v1_local_seed42"
    manifest = json.loads((calibration / "manifest.json").read_text())
    gate_path = bound(manifest["outputs"]["gate.json"])
    candidate_path = bound(manifest["outputs"]["candidates.credit_masked.jsonl"])
    gate = SourceCreditGateV1.load(gate_path)
    candidates = rows(candidate_path)
    input_dir = ROOT / "outputs/audits/source_quality_candidate_bank_v1_inputs_seed42_tensorboard_v1"
    input_manifest = json.loads((input_dir / "manifest.json").read_text())
    inputs_path = bound(input_manifest["outputs"]["inputs.jsonl"], input_dir)
    input_index = {r["question_key"]: r for r in rows(inputs_path)}
    generation_dir = ROOT / "outputs/audits/source_quality_candidate_bank_v1_generated_seed42_local4090_v1"
    gen_manifest = json.loads((generation_dir / "manifest.json").read_text())
    assert gen_manifest["bank_manifest_sha256"] == sha(input_dir / "manifest.json")
    generations_path = bound(gen_manifest["outputs"]["generations.jsonl"], generation_dir)
    generations = {r["candidate_id"]: r for r in rows(generations_path)}
    scored_manifest_path = bound(manifest["parent_validation"]["original_bank_manifest"])
    scored_manifest = json.loads(scored_manifest_path.read_text())
    parent_path = bound(manifest["parent_validation"]["parent_bank"])
    parent = {r["candidate_id"]: r for r in rows(parent_path)}
    silver_path = bound(input_manifest["source_bindings"]["silver"])
    labels = {r["dataset"] + "::" + r["qid"]: r for r in rows(silver_path)}
    policy_dir = ROOT / input_manifest["policy_tokenizer"]["path"]
    rearag_dir = ROOT / input_manifest["rearag_model"]["path"]
    tokenizer_bindings = {}
    for kind, folder in (("policy_tokenizer", policy_dir), ("rearag_model", rearag_dir)):
        for name, binding in input_manifest[kind]["files"].items():
            if "tokenizer" in name or name == "special_tokens_map.json":
                target = bound(binding, folder)
                tokenizer_bindings[str(target)] = identity(target)
    policy_tokenizer = AutoTokenizer.from_pretrained(str(policy_dir), local_files_only=True)
    rearag_tokenizer = AutoTokenizer.from_pretrained(str(rearag_dir), trust_remote_code=True, local_files_only=True)
    archived_reward = scored_manifest_path.parent / "runtime_code/kgproweight/training/reward_function.py"
    bound(scored_manifest["source_bindings"]["code:kgproweight/training/reward_function.py"] | {"path": str(archived_reward)})
    old = archived_reward.read_text()
    old_helper = next(ast.get_source_segment(old, node) for node in ast.parse(old).body
                      if isinstance(node, ast.FunctionDef) and node.name == "source_gate_text_inputs_v1")
    assert old_helper.strip() == inspect.getsource(source_gate_text_inputs_v1).strip()
    selected, reasons, available = choose(candidates, generations)
    checks = []
    raw_calls = 0
    for row in selected:
        cid = row["candidate_id"]
        key = row["dataset"] + "::" + row["qid"]
        frozen, generation = input_index[key], generations[cid]
        assert row["generation"] == generation["generation"] == parent[cid]["generation"]
        assert row["raw_text"] == parent[cid]["raw_text"]
        assert row["input_sha256"] == generation["input_sha256"] == frozen["input_sha256"]
        parent_validated = {**parent[cid], "quality": heuristic_ratio_target(
            parent[cid]["raw_graph"], parent[cid]["raw_text"],
            m_graph=parent[cid]["features"]["m_graph"], trajectory_valid=parent[cid]["trajectory_valid"])}
        assert row["parent_validated_row_sha256"] == canonical_sha256(parent_validated)
        gold = labels[key]["metadata"]
        spec = RewardSpec(**deepcopy(frozen["spec"]), gold_answer=gold["gold_answer"],
                          gold_answer_aliases=gold.get("gold_answer_aliases") or [])
        original_spec_hash = canonical_sha256(spec.__dict__)
        validity = validate_source_gate_trajectory(spec, row["generation"], format_version="v2")
        assert validity["valid"] is row["trajectory_valid"]
        assert validity["contract_version"] == row["format_validation"]["contract_version"]
        prompts, texts = source_gate_text_inputs_v1(spec, validity["steps"])
        ids = generation["response_token_ids"]
        assert policy_tokenizer.decode(ids, skip_special_tokens=True) == row["generation"]
        raw = row["raw_text"]
        masked_m = row["features"]["m_graph"]
        artifact = gate.artifact
        logits = artifact["bias"] + sum(
            weight * (row["features"]["values"][name] - artifact["feature_standardization"]["mean"][name])
            / artifact["feature_standardization"]["scale"][name]
            for name, weight in zip(FEATURE_NAMES, artifact["weights"]))
        learned = 1. / (1. + math.exp(-max(-60., min(60., logits))))
        norm = artifact["normalization"]
        aliases = _canonical_gold_surfaces(spec.gold_answer, spec.gold_answer_aliases)
        answer = extract_final_answer(row["generation"]) or ""
        em = max(canonical_exact_match(answer, alias) for alias in aliases) if validity["valid"] else 0.
        f1 = max(canonical_token_f1(answer, alias) for alias in aliases) if validity["valid"] else 0.
        for arm, mode in (("A", "learned"), ("F", "fixed"), ("T", "text")):
            cache = CachedRealReaRAG(rearag_tokenizer, prompts, texts, raw)
            fn = KGProWeightRewardFunction(
                alpha_gate=Forbidden(), prm_annotator=Forbidden(), text_reward_model=cache,
                tokenizer=policy_tokenizer, outcome_weight=4., text_reward_scale=.3, max_steps=5,
                proofkg_process_reward=True, proofkg_process_version="v2_3", proofkg_process_weight=.2,
                proofkg_f1_weight=.1, proofkg_dynamic_validity=True, mixed_outcome_reward=True,
                mixed_text_reward=True, runtime_contract_version="v2", source_gated_reward_version="v1",
                source_gate_format_version="v2", source_gate_credit_version="v1", source_gate_mode=mode,
                source_quality_gate=gate, center_text_reward=False,
            )
            actual = fn(frozen["prompt"], row["generation"], spec, response_ids=ids)
            valid = validity["valid"]
            alpha = (0. if mode == "text" else norm["fixed_alpha"] if mode == "fixed" else learned) * masked_m if valid else 0.
            outcome = 4. * (em + .1 * f1) if valid else -4.
            clipped = [max(-1., min(1., (value - norm["text_center"]) / norm["text_scale"])) for value in raw] if valid else []
            step_weights = [.3 * (1. - alpha) * value / len(clipped) for value in clipped]
            text_reward = sum(step_weights)
            graph_reward = .2 * alpha * max(-1., min(1., (row["raw_graph"] - norm["graph_center"]) / norm["graph_scale"])) if valid else 0.
            total = outcome + text_reward + graph_reward
            expected_tokens = token_oracle(ids, policy_tokenizer, step_weights, outcome + graph_reward)
            tokens = actual["token_rewards"].cpu().double().numpy()
            assert np.isfinite(tokens).all() and len(tokens) == len(ids)
            assert np.max(np.abs(tokens - expected_tokens)) < 1e-6
            assert abs(tokens.sum() - total) < 1e-6
            assert actual["trajectory_valid"] is valid
            assert actual["source_gate"]["m_graph"] == masked_m
            assert actual["proofkg_process"]["required_steps"] == validity["required_steps"]
            assert abs(actual["source_gate"]["alpha_effective"] - alpha) < 1e-12
            for field, expected in (("outcome", outcome), ("text", text_reward), ("process", graph_reward), ("total", total)):
                assert abs(actual["mixed_reward"][field] - expected) < 1e-10
            assert abs(sum(actual["per_step_rewards"]) - total) < 1e-10
            assert cache.calls == int(valid)
            if valid:
                assert canonical_sha256(source_gate_text_budget_v1(cache, prompts, texts)) == canonical_sha256(row["text_token_budget"])
                assert actual["mixed_reward"]["text_raw_step_scores"] == raw
                if row["source_credit"]["parent_m_graph"]:
                    assert abs(actual["proofkg_process"]["process_score"] - row["raw_graph"]) < 1e-12
            if not masked_m:
                assert alpha == graph_reward == 0.
            assert canonical_sha256(spec.__dict__) == original_spec_hash
            raw_calls += cache.calls
            checks.append({"candidate_id": cid, "selection_reason": reasons[cid], "arm": arm,
                           "valid": valid, "required_steps": validity["required_steps"],
                           "n_steps": len(validity["steps"]), "response_tokens": len(ids),
                           "parent_m_graph": row["source_credit"]["parent_m_graph"], "m_graph": masked_m,
                           "source_status": row["source_credit"]["status"], "alpha": alpha,
                           "outcome": outcome, "text": text_reward, "graph": graph_reward,
                           "total": total, "token_sum": float(tokens.sum()),
                           "token_max_abs_error": float(np.max(np.abs(tokens - expected_tokens))),
                           "cached_real_scores_sha256": canonical_sha256(raw),
                           "text_inputs_sha256": canonical_sha256([prompts, texts]),
                           "generation_sha256": canonical_sha256(row["generation"]), "pass": True})
        print(json.dumps({"candidate_id": cid, "checked": len(checks), "status": "PASS"}), flush=True)
    assert not torch.cuda.is_initialized()
    report = {"schema_version": "source-credit-cached-runtime-check-v1",
              "experiment_id": "SOURCE-CREDIT-RUNTIME-CACHED-V1-LOCAL-SEED42-20260905",
              "status": "PASS_CPU_CACHED_REAL_REARAG_ZERO_UPDATE_NOT_GPU_PPO_PROBE",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": 42,
              "selected_candidates": len(selected), "arms": ["A", "F", "T"], "reward_checks": len(checks),
              "all_checks_passed": True, "selection_population_counts": available,
              "selection_status_counts": dict(Counter(r["source_credit"]["status"] for r in selected)),
              "valid_candidates": sum(r["trajectory_valid"] for r in selected),
              "cached_real_backend_calls": raw_calls, "gold_access": "bound_train_labels_for_outcome_only",
              "gold_written_to_targets_or_report": False, "raw_scores_recomputed": False,
              "backend": "cached-real-ReaRAG; exact frozen raw scores; prompt/step and tokenizer budgets checked",
              "gpu_used": False, "model_weights_loaded": False, "optimizer_updates": 0,
              "ppo_training_started": False, "gpu_a_probe_complete": False,
              "reader_inputs_repaired": False, "source_integrity_clearance": False,
              "scientific_boundary": "CPU reward routing/arithmetic/token allocation integration only; neither PPO utility nor GPU trainer/reference/KL correctness nor input-source repair is established.",
              "max_token_abs_error": max(r["token_max_abs_error"] for r in checks),
              "inputs": {name: identity(path) for name, path in {
                  "calibration_manifest": calibration / "manifest.json", "gate": gate_path,
                  "mask_manifest": gate.mask.manifest_path, "candidate_rows": candidate_path,
                  "scored_parent_manifest": scored_manifest_path, "scored_parent_rows": parent_path,
                  "input_manifest": input_dir / "manifest.json", "input_rows": inputs_path,
                  "generation_manifest": generation_dir / "manifest.json", "generations": generations_path,
                  "train_gold_source": silver_path}.items()},
              "actual_tokenizer_files": tokenizer_bindings,
              "model_identities_from_frozen_generation_contract": {name: input_manifest[name] for name in (
                  "base_model", "rearag_model", "policy_tokenizer")},
              "model_weight_verification_scope": "Inherited SHA bindings from verified generation/scoring; weights neither loaded nor rehashed in this CPU check.",
              "code_bindings": {str(path.relative_to(ROOT)): identity(path) for path in [
                  Path(__file__).resolve(), ROOT / "kgproweight/training/reward_function.py",
                  ROOT / "kgproweight/reward/source_credit_gate_v1.py", ROOT / "kgproweight/reward/source_integrity_v1.py",
                  ROOT / "kgproweight/data/parsers.py", ROOT / "kgproweight/reward/proofkg_process_v2_3.py"]}}
    with (output_dir / "checks.jsonl").open("x") as handle:
        for record in checks:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    with (output_dir / "report.json").open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/audits/source_credit_runtime_cached_v1_local_seed42")
    args = parser.parse_args()
    output_existed_before_call = args.output_dir.exists()
    try:
        result = run(args.output_dir)
    except Exception as exc:
        if not output_existed_before_call and args.output_dir.is_dir() and not (args.output_dir / "FAILED.json").exists():
            (args.output_dir / "FAILED.json").write_text(json.dumps({"status": "FAILED", "type": type(exc).__name__,
                "message": str(exc), "optimizer_updates": 0, "gpu_used": False}, indent=2) + "\n")
        raise
    print(json.dumps({key: result[key] for key in ("status", "selected_candidates", "reward_checks", "max_token_abs_error")}))
