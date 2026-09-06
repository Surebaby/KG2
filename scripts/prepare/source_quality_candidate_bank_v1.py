#!/usr/bin/env python
"""Append-only train-only SFT candidate preparation, CUDA sampling and scoring.

Preparation reads whitelisted train evidence, never exports outcome gold. GPU
stages read only the frozen gold-free bank. A scored bank is a calibration
input, not validation of PPO benefit or independent source reliability.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval.ppo_emf1_development_v1 import (
    ROOT, DATASETS, bind_base_model, bind_tokenizer, canonical_json, digest, file_sha, identity,
    input_hash, key, logical_path, make_renderer, read_rows, write_json, write_rows,
)
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.prepare.materialize_mixed_ppo_three_dataset_v4_proof800 import _ten_safe_passages
from kgproweight.data.prompts import build_rl_messages, build_sft_messages
from kgproweight.data.parsers import parse_steps
from kgproweight.reward.trajectory_source_gate import evaluate_graph_gate
from kgproweight.reward.proofkg_process_v2_3 import SCORER_VERSION, build_execution_trace_v2_3, score_proofkg_v2_3
from kgproweight.reward.source_quality_gate_v1 import FEATURE_VERSION, compute_gate_features
from kgproweight.training.reward_function import validate_source_gate_trajectory_v1, source_gate_text_inputs_v1, source_gate_text_budget_v1
from kgproweight.training.phase3_ppo import _rollout_eos_token_ids, _trim_response_v2, _response_is_length_capped_v2

VERSION = "source-quality-candidate-bank-v1"
PREPARE_VERSION = "source-quality-candidate-preparation-v1"
GENERATION_VERSION = "source-quality-candidate-generation-v1"
TEXT_CONTRACT = "rearag-passage-only-raw-tanh-nll-v1"
DEFAULT_DATA = Path("data/silver_data/mixed_ppo_three_dataset_v4_proof800_n3000_k4_seed42")
DEFAULT_LEDGER = Path("outputs/audits/mixed_ppo_v4_protected_identity_ledger_v2/protected_identities.question_only.jsonl")
DEFAULT_POLICY = Path("checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final")
FORBIDDEN = {"answer", "answers", "gold_answer", "gold_answers", "gold_answer_aliases", "golden_answers", "supporting_facts", "decomposition", "teacher_output", "target", "labels"}
CODE_FILES = ["scripts/train/calibrate_source_quality_gate_v1.py", "kgproweight/training/phase3_ppo.py", "kgproweight/training/reward_function.py", "kgproweight/reward/proofkg_process.py", "scripts/eval/ppo_emf1_development_v1.py", "scripts/prepare/source_quality_candidate_bank_v1.py", "kgproweight/data/prompts.py", "kgproweight/data/parsers.py", "kgproweight/reward/proofkg_process_v2_3.py", "kgproweight/reward/proofkg_process_v2_2.py", "kgproweight/reward/source_quality_gate_v1.py", "kgproweight/reward/trajectory_source_gate.py", "kgproweight/reward/text_reward_model.py"]


def binding(path, root=ROOT):
    info = identity(Path(path))
    return {**info, "path": logical_path(Path(path), root), "origin_path": info["path"]}


def resolve(bound, directory, root=ROOT):
    path = Path(bound["path"])
    for candidate in ([path] if path.is_absolute() else [directory / path, root / path]):
        if candidate.is_file() and file_sha(candidate) == bound["sha256"]:
            return candidate
    raise ValueError(f"bound file missing or hash mismatch: {path}")


def assert_gold_free(value):
    if isinstance(value, dict):
        if FORBIDDEN & set(value):
            raise ValueError("gold/target field in frozen generation/scoring input")
        for item in value.values():
            assert_gold_free(item)
    elif isinstance(value, list):
        for item in value:
            assert_gold_free(item)


@contextmanager
def stage(directory, experiment_id, stage_name):
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "started.json", {"experiment_id": experiment_id, "stage": stage_name,
               "started_at_utc": datetime.now(timezone.utc).isoformat(), "training_started": False})
    try:
        yield
    except BaseException as exc:
        write_json(directory / "FAILED.json", {"experiment_id": experiment_id, "status": "FAILED_NOT_TRAINED",
                   "exception_type": type(exc).__name__, "error": str(exc), "partial_outputs_retained": True})
        raise


def finish(directory, report, files):
    write_json(directory / "report.json", report)
    manifest = {**report, "outputs": {name: {**identity(directory / name), "path": name,
                  "origin_path": str((directory / name).resolve())} for name in [*files, "report.json"]}}
    write_json(directory / "manifest.json", manifest)
    return report


def load_release(directory, expected_schema):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != expected_schema or (directory / "FAILED.json").exists():
        raise ValueError("wrong or failed candidate release")
    for name, info in manifest["outputs"].items():
        if Path(name).name != name or file_sha(directory / name) != info["sha256"]:
            raise ValueError(f"candidate release artifact hash mismatch: {name}")
    return manifest


def row_identity(row):
    return {"dataset": row["dataset"], "qid": row["qid"], "question": row["question"],
            "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
            "family_sha256": family_sha256(row["question"]), "family_version": FAMILY_VERSION}


def isolation(rows, ledger):
    def ids(items):
        result = {name: set() for name in ("qid", "question_sha256", "family_sha256")}
        for row in items:
            key(row)
            item = row_identity(row)
            for name in result:
                result[name].add(item[name])
        return result
    lhs, rhs = ids(rows), ids(ledger)
    overlap = {name: len(lhs[name] & rhs[name]) for name in lhs}
    if any(overlap.values()):
        raise ValueError(f"protected identity isolation failed: {overlap}")
    return overlap


def bind_rearag(model, root):
    index_path = model / "pytorch_model.bin.index.json"
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    if not shards or any(Path(name).name != name for name in shards):
        raise ValueError("invalid ReaRAG shard index")
    files = sorted(set(shards + [index_path.name, "config.json", "tokenizer_config.json", "tokenizer.model"] +
                       [p.name for p in model.glob("*.py")] + [p.name for p in model.glob("generation_config.json")]))
    return {"path": logical_path(model, root), "full_weight_files_hashed": True,
            "files": {name: {**binding(model / name, root), "path": name} for name in files}}


def validate_model(directory, frozen):
    for name, info in frozen["files"].items():
        if Path(name).name != name or file_sha(directory / name) != info["sha256"]:
            raise ValueError(f"model/tokenizer hash mismatch: {name}")


def validate_code(manifest, root):
    for name, info in manifest["source_bindings"].items():
        if name.startswith("code:"):
            resolve(info, root, root)


def prepare_bank(*, output_dir, experiment_id, data_dir=ROOT / DEFAULT_DATA,
                 protected_ledger=ROOT / DEFAULT_LEDGER, policy=ROOT / DEFAULT_POLICY,
                 base_model=ROOT / "models/llama3-8b", rearag_model=ROOT / "models/rearag-9b",
                 controls_per_dataset=10, seed=42, project_root=ROOT):
    with stage(output_dir, experiment_id, "prepare"):
        if controls_per_dataset < 1:
            raise ValueError("ordinary controls per dataset must be positive")
        report = json.loads((data_dir / "report.json").read_text())
        if report.get("status") != "COMPLETE_DATA_NOT_TRAINED" or not report.get("gates") or not all(report["gates"].values()):
            raise ValueError("final v4 data release is not complete/pass")
        paths = {"silver": data_dir / "silver_train.jsonl", "questionkg": data_dir / "question_kg_records.jsonl",
                 "gate": data_dir / "source_gate_records.jsonl", "groups": data_dir / "prompt_groups.jsonl"}
        for name, report_key in (("silver", "silver_train"), ("questionkg", "question_kg_records"),
                                 ("gate", "source_gate_records"), ("groups", "prompt_groups")):
            if file_sha(paths[name]) != report["outputs"][report_key]["sha256"]:
                raise ValueError(f"final v4 source hash mismatch: {name}")
        source_hashes = {name: file_sha(path) for name, path in paths.items()}
        sources = {name: read_rows(path) for name, path in paths.items()}
        indexes = {name: {key(row): row for row in rows} for name, rows in sources.items()}
        if any(len(index) != len(sources[name]) for name, index in indexes.items()):
            raise ValueError("duplicate dataset/qid in source")
        if any(set(index) != set(indexes["groups"]) for index in indexes.values()):
            raise ValueError("source question identities differ")
        if len(indexes["groups"]) != 3000:
            raise ValueError("final v4 requires exactly 3000 prompt groups")
        eligible = [qid for qid, row in indexes["gate"].items() if row["m_graph"] == 1]
        if len(eligible) != 800:
            raise ValueError("final v4 requires all 800 eligible questions")
        selected = set(eligible)
        for dataset in DATASETS:
            pool = [qid for qid, row in indexes["gate"].items() if row["dataset"] == dataset and row["m_graph"] == 0]
            pool.sort(key=lambda qid: digest(["ordinary-controls-v1", seed, qid]))
            if len(pool) < controls_per_dataset:
                raise ValueError("insufficient ordinary controls")
            selected.update(pool[:controls_per_dataset])
        ledger = read_rows(protected_ledger)
        rows = []
        render = make_renderer(policy)
        for question_key in sorted(selected):
            group, silver, record, gate = (indexes[name][question_key] for name in ("groups", "silver", "questionkg", "gate"))
            expected = row_identity(group)
            for item in (silver, record):
                if row_identity(item) != expected:
                    raise ValueError("source identity join mismatch")
            if silver.get("metadata", {}).get("source_split") != "train" or group.get("evaluation_eligible") is not False:
                raise ValueError("candidate must be explicitly train-only")
            if not _ten_safe_passages(silver["retrieved_passages"]) or len(record["kg_subgraph"]) > 12:
                raise ValueError("exact ten safe passages and full graph <=12 required")
            decision = evaluate_graph_gate(record, dataset=group["dataset"], qid=group["qid"], question=group["question"],
                              historical_cutoff=str((record.get("provenance") or {}).get("historical_cutoff") or "")).to_dict()
            for field in ("m_graph", "kg_sha256", "execution_sha256"):
                if decision[field] != gate[field]:
                    raise ValueError("frozen source gate mismatch")
            passages = silver["retrieved_passages"]
            messages = build_rl_messages(group["question"], passages, record["kg_subgraph"], top_k=10, max_kg_triples=12)
            prompt, tokens = render(messages)
            if tokens > 6144:
                raise ValueError(f"token budget exceeded, refusing to drop/truncate question: {question_key} {tokens}")
            row = {**expected, "question_key": question_key, "source_split": "train", "m_graph": decision["m_graph"],
                   "retrieved_passages": passages, "kg_subgraph": record["kg_subgraph"], "source_quality_record": record,
                   "messages": messages, "prompt": prompt, "prompt_tokens": tokens,
                   "source_record_sha256": digest(record), "fullsource_record": record,
                   "spec": {"query": group["question"], "retrieved_passages": passages, "kg_subgraph": record["kg_subgraph"],
                            "metadata": {"dataset": group["dataset"], "qid": group["qid"], "source_quality_record": record}},
                   "source_bindings": {name: {"file_sha256": source_hashes[name], "row_sha256": digest(indexes[name][question_key])}
                                       for name in ("questionkg", "gate", "groups")}}
            assert_gold_free(row)
            row["input_sha256"] = input_hash(row)
            rows.append(row)
        overlaps = isolation(rows, ledger)
        bindings = {name: binding(path, project_root) for name, path in paths.items()}
        bindings.update({"data_report": binding(data_dir / "report.json", project_root), "protected_ledger": binding(protected_ledger, project_root),
                         "policy": binding(policy / "adapter_model.safetensors", project_root),
                         "policy_config": binding(policy / "adapter_config.json", project_root)})
        bindings.update({"code:" + name: binding(ROOT / name, project_root) for name in CODE_FILES})
        generation_config_path = base_model / "generation_config.json"
        config_path = base_model / "config.json"
        policy_stub = SimpleNamespace(generation_config=SimpleNamespace(**json.loads(generation_config_path.read_text())) if generation_config_path.exists() else None,
                                      config=SimpleNamespace(**json.loads(config_path.read_text())))
        effective_eos = list(_rollout_eos_token_ids(policy_stub, None))
        generation = {"eos_token_ids": effective_eos, "eos_contract": "production _rollout_eos_token_ids/_trim_response_v2; retain true first EOS; length cap excludes EOS stop",
                      "sampling_boundary": "same unwarped sampling distribution as PPO; batch1 vs PPO batch4; independent per-candidate seeds, no claim of tokenwise RNG identity",
                      "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 384,
                      "max_input_tokens": 6144, "candidates_per_question": 2, "batch_size": 1, "seed": seed,
                      "seed_contract": "sha256([candidate-seed-v1,seed,question_key,candidate_index]) first 8 hex", "dtype": "bfloat16"}
        scoring = {"text_score_contract": TEXT_CONTRACT, "graph_scorer_version": SCORER_VERSION, "feature_version": FEATURE_VERSION,
                   "rearag_max_tokens": 4096, "text_prompt": "SFT message contents joined by double newline; graph empty; top10 passages; prior step.raw_text joined by newline",
                   "text_aggregation": "raw per-step tanh((2.5-mean_step_NLL)/1.5); trajectory mean; no EMA",
                   "overflow": "fail entire attempt; never truncate/drop questions", "gold_access_for_gate_target": False,
                   "format_contract": "source-gate-runtime-v2-format-v1", "max_steps": 5, "ordinary_min_steps": 3, "min_reasoning_chars": 20}
        write_rows(output_dir / "inputs.jsonl", rows)
        write_json(output_dir / "score_config.json", scoring)
        bindings["score_config"] = binding(output_dir / "score_config.json", project_root)
        report = {"schema_version": PREPARE_VERSION, "status": "TRAIN_ONLY_INPUTS_FROZEN_NOT_GENERATED", "experiment_id": experiment_id,
                  "n_questions": len(rows), "n_candidates": len(rows) * 2, "by_dataset": dict(Counter(row["dataset"] for row in rows)),
                  "graph_eligible": len(eligible), "controls_per_dataset": controls_per_dataset, "seed": seed,
                  "qid_order": [key(row) for row in rows], "generation": generation, "scoring": scoring, "source_bindings": bindings,
                  "policy_path": logical_path(policy, project_root), "policy_tokenizer": bind_tokenizer(policy, project_root),
                  "base_model": bind_base_model(base_model, project_root),
                  "rearag_model": bind_rearag(rearag_model, project_root), "protected_overlap": overlaps,
                  "max_input_tokens_observed": max(row["prompt_tokens"] for row in rows), "gold_access_for_gate_target": False,
                  "training_started": False, "boundary": "Train-only heuristic source credit calibration; all finalv4 graph800 plus balanced ordinary controls. No evaluation gold or synthetic candidates."}
        return finish(output_dir, report, ["inputs.jsonl", "score_config.json"])


def candidate_seed(seed, question_key, index):
    return int(digest(["candidate-seed-v1", seed, question_key, index])[:8], 16)


def require_cuda(device):
    import torch
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; no CPU model fallback")
    return torch


def validate_inputs(directory, manifest):
    rows = read_rows(directory / "inputs.jsonl")
    if [key(row) for row in rows] != manifest["qid_order"]:
        raise ValueError("bank qid/order mismatch")
    for row in rows:
        assert_gold_free(row)
        if input_hash(row) != row["input_sha256"]:
            raise ValueError("bank model-input hash mismatch")
    return rows


def generate_bank(*, bank_dir, output_dir, experiment_id, base_model=None, policy=None, device="cuda:0", project_root=ROOT):
    with stage(output_dir, experiment_id, "generate"):
        bank = load_release(bank_dir, PREPARE_VERSION)
        validate_code(bank, project_root)
        rows = validate_inputs(bank_dir, bank)
        base_model = base_model or project_root / bank["base_model"]["path"]
        policy = policy or project_root / bank["policy_path"]
        validate_model(base_model, bank["base_model"])
        validate_model(policy, bank["policy_tokenizer"])
        for name, filename in (("policy", "adapter_model.safetensors"), ("policy_config", "adapter_config.json")):
            if file_sha(policy / filename) != bank["source_bindings"][name]["sha256"]:
                raise ValueError("frozen SFT adapter changed")
        torch = require_cuda(device)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(policy, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_model, local_files_only=True, torch_dtype=torch.bfloat16).to(device)
        model = PeftModel.from_pretrained(model, policy, local_files_only=True, is_trainable=False).eval()
        effective_eos = _rollout_eos_token_ids(model, tokenizer)
        if list(effective_eos) != bank["generation"]["eos_token_ids"]:
            raise ValueError("runtime model effective EOS differs from frozen production contract")
        n = 0
        with (output_dir / "generations.jsonl").open("x", encoding="utf-8") as handle:
            for row in rows:
                prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
                encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", truncation=False, return_attention_mask=True)
                count = encoded["input_ids"].shape[-1]
                if prompt != row["prompt"] or count != row["prompt_tokens"] or count > bank["generation"]["max_input_tokens"]:
                    raise ValueError("runtime prompt/token budget differs from frozen bank")
                for index in range(2):
                    seed = candidate_seed(bank["seed"], row["question_key"], index)
                    torch.manual_seed(seed)
                    with torch.inference_mode():
                        sequence = model.generate(input_ids=encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device),
                            do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=384,
                            pad_token_id=tokenizer.pad_token_id, eos_token_id=list(effective_eos))
                    raw_response = sequence[0, count:]
                    response = _trim_response_v2(raw_response, eos_token_ids=effective_eos, pad_token_id=tokenizer.pad_token_id, max_new_tokens=384)
                    ids = response.tolist()
                    length_capped = _response_is_length_capped_v2(response, max_new_tokens=384, eos_token_ids=effective_eos)
                    prediction = {"candidate_id": f"{row['question_key']}::k{index}", "dataset": row["dataset"], "qid": row["qid"],
                                  "candidate_index": index, "seed": seed, "input_sha256": row["input_sha256"],
                                  "bank_manifest_sha256": file_sha(bank_dir / "manifest.json"), "generation_contract_sha256": digest(bank["generation"]),
                                  "policy_sha256": bank["source_bindings"]["policy"]["sha256"], "base_model_identity_sha256": digest(bank["base_model"]),
                                  "generation": tokenizer.decode(ids, skip_special_tokens=True), "response_token_ids": ids,
                                  "raw_response_token_ids": raw_response.tolist(), "effective_eos_token_ids": list(effective_eos),
                                  "n_response_tokens": len(ids), "reached_max_new_tokens": length_capped}
                    handle.write(canonical_json(prediction) + "\n"); handle.flush()
                    n += 1
        return finish(output_dir, {"schema_version": GENERATION_VERSION, "status": "FROZEN_SFT_CANDIDATES_GENERATED_NOT_SCORED",
                      "experiment_id": experiment_id, "n_candidates": n, "bank_manifest_sha256": file_sha(bank_dir / "manifest.json"),
                      "generation_contract_sha256": digest(bank["generation"]), "training_started": False}, ["generations.jsonl"])


def score_candidate(row, prediction, scorer):
    """Real v2.3 and gate features plus uncentered passage-only ReaRAG steps."""
    assert_gold_free(row)
    generation, record = prediction["generation"], row["source_quality_record"]
    steps = parse_steps(generation, known_kg=row["kg_subgraph"])
    plan = record.get("query_plan") or {}
    proof = score_proofkg_v2_3(question=row["question"], generation=generation, kg_triples=row["kg_subgraph"],
               execution_trace=build_execution_trace_v2_3(plan, record.get("execution") or {}), planned_hops=len(plan.get("hops") or []))
    spec = SimpleNamespace(**row["spec"])
    validity = validate_source_gate_trajectory_v1(spec, generation)
    features = compute_gate_features(spec, validity["steps"], proof)
    format_validation = {name: validity[name] for name in ("valid", "violations", "all_step_count", "required_steps", "contract_version")}
    raw_text, budget = [], None
    if validity["valid"]:
        prompts, texts = source_gate_text_inputs_v1(spec, validity["steps"])
        budget = source_gate_text_budget_v1(SimpleNamespace(backend=scorer), prompts, texts)
        for prompt, text in zip(prompts, texts):
            value = float(scorer.score_step(prompt, text))
            if not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError("invalid raw ReaRAG score")
            raw_text.append(value)
    return {"schema_version": "source-quality-candidate-row-v1", **row_identity(row),
            "candidate_id": prediction["candidate_id"], "generation": generation,
            "retrieved_passages": row["retrieved_passages"], "source_quality_record": record, "fullsource_record": record,
            "proof_result": proof, "features": features, "raw_graph": float(proof["score"]), "raw_text": raw_text,
            "raw_text_step_mean": sum(raw_text) / len(raw_text) if raw_text else None,
            "source_bindings": row["source_bindings"], "input_sha256": row["input_sha256"],
            "generation_sha256": digest(prediction), "gold_access_for_gate_target": False,
            "trajectory_valid": validity["valid"], "format_validation": format_validation, "text_token_budget": budget}


def score_bank(*, bank_dir, generation_dir, output_dir, experiment_id, rearag_model=None, device="cuda:0", project_root=ROOT):
    with stage(output_dir, experiment_id, "score"):
        bank = load_release(bank_dir, PREPARE_VERSION)
        generated = load_release(generation_dir, GENERATION_VERSION)
        validate_code(bank, project_root)
        bank_sha = file_sha(bank_dir / "manifest.json")
        if generated["bank_manifest_sha256"] != bank_sha:
            raise ValueError("generation and input bank mismatch")
        inputs = validate_inputs(bank_dir, bank)
        predictions = read_rows(generation_dir / "generations.jsonl")
        expected = [(row, index) for row in inputs for index in range(2)]
        if len(predictions) != len(expected):
            raise ValueError("K2 candidate count mismatch")
        for pred, (row, index) in zip(predictions, expected):
            required = {"candidate_id": f"{row['question_key']}::k{index}", "dataset": row["dataset"], "qid": row["qid"],
                        "candidate_index": index, "seed": candidate_seed(bank["seed"], row["question_key"], index),
                        "input_sha256": row["input_sha256"], "bank_manifest_sha256": bank_sha,
                        "generation_contract_sha256": digest(bank["generation"]), "policy_sha256": bank["source_bindings"]["policy"]["sha256"],
                        "base_model_identity_sha256": digest(bank["base_model"])}
            if any(pred.get(name) != value for name, value in required.items()) or not isinstance(pred.get("generation"), str):
                raise ValueError("candidate qid/order/input/model/contract mismatch")
        ledger_path = resolve(bank["source_bindings"]["protected_ledger"], bank_dir, project_root)
        overlaps = isolation(inputs, read_rows(ledger_path))
        rearag_model = rearag_model or project_root / bank["rearag_model"]["path"]
        validate_model(rearag_model, bank["rearag_model"])
        require_cuda(device)
        from kgproweight.reward.text_reward_model import RearagPromptScorer
        scorer = RearagPromptScorer.from_pretrained(str(rearag_model), device=device, dtype="bf16")
        scorer.max_length = bank["scoring"]["rearag_max_tokens"]
        with (output_dir / "candidates.scored.jsonl").open("x", encoding="utf-8") as handle:
            for pred, (row, _) in zip(predictions, expected):
                handle.write(canonical_json(score_candidate(row, pred, scorer)) + "\n"); handle.flush()
        scored_sha = file_sha(output_dir / "candidates.scored.jsonl")
        proof = {"schema_version": "source-quality-bank-isolation-v1", "status": "PASS", "bank_sha256": scored_sha,
                 "family_version": FAMILY_VERSION, "protected_ledger_binding": bank["source_bindings"]["protected_ledger"], "overlap_counts": overlaps}
        write_json(output_dir / "isolation_proof.json", proof)
        bindings = {**bank["source_bindings"], "prepared_bank_manifest": binding(bank_dir / "manifest.json", project_root),
                    "generation_manifest": binding(generation_dir / "manifest.json", project_root)}
        for kind in ("base_model", "rearag_model", "policy_tokenizer"):
            for name, bound in bank[kind]["files"].items():
                bindings[kind + ":" + name] = {**bound, "path": str(Path(bank[kind]["path"]) / name)}
        report = {"schema_version": VERSION, "status": "TRAIN_ONLY_CANDIDATES_FROZEN", "experiment_id": experiment_id,
                  "bank_source": "real_frozen_policy_rollouts", "bank": {"path": "candidates.scored.jsonl", "sha256": scored_sha},
                  "feature_version": FEATURE_VERSION, "graph_scorer_version": SCORER_VERSION, "text_score_contract": TEXT_CONTRACT,
                  "gold_access_for_gate_target": False, "source_bindings": bindings,
                  "isolation_proof": {"path": "isolation_proof.json", "sha256": file_sha(output_dir / "isolation_proof.json")},
                  "n_questions": len(inputs), "n_candidates": len(predictions), "training_started": False,
                  "boundary": "Train-only heuristic alpha credit targets; family split assigned by calibrator; no PPO or EM/F1 result."}
        return finish(output_dir, report, ["candidates.scored.jsonl", "isolation_proof.json"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "generate", "score"):
        command = sub.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--experiment-id", required=True)
        command.add_argument("--project-root", type=Path, default=ROOT)
        if name == "prepare":
            command.add_argument("--data-dir", type=Path, default=ROOT / DEFAULT_DATA)
            command.add_argument("--protected-ledger", type=Path, default=ROOT / DEFAULT_LEDGER)
            command.add_argument("--controls-per-dataset", type=int, default=10)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--policy", type=Path, default=ROOT / DEFAULT_POLICY)
            command.add_argument("--base-model", type=Path, default=ROOT / "models/llama3-8b")
            command.add_argument("--rearag-model", type=Path, default=ROOT / "models/rearag-9b")
        else:
            command.add_argument("--bank-dir", type=Path, required=True)
            command.add_argument("--device", default="cuda:0")
            if name == "generate":
                command.add_argument("--base-model", type=Path)
                command.add_argument("--policy", type=Path)
            else:
                command.add_argument("--generation-dir", type=Path, required=True)
                command.add_argument("--rearag-model", type=Path)
    args = vars(parser.parse_args()); command = args.pop("command")
    result = {"prepare": prepare_bank, "generate": generate_bank, "score": score_bank}[command](**args)
    print(json.dumps({"status": result["status"], "experiment_id": result["experiment_id"]}))


if __name__ == "__main__":
    main()
