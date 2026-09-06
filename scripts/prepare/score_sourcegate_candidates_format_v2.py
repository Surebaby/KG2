"""Versioned format repair scoring of immutable parent SFT candidates.

This stage preserves the exact parent prompts/generations and applies the v2
training-format contract. Known source-label projection issues explicitly block
PPO clearance; the resulting fit is a heuristic diagnostic until a separate
source-repair release is validated. No Gold is read for scores or gate targets.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

from scripts.prepare import source_quality_candidate_bank_v1 as parent
from kgproweight.training.reward_function import (
    source_gate_format_contract_version, validate_source_gate_trajectory,
)

FORMAT_VERSION = "v2"
ALLOWED_PARENT_CODE_CHANGES = {
    "kgproweight/training/reward_function.py", "kgproweight/training/phase3_ppo.py",
    "scripts/train/calibrate_source_quality_gate_v1.py",
}
EXTRA_CODE = ["scripts/prepare/score_sourcegate_candidates_format_v2.py",
              "kgproweight/config/schemas.py", "scripts/train/phase3_ppo.py"]


def verify_parents(bank_dir, generation_dir, snapshot_dir):
    bank = parent.load_release(bank_dir, parent.PREPARE_VERSION)
    generated = parent.load_release(generation_dir, parent.GENERATION_VERSION)
    bank_sha = parent.file_sha(bank_dir / "manifest.json")
    if generated["bank_manifest_sha256"] != bank_sha:
        raise ValueError("generation parent mismatch")
    snapshot = json.loads((snapshot_dir / "manifest.json").read_text())
    if snapshot["parent_bank_manifest_sha256"] != bank_sha:
        raise ValueError("archived parent source identity mismatch")
    bindings = copy.deepcopy(bank["source_bindings"])
    changes = {}
    for name, old in bank["source_bindings"].items():
        if not name.startswith("code:"):
            parent.resolve(old, bank_dir)
            continue
        relative = name.removeprefix("code:")
        saved = snapshot_dir / relative
        if parent.file_sha(saved) != old["sha256"] or snapshot["source_files"][relative]["sha256"] != old["sha256"]:
            raise ValueError(f"parent source snapshot mismatch: {relative}")
        current = parent.binding(parent.ROOT / relative)
        if current["sha256"] != old["sha256"]:
            if relative not in ALLOWED_PARENT_CODE_CHANGES:
                raise ValueError(f"unexpected scientific source mutation: {relative}")
            changes[relative] = {"parent_sha256": old["sha256"], "current_sha256": current["sha256"]}
        bindings[name] = current
    for relative in EXTRA_CODE:
        bindings["code:" + relative] = parent.binding(parent.ROOT / relative)
    inputs = parent.validate_inputs(bank_dir, bank)
    predictions = parent.read_rows(generation_dir / "generations.jsonl")
    if len(predictions) != len(inputs) * 2 or generated["n_candidates"] != len(predictions):
        raise ValueError("full parent K2 population required")
    for n, pred in enumerate(predictions):
        row, index = inputs[n // 2], n % 2
        expected = {"candidate_id": f"{row['question_key']}::k{index}", "dataset": row["dataset"], "qid": row["qid"],
                    "candidate_index": index, "seed": parent.candidate_seed(bank["seed"], row["question_key"], index),
                    "input_sha256": row["input_sha256"], "bank_manifest_sha256": bank_sha,
                    "generation_contract_sha256": parent.digest(bank["generation"]),
                    "policy_sha256": bank["source_bindings"]["policy"]["sha256"],
                    "base_model_identity_sha256": parent.digest(bank["base_model"])}
        if any(pred.get(k) != v for k, v in expected.items()) or not isinstance(pred.get("generation"), str):
            raise ValueError("parent candidate identity/seed/input/model mismatch")
    overlaps = parent.isolation(inputs, parent.read_rows(parent.resolve(bank["source_bindings"]["protected_ledger"], bank_dir)))
    return bank, inputs, predictions, bindings, changes, overlaps


def score_candidate_v2(row, prediction, scorer):
    parent.assert_gold_free(row)
    record, generation = row["source_quality_record"], prediction["generation"]
    plan = record.get("query_plan") or {}
    proof = parent.score_proofkg_v2_3(question=row["question"], generation=generation, kg_triples=row["kg_subgraph"],
            execution_trace=parent.build_execution_trace_v2_3(plan, record.get("execution") or {}),
            planned_hops=len(plan.get("hops") or []))
    spec = SimpleNamespace(**row["spec"])
    validity = validate_source_gate_trajectory(spec, generation, format_version=FORMAT_VERSION)
    features = parent.compute_gate_features(spec, validity["steps"], proof)
    scores, budget = [], None
    if validity["valid"]:
        prompts, texts = parent.source_gate_text_inputs_v1(spec, validity["steps"])
        budget = parent.source_gate_text_budget_v1(SimpleNamespace(backend=scorer), prompts, texts)
        for prompt, text in zip(prompts, texts):
            value = float(scorer.score_step(prompt, text))
            if not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError("invalid original BF16 ReaRAG score")
            scores.append(value)
    return {"schema_version": "source-quality-candidate-row-v1", **parent.row_identity(row),
            "candidate_id": prediction["candidate_id"], "generation": generation,
            "retrieved_passages": row["retrieved_passages"], "source_quality_record": record, "fullsource_record": record,
            "proof_result": proof, "features": features, "raw_graph": float(proof["score"]), "raw_text": scores,
            "raw_text_step_mean": sum(scores) / len(scores) if scores else None,
            "source_bindings": row["source_bindings"], "input_sha256": row["input_sha256"],
            "generation_sha256": parent.digest(prediction), "gold_access_for_gate_target": False,
            "trajectory_valid": validity["valid"],
            "format_validation": {k: validity[k] for k in ("valid", "violations", "all_step_count", "required_steps", "contract_version")},
            "text_token_budget": budget}


def progress(output_dir, status, **details):
    row = {"time_utc": datetime.now(timezone.utc).isoformat(), "status": status, "policy_optimizer_updates": 0, **details}
    encoded = parent.canonical_json(row) + "\n"
    with (output_dir / "events.jsonl").open("a") as handle:
        handle.write(encoded)
    tmp = output_dir / "status.json.tmp"
    tmp.write_text(encoded)
    tmp.replace(output_dir / "status.json")
    print(encoded.strip(), flush=True)


def score(*, bank_dir, generation_dir, snapshot_dir, output_dir, calibration_dir, experiment_id, device="cuda:0"):
    if calibration_dir.exists():
        raise FileExistsError("refusing to overwrite calibration")
    with parent.stage(output_dir, experiment_id, "format_v2_real_scoring"):
        try:
            progress(output_dir, "VERIFYING_PARENTS")
            bank, inputs, predictions, bindings, changes, overlaps = verify_parents(bank_dir, generation_dir, snapshot_dir)
            for name, info in list(bindings.items()):
                if name.startswith("code:"):
                    relative = name.removeprefix("code:")
                    dest = output_dir / "runtime_code" / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes((parent.ROOT / relative).read_bytes())
                    if parent.file_sha(dest) != info["sha256"]:
                        raise ValueError("code changed during freeze")
            revision = {"experiment_id": experiment_id, "format_contract_version": source_gate_format_contract_version(FORMAT_VERSION),
                        "parent_input_manifest": parent.identity(bank_dir / "manifest.json"),
                        "parent_generation_manifest": parent.identity(generation_dir / "manifest.json"),
                        "parent_source_snapshot": parent.identity(snapshot_dir / "manifest.json"), "code_changes": changes,
                        "generation_reused_without_edit": True, "candidates": len(predictions),
                        "generation_sampling_model_inputs_unchanged": True, "features_ratio_split_reward_weights_unchanged": True,
                        "source_integrity_clearance": False, "source_integrity_status": "LABEL_PROJECTION_REPAIR_PENDING",
                        "boundary": "Format repair only; existing source-label issues require independent successor source validation. Fit cannot authorize PPO."}
            parent.write_json(output_dir / "revision.json", revision)
            scoring_config = copy.deepcopy(bank["scoring"])
            scoring_config["format_contract"] = source_gate_format_contract_version(FORMAT_VERSION)
            parent.write_json(output_dir / "score_config.json", scoring_config)
            bindings["parent_score_config"] = bindings["score_config"]
            bindings["score_config"] = parent.binding(output_dir / "score_config.json")
            model_path = parent.ROOT / bank["rearag_model"]["path"]
            progress(output_dir, "VERIFYING_REARAG_MODEL")
            parent.validate_model(model_path, bank["rearag_model"])
            torch = parent.require_cuda(device)
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("BF16 required; no quantized or CPU model fallback")
            from kgproweight.reward.text_reward_model import RearagPromptScorer
            progress(output_dir, "LOADING_REARAG", gpu=torch.cuda.get_device_name(0))
            scorer = RearagPromptScorer.from_pretrained(str(model_path), device=device, dtype="bf16")
            scorer.max_length = 4096
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            valid = step_count = 0
            with (output_dir / "candidates.scored.jsonl").open("x") as handle:
                for n, pred in enumerate(predictions):
                    scored = score_candidate_v2(inputs[n // 2], pred, scorer)
                    handle.write(parent.canonical_json(scored) + "\n")
                    handle.flush()
                    valid += int(scored["trajectory_valid"])
                    step_count += len(scored["raw_text"])
                    if (n + 1) % 20 == 0 or n + 1 == len(predictions):
                        progress(output_dir, "SCORING", completed=n + 1, expected=len(predictions), valid=valid,
                                 text_steps=step_count, elapsed_seconds=round(time.monotonic() - started, 1),
                                 peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
            parent.validate_code({"source_bindings": bindings}, parent.ROOT)
            gpu = {"name": torch.cuda.get_device_name(0), "torch": torch.__version__,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                   "seconds": time.monotonic() - started, "pid": os.getpid()}
            del scorer
            gc.collect()
            torch.cuda.empty_cache()
            scored_sha = parent.file_sha(output_dir / "candidates.scored.jsonl")
            isolation = {"schema_version": "source-quality-bank-isolation-v1", "status": "PASS", "bank_sha256": scored_sha,
                         "family_version": parent.FAMILY_VERSION, "protected_ledger_binding": bank["source_bindings"]["protected_ledger"], "overlap_counts": overlaps}
            parent.write_json(output_dir / "isolation_proof.json", isolation)
            bindings.update({"prepared_bank_manifest": parent.binding(bank_dir / "manifest.json"),
                             "generation_manifest": parent.binding(generation_dir / "manifest.json"),
                             "format_revision": parent.binding(output_dir / "revision.json")})
            for kind in ("base_model", "rearag_model", "policy_tokenizer"):
                for name, bound in bank[kind]["files"].items():
                    bindings[kind + ":" + name] = {**bound, "path": str(Path(bank[kind]["path"]) / name)}
            report = {"schema_version": parent.VERSION, "status": "TRAIN_ONLY_CANDIDATES_FROZEN", "experiment_id": experiment_id,
                      "bank_source": "real_frozen_policy_rollouts", "bank": {"path": "candidates.scored.jsonl", "sha256": scored_sha},
                      "feature_version": parent.FEATURE_VERSION, "graph_scorer_version": parent.SCORER_VERSION,
                      "text_score_contract": parent.TEXT_CONTRACT, "format_contract_version": source_gate_format_contract_version(FORMAT_VERSION),
                      "gold_access_for_gate_target": False, "source_bindings": bindings,
                      "isolation_proof": {"path": "isolation_proof.json", "sha256": parent.file_sha(output_dir / "isolation_proof.json")},
                      "n_questions": len(inputs), "n_candidates": len(predictions), "valid_candidates": valid, "scored_text_steps": step_count,
                      "source_integrity_clearance": False, "source_integrity_status": "LABEL_PROJECTION_REPAIR_PENDING",
                      "training_started": False, "gpu": gpu, "boundary": revision["boundary"]}
            parent.finish(output_dir, report, ["candidates.scored.jsonl", "isolation_proof.json", "revision.json", "score_config.json"])
            progress(output_dir, "SCORING_COMPLETE_CALIBRATING", completed=len(predictions), expected=len(predictions))
            from scripts.train.calibrate_source_quality_gate_v1 import calibrate
            fitted = calibrate(output_dir / "manifest.json", output_dir / "isolation_proof.json", calibration_dir,
                               experiment_id=experiment_id + "-HEURISTIC-FIT", seed=42)
            progress(output_dir, "COMPLETE_SOURCE_REPAIR_AND_PROCESS_UTILITY_PENDING", completed=len(predictions), expected=len(predictions),
                     calibration_status=fitted["status"], heuristic_calibration_clearance=fitted["training_clearance"],
                     source_integrity_clearance=False, ppo_started=False, calibration_dir=str(calibration_dir))
            return fitted
        except BaseException as exc:
            progress(output_dir, "FAILED", exception_type=type(exc).__name__, error=str(exc))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    for name in ("bank-dir", "generation-dir", "snapshot-dir", "output-dir", "calibration-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    score(**vars(parser.parse_args()))
