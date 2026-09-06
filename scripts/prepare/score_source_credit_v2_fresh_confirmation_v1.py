"""Freeze Gold-free process scores for 132 questions, K4 plus one greedy.

ReaRAG is evaluated once per valid candidate step and reused by both gate
variants and all A/F/T views. This program never opens a label table, fits a
gate, changes a calibration artifact, or grants PPO clearance. Per-step and
per-candidate files are immutable checkpoints for interrupted scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.prepare import source_quality_candidate_bank_v1 as bank
from scripts.prepare.score_sourcegate_candidates_format_v2 import score_candidate_v2
from kgproweight.reward.answer_format_objective_v2 import inspect_shortfall_salvage_v2
from kgproweight.reward.source_credit_gate_v2 import SourceCreditGateV2
from kgproweight.reward.source_reward_normalization_v2 import normalize_text_steps_v2
from kgproweight.training.reward_function import validate_source_gate_trajectory


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "source-credit-v2-fresh-process-row-v1"
REPORT_SCHEMA = "source-credit-v2-fresh-process-scoring-v1"
VARIANTS = ("norm_only", "features_v2")
ARMS = ("A", "F", "T")
SCORING_CODE_FILES = sorted(set(bank.CODE_FILES + [
    "scripts/prepare/score_source_credit_v2_fresh_confirmation_v1.py",
    "scripts/prepare/generate_source_credit_v2_fresh_confirmation_v1.py",
    "scripts/prepare/score_sourcegate_candidates_format_v2.py",
    "kgproweight/reward/answer_format_objective_v2.py",
    "kgproweight/reward/source_credit_gate_v1.py",
    "kgproweight/reward/source_credit_gate_v2.py",
    "kgproweight/reward/source_integrity_v1.py",
    "kgproweight/reward/source_reward_normalization_v2.py",
    "kgproweight/reward/source_trajectory_features_v2.py",
]))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def resolve(ref, base=ROOT):
    if not isinstance(ref, dict) or not ref.get("path") or not ref.get("sha256"):
        raise ValueError("required scoring source binding missing")
    path = Path(ref["path"])
    for candidate in ([path] if path.is_absolute() else [Path(base) / path, ROOT / path]):
        if candidate.is_file() and sha(candidate) == ref["sha256"]:
            if "bytes" in ref and candidate.stat().st_size != ref["bytes"]:
                raise ValueError("bound scoring source byte count mismatch")
            return candidate.resolve()
    raise ValueError("bound scoring source is missing or changed")


def write_text(path, value):
    """Exclusive atomic publication; an interrupted attempt remains traceable."""
    path = Path(path)
    temporary = path.with_name(path.name + ".attempt-" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path, value):
    write_text(path, canonical(value) + "\n")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def checked_score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError("raw ReaRAG score must be finite in [-1,1]")
    return float(value)


def process_terms(*, valid, raw_text, raw_graph, features, gate):
    """Apply frozen statistics; invalid candidates never call alpha.predict."""
    if not valid:
        if raw_text:
            raise ValueError("invalid trajectory cannot contain scored Text steps")
        return {arm: {"alpha_effective": 0., "text_component": 0., "graph_component": 0.,
                       "process": 0., "text_step_components": [], "rank_eligible": False}
                for arm in ARMS}
    if not raw_text or isinstance(raw_graph, bool) or not math.isfinite(raw_graph) or not 0 <= raw_graph <= .85 + 1e-12:
        raise ValueError("valid candidate lacks Text or has out-of-range Graph")
    norm = gate.normalization
    normalized = normalize_text_steps_v2(raw_text, norm["text_v2"])
    bounded_steps = normalized["bounded_step_scores"]
    mean_text = math.fsum(bounded_steps) / len(bounded_steps)
    learned = float(gate.predict(features))
    fixed = float(norm["fixed_alpha"])
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (learned, fixed)):
        raise ValueError("invalid frozen gate prediction")
    parent_graph = (features.get("source_credit_mask") or {}).get("parent_m_graph", 0)
    graph_z = (raw_graph - norm["graph_center"]) / norm["graph_scale"] if parent_graph else 0.
    graph_bounded = max(-1., min(1., graph_z))
    m_graph = features["m_graph"]
    terms = {}
    for arm, alpha in (("A", m_graph * learned), ("F", m_graph * fixed), ("T", 0.)):
        text_steps = [.3 * (1 - alpha) * value / len(bounded_steps) for value in bounded_steps]
        # Match the production weighted_reward reduction, including float order.
        text = sum(text_steps)
        graph = .2 * alpha * graph_bounded
        terms[arm] = {
            "alpha_effective": alpha, "text_component": text, "graph_component": graph,
            "process": text + graph, "text_step_components": text_steps, "rank_eligible": True,
            "text_normalized_mean": mean_text, "text_normalized_steps": bounded_steps,
            "graph_normalized": graph_bounded, "graph_normalized_unclipped": graph_z,
        }
    return terms


def score_one(row, prediction, scorer, gates, *, protocol_sha256):
    """Pure per-candidate stage; caller supplies a real scorer or a test double."""
    bank.assert_gold_free(row)
    if set(gates) != set(VARIANTS):
        raise ValueError("both frozen normalization-control and six-feature gates are required")
    # The old function computes a diagnostic raw Graph even for invalid text,
    # but never calls ReaRAG for invalid trajectories. No old CLI/main is used.
    scored = score_candidate_v2(row, prediction, scorer)
    spec = SimpleNamespace(**deepcopy(row["spec"]))
    validity = validate_source_gate_trajectory(spec, prediction["generation"], format_version="v2")
    if validity["valid"] is not scored["trajectory_valid"]:
        raise ValueError("shared format-v2 checker disagreement")
    valid = validity["valid"]
    features, variants = {}, {}
    proof_for_features = scored["proof_result"] if valid else {}
    for name in VARIANTS:
        gate = gates[name]
        original = gate.compute_features(spec, validity["steps"], proof_for_features)
        masked = gate.mask_features(spec, original)
        features[name] = {"original": original, "masked": masked}
        variants[name] = process_terms(valid=valid, raw_text=scored["raw_text"],
                                       raw_graph=scored["raw_graph"], features=masked, gate=gate)
    if features[VARIANTS[0]]["masked"]["m_graph"] != features[VARIANTS[1]]["masked"]["m_graph"]:
        raise ValueError("the two variants must use the same source-credit population")
    shortfall = None
    if not valid:
        contract = inspect_shortfall_salvage_v2(
            prediction["generation"], steps=validity["steps"],
            required_steps=validity["required_steps"], violations=validity["violations"],
            known_passage_ids=range(1, len(row["retrieved_passages"]) + 1),
        )
        shortfall = contract.telemetry()
    result = {
        "schema_version": SCHEMA, **{name: row[name] for name in (
            "question_key", "dataset", "qid", "question", "question_sha256", "family_sha256", "family_version", "input_sha256")},
        **{name: prediction[name] for name in ("candidate_id", "candidate_index", "generation_kind", "generation",
                                              "seed", "n_response_tokens", "reached_max_new_tokens")},
        "protocol_sha256": protocol_sha256, "generation_sha256": digest(prediction),
        "trajectory_valid": valid, "rank_eligible": valid,
        "format_validation": scored["format_validation"],
        "shortfall_salvage": shortfall,
        "raw_text": scored["raw_text"], "raw_text_step_mean": scored["raw_text_step_mean"],
        "raw_graph": scored["raw_graph"], "proof_result": scored["proof_result"],
        "raw_graph_invalid_is_diagnostic_only": not valid,
        "text_token_budget": scored["text_token_budget"], "features": features, "variants": variants,
        "source_record_sha256": row["source_record_sha256"],
        "gold_access": False, "outcome_in_process": False, "model_updates": 0,
    }
    result["process_row_sha256"] = digest(result)
    return result


def rank_questions(processes):
    """Rank only K4 sampled candidates; invalid/all-invalid never get selected."""
    grouped = defaultdict(list)
    for row in processes:
        grouped[row["question_key"]].append(row)
    result = []
    for key, rows in grouped.items():
        by_index = {row["candidate_index"]: row for row in rows}
        if len(rows) != 5 or set(by_index) != set(range(5)):
            raise ValueError("process ranking requires exactly K4 plus one greedy per question")
        if any(by_index[i]["generation_kind"] != ("sampled" if i < 4 else "greedy") for i in range(5)):
            raise ValueError("sampled/greedy kind differs from the frozen index contract")
        sampled = [by_index[index] for index in range(4)]
        eligible = [row for row in sampled if row["trajectory_valid"]]
        rankings = {}
        for variant in VARIANTS:
            rankings[variant] = {}
            for arm in ARMS:
                ordered = sorted(eligible, key=lambda row: (-row["variants"][variant][arm]["process"], row["candidate_index"]))
                if any(not math.isfinite(row["variants"][variant][arm]["process"]) for row in ordered):
                    raise ValueError("nonfinite process rank")
                rankings[variant][arm] = {
                    "selected_candidate_id": ordered[0]["candidate_id"] if ordered else None,
                    "ordered_eligible_candidate_ids": [row["candidate_id"] for row in ordered],
                    "ordered_process_scores": [row["variants"][variant][arm]["process"] for row in ordered],
                }
        result.append({"schema_version": "source-credit-v2-fresh-process-ranking-v1", "question_key": key,
                       "sampled_candidate_ids": [row["candidate_id"] for row in sampled],
                       "invalid_sampled_candidate_ids": [row["candidate_id"] for row in sampled if not row["trajectory_valid"]],
                       "greedy_candidate_id": by_index[4]["candidate_id"], "rankings": rankings,
                       "rank_tie_break": "candidate_index_ascending", "gold_access": False,
                       "greedy_in_ranking": False, "all_sampled_invalid": not eligible})
    return result


class StepCacheScorer:
    """Persist raw step scores once, binding the exact prompt/target/model run."""
    def __init__(self, backend, directory, *, candidate_id, generation_sha256, scoring_binding_sha256):
        self.backend = backend
        self.tokenizer, self.max_length = backend.tokenizer, backend.max_length
        self.directory = Path(directory)
        self.base = {"candidate_id": candidate_id, "generation_sha256": generation_sha256,
                     "scoring_binding_sha256": scoring_binding_sha256}
        self.index = 0
        self.new_calls = 0
        self.cache_hits = 0

    def score_step(self, prompt, text):
        index = self.index
        self.index += 1
        expected = {**self.base, "step_position": index, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
        path = self.directory / (digest([self.base["candidate_id"], index]) + ".json")
        if path.exists():
            cached = json.loads(path.read_text())
            if {key: cached.get(key) for key in expected} != expected or set(cached) != set(expected) | {"raw_score"}:
                raise ValueError("saved ReaRAG step binding changed on resume")
            self.cache_hits += 1
            return checked_score(cached["raw_score"])
        value = checked_score(self.backend.score_step(prompt, text))
        write_json(path, {**expected, "raw_score": value})
        self.new_calls += 1
        return value


def validate_scoring_protocol(p):
    config = p.get("scoring") or {}
    fixed = {"format_version": "v2", "max_steps": 5, "ordinary_min_steps": 3,
             "min_reasoning_chars": 20, "text_backend": "rearag", "dtype": "bf16",
             "max_text_length": 4096, "process_weights": {"text": .3, "graph": .2},
             "rank_samples": 4, "greedy_in_ranking": False, "rank_tie_break": "candidate_index_ascending"}
    if any(config.get(key) != value or type(config.get(key)) is not type(value) for key, value in fixed.items()):
        raise ValueError("scoring protocol differs from frozen format/normalization/ranking contract")
    if set(config.get("gates") or {}) != set(VARIANTS) or not config.get("rearag_model"):
        raise ValueError("scoring protocol lacks both confirmation gates or ReaRAG identity")
    if not set(SCORING_CODE_FILES).issubset(p.get("code_bindings") or {}):
        raise ValueError("scoring dependency code binding missing")
    return config


def run(*, protocol: Path, generation: Path, out: Path, resume=False):
    # Imports are local to keep pure CPU tests independent of GPU execution.
    from scripts.prepare import generate_source_credit_v2_fresh_confirmation_v1 as producer
    context = producer.verify_protocol(protocol, verify_models=False)
    p, inputs = context["protocol"], context["inputs"]
    scoring = validate_scoring_protocol(p)
    generation_manifest = generation / "manifest.json"
    release = json.loads(generation_manifest.read_text())
    if (release.get("schema_version") != "source-credit-v2-fresh-confirmation-generations-v1"
            or release.get("status") != "COMPLETE_GENERATED_NOT_SCORED"
            or release.get("protocol_sha256") != context["protocol_sha256"]):
        raise ValueError("generation release is incomplete or bound to another protocol")
    generated_path = resolve(release["outputs"]["generations.jsonl"], generation)
    predictions = read_rows(generated_path)
    from transformers import AutoTokenizer
    policy_tokenizer = AutoTokenizer.from_pretrained(context["policy_path"], local_files_only=True)
    if policy_tokenizer.pad_token_id is None:
        policy_tokenizer.pad_token_id = policy_tokenizer.eos_token_id
    producer.verify_generation_rows(predictions, context, tokenizer=policy_tokenizer)
    gates = {name: SourceCreditGateV2.load(resolve(scoring["gates"][name], protocol.parent), allow_unvalidated=True)
             for name in VARIANTS}
    if any(gate.artifact.get("training_clearance") is not False for gate in gates.values()):
        raise ValueError("confirmation scoring expects unchanged unconfirmed gate wrappers")
    if gates["norm_only"].mask.payload_sha256 != gates["features_v2"].mask.payload_sha256:
        raise ValueError("confirmation gates use different source-credit masks")
    for name, version in (("norm_only", "source-quality-trajectory-features-v1"),
                          ("features_v2", "source-quality-trajectory-features-v2")):
        if gates[name].artifact["feature_version"] != version:
            raise ValueError("normalization-control/six-feature gate assignments differ")
    model_info = scoring["rearag_model"]
    if model_info != context["input_manifest"]["models"]["rearag_model"]:
        raise ValueError("ReaRAG identity must inherit the original input manifest")
    model_path = Path(model_info["path"])
    if not model_path.is_absolute(): model_path = ROOT / model_path
    bank.validate_model(model_path, model_info)
    prepared = {"schema_version": REPORT_SCHEMA, "protocol": binding(protocol),
                "generation_manifest": binding(generation_manifest), "generations": binding(generated_path),
                "scoring_config_sha256": digest(scoring), "scoring_code": binding(__file__),
                "n_inputs": len(inputs), "n_candidates": len(predictions), "gold_access": False,
                "model_updates": 0, "gate_fitting": False, "ppo_launch_clearance": False}
    if out.exists() and not resume:
        raise FileExistsError("new scoring directory required; --resume must be explicit")
    if not out.exists():
        out.mkdir(parents=True)
        (out / "candidate_rows").mkdir()
        (out / "raw_steps").mkdir()
        write_json(out / "prepared.json", prepared)
    elif json.loads((out / "prepared.json").read_text()) != prepared:
        raise ValueError("resume scoring protocol/generation/model/code binding differs")
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "manifest.json").exists():
            manifest = json.loads((out / "manifest.json").read_text())
            if (manifest.get("schema_version") != REPORT_SCHEMA
                    or manifest.get("status") != "COMPLETE_GOLD_FREE_SCORED_NOT_ANALYZED"
                    or manifest.get("protocol_sha256") != context["protocol_sha256"]
                    or set(manifest.get("outputs") or {}) != {"processes.jsonl", "rankings.jsonl", "report.json", "prepared.json"}):
                raise ValueError("completed scoring seal differs from the frozen contract")
            for ref in manifest["outputs"].values(): resolve(ref, out)
            return json.loads((out / "report.json").read_text())
        input_index = {row["question_key"]: row for row in inputs}
        torch = bank.require_cuda("cuda:0")
        if not torch.cuda.is_bf16_supported(): raise RuntimeError("original BF16 ReaRAG is required")
        from kgproweight.reward.text_reward_model import RearagPromptScorer
        scorer = RearagPromptScorer.from_pretrained(str(model_path), device="cuda:0", dtype="bf16")
        scorer.max_length = 4096
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        processed, new_steps, reused_steps = [], 0, 0
        for index, prediction in enumerate(predictions):
            path = out / "candidate_rows" / f"{index:04d}.json"
            qkey = f"{prediction['dataset']}::{prediction['qid']}"
            if path.exists():
                value = json.loads(path.read_text())
                unsigned = {key: item for key, item in value.items() if key != "process_row_sha256"}
                if (value.get("process_row_sha256") != digest(unsigned)
                        or value.get("generation_sha256") != digest(prediction)
                        or value.get("protocol_sha256") != context["protocol_sha256"]
                        or value.get("input_sha256") != input_index[qkey]["input_sha256"]):
                    raise ValueError("saved process candidate differs from bound generation/input")
            else:
                cached = StepCacheScorer(scorer, out / "raw_steps", candidate_id=prediction["candidate_id"],
                                         generation_sha256=digest(prediction), scoring_binding_sha256=digest(prepared))
                value = score_one(input_index[qkey], prediction, cached, gates,
                                  protocol_sha256=context["protocol_sha256"])
                write_json(path, value)
                new_steps += cached.new_calls
                reused_steps += cached.cache_hits
            processed.append(value)
            if (index + 1) % 10 == 0 or index + 1 == len(predictions):
                print(canonical({"status": "SCORING_GOLD_FREE", "completed": index + 1,
                                 "expected": len(predictions), "new_step_calls": new_steps,
                                 "reused_step_calls": reused_steps,
                                 "seconds": round(time.monotonic() - started, 2)}), flush=True)
        gpu = {"name": torch.cuda.get_device_name(0), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "seconds": time.monotonic() - started}
        del scorer
        gc.collect()
        torch.cuda.empty_cache()
        producer.verify_protocol(protocol, verify_models=False)
        bank.validate_model(model_path, model_info)
        rankings = rank_questions(processed)
        # Completed JSONL files can exist after an interruption between their
        # creation and the manifest. Require exact bytes rather than overwrite.
        for name, values in (("processes.jsonl", processed), ("rankings.jsonl", rankings)):
            expected = "".join(canonical(row) + "\n" for row in values)
            path = out / name
            if path.exists():
                if path.read_text() != expected: raise ValueError("partial publication differs from frozen process records")
            else:
                write_text(path, expected)
        report = {"schema_version": REPORT_SCHEMA, "status": "COMPLETE_GOLD_FREE_SCORED_NOT_ANALYZED",
                  "experiment_id": p["experiment_id"] + "-SCORING", "protocol_sha256": context["protocol_sha256"],
                  "n_questions": len(rankings), "n_candidates": len(processed),
                  "valid_candidates": sum(row["trajectory_valid"] for row in processed),
                  "scored_text_steps": sum(len(row["raw_text"]) for row in processed),
                  "new_step_calls_this_attempt": new_steps, "reused_step_calls_this_attempt": reused_steps,
                  "valid_by_generation_kind": dict(Counter(row["generation_kind"] for row in processed if row["trajectory_valid"])),
                  "gpu": gpu, "gold_access": False, "gate_fitting": False, "model_updates": 0,
                  "ppo_launch_clearance": False, "prepared": binding(out / "prepared.json"),
                  "boundary": "Process-only rankings frozen before independent Gold analysis; no effect-size claim or training clearance."}
        if not (out / "report.json").exists(): write_json(out / "report.json", report)
        else:
            saved_report = json.loads((out / "report.json").read_text())
            attempt_fields = {"new_step_calls_this_attempt", "reused_step_calls_this_attempt", "gpu"}
            if {key: value for key, value in saved_report.items() if key not in attempt_fields} != {
                    key: value for key, value in report.items() if key not in attempt_fields}:
                raise ValueError("saved scoring report population/protocol differs")
            report = saved_report
        manifest = {"schema_version": REPORT_SCHEMA, "status": report["status"],
                    "experiment_id": report["experiment_id"],
                    "protocol_sha256": context["protocol_sha256"], "gold_access": False,
                    "model_updates": 0, "gate_fitting": False, "ppo_launch_clearance": False,
                    "outputs": {name: binding(out / name) for name in ("processes.jsonl", "rankings.jsonl", "report.json", "prepared.json")}}
        write_json(out / "manifest.json", manifest)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(canonical(run(**vars(args))))


if __name__ == "__main__":
    main()
