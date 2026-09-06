"""Frozen cached-component comparison; no generation, optimization or promotion.

Run ``freeze`` first, then the resulting audit.executed.py with ``run``.  The
original format validator is loaded from the immutable pre-revision snapshot.
Gold-free structural decisions are written before train labels are joined.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from scripts.prepare import source_quality_candidate_bank_v1 as bank


ROOT = Path(os.environ.get("KGPW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
OUTPUT = ROOT / "outputs/audits/answer_format_objective_v2_cached_20260906_v1"
GENERATION = ROOT / "outputs/audits/training_format_prompt_v1_paired_probe_20260906_v1"
ASSESSMENT = ROOT / "outputs/audits/training_format_prompt_v1_assessment_20260906_v1"
LEGACY_REWARD = ROOT / "outputs/audits/normalization_representative_bank_v1_seed42_20260906_r2/runtime_code/kgproweight/training/reward_function.py"
EXPERIMENT_ID = "ANSWER-FORMAT-OBJECTIVE-V2-CACHED-20260906-V1"
ARMS = ("legacy", "prompt_v1")


def verify(info):
    if bank.file_sha(Path(info["path"])) != info["sha256"]:
        raise ValueError("immutable artifact changed: " + info["path"])


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def freeze(output):
    output.mkdir(parents=True, exist_ok=False)
    old_protocol = json.loads((ASSESSMENT / "protocol.json").read_text())
    expected = old_protocol["frozen_metric_code"]["kgproweight/training/reward_function.py"]["sha256"]
    if bank.file_sha(LEGACY_REWARD) != expected:
        raise ValueError("legacy validator is not the previously used source")
    snapshots = {}
    for source, name in (
        (Path(__file__), "audit.executed.py"),
        (ROOT / "kgproweight/reward/answer_format_objective_v2.py", "objective.frozen.py"),
        (LEGACY_REWARD, "legacy_reward_function.frozen.py"),
    ):
        (output / name).write_bytes(source.read_bytes())
        snapshots[name] = bank.identity(output / name)
    support = (
        "kgproweight/data/parsers.py", "kgproweight/eval/pred_processing.py",
        "kgproweight/reward/proofkg_process.py", "kgproweight/reward/source_quality_gate_v1.py",
        "kgproweight/reward/trajectory_source_gate.py", "kgproweight/data/prompts.py",
        "scripts/prepare/source_quality_candidate_bank_v1.py", "scripts/eval/ppo_emf1_development_v1.py",
    )
    protocol = {
        "schema_version": "answer-format-objective-v2-cached-protocol-v1",
        "experiment_id": EXPERIMENT_ID,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "already consumed fixed60/K2, legacy and rejected prompt-v1 banks, 240 cached responses",
        "primary_variable": "shortfall-only answer/format objective; generation, input, validator and metrics unchanged",
        "objective": {
            "version": "answer-format-objective-v2", "valid": "unchanged 4*(max_alias_EM+.1*max_alias_F1)",
            "salvage": "exactly2 complete ordered steps, required3, sole old minimum/content violation; unique nonempty terminal Final; Reasoning>=20, all fields once, nonempty Conclusion; no duplicate normalized Reasoning, malformed/unknown KG, out-of-range explicit passage IDs, extra/empty/malformed step header",
            "salvaged_component": "4*(max_alias_PPO_firstline_EM+.1*max_alias_PPO_firstline_F1)-1",
            "other_invalid_component": -4,
            "invalid_process": 0,
            "broad_final_only_proposal": "REJECTED_BEFORE_CACHED_EXECUTION",
        },
        "metrics": {
            "raw_extraction": "frozen double extract_kg_proweight_answer",
            "outcome_extraction": "frozen extract_final_answer then first line; salvage guard must agree",
            "normalization": "frozen canonical_exact_match/canonical_token_f1, max aliases independently",
            "denominator": "all120 per prompt bank; no filtering and no best-of-K",
            "aggregation": "K2 mean per question, equal-question dataset mean, equal3-dataset macro; equal quotas make this equal row means",
        },
        "components_excluded": ["ReaRAG", "Graph", "KL", "loss", "PPO updates"],
        "gold": old_protocol["gold"]["binding"],
        "gold_read_after": "this code/protocol freeze and complete immutable-bank verification and structure_only/before_gold artifacts",
        "source_bindings": {str(path): bank.identity(path) for path in (
            ASSESSMENT / "manifest.json", ASSESSMENT / "candidate_metrics.jsonl", ASSESSMENT / "protocol.json",
            GENERATION / "manifest.json", GENERATION / "protocol.json", GENERATION / "legacy_inputs.jsonl",
            GENERATION / "inputs.jsonl", GENERATION / "baseline_384.jsonl", GENERATION / "generations_prompt_v1.jsonl",
        )},
        "code_snapshots": snapshots,
        "supporting_code": {name: bank.identity(ROOT / name) for name in support},
        "automatic_promotion": False, "ppo_launch_clearance": False, "fresh_confirmation_consumed": False,
    }
    bank.write_json(output / "protocol.json", protocol)
    bank.write_json(output / "prepared.json", {
        "protocol": bank.identity(output / "protocol.json"),
        "status": "FROZEN_BEFORE_CACHED_EXECUTION_AND_GOLD_JOIN",
        "gold_parsed_in_this_experiment": False,
    })
    print(json.dumps({"status": "FROZEN", "experiment_id": EXPERIMENT_ID}))


def summaries(records):
    if not records:
        return {}
    values = defaultdict(list)
    for row in records:
        for name in ("em", "f1", "ppo_raw_em", "ppo_raw_f1", "legacy_component", "v2_component", "component_delta"):
            values[name].append(row[name])
    return {
        "candidates": len(records), "questions": len({r["question_key"] for r in records}),
        "format_valid": sum(r["format_valid"] for r in records),
        "format_valid_fraction": sum(r["format_valid"] for r in records) / len(records),
        "salvaged": sum(r["salvage"]["shortfall_salvage_eligible"] for r in records),
        "salvaged_correct_EM": sum(r["salvage"]["shortfall_salvage_eligible"] and r["ppo_raw_em"] == 1 for r in records),
        "legacy_valid_components_unchanged": all(r["component_delta"] == 0 for r in records if r["format_valid"]),
        "changed_components": sum(r["component_delta"] != 0 for r in records),
        "invalid_reasons": dict(Counter(r["salvage"]["shortfall_salvage_reason"] for r in records if not r["format_valid"])),
        "old_invalid_violations": dict(Counter(v for r in records if not r["format_valid"] for v in r["violations"])),
        "means": {name: math.fsum(items) / len(items) for name, items in values.items()},
    }


def run(output):
    if any((output / name).exists() for name in ("before_gold.json", "structure_only.jsonl", "manifest.json")):
        raise FileExistsError("refusing to overwrite a cached audit")
    prepared = json.loads((output / "prepared.json").read_text())
    verify(prepared["protocol"])
    protocol = json.loads((output / "protocol.json").read_text())
    all_bindings = [*protocol["source_bindings"].values(), *protocol["code_snapshots"].values(), *protocol["supporting_code"].values()]
    for info in all_bindings:
        verify(info)
    if bank.file_sha(Path(__file__)) != protocol["code_snapshots"]["audit.executed.py"]["sha256"]:
        raise ValueError("run must use the frozen audit source")
    bank.load_release(ASSESSMENT, "training-format-prompt-v1-independent-assessment-report")
    bank.load_release(GENERATION, "training-format-prompt-v1-paired-development-probe-report")
    objective = load_module(output / "objective.frozen.py", "cached_answer_format_objective_v2")
    legacy = load_module(output / "legacy_reward_function.frozen.py", "cached_legacy_reward_function_v1")
    from kgproweight.data.parsers import extract_final_answer
    from kgproweight.eval.pred_processing import extract_kg_proweight_answer
    from kgproweight.reward.proofkg_process import canonical_exact_match, canonical_token_f1

    inputs = {"legacy": bank.read_rows(GENERATION / "legacy_inputs.jsonl"), "prompt_v1": bank.read_rows(GENERATION / "inputs.jsonl")}
    wrappers = bank.read_rows(GENERATION / "baseline_384.jsonl")
    predictions = {"legacy": [row["prediction"] for row in wrappers], "prompt_v1": bank.read_rows(GENERATION / "generations_prompt_v1.jsonl")}
    if any(len(inputs[arm]) != 60 or len(predictions[arm]) != 120 for arm in ARMS):
        raise ValueError("fixed60/K2 complete cohort required")
    states, structures = [], []
    for arm in ARMS:
        seen = set()
        for index, prediction in enumerate(predictions[arm]):
            row = inputs[arm][index // 2]
            bank.assert_gold_free(row)
            cid = f"{row['question_key']}::k{index % 2}"
            if (cid in seen or prediction["candidate_id"] != cid or prediction["input_sha256"] != row["input_sha256"]
                    or bank.input_hash(row) != row["input_sha256"]):
                raise ValueError("candidate/input identity mismatch")
            seen.add(cid)
            if arm == "legacy" and wrappers[index]["prediction_sha256"] != bank.digest(prediction):
                raise ValueError("legacy prediction wrapper hash changed")
            validation = legacy.validate_source_gate_trajectory(SimpleNamespace(**row["spec"]), prediction["generation"], format_version="v2")
            salvage = objective.inspect_shortfall_salvage_v2(
                prediction["generation"], steps=validation["steps"], required_steps=validation["required_steps"],
                violations=validation["violations"], known_passage_ids=range(1, min(10, len(row["spec"]["retrieved_passages"])) + 1),
            )
            structure = {
                "arm": arm, "candidate_id": cid, "question_key": row["question_key"], "dataset": row["dataset"],
                "family_sha256": row["family_sha256"], "candidate_index": index % 2,
                "prediction_sha256": bank.digest(prediction), "input_sha256": row["input_sha256"],
                "format_valid": validation["valid"], "violations": validation["violations"],
                "step_count": validation["all_step_count"], "required_steps": validation["required_steps"],
                "salvage": salvage.telemetry(),
            }
            structures.append(structure)
            states.append((row, prediction, validation, salvage))
        if Counter(row["dataset"] for row in inputs[arm]) != {"hotpotqa": 20, "2wikimultihopqa": 20, "musique": 20}:
            raise ValueError("fixed dataset quotas changed")
    bank.write_rows(output / "structure_only.jsonl", structures)
    bank.write_json(output / "before_gold.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "gold_parsed_in_this_experiment": False,
        "structure": bank.identity(output / "structure_only.jsonl"), "protocol": prepared["protocol"],
        "candidates": 240, "old_validator_loaded_from_immutable_snapshot": True,
    })
    verify(protocol["gold"])
    label_rows = bank.read_rows(Path(protocol["gold"]["path"]))
    labels = {bank.key(row): row for row in label_rows}
    if len(labels) != len(label_rows):
        raise ValueError("duplicate frozen train labels")
    old_metrics = {row["candidate_id"]: row for row in bank.read_rows(ASSESSMENT / "candidate_metrics.jsonl")}
    if len(old_metrics) != 120:
        raise ValueError("old assessment cohort mismatch")
    results = []
    for structure, (row, prediction, validation, salvage) in zip(structures, states):
        label = labels[row["question_key"]]
        if label["question"] != row["question"] or label["metadata"]["source_split"] != "train":
            raise ValueError("train-only exact question label join failed")
        surfaces = legacy._canonical_gold_surfaces(label["metadata"]["gold_answer"], label["metadata"].get("gold_answer_aliases"))
        if not surfaces:
            raise ValueError("empty canonical labels")
        generation = prediction["generation"]
        answer = extract_kg_proweight_answer(extract_kg_proweight_answer(generation))
        ppo_answer = (extract_final_answer(generation) or "").split("\n", 1)[0].strip()
        metric = lambda func, text: max(func(text, surface) for surface in surfaces)
        em, f1 = metric(canonical_exact_match, answer), metric(canonical_token_f1, answer)
        ppo_em, ppo_f1 = metric(canonical_exact_match, ppo_answer), metric(canonical_token_f1, ppo_answer)
        old = old_metrics[structure["candidate_id"]][structure["arm"]]
        if (old["valid"] != validation["valid"] or old["violations"] != validation["violations"]
                or old["prediction_sha256"] != structure["prediction_sha256"]
                or old["steps"] != validation["all_step_count"] or old["required_steps"] != validation["required_steps"]):
            raise ValueError("legacy structure differs from original assessment")
        expected = {"em": em, "f1": f1, "ppo_em": ppo_em if validation["valid"] else 0.0, "ppo_f1": ppo_f1 if validation["valid"] else 0.0}
        if any(abs(old[name] - value) > 1e-12 for name, value in expected.items()):
            raise ValueError("legacy answer score differs from original assessment")
        if salvage.eligible and salvage.answer != ppo_answer:
            raise ValueError("salvage changed the PPO answer extractor")
        original = 4.0 * (ppo_em + 0.1 * ppo_f1) if validation["valid"] else -4.0
        revised = objective.compose_answer_format_objective_v2(
            trajectory_valid=bool(validation["valid"]), salvage_contract=salvage,
            outcome_em=ppo_em if validation["valid"] or salvage.eligible else 0.0,
            outcome_f1=ppo_f1 if validation["valid"] or salvage.eligible else 0.0,
        )
        if validation["valid"] and revised.outcome_component != original:
            raise ValueError("valid outcome component changed")
        results.append({**structure, "em": em, "f1": f1, "ppo_raw_em": ppo_em, "ppo_raw_f1": ppo_f1,
                        "legacy_component": original, "v2_component": revised.outcome_component,
                        "component_delta": revised.outcome_component - original, "objective": revised.telemetry()})
    bank.write_rows(output / "candidate_metrics.jsonl", results)
    report = {
        "schema_version": "answer-format-objective-v2-cached-report-v1", "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_CACHED_COMPONENT_DIAGNOSTIC_NOT_TRAINED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "candidates": 240,
        "by_prompt": {arm: summaries([row for row in results if row["arm"] == arm]) for arm in ARMS},
        "by_prompt_dataset": {arm: {dataset: summaries([row for row in results if row["arm"] == arm and row["dataset"] == dataset])
            for dataset in ("hotpotqa", "2wikimultihopqa", "musique")} for arm in ARMS},
        "legacy_structure_and_metrics_reproduced": 240,
        "all_valid_components_unchanged": all(r["component_delta"] == 0 for r in results if r["format_valid"]),
        "all_unsalvaged_invalid_still_minus_four": all(r["v2_component"] == -4 for r in results if not r["format_valid"] and not r["salvage"]["shortfall_salvage_eligible"]),
        "components_excluded": protocol["components_excluded"],
        "interpretation": "reward components changed by construction; no model, baseline EM/F1, process utility or PPO improvement claimed",
        "optimizer_updates": 0, "fresh_confirmation_consumed": False, "ppo_launch_clearance": False,
    }
    for info in [*all_bindings, protocol["gold"], prepared["protocol"]]:
        verify(info)
    bank.finish(output, report, ["protocol.json", "prepared.json", "audit.executed.py", "objective.frozen.py",
        "legacy_reward_function.frozen.py", "structure_only.jsonl", "before_gold.json", "candidate_metrics.jsonl"])
    print(json.dumps({"status": report["status"], "by_prompt": report["by_prompt"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    freeze(args.output_dir) if args.command == "freeze" else run(args.output_dir)
