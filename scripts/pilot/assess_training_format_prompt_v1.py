"""Independent paired assessment, with methods frozen before prompt generation.

Gold is opened only after complete immutable candidate banks and a separate
Gold-free structure artifact have been verified. No reward or prompt fitting.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

from scripts.prepare import source_quality_candidate_bank_v1 as bank
from kgproweight.data.parsers import extract_final_answer, parse_steps
from kgproweight.eval.pred_processing import extract_kg_proweight_answer
from kgproweight.reward.proofkg_process import canonical_answer_normalize, canonical_exact_match, canonical_token_f1
from kgproweight.training.reward_function import _canonical_gold_surfaces, validate_source_gate_trajectory


ROOT = Path(os.environ.get("KGPW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
DEFAULT_OUTPUT = ROOT / "outputs/audits/training_format_prompt_v1_assessment_20260906_v1"
DEFAULT_GENERATIONS = ROOT / "outputs/audits/training_format_prompt_v1_paired_probe_20260906_v1"
ARMS = ("legacy", "prompt_v1")
FIELDS = ("valid", "em", "f1", "format_gated_em", "format_gated_f1", "ppo_em", "ppo_f1",
          "repeated_reasoning", "repeated_conclusion", "repeated_step", "length_capped", "response_tokens")


def verify_binding(info):
    if bank.file_sha(Path(info["path"])) != info["sha256"]:
        raise ValueError(f"frozen binding changed: {info['path']}")


def validate_cohort(records, expected_datasets):
    groups = defaultdict(list)
    for row in records:
        groups[row["question_key"]].append(row)
    families, questions = set(), set()
    for key, pair in groups.items():
        if len(pair) != 2 or {r["candidate_index"] for r in pair} != {0, 1}:
            raise ValueError("exactly K2 required before Gold access")
        identity = {(r["dataset"], r["family_sha256"], r["question_sha256"]) for r in pair}
        if len(identity) != 1:
            raise ValueError("inconsistent question identity")
        _, family, question = next(iter(identity))
        if family in families or question in questions:
            raise ValueError("repeated family or question before Gold access")
        families.add(family); questions.add(question)
    if Counter(pair[0]["dataset"] for pair in groups.values()) != expected_datasets:
        raise ValueError("dataset quota mismatch before Gold access")


def freeze(output_dir):
    """Freeze actual evaluator bytes before completion/Gold, without editing methods."""
    destination = output_dir / "assessment.executed.py"
    if destination.exists() or (output_dir / "scoring_code_freeze.json").exists():
        raise FileExistsError("refusing to overwrite scoring code freeze")
    verify_binding(json.loads((output_dir / "prepared.json").read_text())["protocol"])
    destination.write_bytes(Path(__file__).read_bytes())
    bank.write_json(output_dir / "scoring_code_freeze.json", {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "gold_labels_parsed": False,
        "executed_code": bank.identity(destination), "original_source": bank.identity(Path(__file__)),
        "protocol": bank.identity(output_dir / "protocol.json"),
        "supporting_code": {name:bank.identity(ROOT / name) for name in
                            ("scripts/prepare/source_quality_candidate_bank_v1.py", "scripts/pilot/probe_training_format_prompt_v1.py")}})


def normalized(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def field_content(step, name):
    match = re.search(rf"(?ims)^[ \t]*{name}[ \t]*:[ \t]*(.*?)(?=^[ \t]*(?:Reasoning|Knowledge Used|Conclusion)[ \t]*:|\Z)", step)
    return normalized(match.group(1)) if match else ""


def repeated_nonempty(values):
    values = [v for v in values if v]
    return len(values) != len(set(values))


def structure(row, pred):
    validity = validate_source_gate_trajectory(SimpleNamespace(**row["spec"]), pred["generation"], format_version="v2")
    steps = parse_steps(pred["generation"], known_kg=row["kg_subgraph"])
    answer = extract_kg_proweight_answer(extract_kg_proweight_answer(pred["generation"]))
    return {"valid": bool(validity["valid"]), "violations": validity["violations"],
            "steps": validity["all_step_count"], "required_steps": validity["required_steps"],
            "too_few_steps": validity["all_step_count"] < validity["required_steps"],
            "too_many_steps": validity["all_step_count"] > 5,
            "repeated_reasoning": repeated_nonempty([field_content(s.raw_text, "Reasoning") for s in steps]),
            "repeated_conclusion": repeated_nonempty([field_content(s.raw_text, "Conclusion") for s in steps]),
            "repeated_step": repeated_nonempty([normalized(s.raw_text) for s in steps]),
            "length_capped": pred["reached_max_new_tokens"], "response_tokens": len(pred["response_token_ids"]),
            "generation_sha256": bank.digest(pred["generation"]),
            "normalized_answer_sha256": bank.digest(canonical_answer_normalize(answer)),
            "prediction_sha256": bank.digest(pred)}


def answer_scores(generation, primary, aliases, valid):
    surfaces = _canonical_gold_surfaces(primary, aliases)
    if not surfaces:
        raise ValueError("missing frozen train label")
    answer = extract_kg_proweight_answer(extract_kg_proweight_answer(generation))
    ppo_answer = (extract_final_answer(generation) or "").split("\n", 1)[0].strip()
    score = lambda metric, text: float(max(metric(text, surface) for surface in surfaces))
    em, f1 = score(canonical_exact_match, answer), score(canonical_token_f1, answer)
    return {"em": em, "f1": f1, "format_gated_em": em * valid, "format_gated_f1": f1 * valid,
            "ppo_em": score(canonical_exact_match, ppo_answer) if valid else 0.0,
            "ppo_f1": score(canonical_token_f1, ppo_answer) if valid else 0.0}


def paired_questions(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["question_key"]].append(row)
    questions, families = [], set()
    for key, rows in sorted(groups.items()):
        if len(rows) != 2 or {r["candidate_index"] for r in rows} != {0, 1}:
            raise ValueError("exactly K2, with distinct candidate indices, required")
        if len({(r["dataset"], r["family_sha256"]) for r in rows}) != 1:
            raise ValueError("question metadata inconsistent")
        family = rows[0]["family_sha256"]
        if family in families:
            raise ValueError("this fixed cohort requires unique families per question")
        families.add(family)
        q = {"question_key": key, "dataset": rows[0]["dataset"], "family_sha256": family}
        for arm in ARMS:
            q[arm] = {f: float(np.mean([r[arm][f] for r in rows])) for f in FIELDS}
            q[arm]["identical_K2_generation"] = rows[0][arm]["generation_sha256"] == rows[1][arm]["generation_sha256"]
            q[arm]["same_normalized_K2_answer"] = rows[0][arm]["normalized_answer_sha256"] == rows[1][arm]["normalized_answer_sha256"]
        questions.append(q)
    return questions


def summarize(records):
    questions = paired_questions(records)
    output = {"candidates": len(records), "questions": len(questions), "families": len(questions)}
    for arm in ARMS:
        values = [r[arm] for r in records]
        output[arm] = {f: float(np.mean([q[arm][f] for q in questions])) for f in FIELDS}
        output[arm].update({"valid_count": sum(v["valid"] for v in values),
            "length_capped_count": sum(v["length_capped"] for v in values),
            "response_tokens_sum": sum(v["response_tokens"] for v in values),
            "too_few_steps": sum(v["too_few_steps"] for v in values), "too_many_steps": sum(v["too_many_steps"] for v in values),
            "step_counts": dict(Counter(v["steps"] for v in values)),
            "violations": dict(Counter(e for v in values for e in v["violations"])),
            "identical_K2_generations": sum(q[arm]["identical_K2_generation"] for q in questions),
            "same_normalized_K2_answers": sum(q[arm]["same_normalized_K2_answer"] for q in questions)})
    output["transitions"] = {f"{old}_to_{new}": sum(r["legacy"]["valid"] == old and r["prompt_v1"]["valid"] == new for r in records)
                             for old in (False, True) for new in (False, True)}
    return output


def macro_and_intervals(questions, replicates=10000, seed=42):
    datasets = sorted({q["dataset"] for q in questions})
    strata = [[q for q in questions if q["dataset"] == d] for d in datasets]
    rng = np.random.default_rng(seed)
    # A family is drawn once per replicate, carrying its paired arms and K2 mean.
    indices = [rng.integers(0, len(s), size=(replicates, len(s))) for s in strata]
    result = {}
    for field in FIELDS:
        old = float(np.mean([np.mean([q["legacy"][field] for q in s]) for s in strata]))
        new = float(np.mean([np.mean([q["prompt_v1"][field] for q in s]) for s in strata]))
        draws = np.mean([np.asarray([q["prompt_v1"][field] - q["legacy"][field] for q in s])[idx].mean(axis=1)
                         for s, idx in zip(strata, indices)], axis=0)
        result[field] = {"legacy": old, "prompt_v1": new, "delta": new-old,
                         "delta_95_percentile_interval": [float(v) for v in np.quantile(draws, [.025, .975])]}
    return result


def check_tokens(pred, eos, tokenizer):
    ids = pred["response_token_ids"]
    if not ids or ids != pred["raw_response_token_ids"] or len(ids) != pred["n_response_tokens"] or len(ids) > 384:
        raise ValueError("raw/response token contract mismatch")
    if any(t in eos for t in ids[:-1]) or pred["effective_eos_token_ids"] != eos:
        raise ValueError("EOS contract mismatch")
    capped = len(ids) == 384 and ids[-1] not in eos
    if capped != pred["reached_max_new_tokens"] or not capped and ids[-1] not in eos:
        raise ValueError("invalid stopping condition")
    if tokenizer.decode(ids, skip_special_tokens=True) != pred["generation"]:
        raise ValueError("generation does not decode from frozen token IDs")


def assess(generation_dir, output_dir):
    if any((output_dir / name).exists() for name in ("before_gold.json", "structure_only.jsonl", "manifest.json")):
        raise FileExistsError("refusing to overwrite assessment")
    code_freeze = json.loads((output_dir / "scoring_code_freeze.json").read_text())
    for field in ("executed_code", "original_source", "protocol"):
        verify_binding(code_freeze[field])
    for info in code_freeze["supporting_code"].values():
        verify_binding(info)
    if bank.file_sha(Path(__file__)) != code_freeze["executed_code"]["sha256"]:
        raise ValueError("actual evaluator differs from pre-frozen executed code")
    prepared = json.loads((output_dir / "prepared.json").read_text())
    if (Path(prepared["protocol"]["path"]).resolve() != (output_dir / "protocol.json").resolve()
            or Path(code_freeze["protocol"]["path"]).resolve() != (output_dir / "protocol.json").resolve()
            or prepared["protocol"]["sha256"] != code_freeze["protocol"]["sha256"]):
        raise ValueError("assessment protocol path/hash disagreement")
    verify_binding(prepared["protocol"])
    protocol = json.loads((output_dir / "protocol.json").read_text())
    for field in ("baseline_binding", "input_binding", "selection_binding"):
        verify_binding(protocol[field])
    for info in protocol["frozen_metric_code"].values():
        verify_binding(info)
    release = bank.load_release(generation_dir, "training-format-prompt-v1-paired-development-probe-report")
    required_outputs = {"inputs.jsonl", "legacy_inputs.jsonl", "baseline_384.jsonl", "generations_prompt_v1.jsonl",
                        "selection.question_only.jsonl", "protocol.json", "prepared.json", "execution_environment.json"}
    if (release.get("status") != "COMPLETE_DEVELOPMENT_ONLY" or release.get("all120_candidates_retained") is not True
            or not required_outputs.issubset(release["outputs"])
            or any((generation_dir / name).exists() for name in ("exception.json", "FAILED.json"))):
        raise ValueError("complete successful generation release with every required artifact mandatory")
    from scripts.pilot.probe_training_format_prompt_v1 import verify
    gp = verify(generation_dir)
    if gp["assessment_protocol"]["sha256"] != prepared["protocol"]["sha256"]:
        raise ValueError("generation did not bind pre-frozen assessment protocol")
    for local, field in [("legacy_inputs.jsonl", "input_binding"), ("baseline_384.jsonl", "baseline_binding"),
                         ("selection.question_only.jsonl", "selection_binding")]:
        if bank.file_sha(generation_dir / local) != protocol[field]["sha256"]:
            raise ValueError("not the exact pre-frozen comparison cohort")
    rows = bank.read_rows(generation_dir / "inputs.jsonl")
    legacy = bank.read_rows(generation_dir / "legacy_inputs.jsonl")
    baseline = bank.read_rows(generation_dir / "baseline_384.jsonl")
    generated = bank.read_rows(generation_dir / "generations_prompt_v1.jsonl")
    if len(rows) != 60 or len(legacy) != 60 or len(baseline) != 120 or len(generated) != 120:
        raise ValueError("complete fixed60/K2 cohort required")
    from transformers import AutoTokenizer
    bank.validate_model(Path(gp["policy_path"]), gp["models"]["policy_tokenizer"])
    tokenizer = AutoTokenizer.from_pretrained(gp["policy_path"], local_files_only=True)
    executed = Path(code_freeze["executed_code"]["path"])
    records = []
    for i, (row, old_input) in enumerate(zip(rows, legacy)):
        bank.assert_gold_free(row)
        if bank.input_hash(row) != row["input_sha256"] or bank.input_hash(old_input) != old_input["input_sha256"]:
            raise ValueError("input identity mismatch")
        allowed = {"messages", "prompt", "prompt_tokens", "input_sha256"}
        if {k:v for k,v in row.items() if k not in allowed} != {k:v for k,v in old_input.items() if k not in allowed}:
            raise ValueError("evidence or source spec changed")
        if row["messages"][1:] != old_input["messages"][1:] or row["messages"][0] == old_input["messages"][0]:
            raise ValueError("system-only variable violated")
        for k in range(2):
            wrapper, new = baseline[2*i+k], generated[2*i+k]
            old = wrapper["prediction"]
            cid = f"{row['question_key']}::k{k}"
            expected = {"candidate_id": cid, "candidate_index": k, "dataset": row["dataset"], "qid": row["qid"],
                        "seed": bank.candidate_seed(42, row["question_key"], k),
                        "generation_contract_sha256": bank.digest(gp["generation"]),
                        "policy_sha256": gp["source_bindings"]["policy"]["sha256"],
                        "base_model_identity_sha256": bank.digest(gp["models"]["base_model"])}
            if any(pred.get(key) != value for pred in (old, new) for key, value in expected.items()):
                raise ValueError("candidate identity, model, seed, or generation contract mismatch")
            if (new.get("schema_version") != gp["schema_version"] + "-generation-row"
                    or new.get("experiment_id") != gp["experiment_id"] or new.get("prompt_version") != gp["prompt_version"]):
                raise ValueError("candidate version metadata mismatch")
            if (wrapper["prediction_sha256"] != bank.digest(old) or new["baseline_prediction_sha256"] != bank.digest(old)
                    or new["input_sha256"] != row["input_sha256"] or old["input_sha256"] != old_input["input_sha256"]
                    or new["legacy_input_sha256"] != old_input["input_sha256"]
                    or new["probe_protocol_sha256"] != bank.file_sha(generation_dir / "protocol.json")):
                raise ValueError("paired baseline/prompt binding mismatch")
            for pred in (old, new):
                check_tokens(pred, gp["generation"]["eos_token_ids"], tokenizer)
            records.append({"candidate_id": cid, "candidate_index": k, "question_key": row["question_key"],
                            "dataset": row["dataset"], "family_sha256": row["family_sha256"],
                            "question_sha256": row["question_sha256"], "m_graph": row["m_graph"],
                            "legacy": structure(old_input, old), "prompt_v1": structure(row, new)})
    validate_cohort(records, protocol["cohort"]["datasets"])
    bank.write_rows(output_dir / "structure_only.jsonl", records)
    before_gold = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "gold_labels_parsed": False,
                   "assessment_code": bank.identity(executed), "structure": bank.identity(output_dir / "structure_only.jsonl"),
                   "generation_manifest": bank.identity(generation_dir / "manifest.json"),
                   "generation_rows": bank.identity(generation_dir / "generations_prompt_v1.jsonl"),
                   "actual_inputs": {name:bank.identity(generation_dir / name) for name in required_outputs},
                   "tokenizer": gp["models"]["policy_tokenizer"],
                   "assessment_protocol": bank.identity(output_dir / "protocol.json")}
    bank.write_json(output_dir / "before_gold.json", before_gold)
    verify_binding(protocol["gold"]["binding"])
    labels_list = bank.read_rows(Path(protocol["gold"]["binding"]["path"]))
    labels = {bank.key(row): row for row in labels_list}
    if len(labels) != len(labels_list):
        raise ValueError("duplicate frozen label identity")
    for i, r in enumerate(records):
        label = labels[r["question_key"]]
        source = rows[i//2]
        if label["question"] != source["question"] or label["metadata"]["source_split"] != "train":
            raise ValueError("train label question/source mismatch")
        for arm, pred in [("legacy", baseline[i]["prediction"]), ("prompt_v1", generated[i])]:
            r[arm].update(answer_scores(pred["generation"], label["metadata"]["gold_answer"],
                                       label["metadata"].get("gold_answer_aliases"), r[arm]["valid"]))
    questions = paired_questions(records)
    if Counter(q["dataset"] for q in questions) != protocol["cohort"]["datasets"]:
        raise ValueError("dataset quota mismatch")
    bank.write_rows(output_dir / "candidate_metrics.jsonl", records)
    bank.write_rows(output_dir / "question_metrics.jsonl", questions)
    report = {"schema_version": "training-format-prompt-v1-independent-assessment-report", "experiment_id": protocol["experiment_id"],
              "status": "COMPLETE_DEVELOPMENT_ONLY", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "overall": summarize(records), "macro": macro_and_intervals(questions, **{k:protocol["uncertainty"][k] for k in ("replicates", "seed")}),
              "by_dataset": {d:summarize([r for r in records if r["dataset"] == d]) for d in sorted({r["dataset"] for r in records})},
              "by_m_graph": {str(m):summarize([r for r in records if r["m_graph"] == m]) for m in (0, 1)},
              "code": bank.identity(executed), "generation_manifest": before_gold["generation_manifest"],
              "gold_source": protocol["gold"]["binding"], "optimizer_updates": 0, "baseline_evaluation_changed": False,
              "fresh_confirmation_consumed": False, "ppo_launch_clearance": False,
              "interpretation": "one small consumed-train prompt diagnostic; bootstrap descriptive, not independent model/baseline improvement or noninferiority"}
    for info in protocol["frozen_metric_code"].values():
        verify_binding(info)
    for key in ("assessment_code", "structure", "generation_manifest", "generation_rows", "assessment_protocol"):
        verify_binding(before_gold[key])
    for info in before_gold["actual_inputs"].values():
        verify_binding(info)
    for info in code_freeze["supporting_code"].values():
        verify_binding(info)
    verify_binding(protocol["gold"]["binding"])
    bank.validate_model(Path(gp["policy_path"]), gp["models"]["policy_tokenizer"])
    bank.load_release(generation_dir, "training-format-prompt-v1-paired-development-probe-report")
    verify(generation_dir)
    bank.finish(output_dir, report, ["protocol.json", "prepared.json", "scoring_code_freeze.json", "assessment.executed.py", "structure_only.jsonl",
                "before_gold.json", "candidate_metrics.jsonl", "question_metrics.jsonl"])
    print(json.dumps({"status":report["status"], "macro":report["macro"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--generation-dir", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.output_dir)
    else:
        assess(args.generation_dir, args.output_dir)
