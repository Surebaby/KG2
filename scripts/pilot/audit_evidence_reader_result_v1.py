"""Independent frozen review of the completed consumed20 evidence reader probe.

Freeze before inspecting new generations. Run only after both generation and
primary assessment manifests are complete. Gold is joined after an independent
Gold-free input/identity/token/format artifact, never used to select questions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
from scripts.prepare import source_quality_candidate_bank_v1 as bank
from scripts.pilot.audit_answer_format_objective_v2 import load_module, verify


ROOT = Path(os.environ.get("KGPW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
READER = ROOT / "outputs/audits/evidence_supply_v1_reader_consumed20_20260906_v1"
PARENT = ROOT / "outputs/audits/generation_length_384_512_paired_probe_20260906_v1"
OUTPUT = ROOT / "outputs/audits/evidence_supply_v1_reader_independent_review_20260906_v1"
ARMS = ("legacy", "evidence_supply_v1")
FIELDS = ("valid", "em", "f1", "format_gated_em", "format_gated_f1", "ppo_em", "ppo_f1",
          "repeated_reasoning", "repeated_conclusion", "repeated_step", "length_capped", "response_tokens")


def freeze(output):
    output.mkdir(parents=True, exist_ok=False)
    rp = json.loads((READER / "protocol.json").read_text())
    old_reward = Path(rp["runtime_snapshot"]) / "kgproweight/training/reward_function.py"
    snapshots = {}
    for path, name in ((Path(__file__), "audit.executed.py"), (old_reward, "validator.frozen.py")):
        (output / name).write_bytes(path.read_bytes())
        snapshots[name] = bank.identity(output / name)
    names = ("kgproweight/data/parsers.py", "kgproweight/data/prompts.py", "kgproweight/eval/pred_processing.py",
             "kgproweight/reward/proofkg_process.py", "kgproweight/reward/source_quality_gate_v1.py",
             "kgproweight/reward/trajectory_source_gate.py", "scripts/prepare/source_quality_candidate_bank_v1.py",
             "scripts/eval/ppo_emf1_development_v1.py", "scripts/pilot/audit_answer_format_objective_v2.py")
    protocol = {
        "schema_version": "evidence-reader-independent-review-protocol-v1",
        "experiment_id": "EVIDENCE-SUPPLY-V1-READER-INDEPENDENT-REVIEW-20260906-V1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "new_generations_and_gold_inspected": False,
        "run_after": "reader generation manifest and primary assessment manifest are complete",
        "cohort": "exact previously consumed20 unique MuSiQue families, K2 each; old40+new40 outputs, none dropped",
        "primary_checks": ["all80 raw token sequences decode/EOS/cap", "candidate/input/policy/base-model/protocol/seed identity",
                           "unchanged system/schema/KG and only evidence modification", "all40 paired metrics and unchanged format-v2",
                           "first1200 displayed chars alias-surface coverage", "K2 per-family mean then 10000 paired-family draws seed42"],
        "metrics": {"EM_F1": "double canonical evaluation extraction, max canonical labels independently, all40 denominator",
                    "PPO": "old first-line extraction multiplied by unchanged format validity",
                    "coverage": "nonboolean primary only, match any nonempty canonical alias as whole-token surface within each displayed passage independently; not entailment",
                    "uncertainty": "numpy.default_rng(42).integers(0,20,(10000,20)); pair-family K2 means; percentiles .025/.975"},
        "source_bindings": {str(path): bank.identity(path) for path in (
            READER / "protocol.json", READER / "prepared.json", READER / "probe.executed.py",
            PARENT / "protocol.json", PARENT / "inputs.jsonl", PARENT / "baseline_384.jsonl",
        )},
        "code_snapshots": snapshots, "supporting_code": {name: bank.identity(ROOT / name) for name in names},
        "no_protocol_changes_or_output_selection": True, "optimizer_updates": 0, "ppo_launch_clearance": False,
    }
    bank.write_json(output / "protocol.json", protocol)
    bank.write_json(output / "prepared.json", {"protocol": bank.identity(output / "protocol.json")})
    print(json.dumps({"status": "FROZEN_WAITING_FOR_PRIMARY_ASSESSMENT", "experiment_id": protocol["experiment_id"]}))


def repeat(values):
    values = [" ".join(value.split()).casefold() for value in values if value.strip()]
    return len(values) != len(set(values))


def field_body(text, label):
    match = re.search(rf"(?ims)^[ \t]*{re.escape(label)}[ \t]*:[ \t]*(.*?)(?=^[ \t]*(?:Reasoning|Knowledge Used|Conclusion)[ \t]*:|\Z)", text)
    return match.group(1) if match else ""


def assert_tokens(prediction, tokenizer, eos):
    ids = prediction["response_token_ids"]
    assert ids == prediction["raw_response_token_ids"] and len(ids) == prediction["n_response_tokens"]
    assert 0 < len(ids) <= 384 and all(type(value) is int for value in ids)
    assert prediction["effective_eos_token_ids"] == eos and not any(value in eos for value in ids[:-1])
    capped = len(ids) == 384 and ids[-1] not in eos
    assert capped == prediction["reached_max_new_tokens"]
    assert capped or ids[-1] in eos
    assert tokenizer.decode(ids, skip_special_tokens=True) == prediction["generation"]


def run(output):
    if (output / "before_gold.json").exists() or (output / "manifest.json").exists():
        raise FileExistsError("refusing to overwrite an independent review")
    prepared = json.loads((output / "prepared.json").read_text())
    verify(prepared["protocol"])
    protocol = json.loads((output / "protocol.json").read_text())
    bindings = [*protocol["source_bindings"].values(), *protocol["code_snapshots"].values(), *protocol["supporting_code"].values()]
    for info in bindings:
        verify(info)
    assert bank.file_sha(Path(__file__)) == protocol["code_snapshots"]["audit.executed.py"]["sha256"]
    generation_release = bank.load_release(READER, "evidence-supply-v1-consumed20-reader-report")
    primary_release = bank.load_release(READER / "assessment", "evidence-supply-v1-consumed20-reader-assessment")
    assert generation_release["status"] == primary_release["status"] == "COMPLETE_DEVELOPMENT_ONLY"
    assert generation_release["candidates"] == primary_release["candidates"] == 40
    assert not any((READER / name).exists() for name in ("FAILED.json", "exception.json", "assessment.exception.json"))
    rp = json.loads((READER / "protocol.json").read_text())
    for group in ("source_bindings", "code_bindings", "frozen_artifacts"):
        for info in rp[group].values():
            verify(info)
    for name in ("supply_protocol", "parent_protocol", "metric_code"):
        verify(rp[name])
    supply = Path(rp["supply_dir"])
    supply_manifest = json.loads((supply / "manifest.json").read_text())
    assert supply_manifest["status"] == "COMPLETE_DEVELOPMENT_ONLY"
    for name, info in supply_manifest["outputs"].items():
        verify({**info, "path": str(supply / info["path"])})
    assert rp["generation"]["max_new_tokens"] == 384 and rp["generation"]["candidates_per_question"] == 2
    assert rp["generation"]["do_sample"] is True and rp["generation"]["batch_size"] == 1
    assert {name: rp["generation"][name] for name in ("temperature", "top_p", "top_k", "seed", "dtype")} == {
        "temperature": 1.0, "top_p": 1.0, "top_k": 0, "seed": 42, "dtype": "bfloat16"}
    bank.validate_model(ROOT / rp["models"]["base_model"]["path"], rp["models"]["base_model"])
    bank.validate_model(Path(rp["policy_path"]), rp["models"]["policy_tokenizer"])
    from transformers import AutoTokenizer
    from kgproweight.data.parsers import extract_final_answer, parse_steps
    from kgproweight.data.prompts import build_rl_messages
    from kgproweight.eval.pred_processing import extract_kg_proweight_answer
    from kgproweight.reward.proofkg_process import canonical_answer_normalize, canonical_exact_match, canonical_token_f1
    tokenizer = AutoTokenizer.from_pretrained(rp["policy_path"], local_files_only=True)
    validator = load_module(output / "validator.frozen.py", "independent_evidence_reader_validator")
    old_inputs = bank.read_rows(READER / "legacy_inputs.jsonl")
    new_inputs = bank.read_rows(READER / "inputs.jsonl")
    assert new_inputs == bank.read_rows(supply / "inputs.jsonl")
    assert old_inputs == [row for row in bank.read_rows(PARENT / "inputs.jsonl") if row["dataset"] == "musique"]
    original120 = {row["prediction"]["candidate_id"]: row for row in bank.read_rows(PARENT / "baseline_384.jsonl")}
    baseline, generated = bank.read_rows(READER / "baseline_384.jsonl"), bank.read_rows(READER / "generations.jsonl")
    assert len(old_inputs) == len(new_inputs) == 20 and len(baseline) == len(generated) == 40
    assert len({row["family_sha256"] for row in old_inputs}) == len({row["question_sha256"] for row in old_inputs}) == 20
    assert len({row["question_key"] for row in old_inputs}) == 20
    records, predictions = [], []
    for i, (old, new) in enumerate(zip(old_inputs, new_inputs)):
        for row in (old, new):
            bank.assert_gold_free(row)
            assert bank.input_hash(row) == row["input_sha256"]
            assert row["retrieved_passages"] == row["spec"]["retrieved_passages"] and len(row["retrieved_passages"]) == 10
            messages = build_rl_messages(row["question"], row["retrieved_passages"], row["kg_subgraph"], top_k=10, max_kg_triples=12)
            assert messages == row["messages"]
            assert tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) == row["prompt"]
            assert len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]) == row["prompt_tokens"] <= 6144
        for name in ("question", "question_key", "dataset", "qid", "family_sha256", "question_sha256", "kg_subgraph", "m_graph"):
            assert old[name] == new[name]
        assert old["messages"][0] == new["messages"][0]
        assert {k: v for k, v in old["spec"].items() if k != "retrieved_passages"} == {k: v for k, v in new["spec"].items() if k != "retrieved_passages"}
        for k in range(2):
            wrapper, candidate = baseline[i * 2 + k], generated[i * 2 + k]
            cid = f"{new['question_key']}::k{k}"
            assert wrapper == original120[cid]
            common = {"candidate_id": cid, "candidate_index": k, "dataset": "musique", "qid": new["qid"],
                      "seed": bank.candidate_seed(42, new["question_key"], k), "generation_contract_sha256": bank.digest(rp["generation"]),
                      "policy_sha256": rp["source_bindings"]["policy"]["sha256"], "base_model_identity_sha256": bank.digest(rp["models"]["base_model"])}
            ref = wrapper["prediction"]
            assert all(pred.get(name) == expected for pred in (ref, candidate) for name, expected in common.items())
            assert wrapper["prediction_sha256"] == candidate["baseline_prediction_sha256"] == bank.digest(ref)
            assert ref["input_sha256"] == candidate["legacy_input_sha256"] == old["input_sha256"]
            assert candidate["input_sha256"] == new["input_sha256"] and candidate["probe_protocol_sha256"] == bank.file_sha(READER / "protocol.json")
            record = {"candidate_id": cid, "candidate_index": k, "question_key": new["question_key"], "dataset": "musique",
                      "family_sha256": new["family_sha256"], "question_sha256": new["question_sha256"]}
            for arm, row, pred in ((ARMS[0], old, ref), (ARMS[1], new, candidate)):
                assert_tokens(pred, tokenizer, rp["generation"]["eos_token_ids"])
                validation = validator.validate_source_gate_trajectory(SimpleNamespace(**row["spec"]), pred["generation"], format_version="v2")
                steps = parse_steps(pred["generation"], known_kg=row["kg_subgraph"])
                record[arm] = {"valid": validation["valid"], "violations": validation["violations"],
                    "steps": validation["all_step_count"], "required_steps": validation["required_steps"],
                    "repeated_reasoning": repeat([field_body(step.raw_text, "Reasoning") for step in steps]),
                    "repeated_conclusion": repeat([field_body(step.raw_text, "Conclusion") for step in steps]),
                    "repeated_step": repeat([step.raw_text for step in steps]), "length_capped": pred["reached_max_new_tokens"],
                    "response_tokens": len(pred["response_token_ids"]), "prediction_sha256": bank.digest(pred)}
            records.append(record)
            predictions.append((ref, candidate))
    bank.write_rows(output / "structure_only.jsonl", records)
    completed_bindings = {str(path): bank.identity(path) for path in (READER / "manifest.json", READER / "assessment/manifest.json",
        READER / "generations.jsonl", READER / "inputs.jsonl", READER / "assessment/candidate_metrics.jsonl", READER / "assessment/coverage.jsonl")}
    bank.write_json(output / "before_gold.json", {"gold_labels_parsed": False, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "structure": bank.identity(output / "structure_only.jsonl"), "completed_source_bindings": completed_bindings})
    verify(rp["gold"]["binding"])
    labels_list = bank.read_rows(Path(rp["gold"]["binding"]["path"]))
    labels = {bank.key(row): row for row in labels_list}
    assert len(labels) == len(labels_list)
    for i, record in enumerate(records):
        label = labels[record["question_key"]]
        assert label["question"] == new_inputs[i // 2]["question"] and label["metadata"]["source_split"] == "train"
        surfaces = validator._canonical_gold_surfaces(label["metadata"]["gold_answer"], label["metadata"].get("gold_answer_aliases"))
        assert surfaces
        for arm, pred in zip(ARMS, predictions[i]):
            answer = extract_kg_proweight_answer(extract_kg_proweight_answer(pred["generation"]))
            ppo_answer = (extract_final_answer(pred["generation"]) or "").split("\n", 1)[0].strip()
            metric = lambda fn, text: max(fn(text, surface) for surface in surfaces)
            em, f1, valid = metric(canonical_exact_match, answer), metric(canonical_token_f1, answer), record[arm]["valid"]
            record[arm].update(em=em, f1=f1, format_gated_em=em * valid, format_gated_f1=f1 * valid,
                              ppo_em=metric(canonical_exact_match, ppo_answer) if valid else 0., ppo_f1=metric(canonical_token_f1, ppo_answer) if valid else 0.)
    primary_rows = bank.read_rows(READER / "assessment/candidate_metrics.jsonl")
    assert len(primary_rows) == 40
    for actual, expected in zip(records, primary_rows):
        for name in ("candidate_id", "candidate_index", "question_key", "dataset", "family_sha256", "question_sha256"):
            assert actual[name] == expected[name]
        for arm in ARMS:
            for name, value in actual[arm].items():
                assert value == expected[arm][name], (actual["candidate_id"], arm, name)
    draws = np.random.default_rng(42).integers(0, 20, size=(10000, 20))
    metrics = {}
    for name in FIELDS:
        arrays = [np.array([record[arm][name] for record in records], dtype=float).reshape(20, 2).mean(axis=1) for arm in ARMS]
        delta = arrays[1] - arrays[0]
        metrics[name] = {ARMS[0]: float(arrays[0].mean()), ARMS[1]: float(arrays[1].mean()), "delta": float(delta.mean()),
                         "paired_family_bootstrap95": np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()}
        assert metrics[name] == primary_release["metrics"][name], name
    coverage = []
    for old, new in zip(old_inputs, new_inputs):
        metadata = labels[new["question_key"]]["metadata"]
        aliases = validator._canonical_gold_surfaces(metadata["gold_answer"], metadata.get("gold_answer_aliases"))
        boolean = canonical_answer_normalize(metadata["gold_answer"]) in {"yes", "no", "noanswer"}
        def covered(row):
            for passage in row["spec"]["retrieved_passages"]:
                text = str(passage.get("contents") or passage.get("text") or "").strip()[:1200]
                tokens = " " + canonical_answer_normalize(text) + " "
                if any(" " + canonical_answer_normalize(alias) + " " in tokens for alias in aliases if canonical_answer_normalize(alias)):
                    return True
            return False
        coverage.append({"question_key": new["question_key"], "boolean": boolean,
                         "legacy_surface_present": None if boolean else covered(old), "new_surface_present": None if boolean else covered(new)})
    assert coverage == bank.read_rows(READER / "assessment/coverage.jsonl")
    coverage_counts = {"nonboolean_questions": sum(not row["boolean"] for row in coverage),
                       "legacy_present": sum(row["legacy_surface_present"] is True for row in coverage),
                       "new_present": sum(row["new_surface_present"] is True for row in coverage)}
    assert all(primary_release["surface_coverage"][name] == value for name, value in coverage_counts.items())
    bank.write_rows(output / "candidate_metrics.jsonl", records)
    bank.write_rows(output / "coverage.jsonl", coverage)
    for info in [*bindings, *completed_bindings.values(), rp["gold"]["binding"], prepared["protocol"]]:
        verify(info)
    bank.validate_model(ROOT / rp["models"]["base_model"]["path"], rp["models"]["base_model"])
    bank.validate_model(Path(rp["policy_path"]), rp["models"]["policy_tokenizer"])
    report = {"schema_version": "evidence-reader-independent-review-report-v1", "experiment_id": protocol["experiment_id"],
              "status": "COMPLETE_INDEPENDENT_REPRODUCTION_DEVELOPMENT_ONLY", "questions": 20, "paired_candidates": 40,
              "raw_outputs_checked": 80, "candidate_metrics_and_format_reproduced": 80, "family_bootstrap_fields_reproduced": len(FIELDS),
              "whole_model_and_tokenizer_bindings_reverified": True, "metrics": metrics, "surface_coverage": coverage_counts,
              "optimizer_updates": 0, "fresh_confirmation_consumed": False, "ppo_launch_clearance": False,
              "interpretation": "exact reproduction on consumed20 development families; evidence supply adds upstream retrieval cost; not generalization, PPO gain or verified chain coverage"}
    bank.finish(output, report, ["protocol.json", "prepared.json", "audit.executed.py", "validator.frozen.py", "structure_only.jsonl",
                                "before_gold.json", "candidate_metrics.jsonl", "coverage.jsonl"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    freeze(args.output_dir) if args.command == "freeze" else run(args.output_dir)
