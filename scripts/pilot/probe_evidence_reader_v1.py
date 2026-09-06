"""Frozen consumed20 evidence-only reader probe; Gold opened after generation.

Commands freeze/run/assess preserve failed outputs. This is a development
retrieval comparison, with extra upstream cost, never a PPO evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scripts.prepare import source_quality_candidate_bank_v1 as bank

ROOT = Path(os.environ.get("KGPW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
PARENT = ROOT / "outputs/audits/generation_length_384_512_paired_probe_20260906_v1"
SUPPLY = ROOT / "outputs/audits/evidence_supply_v1_consumed20_20260906_v1"
OUTPUT = ROOT / "outputs/audits/evidence_supply_v1_reader_consumed20_20260906_v1"
METRICS = ROOT / "outputs/audits/training_format_prompt_v1_assessment_20260906_v1/assessment.executed.py"
VERSION = "evidence-supply-v1-consumed20-reader"
EXPERIMENT = "EVIDENCE-SUPPLY-V1-READER-CONSUMED20-20260906-V1"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def helper():
    return load_module("evidence_reader_frozen_helpers", PARENT / "probe.executed.py")


def freeze(directory, supply):
    directory.mkdir(parents=True, exist_ok=False)
    h = helper()
    parent = h.verify(PARENT)
    rows = [r for r in bank.read_rows(PARENT / "inputs.jsonl") if r["dataset"] == "musique"]
    if len(rows) != 20 or len({r["family_sha256"] for r in rows}) != 20:
        raise ValueError("expected exact consumed20 unique MuSiQue families")
    keys = {r["question_key"] for r in rows}
    baseline = [r for r in bank.read_rows(PARENT / "baseline_384.jsonl")
                if r["prediction"]["candidate_id"].rsplit("::k", 1)[0] in keys]
    if len(baseline) != 40:
        raise ValueError("complete original K2 baseline required")
    bank.write_rows(directory / "legacy_inputs.jsonl", rows)
    bank.write_rows(directory / "baseline_384.jsonl", baseline)
    (directory / "probe.executed.py").write_bytes(Path(__file__).read_bytes())
    prior_assessment = json.loads((METRICS.parent / "protocol.json").read_text())
    protocol = {
        "schema_version": VERSION, "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "supply_dir": str(supply), "supply_protocol": bank.identity(supply / "protocol.json"),
        "parent_protocol": bank.identity(PARENT / "protocol.json"),
        "runtime_snapshot": parent["runtime_snapshot"], "models": parent["models"],
        "policy_path": parent["policy_path"], "generation": parent["generation_384"],
        "source_bindings": parent["source_bindings"], "code_bindings": parent["code_bindings"],
        "frozen_artifacts": {n: bank.identity(directory / n) for n in
                             ("legacy_inputs.jsonl", "baseline_384.jsonl", "probe.executed.py")},
        "metric_code": bank.identity(METRICS), "gold": prior_assessment["gold"],
        "single_research_variable": "Gold-free evidence supply, with additional upstream retrieval cost",
        "cohort": {"questions": 20, "K": 2, "families": 20, "dataset": "musique"},
        "reader_contract": "same original system prompt, SFT, max384, seed per candidate; final10 passages",
        "metrics": "frozen canonical double extraction and max-alias EM/F1 over ALL40; unchanged strict format-v2",
        "uncertainty": {"replicates": 10000, "seed": 42, "unit": "paired family with K2 mean"},
        "coverage": "normalized whole-token answer/alias surface in actually displayed first1200chars of each stripped passage content; nonboolean only; no chain-validity claim",
        "no_reselection_or_retries": True, "gold_before_generation": False,
        "optimizer_updates": 0, "fresh_confirmation_consumed": False, "ppo_launch_clearance": False,
    }
    bank.write_json(directory / "protocol.json", protocol)
    bank.write_json(directory / "prepared.json", {"protocol": bank.identity(directory / "protocol.json")})
    h.status(directory, "FROZEN_WAITING_FOR_EVIDENCE_SUPPLY", questions=20, candidates=40)


def verify(directory, models=False):
    h = helper()
    h.require_bindings(json.loads((directory / "prepared.json").read_text()))
    p = json.loads((directory / "protocol.json").read_text())
    generation = p["generation"]
    expected = {"max_new_tokens": 384, "max_input_tokens": 6144,
                "candidates_per_question": 2, "batch_size": 1, "seed": 42,
                "do_sample": True, "temperature": 1., "top_p": 1., "top_k": 0,
                "dtype": "bfloat16"}
    if any(generation.get(k) != v for k, v in expected.items()):
        raise ValueError("fixed384/K2/batch1/unwarped generation contract changed")
    for name in ("source_bindings", "code_bindings", "frozen_artifacts"):
        h.require_bindings(p[name])
    h.require_bindings({n: p[n] for n in ("supply_protocol", "parent_protocol", "metric_code")})
    if bank.file_sha(Path(__file__)) != p["frozen_artifacts"]["probe.executed.py"]["sha256"]:
        raise ValueError("actual executing code differs from pre-frozen probe")
    if Path(bank.__file__).resolve() != Path(p["runtime_snapshot"]) / "scripts/prepare/source_quality_candidate_bank_v1.py":
        raise ValueError("probe must use frozen original runtime imports")
    if models:
        bank.validate_model(ROOT / p["models"]["base_model"]["path"], p["models"]["base_model"])
        bank.validate_model(Path(p["policy_path"]), p["models"]["policy_tokenizer"])
    return p


def supply_inputs(p, directory):
    from kgproweight.data.prompts import build_rl_messages
    supply = Path(p["supply_dir"])
    manifest = json.loads((supply / "manifest.json").read_text())
    if manifest.get("status") != "COMPLETE_DEVELOPMENT_ONLY":
        raise ValueError("supply must finish successfully before any reader generation")
    if any((supply / n).exists() for n in ("exception.json", "FAILED.json")):
        raise ValueError("failed supply cannot be accepted")
    for info in manifest["outputs"].values():
        helper().require_bindings({"supply_output": {**info, "path": str(supply / info["path"])}})
    old = bank.read_rows(directory / "legacy_inputs.jsonl")
    new = bank.read_rows(supply / "inputs.jsonl")
    if len(new) != 20 or [r["question_key"] for r in new] != [r["question_key"] for r in old]:
        raise ValueError("exact full20 ordered cohort required")
    for a, b in zip(old, new):
        bank.assert_gold_free(b)
        if bank.input_hash(a) != a["input_sha256"] or bank.input_hash(b) != b["input_sha256"]:
            raise ValueError("evidence input hash mismatch")
        for field in ("question", "question_key", "question_sha256", "family_sha256", "dataset", "qid", "kg_subgraph", "m_graph"):
            if a[field] != b[field]:
                raise ValueError(f"non-evidence field changed: {field}")
        if a["messages"][0] != b["messages"][0] or len(b["spec"]["retrieved_passages"]) != 10:
            raise ValueError("same original system prompt and final10 passages required")
        if ({k: v for k, v in a["spec"].items() if k != "retrieved_passages"}
                != {k: v for k, v in b["spec"].items() if k != "retrieved_passages"}):
            raise ValueError("non-evidence reward spec changed")
        if b["retrieved_passages"] != b["spec"]["retrieved_passages"]:
            raise ValueError("top-level and spec passages differ")
        expected_messages = build_rl_messages(b["question"], b["retrieved_passages"], b["kg_subgraph"], top_k=10, max_kg_triples=12)
        if b["messages"] != expected_messages:
            raise ValueError("messages differ from frozen original evidence renderer")
        if b["prompt_tokens"] > p["generation"]["max_input_tokens"]:
            raise ValueError("reader input budget exceeded")
    return old, new


def run(directory):
    h = helper()
    h.status(directory, "VERIFYING_EVIDENCE_READER_BINDINGS")
    p = verify(directory, models=True)
    old, rows = supply_inputs(p, directory)
    if (directory / "generations.jsonl").exists():
        raise FileExistsError("no restart/overwrite of candidate outputs")
    bank.write_rows(directory / "inputs.jsonl", rows)
    bank.write_json(directory / "before_generation.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "gold_labels_parsed": False,
        "bindings": {"inputs": bank.identity(directory / "inputs.jsonl"),
                     "supply_manifest": bank.identity(Path(p["supply_dir"]) / "manifest.json"),
                     "protocol": bank.identity(directory / "protocol.json")}})
    baseline = {r["prediction"]["candidate_id"]: r["prediction"]
                for r in bank.read_rows(directory / "baseline_384.jsonl")}
    torch = bank.require_cuda("cuda:0")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import transformers, peft
    tokenizer = AutoTokenizer.from_pretrained(p["policy_path"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    h.status(directory, "LOADING_FROZEN_BF16_SFT")
    model = AutoModelForCausalLM.from_pretrained(ROOT / p["models"]["base_model"]["path"], local_files_only=True,
                                               torch_dtype=torch.bfloat16).to("cuda:0")
    model = PeftModel.from_pretrained(model, p["policy_path"], local_files_only=True, is_trainable=False).eval()
    eos = list(bank._rollout_eos_token_ids(model, tokenizer))
    if eos != p["generation"]["eos_token_ids"]:
        raise ValueError("EOS contract changed")
    bank.write_json(directory / "execution_environment.json", {"python": sys.executable, "torch": torch.__version__,
        "transformers": transformers.__version__, "peft": peft.__version__, "gpu": torch.cuda.get_device_name(),
        "dtype": "bf16", "attention_implementation": model.config._attn_implementation,
        "bank_loaded_from": str(Path(bank.__file__).resolve()), "optimizer_updates": 0})
    start = time.monotonic()
    n = 0
    with (directory / "generations.jsonl").open("x") as output:
        for row in rows:
            encoded = tokenizer(row["prompt"], add_special_tokens=False, truncation=False, return_tensors="pt")
            count = encoded["input_ids"].shape[-1]
            if count != row["prompt_tokens"] or tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True) != row["prompt"]:
                raise ValueError("prompt rendering/tokenization mismatch")
            for k in range(2):
                cid = f"{row['question_key']}::k{k}"
                reference = baseline[cid]
                seed = bank.candidate_seed(42, row["question_key"], k)
                if seed != reference["seed"]:
                    raise ValueError("baseline seed mismatch")
                torch.manual_seed(seed)
                with torch.inference_mode():
                    seq = model.generate(**{a: b.to("cuda:0") for a, b in encoded.items()}, do_sample=True,
                        temperature=1., top_p=1., top_k=0, max_new_tokens=384, pad_token_id=tokenizer.pad_token_id, eos_token_id=eos)
                raw = seq[0, count:]
                response = bank._trim_response_v2(raw, eos_token_ids=eos, pad_token_id=tokenizer.pad_token_id, max_new_tokens=384)
                ids = response.tolist()
                pred = {"candidate_id": cid, "candidate_index": k, "dataset": row["dataset"], "qid": row["qid"], "seed": seed,
                    "input_sha256": row["input_sha256"], "legacy_input_sha256": reference["input_sha256"],
                    "probe_protocol_sha256": bank.file_sha(directory / "protocol.json"),
                    "generation_contract_sha256": bank.digest(p["generation"]), "baseline_prediction_sha256": bank.digest(reference),
                    "policy_sha256": reference["policy_sha256"], "base_model_identity_sha256": reference["base_model_identity_sha256"],
                    "generation": tokenizer.decode(ids, skip_special_tokens=True), "raw_response_token_ids": raw.tolist(),
                    "response_token_ids": ids, "n_response_tokens": len(ids), "effective_eos_token_ids": eos,
                    "reached_max_new_tokens": bank._response_is_length_capped_v2(response, max_new_tokens=384, eos_token_ids=eos)}
                output.write(bank.canonical_json(pred) + "\n"); output.flush()
                n += 1
                h.status(directory, "GENERATING_EVIDENCE_READER", completed=n, expected=40, elapsed_seconds=time.monotonic()-start)
    gpu = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    del model, seq, raw, response, encoded
    gc.collect(); torch.cuda.empty_cache()
    verify(directory, models=True)
    if n != 40:
        raise ValueError("incomplete fixed40 candidates")
    h.require_bindings(json.loads((directory / "before_generation.json").read_text())["bindings"])
    files = ["protocol.json", "prepared.json", "legacy_inputs.jsonl", "inputs.jsonl", "baseline_384.jsonl",
             "probe.executed.py", "before_generation.json", "execution_environment.json", "generations.jsonl"]
    bank.finish(directory, {"schema_version": VERSION+"-report", "experiment_id": EXPERIMENT,
        "status": "COMPLETE_DEVELOPMENT_ONLY", "candidates": n, "gpu": gpu,
        "generation_elapsed_seconds": time.monotonic()-start, "gold_access": False, "optimizer_updates": 0}, files)
    h.status(directory, "GENERATION_COMPLETE_READY_FOR_FROZEN_ASSESSMENT", completed=n)


def assess(directory):
    p = verify(directory)
    dest = directory / "assessment"
    dest.mkdir(exist_ok=False)
    release = bank.load_release(directory, VERSION+"-report")
    if release.get("status") != "COMPLETE_DEVELOPMENT_ONLY" or release.get("candidates") != 40:
        raise ValueError("full successful40 candidate release required")
    if (directory / "exception.json").exists():
        raise ValueError("failed generation cannot be scored as complete")
    old_inputs, new_inputs = supply_inputs(p, directory)
    if bank.read_rows(directory / "inputs.jsonl") != new_inputs:
        raise ValueError("actual generated inputs differ from supply")
    baseline = bank.read_rows(directory / "baseline_384.jsonl")
    generated = bank.read_rows(directory / "generations.jsonl")
    if len(baseline) != 40 or len(generated) != 40:
        raise ValueError("fixed complete K2 pairs required")
    metric = load_module("evidence_reader_frozen_metrics", METRICS)
    from transformers import AutoTokenizer
    bank.validate_model(Path(p["policy_path"]), p["models"]["policy_tokenizer"])
    tokenizer = AutoTokenizer.from_pretrained(p["policy_path"], local_files_only=True)
    records = []
    for i, (old, new) in enumerate(zip(old_inputs, new_inputs)):
        for k in range(2):
            ref = baseline[2*i+k]; a = ref["prediction"]; b = generated[2*i+k]
            cid = f"{new['question_key']}::k{k}"
            common = {"candidate_id": cid, "candidate_index": k, "dataset": "musique", "qid": new["qid"],
                      "seed": bank.candidate_seed(42, new["question_key"], k),
                      "generation_contract_sha256": bank.digest(p["generation"]),
                      "policy_sha256": p["source_bindings"]["policy"]["sha256"],
                      "base_model_identity_sha256": bank.digest(p["models"]["base_model"])}
            if any(pred.get(field) != value for pred in (a, b) for field, value in common.items()):
                raise ValueError("candidate/model/sampling identity mismatch")
            if (ref["prediction_sha256"] != bank.digest(a) or b["baseline_prediction_sha256"] != bank.digest(a)
                or a["input_sha256"] != old["input_sha256"] or b["input_sha256"] != new["input_sha256"]
                or b["legacy_input_sha256"] != old["input_sha256"]
                or b["probe_protocol_sha256"] != bank.file_sha(directory / "protocol.json")):
                raise ValueError("paired evidence binding mismatch")
            for pred in (a, b):
                metric.check_tokens(pred, p["generation"]["eos_token_ids"], tokenizer)
            records.append({"candidate_id": cid, "candidate_index": k, "question_key": new["question_key"],
                "dataset": "musique", "question_sha256": new["question_sha256"], "family_sha256": new["family_sha256"],
                "legacy": metric.structure(old, a), "evidence_supply_v1": metric.structure(new, b)})
    metric.validate_cohort(records, {"musique": 20})
    bank.write_rows(dest / "structure_only.jsonl", records)
    bank.write_json(dest / "before_gold.json", {"gold_labels_parsed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "generation_manifest": bank.identity(directory / "manifest.json"),
        "structure": bank.identity(dest / "structure_only.jsonl"), "executed_code": bank.identity(Path(__file__))})
    helper().require_bindings({"gold": p["gold"]["binding"]})
    labels_list = bank.read_rows(Path(p["gold"]["binding"]["path"]))
    labels = {bank.key(r): r for r in labels_list}
    if len(labels) != len(labels_list):
        raise ValueError("duplicate label identities")
    for i, r in enumerate(records):
        label = labels[r["question_key"]]
        if label["question"] != new_inputs[i//2]["question"] or label["metadata"]["source_split"] != "train":
            raise ValueError("exact train label identity required")
        for arm, pred in (("legacy", baseline[i]["prediction"]), ("evidence_supply_v1", generated[i])):
            r[arm].update(metric.answer_scores(pred["generation"], label["metadata"]["gold_answer"],
                label["metadata"].get("gold_answer_aliases"), r[arm]["valid"]))
    fields = metric.FIELDS
    rng = np.random.default_rng(42)
    draws = rng.integers(0, 20, size=(10000, 20))
    report = {"schema_version": VERSION+"-assessment", "status": "COMPLETE_DEVELOPMENT_ONLY", "questions": 20, "candidates": 40,
              "metrics": {}, "optimizer_updates": 0, "fresh_confirmation_consumed": False, "ppo_launch_clearance": False,
              "interpretation": "paired consumed-train evidence-development comparison with additional retrieval cost; no independent generalization/PPO claim"}
    for f in fields:
        av = np.array([r["legacy"][f] for r in records], dtype=float).reshape(20, 2).mean(axis=1)
        bv = np.array([r["evidence_supply_v1"][f] for r in records], dtype=float).reshape(20, 2).mean(axis=1)
        report["metrics"][f] = {"legacy": float(av.mean()), "evidence_supply_v1": float(bv.mean()),
            "delta": float((bv-av).mean()), "paired_family_bootstrap95": np.quantile((bv-av)[draws].mean(axis=1), [.025, .975]).tolist()}
    coverage = []
    for old, new in zip(old_inputs, new_inputs):
        label = labels[new["question_key"]]["metadata"]
        surfaces = metric._canonical_gold_surfaces(label["gold_answer"], label.get("gold_answer_aliases"))
        norm = metric.canonical_answer_normalize
        boolean = norm(label["gold_answer"]) in {"yes", "no", "noanswer"}
        hit = lambda row: any(" "+norm(s)+" " in " "+norm(str(passage.get("contents", passage.get("text", "")) or "").strip()[:1200])+" "
                              for passage in row["spec"]["retrieved_passages"] for s in surfaces if norm(s))
        coverage.append({"question_key": new["question_key"], "boolean": boolean,
                         "legacy_surface_present": None if boolean else hit(old),
                         "new_surface_present": None if boolean else hit(new)})
    report["surface_coverage"] = {"nonboolean_questions": sum(not r["boolean"] for r in coverage),
        "legacy_present": sum(r["legacy_surface_present"] is True for r in coverage),
        "new_present": sum(r["new_surface_present"] is True for r in coverage),
        "interpretation": "answer surface only, not verified reasoning chain or answerability"}
    bank.write_rows(dest / "candidate_metrics.jsonl", records)
    bank.write_rows(dest / "coverage.jsonl", coverage)
    verify(directory)
    helper().require_bindings({"gold": p["gold"]["binding"]})
    bank.load_release(directory, VERSION+"-report")
    bank.finish(dest, report, ["structure_only.jsonl", "before_gold.json", "candidate_metrics.jsonl", "coverage.jsonl"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "assess"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--supply-dir", type=Path, default=SUPPLY)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.output_dir.resolve(), args.supply_dir.resolve())
    else:
        try:
            globals()[args.command](args.output_dir.resolve())
        except BaseException as exc:
            path = args.output_dir / ("assessment.exception.json" if args.command == "assess" else "exception.json")
            if not path.exists():
                bank.write_json(path, {"error_type": type(exc).__name__, "error": str(exc), "outputs_retained": True})
            raise
