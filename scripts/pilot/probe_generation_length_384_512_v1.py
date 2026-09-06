"""Append-only development format probe: frozen 384 baselines versus 512 tokens.

No Gold, ReaRAG, reward changes, optimizer updates, or automatic PPO clearance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from scripts.prepare import source_quality_candidate_bank_v1 as bank
from kgproweight.training.reward_function import validate_source_gate_trajectory

ROOT = Path(os.environ.get("KGPW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
DEFAULT_PARENT = ROOT / "outputs/audits/normalization_representative_bank_v1_seed42_20260906_r2"
DEFAULT_OUTPUT = ROOT / "outputs/audits/generation_length_384_512_paired_probe_20260906_v1"
VERSION = "generation-length-384-512-paired-development-probe-v1"
EXPERIMENT = "GENERATION-LENGTH-384-512-PAIRED-20260906-V1"
QUOTAS = [("hotpotqa", 0, 20), ("musique", 0, 20), ("2wikimultihopqa", 1, 16), ("2wikimultihopqa", 0, 4)]


def select_inputs(rows):
    if len({bank.key(r) for r in rows}) != len(rows):
        raise ValueError("duplicate question identity")
    selected, families = [], set()
    for dataset, graph, count in QUOTAS:
        pool = [r for r in rows if r["dataset"] == dataset and int(r["m_graph"]) == graph]
        pool.sort(key=lambda r: bank.digest(["format-budget-paired-probe-v1", 42,
                                             r["dataset"], r["qid"], r["family_sha256"]]))
        picked = []
        for row in pool:
            if row["family_sha256"] in families:
                continue
            picked.append(row)
            families.add(row["family_sha256"])
            if len(picked) == count:
                break
        if len(picked) != count:
            raise ValueError("insufficient unique-family quota")
        selected.extend(picked)
    return sorted(selected, key=bank.key)


def prefix_check(old, new, eos=(128001, 128009)):
    """Compare actual generated token IDs, including EOS; never decoded strings."""
    before, after = old["raw_response_token_ids"], new["raw_response_token_ids"]
    if not before or len(before) > 384 or not after or len(after) > 512:
        raise ValueError("empty or over-budget response")
    ended = before[-1] in eos
    if any(token in eos for token in before[:-1]):
        raise ValueError("tokens after baseline EOS")
    if not ended and len(before) != 384:
        raise ValueError("baseline neither EOS nor length capped")
    equal = before == after if ended else before == after[:384]
    mismatch = next((i for i, (a, b) in enumerate(zip(before, after)) if a != b), None)
    if mismatch is None and not equal:
        mismatch = min(len(before), len(after))
    return {"match": equal, "contract": "whole_sequence_equal_after_old_eos" if ended else "first_384_tokens_equal",
            "old_stopped_at_eos": ended, "old_length_capped": not ended,
            "first_mismatch_token_index": mismatch}


def compact_format(row, prediction):
    result = validate_source_gate_trajectory(SimpleNamespace(**row["spec"]), prediction["generation"], format_version="v2")
    return {k: result[k] for k in ("valid", "violations", "all_step_count", "required_steps", "contract_version")}


def paired_summary(records):
    n = len(records)
    transitions = Counter((r["format_384"]["valid"], r["format_512"]["valid"]) for r in records)
    old_tokens = sum(r["tokens_384"] for r in records)
    new_tokens = sum(r["tokens_512"] for r in records)
    violations = {str(cap): dict(Counter(v for r in records for v in r[f"format_{cap}"]["violations"])) for cap in (384, 512)}
    return {"candidates": n, "questions": len({r["question_key"] for r in records}),
            "valid_384": sum(r["format_384"]["valid"] for r in records),
            "valid_512": sum(r["format_512"]["valid"] for r in records),
            "valid_fraction_384": sum(r["format_384"]["valid"] for r in records) / n if n else None,
            "valid_fraction_512": sum(r["format_512"]["valid"] for r in records) / n if n else None,
            "invalid_to_valid": transitions[(False, True)], "valid_to_invalid": transitions[(True, False)],
            "valid_to_valid": transitions[(True, True)], "invalid_to_invalid": transitions[(False, False)],
            "prefix_matches": sum(r["prefix"]["match"] for r in records),
            "cap_384": sum(r["cap_384"] for r in records), "cap_512": sum(r["cap_512"] for r in records),
            "eos_384": sum(r["eos_384"] for r in records), "eos_512": sum(r["eos_512"] for r in records),
            "tokens_384": old_tokens, "tokens_512": new_tokens, "token_delta": new_tokens-old_tokens,
            "token_ratio_512_over_384": new_tokens/old_tokens if old_tokens else None,
            "violations": violations}


def status(directory, state, **details):
    event = {"time_utc": datetime.now(timezone.utc).isoformat(), "status": state,
             "pid": os.getpid(), "optimizer_updates": 0, **details}
    data = bank.canonical_json(event) + "\n"
    with (directory / "events.jsonl").open("a") as handle:
        handle.write(data)
    temporary = directory / "status.json.tmp"
    temporary.write_text(data)
    temporary.replace(directory / "status.json")
    print(data.strip(), flush=True)


def require_bindings(bindings):
    for name, info in bindings.items():
        if bank.file_sha(Path(info["path"])) != info["sha256"]:
            raise ValueError(f"frozen binding changed: {name}")


def prepare(parent_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    status(output_dir, "PREPARING_IDENTITY_ONLY_SELECTION")
    parent = json.loads((parent_dir / "protocol.json").read_text())
    # Check frozen safe inputs, generation manifests and code; never parse label sources.
    require_bindings(parent["code_bindings"])
    for filename, key in [("inputs.all.jsonl", "input_sha256"), ("selection.question_only.jsonl", "selection_sha256")]:
        if bank.file_sha(parent_dir / filename) != parent[key]:
            raise ValueError("representative parent input changed")
    original_input_dir = Path(parent["parent_bindings"]["inputs"]["path"]).parent
    original_generated_dir = Path(parent["parent_bindings"]["generation"]["path"]).parent
    for name in ("inputs", "generation", "assignments"):
        require_bindings({name: parent["parent_bindings"][name]})
    input_manifests = [bank.load_release(original_input_dir, bank.PREPARE_VERSION),
                       bank.load_release(parent_dir / "new_inputs", bank.PREPARE_VERSION)]
    gen_dirs = [original_generated_dir, parent_dir / "generated"]
    gen_manifests = [bank.load_release(path, bank.GENERATION_VERSION) for path in gen_dirs]
    rows = select_inputs(bank.read_rows(parent_dir / "inputs.all.jsonl"))
    old_selections = {r["question_key"]: r for r in bank.read_rows(parent_dir / "selection.question_only.jsonl")}
    assignments = bank.read_rows(Path(parent["parent_bindings"]["assignments"]["path"]))
    forbidden = {r["family_sha256"] for r in assignments if r["split"] != "train"}
    if {r["family_sha256"] for r in rows} & forbidden:
        raise ValueError("consumed calibration/confirmation family in probe")
    ledger = bank.resolve(input_manifests[0]["source_bindings"]["protected_ledger"], original_input_dir, ROOT)
    overlap = bank.isolation(rows, bank.read_rows(ledger))
    generation_maps = [{r["candidate_id"]: r for r in bank.read_rows(path / "generations.jsonl")} for path in gen_dirs]
    baseline, selection, bindings = [], [], {}
    for name, path in [("parent_protocol", parent_dir / "protocol.json"), ("parent_manifest", parent_dir / "manifest.json"),
                       ("parent_inputs", parent_dir / "inputs.all.jsonl"), ("parent_selection", parent_dir / "selection.question_only.jsonl"),
                       ("old_assignments", Path(parent["parent_bindings"]["assignments"]["path"])), ("protected_ledger", ledger)]:
        bindings[name] = bank.identity(path)
    for i, (path, manifest, im) in enumerate(zip(gen_dirs, gen_manifests, input_manifests)):
        input_dir = original_input_dir if i == 0 else parent_dir / "new_inputs"
        if manifest["bank_manifest_sha256"] != bank.file_sha(input_dir / "manifest.json"):
            raise ValueError("generation bank parent changed")
        if im["generation"] != parent["generation"]:
            raise ValueError("mixed baseline generation contracts")
        for name, p in [("manifest", path / "manifest.json"), ("rows", path / "generations.jsonl"),
                        ("input_manifest", input_dir / "manifest.json")]:
            bindings[f"baseline_{i}_{name}"] = bank.identity(p)
    for row in rows:
        bank.assert_gold_free(row)
        if bank.input_hash(row) != row["input_sha256"]:
            raise ValueError("selected input hash mismatch")
        idx = 0 if old_selections[row["question_key"]]["reuse"] else 1
        im = input_manifests[idx]
        selection.append({k: row[k] for k in ("dataset", "qid", "question_key", "question_sha256", "family_sha256", "m_graph")})
        selection[-1]["baseline_origin"] = "old_frozen_train" if idx == 0 else "r2_new_generation"
        for k in range(2):
            cid = f"{row['question_key']}::k{k}"
            pred = generation_maps[idx][cid]
            expected = {"candidate_id": cid, "dataset": row["dataset"], "qid": row["qid"], "candidate_index": k,
                        "seed": bank.candidate_seed(42, row["question_key"], k), "input_sha256": row["input_sha256"],
                        "generation_contract_sha256": bank.digest(parent["generation"]),
                        "policy_sha256": im["source_bindings"]["policy"]["sha256"],
                        "base_model_identity_sha256": bank.digest(im["base_model"]),
                        "bank_manifest_sha256": gen_manifests[idx]["bank_manifest_sha256"]}
            if any(pred.get(key) != value for key, value in expected.items()):
                raise ValueError("baseline identity/model/seed mismatch")
            prefix_check(pred, pred, parent["generation"]["eos_token_ids"])
            baseline.append({"prediction": pred, "origin": bank.identity(gen_dirs[idx] / "generations.jsonl"),
                             "prediction_sha256": bank.digest(pred)})
    bank.write_rows(output_dir / "inputs.jsonl", rows)
    bank.write_rows(output_dir / "selection.question_only.jsonl", selection)
    bank.write_rows(output_dir / "baseline_384.jsonl", baseline)
    executed = output_dir / "probe.executed.py"
    executed.write_bytes(Path(__file__).read_bytes())
    new_contract = deepcopy(parent["generation"])
    new_contract["max_new_tokens"] = 512
    source_bindings = input_manifests[0]["source_bindings"]
    policy = Path(parent["policy_path"])
    for name in ("policy", "policy_config"):
        filename = "adapter_model.safetensors" if name == "policy" else "adapter_config.json"
        bindings[name] = {"path": str(policy / filename), "sha256": source_bindings[name]["sha256"]}
    bindings["base_generation_config"] = bank.identity(ROOT / parent["models"]["base_model"]["path"] / "generation_config.json")
    protocol = {"schema_version": VERSION, "experiment_id": EXPERIMENT, "project_root": str(ROOT),
                "parent_dir": str(parent_dir), "runtime_snapshot": str(parent_dir / "runtime_code"),
                "scope": "development_only_already_consumed_normalization_train_population",
                "selection": {"quotas": QUOTAS, "seed": 42, "K": 2, "questions": 60, "candidates": 120,
                              "sort_hash": "sha256 canonical JSON [format-budget-paired-probe-v1,42,dataset,qid,family_sha256]",
                              "outcome_validity_or_generation_used_for_selection": False},
                "generation_384": parent["generation"], "generation_512": new_contract,
                "source_bindings": bindings, "code_bindings": {**parent["code_bindings"], "probe": bank.identity(executed)},
                "models": {k: parent["models"][k] for k in ("base_model", "policy_tokenizer")}, "policy_path": str(policy),
                "frozen_artifacts": {name: bank.identity(output_dir / name) for name in
                                     ("inputs.jsonl", "selection.question_only.jsonl", "baseline_384.jsonl")},
                "protected_overlap": overlap, "consumed_family_overlap": 0,
                "format_contract": "v2", "production_configuration_changed": False,
                "prefix_rule": "all120 required; old EOS => whole raw sequence identical; old cap => first384 identical",
                "failure_rule": "retain every generation and diagnostic; no seed replacement or sample expansion; no length-only claim if any mismatch",
                "health_reference_valid_fraction": .90, "health_reference_is_clearance_gate": False,
                "ppo_launch_clearance": False, "independent_confirmation": False,
                "optimizer_updates": 0, "gold_access": False, "rearag_loaded": False}
    require_bindings(bindings)
    bank.write_json(output_dir / "protocol.json", protocol)
    bank.write_json(output_dir / "prepared.json", {"protocol": bank.identity(output_dir / "protocol.json"),
                    "experiment_id": EXPERIMENT, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    status(output_dir, "FROZEN_READY_FOR_512", questions=60, candidates=120,
           baseline_origins=dict(Counter(r["baseline_origin"] for r in selection)))


def verify(directory, models=False):
    prepared = json.loads((directory / "prepared.json").read_text())
    require_bindings({"protocol": prepared["protocol"]})
    protocol = json.loads((directory / "protocol.json").read_text())
    for key in ("source_bindings", "code_bindings", "frozen_artifacts"):
        require_bindings(protocol[key])
    if models:
        root = Path(protocol["project_root"])
        bank.validate_model(root / protocol["models"]["base_model"]["path"], protocol["models"]["base_model"])
        bank.validate_model(Path(protocol["policy_path"]), protocol["models"]["policy_tokenizer"])
    return protocol


def run(directory):
    start = time.monotonic()
    status(directory, "VERIFYING_FROZEN_INPUTS_MODELS_AND_CODE")
    protocol = verify(directory, models=True)
    if Path(bank.__file__).resolve() != Path(protocol["runtime_snapshot"]) / "scripts/prepare/source_quality_candidate_bank_v1.py":
        raise ValueError("run must import the frozen parent runtime snapshot")
    if (directory / "generations_512.jsonl").exists():
        raise FileExistsError("append-only experiment cannot restart or overwrite generations")
    torch = bank.require_cuda("cuda:0")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import transformers, peft
    policy = Path(protocol["policy_path"])
    base = Path(protocol["project_root"]) / protocol["models"]["base_model"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(policy, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    status(directory, "LOADING_FROZEN_BF16_SFT")
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, torch_dtype=torch.bfloat16).to("cuda:0")
    model = PeftModel.from_pretrained(model, policy, local_files_only=True, is_trainable=False).eval()
    eos = list(bank._rollout_eos_token_ids(model, tokenizer))
    if eos != protocol["generation_384"]["eos_token_ids"]:
        raise ValueError("effective EOS changed")
    config = model.generation_config
    if config.forced_eos_token_id is not None or config.cache_implementation is not None:
        raise ValueError("unexpected length-dependent logits processor or cache allocation")
    environment = {"python_executable": sys.executable, "torch": torch.__version__, "transformers": transformers.__version__,
                   "peft": peft.__version__, "gpu": torch.cuda.get_device_name(), "dtype": "bf16", "batch_size": 1,
                   "generation_config_before_overrides": config.to_dict(),
                   "attention_implementation": model.config._attn_implementation,
                   "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
                   "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                   "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                   "parent_environment": str(Path(protocol["parent_dir"]) / "execution_environment.actual_kgpaper.json")}
    bank.write_json(directory / "execution_environment.json", environment)
    rows = bank.read_rows(directory / "inputs.jsonl")
    baseline = {r["prediction"]["candidate_id"]: r["prediction"] for r in bank.read_rows(directory / "baseline_384.jsonl")}
    diagnostics = []
    generation_start = time.monotonic()
    with (directory / "generations_512.jsonl").open("x") as output, (directory / "paired_diagnostics.jsonl").open("x") as diag:
        for row in rows:
            prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", truncation=False, return_attention_mask=True)
            count = encoded["input_ids"].shape[-1]
            if prompt != row["prompt"] or count != row["prompt_tokens"] or count > protocol["generation_512"]["max_input_tokens"]:
                raise ValueError("frozen prompt/tokenization changed")
            for k in range(2):
                cid = f"{row['question_key']}::k{k}"
                old = baseline[cid]
                seed = bank.candidate_seed(42, row["question_key"], k)
                if seed != old["seed"]:
                    raise ValueError("candidate seed changed")
                torch.manual_seed(seed)
                call_start = time.monotonic()
                with torch.inference_mode():
                    sequence = model.generate(input_ids=encoded["input_ids"].to("cuda:0"), attention_mask=encoded["attention_mask"].to("cuda:0"),
                        do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=512,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=eos)
                raw = sequence[0, count:]
                response = bank._trim_response_v2(raw, eos_token_ids=eos, pad_token_id=tokenizer.pad_token_id, max_new_tokens=512)
                ids = response.tolist()
                new = {"candidate_id": cid, "dataset": row["dataset"], "qid": row["qid"], "candidate_index": k,
                       "seed": seed, "input_sha256": row["input_sha256"], "generation": tokenizer.decode(ids, skip_special_tokens=True),
                       "response_token_ids": ids, "raw_response_token_ids": raw.tolist(), "n_response_tokens": len(ids),
                       "effective_eos_token_ids": eos, "reached_max_new_tokens": bank._response_is_length_capped_v2(response, max_new_tokens=512, eos_token_ids=eos),
                       "generation_contract_sha256": bank.digest(protocol["generation_512"]), "baseline_prediction_sha256": bank.digest(old),
                       "policy_sha256": old["policy_sha256"], "base_model_identity_sha256": old["base_model_identity_sha256"],
                       "elapsed_seconds": time.monotonic()-call_start}
                output.write(bank.canonical_json(new)+"\n"); output.flush()
                item = {"candidate_id": cid, "question_key": row["question_key"], "dataset": row["dataset"], "original_graph_eligible": bool(row["m_graph"]),
                        "seed": seed, "prefix": prefix_check(old, new, eos),
                        "format_384": compact_format(row, old), "format_512": compact_format(row, new),
                        "tokens_384": old["n_response_tokens"], "tokens_512": len(ids),
                        "cap_384": old["reached_max_new_tokens"], "cap_512": new["reached_max_new_tokens"],
                        "eos_384": old["raw_response_token_ids"][-1] in eos, "eos_512": new["raw_response_token_ids"][-1] in eos}
                diagnostics.append(item)
                diag.write(bank.canonical_json(item)+"\n"); diag.flush()
                status(directory, "GENERATING_512_AND_CHECKING_PAIRED_FORMAT", completed=len(diagnostics), expected=120,
                       prefix_matches=sum(r["prefix"]["match"] for r in diagnostics), elapsed_seconds=time.monotonic()-generation_start)
    gpu = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    del model, sequence, raw, response, encoded
    gc.collect(); torch.cuda.empty_cache()
    status(directory, "VERIFYING_FINAL_BINDINGS_GPU_RELEASED", completed=120)
    verify(directory, models=True)
    if len(diagnostics) != 120:
        raise ValueError("incomplete fixed candidate population")
    matches = sum(r["prefix"]["match"] for r in diagnostics)
    report = {"schema_version": VERSION + "-report", "experiment_id": protocol["experiment_id"],
              "status": "COMPLETE_DEVELOPMENT_ONLY" if matches == 120 else "FAILED_PREFIX_REPRODUCIBILITY_RETAINED",
              "all_120_prefixes_match": matches == 120, "length_only_paired_interpretation_allowed": matches == 120,
              "overall": paired_summary(diagnostics),
              "by_dataset": {d: paired_summary([r for r in diagnostics if r["dataset"] == d]) for d in sorted({r["dataset"] for r in diagnostics})},
              "by_original_graph_eligibility": {str(g): paired_summary([r for r in diagnostics if r["original_graph_eligible"] == g]) for g in (False, True)},
              "by_dataset_and_original_graph_eligibility": {f"{d}::graph{int(g)}": paired_summary([r for r in diagnostics if r["dataset"] == d and r["original_graph_eligible"] == g])
                  for d, g in sorted({(r["dataset"], r["original_graph_eligible"]) for r in diagnostics})},
              "input_model_code_baseline_hashes_verified_before_and_after": True,
              "gpu": gpu, "elapsed_seconds": time.monotonic()-start,
              "generation_elapsed_seconds": time.monotonic()-generation_start,
              "health_reference_valid_fraction": .90, "health_reference_is_clearance_gate": False,
              "production_configuration_changed": False, "optimizer_updates": 0, "ppo_launch_clearance": False,
              "gold_access": False, "rearag_loaded": False, "independent_confirmation": False,
              "scientific_boundary": "Development format diagnosis on consumed normalization-train identities. "
                                     "No reward/gate change, no EM/F1 measurement, no automatic production length change or PPO clearance."}
    if matches != 120:
        bank.write_json(directory / "FAILED.json", {"reason": "not all baseline prefixes reproduced", "prefix_matches": matches,
                                                    "all_outputs_retained": True, "no_replacement_or_expansion": True})
    bank.finish(directory, report, ["protocol.json", "prepared.json", "inputs.jsonl", "selection.question_only.jsonl", "baseline_384.jsonl",
                                   "probe.executed.py", "execution_environment.json", "generations_512.jsonl", "paired_diagnostics.jsonl"])
    status(directory, report["status"], completed=120, prefix_matches=matches, valid_512=report["overall"]["valid_512"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.parent_dir.resolve(), args.output_dir.resolve())
        else:
            run(args.output_dir.resolve())
    except BaseException as exc:
        directory = args.output_dir.resolve()
        if directory.exists():
            failure = {"status": "FAILED_OUTPUTS_RETAINED", "error_type": type(exc).__name__, "error": str(exc), "optimizer_updates": 0}
            bank.write_json(directory / "exception.json", failure)
            status(directory, failure["status"], error=str(exc))
        raise
