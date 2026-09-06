"""One frozen train-only system-prompt candidate, K2, original 384-token budget."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gc
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

from scripts.prepare import source_quality_candidate_bank_v1 as bank
from kgproweight.training.reward_function import validate_source_gate_trajectory

ROOT = Path(os.environ.get("KGPW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
PARENT = ROOT / "outputs/audits/generation_length_384_512_paired_probe_20260906_v1"
DEFAULT_OUTPUT = ROOT / "outputs/audits/training_format_prompt_v1_paired_probe_20260906_v1"
VERSION = "training-format-prompt-v1-paired-development-probe"
EXPERIMENT = "TRAINING-FORMAT-PROMPT-V1-PAIRED-20260906-V1"
PROMPT_SOURCE = ROOT / "kgproweight/data/training_format_prompt_v1.py"
ASSESSMENT_PROTOCOL = ROOT / "outputs/audits/training_format_prompt_v1_assessment_20260906_v1/protocol.json"
ASSESSMENT_PROTOCOL_SHA = "38ee1c25d9a690b054428756df99d16151e027e060d2bfa241d778bdae15f9d3"
PROMPT_SOURCE_SHA = "0ecd28a9ad35448951916b6330d77adb69bcc204734bd6fe4cf47c62e0be617b"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def helpers():
    return load_file("frozen_length_probe_helpers", PARENT / "probe.executed.py")


def make_input(old, module, tokenizer):
    required = validate_source_gate_trajectory(SimpleNamespace(**old["spec"]), "", format_version="v2")["required_steps"]
    messages = module.clarify_training_format_messages_v1(old["messages"], required_steps=required)
    if len(messages) != len(old["messages"]) or messages[1:] != old["messages"][1:]:
        raise ValueError("prompt candidate modified original user/evidence messages")
    if messages[0]["role"] != "system" or messages[0]["content"] == old["messages"][0]["content"]:
        raise ValueError("expected exactly one new system prompt")
    new = deepcopy(old)
    new["messages"] = messages
    new["prompt"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    new["prompt_tokens"] = len(tokenizer(new["prompt"], add_special_tokens=False, truncation=False)["input_ids"])
    new["input_sha256"] = bank.input_hash(new)
    allowed = {"messages", "prompt", "prompt_tokens", "input_sha256"}
    if {k:v for k,v in new.items() if k not in allowed} != {k:v for k,v in old.items() if k not in allowed}:
        raise ValueError("nonprompt source field changed")
    change = {"question_key": old["question_key"], "required_steps": required,
              "legacy_input_sha256": old["input_sha256"], "new_input_sha256": new["input_sha256"],
              "legacy_system_sha256": bank.digest(old["messages"][0]), "new_system_sha256": bank.digest(messages[0]),
              "unchanged_user_messages_sha256": bank.digest(messages[1:]),
              "unchanged_spec_sha256": bank.digest(old["spec"]),
              "legacy_prompt_tokens": old["prompt_tokens"], "new_prompt_tokens": new["prompt_tokens"]}
    return new, change


def format_diagnostic(row, response):
    result = validate_source_gate_trajectory(SimpleNamespace(**row["spec"]), response, format_version="v2")
    steps = bank.parse_steps(response, known_kg=row["kg_subgraph"])
    indices = Counter(step.index for step in steps)
    bodies = Counter(re.sub(r"\s+", " ", step.raw_text).strip() for step in steps)
    repeated_fields = 0
    for step in steps:
        for label in ("Reasoning", "Knowledge Used", "Conclusion"):
            n = len(re.findall(rf"^[ \t]*{re.escape(label)}[ \t]*:", step.raw_text, flags=re.IGNORECASE|re.MULTILINE))
            repeated_fields += max(0, n-1)
    final_count = len(re.findall(r"\[\s*Final Answer\s*\]|^[ \t]*(?:\*\*)?Final Answer(?:\*\*)?[ \t]*[:：]",
                                response, flags=re.IGNORECASE|re.MULTILINE))
    return {**{k:result[k] for k in ("valid", "violations", "all_step_count", "required_steps", "contract_version")},
            "step_indices": [step.index for step in steps], "duplicate_step_indices": sum(n-1 for n in indices.values()),
            "duplicate_exact_step_bodies": sum(n-1 for n in bodies.values()), "repeated_step_fields": repeated_fields,
            "final_marker_count": final_count}


def summarize(records):
    n = len(records)
    transitions = Counter((r["format_legacy"]["valid"],r["format_prompt_v1"]["valid"]) for r in records)
    output = {"candidates": n, "questions": len({r["question_key"] for r in records}),
              "invalid_to_valid": transitions[(False,True)], "valid_to_invalid": transitions[(True,False)],
              "valid_to_valid": transitions[(True,True)], "invalid_to_invalid": transitions[(False,False)]}
    for arm in ("legacy", "prompt_v1"):
        fs = [r[f"format_{arm}"] for r in records]
        output[arm] = {"valid": sum(f["valid"] for f in fs), "valid_fraction": sum(f["valid"] for f in fs)/n if n else None,
                       "length_capped": sum(r[f"cap_{arm}"] for r in records), "eos": sum(r[f"eos_{arm}"] for r in records),
                       "response_tokens": sum(r[f"tokens_{arm}"] for r in records),
                       "step_count_histogram": dict(Counter(f["all_step_count"] for f in fs)),
                       "violations": dict(Counter(v for f in fs for v in f["violations"])),
                       "duplicate_step_indices": sum(f["duplicate_step_indices"] for f in fs),
                       "duplicate_exact_step_bodies": sum(f["duplicate_exact_step_bodies"] for f in fs),
                       "repeated_step_fields": sum(f["repeated_step_fields"] for f in fs),
                       "multiple_final_marker_candidates": sum(f["final_marker_count"] > 1 for f in fs)}
    before, after = output["legacy"]["response_tokens"], output["prompt_v1"]["response_tokens"]
    output["response_token_delta"] = after-before
    output["response_token_ratio"] = after/before if before else None
    return output


def prepare(directory):
    helper = helpers()
    directory.mkdir(parents=True, exist_ok=False)
    helper.status(directory, "PREPARING_FROZEN_PROMPT_VARIANT")
    parent = helper.verify(PARENT)
    if bank.file_sha(ASSESSMENT_PROTOCOL) != ASSESSMENT_PROTOCOL_SHA or bank.file_sha(PROMPT_SOURCE) != PROMPT_SOURCE_SHA:
        raise ValueError("pre-frozen assessment or prompt source changed")
    module = load_file("kgproweight.data.training_format_prompt_v1", PROMPT_SOURCE)
    if module.TRAINING_FORMAT_PROMPT_VERSION != "training-format-prompt-v1":
        raise ValueError("unexpected prompt candidate version")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(parent["policy_path"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    copied = {"inputs.jsonl": "legacy_inputs.jsonl", "selection.question_only.jsonl": "selection.question_only.jsonl", "baseline_384.jsonl": "baseline_384.jsonl"}
    for source, target in copied.items():
        (directory / target).write_bytes((PARENT / source).read_bytes())
        if bank.file_sha(directory / target) != parent["frozen_artifacts"][source]["sha256"]:
            raise ValueError("copied legacy artifact differs from frozen parent")
    new_rows, changes = [], []
    for row in bank.read_rows(directory / "legacy_inputs.jsonl"):
        bank.assert_gold_free(row)
        if bank.input_hash(row) != row["input_sha256"]:
            raise ValueError("legacy input modified")
        new, change = make_input(row, module, tokenizer)
        if new["prompt_tokens"] > parent["generation_384"]["max_input_tokens"]:
            raise ValueError("new prompt exceeds frozen input budget")
        new_rows.append(new); changes.append(change)
    if len(new_rows) != 60:
        raise ValueError("probe must retain exactly the prior fixed 60 questions")
    bank.write_rows(directory / "inputs.jsonl", new_rows)
    bank.write_rows(directory / "prompt_changes.jsonl", changes)
    for source, name in [(PROMPT_SOURCE, "training_format_prompt_v1.executed.py"), (Path(__file__), "probe.executed.py")]:
        (directory / name).write_bytes(source.read_bytes())
    artifacts = {name:bank.identity(directory / name) for name in [*copied.values(), "inputs.jsonl", "prompt_changes.jsonl",
                 "training_format_prompt_v1.executed.py", "probe.executed.py"]}
    protocol = {"schema_version": VERSION, "experiment_id": EXPERIMENT, "scope": "development_only_consumed_train_questions",
                "project_root": str(ROOT), "parent_dir": str(PARENT), "runtime_snapshot": parent["runtime_snapshot"],
                "parent_protocol": bank.identity(PARENT / "protocol.json"), "parent_manifest": bank.identity(PARENT / "manifest.json"),
                "source_bindings": parent["source_bindings"], "code_bindings": parent["code_bindings"],
                "helper_code": bank.identity(Path(helper.__file__).resolve()),
                "prompt_source": bank.identity(PROMPT_SOURCE), "frozen_artifacts": artifacts,
                "assessment_protocol": bank.identity(ASSESSMENT_PROTOCOL),
                "models": parent["models"], "policy_path": parent["policy_path"],
                "selection": {**parent["selection"], "reselection_performed": False, "exact_parent_identity_bytes_reused": True},
                "generation": parent["generation_384"], "prompt_version": module.TRAINING_FORMAT_PROMPT_VERSION,
                "single_research_variable": "one_system_prompt_candidate_only",
                "prompt_candidate_scope": "format clarification plus concise conclusions, anti-padding, and relevant-citation guidance; not merely a restatement of existing validator checks",
                "unchanged_messages": "all messages after first system; byte-identical content and role",
                "required_steps_contract": "frozen format-v2 validator on empty response; original Gold-free spec hard gate, not source-credit mask",
                "required_steps_histogram": dict(Counter(c["required_steps"] for c in changes)),
                "prefix_equality_required": False, "prefix_boundary": "Input prompt changes; same candidate seeds do not imply equal output prefixes.",
                "prompt_candidates": 1, "retry_or_sample_expansion": False, "all_failures_retained": True,
                "format_version": "v2", "reward_mask_gate_or_production_configuration_changed": False,
                "gold_access": False, "rearag_loaded": False, "optimizer_updates": 0,
                "ppo_launch_clearance": False, "independent_confirmation": False,
                "health_reference_valid_fraction": .90, "health_reference_is_clearance_gate": False,
                "evaluation_boundary": "GPU worker does not read Gold or compute EM/F1. Parent independently evaluates after generation."}
    helper.verify(PARENT)
    if bank.file_sha(directory / "training_format_prompt_v1.executed.py") != PROMPT_SOURCE_SHA:
        raise ValueError("prompt module changed during snapshot")
    helper.require_bindings({"assessment_protocol":protocol["assessment_protocol"],"prompt_source":protocol["prompt_source"]})
    bank.write_json(directory / "protocol.json", protocol)
    bank.write_json(directory / "prepared.json", {"protocol":bank.identity(directory / "protocol.json"),"experiment_id":EXPERIMENT})
    helper.status(directory, "FROZEN_READY_FOR_PROMPT_V1", candidates=120, questions=60, max_new_tokens=384,
                  required_steps_histogram=protocol["required_steps_histogram"])


def verify(directory, models=False):
    helper = helpers()
    prepared = json.loads((directory / "prepared.json").read_text())
    helper.require_bindings({"protocol": prepared["protocol"]})
    p = json.loads((directory / "protocol.json").read_text())
    for name in ("source_bindings", "code_bindings", "frozen_artifacts"):
        helper.require_bindings(p[name])
    helper.require_bindings({name:p[name] for name in ("parent_protocol", "parent_manifest", "prompt_source", "assessment_protocol", "helper_code")})
    if str(Path(helper.__file__).resolve()) != p["helper_code"]["path"]:
        raise ValueError("actual helper loaded from unbound path")
    if models:
        bank.validate_model(Path(p["project_root"]) / p["models"]["base_model"]["path"], p["models"]["base_model"])
        bank.validate_model(Path(p["policy_path"]), p["models"]["policy_tokenizer"])
    return p


def run(directory):
    helper = helpers()
    start = time.monotonic()
    helper.status(directory, "VERIFYING_FROZEN_CODE_MODELS_INPUTS_AND_BASELINE")
    p = verify(directory, models=True)
    executed = p["frozen_artifacts"]["probe.executed.py"]
    if str(Path(__file__).resolve()) != executed["path"] or bank.file_sha(Path(__file__)) != executed["sha256"]:
        raise ValueError("actual running producer is not the frozen executed copy")
    if Path(bank.__file__).resolve() != Path(p["runtime_snapshot"]) / "scripts/prepare/source_quality_candidate_bank_v1.py":
        raise ValueError("GPU execution must import frozen runtime snapshot")
    if (directory / "generations_prompt_v1.jsonl").exists():
        raise FileExistsError("no restart/overwrite of generation artifact")
    module = load_file("kgproweight.data.training_format_prompt_v1", directory / "training_format_prompt_v1.executed.py")
    torch = bank.require_cuda("cuda:0")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import transformers, peft
    tokenizer = AutoTokenizer.from_pretrained(p["policy_path"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = Path(p["project_root"]) / p["models"]["base_model"]["path"]
    helper.status(directory, "LOADING_FROZEN_BF16_SFT")
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, torch_dtype=torch.bfloat16).to("cuda:0")
    model = PeftModel.from_pretrained(model, p["policy_path"], local_files_only=True, is_trainable=False).eval()
    eos = list(bank._rollout_eos_token_ids(model, tokenizer))
    if eos != p["generation"]["eos_token_ids"]:
        raise ValueError("EOS changed")
    bank.write_json(directory / "execution_environment.json", {"python_executable":sys.executable, "torch":torch.__version__,
                    "transformers":transformers.__version__, "peft":peft.__version__, "gpu":torch.cuda.get_device_name(),
                    "dtype":"bf16", "batch_size":1, "max_new_tokens":384, "attention_implementation":model.config._attn_implementation,
                    "generation_config_before_overrides":model.generation_config.to_dict(), "prompt_module_loaded_from":str(Path(module.__file__).resolve()),
                    "frozen_bank_module_loaded_from":str(Path(bank.__file__).resolve()), "optimizer_updates":0})
    rows = bank.read_rows(directory / "inputs.jsonl")
    legacy = {r["question_key"]:r for r in bank.read_rows(directory / "legacy_inputs.jsonl")}
    baseline = {r["prediction"]["candidate_id"]:r["prediction"] for r in bank.read_rows(directory / "baseline_384.jsonl")}
    diagnostics = []
    generation_start = time.monotonic()
    with (directory / "generations_prompt_v1.jsonl").open("x") as output, (directory / "paired_diagnostics.jsonl").open("x") as diag:
        for row in rows:
            reconstructed, change = make_input(legacy[row["question_key"]], module, tokenizer)
            if reconstructed != row:
                raise ValueError("frozen new prompt cannot be reconstructed exactly")
            encoded = tokenizer(row["prompt"], add_special_tokens=False, return_tensors="pt", truncation=False, return_attention_mask=True)
            count = encoded["input_ids"].shape[-1]
            if count != row["prompt_tokens"]:
                raise ValueError("prompt tokenization changed")
            for k in range(2):
                cid = f"{row['question_key']}::k{k}"
                old = baseline[cid]
                seed = bank.candidate_seed(42, row["question_key"], k)
                if seed != old["seed"] or old["input_sha256"] != legacy[row["question_key"]]["input_sha256"]:
                    raise ValueError("baseline candidate seed or input mismatch")
                torch.manual_seed(seed)
                call_start = time.monotonic()
                with torch.inference_mode():
                    seq = model.generate(input_ids=encoded["input_ids"].to("cuda:0"), attention_mask=encoded["attention_mask"].to("cuda:0"),
                        do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=384,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=eos)
                raw = seq[0,count:]
                response = bank._trim_response_v2(raw,eos_token_ids=eos,pad_token_id=tokenizer.pad_token_id,max_new_tokens=384)
                ids = response.tolist()
                new = {"schema_version":VERSION+"-generation-row", "experiment_id":EXPERIMENT,
                       "candidate_id":cid, "dataset":row["dataset"], "qid":row["qid"], "candidate_index":k, "seed":seed,
                       "input_sha256":row["input_sha256"], "legacy_input_sha256":old["input_sha256"],
                       "probe_protocol_sha256":bank.file_sha(directory / "protocol.json"), "prompt_version":p["prompt_version"],
                       "generation_contract_sha256":bank.digest(p["generation"]), "baseline_prediction_sha256":bank.digest(old),
                       "policy_sha256":old["policy_sha256"], "base_model_identity_sha256":old["base_model_identity_sha256"],
                       "generation":tokenizer.decode(ids,skip_special_tokens=True), "raw_response_token_ids":raw.tolist(),
                       "response_token_ids":ids, "n_response_tokens":len(ids), "effective_eos_token_ids":eos,
                       "reached_max_new_tokens":bank._response_is_length_capped_v2(response,max_new_tokens=384,eos_token_ids=eos),
                       "elapsed_seconds":time.monotonic()-call_start}
                output.write(bank.canonical_json(new)+"\n"); output.flush()
                item = {"candidate_id":cid,"question_key":row["question_key"],"dataset":row["dataset"],
                        "original_graph_eligible":bool(row["m_graph"]), "required_steps":change["required_steps"],
                        "format_legacy":format_diagnostic(legacy[row["question_key"]],old["generation"]),
                        "format_prompt_v1":format_diagnostic(row,new["generation"]),
                        "tokens_legacy":old["n_response_tokens"],"tokens_prompt_v1":len(ids),
                        "cap_legacy":old["reached_max_new_tokens"],"cap_prompt_v1":new["reached_max_new_tokens"],
                        "eos_legacy":old["raw_response_token_ids"][-1] in eos,"eos_prompt_v1":new["raw_response_token_ids"][-1] in eos}
                if item["format_legacy"]["required_steps"] != change["required_steps"] or item["format_prompt_v1"]["required_steps"] != change["required_steps"]:
                    raise ValueError("required-step contract depends on candidate response")
                diagnostics.append(item)
                diag.write(bank.canonical_json(item)+"\n"); diag.flush()
                helper.status(directory,"GENERATING_PROMPT_V1_AND_CHECKING_FORMAT",completed=len(diagnostics),expected=120,
                              elapsed_seconds=time.monotonic()-generation_start)
    generation_elapsed = time.monotonic()-generation_start
    gpu = {"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()}
    del model,seq,raw,response,encoded
    gc.collect();torch.cuda.empty_cache()
    helper.status(directory,"VERIFYING_FINAL_BINDINGS_GPU_RELEASED",completed=120)
    verify(directory,models=True)
    if len(diagnostics)!=120:
        raise ValueError("incomplete fixed candidate population")
    report = {"schema_version":VERSION+"-report","experiment_id":EXPERIMENT,"status":"COMPLETE_DEVELOPMENT_ONLY",
              "overall":summarize(diagnostics),
              "by_dataset":{d:summarize([r for r in diagnostics if r["dataset"]==d]) for d in sorted({r["dataset"] for r in diagnostics})},
              "by_original_graph_eligibility":{str(g):summarize([r for r in diagnostics if r["original_graph_eligible"]==g]) for g in (False,True)},
              "by_dataset_and_original_graph_eligibility":{f"{d}::graph{int(g)}":summarize([r for r in diagnostics if r["dataset"]==d and r["original_graph_eligible"]==g])
                  for d,g in sorted({(r["dataset"],r["original_graph_eligible"]) for r in diagnostics})},
              "input_token_cost": {"legacy_questions":sum(r["prompt_tokens"] for r in legacy.values()),
                                   "prompt_v1_questions":sum(r["prompt_tokens"] for r in rows),"K":2},
              "gpu":gpu,"generation_elapsed_seconds":generation_elapsed,"elapsed_seconds":time.monotonic()-start,
              "all_input_model_code_baseline_hashes_verified_before_and_after":True,"all120_candidates_retained":True,
              "prefix_equality_required":False,"gold_access":False,"rearag_loaded":False,"optimizer_updates":0,
              "production_configuration_changed":False,"ppo_launch_clearance":False,"independent_confirmation":False,
              "health_reference_valid_fraction":.90,"health_reference_is_clearance_gate":False,
              "scientific_boundary":"One fixed training-only system prompt on consumed train identities at max384. No Gold/EM/F1 in GPU worker, no reward/gate change or production adoption."}
    bank.finish(directory,report,["protocol.json","prepared.json","legacy_inputs.jsonl","inputs.jsonl","selection.question_only.jsonl",
                "baseline_384.jsonl","prompt_changes.jsonl","training_format_prompt_v1.executed.py","probe.executed.py",
                "execution_environment.json","generations_prompt_v1.jsonl","paired_diagnostics.jsonl"])
    helper.status(directory,report["status"],completed=120,valid_prompt_v1=report["overall"]["prompt_v1"]["valid"])


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run"))
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    try:
        globals()[args.command](args.output_dir.resolve())
    except BaseException as exc:
        directory=args.output_dir.resolve()
        if directory.exists():
            bank.write_json(directory/"exception.json",{"status":"FAILED_OUTPUTS_RETAINED","error_type":type(exc).__name__,"error":str(exc),"optimizer_updates":0})
            helpers().status(directory,"FAILED_OUTPUTS_RETAINED",error=str(exc))
        raise
