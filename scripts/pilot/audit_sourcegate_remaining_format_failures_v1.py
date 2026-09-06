"""Gold-free, read-only diagnosis of frozen format-v2 failures and stop events."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import re
from types import SimpleNamespace

from kgproweight.training.reward_function import validate_source_gate_trajectory
from kgproweight.data.parsers import extract_final_answer
from kgproweight.data.prompts import build_rl_messages, SFT_SYSTEM_PROMPT
from transformers import AutoTokenizer, GenerationConfig
from transformers.generation.logits_process import MinLengthLogitsProcessor, MinNewTokensLengthLogitsProcessor

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(binding, base):
    path = Path(binding["path"])
    for candidate in ([path] if path.is_absolute() else [base / path, ROOT / path]):
        if candidate.is_file() and sha(candidate) == binding["sha256"]:
            return candidate
    raise ValueError(f"bound file missing/hash mismatch: {path}")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def detail(response, validity):
    steps = validity["steps"]
    reasons, details = [], []
    if len(steps) < validity["required_steps"]:
        reasons.append("too_few_steps")
    if [s.index for s in steps] != list(range(1, len(steps) + 1)):
        reasons.append("nonsequential_steps")
    if validity["all_step_count"] > 5:
        reasons.append("too_many_steps")
    for s in steps:
        fields = {label: len(re.findall(rf"^[ \t]*{re.escape(label)}[ \t]*:", s.raw_text,
                                       flags=re.I | re.M)) for label in ("Reasoning", "Knowledge Used", "Conclusion")}
        if "Reasoning:" not in s.raw_text:
            reasons.append("missing_exact_reasoning_field")
            content_length = 0
        else:
            content_length = len(re.split(r"Knowledge Used:|Conclusion:|Final Answer:",
                                          s.raw_text.split("Reasoning:", 1)[1])[0].strip())
            if content_length < 20:
                reasons.append("short_reasoning_content")
        details.append({"step": s.index, "field_counts": fields, "reasoning_chars": content_length})
    final_count = len(re.findall(r"\[\s*Final Answer\s*\]|^[ \t]*(?:\*\*)?Final Answer(?:\*\*)?[ \t]*[:：]",
                                response, flags=re.I | re.M))
    if final_count == 0:
        reasons.append("missing_final_marker")
    elif final_count > 1:
        reasons.append("duplicate_final_marker")
    for reason in validity["violations"]:
        if reason not in {"invalid_step_sequence_content_or_minimum", "too_many_steps", "final_field_count_not_one"}:
            reasons.append(re.sub(r"step_\d+_", "step_N_", reason))
    return sorted(set(reasons)), details, final_count


def audit(output):
    output.mkdir(parents=True, exist_ok=False)
    original = ROOT / "outputs/audits/source_quality_candidate_bank_v1_inputs_seed42_tensorboard_v1"
    generated = ROOT / "outputs/audits/source_quality_candidate_bank_v1_generated_seed42_local4090_v1"
    scored = ROOT / "outputs/audits/source_quality_candidates_format_v2_scored_local_seed42"
    bank = json.loads((original / "manifest.json").read_text())
    gen = json.loads((generated / "manifest.json").read_text())
    score = json.loads((scored / "manifest.json").read_text())
    inputs_path = bound(bank["outputs"]["inputs.jsonl"], original)
    generation_path = bound(gen["outputs"]["generations.jsonl"], generated)
    scored_path = bound(score["bank"], scored)
    assert gen["bank_manifest_sha256"] == sha(original / "manifest.json")
    inputs = {r["question_key"]: r for r in read_rows(inputs_path)}
    scored_rows = {r["candidate_id"]: r for r in read_rows(scored_path)}
    tokenizer_dir = ROOT / bank["policy_tokenizer"]["path"]
    tokenizer_files = {name: bound(binding, tokenizer_dir) for name, binding in bank["policy_tokenizer"]["files"].items()}
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    eos = bank["generation"]["eos_token_ids"]
    assert eos == [128001, 128009]
    records = []
    prompt_matches = 0
    for key, row in inputs.items():
        messages = build_rl_messages(row["question"], row["retrieved_passages"], row["kg_subgraph"], top_k=10, max_kg_triples=12)
        assert messages == row["messages"]
        assert tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) == row["prompt"]
        prompt_matches += 1
    for prediction in read_rows(generation_path):
        row = inputs[prediction["dataset"] + "::" + prediction["qid"]]
        assert prediction["input_sha256"] == row["input_sha256"]
        ids = prediction["response_token_ids"]
        assert ids == prediction["raw_response_token_ids"]
        assert len(ids) == prediction["n_response_tokens"] <= 384
        assert tokenizer.decode(ids, skip_special_tokens=True) == prediction["generation"]
        assert prediction["effective_eos_token_ids"] == eos
        assert all(value not in eos for value in ids[:-1])
        capped = len(ids) == 384 and ids[-1] not in eos
        assert prediction["reached_max_new_tokens"] is capped
        assert capped or ids[-1] in eos
        validity = validate_source_gate_trajectory(SimpleNamespace(**row["spec"]), prediction["generation"], format_version="v2")
        stored = scored_rows[prediction["candidate_id"]]
        assert stored["trajectory_valid"] is validity["valid"]
        assert stored["format_validation"]["violations"] == validity["violations"]
        reasons, steps, final_count = detail(prediction["generation"], validity)
        assert bool(reasons) is (not validity["valid"])
        completed_prefix_bad = any(any(n != 1 for n in step["field_counts"].values()) or step["reasoning_chars"] < 20 for step in steps[:-1])
        irreversible_prefix = validity["all_step_count"] > 5 or completed_prefix_bad or "nonsequential_steps" in reasons or final_count > 1
        records.append({"candidate_id": prediction["candidate_id"], "dataset": row["dataset"],
            "m_graph": row["m_graph"], "valid": validity["valid"], "length_capped": capped,
            "last_token": ids[-1], "eos_ended": ids[-1] in eos, "n_tokens": len(ids),
            "steps": validity["all_step_count"], "required_steps": validity["required_steps"],
            "final_marker_count": final_count, "has_extracted_answer": bool(extract_final_answer(prediction["generation"])),
            "reasons": reasons, "original_violations": validity["violations"], "step_details": steps,
            "append_only_extension_cannot_fix_existing_prefix": bool(irreversible_prefix),
            "tail_excerpt": prediction["generation"][-200:] if not validity["valid"] else None})
    invalid = [r for r in records if not r["valid"]]
    combinations = Counter((r["length_capped"], tuple(r["reasons"])) for r in invalid)
    min_lengths = []
    for threshold in (32, 64, 96, 128, 160, 192, 256):
        affected = [r for r in records if r["eos_ended"] and r["n_tokens"] - 1 < threshold]
        min_lengths.append({"min_new_tokens": threshold, "existing_valid_eos_would_be_suppressed": sum(r["valid"] for r in affected),
                            "existing_invalid_eos_would_be_suppressed": sum(not r["valid"] for r in affected),
                            "boundary": "Observed first EOS support only; changed rollout may still fail or worsen. Not a predicted repair count."})
    report = {"schema_version": "sourcegate-remaining-format-failures-v1",
        "experiment_id": "SOURCEGATE-REMAINING-FORMAT-FAILURES-20260905-V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "COMPLETE_READ_ONLY_DIAGNOSTIC",
        "n_candidates": len(records), "valid": len(records)-len(invalid), "invalid": len(invalid),
        "invalid_eos_ended": sum(r["eos_ended"] for r in invalid), "invalid_length_capped": sum(r["length_capped"] for r in invalid),
        "valid_eos_ended": sum(r["valid"] and r["eos_ended"] for r in records),
        "valid_length_capped": sum(r["valid"] and r["length_capped"] for r in records),
        "nonexclusive_invalid_reasons": dict(Counter(reason for r in invalid for reason in r["reasons"])),
        "reason_by_stop": {name: dict(Counter(reason for r in invalid if r["length_capped"] is cap for reason in r["reasons"])) for name,cap in (("length_cap",True),("model_eos",False))},
        "exclusive_combinations": [{"length_capped": cap, "reasons": list(reasons), "n": n} for (cap,reasons),n in combinations.most_common()],
        "shortfall_pairs": dict(Counter(f"{r['steps']}<{r['required_steps']}" for r in invalid if "too_few_steps" in r["reasons"])),
        "only_step_shortfall": sum(r["reasons"] == ["too_few_steps"] for r in invalid),
        "length_capped_with_irreversible_existing_prefix": sum(r["length_capped"] and r["append_only_extension_cannot_fix_existing_prefix"] for r in invalid),
        "min_new_tokens_observed_support": min_lengths,
        "all_generation_identity_decode_eos_checks_passed": True, "frozen_prompts_equal_current_ppo_rendering": prompt_matches,
        "generation_contract": bank["generation"],
        "eos_stop_counts": dict(Counter(str(r["last_token"]) for r in records if r["eos_ended"])),
        "prompt_findings": {"contains_stop_after_final_marker_wording": "Stop generating after [Final Answer]." in SFT_SYSTEM_PROMPT,
                            "explicit_step_minimum_in_system_prompt": bool(re.search(r"(?:at least|minimum|between)\s+[235]", SFT_SYSTEM_PROMPT, re.I)),
                            "prompt_modified": False},
        "hf_min_length_implementation": {"MinLengthLogitsProcessor": inspect.getsource(MinLengthLogitsProcessor.__call__),
                                          "MinNewTokensLengthLogitsProcessor": inspect.getsource(MinNewTokensLengthLogitsProcessor.__call__),
                                          "default_min_length": GenerationConfig().min_length,
                                          "default_min_new_tokens": GenerationConfig().min_new_tokens},
        "gold_access": False, "generated_or_modified_candidates": 0, "runtime_modified": False,
        "bindings": {str(p.relative_to(ROOT)): {"path": str(p), "sha256": sha(p)} for p in [
            inputs_path, generation_path, scored_path, original/'manifest.json', generated/'manifest.json', scored/'manifest.json',
            Path(__file__).resolve(), ROOT/'kgproweight/training/reward_function.py', ROOT/'kgproweight/training/phase3_ppo.py',
            ROOT/'kgproweight/data/parsers.py', ROOT/'kgproweight/data/prompts.py', *tokenizer_files.values()]}}
    with (output/'candidate_diagnostics.jsonl').open('x') as f:
        for row in records:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    with (output/'report.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,sort_keys=True);f.write('\n')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/audits/sourcegate_remaining_format_failures_20260905_v1')
    result=audit(parser.parse_args().output_dir)
    print(json.dumps({k:result[k] for k in ('status','invalid','invalid_eos_ended','invalid_length_capped','nonexclusive_invalid_reasons','shortfall_pairs','length_capped_with_irreversible_existing_prefix','min_new_tokens_observed_support')},ensure_ascii=False,indent=2))
