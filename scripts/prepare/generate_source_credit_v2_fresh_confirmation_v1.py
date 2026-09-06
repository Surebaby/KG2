#!/usr/bin/env python
"""Frozen fresh132 Strong-SFT K4 plus greedy generation; never labels or PPO.

Every committed candidate is immutable. Interrupted attempt files are retained;
resume verifies the exact committed prefix and uses the original slot seed.
The final JSONL is published only after all 660 candidates are verified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.prepare import source_quality_candidate_bank_v1 as bank

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SCHEMA = "source-credit-v2-fresh-confirmation-protocol-v1"
GENERATION_SCHEMA = "source-credit-v2-fresh-confirmation-generations-v1"
ROW_SCHEMA = "source-credit-v2-fresh-confirmation-generation-row-v1"
INPUT_MANIFEST_SHA256 = "e877116411d20a062c4e8a7929c08154c51df0544cdc0c312ac583adb9bab26b"
INPUTS_SHA256 = "c56b68b82a7f1e0460dc2237f3765d1ded21e6eabb4583bc4ace5e6f0bac2a2f"
SOURCE_MANIFEST_SHA256 = "e458e97185e6516c4dcd133bc79dd13195df6943c8861d80d2a9494929baa161"
GENERATION_CODE_FILES = sorted(set(bank.CODE_FILES + [
    "scripts/prepare/generate_source_credit_v2_fresh_confirmation_v1.py",
]))
GENERATION_FIXED = {
    "candidates_per_question": 4, "greedy_per_question": 1,
    "greedy_candidate_index": 4, "batch_size": 1, "dtype": "bfloat16",
    "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
    "max_new_tokens": 384, "max_input_tokens": 6144,
    "eos_token_ids": [128001, 128009],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def resolve_binding(binding, *, directory=ROOT):
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
        raise ValueError("missing file binding")
    path = Path(binding["path"])
    candidates = [path] if path.is_absolute() else [directory / path, ROOT / path]
    for path in candidates:
        if path.is_file() and bank.file_sha(path) == binding.get("sha256"):
            if "bytes" in binding and path.stat().st_size != binding["bytes"]:
                raise ValueError("bound file size mismatch")
            return path.resolve()
    raise ValueError(f"bound file missing or hash mismatch: {binding['path']}")


def model_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_generation_contract(protocol):
    if (protocol.get("schema_version") != PROTOCOL_SCHEMA or protocol.get("status") != "FROZEN"
            or not isinstance(protocol.get("experiment_id"), str) or not protocol["experiment_id"]
            or type(protocol.get("seed")) is not int or protocol["seed"] != 42):
        raise ValueError("fresh confirmation protocol must be frozen with experiment ID and seed42")
    generation = protocol.get("generation") or {}
    if any(generation.get(key) != value or type(generation.get(key)) is not type(value)
           for key, value in GENERATION_FIXED.items()):
        raise ValueError("generation differs from frozen K4+greedy/BF16/384 production contract")
    if not re.fullmatch(r"cuda(?::[0-9]+)?", generation.get("device", "")):
        raise ValueError("explicit CUDA device required")


def verify_protocol(protocol, *, verify_models=True):
    """Read/check the frozen contract without generating or inspecting labels.

    The authority hashes pin this runner to the unchanged original fresh132.
    Model identities are inherited from that input manifest, never caller-picked.
    ``verify_models=False`` is for the downstream scorer, which verifies its own
    ReaRAG weights; adapter and policy tokenizer hashes are still checked here.
    """
    path = Path(protocol).resolve()
    frozen_sha = bank.file_sha(path)
    frozen = json.loads(path.read_text())
    validate_generation_contract(frozen)
    bindings = frozen.get("bindings") or {}
    authorities = {"inputs_manifest": INPUT_MANIFEST_SHA256, "inputs": INPUTS_SHA256,
                   "source_manifest": SOURCE_MANIFEST_SHA256}
    resolved = {}
    for name, expected_sha in authorities.items():
        if (bindings.get(name) or {}).get("sha256") != expected_sha:
            raise ValueError(f"original fresh132 authority changed: {name}")
        resolved[name] = resolve_binding(bindings[name], directory=path.parent)
    code_bindings = frozen.get("code_bindings") or {}
    if not set(GENERATION_CODE_FILES).issubset(code_bindings):
        raise ValueError("generation dependency code binding missing")
    for name, binding in code_bindings.items():
        live = resolve_binding(binding, directory=path.parent)
        if live != (ROOT / name).resolve():
            raise ValueError(f"code binding must match executed repository file: {name}")
    inputs_manifest = json.loads(resolved["inputs_manifest"].read_text())
    if (inputs_manifest.get("schema_version") != "source-credit-v2-fresh-confirmation-inputs-v1"
            or inputs_manifest.get("questions") != 132 or inputs_manifest.get("unique_families") != 132
            or inputs_manifest.get("gold_value_access") is not False
            or inputs_manifest["outputs"]["inputs.jsonl"]["sha256"] != INPUTS_SHA256):
        raise ValueError("wrong fresh132 input scope")
    # Hash-only provenance verification never parses the label-bearing sources.
    for bound in inputs_manifest["source_bindings"].values():
        resolve_binding(bound, directory=resolved["inputs_manifest"].parent)
    source = json.loads(resolved["source_manifest"].read_text())
    if (source.get("schema_version") != "source-credit-v2-fresh-confirmation-source-preparation-v1"
            or source.get("gold_access") is not False or source.get("optimizer_updates") != 0):
        raise ValueError("wrong source-only confirmation preparation")
    for bound in source["source_bindings"].values():
        resolve_binding(bound, directory=resolved["source_manifest"].parent)
    for bound in source["outputs"].values():
        resolve_binding(bound, directory=resolved["source_manifest"].parent)
    rows = bank.read_rows(resolved["inputs"])
    if (len(rows) != 132 or len({row["question_key"] for row in rows}) != 132
            or len({row["family_sha256"] for row in rows}) != 132):
        raise ValueError("exact132 question/family population required")
    for row in rows:
        bank.assert_gold_free(row)
        if (bank.key(row) != row["question_key"] or bank.input_hash(row) != row["input_sha256"]
                or bank.digest(row["source_quality_record"]) != row["source_record_sha256"]
                or row["fullsource_record"] != row["source_quality_record"]
                or row["spec"]["metadata"]["source_quality_record"] != row["source_quality_record"]
                or len(row["retrieved_passages"]) != 10 or len(row["kg_subgraph"]) > 12
                or not 0 < row["prompt_tokens"] <= 6144):
            raise ValueError("frozen input identity/evidence/budget mismatch")
    policy = model_path(inputs_manifest["policy_path"])
    base = model_path(inputs_manifest["models"]["base_model"]["path"])
    if policy != model_path(inputs_manifest["models"]["policy_tokenizer"]["path"]):
        raise ValueError("policy/tokenizer directory identity mismatch")
    for key, filename in (("policy", "adapter_model.safetensors"), ("policy_config", "adapter_config.json")):
        if bank.file_sha(policy / filename) != inputs_manifest["source_bindings"][key]["sha256"]:
            raise ValueError("frozen Strong SFT adapter mismatch")
    bank.validate_model(policy, inputs_manifest["models"]["policy_tokenizer"])
    for name in ("config.json", "generation_config.json"):
        if bank.file_sha(base / name) != inputs_manifest["models"]["base_model"]["files"][name]["sha256"]:
            raise ValueError("frozen base generation config mismatch")
    if verify_models:
        bank.validate_model(base, inputs_manifest["models"]["base_model"])
    stub = SimpleNamespace(generation_config=SimpleNamespace(**json.loads((base / "generation_config.json").read_text())),
                           config=SimpleNamespace(**json.loads((base / "config.json").read_text())))
    if list(bank._rollout_eos_token_ids(stub, None)) != frozen["generation"]["eos_token_ids"]:
        raise ValueError("bound model effective EOS mismatch")
    return {"protocol": frozen, "protocol_path": path, "protocol_sha256": frozen_sha,
            "input_manifest": inputs_manifest, "inputs": rows,
            "input_manifest_path": resolved["inputs_manifest"], "input_path": resolved["inputs"],
            "source_manifest_path": resolved["source_manifest"],
            "policy_path": policy, "base_model_path": base}


def expected_identity(context, row, index):
    frozen, manifest = context["protocol"], context["input_manifest"]
    return {"schema_version": ROW_SCHEMA, "candidate_id": f"{row['question_key']}::k{index}",
            "dataset": row["dataset"], "qid": row["qid"], "question_key": row["question_key"],
            "question_sha256": row["question_sha256"], "family_sha256": row["family_sha256"],
            "candidate_index": index, "generation_kind": "sampled" if index < 4 else "greedy",
            "seed": bank.candidate_seed(frozen["seed"], row["question_key"], index),
            "input_sha256": row["input_sha256"], "protocol_sha256": context["protocol_sha256"],
            "inputs_manifest_sha256": frozen["bindings"]["inputs_manifest"]["sha256"],
            "generation_contract_sha256": bank.digest(frozen["generation"]),
            "policy_sha256": manifest["source_bindings"]["policy"]["sha256"],
            "base_model_identity_sha256": bank.digest(manifest["models"]["base_model"]),
            "effective_eos_token_ids": list(frozen["generation"]["eos_token_ids"])}


def verify_generation_rows(rows, context, *, allow_partial=False, tokenizer=None):
    expected_count = len(context["inputs"]) * 5
    if len(rows) > expected_count or (not allow_partial and len(rows) != expected_count):
        raise ValueError("exact K4 plus greedy candidate count required")
    for position, prediction in enumerate(rows):
        verify_generation_row(prediction, context, position, tokenizer=tokenizer)


def verify_generation_row(prediction, context, position, *, tokenizer=None):
    source = context["inputs"][position // 5]
    expected = expected_identity(context, source, position % 5)
    if any(prediction.get(name) != value for name, value in expected.items()):
        raise ValueError("candidate order/seed/input/model/protocol identity mismatch")
    payload = {key: value for key, value in prediction.items() if key != "candidate_sha256"}
    if bank.digest(payload) != prediction.get("candidate_sha256"):
        raise ValueError("candidate payload hash mismatch")
    bank.assert_gold_free(prediction)
    ids, raw = prediction.get("response_token_ids"), prediction.get("raw_response_token_ids")
    if (not isinstance(prediction.get("generation"), str) or not isinstance(ids, list) or not isinstance(raw, list)
            or not 0 < len(ids) <= len(raw) <= 384
            or any(type(token) is not int or token < 0 for token in raw + ids)
            or prediction.get("n_response_tokens") != len(ids)
            or type(prediction.get("reached_max_new_tokens")) is not bool
            or type(prediction.get("pad_token_id")) is not int):
        raise ValueError("invalid generation token metadata")
    import torch
    trimmed = bank._trim_response_v2(torch.tensor(raw), eos_token_ids=expected["effective_eos_token_ids"],
                pad_token_id=prediction["pad_token_id"], max_new_tokens=384)
    capped = bank._response_is_length_capped_v2(trimmed, max_new_tokens=384,
                                              eos_token_ids=expected["effective_eos_token_ids"])
    if trimmed.tolist() != ids or capped != prediction["reached_max_new_tokens"]:
        raise ValueError("response trim/EOS/cap contract mismatch")
    if tokenizer is not None:
        if tokenizer.pad_token_id != prediction["pad_token_id"] or tokenizer.decode(ids, skip_special_tokens=True) != prediction["generation"]:
            raise ValueError("frozen generation decode/pad identity mismatch")


def encode_frozen_prompt(row, tokenizer):
    prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", truncation=False, return_attention_mask=True)
    count = encoded["input_ids"].shape[-1]
    if prompt != row["prompt"] or count != row["prompt_tokens"] or count > 6144:
        raise ValueError("runtime prompt/token budget differs from frozen input")
    return encoded, count


def generate_one(model, tokenizer, torch, context, row, index, encoded, count):
    """The sampling call and response processing exactly mirror the parent bank."""
    generation = context["protocol"]["generation"]
    effective_eos = bank._rollout_eos_token_ids(model, tokenizer)
    if list(effective_eos) != generation["eos_token_ids"]:
        raise ValueError("runtime model effective EOS differs from frozen production contract")
    seed = bank.candidate_seed(context["protocol"]["seed"], row["question_key"], index)
    torch.manual_seed(seed)
    device = generation["device"]
    with torch.inference_mode():
        if index < 4:
            sequence = model.generate(input_ids=encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device),
                do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=384,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=list(effective_eos))
        else:
            sequence = model.generate(input_ids=encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device),
                do_sample=False, max_new_tokens=384,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=list(effective_eos))
    raw_response = sequence[0, count:]
    response = bank._trim_response_v2(raw_response, eos_token_ids=effective_eos, pad_token_id=tokenizer.pad_token_id, max_new_tokens=384)
    ids = response.tolist()
    length_capped = bank._response_is_length_capped_v2(response, max_new_tokens=384, eos_token_ids=effective_eos)
    prediction = {**expected_identity(context, row, index), "generation": tokenizer.decode(ids, skip_special_tokens=True),
                  "response_token_ids": ids, "raw_response_token_ids": raw_response.tolist(),
                  "n_response_tokens": len(ids), "reached_max_new_tokens": length_capped,
                  "pad_token_id": tokenizer.pad_token_id}
    prediction["candidate_sha256"] = bank.digest(prediction)
    return prediction


def publish_bytes(path, data, attempts_dir):
    """Atomic, exclusive commit; retain interrupted/original write attempts."""
    attempts_dir.mkdir(parents=True, exist_ok=True)
    temporary = attempts_dir / f"{path.name}.{uuid.uuid4().hex}.attempt"
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)  # atomic and refuses replacement of any existing file
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def publish_json(path, value, attempts_dir):
    publish_bytes(path, (bank.canonical_json(value) + "\n").encode(), attempts_dir)


def read_committed(directory, context):
    candidates = directory / "candidates"
    files = sorted(candidates.glob("*.json")) if candidates.exists() else []
    if [path.name for path in files] != [f"{index:08d}.json" for index in range(len(files))]:
        raise ValueError("committed candidate prefix contains gaps/unexpected files")
    rows = [json.loads(path.read_text()) for path in files]
    verify_generation_rows(rows, context, allow_partial=True)
    return rows


@contextmanager
def output_lock(directory, resume):
    if resume:
        if not directory.is_dir() or not (directory / "started.json").is_file():
            raise ValueError("resume requires an existing started generation directory")
    else:
        directory.mkdir(parents=True, exist_ok=False)
    with (directory / ".writer.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run(*, protocol, out, resume=False):
    context = verify_protocol(protocol)
    out = Path(out).resolve()
    with output_lock(out, resume):
        attempts = out / "attempts"
        attempt_id = uuid.uuid4().hex
        frozen_identity = {"schema_version": GENERATION_SCHEMA, "experiment_id": context["protocol"]["experiment_id"],
                           "protocol_sha256": context["protocol_sha256"], "expected_candidates": 660,
                           "gold_access": False, "optimizer_updates": 0}
        if resume:
            started = json.loads((out / "started.json").read_text())
            if any(started.get(key) != value for key, value in frozen_identity.items()):
                raise ValueError("resume protocol/experiment identity mismatch")
        else:
            publish_json(out / "started.json", {**frozen_identity, "started_at_utc": now()}, attempts)
            (out / "candidates").mkdir()
        rows = read_committed(out, context)
        if (out / "manifest.json").exists():
            manifest = json.loads((out / "manifest.json").read_text())
            verify_generation_rows(rows, context)
            for binding in manifest["outputs"].values():
                resolve_binding(binding, directory=out)
            if (any(manifest.get(key) != value for key, value in frozen_identity.items())
                    or manifest.get("status") != "COMPLETE_GENERATED_NOT_SCORED"
                    or manifest.get("n_candidates") != 660
                    or bank.read_rows(out / "generations.jsonl") != rows):
                raise ValueError("completed generation protocol mismatch")
            return manifest
        before = time.monotonic()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(context["policy_path"], local_files_only=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            verify_generation_rows(rows, context, allow_partial=True, tokenizer=tokenizer)
            # Verify every fixed prompt before any fresh model output is revealed.
            for source in context["inputs"]:
                encode_frozen_prompt(source, tokenizer)
            torch = bank.require_cuda(context["protocol"]["generation"]["device"])
            device = context["protocol"]["generation"]["device"]
            model = AutoModelForCausalLM.from_pretrained(context["base_model_path"], local_files_only=True, torch_dtype=torch.bfloat16).to(device)
            model = PeftModel.from_pretrained(model, context["policy_path"], local_files_only=True, is_trainable=False).eval()
            torch.cuda.reset_peak_memory_stats(device)
            encoded, count, previous = None, None, None
            for position in range(len(rows), 660):
                source = context["inputs"][position // 5]
                if previous != source["question_key"]:
                    encoded, count = encode_frozen_prompt(source, tokenizer)
                    previous = source["question_key"]
                prediction = generate_one(model, tokenizer, torch, context, source, position % 5, encoded, count)
                rows.append(prediction)
                verify_generation_row(prediction, context, position, tokenizer=tokenizer)
                publish_json(out / "candidates" / f"{position:08d}.json", prediction, attempts)
                progress = {**frozen_identity, "status": "GENERATING", "completed_candidates": len(rows),
                            "candidate_id": prediction["candidate_id"], "generation_kind": prediction["generation_kind"],
                            "response_tokens": prediction["n_response_tokens"], "updated_at_utc": now(),
                            "attempt_elapsed_seconds": time.monotonic() - before}
                publish_json(out / f"progress_{position:08d}.json", progress, attempts)
                print(json.dumps(progress), flush=True)
            # Detect source/code/model mutations before committing the complete release.
            after = verify_protocol(protocol)
            if after["protocol_sha256"] != context["protocol_sha256"]:
                raise ValueError("protocol changed during generation")
            verify_generation_rows(rows, context, tokenizer=tokenizer)
            contents = "".join(bank.canonical_json(row) + "\n" for row in rows).encode()
            aggregate = out / "generations.jsonl"
            if aggregate.exists():
                if aggregate.read_bytes() != contents:
                    raise ValueError("existing final JSONL differs from committed candidates")
            else:
                publish_bytes(aggregate, contents, attempts)
            report = {**frozen_identity, "status": "COMPLETE_GENERATED_NOT_SCORED", "n_questions": 132,
                      "n_candidates": 660, "n_sampled_candidates": 528, "n_greedy_candidates": 132,
                      "completed_at_utc": now(), "attempt_elapsed_seconds": time.monotonic() - before,
                      "generation_contract_sha256": bank.digest(context["protocol"]["generation"]),
                      "gpu": {"name": torch.cuda.get_device_name(device),
                              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)},
                      "training_started": False, "reward_scoring_started": False,
                      "independent_confirmation_clearance": False, "ppo_launch_clearance": False,
                      "outputs": {"generations.jsonl": bank.identity(aggregate)}}
            publish_json(out / "manifest.json", report, attempts)
            return report
        except BaseException as exc:
            failure = {**frozen_identity, "attempt_id": attempt_id,
                       "status": "INTERRUPTED_PREFIX_RETAINED" if isinstance(exc, KeyboardInterrupt) else "FAILED_PREFIX_RETAINED",
                       "exception_type": type(exc).__name__, "error": str(exc), "time_utc": now(),
                       "committed_candidates": len(list((out / "candidates").glob("*.json")))}
            publish_json(out / f"attempt_{attempt_id}.json", failure, attempts)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps({key: result[key] for key in ("status", "n_candidates", "protocol_sha256")}))


if __name__ == "__main__":
    main()
