#!/usr/bin/env python3
"""Independent read-only replay of reviewed SFT-v3 release provenance.

This auditor does not call the producer runner, its process_one function, its
input/source validator, or DurableCalls. It reconstructs the admission decision
from durable raw producer/reviewer responses, frozen evidence, original labels,
and the declared format/token contracts. A fixture run can never be READY.
Model reviews are not human factual labels; human review remains explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import fcntl
from pathlib import Path
from typing import Any

from kgproweight.data.sft_v3_api import ALLOWED_RESPONSE_MODELS, API_VERSION, RATES
from kgproweight.data.sft_v3_contract import (build_sft_v3_messages, tokenize_sft_v3_example,
                                           validate_sft_v3_trace)
from kgproweight.data.sft_v3_teacher import (build_sft_v3_teacher_messages, build_sft_v3_review_messages,
                                          validate_sft_v3_teacher_response, validate_sft_v3_review_response)
from kgproweight.reward.proofkg_process import canonical_exact_match, canonical_token_f1
from kgproweight.kg.question_kg import question_sha256
from kgproweight.reward.source_integrity_v1 import validate_source_integrity_v1
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256

ROOT = Path(__file__).resolve().parents[2]
VERSION = "sft-v3-independent-release-audit-v1"
RUNNER_VERSION = "sft-v3-reviewed-generation-v1"
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
REQUIRED_CODE = {"kgproweight/data/sft_v3_contract.py", "kgproweight/data/sft_v3_teacher.py",
                 "kgproweight/data/sft_v3_api.py", "kgproweight/data/prompts.py", "kgproweight/data/parsers.py",
                 "kgproweight/reward/proofkg_process.py", "kgproweight/reward/source_integrity_v1.py",
                 "scripts/prepare/freeze_qpeg_v1_protocol.py", "kgproweight/kg/question_kg.py",
                 "scripts/prepare/materialize_qpeg_v1_retrieval.py"}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bind(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_sha(path), "bytes": path.stat().st_size}


def strict_json(raw: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("non-finite JSON value")
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("JSON record must be an object")
    return value


def rows(path: Path) -> list[dict]:
    return [strict_json(line) for line in path.read_text().splitlines() if line.strip()]


def checked_file(ref: dict, root: Path = ROOT) -> Path:
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or not isinstance(ref.get("sha256"), str):
        raise ValueError("missing file SHA binding")
    path = Path(ref["path"])
    if not path.is_absolute():
        path = root / path
    if (file_sha(path) != ref["sha256"] or ("bytes" in ref and path.stat().st_size != ref["bytes"])
            or ("size_bytes" in ref and path.stat().st_size != ref["size_bytes"])):
        raise ValueError(f"bound file differs: {path}")
    return path


def identity(row: dict) -> dict:
    ds, qid, question = row.get("dataset"), row.get("qid"), row.get("question")
    if ds not in DATASETS or not isinstance(qid, str) or not qid or not isinstance(question, str) or not question.strip():
        raise ValueError("incomplete question identity")
    item = {"dataset": ds, "qid": qid, "question": question.strip(),
            "question_sha256": question_sha256(question), "family_sha256": family_sha256(question), "family_version": FAMILY_VERSION}
    for field in ("question_sha256", "family_sha256", "family_version"):
        if row.get(field) and row[field] != item[field]:
            raise ValueError("stored identity hash/family disagrees with independent recomputation")
    if row.get("question_key") and row["question_key"] != f"{ds}::{qid}":
        raise ValueError("question key disagrees with identity")
    return item


def unique_index(items: list[dict], name: str) -> dict[str, dict]:
    result = {}
    for row in items:
        key = row.get("question_key")
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f"duplicate/missing identity in {name}")
        result[key] = row
    return result


def verify_original_labels(labels: list[dict]):
    """Compare unchanged selected checker labels to the exact original raw lines."""
    groups = {}
    for label in labels:
        ref = label.get("source")
        if not isinstance(ref, dict) or type(ref.get("line_number")) is not int or ref["line_number"] < 1:
            raise ValueError("checker label lacks original one-based raw source line")
        path = Path(ref["path"]).resolve()
        if path not in groups:
            checked_file(ref)
            groups[path] = {"sha256": ref["sha256"], "rows": {}}
        if groups[path]["sha256"] != ref["sha256"] or ref["line_number"] in groups[path]["rows"]:
            raise ValueError("conflicting or repeated checker raw source line")
        groups[path]["rows"][ref["line_number"]] = label
    for path, group in groups.items():
        found = set()
        with path.open("rb") as handle:
            for lineno, line in enumerate(handle, 1):
                label = group["rows"].get(lineno)
                if label is None:
                    continue
                raw = strict_json(line.decode())
                if (hashlib.sha256(line).hexdigest() != label["source"].get("line_bytes_sha256")
                        or str(raw.get("id") or raw.get("qid")) != label["qid"]
                        or question_sha256(raw.get("question", "")) != label["question_sha256"]
                        or raw.get("golden_answers") != label["golden_answers"]):
                    raise ValueError("checker labels are not faithful original same-line answers")
                found.add(lineno)
        if found != set(group["rows"]) or file_sha(path) != group["sha256"]:
            raise ValueError("checker source incomplete or changed during audit")


def expected_request(model: str, messages: list[dict], max_tokens: int) -> dict:
    if model not in RATES or type(max_tokens) is not int or not 1 <= max_tokens <= 2400:
        raise ValueError("unrecognized frozen API model/token contract")
    return {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": .2 if model == "deepseek-v4-flash" else 0.,
            "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}, "stream": False}


def cost(model: str, prompt: int, completion: int) -> float:
    a, b = RATES[model]
    return (a * prompt + b * completion) / 1000000


def response_usable(intent: dict, result: dict) -> bool:
    """Recompute provider acceptance; do not trust persisted usable=True."""
    payload = result.get("payload")
    if not isinstance(payload, dict) or result.get("error_class"):
        return False
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return False
    pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if type(pt) is not int or type(ct) is not int or min(pt, ct) < 0:
        return False
    request = intent["request"]
    if pt > intent["prompt_tokens_upper"] or ct > request["max_tokens"]:
        return False
    choices = payload.get("choices")
    if payload.get("model") not in ALLOWED_RESPONSE_MODELS[request["model"]] or not isinstance(choices, list) or len(choices) != 1:
        return False
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop" or choice.get("has_reasoning_content"):
        return False
    message = choice.get("message")
    return (isinstance(message, dict) and isinstance(message.get("content"), str) and bool(message["content"].strip())
            and not message.get("reasoning_content") and not message.get("refusal"))


def read_wal(path: Path, protocol: dict) -> tuple[dict, dict, dict]:
    intents, results = {}, {}
    if not path.is_file():
        return intents, results, {"intents": 0, "results": 0, "unresolved": 0, "charged_upper_usd": 0.}
    for event in rows(path):
        cid, kind = event.get("call_id"), event.get("event")
        if not isinstance(cid, str) or not cid or kind not in {"intent", "result"}:
            raise ValueError("malformed WAL call/event identity")
        table = intents if kind == "intent" else results
        if cid in table:
            raise ValueError("duplicate WAL event identity")
        if kind == "intent":
            req = event.get("request")
            if not isinstance(req, dict) or digest(req) != event.get("request_sha256"):
                raise ValueError("WAL request SHA mismatch")
            if req != expected_request(req.get("model"), req.get("messages"), req.get("max_tokens")) or event.get("api_version") != API_VERSION:
                raise ValueError("WAL API contract mismatch")
            upper = len(canonical(req["messages"]).encode()) + 512
            reserve = cost(req["model"], upper, req["max_tokens"])
            if event.get("prompt_tokens_upper") != upper or not isinstance(event.get("reserved_upper_usd"), (int, float)) or abs(event["reserved_upper_usd"] - reserve) > 1e-12:
                raise ValueError("WAL reserved cost mismatch")
        else:
            if cid not in intents or event.get("request_sha256") != intents[cid]["request_sha256"]:
                raise ValueError("WAL result without exact preceding request")
            if type(event.get("usable")) is not bool or event["usable"] != response_usable(intents[cid], event):
                raise ValueError("WAL usable flag cannot be independently reproduced")
            usage = (event.get("payload") or {}).get("usage") or {}
            pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
            actual = (cost(intents[cid]["request"]["model"], pt, ct)
                      if type(pt) is int and type(ct) is int and min(pt, ct) >= 0 else intents[cid]["reserved_upper_usd"])
            charged = event.get("charged_upper_usd")
            if type(charged) not in (int, float) or not math.isfinite(charged) or abs(charged - actual) > 1e-12:
                raise ValueError("WAL charged cost mismatch")
        table[cid] = event
    committed = sum(results.get(cid, {}).get("charged_upper_usd", item["reserved_upper_usd"]) for cid, item in intents.items())
    if len(intents) > protocol["max_calls"] or committed > protocol["budget_usd"] + 1e-9:
        raise ValueError("frozen API call/budget ceiling exceeded")
    return intents, results, {"intents": len(intents), "results": len(results), "unresolved": len(intents) - len(results), "charged_upper_usd": committed}


def verify_all_blind_requests(inputs, labels, protocol, intents, results):
    """Include unresolved/unused attempts; they must not hide Gold-bearing calls."""
    for call_id, intent in intents.items():
        key, sep, role = call_id.rpartition("/")
        if not sep or key not in inputs or role not in {"producer", "reviewer"}:
            raise ValueError("WAL call outside frozen input/role population")
        item = inputs[key]
        kwargs = {"question": item["question"], "retrieved_passages": item["retrieved_passages"], "kg_triples": item.get("kg_subgraph", [])}
        spec = protocol[role]
        if role == "producer":
            messages = build_sft_v3_teacher_messages(**kwargs)
        else:
            producer_id = key + "/producer"
            if producer_id not in results or not response_usable(intents[producer_id], results[producer_id]):
                raise ValueError("reviewer call lacks a usable durable producer response")
            text = results[producer_id]["payload"]["choices"][0]["message"]["content"]
            validated = validate_sft_v3_teacher_response(text, retrieved_passages=kwargs["retrieved_passages"], kg_triples=kwargs["kg_triples"])
            if not validated["valid"] or not validated["candidate_supported"]:
                raise ValueError("reviewer called for structurally invalid or unsupported producer")
            trace, evidence = validated["response"]["teacher_output"], validated["response"]["evidence"]
            answer = validate_sft_v3_trace(trace, known_kg=kwargs["kg_triples"])["final_answer"]
            if max(canonical_exact_match(answer, gold) for gold in labels[key]["golden_answers"]) != 1.:
                raise ValueError("reviewer called despite producer original-answer mismatch")
            messages = build_sft_v3_review_messages(**kwargs, teacher_output=trace, evidence=evidence)
        if intent["request"] != expected_request(spec["model"], messages, spec["max_tokens"]):
            raise ValueError("WAL request differs from reconstructed blind producer/reviewer context")


class SourceAudit:
    """Independent per-question binding checker; no producer validator reuse."""
    def __init__(self, *, require_real_retrieval=True):
        self.cache = {}
        self.seen = {}
        self.require_real_retrieval = require_real_retrieval
        self.retrieval_verified = set()
        self.asset_stats = {}

    def read(self, ref):
        path = Path(ref.get("path", "")).resolve()
        if path in self.seen and self.seen[path] != ref.get("sha256"):
            raise ValueError("conflicting source hash declarations")
        if path not in self.cache:
            checked_file(ref)
            self.seen[path] = ref["sha256"]
            self.cache[path] = rows(path) if path.suffix == ".jsonl" else strict_json(path.read_text())
        return path, self.cache[path]

    def locate(self, ref, question_key, context=False):
        _, value = self.read(ref)
        items = value.get("contexts") if context and isinstance(value, dict) else value
        if not isinstance(items, list):
            raise ValueError("bound source is not a per-question sequence")
        index = unique_index(items, "source artifact")
        if question_key not in index:
            raise ValueError("bound source omits this question")
        return index[question_key]

    def nested(self, value, parent):
        if isinstance(value, dict):
            declared = value.get("bindings")
            if declared is not None:
                if not isinstance(declared, dict):
                    raise ValueError("malformed nested source bindings")
                for name, sha in declared.items():
                    path = Path(name)
                    candidates = [path] if path.is_absolute() else [ROOT / path, parent / path]
                    path = next((p.resolve() for p in candidates if p.is_file()), None)
                    if path is None:
                        raise ValueError("missing nested source evidence")
                    if path in self.seen and self.seen[path] != sha:
                        raise ValueError("conflicting nested source hash")
                    if path not in self.seen:
                        checked_file({"path": str(path), "sha256": sha})
                        self.seen[path] = sha
            for child in value.values():
                self.nested(child, parent)
        elif isinstance(value, list):
            for child in value:
                self.nested(child, parent)

    def real_retrieval(self, ref):
        path, batch = self.read(ref)
        if path in self.retrieval_verified:
            return
        if not isinstance(batch, dict) or batch.get("schema_version") != "sft-v3-canonical-retrieval-v1":
            raise ValueError("formal audit requires newly sealed canonical retrieval batches")
        attestation = batch.get("backend_attestation") or {}
        if (attestation.get("mode") != "real_full_wiki18" or attestation.get("fallback") is not False
                or attestation.get("load_succeeded") is not True or attestation.get("corpus_documents") != 21015324):
            raise ValueError("retrieval backend is not the real full Wiki18 canonical stack")
        directory = path.parent.parent
        protocol_path, prepared_path = directory / "protocol.json", directory / "prepared.json"
        protocol, prepared = strict_json(protocol_path.read_text()), strict_json(prepared_path.read_text())
        checked_file(prepared["protocol"])
        if Path(prepared["protocol"]["path"]).resolve() != protocol_path.resolve():
            raise ValueError("retrieval preparation binds another protocol path")
        if (protocol.get("schema_version") != "sft-v3-canonical-retrieval-v1"
                or protocol.get("test_double_only") is not False or protocol.get("full_wiki18_documents") != 21015324
                or protocol.get("retrieval") != "E5@100+BM25@100->RRF60@50->bge-reranker-v2-m3@10->pack3860"
                or not protocol.get("assets")):
            raise ValueError("retrieval protocol is incomplete or fixture-only")
        seal_path = path.with_name(path.stem + ".seal.json")
        seal = strict_json(seal_path.read_text())
        checked_file(seal["batch"])
        if Path(seal["batch"]["path"]).resolve() != path or seal.get("protocol_sha256") != file_sha(protocol_path):
            raise ValueError("retrieval batch seal differs from actual protocol/batch")
        if batch.get("protocol_sha256") != seal["protocol_sha256"] or batch.get("contexts_sha256") != digest(batch.get("contexts")):
            raise ValueError("retrieval batch context/protocol SHA mismatch")
        request_path = checked_file(protocol["requests"])
        all_requests = rows(request_path)
        offset = int(path.stem)
        size = protocol["batch_size"]
        source_requests = all_requests[offset:offset + size]
        if offset < 0 or offset % size or not source_requests or batch.get("request_sha256") != digest(source_requests):
            raise ValueError("retrieval batch request membership/hash mismatch")
        if len(source_requests) != len(batch["contexts"]):
            raise ValueError("retrieval batch omitted or added a question")
        for request, context in zip(source_requests, batch["contexts"]):
            if any(context.get(k) != value for k, value in request.items() if k != "schema_version"):
                raise ValueError("retrieval batch request order/identity metadata differs")
        for code_ref in protocol["code"]:
            checked_file(code_ref)
            self.seen[Path(code_ref["path"]).resolve()] = code_ref["sha256"]
        for asset in protocol["assets"]:
            asset_path = Path(asset["path"]).resolve()
            stat = asset_path.stat()
            actual = {"device": stat.st_dev, "inode": stat.st_ino, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if not isinstance(asset.get("sha256"), str) or len(asset["sha256"]) != 64 or actual != asset.get("stat"):
                raise ValueError("retrieval assets differ from their full-SHA freeze stat contract")
            self.asset_stats[asset_path] = actual
        for other in (protocol_path, prepared_path, seal_path, request_path):
            self.seen[other.resolve()] = file_sha(other)
        self.retrieval_verified.add(path)

    def verify(self, item):
        if self.require_real_retrieval:
            self.real_retrieval(item["retrieval_binding"])
        original = self.locate(item["retrieval_binding"], item["question_key"], context=True)
        # Historical source family strings are provenance, never authority.
        original_id = dict(original)
        original_id.pop("family_sha256", None)
        original_id.pop("family_version", None)
        if identity(original_id) != identity(item) or original.get("passages") != item["retrieved_passages"]:
            raise ValueError("source retrieval row does not exactly match input")
        passage_digest = hashlib.sha256(json.dumps(original["passages"], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        if original.get("passages_sha256") and original["passages_sha256"] != passage_digest:
            raise ValueError("source passage hash mismatch")
        if item.get("kg_subgraph"):
            proof = item["kg_source_verification"]
            assignment = self.locate(proof["binding"], item["question_key"])
            record = assignment.get("fullsource_record")
            if not isinstance(record, dict) or identity(assignment) != identity(item) or identity(record) != identity(item):
                raise ValueError("KG source identity does not match")
            if record.get("kg_subgraph") != item["kg_subgraph"] or digest(record) != proof.get("source_record_sha256") or digest(record) != assignment.get("source_record_sha256"):
                raise ValueError("KG source content or record digest differs")
            evidence_ref = proof.get("source_evidence_binding")
            if evidence_ref != assignment.get("source_evidence_binding"):
                raise ValueError("KG evidence binding differs")
            path, evidence = self.read(evidence_ref)
            self.nested(evidence, path.parent)
            actual = validate_source_integrity_v1(record, evidence)
            if proof.get("status") != "PASS" or actual.get("status") != "PASS" or actual.get("clearance") is not True or actual != assignment.get("source_check"):
                raise ValueError("KG integrity PASS not independently reproducible")


def replay_decision(item, label, *, protocol, intents, results, tokenizer):
    kwargs = {"question": item["question"], "retrieved_passages": item["retrieved_passages"], "kg_triples": item.get("kg_subgraph", [])}
    qkey = item["question_key"]
    producer_id, reviewer_id = qkey + "/producer", qkey + "/reviewer"
    producer = protocol["producer"]
    expected = expected_request(producer["model"], build_sft_v3_teacher_messages(**kwargs), producer["max_tokens"])
    if producer_id not in intents or intents[producer_id]["request"] != expected:
        raise ValueError("producer request differs from blind frozen evidence")
    if producer_id not in results:
        raise ValueError("decision has no durable producer response")
    outcome = {"accepted": False, "reason": "producer_transport_or_response_contract"}
    if not response_usable(intents[producer_id], results[producer_id]):
        return outcome
    raw = results[producer_id]["payload"]["choices"][0]["message"]["content"]
    checked = validate_sft_v3_teacher_response(raw, retrieved_passages=kwargs["retrieved_passages"], kg_triples=kwargs["kg_triples"])
    outcome["producer_validation"] = checked
    if not checked["valid"] or not checked["candidate_supported"]:
        outcome["reason"] = "producer_format_evidence_or_insufficient"
        return outcome
    trace, evidence = checked["response"]["teacher_output"], checked["response"]["evidence"]
    structure = validate_sft_v3_trace(trace, known_kg=kwargs["kg_triples"])
    try:
        encoded = tokenize_sft_v3_example(tokenizer, **kwargs, answer_trace=trace)
    except ValueError:
        outcome["reason"] = "target_token_or_template_contract"
        return outcome
    em = max(canonical_exact_match(structure["final_answer"], answer) for answer in label["golden_answers"])
    f1 = max(canonical_token_f1(structure["final_answer"], answer) for answer in label["golden_answers"])
    outcome["answer_check"] = {"em": em, "f1": f1, "checker_label_sha256": digest(label)}
    if em != 1.:
        outcome["reason"] = "original_train_answer_mismatch"
        return outcome
    reviewer = protocol["reviewer"]
    expected = expected_request(reviewer["model"], build_sft_v3_review_messages(**kwargs, teacher_output=trace, evidence=evidence), reviewer["max_tokens"])
    if reviewer_id not in intents or intents[reviewer_id]["request"] != expected:
        raise ValueError("reviewer request differs from blind exact producer/evidence")
    if reviewer_id not in results:
        raise ValueError("decision has no durable reviewer response")
    if not response_usable(intents[reviewer_id], results[reviewer_id]):
        outcome["reason"] = "reviewer_transport_or_response_contract"
        return outcome
    review = validate_sft_v3_review_response(results[reviewer_id]["payload"]["choices"][0]["message"]["content"], step_count=structure["step_count"])
    outcome["review_validation"] = review
    if not review["valid"] or not review["accepted"]:
        outcome["reason"] = "semantic_review_reject_or_uncertain"
        return outcome
    record = {"schema_version": RUNNER_VERSION, "question_key": qkey,
              **{name: item[name] for name in ("dataset", "qid", "question", "question_sha256", "family_sha256", "family_version", "split")},
              "retrieved_passages": item["retrieved_passages"], "kg_subgraph": item.get("kg_subgraph", []),
              "teacher_output": trace, "messages": build_sft_v3_messages(**kwargs, answer_trace=trace),
              "evidence_audit": evidence, "teacher_review": review["response"], "input_sha256": digest(item),
              "producer_call_id": producer_id, "reviewer_call_id": reviewer_id,
              "step_count": structure["step_count"], "graph_citing_steps": sum(bool(s["cited_triples"]) for s in structure["steps"]),
              "token_counts": {k: encoded[k] for k in ("prompt_tokens", "assistant_tokens", "full_tokens")},
              "gold_in_prompt": False, "target_rewritten_from_gold": False}
    outcome.update(accepted=True, reason="accepted_teacher_and_blind_model_review", record=record)
    return outcome


def audit_execution(*, execution_dir: Path, progress_path: Path, candidate_pool_dir: Path,
                    output_dir: Path, tokenizer=None, fixture_only: bool = False) -> dict:
    if tokenizer is not None and not fixture_only:
        raise ValueError("custom tokenizer is allowed only in explicitly fixture-only audit")
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen = {"schema_version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "execution_dir": str(execution_dir.resolve()), "progress": bind(progress_path),
              "candidate_pool_manifest": bind(candidate_pool_dir / "manifest.json"),
              "execution_protocol": bind(execution_dir / "protocol.json"), "code": bind(Path(__file__)),
              "fixture_only": fixture_only, "training_started": False}
    def write(name, value):
        with (output_dir / name).open("x") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    write("protocol.json", frozen)
    lock = None
    try:
        lock_path = execution_dir / "execution.lock"
        if lock_path.exists():
            lock = lock_path.open("r")
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        elif not fixture_only:
            raise ValueError("no execution lock: real teacher execution has not started")
        p = strict_json((execution_dir / "protocol.json").read_text())
        progress = strict_json(progress_path.read_text())
        prepared = strict_json((execution_dir / "prepared.json").read_text())
        if bind(execution_dir / "protocol.json") != prepared["protocol"] or p.get("schema_version") != RUNNER_VERSION:
            raise ValueError("execution protocol/prepared binding differs")
        if progress.get("protocol") != bind(execution_dir / "protocol.json"):
            raise ValueError("progress does not bind exact execution protocol")
        if (p.get("max_length") != 6144 or p.get("max_assistant_tokens") != 384 or p.get("steps") != [2, 5]
                or p.get("passages") != 10):
            raise ValueError("frozen format/token/evidence protocol differs")
        declared_code = {str(Path(ref["path"]).resolve()) for ref in p["code"]}
        if not {str((ROOT / name).resolve()) for name in REQUIRED_CODE} <= declared_code:
            raise ValueError("protocol omits necessary frozen research code")
        for ref in [p["inputs"], p["checker_labels"], p["protected"], *p["code"], *p["tokenizer_files"]]:
            checked_file(ref)
        candidate_manifest = strict_json((candidate_pool_dir / "manifest.json").read_text())
        base_candidate = (candidate_manifest.get("schema_version") == "sft-v3-three-domain-candidate-pool-v1"
                          and candidate_manifest.get("gates") and all(candidate_manifest["gates"].values()))
        combined_candidate = (candidate_manifest.get("schema_version") == "sft-v3-combined-inputs-v1"
                              and candidate_manifest.get("status") == "COMPLETE_COMBINED_IDENTITIES_AND_CHECKER_LABELS_NOT_TEACHER_DATA_NOT_TRAINED"
                              and candidate_manifest.get("original_random_rows_unchanged") is True
                              and candidate_manifest.get("global_families_unique") is True
                              and candidate_manifest.get("protected_overlap") == 0
                              and candidate_manifest.get("typed_source_PASS_reexecuted") is True)
        if (not (base_candidate or combined_candidate) or candidate_manifest.get("complete") is not True
                or (candidate_pool_dir / "FAILED.json").exists()):
            raise ValueError("candidate pool incomplete/failed")
        pool_protocol_path = checked_file(candidate_manifest["outputs"]["protocol.json"])
        pool_protocol = strict_json(pool_protocol_path.read_text())
        if p["protected"]["sha256"] != pool_protocol["protected_identities"]["sha256"]:
            raise ValueError("execution does not use the candidate pool's complete protected ledger")
        candidate_ref = candidate_manifest["outputs"]["candidates.question_only.jsonl"]
        label_ref = candidate_manifest["outputs"]["labels.checker_only.jsonl"]
        candidates = unique_index(rows(checked_file(candidate_ref)), "candidate pool")
        all_labels = unique_index(rows(checked_file(label_ref)), "canonical checker labels")
        inputs = unique_index(rows(Path(p["inputs"]["path"])), "execution inputs")
        labels = unique_index(rows(Path(p["checker_labels"]["path"])), "execution labels")
        if any(key not in all_labels or labels[key] != all_labels[key] for key in labels):
            raise ValueError("execution labels differ from original candidate checker labels")
        verify_original_labels([labels[key] for key in inputs if key in labels])
        protected = rows(Path(p["protected"]["path"]))
        protected_sets = {"qid": set(), "question_sha256": set(), "family_sha256": set()}
        for row in protected:
            item = identity(row)
            for field in protected_sets:
                protected_sets[field].add((item["dataset"], item[field]) if field == "qid" else item[field])
        source = SourceAudit(require_real_retrieval=not fixture_only)
        input_seen = {field: set() for field in protected_sets}
        for key, item in inputs.items():
            ident = identity(item)
            if item.get("gold_access") is not False or item.get("split") not in {"train", "validation"}:
                raise ValueError("input is not explicitly blind train/validation evidence")
            if key not in candidates or key not in labels:
                raise ValueError("execution input outside frozen candidate/label membership")
            candidate = candidates[key]
            if identity(candidate) != ident or candidate["split"] != item["split"] or candidate["within_split_dataset_rank"] != item["selection_rank"]:
                raise ValueError("candidate/input question, split, or ordinal rank differs")
            if labels[key]["question_sha256"] != ident["question_sha256"] or labels[key]["split"] != item["split"]:
                raise ValueError("checker-label input identity mismatch")
            for field in protected_sets:
                value = (ident["dataset"], ident[field]) if field == "qid" else ident[field]
                if value in protected_sets[field] or value in input_seen[field]:
                    raise ValueError("protected or repeated global input identity")
                input_seen[field].add(value)
            source.verify(item)
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(p["tokenizer_path"], local_files_only=True)
        intents, results, wal_stats = read_wal(execution_dir / "api_calls.wal.jsonl", p)
        verify_all_blind_requests(inputs, labels, p, intents, results)
        decisions = {}
        checks = []
        for file in sorted((execution_dir / "decisions").glob("*.json")):
            decision = strict_json(file.read_text())
            key = decision.get("question_key")
            if key not in inputs or key in decisions or file.name != hashlib.sha256(key.encode()).hexdigest() + ".json":
                raise ValueError("decision filename/key membership invalid")
            if decision.get("input_sha256") != digest(inputs[key]):
                raise ValueError("decision input digest mismatch")
            if type(decision.get("accepted")) is not bool or decision.get("model_updates") != 0 or decision.get("gold_in_teacher_prompt") is not False:
                raise ValueError("decision status/Gold/model-update contract differs")
            actual = replay_decision(inputs[key], labels[key], protocol=p, intents=intents, results=results, tokenizer=tokenizer)
            for name, value in actual.items():
                if decision.get(name) != value:
                    raise ValueError(f"decision {key}/{name} differs from independent WAL replay")
            if not actual["accepted"] and "record" in decision:
                raise ValueError("rejected decision contains an admitted training record")
            decisions[key] = actual
            checks.append({"question_key": key, "accepted": actual["accepted"], "reason": actual["reason"],
                           "decision_file": bind(file), "independent_raw_wal_replay_exact": True})
        schedule = sorted(inputs.values(), key=lambda r: ((0, r["selection_rank"], r["split"], r["dataset"])
                          if r.get("kg_subgraph") else (1, r["selection_rank"] / p["quotas_per_domain"][r["split"]], r["split"], r["dataset"])))
        accepted = [decisions[r["question_key"]]["record"] for r in schedule if r["question_key"] in decisions and decisions[r["question_key"]]["accepted"]]
        for split in ("train", "validation"):
            declared = rows(checked_file(progress[split + "_snapshot"]))
            expected = [r for r in accepted if r["split"] == split]
            if declared != expected:
                raise ValueError("final split snapshot differs from independently admitted records/order")
        counts = Counter((r["split"], r["dataset"]) for r in accepted)
        graph = Counter(r["split"] for r in accepted if r["graph_citing_steps"] > 0)
        reported_counts = {f"{split}/{dataset}": count for (split, dataset), count in counts.items()}
        if (progress.get("counts") != reported_counts or progress.get("graph_citing_counts") != dict(graph)
                or progress.get("api_calls") != wal_stats["intents"]
                or abs(progress.get("cost_peak_upper_usd", -1) - wal_stats["charged_upper_usd"]) > 1e-9):
            raise ValueError("progress domain/graph/API/cost statistics differ")
        if progress.get("accepted") != len(accepted) or progress.get("processed") != len(decisions):
            raise ValueError("progress accepted/processed counts differ")
        quotas = all(counts[(split, ds)] == target for split, target in p["quotas_per_domain"].items() for ds in DATASETS)
        graph_met = all(graph[split] >= target for split, target in p["minimum_graph_citing"].items())
        if progress.get("quotas_met") is not quotas or progress.get("graph_coverage_met") is not graph_met:
            raise ValueError("progress quota or graph-clearance flags differ")
        live = (not fixture_only and p.get("execution_mode") == "live_official_deepseek"
                and p.get("test_double_only") is False and progress.get("test_double_only") is False
                and progress.get("execution_mode") == "live_official_deepseek" and wal_stats["intents"] > 0)
        gates = {"all_decisions_recomputed_from_raw_wal": True, "all_snapshot_records_exact": True,
                 "all_input_retrieval_and_kg_source_rows_exact": True, "global_protected_and_internal_identity_overlap_zero": True,
                 "checker_labels_equal_frozen_candidate_originals": True, "checker_labels_equal_original_raw_lines": True,
                 "all_accepted_strict_format_tokens_and_blind_reviews_pass": True,
                 "quotas_met": quotas, "graph_citing_coverage_met": graph_met,
                 "formal_release_scale": p["quotas_per_domain"] == {"train": 2000, "validation": 100},
                 "formal_graph_minima": p["minimum_graph_citing"].get("train", 0) >= 100 and p["minimum_graph_citing"].get("validation", 0) >= 5,
                 "no_unresolved_api_attempts": wal_stats["unresolved"] == 0, "live_official_api_execution": live}
        automated = all(gates.values())
        report = {"schema_version": VERSION, "status": "DATA_READY_AUTOMATICALLY_REVIEWED_NOT_HUMAN_GOLD_NOT_TRAINED" if automated else "PARTIAL_OR_FIXTURE_AUDITED_NOT_READY",
                  "complete": True, "data_ready": automated, "automated_gates_pass": automated,
                  "human_semantic_audit_complete": False,
                  "gates": gates, "accepted": len(accepted), "processed": len(decisions),
                  "accepted_by_dataset_split": {f"{s}/{d}": counts[(s, d)] for s in ("train", "validation") for d in DATASETS},
                  "graph_citing_counts": dict(graph), "wal": wal_stats,
                  "fixture_only": fixture_only, "model_updates": 0,
                  "scientific_boundary": "Format, original-label agreement, source integrity and blind model reviews are replayed; semantic model review is not human ground truth. No human audit is invented, and no training launch is authorized here."}
        with (output_dir / "decision_checks.jsonl").open("x") as handle:
            for item in checks:
                handle.write(canonical(item) + "\n")
        for path, sha in source.seen.items():
            if file_sha(path) != sha:
                raise ValueError("source evidence changed during audit")
        for path, frozen_stat in source.asset_stats.items():
            stat = path.stat()
            if {"device": stat.st_dev, "inode": stat.st_ino, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != frozen_stat:
                raise ValueError("retrieval assets changed during audit")
        for ref in (frozen["progress"], frozen["candidate_pool_manifest"], frozen["execution_protocol"], frozen["code"]):
            checked_file(ref)
        write("report.json", report)
        write("manifest.json", {**report, "outputs": {name: bind(output_dir / name) for name in ("protocol.json", "report.json", "decision_checks.jsonl")}})
        return report
    except BaseException as exc:
        write("FAILED.json", {"status": "AUDIT_FAILED_NOT_READY", "type": type(exc).__name__, "error": str(exc), "partial_outputs_retained": True})
        raise
    finally:
        if lock is not None:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-dir", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(canonical(audit_execution(execution_dir=args.execution_dir, progress_path=args.progress,
                                    candidate_pool_dir=args.candidate_pool, output_dir=args.output_dir)))


if __name__ == "__main__":
    main()
