#!/usr/bin/env python3
"""Create reviewed SFT-v3 targets from immutable inputs, with no model updates.

--prepare_only freezes the exact input, checker-label, code and tokenizer
bindings without reading credentials or using the API. --execute_api is the
explicit paid execution switch; quotas and budget cannot change on resume.
All attempts/rejections are retained. A small/unfinished run is never READY.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from kgproweight.data.sft_v3_api import BudgetStop, DurableCalls, canonical, live_transport, sha_json, utc_now
from kgproweight.data.sft_v3_contract import (
    build_sft_v3_messages, tokenize_sft_v3_example, validate_sft_v3_trace,
)
from kgproweight.data.sft_v3_teacher import (
    build_sft_v3_teacher_messages, build_sft_v3_review_messages,
    validate_sft_v3_teacher_response, validate_sft_v3_review_response,
)
from kgproweight.reward.proofkg_process import canonical_exact_match, canonical_token_f1
from kgproweight.reward.source_integrity_v1 import validate_source_integrity_v1
from scripts.prepare.freeze_sft_v3_protected_ledger_v1 import identity as question_identity, make_index, overlap_reasons

ROOT = Path(__file__).resolve().parents[2]
VERSION = "sft-v3-reviewed-generation-v1"
SELECTION_VERSION = "verified-graph-stratum-first-v1"
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
CODE = [__file__, "kgproweight/data/sft_v3_contract.py", "kgproweight/data/sft_v3_teacher.py",
        "kgproweight/data/sft_v3_api.py", "kgproweight/data/prompts.py", "kgproweight/data/parsers.py",
        "kgproweight/reward/proofkg_process.py", "scripts/prepare/freeze_sft_v3_protected_ledger_v1.py",
        "scripts/prepare/freeze_qpeg_v1_protocol.py", "kgproweight/kg/question_kg.py",
        "kgproweight/reward/source_integrity_v1.py", "scripts/prepare/materialize_qpeg_v1_retrieval.py"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def bind(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_inputs(rows: list[dict], labels: list[dict], protected: list[dict]) -> None:
    """The checker table is never passed to either teacher prompt builder."""
    index = make_index(protected)
    label_index = {r["question_key"]: r for r in labels}
    if len(label_index) != len(labels): raise ValueError("duplicate checker label identity")
    seen_qids, seen_hashes, seen_families = set(), set(), set()
    for row in rows:
        item = question_identity(row)
        qkey = f"{item['dataset']}::{item['qid']}"
        if overlap_reasons(item, index): raise ValueError("protected identity in SFT inputs")
        if (qkey in seen_qids or item["question_sha256"] in seen_hashes
                or item["family_sha256"] in seen_families):
            raise ValueError("duplicate question or global family in SFT inputs")
        seen_qids.add(qkey); seen_hashes.add(item["question_sha256"]); seen_families.add(item["family_sha256"])
        if row.get("gold_access") is not False or row.get("split") not in {"train", "validation"}:
            raise ValueError("input Gold/split contract")
        if any(k in row for k in ("golden_answers", "answers", "gold_answer", "answer", "supporting_facts", "question_decomposition")):
            raise ValueError("checker annotation in teacher input row")
        if type(row.get("selection_rank")) is not int or row["selection_rank"] < 1:
            raise ValueError("frozen selection rank is required")
        label = label_index.get(qkey)
        if (not label or label["question_sha256"] != item["question_sha256"] or label["split"] != row["split"]
                or not isinstance(label.get("golden_answers"), list) or not label["golden_answers"]
                or any(not isinstance(a, str) or not a.strip() for a in label["golden_answers"])):
            raise ValueError("input/checker label identity mismatch or missing original labels")
        # A context must bind its original retrieval batch, never a handmade
        # top-10 target. Hash validation is memoized by prepare() below.
        if not isinstance(row.get("retrieval_binding"), dict):
            raise ValueError("missing canonical retrieval binding")
        if row.get("kg_subgraph"):
            proof = row.get("kg_source_verification") or {}
            if proof.get("status") != "PASS" or not proof.get("binding"):
                raise ValueError("nonempty KG needs a bound source-integrity PASS report")
        build_sft_v3_messages(question=row["question"], retrieved_passages=row["retrieved_passages"],
                              kg_triples=row.get("kg_subgraph", []))
    if not rows: raise ValueError("empty SFT input pool")


def verify_source_bindings(rows: list[dict]) -> None:
    """Replay exact per-question evidence joins, including nested KG sources."""
    checked, documents, indexes = {}, {}, {}

    def load(ref):
        if not isinstance(ref, dict) or not ref.get("path") or not ref.get("sha256"):
            raise ValueError("missing source artifact binding")
        path = Path(ref["path"]).resolve()
        if path in checked and checked[path] != ref["sha256"]:
            raise ValueError("conflicting source artifact SHA bindings")
        if path not in checked:
            if sha(path) != ref["sha256"]:
                raise ValueError("source artifact SHA mismatch")
            if "bytes" in ref and path.stat().st_size != ref["bytes"]:
                raise ValueError("source artifact byte count mismatch")
            checked[path] = ref["sha256"]
        if path not in documents:
            documents[path] = read_rows(path) if path.suffix == ".jsonl" else json.loads(path.read_text())
        return path, documents[path]

    def source_row(ref, key, kind):
        path, document = load(ref)
        cache_key = (path, kind)
        if cache_key not in indexes:
            items = document.get("contexts") if isinstance(document, dict) and kind == "retrieval" else document
            if not isinstance(items, list):
                raise ValueError("source artifact lacks expected per-question rows")
            current = {}
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("non-object source artifact row")
                ident = question_identity(item)
                qkey = f"{ident['dataset']}::{ident['qid']}"
                if qkey in current:
                    raise ValueError("duplicate question in bound source artifact")
                current[qkey] = item
            indexes[cache_key] = current
        if key not in indexes[cache_key]:
            raise ValueError("question missing from its bound source artifact")
        return indexes[cache_key][key]

    def verify_nested(value, evidence_path):
        if isinstance(value, dict):
            bindings = value.get("bindings")
            if bindings is not None:
                if not isinstance(bindings, dict):
                    raise ValueError("malformed nested source-evidence bindings")
                # Complete evidence stores include unused unresolved entities.
                # Validate every declared file; the pure integrity checker
                # separately requires nonempty bindings for this record's
                # global evidence and every entity/edge it actually consumes.
                for name, digest in bindings.items():
                    if not isinstance(name, str) or not isinstance(digest, str):
                        raise ValueError("invalid nested source-evidence binding")
                    raw = Path(name)
                    choices = [raw] if raw.is_absolute() else [ROOT / raw, evidence_path.parent / raw]
                    matched = next((path.resolve() for path in choices if path.is_file()), None)
                    if matched is None:
                        raise ValueError("nested source-evidence file is missing")
                    if matched in checked and checked[matched] != digest:
                        raise ValueError("conflicting nested source-evidence hashes")
                    if matched not in checked:
                        if sha(matched) != digest:
                            raise ValueError("nested source-evidence SHA mismatch")
                        checked[matched] = digest
            for child in value.values():
                verify_nested(child, evidence_path)
        elif isinstance(value, list):
            for child in value:
                verify_nested(child, evidence_path)

    for row in rows:
        key = row["question_key"]
        original = source_row(row["retrieval_binding"], key, "retrieval")
        if question_identity(original) != question_identity(row) or original.get("passages") != row["retrieved_passages"]:
            raise ValueError("retrieved passages or question differ from bound source row")
        # Canonical retrieval inherited _sha_json from materialize_qpeg_v1_retrieval:
        # sorted UTF-8 JSON with DEFAULT separators, not the compact API serializer.
        passage_digest = hashlib.sha256(json.dumps(original["passages"], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        if original.get("passages_sha256") and original["passages_sha256"] != passage_digest:
            raise ValueError("bound source passage digest mismatch")
        if row.get("kg_subgraph"):
            proof = row["kg_source_verification"]
            assignment = source_row(proof["binding"], key, "kg")
            source_record = assignment.get("fullsource_record")
            if not isinstance(source_record, dict) or question_identity(assignment) != question_identity(row):
                raise ValueError("KG assignment question/record missing or mismatched")
            if question_identity(source_record) != question_identity(row) or source_record.get("kg_subgraph") != row["kg_subgraph"]:
                raise ValueError("KG content differs from bound source record")
            digest = sha_json(source_record)
            if digest != assignment.get("source_record_sha256") or digest != proof.get("source_record_sha256"):
                raise ValueError("KG full source record SHA mismatch")
            ref = proof.get("source_evidence_binding")
            if ref != assignment.get("source_evidence_binding"):
                raise ValueError("KG source-evidence binding differs across assignment/input")
            evidence_path, evidence = load(ref)
            verify_nested(evidence, evidence_path)
            actual = validate_source_integrity_v1(source_record, evidence)
            if actual != assignment.get("source_check") or actual.get("status") != "PASS" or actual.get("clearance") is not True:
                raise ValueError("KG source-integrity PASS cannot be reproduced")


def schedule_rows_v3(rows: list[dict], *, quotas_per_domain: dict[str, int]) -> list[dict]:
    """Give source-verified graph supervision an opportunity before quotas fill.

    Source binding/integrity has already been replayed by prepare/run. No Gold,
    teacher output or citation behavior enters this order. Graph rows use their
    original rank, split and dataset; ordinary rows retain proportional ranks.
    A nonempty graph does not require the teacher to cite an irrelevant triple.
    """
    def order(row):
        split, dataset, rank = row["split"], row["dataset"], row["selection_rank"]
        if row.get("kg_subgraph"):
            proof = row.get("kg_source_verification") or {}
            if proof.get("status") != "PASS" or not proof.get("binding"):
                raise ValueError("graph-priority candidate lacks bound source PASS")
            return (0, rank, split, dataset)
        return (1, rank / quotas_per_domain[split], split, dataset)
    return sorted(rows, key=order)


def prepare(*, inputs: Path, labels: Path, protected: Path, out: Path, tokenizer_path: Path,
            budget_usd: float, max_calls: int, train_per_domain: int = 2000,
            validation_per_domain: int = 100, workers: int = 8,
            min_graph_citing_train: int = 100, min_graph_citing_validation: int = 5,
            test_double_only: bool = False) -> dict:
    if out.exists(): raise FileExistsError("new execution directory required")
    if min(train_per_domain, validation_per_domain, workers) < 1 or workers > 32:
        raise ValueError("positive quotas and 1..32 workers required")
    if not 0 < budget_usd <= 1000 or not 0 < max_calls <= 100000:
        raise ValueError("positive bounded API budget required")
    if type(test_double_only) is not bool:
        raise ValueError("test_double_only must be an explicit Boolean")
    rows, gold, exclusions = read_rows(inputs), read_rows(labels), read_rows(protected)
    validate_inputs(rows, gold, exclusions); verify_source_bindings(rows)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    lengths = []
    for row in rows:
        messages = build_sft_v3_messages(question=row["question"], retrieved_passages=row["retrieved_passages"], kg_triples=row.get("kg_subgraph", []))
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(tokens) + 384 > 6144: raise ValueError("input cannot accommodate 384 tokens without dropping evidence")
        lengths.append(len(tokens))
    out.mkdir(parents=True)
    for name in ("decisions", "accepted"): (out / name).mkdir()
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception: commit = None
    protocol = {
        "schema_version": VERSION, "experiment_id": out.name, "frozen_utc": utc_now(),
        "status": "FROZEN_NO_TEACHER_CALLS_NOT_TRAINED", "git_head": commit,
        "execution_mode": "test_double_only" if test_double_only else "live_official_deepseek",
        "test_double_only": test_double_only,
        "code": [bind(ROOT / p) for p in CODE],
        "inputs": bind(inputs), "checker_labels": bind(labels), "protected": bind(protected),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "tokenizer_files": [bind(p) for p in sorted(tokenizer_path.glob("*.json")) if "token" in p.name or p.name == "special_tokens_map.json"],
        "budget_usd": budget_usd, "cost_accounting": "peak_cache_miss_upper_estimate_not_provider_invoice",
        "pricing_source": "https://api-docs.deepseek.com/quick_start/pricing/", "pricing_verified_date": "2026-09-06",
        "max_calls": max_calls, "workers": workers,
        "producer": {"model": "deepseek-v4-flash", "max_tokens": 1800, "thinking": False},
        "reviewer": {"model": "deepseek-v4-pro", "max_tokens": 1000, "thinking": False},
        "quotas_per_domain": {"train": train_per_domain, "validation": validation_per_domain},
        "minimum_graph_citing": {"train": min_graph_citing_train, "validation": min_graph_citing_validation},
        "selection_version": SELECTION_VERSION,
        "selection": "source-PASS nonempty-KG stratum first by original selection_rank/split/dataset; ordinary stratum then by selection_rank/split_quota, split, dataset; first accepted within each split/domain quota; unused reserves retained; citations never forced",
        "graph_stratum_input_counts": dict(Counter(f"{r['split']}/{r['dataset']}" for r in rows if r.get("kg_subgraph"))),
        "gold_policy": "original train labels only after producer; max-alias PPO-canonical EM=1 admission; canonical F1 diagnostic; never sent to producer/reviewer, never rewrite target",
        "semantic_policy": "independent model review of every step and complete chain; uncertain rejects; not human ground truth",
        "retries": "zero automatic semantic/transport retries; unknown paid attempts stop automatic replay",
        "max_length": 6144, "max_assistant_tokens": 384, "steps": [2, 5], "passages": 10,
        "input_counts": dict(Counter(f"{r['split']}/{r['dataset']}" for r in rows)),
        "max_prompt_tokens": max(lengths), "combination_experiment": True,
        "combination_variables": ["three-domain supervision", "ten-passage evidence alignment", "2-5 concise supported steps", "independent teacher review", "verified_graph_supplement"],
        "sft_initialization": "same Llama3-8B-Instruct base; fresh LoRA; init_adapter_path=null",
        "model_updates": 0,
    }
    write(out / "protocol.json", protocol)
    write(out / "prepared.json", {"protocol": bind(out / "protocol.json")})
    return protocol


def verify(out: Path) -> dict:
    prepared = json.loads((out / "prepared.json").read_text())
    if bind(out / "protocol.json") != prepared["protocol"]: raise ValueError("execution protocol drift")
    p = json.loads((out / "protocol.json").read_text())
    for ref in [p["inputs"], p["checker_labels"], p["protected"], *p["code"], *p["tokenizer_files"]]:
        if bind(Path(ref["path"])) != ref: raise ValueError("bound execution input/code/tokenizer drift")
    return p


def process_one(row: dict, gold: dict, *, p: dict, ledger: DurableCalls, tokenizer: Any, transport: Any) -> dict:
    kwargs = {"question": row["question"], "retrieved_passages": row["retrieved_passages"], "kg_triples": row.get("kg_subgraph", [])}
    qkey = row["question_key"]
    decision = {"question_key": qkey, "dataset": row["dataset"], "split": row["split"],
                "input_sha256": sha_json(row), "accepted": False, "reason": None,
                "model_updates": 0, "gold_in_teacher_prompt": False}
    proposal = ledger.complete(call_id=qkey + "/producer", messages=build_sft_v3_teacher_messages(**kwargs),
                               transport=transport, **{k: p['producer'][k] for k in ('model', 'max_tokens')})
    if not proposal["usable"]:
        decision["reason"] = "producer_transport_or_response_contract"; return decision
    raw = proposal["payload"]["choices"][0]["message"]["content"]
    parsed = validate_sft_v3_teacher_response(raw, retrieved_passages=kwargs["retrieved_passages"], kg_triples=kwargs["kg_triples"])
    decision["producer_validation"] = parsed
    if not parsed["valid"] or not parsed["candidate_supported"]:
        decision["reason"] = "producer_format_evidence_or_insufficient"; return decision
    response = parsed["response"]
    trace, evidence = response["teacher_output"], response["evidence"]
    structure = validate_sft_v3_trace(trace, known_kg=kwargs["kg_triples"])
    try:
        encoded = tokenize_sft_v3_example(tokenizer, **kwargs, answer_trace=trace)
    except ValueError as exc:
        decision["reason"] = "target_token_or_template_contract"
        decision["token_error"] = str(exc); return decision
    # This is the first use of answer labels: the teacher text is already
    # durably captured. No answer hint/correction is sent back to the producer.
    em = max(canonical_exact_match(structure["final_answer"], alias) for alias in gold["golden_answers"])
    f1 = max(canonical_token_f1(structure["final_answer"], alias) for alias in gold["golden_answers"])
    decision["answer_check"] = {"em": em, "f1": f1, "checker_label_sha256": sha_json(gold)}
    if em != 1.0:
        decision["reason"] = "original_train_answer_mismatch"; return decision
    review = ledger.complete(call_id=qkey + "/reviewer", messages=build_sft_v3_review_messages(**kwargs, teacher_output=trace, evidence=evidence),
                             transport=transport, **{k: p['reviewer'][k] for k in ('model', 'max_tokens')})
    if not review["usable"]:
        decision["reason"] = "reviewer_transport_or_response_contract"; return decision
    reviewed = validate_sft_v3_review_response(review["payload"]["choices"][0]["message"]["content"], step_count=structure["step_count"])
    decision["review_validation"] = reviewed
    if not reviewed["valid"] or not reviewed["accepted"]:
        decision["reason"] = "semantic_review_reject_or_uncertain"; return decision
    record = {
        "schema_version": VERSION, "question_key": qkey, "dataset": row["dataset"], "qid": row["qid"],
        "question": row["question"], "question_sha256": row["question_sha256"],
        "family_sha256": row["family_sha256"], "family_version": row["family_version"], "split": row["split"],
        "retrieved_passages": row["retrieved_passages"], "kg_subgraph": row.get("kg_subgraph", []),
        "teacher_output": trace, "messages": build_sft_v3_messages(**kwargs, answer_trace=trace),
        "evidence_audit": evidence, "teacher_review": reviewed["response"],
        "input_sha256": sha_json(row), "producer_call_id": qkey + "/producer", "reviewer_call_id": qkey + "/reviewer",
        "step_count": structure["step_count"],
        "graph_citing_steps": sum(bool(s["cited_triples"]) for s in structure["steps"]),
        "token_counts": {k: encoded[k] for k in ("prompt_tokens", "assistant_tokens", "full_tokens")},
        "gold_in_prompt": False, "target_rewritten_from_gold": False,
    }
    decision.update(accepted=True, reason="accepted_teacher_and_blind_model_review", record=record)
    return decision


def run(out: Path, *, env_file: Path, max_questions: int | None = None, transport: Any = None) -> dict:
    if max_questions is not None and (type(max_questions) is not int or max_questions < 1):
        raise ValueError("max_questions must be a positive integer when supplied")
    # Keep lock descriptor alive for the complete operation; no two processes
    # may spend against a shared WAL or race the accepted quota.
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        p = verify(out)
        mode = p.get("execution_mode")
        if mode not in {"test_double_only", "live_official_deepseek"} or p.get("test_double_only") is not (mode == "test_double_only"):
            raise ValueError("execution mode is missing, ambiguous, or changed")
        if transport is not None and mode != "test_double_only":
            raise ValueError("custom transports may run only in an independently frozen test-double directory")
        if transport is None and mode == "test_double_only":
            raise ValueError("test-double execution requires an explicit custom transport; live API forbidden")
        if (out / "manifest.json").exists(): raise FileExistsError("completed dataset release is immutable")
        rows = read_rows(Path(p["inputs"]["path"]))
        labels = {r["question_key"]: r for r in read_rows(Path(p["checker_labels"]["path"]))}
        verify_source_bindings(rows)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(p["tokenizer_path"], local_files_only=True)
        ledger = DurableCalls(out / "api_calls.wal.jsonl", budget_usd=p["budget_usd"], max_calls=p["max_calls"])
        if transport is None: transport = live_transport(env_file)
        existing = {r["question_key"]: r for file in sorted((out / "decisions").glob("*.json")) for r in [json.loads(file.read_text())]}
        inputs_by_key = {r["question_key"]: r for r in rows}
        for key, decision in existing.items():
            if key not in inputs_by_key or decision["input_sha256"] != sha_json(inputs_by_key[key]):
                raise ValueError("decision/input drift on resume")
        counts = Counter((d["split"], d["dataset"]) for d in existing.values() if d["accepted"])
        # Source-backed graph rows must receive a generation opportunity before
        # an ordinary-only prefix fills the domain quota. The remaining rows
        # keep the frozen proportional train/validation schedule.
        schedule = schedule_rows_v3(rows, quotas_per_domain=p["quotas_per_domain"])
        processed, stop_reason = 0, "candidate_pool_exhausted"
        with ThreadPoolExecutor(max_workers=p["workers"]) as executor:
            remaining = deque(schedule)
            while remaining:
                pending, reserved = [], Counter()
                deferred = []
                while remaining and len(pending) < p["workers"]:
                    row = remaining.popleft()
                    key = (row["split"], row["dataset"])
                    quota = p["quotas_per_domain"][row["split"]]
                    if row["question_key"] in existing or counts[key] >= quota:
                        continue
                    if counts[key] + reserved[key] >= quota:
                        # Pending candidates may fail. Keep later candidates
                        # in their original rank order until those outcomes
                        # are known instead of permanently skipping reserves.
                        deferred.append(row)
                        continue
                    if max_questions is not None and processed + len(pending) >= max_questions:
                        remaining.appendleft(row); break
                    pending.append(row); reserved[key] += 1
                remaining.extendleft(reversed(deferred))
                if not pending:
                    if max_questions is not None and processed >= max_questions:
                        stop_reason = "operational_question_limit"; break
                    continue
                futures = [executor.submit(process_one, row, labels[row["question_key"]], p=p, ledger=ledger, tokenizer=tokenizer, transport=transport) for row in pending]
                budget_stopped = False
                for row, future in zip(pending, futures):
                    try: decision = future.result()
                    except BudgetStop:
                        budget_stopped = True; continue
                    filename = hashlib.sha256(row["question_key"].encode()).hexdigest() + ".json"
                    write(out / "decisions" / filename, decision)
                    existing[row["question_key"]] = decision; processed += 1
                    if decision["accepted"]: counts[(row["split"], row["dataset"])] += 1
                print(canonical({"processed_this_attempt": processed, "accepted": {f"{k[0]}/{k[1]}": v for k,v in counts.items()}, "api_calls": len(ledger.intents), "cost_upper_usd": round(ledger.committed_upper_usd(), 6)}), flush=True)
                if budget_stopped: stop_reason = "frozen_api_budget_stop"; break
                if all(counts[(split,ds)] >= quota for split,quota in p["quotas_per_domain"].items() for ds in DATASETS):
                    stop_reason = "all_quotas_met"; break
        verify(out)
        accepted = [existing[r["question_key"]]["record"] for r in schedule if r["question_key"] in existing and existing[r["question_key"]]["accepted"]]
        graph_counts = Counter(r["split"] for r in accepted if r["graph_citing_steps"] > 0)
        quotas_met = all(counts[(split,ds)] == quota for split,quota in p["quotas_per_domain"].items() for ds in DATASETS)
        graph_met = all(graph_counts[split] >= minimum for split,minimum in p["minimum_graph_citing"].items())
        attempt = len(list(out.glob("progress_*.json"))) + 1
        report = {"schema_version": VERSION, "experiment_id": p["experiment_id"],
                  "status": "TEST_DOUBLE_ONLY_NOT_SFT_DATA" if p["test_double_only"] else ("AUTOMATED_GATES_COMPLETE_INDEPENDENT_RELEASE_AUDIT_PENDING" if quotas_met and graph_met else "PARTIAL_TEACHER_DATA_NOT_READY"),
                  "execution_mode": p["execution_mode"], "test_double_only": p["test_double_only"],
                  "stop_reason": stop_reason, "processed": len(existing), "accepted": len(accepted),
                  "counts": {f"{k[0]}/{k[1]}": v for k,v in counts.items()}, "graph_citing_counts": dict(graph_counts),
                  "rejection_reasons": dict(Counter(d["reason"] for d in existing.values() if not d["accepted"])),
                  "api_calls": len(ledger.intents), "cost_peak_upper_usd": ledger.committed_upper_usd(),
                  "model_updates": 0, "quotas_met": quotas_met, "graph_coverage_met": graph_met,
                  "human_semantic_audit_complete": False, "data_ready": False,
                  "protocol": bind(out / "protocol.json")}
        for split in ("train", "validation"):
            path = out / f"accepted_{split}_snapshot_{attempt:04d}.jsonl"
            with path.open("x") as handle:
                for row in accepted:
                    if row["split"] == split: handle.write(canonical(row) + "\n")
            report[f"{split}_snapshot"] = bind(path)
        write(out / f"progress_{attempt:04d}.json", report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--protected", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models/llama3-8b")
    parser.add_argument("--budget_usd", type=float)
    parser.add_argument("--max_calls", type=int, default=33000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_questions", type=int)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare_only", action="store_true")
    group.add_argument("--execute_api", action="store_true")
    parser.add_argument("--env_file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()
    if args.prepare_only:
        if any(x is None for x in (args.inputs, args.labels, args.protected, args.budget_usd)):
            parser.error("prepare requires inputs, labels, protected and budget_usd")
        result = prepare(inputs=args.inputs, labels=args.labels, protected=args.protected, out=args.out,
                         tokenizer_path=args.tokenizer, budget_usd=args.budget_usd, max_calls=args.max_calls, workers=args.workers)
    else:
        result = run(args.out, env_file=args.env_file, max_questions=args.max_questions)
    print(canonical(result))


if __name__ == "__main__": main()
