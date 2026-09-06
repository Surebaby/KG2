from collections import Counter
from copy import deepcopy
import hashlib
import json

import pytest

from kgproweight.data.sft_v3_api import DurableCalls
from kgproweight.data.sft_v3_teacher import TEACHER_SCHEMA_VERSION, REVIEW_SCHEMA_VERSION
from kgproweight.data.sft_v3_contract import EVIDENCE_SCHEMA_VERSION
from scripts.prepare import audit_sft_v3_release_v1 as audit
from scripts.prepare import generate_sft_v3_teacher_v1 as producer


TRACE = """[Step 1]
Reasoning: The first passage names the initial fact.
Knowledge Used: []
Conclusion: The initial fact is Alpha.

[Step 2]
Reasoning: The second passage gives the final response.
Knowledge Used: []
Conclusion: The final response is Beta.

[Final Answer]
Beta"""


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(m["role"] + ":" + m["content"] + "\n" for m in messages)
        if add_generation_prompt:
            text += "assistant:"
        return [ord(c) for c in text] if tokenize else text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def write_rows(path, rows):
    path.write_text("".join(audit.canonical(row) + "\n" for row in rows))


def teacher_payload():
    evidence = {"schema_version": EVIDENCE_SCHEMA_VERSION, "steps": [
        {"step_index": 1, "supports": [{"passage_index": 1, "quote": "The initial fact is Alpha."}], "derivation_from_steps": []},
        {"step_index": 2, "supports": [{"passage_index": 2, "quote": "The final response is Beta."}], "derivation_from_steps": [1]},
    ]}
    return {"schema_version": TEACHER_SCHEMA_VERSION, "status": "supported", "teacher_output": TRACE, "evidence": evidence}


def transport(body):
    review = {"schema_version": REVIEW_SCHEMA_VERSION, "verdict": "accept", "steps": [
        {"step_index": 1, "verdict": "accept", "reason": "Visible first fact."},
        {"step_index": 2, "verdict": "accept", "reason": "Visible final fact."}],
        "chain_complete": True, "final_supported": True, "concise_answer": True, "reason": "Supported fixture."}
    payload = teacher_payload() if body["model"] == "deepseek-v4-flash" else review
    return {"id": "fixture-call", "model": body["model"], "usage": {"prompt_tokens": 100, "completion_tokens": 100},
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": audit.canonical(payload)}}]}


def fixture_release(tmp_path):
    execution, candidates_dir = tmp_path / "execution", tmp_path / "candidates"
    execution.mkdir(); candidates_dir.mkdir(); (execution / "decisions").mkdir()
    (execution / "execution.lock").touch()
    protected = tmp_path / "protected.jsonl"; protected.write_text("")
    items, labels, candidates, contexts = [], [], [], []
    for dataset in audit.DATASETS:
        raw = [{"id": split, "question": f"what response links {dataset} {split} evidence?", "golden_answers": ["Beta", "BETA"]}
               for split in ("train", "validation")]
        raw_path = tmp_path / (dataset + ".jsonl"); write_rows(raw_path, raw)
        for line_number, (split, raw_row) in enumerate(zip(("train", "validation"), raw), 1):
            ident = audit.identity({"dataset": dataset, "qid": split, "question": raw_row["question"]})
            qkey = f"{dataset}::{split}"
            candidate = {**ident, "question_key": qkey, "split": split, "within_split_dataset_rank": 1}
            candidates.append(candidate)
            passages = [{"id": f"p{i}", "contents": text} for i, text in enumerate(
                ["The initial fact is Alpha.", "The final response is Beta."] + [f"Other visible passage {j}." for j in range(8)])]
            contexts.append({**ident, "question_key": qkey, "passages": passages,
                             "passages_sha256": hashlib.sha256(json.dumps(passages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()})
            items.append({**ident, "question_key": qkey, "split": split, "selection_rank": 1, "gold_access": False,
                          "retrieved_passages": passages, "kg_subgraph": []})
            raw_line = raw_path.read_bytes().splitlines(keepends=True)[line_number - 1]
            labels.append({"question_key": qkey, "dataset": dataset, "qid": split, "question_sha256": ident["question_sha256"],
                           "family_sha256": ident["family_sha256"], "split": split, "golden_answers": raw_row["golden_answers"],
                           "source": {**audit.bind(raw_path), "line_number": line_number,
                                      "line_bytes_sha256": hashlib.sha256(raw_line).hexdigest()}})
    retrieval = tmp_path / "retrieval.json"; write(retrieval, {"contexts": contexts})
    for item in items:
        item["retrieval_binding"] = audit.bind(retrieval)
    input_path = tmp_path / "inputs.jsonl"; write_rows(input_path, items)
    candidate_path = candidates_dir / "candidates.question_only.jsonl"; write_rows(candidate_path, candidates)
    label_path = candidates_dir / "labels.checker_only.jsonl"; write_rows(label_path, labels)
    pool_protocol = candidates_dir / "protocol.json"; write(pool_protocol, {"protected_identities": audit.bind(protected)})
    write(candidates_dir / "manifest.json", {"schema_version": "sft-v3-three-domain-candidate-pool-v1", "complete": True,
          "gates": {"isolated": True}, "outputs": {p.name: audit.bind(p) for p in (candidate_path, label_path, pool_protocol)}})
    p = {"schema_version": producer.VERSION, "experiment_id": "fixture", "inputs": audit.bind(input_path),
         "checker_labels": audit.bind(label_path), "protected": audit.bind(protected),
         "code": [audit.bind(audit.ROOT / name) for name in sorted(audit.REQUIRED_CODE)], "tokenizer_files": [],
         "budget_usd": 1., "max_calls": 20, "producer": {"model": "deepseek-v4-flash", "max_tokens": 1800},
         "reviewer": {"model": "deepseek-v4-pro", "max_tokens": 1000},
         "quotas_per_domain": {"train": 1, "validation": 1}, "minimum_graph_citing": {"train": 0, "validation": 0},
         "execution_mode": "test_double_only", "test_double_only": True,
         "max_length": 6144, "max_assistant_tokens": 384, "steps": [2, 5], "passages": 10,
         "selection_version": "verified-graph-stratum-first-v1"}
    write(execution / "protocol.json", p)
    write(execution / "prepared.json", {"protocol": audit.bind(execution / "protocol.json")})
    wal = DurableCalls(execution / "api_calls.wal.jsonl", budget_usd=1., max_calls=20)
    decisions = []
    for item, label in zip(items, labels):
        result = producer.process_one(item, label, p=p, ledger=wal, tokenizer=CharacterTokenizer(), transport=transport)
        assert result["accepted"], result
        name = hashlib.sha256(item["question_key"].encode()).hexdigest() + ".json"
        write(execution / "decisions" / name, result); decisions.append(result)
    records = [d["record"] for d in decisions]
    records.sort(key=lambda r: (r["split"], r["dataset"]))
    snapshots = {}
    for split in ("train", "validation"):
        path = execution / (split + ".jsonl")
        write_rows(path, [r for r in records if r["split"] == split]); snapshots[split + "_snapshot"] = audit.bind(path)
    progress = {"protocol": audit.bind(execution / "protocol.json"), "accepted": 6, "processed": 6,
                "counts": {f"{s}/{d}": 1 for s in ("train", "validation") for d in audit.DATASETS},
                "graph_citing_counts": {}, "api_calls": 12, "cost_peak_upper_usd": wal.committed_upper_usd(),
                "quotas_met": True, "graph_coverage_met": True, "execution_mode": "test_double_only", "test_double_only": True,
                **snapshots}
    progress_path = execution / "progress_0001.json"; write(progress_path, progress)
    return {"execution_dir": execution, "progress_path": progress_path, "candidate_pool_dir": candidates_dir,
            "output_dir": tmp_path / "audit", "tokenizer": CharacterTokenizer(), "fixture_only": True}


def test_end_to_end_fixture_wal_is_replayed_but_can_never_be_ready(tmp_path):
    args = fixture_release(tmp_path)
    report = audit.audit_execution(**args)
    assert report["accepted"] == 6
    assert report["gates"]["all_decisions_recomputed_from_raw_wal"]
    assert report["gates"]["quotas_met"]
    assert not report["data_ready"]
    assert not report["gates"]["live_official_api_execution"]
    assert not report["gates"]["formal_release_scale"]
    assert report["human_semantic_audit_complete"] is False
    assert report["wal"]["intents"] == 12


@pytest.mark.parametrize("field,changed", [("teacher_output", TRACE.replace("Beta", "Delta")),
                                           ("graph_citing_steps", 1), ("evidence_audit", {})])
def test_forged_accepted_record_cannot_hide_behind_original_decision_bool(tmp_path, field, changed):
    args = fixture_release(tmp_path)
    decision_path = next((args["execution_dir"] / "decisions").glob("*.json"))
    decision = json.loads(decision_path.read_text()); decision["record"][field] = changed; write(decision_path, decision)
    with pytest.raises(ValueError, match="independent WAL replay"):
        audit.audit_execution(**args)
    assert (args["output_dir"] / "FAILED.json").exists()


def test_persisted_usable_flag_is_not_trusted(tmp_path):
    args = fixture_release(tmp_path)
    path = args["execution_dir"] / "api_calls.wal.jsonl"
    events = audit.rows(path)
    result = next(r for r in events if r["event"] == "result")
    result["payload"]["choices"][0]["finish_reason"] = "length"
    write_rows(path, events)
    with pytest.raises(ValueError, match="usable flag"):
        audit.audit_execution(**args)


def test_extra_gold_hint_in_durable_request_is_rejected_even_with_updated_sha(tmp_path):
    args = fixture_release(tmp_path)
    path = args["execution_dir"] / "api_calls.wal.jsonl"
    events = audit.rows(path)
    intent = next(r for r in events if r["event"] == "intent")
    intent["request"]["messages"][-1]["content"] += "\nGold answer hint: Beta"
    intent["request_sha256"] = audit.digest(intent["request"])
    intent["prompt_tokens_upper"] = len(audit.canonical(intent["request"]["messages"]).encode()) + 512
    intent["reserved_upper_usd"] = audit.cost(intent["request"]["model"], intent["prompt_tokens_upper"], intent["request"]["max_tokens"])
    for result in events:
        if result["event"] == "result" and result["call_id"] == intent["call_id"]:
            result["request_sha256"] = intent["request_sha256"]
    write_rows(path, events)
    with pytest.raises(ValueError, match="reconstructed blind"):
        audit.audit_execution(**args)


def test_snapshot_is_not_trusted_even_when_its_manifest_sha_is_updated(tmp_path):
    args = fixture_release(tmp_path)
    progress = json.loads(args["progress_path"].read_text())
    path = args["execution_dir"] / "train.jsonl"
    write_rows(path, audit.rows(path)[:2])
    progress["train_snapshot"] = audit.bind(path); write(args["progress_path"], progress)
    with pytest.raises(ValueError, match="snapshot differs"):
        audit.audit_execution(**args)


def test_custom_tokenizer_cannot_claim_nonfixture_audit(tmp_path):
    args = fixture_release(tmp_path); args["fixture_only"] = False
    with pytest.raises(ValueError, match="custom tokenizer"):
        audit.audit_execution(**args)


def test_actual_raw_label_changes_are_detected(tmp_path):
    args = fixture_release(tmp_path)
    path = tmp_path / "hotpotqa.jsonl"
    raw = audit.rows(path); raw[0]["golden_answers"] = ["Delta"]; write_rows(path, raw)
    with pytest.raises(ValueError, match="bound file differs"):
        audit.audit_execution(**args)


def test_source_exact_content_is_checked_independently(tmp_path):
    args = fixture_release(tmp_path)
    p = json.loads((args["execution_dir"] / "protocol.json").read_text())
    item = audit.rows(tmp_path / "inputs.jsonl")[0]
    checker = audit.SourceAudit(require_real_retrieval=False)
    checker.verify(item)
    wrong = deepcopy(item); wrong["retrieved_passages"][0]["contents"] = "Invented passage after binding."
    with pytest.raises(ValueError, match="exactly match"):
        checker.verify(wrong)


def test_unsealed_fake_retrieval_cannot_pass_formal_source_audit(tmp_path):
    fixture_release(tmp_path)
    item = audit.rows(tmp_path / "inputs.jsonl")[0]
    with pytest.raises(ValueError, match="sealed canonical"):
        audit.SourceAudit().verify(item)


def test_duplicate_and_nonfinite_wal_json_fail():
    with pytest.raises(ValueError, match="duplicate"):
        audit.strict_json('{"call_id":"a","call_id":"b"}')
    with pytest.raises(ValueError, match="non-finite"):
        audit.strict_json('{"cost":NaN}')
