"""Offline integration: admission boundaries, paid-call accounting and quotas.

All provider responses are explicit test doubles; none are research evidence.
No tokenizer/model downloads or external requests are used.
"""

from collections import Counter
from copy import deepcopy
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from kgproweight.data.sft_v3_api import DurableCalls
from scripts.prepare import generate_sft_v3_teacher_v1 as runner


class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.lock = threading.Lock()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = "".join(f"<{m['role']}> {m['content']} <eot> " for m in messages)
        text += "<assistant> " if add_generation_prompt else ""
        return self(text, add_special_tokens=False)["input_ids"] if tokenize else text

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        with self.lock:
            for word in text.split():
                self.vocab.setdefault(word, len(self.vocab) + 1)
            return {"input_ids": [self.vocab[word] for word in text.split()]}


def make_row(dataset="hotpotqa", split="train", rank=1):
    row = {"dataset": dataset, "qid": f"{split}_{rank}",
           "question": f"Which fact holds for {dataset} {split} rank {rank}?"}
    row.update(runner.question_identity(row))
    row.update(question_key=f"{dataset}::{row['qid']}", split=split, selection_rank=rank,
               gold_access=False, kg_subgraph=[],
               retrieved_passages=[{"id": str(i), "contents": f"Subject {i} has a documented fact."} for i in range(1, 11)])
    return row


def teacher(answer="Example", long=False):
    trace = "\n\n".join(
        f"[Step {i}]\nReasoning: Subject {i} has a documented fact in the visible evidence.\n"
        f"Knowledge Used: []\nConclusion: Subject {i} is documented."
        for i in (1, 2)
    ) + f"\n\n[Final Answer]\n{answer}"
    if long:
        trace = trace.replace("Subject 1 has a documented fact in the visible evidence.", "A fact " + "word " * 400)
    return {"schema_version": "sft-v3-teacher-v1", "status": "supported", "teacher_output": trace,
            "evidence": {"schema_version": "sft-v3-evidence-v1", "steps": [
                {"step_index": i, "supports": [{"passage_index": i, "quote": f"Subject {i} has a documented fact."}],
                 "derivation_from_steps": []} for i in (1, 2)
            ]}}


def review(verdict="accept"):
    return {"schema_version": "sft-v3-review-v1", "verdict": verdict,
            "steps": [{"step_index": i, "verdict": verdict, "reason": "Synthetic review fixture."} for i in (1, 2)],
            "chain_complete": verdict == "accept", "final_supported": verdict == "accept",
            "concise_answer": True, "reason": "Synthetic review fixture."}


def protocol():
    return {"producer": {"model": "deepseek-v4-flash", "max_tokens": 1800},
            "reviewer": {"model": "deepseek-v4-pro", "max_tokens": 1000}}


def envelope(body, response):
    return {"model": body["model"], "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(response)}}]}


def execute_one(tmp_path, *, proposed=None, reviewed=None, labels=None, mutate_provider=None):
    row = make_row()
    gold = {"question_key": row["question_key"], "golden_answers": labels or ["example", "LABEL_ONLY_SENTINEL"]}
    calls = []
    def provider(body):
        calls.append(deepcopy(body))
        value = envelope(body, (proposed or teacher()) if body["model"] == "deepseek-v4-flash" else (reviewed or review()))
        if mutate_provider:
            mutate_provider(value)
        return value
    ledger = DurableCalls(tmp_path / "wal.jsonl", budget_usd=1, max_calls=10)
    decision = runner.process_one(row, gold, p=protocol(), ledger=ledger, tokenizer=Tokenizer(), transport=provider)
    return decision, calls, ledger, row


def test_gold_remains_checker_only_and_teacher_final_is_never_replaced(tmp_path):
    proposal = teacher("EXAMPLE")
    decision, calls, ledger, row = execute_one(tmp_path, proposed=proposal)
    assert decision["accepted"]
    assert len(calls) == 2
    assert "LABEL_ONLY_SENTINEL" not in json.dumps(calls)
    record = decision["record"]
    assert record["teacher_output"] == proposal["teacher_output"]
    assert record["messages"][-1]["content"] == proposal["teacher_output"]
    assert record["target_rewritten_from_gold"] is False
    assert "golden_answers" not in record
    assert record["retrieved_passages"] == row["retrieved_passages"]
    assert set(ledger.results) == {row["question_key"] + "/producer", row["question_key"] + "/reviewer"}


def test_label_access_happens_only_after_producer_is_durable(tmp_path):
    row = make_row()
    ledger = DurableCalls(tmp_path / "wal", budget_usd=1, max_calls=3)
    class TrackedGold(dict):
        def __getitem__(self, key):
            if key == "golden_answers":
                assert row["question_key"] + "/producer" in ledger.results
                assert len((tmp_path / "wal").read_text().splitlines()) >= 2
            return super().__getitem__(key)
    gold = TrackedGold(golden_answers=["Example"])
    result = runner.process_one(row, gold, p=protocol(), ledger=ledger, tokenizer=Tokenizer(),
                                transport=lambda b: envelope(b, teacher() if b["model"] == "deepseek-v4-flash" else review()))
    assert result["accepted"]


def test_wrong_answer_skips_paid_review_and_is_not_corrected(tmp_path):
    decision, calls, _, _ = execute_one(tmp_path, labels=["Some different label"])
    assert not decision["accepted"]
    assert decision["reason"] == "original_train_answer_mismatch"
    assert len(calls) == 1
    assert decision["producer_validation"]["response"]["teacher_output"].endswith("Example")


def test_boolean_f1_uses_current_ppo_canonical_guard(tmp_path):
    decision, calls, _, _ = execute_one(tmp_path, proposed=teacher("not yes"), labels=["yes"])
    assert decision["answer_check"]["em"] == decision["answer_check"]["f1"] == 0
    assert len(calls) == 1


def test_punctuation_article_order_uses_current_ppo_canonical_em(tmp_path):
    # Generic legacy metrics remove 'The' before joining the hyphen and would
    # incorrectly admit this fixture against 'Example'. PPO removes punctuation
    # first, yielding 'theexample', which is correctly a mismatch here.
    decision, calls, _, _ = execute_one(tmp_path, proposed=teacher("The-Example"), labels=["Example"])
    assert decision["answer_check"]["em"] == 0
    assert not decision["accepted"] and len(calls) == 1


def test_answer_aliases_use_max_canonical_match(tmp_path):
    decision, calls, _, _ = execute_one(tmp_path, labels=["different", "The Example"])
    assert decision["accepted"]
    assert decision["answer_check"]["em"] == 1
    assert len(calls) == 2


@pytest.mark.parametrize("proposal", [
    {"not": "the required schema"},
    {"schema_version": "sft-v3-teacher-v1", "status": "insufficient_evidence",
     "teacher_output": "", "evidence": {"schema_version": "sft-v3-evidence-v1", "steps": []}},
    dict(teacher(), teacher_output="[Final Answer]\nExample"),
])
def test_producer_invalid_or_insufficient_skips_review(tmp_path, proposal):
    decision, calls, _, _ = execute_one(tmp_path, proposed=proposal)
    assert not decision["accepted"]
    assert decision["reason"] == "producer_format_evidence_or_insufficient"
    assert "answer_check" not in decision
    assert len(calls) == 1


def test_over_384_target_skips_gold_and_review(tmp_path):
    decision, calls, _, _ = execute_one(tmp_path, proposed=teacher(long=True))
    assert decision["reason"] == "target_token_or_template_contract"
    assert "answer_check" not in decision
    assert len(calls) == 1


@pytest.mark.parametrize("verdict", ["reject", "uncertain"])
def test_semantic_review_failure_rejects_even_when_answer_correct(tmp_path, verdict):
    decision, calls, _, _ = execute_one(tmp_path, reviewed=review(verdict))
    assert not decision["accepted"]
    assert decision["answer_check"]["em"] == 1
    assert decision["reason"] == "semantic_review_reject_or_uncertain"
    assert len(calls) == 2


def test_wrong_provider_model_is_rejected_before_target_processing(tmp_path):
    decision, calls, _, _ = execute_one(tmp_path, mutate_provider=lambda p: p.update(model="different-provider-model"))
    assert decision["reason"] == "producer_transport_or_response_contract"
    assert len(calls) == 1


def setup_run(tmp_path, monkeypatch, *, workers=8, max_calls=100):
    out = tmp_path / "run"
    out.mkdir()
    (out / "decisions").mkdir()
    (out / "accepted").mkdir()
    rows = [make_row(dataset, split, rank) for rank in (1, 2, 3)
            for split in ("train", "validation") for dataset in runner.DATASETS]
    labels = [{"question_key": r["question_key"], "golden_answers": ["Example"]} for r in rows]
    inputs, label_path = tmp_path / "inputs.jsonl", tmp_path / "labels.jsonl"
    inputs.write_text("".join(json.dumps(r) + "\n" for r in rows))
    label_path.write_text("".join(json.dumps(r) + "\n" for r in labels))
    p = protocol() | {"inputs": {"path": str(inputs)}, "checker_labels": {"path": str(label_path)},
                     "execution_mode": "test_double_only", "test_double_only": True,
                     "tokenizer_path": "OFFLINE_TEST_DOUBLE", "budget_usd": 1,
                     "max_calls": max_calls, "quotas_per_domain": {"train": 1, "validation": 1},
                     "minimum_graph_citing": {"train": 0, "validation": 0},
                     "workers": workers, "experiment_id": "synthetic_runner_test"}
    (out / "protocol.json").write_text(json.dumps(p))
    monkeypatch.setattr(runner, "verify", lambda _: p)
    monkeypatch.setattr(runner, "verify_source_bindings", lambda _: None)
    tokenizer = Tokenizer()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)))
    return out, p, rows


class Provider:
    def __init__(self, reject_first=False):
        self.calls = []
        self.lock = threading.Lock()
        self.reject_first = reject_first

    def __call__(self, body):
        with self.lock:
            self.calls.append(deepcopy(body))
        is_producer = body["model"] == "deepseek-v4-flash"
        response = teacher() if is_producer else review()
        if (is_producer and self.reject_first
                and "Question: Which fact holds for hotpotqa train rank 1?" in body["messages"][1]["content"]):
            response = teacher("Wrong")
        return envelope(body, response)


def test_concurrent_quota_does_not_overconsume_reserves(tmp_path, monkeypatch):
    out, _, _ = setup_run(tmp_path, monkeypatch)
    provider = Provider()
    report = runner.run(out, env_file=tmp_path / "NONEXISTENT", transport=provider)
    assert report["accepted"] == 6
    assert all(v == 1 for v in report["counts"].values())
    assert report["quotas_met"]
    assert report["api_calls"] == len(provider.calls) == 12
    assert report["status"] == "TEST_DOUBLE_ONLY_NOT_SFT_DATA"
    assert report["data_ready"] is False
    producer_users = [body["messages"][1]["content"] for body in provider.calls if body["model"] == "deepseek-v4-flash"]
    assert all("rank 1?" in user for user in producer_users)


def test_rejected_pending_candidate_keeps_later_reserve_available(tmp_path, monkeypatch):
    out, _, _ = setup_run(tmp_path, monkeypatch)
    provider = Provider(reject_first=True)
    report = runner.run(out, env_file=tmp_path / "NONEXISTENT", transport=provider)
    assert report["quotas_met"]
    assert report["accepted"] == 6 and report["processed"] == 7
    assert report["api_calls"] == len(provider.calls) == 13
    decisions = [json.loads(f.read_text()) for f in (out / "decisions").glob("*.json")]
    hotpot_train = [d for d in decisions if d["dataset"] == "hotpotqa" and d["split"] == "train"]
    assert {d["question_key"]: d["accepted"] for d in hotpot_train} == {
        "hotpotqa::train_1": False, "hotpotqa::train_2": True,
    }


def test_partial_resume_fills_quotas_without_repaying_completed_calls(tmp_path, monkeypatch):
    out, _, _ = setup_run(tmp_path, monkeypatch)
    provider = Provider()
    first = runner.run(out, env_file=tmp_path / "NONEXISTENT", max_questions=2, transport=provider)
    assert first["accepted"] == 2 and first["api_calls"] == 4
    second = runner.run(out, env_file=tmp_path / "NONEXISTENT", transport=provider)
    assert second["accepted"] == 6 and second["api_calls"] == 12
    assert len(provider.calls) == 12
    third = runner.run(out, env_file=tmp_path / "NONEXISTENT", transport=provider)
    assert third["accepted"] == 6 and len(provider.calls) == 12
    assert len(list(out.glob("progress_*.json"))) == 3


def test_concurrent_paid_call_limit_never_exceeded(tmp_path, monkeypatch):
    out, _, _ = setup_run(tmp_path, monkeypatch, max_calls=5)
    provider = Provider()
    report = runner.run(out, env_file=tmp_path / "NONEXISTENT", transport=provider)
    assert report["stop_reason"] == "frozen_api_budget_stop"
    assert report["api_calls"] == len(provider.calls) == 5
    assert report["accepted"] <= 2
    assert report["data_ready"] is False


def test_live_mode_cannot_accept_custom_provider(tmp_path, monkeypatch):
    out, p, _ = setup_run(tmp_path, monkeypatch)
    p.update(execution_mode="live_official_deepseek", test_double_only=False)
    with pytest.raises(ValueError, match="custom transports"):
        runner.run(out, env_file=tmp_path / "NONEXISTENT", transport=Provider())


def test_test_double_mode_cannot_read_live_credentials(tmp_path, monkeypatch):
    out, _, _ = setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "live_transport", lambda _: pytest.fail("no credential access"))
    with pytest.raises(ValueError, match="live API forbidden"):
        runner.run(out, env_file=tmp_path / "NONEXISTENT")


def as_verified_graph(row):
    result = deepcopy(row)
    result["kg_subgraph"] = [["Subject 1", "has", "documented fact"]]
    result["kg_source_verification"] = {"status": "PASS", "binding": {"path": "OFFLINE_TEST_DOUBLE_SOURCE"}}
    return result


def test_verified_graph_stratum_precedes_small_ordinary_ranks_without_gold():
    ordinary_train = make_row("2wikimultihopqa", "train", 1)
    ordinary_val = make_row("hotpotqa", "validation", 1)
    graph_late = as_verified_graph(make_row("2wikimultihopqa", "train", 4001))
    graph_earlier = as_verified_graph(make_row("musique", "validation", 1200))
    rows = [ordinary_val, graph_late, ordinary_train, graph_earlier]
    expected = [graph_earlier, graph_late, ordinary_train, ordinary_val]
    scheduled = runner.schedule_rows_v3(rows, quotas_per_domain={"train": 2000, "validation": 100})
    assert scheduled == expected
    # The same generic evidence gate, without a 2Wiki branch, applies to all
    # datasets. Ordinary validation retains the prior proportional schedule.
    assert scheduled[0]["dataset"] == "musique"
    assert rows[0] is ordinary_val  # source order/records were not rewritten


def test_graph_rank_then_split_dataset_ties_are_frozen():
    rows = [as_verified_graph(make_row(ds, split, 7))
            for ds in reversed(runner.DATASETS) for split in ("validation", "train")]
    scheduled = runner.schedule_rows_v3(rows, quotas_per_domain={"train": 2000, "validation": 100})
    assert [(row["selection_rank"], row["split"], row["dataset"]) for row in scheduled] == sorted(
        (row["selection_rank"], row["split"], row["dataset"]) for row in rows)


def test_nonempty_unverified_graph_cannot_gain_priority():
    bad = as_verified_graph(make_row())
    bad["kg_source_verification"]["status"] = "UNVERIFIED"
    with pytest.raises(ValueError, match="source PASS"):
        runner.schedule_rows_v3([bad], quotas_per_domain={"train": 1, "validation": 1})


def test_late_graph_candidate_gets_quota_opportunity_and_citation_is_not_forced(tmp_path, monkeypatch):
    out, p, rows = setup_run(tmp_path, monkeypatch)
    replaced = [as_verified_graph(row) if row["question_key"] == "2wikimultihopqa::train_3" else row for row in rows]
    input_path = runner.Path(p["inputs"]["path"])
    input_path.write_text("".join(json.dumps(row) + "\n" for row in replaced))
    p["minimum_graph_citing"]["train"] = 1
    provider = Provider()
    result = runner.run(out, env_file=tmp_path / "NONEXISTENT", transport=provider)
    decisions = [json.loads(path.read_text()) for path in (out / "decisions").glob("*.json")]
    selected = [item for item in decisions if item["dataset"] == "2wikimultihopqa" and item["split"] == "train"]
    assert len(selected) == 1
    assert selected[0]["question_key"] == "2wikimultihopqa::train_3"
    assert selected[0]["accepted"] is True
    assert selected[0]["record"]["graph_citing_steps"] == 0
    assert result["quotas_met"] is True
    assert result["graph_coverage_met"] is False
    assert result["data_ready"] is False
    # Nonempty KG is not an instruction to force an irrelevant citation.
    assert "Knowledge Used: []" in selected[0]["record"]["teacher_output"]
    assert len(provider.calls) == 12
