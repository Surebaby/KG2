"""Regression tests for matched full-trajectory PPO supervised replay."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from kgproweight.training.phase3_ppo import (
    _advance_replay_credit,
    _prepare_sft_anchor_data,
)


class _CharTokenizer:
    """Tiny deterministic tokenizer with a chat-template prefix invariant."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, *, truncation=False, max_length=None, **kwargs):
        ids = [ord(ch) for ch in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def _write_silver(path: Path) -> None:
    row = {
        "qid": "train_1",
        "question": "Who wrote the work?",
        "answer": "Alice",  # bare answer: must never be the replay target alone
        "dataset": "hotpotqa",
        "accepted": True,
        "steps": [
            {
                "index": 1,
                "text": "Reasoning: Find the author.\nConclusion: The author is Alice.",
                "label": 1.0,
                "cited_triples": [["Work", "author", "Alice"]],
            },
            {
                "index": 2,
                "text": "Reasoning: Verify the name.\nConclusion: Alice is confirmed.",
                "label": 0.0,
                "cited_triples": [],
            },
        ],
        "kg_subgraph": [["Work", "author", "Alice"]],
        "retrieved_passages": [],
        "metadata": {"gold_answer": "Alice"},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_replay_target_is_full_standard_trajectory(tmp_path: Path):
    silver = tmp_path / "silver.jsonl"
    _write_silver(silver)
    cfg = SimpleNamespace(
        split=None,
        build_split_spec=lambda: None,
        ppo_max_passages=15,
        ppo_max_kg_triples=12,
        max_input_length=4096,
        max_new_tokens=256,
        seed=42,
    )

    rows = _prepare_sft_anchor_data(str(silver), _CharTokenizer(), cfg)

    assert len(rows) == 1
    assert rows[0]["qid"] == "train_1"
    trace = rows[0]["answer_trace"]
    assert "[Step 1]" in trace and "[Step 2]" in trace
    assert "[Final Answer]\nAlice" in trace
    assert trace.strip() != "Alice"
    assert any(label == -100 for label in rows[0]["labels"])
    assert any(label != -100 for label in rows[0]["labels"])


def test_replay_drops_passages_instead_of_truncating_final_answer(tmp_path: Path):
    silver = tmp_path / "silver.jsonl"
    _write_silver(silver)
    row = json.loads(silver.read_text(encoding="utf-8"))
    row["retrieved_passages"] = [
        {"title": "Too long", "text": "x" * 5000, "score": 1.0},
    ]
    silver.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cfg = SimpleNamespace(
        split=None,
        build_split_spec=lambda: None,
        ppo_max_passages=15,
        ppo_max_kg_triples=12,
        max_input_length=1600,
        max_new_tokens=512,
        seed=42,
    )

    rows = _prepare_sft_anchor_data(str(silver), _CharTokenizer(), cfg)

    assert len(rows) == 1
    assert rows[0]["num_passages"] == 0
    supervised = [
        token for token, label in zip(rows[0]["input_ids"], rows[0]["labels"])
        if label != -100
    ]
    assert "[Final Answer]\nAlice" in "".join(chr(token) for token in supervised)


def test_independent_replay_source_does_not_apply_rollout_question_kg(
    tmp_path: Path, monkeypatch,
):
    silver = tmp_path / "hotpot_replay.jsonl"
    _write_silver(silver)
    cfg = SimpleNamespace(
        split=None,
        build_split_spec=lambda: None,
        ppo_max_passages=15,
        ppo_max_kg_triples=12,
        max_input_length=4096,
        max_new_tokens=256,
        seed=42,
        question_kg_records_path="automatic_2wiki_only.jsonl",
    )
    monkeypatch.setattr(
        "kgproweight.training.phase3_ppo.read_question_kg_records",
        lambda _path: (_ for _ in ()).throw(AssertionError("must not join rollout KG")),
    )

    rows = _prepare_sft_anchor_data(
        str(silver), _CharTokenizer(), cfg,
        apply_rollout_question_kg=False,
    )
    assert len(rows) == 1
    assert rows[0]["qid"] == "train_1"


def test_ten_percent_replay_is_not_truncated_to_zero_at_batch_four():
    credit = 0.0
    due = []
    for _ in range(25):  # 100 PPO samples
        n, credit = _advance_replay_credit(
            credit, batch_size=4, replay_ratio=0.10,
        )
        due.append(n)

    assert due[:5] == [0, 0, 1, 0, 1]
    assert sum(due) == 10
    assert credit == 0.0


def test_replay_ratio_validation():
    for bad in (-0.01, 1.01):
        try:
            _advance_replay_credit(0.0, batch_size=4, replay_ratio=bad)
        except ValueError as exc:
            assert "sft_replay_ratio" in str(exc)
        else:
            raise AssertionError(f"invalid ratio {bad} was accepted")
