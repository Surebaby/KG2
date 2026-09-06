import json

from kgproweight.training.query_planner import (
    balanced_sample,
    encode_record,
    planner_messages,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize
        text = "".join(f"{row['role']}:{row['content']}\n" for row in messages)
        if add_generation_prompt:
            text += "assistant:"
        return text

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": list(text.encode())}


def _record(dataset, qid):
    return {
        "question_key": f"{dataset}::{qid}",
        "dataset": dataset,
        "qid": qid,
        "question": "Who directed Film A?",
        "target_type": "relation_graph" if dataset == "2wikimultihopqa" else "subquery_graph",
        "target": {"anchors": ["Film A"], "steps": []} if dataset == "2wikimultihopqa" else {"steps": []},
    }


def test_planner_prompt_does_not_include_target_or_answer():
    record = _record("2wikimultihopqa", "q1")
    record["target"]["secret"] = "SHOULD_NOT_APPEAR"
    messages = planner_messages(record, include_target=False)
    assert "SHOULD_NOT_APPEAR" not in json.dumps(messages)
    assert "Do not answer" in messages[0]["content"]


def test_encode_masks_prompt_and_keeps_only_assistant_target():
    record = _record("2wikimultihopqa", "q1")
    encoded = encode_record(record, FakeTokenizer(), max_seq_length=4096)
    first_supervised = next(index for index, value in enumerate(encoded["labels"]) if value != -100)
    assert all(value == -100 for value in encoded["labels"][:first_supervised])
    assert encoded["labels"][first_supervised:] == encoded["input_ids"][first_supervised:]


def test_balanced_sample_is_deterministic(tmp_path):
    path = tmp_path / "split.jsonl"
    rows = [
        _record(dataset, f"q{index}")
        for dataset in ("2wikimultihopqa", "musique")
        for index in range(5)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    first = balanced_sample(path, per_dataset=3, seed=42)
    second = balanced_sample(path, per_dataset=3, seed=42)
    assert [row["question_key"] for row in first] == [row["question_key"] for row in second]
    assert {dataset: sum(row["dataset"] == dataset for row in first) for dataset in ("2wikimultihopqa", "musique")} == {
        "2wikimultihopqa": 3, "musique": 3,
    }
