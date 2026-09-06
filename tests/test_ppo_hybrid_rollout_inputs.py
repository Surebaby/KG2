"""Fail-fast checks for versioned PPO passage overrides and rollout schedules."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from kgproweight.training.phase3_ppo import (
    Phase3PPOConfig,
    _load_hybrid_rollout_inputs,
    _prepare_prompts,
)
from kgproweight.data.silver_dataset import SilverDatasetReader


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class _Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        return "\n".join(message["content"] for message in messages)

    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(len(text.split())))}


def _assets(tmp_path: Path):
    silver = tmp_path / "silver.jsonl"
    silver_rows = [
        {
            "qid": f"q{i}",
            "question": f"question {i}",
            "answer": f"answer {i}",
            "dataset": "hotpotqa",
            "accepted": True,
            "steps": [],
            "kg_subgraph": [],
            "retrieved_passages": [{"title": "old", "text": "old evidence"}],
            "metadata": {"gold_answer": f"answer {i}"},
        }
        for i in range(5)
    ]
    _write_jsonl(silver, silver_rows)

    # Mirror the legacy sampler for two batches. Replay is disabled in this
    # unit fixture; its shared-RNG path is covered by test_ppo_rollout_schedule.
    generator = torch.Generator().manual_seed(42)
    chosen = []
    for _ in range(2):
        chosen.extend(torch.randint(0, 5, (2,), generator=generator).tolist())

    schedule = tmp_path / "schedule.jsonl"
    _write_jsonl(
        schedule,
        [
            {"rollout_index": index, "qid": f"q{sample_index}"}
            for index, sample_index in enumerate(chosen, start=1)
        ],
    )
    overrides = tmp_path / "overrides.jsonl"
    _write_jsonl(
        overrides,
        [
            {
                "qid": f"q{i}",
                "question": f"question {i}",
                "retrieved_passages": [
                    {"title": "hybrid", "text": f"bridge evidence {i}"}
                ],
            }
            for i in sorted(set(chosen))
        ],
    )
    return silver, overrides, schedule, chosen


def test_hybrid_inputs_reproduce_current_sampler(tmp_path: Path):
    silver, overrides, schedule, chosen = _assets(tmp_path)
    cfg = Phase3PPOConfig(
        silver_path=str(silver),
        output_dir=str(tmp_path / "output"),
        total_steps=4,
        batch_size=2,
        seed=42,
        split=None,
        sft_replay_ratio=0.0,
        sft_anchor_weight=0.0,
        passage_overrides_path=str(overrides),
        rollout_schedule_path=str(schedule),
    )

    loaded, qids, metadata = _load_hybrid_rollout_inputs(cfg)

    assert qids == [f"q{i}" for i in chosen]
    assert set(loaded) == set(qids)
    assert metadata["scheduled_rollouts"] == 4
    assert metadata["scheduled_unique_qids"] == len(set(qids))


def test_hybrid_paths_must_be_paired(tmp_path: Path):
    silver, overrides, _, _ = _assets(tmp_path)
    cfg = Phase3PPOConfig(
        silver_path=str(silver),
        output_dir=str(tmp_path / "output"),
        passage_overrides_path=str(overrides),
    )

    try:
        _load_hybrid_rollout_inputs(cfg)
    except ValueError as exc:
        assert "must be provided together" in str(exc)
    else:
        raise AssertionError("unpaired hybrid input was accepted")


def test_schedule_drift_is_rejected_before_model_allocation(tmp_path: Path):
    silver, overrides, schedule, _ = _assets(tmp_path)
    rows = [json.loads(line) for line in schedule.read_text(encoding="utf-8").splitlines()]
    rows[0]["qid"] = "q4" if rows[0]["qid"] != "q4" else "q3"
    _write_jsonl(schedule, rows)
    cfg = Phase3PPOConfig(
        silver_path=str(silver),
        output_dir=str(tmp_path / "output"),
        total_steps=4,
        batch_size=2,
        seed=42,
        split=None,
        sft_replay_ratio=0.0,
        sft_anchor_weight=0.0,
        passage_overrides_path=str(overrides),
        rollout_schedule_path=str(schedule),
    )

    try:
        _load_hybrid_rollout_inputs(cfg)
    except ValueError as exc:
        assert (
            "missing passage overrides" in str(exc)
            or "first mismatch at rollout 1" in str(exc)
        )
    else:
        raise AssertionError("drifted rollout schedule was accepted")


def test_override_reaches_policy_prompt_and_reward_evidence(tmp_path: Path):
    silver, overrides, schedule, _ = _assets(tmp_path)
    cfg = Phase3PPOConfig(
        silver_path=str(silver),
        output_dir=str(tmp_path / "output"),
        total_steps=4,
        batch_size=2,
        seed=42,
        split=None,
        sft_replay_ratio=0.0,
        sft_anchor_weight=0.0,
        passage_overrides_path=str(overrides),
        rollout_schedule_path=str(schedule),
        max_input_length=4096,
    )
    loaded, _, _ = _load_hybrid_rollout_inputs(cfg)

    rows = _prepare_prompts(
        SilverDatasetReader(silver),
        _Tokenizer(),
        cfg,
        passage_overrides=loaded,
    )
    row = next(item for item in rows if item["spec"].metadata["qid"] in loaded)

    assert "bridge evidence" in row["prompt"]
    assert "old evidence" not in row["prompt"]
    assert row["spec"].retrieved_passages == loaded[row["spec"].metadata["qid"]][
        "retrieved_passages"
    ]
    assert row["spec"].metadata["passage_override_applied"] is True
