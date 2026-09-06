from kgproweight.data.silver_dataset import SilverTrajectory
from scripts.pilot.build_ppo_rollout_schedule import build_schedule
from kgproweight.training.phase3_ppo import _sample_rollout_indices


def _item(qid):
    return SilverTrajectory(
        qid=qid,
        question=f"question {qid}",
        answer="answer",
        dataset="hotpotqa",
        steps=[],
    )


def test_rollout_schedule_is_deterministic_and_counts_replay_draws():
    samples = [_item(f"q{i}") for i in range(20)]
    first, first_replay = build_schedule(
        samples, seed=42, batch_size=4, total_steps=20, replay_ratio=0.10
    )
    second, second_replay = build_schedule(
        samples, seed=42, batch_size=4, total_steps=20, replay_ratio=0.10
    )

    assert [item.qid for item in first] == [item.qid for item in second]
    assert len(first) == 20
    assert first_replay == second_replay == 2


def test_rollout_schedule_rejects_partial_batch():
    try:
        build_schedule([_item("q")], seed=1, batch_size=4, total_steps=6, replay_ratio=0)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_same_prompt_rollouts_are_contiguous_groups_and_deterministic():
    import torch

    first = _sample_rollout_indices(
        20, batch_size=8, rollouts_per_prompt=4,
        generator=torch.Generator().manual_seed(42),
    )
    second = _sample_rollout_indices(
        20, batch_size=8, rollouts_per_prompt=4,
        generator=torch.Generator().manual_seed(42),
    )
    assert first == second
    assert first[:4] == [first[0]] * 4
    assert first[4:] == [first[4]] * 4


def test_same_prompt_rollouts_require_exact_batch_divisibility():
    import pytest
    import torch

    with pytest.raises(ValueError, match="divide batch_size"):
        _sample_rollout_indices(
            20, batch_size=6, rollouts_per_prompt=4,
            generator=torch.Generator().manual_seed(42),
        )


def test_schedule_builder_supports_same_prompt_k4():
    samples = [_item(f"q{i}") for i in range(20)]
    schedule, _ = build_schedule(
        samples, seed=42, batch_size=4, total_steps=8,
        replay_ratio=0.0, rollouts_per_prompt=4,
    )
    assert [row.qid for row in schedule[:4]] == [schedule[0].qid] * 4
    assert [row.qid for row in schedule[4:]] == [schedule[4].qid] * 4
