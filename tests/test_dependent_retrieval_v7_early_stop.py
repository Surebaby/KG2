from __future__ import annotations

from pathlib import Path

import pytest

from scripts.diagnose import audit_dependent_retrieval_v7_depth1_upper_bound as audit


def _state(*, key: str, dataset: str, plan: dict, b_slots: dict[str, str]) -> dict:
    return {
        "question_key": key,
        "dataset": dataset,
        "qid": key.split("::", 1)[1],
        "schedule": audit._recompute_schedule(plan),
        "slot_values_B": b_slots,
    }


def _task(key: str, slot: str, *, dataset: str) -> dict:
    return {
        "task_id": f"task::{key}::{slot}",
        "question_key": key,
        "dataset": dataset,
        "qid": key.split("::", 1)[1],
        "producer_slot": slot,
    }


def _answer(task: dict, *, verified: bool) -> dict:
    return {**task, "verified": verified}


def test_question_upper_bound_propagates_only_from_current_verified_roots() -> None:
    plan = {
        "steps": [
            {"output_slot": "hop_1", "dependencies": []},
            {"output_slot": "hop_2", "dependencies": ["hop_1"]},
            {"output_slot": "hop_3", "dependencies": ["hop_2"]},
        ]
    }
    state = _state(
        key="hotpotqa::q1",
        dataset="hotpotqa",
        plan=plan,
        b_slots={"slot_1": "entity"},
    )
    task = _task("hotpotqa::q1", "slot_1", dataset="hotpotqa")
    result = audit.question_upper_bound(
        state,
        task_by_slot={"slot_1": task},
        answer_by_slot={"slot_1": _answer(task, verified=True)},
    )

    assert result["current_root_reader_attempts"] == 1
    assert result["current_verified_root_slots"] == 1
    assert result["future_reachable_dependent_hop_ids"] == ["slot_2", "slot_3"]
    # Only slot_2 has a consumer and therefore creates another reader call.
    assert result["future_reader_attempt_slot_ids"] == ["slot_2"]
    assert result["final_reader_attempts_max"] == 2
    assert result["final_mechanically_verified_max"] == 2
    assert result["mechanically_verified_rate_upper_bound"] == 1.0


def test_failed_root_is_permanent_and_blocks_its_descendants() -> None:
    plan = {
        "steps": [
            {"output_slot": "hop_1", "dependencies": []},
            {"output_slot": "hop_2", "dependencies": ["hop_1"]},
        ]
    }
    state = _state(
        key="musique::q1",
        dataset="musique",
        plan=plan,
        b_slots={"slot_1": "entity"},
    )
    task = _task("musique::q1", "slot_1", dataset="musique")
    result = audit.question_upper_bound(
        state,
        task_by_slot={"slot_1": task},
        answer_by_slot={"slot_1": _answer(task, verified=False)},
    )

    assert result["future_reachable_dependent_hops_max"] == 0
    assert result["future_reader_attempts_max"] == 0
    assert result["mechanically_verified_rate_upper_bound"] == 0.0
    assert result["blocked_dependent_steps_even_under_future_success"] == [
        {
            "slot": "slot_2",
            "dependency_depth": 2,
            "missing_B": [],
            "missing_C": ["slot_1"],
        }
    ]


def test_multiple_root_denominator_and_optimistic_future_success() -> None:
    plan = {
        "steps": [
            {"output_slot": "hop_1", "dependencies": []},
            {"output_slot": "hop_2", "dependencies": []},
            {"output_slot": "hop_3", "dependencies": ["hop_1"]},
            {"output_slot": "hop_4", "dependencies": ["hop_2", "hop_3"]},
        ]
    }
    state = _state(
        key="hotpotqa::q2",
        dataset="hotpotqa",
        plan=plan,
        b_slots={"slot_1": "e1", "slot_2": "e2"},
    )
    task1 = _task("hotpotqa::q2", "slot_1", dataset="hotpotqa")
    task2 = _task("hotpotqa::q2", "slot_2", dataset="hotpotqa")
    result = audit.question_upper_bound(
        state,
        task_by_slot={"slot_1": task1, "slot_2": task2},
        answer_by_slot={
            "slot_1": _answer(task1, verified=True),
            "slot_2": _answer(task2, verified=False),
        },
    )

    assert result["current_root_reader_attempts"] == 2
    assert result["current_verified_root_slots"] == 1
    assert result["future_reader_attempts_max"] == 1
    assert result["final_mechanically_verified_max"] == 2
    assert result["mechanically_verified_rate_upper_bound"] == pytest.approx(2 / 3)


def test_compute_upper_bounds_stops_if_either_dataset_cannot_reach_gate() -> None:
    reachable_plan = {
        "steps": [
            {"output_slot": "hop_1", "dependencies": []},
            {"output_slot": "hop_2", "dependencies": ["hop_1"]},
        ]
    }
    states = [
        _state(
            key="hotpotqa::q1",
            dataset="hotpotqa",
            plan=reachable_plan,
            b_slots={"slot_1": "e"},
        ),
        _state(
            key="musique::q1",
            dataset="musique",
            plan=reachable_plan,
            b_slots={"slot_1": "e"},
        ),
    ]
    tasks = [
        _task("hotpotqa::q1", "slot_1", dataset="hotpotqa"),
        _task("musique::q1", "slot_1", dataset="musique"),
    ]
    answers = [
        _answer(tasks[0], verified=True),
        _answer(tasks[1], verified=False),
    ]

    _, by_dataset, status = audit.compute_upper_bounds(
        states, tasks, answers, threshold=0.4
    )

    assert by_dataset["hotpotqa"]["upper_bound_reaches_gate"] is True
    assert by_dataset["musique"]["upper_bound_reaches_gate"] is False
    assert status == audit.FAIL_STATUS


def test_file_lock_detects_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "value.json"
    path.write_text("{}\n", encoding="utf-8")
    expected = audit.file_lock(path)
    path.write_text('{"changed":true}\n', encoding="utf-8")

    with pytest.raises(audit.AuditIntegrityError, match="lock mismatch"):
        audit._assert_file_lock(expected, path, label="fixture")


def test_recursive_gold_key_is_rejected() -> None:
    with pytest.raises(audit.AuditIntegrityError, match="forbidden Gold key"):
        audit._assert_gold_free(
            {"safe": [{"supporting_facts": []}]}, location="fixture"
        )


def test_generator_count_recomputation_is_dataset_specific() -> None:
    rows = [
        {
            "dataset": "hotpotqa",
            "verified": True,
            "telemetry": {
                "strict_parse": {"valid": True},
                "verification": {"reason": "verified"},
            },
        },
        {
            "dataset": "musique",
            "verified": False,
            "telemetry": {
                "strict_parse": {"valid": False},
                "verification": {"reason": "parse_error"},
            },
        },
    ]

    counts = audit._recompute_generator_counts(rows)

    assert counts["tasks"] == 2
    assert counts["mechanically_verified"] == 1
    assert counts["by_dataset"]["hotpotqa"]["mechanically_verified_rate"] == 1.0
    assert counts["by_dataset"]["musique"]["mechanically_verified_rate"] == 0.0
