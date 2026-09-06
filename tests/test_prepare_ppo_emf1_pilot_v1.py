from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.prepare import prepare_ppo_emf1_pilot_v1 as pilot


def _parent_bytes(groups=150):
    lines = []
    for group in range(1, groups + 1):
        for within in range(1, 5):
            row = {
                "dataset": pilot.DATASETS[(group - 1) % 3], "qid": f"题目-{group}",
                "question_sha256": f"hash-{group}", "prompt_group_index": group,
                "rollout_index": (group - 1) * 4 + within, "within_group_rollout": within,
                "process_reward_eligible": group % 3 == 2,
            }
            # Deliberately preserve unusual whitespace, Unicode and CRLF.  A
            # json.loads/json.dumps rewrite would fail the exact-byte check.
            lines.append((json.dumps(row, ensure_ascii=False, separators=(", ", " : ")) + "\r\n").encode())
    return b"".join(lines)


@pytest.mark.parametrize(("count", "quota"), [(12, 1), (600, 50)])
def test_prefix_preserves_parent_bytes_and_balanced_complete_groups(count, quota):
    parent = _parent_bytes()
    prefix, stats = pilot.schedule_prefix(parent, count, quota)
    expected = b"".join(parent.splitlines(keepends=True)[:count])
    assert prefix == expected
    assert prefix.endswith(b"\r\n")
    assert "题目".encode() in prefix
    assert stats["groups_by_dataset"] == {name: quota for name in pilot.DATASETS}
    assert stats["prompt_groups"] == count // 4
    assert stats["graph_groups"] == quota


@pytest.mark.parametrize(("field", "value", "match"), [
    ("qid", "other-question", "mixed or missing question identities"),
    ("within_group_rollout", 4, "within-group"),
    ("prompt_group_index", 3, "group indices"),
    ("rollout_index", 99, "rollout indices"),
    ("process_reward_eligible", True, "inconsistent source"),
])
def test_prefix_rejects_incomplete_or_corrupted_k4_contract(field, value, match):
    rows = [json.loads(line) for line in _parent_bytes().splitlines()]
    rows[0][field] = value
    parent = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    with pytest.raises(ValueError, match=match):
        pilot.schedule_prefix(parent, 12, 1)


def test_prefix_rejects_dataset_imbalance_and_non_group_lengths():
    with pytest.raises(ValueError, match="not balanced"):
        pilot.schedule_prefix(_parent_bytes(), 12, 50)
    with pytest.raises(ValueError, match="complete K4"):
        pilot.schedule_prefix(_parent_bytes(), 11, 1)


@pytest.fixture(scope="module")
def actual_configs():
    # The actual CLI resolver replaces only the final training call; no models
    # are loaded and this verifies forwarded values, rather than YAML alone.
    return {arm: pilot.resolve_phase3_ppo_runtime_config(path) for arm, path in pilot.CONFIGS.items()}


@pytest.mark.parametrize("arm", ["outcome12000", "probe12", "smoke600"])
def test_real_cli_configs_obey_outcome_only_pilot_contract(actual_configs, arm):
    pilot.validate_runtime_config(
        actual_configs[arm], arm=arm,
        parent_dir=pilot.ROOT / pilot.DEFAULT_DATA_DIR,
        replay_dir=pilot.ROOT / pilot.DEFAULT_REPLAY_DIR,
    )


@pytest.mark.parametrize(("field", "bad"), [
    ("runtime_contract_version", "v1"), ("gamma", 0.95), ("lam", 0.95),
    ("mixed_text_reward", True), ("proofkg_process_reward", True),
    ("sft_replay_ratio", 0.0), ("sft_anchor_weight", 0.0),
    ("ppo_max_passages", 10), ("alpha_gate_path", "old-alpha.pt"),
    ("total_steps", 600), ("rollouts_per_prompt", 2),
    ("fixed_rollout_schedule_path", "old/schedule.jsonl"),
])
def test_runtime_contract_rejects_silent_forwarding_or_source_drift(actual_configs, field, bad):
    cfg = copy.deepcopy(actual_configs["probe12"])
    cfg[field] = bad
    with pytest.raises(ValueError, match="runtime config contract mismatch"):
        pilot.validate_runtime_config(
            cfg, arm="probe12", parent_dir=pilot.ROOT / pilot.DEFAULT_DATA_DIR,
            replay_dir=pilot.ROOT / pilot.DEFAULT_REPLAY_DIR,
        )


def test_nonempty_output_refuses_before_reading_any_upstream_assets(tmp_path, monkeypatch):
    existing = tmp_path / "keep.json"
    existing.write_bytes(b"original research record")
    monkeypatch.setattr(pilot, "_config_sources", lambda *_: pytest.fail("must reject before upstream reads"))
    with pytest.raises(FileExistsError):
        pilot.prepare(tmp_path, experiment_id="UNIT-NONEMPTY")
    assert existing.read_bytes() == b"original research record"
    assert list(tmp_path.iterdir()) == [existing]


def test_failure_is_preserved_and_cannot_be_overwritten(tmp_path, monkeypatch):
    def unavailable(*_):
        raise ValueError("synthetic missing config prerequisite")

    monkeypatch.setattr(pilot, "_config_sources", unavailable)
    with pytest.raises(ValueError, match="synthetic missing"):
        pilot.prepare(tmp_path, experiment_id="UNIT-FAILURE-RECORD")
    failure_path = tmp_path / "FAILED_PREPARATION.json"
    failure = json.loads(failure_path.read_text())
    assert failure["experiment_id"] == "UNIT-FAILURE-RECORD"
    assert failure["status"] == "FAIL_CONFIG_PREPARATION_NOT_TRAINED"
    assert failure["training_started"] is False
    assert not (tmp_path / "manifest.json").exists()
    previous = failure_path.read_bytes()
    with pytest.raises(FileExistsError):
        pilot.prepare(tmp_path, experiment_id="UNIT-RETRY-NOT-ALLOWED")
    assert failure_path.read_bytes() == previous
