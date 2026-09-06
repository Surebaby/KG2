"""Fail-closed sequencing and immutable publication for the fresh confirmation."""
import json
import fcntl
import signal
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.pilot import run_source_credit_v2_fresh_confirmation_v1 as runner
from scripts.prepare import freeze_source_credit_v2_fresh_confirmation_v1 as freezer


def protocol(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({"schema_version": freezer.SCHEMA, "status": "FROZEN",
        "experiment_id": "TEST-ONLY", "code_bindings": {
            "scripts/pilot/run_source_credit_v2_fresh_confirmation_v1.py": {
                "sha256": runner.sha(runner.__file__)}}}))
    return path


def test_exclusive_protocol_publication(tmp_path):
    path = tmp_path / "frozen.json"
    freezer.write_new(path, {"value": 1})
    with pytest.raises(FileExistsError):
        freezer.write_new(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


def test_binding_fails_when_source_changes(tmp_path):
    path = tmp_path / "source"
    path.write_text("original")
    binding = freezer.bind(path)
    path.write_text("altered")
    with pytest.raises(ValueError, match="changed"):
        freezer.checked(binding)


def test_generation_contract_matches_production():
    from scripts.prepare.generate_source_credit_v2_fresh_confirmation_v1 import validate_generation_contract
    validate_generation_contract({"schema_version": freezer.SCHEMA, "status": "FROZEN",
        "experiment_id": "TEST", "seed": 42, "generation": freezer.generation_contract()})


def test_failed_generation_never_reaches_score_or_gold(tmp_path):
    p = protocol(tmp_path)
    out = tmp_path / "run"
    with patch.object(runner.subprocess, "Popen") as start:
        start.return_value.pid = 123
        start.return_value.wait.return_value = 7
        assert runner.run(p, out) == 7
    assert start.call_count == 1
    assert "generate_source_credit" in start.call_args.args[0][2]
    assert json.loads((out / "status.json").read_text())["status"] == "FAILED_PREFIX_RETAINED"
    assert not (out / "analysis").exists()


def test_failed_scoring_never_reads_gold(tmp_path):
    p = protocol(tmp_path)
    out = tmp_path / "run"
    with patch.object(runner.subprocess, "Popen") as start:
        start.return_value.pid = 123
        start.return_value.wait.side_effect = [0, 9]
        assert runner.run(p, out) == 9
    assert start.call_count == 2
    assert all("analyze_" not in str(call.args[0]) for call in start.call_args_list)


def test_completed_pipeline_cannot_start_ppo(tmp_path):
    p = protocol(tmp_path)
    out = tmp_path / "run"
    with patch.object(runner.subprocess, "Popen") as start:
        start.return_value.pid = 123
        start.return_value.wait.return_value = 0
        assert runner.run(p, out) == 0
    assert start.call_count == 3
    status = json.loads((out / "status.json").read_text())
    assert status["optimizer_updates"] == 0 and status["ppo_started"] is False


def test_resume_does_not_repeat_label_analysis(tmp_path):
    p = protocol(tmp_path)
    out = tmp_path / "run"
    out.mkdir()
    (out / "started.json").write_text(json.dumps({"experiment_id": "TEST-ONLY",
        "protocol_sha256": runner.sha(p), "protocol": str(p), "optimizer_updates": 0}))
    (out / "analysis").mkdir()
    with pytest.raises(ValueError, match="analysis already attempted"):
        runner.run(p, out, resume=True)


def test_resume_only_committed_started_stages(tmp_path):
    out = tmp_path / "run"
    (out / "generation").mkdir(parents=True)
    (out / "generation" / "started.json").write_text("{}")
    stages = dict(runner.commands(tmp_path / "p.json", out, resume=True))
    assert "--resume" in stages["generation"]
    assert "--resume" not in stages["scoring"]
    assert "--resume" not in stages["analysis"]


def test_parallel_controller_is_rejected_before_subprocess(tmp_path):
    p = protocol(tmp_path)
    out = tmp_path / "run"
    with (tmp_path / "run.controller.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with patch.object(runner.subprocess, "Popen") as start:
            with pytest.raises(BlockingIOError):
                runner.run(p, out)
            start.assert_not_called()


def test_sigterm_stops_child_and_preserves_interruption(tmp_path):
    p = protocol(tmp_path)
    out = tmp_path / "run"
    original = signal.getsignal(signal.SIGTERM)
    calls = 0

    def waiting():
        nonlocal calls
        calls += 1
        if calls == 1:
            signal.raise_signal(signal.SIGTERM)
        return 130

    with patch.object(runner.subprocess, "Popen") as start:
        start.return_value.pid = 123
        start.return_value.poll.return_value = None
        start.return_value.wait.side_effect = waiting
        assert runner.run(p, out) == 130
        start.return_value.send_signal.assert_called_once_with(signal.SIGINT)
        assert calls == 2
    assert signal.getsignal(signal.SIGTERM) == original
    assert json.loads((out / "status.json").read_text())["status"] == "INTERRUPTED_PREFIX_RETAINED"
