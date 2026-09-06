"""Exercise orchestration boundaries without loading models or running CUDA."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.train import launch_sourcegate_preparation_v1 as launcher


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    asset = tmp_path / "frozen.txt"
    asset.write_text("frozen")
    manifest = tmp_path / launcher.RELEASE
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"files": {"frozen.txt": {
        "size_bytes": asset.stat().st_size,
        "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
    }}}))
    bank_manifest = tmp_path / launcher.BANK / "manifest.json"
    bank_manifest.parent.mkdir(parents=True)
    bank_manifest.write_text("{}")

    class Matrix:
        def __matmul__(self, other):
            return self

        def __getitem__(self, key):
            return 32.0

    fake_torch = SimpleNamespace(
        __version__="test-only",
        bfloat16="bf16",
        ones=lambda *args, **kwargs: Matrix(),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda: True,
            get_device_properties=lambda index: SimpleNamespace(total_memory=96 * 1024**3),
            get_device_name=lambda index: "test GPU",
            empty_cache=lambda: None,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "scripts.prepare.source_quality_candidate_bank_v1", SimpleNamespace(
        load_release=lambda *args: {}, PREPARE_VERSION="test-only",
        validate_code=lambda *args: None,
        validate_inputs=lambda *args: [{}] * 830,
    ))
    state = SimpleNamespace(calls=[], exits={}, counts={}, gate_clearance=True, report_clearance=True)

    class Process:
        def __init__(self, cmd, *, cwd, stdin, stdout, stderr):
            module = cmd[3]
            stage = "calibrate" if module.endswith("calibrate_source_quality_gate_v1") else cmd[4]
            state.calls.append(stage)
            output = Path(cwd) / cmd[cmd.index("--output-dir") + 1]
            output.mkdir(parents=True, exist_ok=False)
            if stage == "calibrate":
                (output / "gate.json").write_text(json.dumps({"training_clearance": state.gate_clearance}))
                (output / "report.json").write_text(json.dumps({"training_clearance": state.report_clearance}))
            else:
                filename = "generations.jsonl" if stage == "generate" else "candidates.scored.jsonl"
                (output / filename).write_text("{}\n" * state.counts.get(stage, 1660))
            stdout.write((stage + " child output\n").encode())
            self.stage = stage
            self.pid = 1000 + len(state.calls)
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = state.exits.get(self.stage, 0)
            return self.returncode

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    return tmp_path, state


def read_status(root):
    return json.loads((root / "run" / "status.json").read_text())


def test_success_stops_before_ppo_and_records_progress(release):
    root, state = release
    assert launcher.run(root / "run") == 0
    assert state.calls == ["generate", "score", "calibrate"]
    status = read_status(root)
    assert status["status"] == "AWAITING_PROCESS_UTILITY_CHECK"
    assert status["ppo_started"] is False
    assert status["research_policy_updates"] == 0
    events = [json.loads(line) for line in (root / "run/events.jsonl").read_text().splitlines()]
    progress = [row for row in events if row.get("stage") == "generate" and "expected" in row]
    assert progress and progress[0]["completed"] == progress[0]["expected"] == 1660
    assert "generate child output" in (root / "run/generate.log").read_text()
    manifest = json.loads((root / "run/manifest.json").read_text())
    assert manifest["file_verification"]["all_sha256_match"] is True
    assert manifest["runner_sha256"] == launcher.file_sha(launcher.__file__)
    assert "outcome" not in json.dumps(manifest["stages"])
    assert all("phase3_ppo" not in " ".join(stage["command"]) for stage in manifest["stages"])


@pytest.mark.parametrize("stage,index", [("generate", 1), ("score", 2), ("calibrate", 3)])
def test_nonzero_child_exit_stops_later_stages(release, stage, index):
    root, state = release
    state.exits[stage] = 17
    with pytest.raises(RuntimeError, match=f"{stage} exited 17"):
        launcher.run(root / "run")
    assert state.calls == ["generate", "score", "calibrate"][:index]
    assert read_status(root)["status"] == "FAILED"
    assert read_status(root)["stage"] == stage
    assert (root / "run" / (stage + ".log")).exists()


@pytest.mark.parametrize("gate,report", [(False, True), (True, False), (False, False)])
def test_exit_zero_with_failed_calibration_is_not_clearance(release, gate, report):
    root, state = release
    state.gate_clearance, state.report_clearance = gate, report
    assert launcher.run(root / "run") == 2
    assert state.calls == ["generate", "score", "calibrate"]
    assert read_status(root)["status"] == "STOPPED_CALIBRATION_FAILED"


@pytest.mark.parametrize("stage,index", [("generate", 1), ("score", 2)])
def test_exit_zero_with_incomplete_population_stops(release, stage, index):
    root, state = release
    state.counts[stage] = 1659
    with pytest.raises(RuntimeError, match="incomplete candidate population"):
        launcher.run(root / "run")
    assert state.calls == ["generate", "score"][:index]
    assert read_status(root)["status"] == "FAILED"


@pytest.mark.parametrize("existing", [launcher.GENERATED, launcher.SCORED, launcher.CALIBRATION, "run"])
def test_previous_assets_are_preserved_and_no_child_starts(release, existing):
    root, state = release
    output = root / existing
    output.mkdir(parents=True)
    sentinel = output / "retained.txt"
    sentinel.write_text("previous research result")
    with pytest.raises(FileExistsError):
        launcher.run(root / "run")
    assert sentinel.read_text() == "previous research result"
    assert state.calls == []


def test_changed_release_asset_blocks_all_gpu_children(release):
    root, state = release
    (root / "frozen.txt").write_text("broken")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        launcher.run(root / "run")
    assert state.calls == []
    assert read_status(root)["status"] == "FAILED"


def test_line_count_handles_absent_and_partial_records(tmp_path):
    assert launcher.line_count(None) == 0
    path = tmp_path / "progress.jsonl"
    assert launcher.line_count(path) == 0
    path.write_bytes(b"{}\n{}\npartial")
    assert launcher.line_count(path) == 2
