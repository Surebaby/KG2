"""Local orchestration checks; synthetic records only, no CUDA/model execution."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts.prepare import source_quality_candidate_bank_v1 as bank
from scripts.train import launch_local_sourcegate_generation_v1 as launcher


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    runner = tmp_path / "scripts/train/launch_local_sourcegate_generation_v1.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("synthetic runner source")
    monkeypatch.setattr(launcher, "__file__", str(runner))
    for name in ("scripts/train/launch_sourcegate_preparation_v1.py",
                 "scripts/prepare/source_quality_candidate_bank_v1.py"):
        source = tmp_path / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("synthetic source identity")
    bank_manifest = tmp_path / launcher.BANK / "manifest.json"
    bank_manifest.parent.mkdir(parents=True)
    bank_manifest.write_text("{}")
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda: True,
            get_device_name=lambda index: "synthetic 4090",
            get_device_properties=lambda index: SimpleNamespace(total_memory=24 * 1024**3),
        ),
        version=SimpleNamespace(cuda="synthetic CUDA"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(launcher.importlib.metadata, "version", lambda name: "test-only")
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="test-driver\n"))
    state = SimpleNamespace(calls=[], children=[], count=1660, reported_count=1660, exit_code=0,
                            corrupt=False, failed_release=False, schema=bank.GENERATION_VERSION,
                            wait_error=None, terminate_timeout=False, loads=[])
    original_load = bank.load_release

    def checked_load(directory, expected_schema):
        state.loads.append((directory, expected_schema))
        return original_load(directory, expected_schema)

    monkeypatch.setattr(bank, "load_release", checked_load)

    class Process:
        def __init__(self, command, *, cwd, stdin, stdout, stderr):
            state.calls.append(command)
            state.children.append(self)
            self.pid = 7070
            self.returncode = None
            self.terminated = self.killed = False
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir(parents=True, exist_ok=False)
            (output / "generations.jsonl").write_text("{}\n" * state.count)
            bank.finish(output, {
                "schema_version": state.schema,
                "status": "FROZEN_SFT_CANDIDATES_GENERATED_NOT_SCORED",
                "n_candidates": state.reported_count,
                "bank_manifest_sha256": launcher.file_sha(bank_manifest),
                "training_started": False,
            }, ["generations.jsonl"])
            if state.corrupt:
                with (output / "generations.jsonl").open("a") as handle:
                    handle.write("tampered\n")
            if state.failed_release:
                (output / "FAILED.json").write_text("{}")
            stdout.write(b"synthetic generation output\n")

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if self.killed:
                self.returncode = -9
            elif self.terminated:
                if state.terminate_timeout:
                    raise launcher.subprocess.TimeoutExpired("owned child", timeout)
                self.returncode = -15
            elif state.wait_error is not None:
                error, state.wait_error = state.wait_error, None
                raise error
            else:
                self.returncode = state.exit_code
            return self.returncode

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    return tmp_path, state, fake_torch


def launch(root):
    return launcher.run(root / "run", root / "candidates", "LOCAL-FROM-SCRATCH-TEST")


def status(root):
    return json.loads((root / "run/status.json").read_text())


def test_from_zero_uses_original_generator_and_finishes_before_scoring(release):
    root, state, _ = release
    launch(root)
    assert len(state.calls) == 1
    command = state.calls[0]
    assert command[:5] == [sys.executable, "-u", "-m",
                           "scripts.prepare.source_quality_candidate_bank_v1", "generate"]
    assert "--prefix-dir" not in command and "--generation-dir" not in command
    assert command[command.index("--output-dir") + 1] == str(root / "candidates")
    assert command[command.index("--project-root") + 1] == str(root)
    assert command[command.index("--device") + 1] == "cuda:0"
    assert state.loads == [(root / "candidates", bank.GENERATION_VERSION)]
    manifest = json.loads((root / "run/manifest.json").read_text())
    assert manifest["remote_candidates_reused"] == 0
    assert manifest["research_policy_updates"] == 0
    assert manifest["candidate_bank_sha256"] == launcher.file_sha(root / launcher.BANK / "manifest.json")
    assert manifest["versions"]["torch"] == "test-only"
    assert manifest["gpu_memory_bytes"] == 24 * 1024**3
    events = [json.loads(line) for line in (root / "run/events.jsonl").read_text().splitlines()]
    assert any(row["status"] == "RUNNING" and row["expected"] == 1660 for row in events)
    assert all(row.get("remote_candidates_reused", 0) == 0 for row in events)
    assert status(root)["status"] == "LOCAL_GENERATION_COMPLETE_SCORING_PENDING"
    assert status(root)["completed"] == status(root)["expected"] == 1660
    assert "synthetic generation output" in (root / "run/generate.log").read_text()


@pytest.mark.parametrize("count,reported", [(1659, 1660), (1660, 1659), (1661, 1660)])
def test_incomplete_or_extra_population_never_reports_success(release, count, reported):
    root, state, _ = release
    state.count, state.reported_count = count, reported
    with pytest.raises(RuntimeError, match="complete frozen population"):
        launch(root)
    assert status(root)["status"] == "FAILED"
    assert (root / "candidates/generations.jsonl").exists()


@pytest.mark.parametrize("fault", ["corrupt", "failed_release", "schema"])
def test_success_requires_valid_frozen_manifest_and_artifact_hashes(release, fault):
    root, state, _ = release
    setattr(state, fault, "wrong-version" if fault == "schema" else True)
    with pytest.raises(ValueError):
        launch(root)
    assert status(root)["status"] == "FAILED"
    assert len(state.loads) == 1


def test_nonzero_child_exit_retains_logs_and_does_not_validate_as_success(release):
    root, state, _ = release
    state.exit_code = 17
    with pytest.raises(RuntimeError, match="generator exited 17"):
        launch(root)
    assert state.loads == []
    assert status(root)["status"] == "FAILED"
    assert (root / "run/generate.log").is_file()


@pytest.mark.parametrize("error", [RuntimeError("monitor interrupted"), KeyboardInterrupt()])
@pytest.mark.parametrize("force_kill", [False, True])
def test_exception_cleans_up_only_owned_child(release, error, force_kill):
    root, state, _ = release
    state.wait_error, state.terminate_timeout = error, force_kill
    with pytest.raises(type(error)):
        launch(root)
    assert len(state.children) == 1
    child = state.children[0]
    assert child.terminated is True
    assert child.killed is force_kill
    assert child.poll() is not None
    assert status(root)["status"] == "FAILED"
    assert (root / "candidates/generations.jsonl").is_file()


@pytest.mark.parametrize("existing", ["run", "candidates"])
def test_existing_assets_never_overwritten(release, existing):
    root, state, _ = release
    path = root / existing
    path.mkdir()
    sentinel = path / "retained.txt"
    sentinel.write_text("previous research asset")
    with pytest.raises(FileExistsError):
        launch(root)
    assert sentinel.read_text() == "previous research asset"
    assert state.calls == []


@pytest.mark.parametrize("run_path,output_path", [
    ("run", "run"), ("run", "run/nested"), ("candidates/nested", "candidates"),
    ("../outside-run", "candidates"), ("run", "../outside-candidates"),
])
def test_unsafe_or_nested_paths_rejected_before_launch(release, run_path, output_path):
    root, state, _ = release
    with pytest.raises(ValueError):
        launcher.run(root / run_path, root / output_path, "NO-LAUNCH")
    assert state.calls == []


@pytest.mark.parametrize("attribute", ["is_available", "is_bf16_supported"])
def test_missing_cuda_or_bf16_fails_without_model_process(release, attribute):
    root, state, fake_torch = release
    setattr(fake_torch.cuda, attribute, lambda: False)
    with pytest.raises(RuntimeError, match="CUDA/BF16"):
        launch(root)
    assert state.calls == []
    assert status(root)["status"] == "FAILED"
