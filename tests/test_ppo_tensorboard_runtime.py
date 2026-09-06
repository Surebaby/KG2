import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from kgproweight.training.ppo_tensorboard_runtime import create_ppo_writer, log_runtime


def test_autodl_root_isolates_runs_and_records_pointer(tmp_path, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    root = tmp_path / "tf-logs" / "kgpaper"
    writer, record = create_ppo_writer(tmp_path / "outputs" / "a", "ppo_a_probe", environ={"KGPW_TB_ROOT": str(root)})
    assert Path(record["log_dir"]).parent == root / "ppo_a_probe"
    trainer = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"lr": 1e-6}]))
    log_runtime(writer, step=4, update_index=1, batch_started=0,
                response_lengths=[20, 30, 40, 50], trainer=trainer)
    writer.close()
    events = EventAccumulator(record["log_dir"]).Reload()
    assert events.Scalars("progress/rollout_trajectories")[0].value == 4
    assert events.Scalars("progress/ppo_batches")[0].step == 4
    assert events.Scalars("optimizer/learning_rate_group_0")[0].value == pytest.approx(1e-6)
    assert json.loads((tmp_path / "outputs/a/tensorboard_run.json").read_text()) == record
    assert record["histogram_initial_ppo_batches"] == 3
    assert record["histogram_every_ppo_batches"] == 10
    other, other_record = create_ppo_writer(tmp_path / "outputs" / "b", "ppo_a_probe", environ={"KGPW_TB_ROOT": str(root)})
    other.close()
    assert other_record["log_dir"] != record["log_dir"]


def test_explicit_logdir_never_merges_existing_events(tmp_path):
    env = {"KGPW_TB_DIR": str(tmp_path / "explicit")}
    writer, record = create_ppo_writer(tmp_path / "a", "a", environ=env)
    writer.close()
    with pytest.raises(FileExistsError):
        create_ppo_writer(tmp_path / "b", "b", environ=env)
    assert Path(record["log_dir"]) == tmp_path / "explicit"


def test_local_default_stays_in_output_directory(tmp_path):
    writer, record = create_ppo_writer(tmp_path / "local", "a", environ={})
    writer.close()
    assert Path(record["log_dir"]).parent == tmp_path / "local" / "tensorboard"


def test_gpu_memory_tags_are_written_for_each_visible_device_without_cuda_execution(tmp_path, monkeypatch):
    """Exercise the GPU telemetry branch using readers, never model/CUDA work."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    monkeypatch.setattr("torch.cuda.memory_allocated", lambda device: (device + 1) * 1024**3)
    monkeypatch.setattr("torch.cuda.memory_reserved", lambda device: (device + 2) * 1024**3)
    monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda device: (device + 3) * 1024**3)
    writer, record = create_ppo_writer(tmp_path / "output", "CPU_GPU_READER_DOUBLE", environ={})
    trainer = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"lr": 1e-6}, {"lr": 2e-6}]))
    log_runtime(writer, step=4, update_index=1, batch_started=0,
                response_lengths=[4, 4, 4, 4], trainer=trainer)
    writer.close()
    events = EventAccumulator(record["log_dir"]).Reload()
    for device in range(2):
        for name, expected in (("allocated_gib", device + 1), ("reserved_gib", device + 2),
                               ("peak_allocated_gib", device + 3)):
            point = events.Scalars(f"system/gpu_{device}/{name}")[0]
            assert point.step == 4 and point.value == expected
    assert events.Scalars("optimizer/learning_rate_group_1")[0].value == pytest.approx(2e-6)
