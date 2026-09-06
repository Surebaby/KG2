"""TensorBoard run isolation and runtime telemetry; never changes optimization."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import time
from uuid import uuid4

from kgproweight.training.ppo_tensorboard import HISTOGRAM_EVERY_BATCHES, HISTOGRAM_INITIAL_BATCHES


def create_ppo_writer(output_dir, experiment_id, *, environ=None):
    from torch.utils.tensorboard import SummaryWriter

    env = os.environ if environ is None else environ
    output_dir = Path(output_dir)
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    if env.get("KGPW_TB_DIR"):
        log_dir = Path(env["KGPW_TB_DIR"])
        if log_dir.exists() and any(log_dir.glob("events.out.tfevents.*")):
            raise FileExistsError("KGPW_TB_DIR already contains events; choose a fresh run directory")
    elif env.get("KGPW_TB_ROOT"):
        if Path(experiment_id).name != experiment_id or experiment_id in {"", ".", ".."}:
            raise ValueError("TensorBoard experiment_id must be a single path component")
        log_dir = Path(env["KGPW_TB_ROOT"]) / experiment_id / session
    else:
        log_dir = output_dir / "tensorboard" / session
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {"experiment_id": experiment_id, "session": session,
              "log_dir": str(log_dir.resolve()), "step_unit": "completed_rollout_trajectories",
              "histogram_initial_ppo_batches": HISTOGRAM_INITIAL_BATCHES,
              "histogram_every_ppo_batches": HISTOGRAM_EVERY_BATCHES, "flush_every_ppo_batch": True}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "tensorboard_run.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    return SummaryWriter(log_dir=str(log_dir), flush_secs=10, max_queue=10), record


def log_runtime(writer, *, step, update_index, batch_started, response_lengths, trainer):
    import torch

    seconds = max(time.perf_counter() - batch_started, 1e-9)
    values = {"progress/rollout_trajectories": step, "progress/ppo_batches": update_index,
              "runtime/batch_seconds": seconds,
              "runtime/rollout_trajectories_per_second": len(response_lengths) / seconds,
              "runtime/response_tokens_per_second": sum(response_lengths) / seconds,
              "runtime/host_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    optimizer = getattr(trainer, "optimizer", None)
    for index, group in enumerate(getattr(optimizer, "param_groups", [])):
        values[f"optimizer/learning_rate_group_{index}"] = float(group["lr"])
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            for name, reader in (("allocated_gib", torch.cuda.memory_allocated),
                                 ("reserved_gib", torch.cuda.memory_reserved),
                                 ("peak_allocated_gib", torch.cuda.max_memory_allocated)):
                values[f"system/gpu_{device}/{name}"] = reader(device) / 1024**3
    for tag, value in values.items():
        writer.add_scalar(tag, value, step)
