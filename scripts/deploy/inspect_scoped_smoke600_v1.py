"""Read-only status summary for the fixed A-smoke600 run; never starts training."""
from pathlib import Path
from datetime import datetime, timezone
import json
import math
import subprocess


def inspect():
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/ppo_mixed4_answer_format_v2_a_smoke600_scoped_seed42_20260906_v1"
    supervision = root / "outputs/launches/ppo_a_smoke600_scoped_20260906_v1_supervision"
    def read(path):
        return json.loads(path.read_text()) if path.is_file() else None
    rows = []
    history = out / "history.jsonl"
    if history.is_file():
        for line in history.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break  # A trailing in-progress append is retried on the next inspection.
    result = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "supervisor": read(supervision / "status.json"),
              "manifest_status": (read(out / "manifest.json") or {}).get("status"),
              "step": rows[-1]["step"] if rows else 0, "target": 600,
              "batches": len(rows), "tensorboard": read(out / "tensorboard_run.json"),
              "checkpoints": sorted(p.name for p in out.iterdir() if p.is_dir() and (p.name.startswith("step_") or p.name.startswith("aborted_step_"))) if out.is_dir() else [],
              "final_exists": (out / "final/adapter_model.safetensors").is_file()}
    keys = ("mean_reward", "ppo_mean_kl", "valid_rate", "length_capped_frac", "loss_total", "loss_policy", "loss_value", "mixed_outcome_em_mean", "mixed_outcome_f1_mean", "source_gate_alpha_effective_mean", "source_gate_text_component_mean", "source_gate_graph_component_mean", "sft_replay_items_seen")
    if rows:
        result["last"] = {k: rows[-1].get(k) for k in keys}
        window = rows[-15:]
        result["rolling15"] = {k: sum(float(x[k]) for x in window) / len(window) for k in ("valid_rate", "length_capped_frac", "ppo_mean_kl")}
        result["cumulative"] = {"valid": sum(x["n_valid"] for x in rows),
                                "text_steps": sum(x["mixed_text_step_count"] for x in rows),
                                "nonzero_alpha_records": sum(r["alpha_effective"] > 0 for x in rows for r in x["source_gate_records"]),
                                "replay_items": sum(x["sft_replay_items"] for x in rows)}
        result["nonfinite_batches"] = [x["step"] for x in rows if any(isinstance(x.get(k), (float, int)) and not math.isfinite(x[k]) for k in keys)]
    result["history_persisted_step"] = result["step"]
    if result["tensorboard"]:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(result["tensorboard"]["log_dir"], size_guidance={"scalars": 0})
        events.Reload()
        names = events.Tags().get("scalars", [])
        selected = ("progress/rollout_trajectories", "progress/ppo_batches", "ppo/loss/total", "ppo/loss/policy", "ppo/loss/value", "objective/kl", "custom/valid_rate", "custom/length_capped_frac", "gate/all/eligible_valid_count", "gate/eligible_valid/alpha_effective_mean", "custom/sft_replay_actual_ratio", "custom/sft_replay_loss")
        result["live_scalars"] = {name: {"step": events.Scalars(name)[-1].step, "value": events.Scalars(name)[-1].value} for name in selected if name in names and events.Scalars(name)}
        result["live_scalar_tags"] = len(names)
        result["live_histogram_tags"] = len(events.Tags().get("histograms", []))
        progress = result["live_scalars"].get("progress/rollout_trajectories")
        if progress:
            result["step"] = max(result["step"], int(progress["value"]))
        result["live_rolling15"] = {name: sum(x.value for x in events.Scalars(name)[-15:]) / min(15, len(events.Scalars(name))) for name in ("custom/valid_rate", "custom/length_capped_frac", "objective/kl") if name in names and events.Scalars(name)}
        result["live_nonfinite_tags"] = [name for name in names if any(not math.isfinite(x.value) for x in events.Scalars(name))]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    result["gpu_csv"] = gpu.stdout.strip()
    return result


if __name__ == "__main__":
    print(json.dumps(inspect(), ensure_ascii=False))
