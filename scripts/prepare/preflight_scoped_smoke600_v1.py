"""CPU verification of the deployed, frozen A-smoke600; never loads model tensors."""
from pathlib import Path
from dataclasses import replace
from datetime import datetime, timezone
import json
import hashlib
import subprocess
import urllib.request


def main():
    root = Path(__file__).resolve().parents[2]
    audit = root / "outputs/audits/ppo_smoke600_remote_prelaunch_20260906_v1"
    release = json.loads((audit / "manifest.json").read_text())
    for name, frozen in release["files"].items():
        path = root / name
        assert path.stat().st_size == frozen["bytes"], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == frozen["sha256"], name
    from scripts.prepare.resolve_phase3_ppo_runtime_config import resolve_phase3_ppo_runtime_config
    from kgproweight.training.phase3_ppo import Phase3PPOConfig
    from kgproweight.reward.source_credit_gate_v2 import SourceCreditGateV2
    from kgproweight.reward.source_gate_bounded_dispatch_v1 import load_referenced_bounded_before_dispatch
    cfg = Phase3PPOConfig(**resolve_phase3_ppo_runtime_config(root / "configs/training/phase3_ppo_mixed4_answer_format_v2_a_smoke600_scoped_seed42.yaml"))
    gate = load_referenced_bounded_before_dispatch(cfg)
    checks = []
    for name, changes in [("limit12", {"total_steps":12}), ("limit601", {"total_steps":601}), ("limit12000", {"total_steps":12000}), ("mini2", {"mini_batch_size":2}), ("lr_override", {"learning_rate":2e-6}), ("disable_text", {"mixed_text_reward":False}), ("fixed_alpha", {"source_gate_mode":"fixed"}), ("disable_cost_guard", {"health_guard_after_steps":0}), ("relax_valid_threshold", {"health_guard_min_valid_rate":.5})]:
        try:
            load_referenced_bounded_before_dispatch(replace(cfg, **changes))
        except ValueError:
            checks.append({"case":name,"rejected":True})
        else:
            raise AssertionError(name)
    try:
        SourceCreditGateV2.load(cfg.source_gate_calibration_path)
    except ValueError:
        checks.append({"case":"default_without_runtime_config","rejected":True})
    else:
        raise AssertionError("default scope bypass")
    assert not (root / cfg.output_dir).exists(), "existing run must be inspected, not restarted"
    with urllib.request.urlopen("http://127.0.0.1:6007/data/plugin/scalars/tags", timeout=10) as response:
        tags = json.load(response)
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    result = {"status":"REMOTE_CPU_PREFLIGHT_PASS_MANUAL_A_SMOKE600_NOT_STARTED", "created_at_utc":datetime.now(timezone.utc).isoformat(), "release_file_count":len(release["files"]), "all_release_sha256_match":True, "exact_runtime_and_actual_base_rearag_paths_pass":True, "scope":gate.execution_scope_validation, "negative_cases":checks, "tensorboard_http_accessible":True, "tensorboard_run_count_before_start":len(tags), "gpu_csv":gpu, "training_started":False, "model_tensors_loaded":False, "gold_values_opened":False}
    with (audit / "remote_preflight.json").open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
