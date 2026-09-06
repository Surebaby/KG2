"""Run only the frozen A-smoke600 and retain its exit status and GPU observations."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    project = Path(__file__).resolve().parents[2]
    run = project / "outputs/launches/ppo_a_smoke600_scoped_20260906_v1_supervision"
    output = project / "outputs/ppo_mixed4_answer_format_v2_a_smoke600_scoped_seed42_20260906_v1"
    log_path = project / "outputs/launches/ppo_a_smoke600_scoped_20260906_v1.log"
    config = "configs/training/phase3_ppo_mixed4_answer_format_v2_a_smoke600_scoped_seed42.yaml"
    if output.exists() or log_path.exists():
        raise FileExistsError("Existing training output/log retained; inspect it instead of restarting")
    run.mkdir(parents=True, exist_ok=False)
    lock = (run / "supervisor.lock").open("x")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    command = ["bash", "scripts/sourcegate_python.sh", "scripts/train/phase3_ppo.py", "--config", config]
    env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
    launch = {"experiment_id": "SOURCE-CREDIT-V2-A-SMOKE600-GPU-SUPERVISED-20260906-V1",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "supervisor_pid": os.getpid(), "command": command,
              "config_sha256": hashlib.sha256((project / config).read_bytes()).hexdigest(),
              "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "release_manifest_sha256": hashlib.sha256((project / "outputs/audits/ppo_smoke600_remote_prelaunch_20260906_v1/manifest.json").read_bytes()).hexdigest(),
              "scope_sha256": hashlib.sha256((project / "outputs/calibration/source_credit_gate_v2_smoke600_scoped_20260906_v1/scope.json").read_bytes()).hexdigest(),
              "trajectory_limit": 600, "automatic_restart": False, "automatic_smoke_or_full": False}
    with log_path.open("xb") as log:
        child = subprocess.Popen(command, cwd=project, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    def forward(signum, _frame):
        if child.poll() is None:
            os.killpg(child.pid, signum)
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    launch["training_pid"] = child.pid
    try:
        (run / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise
    while True:
        code = child.poll()
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10)
        except (subprocess.TimeoutExpired, OSError) as exc:
            gpu = subprocess.CompletedProcess(args=["nvidia-smi"], returncode=-1, stdout="", stderr=type(exc).__name__)
        record = {"observed_at_utc": datetime.now(timezone.utc).isoformat(), "training_pid": child.pid,
                  "status": "RUNNING" if code is None else ("PROCESS_EXITED_ZERO_AWAITING_AUDIT" if code == 0 else "PROCESS_FAILED"),
                  "exit_code": code, "gpu_csv": gpu.stdout.strip(), "gpu_query_returncode": gpu.returncode, "gpu_query_error": gpu.stderr.strip(),
                  "log_bytes": log_path.stat().st_size, "final_checkpoint_exists": (output / "final").is_dir()}
        with (run / "observations.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        temp = run / "status.tmp.json"
        temp.write_text(json.dumps(record, indent=2) + "\n")
        temp.replace(run / "status.json")
        if code is not None:
            return code
        time.sleep(15)


if __name__ == "__main__":
    sys.exit(main())
