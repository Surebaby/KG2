"""Run the frozen SFT/200/300 development comparison; retain every failed job."""
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


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/evals/ppo_a_stopped300_development150_20260906_v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_code(protocol):
    for name, frozen in protocol["code_bindings"].items():
        if sha(ROOT / name) != frozen["sha256"]:
            raise ValueError(f"frozen evaluation source changed: {name}")


def save_status(state):
    path = RUN / "status.tmp.json"
    path.write_text(json.dumps({**state, "observed_at_utc": now()}, ensure_ascii=False, indent=2) + "\n")
    path.replace(RUN / "status.json")


def verify_complete_generation(directory, bank_sha):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, frozen in manifest["outputs"].items():
        if Path(name).name != name or sha(directory / name) != frozen["sha256"]:
            raise ValueError("generation artifact seal differs")
    report = json.loads((directory / "report.json").read_text())
    if (manifest["status"] != "COMPLETE_DEVELOPMENT_GENERATION_NOT_MAIN_TABLE"
            or report["status"] != manifest["status"] or report["n"] != 150
            or report["bank_manifest_sha256"] != bank_sha):
        raise ValueError("only sealed complete150 generation can enter scoring")


def main():
    protocol = json.loads((RUN / "execution.json").read_text())
    check_code(protocol)
    for name, frozen in protocol["bank_files"].items():
        if sha(RUN / "bank" / name) != frozen["sha256"]:
            raise ValueError(f"frozen evaluation bank changed: {name}")
    lock = (RUN / "execution.lock").open("x")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if any((RUN / name).exists() for name in ("generations", "scores", "selection", "status.json")):
        raise FileExistsError("existing evaluation retained; automatic restart is forbidden")
    for name in ("generations", "scores", "logs"):
        (RUN / name).mkdir(exist_ok=False)
    env = {**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT / 'flashrag_src'}",
           "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "KGPW_PROJECT_ROOT": str(ROOT), "KGPW_KG_OFFLINE": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    state = {"experiment_id": protocol["experiment_id"], "status": "RUNNING",
             "supervisor_pid": os.getpid(), "started_at_utc": now(),
             "execution_sha256": sha(RUN / "execution.json"), "completed_jobs": [],
             "automatic_restart": False, "automatic_canonical_evaluation": False}
    save_status(state)
    child = None
    def forward(signum, _frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum)
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    base = [sys.executable, "-m", "scripts.eval.ppo_emf1_development_v1"]
    jobs = []
    for candidate in protocol["candidates"]:
        for view in protocol["views"]:
            key = f"{candidate['model_id']}__{view}"
            common = ["--bank-dir", str(RUN / "bank"), "--model-id", candidate["model_id"],
                      "--checkpoint", str(ROOT / candidate["checkpoint_path"]), "--view", view,
                      "--project-root", str(ROOT), "--experiment-id", f"{protocol['experiment_id']}__{key}"]
            jobs.append((f"generate__{key}", base + ["generate", *common, "--output-dir", str(RUN / "generations" / key),
                         "--base-model", str(ROOT / protocol["base_model"]), "--tokenizer-path", str(ROOT / protocol["tokenizer"]), "--device", "cuda:0"]))
    # All candidates/views finish generation before scorer-only labels are read.
    score_dirs = []
    for candidate in protocol["candidates"]:
        for view in protocol["views"]:
            key = f"{candidate['model_id']}__{view}"
            directory = RUN / "scores" / key
            score_dirs.append(directory)
            jobs.append((f"score__{key}", base + ["score", "--bank-dir", str(RUN / "bank"),
                         "--model-id", candidate["model_id"], "--checkpoint", str(ROOT / candidate["checkpoint_path"]),
                         "--view", view, "--project-root", str(ROOT), "--predictions", str(RUN / "generations" / key / "predictions.jsonl"),
                         "--output-dir", str(directory), "--experiment-id", f"{protocol['experiment_id']}__score__{key}"]))
    select = base + ["select", "--bank-dir", str(RUN / "bank"), "--output-dir", str(RUN / "selection"),
                     "--experiment-id", f"{protocol['experiment_id']}__selection"]
    for directory in score_dirs:
        select += ["--score-dir", str(directory)]
    jobs.append(("select", select))
    try:
        for job, command in jobs:
            check_code(protocol)
            if job.startswith("score__"):
                verify_complete_generation(RUN / "generations" / job.removeprefix("score__"),
                                           sha(RUN / "bank/manifest.json"))
            print(f"{now()} START {job}", flush=True)
            with (RUN / "logs" / f"{job}.log").open("xb") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(current_job=job, child_pid=child.pid, current_job_started_at_utc=now())
            while True:
                code = child.poll()
                state["generation_rows"] = {
                    p.parent.name: sum(1 for line in p.open() if line.strip())
                    for p in (RUN / "generations").glob("*/predictions.jsonl")}
                state["generated_predictions"] = sum(state["generation_rows"].values())
                save_status(state)
                if code is not None:
                    break
                time.sleep(10)
            if code:
                state.update(status="FAILED_RETAINED", failed_job=job, child_exit_code=code, completed_at_utc=now())
                save_status(state)
                print(f"{now()} FAILED {job} exit={code}", flush=True)
                return code
            if job.startswith("generate__"):
                verify_complete_generation(RUN / "generations" / job.removeprefix("generate__"),
                                           sha(RUN / "bank/manifest.json"))
            state["completed_jobs"].append(job)
            print(f"{now()} COMPLETE {job}", flush=True)
        state.update(status="COMPLETE_DEVELOPMENT_COMPARISON_NOT_MAIN_TABLE", completed_at_utc=now(), child_pid=None)
        save_status(state)
        return 0
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        state.update(status="SUPERVISOR_FAILED_RETAINED", error_type=type(exc).__name__, error=str(exc), completed_at_utc=now())
        save_status(state)
        raise


if __name__ == "__main__":
    sys.exit(main())
