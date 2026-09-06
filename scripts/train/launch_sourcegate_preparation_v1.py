"""Run the authorized GPU prerequisites for full PPO-A, with persistent logs.

Calibration success is not process-utility clearance. This runner explicitly
stops at that scientific boundary; it never silently launches outcome-only PPO.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RELEASE = "outputs/audits/source_gated_mixed4_emf1_v1_release_tensorboard_v1/manifest.json"
BANK = "outputs/audits/source_quality_candidate_bank_v1_inputs_seed42_tensorboard_v1"
GENERATED = "outputs/audits/source_quality_candidate_bank_v1_generated_seed42"
SCORED = "outputs/audits/source_quality_candidate_bank_v1_scored_seed42"
CALIBRATION = "outputs/calibration/source_quality_gate_v1_seed42"


def commands():
    prefix = [sys.executable, "-u", "-m"]
    module = "scripts.prepare.source_quality_candidate_bank_v1"
    return [
        ("generate", prefix + [module, "generate", "--bank-dir", BANK, "--output-dir", GENERATED,
                              "--experiment-id", "SOURCE-QUALITY-CANDIDATE-BANK-V1-GENERATE-SEED42"],
         GENERATED + "/generations.jsonl", 1660),
        ("score", prefix + [module, "score", "--bank-dir", BANK, "--generation-dir", GENERATED,
                           "--output-dir", SCORED,
                           "--experiment-id", "SOURCE-QUALITY-CANDIDATE-BANK-V1-SCORE-SEED42"],
         SCORED + "/candidates.scored.jsonl", 1660),
        ("calibrate", prefix + ["scripts.train.calibrate_source_quality_gate_v1",
                               "--bank-manifest", SCORED + "/manifest.json",
                               "--isolation-proof", SCORED + "/isolation_proof.json",
                               "--output-dir", CALIBRATION,
                               "--experiment-id", "SOURCE-QUALITY-GATE-V1-CALIBRATION-SEED42"], None, None),
    ]


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def line_count(path):
    if not path or not Path(path).is_file():
        return 0
    with Path(path).open("rb") as handle:
        return sum(block.count(b"\n") for block in iter(lambda: handle.read(1024 * 1024), b""))


def event(run_dir, status, **details):
    record = {"time_utc": datetime.now(timezone.utc).isoformat(), "status": status,
              "research_policy_updates": 0, **details}
    text = json.dumps(record, ensure_ascii=False)
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")
    # This is a live status pointer; the append-only events retain every state.
    temporary = run_dir / "status.json.tmp"
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(run_dir / "status.json")
    print(text, flush=True)


def run(run_dir):
    os.chdir(ROOT)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    current_stage = "preflight"
    try:
        for relative in (GENERATED, SCORED, CALIBRATION):
            if (ROOT / relative).exists():
                raise FileExistsError(f"refusing to overwrite previous stage: {relative}")
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA and BF16 are required")
        memory = torch.cuda.get_device_properties(0).total_memory
        if memory < 80 * 1024**3:
            raise RuntimeError("this full-method launch was prepared for the 96 GiB GPU")
        # Real CUDA kernel check, no research model or optimizer involved.
        x = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
        assert float((x @ x)[0, 0]) == 32.0
        del x
        torch.cuda.empty_cache()
        manifest = json.loads((ROOT / RELEASE).read_text())
        event(run_dir, "VERIFYING_FROZEN_RELEASE", stage=current_stage)
        from scripts.prepare.verify_sourcegate_deployment_v1 import verify_files
        verified = verify_files(ROOT, manifest["files"])
        from scripts.prepare.source_quality_candidate_bank_v1 import (
            load_release, PREPARE_VERSION, validate_code, validate_inputs,
        )
        bank = load_release(ROOT / BANK, PREPARE_VERSION)
        validate_code(bank, ROOT)
        if len(validate_inputs(ROOT / BANK, bank)) != 830:
            raise RuntimeError("candidate bank size differs")
        launch = {"experiment_id": run_dir.name, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "pid": os.getpid(), "gpu": torch.cuda.get_device_name(0), "gpu_memory_bytes": memory,
                  "torch": torch.__version__, "release_manifest": RELEASE,
                  "release_sha256": file_sha(ROOT / RELEASE), "file_verification": verified,
                  "runner_sha256": file_sha(__file__), "candidate_bank": BANK,
                  "candidate_bank_sha256": file_sha(ROOT / BANK / "manifest.json"),
                  "stages": [{"name": name, "command": cmd} for name, cmd, _, _ in commands()],
                  "environment": {key: os.environ.get(key) for key in
                      ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TOKENIZERS_PARALLELISM",
                       "KGPW_TB_ROOT", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")},
                  "stop_boundary": "real process utility must be checked before PPO-A probe",
                  "research_policy_updates": 0}
        (run_dir / "manifest.json").write_text(json.dumps(launch, indent=2) + "\n")
        for name, cmd, progress_file, expected in commands():
            current_stage = name
            log = run_dir / (name + ".log")
            start = time.monotonic()
            with log.open("xb") as handle:
                process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                                           stdout=handle, stderr=subprocess.STDOUT)
                event(run_dir, "RUNNING", stage=name, pid=process.pid, log=str(log))
                while process.poll() is None:
                    completed = line_count(progress_file)
                    elapsed = time.monotonic() - start
                    event(run_dir, "RUNNING", stage=name, pid=process.pid,
                          completed=completed, expected=expected, elapsed_seconds=round(elapsed, 1),
                          eta_seconds=round(elapsed * (expected - completed) / completed) if expected and completed else None)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
            if process.returncode:
                raise RuntimeError(f"{name} exited {process.returncode}; see {log}")
            if expected and line_count(progress_file) != expected:
                raise RuntimeError(f"{name} returned an incomplete candidate population")
            event(run_dir, "STAGE_COMPLETE", stage=name, completed=line_count(progress_file))
        gate = json.loads((ROOT / CALIBRATION / "gate.json").read_text())
        report = json.loads((ROOT / CALIBRATION / "report.json").read_text())
        if not gate.get("training_clearance") or not report.get("training_clearance"):
            event(run_dir, "STOPPED_CALIBRATION_FAILED", stage="calibrate", report=CALIBRATION + "/report.json")
            return 2
        event(run_dir, "AWAITING_PROCESS_UTILITY_CHECK", stage="pre_ppo_review",
              report=CALIBRATION + "/report.json", ppo_started=False,
              next_config="configs/training/phase3_ppo_mixed4_sourcegate_v1_a_probe_seed42.yaml")
        return 0
    except BaseException as exc:
        event(run_dir, "FAILED", stage=current_stage, error_type=type(exc).__name__, error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    return run(args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
