"""Local, from-scratch generation of the frozen SourceGate K2 candidate bank.

Calls the unchanged production generator. Remote partial candidates are never
loaded, and this launcher does not start ReaRAG, calibration, or PPO implicitly.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.train.launch_sourcegate_preparation_v1 import event, file_sha, line_count

ROOT = Path(__file__).resolve().parents[2]
BANK = "outputs/audits/source_quality_candidate_bank_v1_inputs_seed42_tensorboard_v1"


def run(run_dir: Path, output_dir: Path, experiment_id: str):
    os.chdir(ROOT)
    run_dir, output_dir = run_dir.resolve(), output_dir.resolve()
    if not run_dir.is_relative_to(ROOT) or not output_dir.is_relative_to(ROOT):
        raise ValueError("local launch paths must stay inside the project")
    if run_dir == output_dir or run_dir in output_dir.parents or output_dir in run_dir.parents:
        raise ValueError("run and candidate output directories must be separate")
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite existing candidates")
    run_dir.mkdir(parents=True, exist_ok=False)
    child = None
    try:
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("local generation requires working CUDA/BF16")
        bank = ROOT / BANK
        command = [sys.executable, "-u", "-m", "scripts.prepare.source_quality_candidate_bank_v1",
                   "generate", "--bank-dir", str(bank), "--output-dir", str(output_dir),
                   "--experiment-id", experiment_id, "--project-root", str(ROOT), "--device", "cuda:0"]
        manifest = {"experiment_id": experiment_id, "run_id": run_dir.name,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                    "command": command, "project_root": str(ROOT), "candidate_output": str(output_dir),
                    "candidate_bank": BANK, "candidate_bank_sha256": file_sha(bank / "manifest.json"),
                    "versions": {name: importlib.metadata.version(name) for name in
                                 ("torch", "transformers", "peft", "trl", "accelerate")},
                    "gpu": torch.cuda.get_device_name(0), "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                    "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                             capture_output=True, text=True, check=True).stdout.strip(),
                    "cuda_runtime": torch.version.cuda,
                    "source_bindings": {str(path.relative_to(ROOT)): file_sha(path) for path in
                                        (Path(__file__), ROOT / "scripts/train/launch_sourcegate_preparation_v1.py",
                                         ROOT / "scripts/prepare/source_quality_candidate_bank_v1.py")},
                    "environment": {name: os.environ.get(name) for name in
                                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                     "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM")},
                    "remote_candidates_reused": 0, "research_policy_updates": 0,
                    "boundary": "full regeneration under frozen inputs/seed/sampling; cross-GPU bitwise identity is not claimed"}
        (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        started = time.monotonic()
        with (run_dir / "generate.log").open("xb") as handle:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=handle, stderr=subprocess.STDOUT)
            while child.poll() is None:
                completed = line_count(output_dir / "generations.jsonl")
                elapsed = time.monotonic() - started
                event(run_dir, "RUNNING", stage="local_generate", pid=child.pid,
                      completed=completed, expected=1660, remote_candidates_reused=0,
                      elapsed_seconds=round(elapsed, 1),
                      eta_seconds=round(elapsed * (1660-completed) / completed) if completed else None)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        if child.returncode:
            raise RuntimeError(f"generator exited {child.returncode}; see {run_dir / 'generate.log'}")
        from scripts.prepare.source_quality_candidate_bank_v1 import load_release, GENERATION_VERSION
        generated = load_release(output_dir, GENERATION_VERSION)
        if generated.get("n_candidates") != 1660 or line_count(output_dir / "generations.jsonl") != 1660:
            raise RuntimeError("local generation did not produce the complete frozen population")
        event(run_dir, "LOCAL_GENERATION_COMPLETE_SCORING_PENDING", stage="local_generate",
              completed=1660, expected=1660, output_dir=str(output_dir), remote_candidates_reused=0)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        event(run_dir, "FAILED", stage="local_generate", error_type=type(exc).__name__, error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    run(args.run_dir, args.output_dir, args.experiment_id)


if __name__ == "__main__":
    main()
