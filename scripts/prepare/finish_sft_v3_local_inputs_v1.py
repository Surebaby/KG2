#!/usr/bin/env python3
"""Wait for the two authorized local retrieval jobs and assemble all inputs.

This process has no teacher API code, credentials or model-training path.
Existing failed/complete assets are never overwritten. It stops after 24h or
on source/code failure; otherwise the exact all-input snapshot is materialized.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def binding(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined_manifest", type=Path, required=True)
    parser.add_argument("--retrieval_run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--watch_dir", type=Path, required=True)
    args = parser.parse_args()
    args.watch_dir.mkdir(parents=True, exist_ok=False)
    code = ROOT / "scripts/prepare/assemble_sft_v3_inputs_v1.py"
    frozen = {"schema_version": "sft-v3-local-input-finisher-v1",
              "combined_manifest": binding(args.combined_manifest), "assembler": binding(code),
              "runner": binding(Path(__file__)),
              "retrieval_runs": [str(p.resolve()) for p in args.retrieval_run],
              "out": str(args.out.resolve()), "teacher_calls": 0, "model_updates": 0}
    (args.watch_dir / "protocol.json").write_text(json.dumps(frozen, indent=2) + "\n")
    start = time.monotonic()
    try:
        while True:
            if binding(code) != frozen["assembler"] or binding(args.combined_manifest) != frozen["combined_manifest"]:
                raise RuntimeError("frozen local preparation code/input changed")
            ready = [bool((p / "manifest.json").is_file()) for p in args.retrieval_run]
            if any(list(p.glob("failure_*.json")) for p in args.retrieval_run):
                raise RuntimeError("retrieval failure retained; local finisher stopped for diagnosis")
            if all(ready): break
            if time.monotonic() - start > 24 * 3600:
                raise TimeoutError("24h local preparation deadline reached; upstream jobs untouched")
            print(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "status": "WAITING_FOR_LOCAL_RETRIEVAL_ONLY", "complete": ready, "teacher_calls": 0}), flush=True)
            time.sleep(30)
        command = [sys.executable, str(code), "materialize", "--combined_manifest", str(args.combined_manifest), "--out", str(args.out)]
        for directory in args.retrieval_run: command.extend(["--retrieval_run", str(directory)])
        subprocess.run(command, cwd=ROOT, check=True)
        result = json.loads((args.out / "manifest.json").read_text())
        if result.get("all_combined_inputs_prepared") is not True or result.get("ready_inputs") != 16560:
            raise RuntimeError("local final snapshot does not cover all frozen 16560 inputs")
        report = {"status": "ALL_LOCAL_INPUTS_PREPARED_TEACHER_BUDGET_PENDING", "inputs_manifest": binding(args.out / "manifest.json"), "teacher_calls": 0, "qualified_sft_examples": 0, "model_updates": 0}
        (args.watch_dir / "completed.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    except BaseException as exc:
        (args.watch_dir / "failure.json").write_text(json.dumps({"status": "LOCAL_FINISHER_FAILED_ASSETS_RETAINED", "type": type(exc).__name__, "message": str(exc), "teacher_calls": 0}) + "\n")
        raise


if __name__ == "__main__": main()
