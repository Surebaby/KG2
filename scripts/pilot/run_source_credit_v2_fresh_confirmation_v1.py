#!/usr/bin/env python
"""Run the authorized fresh132 stages serially, releasing GPU memory between them.

This bounded runner never fits a gate, starts PPO/SFT or authorizes training.
Generation and scoring retain exact committed prefixes for explicit resumption.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def commands(protocol, out, resume=False):
    common = ["--protocol", str(protocol)]
    generation = [sys.executable, "-u", str(ROOT / "scripts/prepare/generate_source_credit_v2_fresh_confirmation_v1.py"),
                  *common, "--out", str(out / "generation")]
    scoring = [sys.executable, "-u", str(ROOT / "scripts/prepare/score_source_credit_v2_fresh_confirmation_v1.py"),
               *common, "--generation", str(out / "generation"), "--out", str(out / "scoring")]
    analysis = [sys.executable, "-u", str(ROOT / "scripts/pilot/analyze_source_credit_v2_fresh_confirmation_v1.py"),
                *common, "--scoring", str(out / "scoring"), "--out", str(out / "analysis")]
    if resume:
        if (out / "generation" / "started.json").exists():
            generation.append("--resume")
        if (out / "scoring" / "prepared.json").exists():
            scoring.append("--resume")
    return [("generation", generation), ("scoring", scoring), ("analysis", analysis)]


def run(protocol, out, resume=False):
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # The sibling lock does not create a new stage directory before its own
    # exclusive creation, and covers simultaneous explicit resume controllers.
    with (out.parent / f"{out.name}.controller.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous_term = signal.getsignal(signal.SIGTERM)

        def interrupted(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
        try:
            return _run(protocol, out, resume)
        finally:
            signal.signal(signal.SIGTERM, previous_term)


def _run(protocol, out, resume=False):
    protocol, out = Path(protocol).resolve(), Path(out).resolve()
    frozen = json.loads(protocol.read_text())
    if frozen.get("status") != "FROZEN" or frozen.get("schema_version") != "source-credit-v2-fresh-confirmation-protocol-v1":
        raise ValueError("bounded confirmation requires the frozen protocol")
    binding = frozen["code_bindings"]["scripts/pilot/run_source_credit_v2_fresh_confirmation_v1.py"]
    if sha(__file__) != binding["sha256"]:
        raise ValueError("runner differs from preregistered code")
    protocol_sha = sha(protocol)
    if not resume:
        out.mkdir(parents=True, exist_ok=False)
    elif not out.is_dir() or not (out / "started.json").is_file():
        raise ValueError("resume requires a started pipeline")
    identity = {"experiment_id": frozen["experiment_id"], "protocol_sha256": protocol_sha,
                "protocol": str(protocol), "optimizer_updates": 0}
    if resume:
        previous = json.loads((out / "started.json").read_text())
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError("pipeline resume identity mismatch")
        if (out / "analysis").exists():
            raise ValueError("analysis already attempted; inspect preserved result, never rerun automatically")
    else:
        with (out / "started.json").open("x") as handle:
            json.dump({**identity, "started_at_utc": now(), "pid": os.getpid()}, handle, indent=2)
    attempt = uuid.uuid4().hex

    def status(stage, state, **details):
        record = {**identity, "attempt": attempt, "stage": stage, "status": state,
                  "updated_at_utc": now(), "pid": os.getpid(), **details}
        with (out / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
        temporary = out / f"status.{attempt}.tmp"
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(out / "status.json")
        print(json.dumps(record, allow_nan=False), flush=True)

    env = {**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT / 'flashrag_src'}",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    for stage, command in commands(protocol, out, resume):
        if sha(protocol) != protocol_sha:
            status(stage, "FAILED_PROTOCOL_CHANGED")
            return 2
        log = out / f"{stage}.{attempt}.log"
        status(stage, "RUNNING", log=str(log), command=command)
        child = None
        try:
            with log.open("x") as handle:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                status(stage, "RUNNING", log=str(log), child_pid=child.pid)
                code = child.wait()
        except KeyboardInterrupt:
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGINT)
                child.wait()
            status(stage, "INTERRUPTED_PREFIX_RETAINED", log=str(log))
            return 130
        if code:
            status(stage, "FAILED_PREFIX_RETAINED", returncode=code, log=str(log))
            return code
        status(stage, "COMPLETE", log=str(log))
    status("pipeline", "COMPLETE_CONFIRMATION_REQUIRES_REVIEW", ppo_started=False,
           boundary="No automatic gate clearance, resampling or PPO launch")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(**vars(args)))


if __name__ == "__main__":
    main()
