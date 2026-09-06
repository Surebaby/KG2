"""Use AutoDL's /root/tf-logs and port 6007 without deleting logs or killing services."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


def server_logdir(port):
    with urlopen(f"http://127.0.0.1:{port}/data/logdir", timeout=3) as response:
        result = json.load(response)
    return result.get("logdir") if isinstance(result, dict) else result


def setup(logdir: Path, port: int):
    logdir = logdir.resolve()
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "kgpaper").mkdir(exist_ok=True)
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            occupied = True
    except OSError:
        occupied = False
    if occupied:
        try:
            current = server_logdir(port)
        except Exception as exc:
            raise RuntimeError(f"Port {port} is occupied by an unverified service; left untouched") from exc
        if not current or Path(current).resolve() != logdir:
            raise RuntimeError(f"Existing TensorBoard uses {current!r}; expected {str(logdir)!r}; left untouched")
        return {"status": "REUSED", "port": port, "logdir": str(logdir)}
    service_dir = logdir / ".kgpaper-service"
    service_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    service_log = service_dir / f"tensorboard_{stamp}.log"
    command = [sys.executable, "-m", "tensorboard.main", "--port", str(port),
               "--logdir", str(logdir), "--host", "0.0.0.0", "--reload_interval", "10"]
    with service_log.open("xb") as handle:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    record = {"status": "STARTING", "pid": process.pid, "command": command,
              "port": port, "logdir": str(logdir), "service_log": str(service_log)}
    record_path = service_dir / f"service_{stamp}.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"TensorBoard exited with {process.returncode}; see {service_log}")
        try:
            if Path(server_logdir(port)).resolve() == logdir:
                record["status"] = "RUNNING"
                record_path.write_text(json.dumps(record, indent=2) + "\n")
                return record
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"TensorBoard startup unconfirmed; retained PID {process.pid} and log {service_log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, default=Path("/root/tf-logs"))
    parser.add_argument("--port", type=int, default=6007)
    args = parser.parse_args()
    print(json.dumps(setup(args.logdir, args.port), ensure_ascii=False))


if __name__ == "__main__":
    main()
