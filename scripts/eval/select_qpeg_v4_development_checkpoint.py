#!/usr/bin/env python
"""Select the earliest QPEG-v4 checkpoint passing every frozen dev gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select(reports: list[dict]) -> int | None:
    for row in sorted(reports, key=lambda item: int(item["checkpoint_step"])):
        if bool(row.get("all_development_gates_pass")):
            return int(row["checkpoint_step"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite checkpoint selection: {output}")
    candidates = []
    seen_steps = set()
    for raw_path in args.candidate:
        path = Path(raw_path).resolve()
        row = json.loads(path.read_text(encoding="utf-8"))
        step = int(row["checkpoint_step"])
        if step in seen_steps or step not in {25, 50, 75}:
            raise SystemExit(f"duplicate/invalid checkpoint step: {step}")
        if row.get("confirmation_opened") is not False:
            raise SystemExit("candidate does not prove confirmation remained unopened")
        seen_steps.add(step)
        candidates.append({
            "checkpoint_step": step,
            "path": str(path),
            "sha256": _sha256(path),
            "status": row["status"],
            "all_development_gates_pass": bool(row["all_development_gates_pass"]),
            "macro": row["macro"],
            "gate_checks": row["gate_checks"],
        })
    selected = select(candidates)
    result = {
        "schema_version": "qpeg-v4-development-checkpoint-selection-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_OPEN_CONFIRMATION_ONCE" if selected is not None else "FAIL_STOP_DEVELOPMENT",
        "selection_rule": "earliest checkpoint passing every frozen development gate",
        "selected_checkpoint_step": selected,
        "confirmation_opened": False,
        "candidates": sorted(candidates, key=lambda row: row["checkpoint_step"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
