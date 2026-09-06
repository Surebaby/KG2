#!/usr/bin/env python
"""Append one trained QPEG-v4 checkpoint identity to development inputs.

This creates a derivative protocol. It does not modify the frozen input
protocol and does not inspect or open the confirmation cohort.
"""

from __future__ import annotations

import argparse
import copy
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent_protocol", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--checkpoint_step", type=int, choices=(25, 50, 75), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    parent_path = Path(args.parent_protocol).resolve()
    adapter = Path(args.adapter).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite checkpoint registry: {output}")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("schema_version") != "qpeg-v4-development-paired-input-protocol-v1":
        raise SystemExit("unexpected parent protocol")
    if parent.get("integrity", {}).get("confirmation_opened") is not False:
        raise SystemExit("parent does not prove confirmation remained unopened")
    for key in ("arm_legacy", "arm_proof"):
        identity = parent["inputs"][key]
        path = Path(identity["path"]).resolve()
        if not path.is_file() or _sha256(path) != identity["sha256"]:
            raise SystemExit(f"frozen input missing or changed: {key}")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / name).is_file():
            raise SystemExit(f"incomplete adapter checkpoint: {adapter / name}")

    label = f"adapted_step{args.checkpoint_step}"
    derived = copy.deepcopy(parent)
    derived["schema_version"] = "qpeg-v4-development-checkpoint-protocol-v1"
    derived["status"] = "FROZEN_DEVELOPMENT_MODEL_NOT_EVALUATED"
    derived["parent_protocol"] = {"path": str(parent_path), "sha256": _sha256(parent_path)}
    derived["checkpoint_registration"] = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_step": args.checkpoint_step,
        "single_change": "append already-trained checkpoint identity; evaluation inputs and decoding unchanged",
        "confirmation_opened": False,
        "evaluator": {
            "path": "scripts/eval/evaluate_a1_fixed_context_kg.py",
            "sha256": _sha256(Path("scripts/eval/evaluate_a1_fixed_context_kg.py")),
        },
    }
    derived["models"][label] = {
        "path": str(adapter),
        "adapter_config_sha256": _sha256(adapter / "adapter_config.json"),
        "adapter_model_sha256": _sha256(adapter / "adapter_model.safetensors"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(derived, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": derived["status"],
        "model_label": label,
        "checkpoint_step": args.checkpoint_step,
        "output": str(output),
        "sha256": _sha256(output),
        "confirmation_opened": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
