#!/usr/bin/env python
"""Create an append-only model-registration derivative of a frozen paired-KG protocol."""

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


def _resolve_protocol_path(parent: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # Frozen protocols in this project store paths relative to project root.
    project_root = Path.cwd()
    candidate = project_root / path
    return candidate.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent_protocol", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    parent_path = Path(args.parent_protocol).resolve()
    adapter = Path(args.adapter).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite derived protocol: {output}")
    if not (adapter / "adapter_config.json").is_file() or not (
        adapter / "adapter_model.safetensors"
    ).is_file():
        raise SystemExit(f"incomplete PEFT adapter: {adapter}")

    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if not str(parent.get("status") or "").startswith("FROZEN_BEFORE_MODEL_INFERENCE"):
        raise SystemExit("parent protocol was not frozen before inference")
    if args.model_label in (parent.get("models") or {}):
        raise SystemExit(f"model label already exists in parent: {args.model_label}")
    for arm in ("arm_legacy", "arm_proof"):
        identity = parent["inputs"][arm]
        path = _resolve_protocol_path(parent_path, identity["path"])
        if not path.is_file() or _sha256(path) != identity["sha256"]:
            raise SystemExit(f"frozen {arm} input missing or changed: {path}")
    evaluator = Path("scripts/eval/evaluate_a1_fixed_context_kg.py").resolve()
    frozen_evaluator_hash = str(parent.get("implementation", {}).get("evaluator_sha256") or "")
    if not frozen_evaluator_hash or _sha256(evaluator) != frozen_evaluator_hash:
        raise SystemExit("paired evaluator differs from the frozen implementation")

    derived = copy.deepcopy(parent)
    derived["status"] = "FROZEN_BEFORE_MODEL_INFERENCE_DERIVED_MODEL_REGISTRY"
    derived["parent_protocol"] = {
        "path": str(parent_path),
        "sha256": _sha256(parent_path),
    }
    derived["model_registration"] = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "single_change": "append one already-trained model identity; inputs/generation/metrics/gates unchanged",
    }
    derived.setdefault("models", {})[args.model_label] = {
        "adapter_path": str(adapter),
        "adapter_config_sha256": _sha256(adapter / "adapter_config.json"),
        "adapter_model_sha256": _sha256(adapter / "adapter_model.safetensors"),
        "experiment_id": args.experiment_id,
    }
    derived.setdefault("scientific_boundary", {})["derived_protocol_changes_evaluation"] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(derived, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "status": derived["status"],
        "output": str(output),
        "sha256": _sha256(output),
        "parent_sha256": derived["parent_protocol"]["sha256"],
        "model_label": args.model_label,
        "model": derived["models"][args.model_label],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
