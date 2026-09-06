#!/usr/bin/env python
"""Materialize a runnable light20 PPO config only after SFT selection PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from kgproweight.config import ProjectConfig, load_config


EXPECTED_EXPERIMENT = "SFT-PROOFKG-LIGHT20-V2-CHECKPOINT-SELECTION"
EXPECTED_THRESHOLDS = {
    "expected_n": 200,
    "min_parse_rate": 0.995,
    "min_em": 0.770,
    "min_hidden_em": 0.545,
}
LABEL_TO_DIR = {
    "step40": "checkpoint-40",
    "step80": "checkpoint-80",
    "step120": "checkpoint-120",
    "final": "final",
}
DATA_SHA256 = {
    "data/silver_data/proofkg_curriculum_light20_v2_n5000_seed42_20260831/silver_curriculum.jsonl":
        "1cfd5f989f65a476390151e74ae4511e6199129d4394d884276fea92487a067b",
    "data/silver_data/proofkg_curriculum_light20_v2_n5000_seed42_20260831/question_kg_records.jsonl":
        "8434ed430a644d6e5b175a2eba1799ba2ac33075731b2af7456b60786fd32521",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_selected_checkpoint(report: dict[str, Any], checkpoint_root: Path) -> tuple[str, Path]:
    if report.get("experiment_id") != EXPECTED_EXPERIMENT:
        raise ValueError("unexpected SFT selection experiment_id")
    if report.get("status") != "PASS":
        raise ValueError("SFT checkpoint selection did not PASS; PPO remains blocked")
    if report.get("thresholds") != EXPECTED_THRESHOLDS:
        raise ValueError("SFT selection thresholds differ from the frozen light20 gate")
    label = str(report.get("selected") or "")
    if label not in LABEL_TO_DIR:
        raise ValueError(f"unsupported selected SFT label: {label!r}")
    candidate = next(
        (row for row in report.get("candidates", []) if row.get("label") == label),
        None,
    )
    if not candidate or candidate.get("passes_gate") is not True:
        raise ValueError("selected SFT candidate is missing or did not pass its gate")
    adapter = checkpoint_root / LABEL_TO_DIR[label]
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        if not (adapter / name).is_file():
            raise FileNotFoundError(f"selected adapter is incomplete: {adapter / name}")
    return label, adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_report", required=True)
    parser.add_argument(
        "--checkpoint_root",
        default="checkpoints/sft_proofkg_curriculum_light20_v2_n5000_seed42",
    )
    parser.add_argument(
        "--template",
        default=(
            "configs/training/"
            "phase3_ppo_proofkg_curriculum_light20_v2_smoke600_seed42.template.yaml"
        ),
    )
    parser.add_argument("--output_config", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selection_path = Path(args.selection_report)
    output_path = Path(args.output_config)
    template_path = Path(args.template)
    checkpoint_root = Path(args.checkpoint_root)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite resolved PPO config: {output_path}")
    if output_path.parent.resolve() != template_path.parent.resolve():
        raise ValueError("output_config must stay beside the template so includes resolve identically")

    report = json.loads(selection_path.read_text(encoding="utf-8"))
    label, adapter = resolve_selected_checkpoint(report, checkpoint_root)
    manifest_path = checkpoint_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise ValueError("light20 SFT manifest is not COMPLETE")

    for raw_path, expected in DATA_SHA256.items():
        path = Path(raw_path)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"light20 training artifact hash mismatch: {path}")

    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    training = config["training"]
    output_dir = f"outputs/ppo_proofkg_curriculum_light20_v2_smoke600_seed42_from_{label}"
    if Path(output_dir).exists():
        raise FileExistsError(f"PPO output already exists: {output_dir}")
    training["sft_checkpoint"] = str(adapter)
    training["sft_selection_report_path"] = str(selection_path)
    training["output_dir"] = output_dir

    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    resolved = load_config(output_path, validate=ProjectConfig).training
    if resolved.sft_checkpoint != str(adapter) or resolved.ppo.total_ppo_steps != 600:
        output_path.unlink(missing_ok=True)
        raise ValueError("resolved PPO config failed post-write validation")
    print(json.dumps({
        "status": "READY_NOT_STARTED",
        "selected_label": label,
        "sft_checkpoint": str(adapter),
        "selection_report": str(selection_path),
        "output_config": str(output_path),
        "ppo_output_dir": output_dir,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
