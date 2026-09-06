#!/usr/bin/env python
"""CPU-only preflight and effective-config lock for SAEG-v1 continued-SFT."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from peft import PeftConfig
from safetensors import safe_open

from kgproweight.config import ProjectConfig, load_config
from kgproweight.utils.logging import dump_manifest
from kgproweight.utils.paths import model_path
from scripts.prepare.materialize_saeg_v1_sft_dataset import validate_trajectory


EXPERIMENT_ID = "SAEG-V1-SFT-BALANCED-EPOCH4860-SEED42"
EPOCH_STATUS = "FROZEN_TRAIN_ONLY_SFT_EPOCH_NOT_TRAINED"
RELEASE_STATUS = "COMPLETE_TRAIN_ONLY_FAMILY_DISJOINT_NOT_TRAINED"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def without_epoch_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    output = deepcopy(dict(row))
    metadata = deepcopy(dict(output.get("metadata") or {}))
    for key in list(metadata):
        if key.startswith("sft_epoch_"):
            metadata.pop(key)
    output["metadata"] = metadata
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/training/phase3_sft_saeg_v1_balanced_epoch4860_seed42.yaml"
        ),
    )
    parser.add_argument(
        "--epoch_report",
        type=Path,
        default=Path(
            "data/silver_data/saeg_v1_sft_balanced_epoch4860_seed42_v1/report.json"
        ),
    )
    parser.add_argument(
        "--source_release",
        type=Path,
        default=Path(
            "data/silver_data/saeg_v1_train4860_family_disjoint_seed42_v2/silver_train.jsonl"
        ),
    )
    parser.add_argument(
        "--source_release_report",
        type=Path,
        default=Path(
            "data/silver_data/saeg_v1_train4860_family_disjoint_seed42_v2/report.json"
        ),
    )
    parser.add_argument(
        "--dataset_release_audit",
        type=Path,
        default=Path("outputs/audits/saeg_v1_dataset_release_audit/report.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/audits/saeg_v1_sft_balanced_epoch4860_preflight_v1"),
    )
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite SAEG SFT preflight: {args.out}")
    for path in (
        args.config,
        args.epoch_report,
        args.source_release,
        args.source_release_report,
        args.dataset_release_audit,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    cfg = load_config(args.config, validate=ProjectConfig).training
    epoch_report = json.loads(args.epoch_report.read_text(encoding="utf-8"))
    release_report = json.loads(args.source_release_report.read_text(encoding="utf-8"))
    dataset_audit = json.loads(args.dataset_release_audit.read_text(encoding="utf-8"))
    silver_path = Path(cfg.silver_path)
    adapter_path = Path(cfg.sft_init_adapter_path)
    output_dir = Path(cfg.output_dir)
    base_path = Path(model_path(cfg.base_model))
    for path in (silver_path, adapter_path, base_path):
        if not path.exists():
            raise FileNotFoundError(path)

    epoch = read_jsonl(silver_path)
    source = read_jsonl(args.source_release)
    source_by_qid = {str(row["qid"]): row for row in source}
    if len(source_by_qid) != len(source):
        raise ValueError("source release must have unique variant qids")

    content_exact = 0
    parser_valid = 0
    for row in epoch:
        source_row = source_by_qid.get(str(row["qid"]))
        content_exact += source_row is not None and without_epoch_metadata(row) == source_row
        try:
            validate_trajectory(row)
        except ValueError:
            continue
        parser_valid += 1

    group_counts = Counter(
        f"{row['dataset']}::{row['evidence_mode']}" for row in epoch
    )
    expected_group_counts = {
        str(key): int(value)
        for key, value in (epoch_report.get("sampler") or {}).get("exact_group_quotas", {}).items()
    }
    sample_indices = [
        int((row.get("metadata") or {}).get("sft_epoch_sample_index", -1))
        for row in epoch
    ]
    peft_cfg = PeftConfig.from_pretrained(adapter_path)
    adapter_file = adapter_path / "adapter_model.safetensors"
    with safe_open(adapter_file, framework="pt", device="cpu") as handle:
        adapter_tensor_count = len(handle.keys())

    effective_batch = int(cfg.sft_batch_size) * int(cfg.sft_grad_accum)
    expected_updates = math.ceil(len(epoch) / effective_batch)
    token_gate = (dataset_audit.get("token_audit") or {}).get("train_full_top10") or {}
    epoch_output = ((epoch_report.get("outputs") or {}).get("silver_train") or {})
    release_output = ((release_report.get("outputs") or {}).get("silver_train") or {})

    checks = {
        "epoch_status_frozen_not_trained": epoch_report.get("status") == EPOCH_STATUS,
        "release_status_complete_not_trained": release_report.get("status") == RELEASE_STATUS,
        "dataset_release_gate_passed": dataset_audit.get("status") == "PASS_DATASET_RELEASE_NOT_TRAINED_NOT_EVALUATED",
        "epoch_hash_matches_report": sha256_file(silver_path) == epoch_output.get("sha256"),
        "source_hash_matches_report": sha256_file(args.source_release) == release_output.get("sha256"),
        "epoch_rows_4860": len(epoch) == 4860,
        "source_rows_4860": len(source) == 4860,
        "exact_group_quotas": dict(group_counts) == expected_group_counts,
        "sample_indices_exact": sorted(sample_indices) == list(range(4860)),
        "all_epoch_content_matches_source": content_exact == len(epoch),
        "all_saeg_trajectories_parser_valid": parser_valid == len(epoch),
        "train_eval_qid_overlap_zero": (dataset_audit.get("integrity") or {}).get("train_eval_qid_overlap") == 0,
        "train_dev_family_overlap_zero": (dataset_audit.get("integrity") or {}).get("train_development_family_overlap") == 0,
        "train_confirmation_family_overlap_zero": (dataset_audit.get("integrity") or {}).get("train_confirmation_family_overlap") == 0,
        "token_audit_over_4096_zero": int(token_gate.get("over_4096", -1)) == 0,
        "token_audit_max_below_4096": int(token_gate.get("max", 4097)) <= 4096,
        "base_model_exists": base_path.is_dir(),
        "start_adapter_exists": adapter_path.is_dir(),
        "adapter_safetensors_readable": adapter_tensor_count == 256,
        "lora_r_32": int(peft_cfg.r) == 32,
        "lora_alpha_64": int(peft_cfg.lora_alpha) == 64,
        "lora_dropout_0_05": float(peft_cfg.lora_dropout) == 0.05,
        "lora_targets_exact": set(peft_cfg.target_modules or []) == {"q_proj", "k_proj", "v_proj", "o_proj"},
        "continued_sft_lr_2e_6": float(cfg.sft_lr) == 2e-6,
        "one_epoch": int(cfg.sft_epochs) == 1,
        "effective_batch_32": effective_batch == 32,
        "max_length_4096": int(cfg.sft_max_length) == 4096,
        "expected_updates_152": expected_updates == 152,
        "save_every_38": cfg.sft_save_strategy == "steps" and int(cfg.sft_save_steps) == 38,
        "tensorboard_enabled": cfg.sft_log_with == "tensorboard",
        "tensorboard_autodl_path": str(cfg.sft_logging_dir or "").startswith("/root/tf-logs/"),
        "derived_whole_file_explicit": cfg.split is None and bool(cfg.split_allow_none),
        "output_dir_absent": not output_dir.exists(),
    }
    status = "PASS_NOT_TRAINED" if all(checks.values()) else "FAIL_STOP"
    effective_config = {
        "experiment_id": EXPERIMENT_ID,
        "silver_path": str(silver_path),
        "silver_sha256": sha256_file(silver_path),
        "base_model": cfg.base_model,
        "base_model_resolved": str(base_path),
        "init_adapter_path": str(adapter_path),
        "init_adapter_sha256": sha256_file(adapter_file),
        "output_dir": str(output_dir),
        "seed": int(cfg.seed),
        "epochs": int(cfg.sft_epochs),
        "lr": float(cfg.sft_lr),
        "batch_size": int(cfg.sft_batch_size),
        "grad_accum": int(cfg.sft_grad_accum),
        "effective_batch": effective_batch,
        "max_length": int(cfg.sft_max_length),
        "save_strategy": str(cfg.sft_save_strategy),
        "save_steps": int(cfg.sft_save_steps),
        "log_with": cfg.sft_log_with,
        "logging_dir": cfg.sft_logging_dir,
        "expected_updates": expected_updates,
        "expected_checkpoint_steps": [38, 76, 114, 152],
        "split": cfg.split,
        "split_allow_none": bool(cfg.split_allow_none),
        "lora": {"r": 32, "alpha": 64, "dropout": 0.05, "targets": ["k_proj", "o_proj", "q_proj", "v_proj"]},
    }
    report = {
        "schema_version": "saeg-v1-sft-preflight-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "counts": {
            "epoch_rows": len(epoch),
            "source_rows": len(source),
            "content_exact": content_exact,
            "parser_valid": parser_valid,
            "adapter_tensor_count": adapter_tensor_count,
            "group_counts": dict(sorted(group_counts.items())),
        },
        "effective_config": effective_config,
        "checks": checks,
        "inputs": {
            "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
            "epoch_report": {"path": str(args.epoch_report), "sha256": sha256_file(args.epoch_report)},
            "source_release": {"path": str(args.source_release), "sha256": sha256_file(args.source_release)},
            "source_release_report": {"path": str(args.source_release_report), "sha256": sha256_file(args.source_release_report)},
            "dataset_release_audit": {"path": str(args.dataset_release_audit), "sha256": sha256_file(args.dataset_release_audit)},
        },
        "scientific_boundary": (
            "CPU-only readiness and effective-config lock. No model was trained, no evaluation cohort was opened, "
            "and no reward, target, answer, evidence, or evaluation protocol was changed."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "effective_config.lock.json").write_text(
        json.dumps(effective_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(args.out, extra={"phase": "saeg_v1_sft_preflight", **report}, status=status)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status != "PASS_NOT_TRAINED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
