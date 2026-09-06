"""Read-only CPU verification of the portable full-method release.

This verifies bytes and production consumers. It deliberately does not claim
GPU, calibration, reward rankability, or PPO clearance from CPU checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RELEASE = "outputs/audits/source_gated_mixed4_emf1_v1_release/manifest.json"
DATA = "data/silver_data/mixed_ppo_three_dataset_v4_proof800_n3000_k4_seed42"
REPLAY = "data/silver_data/sft_replay_legacy_strong_train_rendered3to5_n2000_seed42_v2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_files(root: Path, files: dict) -> dict:
    if not files:
        raise ValueError("empty source/data/model lock")
    for name, identity in files.items():
        logical = Path(name)
        if logical.is_absolute() or ".." in logical.parts or logical.as_posix() != name:
            raise ValueError(f"nonportable lock path: {name}")
        path = root / logical
        if not path.is_file() or path.stat().st_size != identity["size_bytes"]:
            raise ValueError(f"file size/missing asset: {name}")
        if sha256(path) != identity["sha256"]:
            raise ValueError(f"file SHA256 mismatch: {name}")
    return {"files": len(files), "bytes": sum(x["size_bytes"] for x in files.values()), "all_sha256_match": True}


def inspect_consumers(root: Path) -> dict:
    from kgproweight.data.silver_dataset import SilverDatasetReader
    from kgproweight.training.phase3_ppo import _load_fixed_rollout_schedule, _load_rollout_sampling_weights
    from scripts.prepare.resolve_phase3_ppo_runtime_config import resolve_phase3_ppo_runtime_config
    import torch
    from transformers import AutoTokenizer

    population = SilverDatasetReader(root / DATA / "silver_train.jsonl").trajectories
    replay = SilverDatasetReader(root / REPLAY / "silver_train.jsonl").trajectories
    counts = dict(Counter(x.dataset for x in population))
    if counts != {"hotpotqa": 1000, "2wikimultihopqa": 1000, "musique": 1000}:
        raise ValueError("v4 population counts differ from the frozen release")
    if len(replay) != 2000 or any(len(x.retrieved_passages) != 10 for x in population):
        raise ValueError("v4/replay consumer count mismatch")
    _, weights = _load_rollout_sampling_weights(root / DATA / "sampling_weights.jsonl", population)
    resolved, pending = {}, set()
    for stage, n in (("probe", 12), ("smoke", 600), ("full", 12000)):
        stage_configs = []
        for arm, mode in (("a", "learned"), ("f", "fixed"), ("t", "text")):
            relative = f"configs/training/phase3_ppo_mixed4_sourcegate_v1_{arm}_{stage}_seed42.yaml"
            cfg = resolve_phase3_ppo_runtime_config(root / relative)
            expected = {"runtime_contract_version": "v2", "source_gated_reward_version": "v1",
                        "source_gate_mode": mode, "mixed_outcome_reward": True, "mixed_text_reward": True,
                        "proofkg_process_reward": True, "proofkg_process_version": "v2_3",
                        "center_text_reward": False, "gamma": 1., "lam": .99,
                        "sft_replay_ratio": .1, "sft_anchor_weight": .1, "total_steps": n}
            for key, value in expected.items():
                if cfg.get(key) != value:
                    raise ValueError(f"full-method forwarding mismatch: {relative}: {key}")
            _load_fixed_rollout_schedule(root / cfg["fixed_rollout_schedule_path"], population,
                                        total_steps=n, rollouts_per_prompt=4, sampling_records=weights)
            if not (root / cfg["source_gate_calibration_path"]).is_file():
                pending.add("REAL_SOURCE_GATE_CALIBRATION_ARTIFACT_NOT_CREATED")
            resolved[f"{arm}_{stage}"] = cfg
            stage_configs.append({k: v for k, v in cfg.items() if k not in {"source_gate_mode", "output_dir"}})
        if not all(cfg == stage_configs[0] for cfg in stage_configs):
            raise ValueError(f"A/F/T differ beyond mode/output at stage {stage}")
    tokenizers = {}
    for model, trust in (("llama3-8b", False), ("rearag-9b", True)):
        tokenizer = AutoTokenizer.from_pretrained(root / "models" / model, trust_remote_code=trust,
                                                 local_files_only=True)
        ids = tokenizer("Question: Where was the bridge built?", add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"empty tokenizer output: {model}")
        tokenizers[model] = {"class": type(tokenizer).__name__, "probe_tokens": len(ids)}
    if not torch.cuda.is_available():
        pending.add("CUDA_NOT_AVAILABLE")
    pending.add("REAL_CANDIDATE_PROCESS_RANKABILITY_AND_GPU_PROBE_NOT_RUN")
    return {"rollout_questions": len(population), "by_dataset": counts, "replay_questions": len(replay),
            "replay_passage_counts": dict(Counter(len(x.retrieved_passages) for x in replay)),
            "resolved_configs": resolved, "tokenizers": tokenizers,
            "cuda_available": torch.cuda.is_available(), "remaining_gpu_work": sorted(pending)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, default=ROOT / RELEASE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("refusing to overwrite deployment verification")
    os.chdir(ROOT)
    manifest = json.loads(args.release_manifest.read_text())
    verified = verify_files(ROOT, manifest["files"])
    consumers = inspect_consumers(ROOT)
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "trl", "peft", "accelerate", "pydantic", "pytest", "tensorboard")}
    report = {"experiment_id": "SOURCE-GATED-MIXED4-EMF1-V1-DEPLOYMENT-CPU-VERIFY",
              "status": "CPU_DEPLOYMENT_VERIFIED_CALIBRATION_GPU_PENDING", "training_started": False,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "project_root": str(ROOT), "versions": versions, "release_sha256": sha256(args.release_manifest),
              "file_verification": verified, "consumers": consumers}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "consumers"}, ensure_ascii=False))
    print(json.dumps({"remaining_gpu_work": consumers["remaining_gpu_work"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
