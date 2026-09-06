#!/usr/bin/env python3
"""Freeze the outcome-only PPO pilot configs and exact parent schedule prefixes.

CPU/read-only upstream validation only: no model loading, CUDA calls, training,
or mutation of the parent v4 data/preflight.  Existing output files are never
overwritten.  Failed preparation leaves a uniquely identified failure record.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import yaml

from scripts.prepare.preflight_mixed_ppo_three_dataset_v4_proof800 import (
    DEFAULT_DATA_DIR,
    DEFAULT_PROTECTED_LEDGER_DIR,
    DEFAULT_REPLAY_DIR,
    PREFLIGHT_STATUS_PASS,
    REQUIRED_DATA_FILES,
    REQUIRED_REPLAY_FILES,
    run_preflight,
)
from scripts.prepare.resolve_phase3_ppo_runtime_config import resolve_phase3_ppo_runtime_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "outputs/audits/ppo_mixed4_emf1_v1_seed42"
EXPERIMENT_ID = "PPO-MIXED4-EMF1-V1-CONFIG-LOCK-SEED42"
STATUS = "CONFIG_READY_GPU_PROBE_NOT_STARTED"
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
CONFIGS = {
    "outcome12000": ROOT / "configs/training/phase3_ppo_mixed4_emf1_v1_outcome_seed42.yaml",
    "probe12": ROOT / "configs/training/phase3_ppo_mixed4_emf1_v1_outcome_probe12_seed42.yaml",
    "smoke600": ROOT / "configs/training/phase3_ppo_mixed4_emf1_v1_outcome_smoke600_seed42.yaml",
}
TOTALS = {"outcome12000": 12000, "probe12": 12, "smoke600": 600}


def identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def schedule_prefix(parent_bytes: bytes, count: int, groups_per_dataset: int) -> tuple[bytes, dict[str, Any]]:
    """Return an exact byte prefix after independent K4/identity/quota checks."""
    lines = parent_bytes.splitlines(keepends=True)
    if count <= 0 or count % 4 or len(lines) < count:
        raise ValueError("schedule prefix must contain complete K4 groups")
    prefix = b"".join(lines[:count])
    rows = [json.loads(line) for line in lines[:count]]
    if [r.get("rollout_index") for r in rows] != list(range(1, count + 1)):
        raise ValueError("parent prefix rollout indices are not contiguous from one")
    counts: Counter[str] = Counter()
    graph_groups = 0
    for start in range(0, count, 4):
        group = rows[start:start + 4]
        expected_group = start // 4 + 1
        identities = {(r.get("dataset"), r.get("qid"), r.get("question_sha256")) for r in group}
        if len(identities) != 1 or any(not part for part in next(iter(identities))):
            raise ValueError("K4 group contains mixed or missing question identities")
        if [r.get("within_group_rollout") for r in group] != [1, 2, 3, 4]:
            raise ValueError("K4 within-group indices differ from 1,2,3,4")
        if any(r.get("prompt_group_index") != expected_group for r in group):
            raise ValueError("K4 prompt group indices are not sequential")
        if len({r.get("process_reward_eligible") for r in group}) != 1:
            raise ValueError("K4 group has inconsistent source eligibility")
        counts[group[0]["dataset"]] += 1
        graph_groups += group[0].get("process_reward_eligible") is True
    if dict(counts) != {dataset: groups_per_dataset for dataset in DATASETS}:
        raise ValueError(f"prefix is not balanced across three datasets: {dict(counts)}")
    if not parent_bytes.startswith(prefix):
        raise AssertionError("prefix serialization changed parent bytes")
    return prefix, {
        "trajectories": count, "prompt_groups": count // 4,
        "groups_by_dataset": dict(counts), "graph_groups": graph_groups,
        "exact_parent_byte_prefix": True, "k4_identity_groups": True,
        "sha256": hashlib.sha256(prefix).hexdigest(),
    }


def validate_runtime_config(cfg: Mapping[str, Any], *, arm: str, parent_dir: Path, replay_dir: Path) -> None:
    expected = {
        "runtime_contract_version": "v2", "mixed_outcome_reward": True,
        "mixed_text_reward": False, "proofkg_process_reward": False,
        "proofkg_outcome_only_reward": False, "pure_em_reward": False,
        "alpha_gate_path": None, "alpha_override": None,
        "gamma": 1.0, "lam": 0.99, "sft_replay_ratio": 0.10,
        "sft_anchor_weight": 0.10, "sft_anchor_interval": 0,
        "ppo_max_passages": 15, "rollouts_per_prompt": 4, "batch_size": 4,
        "mini_batch_size": 1, "seed": 42, "outcome_weight": 4.0,
        "proofkg_f1_weight": 0.10, "total_steps": TOTALS[arm],
    }
    mismatch = {key: {"expected": value, "actual": cfg.get(key)} for key, value in expected.items() if cfg.get(key) != value}
    path_expectations = {
        "silver_path": parent_dir / "silver_train.jsonl",
        "question_kg_records_path": parent_dir / "question_kg_records.jsonl",
        "rollout_sampling_weights_path": parent_dir / "sampling_weights.jsonl",
        "sft_replay_silver_path": replay_dir / "silver_train.jsonl",
        "fixed_rollout_schedule_path": parent_dir / "fixed_rollout_schedule.jsonl" if arm == "outcome12000" else DEFAULT_OUTPUT_DIR / f"{arm}_schedule.jsonl",
    }
    for key, expected_path in path_expectations.items():
        actual = cfg.get(key)
        if not actual or (ROOT / str(actual)).resolve() != expected_path.resolve():
            mismatch[key] = {"expected": str(expected_path.resolve()), "actual": actual}
    if mismatch:
        raise ValueError(f"{arm} runtime config contract mismatch: {mismatch}")


def _config_sources(path: Path, seen: set[Path] | None = None) -> set[Path]:
    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        return seen
    seen.add(path)
    contents = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for included in contents.get("includes") or []:
        _config_sources(path.parent / included, seen)
    return seen


def prepare(output_dir: Path = DEFAULT_OUTPUT_DIR, *, experiment_id: str = EXPERIMENT_ID) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"refusing nonempty/existing-file output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    try:
        parent_dir, replay_dir = (ROOT / DEFAULT_DATA_DIR).resolve(), (ROOT / DEFAULT_REPLAY_DIR).resolve()
        parent_paths = [parent_dir / name for name in [*REQUIRED_DATA_FILES.values(), "report.json", "manifest.json"]]
        replay_paths = [replay_dir / name for name in REQUIRED_REPLAY_FILES]
        config_paths = sorted(set().union(*(_config_sources(p) for p in CONFIGS.values())))
        initial_inputs = {str(p): identity(p) for p in [*parent_paths, *replay_paths, *config_paths]}
        resolved = {arm: resolve_phase3_ppo_runtime_config(str(path)) for arm, path in CONFIGS.items()}
        for arm, cfg in resolved.items():
            validate_runtime_config(cfg, arm=arm, parent_dir=parent_dir, replay_dir=replay_dir)
        allowed_differences = {"output_dir", "total_steps", "save_every_steps", "fixed_rollout_schedule_path"}
        common = {key: value for key, value in resolved["outcome12000"].items() if key not in allowed_differences}
        if any({k: v for k, v in cfg.items() if k not in allowed_differences} != common for cfg in resolved.values()):
            raise ValueError("pilot/full resolved configs differ beyond output/schedule/budget/checkpoint cadence")
        parent_bytes = (parent_dir / "fixed_rollout_schedule.jsonl").read_bytes()
        if len(parent_bytes.splitlines()) != 12000:
            raise ValueError("authoritative parent schedule does not have 12000 trajectories")
        prefixes = {arm: schedule_prefix(parent_bytes, n, n // 12) for arm, n in (("probe12", 12), ("smoke600", 600))}
        silver = [json.loads(line) for line in (parent_dir / "silver_train.jsonl").read_bytes().splitlines()]
        passage_counts = Counter(len(row.get("retrieved_passages") or []) for row in silver)
        if len(silver) != 3000 or dict(passage_counts) != {10: 3000}:
            raise ValueError("v4 rollout silver must have exactly 3000 rows with 10 passages each")
        preflight = run_preflight(data_dir=parent_dir, replay_dir=replay_dir, protected_ledger_dir=(ROOT / DEFAULT_PROTECTED_LEDGER_DIR).resolve())
        checks = preflight.get("checks") or {}
        if preflight.get("status") != PREFLIGHT_STATUS_PASS or len(checks) != 46 or not all(value is True for value in checks.values()):
            raise ValueError("read-only parent v4 preflight did not pass exactly 46 checks")
        if initial_inputs != {str(p): identity(p) for p in [*parent_paths, *replay_paths, *config_paths]}:
            raise ValueError("parent/config/replay inputs changed while preparing the lock")
        for arm, (payload, _stats) in prefixes.items():
            with (output_dir / f"{arm}_schedule.jsonl").open("xb") as handle:
                handle.write(payload)
        for arm, cfg in resolved.items():
            _write_json(output_dir / f"{arm}.resolved_config.json", cfg)
        _write_json(output_dir / "parent_v4_readonly_preflight.json", preflight)
        protocol = {
            "schema_version": "ppo-emf1-pilot-config-protocol-v1", "experiment_id": experiment_id,
            "status": STATUS, "created_at_utc": started, "seed": 42,
            "experiment_kind": "combined_performance_pilot_not_alpha_ablation",
            "reward": "valid:4*(canonical_alias_EM+0.1*canonical_alias_F1); invalid:-4; outcome_only",
            "runtime_contract": common,
            "schedule_rule": "Exact byte prefixes of the complete frozen v4 K4 schedule; no resampling or serialization",
            "schedule_stats": {arm: stats for arm, (_payload, stats) in prefixes.items()},
            "v4_rollout_passage_counts": dict(passage_counts),
            "loader_passage_limit": 15,
            "source_bindings": initial_inputs,
            "config_roles": {arm: str(path) for arm, path in CONFIGS.items()},
            "parent_preflight_checks": len(checks), "parent_preflight_pass": True,
            "training_started": False, "gpu_probe_started": False,
            "scientific_boundary": "Configuration/data preparation only. No EM/F1, PPO stability, process-reward utility, or learned-alpha conclusion. Formal GPU training still follows the registered probe/smoke approval sequence.",
        }
        _write_json(output_dir / "protocol.json", protocol)
        outputs = {path.name: identity(path) for path in sorted(output_dir.iterdir()) if path.is_file()}
        manifest = {
            "schema_version": "ppo-emf1-pilot-config-manifest-v1", "experiment_id": experiment_id,
            "status": STATUS, "started_at_utc": started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "code_state": "Working-tree source bindings; root final release manifest must bind all final runtime sources",
            "code_bindings": {str(p): identity(p) for p in [Path(__file__), ROOT / "scripts/prepare/resolve_phase3_ppo_runtime_config.py", ROOT / "scripts/prepare/preflight_mixed_ppo_three_dataset_v4_proof800.py"]},
            "inputs": initial_inputs, "outputs": outputs,
            "model_version": common.get("base_model"), "sft_checkpoint": common.get("sft_checkpoint"),
            "data_version": "mixed-ppo-three-dataset-v4-proof800", "seed": 42,
            "evaluation_protocol": "parent-v4-preflight-v1-plus-emf1-pilot-config-contract-v1",
            "optimizer_updates": 0, "training_started": False,
            "relocated_audit_copy": output_dir != DEFAULT_OUTPUT_DIR.resolve(),
            "runtime_schedule_paths_rewritten": False,
        }
        _write_json(output_dir / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        _write_json(output_dir / "FAILED_PREPARATION.json", {
            "experiment_id": experiment_id, "status": "FAIL_CONFIG_PREPARATION_NOT_TRAINED",
            "started_at_utc": started, "error_type": type(exc).__name__, "error": str(exc),
            "training_started": False,
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    args = parser.parse_args()
    manifest = prepare(args.output_dir, experiment_id=args.experiment_id)
    print(json.dumps({"status": manifest["status"], "experiment_id": manifest["experiment_id"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
