#!/usr/bin/env python
"""Sanity checks before full-scale evaluation.

Runs fast, CPU-only regression tests for answer extraction and prompt schema,
then optionally launches a small GPU eval (200 samples) when data/indexes are
available.

Usage:
    python scripts/eval/sanity_check.py
    python scripts/eval/sanity_check.py --run-eval --test_sample_num 200
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-eval",
        action="store_true",
        help="Also run a small KG-ProWeight eval if data/indexes/checkpoints exist.",
    )
    p.add_argument("--test_sample_num", type=int, default=200)
    p.add_argument("--dataset", default="hotpotqa")
    p.add_argument("--gpu_id", default="0")
    p.add_argument(
        "--kg2-fallback",
        action="store_true",
        help="Point KGPW_* dirs at ../kg2/ when kgpaper dirs are missing.",
    )
    return p.parse_args()


def _maybe_use_kg2_paths(project_root: Path) -> None:
    import os

    kg2 = project_root.parent / "kg2"
    mapping = {
        "KGPW_DATA_DIR": kg2 / "data",
        "KGPW_INDEX_DIR": kg2 / "indexes",
        "KGPW_CHECKPOINT_DIR": kg2 / "checkpoints",
        "KGPW_OUTPUT_DIR": project_root / "outputs",
    }
    for env_var, path in mapping.items():
        if env_var not in os.environ and path.exists():
            os.environ[env_var] = str(path)


def _run_pytest(project_root: Path) -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_pipeline_inference.py",
        "tests/test_parse_teacher_output.py",
        "-q",
    ]
    print("Running unit tests:", " ".join(cmd))
    return subprocess.call(cmd, cwd=project_root)


def _paths_ready(project_root: Path) -> bool:
    import os

    data_dir = Path(os.environ.get("KGPW_DATA_DIR", project_root / "data"))
    index_dir = Path(os.environ.get("KGPW_INDEX_DIR", project_root / "indexes"))
    ckpt_dir = Path(os.environ.get("KGPW_CHECKPOINT_DIR", project_root / "checkpoints"))
    dev = data_dir / "hotpotqa" / "dev.jsonl"
    index = index_dir / "e5_Flat.index"
    ckpt = ckpt_dir / "kg_proweight_final" / "final"
    return dev.exists() and index.exists() and ckpt.exists()


def _run_small_eval(project_root: Path, args) -> int:
    cmd = [
        sys.executable,
        "scripts/eval/run_kg_proweight.py",
        "--datasets",
        args.dataset,
        "--split",
        "dev",
        "--test_sample_num",
        str(args.test_sample_num),
        "--seeds",
        "42",
        "--gpu_id",
        args.gpu_id,
    ]
    print("Running small eval:", " ".join(cmd))
    return subprocess.call(cmd, cwd=project_root)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.kg2_fallback:
        _maybe_use_kg2_paths(project_root)

    rc = _run_pytest(project_root)
    if rc != 0:
        print("Sanity check FAILED: unit tests did not pass.")
        return rc

    print("Sanity check PASSED: answer extraction and prompt schema tests OK.")

    if not args.run_eval:
        print("Skipping GPU eval (pass --run-eval to launch a small run).")
        return 0

    if not _paths_ready(project_root):
        print(
            "Skipping GPU eval: data/index/checkpoint not found. "
            "Set KGPW_DATA_DIR / KGPW_INDEX_DIR / KGPW_CHECKPOINT_DIR or use --kg2-fallback."
        )
        return 0

    rc = _run_small_eval(project_root, args)
    if rc != 0:
        print("Small GPU eval failed (exit code %d)." % rc)
        return rc

    print("Small GPU eval completed. Inspect outputs/kg_proweight/ for metric_score.txt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
