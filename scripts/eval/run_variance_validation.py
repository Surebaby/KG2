#!/usr/bin/env python
"""Empirical validation of Theorem 2 (paper §6.2).

Runs short PPO loops under different α strategies (dynamic vs fixed
0.0 / 0.5 / 1.0) and logs the per-update advantage variance. The
expected output (per paper §6.2) is

    Var(dynamic) ≤ Var(fixed)

with a visible gap that grows on KG-poor batches.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from kgproweight.eval.variance_validation import VarianceLog, summarise_variance_logs
from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import checkpoint_dir, output_dir

configure_logging("INFO")
logger = get_logger(__name__)


_STRATEGIES = {
    "dynamic": None,
    "fixed_0.0": 0.0,
    "fixed_0.5": 0.5,
    "fixed_1.0": 1.0,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument(
        "--strategies",
        nargs="+",
        default=list(_STRATEGIES.keys()),
        choices=list(_STRATEGIES.keys()),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_root", default=None)
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def _extract_variance(history_jsonl: Path) -> VarianceLog:
    log = VarianceLog(name=history_jsonl.parent.name, steps=[], variances=[])
    if not history_jsonl.exists():
        return log
    with open(history_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            log.append(obj.get("step", 0), obj.get("advantage_var", 0.0))
    return log


def main():
    args = parse_args()
    save_root = Path(args.save_root) if args.save_root else Path(output_dir()) / "rigor" / "variance"
    save_root.mkdir(parents=True, exist_ok=True)
    logs: List[VarianceLog] = []
    for name in args.strategies:
        override = _STRATEGIES[name]
        out = save_root / name
        ppo_cmd = [
            args.python,
            "scripts/train/phase3_ppo.py",
            "--output_dir",
            str(out),
            "--total_steps",
            str(args.max_steps),
            "--seed",
            str(args.seed),
        ]
        if override is not None:
            ppo_cmd += ["--alpha_override", str(override)]
        logger.info("Variance run %s: %s", name, ppo_cmd)
        try:
            subprocess.run(ppo_cmd, check=True)
        except subprocess.CalledProcessError as exc:
            logger.error("Variance run %s failed: %s", name, exc)
            continue
        log = _extract_variance(out / "history.jsonl")
        log.name = name
        log.to_csv(save_root / f"{name}.csv")
        logs.append(log)

    summary = summarise_variance_logs(logs)
    (save_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Variance summary → %s", save_root)


if __name__ == "__main__":
    main()
