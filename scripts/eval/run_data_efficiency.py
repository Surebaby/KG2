#!/usr/bin/env python
"""Data-efficiency rigour scan (paper §5.5 indicator 4)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from kgproweight.eval.data_efficiency import f1_curve_from_summary, make_subset_file
from kgproweight.eval.stats import mean_std_ci
from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import checkpoint_dir, data_dir, output_dir

configure_logging("INFO")
logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 5000, 10000, 15000])
    p.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 2024])
    p.add_argument(
        "--silver",
        default=None,
        help="Path to silver_with_logprobs.jsonl (defaults to checkpoint_dir/prm_alpha_gate/...).",
    )
    p.add_argument("--dataset", default="hotpotqa", help="Dataset for evaluation.")
    p.add_argument("--split", default="dev")
    p.add_argument("--total_steps", type=int, default=1500)
    p.add_argument("--save_root", default=None)
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def _read_metric(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return {}


def main():
    args = parse_args()
    silver = args.silver or str(Path(checkpoint_dir()) / "prm_alpha_gate" / "silver_with_logprobs.jsonl")
    save_root = Path(args.save_root) if args.save_root else Path(output_dir()) / "rigor" / "data_eff"
    save_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[int, Dict[str, float]] = {}
    for n in args.sizes:
        f1_values: List[float] = []
        for seed in args.seeds:
            run_dir = save_root / f"size_{n}_seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            subset_path = run_dir / "silver_subset.jsonl"
            make_subset_file(silver, n=n, seed=seed, output_path=subset_path)

            ppo_out = run_dir / "ppo"
            ppo_cmd = [
                args.python,
                "scripts/train/phase3_ppo.py",
                "--silver_data",
                str(subset_path),
                "--output_dir",
                str(ppo_out),
                "--total_steps",
                str(args.total_steps),
                "--seed",
                str(seed),
            ]
            logger.info("Training subset N=%d seed=%d", n, seed)
            try:
                subprocess.run(ppo_cmd, check=True)
            except subprocess.CalledProcessError as exc:
                logger.error("PPO failed N=%d seed=%d: %s", n, seed, exc)
                continue

            eval_dir = run_dir / "eval"
            eval_cmd = [
                args.python,
                "scripts/eval/run_kg_proweight.py",
                "--checkpoint",
                str(ppo_out / "final"),
                "--datasets",
                args.dataset,
                "--split",
                args.split,
                "--save_root",
                str(eval_dir),
                "--seeds",
                str(seed),
            ]
            try:
                subprocess.run(eval_cmd, check=True)
            except subprocess.CalledProcessError as exc:
                logger.error("Eval failed N=%d seed=%d: %s", n, seed, exc)
                continue

            metric_paths = list(eval_dir.rglob("metric_score.json"))
            if not metric_paths:
                logger.warning("No metric_score.json found in %s", eval_dir)
                continue
            metric = _read_metric(metric_paths[0])
            if "f1" in metric:
                f1_values.append(float(metric["f1"]))

        summary[n] = {**mean_std_ci(f1_values), "f1_values": f1_values}

    curve = f1_curve_from_summary({n: {k: v for k, v in s.items() if isinstance(v, (int, float))} for n, s in summary.items()})
    out = {"curve": curve, "summary": summary}
    out_path = save_root / "data_efficiency.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    logger.info("Data efficiency curve → %s", out_path)


if __name__ == "__main__":
    main()
