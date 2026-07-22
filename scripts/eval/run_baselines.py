#!/usr/bin/env python
"""Evaluate every baseline under the same hybrid RRF top-50 retrieval.

Fixes bug #12 — the legacy code used single-route retrieval for baselines
and a hybrid setup for KG-ProWeight, which made the comparison unfair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from kgproweight.eval.baselines import BASELINES, BaselineSpec, baseline_config
from kgproweight.eval.runner import run_evaluation
from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import output_dir

configure_logging("INFO")
logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", default=[b.name for b in BASELINES])
    p.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihopqa", "musique"])
    p.add_argument("--split", default="dev")
    p.add_argument("--test_sample_num", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 2024])
    p.add_argument("--gpu_id", default="0")
    p.add_argument("--save_root", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    save_root = Path(args.save_root) if args.save_root else Path(output_dir()) / "baselines"
    save_root.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    spec_by_name = {b.name: b for b in BASELINES}
    unknown = [m for m in args.methods if m not in spec_by_name]
    if unknown:
        raise ValueError(f"Unknown baselines: {unknown}; available: {list(spec_by_name.keys())}")

    for method in args.methods:
        spec: BaselineSpec = spec_by_name[method]
        for ds in args.datasets:
            for seed in args.seeds:
                save_dir = str(save_root / method / ds / f"seed_{seed}")
                cfg = baseline_config(
                    spec,
                    dataset_name=ds,
                    save_dir=save_dir,
                    split=args.split,
                    test_sample_num=args.test_sample_num,
                    seed=seed,
                    gpu_id=args.gpu_id,
                )
                logger.info("Running %s / %s / seed=%d → %s", method, ds, seed, save_dir)
                try:
                    out = run_evaluation(
                        flashrag_cfg=cfg,
                        pipeline_module=spec.pipeline_module,
                        pipeline_class=spec.pipeline_class,
                        seed=seed,
                        run_mode=spec.run_mode,
                        system_prompt=spec.system_prompt,
                        user_prompt=spec.user_prompt,
                        pred_process_fun=spec.extras.get("pred_process_fun"),
                    )
                except Exception as exc:
                    logger.error("Eval failed for %s/%s seed=%d: %s", method, ds, seed, exc)
                    summary.setdefault(method, {}).setdefault(ds, {})[f"seed_{seed}"] = {"error": str(exc)}
                    continue
                summary.setdefault(method, {}).setdefault(ds, {})[f"seed_{seed}"] = {
                    "save_dir": out.get("save_dir"),
                }

    summary_path = save_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Baselines summary → %s", summary_path)


if __name__ == "__main__":
    main()
