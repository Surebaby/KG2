#!/usr/bin/env python
"""Preflight or train the dedicated observation-conditioned Query Controller."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json

from kgproweight.training.query_controller import (
    dry_run,
    load_config,
    run_query_controller_sft,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Versioned Controller YAML")
    parser.add_argument("--dry_run", action="store_true", help="CPU-only data/token preflight")
    parser.add_argument("--probe", action="store_true", help="Mark this run as a probe")
    parser.add_argument("--probe_steps", type=int, help="Override max_steps for a unique probe")
    parser.add_argument("--output_dir", help="Unique output override; existing paths are refused")
    parser.add_argument("--experiment_id", help="Unique Experiment ID override")
    parser.add_argument(
        "--initialization", choices=("base_instruct", "planner_adapter"),
        help="Explicit initialization override",
    )
    parser.add_argument("--init_adapter_path", help="Required with planner_adapter")
    args = parser.parse_args()

    if args.probe_steps is not None and args.probe_steps <= 0:
        raise SystemExit("--probe_steps must be positive")
    if args.probe_steps is not None and not args.probe:
        raise SystemExit("--probe_steps requires --probe")
    if args.probe_steps is not None and not args.output_dir:
        raise SystemExit("an overridden probe requires a unique --output_dir")
    if args.output_dir and not args.dry_run and not (args.experiment_id or args.probe_steps):
        raise SystemExit("an output override requires a unique --experiment_id")

    cfg = load_config(
        args.config,
        output_override=args.output_dir,
        max_steps_override=args.probe_steps,
    )
    if args.experiment_id:
        cfg = replace(cfg, experiment_id=args.experiment_id)
    elif args.probe_steps is not None:
        cfg = replace(cfg, experiment_id=f"{cfg.experiment_id}-STEPS{args.probe_steps}")
    if args.initialization or args.init_adapter_path:
        initialization = args.initialization or cfg.initialization
        init_adapter = args.init_adapter_path or cfg.init_adapter_path
        cfg = replace(
            cfg,
            initialization=initialization,
            init_adapter_path=init_adapter,
        )

    if args.dry_run:
        report = dry_run(cfg)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    result = run_query_controller_sft(cfg, probe=args.probe)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
