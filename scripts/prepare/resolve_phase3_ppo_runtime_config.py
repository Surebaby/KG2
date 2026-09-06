"""Resolve the exact Phase-3 PPO CLI dataclass without starting training.

The YAML schema alone is not the runtime contract: ``scripts/train/phase3_ppo.py``
performs CLI/default forwarding before calling the trainer.  This helper
captures that exact dataclass by replacing only the final training call.  It is
used by versioned preflights and lock files; it never loads model weights.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import scripts.train.phase3_ppo as phase3_cli


def resolve_phase3_ppo_runtime_config(config_path: str | Path) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    original_argv = sys.argv
    original_runner = phase3_cli.run_phase3_ppo

    def _capture(cfg: Any) -> None:
        captured["cfg"] = cfg

    try:
        phase3_cli.run_phase3_ppo = _capture
        sys.argv = ["phase3_ppo.py", "--config", str(config_path)]
        phase3_cli.main()
    finally:
        phase3_cli.run_phase3_ppo = original_runner
        sys.argv = original_argv

    if "cfg" not in captured:
        raise RuntimeError("Phase-3 PPO CLI did not construct a runtime config")
    return asdict(captured["cfg"])

