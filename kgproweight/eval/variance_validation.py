"""Theorem 2 empirical validation.

Logs the variance of the per-update PPO advantage under fixed-α vs
dynamic-α regimes. Theorem 2 (paper §6.2) predicts

    V_dynamic ≤ V_fixed − p_miss · (1 − p_miss) · Δ_R² / 4,

so the curves should be visibly lower under the dynamic schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class VarianceLog:
    name: str
    steps: List[int]
    variances: List[float]

    def append(self, step: int, advantage_var: float) -> None:
        self.steps.append(int(step))
        self.variances.append(float(advantage_var))

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("step,advantage_var\n")
            for s, v in zip(self.steps, self.variances):
                fh.write(f"{s},{v}\n")


def summarise_variance_logs(logs: List[VarianceLog]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for log in logs:
        arr = np.asarray(log.variances, dtype=np.float64)
        out[log.name] = {
            "mean": float(arr.mean()) if arr.size else 0.0,
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()) if arr.size else 0.0,
            "max": float(arr.max()) if arr.size else 0.0,
            "n_updates": int(arr.size),
        }
    return out
