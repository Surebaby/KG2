"""α distribution analysis (D_std vs D_dropout)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def _load_alpha_jsonl(path: str | Path) -> List[float]:
    out: List[float] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for a in obj.get("alpha_values", []) or []:
                out.append(float(a))
    return out


def summarise_alpha(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def compare_alpha_distributions(
    d_std_path: str | Path,
    d_dropout_path: str | Path,
) -> Dict[str, Dict[str, float] | float]:
    """Return summary stats for both distributions plus a Welch t-test."""
    from scipy import stats

    std_alpha = _load_alpha_jsonl(d_std_path)
    drop_alpha = _load_alpha_jsonl(d_dropout_path)

    out: Dict[str, Dict[str, float] | float] = {
        "d_std": summarise_alpha(std_alpha),
        "d_dropout": summarise_alpha(drop_alpha),
    }
    if std_alpha and drop_alpha:
        t = stats.ttest_ind(std_alpha, drop_alpha, equal_var=False)
        out["welch_t"] = float(t.statistic)
        out["welch_p_value"] = float(t.pvalue)
    return out
