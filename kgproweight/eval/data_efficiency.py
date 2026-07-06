"""Data-efficiency rigour utilities.

Given a silver dataset, produce reproducible random subsets at multiple
sizes and report the trained model's F1. The CLI lives in
``scripts/eval/run_data_efficiency.py``; this module contains the pure
helpers so unit tests can exercise them in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from kgproweight.data.silver_dataset import SilverDatasetReader, SilverTrajectory


def make_subset_file(
    silver_path: str | Path,
    n: int,
    seed: int,
    output_path: str | Path,
) -> str:
    reader = SilverDatasetReader(silver_path)
    subset: List[SilverTrajectory] = reader.subset(n, seed=seed)
    output_path = Path(output_path)
    SilverDatasetReader.write_jsonl(output_path, subset)
    return str(output_path)


def f1_curve_from_summary(summary: Dict[int, Dict[str, float]]) -> List[Dict[str, float]]:
    """Convert ``{N: {"f1": ..., "f1_std": ...}}`` to a list of points sorted by N."""
    out: List[Dict[str, float]] = []
    for n in sorted(summary.keys()):
        out.append({"N": n, **summary[n]})
    return out
