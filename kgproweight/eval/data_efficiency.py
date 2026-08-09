"""Data-efficiency rigour utilities.

Given a silver dataset, produce reproducible random subsets at multiple
sizes and report the trained model's F1. The CLI lives in
``scripts/eval/run_data_efficiency.py``; this module contains the pure
helpers so unit tests can exercise them in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List

from kgproweight.data.silver_dataset import SilverDatasetReader, SilverTrajectory

if TYPE_CHECKING:
    from kgproweight.data.silver_split import SplitSpec


def make_subset_file(
    silver_path: str | Path,
    n: int,
    seed: int,
    output_path: str | Path,
    split: str | None = None,
    split_spec: "SplitSpec | None" = None,
) -> str:
    """Write a reproducible ``n``-trajectory subset for a data-efficiency point.

    ``split`` should be set to the same fold the real runs use. The subsets this
    writes are TRAINING data for the scan's models, so drawing from the whole
    file would put val/test trajectories into training at every point on the
    curve — making the curve's own held-out evaluation invalid. Left at ``None``
    for backward compatibility with the pre-split scans.
    """
    reader = SilverDatasetReader(silver_path, split=split, split_spec=split_spec)
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
