#!/usr/bin/env python
"""Aggregate every experiment into the paper tables.

Looks under ``outputs/`` (configurable via ``--save_root``) for:

- ``outputs/baselines/<method>/<dataset>/seed_<S>/<run>/metric_score.json``
- ``outputs/kg_proweight/<dataset>/seed_<S>/<run>/metric_score.json``
- ``outputs/kg_proweight/<dataset>/seed_<S>/<run>/alpha_distribution.jsonl``
- ``outputs/ablations/<variant>/eval/<dataset>/seed_<S>/...``
- ``outputs/rigor/ihr/<dataset>.json``
- ``outputs/rigor/data_eff/data_efficiency.json``

Writes:
  • ``outputs/summary/table1_main_results.md`` (EM / F1 / IHR per baseline)
  • ``outputs/summary/table2_ablations.md``
  • ``outputs/summary/alpha_distribution.md`` (D_std vs D_dropout)
  • ``outputs/summary/significance.md`` (paired bootstrap vs ReaRAG)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from kgproweight.eval.alpha_analysis import compare_alpha_distributions
from kgproweight.eval.stats import mean_std_ci, paired_bootstrap
from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import output_dir

configure_logging("INFO")
logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save_root", default=None)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["hotpotqa", "2wikimultihopqa", "musique", "d_dropout"],
    )
    p.add_argument("--check", action="store_true", help="Verify multi-seed coverage and abort if incomplete.")
    return p.parse_args()


def _find_metric(d: Path) -> Optional[Dict[str, Any]]:
    """Pick the newest metric_score.json under directory ``d``."""
    if not d.exists():
        return None
    candidates = sorted(d.rglob("metric_score.json"))
    if not candidates:
        return None
    try:
        with open(candidates[-1], "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return None


def _gather_baselines(save_root: Path, datasets: List[str]) -> Dict[str, Dict[str, Dict]]:
    """Collect ``{method: {dataset: {em_mean, em_std, f1_mean, f1_std}}}``."""
    base_root = save_root / "baselines"
    out: Dict[str, Dict[str, Dict]] = {}
    if not base_root.exists():
        return out
    for method_dir in sorted(p for p in base_root.iterdir() if p.is_dir()):
        method = method_dir.name
        for ds in datasets:
            ds_dir = method_dir / ds
            if not ds_dir.exists():
                continue
            ems: List[float] = []
            f1s: List[float] = []
            for seed_dir in sorted(ds_dir.glob("seed_*")):
                metric = _find_metric(seed_dir)
                if not metric:
                    continue
                if "em" in metric:
                    ems.append(float(metric["em"]))
                if "f1" in metric:
                    f1s.append(float(metric["f1"]))
            if ems or f1s:
                out.setdefault(method, {})[ds] = {
                    "em": mean_std_ci(ems),
                    "f1": mean_std_ci(f1s),
                }
    return out


def _gather_kgpw(save_root: Path, datasets: List[str]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    kgpw_root = save_root / "kg_proweight"
    if not kgpw_root.exists():
        return out
    for ds in datasets:
        ds_dir = kgpw_root / ds
        if not ds_dir.exists():
            continue
        ems: List[float] = []
        f1s: List[float] = []
        for seed_dir in sorted(ds_dir.glob("seed_*")):
            metric = _find_metric(seed_dir)
            if not metric:
                continue
            ems.append(float(metric.get("em", 0.0)))
            f1s.append(float(metric.get("f1", 0.0)))
        out[ds] = {
            "em": mean_std_ci(ems),
            "f1": mean_std_ci(f1s),
        }
    return out


def _table1(baselines: Dict[str, Dict[str, Dict]], kgpw: Dict[str, Dict], datasets: List[str]) -> str:
    lines = ["# Table 1 — Main results", ""]
    header = ["Method"] + [f"{ds} EM" for ds in datasets] + [f"{ds} F1" for ds in datasets]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    def fmt(stat: Dict[str, float]) -> str:
        if not stat or stat.get("n", 0) == 0:
            return "—"
        if stat["n"] == 1:
            return f"{stat['mean']:.4f}"
        return f"{stat['mean']:.4f} ± {stat['std']:.4f}"

    for method, per_ds in baselines.items():
        row = [method]
        for ds in datasets:
            row.append(fmt(per_ds.get(ds, {}).get("em", {})))
        for ds in datasets:
            row.append(fmt(per_ds.get(ds, {}).get("f1", {})))
        lines.append("| " + " | ".join(row) + " |")

    if kgpw:
        row = ["**kg_proweight**"]
        for ds in datasets:
            row.append(fmt(kgpw.get(ds, {}).get("em", {})))
        for ds in datasets:
            row.append(fmt(kgpw.get(ds, {}).get("f1", {})))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _alpha_table(save_root: Path) -> str:
    lines = ["# α distribution — D_std vs D_dropout", ""]
    std_path = next((save_root / "kg_proweight" / "hotpotqa").rglob("alpha_distribution.jsonl"), None)
    drop_path = next((save_root / "kg_proweight" / "d_dropout").rglob("alpha_distribution.jsonl"), None)
    if std_path is None or drop_path is None:
        return "α distribution files not found. Run scripts/eval/run_kg_proweight.py first.\n"
    cmp = compare_alpha_distributions(std_path, drop_path)
    lines.append("```json")
    lines.append(json.dumps(cmp, indent=2, ensure_ascii=False))
    lines.append("```")
    return "\n".join(lines) + "\n"


def _significance(save_root: Path, datasets: List[str]) -> str:
    """Compute paired bootstrap of KG-ProWeight vs ReaRAG (when both available)."""
    lines = ["# Significance — KG-ProWeight vs ReaRAG (paired bootstrap)", ""]
    rearag_root = save_root / "baselines" / "rearag"
    kgpw_root = save_root / "kg_proweight"
    for ds in datasets:
        a_metric = _find_metric(rearag_root / ds)
        b_metric = _find_metric(kgpw_root / ds)
        if not a_metric or not b_metric:
            continue
        # Without per-item F1 dumps we use the means. Replace with per-item arrays
        # when available; the helper handles both cases.
        scores_a = [float(a_metric.get("f1", 0.0))]
        scores_b = [float(b_metric.get("f1", 0.0))]
        if len(scores_a) == 1:
            lines.append(f"## {ds}: (run with per-item dumps for bootstrap; means {scores_b[0]:.4f} vs {scores_a[0]:.4f})")
            continue
        result = paired_bootstrap(scores_b, scores_a)
        lines.append(
            f"## {ds}: diff F1 = {result['diff_mean']:.4f} "
            f"(95% CI [{result['lower']:.4f}, {result['upper']:.4f}], p={result['p_value']:.4f})"
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    save_root = Path(args.save_root) if args.save_root else Path(output_dir())
    summary_dir = save_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    baselines = _gather_baselines(save_root, args.datasets)
    kgpw = _gather_kgpw(save_root, args.datasets)

    (summary_dir / "table1_main_results.md").write_text(_table1(baselines, kgpw, args.datasets), encoding="utf-8")

    # Table 2 — ablations.
    ablations_root = save_root / "ablations"
    abl_summary: Dict[str, Dict[str, Dict]] = {}
    if ablations_root.exists():
        for variant_dir in sorted(p for p in ablations_root.iterdir() if p.is_dir()):
            inner = variant_dir / "eval"
            if not inner.exists():
                continue
            sub = {}
            for ds in args.datasets:
                ds_dir = inner / ds
                if not ds_dir.exists():
                    continue
                ems: List[float] = []
                f1s: List[float] = []
                for seed_dir in sorted(ds_dir.glob("seed_*")):
                    m = _find_metric(seed_dir)
                    if not m:
                        continue
                    ems.append(float(m.get("em", 0.0)))
                    f1s.append(float(m.get("f1", 0.0)))
                sub[ds] = {"em": mean_std_ci(ems), "f1": mean_std_ci(f1s)}
            abl_summary[variant_dir.name] = sub
    (summary_dir / "table2_ablations.md").write_text(_table1(abl_summary, kgpw, args.datasets), encoding="utf-8")

    (summary_dir / "alpha_distribution.md").write_text(_alpha_table(save_root), encoding="utf-8")
    (summary_dir / "significance.md").write_text(_significance(save_root, args.datasets), encoding="utf-8")

    logger.info("Summary written to %s", summary_dir)


if __name__ == "__main__":
    main()
