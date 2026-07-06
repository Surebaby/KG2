#!/usr/bin/env python
"""GPT-4o LLM-as-Judge IHR (paper §5.5 indicator 3).

Reads a FlashRAG-style ``intermediate_data.json`` (pred/golden_answers/...)
and calls GPT-4o per parsed step. Outputs a JSON with per-item IHR plus
an optional Cohen κ against a small human-labelled subset.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from kgproweight.data.parsers import parse_steps
from kgproweight.reward.ihr_judge import IHRJudge, compute_cohen_kappa
from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import output_dir

configure_logging("INFO")
logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, help="Path to FlashRAG intermediate_data.json (or list of preds).")
    p.add_argument("--sample", type=int, default=200, help="Number of items to judge.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge_model", default="gpt-4o-2024-08-06")
    p.add_argument("--output", default=None)
    p.add_argument(
        "--human_csv",
        default=None,
        help="Optional CSV with columns: id,step_index,human_label (0/1).",
    )
    return p.parse_args()


def _load_predictions(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return list(data)


def _load_human(path: Path) -> Dict[str, Dict[int, int]]:
    import csv

    out: Dict[str, Dict[int, int]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out.setdefault(row["id"], {})[int(row["step_index"])] = int(row["human_label"])
    return out


def main():
    args = parse_args()
    random.seed(args.seed)

    preds = _load_predictions(Path(args.predictions))
    if args.sample and args.sample < len(preds):
        preds = random.sample(preds, args.sample)
    logger.info("Judging %d items with %s", len(preds), args.judge_model)

    judge = IHRJudge(model=args.judge_model)
    results: List[Dict] = []
    item_ihrs: List[float] = []
    human_labels = _load_human(Path(args.human_csv)) if args.human_csv else None

    human_vec: List[int] = []
    llm_vec: List[int] = []

    for item in preds:
        item_id = str(item.get("id") or "")
        question = item.get("question") or ""
        gold_list = item.get("golden_answers") or []
        gold = gold_list[0] if gold_list else ""
        pred = item.get("pred") or item.get("raw_output") or ""
        if not pred or not isinstance(pred, str):
            # pred may be nested inside item["output"]
            out = item.get("output") or {}
            pred = (out.get("raw_output") or out.get("pred") or "" if isinstance(out, dict) else "")
        steps = parse_steps(pred) if isinstance(pred, str) else []
        if not steps:
            continue
        per_step = judge.judge_trajectory(question, gold, [s.raw_text for s in steps])
        ihr = IHRJudge.aggregate_ihr(per_step)
        item_ihrs.append(ihr)
        records = [
            {
                "step_index": j.step_index,
                "hallucination": j.is_hallucination,
                "confidence": j.confidence,
                "reason": j.reason,
            }
            for j in per_step
        ]
        results.append({"id": item_id, "ihr": ihr, "steps": records})

        if human_labels and item_id in human_labels:
            human_map = human_labels[item_id]
            for j in per_step:
                if j.step_index in human_map:
                    human_vec.append(human_map[j.step_index])
                    llm_vec.append(int(j.is_hallucination))

    kappa: Optional[float] = None
    if human_vec:
        kappa = compute_cohen_kappa(human_vec, llm_vec)

    mean_ihr = sum(item_ihrs) / len(item_ihrs) if item_ihrs else 0.0
    out_payload = {
        "judge_model": args.judge_model,
        "n_items": len(results),
        "mean_ihr": mean_ihr,
        "kappa_vs_human": kappa,
        "items": results,
    }
    out_path = Path(args.output) if args.output else Path(output_dir()) / "rigor" / "ihr" / "ihr_judge.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Mean IHR = %.4f (n=%d). Saved → %s", mean_ihr, len(results), out_path)


if __name__ == "__main__":
    main()
