#!/usr/bin/env python
"""IHR (LLM-as-Judge) for reasoning baselines.

The canonical ``run_ihr_judge.py`` parses the KG-ProWeight ``[Step N]`` schema
via :func:`kgproweight.data.parsers.parse_steps`. The reasoning baselines emit
*other* formats, so this script extracts their reasoning steps and reuses the
same :class:`IHRJudge` (deepseek-v4-pro) so the numbers are comparable.

Supported step sources:

* ``rearag``  — ``Thought N`` segments from ``output.messages`` (assistant turns).
* ``ircot``   — the per-iteration ``new_thought`` sentences (``trace`` baseline).
* ``r1_searcher`` — the first ``<think>…</think>`` reasoning block (``r1-searcher``).

The non-reasoning baselines (zero_shot / naive_rag / corag) and self_rag emit no
extractable reasoning steps, so IHR is not applicable to them.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

from kgproweight.reward.ihr_judge import IHRJudge, compute_cohen_kappa
from kgproweight.utils.logging import configure_logging, get_logger

configure_logging("INFO")
logger = get_logger(__name__)


def _load_items(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return list(data)


def _item_output(item: Dict) -> Dict:
    out = item.get("output") or {}
    return out if isinstance(out, dict) else {}


def _item_question(item: Dict) -> str:
    return str(item.get("question") or "")


def _item_gold(item: Dict) -> str:
    gold_list = item.get("golden_answers") or []
    return gold_list[0] if gold_list else ""


def extract_rearag_steps(item: Dict) -> List[str]:
    """Extract ``Thought N`` reasoning text from a ReaRAG trajectory."""
    out = _item_output(item)
    steps: List[str] = []
    for msg in out.get("messages") or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        # Each assistant turn is "Thought N: ... \nAction N: ...". The Thought
        # carries the factual reasoning the judge should score; the Action is a
        # search/finish call, not a claim.
        for m in re.finditer(
            r"Thought\s+\d+\s*:\s*(.+?)(?=\nAction\s+\d+\s*:|\Z)", content, re.DOTALL
        ):
            text = m.group(1).strip()
            if text:
                steps.append(text)
    return steps


def extract_ircot_steps(item: Dict) -> List[str]:
    """Extract per-iteration ``new_thought`` sentences from an IRCoT trace."""
    out = _item_output(item)
    steps: List[str] = []
    for key, val in out.items():
        if not key.startswith("intermediate_output_iter"):
            continue
        if not isinstance(val, dict):
            continue
        thought = val.get("new_thought")
        if isinstance(thought, str) and thought.strip():
            steps.append(thought.strip())
    return steps


def extract_r1_searcher_steps(item: Dict) -> List[str]:
    """Extract reasoning steps from an r1-searcher trajectory.

    The raw generation is ``<think>reasoning…</think> <answer>…</answer>``, but the
    model degenerates into repeating the same chain (and ``</assistant><answer>…``
    loops) until ``max_tokens``, so only the *first* reasoning block — everything
    before the first ``</think>`` / ``<answer>`` marker — is real; the rest is
    repetition and is discarded. The block is then split into sentence-level steps,
    mirroring IRCoT's one-claim-per-``new_thought`` granularity.
    """
    out = _item_output(item)
    raw = out.get("raw_pred") or ""
    if not raw:
        return []
    # Cut at the first answer/think-close marker; everything after is repetition.
    cut = len(raw)
    for marker in ("<answer>", "</think>"):
        idx = raw.lower().find(marker)
        if idx != -1:
            cut = min(cut, idx)
    reasoning = raw[:cut]
    reasoning = re.sub(r"</?think>", " ", reasoning, flags=re.IGNORECASE)
    steps: List[str] = []
    for line in reasoning.split("\n"):
        line = line.strip()
        if not line:
            continue
        for sent in re.split(r"(?<=[.!?])\s+", line):
            sent = sent.strip()
            # Drop empty / too-short fragments and pure tag remnants.
            if len(sent) >= 8 and not re.fullmatch(r"[<>\s]+", sent):
                steps.append(sent)
    return steps


def extract_steps(item: Dict, method: str) -> List[str]:
    if method == "rearag":
        return extract_rearag_steps(item)
    if method == "ircot":
        return extract_ircot_steps(item)
    if method == "r1_searcher":
        return extract_r1_searcher_steps(item)
    raise ValueError(f"Unknown step source: {method!r}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, help="Path to intermediate_data.json.")
    p.add_argument("--method", required=True, choices=["rearag", "ircot", "r1_searcher"], help="Step source.")
    p.add_argument("--sample", type=int, default=50, help="Number of items to judge.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge_model", default="deepseek-v4-pro")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    items = _load_items(Path(args.predictions))
    if args.sample and args.sample < len(items):
        items = random.sample(items, args.sample)

    judge = IHRJudge(model=args.judge_model)
    results: List[Dict] = []
    item_ihrs: List[float] = []

    for item in items:
        item_id = str(item.get("id") or "")
        question = _item_question(item)
        gold = _item_gold(item)
        steps = extract_steps(item, args.method)
        if not steps:
            continue
        per_step = judge.judge_trajectory(question, gold, steps)
        item_ihrs.append(IHRJudge.aggregate_ihr(per_step))
        results.append(
            {
                "id": item_id,
                "ihr": IHRJudge.aggregate_ihr(per_step),
                "steps": [
                    {
                        "step_index": j.step_index,
                        "hallucination": j.is_hallucination,
                        "confidence": j.confidence,
                        "reason": j.reason,
                    }
                    for j in per_step
                ],
            }
        )

    mean_ihr = sum(item_ihrs) / len(item_ihrs) if item_ihrs else 0.0
    out_payload = {
        "judge_model": args.judge_model,
        "method": args.method,
        "n_items": len(results),
        "mean_ihr": mean_ihr,
        "items": results,
    }
    out_path = Path(args.output) if args.output else Path(args.predictions).with_name(
        f"ihr_result_{args.method}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Mean IHR = %.4f (n=%d, method=%s). Saved → %s", mean_ihr, len(results), args.method, out_path)


if __name__ == "__main__":
    main()
