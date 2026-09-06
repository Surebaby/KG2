#!/usr/bin/env python
"""Score the frozen hidden33/hard25 old-KG versus KG-v2 model evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scripts.pilot.score_zero_training_retrieval import _metrics, _paired
from kgproweight.utils.logging import dump_manifest


ARM_NAMES = (
    "hidden_sft_old", "hidden_sft_v2", "hidden_ppo_old", "hidden_ppo_v2",
    "hard_sft_old", "hard_sft_v2", "hard_ppo_old", "hard_ppo_v2",
)
PAIR_NAMES = {
    "hidden_sft_v2_vs_old": ("hidden_sft_old", "hidden_sft_v2"),
    "hidden_ppo_v2_vs_old": ("hidden_ppo_old", "hidden_ppo_v2"),
    "hard_sft_v2_vs_old": ("hard_sft_old", "hard_sft_v2"),
    "hard_ppo_v2_vs_old": ("hard_ppo_old", "hard_ppo_v2"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate_advancement_gate(
    pair_reports: Dict[str, Dict[str, Any]],
    new_gold_improvements: int,
) -> Dict[str, Any]:
    """Apply the pre-run exploratory advancement gate.

    This gate is intentionally count-based because n=25/33 is too small for a
    formal efficacy claim.  Passing permits an independent confirmation only;
    it never permits silent protocol replacement or PPO training.
    """
    nets = {name: int(report["mcnemar"]["net"]) for name, report in pair_reports.items()}
    checks = {
        "no_pair_loses_two_or_more_questions": min(nets.values()) >= -1,
        "pooled_net_gain_at_least_two": sum(nets.values()) >= 2,
        "at_least_one_new_gold_kg_question_improves": new_gold_improvements >= 1,
    }
    return {
        "status": "PASS_EXPLORATORY_ADVANCE" if all(checks.values()) else "FAIL_STOP_KG_ROUTE",
        "checks": checks,
        "pair_net_correct": nets,
        "pooled_net_correct": sum(nets.values()),
        "new_gold_kg_improvements": new_gold_improvements,
        "interpretation": "pass allows independent confirmation only; not PPO training",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ARM_NAMES:
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--hidden_cohort", required=True)
    parser.add_argument("--hard_cohort", required=True)
    parser.add_argument("--hard_structural_report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    run_dir = Path(args.run_dir).resolve()
    for target in (output_path, run_dir):
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing path: {target}")

    hidden_qids = [str(row["qid"]) for row in _read(Path(args.hidden_cohort).resolve())]
    hard_qids = [str(row["qid"]) for row in _read(Path(args.hard_cohort).resolve())]
    if len(hidden_qids) != len(set(hidden_qids)) or len(hard_qids) != len(set(hard_qids)):
        raise SystemExit("cohort qids must be unique")

    arms: Dict[str, List[Dict[str, Any]]] = {}
    sources: Dict[str, Any] = {}
    for name in ARM_NAMES:
        path = Path(getattr(args, name)).resolve()
        rows = _read(path)
        expected = hidden_qids if name.startswith("hidden_") else hard_qids
        by_qid = {str(row.get("qid") or ""): row for row in rows}
        missing = [qid for qid in expected if qid not in by_qid]
        if missing:
            raise SystemExit(f"{name} missing qids: {missing}")
        arms[name] = [by_qid[qid] for qid in expected]
        sources[name] = {"path": str(path), "sha256": _sha256(path)}

    pair_reports = {
        name: _paired(arms[old_name], arms[new_name])
        for name, (old_name, new_name) in PAIR_NAMES.items()
    }
    structural = json.loads(Path(args.hard_structural_report).read_text(encoding="utf-8"))
    old_detail = {
        row["qid"]: row for row in structural["details"]["stored_kg"]
    }
    new_detail = {
        row["qid"]: row for row in structural["details"]["kg_v2_local_roots"]
    }
    new_gold_qids = [
        qid for qid in hard_qids
        if not old_detail[qid]["answer_hit"] and new_detail[qid]["answer_hit"]
    ]
    new_gold_rows: Dict[str, Any] = {"qids": new_gold_qids, "models": {}}
    total_new_gold_improvements = 0
    for model in ("sft", "ppo"):
        old_rows = {row["qid"]: row for row in arms[f"hard_{model}_old"]}
        new_rows = {row["qid"]: row for row in arms[f"hard_{model}_v2"]}
        improved = [
            qid for qid in new_gold_qids
            if float(old_rows[qid]["em"]) == 0 and float(new_rows[qid]["em"]) == 1
        ]
        degraded = [
            qid for qid in new_gold_qids
            if float(old_rows[qid]["em"]) == 1 and float(new_rows[qid]["em"]) == 0
        ]
        total_new_gold_improvements += len(improved)
        new_gold_rows["models"][model] = {"improved": improved, "degraded": degraded}

    gate = evaluate_advancement_gate(pair_reports, total_new_gold_improvements)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "status": "COMPLETE_EXPLORATORY",
        "scope": "fixed hidden33 val + hard25 train diagnostic; no training",
        "protocol": {
            "decode": "greedy",
            "max_new_tokens": 512,
            "prompt_passages": 15,
            "research_variable": "stored question KG versus passage-aware KG v2",
            "same_passages_within_each_pair": True,
            "hard25_is_train_diagnostic_not_heldout": True,
            "advancement_gate_preregistered_before_model_run": True,
        },
        "sources": sources,
        "arms": {name: _metrics(rows) for name, rows in arms.items()},
        "paired": pair_reports,
        "new_gold_kg_questions": new_gold_rows,
        "advancement_gate": gate,
        "scientific_verdict": "INDEPENDENT_CONFIRMATION_REQUIRED" if gate["status"].startswith("PASS") else "STOP_KG_ROUTE",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        run_dir,
        extra={
            "experiment_id": args.experiment_id,
            "phase": "kg_v2_zero_train_model_eval_scoring",
            "scope": report["scope"],
            "protocol": report["protocol"],
            "sources": sources,
            "advancement_gate": gate,
        },
    )
    print(json.dumps({"arms": report["arms"], "paired": pair_reports, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
