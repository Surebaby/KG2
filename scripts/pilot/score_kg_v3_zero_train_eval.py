#!/usr/bin/env python
"""Score frozen old-KG arms against precision-first KG-v3 arms.

The old arms may be reused from the immutable KG-v2 run because the cohort,
adapter, passages, decoding configuration, and evaluator are identical.  Only
the four v3 arms require new model inference.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from kgproweight.utils.logging import dump_manifest
from scripts.pilot.score_zero_training_retrieval import _metrics, _paired


ARM_NAMES = (
    "hidden_sft_old", "hidden_sft_v3", "hidden_ppo_old", "hidden_ppo_v3",
    "hard_sft_old", "hard_sft_v3", "hard_ppo_old", "hard_ppo_v3",
)
PAIR_NAMES = {
    "hidden_sft_v3_vs_old": ("hidden_sft_old", "hidden_sft_v3"),
    "hidden_ppo_v3_vs_old": ("hidden_ppo_old", "hidden_ppo_v3"),
    "hard_sft_v3_vs_old": ("hard_sft_old", "hard_sft_v3"),
    "hard_ppo_v3_vs_old": ("hard_ppo_old", "hard_ppo_v3"),
}
REQUIRED_RETAINED_QIDS = ("train_11904", "train_14764")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate_v3_advancement_gate(
    pair_reports: Dict[str, Dict[str, Any]],
    arm_metrics: Dict[str, Dict[str, Any]],
    retained_improvements: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Apply the gate frozen before the KG-v3 model run.

    This small, v2-informed diagnostic is intentionally stricter on safety
    than on statistical significance.  Passing authorizes an independent
    val200 confirmation only, never training or a protocol replacement.
    """
    nets = {name: int(report["mcnemar"]["net"]) for name, report in pair_reports.items()}
    parse_deltas: Dict[str, float] = {}
    for new_name, old_name in (
        ("hidden_sft_v3", "hidden_sft_old"),
        ("hidden_ppo_v3", "hidden_ppo_old"),
        ("hard_sft_v3", "hard_sft_old"),
        ("hard_ppo_v3", "hard_ppo_old"),
    ):
        parse_deltas[new_name] = (
            float(arm_metrics[new_name]["parse_rate"])
            - float(arm_metrics[old_name]["parse_rate"])
        )
    required = set(REQUIRED_RETAINED_QIDS)
    retained_by_both = required.issubset(set(retained_improvements.get("sft", []))) and required.issubset(
        set(retained_improvements.get("ppo", []))
    )
    checks = {
        "no_pair_has_net_em_degradation": min(nets.values()) >= 0,
        "no_v3_arm_has_parse_rate_degradation": min(parse_deltas.values()) >= 0.0,
        "pooled_net_em_gain_at_least_two": sum(nets.values()) >= 2,
        "both_v2_verified_gold_kg_gains_retained_for_sft_and_ppo": retained_by_both,
    }
    return {
        "status": "PASS_TO_INDEPENDENT_VAL200" if all(checks.values()) else "FAIL_STOP_KG_V3_ROUTE",
        "checks": checks,
        "pair_net_correct": nets,
        "pooled_net_correct": sum(nets.values()),
        "parse_rate_delta": parse_deltas,
        "required_retained_qids": list(REQUIRED_RETAINED_QIDS),
        "retained_improvements": retained_improvements,
        "interpretation": "pass allows independent val200 confirmation only; not PPO training",
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
    arm_metrics = {name: _metrics(rows) for name, rows in arms.items()}

    structural_path = Path(args.hard_structural_report).resolve()
    structural = json.loads(structural_path.read_text(encoding="utf-8"))
    old_detail = {row["qid"]: row for row in structural["details"]["stored_kg"]}
    v3_labels = [label for label in structural["details"] if label != "stored_kg"]
    if len(v3_labels) != 1:
        raise SystemExit(f"expected exactly one v3 structural arm, got {v3_labels}")
    new_detail = {row["qid"]: row for row in structural["details"][v3_labels[0]]}
    new_gold_qids = [
        qid for qid in hard_qids
        if not old_detail[qid]["answer_hit"] and new_detail[qid]["answer_hit"]
    ]
    retained: Dict[str, List[str]] = {}
    new_gold_rows: Dict[str, Any] = {"qids": new_gold_qids, "models": {}}
    for model in ("sft", "ppo"):
        old_rows = {row["qid"]: row for row in arms[f"hard_{model}_old"]}
        new_rows = {row["qid"]: row for row in arms[f"hard_{model}_v3"]}
        improved = [
            qid for qid in new_gold_qids
            if float(old_rows[qid]["em"]) == 0 and float(new_rows[qid]["em"]) == 1
        ]
        degraded = [
            qid for qid in new_gold_qids
            if float(old_rows[qid]["em"]) == 1 and float(new_rows[qid]["em"]) == 0
        ]
        retained[model] = improved
        new_gold_rows["models"][model] = {"improved": improved, "degraded": degraded}

    gate = evaluate_v3_advancement_gate(pair_reports, arm_metrics, retained)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "status": "COMPLETE_EXPLORATORY",
        "scope": "fixed hidden33 val + hard25 train diagnostic; no training",
        "protocol": {
            "decode": "greedy",
            "max_new_tokens": 512,
            "prompt_passages": 15,
            "research_variable": "stored question KG versus precision-first additive KG v3",
            "same_passages_within_each_pair": True,
            "hard25_is_train_diagnostic_not_heldout": True,
            "old_arms_reused_from_frozen_kg_v2_run": True,
            "advancement_gate_preregistered_before_v3_model_run": True,
            "gate_was_informed_by_v2_diagnostic": True,
        },
        "sources": sources,
        "hard_structural_report": {"path": str(structural_path), "sha256": _sha256(structural_path)},
        "arms": arm_metrics,
        "paired": pair_reports,
        "new_gold_kg_questions": new_gold_rows,
        "advancement_gate": gate,
        "scientific_verdict": "INDEPENDENT_VAL200_REQUIRED" if gate["status"].startswith("PASS") else "STOP_KG_V3_ROUTE",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(
        run_dir,
        extra={
            "experiment_id": args.experiment_id,
            "phase": "kg_v3_zero_train_model_eval_scoring",
            "scope": report["scope"],
            "protocol": report["protocol"],
            "sources": sources,
            "advancement_gate": gate,
        },
    )
    print(json.dumps({"arms": arm_metrics, "paired": pair_reports, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
