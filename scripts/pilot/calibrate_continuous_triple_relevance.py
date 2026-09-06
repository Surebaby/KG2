#!/usr/bin/env python
"""Fit Platt ``b/T`` from reviewed relevance pairs and emit pilot labels.

This script never edits its input or production silver data.  It refuses to
fit when review labels or the held-out evaluation split are insufficient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from kgproweight.utils.logging import dump_manifest


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float | int]:
    probs = logits.sigmoid()
    return {
        "n": int(targets.numel()),
        "soft_bce": float(F.binary_cross_entropy_with_logits(logits, targets).item()),
        "mae": float((probs - targets).abs().mean().item()),
        "brier": float(((probs - targets) ** 2).mean().item()),
        "prediction_mean": float(probs.mean().item()),
        "target_mean": float(targets.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed_pairs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_fit", type=int, default=20)
    parser.add_argument("--min_eval", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    input_path = Path(args.reviewed_pairs).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    def gate_fail(reason: str, **accounting: Any) -> None:
        failure = {
            "status": "BLOCKED_BY_CALIBRATION_GATE",
            "reason": reason,
            "source": {"path": str(input_path), "md5": _md5(input_path)},
            "accounting": accounting,
        }
        failure_path = output_dir / "gate_failure.json"
        failure_path.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        dump_manifest(output_dir / "run", extra={
            "experiment": "continuous_triple_relevance_platt_calibration",
            "status": failure["status"],
            "failure": str(failure_path),
            **accounting,
        })
        raise SystemExit(reason)

    labelled: List[Dict[str, Any]] = []
    for row in rows:
        value = row.get("human_relevance")
        if value is None:
            continue
        value = float(value)
        if not 0.0 <= value <= 1.0:
            gate_fail(
                f"human_relevance outside [0,1] for {row.get('pair_id')}",
                all_pairs=len(rows),
                reviewed=len(labelled),
            )
        if row.get("review_split") not in {"fit", "eval"}:
            gate_fail(
                f"missing review_split for {row.get('pair_id')}",
                all_pairs=len(rows),
                reviewed=len(labelled),
            )
        labelled.append(row)

    fit = [row for row in labelled if row["review_split"] == "fit"]
    evaluate = [row for row in labelled if row["review_split"] == "eval"]
    if len(fit) < args.min_fit or len(evaluate) < args.min_eval:
        gate_fail(
            f"insufficient reviewed pairs: fit={len(fit)} (need {args.min_fit}), "
            f"eval={len(evaluate)} (need {args.min_eval})",
            all_pairs=len(rows),
            reviewed=len(labelled),
            fit=len(fit),
            eval=len(evaluate),
        )
    fit_targets_list = [float(row["human_relevance"]) for row in fit]
    if max(fit_targets_list) - min(fit_targets_list) < 0.25:
        gate_fail(
            "fit labels lack relevance variation (range < 0.25)",
            all_pairs=len(rows),
            reviewed=len(labelled),
            fit=len(fit),
            eval=len(evaluate),
        )

    fit_x = torch.tensor([float(row["raw_cross_encoder_logit"]) for row in fit])
    fit_y = torch.tensor(fit_targets_list)
    eval_x = torch.tensor([float(row["raw_cross_encoder_logit"]) for row in evaluate])
    eval_y = torch.tensor([float(row["human_relevance"]) for row in evaluate])
    b = torch.nn.Parameter(fit_x.median().detach().clone())
    log_t = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([b, log_t], lr=args.lr)
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        temperature = log_t.exp().clamp(0.05, 20.0)
        calibrated_logits = (fit_x - b) / temperature
        loss = F.binary_cross_entropy_with_logits(calibrated_logits, fit_y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_t.clamp_(math.log(0.05), math.log(20.0))

    bias = float(b.detach().item())
    temperature = float(log_t.detach().exp().item())
    calibrated_rows: List[Dict[str, Any]] = []
    step_values: Dict[tuple, List[float]] = defaultdict(list)
    for row in rows:
        raw = float(row["raw_cross_encoder_logit"])
        probability = torch.sigmoid(torch.tensor((raw - bias) / temperature)).item()
        out = {**row, "calibrated_relevance": float(probability), "calibration_status": "CALIBRATED_PILOT"}
        calibrated_rows.append(out)
        if row.get("pair_type") == "cited":
            step_values[(row.get("qid"), row.get("step_position"))].append(float(probability))

    calibrated_path = output_dir / "calibrated_pairs.jsonl"
    with calibrated_path.open("w", encoding="utf-8") as fh:
        for row in calibrated_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    step_path = output_dir / "continuous_step_labels.candidate.jsonl"
    with step_path.open("w", encoding="utf-8") as fh:
        for (qid, step_position), values in sorted(step_values.items(), key=lambda item: (str(item[0][0]), item[0][1])):
            fh.write(json.dumps({
                "qid": qid,
                "step_position": step_position,
                "r_kg_candidate": sum(values) / len(values),
                "n_verified_citations": len(values),
                "status": "PILOT_NOT_PRODUCTION",
            }, ensure_ascii=False) + "\n")

    fit_logits = (fit_x - bias) / temperature
    eval_logits = (eval_x - bias) / temperature
    report = {
        "status": "CALIBRATED_PILOT_NOT_PRODUCTION",
        "source": {"path": str(input_path), "md5": _md5(input_path)},
        "accounting": {"all_pairs": len(rows), "reviewed": len(labelled), "fit": len(fit), "eval": len(evaluate)},
        "parameters": {"b": bias, "T": temperature, "formula": "sigmoid((raw_logit-b)/T)"},
        "metrics": {"fit": _metrics(fit_logits, fit_y), "held_out_eval": _metrics(eval_logits, eval_y)},
        "outputs": {"calibrated_pairs": str(calibrated_path), "candidate_step_labels": str(step_path)},
        "constraints": [
            "Human relevance is semantic relevance, not dataset gold-answer supervision.",
            "Held-out metrics must be reviewed before any annotator integration.",
            "Contradiction and no-citation branches are not changed by this pilot.",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(output_dir / "run", extra={
        "experiment": "continuous_triple_relevance_platt_calibration",
        "report": str(report_path),
        "status": report["status"],
        "reviewed_pairs": len(labelled),
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
