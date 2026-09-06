#!/usr/bin/env python
"""Score one QPEG-v4 development 2x2 checkpoint interaction.

Inputs are paired-evaluator prediction JSONLs for the frozen strong SFT and
one adapted checkpoint.  This script performs no generation and never reads
the confirmation cohort.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _index(rows: list[Mapping[str, Any]], expected_label: str) -> dict[tuple[str, str], Mapping[str, Any]]:
    if len(rows) != 300:
        raise ValueError(f"expected 300 predictions (150 qid x 2 arms), got {len(rows)}")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("model_label")) != expected_label:
            raise ValueError(f"unexpected model label: {row.get('model_label')} != {expected_label}")
        arm = str(row.get("arm"))
        if arm not in {"legacy", "proof"}:
            raise ValueError(f"unexpected arm: {arm}")
        key = (str(row.get("row_id")), arm)
        if not key[0] or key in result:
            raise ValueError(f"missing/duplicate prediction key: {key}")
        result[key] = row
    return result


def _mean(rows: list[Mapping[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / max(1, len(rows))


def score(
    strong_rows: list[Mapping[str, Any]],
    adapted_rows: list[Mapping[str, Any]],
    *,
    strong_label: str,
    adapted_label: str,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    strong = _index(strong_rows, strong_label)
    adapted = _index(adapted_rows, adapted_label)
    if set(strong) != set(adapted):
        raise ValueError("strong/adapted qid x arm grids differ")

    cells: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    row_effects: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row_id in sorted({key[0] for key in strong}):
        a = strong[(row_id, "legacy")]
        b = strong[(row_id, "proof")]
        d = adapted[(row_id, "legacy")]
        c = adapted[(row_id, "proof")]
        dataset = str(a["dataset"])
        if dataset not in DATASETS:
            raise ValueError(f"unexpected dataset: {dataset}")
        identity = ("dataset", "qid", "question", "gold_answers")
        if any(tuple(row.get(field) for field in identity) != tuple(a.get(field) for field in identity)
               for row in (b, c, d)):
            raise ValueError(f"cell identity mismatch: {row_id}")
        cells[f"{dataset}:A"].append(a)
        cells[f"{dataset}:B"].append(b)
        cells[f"{dataset}:C"].append(c)
        cells[f"{dataset}:D"].append(d)
        row_effects[dataset].append({
            "interaction_em": (float(c["em"]) - float(d["em"])) - (float(b["em"]) - float(a["em"])),
            "interaction_f1": (float(c["f1"]) - float(d["f1"])) - (float(b["f1"]) - float(a["f1"])),
            "c_minus_b_em": float(c["em"]) - float(b["em"]),
            "d_minus_a_em": float(d["em"]) - float(a["em"]),
        })

    per_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        if len(row_effects[dataset]) != 50:
            raise ValueError(f"{dataset}: expected 50 qids, got {len(row_effects[dataset])}")
        summaries = {}
        for arm in "ABCD":
            arm_rows = cells[f"{dataset}:{arm}"]
            summaries[arm] = {
                "em": _mean(arm_rows, "em"),
                "f1": _mean(arm_rows, "f1"),
                "parse_rate": sum(bool(row.get("well_formed")) for row in arm_rows) / len(arm_rows),
            }
        effects = row_effects[dataset]
        no_graph_net_loss = sum(float(x["d_minus_a_em"]) < 0 for x in effects) - sum(
            float(x["d_minus_a_em"]) > 0 for x in effects
        )
        per_dataset[dataset] = {
            "cells": summaries,
            "interaction_em": _mean(effects, "interaction_em"),
            "interaction_f1": _mean(effects, "interaction_f1"),
            "C_minus_B_em": _mean(effects, "c_minus_b_em"),
            "D_minus_A_em": _mean(effects, "d_minus_a_em"),
            "no_graph_net_correct_loss": no_graph_net_loss,
            "parse_drop_C_vs_B": summaries["B"]["parse_rate"] - summaries["C"]["parse_rate"],
            "parse_drop_D_vs_A": summaries["A"]["parse_rate"] - summaries["D"]["parse_rate"],
        }

    macro = {
        field: sum(float(per_dataset[d][field]) for d in DATASETS) / len(DATASETS)
        for field in ("interaction_em", "interaction_f1", "C_minus_B_em", "D_minus_A_em")
    }
    positive_datasets = sum(per_dataset[d]["interaction_em"] > 0 for d in DATASETS)
    checks = {
        "macro_interaction_em": macro["interaction_em"] > float(gates["macro_interaction_em_gt"]),
        "macro_interaction_f1": macro["interaction_f1"] > float(gates["macro_interaction_f1_gt"]),
        "positive_interaction_datasets": positive_datasets >= int(gates["positive_interaction_datasets_ge"]),
        "macro_C_minus_B_em": macro["C_minus_B_em"] > float(gates["macro_C_minus_B_em_gt"]),
        "macro_D_minus_A_em": macro["D_minus_A_em"] >= float(gates["macro_D_minus_A_em_ge"]),
        "no_graph_net_loss": all(
            per_dataset[d]["no_graph_net_correct_loss"] <= int(gates["max_no_graph_net_loss_per_dataset"])
            for d in DATASETS
        ),
        "parse_drop": all(
            max(per_dataset[d]["parse_drop_C_vs_B"], per_dataset[d]["parse_drop_D_vs_A"])
            <= float(gates["max_parse_rate_drop"])
            for d in DATASETS
        ),
    }
    return {
        "per_dataset": per_dataset,
        "macro": macro,
        "positive_interaction_datasets": positive_datasets,
        "gate_checks": checks,
        "all_development_gates_pass": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_protocol", required=True)
    parser.add_argument("--strong_predictions", required=True)
    parser.add_argument("--adapted_predictions", required=True)
    parser.add_argument("--strong_label", default="strong_sft")
    parser.add_argument("--adapted_label", required=True)
    parser.add_argument("--checkpoint_step", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.experiment_protocol).resolve()
    strong_path = Path(args.strong_predictions).resolve()
    adapted_path = Path(args.adapted_predictions).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite development score: {output}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "qpeg-v4-schema-adaptation-protocol-v1":
        raise SystemExit("unexpected experiment protocol")
    result = score(
        _read_jsonl(strong_path),
        _read_jsonl(adapted_path),
        strong_label=args.strong_label,
        adapted_label=args.adapted_label,
        gates=protocol["development_gates"],
    )
    report = {
        "schema_version": "qpeg-v4-development-interaction-score-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_DEVELOPMENT" if result["all_development_gates_pass"] else "FAIL_DEVELOPMENT",
        "checkpoint_step": args.checkpoint_step,
        "confirmation_opened": False,
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "inputs": {
            "strong_predictions": {"path": str(strong_path), "sha256": _sha256(strong_path)},
            "adapted_predictions": {"path": str(adapted_path), "sha256": _sha256(adapted_path)},
        },
        **result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
