#!/usr/bin/env python
"""Score the two frozen model runs in the A1 paired KG-utilization pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _mcnemar_exact(gained: int, lost: int) -> float:
    discordant = gained + lost
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(0, min(gained, lost) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _bootstrap_ci(values: List[float], seed: int = 20260829, draws: int = 10000) -> List[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(draws)
    )
    return [means[int(0.025 * draws)], means[int(0.975 * draws) - 1]]


def paired_metrics(rows: List[Mapping[str, Any]], model_label: str) -> Dict[str, Any]:
    model_rows = [row for row in rows if row["model_label"] == model_label]
    by_key = {(str(row["qid"]), str(row["arm"])): row for row in model_rows}
    qids = [str(row["qid"]) for row in model_rows if row["arm"] == "legacy"]
    pairs = [(by_key[(qid, "legacy")], by_key[(qid, "proof")]) for qid in qids]
    gained = sum(left["em"] < right["em"] for left, right in pairs)
    lost = sum(left["em"] > right["em"] for left, right in pairs)
    em_diffs = [float(right["em"]) - float(left["em"]) for left, right in pairs]
    f1_diffs = [float(right["f1"]) - float(left["f1"]) for left, right in pairs]
    changed = sum(str(left["prediction"]).strip() != str(right["prediction"]).strip() for left, right in pairs)
    def rate(key: str, arm_index: int) -> float:
        return sum(bool(pair[arm_index][key]) for pair in pairs) / max(1, len(pairs))
    legacy_em = sum(float(left["em"]) for left, _ in pairs) / max(1, len(pairs))
    proof_em = sum(float(right["em"]) for _, right in pairs) / max(1, len(pairs))
    legacy_f1 = sum(float(left["f1"]) for left, _ in pairs) / max(1, len(pairs))
    proof_f1 = sum(float(right["f1"]) for _, right in pairs) / max(1, len(pairs))
    return {
        "n": len(pairs),
        "legacy_em": legacy_em,
        "proof_em": proof_em,
        "delta_em": proof_em - legacy_em,
        "delta_em_bootstrap_95ci": _bootstrap_ci(em_diffs),
        "legacy_f1": legacy_f1,
        "proof_f1": proof_f1,
        "delta_f1": proof_f1 - legacy_f1,
        "delta_f1_bootstrap_95ci": _bootstrap_ci(f1_diffs, seed=20260830),
        "gained_correct": gained,
        "lost_correct": lost,
        "net_correct": gained - lost,
        "mcnemar_exact_p": _mcnemar_exact(gained, lost),
        "prediction_changed": changed,
        "legacy_parse_rate": rate("well_formed", 0),
        "proof_parse_rate": rate("well_formed", 1),
        "parse_count_delta": sum(bool(right["well_formed"]) - bool(left["well_formed"]) for left, right in pairs),
        "legacy_known_citation_response_rate": rate("known_citation_response", 0),
        "proof_known_citation_response_rate": rate("known_citation_response", 1),
        "citation_utilization_gain": rate("known_citation_response", 1) - rate("known_citation_response", 0),
        "legacy_contract_error_rate": rate("citation_contract_error", 0),
        "proof_contract_error_rate": rate("citation_contract_error", 1),
        "contract_error_delta": rate("citation_contract_error", 1) - rate("citation_contract_error", 0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--sft_predictions", required=True)
    parser.add_argument("--hybrid_predictions", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = {
        "sft": Path(args.sft_predictions).resolve(),
        "hybrid_ppo": Path(args.hybrid_predictions).resolve(),
    }
    rows: List[Dict[str, Any]] = []
    input_hashes = {
        "legacy": protocol["inputs"]["arm_legacy"]["sha256"],
        "proof": protocol["inputs"]["arm_proof"]["sha256"],
    }
    for label, path in paths.items():
        current = _read_jsonl(path)
        if len(current) != 2 * int(protocol["n"]):
            raise SystemExit(f"{label} output row count mismatch")
        if any(row["model_label"] != label for row in current):
            raise SystemExit(f"{label} output contains wrong model label")
        if any(row["input_sha256"] != input_hashes[row["arm"]] for row in current):
            raise SystemExit(f"{label} output input identity mismatch")
        rows.extend(current)
    metrics = {label: paired_metrics(rows, label) for label in paths}
    gates = protocol["decision_gates"]
    checks = {
        "all_arms_complete": all(value["n"] == int(protocol["n"]) for value in metrics.values()),
        "parse_rate": all(
            min(value["legacy_parse_rate"], value["proof_parse_rate"])
            >= gates["each_arm_parse_rate_min"]
            for value in metrics.values()
        ),
        "parse_no_material_regression": all(
            value["parse_count_delta"] >= -gates["max_parse_response_loss"]
            for value in metrics.values()
        ),
        "em_no_material_regression": all(
            value["net_correct"] >= -gates["max_net_correct_loss_per_model"]
            for value in metrics.values()
        ),
        "citation_contract_no_material_regression": all(
            value["contract_error_delta"] <= gates["max_contract_error_rate_increase"]
            for value in metrics.values()
        ),
        "model_utility_signal": any(
            value["net_correct"] >= gates["utility_net_correct_gain_in_any_model"]
            or (
                value["proof_known_citation_response_rate"]
                >= gates["proof_known_citation_response_rate_min"]
                and value["citation_utilization_gain"]
                >= gates["citation_utilization_gain_in_any_model"]
            )
            for value in metrics.values()
        ),
    }
    all_pass = all(checks.values())
    status = "ADVANCE_TO_UNSEEN_CONFIRMATION_DESIGN" if all_pass else "STOP_NO_MODEL_UTILITY_SIGNAL"
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": protocol["scope"],
        "scientific_boundary": protocol["scientific_boundary"],
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "inputs": {
            label: {"path": str(path), "sha256": _sha256(path)} for label, path in paths.items()
        },
        "metrics": metrics,
        "gates": {"thresholds": gates, "checks": checks, "all_pass": all_pass},
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
