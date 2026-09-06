#!/usr/bin/env python
"""Generic paired-KG model scorer (no hardcoded model names).

Scores the legacy-vs-proof two-arm outputs of one or more models against the
frozen paired-KG protocol, then compares a candidate model against a baseline
(SFT) to separate "supply gain" from "training gain".

    python scripts/pilot/score_paired_kg_eval_models.py \
      --protocol <derived_protocol.json> \
      --baseline sft \
      --candidate ppo_automatic_proofkg \
      --model sft:<sft_predictions.jsonl> \
      --model ppo_automatic_proofkg:<ppo_predictions.jsonl> \
      --output <report.json>
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def _mcnemar_exact(gained: int, lost: int) -> float:
    discordant = gained + lost
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, v) for v in range(0, min(gained, lost) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def _bootstrap_ci(values: Sequence[float], seed: int = 20260829, draws: int = 10000) -> List[float]:
    values = list(values)
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(draws)
    )
    return [means[int(0.025 * draws)], means[int(0.975 * draws) - 1]]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pair_rows(model_rows: List[Mapping[str, Any]]) -> List[tuple]:
    by_key = {(str(r["qid"]), str(r["arm"])): r for r in model_rows}
    qids = [str(r["qid"]) for r in model_rows if r["arm"] == "legacy"]
    return [(by_key[(q, "legacy")], by_key[(q, "proof")]) for q in qids]


def paired_metrics(rows: List[Mapping[str, Any]], model_label: str) -> Dict[str, Any]:
    model_rows = [r for r in rows if r["model_label"] == model_label]
    pairs = _pair_rows(model_rows)

    def em(arm: int) -> float:
        return sum(float(p[arm]["em"]) for p in pairs) / max(1, len(pairs))

    def f1(arm: int) -> float:
        return sum(float(p[arm]["f1"]) for p in pairs) / max(1, len(pairs))

    def rate(key: str, arm: int) -> float:
        return sum(bool(p[arm].get(key)) for p in pairs) / max(1, len(pairs))

    gained = sum(float(l["em"]) < float(r["em"]) for l, r in pairs)
    lost = sum(float(l["em"]) > float(r["em"]) for l, r in pairs)
    em_diffs = [float(r["em"]) - float(l["em"]) for l, r in pairs]
    f1_diffs = [float(r["f1"]) - float(l["f1"]) for l, r in pairs]
    return {
        "n": len(pairs),
        "legacy_em": em(0),
        "proof_em": em(1),
        "delta_em": em(1) - em(0),
        "delta_em_bootstrap_95ci": _bootstrap_ci(em_diffs),
        "legacy_f1": f1(0),
        "proof_f1": f1(1),
        "delta_f1": f1(1) - f1(0),
        "delta_f1_bootstrap_95ci": _bootstrap_ci(f1_diffs, seed=20260830),
        "gained_correct": gained,
        "lost_correct": lost,
        "net_correct": gained - lost,
        "mcnemar_exact_p": _mcnemar_exact(gained, lost),
        "legacy_parse_rate": rate("well_formed", 0),
        "proof_parse_rate": rate("well_formed", 1),
        "legacy_known_citation_response_rate": rate("known_citation_response", 0),
        "proof_known_citation_response_rate": rate("known_citation_response", 1),
        "citation_utilization_gain": rate("known_citation_response", 1) - rate("known_citation_response", 0),
        "legacy_contract_error_rate": rate("citation_contract_error", 0),
        "proof_contract_error_rate": rate("citation_contract_error", 1),
        "contract_error_delta": rate("citation_contract_error", 1) - rate("citation_contract_error", 0),
    }


def _arm_em(rows: List[Mapping[str, Any]], model_label: str, arm: str, predicate=None) -> tuple:
    rows = [r for r in rows if r["model_label"] == model_label and r["arm"] == arm]
    if predicate is not None:
        rows = [r for r in rows if predicate(r)]
    if not rows:
        return 0, 0.0
    return len(rows), sum(float(r["em"]) for r in rows) / len(rows)


def stratified(rows: List[Mapping[str, Any]], baseline: str, candidate: str) -> Dict[str, Any]:
    strata = {
        "passages_visible": lambda r: r.get("gold_in_passages"),
        "passages_hidden": lambda r: not r.get("gold_in_passages"),
        "tail_visible": lambda r: r.get("gold_in_kg_tail"),
        "tail_hidden": lambda r: not r.get("gold_in_kg_tail"),
    }
    out = {}
    for name, pred in strata.items():
        bn, be = _arm_em(rows, baseline, "legacy", pred)
        bp, bp_em = _arm_em(rows, baseline, "proof", pred)
        cn, ce = _arm_em(rows, candidate, "legacy", pred)
        cp, cp_em = _arm_em(rows, candidate, "proof", pred)
        out[name] = {
            "n": bp,
            "baseline_legacy_em": be,
            "baseline_proof_em": bp_em,
            "baseline_delta": bp_em - be,
            "candidate_legacy_em": ce,
            "candidate_proof_em": cp_em,
            "candidate_delta": cp_em - ce,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--baseline", required=True, help="model label treated as the frozen SFT baseline")
    ap.add_argument("--candidate", required=True, help="model label of the candidate (PPO)")
    ap.add_argument("--model", action="append", required=True, metavar="LABEL:PATH",
                    help="repeatable: register one model's two-arm predictions")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    protocol = json.loads(Path(args.protocol).resolve().read_text(encoding="utf-8"))
    labels = []
    rows: List[Dict[str, Any]] = []
    input_hashes = {
        "legacy": protocol["inputs"]["arm_legacy"]["sha256"],
        "proof": protocol["inputs"]["arm_proof"]["sha256"],
    }
    for spec in args.model:
        label, _, path = spec.partition(":")
        if not path:
            raise SystemExit(f"--model must be LABEL:PATH, got {spec!r}")
        current = _read_jsonl(Path(path).resolve())
        if len(current) != 2 * int(protocol["n"]):
            raise SystemExit(f"{label} output row count mismatch")
        if any(r["model_label"] != label for r in current):
            raise SystemExit(f"{label} output contains wrong model label")
        if any(r["input_sha256"] != input_hashes[r["arm"]] for r in current):
            raise SystemExit(f"{label} output input identity mismatch")
        labels.append(label)
        rows.extend(current)
    if args.baseline not in labels or args.candidate not in labels:
        raise SystemExit(f"baseline/candidate not among registered labels: {labels}")

    per_model = {label: paired_metrics(rows, label) for label in labels}
    b, c = per_model[args.baseline], per_model[args.candidate]

    def _arm_em_lookup(metrics: Dict[str, Any], arm: str) -> float:
        return metrics[f"{arm}_em"]

    proof_diff = _arm_em_lookup(c, "proof") - _arm_em_lookup(b, "proof")
    legacy_diff = _arm_em_lookup(c, "legacy") - _arm_em_lookup(b, "legacy")
    did = (c["delta_em"] - b["delta_em"])

    # DID bootstrap: per-qid [[cand_proof - cand_legacy] - [base_proof - base_legacy]]
    b_pairs = _pair_rows([r for r in rows if r["model_label"] == args.baseline])
    c_pairs = _pair_rows([r for r in rows if r["model_label"] == args.candidate])
    b_by_q = {(str(l["qid"])): (l, r) for l, r in b_pairs}
    c_by_q = {(str(l["qid"])): (l, r) for l, r in c_pairs}
    did_diffs = []
    for q, (cl, cr) in c_by_q.items():
        bl, br = b_by_q[q]
        did_diffs.append(
            (float(cr["em"]) - float(cl["em"])) - (float(br["em"]) - float(bl["em"]))
        )

    report = {
        "protocol": args.protocol,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "per_model": per_model,
        "cross_model": {
            "candidate_minus_baseline_proof_em": proof_diff,
            "candidate_minus_baseline_legacy_em": legacy_diff,
            "difference_in_differences_em": did,
            "difference_in_differences_bootstrap_95ci": _bootstrap_ci(did_diffs),
        },
        "stratified": stratified(rows, args.baseline, args.candidate),
        "frozen_gates": protocol.get("decision_gates", {}),
        "baseline_frozen_utilization": {
            "legacy_em": b["legacy_em"],
            "proof_em": b["proof_em"],
            "utilization_gain": b["delta_em"],
        },
    }
    out = Path(args.output).resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
