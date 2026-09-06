#!/usr/bin/env python
"""Score frozen SFT vs candidate legacy/proof-KG paired predictions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.pilot.score_a1_fixed_context_kg import (
    _bootstrap_ci,
    _mcnemar_exact,
    paired_metrics,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validate_predictions(
    rows: list[Mapping[str, Any]],
    *,
    label: str,
    n: int,
    input_hashes: Mapping[str, str],
) -> None:
    if len(rows) != 2 * n:
        raise SystemExit(f"{label} predictions have {len(rows)} rows; expected {2*n}")
    if any(str(row.get("model_label")) != label for row in rows):
        raise SystemExit(f"{label} predictions contain a different model_label")
    if any(str(row.get("input_sha256")) != input_hashes[str(row.get("arm"))] for row in rows):
        raise SystemExit(f"{label} predictions contain a different frozen input hash")


def _by_qid(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["qid"]), str(row["arm"])): row for row in rows}


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _model_comparison(
    baseline: list[Mapping[str, Any]], candidate: list[Mapping[str, Any]]
) -> dict[str, Any]:
    base = _by_qid(baseline)
    cand = _by_qid(candidate)
    qids = [str(row["qid"]) for row in baseline if row["arm"] == "legacy"]
    if set(base) != set(cand):
        raise SystemExit("baseline and candidate prediction identities differ")
    proof_diffs: list[float] = []
    legacy_diffs: list[float] = []
    did: list[float] = []
    gained = lost = 0
    for qid in qids:
        bl, bp = base[(qid, "legacy")], base[(qid, "proof")]
        cl, cp = cand[(qid, "legacy")], cand[(qid, "proof")]
        proof_diff = float(cp["em"]) - float(bp["em"])
        legacy_diff = float(cl["em"]) - float(bl["em"])
        proof_diffs.append(proof_diff)
        legacy_diffs.append(legacy_diff)
        did.append((float(cp["em"]) - float(cl["em"])) - (float(bp["em"]) - float(bl["em"])))
        gained += proof_diff > 0
        lost += proof_diff < 0
    return {
        "n": len(qids),
        "candidate_minus_sft_proof_em": _mean(proof_diffs),
        "candidate_minus_sft_proof_em_bootstrap_95ci": _bootstrap_ci(proof_diffs, seed=20260901),
        "candidate_minus_sft_legacy_em": _mean(legacy_diffs),
        "candidate_minus_sft_legacy_em_bootstrap_95ci": _bootstrap_ci(legacy_diffs, seed=20260902),
        "utilization_difference_in_differences": _mean(did),
        "utilization_difference_in_differences_bootstrap_95ci": _bootstrap_ci(did, seed=20260903),
        "proof_arm_gained_correct": gained,
        "proof_arm_lost_correct": lost,
        "proof_arm_net_correct": gained - lost,
        "proof_arm_mcnemar_exact_p": _mcnemar_exact(gained, lost),
    }


def _subgroups(rows: list[Mapping[str, Any]], label: str) -> dict[str, Any]:
    legacy = [row for row in rows if row["arm"] == "legacy"]
    groups = {
        "passage_gold_visible": {str(row["qid"]) for row in legacy if row["gold_in_passages"]},
        "passage_gold_hidden": {str(row["qid"]) for row in legacy if not row["gold_in_passages"]},
    }
    proof = [row for row in rows if row["arm"] == "proof"]
    groups["proof_kg_gold_tail_visible"] = {
        str(row["qid"]) for row in proof if row["gold_in_kg_tail"]
    }
    groups["proof_kg_gold_tail_hidden"] = {
        str(row["qid"]) for row in proof if not row["gold_in_kg_tail"]
    }
    return {
        name: paired_metrics([row for row in rows if str(row["qid"]) in qids], label)
        for name, qids in groups.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis_protocol", required=True)
    parser.add_argument("--baseline_predictions", required=True)
    parser.add_argument("--candidate_predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    analysis_path = Path(args.analysis_protocol).resolve()
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    derived_path = Path(analysis["derived_protocol"]["path"]).resolve()
    if _sha256(derived_path) != analysis["derived_protocol"]["sha256"]:
        raise SystemExit("derived protocol changed after analysis-plan freeze")
    protocol = json.loads(derived_path.read_text(encoding="utf-8"))
    n = int(protocol["n"])
    input_hashes = {
        arm: protocol["inputs"][f"arm_{arm}"]["sha256"] for arm in ("legacy", "proof")
    }
    baseline_path = Path(args.baseline_predictions).resolve()
    candidate_path = Path(args.candidate_predictions).resolve()
    if _sha256(baseline_path) != analysis["baseline"]["predictions_sha256"]:
        raise SystemExit("baseline predictions changed after analysis-plan freeze")
    baseline = _read_jsonl(baseline_path)
    candidate = _read_jsonl(candidate_path)
    baseline_label = analysis["baseline"]["model_label"]
    candidate_label = analysis["candidate"]["model_label"]
    _validate_predictions(baseline, label=baseline_label, n=n, input_hashes=input_hashes)
    _validate_predictions(candidate, label=candidate_label, n=n, input_hashes=input_hashes)

    baseline_metrics = paired_metrics(baseline, baseline_label)
    candidate_metrics = paired_metrics(candidate, candidate_label)
    comparison = _model_comparison(baseline, candidate)
    gates = analysis["decision_gates"]
    supply_checks = {
        "candidate_parse_rate": min(
            candidate_metrics["legacy_parse_rate"], candidate_metrics["proof_parse_rate"]
        ) >= gates["each_candidate_arm_parse_rate_min"],
        "candidate_proof_minus_legacy_em": candidate_metrics["delta_em"]
        >= gates["candidate_proof_minus_legacy_em_min"],
        "candidate_net_correct": candidate_metrics["net_correct"]
        >= gates["candidate_proof_minus_legacy_net_correct_min"],
        "candidate_proof_citation_response": candidate_metrics["proof_known_citation_response_rate"]
        >= gates["candidate_proof_known_citation_response_rate_min"],
        "candidate_contract_error": candidate_metrics["contract_error_delta"]
        <= gates["candidate_citation_contract_error_rate_increase_max"],
    }
    added_checks = {
        "candidate_minus_sft_proof_em": comparison["candidate_minus_sft_proof_em"]
        >= gates["candidate_minus_sft_proof_em_min"],
        "candidate_minus_sft_legacy_em": comparison["candidate_minus_sft_legacy_em"]
        >= gates["candidate_minus_sft_legacy_em_min"],
        "utilization_difference_in_differences": comparison["utilization_difference_in_differences"]
        >= gates["utilization_difference_in_differences_min"],
    }
    ci_confirmed = (
        comparison["candidate_minus_sft_proof_em_bootstrap_95ci"][0]
        > gates["confirmed_added_utility_requires_bootstrap_ci_lower_gt"]
        and comparison["utilization_difference_in_differences_bootstrap_95ci"][0]
        > gates["confirmed_added_utility_requires_bootstrap_ci_lower_gt"]
    )
    supply_pass = all(supply_checks.values())
    point_added_pass = all(added_checks.values())
    status = (
        "PASS_CONFIRMED_PPO_ADDED_PROOFKG_UTILITY"
        if supply_pass and point_added_pass and ci_confirmed
        else "PASS_DIRECTIONAL_PPO_ADDED_PROOFKG_UTILITY"
        if supply_pass and point_added_pass
        else "PASS_SUPPLY_ONLY_NO_PPO_ADDED_UTILITY"
        if supply_pass
        else "FAIL_PPO_PROOFKG_UTILITY"
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": analysis["scope"],
        "scientific_boundary": analysis["scientific_boundary"],
        "analysis_protocol": {"path": str(analysis_path), "sha256": _sha256(analysis_path)},
        "inputs": {
            "baseline_predictions": {"path": str(baseline_path), "sha256": _sha256(baseline_path)},
            "candidate_predictions": {"path": str(candidate_path), "sha256": _sha256(candidate_path)},
        },
        "metrics": {
            "sft": baseline_metrics,
            "proofkg_ppo": candidate_metrics,
            "model_comparison": comparison,
            "proofkg_ppo_subgroups": _subgroups(candidate, candidate_label),
        },
        "gates": {
            "thresholds": gates,
            "supply_checks": supply_checks,
            "added_utility_checks": added_checks,
            "supply_pass": supply_pass,
            "point_added_utility_pass": point_added_pass,
            "ci_confirmed": ci_confirmed,
        },
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite score report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
