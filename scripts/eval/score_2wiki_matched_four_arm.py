#!/usr/bin/env python
"""Score the frozen 2Wiki SFT/PPO x legacy/ProofKG matched control.

The two SFT generations predate this four-arm completion.  This scorer reuses
their frozen raw generations, consumes the newly generated PPO two-arm rows,
and re-scores all four cells with the canonical pipeline answer extractor and
the same EM/F1 implementation.  No retrieval or model generation happens here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.data.parsers import extract_final_answer, parse_steps
from kgproweight.eval.metrics import compute_em, compute_f1
from kgproweight.eval.pred_processing import extract_kg_proweight_answer


SCORER_VERSION = "2wiki-matched-four-arm-canonical-scorer-1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _bootstrap_ci(
    values: Sequence[float], *, seed: int, draws: int
) -> list[float]:
    values = list(values)
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(draws)
    )
    return [means[int(0.025 * draws)], means[int(0.975 * draws) - 1]]


def _mcnemar_exact(gained: int, lost: int) -> float:
    discordant = gained + lost
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(gained, lost) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _validate_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise SystemExit(f"{label} SHA256 mismatch: expected {expected}, got {actual}")


def _common_without_kg(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "kg_subgraph"}


def _canonical_score(
    *, model: str, arm: str, input_row: Mapping[str, Any], generation: str
) -> dict[str, Any]:
    golds = [str(value) for value in input_row.get("gold_answers") or [] if str(value).strip()]
    prediction = extract_kg_proweight_answer(generation)
    kg = list(input_row.get("kg_subgraph") or [])
    steps = parse_steps(generation, known_kg=kg)
    strict_answer = (extract_final_answer(generation) or "").split("\n", 1)[0].strip()
    return {
        "model": model,
        "arm": arm,
        "qid": str(input_row["qid"]),
        "question": str(input_row["question"]),
        "gold_answers": golds,
        "prediction": prediction,
        "em": compute_em(prediction, golds) if prediction and golds else 0.0,
        "f1": compute_f1(prediction, golds) if prediction and golds else 0.0,
        "strict_final_answer_present": bool(strict_answer),
        "parsed_step_count": len(steps),
        "generation": generation,
    }


def _load_sft_rows(
    legacy_inputs: list[Mapping[str, Any]],
    proof_inputs: list[Mapping[str, Any]],
    legacy_predictions_path: Path,
    proof_intermediate_path: Path,
) -> list[dict[str, Any]]:
    legacy_predictions = _read_jsonl(legacy_predictions_path)
    proof_intermediate = json.loads(proof_intermediate_path.read_text(encoding="utf-8"))
    legacy_by_qid = {str(row["qid"]): row for row in legacy_predictions}
    proof_by_qid = {str(row["id"]): row for row in proof_intermediate}
    expected_qids = [str(row["qid"]) for row in legacy_inputs]
    if set(legacy_by_qid) != set(expected_qids) or set(proof_by_qid) != set(expected_qids):
        raise SystemExit("frozen SFT artifacts do not match the canonical qid set")

    rows: list[dict[str, Any]] = []
    for legacy_input, proof_input in zip(legacy_inputs, proof_inputs):
        qid = str(legacy_input["qid"])
        legacy = legacy_by_qid[qid]
        proof = proof_by_qid[qid]
        if str(legacy.get("question")) != str(legacy_input["question"]):
            raise SystemExit(f"SFT legacy question mismatch for {qid}")
        if str(proof.get("question")) != str(proof_input["question"]):
            raise SystemExit(f"SFT proof question mismatch for {qid}")
        rows.append(
            _canonical_score(
                model="sft",
                arm="legacy",
                input_row=legacy_input,
                generation=str(legacy["generation"]),
            )
        )
        rows.append(
            _canonical_score(
                model="sft",
                arm="proof",
                input_row=proof_input,
                generation=str(proof["output"]["raw_output"]),
            )
        )
    return rows


def _load_ppo_rows(
    legacy_inputs: list[Mapping[str, Any]],
    proof_inputs: list[Mapping[str, Any]],
    predictions_path: Path,
    *,
    expected_label: str,
    input_hashes: Mapping[str, str],
) -> list[dict[str, Any]]:
    predictions = _read_jsonl(predictions_path)
    n = len(legacy_inputs)
    if len(predictions) != 2 * n:
        raise SystemExit(f"PPO predictions have {len(predictions)} rows; expected {2*n}")
    by_key = {(str(row["qid"]), str(row["arm"])): row for row in predictions}
    expected_keys = {
        (str(row["qid"]), arm) for row in legacy_inputs for arm in ("legacy", "proof")
    }
    if set(by_key) != expected_keys:
        raise SystemExit("PPO predictions do not match the frozen qid x arm grid")

    rows: list[dict[str, Any]] = []
    input_by_arm = {"legacy": legacy_inputs, "proof": proof_inputs}
    for index in range(n):
        for arm in ("legacy", "proof"):
            input_row = input_by_arm[arm][index]
            pred = by_key[(str(input_row["qid"]), arm)]
            if str(pred.get("model_label")) != expected_label:
                raise SystemExit("PPO model label differs from the frozen protocol")
            if str(pred.get("input_sha256")) != input_hashes[arm]:
                raise SystemExit("PPO row input hash differs from the frozen protocol")
            rows.append(
                _canonical_score(
                    model="proofkg_ppo",
                    arm=arm,
                    input_row=input_row,
                    generation=str(pred["generation"]),
                )
            )
    return rows


def _by_cell(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    result: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        result.setdefault((str(row["model"]), str(row["arm"])), []).append(row)
    return result


def _cell_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "em": sum(float(row["em"]) for row in rows) / max(1, n),
        "f1": sum(float(row["f1"]) for row in rows) / max(1, n),
        "strict_final_answer_rate": sum(bool(row["strict_final_answer_present"]) for row in rows)
        / max(1, n),
        "parsed_step_rate": sum(int(row["parsed_step_count"]) > 0 for row in rows) / max(1, n),
    }


def _paired_effect(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    left_by_qid = {str(row["qid"]): row for row in left}
    right_by_qid = {str(row["qid"]): row for row in right}
    if set(left_by_qid) != set(right_by_qid):
        raise SystemExit("paired effect qid sets differ")
    qids = list(left_by_qid)
    em_diffs = [float(right_by_qid[q]["em"]) - float(left_by_qid[q]["em"]) for q in qids]
    f1_diffs = [float(right_by_qid[q]["f1"]) - float(left_by_qid[q]["f1"]) for q in qids]
    gained = sum(value > 0 for value in em_diffs)
    lost = sum(value < 0 for value in em_diffs)
    return {
        "n": len(qids),
        "em_delta": sum(em_diffs) / max(1, len(em_diffs)),
        "em_bootstrap_ci95": _bootstrap_ci(em_diffs, seed=seed, draws=draws),
        "f1_delta": sum(f1_diffs) / max(1, len(f1_diffs)),
        "f1_bootstrap_ci95": _bootstrap_ci(f1_diffs, seed=seed + 1, draws=draws),
        "em_gained": gained,
        "em_lost": lost,
        "em_tied": len(qids) - gained - lost,
        "mcnemar_exact_p": _mcnemar_exact(gained, lost),
    }


def _interaction(
    cells: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    *,
    metric: str,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    indexed = {
        key: {str(row["qid"]): float(row[metric]) for row in value}
        for key, value in cells.items()
    }
    qids = set(indexed[("sft", "legacy")])
    if any(set(value) != qids for value in indexed.values()):
        raise SystemExit("four-arm qid sets differ")
    diffs = [
        (indexed[("proofkg_ppo", "proof")][qid] - indexed[("proofkg_ppo", "legacy")][qid])
        - (indexed[("sft", "proof")][qid] - indexed[("sft", "legacy")][qid])
        for qid in sorted(qids)
    ]
    return {
        "delta": sum(diffs) / max(1, len(diffs)),
        "bootstrap_ci95": _bootstrap_ci(diffs, seed=seed, draws=draws),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--ppo_predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol).resolve()
    ppo_path = Path(args.ppo_predictions).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite result: {output}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not str(protocol.get("status", "")).startswith("FROZEN_BEFORE_PPO_INFERENCE"):
        raise SystemExit("protocol was not frozen before PPO inference")
    if _sha256(Path(__file__).resolve()) != protocol["implementation"]["four_arm_scorer_sha256"]:
        raise SystemExit("four-arm scorer differs from frozen protocol")

    legacy_input_path = Path(protocol["inputs"]["arm_legacy"]["path"]).resolve()
    proof_input_path = Path(protocol["inputs"]["arm_proof"]["path"]).resolve()
    sft_legacy_path = Path(protocol["existing_sft_artifacts"]["legacy_predictions"]["path"]).resolve()
    sft_proof_path = Path(protocol["existing_sft_artifacts"]["proof_intermediate"]["path"]).resolve()
    for label, path, expected in (
        ("legacy input", legacy_input_path, protocol["inputs"]["arm_legacy"]["sha256"]),
        ("proof input", proof_input_path, protocol["inputs"]["arm_proof"]["sha256"]),
        (
            "SFT legacy predictions",
            sft_legacy_path,
            protocol["existing_sft_artifacts"]["legacy_predictions"]["sha256"],
        ),
        (
            "SFT proof intermediate",
            sft_proof_path,
            protocol["existing_sft_artifacts"]["proof_intermediate"]["sha256"],
        ),
    ):
        _validate_hash(path, expected, label)

    legacy_inputs = _read_jsonl(legacy_input_path)
    proof_inputs = _read_jsonl(proof_input_path)
    n = int(protocol["n"])
    if len(legacy_inputs) != n or len(proof_inputs) != n:
        raise SystemExit("frozen input row count mismatch")
    if any(
        _common_without_kg(left) != _common_without_kg(right)
        for left, right in zip(legacy_inputs, proof_inputs)
    ):
        raise SystemExit("matched inputs differ outside kg_subgraph")

    rows = _load_sft_rows(legacy_inputs, proof_inputs, sft_legacy_path, sft_proof_path)
    rows.extend(
        _load_ppo_rows(
            legacy_inputs,
            proof_inputs,
            ppo_path,
            expected_label=protocol["models"]["proofkg_ppo"]["model_label"],
            input_hashes={
                "legacy": protocol["inputs"]["arm_legacy"]["sha256"],
                "proof": protocol["inputs"]["arm_proof"]["sha256"],
            },
        )
    )
    cells = _by_cell(rows)
    expected_cells = {
        ("sft", "legacy"),
        ("sft", "proof"),
        ("proofkg_ppo", "legacy"),
        ("proofkg_ppo", "proof"),
    }
    if set(cells) != expected_cells or any(len(value) != n for value in cells.values()):
        raise SystemExit("four-arm grid is incomplete")

    stats = protocol["statistics"]
    draws = int(stats["bootstrap_draws"])
    seed = int(stats["bootstrap_seed"])
    effects = {
        "supply_at_sft": _paired_effect(
            cells[("sft", "legacy")], cells[("sft", "proof")], seed=seed, draws=draws
        ),
        "supply_at_ppo": _paired_effect(
            cells[("proofkg_ppo", "legacy")],
            cells[("proofkg_ppo", "proof")],
            seed=seed + 10,
            draws=draws,
        ),
        "ppo_at_legacy": _paired_effect(
            cells[("sft", "legacy")],
            cells[("proofkg_ppo", "legacy")],
            seed=seed + 20,
            draws=draws,
        ),
        "ppo_at_proof": _paired_effect(
            cells[("sft", "proof")],
            cells[("proofkg_ppo", "proof")],
            seed=seed + 30,
            draws=draws,
        ),
    }
    interactions = {
        metric: _interaction(cells, metric=metric, seed=seed + 40 + index, draws=draws)
        for index, metric in enumerate(("em", "f1"))
    }
    ppo_proof = effects["ppo_at_proof"]
    if ppo_proof["em_delta"] > 0 and ppo_proof["em_bootstrap_ci95"][0] > 0:
        status = "PPO_ABOVE_SFT_ON_ALIGNED_PROOFKG_CONFIRMED_SINGLE_SEED"
    elif ppo_proof["em_delta"] > 0:
        status = "PPO_ABOVE_SFT_ON_ALIGNED_PROOFKG_DIRECTIONAL_ONLY"
    else:
        status = "NO_PPO_GAIN_OVER_SFT_ON_ALIGNED_PROOFKG"

    result = {
        "schema_version": "2wiki-matched-four-arm-result-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": protocol["experiment_id"],
        "status": status,
        "scope": protocol["scope"],
        "scientific_boundary": protocol["scientific_boundary"],
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "artifacts": {
            "ppo_predictions": {"path": str(ppo_path), "sha256": _sha256(ppo_path)},
        },
        "cells": {
            f"{model}__{arm}": _cell_summary(value)
            for (model, arm), value in sorted(cells.items())
        },
        "paired_effects": effects,
        "interactions": interactions,
        "claim_guard": {
            "proofkg_supply_utility": "identified by within-checkpoint legacy vs ProofKG contrasts",
            "ppo_added_utility": "identified by within-KG-arm PPO vs SFT contrasts",
            "kg_process_reward_causality": "NOT_IDENTIFIED_WITHOUT_MATCHED_PPO_OUTCOME_ONLY_CONTROL",
            "multi_seed_confirmation": "NOT_RUN",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    detail_path = output.with_name("four_arm_scored_rows.jsonl")
    if detail_path.exists():
        raise SystemExit(f"refusing to overwrite details: {detail_path}")
    with detail_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
