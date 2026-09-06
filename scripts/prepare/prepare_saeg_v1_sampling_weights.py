#!/usr/bin/env python
"""Freeze source/dataset-balanced weights for the SAEG-v1 candidate pool."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.utils.logging import dump_manifest


EXPERIMENT_ID = "SAEG-V1-TRAINING-SAMPLING-WEIGHTS-SEED42"
STATUS = "FROZEN_TRAIN_ONLY_SAMPLING_WEIGHTS_NOT_TRAINED"
DATASET_PRIOR = {
    "hotpotqa": 1 / 3,
    "2wikimultihopqa": 1 / 3,
    "musique": 1 / 3,
}
CONDITIONAL_SOURCE_PRIOR = {
    "hotpotqa": {"P_ONLY": 0.90, "N_REPLAY": 0.10},
    "2wikimultihopqa": {"P_ONLY": 0.20, "W_ONLY": 0.30, "P_W_FUSED": 0.40, "N_REPLAY": 0.10},
    "musique": {"P_ONLY": 0.90, "N_REPLAY": 0.10},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_sampling_probabilities(
    candidates: Sequence[Mapping[str, Any]],
    dataset_prior: Mapping[str, float] = DATASET_PRIOR,
    conditional_source_prior: Mapping[str, Mapping[str, float]] = CONDITIONAL_SOURCE_PRIOR,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[(str(row["dataset"]), str(row["source_mode"]))].append(row)
    if abs(sum(dataset_prior.values()) - 1.0) > 1e-9:
        raise ValueError("dataset prior must sum to one")
    results: list[dict[str, Any]] = []
    for dataset, dataset_probability in dataset_prior.items():
        source_prior = conditional_source_prior[dataset]
        if abs(sum(source_prior.values()) - 1.0) > 1e-9:
            raise ValueError(f"{dataset}: conditional source prior must sum to one")
        for mode, conditional_probability in source_prior.items():
            values = groups.get((dataset, mode), [])
            if not values:
                raise ValueError(f"missing candidate group: {dataset}/{mode}")
            probability = dataset_probability * conditional_probability / len(values)
            for row in values:
                results.append({
                    "candidate_id": str(row["candidate_id"]),
                    "dataset": dataset,
                    "source_mode": mode,
                    "sampling_probability": probability,
                    "sampling_weight": probability,
                })
    candidate_ids = {str(row["candidate_id"]) for row in candidates}
    result_ids = {row["candidate_id"] for row in results}
    if candidate_ids != result_ids:
        raise ValueError("sampling policy does not cover the candidate pool exactly")
    total = sum(row["sampling_probability"] for row in results)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"sampling probabilities sum to {total}, not one")
    return sorted(results, key=lambda row: row["candidate_id"])


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate_pool",
        type=Path,
        default=Path("outputs/audits/saeg_v1_training_candidate_pool_v1/candidate_pool.identity_only.jsonl"),
    )
    parser.add_argument(
        "--graph_assets",
        type=Path,
        default=Path("data/derived/saeg_v1_training_graph_assets_v1/question_graph_records.jsonl"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/audits/saeg_v1_training_sampling_weights_v1")
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite sampling protocol: {args.out}")
    for path in (args.candidate_pool, args.graph_assets):
        if not path.is_file():
            raise FileNotFoundError(path)
    candidates = read_jsonl(args.candidate_pool)
    assets = read_jsonl(args.graph_assets)
    asset_ids = {str(row["record_id"]) for row in assets}
    candidate_ids = {str(row["candidate_id"]) for row in candidates}
    if asset_ids != candidate_ids:
        raise ValueError("candidate/graph-asset identity join is not 1.0")
    rows = assign_sampling_probabilities(candidates)

    args.out.mkdir(parents=True, exist_ok=False)
    weights_path = args.out / "sampling_weights.jsonl"
    write_jsonl(weights_path, rows)
    realised_dataset = defaultdict(float)
    realised_source = defaultdict(float)
    realised_group = defaultdict(float)
    for row in rows:
        probability = float(row["sampling_probability"])
        realised_dataset[row["dataset"]] += probability
        realised_source[row["source_mode"]] += probability
        realised_group[f"{row['dataset']}::{row['source_mode']}"] += probability
    report = {
        "schema_version": "saeg-training-sampling-weights-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "sampler": {
            "type": "weighted_random_with_replacement",
            "seed": 42,
            "num_samples": "NOT_YET_FROZEN_REQUIRES_SFT_CONFIG_APPROVAL",
            "dataset_prior": DATASET_PRIOR,
            "conditional_source_prior": CONDITIONAL_SOURCE_PRIOR,
        },
        "counts": {
            "candidate_records": len(candidates),
            "weights": len(rows),
            "candidate_groups": dict(Counter(
                f"{row['dataset']}::{row['source_mode']}" for row in candidates
            )),
        },
        "realised_probability": {
            "by_dataset": dict(realised_dataset),
            "by_source_mode": dict(realised_source),
            "by_dataset_source": dict(realised_group),
            "total": sum(realised_dataset.values()),
        },
        "integrity": {
            "candidate_graph_identity_join_rate": 1.0,
            "weight_identity_unique": len(rows) == len({row["candidate_id"] for row in rows}),
            "all_candidates_covered": len(rows) == len(candidates),
            "probability_sum_one": abs(sum(realised_dataset.values()) - 1.0) < 1e-9,
        },
        "inputs": {
            "candidate_pool": {"path": str(args.candidate_pool), "sha256": sha256_file(args.candidate_pool)},
            "graph_assets": {"path": str(args.graph_assets), "sha256": sha256_file(args.graph_assets)},
        },
        "output": {"path": str(weights_path), "sha256": sha256_file(weights_path)},
        "scientific_boundary": (
            "This freezes relative sampling probabilities only. It does not freeze update count, "
            "SFT targets, checkpoint, optimizer, loss, reward, evaluation protocol, or authorize training."
        ),
    }
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(args.out, extra=report, status=STATUS)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
