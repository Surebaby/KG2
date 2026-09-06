#!/usr/bin/env python
"""Freeze one exact, deterministic SAEG-v1 continued-SFT epoch.

The source release stores one row per available evidence variant and a
per-row sampling probability.  Feeding that file directly to the ordinary
Hugging Face Trainer would ignore those probabilities and expose 2Wiki far
more often than HotpotQA or MuSiQue.  This script materialises the approved
dataset/source distribution as a versioned epoch file that the unchanged SFT
loader can consume.

Sampling is deterministic and coverage-first within each dataset/source
stratum: every candidate is visited once before a second visit is allowed.
The final epoch is shuffled with the frozen seed.  No target, answer, passage,
or evidence content is edited.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.utils.logging import dump_manifest


EXPERIMENT_ID = "SAEG-V1-SFT-BALANCED-EPOCH4860-SEED42"
STATUS = "FROZEN_TRAIN_ONLY_SFT_EPOCH_NOT_TRAINED"

# This is the already approved two-level distribution written as exact global
# fractions.  Keeping rational values avoids floating-point quota drift.
GROUP_PRIOR: dict[tuple[str, str], Fraction] = {
    ("hotpotqa", "P_ONLY"): Fraction(3, 10),
    ("hotpotqa", "N_REPLAY"): Fraction(1, 30),
    ("2wikimultihopqa", "P_ONLY"): Fraction(1, 15),
    ("2wikimultihopqa", "W_ONLY"): Fraction(1, 10),
    ("2wikimultihopqa", "P_W_FUSED"): Fraction(2, 15),
    ("2wikimultihopqa", "N_REPLAY"): Fraction(1, 30),
    ("musique", "P_ONLY"): Fraction(3, 10),
    ("musique", "N_REPLAY"): Fraction(1, 30),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_group_quotas(epoch_size: int) -> dict[tuple[str, str], int]:
    """Convert exact priors to integer quotas using deterministic remainders."""

    if epoch_size <= 0:
        raise ValueError("epoch_size must be positive")
    if sum(GROUP_PRIOR.values(), Fraction()) != 1:
        raise ValueError("GROUP_PRIOR must sum to one")
    raw = {key: prior * epoch_size for key, prior in GROUP_PRIOR.items()}
    quotas = {key: value.numerator // value.denominator for key, value in raw.items()}
    missing = epoch_size - sum(quotas.values())
    remainder_order = sorted(
        raw,
        key=lambda key: (
            -(raw[key] - quotas[key]),
            key[0],
            key[1],
        ),
    )
    for key in remainder_order[:missing]:
        quotas[key] += 1
    if sum(quotas.values()) != epoch_size:
        raise AssertionError("integer quota allocation failed")
    return quotas


def _group_seed(seed: int, key: tuple[str, str], cycle: int) -> int:
    value = f"{seed}::{key[0]}::{key[1]}::{cycle}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def coverage_first_sample(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    *,
    seed: int,
    key: tuple[str, str],
) -> list[dict[str, Any]]:
    """Draw uniformly by shuffled cycles, never repeating before coverage."""

    if count < 0:
        raise ValueError("sample count cannot be negative")
    if count and not rows:
        raise ValueError(f"empty candidate stratum for non-zero quota: {key}")
    stable = sorted((deepcopy(dict(row)) for row in rows), key=lambda row: str(row["qid"]))
    output: list[dict[str, Any]] = []
    cycle = 0
    while len(output) < count:
        values = [deepcopy(row) for row in stable]
        random.Random(_group_seed(seed, key, cycle)).shuffle(values)
        output.extend(values[: count - len(output)])
        cycle += 1
    return output


def build_epoch(
    rows: Sequence[Mapping[str, Any]],
    *,
    epoch_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {
        key: [] for key in GROUP_PRIOR
    }
    for row in rows:
        key = (str(row.get("dataset") or ""), str(row.get("evidence_mode") or ""))
        if key not in groups:
            raise ValueError(f"unexpected SAEG dataset/source stratum: {key}")
        groups[key].append(row)
    if any(not values for values in groups.values()):
        missing = [key for key, values in groups.items() if not values]
        raise ValueError(f"missing SAEG strata: {missing}")

    quotas = exact_group_quotas(epoch_size)
    sampled: list[dict[str, Any]] = []
    for key in sorted(groups):
        sampled.extend(
            coverage_first_sample(groups[key], quotas[key], seed=seed, key=key)
        )
    random.Random(seed).shuffle(sampled)
    for index, row in enumerate(sampled):
        metadata = deepcopy(dict(row.get("metadata") or {}))
        metadata.update({
            "sft_epoch_experiment_id": EXPERIMENT_ID,
            "sft_epoch_sample_index": index,
            "sft_epoch_seed": seed,
            "sft_epoch_size": epoch_size,
            "sft_epoch_source_qid": str(row["qid"]),
            "sft_epoch_sampling": "exact_stratum_quota_coverage_first_then_seeded_shuffle",
        })
        row["metadata"] = metadata
    return sampled, quotas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/silver_data/saeg_v1_train4860_family_disjoint_seed42_v2/silver_train.jsonl"
        ),
    )
    parser.add_argument(
        "--release_report",
        type=Path,
        default=Path(
            "data/silver_data/saeg_v1_train4860_family_disjoint_seed42_v2/report.json"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "data/silver_data/saeg_v1_sft_balanced_epoch4860_seed42_v1"
        ),
    )
    parser.add_argument("--epoch_size", type=int, default=4860)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite frozen SFT epoch: {args.out}")
    for path in (args.input, args.release_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    release = json.loads(args.release_report.read_text(encoding="utf-8"))
    if release.get("status") != "COMPLETE_TRAIN_ONLY_FAMILY_DISJOINT_NOT_TRAINED":
        raise ValueError("input is not the complete family-disjoint SAEG release")
    expected_sha = str(((release.get("outputs") or {}).get("silver_train") or {}).get("sha256") or "")
    actual_sha = sha256_file(args.input)
    if expected_sha != actual_sha:
        raise ValueError("input SHA256 does not match the frozen release report")

    source = read_jsonl(args.input)
    epoch, quotas = build_epoch(source, epoch_size=args.epoch_size, seed=args.seed)
    realised = Counter((row["dataset"], row["evidence_mode"]) for row in epoch)
    if dict(realised) != quotas:
        raise AssertionError("realised epoch does not match exact quotas")
    if any(not row.get("accepted") for row in epoch):
        raise ValueError("frozen epoch contains a rejected trajectory")

    args.out.mkdir(parents=True, exist_ok=False)
    data_path = args.out / "silver_train.jsonl"
    write_jsonl(data_path, epoch)
    source_occurrences = Counter(str(row["qid"]) for row in epoch)
    report = {
        "schema_version": "saeg-sft-balanced-epoch-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "sampler": {
            "type": "exact_stratum_quota_coverage_first_then_seeded_shuffle",
            "seed": args.seed,
            "epoch_size": args.epoch_size,
            "replacement": "only_after_full_stratum_coverage",
            "group_prior": {
                f"{dataset}::{mode}": f"{prior.numerator}/{prior.denominator}"
                for (dataset, mode), prior in GROUP_PRIOR.items()
            },
            "exact_group_quotas": {
                f"{dataset}::{mode}": count
                for (dataset, mode), count in sorted(quotas.items())
            },
        },
        "counts": {
            "source_variants": len(source),
            "epoch_rows": len(epoch),
            "unique_source_variants_exposed": len(source_occurrences),
            "repeated_epoch_rows": len(epoch) - len(source_occurrences),
            "max_occurrences_per_source_variant": max(source_occurrences.values()),
            "unique_dataset_source_qids": len({
                (row["dataset"], row.get("source_qid")) for row in epoch
            }),
        },
        "integrity": {
            "input_release_sha256_match": True,
            "exact_epoch_size": len(epoch) == args.epoch_size,
            "exact_group_quotas": dict(realised) == quotas,
            "all_rows_accepted": True,
            "targets_unchanged": True,
            "answers_unchanged": True,
            "evidence_unchanged": True,
        },
        "inputs": {
            "release": {"path": str(args.input), "sha256": actual_sha},
            "release_report": {
                "path": str(args.release_report),
                "sha256": sha256_file(args.release_report),
            },
        },
        "outputs": {
            "silver_train": {"path": str(data_path), "sha256": sha256_file(data_path)}
        },
        "scientific_boundary": (
            "This file freezes one train-only SFT exposure schedule. It changes no target, answer, "
            "passage, graph edge, reward, evaluation input, or Gold label and does not itself train a model."
        ),
    }
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(args.out, extra=report, status=STATUS)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
