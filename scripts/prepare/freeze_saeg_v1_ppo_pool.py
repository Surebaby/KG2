#!/usr/bin/env python
"""Freeze exact SAEG-v1 PPO identities only after the utility gate passes.

The output remains answer-free and contains no graph materialisation.  It
freezes a balanced 5,400-prompt main schedule plus fresh SFT/eval/legacy-reward
family-disjoint reward development and confirmation cohorts (100 each).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_sha256
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
SEED = 42
EXPERIMENT_ID = "SAEG-V1-PPO-POOL-N5400-K4-SEED42"
MAIN_PER_DATASET = 1800
REWARD_DEV = {"hotpotqa": 34, "2wikimultihopqa": 33, "musique": 33}
REWARD_CONFIRMATION = {"hotpotqa": 33, "2wikimultihopqa": 34, "musique": 33}


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


def rank(label: str, dataset: str, qid: str) -> str:
    return hashlib.sha256(f"{SEED}\0{label}\0{dataset}\0{qid}".encode()).hexdigest()


def source_qid(row: Mapping[str, Any]) -> str:
    return str(row.get("source_qid") or row.get("qid") or row.get("id") or "")


def select_family_unique(
    rows: Sequence[Mapping[str, Any]],
    n: int,
    *,
    label: str,
    blocked_families: set[str],
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda x: rank(label, str(x["dataset"]), str(x["qid"]))):
        family = str(row["family_sha256"])
        if family in blocked_families:
            continue
        chosen.append(dict(row))
        blocked_families.add(family)
        if len(chosen) == n:
            return chosen
    raise ValueError(f"only {len(chosen)}/{n} family-unique rows available for {label}")


def interleave_datasets(rows_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    lengths = {dataset: len(rows_by_dataset[dataset]) for dataset in DATASETS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"dataset schedule is not balanced: {lengths}")
    output: list[dict[str, Any]] = []
    for index in range(next(iter(lengths.values()))):
        for dataset in DATASETS:
            row = dict(rows_by_dataset[dataset][index])
            row["schedule_index"] = len(output)
            output.append(row)
    return output


def question_only(dataset: str, qid: str, question: str, *, role: str, route: str) -> dict[str, Any]:
    return {
        "schema_version": "saeg-ppo-question-only-v1",
        "question_key": f"{dataset}::{qid}",
        "dataset": dataset,
        "qid": qid,
        "question": question,
        "question_sha256": question_sha256(question),
        "family_sha256": family_sha256(question),
        "role": role,
        "source_route": route,
        "gold_access": False,
        "materialized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--utility_report", type=Path,
        default=Path("outputs/validation/saeg_v1_development_strong_sft_npdf_v2_attempt2/report.json"),
    )
    parser.add_argument(
        "--sft_release", type=Path,
        default=Path("data/silver_data/saeg_v1_train4860_family_disjoint_seed42_v2/silver_train.jsonl"),
    )
    parser.add_argument(
        "--eval_dir", type=Path,
        default=Path("data/derived/saeg_v1_evaluation_inputs_seed42_v1"),
    )
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/audits/saeg_v1_ppo_pool_n5400_k4_seed42_v1"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite PPO identity freeze: {args.out}")

    eval_paths = [
        args.eval_dir / "development.answer_free.jsonl",
        args.eval_dir / "confirmation.answer_free.jsonl",
        args.eval_dir / "canonical_reporting.answer_free.jsonl",
    ]
    consumed_paths = [
        Path("outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_train.question_only.jsonl"),
        Path("outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_dev.question_only.jsonl"),
        Path("outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_confirmation.question_only.jsonl"),
        Path("outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_reserve.question_only.jsonl"),
        Path("outputs/audits/2wiki_train_only_rankability_n150_v1/cohort.question_only.jsonl"),
        Path("outputs/audits/2wiki_train_only_rankability_confirmation_n100_v1/cohort.question_only.jsonl"),
    ]
    raw_paths = {dataset: args.data_root / dataset / "train.jsonl" for dataset in DATASETS}
    required = [args.utility_report, args.sft_release, *eval_paths, *consumed_paths, *raw_paths.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing freeze inputs: {missing}")
    utility = json.loads(args.utility_report.read_text(encoding="utf-8"))
    if utility.get("status") != "PASS_ZERO_TRAINING_UTILITY":
        raise PermissionError(f"PPO identity freeze blocked by utility status: {utility.get('status')}")

    heldout_families: set[str] = set()
    heldout_qids: set[tuple[str, str]] = set()
    for path in eval_paths:
        for row in read_jsonl(path):
            dataset, row_qid = str(row["dataset"]), source_qid(row)
            heldout_qids.add((dataset, row_qid))
            heldout_families.add(family_sha256(str(row["question"])))
    consumed_families: set[str] = set()
    consumed_qids: set[tuple[str, str]] = set()
    for path in consumed_paths:
        for row in read_jsonl(path):
            dataset, row_qid = str(row.get("dataset") or "2wikimultihopqa"), source_qid(row)
            consumed_qids.add((dataset, row_qid))
            consumed_families.add(family_sha256(str(row["question"])))

    existing: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(args.sft_release):
        key = (str(row["dataset"]), source_qid(row))
        current = existing.setdefault(key, {
            "dataset": key[0], "qid": key[1], "question": str(row["question"]), "modes": set(),
        })
        current["modes"].add(str(row["evidence_mode"]))
    existing_families = {family_sha256(row["question"]) for row in existing.values()}

    raw_safe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dataset, path in raw_paths.items():
        for raw in read_jsonl(path):
            row_qid, question = source_qid(raw), str(raw["question"]).strip()
            key, family = (dataset, row_qid), family_sha256(question)
            if (
                not row_qid or key in existing or key in heldout_qids or family in heldout_families
                or key in consumed_qids or family in consumed_families
            ):
                continue
            raw_safe[dataset].append(question_only(
                dataset, row_qid, question, role="unassigned_train_only", route="TO_MATERIALIZE"
            ))

    # Reserve fresh, policy-unseen reward gates before choosing main additions.
    reward_blocked = set(heldout_families) | set(consumed_families) | set(existing_families)
    reward_dev: list[dict[str, Any]] = []
    reward_confirmation: list[dict[str, Any]] = []
    for dataset in DATASETS:
        selected = select_family_unique(
            raw_safe[dataset], REWARD_DEV[dataset], label="reward-development",
            blocked_families=reward_blocked,
        )
        for row in selected:
            row.update(role="reward_development", source_route="TO_MATERIALIZE_SOURCE_AWARE")
        reward_dev.extend(selected)
        remaining = [row for row in raw_safe[dataset] if row["qid"] not in {x["qid"] for x in selected}]
        selected_confirmation = select_family_unique(
            remaining, REWARD_CONFIRMATION[dataset], label="reward-confirmation",
            blocked_families=reward_blocked,
        )
        for row in selected_confirmation:
            row.update(role="reward_confirmation", source_route="TO_MATERIALIZE_SOURCE_AWARE")
        reward_confirmation.extend(selected_confirmation)

    # Build the exact 2Wiki existing-source allocation on distinct qids.
    main_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    proof = [row for row in existing.values() if row["dataset"] == "2wikimultihopqa" and {"W_ONLY", "P_W_FUSED"} <= row["modes"]]
    proof = sorted(proof, key=lambda row: rank("2wiki-proof-route", row["dataset"], row["qid"]))
    if len(proof) < 1230:
        raise ValueError(f"2Wiki proof qids insufficient: {len(proof)} < 1230")
    for row, route in [
        *((row, "P_W_FUSED") for row in proof[:720]),
        *((row, "W_ONLY") for row in proof[720:1230]),
    ]:
        main_by_dataset["2wikimultihopqa"].append(question_only(
            row["dataset"], row["qid"], row["question"], role="ppo_main", route=route
        ))
    proof_qids = {row["qid"] for row in proof[:1230]}
    nonproof = [row for row in existing.values() if row["dataset"] == "2wikimultihopqa" and row["qid"] not in proof_qids]
    n_capable = sorted(
        [row for row in nonproof if "N_REPLAY" in row["modes"]],
        key=lambda row: rank("2wiki-n", row["dataset"], row["qid"]),
    )[:180]
    n_qids = {row["qid"] for row in n_capable}
    p_capable = sorted(
        [row for row in nonproof if row["qid"] not in n_qids and "P_ONLY" in row["modes"]],
        key=lambda row: rank("2wiki-p", row["dataset"], row["qid"]),
    )[:390]
    if len(n_capable) != 180 or len(p_capable) != 390:
        raise ValueError("2Wiki P/N allocation cannot be filled")
    for row, route in [*((row, "N_REPLAY") for row in n_capable), *((row, "P_ONLY") for row in p_capable)]:
        main_by_dataset["2wikimultihopqa"].append(question_only(
            row["dataset"], row["qid"], row["question"], role="ppo_main", route=route
        ))

    # Hotpot/MuSiQue keep all 600 existing qids as P, then add 1,020 fresh P
    # and 180 fresh N qids.  Reward-family cohorts are excluded first.
    reward_families = {row["family_sha256"] for row in reward_dev + reward_confirmation}
    for dataset in ("hotpotqa", "musique"):
        existing_rows = [row for row in existing.values() if row["dataset"] == dataset]
        if len(existing_rows) != 600 or any("P_ONLY" not in row["modes"] for row in existing_rows):
            raise ValueError(f"{dataset} existing P capacity differs from 600")
        for row in existing_rows:
            main_by_dataset[dataset].append(question_only(
                dataset, row["qid"], row["question"], role="ppo_main", route="P_ONLY"
            ))
        fresh = [row for row in raw_safe[dataset] if row["family_sha256"] not in reward_families]
        fresh = sorted(fresh, key=lambda row: rank("ppo-main-fresh", dataset, str(row["qid"])))[:1200]
        if len(fresh) != 1200:
            raise ValueError(f"{dataset} fresh main capacity differs from 1200")
        for index, row in enumerate(fresh):
            item = dict(row)
            item.update(role="ppo_main", source_route="P_ONLY" if index < 1020 else "N_REPLAY")
            main_by_dataset[dataset].append(item)

    # Deterministic within-dataset shuffle and round-robin dataset interleave.
    for dataset in DATASETS:
        main_by_dataset[dataset] = sorted(
            main_by_dataset[dataset], key=lambda row: rank("ppo-main-order", dataset, str(row["qid"]))
        )
        if len(main_by_dataset[dataset]) != MAIN_PER_DATASET:
            raise ValueError(f"{dataset} main count is {len(main_by_dataset[dataset])}, expected 1800")
    main_rows = interleave_datasets(main_by_dataset)

    main_keys = {(row["dataset"], row["qid"]) for row in main_rows}
    dev_keys = {(row["dataset"], row["qid"]) for row in reward_dev}
    confirmation_keys = {(row["dataset"], row["qid"]) for row in reward_confirmation}
    main_families = {row["family_sha256"] for row in main_rows}
    dev_families = {row["family_sha256"] for row in reward_dev}
    confirmation_families = {row["family_sha256"] for row in reward_confirmation}
    checks = {
        "main_rows_5400": len(main_rows) == 5400,
        "main_unique_qid": len(main_keys) == 5400,
        "main_dataset_balanced": Counter(row["dataset"] for row in main_rows) == Counter({dataset: 1800 for dataset in DATASETS}),
        "smoke_prefix_balanced": Counter(row["dataset"] for row in main_rows[:300]) == Counter({dataset: 100 for dataset in DATASETS}),
        "reward_dev_rows_100": len(reward_dev) == 100,
        "reward_confirmation_rows_100": len(reward_confirmation) == 100,
        "qid_disjoint": not (main_keys & dev_keys or main_keys & confirmation_keys or dev_keys & confirmation_keys),
        "family_disjoint": not (main_families & dev_families or main_families & confirmation_families or dev_families & confirmation_families),
        "eval_qid_disjoint": not (main_keys & heldout_qids or dev_keys & heldout_qids or confirmation_keys & heldout_qids),
        "all_answer_free": all(row["gold_access"] is False for row in main_rows + reward_dev + reward_confirmation),
        "no_materialized_graph_claim": all(row["materialized"] is False for row in main_rows + reward_dev + reward_confirmation),
    }
    if not all(checks.values()):
        raise ValueError(f"PPO identity freeze checks failed: {checks}")

    args.out.mkdir(parents=True, exist_ok=False)
    outputs = {
        "main": args.out / "ppo_main.question_only.jsonl",
        "reward_development": args.out / "reward_development.question_only.jsonl",
        "reward_confirmation": args.out / "reward_confirmation.question_only.jsonl",
    }
    write_jsonl(outputs["main"], main_rows)
    write_jsonl(outputs["reward_development"], sorted(reward_dev, key=lambda row: (row["dataset"], rank("reward-dev-order", row["dataset"], row["qid"]))))
    write_jsonl(outputs["reward_confirmation"], sorted(reward_confirmation, key=lambda row: (row["dataset"], rank("reward-confirm-order", row["dataset"], row["qid"]))))
    report = {
        "schema_version": "saeg-v1-ppo-pool-freeze-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_IDENTITIES_NOT_MATERIALIZED_NOT_TRAINED",
        "counts": {
            "main": len(main_rows), "reward_development": len(reward_dev),
            "reward_confirmation": len(reward_confirmation),
            "main_by_dataset_route": dict(Counter(f"{row['dataset']}::{row['source_route']}" for row in main_rows)),
        },
        "schedule": {"K": 4, "smoke_prompts": 300, "mid_prompts": 2000, "formal_prompts": 5400},
        "checks": checks,
        "inputs": {
            "utility": {"path": str(args.utility_report), "sha256": sha256_file(args.utility_report)},
            "sft_release": {"path": str(args.sft_release), "sha256": sha256_file(args.sft_release)},
        },
        "outputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in outputs.items()},
        "scientific_boundary": (
            "Answer-free identity freeze only. Graph/context/outcome-label materialisation and both PPO arms "
            "remain forbidden until their own preflight and reward rankability gates pass."
        ),
    }
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra=report, status=report["status"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
