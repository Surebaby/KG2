#!/usr/bin/env python
"""Freeze the answer-free identities and exact K=4 schedule for mixed PPO v1.

This stage deliberately runs before train Gold answers are joined.  It reuses
the frozen canonical retrieval cohort for ordinary prompts and the already
frozen 208-qid 2Wiki hard-ProofKG curriculum.  Failed Passage-QPEG/SAEG-P edges
are never read by this script and cannot enter the resulting population.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256


SEED = 42
K = 4
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
POPULATION_COUNTS = {"hotpotqa": 600, "2wikimultihopqa": 600, "musique": 599}
EXPERIMENT_ID = "MIXED-PPO-THREE-DATASET-V1-N1799-K4-7200-SEED42-PROTOCOL"
STATUS = "FROZEN_ANSWER_FREE_NOT_MATERIALIZED_NOT_TRAINED"


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


def ref(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def rank(label: str, dataset: str, qid: str, *, seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}\0{label}\0{dataset}\0{qid}".encode("utf-8")).hexdigest()


def source_qid(row: Mapping[str, Any]) -> str:
    return str(row.get("source_qid") or row.get("qid") or row.get("id") or "").strip()


def _identity_set(paths: Sequence[Path]) -> tuple[set[tuple[str, str]], set[str]]:
    qids: set[tuple[str, str]] = set()
    families: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            dataset = str(row.get("dataset") or "2wikimultihopqa").strip()
            qid = source_qid(row)
            question = str(row.get("question") or "").strip()
            if not dataset or not qid or not question:
                raise ValueError(f"incomplete identity in {path}: dataset={dataset!r}, qid={qid!r}")
            qids.add((dataset, qid))
            families.add(str(row.get("family_sha256") or family_sha256(question)))
    return qids, families


def _question_only(row: Mapping[str, Any], *, route: str, eligible: bool) -> dict[str, Any]:
    dataset = str(row["dataset"])
    qid = str(row["qid"])
    question = str(row["question"]).strip()
    return {
        "schema_version": "mixed-ppo-question-only-v1",
        "question_key": f"{dataset}::{qid}",
        "dataset": dataset,
        "qid": qid,
        "question": question,
        "question_sha256": str(row["question_sha256"]),
        "family_sha256": str(row["family_sha256"]),
        "route": route,
        "process_reward_eligible": bool(eligible),
        "gold_access": False,
        "evaluation_eligible": False,
    }


def select_ordinary_2wiki(
    candidates: Sequence[Mapping[str, Any]],
    *,
    n: int,
    blocked_qids: set[tuple[str, str]],
    blocked_families: set[str],
) -> list[dict[str, Any]]:
    eligible = [
        dict(row)
        for row in candidates
        if str(row.get("dataset")) == "2wikimultihopqa"
        and ("2wikimultihopqa", str(row.get("qid") or "")) not in blocked_qids
        and str(row.get("family_sha256") or "") not in blocked_families
    ]
    eligible.sort(key=lambda row: (
        rank("ordinary-2wiki-population", "2wikimultihopqa", str(row["qid"])),
        str(row["qid"]),
    ))
    if len(eligible) < n:
        raise ValueError(f"only {len(eligible)}/{n} safe ordinary 2Wiki rows")
    selected = eligible[:n]
    if len({str(row["family_sha256"]) for row in selected}) != len(selected):
        raise ValueError("ordinary 2Wiki selection is not family-unique")
    return selected


def build_sampling_weights(population: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    strata = Counter(str(row["route"]) for row in population)
    target_mass = {
        "hotpotqa_outcome": 1.0 / 3.0,
        "musique_outcome": 1.0 / 3.0,
        "2wiki_ordinary_outcome": 1.0 / 6.0,
        "2wiki_hard_recovery": 1.0 / 12.0,
        "2wiki_hard_stability": 1.0 / 12.0,
    }
    if set(strata) != set(target_mass):
        raise ValueError(f"unexpected population strata: {dict(strata)}")
    rows = []
    for row in population:
        stratum = str(row["route"])
        rows.append({
            "schema_version": "mixed-ppo-rollout-sampling-weight-v1",
            "dataset": row["dataset"],
            "qid": row["qid"],
            "question_sha256": row["question_sha256"],
            "stratum": stratum,
            "process_reward_eligible": bool(row["process_reward_eligible"]),
            "sampling_probability": target_mass[stratum] / strata[stratum],
        })
    if abs(sum(float(row["sampling_probability"]) for row in rows) - 1.0) > 1e-12:
        raise ValueError("sampling probability mass is not 1.0")
    return rows


def _interleave(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(left) != len(right):
        raise ValueError(f"cannot interleave unequal lists: {len(left)} != {len(right)}")
    out: list[dict[str, Any]] = []
    for a, b in zip(left, right):
        out.extend((dict(a), dict(b)))
    return out


def build_prompt_group_schedule(population: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return exactly 1,800 prompt groups with the frozen balanced allocation."""

    by_route: dict[str, list[dict[str, Any]]] = {}
    for route in {
        "hotpotqa_outcome", "musique_outcome", "2wiki_ordinary_outcome",
        "2wiki_hard_recovery", "2wiki_hard_stability",
    }:
        by_route[route] = sorted(
            [dict(row) for row in population if row["route"] == route],
            key=lambda row: (rank(f"schedule-{route}", str(row["dataset"]), str(row["qid"])), str(row["qid"])),
        )

    hotpot = by_route["hotpotqa_outcome"]
    musique = by_route["musique_outcome"]
    ordinary = by_route["2wiki_ordinary_outcome"][:300]
    recovery = by_route["2wiki_hard_recovery"]
    stability = by_route["2wiki_hard_stability"][:150]
    if not (len(hotpot) == 600 and len(musique) == 599 and len(ordinary) == 300):
        raise ValueError("unexpected ordinary/dataset population capacity")
    if len(recovery) != 25 or len(stability) != 150:
        raise ValueError(
            f"hard schedule requires recovery=25 and selected stability=150, got {len(recovery)}/{len(stability)}"
        )

    # Six deterministic passes expose each of the 25 recovery prompts exactly
    # six times.  Stability and ordinary prompts selected for the schedule are
    # each seen once.  This realizes a 150/150 hard split and a 300/300
    # hard/ordinary split within 2Wiki without claiming full population cover.
    recovery_exposures = [dict(row) for _cycle in range(6) for row in recovery]
    hard = _interleave(recovery_exposures, stability)
    two_wiki = _interleave(ordinary, hard)

    repeat = min(
        musique,
        key=lambda row: (rank("musique-single-repeat", str(row["dataset"]), str(row["qid"])), str(row["qid"])),
    )
    musique = [*musique, dict(repeat)]
    if not (len(hotpot) == len(two_wiki) == len(musique) == 600):
        raise ValueError("dataset prompt-group schedules are not exactly balanced at 600")

    groups: list[dict[str, Any]] = []
    for index in range(600):
        for row in (hotpot[index], two_wiki[index], musique[index]):
            item = dict(row)
            item["prompt_group_index"] = len(groups) + 1
            groups.append(item)
    return groups


def expand_k4_schedule(groups: Sequence[Mapping[str, Any]], *, k: int = K) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups, start=1):
        for within in range(1, k + 1):
            rows.append({
                "schema_version": "mixed-ppo-fixed-rollout-schedule-v1",
                "rollout_index": len(rows) + 1,
                "prompt_group_index": group_index,
                "within_group_rollout": within,
                "dataset": group["dataset"],
                "qid": group["qid"],
                "question_sha256": group["question_sha256"],
                "stratum": group["route"],
                "process_reward_eligible": bool(group["process_reward_eligible"]),
            })
    return rows


def _schedule_checks(groups: Sequence[Mapping[str, Any]], schedule: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    group_counts = Counter(str(row["dataset"]) for row in groups)
    first_300 = Counter(str(row["dataset"]) for row in groups[:300])
    wiki_routes = Counter(str(row["route"]) for row in groups if row["dataset"] == "2wikimultihopqa")
    first_wiki = [row for row in groups[:300] if row["dataset"] == "2wikimultihopqa"]
    first_wiki_routes = Counter(str(row["route"]) for row in first_wiki)
    return {
        "prompt_groups_1800": len(groups) == 1800,
        "rollouts_7200": len(schedule) == 7200,
        "dataset_groups_600_each": group_counts == Counter({dataset: 600 for dataset in DATASETS}),
        "smoke_prefix_groups_300_balanced": first_300 == Counter({dataset: 100 for dataset in DATASETS}),
        "2wiki_full_schedule_ordinary300_recovery150_stability150": wiki_routes == Counter({
            "2wiki_ordinary_outcome": 300,
            "2wiki_hard_recovery": 150,
            "2wiki_hard_stability": 150,
        }),
        "2wiki_smoke_prefix_ordinary50_recovery25_stability25": first_wiki_routes == Counter({
            "2wiki_ordinary_outcome": 50,
            "2wiki_hard_recovery": 25,
            "2wiki_hard_stability": 25,
        }),
        "k4_same_prompt_groups": all(
            len({(row["dataset"], row["qid"]) for row in schedule[start:start + K]}) == 1
            for start in range(0, len(schedule), K)
        ),
        "rollout_indices_contiguous": [int(row["rollout_index"]) for row in schedule] == list(range(1, 7201)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--effective_train", type=Path,
        default=Path("outputs/audits/saeg_p_hard_negative_alignment_v2_isolation_addendum/effective_train.question_only.jsonl"),
    )
    parser.add_argument(
        "--retrieval_contexts", type=Path,
        default=Path("outputs/audits/saeg_p_alignment_v2_train1800_retrieval/retrieval_contexts.jsonl"),
    )
    parser.add_argument(
        "--hard_curriculum", type=Path,
        default=Path("outputs/audits/2wiki_hard_curriculum_v1_protocol_v2/train_contrastive_qids.jsonl"),
    )
    parser.add_argument(
        "--hard_silver", type=Path,
        default=Path("data/silver_data/automatic_proofkg_2wiki_hard_contrastive_v1/silver_train.jsonl"),
    )
    parser.add_argument(
        "--hard_kg", type=Path,
        default=Path("data/silver_data/automatic_proofkg_2wiki_hard_contrastive_v1/question_kg_records.with_execution.jsonl"),
    )
    parser.add_argument(
        "--eval_dir", type=Path,
        default=Path("outputs/audits/saeg_v1_evaluation_protocol_v1"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/audits/mixed_ppo_three_dataset_v1_n1799_k4_seed42_protocol"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite frozen protocol: {args.out}")

    eval_paths = [
        args.eval_dir / f"{role}.question_only.jsonl"
        for role in ("development", "confirmation", "canonical_reporting")
    ]
    consumed_paths = [
        Path("outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_train.question_only.jsonl"),
        Path("outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_dev.question_only.jsonl"),
        Path("outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_confirmation.question_only.jsonl"),
        Path("outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_reserve.question_only.jsonl"),
        Path("outputs/audits/2wiki_train_only_rankability_n150_v1/cohort.question_only.jsonl"),
        Path("outputs/audits/2wiki_train_only_rankability_confirmation_n100_v1/cohort.question_only.jsonl"),
    ]
    input_paths = [
        args.effective_train, args.retrieval_contexts, args.hard_curriculum,
        args.hard_silver, args.hard_kg, *eval_paths, *consumed_paths,
    ]
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing mixed-PPO freeze inputs: {missing}")

    effective = read_jsonl(args.effective_train)
    retrieval = read_jsonl(args.retrieval_contexts)
    retrieval_by_key = {str(row["question_key"]): row for row in retrieval}
    if len(retrieval_by_key) != len(retrieval):
        raise ValueError("canonical retrieval contains duplicate question_key")
    for row in effective:
        found = retrieval_by_key.get(str(row["question_key"]))
        if found is None or found.get("question_sha256") != row.get("question_sha256"):
            raise ValueError(f"effective/retrieval identity mismatch: {row['question_key']}")

    hard = read_jsonl(args.hard_curriculum)
    if len(hard) != 208 or Counter(str(row["stratum"]) for row in hard) != Counter({"recovery": 25, "stability": 183}):
        raise ValueError("hard curriculum is not the frozen 25 recovery + 183 stability cohort")
    hard_qids = {(str(row["dataset"]), str(row["qid"])) for row in hard}
    hard_families = {str(row["family_sha256"]) for row in hard}
    eval_qids, eval_families = _identity_set(eval_paths)
    consumed_qids, consumed_families = _identity_set(consumed_paths)
    blocked_qids = hard_qids | eval_qids | consumed_qids
    blocked_families = hard_families | eval_families | consumed_families

    hotpot = [row for row in effective if row["dataset"] == "hotpotqa"]
    musique = [row for row in effective if row["dataset"] == "musique"]
    if len(hotpot) != 600 or len(musique) != 599:
        raise ValueError(f"effective HP/Mu counts changed: {len(hotpot)}/{len(musique)}")
    ordinary = select_ordinary_2wiki(
        effective, n=392, blocked_qids=blocked_qids, blocked_families=blocked_families,
    )

    population = [
        *(_question_only(row, route="hotpotqa_outcome", eligible=False) for row in hotpot),
        *(_question_only(row, route="musique_outcome", eligible=False) for row in musique),
        *(_question_only(row, route="2wiki_ordinary_outcome", eligible=False) for row in ordinary),
        *(
            _question_only(
                row,
                route=f"2wiki_hard_{row['stratum']}",
                eligible=True,
            )
            for row in hard
        ),
    ]
    population.sort(key=lambda row: (
        DATASETS.index(str(row["dataset"])),
        rank("population-order", str(row["dataset"]), str(row["qid"])),
        str(row["qid"]),
    ))
    keys = [str(row["question_key"]) for row in population]
    population_counts = Counter(str(row["dataset"]) for row in population)
    if len(population) != 1799 or len(keys) != len(set(keys)) or population_counts != Counter(POPULATION_COUNTS):
        raise ValueError(f"invalid mixed population: n={len(population)}, counts={dict(population_counts)}")

    weights = build_sampling_weights(population)
    groups = build_prompt_group_schedule(population)
    schedule = expand_k4_schedule(groups)
    schedule_checks = _schedule_checks(groups, schedule)
    if not all(schedule_checks.values()):
        raise ValueError(f"fixed schedule checks failed: {schedule_checks}")

    population_qids = {(str(row["dataset"]), str(row["qid"])) for row in population}
    population_families = {str(row["family_sha256"]) for row in population}
    isolation_checks = {
        "population_eval_qid_overlap_zero": not (population_qids & eval_qids),
        "population_eval_family_overlap_zero": not (population_families & eval_families),
        "ordinary_consumed_qid_overlap_zero": not ({("2wikimultihopqa", str(row["qid"])) for row in ordinary} & consumed_qids),
        "ordinary_consumed_family_overlap_zero": not ({str(row["family_sha256"]) for row in ordinary} & consumed_families),
        "ordinary_hard_qid_overlap_zero": not ({("2wikimultihopqa", str(row["qid"])) for row in ordinary} & hard_qids),
        "ordinary_hard_family_overlap_zero": not ({str(row["family_sha256"]) for row in ordinary} & hard_families),
    }
    if not all(isolation_checks.values()):
        raise ValueError(f"mixed population isolation failed: {isolation_checks}")

    args.out.mkdir(parents=True, exist_ok=False)
    output_paths = {
        "population": args.out / "population.question_only.jsonl",
        "sampling_weights": args.out / "sampling_weights.question_only.jsonl",
        "prompt_groups": args.out / "prompt_groups.question_only.jsonl",
        "fixed_rollout_schedule": args.out / "fixed_rollout_schedule.question_only.jsonl",
    }
    write_jsonl(output_paths["population"], population)
    write_jsonl(output_paths["sampling_weights"], weights)
    write_jsonl(output_paths["prompt_groups"], groups)
    write_jsonl(output_paths["fixed_rollout_schedule"], schedule)

    schedule_unique = {(str(row["dataset"]), str(row["qid"])) for row in groups}
    report = {
        "schema_version": "mixed-ppo-three-dataset-protocol-v1",
        "experiment_id": EXPERIMENT_ID,
        "researcher_approval": "USER_APPROVED_2026-09-03_MIXED_PPO_7200_REWARD_REVIEW",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "population": {
            "unique_total": len(population),
            "unique_by_dataset": dict(sorted(population_counts.items())),
            "2wiki_route_counts": dict(sorted(Counter(
                str(row["route"]) for row in population if row["dataset"] == "2wikimultihopqa"
            ).items())),
        },
        "schedule": {
            "prompt_groups": len(groups),
            "rollouts_per_prompt": K,
            "trajectories": len(schedule),
            "actual_scheduled_unique_total": len(schedule_unique),
            "actual_scheduled_unique_by_dataset": dict(sorted(Counter(dataset for dataset, _qid in schedule_unique).items())),
            "group_exposures_by_dataset": dict(sorted(Counter(str(row["dataset"]) for row in groups).items())),
            "group_exposures_by_stratum": dict(sorted(Counter(str(row["route"]) for row in groups).items())),
            "musique_repeated_qid": next(
                qid for (dataset, qid), count in Counter((str(row["dataset"]), str(row["qid"])) for row in groups).items()
                if dataset == "musique" and count == 2
            ),
            "not_a_full_unique_population_pass": True,
            "checks": schedule_checks,
        },
        "isolation_checks": isolation_checks,
        "scientific_boundary": {
            "answer_free_freeze": True,
            "gold_access": False,
            "failed_qpeg_or_saeg_p_edges_consumed": False,
            "hard_2wiki_source": "previously frozen complete automatic ProofKG only",
            "ordinary_source": "frozen canonical retrieval passages with no graph",
            "training_started": False,
            "runtime_readiness": "NOT_RUNNABLE_UNTIL_FIXED_SCHEDULE_IS_CONSUMED_AND_PREFLIGHTED",
        },
        "inputs": {
            "effective_train": ref(args.effective_train),
            "retrieval_contexts": ref(args.retrieval_contexts),
            "hard_curriculum": ref(args.hard_curriculum),
            "hard_silver": ref(args.hard_silver),
            "hard_question_kg": ref(args.hard_kg),
            "evaluation": [ref(path) for path in eval_paths],
            "historical_consumed": [ref(path) for path in consumed_paths],
        },
        "outputs": {name: ref(path) for name, path in output_paths.items()},
    }
    protocol_path = args.out / "protocol.json"
    protocol_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, status=STATUS, extra={
        "phase": "mixed_ppo_answer_free_protocol_freeze",
        "experiment_id": EXPERIMENT_ID,
        "protocol_sha256": sha256_file(protocol_path),
    })
    print(json.dumps({
        "status": STATUS,
        "population": report["population"],
        "schedule": report["schedule"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
