#!/usr/bin/env python
"""Freeze the answer-free mixed PPO v2 population and exact K=4 schedule.

V2 corrects the v1 family-isolation claim by recomputing every family with the
single ``answer-free-lexical-family-v1`` implementation.  The protected A
class is the canonical reporting cohort plus the unopened confirmation cohort.
No Gold answer, support fact, decomposition, or generated trajectory is read.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_sha256, validate_question_kg_record
from kgproweight.reward.proofkg_process import is_automatic_proofkg
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_mixed_ppo_three_dataset_v1 import rank, read_jsonl, ref, sha256_file
from scripts.prepare.freeze_qpeg_v1_protocol import (
    FAMILY_VERSION,
    family_sha256,
    question_family_signature,
)


SEED = 42
K = 4
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
QTYPES = ("inference", "comparison", "compositional", "bridge_comparison")
EXPERIMENT_ID = "MIXED-PPO-THREE-DATASET-V2-PROOF400-N1799-K4-7200-SEED42-PROTOCOL"
STATUS = "FROZEN_ANSWER_FREE_LEXICAL_FAMILY_V1_NOT_MATERIALIZED_NOT_TRAINED"
SCHEMA = "mixed-ppo-question-only-v2-proof400"


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _normalized(row: Mapping[str, Any], *, route: str, eligible: bool,
                proof_source: str = "none", qtype: str = "unknown") -> dict[str, Any]:
    dataset = str(row["dataset"])
    qid = str(row["qid"])
    question = str(row["question"]).strip()
    qhash = question_sha256(question)
    if row.get("question_sha256") not in (None, qhash):
        raise ValueError(f"question hash mismatch: {dataset}::{qid}")
    return {
        "schema_version": SCHEMA,
        "question_key": f"{dataset}::{qid}",
        "dataset": dataset,
        "qid": qid,
        "question": question,
        "question_sha256": qhash,
        "family_version": FAMILY_VERSION,
        "family_sha256": family_sha256(question),
        "question_type": qtype,
        "route": route,
        "proof_source": proof_source,
        "process_reward_eligible": bool(eligible),
        "gold_access": False,
        "evaluation_eligible": False,
    }


def _a_class(rows: Sequence[Mapping[str, Any]], role: str) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        item = _normalized(row, route=f"protected_a_{role}", eligible=False)
        item["protected_role"] = role
        result.append(item)
    return sorted(result, key=lambda row: (DATASETS.index(row["dataset"]), row["qid"]))


def _identity(rows: Iterable[Mapping[str, Any]]) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    return (
        {(str(row["dataset"]), str(row["qid"])) for row in rows},
        {(str(row["dataset"]), str(row["family_sha256"])) for row in rows},
    )


def _choose_max_family(
    candidates: Sequence[Mapping[str, Any]], *, n: int, label: str,
) -> list[dict[str, Any]]:
    """Choose one per family first, then deterministic repeat-family rows."""
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in candidates:
        row = dict(raw)
        by_family[str(row["family_sha256"])].append(row)
    for family in by_family:
        by_family[family].sort(key=lambda row: (rank(label, row["dataset"], row["qid"]), row["qid"]))
    family_order = sorted(
        by_family,
        key=lambda family: (rank(f"{label}-family", "2wikimultihopqa", family), family),
    )
    first = [by_family[family][0] for family in family_order]
    selected = first[:n]
    if len(selected) < n:
        selected_qids = {row["qid"] for row in selected}
        remainder = sorted(
            (row for rows in by_family.values() for row in rows if row["qid"] not in selected_qids),
            key=lambda row: (rank(f"{label}-repeat", row["dataset"], row["qid"]), row["qid"]),
        )
        selected.extend(remainder[: n - len(selected)])
    if len(selected) != n or len({row["qid"] for row in selected}) != n:
        raise ValueError(f"{label}: only {len(selected)}/{n} deterministic candidates")
    return selected


def select_proof400(
    hard_rows: Sequence[Mapping[str, Any]], complete_rows: Sequence[Mapping[str, Any]],
    *, protected_qids: set[tuple[str, str]], protected_families: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def safe(row: Mapping[str, Any]) -> bool:
        return (
            (row["dataset"], row["qid"]) not in protected_qids
            and (row["dataset"], row["family_sha256"]) not in protected_families
        )

    hard = [dict(row) for row in hard_rows if safe(row)]
    hard_qids_all = {str(row["qid"]) for row in hard_rows}
    hard_counts = Counter(row["question_type"] for row in hard)
    if len(hard) != 125:
        raise ValueError(f"expected 125 A-safe hard rows, got {len(hard)}")
    selected = list(hard)
    additions: list[dict[str, Any]] = []
    quotas: dict[str, int] = {}
    for qtype in QTYPES:
        need = 100 - hard_counts[qtype]
        quotas[qtype] = need
        candidates = [
            dict(row) for row in complete_rows
            if row["question_type"] == qtype and safe(row) and row["qid"] not in hard_qids_all
        ]
        chosen = _choose_max_family(candidates, n=need, label=f"proof400-{qtype}")
        for row in chosen:
            row["route"] = f"2wiki_proof_expansion_{qtype}"
            row["proof_source"] = "automatic_proofkg_2wiki_train_k4_v1"
        additions.extend(chosen)
    selected.extend(additions)
    if len(selected) != 400 or Counter(row["question_type"] for row in selected) != Counter({q: 100 for q in QTYPES}):
        raise ValueError("Proof400 does not have four exact 100-row question-type strata")
    return selected, {
        "safe_hard": len(hard),
        "excluded_hard": len(hard_rows) - len(hard),
        "safe_hard_by_question_type": dict(sorted(hard_counts.items())),
        "fill_quotas": quotas,
        "fill_selected": len(additions),
        "proof400_by_question_type": dict(sorted(Counter(row["question_type"] for row in selected).items())),
        "proof400_unique_families": len({row["family_sha256"] for row in selected}),
    }


def build_groups(population: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_dataset = {
        dataset: sorted(
            (dict(row) for row in population if row["dataset"] == dataset),
            key=lambda row: (rank(f"v2-schedule-{dataset}", dataset, row["qid"]), row["qid"]),
        )
        for dataset in DATASETS
    }
    proof = sorted(
        (row for row in by_dataset["2wikimultihopqa"] if row["process_reward_eligible"]),
        key=lambda row: (rank("v2-schedule-proof", row["dataset"], row["qid"]), row["qid"]),
    )
    ordinary = sorted(
        (row for row in by_dataset["2wikimultihopqa"] if not row["process_reward_eligible"]),
        key=lambda row: (rank("v2-schedule-ordinary", row["dataset"], row["qid"]), row["qid"]),
    )
    two_wiki: list[dict[str, Any]] = []
    for i in range(200):
        two_wiki.extend((dict(proof[2 * i]), dict(ordinary[i]), dict(proof[2 * i + 1])))
    musique = by_dataset["musique"]
    repeat = min(musique, key=lambda row: (rank("v2-musique-repeat", row["dataset"], row["qid"]), row["qid"]))
    musique = [*musique, dict(repeat)]
    if not (len(by_dataset["hotpotqa"]) == len(two_wiki) == len(musique) == 600):
        raise ValueError("fixed schedule is not 600 prompt groups per dataset")
    groups = []
    for i in range(600):
        for row in (by_dataset["hotpotqa"][i], two_wiki[i], musique[i]):
            item = dict(row)
            item["prompt_group_index"] = len(groups) + 1
            groups.append(item)
    return groups


def expand_k4(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    schedule = []
    for group_index, group in enumerate(groups, start=1):
        for within in range(1, K + 1):
            schedule.append({
                "schema_version": "mixed-ppo-fixed-rollout-schedule-v2-proof400",
                "rollout_index": len(schedule) + 1,
                "prompt_group_index": group_index,
                "within_group_rollout": within,
                "dataset": group["dataset"],
                "qid": group["qid"],
                "question_sha256": group["question_sha256"],
                "stratum": group["route"],
                "process_reward_eligible": group["process_reward_eligible"],
            })
    return schedule


def build_weights(population: Sequence[Mapping[str, Any]], groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    exposures = Counter((row["dataset"], row["qid"]) for row in groups)
    rows = []
    for row in population:
        rows.append({
            "schema_version": "mixed-ppo-rollout-sampling-weight-v2-proof400",
            "dataset": row["dataset"], "qid": row["qid"],
            "question_sha256": row["question_sha256"],
            "stratum": row["route"],
            "process_reward_eligible": row["process_reward_eligible"],
            "scheduled_prompt_group_exposures": exposures[(row["dataset"], row["qid"])],
            "sampling_probability": exposures[(row["dataset"], row["qid"])] / len(groups),
        })
    if abs(sum(row["sampling_probability"] for row in rows) - 1.0) > 1e-12:
        raise ValueError("sampling weights do not sum to one")
    return rows


def _function_hash() -> str:
    source = inspect.getsource(question_family_signature) + "\n" + inspect.getsource(family_sha256)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1_protocol", type=Path, default=Path("outputs/audits/mixed_ppo_three_dataset_v1_n1799_k4_seed42_protocol/protocol.json"))
    parser.add_argument("--hard", type=Path, default=Path("outputs/audits/2wiki_hard_curriculum_v1_protocol_v2/train_contrastive_qids.jsonl"))
    parser.add_argument("--complete_cohort", type=Path, default=Path("outputs/audits/automatic_proofkg_2wiki_train_k4_v1_n1500_seed42_preregistration/cohort.question_only.jsonl"))
    parser.add_argument("--complete_kg", type=Path, default=Path("data/silver_data/automatic_proofkg_2wiki_train_k4_v1/question_kg_records.jsonl"))
    parser.add_argument("--runtime_details", type=Path, default=Path("outputs/audits/automatic_proofkg_2wiki_train_k4_v1_n1500_seed42_historical_stage3_runtime/runtime_details.jsonl"))
    parser.add_argument("--eval_dir", type=Path, default=Path("outputs/audits/saeg_v1_evaluation_protocol_v1"))
    parser.add_argument("--out", type=Path, default=Path("outputs/audits/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_protocol"))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite frozen protocol: {args.out}")

    main_path = args.eval_dir / "canonical_reporting.question_only.jsonl"
    confirmation_path = args.eval_dir / "confirmation.question_only.jsonl"
    family_code_path = Path("scripts/prepare/freeze_qpeg_v1_protocol.py")
    inputs = [args.v1_protocol, args.hard, args.complete_cohort, args.complete_kg,
              args.runtime_details, main_path, confirmation_path, family_code_path]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    v1 = json.loads(args.v1_protocol.read_text(encoding="utf-8"))
    v1_population = read_jsonl(Path(v1["outputs"]["population"]["path"]))
    main_a = _a_class(read_jsonl(main_path), "canonical_main")
    confirmation_a = _a_class(read_jsonl(confirmation_path), "unopened_confirmation")
    protected = [*main_a, *confirmation_a]
    protected_qids, protected_families = _identity(protected)

    hard = []
    for row in read_jsonl(args.hard):
        item = _normalized(
            row, route=f"2wiki_hard_{row['stratum']}", eligible=True,
            proof_source="automatic_proofkg_2wiki_hard_contrastive_v1",
            qtype=str(row["question_type"]),
        )
        hard.append(item)

    cohort = {str(row["qid"]): row for row in read_jsonl(args.complete_cohort)}
    runtime = {str(row["question_key"]): row for row in read_jsonl(args.runtime_details)}
    complete = []
    for kg in read_jsonl(args.complete_kg):
        validate_question_kg_record(kg)
        source = cohort.get(str(kg["qid"]))
        trace = runtime.get(str(kg["question_key"]))
        if source is None or trace is None:
            raise ValueError(f"complete proof source join miss: {kg['question_key']}")
        if trace.get("question_sha256") != kg.get("question_sha256") or trace.get("kg_subgraph") != kg.get("kg_subgraph"):
            raise ValueError(f"complete proof trace identity mismatch: {kg['question_key']}")
        if not is_automatic_proofkg(trace, kg.get("kg_subgraph") or []):
            raise ValueError(f"source is not complete Gold-free ProofKG: {kg['question_key']}")
        complete.append(_normalized(
            source, route=f"2wiki_proof_expansion_{source['question_type']}", eligible=True,
            proof_source="automatic_proofkg_2wiki_train_k4_v1", qtype=str(source["question_type"]),
        ))
    if len(complete) != 1299:
        raise ValueError(f"expected 1299 complete automatic proofs, got {len(complete)}")

    proof400, proof_stats = select_proof400(
        hard, complete, protected_qids=protected_qids, protected_families=protected_families,
    )
    proof_qids = {(row["dataset"], row["qid"]) for row in proof400}
    hp_mu = [
        _normalized(row, route=str(row["route"]), eligible=False)
        for row in v1_population if row["dataset"] in {"hotpotqa", "musique"}
    ]
    ordinary_candidates = [
        _normalized(row, route="2wiki_ordinary_outcome", eligible=False)
        for row in v1_population if row["route"] == "2wiki_ordinary_outcome"
    ]
    ordinary_candidates = [
        row for row in ordinary_candidates
        if (row["dataset"], row["qid"]) not in protected_qids
        and (row["dataset"], row["family_sha256"]) not in protected_families
        and (row["dataset"], row["qid"]) not in proof_qids
    ]
    ordinary = sorted(
        ordinary_candidates,
        key=lambda row: (rank("v2-ordinary200", row["dataset"], row["qid"]), row["qid"]),
    )[:200]
    if len(ordinary) != 200:
        raise ValueError(f"only {len(ordinary)}/200 safe ordinary rows")

    population = [*hp_mu, *ordinary, *proof400]
    population.sort(key=lambda row: (DATASETS.index(row["dataset"]), rank("v2-population", row["dataset"], row["qid"]), row["qid"]))
    pop_qids, pop_families = _identity(population)
    counts = Counter(row["dataset"] for row in population)
    if len(population) != len(pop_qids) or counts != Counter({"hotpotqa": 600, "2wikimultihopqa": 600, "musique": 599}):
        raise ValueError(f"invalid v2 population: {len(population)} {counts}")
    if pop_qids & protected_qids or pop_families & protected_families:
        raise ValueError("v2 population overlaps protected A-class qid/family")

    groups = build_groups(population)
    schedule = expand_k4(groups)
    weights = build_weights(population, groups)
    scheduled = Counter((row["dataset"], row["qid"]) for row in groups)
    scheduled_eligible = sum(bool(row["process_reward_eligible"]) for row in schedule)
    schedule_gates = {
        "groups_1800": len(groups) == 1800,
        "dataset_groups_600_each": Counter(row["dataset"] for row in groups) == Counter({d: 600 for d in DATASETS}),
        "all_1799_unique_scheduled": len(scheduled) == 1799,
        "only_one_musique_repeat": Counter(scheduled.values()) == Counter({1: 1798, 2: 1}),
        "rollouts_7200": len(schedule) == 7200,
        "k4_contiguous": all(len({(r["dataset"], r["qid"]) for r in schedule[i:i + 4]}) == 1 for i in range(0, 7200, 4)),
        "proof_groups_400": sum(bool(row["process_reward_eligible"]) for row in groups) == 400,
        "proof_trajectories_1600": scheduled_eligible == 1600,
    }
    if not all(schedule_gates.values()):
        raise ValueError(schedule_gates)

    args.out.mkdir(parents=True, exist_ok=False)
    output_paths = {
        "population": args.out / "population.question_only.jsonl",
        "proof400": args.out / "proof400.question_only.jsonl",
        "ordinary200": args.out / "ordinary200.question_only.jsonl",
        "protected_a_canonical_main": args.out / "protected_a_canonical_main.question_only.jsonl",
        "protected_a_unopened_confirmation": args.out / "protected_a_unopened_confirmation.question_only.jsonl",
        "sampling_weights": args.out / "sampling_weights.question_only.jsonl",
        "prompt_groups": args.out / "prompt_groups.question_only.jsonl",
        "fixed_rollout_schedule": args.out / "fixed_rollout_schedule.question_only.jsonl",
    }
    for name, rows in (
        ("population", population), ("proof400", proof400), ("ordinary200", ordinary),
        ("protected_a_canonical_main", main_a),
        ("protected_a_unopened_confirmation", confirmation_a),
        ("sampling_weights", weights), ("prompt_groups", groups),
        ("fixed_rollout_schedule", schedule),
    ):
        write_jsonl(output_paths[name], rows)

    hard_safe_qids = {row["qid"] for row in proof400 if row["proof_source"] == "automatic_proofkg_2wiki_hard_contrastive_v1"}
    report = {
        "schema_version": "mixed-ppo-three-dataset-protocol-v2-proof400",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "population": {
            "unique_total": len(population),
            "unique_by_dataset": dict(sorted(counts.items())),
            "2wiki_ordinary": 200,
            "2wiki_complete_proofkg": 400,
            "2wiki_proof_source_counts": dict(sorted(Counter(row["proof_source"] for row in proof400).items())),
            **proof_stats,
        },
        "protected_a_class": {
            "definition": "canonical main reporting cohort plus unopened confirmation cohort",
            "family_version": FAMILY_VERSION,
            "family_function_sha256": _function_hash(),
            "family_implementation_file": ref(family_code_path),
            "rows_by_role_and_dataset": {
                "canonical_main": dict(sorted(Counter(row["dataset"] for row in main_a).items())),
                "unopened_confirmation": dict(sorted(Counter(row["dataset"] for row in confirmation_a).items())),
            },
            "population_qid_overlap": len(pop_qids & protected_qids),
            "population_family_overlap": len(pop_families & protected_families),
            "proof400_qid_overlap": len(proof_qids & protected_qids),
            "proof400_family_overlap": len({(row["dataset"], row["family_sha256"]) for row in proof400} & protected_families),
        },
        "schedule": {
            "prompt_groups": len(groups), "rollouts_per_prompt": K,
            "trajectories": len(schedule), "scheduled_unique": len(scheduled),
            "process_eligible_groups": scheduled_eligible // K,
            "process_eligible_trajectories": scheduled_eligible,
            "checks": schedule_gates,
        },
        "scientific_boundary": {
            "answer_free_freeze": True,
            "gold_access": False,
            "failed_qpeg_or_saeg_p_edges_consumed": False,
            "training_started": False,
            "v1_family_gate_status": "SUPERSEDED_NAMESPACE_INCOMPARABLE",
            "v1_data_and_results_preserved": True,
            "v2_family_isolation_scope": "protected A-class only; train-side family repeats are allowed",
            "train_side_consumed_qids_allowed": True,
            "consumed_rankability_or_verifier_results_after_training_use": "development/consumed only, never independent confirmation",
            "hard208_policy": f"{len(hard_safe_qids)} A-safe rows retained; unsafe hard rows excluded rather than forced",
        },
        "inputs": {
            "v1_protocol": ref(args.v1_protocol), "hard": ref(args.hard),
            "complete_cohort": ref(args.complete_cohort), "complete_kg": ref(args.complete_kg),
            "runtime_details": ref(args.runtime_details),
            "canonical_main": ref(main_path), "unopened_confirmation": ref(confirmation_path),
        },
        "outputs": {name: ref(path) for name, path in output_paths.items()},
    }
    protocol_path = args.out / "protocol.json"
    protocol_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    supersession = {
        "schema_version": "mixed-ppo-family-namespace-supersession-1",
        "status": "V1_FAMILY_GATE_SUPERSEDED_DATA_AND_RESULTS_PRESERVED",
        "superseded_claim_only": "v1 population_eval_family_overlap_zero",
        "reason": "v1 compared stored family hashes produced by incompatible namespaces",
        "replacement_family_version": FAMILY_VERSION,
        "replacement_protocol": ref(protocol_path),
        "v1_protocol": ref(args.v1_protocol),
        "v1_files_deleted_or_overwritten": False,
    }
    supersession_path = args.out / "v1_family_gate_supersession.json"
    supersession_path.write_text(json.dumps(supersession, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, status=STATUS, extra={
        "phase": "mixed_ppo_v2_answer_free_protocol_freeze",
        "experiment_id": EXPERIMENT_ID,
        "protocol_sha256": sha256_file(protocol_path),
        "v1_family_gate_supersession_sha256": sha256_file(supersession_path),
    })
    print(json.dumps({"status": STATUS, "population": report["population"],
                      "protected_a_class": report["protected_a_class"],
                      "schedule": report["schedule"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
