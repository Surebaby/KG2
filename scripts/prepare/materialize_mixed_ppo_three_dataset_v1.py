#!/usr/bin/env python
"""Materialize the frozen 1,799-qid mixed PPO population.

Only the train Gold answer is added, as an outcome-reward label.  It is not
inserted into questions, passages, KG records, plans, or supervised steps.
The 208 frozen hard 2Wiki questions retain their complete automatic ProofKG
and Gold-free executor traces.  Every other question receives an explicit
identity-safe empty KG record and is process-reward ineligible.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.kg.question_kg import (
    make_question_kg_record,
    question_sha256,
    validate_question_kg_record,
)
from kgproweight.kg.training_question_kg import (
    apply_training_question_kg,
    read_question_kg_records,
)
from kgproweight.reward.proofkg_process import is_automatic_proofkg
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_mixed_ppo_three_dataset_v1 import read_jsonl, sha256_file


EXPERIMENT_ID = "MIXED-PPO-THREE-DATASET-V1-N1799-K4-7200-SEED42-DATA"
STATUS = "COMPLETE_DATA_NOT_TRAINED_RUNTIME_SCHEDULE_BLOCKED"
PROTOCOL_STATUS = "FROZEN_ANSWER_FREE_NOT_MATERIALIZED_NOT_TRAINED"
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def ref(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _verify_ref(identity: Mapping[str, Any]) -> Path:
    path = Path(str(identity["path"]))
    if not path.is_file() or sha256_file(path) != str(identity["sha256"]):
        raise ValueError(f"frozen input missing or hash mismatch: {path}")
    return path


def _by_question_key(rows: Iterable[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for value in rows:
        row = dict(value)
        key = str(row.get("question_key") or "")
        if not key or key in output:
            raise ValueError(f"{label} contains empty/duplicate question_key: {key!r}")
        output[key] = row
    return output


def freeze_raw_source_addendum(
    *, protocol_path: Path, raw_paths: Mapping[str, Path], output_dir: Path,
) -> dict[str, Any]:
    """Bind raw train files by hash before any Gold-bearing row is parsed."""

    expected = {
        "schema_version": "mixed-ppo-raw-source-addendum-v1",
        "experiment_id": f"{EXPERIMENT_ID}-RAW-SOURCE-ADDENDUM",
        "status": "FROZEN_BEFORE_GOLD_LABEL_JOIN",
        "parent_protocol": ref(protocol_path),
        "raw_train": {dataset: ref(path) for dataset, path in raw_paths.items()},
        "scientific_boundary": (
            "This append-only addendum binds source files only. No row, answer, "
            "supporting fact, decomposition, passage, or KG has been selected or modified."
        ),
    }
    if output_dir.exists():
        path = output_dir / "raw_source_addendum.json"
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in ("schema_version", "experiment_id", "status", "parent_protocol", "raw_train"):
            if existing.get(key) != expected.get(key):
                raise ValueError(f"existing raw-source addendum changed at field {key}")
        return existing
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = {**expected, "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    path = output_dir / "raw_source_addendum.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(output_dir, status=payload["status"], extra={
        "phase": "mixed_ppo_raw_source_freeze",
        "experiment_id": payload["experiment_id"],
        "addendum_sha256": sha256_file(path),
    })
    return payload


def load_selected_raw(
    raw_paths: Mapping[str, Path], population: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    wanted = {
        dataset: {str(row["qid"]) for row in population if row["dataset"] == dataset}
        for dataset in DATASETS
    }
    output: dict[str, dict[str, Any]] = {}
    for dataset, path in raw_paths.items():
        found: set[str] = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                qid = str(row.get("id") or row.get("qid") or "").strip()
                if qid not in wanted[dataset]:
                    continue
                key = f"{dataset}::{qid}"
                if qid in found or key in output:
                    raise ValueError(f"duplicate selected raw identity: {key}")
                found.add(qid)
                output[key] = row
        missing = sorted(wanted[dataset] - found)
        if missing:
            raise ValueError(f"raw {dataset} missing {len(missing)} selected qids: {missing[:5]}")
    return output


def primary_gold_answers(raw: Mapping[str, Any]) -> tuple[str, list[str]]:
    aliases = [str(value).strip() for value in (raw.get("golden_answers") or []) if str(value).strip()]
    if not aliases:
        raise ValueError(f"raw train row has no Gold answer: {raw.get('id') or raw.get('qid')}")
    return aliases[0], aliases


def make_outcome_only_kg_record(identity: Mapping[str, Any]) -> dict[str, Any]:
    record = make_question_kg_record(
        dataset=str(identity["dataset"]),
        qid=str(identity["qid"]),
        question=str(identity["question"]),
        triples=[],
        query_plan={},
        provenance={
            "builder_version": "mixed-ppo-outcome-only-empty-kg-v1",
            "gold_access": False,
            "complete_plan_execution": False,
            "process_reward_eligible": False,
            "ineligible_reason": "outcome_only_no_trusted_proofkg",
            "failed_qpeg_or_saeg_p_edges_included": False,
        },
    )
    record.update({
        "execution": {},
        "runtime_error": None,
        "process_reward_eligible": False,
    })
    return record


def _question_type(dataset: str, raw: Mapping[str, Any]) -> str:
    metadata = raw.get("metadata") or {}
    if dataset == "musique":
        nested = metadata.get("metadata") or {}
        decomposition = list(nested.get("question_decomposition") or [])
        return f"decomposition_{len(decomposition)}hop" if decomposition else "unknown"
    return str(metadata.get("type") or "unknown")


def build_silver_row(
    identity: Mapping[str, Any],
    *,
    raw: Mapping[str, Any],
    retrieved_passages: Sequence[Mapping[str, Any]],
    kg_subgraph: Sequence[Sequence[str]],
) -> dict[str, Any]:
    dataset = str(identity["dataset"])
    gold, aliases = primary_gold_answers(raw)
    if str(raw.get("question") or "").strip() != str(identity["question"]).strip():
        raise ValueError(f"raw question mismatch: {identity['question_key']}")
    if question_sha256(str(raw["question"])) != str(identity["question_sha256"]):
        raise ValueError(f"raw question hash mismatch: {identity['question_key']}")
    passages = [dict(value) for value in retrieved_passages]
    if len(passages) != 10 or any(not str(row.get("contents") or "").strip() for row in passages):
        raise ValueError(f"expected exactly ten nonempty passages: {identity['question_key']}")
    forbidden_passage_fields = {"answer", "answers", "golden_answers", "supporting_facts", "decomposition"}
    leaked = sorted({key for row in passages for key in row if key in forbidden_passage_fields})
    if leaked:
        raise ValueError(f"explicit Gold/support fields in prompt passages for {identity['question_key']}: {leaked}")
    eligible = bool(identity["process_reward_eligible"])
    return {
        "qid": str(identity["qid"]),
        "question": str(identity["question"]),
        "answer": gold,
        "dataset": dataset,
        # No same-file CE replay: this asset is rollout/outcome supervision only.
        "steps": [],
        "kg_subgraph": [[str(part) for part in triple] for triple in kg_subgraph],
        "retrieved_passages": passages,
        "accepted": True,
        "metadata": {
            "gold_answer": gold,
            "gold_answer_aliases": aliases,
            "question_type": _question_type(dataset, raw),
            "source_split": "train",
            "rollout_only": True,
            "source_gold_trace_removed": True,
            "evaluation_eligible": False,
            "family_sha256": str(identity["family_sha256"]),
            "mixed_ppo_data_version": "mixed-ppo-three-dataset-v1",
            "mixed_ppo_route": str(identity["route"]),
            "process_reward_eligible": eligible,
            "gold_use": "outcome_reward_label_only",
            "failed_qpeg_or_saeg_p_edges_included": False,
        },
        "teacher_output": "",
        "teacher_model": "none_ppo_rollout_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("outputs/audits/mixed_ppo_three_dataset_v1_n1799_k4_seed42_protocol/protocol.json"),
    )
    parser.add_argument(
        "--raw_source_addendum_dir", type=Path,
        default=Path("outputs/audits/mixed_ppo_three_dataset_v1_n1799_k4_seed42_raw_source_addendum"),
    )
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("data/silver_data/mixed_ppo_three_dataset_v1_n1799_k4_seed42"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite versioned data: {args.output_dir}")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != PROTOCOL_STATUS or protocol.get("experiment_id") != (
        "MIXED-PPO-THREE-DATASET-V1-N1799-K4-7200-SEED42-PROTOCOL"
    ):
        raise ValueError("unexpected or unfrozen mixed-PPO protocol")
    protocol_dir = args.protocol.parent
    population_path = _verify_ref(protocol["outputs"]["population"])
    weights_path = _verify_ref(protocol["outputs"]["sampling_weights"])
    groups_path = _verify_ref(protocol["outputs"]["prompt_groups"])
    schedule_path = _verify_ref(protocol["outputs"]["fixed_rollout_schedule"])
    retrieval_path = _verify_ref(protocol["inputs"]["retrieval_contexts"])
    hard_silver_path = _verify_ref(protocol["inputs"]["hard_silver"])
    hard_kg_path = _verify_ref(protocol["inputs"]["hard_question_kg"])
    if args.protocol != protocol_dir / "protocol.json":
        raise ValueError("protocol path must be its canonical protocol.json location")

    raw_paths = {dataset: args.data_root / dataset / "train.jsonl" for dataset in DATASETS}
    addendum = freeze_raw_source_addendum(
        protocol_path=args.protocol,
        raw_paths=raw_paths,
        output_dir=args.raw_source_addendum_dir,
    )
    for dataset, identity in addendum["raw_train"].items():
        if sha256_file(raw_paths[dataset]) != identity["sha256"]:
            raise ValueError(f"raw source changed after addendum freeze: {dataset}")

    population = read_jsonl(population_path)
    if len(population) != 1799:
        raise ValueError(f"expected 1799 population rows, got {len(population)}")
    raw_by_key = load_selected_raw(raw_paths, population)
    retrieval_by_key = _by_question_key(read_jsonl(retrieval_path), "canonical retrieval")
    hard_silver_by_key = {
        f"{row['dataset']}::{row['qid']}": row for row in read_jsonl(hard_silver_path)
    }
    hard_kg_by_key = _by_question_key(read_jsonl(hard_kg_path), "hard question KG")
    if len(hard_silver_by_key) != 208 or len(hard_kg_by_key) != 208:
        raise ValueError("hard source is no longer exactly 208 identities")

    silver_rows: list[dict[str, Any]] = []
    kg_rows: list[dict[str, Any]] = []
    source_passage_counts: Counter[str] = Counter()
    multi_alias_count = 0
    for identity in population:
        key = str(identity["question_key"])
        raw = raw_by_key[key]
        _gold, aliases = primary_gold_answers(raw)
        multi_alias_count += int(len(aliases) > 1)
        if identity["process_reward_eligible"]:
            source_silver = hard_silver_by_key.get(key)
            source_kg = hard_kg_by_key.get(key)
            if source_silver is None or source_kg is None:
                raise ValueError(f"eligible hard identity missing source: {key}")
            if source_kg.get("question_sha256") != identity["question_sha256"]:
                raise ValueError(f"hard KG question hash mismatch: {key}")
            kg = list(source_kg.get("kg_subgraph") or [])
            runtime = {
                "question_key": key,
                "query_plan": source_kg.get("query_plan") or {},
                "provenance": source_kg.get("provenance") or {},
                "execution": source_kg.get("execution") or {},
            }
            if not is_automatic_proofkg(runtime, kg):
                raise ValueError(f"hard source is not complete Gold-free automatic ProofKG: {key}")
            planned = list(runtime["query_plan"].get("hops") or [])
            executed = list(runtime["execution"].get("hops") or [])
            if len(executed) < len(planned) or source_kg.get("runtime_error") is not None:
                raise ValueError(f"hard source has incomplete execution: {key}")
            record = dict(source_kg)
            provenance = dict(record.get("provenance") or {})
            provenance.update({
                "mixed_ppo_data_version": "mixed-ppo-three-dataset-v1",
                "process_reward_eligible": True,
                "failed_qpeg_or_saeg_p_edges_included": False,
            })
            record["provenance"] = provenance
            record["process_reward_eligible"] = True
            passages = list(source_silver.get("retrieved_passages") or [])
            source_passage_counts["hard_2wiki_frozen_train_context"] += 1
        else:
            retrieval = retrieval_by_key.get(key)
            if retrieval is None or retrieval.get("question_sha256") != identity["question_sha256"]:
                raise ValueError(f"ordinary canonical retrieval join mismatch: {key}")
            record = make_outcome_only_kg_record(identity)
            kg = []
            passages = list(retrieval.get("passages") or [])
            source_passage_counts["canonical_wiki18_rrf_reranked_top10"] += 1
        validate_question_kg_record(record)
        if bool(record.get("process_reward_eligible")) != bool(identity["process_reward_eligible"]):
            raise ValueError(f"process eligibility marker mismatch: {key}")
        row = build_silver_row(
            identity,
            raw=raw,
            retrieved_passages=passages,
            kg_subgraph=kg,
        )
        silver_rows.append(row)
        kg_rows.append(record)

    # Validate the exact runtime join using the production loader/applicator.
    # This mutates only fresh in-memory trajectory objects, never the source or
    # derived JSONL rows about to be written.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "silver_train": args.output_dir / "silver_train.jsonl",
        "question_kg_records": args.output_dir / "question_kg_records.jsonl",
        "sampling_weights": args.output_dir / "sampling_weights.jsonl",
        "prompt_groups": args.output_dir / "prompt_groups.jsonl",
        "fixed_rollout_schedule": args.output_dir / "fixed_rollout_schedule.jsonl",
    }
    write_jsonl(outputs["silver_train"], silver_rows)
    write_jsonl(outputs["question_kg_records"], kg_rows)
    shutil.copyfile(weights_path, outputs["sampling_weights"])
    shutil.copyfile(groups_path, outputs["prompt_groups"])
    shutil.copyfile(schedule_path, outputs["fixed_rollout_schedule"])

    reader = SilverDatasetReader(outputs["silver_train"])
    records = read_question_kg_records(outputs["question_kg_records"])
    join_stats = apply_training_question_kg(
        reader.accepted(), records, min_coverage=1.0, require_nonempty=False,
    ).to_dict()
    runtime_eligible = sum(
        is_automatic_proofkg(
            trajectory.metadata.get("question_kg_runtime") or {},
            trajectory.kg_subgraph,
        )
        for trajectory in reader.accepted()
    )
    explicit_eligible = sum(bool(row["metadata"]["process_reward_eligible"]) for row in silver_rows)
    no_graph = sum(not row["kg_subgraph"] for row in silver_rows)
    output_keys = {f"{row['dataset']}::{row['qid']}" for row in silver_rows}
    record_keys = {str(row["question_key"]) for row in kg_rows}
    schedule = read_jsonl(outputs["fixed_rollout_schedule"])
    scheduled_eligible = sum(bool(row["process_reward_eligible"]) for row in schedule)
    gates = {
        "population_1799_unique": len(silver_rows) == len(output_keys) == 1799,
        "dataset_unique_counts_exact": Counter(row["dataset"] for row in silver_rows) == Counter({
            "hotpotqa": 600, "2wikimultihopqa": 600, "musique": 599,
        }),
        "identity_question_kg_join_rate_1": output_keys == record_keys and join_stats["coverage_rate"] == 1.0,
        "all_gold_outcome_labels_nonempty": all(str(row["metadata"]["gold_answer"]).strip() for row in silver_rows),
        "all_steps_empty": all(row["steps"] == [] for row in silver_rows),
        "all_exactly_ten_passages": all(len(row["retrieved_passages"]) == 10 for row in silver_rows),
        "hard_eligible_exact_208": runtime_eligible == explicit_eligible == 208,
        "outcome_only_empty_kg_exact_1591": no_graph == 1591,
        "failed_qpeg_or_saeg_p_edges_zero": all(
            row["metadata"].get("failed_qpeg_or_saeg_p_edges_included") is False
            and not row.get("passage_evidence")
            and row.get("evidence_mode") is None
            for row in silver_rows
        ),
        "schedule_7200_k4": len(schedule) == 7200 and all(
            len({(row["dataset"], row["qid"]) for row in schedule[start:start + 4]}) == 1
            for start in range(0, len(schedule), 4)
        ),
        "scheduled_process_eligible_trajectories_1200": scheduled_eligible == 1200,
        "frozen_schedule_and_weights_byte_identical": (
            sha256_file(outputs["fixed_rollout_schedule"]) == sha256_file(schedule_path)
            and sha256_file(outputs["sampling_weights"]) == sha256_file(weights_path)
        ),
    }
    if not all(gates.values()):
        failure_path = args.output_dir / "FAILED_MATERIALIZATION_GATES.json"
        failure_path.write_text(json.dumps(gates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"mixed PPO materialization gates failed: {gates}")

    report = {
        "schema_version": "mixed-ppo-three-dataset-materialization-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "counts": {
            "unique_population": len(silver_rows),
            "unique_by_dataset": dict(sorted(Counter(row["dataset"] for row in silver_rows).items())),
            "unique_by_route": dict(sorted(Counter(row["metadata"]["mixed_ppo_route"] for row in silver_rows).items())),
            "process_reward_eligible_unique": explicit_eligible,
            "outcome_only_no_graph_unique": no_graph,
            "selected_rows_with_multiple_gold_aliases": multi_alias_count,
            "passage_source": dict(sorted(source_passage_counts.items())),
            "scheduled_prompt_groups": 1800,
            "scheduled_trajectories": len(schedule),
            "scheduled_process_eligible_prompt_groups": scheduled_eligible // 4,
            "scheduled_process_eligible_trajectories": scheduled_eligible,
        },
        "gates": gates,
        "question_kg_join_stats": join_stats,
        "scientific_boundary": {
            "train_only_gold_answer_use": "outcome label only",
            "gold_answer_inserted_into_prompt_fields": False,
            "natural_answer_mentions_inside_retrieved_wikipedia_text": "allowed and not equivalent to Gold-field leakage",
            "gold_process_steps_copied": 0,
            "failed_qpeg_or_saeg_p_edges_consumed": False,
            "non_2wiki_hard_process_reward": False,
            "training_started": False,
            "schedule_semantics": (
                "1800 prompt groups is a fixed exposure schedule, not a full pass over all 1799 unique prompts; "
                "it schedules 1674 unique identities and intentionally repeats hard recovery/MuSiQue rows."
            ),
            "runtime_readiness": (
                "NOT_RUNNABLE until phase3_ppo actually consumes fixed_rollout_schedule_path and a paired CPU/GPU preflight passes"
            ),
        },
        "inputs": {
            "protocol": ref(args.protocol),
            "raw_source_addendum": ref(args.raw_source_addendum_dir / "raw_source_addendum.json"),
            "raw_train": {dataset: ref(path) for dataset, path in raw_paths.items()},
            "canonical_retrieval": ref(retrieval_path),
            "hard_silver": ref(hard_silver_path),
            "hard_question_kg": ref(hard_kg_path),
        },
        "outputs": {name: ref(path) for name, path in outputs.items()},
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.output_dir, status=STATUS, extra={
        "phase": "mixed_ppo_data_materialization",
        "experiment_id": EXPERIMENT_ID,
        "report_sha256": sha256_file(report_path),
    })
    print(json.dumps({
        "status": STATUS,
        "counts": report["counts"],
        "gates": gates,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
