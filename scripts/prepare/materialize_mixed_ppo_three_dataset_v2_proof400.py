#!/usr/bin/env python
"""Materialize the frozen mixed PPO v2 Proof400 population (CPU only).

Train Gold answers are joined only as outcome labels after the answer-free
population has been frozen.  The prompt-facing question, ten passages, and KG
never receive an explicit Gold field.  Exactly 400 2Wiki rows carry complete,
Gold-free automatic ProofKG execution traces; all other rows carry an explicit
identity-safe empty graph and are process-reward ineligible.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.kg.question_kg import validate_question_kg_record
from kgproweight.kg.training_question_kg import apply_training_question_kg, read_question_kg_records
from kgproweight.reward.proofkg_process import (
    is_automatic_proofkg,
    is_identity_safe_automatic_proofkg,
)
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_mixed_ppo_three_dataset_v1 import read_jsonl, sha256_file
from scripts.prepare.freeze_mixed_ppo_three_dataset_v2_proof400 import (
    EXPERIMENT_ID as PROTOCOL_EXPERIMENT_ID,
    FAMILY_VERSION,
    STATUS as PROTOCOL_STATUS,
)
from scripts.prepare.materialize_mixed_ppo_three_dataset_v1 import (
    _by_question_key,
    build_silver_row,
    load_selected_raw,
    make_outcome_only_kg_record,
    primary_gold_answers,
)


EXPERIMENT_ID = "MIXED-PPO-THREE-DATASET-V2-PROOF400-N1799-K4-7200-SEED42-DATA"
STATUS = "COMPLETE_DATA_NOT_TRAINED"
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
QTYPES = ("inference", "comparison", "compositional", "bridge_comparison")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def ref(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def verify_ref(identity: Mapping[str, Any]) -> Path:
    path = Path(str(identity["path"]))
    if not path.is_file() or sha256_file(path) != str(identity["sha256"]):
        raise ValueError(f"frozen input missing/hash mismatch: {path}")
    return path


def freeze_raw_addendum(protocol_path: Path, raw_paths: Mapping[str, Path], out: Path) -> dict[str, Any]:
    expected = {
        "schema_version": "mixed-ppo-raw-source-addendum-v2-proof400",
        "experiment_id": f"{EXPERIMENT_ID}-RAW-SOURCE-ADDENDUM",
        "status": "FROZEN_BEFORE_GOLD_LABEL_JOIN",
        "parent_protocol": ref(protocol_path),
        "raw_train": {dataset: ref(path) for dataset, path in raw_paths.items()},
        "scientific_boundary": "Binds raw train files only; it does not select or modify Gold labels.",
    }
    if out.exists():
        existing = json.loads((out / "raw_source_addendum.json").read_text(encoding="utf-8"))
        for key in ("schema_version", "experiment_id", "status", "parent_protocol", "raw_train"):
            if existing.get(key) != expected.get(key):
                raise ValueError(f"existing raw addendum changed at {key}")
        return existing
    out.mkdir(parents=True, exist_ok=False)
    payload = {**expected, "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    path = out / "raw_source_addendum.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(out, status=payload["status"], extra={
        "phase": "mixed_ppo_v2_raw_source_freeze",
        "experiment_id": payload["experiment_id"],
        "addendum_sha256": sha256_file(path),
    })
    return payload


def _validate_identity(identity: Mapping[str, Any], record: Mapping[str, Any], label: str) -> None:
    validate_question_kg_record(record)
    for field in ("question_key", "dataset", "qid", "question", "question_sha256"):
        if str(record.get(field)) != str(identity.get(field)):
            raise ValueError(f"{label} identity mismatch at {field}: {identity['question_key']}")


def _complete_record(
    identity: Mapping[str, Any], base: Mapping[str, Any], trace: Mapping[str, Any] | None,
) -> dict[str, Any]:
    record = dict(base)
    _validate_identity(identity, record, "ProofKG")
    if trace is not None:
        _validate_identity(identity, trace, "runtime trace")
        for field in ("kg_subgraph", "query_plan"):
            if trace.get(field) != record.get(field):
                raise ValueError(f"runtime/base {field} mismatch: {identity['question_key']}")
        record["execution"] = dict(trace.get("execution") or {})
        record["runtime_error"] = trace.get("runtime_error")
    execution = record.get("execution") or {}
    planned = list((record.get("query_plan") or {}).get("hops") or [])
    executed = list(execution.get("hops") or [])
    kg = list(record.get("kg_subgraph") or [])
    provenance = dict(record.get("provenance") or {})
    if (
        record.get("runtime_error") is not None
        or provenance.get("gold_access") is not False
        or provenance.get("complete_plan_execution") is not True
        or execution.get("complete_plan_execution") is not True
        or not planned
        or len(executed) < len(planned)
        or any(not list(hop.get("matches") or []) for hop in executed[: len(planned)])
        or not is_automatic_proofkg(record, kg)
    ):
        raise ValueError(f"incomplete/non-Gold-free execution: {identity['question_key']}")
    provenance.update({
        "mixed_ppo_data_version": "mixed-ppo-three-dataset-v2-proof400",
        "family_version": FAMILY_VERSION,
        "process_reward_eligible": True,
        "failed_qpeg_or_saeg_p_edges_included": False,
    })
    record["provenance"] = provenance
    record["process_reward_eligible"] = True
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("outputs/audits/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_protocol/protocol.json"))
    parser.add_argument("--raw_source_addendum_dir", type=Path, default=Path("outputs/audits/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_raw_source_addendum"))
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--auto_silver", type=Path, default=Path("data/silver_data/automatic_proofkg_2wiki_train_k4_v1/silver_train.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/silver_data/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite versioned data: {args.output_dir}")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != PROTOCOL_STATUS or protocol.get("experiment_id") != PROTOCOL_EXPERIMENT_ID:
        raise ValueError("unexpected/unfrozen mixed PPO v2 protocol")
    population_path = verify_ref(protocol["outputs"]["population"])
    weights_path = verify_ref(protocol["outputs"]["sampling_weights"])
    groups_path = verify_ref(protocol["outputs"]["prompt_groups"])
    schedule_path = verify_ref(protocol["outputs"]["fixed_rollout_schedule"])
    auto_kg_path = verify_ref(protocol["inputs"]["complete_kg"])
    runtime_path = verify_ref(protocol["inputs"]["runtime_details"])
    v1_protocol_path = verify_ref(protocol["inputs"]["v1_protocol"])
    v1 = json.loads(v1_protocol_path.read_text(encoding="utf-8"))
    retrieval_path = verify_ref(v1["inputs"]["retrieval_contexts"])
    hard_silver_path = verify_ref(v1["inputs"]["hard_silver"])
    hard_kg_path = verify_ref(v1["inputs"]["hard_question_kg"])
    if not args.auto_silver.is_file():
        raise FileNotFoundError(args.auto_silver)

    population = read_jsonl(population_path)
    raw_paths = {dataset: args.data_root / dataset / "train.jsonl" for dataset in DATASETS}
    addendum = freeze_raw_addendum(args.protocol, raw_paths, args.raw_source_addendum_dir)
    raw_by_key = load_selected_raw(raw_paths, population)
    retrieval = _by_question_key(read_jsonl(retrieval_path), "canonical retrieval")
    hard_silver = {f"{row['dataset']}::{row['qid']}": row for row in read_jsonl(hard_silver_path)}
    hard_kg = _by_question_key(read_jsonl(hard_kg_path), "hard ProofKG")
    auto_silver = {f"{row['dataset']}::{row['qid']}": row for row in read_jsonl(args.auto_silver)}
    auto_kg = _by_question_key(read_jsonl(auto_kg_path), "automatic ProofKG")
    runtime = _by_question_key(read_jsonl(runtime_path), "stage3 runtime")

    silver_rows: list[dict[str, Any]] = []
    kg_rows: list[dict[str, Any]] = []
    passage_sources: Counter[str] = Counter()
    multi_alias = 0
    for identity in population:
        key = str(identity["question_key"])
        raw = raw_by_key[key]
        _gold, aliases = primary_gold_answers(raw)
        multi_alias += int(len(aliases) > 1)
        if identity["process_reward_eligible"]:
            source = str(identity["proof_source"])
            if source == "automatic_proofkg_2wiki_hard_contrastive_v1":
                base, source_silver, trace = hard_kg.get(key), hard_silver.get(key), None
                passage_label = "hard_2wiki_frozen_train_context"
            elif source == "automatic_proofkg_2wiki_train_k4_v1":
                base, source_silver, trace = auto_kg.get(key), auto_silver.get(key), runtime.get(key)
                passage_label = "automatic_2wiki_frozen_train_context"
            else:
                raise ValueError(f"unknown eligible proof source: {source}")
            if base is None or source_silver is None:
                raise ValueError(f"eligible source join miss: {key}")
            record = _complete_record(identity, base, trace)
            kg = list(record["kg_subgraph"])
            passages = list(source_silver.get("retrieved_passages") or [])
            passage_sources[passage_label] += 1
        else:
            context = retrieval.get(key)
            if context is None or context.get("question_sha256") != identity.get("question_sha256"):
                raise ValueError(f"canonical retrieval join mismatch: {key}")
            record = make_outcome_only_kg_record(identity)
            provenance = dict(record["provenance"])
            provenance.update({
                "builder_version": "mixed-ppo-outcome-only-empty-kg-v2-proof400",
                "mixed_ppo_data_version": "mixed-ppo-three-dataset-v2-proof400",
                "family_version": FAMILY_VERSION,
            })
            record["provenance"] = provenance
            kg = []
            passages = list(context.get("passages") or [])
            passage_sources["canonical_wiki18_rrf_reranked_top10"] += 1
        row = build_silver_row(identity, raw=raw, retrieved_passages=passages, kg_subgraph=kg)
        row["metadata"].update({
            "question_type": identity.get("question_type") or row["metadata"].get("question_type"),
            "family_version": FAMILY_VERSION,
            "family_sha256": identity["family_sha256"],
            "mixed_ppo_data_version": "mixed-ppo-three-dataset-v2-proof400",
            "mixed_ppo_route": identity["route"],
            "proof_source": identity["proof_source"],
        })
        silver_rows.append(row)
        kg_rows.append(record)

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
    join = apply_training_question_kg(reader.accepted(), records, min_coverage=1.0, require_nonempty=False).to_dict()
    runtime_eligible = sum(
        is_identity_safe_automatic_proofkg(
            trajectory.metadata.get("question_kg_runtime") or {}, trajectory.kg_subgraph,
            dataset=trajectory.dataset, qid=trajectory.qid,
        )
        for trajectory in reader.accepted()
    )
    explicit_eligible = sum(bool(row["metadata"]["process_reward_eligible"]) for row in silver_rows)
    empty = sum(not row["kg_subgraph"] for row in silver_rows)
    schedule = read_jsonl(outputs["fixed_rollout_schedule"])
    scheduled_eligible = sum(bool(row["process_reward_eligible"]) for row in schedule)
    keys = {f"{row['dataset']}::{row['qid']}" for row in silver_rows}
    record_keys = {row["question_key"] for row in kg_rows}
    qtypes = Counter(row["metadata"]["question_type"] for row in silver_rows if row["metadata"]["process_reward_eligible"])
    gates = {
        "population_1799_unique": len(silver_rows) == len(keys) == 1799,
        "dataset_counts_exact": Counter(row["dataset"] for row in silver_rows) == Counter({"hotpotqa": 600, "2wikimultihopqa": 600, "musique": 599}),
        "identity_question_kg_join_rate_1": keys == record_keys and join["coverage_rate"] == 1.0,
        "complete_proofkg_eligible_exact_400": explicit_eligible == runtime_eligible == 400,
        "proofkg_qtype_100_each": qtypes == Counter({qtype: 100 for qtype in QTYPES}),
        "outcome_only_empty_kg_exact_1399": empty == 1399,
        "all_gold_outcome_labels_nonempty": all(str(row["metadata"]["gold_answer"]).strip() for row in silver_rows),
        "all_steps_empty": all(row["steps"] == [] for row in silver_rows),
        "all_exactly_ten_passages": all(len(row["retrieved_passages"]) == 10 for row in silver_rows),
        "failed_qpeg_or_saeg_p_edges_zero": all(
            row["metadata"].get("failed_qpeg_or_saeg_p_edges_included") is False
            and not row.get("passage_evidence") and row.get("evidence_mode") is None
            for row in silver_rows
        ),
        "schedule_7200_k4": len(schedule) == 7200 and all(
            len({(row["dataset"], row["qid"]) for row in schedule[i:i + 4]}) == 1
            for i in range(0, len(schedule), 4)
        ),
        "scheduled_process_eligible_1600": scheduled_eligible == 1600,
        "frozen_schedule_and_weights_identical": (
            sha256_file(outputs["fixed_rollout_schedule"]) == sha256_file(schedule_path)
            and sha256_file(outputs["sampling_weights"]) == sha256_file(weights_path)
        ),
    }
    if not all(gates.values()):
        (args.output_dir / "FAILED_MATERIALIZATION_GATES.json").write_text(
            json.dumps(gates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(gates)

    report = {
        "schema_version": "mixed-ppo-three-dataset-materialization-v2-proof400",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "counts": {
            "unique_population": len(silver_rows),
            "unique_by_dataset": dict(sorted(Counter(row["dataset"] for row in silver_rows).items())),
            "unique_by_route": dict(sorted(Counter(row["metadata"]["mixed_ppo_route"] for row in silver_rows).items())),
            "process_reward_eligible_unique": explicit_eligible,
            "process_reward_eligible_by_question_type": dict(sorted(qtypes.items())),
            "outcome_only_no_graph_unique": empty,
            "selected_rows_with_multiple_gold_aliases": multi_alias,
            "passage_source": dict(sorted(passage_sources.items())),
            "scheduled_prompt_groups": len(schedule) // 4,
            "scheduled_trajectories": len(schedule),
            "scheduled_process_eligible_prompt_groups": scheduled_eligible // 4,
            "scheduled_process_eligible_trajectories": scheduled_eligible,
        },
        "gates": gates,
        "question_kg_join_stats": join,
        "scientific_boundary": {
            "train_only_gold_answer_use": "outcome label only",
            "gold_answer_inserted_into_prompt_fields": False,
            "gold_process_steps_copied": 0,
            "failed_qpeg_or_saeg_p_edges_consumed": False,
            "protected_a_qid_and_lexical_family_overlap": 0,
            "v1_family_gate_status": "SUPERSEDED_NAMESPACE_INCOMPARABLE",
            "training_started": False,
        },
        "inputs": {
            "protocol": ref(args.protocol),
            "raw_source_addendum": ref(args.raw_source_addendum_dir / "raw_source_addendum.json"),
            "raw_train": {dataset: ref(path) for dataset, path in raw_paths.items()},
            "canonical_retrieval": ref(retrieval_path),
            "hard_silver": ref(hard_silver_path), "hard_question_kg": ref(hard_kg_path),
            "automatic_silver": ref(args.auto_silver), "automatic_question_kg": ref(auto_kg_path),
            "stage3_runtime_details": ref(runtime_path),
        },
        "outputs": {name: ref(path) for name, path in outputs.items()},
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.output_dir, status=STATUS, extra={
        "phase": "mixed_ppo_v2_data_materialization",
        "experiment_id": EXPERIMENT_ID,
        "report_sha256": sha256_file(report_path),
    })
    print(json.dumps({"status": STATUS, "counts": report["counts"], "gates": gates}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
