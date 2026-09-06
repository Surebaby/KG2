#!/usr/bin/env python3
"""Bind the frozen official-raw 2Wiki cohort to a planner execution input.

The upstream candidate release intentionally contains only question identity
and answer-free stratification metadata.  The mature planner runner additionally
requires ``row_id`` and ``question_key``.  This script derives those two fields
without reading the official raw source, answers, evidence, passages, or Gold
steps.  It also binds the final clean replay-v2 identity ledger and proves that
the candidate cohort remains disjoint from it.

No model is loaded and no planner inference, retrieval, network access, or
training is performed here.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.utils.logging import artifact_identity, dump_manifest


DATASET = "2wikimultihopqa"
EXPECTED_N = 1500
EXPECTED_QUOTAS = {
    "bridge_comparison": 390,
    "comparison": 390,
    "compositional": 389,
    "inference": 331,
}
EXPECTED_CANDIDATE_FIELDS = frozenset(
    {
        "schema_version",
        "dataset",
        "qid",
        "question",
        "question_sha256",
        "family_version",
        "family_sha256",
        "question_type",
        "target_type",
        "gold_access",
    }
)
RUNTIME_FIELDS = (
    "row_id",
    "question_key",
    "dataset",
    "qid",
    "question",
    "question_sha256",
    "target_type",
    "question_type",
    "family_version",
    "family_sha256",
    "gold_access",
)
PROHIBITED_FIELDS = frozenset(
    {
        "answer",
        "answers",
        "golden_answers",
        "target",
        "steps",
        "supporting_facts",
        "evidence",
        "evidences",
        "paragraph_text",
        "retrieved_passages",
        "context",
        "decomposition",
        "question_decomposition",
    }
)

DEFAULT_CANDIDATE_DIR = Path(
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "seed42_preregistration"
)
DEFAULT_REPLAY_DIR = Path(
    "data/silver_data/sft_replay_legacy_strong_train_rendered3to5_"
    "n2000_seed42_v2"
)
DEFAULT_HISTORICAL_PROTOCOL = Path(
    "outputs/audits/automatic_proofkg_2wiki_train_k4_v1_n1500_seed42_"
    "preregistration/protocol.json"
)
DEFAULT_ADAPTER = Path("checkpoints/query_planner_learned_scale_v1_1_seed42/final")
DEFAULT_CONFIG = Path("configs/training/query_planner_learned_scale_v1_1_seed42.yaml")
DEFAULT_OUT = Path(
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "seed42_planner_execution_v1_preregistration"
)
EXPERIMENT_ID = (
    "2WIKI-PROOFKG-OFFICIAL-RAW-V2-CANDIDATE-POOL-N1500-SEED42-"
    "PLANNER-EXECUTION-V1"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_answer_free(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        present = PROHIBITED_FIELDS.intersection(map(str, value))
        if present:
            raise ValueError(f"prohibited fields at {location}: {sorted(present)}")
        for key, child in value.items():
            _assert_answer_free(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_answer_free(child, location=f"{location}[{index}]")


def _identity_sets(rows: Iterable[Mapping[str, Any]]) -> dict[str, set[tuple[str, str]]]:
    result: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        dataset = str(row.get("dataset") or "").strip().lower()
        if not dataset:
            raise ValueError("identity record has empty dataset")
        for name in ("qid", "question_sha256", "family_sha256"):
            value = str(row.get(name) or "").strip()
            if not value:
                raise ValueError(f"identity record lacks {name}")
            result[name].add((dataset, value))
    return result


def _derive_runtime_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    derived: list[dict[str, Any]] = []
    for index, raw in enumerate(rows, start=1):
        row = dict(raw)
        if set(row) != EXPECTED_CANDIDATE_FIELDS:
            raise ValueError(
                f"candidate field mismatch at row {index}: "
                f"{sorted(set(row) ^ EXPECTED_CANDIDATE_FIELDS)}"
            )
        _assert_answer_free(row, location=f"candidate[{index}]")
        dataset = str(row["dataset"]).strip().lower()
        qid = str(row["qid"]).strip()
        question = str(row["question"]).strip()
        if dataset != DATASET or not qid or not question:
            raise ValueError(f"invalid candidate identity at row {index}")
        if question_sha256(question) != str(row["question_sha256"]):
            raise ValueError(f"question hash mismatch for {dataset}::{qid}")
        if row["gold_access"] is not False:
            raise ValueError(f"gold_access must be false for {dataset}::{qid}")
        if str(row["target_type"]) != "relation_graph":
            raise ValueError(f"non relation_graph target for {dataset}::{qid}")
        runtime = {
            "row_id": f"OFFICIAL-RAW-V2-{index:04d}",
            "question_key": question_key(dataset, qid),
            **{key: row[key] for key in RUNTIME_FIELDS if key not in {"row_id", "question_key"}},
        }
        if set(runtime) != set(RUNTIME_FIELDS):
            raise AssertionError("runtime field construction drifted")
        derived.append(runtime)
    return derived


def _balanced_preflight(rows: list[Mapping[str, Any]], *, per_type: int = 2) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_type"])].append(dict(row))
    selected: list[dict[str, Any]] = []
    for qtype in EXPECTED_QUOTAS:
        if len(grouped[qtype]) < per_type:
            raise ValueError(f"not enough {qtype} rows for preflight")
        selected.extend(grouped[qtype][:per_type])
    return selected


def materialize(
    *,
    candidate_dir: Path,
    replay_dir: Path,
    historical_protocol_path: Path,
    adapter: Path,
    config: Path,
    output_dir: Path,
    experiment_id: str = EXPERIMENT_ID,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite versioned output: {output_dir}")
    cohort_path = candidate_dir / "cohort.question_only.jsonl"
    candidate_protocol_path = candidate_dir / "protocol.json"
    replay_records_path = replay_dir / "selection_records.jsonl"
    replay_report_path = replay_dir / "report.json"
    required = (
        cohort_path,
        candidate_protocol_path,
        replay_records_path,
        replay_report_path,
        historical_protocol_path,
        adapter / "adapter_model.safetensors",
        config,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    candidate_protocol = json.loads(candidate_protocol_path.read_text(encoding="utf-8"))
    historical = json.loads(historical_protocol_path.read_text(encoding="utf-8"))
    replay_report = json.loads(replay_report_path.read_text(encoding="utf-8"))
    if candidate_protocol.get("status") != (
        "FROZEN_GOLD_FREE_BEFORE_PLANNER_NOT_MATERIALIZED_NOT_TRAINED"
    ):
        raise ValueError("candidate protocol is not a frozen pre-planner release")
    if candidate_protocol.get("execution_boundary", {}).get("planner_started") is not False:
        raise ValueError("candidate protocol planner boundary is not false")
    expected_cohort_sha = str(candidate_protocol.get("output", {}).get("sha256") or "")
    if expected_cohort_sha != _sha256(cohort_path):
        raise ValueError("candidate protocol/cohort SHA256 mismatch")
    if replay_report.get("status") != "COMPLETE_DATA_NOT_TRAINED":
        raise ValueError("final replay-v2 release is not COMPLETE_DATA_NOT_TRAINED")
    replay_ref = replay_report.get("outputs", {}).get("selection_records", {})
    if str(replay_ref.get("sha256") or "") != _sha256(replay_records_path):
        raise ValueError("replay-v2 report/selection-record SHA256 mismatch")

    historical_planner = historical.get("planner") or {}
    if str(historical_planner.get("adapter_model_sha256") or "") != _sha256(
        adapter / "adapter_model.safetensors"
    ):
        raise ValueError("planner adapter differs from the mature n1500 protocol")
    if str(historical_planner.get("config_sha256") or "") != _sha256(config):
        raise ValueError("planner config differs from the mature n1500 protocol")
    if historical_planner.get("decoding") != "greedy":
        raise ValueError("historical planner protocol was not greedy")

    candidates = _read_jsonl(cohort_path)
    replay = _read_jsonl(replay_records_path)
    runtime = _derive_runtime_rows(candidates)
    counts = Counter(str(row["question_type"]) for row in runtime)
    candidate_ids, replay_ids = _identity_sets(runtime), _identity_sets(replay)
    overlaps = {
        key: len(candidate_ids[key].intersection(replay_ids[key]))
        for key in ("qid", "question_sha256", "family_sha256")
    }
    gates = {
        "candidate_n_exact": len(runtime) == EXPECTED_N,
        "question_type_quotas_exact": counts == Counter(EXPECTED_QUOTAS),
        "dataset_scoped_qid_unique": len(candidate_ids["qid"]) == len(runtime),
        "dataset_scoped_question_hash_unique": len(candidate_ids["question_sha256"]) == len(runtime),
        "question_key_unique": len({row["question_key"] for row in runtime}) == len(runtime),
        "row_id_unique": len({row["row_id"] for row in runtime}) == len(runtime),
        "gold_access_false": all(row["gold_access"] is False for row in runtime),
        "relation_graph_only": all(row["target_type"] == "relation_graph" for row in runtime),
        "final_replay_v2_overlap_zero": all(value == 0 for value in overlaps.values()),
        "adapter_matches_mature_protocol": True,
        "config_matches_mature_protocol": True,
        "no_planner_or_training_started": True,
    }
    if not all(gates.values()):
        raise ValueError(f"planner execution input gates failed: {gates}")

    output_dir.mkdir(parents=True, exist_ok=False)
    runtime_path = output_dir / "planner_input.question_only.jsonl"
    preflight_path = output_dir / "planner_input.preflight8.question_only.jsonl"
    _write_jsonl(runtime_path, runtime)
    _write_jsonl(preflight_path, _balanced_preflight(runtime, per_type=2))
    generated_at = datetime.now(timezone.utc).isoformat()
    protocol = {
        "schema_version": "2wiki-official-raw-planner-execution-v1",
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at,
        "status": "FROZEN_GOLD_FREE_PLANNER_EXECUTION_NOT_RUN_NOT_TRAINED",
        "scope": "2Wiki official-raw candidate pool n1500; answer-free planner inference only",
        "generation": {
            "runner": "scripts/eval/generate_query_plans_unseen.py",
            "decode": "greedy",
            "max_new_tokens": 512,
            "batch_size": 8,
            "preflight_n": 8,
            "preflight_per_question_type": 2,
        },
        "planner": {
            "adapter": artifact_identity(adapter),
            "adapter_model": artifact_identity(adapter / "adapter_model.safetensors"),
            "config": artifact_identity(config),
            "historical_mature_n1500_protocol": artifact_identity(historical_protocol_path),
        },
        "input_binding": {
            "candidate_protocol": artifact_identity(candidate_protocol_path),
            "candidate_cohort": artifact_identity(cohort_path),
            "final_replay_v2_report": artifact_identity(replay_report_path),
            "final_replay_v2_identity_records": artifact_identity(replay_records_path),
            "final_replay_v2_overlap": overlaps,
        },
        "runtime_projection": {
            "derived_fields": ["row_id", "question_key"],
            "copied_answer_free_fields": [
                key for key in RUNTIME_FIELDS if key not in {"row_id", "question_key"}
            ],
            "prohibited_fields": sorted(PROHIBITED_FIELDS),
            "official_raw_source_opened": False,
            "gold_or_evidence_opened": False,
        },
        "counts": {
            "n": len(runtime),
            "by_question_type": dict(sorted(counts.items())),
        },
        "gates": gates,
        "outputs": {
            "planner_input": artifact_identity(runtime_path),
            "preflight_input": artifact_identity(preflight_path),
        },
        "execution_boundary": {
            "planner_started": False,
            "retrieval_started": False,
            "network_accessed": False,
            "training_started": False,
        },
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": "2wiki-official-raw-planner-input-report-v1",
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at,
        "status": protocol["status"],
        "counts": protocol["counts"],
        "final_replay_v2_overlap": overlaps,
        "gates": gates,
        "outputs": {
            **protocol["outputs"],
            "protocol": artifact_identity(protocol_path),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        output_dir,
        status=protocol["status"],
        extra={
            "phase": "materialize_2wiki_official_raw_planner_inputs_v1",
            "experiment_id": experiment_id,
            "planner_started": False,
            "training_started": False,
            "gates": gates,
            "report": artifact_identity(report_path),
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument(
        "--historical-protocol", type=Path, default=DEFAULT_HISTORICAL_PROTOCOL
    )
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    args = parser.parse_args()
    report = materialize(
        candidate_dir=args.candidate_dir,
        replay_dir=args.replay_dir,
        historical_protocol_path=args.historical_protocol,
        adapter=args.adapter,
        config=args.config,
        output_dir=args.out,
        experiment_id=args.experiment_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
