#!/usr/bin/env python
"""Freeze QPEG pilot A/B inputs; Gold is joined only after graph construction."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.kg.qpeg import validate_qpeg_record
from kgproweight.utils.logging import dump_manifest


DEFAULT_EXPERIMENT_ID = "QPEG-V1-PILOT50X3-AB-INPUTS-SEED42"
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "kg_subgraph"}


def _adapter_hashes(path: Path) -> dict[str, str]:
    return {
        "adapter_config_sha256": _sha256(path / "adapter_config.json"),
        "adapter_model_sha256": _sha256(path / "adapter_model.safetensors"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", type=Path,
        default=Path("outputs/audits/qpeg_v1_n1350_seed42_preregistration/pilot.question_only.jsonl"),
    )
    parser.add_argument(
        "--contexts", type=Path,
        default=Path("outputs/audits/qpeg_v1_pilot_confirmation_retrieval_seed42_canonical_v2/retrieval_contexts.jsonl"),
    )
    parser.add_argument(
        "--qpeg", type=Path,
        default=Path("data/derived/qpeg_v1_1_precision_n1350_seed42/question_graph_records.jsonl"),
    )
    parser.add_argument(
        "--adapter", type=Path,
        default=Path("checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"),
    )
    parser.add_argument("--base_model", type=Path, default=Path("models/llama3-8b"))
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/audits/qpeg_v1_1_precision_pilot50x3_ab_inputs_seed42"),
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--role", choices=["pilot", "confirmation"], default="pilot")
    parser.add_argument("--n_per_dataset", type=int, default=50)
    parser.add_argument("--experiment_id", default=DEFAULT_EXPERIMENT_ID)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite frozen A/B inputs: {args.out}")
    args.out.mkdir(parents=True)

    cohort = _read_jsonl(args.cohort)
    contexts = {
        str(row["question_key"]): row
        for row in _read_jsonl(args.contexts)
        if row.get("role") == args.role
    }
    qpeg = {
        str(row["question_key"]): row
        for row in _read_jsonl(args.qpeg)
        if row.get("role") == args.role
    }
    keys = [str(row["question_key"]) for row in cohort]
    expected_n = args.n_per_dataset * len(DATASETS)
    if len(cohort) != expected_n or len(set(keys)) != expected_n:
        raise ValueError(f"expected {expected_n} unique {args.role} rows, got {len(cohort)}/{len(set(keys))}")
    if set(keys) != set(contexts) or set(keys) != set(qpeg):
        raise ValueError("pilot cohort, retrieval contexts, and QPEG keys differ")

    # Validate all answer-free assets before opening any evaluation labels.
    for frozen in cohort:
        key = str(frozen["question_key"])
        context = contexts[key]
        graph = qpeg[key]
        if str(context["question"]).strip() != str(frozen["question"]).strip():
            raise ValueError(f"cohort/context question mismatch: {key}")
        if str(graph["question"]).strip() != str(frozen["question"]).strip():
            raise ValueError(f"cohort/QPEG question mismatch: {key}")
        validate_qpeg_record(graph, passages=context["passages"])
        if graph.get("gold_access") is not False:
            raise ValueError(f"QPEG gold_access is not false: {key}")

    selected = {dataset: set() for dataset in DATASETS}
    for row in cohort:
        selected[str(row["dataset"])].add(str(row["qid"]))
    gold: dict[str, list[str]] = {}
    raw_hashes: dict[str, str] = {}
    for dataset in DATASETS:
        path = args.data_root / dataset / "dev.jsonl"
        raw_hashes[str(path)] = _sha256(path)
        for row in _read_jsonl(path):
            qid = str(row.get("id") or "")
            if qid in selected[dataset]:
                key = f"{dataset}::{qid}"
                if str(row.get("question") or "").strip() != str(contexts[key]["question"]).strip():
                    raise ValueError(f"raw/context question mismatch: {key}")
                gold[key] = [str(value) for value in row.get("golden_answers") or []]
    if set(gold) != set(keys):
        raise ValueError(f"Gold join incomplete: {len(gold)}/{len(keys)}")

    arm_no_qpeg: list[dict[str, Any]] = []
    arm_qpeg: list[dict[str, Any]] = []
    for frozen in cohort:
        key = str(frozen["question_key"])
        context = contexts[key]
        graph = qpeg[key]
        common = {
            "row_id": key,
            "question_key": key,
            "dataset": str(frozen["dataset"]),
            "qid": str(frozen["qid"]),
            "question": str(frozen["question"]),
            "gold_answers": gold[key],
            "retrieved_passages": list(context["passages"]),
            "passages_sha256": str(context["passages_sha256"]),
            "qpeg_sha256": str(graph["qpeg_sha256"]),
            "qpeg_edge_count": len(graph["kg_subgraph"]),
            "role": args.role,
        }
        arm_no_qpeg.append({**common, "kg_subgraph": []})
        arm_qpeg.append({**common, "kg_subgraph": list(graph["kg_subgraph"])})
    if any(_projection(left) != _projection(right) for left, right in zip(arm_no_qpeg, arm_qpeg)):
        raise ValueError("paired inputs differ outside kg_subgraph")

    no_qpeg_path = args.out / "arm_no_qpeg.jsonl"
    qpeg_path = args.out / "arm_qpeg.jsonl"
    _write_jsonl(no_qpeg_path, arm_no_qpeg)
    _write_jsonl(qpeg_path, arm_qpeg)
    qid_order_sha256 = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    protocol = {
        "schema_version": "qpeg-pilot-ab-protocol-v1",
        "experiment_id": args.experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_INPUTS_NOT_EVALUATED",
        "scope": f"QPEG matched {args.role}{args.n_per_dataset}x3; strong SFT; same passages; only graph block differs",
        "n": len(cohort),
        "per_dataset_n": args.n_per_dataset,
        "arm_semantics": {"no_qpeg": "passages only", "qpeg": "same passages + QPEG-v1"},
        "qid_order_sha256": qid_order_sha256,
        "generation": {
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
            "top_k_passages": 10,
        },
        "models": {"strong_sft": {"path": str(args.adapter), **_adapter_hashes(args.adapter)}},
        "base_model": {
            "path": str(args.base_model),
            "config_sha256": _sha256(args.base_model / "config.json"),
            "model_index_sha256": _sha256(args.base_model / "model.safetensors.index.json"),
        },
        "inputs": {
            "cohort": {"path": str(args.cohort), "sha256": _sha256(args.cohort)},
            "contexts": {"path": str(args.contexts), "sha256": _sha256(args.contexts)},
            "qpeg": {"path": str(args.qpeg), "sha256": _sha256(args.qpeg)},
            "arm_no_qpeg": {"path": str(no_qpeg_path), "sha256": _sha256(no_qpeg_path)},
            "arm_qpeg": {"path": str(qpeg_path), "sha256": _sha256(qpeg_path)},
            "raw_dev_post_graph_build_only": raw_hashes,
        },
        "decision_gates": {
            "macro_delta_em_gt": 0.0,
            "max_net_correct_loss_per_dataset": 2,
            "one_generic_rule_revision_allowed": True,
            "per_question_patches_forbidden": True,
        },
        "integrity": {
            "paired_non_graph_fields_identical": True,
            "identity_join_rate": 1.0,
            "passage_provenance_rate": 1.0,
            "gold_access": False,
            "gold_joined_after_graph_freeze": True,
        },
    }
    protocol_path = args.out / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra={"phase": "qpeg_pilot_ab_input_freeze", **protocol}, status=protocol["status"])
    print(json.dumps({
        "status": protocol["status"],
        "n": len(cohort),
        "counts": {dataset: len(selected[dataset]) for dataset in DATASETS},
        "arm_no_qpeg_sha256": protocol["inputs"]["arm_no_qpeg"]["sha256"],
        "arm_qpeg_sha256": protocol["inputs"]["arm_qpeg"]["sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
