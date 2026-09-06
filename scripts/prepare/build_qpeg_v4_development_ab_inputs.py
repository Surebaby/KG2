#!/usr/bin/env python
"""Freeze QPEG-v4 development A/B inputs for the strong-SFT baseline.

The answer-free retrieval and graph assets are validated before Gold answers
are joined.  The two output arms are byte-identical outside ``kg_subgraph``.
This script opens development only; confirmation rows are rejected.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.kg.qpeg import validate_qpeg_record
from kgproweight.utils.logging import dump_manifest


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
EXPECTED_PER_DATASET = 50


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


def _artifact_hashes(adapter: Path, base_model: Path) -> dict[str, Any]:
    return {
        "strong_sft": {
            "path": str(adapter),
            "adapter_config_sha256": _sha256(adapter / "adapter_config.json"),
            "adapter_model_sha256": _sha256(adapter / "adapter_model.safetensors"),
        },
        "base_model": {
            "path": str(base_model),
            "config_sha256": _sha256(base_model / "config.json"),
            "model_index_sha256": _sha256(base_model / "model.safetensors.index.json"),
            "tokenizer_sha256": _sha256(base_model / "tokenizer.json"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment_protocol", type=Path,
        default=Path("outputs/audits/qpeg_v4_schema_adaptation_protocol_v1/protocol.json"),
    )
    parser.add_argument(
        "--contexts", type=Path,
        default=Path("outputs/audits/qpeg_v4_schema_adaptation_eval_retrieval_v1/retrieval_contexts.jsonl"),
    )
    parser.add_argument(
        "--graphs", type=Path,
        default=Path("data/derived/qpeg_v4_schema_adaptation_eval450_seed42/question_graph_records.jsonl"),
    )
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument(
        "--adapter", type=Path,
        default=Path("checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"),
    )
    parser.add_argument("--base_model", type=Path, default=Path("models/llama3-8b"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/audits/qpeg_v4_schema_adaptation_development_inputs_v1"),
    )
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite frozen development inputs: {args.out}")

    experiment_protocol = json.loads(args.experiment_protocol.read_text(encoding="utf-8"))
    if experiment_protocol.get("schema_version") != "qpeg-v4-schema-adaptation-protocol-v1":
        raise ValueError("unexpected QPEG-v4 experiment protocol")
    if not str(experiment_protocol.get("researcher_approval_id") or "").strip():
        raise ValueError("QPEG-v4 experiment is missing researcher approval")

    all_contexts = _read_jsonl(args.contexts)
    all_graph_rows = _read_jsonl(args.graphs)
    development_contexts = [row for row in all_contexts if row.get("role") == "development"]
    development_graphs = {
        str(row["question_key"]): row
        for row in all_graph_rows
        if row.get("role") == "development"
    }
    if len(development_contexts) != 150 or len(development_graphs) != 150:
        raise ValueError(
            f"expected 150 development contexts/graphs, got "
            f"{len(development_contexts)}/{len(development_graphs)}"
        )
    if any(row.get("role") != "development" for row in development_contexts):
        raise ValueError("non-development row reached development freeze")

    counts: Counter[str] = Counter()
    selected: dict[str, set[str]] = {dataset: set() for dataset in DATASETS}
    for context in development_contexts:
        key = str(context["question_key"])
        graph = development_graphs.get(key)
        if graph is None:
            raise ValueError(f"missing development graph: {key}")
        if context.get("gold_access") is not False or graph.get("gold_access") is not False:
            raise ValueError(f"Gold access is not false before Gold join: {key}")
        if graph.get("role") != "development":
            raise ValueError(f"non-development graph row: {key}")
        if str(context["question"]).strip() != str(graph["question"]).strip():
            raise ValueError(f"question mismatch: {key}")
        if str(context["passages_sha256"]) != str(graph["passages_sha256"]):
            raise ValueError(f"passage hash mismatch: {key}")
        validate_qpeg_record(graph, passages=context["passages"])
        dataset = str(context["dataset"])
        if dataset not in selected:
            raise ValueError(f"unexpected dataset: {dataset}")
        counts[dataset] += 1
        selected[dataset].add(str(context["qid"]))
    if any(counts[dataset] != EXPECTED_PER_DATASET for dataset in DATASETS):
        raise ValueError(f"unexpected development counts: {dict(counts)}")

    # Gold is opened only after all answer-free joins and provenance checks pass.
    gold: dict[str, list[str]] = {}
    raw_hashes: dict[str, str] = {}
    for dataset in DATASETS:
        raw_path = args.data_root / dataset / "dev.jsonl"
        raw_hashes[str(raw_path)] = _sha256(raw_path)
        for row in _read_jsonl(raw_path):
            qid = str(row.get("id") or "")
            if qid in selected[dataset]:
                gold[f"{dataset}::{qid}"] = [
                    str(value) for value in row.get("golden_answers") or [] if str(value).strip()
                ]
    if len(gold) != 150 or any(not answers for answers in gold.values()):
        raise ValueError(f"Gold join incomplete: {len(gold)}/150")

    no_graph_rows: list[dict[str, Any]] = []
    qpeg_rows: list[dict[str, Any]] = []
    for context in development_contexts:
        key = str(context["question_key"])
        graph = development_graphs[key]
        common = {
            "row_id": key,
            "question_key": key,
            "dataset": context["dataset"],
            "qid": context["qid"],
            "question": context["question"],
            "gold_answers": gold[key],
            "retrieved_passages": context["passages"],
            "passages_sha256": context["passages_sha256"],
            "qpeg_sha256": graph["qpeg_sha256"],
            "qpeg_edge_count": len(graph["kg_subgraph"]),
            "role": "development",
        }
        no_graph_rows.append({**common, "kg_subgraph": []})
        qpeg_rows.append({**common, "kg_subgraph": graph["kg_subgraph"]})
    if any(
        {key: value for key, value in left.items() if key != "kg_subgraph"}
        != {key: value for key, value in right.items() if key != "kg_subgraph"}
        for left, right in zip(no_graph_rows, qpeg_rows)
    ):
        raise ValueError("paired development inputs differ outside kg_subgraph")

    args.out.mkdir(parents=True)
    no_graph_path = args.out / "arm_no_graph.jsonl"
    qpeg_path = args.out / "arm_qpeg.jsonl"
    _write_jsonl(no_graph_path, no_graph_rows)
    _write_jsonl(qpeg_path, qpeg_rows)
    models = _artifact_hashes(args.adapter, args.base_model)
    qid_order = [str(row["qid"]) for row in no_graph_rows]
    protocol = {
        "schema_version": "qpeg-v4-development-paired-input-protocol-v1",
        "experiment_id": "QPEG-V4-DEVELOPMENT-STRONG-SFT-AB-SEED42",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_DEVELOPMENT_INPUTS_NOT_EVALUATED",
        "researcher_approval_id": experiment_protocol["researcher_approval_id"],
        "scope": "QPEG-v4 fresh development50x3; confirmation remains unopened",
        "n": 150,
        "per_dataset_n": 50,
        "single_variable": "same checkpoint/passages/decoding; proof arm adds answer-free QPEG",
        "generation": {
            "seed": 42,
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 0.0,
            "top_k_passages": 10,
        },
        "models": {"strong_sft": models["strong_sft"]},
        "base_model": models["base_model"],
        "qid_order_sha256": hashlib.sha256("\n".join(qid_order).encode()).hexdigest(),
        "inputs": {
            # Compatibility names for the generic paired evaluator.  Their
            # semantics in this experiment are no-graph and QPEG, respectively.
            "arm_legacy": {"path": str(no_graph_path), "sha256": _sha256(no_graph_path)},
            "arm_proof": {"path": str(qpeg_path), "sha256": _sha256(qpeg_path)},
            "contexts": {"path": str(args.contexts), "sha256": _sha256(args.contexts)},
            "graphs": {"path": str(args.graphs), "sha256": _sha256(args.graphs)},
            "experiment_protocol": {
                "path": str(args.experiment_protocol), "sha256": _sha256(args.experiment_protocol)
            },
            "raw_dev_post_graph_freeze_only": raw_hashes,
        },
        "integrity": {
            "development_only": True,
            "confirmation_opened": False,
            "paired_non_graph_fields_identical": True,
            "identity_join_rate": 1.0,
            "gold_access_during_graph_build": False,
            "gold_joined_after_graph_validation": True,
        },
    }
    protocol_path = args.out / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(
        args.out,
        extra={"phase": "qpeg_v4_development_input_freeze", **protocol},
        status=protocol["status"],
    )
    print(json.dumps({
        "status": protocol["status"],
        "counts": dict(counts),
        "arm_no_graph_sha256": protocol["inputs"]["arm_legacy"]["sha256"],
        "arm_qpeg_sha256": protocol["inputs"]["arm_proof"]["sha256"],
        "confirmation_opened": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
