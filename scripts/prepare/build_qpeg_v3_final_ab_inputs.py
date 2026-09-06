#!/usr/bin/env python
"""Freeze QPEG-v3 final300x3 matched A/B inputs after explicit approval.

All answer-free graph/context assets are validated before the dev Gold join.
Running this script creates a new evaluation protocol, so an explicit approval
identifier is mandatory.  It never modifies prior baseline artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.kg.qpeg import validate_qpeg_record
from kgproweight.utils.logging import dump_manifest


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
EXPECTED_PER_DATASET = 300


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


def _validate_prerequisites(
    *, gate_addendum: Mapping[str, Any], materialization: Mapping[str, Any], reuse: Mapping[str, Any]
) -> None:
    if gate_addendum.get("status") != "PASS_TRAIN_HOLDOUT_ADVANCE_CORRECTED_RUNTIME_CAP":
        raise ValueError("QPEG-v3 train-only gate did not pass")
    if materialization.get("status") != "COMPLETE_NOT_EVALUATED":
        raise ValueError("QPEG-v3 final graph materialization is not complete")
    integrity = materialization.get("integrity") or {}
    if not (
        materialization.get("n") == 900
        and integrity.get("identity_unique") is True
        and integrity.get("gold_access_false") is True
        and integrity.get("provenance_validated") is True
        and int(integrity.get("max_edges", 99)) <= 4
    ):
        raise ValueError("QPEG-v3 final graph integrity gate failed")
    if reuse.get("status") != "PASS_REUSE_EXACT_PROMPTS" or reuse.get("gates", {}).get("all_pass") is not True:
        raise ValueError("historical no-graph prompt reuse gate failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contexts", type=Path,
        default=Path("outputs/audits/qpeg_v1_n1350_seed42_preregistration/final.retrieval_contexts.jsonl"),
    )
    parser.add_argument(
        "--qpeg", type=Path,
        default=Path("data/derived/qpeg_v3_sentence_selector_final900_seed42_v2/question_graph_records.jsonl"),
    )
    parser.add_argument(
        "--gate_addendum", type=Path,
        default=Path("outputs/audits/qpeg_v3_sentence_selector_runtime_cap_correction_v1/decision_addendum.json"),
    )
    parser.add_argument(
        "--materialization_report", type=Path,
        default=Path("data/derived/qpeg_v3_sentence_selector_final900_seed42_v2/report.json"),
    )
    parser.add_argument(
        "--reuse_report", type=Path,
        default=Path("outputs/audits/qpeg_final_no_graph_prompt_reuse_v1/report.json"),
    )
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument(
        "--adapter", type=Path,
        default=Path("checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"),
    )
    parser.add_argument("--base_model", type=Path, default=Path("models/llama3-8b"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/audits/qpeg_v3_final300x3_ab_inputs_seed42"),
    )
    parser.add_argument("--researcher_approval_id", required=True)
    args = parser.parse_args()
    if not str(args.researcher_approval_id).strip():
        raise ValueError("researcher approval identifier must be non-empty")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite frozen final A/B inputs: {args.out}")
    args.out.mkdir(parents=True)

    gate = json.loads(args.gate_addendum.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    reuse = json.loads(args.reuse_report.read_text(encoding="utf-8"))
    _validate_prerequisites(gate_addendum=gate, materialization=materialization, reuse=reuse)
    if _sha256(args.qpeg) != materialization["outputs"]["records"]["sha256"]:
        raise ValueError("QPEG graph file differs from materialization report")
    if _sha256(args.contexts) != materialization["inputs"]["contexts"]["sha256"]:
        raise ValueError("frozen contexts differ from materialization report")

    contexts = _read_jsonl(args.contexts)
    graphs = {str(row["question_key"]): row for row in _read_jsonl(args.qpeg)}
    if len(contexts) != 900 or len(graphs) != 900:
        raise ValueError("expected final300x3 answer-free assets")
    counts: dict[str, int] = {dataset: 0 for dataset in DATASETS}
    for context in contexts:
        key = str(context["question_key"])
        graph = graphs[key]
        if context.get("role") != "final" or graph.get("role") != "final":
            raise ValueError(f"non-final row in final cohort: {key}")
        if graph.get("gold_access") is not False:
            raise ValueError(f"graph Gold access is not false: {key}")
        if str(context["question"]).strip() != str(graph["question"]).strip():
            raise ValueError(f"question mismatch: {key}")
        validate_qpeg_record(graph, passages=context["passages"])
        counts[str(context["dataset"])] += 1
    if any(counts[dataset] != EXPECTED_PER_DATASET for dataset in DATASETS):
        raise ValueError(f"unexpected per-dataset final counts: {counts}")

    # Gold is opened only after every answer-free identity/provenance gate above.
    selected = {dataset: set() for dataset in DATASETS}
    for context in contexts:
        selected[str(context["dataset"])].add(str(context["qid"]))
    gold: dict[str, list[str]] = {}
    raw_hashes: dict[str, str] = {}
    for dataset in DATASETS:
        path = args.data_root / dataset / "dev.jsonl"
        raw_hashes[str(path)] = _sha256(path)
        for row in _read_jsonl(path):
            qid = str(row.get("id") or "")
            if qid not in selected[dataset]:
                continue
            key = f"{dataset}::{qid}"
            gold[key] = [str(value) for value in row.get("golden_answers") or []]
    if len(gold) != 900:
        raise ValueError(f"Gold join incomplete: {len(gold)}/900")

    arm_no_graph: list[dict[str, Any]] = []
    arm_qpeg: list[dict[str, Any]] = []
    for context in contexts:
        key = str(context["question_key"])
        graph = graphs[key]
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
            "role": "final",
        }
        arm_no_graph.append({**common, "kg_subgraph": []})
        arm_qpeg.append({**common, "kg_subgraph": graph["kg_subgraph"]})
    if any(
        {key: value for key, value in left.items() if key != "kg_subgraph"}
        != {key: value for key, value in right.items() if key != "kg_subgraph"}
        for left, right in zip(arm_no_graph, arm_qpeg)
    ):
        raise ValueError("final paired inputs differ outside graph block")

    no_graph_path = args.out / "arm_no_graph.jsonl"
    qpeg_path = args.out / "arm_qpeg_v3.jsonl"
    _write_jsonl(no_graph_path, arm_no_graph)
    _write_jsonl(qpeg_path, arm_qpeg)
    qid_order = [str(row["question_key"]) for row in arm_no_graph]
    models = _artifact_hashes(args.adapter, args.base_model)
    protocol = {
        "schema_version": "qpeg-v3-final-matched-ab-protocol-v1",
        "experiment_id": "QPEG-V3-FINAL300X3-MATCHED-AB-SEED42",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_INPUTS_NOT_EVALUATED",
        "researcher_approval_id": str(args.researcher_approval_id).strip(),
        "n": 900,
        "per_dataset_n": 300,
        "single_variable": "same frozen passages; B adds QPEG-v3 typed evidence graph",
        "arm_semantics": {
            "no_graph": "strong SFT, frozen passages only; exact historical prompts/predictions reused",
            "qpeg_v3": "same strong SFT and passages plus QPEG-v3 full-sentence typed evidence edges",
        },
        "generation": {
            "seed": 42, "max_new_tokens": 512, "do_sample": False,
            "temperature": 0.0, "top_k_passages": 10,
        },
        "models": models,
        "qid_order_sha256": hashlib.sha256("\n".join(qid_order).encode()).hexdigest(),
        "inputs": {
            "contexts": {"path": str(args.contexts), "sha256": _sha256(args.contexts)},
            "qpeg": {"path": str(args.qpeg), "sha256": _sha256(args.qpeg)},
            "arm_no_graph": {"path": str(no_graph_path), "sha256": _sha256(no_graph_path)},
            "arm_qpeg_v3": {"path": str(qpeg_path), "sha256": _sha256(qpeg_path)},
            "gate_addendum": {"path": str(args.gate_addendum), "sha256": _sha256(args.gate_addendum)},
            "materialization_report": {"path": str(args.materialization_report), "sha256": _sha256(args.materialization_report)},
            "reuse_report": {"path": str(args.reuse_report), "sha256": _sha256(args.reuse_report)},
            "historical_no_graph_predictions": reuse["inputs"],
            "raw_dev_post_graph_freeze_only": raw_hashes,
        },
        "decision_gates": {
            "macro_delta_em_gt": 0.0,
            "macro_delta_f1_gt": 0.0,
            "max_net_correct_loss_per_dataset": 6,
            "max_parse_rate_drop": 0.01,
        },
        "reporting": [
            "per-dataset and macro EM/F1", "paired bootstrap 95% CI", "McNemar",
            "gained/lost/tied", "parse rate", "nonempty/empty graph strata",
        ],
        "integrity": {
            "paired_non_graph_fields_identical": True,
            "identity_join_rate": 1.0,
            "provenance_rate": 1.0,
            "gold_access_during_graph_build": False,
            "gold_joined_after_graph_freeze": True,
            "no_post_final_tuning": True,
        },
    }
    protocol_path = args.out / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra={"phase": "qpeg_v3_final_ab_input_freeze", **protocol}, status=protocol["status"])
    print(json.dumps({
        "status": protocol["status"], "n": 900, "counts": counts,
        "arm_no_graph_sha256": protocol["inputs"]["arm_no_graph"]["sha256"],
        "arm_qpeg_v3_sha256": protocol["inputs"]["arm_qpeg_v3"]["sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
