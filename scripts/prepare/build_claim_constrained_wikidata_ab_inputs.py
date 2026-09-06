#!/usr/bin/env python
"""Freeze same-passage A/B inputs for the claim-constrained Wikidata pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_triples(proof: list[list[str]], legacy: list[list[str]], cap: int = 12) -> list[list[str]]:
    merged: list[list[str]] = []
    seen = set()
    for value in [*proof, *legacy]:
        triple = tuple(str(part).strip() for part in value)
        if len(triple) != 3 or not all(triple) or triple in seen:
            continue
        seen.add(triple)
        merged.append(list(triple))
        if len(merged) >= cap:
            break
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof_records", type=Path, required=True)
    parser.add_argument("--retrieval_contexts", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--parent_protocol", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    proof_rows = _read_jsonl(args.proof_records)
    retrieval = {
        (str(row["dataset"]), str(row["qid"])): row
        for row in _read_jsonl(args.retrieval_contexts)
    }
    gold: dict[tuple[str, str], list[str]] = {}
    for dataset in ("hotpotqa", "musique"):
        for row in _read_jsonl(ROOT / "data" / dataset / "dev.jsonl"):
            gold[(dataset, str(row["id"]))] = [str(value) for value in row.get("golden_answers") or []]

    model = ROOT / "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"
    base = ROOT / "models/llama3-8b"
    reports: dict[str, Any] = {}
    for dataset in ("hotpotqa", "musique"):
        selected = [row for row in proof_rows if row["dataset"] == dataset]
        arm_a: list[dict[str, Any]] = []
        arm_b: list[dict[str, Any]] = []
        fallback = 0
        for proof in selected:
            key = (dataset, str(proof["qid"]))
            context = retrieval[key]
            if context["question_sha256"] != proof["question_sha256"]:
                raise ValueError(f"question hash mismatch for {key}")
            common = {
                "row_id": f"claim-constrained-wikidata::{dataset}::{proof['qid']}",
                "dataset": dataset,
                "qid": proof["qid"],
                "question": proof["question"],
                "gold_answers": gold[key],
                "retrieved_passages": list(context["passages"]),
                "scope": "development-only claim-constrained Wikidata pilot30",
            }
            legacy = list(context.get("legacy_kg") or [])
            new = list(proof.get("kg_subgraph") or [])
            a = {**common, "kg_subgraph": legacy}
            b = {**common, "kg_subgraph": merge_triples(new, legacy)}
            arm_a.append(a)
            arm_b.append(b)
            fallback += int(not new)

        dataset_dir = args.output_dir / dataset
        dataset_dir.mkdir()
        a_path, b_path = dataset_dir / "arm_legacy.jsonl", dataset_dir / "arm_legacy_plus_claim.jsonl"
        _write_jsonl(a_path, arm_a)
        _write_jsonl(b_path, arm_b)
        qid_order_sha = hashlib.sha256("\n".join(str(row["qid"]) for row in arm_a).encode()).hexdigest()
        protocol = {
            "schema_version": "claim-constrained-wikidata-ab-protocol-1",
            "experiment_id": f"CLAIM-CONSTRAINED-WIKIDATA-{dataset.upper()}-PILOT30-AB-V1",
            "status": "FROZEN_BEFORE_MODEL_INFERENCE",
            "scope": "development-only pilot30; same frozen passages; zero training",
            "parent_protocol": {"path": str(args.parent_protocol), "sha256": _sha256(args.parent_protocol)},
            "dataset": dataset,
            "n": len(arm_a),
            "qid_order_sha256": qid_order_sha,
            "inputs": {
                "arm_legacy": {"path": str(a_path), "sha256": _sha256(a_path)},
                "arm_proof": {"path": str(b_path), "sha256": _sha256(b_path)},
            },
            "models": {"sft": {
                "adapter_config_sha256": _sha256(model / "adapter_config.json"),
                "adapter_model_sha256": _sha256(model / "adapter_model.safetensors"),
            }},
            "base_model": {
                "config_sha256": _sha256(base / "config.json"),
                "model_index_sha256": _sha256(base / "model.safetensors.index.json"),
            },
            "generation": {"max_new_tokens": 512, "seed": 42, "top_k_passages": 10, "greedy": True},
            "single_variable": "KG block: legacy vs claim-constrained Wikidata triples prepended to legacy; same passages/model/decoding/scorer",
            "fallback_rows": fallback,
            "gold_usage": "Gold joined only after KG records were frozen and used only by scorer.",
        }
        protocol_path = dataset_dir / "protocol.json"
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        reports[dataset] = {"n": len(arm_a), "fallback": fallback, "protocol_sha256": _sha256(protocol_path)}
    (args.output_dir / "report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
