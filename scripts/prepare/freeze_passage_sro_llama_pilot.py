#!/usr/bin/env python
"""Freeze answer-free HotpotQA/MuSiQue inputs for a passage-SRO pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.kg.passage_sro import ALLOWED_RELATIONS, MAX_EDGES
from kgproweight.utils.logging import dump_manifest


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hotpot_plans", type=Path, required=True)
    parser.add_argument("--musique_plans", type=Path, required=True)
    parser.add_argument("--retrieval_contexts", type=Path, required=True)
    parser.add_argument("--base_model", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    retrieval = {
        (str(row["dataset"]), str(row["qid"])): row
        for row in _read_jsonl(args.retrieval_contexts)
    }
    rows: list[dict[str, Any]] = []
    for dataset, source in (("hotpotqa", args.hotpot_plans), ("musique", args.musique_plans)):
        plans = _read_jsonl(source)
        if len(plans) != 30:
            raise ValueError(f"expected 30 frozen plan rows for {dataset}, got {len(plans)}")
        for plan in plans:
            key = (dataset, str(plan["qid"]))
            context = retrieval.get(key)
            if context is None or context.get("question_sha256") != plan.get("question_sha256"):
                raise ValueError(f"missing/hash-mismatched retrieval context: {key}")
            rows.append({
                "dataset": dataset,
                "qid": str(plan["qid"]),
                "question": str(plan["question"]),
                "question_sha256": str(plan["question_sha256"]),
                "passages": list(context.get("passages") or [])[:10],
                "passages_sha256": str(context["passages_sha256"]),
                "legacy_kg": list(context.get("legacy_kg") or []),
            })
    if len(rows) != 60 or len({(row["dataset"], row["qid"]) for row in rows}) != 60:
        raise ValueError("pilot identity/count gate failed")
    cohort_path = args.output_dir / "cohort.answer_free.jsonl"
    _write_jsonl(cohort_path, rows)

    model = args.base_model
    protocol = {
        "schema_version": "passage-sro-llama-pilot-protocol-1",
        "experiment_id": "PASSAGE-SRO-LLAMA-HOTPOT-MUSIQUE-PILOT30-V1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PREREGISTERED_BEFORE_MODEL_EXTRACTION",
        "development_only": True,
        "confirmation_locked": True,
        "research_question": "Can a local model extract small, exact-span-verifiable SRO graphs from the already-retrieved passages that improve the strong SFT model under matched inputs?",
        "novel_variable": "model-assisted canonical SRO extraction with exact quote/head/tail/trigger validation; this is not QPEG-v1 regex extraction or QPEG-v3 full-sentence edges",
        "datasets": {"hotpotqa": 30, "musique": 30},
        "inputs": {
            "cohort": {"path": str(cohort_path), "sha256": _sha256(cohort_path)},
            "retrieval_contexts": {"path": str(args.retrieval_contexts), "sha256": _sha256(args.retrieval_contexts)},
            "hotpot_plans": {"path": str(args.hotpot_plans), "sha256": _sha256(args.hotpot_plans)},
            "musique_plans": {"path": str(args.musique_plans), "sha256": _sha256(args.musique_plans)},
        },
        "extractor": {
            "base_model": str(model),
            "config_sha256": _sha256(model / "config.json"),
            "model_index_sha256": _sha256(model / "model.safetensors.index.json"),
            "greedy": True,
            "seed": 42,
            "max_new_tokens": 512,
            "max_edges": MAX_EDGES,
            "allowed_relations": sorted(ALLOWED_RELATIONS),
        },
        "accepted_edge_requirements": [
            "passage rank refers to one frozen top-10 passage",
            "evidence quote occurs in that passage after whitespace normalization",
            "tail and relation trigger occur in the quote",
            "head occurs in the quote or exactly equals the passage title",
            "relation is in the frozen canonical vocabulary",
            "no self-loop, duplicate edge, or edge beyond the per-question cap",
        ],
        "structural_gates": {
            "identity_join": 1.0,
            "runtime_errors": 0,
            "accepted_edge_provenance_valid_rate": 1.0,
            "per_dataset_parse_rate_min": 0.9,
            "per_dataset_nonempty_rate_min": 0.5,
            "max_edges_per_question": MAX_EDGES,
        },
        "utility_gate": {
            "comparison": "same qid, passages, strong SFT checkpoint, decoding and scorer; A=legacy, B=accepted SRO prepended to legacy",
            "per_dataset_net_correct_min": 1,
            "per_dataset_lost_correct_max": 1,
            "per_dataset_f1_delta_min": 0.0,
            "fallback_prediction_identity": 1.0,
        },
        "forbidden": [
            "gold answers", "supporting-fact labels", "decomposition answers",
            "invented unverified free-form triples", "Wikidata/DBpedia/network access",
            "per-qid patching", "opening confirmation data",
        ],
        "prior_boundaries": {
            "qpeg_v1_1": "regex SRO macro delta EM -0.67pp; stopped",
            "qpeg_v3": "full-sentence evidence graph macro delta EM -1.22pp; stopped",
            "claim_constrained_wikidata": "HotpotQA/MuSiQue each gained 0 and lost 1; stopped",
        },
    }
    protocol_path = args.output_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.output_dir, status=protocol["status"], extra={
        "experiment_id": protocol["experiment_id"],
        "phase": "passage_sro_llama_pilot_preregistration",
        "protocol_sha256": _sha256(protocol_path),
    })
    print(json.dumps({"status": protocol["status"], "n": len(rows), "protocol": str(protocol_path)}, indent=2))


if __name__ == "__main__":
    main()
