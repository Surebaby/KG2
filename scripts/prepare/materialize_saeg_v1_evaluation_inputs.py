#!/usr/bin/env python
"""Materialize answer-free SAEG evaluation inputs from the frozen protocol.

The output is a master dataset, not model predictions.  It contains frozen
passages, passage evidence objects, standard Wikidata triples where eligible,
and explicit arm eligibility.  Gold answers are prohibited and exported by a
separate scorer-only script.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.kg.qpeg import compute_passages_sha256
from kgproweight.kg.source_adaptive_evidence_graph import sha256_json
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_saeg_v1_evaluation_protocol import assert_answer_free


EXPERIMENT_ID = "SAEG-V1-EVALUATION-INPUTS-SEED42"
STATUS = "COMPLETE_ANSWER_FREE_CONFIRMATION_UNOPENED"


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


def index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        key = str(row["question_key"])
        if key in output:
            raise ValueError(f"duplicate {label}: {key}")
        output[key] = row
    return output


def passage_items(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = []
    for ordinal, edge in enumerate(graph.get("edges") or [], start=1):
        if str(edge.get("relation_surface")) != "evidence sentence":
            raise ValueError(f"{graph['question_key']}: unexpected QPEG relation")
        item = {
            "passage_id": f"P{ordinal}",
            "title": str(edge["head_surface"]),
            "sentence": str(edge["tail_surface"]),
            "source_passage_id": str(edge["passage_id"]),
            "passage_rank": int(edge["passage_rank"]),
            "sentence_index": int(edge["sentence_index"]),
            "sentence_sha256": str(edge["sentence_sha256"]),
            "selector_score": float(edge["relevance_score"]),
            "construction_gold_access": False,
        }
        items.append(item)
    return items


def standard_wikidata_triples(proof: Mapping[str, Any] | None) -> list[list[str]]:
    triples = []
    for raw in (proof or {}).get("kg_subgraph") or []:
        if not isinstance(raw, list) or len(raw) != 3:
            raise ValueError("malformed Wikidata triple")
        triple = [" ".join(str(value).split()) for value in raw]
        if not all(triple) or triple[1].casefold() == "evidence sentence":
            raise ValueError("passage pseudo-triple found in Wikidata branch")
        triples.append(triple)
    return triples


def materialize_row(
    cohort: Mapping[str, Any],
    context: Mapping[str, Any],
    graph: Mapping[str, Any],
    proof: Mapping[str, Any] | None,
    *,
    proof_eligible: bool,
) -> dict[str, Any]:
    key = str(cohort["question_key"])
    for other in (context, graph):
        if str(other["question_key"]) != key:
            raise ValueError(f"{key}: identity mismatch")
        if str(other["question_sha256"]) != cohort["question_sha256"]:
            raise ValueError(f"{key}: question hash mismatch")
    if context["passages_sha256"] != cohort["passages_sha256"]:
        raise ValueError(f"{key}: context/cohort passage hash mismatch")
    if compute_passages_sha256(context.get("passages") or []) != context["passages_sha256"]:
        # QPEG retrieval uses the same title+contents semantic hash helper; a
        # mismatch here would make graph/prompt comparisons invalid.
        raise ValueError(f"{key}: recomputed passage hash mismatch")
    passages = list(context.get("passages") or [])
    p_evidence = passage_items(graph)
    candidate_w_triples = standard_wikidata_triples(proof)
    w_triples = candidate_w_triples if proof_eligible else []
    dataset = str(cohort["dataset"])
    w_eligible = dataset == "2wikimultihopqa" and proof is not None and proof_eligible
    if proof is not None:
        if str(proof["question_key"]) != key or str(proof["question_sha256"]) != cohort["question_sha256"]:
            raise ValueError(f"{key}: ProofKG identity/hash mismatch")
        provenance = proof.get("provenance") or {}
        if provenance.get("gold_access") is not False:
            raise ValueError(f"{key}: ProofKG does not prove gold_access=false")
    if dataset != "2wikimultihopqa" and proof is not None:
        raise ValueError(f"{key}: ineligible dataset unexpectedly has Wikidata proof")
    arms = {
        "A_no_graph": {"eligible": True, "sources": []},
        "B_passage": {"eligible": bool(p_evidence), "sources": ["passage"] if p_evidence else []},
        "C_wikidata": {"eligible": bool(w_eligible and w_triples), "sources": ["wikidata"] if w_triples else []},
        "D_fused": {
            "eligible": bool(p_evidence or (w_eligible and w_triples)),
            "sources": (["passage"] if p_evidence else []) + (["wikidata"] if w_eligible and w_triples else []),
        },
    }
    row = {
        "schema_version": "saeg-eval-input-v1",
        "question_key": key,
        "dataset": dataset,
        "qid": str(cohort["qid"]),
        "question": str(cohort["question"]),
        "question_sha256": str(cohort["question_sha256"]),
        "family_sha256": str(cohort["family_sha256"]),
        "role": str(cohort["role"]),
        "gold_access": False,
        "passages": passages,
        "passages_sha256": str(context["passages_sha256"]),
        "passage_evidence": p_evidence,
        "wikidata_kg": w_triples,
        "wikidata_provenance": dict((proof or {}).get("provenance") or {}),
        "wikidata_candidate_triple_count": len(candidate_w_triples),
        "source_status": {
            "passage": "nonempty" if p_evidence else "empty_fail_closed",
            "wikidata": (
                "nonempty" if w_eligible and w_triples else
                "empty_fail_closed" if w_eligible else
                "not_eligible_frozen_structural_failure"
            ),
        },
        "arms": arms,
    }
    row["evidence_sha256"] = sha256_json({
        "passage_evidence": p_evidence,
        "wikidata_kg": w_triples,
        "source_status": row["source_status"],
    })
    assert_answer_free(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol_dir", type=Path, default=Path(
        "outputs/audits/saeg_v1_evaluation_protocol_v1"))
    parser.add_argument("--fresh_contexts", type=Path, default=Path(
        "outputs/audits/qpeg_v4_schema_adaptation_eval_retrieval_v1/retrieval_contexts.jsonl"))
    parser.add_argument("--fresh_passage_graphs", type=Path, default=Path(
        "data/derived/qpeg_v4_schema_adaptation_eval450_seed42/question_graph_records.jsonl"))
    parser.add_argument("--fresh_2wiki_proof", type=Path, default=Path(
        "data/derived/saeg_v1_2wiki_dev_confirmation_proofkg_v1/question_kg_records.jsonl"))
    parser.add_argument("--fresh_2wiki_proof_report", type=Path, default=Path(
        "data/derived/saeg_v1_2wiki_dev_confirmation_proofkg_v1/report.json"))
    parser.add_argument("--canonical_contexts", type=Path, default=Path(
        "outputs/audits/qpeg_v1_n1350_seed42_preregistration/final.retrieval_contexts.jsonl"))
    parser.add_argument("--canonical_passage_graphs", type=Path, default=Path(
        "data/derived/qpeg_v3_sentence_selector_final900_seed42_v2/question_graph_records.jsonl"))
    parser.add_argument("--canonical_2wiki_proof", type=Path, default=Path(
        "data/derived/inference_proofkg_v1_2wiki_dev_n300_v1/question_kg_records.jsonl"))
    parser.add_argument("--out", type=Path, default=Path(
        "data/derived/saeg_v1_evaluation_inputs_seed42_v1"))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite SAEG evaluation inputs: {args.out}")
    cohort_paths = {
        "development": args.protocol_dir / "development.question_only.jsonl",
        "confirmation": args.protocol_dir / "confirmation.question_only.jsonl",
        "canonical_reporting": args.protocol_dir / "canonical_reporting.question_only.jsonl",
    }
    inputs = {
        **cohort_paths,
        "protocol": args.protocol_dir / "protocol.json",
        "fresh_contexts": args.fresh_contexts,
        "fresh_passage_graphs": args.fresh_passage_graphs,
        "fresh_2wiki_proof": args.fresh_2wiki_proof,
        "fresh_2wiki_proof_report": args.fresh_2wiki_proof_report,
        "canonical_contexts": args.canonical_contexts,
        "canonical_passage_graphs": args.canonical_passage_graphs,
        "canonical_2wiki_proof": args.canonical_2wiki_proof,
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol = json.loads(inputs["protocol"].read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_ANSWER_FREE_BEFORE_SAEG_DEVELOPMENT":
        raise ValueError("unexpected SAEG evaluation protocol")

    fresh_contexts = index(read_jsonl(args.fresh_contexts), "fresh context")
    fresh_graphs = index(read_jsonl(args.fresh_passage_graphs), "fresh graph")
    fresh_proof = index(read_jsonl(args.fresh_2wiki_proof), "fresh ProofKG")
    fresh_proof_report = json.loads(args.fresh_2wiki_proof_report.read_text(encoding="utf-8"))
    fresh_proof_eligible = fresh_proof_report.get("status") == "PASS_STRUCTURAL_NOT_MODEL_EVALUATED"
    canonical_contexts = index(read_jsonl(args.canonical_contexts), "canonical context")
    canonical_graphs = index(read_jsonl(args.canonical_passage_graphs), "canonical graph")
    canonical_proof = index(read_jsonl(args.canonical_2wiki_proof), "canonical ProofKG")

    outputs_by_role: dict[str, list[dict[str, Any]]] = {}
    for role in ("development", "confirmation"):
        rows = []
        for cohort in read_jsonl(cohort_paths[role]):
            key = str(cohort["question_key"])
            proof = fresh_proof.get(key) if cohort["dataset"] == "2wikimultihopqa" else None
            if cohort["dataset"] == "2wikimultihopqa" and proof is None:
                raise ValueError(f"{key}: missing fresh 2Wiki ProofKG record")
            rows.append(materialize_row(
                cohort, fresh_contexts[key], fresh_graphs[key], proof,
                proof_eligible=fresh_proof_eligible,
            ))
        outputs_by_role[role] = rows
    reporting = []
    for cohort in read_jsonl(cohort_paths["canonical_reporting"]):
        key = str(cohort["question_key"])
        proof = canonical_proof.get(key) if cohort["dataset"] == "2wikimultihopqa" else None
        reporting.append(materialize_row(
            cohort, canonical_contexts[key], canonical_graphs[key], proof,
            proof_eligible=True,
        ))
    outputs_by_role["canonical_reporting"] = reporting

    args.out.mkdir(parents=True, exist_ok=False)
    output_paths = {}
    for role, rows in outputs_by_role.items():
        path = args.out / f"{role}.answer_free.jsonl"
        write_jsonl(path, rows)
        output_paths[role] = path
    all_rows = [row for rows in outputs_by_role.values() for row in rows]
    if len(all_rows) != 1350 or len({(row["role"], row["question_key"]) for row in all_rows}) != 1350:
        raise RuntimeError("expected 1350 role-qualified unique evaluation inputs")
    counts = Counter()
    for row in all_rows:
        counts[f"role::{row['role']}"] += 1
        counts[f"dataset::{row['role']}::{row['dataset']}"] += 1
        counts[f"P::{row['source_status']['passage']}"] += 1
        counts[f"W::{row['source_status']['wikidata']}"] += 1
    report = {
        "schema_version": "saeg-evaluation-input-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "counts": dict(sorted(counts.items())),
        "integrity": {
            "records": len(all_rows),
            "answer_or_gold_fields": 0,
            "passage_pseudo_triples_in_wikidata_kg": 0,
            "identity_and_hash_join_rate": 1.0,
            "confirmation_opened_for_model_evaluation": False,
            "canonical_reporting_is_nonconfirmatory": True,
            "fresh_2wiki_wikidata_branch_eligible": fresh_proof_eligible,
        },
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in inputs.items()},
        "outputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in output_paths.items()},
        "scientific_boundary": (
            "Answer-free inference inputs only. Gold answers live in a separate scorer-only dataset. "
            "Canonical reporting is historical/consumed and cannot select checkpoints."
        ),
    }
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(args.out, extra=report, status=STATUS)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
