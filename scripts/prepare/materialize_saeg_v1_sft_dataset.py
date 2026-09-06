#!/usr/bin/env python
"""Materialize the train-only SAEG-v1 SFT dataset with two citation fields.

Passage evidence is rendered as ``[P<n>]`` text and never appears in
``kg_subgraph`` or ``Knowledge Used``.  Wikidata facts remain standard
``(head, relation, tail)`` triples.  No teacher API is called.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.data.saeg_parsers import parse_saeg_steps
from kgproweight.kg.question_kg import question_sha256
from kgproweight.kg.source_adaptive_evidence_graph import canonical_triple
from kgproweight.utils.logging import dump_manifest


EXPERIMENT_ID = "SAEG-V1-TRAIN4862-DUAL-CITATION-SEED42"
STATUS = "COMPLETE_TRAIN_ONLY_NOT_TRAINED"


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


def source_qid(row: Mapping[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("source_qid") or str(row.get("qid") or "").split("::", 1)[0])


def passage_evidence(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for edge in record.get("edges") or []:
        if edge.get("source_type") != "passage":
            continue
        title, relation, sentence = edge["triple"]
        if relation != "evidence sentence":
            raise ValueError(f"{record['record_id']}: unexpected internal passage relation")
        provenance = edge.get("provenance") or {}
        output.append({
            "passage_id": str(edge["edge_id"]),
            "title": str(title),
            "sentence": str(sentence),
            "source_passage_id": str(provenance["passage_id"]),
            "passage_rank": int(provenance["passage_rank"]),
            "sentence_index": int(provenance["sentence_index"]),
            "sentence_sha256": str(provenance["sentence_sha256"]),
            "construction_gold_access": bool(edge.get("construction_gold_access")),
        })
    return output


def wikidata_edges(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(edge) for edge in (record.get("edges") or []) if edge.get("source_type") == "wikidata"]


def _add_passage_field(text: str, ids: Sequence[str]) -> str:
    field = f"Passage Used: [{', '.join(ids)}]" if ids else "Passage Used: []"
    replaced, count = re.subn(
        r"(?im)^[ \t]*Knowledge Used\s*:[^\r\n]*$",
        f"Knowledge Used: []\n{field}",
        str(text).strip(),
        count=1,
    )
    if count != 1:
        raise ValueError("source step does not contain exactly one Knowledge Used field")
    return replaced


def p_or_n_steps(
    trajectory: Mapping[str, Any],
    p_evidence: Sequence[Mapping[str, Any]],
    *,
    cite_passages: bool,
) -> list[dict[str, Any]]:
    triple_to_id = {
        tuple(canonical_triple((item["title"], "evidence sentence", item["sentence"]))): str(item["passage_id"])
        for item in p_evidence
    }
    output = []
    for raw in trajectory.get("steps") or []:
        cited_ids: list[str] = []
        if cite_passages:
            for triple in raw.get("cited_triples") or []:
                key = tuple(canonical_triple(triple))
                if key not in triple_to_id:
                    raise ValueError(f"{trajectory['qid']}: passage citation is not visible")
                cited_ids.append(triple_to_id[key])
        cited_ids = list(dict.fromkeys(cited_ids))
        output.append({
            "index": int(raw["index"]),
            "text": _add_passage_field(str(raw["text"]), cited_ids),
            "label": float(raw.get("label", 0.0)) if cite_passages else 0.0,
            "cited_triples": [],
            **({"cited_edge_ids": cited_ids, "cited_passage_ids": cited_ids} if cited_ids else {}),
        })
    return output


def _render_kg_list(edges: Sequence[Mapping[str, Any]]) -> str:
    return "[" + ", ".join(f"({', '.join(map(str, edge['triple']))})" for edge in edges) + "]"


def w_or_fused_steps(
    record: Mapping[str, Any],
    alignment: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], str]:
    by_hop: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in wikidata_edges(record):
        by_hop[int(edge["provenance"]["hop_index"])].append(edge)
    if not by_hop:
        raise ValueError(f"{record['record_id']}: W target has no Wikidata hop")
    aligned: dict[int, str] = {}
    target_route = "W_ONLY"
    if alignment is not None and alignment.get("all_hops_one_to_one_aligned"):
        aligned = {
            int(item["hop_index"]): str(item["passage_edge_id"])
            for item in alignment.get("alignments") or []
        }
        target_route = "P_W_JOINT"
    elif alignment is not None:
        target_route = "W_ONLY_FAIL_CLOSED"
    output = []
    for output_index, (hop_index, edges) in enumerate(sorted(by_hop.items()), start=1):
        triples = [list(map(str, edge["triple"])) for edge in edges]
        edge_ids = [str(edge["edge_id"]) for edge in edges]
        passage_ids = [aligned[hop_index]] if hop_index in aligned else []
        claims = "; ".join(" ".join(triple) for triple in triples)
        reasoning = f"The supplied Wikidata fact establishes: {claims}."
        if passage_ids:
            reasoning += " The aligned passage evidence independently supports this hop."
        output.append({
            "index": output_index,
            "text": (
                f"Reasoning: {reasoning}\n"
                f"Knowledge Used: {_render_kg_list(edges)}\n"
                f"Passage Used: [{', '.join(passage_ids)}]\n"
                f"Conclusion: {claims}."
            ),
            "label": 1.0,
            "cited_triples": triples,
            "cited_edge_ids": edge_ids + passage_ids,
            **({"cited_passage_ids": passage_ids} if passage_ids else {}),
            "required_hop_index": hop_index,
        })
    return output, target_route


def final_step(index: int, answer: str) -> dict[str, Any]:
    return {
        "index": index,
        "text": (
            "Reasoning: Combining the preceding evidence resolves the multi-hop question.\n"
            "Knowledge Used: []\n"
            "Passage Used: []\n"
            f"Conclusion: {answer}"
        ),
        "label": 0.0,
        "cited_triples": [],
    }


def assistant_trace(steps: Sequence[Mapping[str, Any]], answer: str) -> str:
    chunks = [f"[Step {index}]\n{step['text']}" for index, step in enumerate(steps, start=1)]
    chunks.append(f"[Final Answer]\n{answer}")
    return "\n\n".join(chunks)


def validate_trajectory(row: Mapping[str, Any]) -> None:
    kg = row.get("kg_subgraph") or []
    p_ids = [str(item["passage_id"]) for item in (row.get("passage_evidence") or [])]
    if any(len(triple) != 3 or str(triple[1]) == "evidence sentence" for triple in kg):
        raise ValueError(f"{row['qid']}: kg_subgraph contains non-standard passage pseudo-triple")
    parsed = parse_saeg_steps(row["teacher_output"], known_kg=kg, known_passage_ids=p_ids)
    if len(parsed) != len(row.get("steps") or []):
        raise ValueError(f"{row['qid']}: parsed step count mismatch")
    if not parsed or any(not step.citation_contract_valid for step in parsed):
        raise ValueError(f"{row['qid']}: invalid SAEG citation contract")
    if not str(row.get("answer") or "").strip():
        raise ValueError(f"{row['qid']}: empty train answer")
    for raw, parsed_step in zip(row["steps"], parsed):
        # The exact-match parser removes longer rendered surfaces first to
        # handle comma-bearing values, so two valid triples in one step may be
        # returned in length order rather than prompt order.
        parsed_w = Counter(tuple(value) for value in parsed_step.cited_triples)
        sidecar_w = Counter(tuple(map(str, value)) for value in (raw.get("cited_triples") or []))
        if parsed_w != sidecar_w:
            raise ValueError(f"{row['qid']}: visible W citation and sidecar mismatch")
        if parsed_step.cited_passage_ids != list(raw.get("cited_passage_ids") or []):
            raise ValueError(f"{row['qid']}: visible P citation and sidecar mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph_assets", type=Path, default=Path(
        "data/derived/saeg_v1_training_graph_assets_v1/question_graph_records.jsonl"))
    parser.add_argument("--alignment", type=Path, default=Path(
        "outputs/audits/saeg_v1_cross_source_alignment_v1/alignment_rows.identity_only.jsonl"))
    parser.add_argument("--qpeg_silver", type=Path, default=Path(
        "data/silver_data/qpeg_v4_schema_adaptation_n2400_seed42_v2/silver_curriculum.jsonl"))
    parser.add_argument("--proof_silver", type=Path, default=Path(
        "data/silver_data/automatic_proofkg_2wiki_train_k4_v1/silver_train.jsonl"))
    parser.add_argument("--sampling_weights", type=Path, default=Path(
        "outputs/audits/saeg_v1_training_sampling_weights_v1/sampling_weights.jsonl"))
    parser.add_argument("--out", type=Path, default=Path(
        "data/silver_data/saeg_v1_train4862_seed42_v1"))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite SAEG SFT data: {args.out}")
    inputs = {
        "graph_assets": args.graph_assets,
        "alignment": args.alignment,
        "qpeg_silver": args.qpeg_silver,
        "proof_silver": args.proof_silver,
        "sampling_weights": args.sampling_weights,
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    records = read_jsonl(args.graph_assets)
    qpeg = read_jsonl(args.qpeg_silver)
    qpeg_lookup = {}
    for row in qpeg:
        variant = str((row.get("metadata") or {}).get("curriculum_variant") or "")
        mode = "P_ONLY" if variant == "qpeg" else "N_REPLAY"
        qpeg_lookup[(str(row["dataset"]), source_qid(row), mode)] = row
    proof = {str(row["qid"]): row for row in read_jsonl(args.proof_silver)}
    alignment = {str(row["record_id"]): row for row in read_jsonl(args.alignment)}
    weights = {str(row["candidate_id"]): row for row in read_jsonl(args.sampling_weights)}

    trajectories = []
    counts: Counter[str] = Counter()
    for record in records:
        dataset, qid = str(record["dataset"]), str(record["qid"])
        mode = str((record.get("routing") or {})["mode"])
        p_items = passage_evidence(record)
        w_edges = wikidata_edges(record)
        if mode in {"P_ONLY", "N_REPLAY"}:
            source = qpeg_lookup[(dataset, qid, mode)]
            answer = str(source["answer"]).strip()
            passages = list(source.get("retrieved_passages") or [])
            steps = p_or_n_steps(source, p_items, cite_passages=mode == "P_ONLY")
            target_route = mode
        else:
            source = proof[qid]
            answer = str(source["answer"]).strip()
            passages = list(source.get("retrieved_passages") or [])
            steps, target_route = w_or_fused_steps(record, alignment.get(str(record["record_id"])))
        steps.append(final_step(len(steps) + 1, answer)) if mode in {"W_ONLY", "P_W_FUSED"} else None
        if question_sha256(str(source["question"])) != record["question_sha256"]:
            raise ValueError(f"{record['record_id']}: source question hash mismatch")
        weight = weights[str(record["record_id"])]
        trajectory = {
            "schema_version": "saeg-silver-trajectory-v1",
            "qid": str(record["record_id"]),
            "source_qid": qid,
            "question_key": str(record["question_key"]),
            "question": str(record["question"]),
            "answer": answer,
            "dataset": dataset,
            "evidence_mode": mode,
            "steps": steps,
            "kg_subgraph": [list(map(str, edge["triple"])) for edge in w_edges],
            "passage_evidence": p_items,
            "retrieved_passages": passages,
            "accepted": True,
            "metadata": {
                "experiment_id": EXPERIMENT_ID,
                "source_graph_record_id": str(record["record_id"]),
                "source_graph_sha256": str(record["graph_sha256"]),
                "source_mode": mode,
                "sft_target_route": target_route,
                "sampling_probability": float(weight["sampling_probability"]),
                "gold_train_only": True,
                "evaluation_eligible": False,
                "teacher_api_used": False,
                "wikidata_construction_gold_access": False,
                "passage_construction_gold_access": bool(p_items),
            },
            "teacher_output": assistant_trace(steps, answer),
            "teacher_model": "deterministic_saeg_dual_citation_adapter_v1",
        }
        validate_trajectory(trajectory)
        trajectories.append(trajectory)
        counts[f"mode::{mode}"] += 1
        counts[f"dataset::{dataset}"] += 1
        counts[f"target::{target_route}"] += 1
        counts["wikidata_triples"] += len(trajectory["kg_subgraph"])
        counts["passage_evidence"] += len(p_items)

    if len(trajectories) != 4862 or len({row["qid"] for row in trajectories}) != 4862:
        raise RuntimeError("expected exactly 4862 unique train-only SAEG variants")
    if abs(sum(float(row["metadata"]["sampling_probability"]) for row in trajectories) - 1.0) > 1e-9:
        raise RuntimeError("sampling probabilities do not sum to one")

    args.out.mkdir(parents=True, exist_ok=False)
    data_path = args.out / "silver_train.jsonl"
    write_jsonl(data_path, trajectories)
    report = {
        "schema_version": "saeg-sft-dataset-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "counts": dict(sorted(counts.items())),
        "integrity": {
            "trajectory_count": len(trajectories),
            "unique_trajectory_id": True,
            "all_citation_contracts_valid": True,
            "passage_pseudo_triples_in_kg_subgraph": 0,
            "all_answers_train_only": True,
            "teacher_api_calls": 0,
            "sampling_probability_sum": sum(
                float(row["metadata"]["sampling_probability"]) for row in trajectories),
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)} for name, path in inputs.items()
        },
        "outputs": {"silver_train": {"path": str(data_path), "sha256": sha256_file(data_path)}},
        "scientific_boundary": (
            "Train-only Gold support/decomposition may construct passage evidence and targets; "
            "Wikidata graph construction is Gold-free. Not evaluation eligible and not yet trained."
        ),
    }
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(args.out, extra=report, status=STATUS)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
