#!/usr/bin/env python
"""Materialise train-only SAEG P/W/fused/no-graph records from audited assets.

The output deliberately excludes answers and SFT target trajectories.  It is a
source/provenance-complete graph asset pool used to decide the later sampling
and target-serialization protocol.  Raw train support annotations are used for
the passage branch and recorded as construction_gold_access=True.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.qpeg import passage_sentences
from kgproweight.kg.source_adaptive_evidence_graph import (
    fuse_edges,
    make_passage_edge,
    make_record,
    make_wikidata_edge,
    passages_sha256,
    validate_record,
)
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.build_qpeg_v4_schema_adaptation_data import (
    _best_answer_sentence,
    _clean,
    _norm,
    build_trajectory,
)


EXPERIMENT_ID = "SAEG-V1-TRAINING-GRAPH-ASSETS"
STATUS = "COMPLETE_TRAIN_ONLY_GRAPH_ASSETS_NOT_SFT_NOT_TRAINED"
CUTOFF = "2020-12-09T23:59:59Z"
ASSET_CAP = 12


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


def _source_qid(row: Mapping[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("source_qid") or "")


def _raw_rows(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("id") or row.get("qid") or "")
            if qid in wanted:
                selected[qid] = row
    if set(selected) != wanted:
        raise ValueError(f"{path}: missing {len(wanted - set(selected))} requested raw qids")
    return selected


def passage_edge_specs(
    dataset: str,
    raw: Mapping[str, Any],
    trajectory: Mapping[str, Any],
) -> list[tuple[list[str], str, int, int]]:
    """Reproduce QPEG-v4 edge selection while retaining source coordinates."""
    passages = list(trajectory.get("retrieved_passages") or [])
    specs: list[tuple[list[str], str, int, int]] = []
    seen: set[tuple[str, str, str]] = set()
    if dataset in {"hotpotqa", "2wikimultihopqa"}:
        metadata = raw.get("metadata") or {}
        context = metadata.get("context") or {}
        titles = list(context.get("title") or [])
        sentence_key = "sentences" if dataset == "hotpotqa" else "content"
        contents = list(context.get(sentence_key) or [])
        by_title = {
            _norm(title): (rank, str(title), [str(value) for value in sentences])
            for rank, (title, sentences) in enumerate(zip(titles, contents))
        }
        support = metadata.get("supporting_facts") or {}
        for title, sentence_index in zip(support.get("title") or [], support.get("sent_id") or []):
            source = by_title.get(_norm(title))
            if source is None:
                continue
            passage_rank, canonical_title, sentences = source
            index = int(sentence_index)
            if not 0 <= index < len(sentences):
                continue
            triple = (_clean(canonical_title), "evidence sentence", _clean(sentences[index]))
            if not all(triple) or triple in seen:
                continue
            seen.add(triple)
            specs.append((list(triple), str(passages[passage_rank]["id"]), passage_rank, index))
            if len(specs) == 4:
                break
    else:
        decomposition = (((raw.get("metadata") or {}).get("metadata") or {}).get("question_decomposition") or [])
        passage_rank = 0
        for step in decomposition:
            support = step.get("support_paragraph") or {}
            title = _clean(support.get("title") or f"support-{passage_rank}")
            text = _clean(support.get("paragraph_text"))
            if not text:
                continue
            sentence = _best_answer_sentence(text, _clean(step.get("answer")))
            triple = (title, "evidence sentence", sentence)
            current_passage = passages[passage_rank]
            sentence_values = passage_sentences(current_passage)
            matches = [
                index for index, value in enumerate(sentence_values)
                if _clean(value) == sentence
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{dataset}/{raw.get('id')}: passage sentence provenance is not unique"
                )
            if all(triple) and triple not in seen:
                seen.add(triple)
                specs.append((list(triple), str(current_passage["id"]), passage_rank, matches[0]))
                if len(specs) == 4:
                    break
            passage_rank += 1
    expected = [list(value) for value in trajectory.get("kg_subgraph") or []]
    actual = [value[0] for value in specs]
    if actual != expected:
        raise ValueError(f"{dataset}/{raw.get('id')}: passage provenance replay drift")
    return specs


def make_passage_edges(
    dataset: str,
    raw: Mapping[str, Any],
    trajectory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        make_passage_edge(
            edge_index=index,
            triple=triple,
            passage_id=passage_id,
            passage_rank=passage_rank,
            sentence_index=sentence_index,
            construction_gold_access=True,
        )
        for index, (triple, passage_id, passage_rank, sentence_index) in enumerate(
            passage_edge_specs(dataset, raw, trajectory), start=1
        )
    ]


def make_wikidata_edges(
    record: Mapping[str, Any], runtime: Mapping[str, Any]
) -> list[dict[str, Any]]:
    runtime_hops = list((runtime.get("execution") or {}).get("hops") or [])
    builder_version = str((runtime.get("provenance") or {}).get("builder_version") or "")
    edges: list[dict[str, Any]] = []
    for index, triple in enumerate(record.get("kg_subgraph") or [], start=1):
        matching_hops = [hop for hop in runtime_hops if list(triple) in (hop.get("matches") or [])]
        if len(matching_hops) != 1:
            raise ValueError(f"{record.get('qid')}: Wikidata triple must map to exactly one runtime hop")
        hop = matching_hops[0]
        pids = list(hop.get("pids") or [])
        input_qids = [
            str(entity.get("qid")) for entity in (hop.get("input_entities") or [])
            if entity.get("qid")
        ]
        if len(pids) != 1:
            raise ValueError(f"{record.get('qid')}: runtime hop does not have exactly one PID")
        tail_surface = _norm(triple[2])
        output_qids = [
            str(entity.get("qid")) for entity in (hop.get("output_entities") or [])
            if entity.get("qid") and _norm(entity.get("surface") or entity.get("label")) == tail_surface
        ]
        edges.append(make_wikidata_edge(
            edge_index=index,
            triple=triple,
            hop_index=int(hop["hop_index"]),
            input_qids=input_qids,
            pid=str(pids[0]),
            tail_qid=output_qids[0] if len(output_qids) == 1 else None,
            cutoff=CUTOFF,
            builder_version=builder_version,
        ))
    return edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate_pool",
        type=Path,
        default=Path("outputs/audits/saeg_v1_training_candidate_pool_v1/candidate_pool.identity_only.jsonl"),
    )
    parser.add_argument(
        "--qpeg_silver",
        type=Path,
        default=Path("data/silver_data/qpeg_v4_schema_adaptation_n2400_seed42_v2/silver_curriculum.jsonl"),
    )
    parser.add_argument(
        "--proof_silver",
        type=Path,
        default=Path("data/silver_data/automatic_proofkg_2wiki_train_k4_v1/silver_train.jsonl"),
    )
    parser.add_argument(
        "--proof_records",
        type=Path,
        default=Path("data/silver_data/automatic_proofkg_2wiki_train_k4_v1/question_kg_records.jsonl"),
    )
    parser.add_argument(
        "--proof_runtime",
        type=Path,
        default=Path("outputs/audits/automatic_proofkg_2wiki_train_k4_v1_n1500_seed42_historical_stage3_runtime/runtime_details.jsonl"),
    )
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument(
        "--out", type=Path, default=Path("data/derived/saeg_v1_training_graph_assets_v1")
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite graph assets: {args.out}")
    inputs = {
        "candidate_pool": args.candidate_pool,
        "qpeg_silver": args.qpeg_silver,
        "proof_silver": args.proof_silver,
        "proof_records": args.proof_records,
        "proof_runtime": args.proof_runtime,
    }
    for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
        inputs[f"raw_{dataset}_train"] = args.data_root / dataset / "train.jsonl"
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    candidates = read_jsonl(args.candidate_pool)
    qpeg_rows = read_jsonl(args.qpeg_silver)
    qpeg_by_mode_qid: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in qpeg_rows:
        variant = str((row.get("metadata") or {}).get("curriculum_variant") or "")
        mode = "P_ONLY" if variant == "qpeg" else "N_REPLAY"
        key = (str(row["dataset"]), _source_qid(row), mode)
        qpeg_by_mode_qid[key] = row
    proof_silver = {str(row["qid"]): row for row in read_jsonl(args.proof_silver)}
    proof_records = {str(row["qid"]): row for row in read_jsonl(args.proof_records)}
    proof_runtime = {str(row["qid"]): row for row in read_jsonl(args.proof_runtime)}

    raw_wanted: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        if candidate["source_mode"] in {"P_ONLY", "P_W_FUSED"}:
            raw_wanted[str(candidate["dataset"])].add(str(candidate["qid"]))
    raw_by_dataset = {
        dataset: _raw_rows(inputs[f"raw_{dataset}_train"], wanted)
        for dataset, wanted in raw_wanted.items()
    }

    records: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    fusion_telemetry: Counter[str] = Counter()
    for candidate in candidates:
        dataset = str(candidate["dataset"])
        qid = str(candidate["qid"])
        mode = str(candidate["source_mode"])
        if mode in {"P_ONLY", "N_REPLAY"}:
            trajectory = qpeg_by_mode_qid[(dataset, qid, mode)]
        else:
            trajectory = proof_silver[qid]
        question = str(trajectory["question"])
        passages = list(trajectory.get("retrieved_passages") or [])
        p_edges: list[dict[str, Any]] = []
        w_edges: list[dict[str, Any]] = []
        if mode == "P_ONLY":
            p_edges = make_passage_edges(dataset, raw_by_dataset[dataset][qid], trajectory)
            edges = p_edges
            routing = {"cap": ASSET_CAP, "passage_status": "gold_train_support", "wikidata_status": "not_requested"}
        elif mode == "N_REPLAY":
            edges = []
            routing = {"cap": ASSET_CAP, "fallback_reason": "balanced_no_graph_training_replay"}
        else:
            proof_record = proof_records[qid]
            runtime = proof_runtime[qid]
            if proof_record.get("question_sha256") != runtime.get("question_sha256"):
                raise ValueError(f"{qid}: ProofKG runtime question hash mismatch")
            if proof_record.get("kg_subgraph") != runtime.get("kg_subgraph"):
                raise ValueError(f"{qid}: ProofKG runtime graph mismatch")
            if not (runtime.get("execution") or {}).get("complete_plan_execution"):
                raise ValueError(f"{qid}: ProofKG runtime is incomplete")
            if (runtime.get("provenance") or {}).get("gold_access") is not False:
                raise ValueError(f"{qid}: ProofKG runtime used Gold")
            w_edges = make_wikidata_edges(proof_record, runtime)
            if mode == "W_ONLY":
                edges = w_edges
                routing = {"cap": ASSET_CAP, "passage_status": "not_requested", "wikidata_status": "complete"}
            else:
                rebuilt = build_trajectory(dataset, raw_by_dataset[dataset][qid], graph=True)
                if rebuilt["retrieved_passages"] != passages:
                    # IDs/source fields differ across old builders, but the
                    # semantic sequence must be identical before P ranks can be
                    # transferred to ProofKG's canonical context.
                    if passages_sha256(rebuilt["retrieved_passages"]) != passages_sha256(passages):
                        raise ValueError(f"{qid}: ordered passage context mismatch")
                    rebuilt = dict(rebuilt)
                    rebuilt["retrieved_passages"] = passages
                p_edges = make_passage_edges(dataset, raw_by_dataset[dataset][qid], rebuilt)
                edges, telemetry = fuse_edges(w_edges, p_edges, cap=ASSET_CAP)
                if {edge["source_type"] for edge in edges} != {"passage", "wikidata"}:
                    raise ValueError(f"{qid}: fused asset lost one source")
                for key, value in telemetry.items():
                    fusion_telemetry[key] += value
                routing = {
                    "cap": ASSET_CAP,
                    "passage_status": "gold_train_support",
                    "wikidata_status": "complete",
                    "fusion_telemetry": telemetry,
                }
        record = make_record(
            dataset=dataset,
            qid=qid,
            question=question,
            passages=passages,
            routing_mode=mode,
            edges=edges,
            routing=routing,
        )
        if record["question_sha256"] != candidate["question_sha256"]:
            raise ValueError(f"{record['record_id']}: candidate question hash mismatch")
        validate_record(record)
        records.append(record)
        counters[f"mode::{mode}"] += 1
        counters[f"dataset::{dataset}"] += 1
        counters[f"source_edges::passage"] += len(p_edges)
        counters[f"source_edges::wikidata"] += len(w_edges)

    if len(records) != len(candidates):
        raise RuntimeError("candidate/record count mismatch")
    if len({record["record_id"] for record in records}) != len(records):
        raise RuntimeError("duplicate SAEG record_id")
    args.out.mkdir(parents=True, exist_ok=False)
    records_path = args.out / "question_graph_records.jsonl"
    write_jsonl(records_path, records)
    report = {
        "schema_version": "saeg-training-graph-assets-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "counts": {
            "records": len(records),
            "unique_record_ids": len({record["record_id"] for record in records}),
            "unique_dataset_qids": len({record["question_key"] for record in records}),
            "by_mode": dict(Counter((record["routing"] or {})["mode"] for record in records)),
            "by_dataset": dict(Counter(record["dataset"] for record in records)),
            "edges_by_source_in_record_outputs": dict(Counter(
                edge["source_type"] for record in records for edge in record["edges"]
            )),
            "empty_graph_records": sum(not record["edges"] for record in records),
        },
        "integrity": {
            "schema_valid_rate": 1.0,
            "identity_unique": True,
            "question_hash_join_rate": 1.0,
            "graph_hash_valid_rate": 1.0,
            "all_evaluation_eligible_false": all(record["evaluation_eligible"] is False for record in records),
            "wikidata_gold_access_false": True,
            "passage_gold_use_explicit": all(
                edge["construction_gold_access"] is True
                for record in records for edge in record["edges"]
                if edge["source_type"] == "passage"
            ),
            "fused_records_retain_both_sources": all(
                {edge["source_type"] for edge in record["edges"]} == {"passage", "wikidata"}
                for record in records if record["routing"]["mode"] == "P_W_FUSED"
            ),
            "asset_cap": ASSET_CAP,
            "prompt_cap": "NOT_YET_FROZEN",
        },
        "fusion_telemetry_aggregate": dict(fusion_telemetry),
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in inputs.items()},
        "output": {"path": str(records_path), "sha256": sha256_file(records_path)},
        "scientific_boundary": (
            "These are train-only graph assets without answers or SFT targets. Passage edges use raw-train "
            "Gold support and are not answer-free automatic construction. Wikidata edges are answer-free. "
            "No sampling policy, prompt serialization, loss, reward, evaluation protocol, or training run is authorized here."
        ),
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra=report, status=STATUS)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
