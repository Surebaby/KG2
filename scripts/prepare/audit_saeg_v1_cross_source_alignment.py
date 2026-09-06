#!/usr/bin/env python
"""Audit answer-free hop alignment between P and W edges in fused SAEG assets."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Mapping, Sequence

from kgproweight.utils.logging import dump_manifest


EXPERIMENT_ID = "SAEG-V1-CROSS-SOURCE-ALIGNMENT-AUDIT"
STATUS = "COMPLETE_TRAIN_ONLY_ALIGNMENT_AUDIT_NOT_SFT"


def _tokens(value: object) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", str(value)).casefold()))


def _contains(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    return bool(needle) and any(
        haystack[index : index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


def endpoint_score(
    wikidata_edges: Sequence[Mapping[str, Any]],
    passage_edge: Mapping[str, Any],
    *,
    hop_index: int,
    passage_index: int,
) -> tuple[float, dict[str, bool]]:
    passage_head, _, passage_tail = map(_tokens, passage_edge["triple"])
    passage_blob = passage_head + passage_tail
    tail_match = False
    head_match = False
    title_match = False
    for edge in wikidata_edges:
        w_head, _, w_tail = map(_tokens, edge["triple"])
        tail_match = tail_match or _contains(w_tail, passage_blob)
        head_match = head_match or _contains(w_head, passage_blob)
        title_match = title_match or _contains(passage_head, w_head + w_tail)
    components = {
        "wikidata_tail_in_passage": tail_match,
        "wikidata_head_in_passage": head_match,
        "passage_title_in_wikidata_endpoints": title_match,
        "same_ordinal": hop_index == passage_index,
    }
    score = 4 * tail_match + 2 * head_match + title_match + 0.25 * components["same_ordinal"]
    return float(score), components


def align_record(record: Mapping[str, Any]) -> dict[str, Any]:
    passage_edges = [edge for edge in record["edges"] if edge["source_type"] == "passage"]
    by_hop: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for edge in record["edges"]:
        if edge["source_type"] == "wikidata":
            by_hop[int(edge["provenance"]["hop_index"])].append(edge)
    unused = set(range(1, len(passage_edges) + 1))
    alignments = []
    for hop_index, values in sorted(by_hop.items()):
        candidates = []
        for passage_index in sorted(unused):
            score, components = endpoint_score(
                values,
                passage_edges[passage_index - 1],
                hop_index=hop_index,
                passage_index=passage_index,
            )
            semantic_score = score - (0.25 if components["same_ordinal"] else 0.0)
            candidates.append((score, semantic_score, passage_index, components))
        candidates.sort(key=lambda value: (-value[0], value[2]))
        if candidates and candidates[0][1] > 0:
            score, semantic_score, passage_index, components = candidates[0]
            unused.remove(passage_index)
            alignments.append({
                "hop_index": hop_index,
                "wikidata_edge_ids": [str(edge["edge_id"]) for edge in values],
                "passage_edge_id": str(passage_edges[passage_index - 1]["edge_id"]),
                "score": score,
                "semantic_score": semantic_score,
                "components": components,
            })
        else:
            alignments.append({
                "hop_index": hop_index,
                "wikidata_edge_ids": [str(edge["edge_id"]) for edge in values],
                "passage_edge_id": None,
                "score": 0.0,
                "semantic_score": 0.0,
                "components": {},
            })
    aligned = sum(value["passage_edge_id"] is not None for value in alignments)
    return {
        "record_id": str(record["record_id"]),
        "dataset": str(record["dataset"]),
        "qid": str(record["qid"]),
        "question_sha256": str(record["question_sha256"]),
        "n_wikidata_hops": len(by_hop),
        "n_passage_edges": len(passage_edges),
        "aligned_hops": aligned,
        "all_hops_one_to_one_aligned": aligned == len(by_hop),
        "sft_target_route": "P_W_JOINT" if aligned == len(by_hop) else "W_ONLY_FAIL_CLOSED",
        "alignments": alignments,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph_assets",
        type=Path,
        default=Path("data/derived/saeg_v1_training_graph_assets_v1/question_graph_records.jsonl"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/audits/saeg_v1_cross_source_alignment_v1")
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite alignment audit: {args.out}")
    if not args.graph_assets.is_file():
        raise FileNotFoundError(args.graph_assets)
    fused = []
    with args.graph_assets.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if (record.get("routing") or {}).get("mode") == "P_W_FUSED":
                fused.append(record)
    rows = [align_record(record) for record in fused]
    args.out.mkdir(parents=True, exist_ok=False)
    rows_path = args.out / "alignment_rows.identity_only.jsonl"
    with rows_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    total_hops = sum(row["n_wikidata_hops"] for row in rows)
    aligned_hops = sum(row["aligned_hops"] for row in rows)
    jointly_aligned = sum(row["all_hops_one_to_one_aligned"] for row in rows)
    report = {
        "schema_version": "saeg-cross-source-alignment-audit-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "method": (
            "Answer-free endpoint matching; W tail match weight=4, W head=2, passage-title=1, "
            "same-ordinal tie break=0.25; one passage edge can align to at most one hop."
        ),
        "counts": {
            "fused_records": len(rows),
            "wikidata_hops": total_hops,
            "one_to_one_aligned_hops": aligned_hops,
            "all_hops_aligned_records": jointly_aligned,
            "fail_closed_w_only_records": len(rows) - jointly_aligned,
        },
        "rates": {
            "one_to_one_hop_alignment": aligned_hops / total_hops,
            "all_hops_aligned_record": jointly_aligned / len(rows),
        },
        "integrity": {
            "gold_answer_access": False,
            "question_text_written_to_output": False,
            "edge_text_written_to_output": False,
            "identity_rows_unique": len(rows) == len({row["record_id"] for row in rows}),
        },
        "input": {"path": str(args.graph_assets), "sha256": _sha256(args.graph_assets)},
        "output": {"path": str(rows_path), "sha256": _sha256(rows_path)},
        "recommendation_not_yet_frozen": (
            "Use joint P+W hop targets only for all-hops one-to-one aligned records; route the remaining "
            "records to W-only target supervision. Keep their fused input graph for a separate distractor-robustness ablation."
        ),
        "scientific_boundary": "Mechanical train-only alignment diagnostic; not semantic correctness and not model utility.",
    }
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(args.out, extra=report, status=STATUS)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
