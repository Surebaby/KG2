#!/usr/bin/env python
"""Audit whether existing QPEG and ProofKG assets can form SAEG train pairs.

This is a CPU-only, non-materialising audit.  It never writes questions,
answers, passages, support annotations, or graph contents to its outputs.  The
only per-question output contains identities, hashes, booleans, and counts.

The key distinction is between:

* direct pairs: an existing QPEG-v4 graph row and ProofKG row share a qid;
* rebuildable pairs: a ProofKG qid can deterministically obtain a passage graph
  from the raw *training* support annotations used by QPEG-v4.

Rebuildability is a capacity estimate, not a materialised dataset and not an
evaluation result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.utils.logging import dump_manifest
from scripts.prepare.build_qpeg_v4_schema_adaptation_data import build_trajectory
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256 as lexical_family_sha256


EXPERIMENT_ID = "SAEG-V1-TRAINING-ASSET-OVERLAP-AUDIT"
STATUS = "COMPLETE_DATASET_PREPARATION_AUDIT_NOT_MATERIALIZED"
_SPACE = re.compile(r"\s+")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return _SPACE.sub(" ", text).strip()


def passage_title(passage: Mapping[str, Any]) -> str:
    title = normalise_text(passage.get("title"))
    if title:
        return title
    contents = str(passage.get("contents") or "")
    return normalise_text(contents.splitlines()[0] if contents else "")


def passage_body(passage: Mapping[str, Any]) -> str:
    contents = str(passage.get("contents") or "")
    lines = contents.splitlines()
    if lines and normalise_text(lines[0]) == passage_title(passage):
        contents = " ".join(lines[1:])
    return normalise_text(contents)


def passage_signatures(passages: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    titles = {passage_title(passage) for passage in passages if passage_title(passage)}
    records = {
        f"{passage_title(passage)}\0{passage_body(passage)}"
        for passage in passages
        if passage_title(passage) or passage_body(passage)
    }
    return titles, records


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _qid(row: Mapping[str, Any]) -> str:
    return str(row.get("id") or row.get("qid") or "")


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def _input(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qpeg_silver",
        type=Path,
        default=Path("data/silver_data/qpeg_v4_schema_adaptation_n2400_seed42_v2/silver_curriculum.jsonl"),
    )
    parser.add_argument(
        "--qpeg_protocol_dir",
        type=Path,
        default=Path("outputs/audits/qpeg_v4_schema_adaptation_protocol_v1"),
    )
    parser.add_argument(
        "--proofkg_silver",
        type=Path,
        default=Path("data/silver_data/automatic_proofkg_2wiki_train_k4_v1/silver_train.jsonl"),
    )
    parser.add_argument(
        "--proofkg_records",
        type=Path,
        default=Path("data/silver_data/automatic_proofkg_2wiki_train_k4_v1/question_kg_records.jsonl"),
    )
    parser.add_argument(
        "--proofkg_cohort",
        type=Path,
        default=Path("outputs/audits/automatic_proofkg_2wiki_train_k4_v1_n1500_seed42_preregistration/cohort.question_only.jsonl"),
    )
    parser.add_argument(
        "--raw_2wiki_train", type=Path, default=Path("data/2wikimultihopqa/train.jsonl")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/audits/saeg_v1_training_asset_overlap_v2")
    )
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite audit output: {args.out}")
    input_paths = {
        "qpeg_silver": args.qpeg_silver,
        "qpeg_train": args.qpeg_protocol_dir / "train.question_only.jsonl",
        "qpeg_development": args.qpeg_protocol_dir / "development.question_only.jsonl",
        "qpeg_confirmation": args.qpeg_protocol_dir / "confirmation.question_only.jsonl",
        "proofkg_silver": args.proofkg_silver,
        "proofkg_records": args.proofkg_records,
        "proofkg_cohort": args.proofkg_cohort,
        "raw_2wiki_train": args.raw_2wiki_train,
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    args.out.mkdir(parents=True, exist_ok=False)

    qpeg_silver = read_jsonl(args.qpeg_silver)
    qpeg_graph = [
        row for row in qpeg_silver
        if (row.get("metadata") or {}).get("curriculum_variant") == "qpeg"
    ]
    qpeg_replay = [
        row for row in qpeg_silver
        if (row.get("metadata") or {}).get("curriculum_variant") == "no_graph_replay"
    ]
    qpeg_train = read_jsonl(input_paths["qpeg_train"])
    qpeg_development = read_jsonl(input_paths["qpeg_development"])
    qpeg_confirmation = read_jsonl(input_paths["qpeg_confirmation"])

    proof_silver_rows = read_jsonl(args.proofkg_silver)
    proof_record_rows = read_jsonl(args.proofkg_records)
    proof_cohort_rows = read_jsonl(args.proofkg_cohort)

    proof_silver = {_qid(row): row for row in proof_silver_rows}
    proof_records = {_qid(row): row for row in proof_record_rows}
    proof_cohort = {_qid(row): row for row in proof_cohort_rows}
    proof_qids = set(proof_silver)
    if len(proof_silver) != len(proof_silver_rows):
        raise ValueError("duplicate qid in ProofKG silver")
    if len(proof_records) != len(proof_record_rows):
        raise ValueError("duplicate qid in ProofKG records")
    if proof_qids != set(proof_records):
        raise ValueError("ProofKG silver/question-record qid sets differ")
    if not proof_qids <= set(proof_cohort):
        raise ValueError("ProofKG materialized qids are not a subset of the frozen cohort")

    qpeg_graph_by_source = {
        str((row.get("metadata") or {}).get("source_qid") or ""): row for row in qpeg_graph
    }
    if "" in qpeg_graph_by_source or len(qpeg_graph_by_source) != len(qpeg_graph):
        raise ValueError("missing or duplicate QPEG graph source_qid")
    qpeg_2wiki_qids = {
        qid for qid, row in qpeg_graph_by_source.items()
        if row.get("dataset") == "2wikimultihopqa"
    }
    qpeg_family = {
        _qid(row): lexical_family_sha256(str(row["question"]))
        for row in qpeg_train if row.get("dataset") == "2wikimultihopqa"
    }
    # The planner split's stored family hash includes target PID structure,
    # while QPEG uses an answer-free lexical family.  Those hash namespaces are
    # intentionally different and must never be compared directly.  Recompute
    # every cross-protocol comparison with QPEG's answer-free lexical function.
    proof_family = {
        qid: lexical_family_sha256(str(proof_records[qid]["question"]))
        for qid in proof_qids
    }
    dev_qids = {_qid(row) for row in qpeg_development if row.get("dataset") == "2wikimultihopqa"}
    confirmation_qids = {
        _qid(row) for row in qpeg_confirmation if row.get("dataset") == "2wikimultihopqa"
    }
    dev_families = {
        lexical_family_sha256(str(row["question"]))
        for row in qpeg_development if row.get("dataset") == "2wikimultihopqa"
    }
    confirmation_families = {
        lexical_family_sha256(str(row["question"]))
        for row in qpeg_confirmation if row.get("dataset") == "2wikimultihopqa"
    }
    heldout_families = dev_families | confirmation_families

    raw_selected: dict[str, dict[str, Any]] = {}
    with args.raw_2wiki_train.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = _qid(row)
            if qid in proof_qids:
                raw_selected[qid] = row
    if set(raw_selected) != proof_qids:
        missing_raw = sorted(proof_qids - set(raw_selected))
        raise ValueError(f"ProofKG qids missing from raw 2Wiki train: {missing_raw[:10]}")

    audit_rows: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    title_jaccards: list[float] = []
    body_jaccards: list[float] = []
    direct_overlap = proof_qids & qpeg_2wiki_qids
    direct_question_exact = 0
    direct_raw_passage_json_exact = 0
    all_complete = True
    all_gold_false = True
    for qid in sorted(proof_qids):
        silver = proof_silver[qid]
        record = proof_records[qid]
        cohort = proof_cohort[qid]
        if str(silver.get("question", "")).strip() != str(record.get("question", "")).strip():
            raise ValueError(f"ProofKG question mismatch: {qid}")
        if str(record.get("question_sha256")) != str(cohort.get("question_sha256")):
            raise ValueError(f"ProofKG question hash mismatch: {qid}")
        provenance = record.get("provenance") or {}
        complete = provenance.get("complete_plan_execution") is True
        gold_false = provenance.get("gold_access") is False
        all_complete = all_complete and complete
        all_gold_false = all_gold_false and gold_false

        rebuilt = False
        failure_reason = None
        p_edges = 0
        titles_exact = False
        bodies_exact = False
        title_score = 0.0
        body_score = 0.0
        try:
            trajectory = build_trajectory("2wikimultihopqa", raw_selected[qid], graph=True)
            rebuilt = True
            p_edges = len(trajectory["kg_subgraph"])
            p_titles, p_bodies = passage_signatures(trajectory["retrieved_passages"])
            w_titles, w_bodies = passage_signatures(silver.get("retrieved_passages") or [])
            titles_exact = p_titles == w_titles
            bodies_exact = p_bodies == w_bodies
            title_score = jaccard(p_titles, w_titles)
            body_score = jaccard(p_bodies, w_bodies)
            title_jaccards.append(title_score)
            body_jaccards.append(body_score)
            if qid in direct_overlap:
                existing = qpeg_graph_by_source[qid]
                direct_question_exact += int(
                    str(existing.get("question", "")).strip() == str(silver.get("question", "")).strip()
                )
                direct_raw_passage_json_exact += int(
                    existing.get("retrieved_passages") == silver.get("retrieved_passages")
                )
        except ValueError as exc:
            failure_reason = str(exc).split(": ", 1)[-1]
            failures[failure_reason] += 1

        audit_rows.append({
            "dataset": "2wikimultihopqa",
            "qid": qid,
            "question_sha256": str(record["question_sha256"]),
            "family_sha256": proof_family[qid],
            "direct_existing_qpeg": qid in direct_overlap,
            "passage_branch_rebuildable": rebuilt,
            "rebuild_failure_reason": failure_reason,
            "passage_edge_count": p_edges,
            "wikidata_edge_count": len(record.get("kg_subgraph") or []),
            "context_title_set_exact": titles_exact,
            "context_title_jaccard": round(title_score, 6),
            "context_body_set_exact": bodies_exact,
            "context_body_jaccard": round(body_score, 6),
            "excluded_by_current_qpeg_v4_eval_qid": qid in dev_qids or qid in confirmation_qids,
            "excluded_by_current_qpeg_v4_eval_family": proof_family[qid] in heldout_families,
            "proof_complete": complete,
            "proof_gold_access_false": gold_false,
        })

    rebuildable = [row for row in audit_rows if row["passage_branch_rebuildable"]]
    safe_rebuildable = [
        row for row in rebuildable if not row["excluded_by_current_qpeg_v4_eval_family"]
    ]
    proof_families = set(proof_family.values())
    qpeg_2wiki_families = {qpeg_family[qid] for qid in qpeg_2wiki_qids}
    graph_source_qids = {
        str((row.get("metadata") or {}).get("source_qid")) for row in qpeg_graph
    }
    replay_source_qids = {
        str((row.get("metadata") or {}).get("source_qid")) for row in qpeg_replay
    }

    rows_path = args.out / "audit_rows.jsonl"
    write_jsonl(rows_path, audit_rows)
    report = {
        "schema_version": "saeg-training-asset-overlap-audit-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "scope": "CPU-only data preparation audit; no training set or evaluation cohort was materialized",
        "inputs": {name: _input(path) for name, path in input_paths.items()},
        "integrity": {
            "proof_silver_record_identity_join_rate": 1.0,
            "proof_materialized_subset_of_frozen_cohort": True,
            "proof_raw_train_qid_resolution_rate": len(raw_selected) / len(proof_qids),
            "all_proof_complete": all_complete,
            "all_proof_gold_access_false": all_gold_false,
            "audit_rows_contain_no_question_answer_passage_or_graph_text": True,
            "cross_protocol_family_hash": "recomputed with answer-free lexical family v1",
            "stored_qpeg_and_planner_family_hashes_compared_directly": False,
        },
        "existing_assets": {
            "qpeg_graph_total": len(qpeg_graph),
            "qpeg_graph_by_dataset": dict(Counter(str(row["dataset"]) for row in qpeg_graph)),
            "qpeg_no_graph_replay_total": len(qpeg_replay),
            "qpeg_replay_source_qids_also_have_graph": len(replay_source_qids & graph_source_qids),
            "proofkg_complete_2wiki_total": len(proof_qids),
        },
        "overlap": {
            "qpeg_2wiki_qids": len(qpeg_2wiki_qids),
            "proofkg_qids": len(proof_qids),
            "direct_qid_overlap": len(direct_overlap),
            "direct_question_exact": direct_question_exact,
            "direct_raw_passage_json_exact": direct_raw_passage_json_exact,
            "family_overlap": len(qpeg_2wiki_families & proof_families),
            "proof_qid_overlap_current_qpeg_v4_development": len(proof_qids & dev_qids),
            "proof_qid_overlap_current_qpeg_v4_confirmation": len(proof_qids & confirmation_qids),
            "proof_family_overlap_current_qpeg_v4_development": len(proof_families & dev_families),
            "proof_family_overlap_current_qpeg_v4_confirmation": len(proof_families & confirmation_families),
        },
        "rebuild_capacity": {
            "proofkg_qids_audited": len(proof_qids),
            "passage_branch_rebuildable": len(rebuildable),
            "passage_branch_rebuildable_rate": len(rebuildable) / len(proof_qids),
            "safe_after_excluding_current_qpeg_v4_eval_families": len(safe_rebuildable),
            "failures": dict(failures),
            "context_title_set_exact": sum(row["context_title_set_exact"] for row in rebuildable),
            "context_body_set_exact": sum(row["context_body_set_exact"] for row in rebuildable),
            "context_title_jaccard_mean": sum(title_jaccards) / len(title_jaccards),
            "context_title_jaccard_p50": _percentile(title_jaccards, 0.5),
            "context_body_jaccard_mean": sum(body_jaccards) / len(body_jaccards),
            "context_body_jaccard_p50": _percentile(body_jaccards, 0.5),
            "passage_edge_count_distribution": dict(Counter(row["passage_edge_count"] for row in rebuildable)),
            "wikidata_edge_count_distribution": dict(Counter(row["wikidata_edge_count"] for row in audit_rows)),
        },
        "candidate_inventory_not_materialized": {
            "P_only_existing": {
                "hotpotqa": 600,
                "2wikimultihopqa": 600,
                "musique": 600,
            },
            "W_only_existing_2wiki": len(proof_qids),
            "P_plus_W_direct_existing_2wiki": len(direct_overlap),
            "P_plus_W_rebuildable_2wiki": len(rebuildable),
            "P_plus_W_rebuildable_safe_against_current_qpeg_v4_eval_families": len(safe_rebuildable),
            "N_existing_replay": len(qpeg_replay),
        },
        "decision": {
            "direct_join_sufficient": len(direct_overlap) >= 500,
            "rebuild_required_for_substantive_fused_arm": len(direct_overlap) < 500,
            "recommended_next_step": (
                "Freeze a separate SAEG-v1 training-data protocol, then materialize passage edges "
                "for eligible ProofKG qids from raw-train support annotations. Choose and hash one "
                "canonical passage context before fusion; do not mix unequal contexts silently."
            ),
        },
        "scientific_boundary": (
            "Passage rebuildability uses Gold supporting facts from raw TRAIN only and is SFT-schema "
            "supervision. It is not answer-free automatic graph construction, not an evaluation result, "
            "and not evidence that a trained model benefits from fusion."
        ),
        "outputs": {"audit_rows": {"path": str(rows_path), "sha256": sha256_file(rows_path)}},
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra=report, status=STATUS)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
