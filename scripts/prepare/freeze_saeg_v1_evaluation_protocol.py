#!/usr/bin/env python
"""Freeze SAEG-v1 development, untouched confirmation, and reporting cohorts.

This stage is answer-free.  It binds question/retrieval/Passage-QPEG identities
and creates the 2Wiki-only relation-planner cohort.  Gold answers are exported
later into a physically separate scorer-only file.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.utils.logging import dump_manifest


EXPERIMENT_ID = "SAEG-V1-EVALUATION-PROTOCOL-SEED42"
STATUS = "FROZEN_ANSWER_FREE_BEFORE_SAEG_DEVELOPMENT"
FORBIDDEN = {
    "answer", "answers", "golden_answers", "target", "supporting_facts",
    "question_decomposition", "evidences",
}


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


def assert_answer_free(value: Any, location: str = "row") -> None:
    if isinstance(value, Mapping):
        bad = FORBIDDEN.intersection(map(str, value.keys()))
        if bad:
            raise ValueError(f"forbidden answer/Gold fields at {location}: {sorted(bad)}")
        for key, child in value.items():
            assert_answer_free(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_answer_free(child, f"{location}[{index}]")


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        key = str(row["question_key"])
        if key in output:
            raise ValueError(f"duplicate {label} question_key: {key}")
        output[key] = row
    return output


def _cohort_row(context: Mapping[str, Any], graph: Mapping[str, Any], role: str) -> dict[str, Any]:
    if context["question_sha256"] != graph["question_sha256"]:
        raise ValueError(f"{context['question_key']}: question hash mismatch")
    if context["passages_sha256"] != graph["passages_sha256"]:
        raise ValueError(f"{context['question_key']}: passages hash mismatch")
    if context.get("gold_access") is not False or graph.get("gold_access") is not False:
        raise ValueError(f"{context['question_key']}: graph/context accessed Gold")
    row = {
        "schema_version": "saeg-eval-cohort-v1",
        "question_key": str(context["question_key"]),
        "dataset": str(context["dataset"]),
        "qid": str(context["qid"]),
        "question": str(context["question"]),
        "question_sha256": str(context["question_sha256"]),
        "family_sha256": str(context["family_sha256"]),
        "role": role,
        "gold_access": False,
        "passages_sha256": str(context["passages_sha256"]),
        "passage_graph_sha256": str(graph["qpeg_sha256"]),
        "passage_graph_nonempty": bool(graph.get("edges")),
    }
    assert_answer_free(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh_contexts", type=Path, default=Path(
        "outputs/audits/qpeg_v4_schema_adaptation_eval_retrieval_v1/retrieval_contexts.jsonl"))
    parser.add_argument("--fresh_passage_graphs", type=Path, default=Path(
        "data/derived/qpeg_v4_schema_adaptation_eval450_seed42/question_graph_records.jsonl"))
    parser.add_argument("--canonical_contexts", type=Path, default=Path(
        "outputs/audits/qpeg_v1_n1350_seed42_preregistration/final.retrieval_contexts.jsonl"))
    parser.add_argument("--canonical_passage_graphs", type=Path, default=Path(
        "data/derived/qpeg_v3_sentence_selector_final900_seed42_v2/question_graph_records.jsonl"))
    parser.add_argument("--canonical_2wiki_proof", type=Path, default=Path(
        "data/derived/inference_proofkg_v1_2wiki_dev_n300_v1/question_kg_records.jsonl"))
    parser.add_argument("--qpeg_v4_selection", type=Path, default=Path(
        "outputs/audits/qpeg_v4_schema_adaptation_development_scores_v1/checkpoint_selection.json"))
    parser.add_argument("--out", type=Path, default=Path(
        "outputs/audits/saeg_v1_evaluation_protocol_v1"))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite SAEG evaluation protocol: {args.out}")
    inputs = {
        "fresh_contexts": args.fresh_contexts,
        "fresh_passage_graphs": args.fresh_passage_graphs,
        "canonical_contexts": args.canonical_contexts,
        "canonical_passage_graphs": args.canonical_passage_graphs,
        "canonical_2wiki_proof": args.canonical_2wiki_proof,
        "qpeg_v4_selection": args.qpeg_v4_selection,
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    selection = json.loads(args.qpeg_v4_selection.read_text(encoding="utf-8"))
    if selection.get("status") != "FAIL_STOP_DEVELOPMENT" or selection.get("confirmation_opened") is not False:
        raise ValueError("QPEG-v4 must be fail-stopped with confirmation unopened")

    fresh_contexts = _index(read_jsonl(args.fresh_contexts), "fresh context")
    fresh_graphs = _index(read_jsonl(args.fresh_passage_graphs), "fresh graph")
    if set(fresh_contexts) != set(fresh_graphs):
        raise ValueError("fresh context/graph identity join is not 1.0")
    role_rows = {"development": [], "confirmation": []}
    for key, context in fresh_contexts.items():
        role = str(context["role"])
        if role not in role_rows or fresh_graphs[key].get("role") != role:
            raise ValueError(f"{key}: invalid fresh role")
        role_rows[role].append(_cohort_row(context, fresh_graphs[key], role))

    canonical_contexts = _index(read_jsonl(args.canonical_contexts), "canonical context")
    canonical_graphs = _index(read_jsonl(args.canonical_passage_graphs), "canonical graph")
    if set(canonical_contexts) != set(canonical_graphs):
        raise ValueError("canonical context/graph identity join is not 1.0")
    reporting = [
        _cohort_row(canonical_contexts[key], canonical_graphs[key], "reporting_only_nonconfirmatory")
        for key in canonical_contexts
    ]
    proof = _index(read_jsonl(args.canonical_2wiki_proof), "canonical ProofKG")
    expected_2wiki = {row["question_key"] for row in reporting if row["dataset"] == "2wikimultihopqa"}
    if set(proof) != expected_2wiki:
        raise ValueError("canonical 2Wiki ProofKG/reporting join is not 1.0")
    if any((row.get("provenance") or {}).get("gold_access") is not False for row in proof.values()):
        raise ValueError("canonical ProofKG includes a record without gold_access=false")

    for role in role_rows:
        role_rows[role].sort(key=lambda row: (row["dataset"], row["question_key"]))
    reporting.sort(key=lambda row: (row["dataset"], row["question_key"]))
    dev_families = {row["family_sha256"] for row in role_rows["development"]}
    confirmation_families = {row["family_sha256"] for row in role_rows["confirmation"]}
    if dev_families & confirmation_families:
        raise ValueError("development/confirmation family overlap")

    planner_rows = []
    for role in ("development", "confirmation"):
        for row in role_rows[role]:
            if row["dataset"] != "2wikimultihopqa":
                continue
            planner_rows.append({
                "schema_version": "query-planner-supervision-1",
                "row_id": f"saeg-v1::{role}::{row['question_key']}",
                "question_key": row["question_key"],
                "dataset": row["dataset"],
                "qid": row["qid"],
                "question": row["question"],
                "question_sha256": row["question_sha256"],
                "target_type": "relation_graph",
                "role": role,
                "gold_access": False,
            })
    if len(planner_rows) != 150:
        raise ValueError(f"expected 150 2Wiki planner rows, got {len(planner_rows)}")
    for row in planner_rows:
        assert_answer_free(row)

    args.out.mkdir(parents=True, exist_ok=False)
    outputs = {
        "development": args.out / "development.question_only.jsonl",
        "confirmation": args.out / "confirmation.question_only.jsonl",
        "reporting": args.out / "canonical_reporting.question_only.jsonl",
        "planner_2wiki": args.out / "2wiki_dev_confirmation_planner.question_only.jsonl",
    }
    write_jsonl(outputs["development"], role_rows["development"])
    write_jsonl(outputs["confirmation"], role_rows["confirmation"])
    write_jsonl(outputs["reporting"], reporting)
    write_jsonl(outputs["planner_2wiki"], planner_rows)

    counts = Counter((row["role"], row["dataset"]) for rows in [*role_rows.values(), reporting] for row in rows)
    protocol = {
        "schema_version": "saeg-evaluation-protocol-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "roles": {
            "development": "150 rows (50/dataset); method/checkpoint decisions allowed; already consumed only by failed QPEG-v4, never confirmation",
            "confirmation": "300 rows (100/dataset); unopened; may be evaluated once only after SAEG development gates pass",
            "canonical_reporting": "900 rows (300/dataset); historical/consumed; baseline-comparable reporting only; never tune or select",
        },
        "citation_contract": {
            "wikidata": "standard (head, relation, tail) triples in Knowledge Used",
            "passage": "[P<n>] evidence sentences in Passage Used; never KG triples",
        },
        "resource_policy": {
            "same_resource_main": "Passage evidence constructed only from each row's frozen Top-10 retrieval snapshot",
            "extra_resource": "Wikidata ProofKG reported as a separate arm; unavailable/failed branches fail closed",
            "hotpotqa_wikidata": "NOT_ELIGIBLE after frozen structural failure",
            "musique_wikidata": "NOT_ELIGIBLE after frozen structural failure",
            "2wiki_reporting_wikidata": "materialized 300/300 Gold-free ProofKG records",
            "2wiki_fresh_wikidata": "PENDING from this frozen answer-free 150-row planner cohort",
        },
        "counts": {f"{role}::{dataset}": n for (role, dataset), n in sorted(counts.items())},
        "integrity": {
            "fresh_identity_join_rate": 1.0,
            "canonical_identity_join_rate": 1.0,
            "development_confirmation_family_overlap": 0,
            "gold_in_protocol_or_cohorts": False,
            "confirmation_opened": False,
            "qpeg_v4_failure_preserved": True,
        },
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in inputs.items()},
        "outputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in outputs.items()},
        "scientific_boundary": (
            "This freezes answer-free cohorts only. It does not open confirmation, change a baseline, "
            "approve training, or claim SAEG utility. Gold export and final eval inputs are later stages."
        ),
    }
    (args.out / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(args.out, extra=protocol, status=STATUS)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
