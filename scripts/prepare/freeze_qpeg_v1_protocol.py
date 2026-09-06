#!/usr/bin/env python
"""Freeze QPEG-v1 pilot/confirmation/final cohorts before graph construction.

The historical canonical n=300 per dataset remains the untouched final cohort.
Pilot50 and confirmation100 are selected from the remaining dev rows after
excluding every final qid and every answer-free question family represented in
final.  Only qid/question/hash fields are emitted for new retrieval; gold and
dataset annotations never enter the frozen artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.utils.logging import dump_manifest


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
SCHEMA_VERSION = "qpeg-v1-preregistration-1"
FAMILY_VERSION = "answer-free-lexical-family-v1"
EXPERIMENT_ID = "QPEG-V1-N1350-SEED42-PREREGISTRATION"
PILOT_PER_DATASET = 50
CONFIRMATION_PER_DATASET = 100
FINAL_PER_DATASET = 300
FORBIDDEN_FIELDS = {
    "golden_answers", "answer", "answers", "supporting_facts", "support",
    "decomposition", "question_decomposition", "evidence", "reasoning", "sp",
}

_FINAL_CONTEXTS = Path(
    "outputs/audits/inference_proofkg_v1_n900_seed42_preregistration/retrieval_contexts.jsonl"
)
_DEFAULT_OUT = Path("outputs/audits/qpeg_v1_n1350_seed42_preregistration")
_ENTITY_SPAN = re.compile(r"\b(?:[A-Z][\w'’-]*)(?:\s+(?:[A-Z][\w'’-]*|of|the|and|&)){0,5}\b")
_QUOTED = re.compile(r"([\"“][^\"”]+[\"”]|'[^']{2,}')")
_NUMBER = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
_SPACE = re.compile(r"\s+")
_QUESTION_OPENERS = {
    "who", "what", "when", "where", "which", "why", "how", "were", "was",
    "are", "is", "did", "do", "does", "name",
}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_json(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def question_family_signature(question: str) -> str:
    """Return a deterministic answer-free lexical/template family signature."""
    text = unicodedata.normalize("NFKC", str(question or "")).strip()
    text = _QUOTED.sub(" <entity> ", text)
    text = _NUMBER.sub(" <num> ", text)

    def replace_entity(match: re.Match[str]) -> str:
        value = match.group(0)
        if value.casefold() in _QUESTION_OPENERS:
            return value
        return " <entity> "

    text = _ENTITY_SPAN.sub(replace_entity, text)
    text = text.casefold()
    text = re.sub(r"[^a-z0-9<>]+", " ", text)
    text = _SPACE.sub(" ", text).strip()
    return text


def family_sha256(question: str) -> str:
    return hashlib.sha256(question_family_signature(question).encode("utf-8")).hexdigest()


def _question_only(row: Mapping[str, Any], role: str) -> dict[str, Any]:
    return {
        "schema_version": "qpeg-question-only-v1",
        "question_key": question_key(str(row["dataset"]), str(row["qid"])),
        "dataset": str(row["dataset"]),
        "qid": str(row["qid"]),
        "question": str(row["question"]),
        "question_sha256": str(row["question_sha256"]),
        "family_sha256": str(row["family_sha256"]),
        "role": role,
        "gold_access": False,
    }


def choose_disjoint_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded_qids: set[str],
    excluded_families: set[str],
    n: int,
    dataset: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Choose one row per answer-free family, deterministically."""
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        qid = str(raw.get("id") or raw.get("qid") or "")
        question = str(raw.get("question") or "").strip()
        if not qid or not question or qid in excluded_qids:
            continue
        family = family_sha256(question)
        if family in excluded_families:
            continue
        by_family[family].append({
            "dataset": dataset,
            "qid": qid,
            "question": question,
            "question_sha256": question_sha256(question),
            "family_sha256": family,
        })
    ordered_families = sorted(
        by_family,
        key=lambda family: hashlib.sha256(f"{seed}\0{dataset}\0{family}".encode()).hexdigest(),
    )
    if len(ordered_families) < n:
        raise ValueError(f"{dataset}: only {len(ordered_families)} eligible families; need {n}")
    selected: list[dict[str, Any]] = []
    for family in ordered_families[:n]:
        candidates = sorted(by_family[family], key=lambda row: (row["question_sha256"], row["qid"]))
        selected.append(candidates[0])
    return selected


def validate_partition(
    pilot: Sequence[Mapping[str, Any]],
    confirmation: Sequence[Mapping[str, Any]],
    final: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    role_rows = {"pilot": pilot, "confirmation": confirmation, "final": final}
    qids = {role: {str(row["qid"]) for row in rows} for role, rows in role_rows.items()}
    families = {
        role: {str(row["family_sha256"]) for row in rows}
        for role, rows in role_rows.items()
    }
    return {
        "qid_counts": {role: len(values) for role, values in qids.items()},
        "family_counts": {role: len(values) for role, values in families.items()},
        "qid_overlap": {
            "pilot_confirmation": len(qids["pilot"] & qids["confirmation"]),
            "pilot_final": len(qids["pilot"] & qids["final"]),
            "confirmation_final": len(qids["confirmation"] & qids["final"]),
        },
        "family_overlap": {
            "pilot_confirmation": len(families["pilot"] & families["confirmation"]),
            "pilot_final": len(families["pilot"] & families["final"]),
            "confirmation_final": len(families["confirmation"] & families["final"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--final_contexts", type=Path, default=_FINAL_CONTEXTS)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite frozen output: {args.out}")
    if not args.final_contexts.is_file():
        raise FileNotFoundError(args.final_contexts)
    args.out.mkdir(parents=True)

    final_context_rows = _read_jsonl(args.final_contexts)
    final_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for context in final_context_rows:
        final_by_dataset[str(context["dataset"])].append(context)

    all_pilot: list[dict[str, Any]] = []
    all_confirmation: list[dict[str, Any]] = []
    all_final: list[dict[str, Any]] = []
    final_context_out: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    raw_hashes: dict[str, str] = {}

    for dataset in DATASETS:
        raw_path = args.data_root / dataset / "dev.jsonl"
        raw_rows = _read_jsonl(raw_path)
        raw_hashes[dataset] = _sha_file(raw_path)
        raw_by_qid = {str(row.get("id") or row.get("qid")): row for row in raw_rows}
        contexts = final_by_dataset[dataset]
        if len(contexts) != FINAL_PER_DATASET:
            raise ValueError(f"{dataset}: expected {FINAL_PER_DATASET} final contexts, got {len(contexts)}")

        final_rows: list[dict[str, Any]] = []
        for context in contexts:
            qid = str(context["qid"])
            raw = raw_by_qid.get(qid)
            if raw is None:
                raise ValueError(f"{dataset}/{qid}: final qid missing from raw dev")
            question = str(raw.get("question") or "").strip()
            if question_sha256(question) != str(context["question_sha256"]):
                raise ValueError(f"{dataset}/{qid}: final question hash mismatch")
            family = family_sha256(question)
            final_row = {
                "dataset": dataset,
                "qid": qid,
                "question": question,
                "question_sha256": question_sha256(question),
                "family_sha256": family,
            }
            final_rows.append(final_row)
            final_context_out.append({
                **_question_only(final_row, "final"),
                "passages": context["passages"],
                "passages_sha256": context["passages_sha256"],
                "retrieval_source": "historical_canonical_n300_frozen",
            })

        final_qids = {row["qid"] for row in final_rows}
        final_families = {row["family_sha256"] for row in final_rows}
        pilot_rows = choose_disjoint_rows(
            raw_rows,
            excluded_qids=final_qids,
            excluded_families=final_families,
            n=PILOT_PER_DATASET,
            dataset=dataset,
            seed=args.seed,
        )
        pilot_qids = final_qids | {row["qid"] for row in pilot_rows}
        pilot_families = final_families | {row["family_sha256"] for row in pilot_rows}
        confirmation_rows = choose_disjoint_rows(
            raw_rows,
            excluded_qids=pilot_qids,
            excluded_families=pilot_families,
            n=CONFIRMATION_PER_DATASET,
            dataset=dataset,
            seed=args.seed + 1,
        )
        partition = validate_partition(pilot_rows, confirmation_rows, final_rows)
        expected_counts = {
            "pilot": PILOT_PER_DATASET,
            "confirmation": CONFIRMATION_PER_DATASET,
            "final": FINAL_PER_DATASET,
        }
        if partition["qid_counts"] != expected_counts:
            raise ValueError(f"{dataset}: qid counts failed: {partition['qid_counts']}")
        if any(partition["qid_overlap"].values()) or any(partition["family_overlap"].values()):
            raise ValueError(f"{dataset}: split overlap: {partition}")
        reports[dataset] = {
            "raw_dev_rows": len(raw_rows),
            "partition": partition,
            "final_passage_hashes_valid": True,
        }
        all_pilot.extend(_question_only(row, "pilot") for row in pilot_rows)
        all_confirmation.extend(_question_only(row, "confirmation") for row in confirmation_rows)
        all_final.extend(_question_only(row, "final") for row in final_rows)

    retrieval_requests = sorted(
        all_pilot + all_confirmation,
        key=lambda row: (row["dataset"], row["role"], row["question_sha256"]),
    )
    _write_jsonl(args.out / "pilot.question_only.jsonl", all_pilot)
    _write_jsonl(args.out / "confirmation.question_only.jsonl", all_confirmation)
    _write_jsonl(args.out / "final.question_only.jsonl", all_final)
    _write_jsonl(args.out / "retrieval_requests.jsonl", retrieval_requests)
    _write_jsonl(args.out / "final.retrieval_contexts.jsonl", final_context_out)

    forbidden_present = Counter()
    for row in all_pilot + all_confirmation + all_final:
        for field in FORBIDDEN_FIELDS & set(row):
            forbidden_present[field] += 1
    gates = {
        "qid_overlap_zero": all(
            value == 0 for report in reports.values()
            for value in report["partition"]["qid_overlap"].values()
        ),
        "family_overlap_zero": all(
            value == 0 for report in reports.values()
            for value in report["partition"]["family_overlap"].values()
        ),
        "forbidden_fields_zero": not forbidden_present,
        "final_passages_frozen_900": len(final_context_out) == 900,
        "retrieval_requests_450": len(retrieval_requests) == 450,
        "gold_access_false": all(row["gold_access"] is False for row in all_pilot + all_confirmation + all_final),
    }
    if not all(gates.values()):
        raise SystemExit(f"QPEG preregistration gates failed: {gates}")

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_QPEG_BUILD_OR_NEW_RETRIEVAL",
        "scope": "same-resource QPEG-v1; three datasets; disjoint pilot50/confirmation100/final300",
        "roles": {
            "pilot": "50/dataset; one generic extractor correction allowed",
            "confirmation": "100/dataset; unseen after pilot; no protocol changes",
            "final": "historical canonical 300/dataset; untouched by pilot/confirmation selection",
        },
        "selection": {
            "seed": args.seed,
            "family_version": FAMILY_VERSION,
            "one_question_per_family": True,
            "final_excluded_from_development_by_qid_and_family": True,
        },
        "retrieval": {
            "pilot_confirmation": "E5@100 + BM25@100 -> RRF(k=60)@50 -> bge-reranker-v2-m3@10",
            "final": "reuse frozen historical canonical passages; no re-retrieval",
            "underlying_corpus": "Wiki18 corpus_flashrag",
        },
        "qpeg": {
            "extractor_version": "qpeg-deterministic-passage-v1",
            "knowledge_inputs": ["question", "retrieved_top10_passages"],
            "forbidden_inputs": sorted(FORBIDDEN_FIELDS | {"wikidata", "legacy_kg"}),
            "max_edges": 12,
            "fallback": "explicit no-graph if no passage-backed edge",
        },
        "external_baseline_comparison": {
            "main_table": "native end-to-end methods on same final qids/Wiki18/n300/seed42/greedy/canonical scorer",
            "qpeg_is_method_component": True,
            "reuse_old_prediction_only_if_qids_and_scorer_match": True,
            "strict_attribution": "internal A-F matched arms with frozen passages",
            "extra_wikidata_proofkg": "separate resource-augmented table",
        },
        "day2_go_gates": {
            "qpeg_nonempty_per_dataset": ">=0.80",
            "identity_hash_join": "==1.0",
            "passage_provenance": "==1.0",
            "gold_access": "false",
            "max_edges": "<=12",
        },
        "inputs": {
            "raw_dev_sha256": raw_hashes,
            "final_contexts": str(args.final_contexts),
            "final_contexts_sha256": _sha_file(args.final_contexts),
        },
        "reports": reports,
        "gates": gates,
    }
    (args.out / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": "qpeg-v1-preregistration-report-1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS",
        "counts": {"pilot": 150, "confirmation": 300, "final": 900, "retrieval_requests": 450},
        "datasets": reports,
        "gates": gates,
        "forbidden_fields_present": dict(forbidden_present),
        "cohort_hashes": {
            "pilot": _sha_file(args.out / "pilot.question_only.jsonl"),
            "confirmation": _sha_file(args.out / "confirmation.question_only.jsonl"),
            "final": _sha_file(args.out / "final.question_only.jsonl"),
            "retrieval_requests": _sha_file(args.out / "retrieval_requests.jsonl"),
            "final_contexts": _sha_file(args.out / "final.retrieval_contexts.jsonl"),
        },
    }
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        args.out,
        extra={"experiment_id": EXPERIMENT_ID, "phase": "qpeg_preregistration", **report},
        status="COMPLETE",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
