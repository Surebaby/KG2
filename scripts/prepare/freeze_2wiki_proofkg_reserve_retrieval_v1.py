#!/usr/bin/env python3
"""Freeze canonical-retrieval requests for the 2Wiki ProofKG reserve50.

This CPU-only preregistration projects the already-frozen reserve cohort onto
dataset/qid/question identities.  It neither reads outcomes/support labels nor
runs retrieval.  The resulting protocol is the sole authorised input for the
append-only retrieval materializer.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_2wiki_proofkg_extension_v1 import (
    file_ref,
    read_jsonl,
    write_jsonl,
)
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.prepare.materialize_qpeg_v1_retrieval import FORBIDDEN_FIELDS


DATASET = "2wikimultihopqa"
EXPECTED_SOURCE_SCHEMA = "2wiki-proofkg-extension-reserve-protocol-v1"
EXPECTED_SOURCE_STATUS = "FROZEN_TRAIN_ONLY_BEFORE_PLANNER_NOT_MATERIALIZED"
SCHEMA_VERSION = "2wiki-proofkg-reserve-retrieval-preregistration-v1"
REQUEST_SCHEMA = "2wiki-proofkg-reserve-retrieval-request-v1"
STATUS = "FROZEN_ANSWER_FREE_2WIKI_RESERVE_RETRIEVAL_NOT_MATERIALIZED"
EXPERIMENT_ID = "2WIKI-PROOFKG-EXTENSION-RESERVE-V1-N50-RETRIEVAL-PREREG"
RETRIEVAL_STACK = (
    "E5@100+BM25@100->RRF60@50->bge-reranker-v2-m3@10->pack3860"
)
DEFAULT_SOURCE = Path(
    "outputs/audits/2wiki_proofkg_extension_reserve_v1_n50_seed42_"
    "preregistration/protocol.json"
)
DEFAULT_OUT = Path(
    "outputs/audits/2wiki_proofkg_extension_reserve_v1_n50_retrieval_"
    "preregistration"
)
EXPECTED_QUOTAS = {"inference": 30, "compositional": 12, "comparison": 8}


def build_requests(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != 50:
        raise ValueError(f"expected reserve50 cohort, got {len(rows)}")
    output: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    seen_hashes: set[str] = set()
    for index, value in enumerate(rows, start=1):
        row = dict(value)
        dataset = str(row.get("dataset") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        question = str(row.get("question") or "").strip()
        key = question_key(dataset, qid)
        qhash = question_sha256(question)
        family = family_sha256(question)
        if (
            dataset != DATASET
            or not qid
            or not question
            or row.get("question_key") != key
            or row.get("question_sha256") != qhash
            or row.get("family_version") != FAMILY_VERSION
            or row.get("family_sha256") != family
            or row.get("gold_access") is not False
            or row.get("evaluation_eligible") is not False
            or row.get("source_role")
            != "automatic_proofkg_extension_reserve_candidate"
        ):
            raise ValueError(f"reserve cohort identity/boundary invalid at row {index}")
        present = FORBIDDEN_FIELDS & set(row)
        if present:
            raise ValueError(f"{key}: forbidden fields in source: {sorted(present)}")
        if key in seen_keys or qhash in seen_hashes:
            raise ValueError(f"duplicate reserve identity/hash: {key}")
        seen_keys.add(key)
        seen_hashes.add(qhash)
        output.append(
            {
                "schema_version": REQUEST_SCHEMA,
                "question_key": key,
                "dataset": DATASET,
                "qid": qid,
                "question": question,
                "question_sha256": qhash,
                "family_version": FAMILY_VERSION,
                "family_sha256": family,
                "question_type": str(row.get("question_type") or ""),
                "role": "rollout_retrieval",
                "gold_access": False,
                "evaluation_eligible": False,
            }
        )
    if Counter(row["question_type"] for row in output) != Counter(EXPECTED_QUOTAS):
        raise ValueError("reserve retrieval question-type quotas drifted")
    return output


def freeze(
    *, source_protocol_path: Path, output_dir: Path, experiment_id: str
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite retrieval prereg: {output_dir}")
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    if (
        source_protocol.get("schema_version") != EXPECTED_SOURCE_SCHEMA
        or source_protocol.get("status") != EXPECTED_SOURCE_STATUS
    ):
        raise ValueError("reserve source protocol schema/status invalid")
    cohort_ref = source_protocol.get("output_cohort")
    if not isinstance(cohort_ref, Mapping):
        raise ValueError("reserve source protocol does not bind output_cohort")
    cohort_path = Path(str(cohort_ref.get("path") or "")).resolve()
    if not cohort_path.is_file() or file_ref(cohort_path) != dict(cohort_ref):
        raise ValueError("reserve source cohort hash/size binding invalid")
    requests = build_requests(read_jsonl(cohort_path))
    output_dir.mkdir(parents=True, exist_ok=False)
    requests_path = output_dir / "retrieval_requests.question_only.jsonl"
    write_jsonl(requests_path, requests)
    generated_at = datetime.now(timezone.utc).isoformat()
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(experiment_id).strip(),
        "generated_at_utc": generated_at,
        "status": STATUS,
        "population": {
            "n": 50,
            "by_dataset": {DATASET: 50},
            "by_question_type": dict(EXPECTED_QUOTAS),
        },
        "retrieval": {
            "stack": RETRIEVAL_STACK,
            "top_k": 10,
            "token_budget": 3860,
            "reranker": "models/bge-reranker-v2-m3",
            "backend_fallback_allowed": False,
        },
        "inputs": {
            "reserve_protocol": file_ref(source_protocol_path),
            "reserve_cohort": file_ref(cohort_path),
        },
        "outputs": {"retrieval_requests": file_ref(requests_path)},
        "scientific_boundary": {
            "question_only": True,
            "gold_access": False,
            "outcome_or_support_labels_read": False,
            "retrieval_started": False,
            "training_started": False,
        },
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    gates = {
        "n_exact_50": len(requests) == 50,
        "question_type_quotas_exact": Counter(
            row["question_type"] for row in requests
        )
        == Counter(EXPECTED_QUOTAS),
        "identity_unique": len({row["question_key"] for row in requests}) == 50,
        "question_hash_unique": len(
            {row["question_sha256"] for row in requests}
        )
        == 50,
        "gold_access_false": all(row["gold_access"] is False for row in requests),
        "forbidden_fields_zero": all(
            not (FORBIDDEN_FIELDS & set(row)) for row in requests
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"reserve retrieval prereg gates failed: {gates}")
    report = {
        "schema_version": f"{SCHEMA_VERSION}-report",
        "experiment_id": protocol["experiment_id"],
        "generated_at_utc": generated_at,
        "status": STATUS,
        "counts": protocol["population"],
        "gates": gates,
        "inputs": protocol["inputs"],
        "outputs": {
            "retrieval_requests": file_ref(requests_path),
            "protocol": file_ref(protocol_path),
        },
        "training_started": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        output_dir,
        status=STATUS,
        extra={
            "phase": "freeze_2wiki_proofkg_reserve_retrieval_v1",
            "experiment_id": protocol["experiment_id"],
            "protocol": file_ref(protocol_path),
            "report": file_ref(report_path),
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-protocol", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    args = parser.parse_args()
    report = freeze(
        source_protocol_path=args.source_protocol,
        output_dir=args.out,
        experiment_id=args.experiment_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
