#!/usr/bin/env python3
"""Materialize canonical passages for the strict official-raw 2Wiki scope.

This is deliberately a separate release from the historical reserve50
retrieval.  It consumes only the append-only, answer-free strict-eligibility
scope and runs the canonical retrieval stack.  It neither selects Proof800
nor constructs/changes a ProofKG record.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_2wiki_official_raw_canonical_retrieval_v1 import (
    DATASET,
    FAMILY_VERSION,
    MIN_PER_TYPE,
    QTYPES,
    REQUEST_SCHEMA,
    RETRIEVAL_STACK,
    SCOPE_SCHEMA as EXPECTED_SCOPE_SCHEMA,
    SCOPE_STATUS as EXPECTED_SCOPE_STATUS,
    family_sha256,
)
from scripts.prepare.materialize_mixed3_v4_expansion_retrieval import (
    _attest_cross_encoder_backend,
)
from scripts.prepare.materialize_qpeg_v1_retrieval import (
    FORBIDDEN_FIELDS,
    _sha_json,
    materialize_dataset,
)


SCHEMA_VERSION = "2wiki-official-raw-canonical-retrieval-context-v1"
REPORT_SCHEMA_VERSION = "2wiki-official-raw-canonical-retrieval-report-v1"
STATUS = "COMPLETE_ANSWER_FREE_2WIKI_OFFICIAL_RAW_CANONICAL_RETRIEVAL_NOT_TRAINED"
EXPERIMENT_ID = "2WIKI-OFFICIAL-RAW-N1500-STRICT-SCOPE-CANONICAL-RETRIEVAL-V1"
DEFAULT_PROTOCOL = Path(
    "outputs/audits/2wiki_official_raw_canonical_retrieval_v1_scope_"
    "preregistration/protocol.json"
)
DEFAULT_OUT = Path(
    "outputs/audits/2wiki_official_raw_canonical_retrieval_v1"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _resolve_ref(value: Mapping[str, Any], *, label: str) -> Path:
    raw = str(value.get("path") or "").strip()
    if not raw:
        raise ValueError(f"{label}: empty path")
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file() or _sha256(path) != str(value.get("sha256") or ""):
        raise ValueError(f"{label}: missing file or SHA256 drift")
    if value.get("size_bytes") is not None and path.stat().st_size != int(
        value["size_bytes"]
    ):
        raise ValueError(f"{label}: size drift")
    return path


def _validate_requests(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    seen_keys: set[str] = set()
    seen_hashes: set[str] = set()
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        dataset = str(row.get("dataset") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        question = str(row.get("question") or "").strip()
        key = question_key(dataset, qid)
        qtype = str(row.get("question_type") or "")
        if (
            row.get("schema_version") != REQUEST_SCHEMA
            or dataset != DATASET
            or not qid
            or not question
            or row.get("question_key") != key
            or row.get("question_sha256") != question_sha256(question)
            or row.get("family_version") != FAMILY_VERSION
            or row.get("family_sha256") != family_sha256(question)
            or qtype not in QTYPES
            or row.get("role") != "official_raw_proofkg_rollout_retrieval"
            or row.get("gold_access") is not False
            or row.get("evaluation_eligible") is not False
            or bool(FORBIDDEN_FIELDS & set(row))
        ):
            raise ValueError(f"official-raw retrieval request invalid at row {index}")
        for field in (
            "closure_runtime_record_sha256",
            "closure_kg_sha256",
            "closure_execution_sha256",
        ):
            value = str(row.get(field) or "")
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"request lacks closure hash {field}: {key}")
        qhash = str(row["question_sha256"])
        if key in seen_keys or qhash in seen_hashes:
            raise ValueError(f"duplicate request identity/hash: {key}")
        seen_keys.add(key)
        seen_hashes.add(qhash)
        counts[qtype] += 1
    if not rows or any(counts[qtype] < MIN_PER_TYPE for qtype in QTYPES):
        raise ValueError(
            f"official-raw scope lacks >= {MIN_PER_TYPE} rows/type: {dict(counts)}"
        )
    return counts


def _valid_passages(passages: Any) -> bool:
    return isinstance(passages, list) and len(passages) == 10 and all(
        isinstance(row, Mapping)
        and all(str(row.get(field) or "").strip() for field in ("id", "source", "contents"))
        and not (FORBIDDEN_FIELDS & set(row))
        for row in passages
    )


def _validate_contexts(
    requests: Sequence[Mapping[str, Any]], contexts: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    expected = {str(row["question_key"]): row for row in requests}
    accepted: dict[str, dict[str, Any]] = {}
    for value in contexts:
        row = dict(value)
        key = question_key(row.get("dataset"), row.get("qid"))
        request = expected.get(key)
        if request is None or key in accepted:
            raise ValueError(f"retrieval output outside/duplicate scope: {key}")
        for field in (
            "question_key",
            "dataset",
            "qid",
            "question",
            "question_sha256",
            "family_sha256",
            "role",
            "gold_access",
        ):
            if row.get(field) != request.get(field):
                raise ValueError(f"retrieval/request mismatch at {field}: {key}")
        passages = row.get("passages")
        if (
            bool(FORBIDDEN_FIELDS & set(row))
            or not _valid_passages(passages)
            or row.get("passages_sha256") != _sha_json(passages)
            or row.get("retrieval_source") != RETRIEVAL_STACK
        ):
            raise ValueError(f"canonical passage/backend contract failed: {key}")
        # Carry the closure identities through to the release.  These are hashes,
        # not answer-bearing execution traces.
        for field in (
            "question_type",
            "family_version",
            "evaluation_eligible",
            "closure_runtime_record_sha256",
            "closure_kg_sha256",
            "closure_execution_sha256",
        ):
            row[field] = request[field]
        row["schema_version"] = SCHEMA_VERSION
        accepted[key] = row
    if set(accepted) != set(expected):
        raise ValueError(
            f"retrieval output misses {len(set(expected) - set(accepted))} scoped rows"
        )
    return [accepted[str(row["question_key"])] for row in requests]


def materialize(
    *,
    protocol_path: Path,
    output_dir: Path,
    batch_size: int,
    experiment_id: str,
    retrieval_fn: Callable[[str, list[dict[str, Any]], int], list[dict[str, Any]]]
    | None = None,
    backend_attestation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite retrieval release: {output_dir}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    protocol = _read_json(protocol_path)
    if (
        protocol.get("schema_version") != EXPECTED_SCOPE_SCHEMA
        or protocol.get("status") != EXPECTED_SCOPE_STATUS
        or not all(bool(v) for v in (protocol.get("gates") or {}).values())
        or (protocol.get("retrieval") or {}).get("stack") != RETRIEVAL_STACK
        or (protocol.get("retrieval") or {}).get("backend_fallback_allowed") is not False
        or (protocol.get("scientific_boundary") or {}).get("retrieval_started") is not False
        or (protocol.get("scientific_boundary") or {}).get("training_started") is not False
    ):
        raise ValueError("official-raw retrieval scope schema/status/gates invalid")
    requests_ref = (protocol.get("outputs") or {}).get("retrieval_requests")
    if not isinstance(requests_ref, Mapping):
        raise ValueError("scope does not bind retrieval_requests")
    requests_path = _resolve_ref(requests_ref, label="retrieval requests")
    requests = _read_jsonl(requests_path)
    type_counts = _validate_requests(requests)
    declared_counts = (protocol.get("population") or {}).get(
        "strict_scope_by_question_type"
    ) or {}
    if any(int(declared_counts.get(qtype, -1)) != type_counts[qtype] for qtype in QTYPES):
        raise ValueError("scope protocol/request question-type counts drifted")
    if int((protocol.get("population") or {}).get("strict_scope_total", -1)) != len(
        requests
    ):
        raise ValueError("scope protocol/request total drifted")

    if retrieval_fn is None:
        attestation = _attest_cross_encoder_backend()
        runner = materialize_dataset
    else:
        runner = retrieval_fn
        attestation = dict(
            backend_attestation
            or {
                "mode": "injected_test_double",
                "load_succeeded": True,
                "backend_fallback": False,
            }
        )
    if attestation.get("load_succeeded") is not True or attestation.get(
        "backend_fallback"
    ) is not False:
        raise RuntimeError(f"canonical BGE attestation failed: {attestation}")

    contexts = _validate_contexts(
        requests, runner(DATASET, list(requests), batch_size)
    )
    gates = {
        "scope_and_request_hash_bound": True,
        "n_exact_scope": len(contexts) == len(requests),
        "each_question_type_ge_200": all(
            type_counts[qtype] >= MIN_PER_TYPE for qtype in QTYPES
        ),
        "identity_join_rate_1": {row["question_key"] for row in contexts}
        == {row["question_key"] for row in requests},
        "request_order_preserved": [row["question_key"] for row in contexts]
        == [row["question_key"] for row in requests],
        "closure_hashes_propagated_exact": all(
            all(context[field] == request[field] for field in (
                "closure_runtime_record_sha256",
                "closure_kg_sha256",
                "closure_execution_sha256",
            ))
            for request, context in zip(requests, contexts)
        ),
        "gold_access_false": all(row.get("gold_access") is False for row in contexts),
        "forbidden_fields_zero": all(
            not (FORBIDDEN_FIELDS & set(row))
            and all(not (FORBIDDEN_FIELDS & set(p)) for p in row["passages"])
            for row in contexts
        ),
        "all_exactly_ten_safe_passages": all(
            _valid_passages(row.get("passages")) for row in contexts
        ),
        "canonical_retrieval_stack_exact": all(
            row.get("retrieval_source") == RETRIEVAL_STACK for row in contexts
        ),
        "backend_load_attested": attestation.get("load_succeeded") is True,
        "backend_fallback_false": attestation.get("backend_fallback") is False,
        "reserve50_schema_or_status_not_reused": True,
        "selection_not_performed": True,
        "training_not_started": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"official-raw retrieval gates failed: {gates}")

    output_dir.mkdir(parents=True, exist_ok=False)
    contexts_path = output_dir / "retrieval_contexts.jsonl"
    _write_jsonl(contexts_path, contexts)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment_id": str(experiment_id).strip(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "counts": {
            "requests": len(requests),
            "contexts": len(contexts),
            "by_dataset": {DATASET: len(contexts)},
            "by_question_type": {qtype: type_counts[qtype] for qtype in QTYPES},
        },
        "retrieval": RETRIEVAL_STACK,
        "backend_attestation": attestation,
        "gates": gates,
        "inputs": {
            "scope_protocol": _identity(protocol_path),
            "retrieval_requests": _identity(requests_path),
        },
        "outputs": {"retrieval_contexts": _identity(contexts_path)},
        "scientific_boundary": {
            "scope_selection_performed": False,
            "proof800_selection_performed": False,
            "gold_fields_read_or_written": False,
            "kg_constructed_or_changed": False,
            "model_updated": False,
            "training_started": False,
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
            "phase": "materialize_2wiki_official_raw_canonical_retrieval_v1",
            "experiment_id": report["experiment_id"],
            "report": _identity(report_path),
            "retrieval_contexts": report["outputs"]["retrieval_contexts"],
            "backend_attestation": attestation,
            "gold_access": False,
            "proof800_selected": False,
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    args = parser.parse_args()
    report = materialize(
        protocol_path=args.protocol,
        output_dir=args.out,
        batch_size=args.batch_size,
        experiment_id=args.experiment_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
