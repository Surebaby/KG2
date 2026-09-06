#!/usr/bin/env python3
"""Materialize canonical ten-passage retrieval for frozen 2Wiki reserve50.

The release is append-only and fail-closed.  Formal execution preloads and
attests the exact local BGE reranker, forbids its historical BM25 fallback,
and checks an exact identity join to the answer-free preregistration.
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
from scripts.prepare.freeze_2wiki_proofkg_extension_v1 import file_ref
from scripts.prepare.freeze_2wiki_proofkg_reserve_retrieval_v1 import (
    DATASET,
    RETRIEVAL_STACK,
    SCHEMA_VERSION as EXPECTED_PROTOCOL_SCHEMA,
    STATUS as EXPECTED_PROTOCOL_STATUS,
)
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.prepare.materialize_mixed3_v4_expansion_retrieval import (
    _attest_cross_encoder_backend,
)
from scripts.prepare.materialize_qpeg_v1_retrieval import (
    FORBIDDEN_FIELDS,
    _sha_json,
    materialize_dataset,
)


SCHEMA_VERSION = "2wiki-proofkg-reserve-retrieval-context-v1"
REPORT_SCHEMA_VERSION = "2wiki-proofkg-reserve-retrieval-report-v1"
STATUS = "COMPLETE_ANSWER_FREE_2WIKI_RESERVE_RETRIEVAL_NOT_TRAINED"
EXPERIMENT_ID = "2WIKI-PROOFKG-EXTENSION-RESERVE-V1-N50-RETRIEVAL"
DEFAULT_PROTOCOL = Path(
    "outputs/audits/2wiki_proofkg_extension_reserve_v1_n50_retrieval_"
    "preregistration/protocol.json"
)
DEFAULT_OUT = Path(
    "outputs/audits/2wiki_proofkg_extension_reserve_v1_n50_retrieval_v1"
)


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
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_ref(identity: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(identity.get("path") or "")).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if _sha256(path) != str(identity.get("sha256") or ""):
        raise ValueError(f"{label} SHA256 mismatch: {path}")
    if identity.get("size_bytes") is not None and path.stat().st_size != int(
        identity["size_bytes"]
    ):
        raise ValueError(f"{label} size mismatch: {path}")
    return path


def _valid_passages(passages: Any) -> bool:
    return isinstance(passages, list) and len(passages) == 10 and all(
        isinstance(row, Mapping)
        and all(str(row.get(field) or "").strip() for field in ("id", "source", "contents"))
        and not (FORBIDDEN_FIELDS & set(row))
        for row in passages
    )


def _validate_requests(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 50:
        raise ValueError(f"expected 50 frozen requests, got {len(rows)}")
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        dataset = str(row.get("dataset") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        question = str(row.get("question") or "").strip()
        key = question_key(dataset, qid)
        if (
            dataset != DATASET
            or row.get("question_key") != key
            or row.get("question_sha256") != question_sha256(question)
            or row.get("family_version") != FAMILY_VERSION
            or row.get("family_sha256") != family_sha256(question)
            or row.get("role") != "rollout_retrieval"
            or row.get("gold_access") is not False
            or FORBIDDEN_FIELDS & set(row)
        ):
            raise ValueError(f"retrieval request identity/boundary invalid at row {index}")
        if key in seen:
            raise ValueError(f"duplicate retrieval request: {key}")
        seen.add(key)


def _validate_contexts(
    requests: Sequence[Mapping[str, Any]], contexts: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    expected = {str(row["question_key"]): row for row in requests}
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in contexts:
        row = dict(value)
        key = question_key(row.get("dataset"), row.get("qid"))
        request = expected.get(key)
        if request is None or key in seen:
            raise ValueError(f"retrieval output outside/duplicate frozen identity: {key}")
        seen.add(key)
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
            FORBIDDEN_FIELDS & set(row)
            or not _valid_passages(passages)
            or row.get("passages_sha256") != _sha_json(passages)
            or row.get("retrieval_source") != RETRIEVAL_STACK
        ):
            raise ValueError(f"retrieval passage/backend contract failed: {key}")
        row["schema_version"] = SCHEMA_VERSION
        output.append(row)
    if seen != set(expected):
        raise ValueError(
            f"retrieval output misses {len(set(expected) - seen)} frozen identities"
        )
    # Preserve the preregistered row order irrespective of backend batching.
    indexed = {row["question_key"]: row for row in output}
    return [indexed[str(request["question_key"])] for request in requests]


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
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version") != EXPECTED_PROTOCOL_SCHEMA
        or protocol.get("status") != EXPECTED_PROTOCOL_STATUS
        or ((protocol.get("retrieval") or {}).get("stack") != RETRIEVAL_STACK)
        or ((protocol.get("retrieval") or {}).get("backend_fallback_allowed") is not False)
    ):
        raise ValueError("reserve retrieval protocol schema/status/stack invalid")
    requests_ref = (protocol.get("outputs") or {}).get("retrieval_requests")
    if not isinstance(requests_ref, Mapping):
        raise ValueError("protocol does not bind retrieval_requests")
    requests_path = _resolve_ref(requests_ref, label="retrieval requests")
    requests = _read_jsonl(requests_path)
    _validate_requests(requests)
    if retrieval_fn is None:
        # Attest before creating any output, so a missing reranker cannot leave
        # a misleading partial release.
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
    if attestation.get("backend_fallback") is not False or attestation.get(
        "load_succeeded"
    ) is not True:
        raise RuntimeError(f"canonical reranker attestation failed: {attestation}")
    contexts = _validate_contexts(
        requests, runner(DATASET, list(requests), batch_size)
    )
    gates = {
        "protocol_and_request_hash_bound": True,
        "n_exact_50": len(contexts) == 50,
        "identity_join_rate_1": {row["question_key"] for row in contexts}
        == {row["question_key"] for row in requests},
        "request_order_preserved": [row["question_key"] for row in contexts]
        == [row["question_key"] for row in requests],
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
    }
    if not all(gates.values()):
        raise RuntimeError(f"reserve retrieval gates failed: {gates}")
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
            "by_dataset": dict(Counter(row["dataset"] for row in contexts)),
        },
        "retrieval": RETRIEVAL_STACK,
        "backend_attestation": attestation,
        "gates": gates,
        "inputs": {
            "protocol": file_ref(protocol_path),
            "retrieval_requests": file_ref(requests_path),
        },
        "outputs": {"retrieval_contexts": file_ref(contexts_path)},
        "scientific_boundary": {
            "identity_selection_performed": False,
            "gold_fields_read_or_written": False,
            "kg_constructed": False,
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
            "phase": "materialize_2wiki_proofkg_reserve_retrieval_v1",
            "experiment_id": report["experiment_id"],
            "report": file_ref(report_path),
            "retrieval_contexts": report["outputs"]["retrieval_contexts"],
            "backend_attestation": attestation,
            "gold_access": False,
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
