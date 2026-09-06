#!/usr/bin/env python3
"""Materialise canonical retrieval for the frozen mixed3-v4 H/M additions.

The input is the answer-free ``retrieval_requests`` artifact bound by the v4
population protocol.  This script does not select questions, read Gold labels,
construct a KG, or update a model.  It only runs the same canonical retrieval
function used by QPEG-v1 (E5 + BM25 -> RRF -> BGE reranker -> top 10) and writes
an append-only, identity-checked retrieval release.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.prepare.materialize_qpeg_v1_retrieval import (
    FORBIDDEN_FIELDS,
    _sha_json,
    materialize_dataset,
)


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("hotpotqa", "musique")
EXPECTED_PROTOCOL_SCHEMA = "mixed-ppo-three-dataset-protocol-v4-proof800"
EXPECTED_PROTOCOL_STATUS = "FROZEN_ANSWER_FREE_NOT_MATERIALIZED_NOT_TRAINED"
HM_PREREG_PROTOCOL_SCHEMA = (
    "mixed-ppo-three-dataset-v4-hm-expansion-preregistration-v1"
)
HM_PREREG_PROTOCOL_STATUS = (
    "FROZEN_HM_IDENTITIES_RETRIEVAL_NOT_MATERIALIZED_"
    "2WIKI_UNRESOLVED_NOT_TRAINED"
)
ACCEPTED_PROTOCOL_CONTRACTS = {
    EXPECTED_PROTOCOL_SCHEMA: EXPECTED_PROTOCOL_STATUS,
    HM_PREREG_PROTOCOL_SCHEMA: HM_PREREG_PROTOCOL_STATUS,
}
SCHEMA_VERSION = "mixed3-v4-expansion-retrieval-v1"
REPORT_SCHEMA_VERSION = "mixed3-v4-expansion-retrieval-report-v1"
STATUS = "COMPLETE_ANSWER_FREE_RETRIEVAL_NOT_TRAINED"
EXPERIMENT_ID = "MIXED3-V4-EXPANSION-RETRIEVAL-H417-M401-SEED42-V1"
RETRIEVAL_STACK = (
    "E5@100+BM25@100->RRF60@50->bge-reranker-v2-m3@10->pack3860"
)
RERANKER_MODEL_DIR = ROOT / "models" / "bge-reranker-v2-m3"
RERANKER_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _attest_cross_encoder_backend() -> dict[str, Any]:
    """Load the exact local reranker before retrieval and forbid fallback.

    ``rerank_with_cross_encoder`` historically falls back to BM25 when model
    loading fails.  That behaviour is useful for interactive inference but is
    not acceptable for a frozen research-data release: the declared and
    executed retrieval stacks must be identical.  Preloading the model puts it
    in the shared CE cache, so the subsequent canonical calls cannot take the
    load-failure fallback branch.
    """

    model_dir = RERANKER_MODEL_DIR.resolve()
    missing = [name for name in RERANKER_REQUIRED_FILES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"canonical BGE reranker is incomplete at {model_dir}: {missing}"
        )
    from kgproweight.retrieval.reranker import get_cross_encoder

    model = get_cross_encoder(str(model_dir))
    if model is None:
        raise RuntimeError("canonical BGE reranker load returned no model")
    return {
        "mode": "cross_encoder",
        "requested_backend": "bge-reranker-v2-m3",
        "resolved_model_path": str(model_dir),
        "config": _identity(model_dir / "config.json"),
        "weights": _identity(model_dir / "model.safetensors"),
        "tokenizer": _identity(model_dir / "tokenizer.json"),
        "load_succeeded": True,
        "backend_fallback": False,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


def _resolve_bound_file(
    identity: Mapping[str, Any], *, label: str, root: Path = ROOT
) -> Path:
    raw_path = str(identity.get("path") or "").strip()
    expected_hash = str(identity.get("sha256") or "").strip()
    if not raw_path or not expected_hash:
        raise ValueError(f"{label} identity must bind path and sha256")
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen {label}: {path}")
    if _sha256(path) != expected_hash:
        raise ValueError(f"frozen {label} SHA256 mismatch: {path}")
    if identity.get("size_bytes") is not None and path.stat().st_size != int(
        identity["size_bytes"]
    ):
        raise ValueError(f"frozen {label} size mismatch: {path}")
    return path


def _frozen_counts(protocol: Mapping[str, Any]) -> dict[str, int]:
    raw = (protocol.get("population") or {}).get(
        "retrieval_requests_by_dataset"
    )
    if not isinstance(raw, Mapping) or set(raw) != set(DATASETS):
        raise ValueError(
            "protocol.population.retrieval_requests_by_dataset must contain "
            "exactly hotpotqa and musique"
        )
    counts = {dataset: int(raw[dataset]) for dataset in DATASETS}
    if any(value <= 0 for value in counts.values()):
        raise ValueError(f"frozen retrieval counts must be positive: {counts}")
    return counts


def _validate_requests(
    rows: Sequence[Mapping[str, Any]], expected_counts: Mapping[str, int]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for index, value in enumerate(rows, start=1):
        row = dict(value)
        dataset = str(row.get("dataset") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        question = str(row.get("question") or "").strip()
        if dataset not in DATASETS:
            raise ValueError(
                f"retrieval request {index} has forbidden dataset {dataset!r}; "
                "only hotpotqa/musique expansion is allowed"
            )
        if not qid or not question:
            raise ValueError(f"retrieval request {index} has empty qid/question")
        key = question_key(dataset, qid)
        if key in seen:
            raise ValueError(f"duplicate retrieval request identity: {key}")
        seen.add(key)
        present = sorted(FORBIDDEN_FIELDS & set(row))
        if present:
            raise ValueError(f"{key}: forbidden Gold/evidence fields: {present}")
        expected_identity = {
            "question_key": key,
            "question_sha256": question_sha256(question),
            "family_version": FAMILY_VERSION,
            "family_sha256": family_sha256(question),
            "role": "rollout_retrieval",
            "gold_access": False,
        }
        for field, expected in expected_identity.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"{key}: request {field}={row.get(field)!r}, expected {expected!r}"
                )
        grouped[dataset].append(row)

    actual_counts = Counter(str(row["dataset"]) for row in rows)
    if dict(actual_counts) != dict(expected_counts):
        raise ValueError(
            f"retrieval request counts {dict(actual_counts)} != frozen "
            f"{dict(expected_counts)}"
        )
    return {dataset: grouped[dataset] for dataset in DATASETS}


def _passages_valid(passages: Any) -> bool:
    if not isinstance(passages, list) or len(passages) != 10:
        return False
    for passage in passages:
        if not isinstance(passage, Mapping):
            return False
        if any(not str(passage.get(field) or "").strip() for field in (
            "id", "source", "contents"
        )):
            return False
        if FORBIDDEN_FIELDS & set(passage):
            return False
    return True


def _validate_contexts(
    requests: Sequence[Mapping[str, Any]], contexts: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    expected = {str(row["question_key"]): dict(row) for row in requests}
    if len(expected) != len(requests):
        raise ValueError("request identities became non-unique before retrieval")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(contexts, start=1):
        row = dict(value)
        dataset = str(row.get("dataset") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        key = question_key(dataset, qid)
        if key in seen:
            raise ValueError(f"duplicate materialized retrieval identity: {key}")
        seen.add(key)
        request = expected.get(key)
        if request is None:
            raise ValueError(f"retrieval output outside frozen requests: {key}")
        present = sorted(FORBIDDEN_FIELDS & set(row))
        if present:
            raise ValueError(f"{key}: retrieval output has forbidden fields: {present}")
        for field in (
            "question_key", "dataset", "qid", "question", "question_sha256",
            "family_sha256", "role", "gold_access",
        ):
            if row.get(field) != request.get(field):
                raise ValueError(
                    f"{key}: retrieval/request identity mismatch at {field}"
                )
        passages = row.get("passages")
        if not _passages_valid(passages):
            raise ValueError(f"{key}: expected ten safe nonempty passages")
        if row.get("passages_sha256") != _sha_json(passages):
            raise ValueError(f"{key}: passages_sha256 mismatch")
        if row.get("retrieval_source") != RETRIEVAL_STACK:
            raise ValueError(f"{key}: canonical retrieval stack drifted")
        row["schema_version"] = SCHEMA_VERSION
        output.append(row)
    if seen != set(expected):
        missing = sorted(set(expected) - seen)
        raise ValueError(
            f"retrieval output misses {len(missing)} frozen identities: {missing[:5]}"
        )
    return output


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
    """Execute and validate the append-only retrieval release."""

    protocol_path = protocol_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite retrieval output: {output_dir}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not str(experiment_id).strip():
        raise ValueError("a nonempty Experiment ID is required")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_schema = str(protocol.get("schema_version") or "")
    expected_status = ACCEPTED_PROTOCOL_CONTRACTS.get(protocol_schema)
    if expected_status is None:
        raise ValueError(
            f"unexpected v4 protocol schema: {protocol_schema!r}"
        )
    if protocol.get("status") != expected_status:
        raise ValueError(
            f"unexpected/unfrozen v4 protocol status: {protocol.get('status')!r}"
        )
    expected_counts = _frozen_counts(protocol)
    request_identity = (protocol.get("outputs") or {}).get("retrieval_requests")
    if not isinstance(request_identity, Mapping):
        raise ValueError("v4 protocol does not bind outputs.retrieval_requests")
    requests_path = _resolve_bound_file(
        request_identity, label="retrieval_requests"
    )
    requests = _read_jsonl(requests_path)
    grouped = _validate_requests(requests, expected_counts)

    if retrieval_fn is None:
        # This is intentionally executed before the output directory is
        # created.  An unavailable canonical reranker therefore leaves no
        # partial release behind and can never be mislabeled as a BGE run.
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
    if attestation.get("backend_fallback") is not False:
        raise RuntimeError(f"reranker fallback is forbidden: {attestation}")
    contexts_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        raw_contexts = runner(dataset, grouped[dataset], batch_size)
        contexts_by_dataset[dataset] = _validate_contexts(
            grouped[dataset], raw_contexts
        )
    contexts = [
        row for dataset in DATASETS for row in contexts_by_dataset[dataset]
    ]
    output_counts = Counter(str(row["dataset"]) for row in contexts)
    gates = {
        "protocol_frozen": True,
        "request_hash_and_size_match": True,
        "request_counts_match_protocol": dict(output_counts) == expected_counts,
        "only_hotpotqa_and_musique": set(output_counts) == set(DATASETS),
        "identity_join_rate_1": len(contexts) == len(requests)
        and {row["question_key"] for row in contexts}
        == {row["question_key"] for row in requests},
        "all_gold_access_false": all(row.get("gold_access") is False for row in contexts),
        "forbidden_fields_zero": all(
            not (FORBIDDEN_FIELDS & set(row))
            and all(not (FORBIDDEN_FIELDS & set(p)) for p in row["passages"])
            for row in contexts
        ),
        "all_exactly_ten_safe_passages": all(
            _passages_valid(row.get("passages")) for row in contexts
        ),
        "canonical_retrieval_stack_exact": all(
            row.get("retrieval_source") == RETRIEVAL_STACK for row in contexts
        ),
        "backend_fallback_false": attestation.get("backend_fallback") is False,
        "backend_load_attested": attestation.get("load_succeeded") is True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"mixed3-v4 retrieval gates failed: {gates}")

    output_dir.mkdir(parents=True, exist_ok=False)
    output_paths = {
        dataset: output_dir / f"{dataset}.retrieval_contexts.jsonl"
        for dataset in DATASETS
    }
    output_paths["combined"] = output_dir / "retrieval_contexts.jsonl"
    for dataset in DATASETS:
        _write_jsonl(output_paths[dataset], contexts_by_dataset[dataset])
    _write_jsonl(output_paths["combined"], contexts)

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment_id": str(experiment_id).strip(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "counts": {
            "requests_total": len(requests),
            "contexts_total": len(contexts),
            "by_dataset": dict(output_counts),
        },
        "retrieval": RETRIEVAL_STACK,
        "backend_attestation": attestation,
        "gates": gates,
        "scientific_boundary": {
            "identity_selection_performed": False,
            "gold_fields_read_or_written": False,
            "kg_constructed": False,
            "model_updated": False,
            "training_started": False,
        },
        "inputs": {
            "protocol": _identity(protocol_path),
            "retrieval_requests": _identity(requests_path),
        },
        "outputs": {
            name: _identity(path) for name, path in output_paths.items()
        },
        "training_started": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    dump_manifest(
        output_dir,
        status=STATUS,
        extra={
            "phase": "mixed3_v4_expansion_retrieval",
            "experiment_id": report["experiment_id"],
            "protocol": report["inputs"]["protocol"],
            "retrieval_requests": report["inputs"]["retrieval_requests"],
            "outputs": {
                **report["outputs"],
                "report": _identity(report_path),
            },
            "counts": report["counts"],
            "retrieval": RETRIEVAL_STACK,
            "backend_attestation": attestation,
            "gold_access": False,
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "outputs/audits/"
            "mixed_ppo_three_dataset_v4_hm_expansion_"
            "h1000_m1000_seed42_preregistration/"
            "protocol.json"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/audits/"
            "mixed3_v4_expansion_retrieval_h417_m401_seed42_v1"
        ),
    )
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
