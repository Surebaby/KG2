#!/usr/bin/env python3
"""Materialize the official-raw 2Wiki unified ProofKG candidate supply v3.

Unlike v2, this release has no historical extension/reserve source ambiguity.
Every row must belong to the answer-free strict closure-v3 scope, have a
canonical official-raw retrieval context, and retain an exact hash join to its
closure runtime.  Gold answers are read only from official *train* rows after
the scope is frozen and are carried solely as PPO outcome labels; they are
never used to build/rank/filter the graph candidate pool.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.reward.trajectory_source_gate import make_source_gate_record
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.audit_auto1500_v4_clean_reproducibility import (
    DEFAULT_PROTECTED_LEDGER_DIR,
    PROTECTED_LEDGER_SCHEMA_VERSION,
    canonical_sha256,
    validate_protected_ledger_release,
)
from scripts.prepare.freeze_2wiki_official_raw_canonical_retrieval_v1 import (
    DATASET,
    MIN_PER_TYPE,
    QTYPES,
    SCOPE_SCHEMA,
    SCOPE_STATUS,
)
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.prepare.materialize_2wiki_official_raw_canonical_retrieval_v1 import (
    REPORT_SCHEMA_VERSION as RETRIEVAL_REPORT_SCHEMA,
    SCHEMA_VERSION as RETRIEVAL_CONTEXT_SCHEMA,
    STATUS as RETRIEVAL_STATUS,
)
from scripts.prepare import materialize_2wiki_proofkg_unified_v2 as v2


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_SCHEMA_VERSION = "2wiki-unified-proofkg-official-raw-contract-v3"
CONTRACT_STATUS = "FROZEN_OFFICIAL_RAW_SOURCE_CONTRACT_NOT_MATERIALIZED_NOT_TRAINED"
SCHEMA_VERSION = "2wiki-unified-proofkg-official-raw-candidate-supply-v3"
STATUS = "COMPLETE_STRICT_OFFICIAL_RAW_CANDIDATE_SUPPLY_NOT_SELECTED_NOT_TRAINED"
CANDIDATE_SCHEMA_VERSION = "2wiki-unified-proofkg-official-raw-candidate-wrapper-v3"
SOURCE_RELEASE = "2wiki_official_raw_canonical_retrieval_v1"
REQUIRED_OUTPUTS = (
    "silver_train",
    "question_kg_records",
    "source_gate_records",
    "proof_candidates",
)
DEFAULT_CONTRACT = ROOT / (
    "outputs/audits/2wiki_unified_proofkg_official_raw_v3_contract/"
    "unified_contract.json"
)
DEFAULT_SCOPE = ROOT / (
    "outputs/audits/2wiki_official_raw_canonical_retrieval_v1_scope_"
    "preregistration"
)
DEFAULT_CLOSURE = ROOT / (
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "clean_closure_v3_result"
)
DEFAULT_RETRIEVAL = ROOT / (
    "outputs/audits/2wiki_official_raw_canonical_retrieval_v1"
)
DEFAULT_RAW = ROOT / "data/2wikimultihopqa/train.jsonl"
DEFAULT_OUT = ROOT / "data/derived/2wiki_unified_proofkg_official_raw_v3"
EXPERIMENT_ID = "2WIKI-UNIFIED-PROOFKG-OFFICIAL-RAW-V3-SEED42"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return v2._read_jsonl(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    v2._write_jsonl(path, rows)


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


def _same_identity(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(actual.get(field) == expected.get(field) for field in ("sha256", "size_bytes"))


def _resolve_identity(value: Mapping[str, Any], *, label: str) -> Path:
    raw = str(value.get("path") or "").strip()
    if not raw:
        raise ValueError(f"{label}: empty path")
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file() or _sha256(path) != str(value.get("sha256") or ""):
        raise ValueError(f"{label}: missing file or SHA256 drift")
    if value.get("size_bytes") is not None and path.stat().st_size != int(
        value["size_bytes"]
    ):
        raise ValueError(f"{label}: size drift")
    return path


def validate_contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _read_json(path)
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or contract.get("status") != CONTRACT_STATUS
        or contract.get("release_schema_version") != SCHEMA_VERSION
        or contract.get("release_status") != STATUS
        or contract.get("candidate_wrapper_schema_version")
        != CANDIDATE_SCHEMA_VERSION
        or tuple(contract.get("required_outputs") or ()) != REQUIRED_OUTPUTS
        or contract.get("training_started") is not False
    ):
        raise ValueError("official-raw unified-v3 source contract drifted")
    implementation = contract.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("unified-v3 contract does not bind implementation")
    implementation_path = _resolve_identity(
        implementation, label="unified-v3 materializer"
    )
    if implementation_path != Path(__file__).resolve():
        raise ValueError("unified-v3 contract binds a different materializer")
    return contract, {
        "contract": _identity(path),
        "implementation": _identity(implementation_path),
    }


def validate_scope_release(
    directory: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    protocol_path = directory / "protocol.json"
    report_path = directory / "report.json"
    manifest_path = directory / "manifest.json"
    protocol = _read_json(protocol_path)
    report = _read_json(report_path)
    manifest = _read_json(manifest_path)
    if (
        protocol.get("schema_version") != SCOPE_SCHEMA
        or protocol.get("status") != SCOPE_STATUS
        or report.get("status") != SCOPE_STATUS
        or manifest.get("status") != SCOPE_STATUS
        or not all(v is True for v in (protocol.get("gates") or {}).values())
        or (protocol.get("scientific_boundary") or {}).get("answer_or_support_fields_read")
        is not False
        or (protocol.get("scientific_boundary") or {}).get("retrieval_started")
        is not False
        or (protocol.get("scientific_boundary") or {}).get("training_started")
        is not False
    ):
        raise ValueError("official-raw retrieval scope release failed")
    requests_ref = (protocol.get("outputs") or {}).get("retrieval_requests")
    if not isinstance(requests_ref, Mapping):
        raise ValueError("scope release does not bind requests")
    requests_path = _resolve_identity(requests_ref, label="scope requests")
    requests = _read_jsonl(requests_path)
    request_index = v2._index(requests, label="scope requests")
    by_type = Counter(str(row.get("question_type") or "") for row in requests)
    if (
        len(requests) != int((protocol.get("population") or {}).get("strict_scope_total", -1))
        or any(by_type[qtype] < MIN_PER_TYPE for qtype in QTYPES)
        or any(
            int(((protocol.get("population") or {}).get("strict_scope_by_question_type") or {}).get(qtype, -1))
            != by_type[qtype]
            for qtype in QTYPES
        )
    ):
        raise ValueError("scope release population/counts drifted")
    for key, row in request_index.items():
        question = str(row.get("question") or "").strip()
        if (
            str(row.get("dataset") or "").strip().lower() != DATASET
            or str(row.get("question_key") or "") != key
            or str(row.get("question_sha256") or "") != question_sha256(question)
            or row.get("family_version") != FAMILY_VERSION
            or str(row.get("family_sha256") or "") != family_sha256(question)
            or str(row.get("question_type") or "") not in QTYPES
            or row.get("gold_access") is not False
            or row.get("evaluation_eligible") is not False
        ):
            raise ValueError(f"scope request identity boundary failed: {key}")
    manifest_report = (manifest.get("run") or {}).get("report") or {}
    if not isinstance(manifest_report, Mapping) or not _same_identity(
        manifest_report, _identity(report_path)
    ):
        raise ValueError("scope manifest/report binding drifted")
    closure_binding = (protocol.get("inputs") or {}).get("closure_v3_release")
    if not isinstance(closure_binding, Mapping):
        raise ValueError("scope does not bind closure-v3 release")
    return requests, request_index, {
        "protocol": _identity(protocol_path),
        "report": _identity(report_path),
        "manifest": _identity(manifest_path),
        "retrieval_requests": _identity(requests_path),
        "closure_v3_release": dict(closure_binding),
    }


def validate_retrieval_release(
    directory: Path,
    *,
    scope_binding: Mapping[str, Mapping[str, Any]],
    expected_keys: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contexts_path = directory / "retrieval_contexts.jsonl"
    report_path = directory / "report.json"
    manifest_path = directory / "manifest.json"
    report = _read_json(report_path)
    manifest = _read_json(manifest_path)
    if (
        report.get("schema_version") != RETRIEVAL_REPORT_SCHEMA
        or report.get("status") != RETRIEVAL_STATUS
        or manifest.get("status") != RETRIEVAL_STATUS
        or not all(v is True for v in (report.get("gates") or {}).values())
        or report.get("training_started") is not False
    ):
        raise ValueError("official-raw retrieval release schema/status/gates failed")
    attestation = report.get("backend_attestation") or {}
    if not (
        attestation.get("mode") == "cross_encoder"
        and attestation.get("requested_backend") == "bge-reranker-v2-m3"
        and attestation.get("load_succeeded") is True
        and attestation.get("backend_fallback") is False
    ):
        raise ValueError("official-raw retrieval lacks exact BGE attestation")
    scope_ref = (report.get("inputs") or {}).get("scope_protocol") or {}
    if not isinstance(scope_ref, Mapping) or not _same_identity(
        scope_ref, scope_binding["protocol"]
    ):
        raise ValueError("retrieval release/scope protocol binding mismatch")
    contexts_ref = (report.get("outputs") or {}).get("retrieval_contexts") or {}
    if not isinstance(contexts_ref, Mapping) or not _same_identity(
        contexts_ref, _identity(contexts_path)
    ):
        raise ValueError("retrieval release/context binding mismatch")
    contexts = _read_jsonl(contexts_path)
    context_index = v2._index(contexts, label="official-raw retrieval")
    if set(context_index) != expected_keys:
        raise ValueError("retrieval/scope identity join is not exact")
    for key, row in context_index.items():
        if (
            row.get("schema_version") != RETRIEVAL_CONTEXT_SCHEMA
            or len(row.get("passages") or []) != 10
            or row.get("gold_access") is not False
            or row.get("retrieval_source") != v2.CANONICAL_RETRIEVAL_STACK
            or row.get("passages_sha256") != v2._json_sha256(row.get("passages") or [])
        ):
            raise ValueError(f"retrieval context contract failed: {key}")
    manifest_report = (manifest.get("run") or {}).get("report") or {}
    if not isinstance(manifest_report, Mapping) or not _same_identity(
        manifest_report, _identity(report_path)
    ):
        raise ValueError("retrieval manifest/report binding drifted")
    return contexts, {
        "report": _identity(report_path),
        "manifest": _identity(manifest_path),
        "retrieval_contexts": _identity(contexts_path),
    }


def _assert_closure_binding(
    scoped: Mapping[str, Mapping[str, Any]], actual: Mapping[str, Mapping[str, Any]]
) -> None:
    # Scope freezes the four decisive closure files.  Selector-v2's stricter
    # validator additionally exposes the upstream execution lock, raw closure
    # report and runtime report; those extra attestations are welcome, but the
    # four identities already frozen by the retrieval scope must remain exact.
    if not set(scoped).issubset(actual):
        raise ValueError("scope closure bindings are absent from live closure-v3")
    for name in scoped:
        if not _same_identity(scoped[name], actual[name]):
            raise ValueError(f"scope/closure SHA binding drifted: {name}")


def build_official_raw_supply(
    *,
    requests: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    retrieval_rows: Sequence[Mapping[str, Any]],
    blocked_qids: set[str],
    blocked_hashes: set[str],
    blocked_families: set[str],
    cutoff: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    request_index = v2._index(requests, label="official-raw scope")
    runtime_index = v2._index(runtime_rows, label="closure runtime")
    if set(request_index) != set(runtime_index):
        raise ValueError("scope/runtime identity join is not exact")
    for key, request in request_index.items():
        trace = runtime_index[key]
        if (
            str(request.get("closure_runtime_record_sha256") or "")
            != canonical_sha256(trace)
            or str(request.get("closure_kg_sha256") or "")
            != canonical_sha256(trace.get("kg_subgraph") or [])
            or str(request.get("closure_execution_sha256") or "")
            != canonical_sha256(trace.get("execution") or {})
        ):
            raise ValueError(f"scope/runtime closure hash mismatch: {key}")

    silver, records, _old_gates, stats = v2.build_supply(
        old_silver=[],
        old_records=[],
        old_runtime=[],
        new_source=[],
        new_cohort=requests,
        new_runtime=runtime_rows,
        blocked_qids=blocked_qids,
        blocked_hashes=blocked_hashes,
        blocked_families=blocked_families,
        cutoff=cutoff,
        new_raw=raw_rows,
        new_retrieval=retrieval_rows,
    )
    if len(silver) != len(requests):
        raise RuntimeError(
            f"official-raw strict scope lost rows during unified gate: {len(silver)}/{len(requests)}; {stats}"
        )

    normalized_records: list[dict[str, Any]] = []
    normalized_silver: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    for silver_row, record in zip(silver, records):
        record = dict(record)
        provenance = dict(record.get("provenance") or {})
        provenance.update(
            {
                "unified_supply_version": SCHEMA_VERSION,
                "unified_source_release": SOURCE_RELEASE,
                "canonical_retrieval_release": RETRIEVAL_STATUS,
                "source_gold_steps_copied": 0,
                "gold_use": "outcome_reward_label_only",
            }
        )
        record["provenance"] = provenance
        key = str(record["question_key"])
        request = request_index[key]
        gate = make_source_gate_record(
            record,
            dataset=DATASET,
            qid=str(record["qid"]),
            question=str(record["question"]),
            text_evidence_available=True,
            historical_cutoff=cutoff,
        )
        if gate.get("m_graph") != 1 or not all(
            value is True for value in (gate.get("eligibility_checks") or {}).values()
        ):
            raise RuntimeError(f"normalized official-raw graph gate failed: {key}")
        silver_row = dict(silver_row)
        metadata = dict(silver_row.get("metadata") or {})
        metadata.update(
            {
                "proof_source": SOURCE_RELEASE,
                "unified_supply_version": SCHEMA_VERSION,
                "question_type": str(request["question_type"]),
                "gold_use": "outcome_reward_label_only",
            }
        )
        silver_row["metadata"] = metadata
        normalized_silver.append(silver_row)
        normalized_records.append(record)
        gates.append(gate)

    wrappers = v2.build_candidate_wrappers(
        normalized_silver, normalized_records, gates
    )
    for wrapper in wrappers:
        wrapper["schema_version"] = CANDIDATE_SCHEMA_VERSION
        wrapper["source_release"] = SOURCE_RELEASE
    # v2's stable construction helper uses its historical reserve label while
    # assembling fallback rows.  Never expose that internal compatibility label
    # in the v3 release telemetry.
    stats = dict(stats)
    stats["eligible_by_source"] = {SOURCE_RELEASE: len(normalized_silver)}
    return normalized_silver, normalized_records, gates, wrappers, stats


def materialize(
    *,
    contract_path: Path,
    scope_dir: Path,
    closure_dir: Path,
    retrieval_dir: Path,
    raw_path: Path,
    protected_ledger_dir: Path,
    cutoff: str,
    output_dir: Path,
    experiment_id: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite unified-v3 release: {output_dir}")
    _contract, contract_binding = validate_contract(contract_path)
    requests, request_index, scope_binding = validate_scope_release(scope_dir)

    # Import here to keep the source-contract implementation independent from
    # selector-v2 while sharing the exact closure-v3 validator.
    from scripts.prepare.select_2wiki_proof800_v2 import validate_closure_v3_release

    _telemetry, closure_binding = validate_closure_v3_release(closure_dir)
    _assert_closure_binding(
        scope_binding["closure_v3_release"], closure_binding
    )
    runtime_path = _resolve_identity(
        closure_binding["runtime_details"], label="closure runtime"
    )
    runtime_index = v2._index(_read_jsonl(runtime_path), label="closure runtime")
    runtime_rows = [runtime_index[str(row["question_key"])] for row in requests]

    retrieval_rows, retrieval_binding = validate_retrieval_release(
        retrieval_dir,
        scope_binding=scope_binding,
        expected_keys=set(request_index),
    )
    ledger_path, ledger_report_path, ledger_manifest_path, _ledger_report = (
        validate_protected_ledger_release(protected_ledger_dir)
    )
    ledger_binding = {
        "ledger": _identity(ledger_path),
        "report": _identity(ledger_report_path),
        "manifest": _identity(ledger_manifest_path),
    }
    blocked_qids, blocked_hashes, blocked_families = v2._blocked([ledger_path])
    selected_qids = {str(row["qid"]) for row in requests}
    raw_rows = v2._selected_raw(raw_path, qids=selected_qids)
    silver, records, gates, wrappers, stats = build_official_raw_supply(
        requests=requests,
        runtime_rows=runtime_rows,
        raw_rows=raw_rows,
        retrieval_rows=retrieval_rows,
        blocked_qids=blocked_qids,
        blocked_hashes=blocked_hashes,
        blocked_families=blocked_families,
        cutoff=cutoff,
    )

    type_counts = Counter(
        str((row.get("metadata") or {}).get("question_type") or "")
        for row in silver
    )
    key_sets = [
        {question_key(str(row.get("dataset") or ""), str(row.get("qid") or "")) for row in rows}
        for rows in (silver, records, gates, wrappers)
    ]
    checks = {
        "source_contract_and_implementation_hash_bound": True,
        "scope_closure_retrieval_hash_join_exact": all(keys == set(request_index) for keys in key_sets),
        "strict_candidates_equal_frozen_scope": len(silver) == len(requests),
        "each_question_type_at_least_200": all(type_counts[qtype] >= MIN_PER_TYPE for qtype in QTYPES),
        "all_graph_gate_pass": all(
            row.get("m_graph") == 1
            and all(v is True for v in (row.get("eligibility_checks") or {}).values())
            for row in gates
        ),
        "all_steps_empty": all(row.get("steps") == [] for row in silver),
        "all_exactly_ten_canonical_passages": all(
            len(row.get("retrieved_passages") or []) == 10
            and (row.get("metadata") or {}).get("retrieved_passages_sha256")
            == v2._json_sha256(row.get("retrieved_passages") or [])
            for row in silver
        ),
        "source_release_unambiguous_official_raw": all(
            (row.get("provenance") or {}).get("unified_source_release") == SOURCE_RELEASE
            for row in records
        ) and all(
            (row.get("metadata") or {}).get("proof_source") == SOURCE_RELEASE
            for row in silver
        ),
        "candidate_wrapper_schema_v3": all(
            row.get("schema_version") == CANDIDATE_SCHEMA_VERSION for row in wrappers
        ),
        "source_gold_steps_or_kg_copied_false": all(
            (row.get("provenance") or {}).get("source_gold_steps_copied") == 0
            for row in records
        ),
        "protected_qid_hash_family_overlap_zero": all(
            row["qid"] not in blocked_qids
            and question_sha256(row["question"]) not in blocked_hashes
            and family_sha256(row["question"]) not in blocked_families
            for row in silver
        ),
        "final_proof800_selection_not_performed": True,
        "training_not_started": True,
    }
    if not all(checks.values()):
        raise RuntimeError(
            {"checks": checks, "counts": dict(type_counts), "stats": stats}
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    output_paths = {
        "silver_train": output_dir / "silver_train.jsonl",
        "question_kg_records": output_dir / "question_kg_records.jsonl",
        "source_gate_records": output_dir / "source_gate_records.jsonl",
        "proof_candidates": output_dir / "proof_candidates.jsonl",
    }
    for name, rows in zip(
        REQUIRED_OUTPUTS, (silver, records, gates, wrappers)
    ):
        _write_jsonl(output_paths[name], rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(experiment_id).strip(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "source_contract": contract_binding["contract"],
        "counts": {
            "strict_candidates": len(silver),
            "by_question_type": {qtype: type_counts[qtype] for qtype in QTYPES},
            **stats,
        },
        "checks": checks,
        "scientific_boundary": {
            "train_only": True,
            "scope_and_graph_generation_gold_access": False,
            "source_gold_steps_or_kg_copied": False,
            "gold_answer_use": "outcome_reward_label_only_after_scope_freeze",
            "proof800_selection_performed": False,
            "canonical_passages_used_for_selection": False,
            "training_started": False,
        },
        "protected_ledger": {
            "version": PROTECTED_LEDGER_SCHEMA_VERSION,
            "complete": True,
            "current_family_recomputed": True,
            **ledger_binding,
        },
        "inputs": {
            "source_contract": contract_binding["contract"],
            "scope_release": scope_binding,
            "closure_v3_release": closure_binding,
            "canonical_retrieval_release": retrieval_binding,
            "official_raw_train": _identity(raw_path),
        },
        "outputs": {name: _identity(path) for name, path in output_paths.items()},
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
            "phase": "unified_2wiki_proofkg_official_raw_v3_candidate_supply",
            "experiment_id": report["experiment_id"],
            "report": _identity(report_path),
            "source_contract": contract_binding["contract"],
            "proof800_selected": False,
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--scope-dir", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--closure-dir", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--retrieval-dir", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--protected-ledger-dir", type=Path, default=DEFAULT_PROTECTED_LEDGER_DIR
    )
    parser.add_argument("--cutoff", default="2020-12-09T23:59:59Z")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    args = parser.parse_args()
    report = materialize(
        contract_path=args.contract,
        scope_dir=args.scope_dir,
        closure_dir=args.closure_dir,
        retrieval_dir=args.retrieval_dir,
        raw_path=args.raw,
        protected_ledger_dir=args.protected_ledger_dir,
        cutoff=args.cutoff,
        output_dir=args.out,
        experiment_id=args.experiment_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
