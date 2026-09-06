#!/usr/bin/env python3
"""Materialise the full-ledger-safe H/M canonical retrieval successor.

This append-only materialiser consumes the frozen H/M reconciliation protocol.
It reuses exactly 812 passage blocks from the previously attested canonical
release, retrieves only the 11 newly selected MuSiQue questions, retires the
six explicitly removed contexts, and emits the exact H417/M406=823 release.

No raw dataset, answer, support label, KG, or model-training asset is read.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from kgproweight.kg.question_kg import question_key
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.materialize_mixed3_v4_expansion_retrieval import (
    DATASETS,
    RETRIEVAL_STACK,
    STATUS,
    _attest_cross_encoder_backend,
    _identity,
    _passages_valid,
    _read_jsonl,
    _resolve_bound_file,
    _sha256,
    _sha_json,
    _validate_contexts,
    _validate_requests,
    _write_jsonl,
    materialize_dataset,
)
from scripts.prepare.materialize_qpeg_v1_retrieval import FORBIDDEN_FIELDS


ROOT = Path(__file__).resolve().parents[2]
RECONCILIATION_SCHEMA = "mixed-ppo-v4-hm-full-ledger-reconciliation-v2"
RECONCILIATION_STATUS = (
    "FROZEN_HM_FULL_LEDGER_DELTA_RETRIEVAL_NOT_RUN_NOT_TRAINED"
)
REPORT_SCHEMA_VERSION = "mixed3-v4-reconciled-retrieval-report-v2"
EXPERIMENT_ID = "MIXED3-V4-EXPANSION-RETRIEVAL-H417-M406-SEED42-V2"
DEFAULT_PROTOCOL = Path(
    "outputs/audits/"
    "mixed_ppo_v4_hm_full_ledger_reconciliation_v2_seed42_preregistration/"
    "protocol.json"
)
DEFAULT_OUT = Path(
    "outputs/audits/mixed3_v4_expansion_retrieval_h417_m406_seed42_v2"
)
EXPECTED_COUNTS = {"hotpotqa": 417, "musique": 406}


def _index(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        dataset = str(row.get("dataset") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        if dataset not in DATASETS or not qid:
            raise ValueError(f"{label}: invalid identity {dataset!r}/{qid!r}")
        key = question_key(dataset, qid)
        if key in result:
            raise ValueError(f"{label}: duplicate identity {key}")
        result[key] = row
    return result


def _load_protocol_output(
    protocol: Mapping[str, Any], name: str
) -> tuple[Path, list[dict[str, Any]]]:
    identity = (protocol.get("outputs") or {}).get(name)
    if not isinstance(identity, Mapping):
        raise ValueError(f"reconciliation protocol does not bind outputs.{name}")
    path = _resolve_bound_file(identity, label=f"reconciliation {name}")
    return path, _read_jsonl(path)


def _same_identity(
    left: Mapping[str, Any], right: Mapping[str, Any], *, label: str
) -> None:
    for field in (
        "dataset",
        "qid",
        "question",
        "question_sha256",
        "family_sha256",
    ):
        if str(left.get(field) or "") != str(right.get(field) or ""):
            raise ValueError(f"{label}: identity mismatch at {field}")


def _validate_prior_release(
    protocol: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any], list[dict[str, Any]]]:
    inputs = protocol.get("inputs") or {}
    report_identity = inputs.get("completed_retrieval_report")
    contexts_identity = inputs.get("completed_retrieval_contexts")
    if not isinstance(report_identity, Mapping) or not isinstance(
        contexts_identity, Mapping
    ):
        raise ValueError("reconciliation protocol does not bind prior retrieval release")
    report_path = _resolve_bound_file(
        report_identity, label="completed retrieval report"
    )
    contexts_path = _resolve_bound_file(
        contexts_identity, label="completed retrieval contexts"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    attestation = report.get("backend_attestation") or {}
    combined = (report.get("outputs") or {}).get("combined") or {}
    if (
        report.get("status") != STATUS
        or report.get("retrieval") != RETRIEVAL_STACK
        or not all(bool(value) for value in (report.get("gates") or {}).values())
        or attestation.get("mode") != "cross_encoder"
        or attestation.get("requested_backend") != "bge-reranker-v2-m3"
        or attestation.get("load_succeeded") is not True
        or attestation.get("backend_fallback") is not False
        or str(combined.get("sha256") or "") != _sha256(contexts_path)
    ):
        raise ValueError("prior canonical retrieval release attestation failed")
    rows = _read_jsonl(contexts_path)
    if len(rows) != 818 or Counter(row.get("dataset") for row in rows) != Counter(
        {"hotpotqa": 417, "musique": 401}
    ):
        raise ValueError("prior retrieval release is not the frozen H417/M401=818")
    return report_path, contexts_path, report, rows


def _backend_identity(attestation: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        attestation.get("requested_backend"),
        ((attestation.get("config") or {}).get("sha256")),
        ((attestation.get("weights") or {}).get("sha256")),
        ((attestation.get("tokenizer") or {}).get("sha256")),
    )


def _validate_reused_context(
    context: Mapping[str, Any],
    binding: Mapping[str, Any],
    requirement: Mapping[str, Any],
) -> dict[str, Any]:
    key = question_key(str(requirement["dataset"]), str(requirement["qid"]))
    _same_identity(context, requirement, label=f"{key} prior/requirement")
    _same_identity(binding, requirement, label=f"{key} binding/requirement")
    if binding.get("reuse_existing_context") is not True:
        raise ValueError(f"{key}: reuse binding is not affirmative")
    if context.get("gold_access") is not False or binding.get("gold_access") is not False:
        raise ValueError(f"{key}: reused context/binding must be Gold-free")
    if FORBIDDEN_FIELDS & set(context) or FORBIDDEN_FIELDS & set(binding):
        raise ValueError(f"{key}: reused context/binding contains forbidden fields")
    passages = context.get("passages")
    if not _passages_valid(passages):
        raise ValueError(f"{key}: reused context lacks ten safe passages")
    actual_hash = _sha_json(passages)
    if (
        str(context.get("passages_sha256") or "") != actual_hash
        or str(binding.get("passages_sha256") or "") != actual_hash
    ):
        raise ValueError(f"{key}: reused passage hash drifted")
    if context.get("retrieval_source") != RETRIEVAL_STACK:
        raise ValueError(f"{key}: reused retrieval stack drifted")
    return dict(context)


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
    """Retrieve the 11-row delta and emit the exact reconciled release."""

    protocol_path = protocol_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite retrieval output: {output_dir}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not str(experiment_id).strip():
        raise ValueError("a nonempty Experiment ID is required")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version") != RECONCILIATION_SCHEMA
        or protocol.get("status") != RECONCILIATION_STATUS
        or not all(bool(value) for value in (protocol.get("gates") or {}).values())
    ):
        raise ValueError("H/M reconciliation protocol schema/status/gates failed")
    reconciliation_manifest_path = protocol_path.parent / "manifest.json"
    if not reconciliation_manifest_path.is_file():
        raise FileNotFoundError(reconciliation_manifest_path)
    reconciliation_manifest = json.loads(
        reconciliation_manifest_path.read_text(encoding="utf-8")
    )
    if (
        reconciliation_manifest.get("status") != RECONCILIATION_STATUS
        or (reconciliation_manifest.get("run") or {}).get("protocol_sha256")
        != _sha256(protocol_path)
    ):
        raise ValueError("H/M reconciliation manifest/protocol binding drifted")
    frozen = protocol.get("retrieval_reconciliation") or {}
    if (
        int(frozen.get("required_for_reconciled_population", -1)) != 823
        or int(frozen.get("reused", -1)) != 812
        or int(frozen.get("new_requests", -1)) != 11
        or int(frozen.get("retired", -1)) != 6
        or frozen.get("retrieval_executed_in_this_stage") is not False
    ):
        raise ValueError("H/M reconciliation does not freeze exact 812/11/6 counts")

    requirement_path, requirement_rows = _load_protocol_output(
        protocol, "retrieval_requirements"
    )
    reused_path, reused_rows = _load_protocol_output(
        protocol, "reused_context_bindings"
    )
    new_path, new_rows = _load_protocol_output(protocol, "new_retrieval_requests")
    retired_path, retired_rows = _load_protocol_output(protocol, "retired_contexts")
    _validate_requests(requirement_rows, EXPECTED_COUNTS)
    _validate_requests(new_rows, {"musique": 11})
    requirements = _index(requirement_rows, label="retrieval requirements")
    reused = _index(reused_rows, label="reused context bindings")
    new = _index(new_rows, label="new retrieval requests")
    retired = _index(retired_rows, label="retired contexts")
    if (
        len(requirements) != 823
        or len(reused) != 812
        or len(new) != 11
        or len(retired) != 6
        or set(reused) & set(new)
        or set(reused) | set(new) != set(requirements)
    ):
        raise ValueError("reconciliation identity sets are not exact 812+11=823")

    prior_report_path, prior_contexts_path, prior_report, prior_rows = (
        _validate_prior_release(protocol)
    )
    prior = _index(prior_rows, label="prior retrieval contexts")
    if set(prior) != set(reused) | set(retired) or set(reused) & set(retired):
        raise ValueError("prior contexts are not exact reused812 + retired6")
    reused_contexts = {
        key: _validate_reused_context(prior[key], binding, requirements[key])
        for key, binding in reused.items()
    }
    for key, row in retired.items():
        _same_identity(prior[key], row, label=f"{key} retired/prior")
        if row.get("gold_access") is not False:
            raise ValueError(f"{key}: retired identity is not Gold-free")

    if retrieval_fn is None:
        # Fail before creating the release if the exact local BGE backend is
        # unavailable.  The canonical runner is called for the 11 new rows only.
        attestation = _attest_cross_encoder_backend()
        runner = materialize_dataset
    else:
        runner = retrieval_fn
        attestation = dict(
            backend_attestation
            or {
                "mode": "injected_test_double",
                "requested_backend": "bge-reranker-v2-m3",
                "load_succeeded": True,
                "backend_fallback": False,
            }
        )
    if attestation.get("backend_fallback") is not False:
        raise RuntimeError(f"reranker fallback is forbidden: {attestation}")
    raw_new_contexts = runner("musique", list(new.values()), batch_size)
    validated_new_rows = _validate_contexts(list(new.values()), raw_new_contexts)
    new_contexts = _index(validated_new_rows, label="new retrieval contexts")
    if set(new_contexts) != set(new):
        raise ValueError("new retrieval output identity join is not exact")

    combined_by_key = {**reused_contexts, **new_contexts}
    ordered = [combined_by_key[str(row["question_key"])] for row in requirement_rows]
    contexts = _validate_contexts(requirement_rows, ordered)
    contexts_by_dataset = {
        dataset: [row for row in contexts if row["dataset"] == dataset]
        for dataset in DATASETS
    }
    output_counts = Counter(row["dataset"] for row in contexts)
    prior_attestation = prior_report["backend_attestation"]
    backend_matches_prior = _backend_identity(attestation) == _backend_identity(
        prior_attestation
    )
    # Injected test doubles do not carry real asset hashes; production must.
    if retrieval_fn is not None and attestation.get("mode") == "injected_test_double":
        backend_matches_prior = True
    gates = {
        "reconciliation_protocol_frozen": True,
        "reconciliation_protocol_hash_bound": True,
        "reconciliation_manifest_hash_bound": True,
        "requirements_h417_m406_exact": output_counts == Counter(EXPECTED_COUNTS),
        "reused_contexts_812_exact": len(reused_contexts) == 812,
        "newly_retrieved_contexts_11_exact": len(new_contexts) == 11,
        "retired_contexts_6_excluded": len(retired) == 6
        and not (set(retired) & set(combined_by_key)),
        "identity_join_rate_1": set(combined_by_key) == set(requirements),
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
        "reused_passage_hashes_unchanged": all(
            _sha_json(reused_contexts[key]["passages"])
            == str(reused[key]["passages_sha256"])
            for key in reused
        ),
        "new_backend_matches_reused_backend": backend_matches_prior,
        "backend_fallback_false": attestation.get("backend_fallback") is False,
        "backend_load_attested": attestation.get("load_succeeded") is True,
        "training_not_started": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"reconciled H/M retrieval gates failed: {gates}")

    output_dir.mkdir(parents=True, exist_ok=False)
    output_paths = {
        "hotpotqa": output_dir / "hotpotqa.retrieval_contexts.jsonl",
        "musique": output_dir / "musique.retrieval_contexts.jsonl",
        "newly_retrieved": output_dir / "newly_retrieved_contexts.jsonl",
        "combined": output_dir / "retrieval_contexts.jsonl",
    }
    for dataset in DATASETS:
        _write_jsonl(output_paths[dataset], contexts_by_dataset[dataset])
    _write_jsonl(output_paths["newly_retrieved"], validated_new_rows)
    _write_jsonl(output_paths["combined"], contexts)

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment_id": str(experiment_id).strip(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "counts": {
            "requests_total": len(requirements),
            "contexts_total": len(contexts),
            "by_dataset": dict(output_counts),
        },
        "reconciliation": {
            "reused_contexts": len(reused_contexts),
            "newly_retrieved_contexts": len(new_contexts),
            "retired_contexts": len(retired),
            "prior_contexts": len(prior),
        },
        "retrieval": RETRIEVAL_STACK,
        "backend_attestation": attestation,
        "reused_backend_attestation": prior_attestation,
        "gates": gates,
        "scientific_boundary": {
            "identity_selection_performed": False,
            "gold_fields_read_or_written": False,
            "raw_dataset_read": False,
            "only_frozen_new_requests_retrieved": True,
            "kg_constructed": False,
            "model_updated": False,
            "training_started": False,
            "prior_release_modified_or_deleted": False,
        },
        "inputs": {
            "hm_reconciliation_protocol": _identity(protocol_path),
            "hm_reconciliation_manifest": _identity(
                reconciliation_manifest_path
            ),
            "retrieval_requirements": _identity(requirement_path),
            "reused_context_bindings": _identity(reused_path),
            "new_retrieval_requests": _identity(new_path),
            "retired_contexts": _identity(retired_path),
            "prior_retrieval_report": _identity(prior_report_path),
            "prior_retrieval_contexts": _identity(prior_contexts_path),
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
            "phase": "mixed3_v4_reconciled_expansion_retrieval",
            "experiment_id": report["experiment_id"],
            "hm_reconciliation_protocol": report["inputs"][
                "hm_reconciliation_protocol"
            ],
            "reconciliation": report["reconciliation"],
            "outputs": {
                **report["outputs"],
                "report": _identity(report_path),
            },
            "retrieval": RETRIEVAL_STACK,
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
    parser.add_argument("--batch-size", type=int, default=16)
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
