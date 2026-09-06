#!/usr/bin/env python3
"""Build one leakage-controlled, strict 2Wiki ProofKG candidate supply.

The release merges the previously generated automatic train ProofKG with a
fresh question-only extension runtime.  It never copies the Gold-derived
source steps or source KG.  A row enters the output only after the same strict
trajectory Graph gate used by the PPO data release returns ``m_graph=1``.

This is a candidate supply, not the final mixed-PPO selection.  The downstream
answer-free v4 freeze selects an exact 800-row, four-stratum subset from the
explicit ``proof_candidates.jsonl`` wrapper output.  The wrapper keeps
question-type selection metadata outside the canonical question-KG record.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_key, question_sha256, validate_question_kg_record
from kgproweight.reward.trajectory_source_gate import make_source_gate_record
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.prepare.audit_auto1500_v4_clean_reproducibility import (
    DEFAULT_PROTECTED_LEDGER_DIR,
    PROTECTED_LEDGER_SCHEMA_VERSION,
    QUESTION_EDGE_SCHEMA_VERSION,
    QUESTION_ROOT_SCHEMA_VERSION,
    SCHEMA_VERSION as OLD_ATTESTATION_RELEASE_SCHEMA_VERSION,
    STATUS as OLD_ATTESTATION_RELEASE_STATUS,
    canonical_sha256,
    validate_protected_ledger_release,
)


SCHEMA_VERSION = "2wiki-unified-proofkg-candidate-supply-v2"
STATUS = "COMPLETE_STRICT_CANDIDATE_SUPPLY_NOT_SELECTED_NOT_TRAINED"
QTYPES = ("bridge_comparison", "comparison", "compositional", "inference")
CANDIDATE_SCHEMA_VERSION = "2wiki-unified-proofkg-candidate-wrapper-v2"
CANONICAL_RETRIEVAL_STACK = (
    "E5@100+BM25@100->RRF60@50->bge-reranker-v2-m3@10->pack3860"
)
EXPECTED_RETRIEVAL_STATUS = (
    "COMPLETE_ANSWER_FREE_2WIKI_RESERVE_RETRIEVAL_NOT_TRAINED"
)
FORBIDDEN_RETRIEVAL_FIELDS = {
    "answer",
    "answers",
    "gold_answer",
    "golden_answers",
    "supporting_facts",
    "support",
    "decomposition",
    "question_decomposition",
    "evidence",
    "reasoning",
    "sp",
}
CLEAN_REEXECUTION_SCHEMA_VERSION = "old-auto1500-clean-proofkg-reexecution-v1"
COMPLETE_PROTECTED_LEDGER_VERSION = PROTECTED_LEDGER_SCHEMA_VERSION
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_OLD_ATTESTATION_DIR = Path(
    "outputs/audits/auto1500_clean_edge_root_attestation_v4_complete_ledger"
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
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _json_sha256(value: Any) -> str:
    # Match the canonical retrieval materializer's passages hash contract.
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _index(rows: Iterable[Mapping[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in rows:
        row = dict(value)
        dataset = str(row.get("dataset") or "2wikimultihopqa").strip().lower()
        qid = str(row.get("qid") or row.get("source_qid") or "").strip()
        key = str(row.get("question_key") or question_key(dataset, qid))
        if not qid or key in result:
            raise ValueError(f"{label}: empty/duplicate identity {key!r}")
        result[key] = row
    return result


def _attestation_index(
    rows: Iterable[Mapping[str, Any]], *, label: str, schema_version: str
) -> dict[str, dict[str, Any]]:
    result = _index(rows, label=label)
    for key, row in result.items():
        if row.get("schema_version") != schema_version:
            raise ValueError(f"{label}: unexpected schema for {key}")
    return result


def _old_attestation_admission(
    *,
    trace: Mapping[str, Any],
    edge: Mapping[str, Any] | None,
    root: Mapping[str, Any] | None,
    cutoff: str,
) -> tuple[dict[str, Any] | None, str]:
    """Validate an old trace against two independently frozen attestations."""

    if edge is None:
        return None, "missing_edge_attestation"
    if root is None:
        return None, "missing_root_attestation"
    key = str(trace.get("question_key") or "")
    qhash = str(trace.get("question_sha256") or "")
    plan_sha = canonical_sha256(trace.get("query_plan") or {})
    execution_sha = canonical_sha256(trace.get("execution") or {})
    runtime_sha = canonical_sha256(trace)
    identity = {
        "question_key": key,
        "dataset": str(trace.get("dataset") or "").strip().lower(),
        "qid": str(trace.get("qid") or "").strip(),
        "question_sha256": qhash,
        "old_runtime_sha256": runtime_sha,
        "old_plan_sha256": plan_sha,
        "old_execution_sha256": execution_sha,
        "historical_cutoff": cutoff,
    }
    for label, row in (("edge", edge), ("root", root)):
        for field, expected in identity.items():
            if row.get(field) != expected:
                return None, f"{label}_attestation_{field}_mismatch"
        if row.get("gold_access") is not False:
            return None, f"{label}_attestation_gold_boundary"
    if edge.get("kg_sha256") != canonical_sha256(trace.get("kg_subgraph") or []):
        return None, "edge_attestation_kg_sha256_mismatch"
    edge_count = int(edge.get("executed_edge_count") or 0)
    if (
        edge.get("all_executed_edges_independently_reproduced") is not True
        or edge_count <= 0
        or int(edge.get("independently_reproduced_edge_count") or -1) != edge_count
        or not _HEX64.fullmatch(str(edge.get("edge_details_sha256") or ""))
    ):
        return None, "edge_attestation_not_complete"
    root_count = int(root.get("root_anchor_count") or 0)
    if (
        root.get("all_root_anchors_independently_attested") is not True
        or root_count <= 0
        or int(root.get("independently_attested_root_count") or -1) != root_count
        or not _HEX64.fullmatch(str(root.get("root_details_sha256") or ""))
    ):
        return None, "root_attestation_not_complete"
    return {
        "mode": "independent_edge_plus_root_attestation",
        "edge_attestation_sha256": canonical_sha256(edge),
        "root_attestation_sha256": canonical_sha256(root),
        "old_runtime_sha256": runtime_sha,
        "old_plan_sha256": plan_sha,
        "old_execution_sha256": execution_sha,
    }, "eligible"


def _clean_reexecution_admission(
    trace: Mapping[str, Any], *, cutoff: str
) -> tuple[dict[str, Any] | None, str]:
    """Require a fully bound clean-runtime contract, not a boolean self-label."""

    provenance = trace.get("provenance") or {}
    contract = provenance.get("clean_reexecution_contract")
    if not isinstance(contract, Mapping):
        return None, "clean_runtime_contract_missing"
    required = {
        "schema_version": CLEAN_REEXECUTION_SCHEMA_VERSION,
        "root_resolver_input": "question_and_root_anchor_surfaces_only",
        "expected_old_qids_used_as_resolver_targets": False,
        "old_v2_store_used": False,
        "gold_access": False,
        "historical_cutoff": cutoff,
    }
    for field, expected in required.items():
        if contract.get(field) != expected:
            return None, f"clean_runtime_contract_{field}_mismatch"
    for field in (
        "clean_store_manifest_sha256",
        "historical_cache_sha256",
        "resolver_code_sha256",
        "executor_code_sha256",
    ):
        if not _HEX64.fullmatch(str(contract.get(field) or "")):
            return None, f"clean_runtime_contract_{field}_missing"
    if provenance.get("gold_access") is not False:
        return None, "clean_runtime_provenance_gold_boundary"
    return {
        "mode": "new_clean_reexecuted_runtime",
        "clean_reexecution_contract_sha256": canonical_sha256(contract),
        "clean_runtime_sha256": canonical_sha256(trace),
    }, "eligible"


def _selected_raw(path: Path, *, qids: set[str]) -> list[dict[str, Any]]:
    """Stream only selected 2Wiki raw rows instead of loading the 689 MB file."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("id") or row.get("qid") or "").strip()
            if qid not in qids:
                continue
            if qid in seen:
                raise ValueError(f"raw source duplicate qid at {path}:{line_number}: {qid}")
            seen.add(qid)
            # Project immediately onto the only fields authorised downstream.
            # The official row also contains support annotations and passages;
            # those must never enter the unified Proof supply.
            rows.append(
                {
                    "id": qid,
                    "question": str(row.get("question") or "").strip(),
                    "golden_answers": list(row.get("golden_answers") or []),
                }
            )
    missing = sorted(qids - seen)
    if missing:
        raise ValueError(f"raw source misses {len(missing)} selected qids: {missing[:5]}")
    return rows


def _raw_index(rows: Iterable[Mapping[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in rows:
        row = dict(value)
        qid = str(row.get("id") or row.get("qid") or "").strip()
        question = str(row.get("question") or "").strip()
        key = question_key("2wikimultihopqa", qid)
        if not qid or not question or key in result:
            raise ValueError(f"{label}: empty/duplicate identity {key!r}")
        result[key] = row
    return result


def _fallback_source_from_raw_retrieval(
    *,
    cohort: Mapping[str, Any],
    raw: Mapping[str, Any],
    retrieval: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the minimal rollout source for a new official-train identity."""

    dataset = str(cohort.get("dataset") or "").strip().lower()
    qid = str(cohort.get("qid") or "").strip()
    question = str(cohort.get("question") or "").strip()
    qhash = question_sha256(question)
    key = question_key(dataset, qid)
    raw_qid = str(raw.get("id") or raw.get("qid") or "").strip()
    if (
        dataset != "2wikimultihopqa"
        or raw_qid != qid
        or str(raw.get("question") or "").strip() != question
    ):
        raise ValueError(f"raw/cohort identity mismatch: {key}")
    for field, expected in (
        ("question_key", key),
        ("dataset", dataset),
        ("qid", qid),
        ("question", question),
        ("question_sha256", qhash),
        ("gold_access", False),
    ):
        if retrieval.get(field) != expected:
            raise ValueError(f"retrieval/cohort mismatch at {field}: {key}")
    if FORBIDDEN_RETRIEVAL_FIELDS & set(retrieval):
        raise ValueError(f"retrieval record contains forbidden fields: {key}")
    if retrieval.get("retrieval_source") != CANONICAL_RETRIEVAL_STACK:
        raise ValueError(f"retrieval backend drift: {key}")
    passages = [dict(row) for row in (retrieval.get("passages") or [])]
    if (
        len(passages) != 10
        or any(
            not str(row.get("id") or "").strip()
            or not str(row.get("source") or "").strip()
            or not str(row.get("contents") or "").strip()
            or bool(FORBIDDEN_RETRIEVAL_FIELDS & set(row))
            for row in passages
        )
        or retrieval.get("passages_sha256") != _json_sha256(passages)
    ):
        raise ValueError(f"retrieval ten-passage/hash contract failed: {key}")
    aliases = [
        str(value).strip()
        for value in (raw.get("golden_answers") or [])
        if str(value).strip()
    ]
    if not aliases:
        raise ValueError(f"raw outcome label missing: {key}")
    return {
        "dataset": dataset,
        "qid": qid,
        "question": question,
        "answer": aliases[0],
        "retrieved_passages": passages,
        "metadata": {
            "gold_answer": aliases[0],
            "question_type": str(cohort.get("question_type") or ""),
            "train_only": True,
            "passage_source": "canonical_wiki18_rrf_reranked_top10",
            "gold_use": "outcome_reward_label_only",
        },
    }


def _uses_retrieval_fallback(
    cohort: Mapping[str, Any], *, source_available: bool
) -> bool:
    """Resolve the frozen extension source route without silent substitution.

    The combined350 release has two explicit parents.  v1b rows must keep their
    previously frozen silver passages; reserve rows were selected from official
    train beyond that curriculum and must use the separately versioned canonical
    retrieval release.  The final branch is retained solely for legacy unit/API
    callers whose rows predate ``source_role``.
    """

    role = str(cohort.get("source_role") or "").strip()
    schema = str(cohort.get("schema_version") or "").strip()
    if role == "automatic_proofkg_extension_reserve_candidate" or schema == (
        "automatic-proofkg-extension-reserve-question-only-v1"
    ):
        return True
    if role == "automatic_proofkg_extension_candidate" or schema == (
        "automatic-proofkg-extension-question-only-v1"
    ):
        if not source_available:
            raise ValueError(
                "frozen v1b row is missing from --new-source; canonical retrieval "
                "fallback is reserved for reserve-v1 rows"
            )
        return False
    return not source_available


def _validate_retrieval_release(directory: Path) -> tuple[Path, list[Path]]:
    """Validate the append-only reserve retrieval release and return its data."""

    if not directory.is_dir():
        raise ValueError(
            "--new-retrieval must be a versioned directory containing "
            "retrieval_contexts.jsonl, report.json, and manifest.json"
        )
    contexts = directory / "retrieval_contexts.jsonl"
    report_path = directory / "report.json"
    manifest_path = directory / "manifest.json"
    for path in (contexts, report_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != EXPECTED_RETRIEVAL_STATUS:
        raise ValueError("reserve retrieval release has an unexpected status")
    if report.get("retrieval") != CANONICAL_RETRIEVAL_STACK:
        raise ValueError("reserve retrieval release has a noncanonical stack")
    if not all(bool(value) for value in (report.get("gates") or {}).values()):
        raise ValueError("reserve retrieval release contains a failed gate")
    attestation = report.get("backend_attestation")
    if not isinstance(attestation, Mapping) or not (
        attestation.get("mode") == "cross_encoder"
        and attestation.get("requested_backend") == "bge-reranker-v2-m3"
        and attestation.get("load_succeeded") is True
        and attestation.get("backend_fallback") is False
    ):
        raise ValueError("reserve retrieval release lacks exact BGE attestation")
    combined = (report.get("outputs") or {}).get("retrieval_contexts")
    if not isinstance(combined, Mapping) or str(combined.get("sha256") or "") != _sha256(
        contexts
    ):
        raise ValueError("reserve retrieval report/context SHA256 mismatch")
    return contexts, [report_path, manifest_path]


def _validate_old_attestation_release(
    directory: Path,
    *,
    protected_ledger_binding: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Path, Path, list[Path]]:
    """Bind old trace reuse to the complete append-only audit release."""

    if not directory.is_dir():
        raise ValueError("--old-attestation-dir must be a versioned audit directory")
    edge_path = directory / "question_edge_attestations.jsonl"
    root_path = directory / "question_root_attestations.jsonl"
    report_path = directory / "report.json"
    manifest_path = directory / "manifest.json"
    for path in (edge_path, root_path, report_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != OLD_ATTESTATION_RELEASE_SCHEMA_VERSION
        or report.get("status") != OLD_ATTESTATION_RELEASE_STATUS
        or manifest.get("status") != OLD_ATTESTATION_RELEASE_STATUS
        or not all(bool(value) for value in (report.get("checks") or {}).values())
    ):
        raise ValueError("old attestation release status/schema/checks failed")
    ledger = report.get("protected_ledger") or {}
    if (
        ledger.get("version") != COMPLETE_PROTECTED_LEDGER_VERSION
        or ledger.get("complete") is not True
        or ledger.get("current_family_recomputed") is not True
    ):
        raise ValueError("old attestation release does not bind the complete protected ledger")
    if protected_ledger_binding is None:
        raise ValueError("old attestation validation requires the live protected-ledger binding")
    for name in ("ledger", "report", "manifest"):
        actual = ledger.get(name)
        expected = protected_ledger_binding.get(name)
        if (
            not isinstance(actual, Mapping)
            or not isinstance(expected, Mapping)
            or actual.get("sha256") != expected.get("sha256")
        ):
            raise ValueError(f"old attestation protected-ledger hash mismatch: {name}")
    counts = report.get("counts") or {}
    population = int(counts.get("protected_safe_strict_questions", -1))
    executed_edges = int(counts.get("executed_edges", -1))
    reproduced_edges = int(counts.get("independently_reproduced_edges", -1))
    edge_questions = int(counts.get("all_edges_attested_questions", -1))
    if (
        population <= 0
        or executed_edges <= 0
        or reproduced_edges != executed_edges
        or edge_questions != population
    ):
        raise ValueError("old attestation release frozen counts drifted")
    all_roots = int(counts.get("all_roots_attested_questions", -1))
    worklist = int(counts.get("reresolution_worklist_questions", -1))
    if all_roots < 0 or worklist < 0 or all_roots + worklist != population:
        raise ValueError("old attestation release root/worklist partition drifted")
    edge_rows = _read_jsonl(edge_path)
    root_rows = _read_jsonl(root_path)
    if (
        len(edge_rows) != population
        or len(root_rows) != population
        or {str(row.get("question_key") or "") for row in edge_rows}
        != {str(row.get("question_key") or "") for row in root_rows}
        or any(
            row.get("schema_version") != QUESTION_EDGE_SCHEMA_VERSION
            or row.get("all_executed_edges_independently_reproduced") is not True
            for row in edge_rows
        )
        or sum(
            row.get("all_root_anchors_independently_attested") is True
            for row in root_rows
        )
        != all_roots
    ):
        raise ValueError("old attestation release row population/content drifted")
    outputs = report.get("outputs") or {}
    for name, path in (
        ("question_edge_attestations", edge_path),
        ("question_root_attestations", root_path),
    ):
        identity = outputs.get(name)
        if not isinstance(identity, Mapping) or identity.get("sha256") != _sha256(path):
            raise ValueError(f"old attestation release hash mismatch: {name}")
    manifest_run = manifest.get("run") or {}
    manifest_report = manifest_run.get("report") or {}
    if (
        not isinstance(manifest_report, Mapping)
        or manifest_report.get("sha256") != _sha256(report_path)
        or manifest_run.get("training_started") is not False
    ):
        raise ValueError("old attestation release manifest/report binding drifted")
    return edge_path, root_path, [report_path, manifest_path]


def _blocked(paths: Sequence[Path]) -> tuple[set[str], set[str], set[str]]:
    qids: set[str] = set()
    hashes: set[str] = set()
    families: set[str] = set()
    for path in paths:
        for row in _read_jsonl(path):
            if str(row.get("dataset") or "2wikimultihopqa").strip().lower() != "2wikimultihopqa":
                continue
            qid = str(row.get("qid") or row.get("source_qid") or "").strip()
            question = str(row.get("question") or "").strip()
            if not qid or not question:
                raise ValueError(f"incomplete exclusion identity in {path}")
            qids.add(qid)
            hashes.add(question_sha256(question))
            families.add(family_sha256(question))
    return qids, hashes, families


def _attach_trace(
    *, base: Mapping[str, Any] | None, trace: Mapping[str, Any], cutoff: str,
    source_release: str, admission: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = dict(trace)
    if base is not None:
        for field in ("question_key", "dataset", "qid", "question", "question_sha256", "kg_subgraph", "query_plan"):
            if base.get(field) != trace.get(field):
                raise ValueError(f"base/runtime mismatch at {field}: {trace.get('question_key')}")
    validate_question_kg_record(record)
    provenance = dict(record.get("provenance") or {})
    provenance.update(
        {
            "historical_cutoff": cutoff,
            "unified_supply_version": SCHEMA_VERSION,
            "unified_source_release": source_release,
            "process_reward_eligible": True,
            "source_gold_steps_copied": 0,
            "old_trace_admission": dict(admission or {"mode": "not_old_source"}),
        }
    )
    record["provenance"] = provenance
    record["process_reward_eligible"] = True
    return record


def _clean_silver(
    *, source: Mapping[str, Any], record: Mapping[str, Any], qtype: str,
    source_release: str,
) -> dict[str, Any]:
    question = str(source.get("question") or "").strip()
    qid = str(source.get("qid") or "").strip()
    answer = str((source.get("metadata") or {}).get("gold_answer") or source.get("answer") or "").strip()
    passages = [dict(row) for row in (source.get("retrieved_passages") or [])]
    if not qid or not question or not answer:
        raise ValueError(f"source silver identity/outcome missing: {qid!r}")
    if len(passages) != 10 or any(not str(row.get("contents") or "").strip() for row in passages):
        raise ValueError(f"source silver does not have ten passages: {qid}")
    return {
        "qid": qid,
        "question": question,
        "answer": answer,
        "dataset": "2wikimultihopqa",
        "steps": [],
        "kg_subgraph": [list(triple) for triple in (record.get("kg_subgraph") or [])],
        "retrieved_passages": passages,
        "accepted": True,
        "metadata": {
            "gold_answer": answer,
            "question_type": qtype,
            "source_split": "train",
            "rollout_only": True,
            "source_gold_trace_removed": True,
            "source_gold_kg_removed": True,
            "evaluation_eligible": False,
            "family_version": FAMILY_VERSION,
            "family_sha256": family_sha256(question),
            "proof_source": source_release,
            "process_reward_eligible": True,
            "retrieved_passages_sha256": _json_sha256(passages),
            "gold_use": "outcome_reward_label_only",
        },
        "teacher_output": "",
        "teacher_model": "none_ppo_rollout_only",
    }


def build_supply(
    *, old_silver: Sequence[Mapping[str, Any]], old_records: Sequence[Mapping[str, Any]],
    old_runtime: Sequence[Mapping[str, Any]], new_source: Sequence[Mapping[str, Any]],
    new_cohort: Sequence[Mapping[str, Any]], new_runtime: Sequence[Mapping[str, Any]],
    blocked_qids: set[str], blocked_hashes: set[str], blocked_families: set[str],
    cutoff: str, new_raw: Sequence[Mapping[str, Any]] = (),
    new_retrieval: Sequence[Mapping[str, Any]] = (),
    old_edge_attestations: Sequence[Mapping[str, Any]] = (),
    old_root_attestations: Sequence[Mapping[str, Any]] = (),
    old_clean_runtime: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    old_silver_by_key = _index(old_silver, label="old silver")
    old_base_by_key = _index(old_records, label="old KG")
    old_runtime_by_key = _index(old_runtime, label="old runtime")
    old_edge_by_key = _attestation_index(
        old_edge_attestations,
        label="old edge attestation",
        schema_version=QUESTION_EDGE_SCHEMA_VERSION,
    )
    old_root_by_key = _attestation_index(
        old_root_attestations,
        label="old root attestation",
        schema_version=QUESTION_ROOT_SCHEMA_VERSION,
    )
    old_clean_by_key = _index(old_clean_runtime, label="old clean runtime")
    unknown_clean = set(old_clean_by_key) - set(old_silver_by_key)
    if unknown_clean:
        raise ValueError(f"clean runtime contains non-old identities: {sorted(unknown_clean)[:5]}")
    new_source_by_key = _index(
        (row for row in new_source if str(row.get("dataset")) == "2wikimultihopqa"),
        label="new source silver",
    )
    new_cohort_by_key = _index(new_cohort, label="new cohort")
    new_runtime_by_key = _index(new_runtime, label="new runtime")
    if set(new_runtime_by_key) != set(new_cohort_by_key):
        raise ValueError("new runtime/cohort identity join is not exact")
    new_raw_by_key = _raw_index(new_raw, label="new raw fallback")
    new_retrieval_by_key = _index(new_retrieval, label="new retrieval fallback")
    fallback_keys: set[str] = set()
    for key, cohort in new_cohort_by_key.items():
        dataset = str(cohort.get("dataset") or "").strip().lower()
        qid = str(cohort.get("qid") or "").strip()
        question = str(cohort.get("question") or "").strip()
        if (
            dataset != "2wikimultihopqa"
            or str(cohort.get("question_key") or key) != key
            or str(cohort.get("question_sha256") or "")
            != question_sha256(question)
            or cohort.get("gold_access") not in (None, False)
        ):
            raise ValueError(f"new cohort identity/gold boundary invalid: {key}")
        trace = new_runtime_by_key[key]
        for field, expected in (
            ("dataset", dataset),
            ("qid", qid),
            ("question", question),
            ("question_sha256", question_sha256(question)),
        ):
            if trace.get(field) != expected:
                raise ValueError(f"new runtime/cohort mismatch at {field}: {key}")
        if _uses_retrieval_fallback(
            cohort, source_available=key in new_source_by_key
        ):
            fallback_keys.add(key)
    if set(new_raw_by_key) != fallback_keys or set(new_retrieval_by_key) != fallback_keys:
        raise ValueError(
            "raw/retrieval fallback must exactly cover new cohort identities absent "
            f"from new-source silver: need={len(fallback_keys)} "
            f"raw={len(new_raw_by_key)} retrieval={len(new_retrieval_by_key)}"
        )

    candidates: list[tuple[str, str, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any] | None, str]] = []
    for key, source in old_silver_by_key.items():
        if key not in old_base_by_key or key not in old_runtime_by_key:
            raise ValueError(f"old supply join miss: {key}")
        candidates.append(("automatic_proofkg_train_k4_v1", str((source.get("metadata") or {}).get("question_type") or ""), source, old_runtime_by_key[key], old_base_by_key[key], "old"))
    for key, cohort in new_cohort_by_key.items():
        if key in fallback_keys:
            source = _fallback_source_from_raw_retrieval(
                cohort=cohort,
                raw=new_raw_by_key[key],
                retrieval=new_retrieval_by_key[key],
            )
            source_release = "automatic_proofkg_extension_reserve_v1"
        else:
            source = new_source_by_key[key]
            source_release = "automatic_proofkg_extension_v1"
        candidates.append((source_release, str(cohort.get("question_type") or ""), source, new_runtime_by_key[key], None, "new"))

    silver_out: list[dict[str, Any]] = []
    records_out: list[dict[str, Any]] = []
    gates_out: list[dict[str, Any]] = []
    excluded = Counter()
    source_counts = Counter()
    type_counts = Counter()
    admission_counts = Counter()
    seen_keys: set[str] = set()
    for source_release, qtype, source, trace, base, origin in candidates:
        qid = str(source.get("qid") or "").strip()
        question = str(source.get("question") or "").strip()
        qhash = question_sha256(question)
        family = family_sha256(question)
        if qid in blocked_qids:
            excluded["qid"] += 1
            continue
        if qhash in blocked_hashes:
            excluded["question_sha256"] += 1
            continue
        if family in blocked_families:
            excluded["family_sha256"] += 1
            continue
        if qtype not in QTYPES:
            excluded["unsupported_question_type"] += 1
            continue
        key = question_key("2wikimultihopqa", qid)
        admission: dict[str, Any] | None = None
        if origin == "old":
            clean_trace = old_clean_by_key.get(key)
            if clean_trace is not None:
                admission, reason = _clean_reexecution_admission(
                    clean_trace, cutoff=cutoff
                )
                if admission is None:
                    excluded[f"old_admission:{reason}"] += 1
                    continue
                trace = clean_trace
                # A clean re-execution is intentionally allowed to differ
                # from the old question-KG/runtime graph.
                base = None
            else:
                admission, reason = _old_attestation_admission(
                    trace=trace,
                    edge=old_edge_by_key.get(key),
                    root=old_root_by_key.get(key),
                    cutoff=cutoff,
                )
                if admission is None:
                    excluded[f"old_admission:{reason}"] += 1
                    continue
        if key in seen_keys:
            raise ValueError(f"duplicate unified candidate: {key}")
        if str(trace.get("question") or "").strip() != question or str(trace.get("question_sha256") or "") != qhash:
            raise ValueError(f"source/runtime identity mismatch: {key}")
        record = _attach_trace(
            base=base,
            trace=trace,
            cutoff=cutoff,
            source_release=source_release,
            admission=admission,
        )
        gate = make_source_gate_record(
            record,
            dataset="2wikimultihopqa",
            qid=qid,
            question=question,
            text_evidence_available=True,
            historical_cutoff=cutoff,
        )
        if gate["m_graph"] != 1:
            excluded[f"strict_gate:{gate['routing_reason']}"] += 1
            continue
        seen_keys.add(key)
        records_out.append(record)
        gates_out.append(gate)
        silver_out.append(_clean_silver(source=source, record=record, qtype=qtype, source_release=source_release))
        source_counts[source_release] += 1
        type_counts[qtype] += 1
        admission_counts[str((admission or {}).get("mode") or "new_source_not_applicable")] += 1
    order = sorted(range(len(silver_out)), key=lambda i: (silver_out[i]["metadata"]["question_type"], silver_out[i]["qid"]))
    silver_out = [silver_out[i] for i in order]
    records_out = [records_out[i] for i in order]
    gates_out = [gates_out[i] for i in order]
    return silver_out, records_out, gates_out, {
        "eligible_by_source": dict(sorted(source_counts.items())),
        "eligible_by_question_type": dict(sorted(type_counts.items())),
        "excluded": dict(sorted(excluded.items())),
        "old_trace_admission_modes": dict(sorted(admission_counts.items())),
    }


def build_candidate_wrappers(
    silver_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the answer-free interface consumed by the v4 protocol freeze.

    ``question_kg_records.jsonl`` intentionally remains a canonical runtime
    artifact and therefore does not carry sampling strata.  The v4 freezer,
    however, must know the frozen 2Wiki question type while re-running the hard
    Graph gate.  This wrapper binds those two pieces without copying an answer,
    source reasoning step, or source Gold KG.
    """

    if not (len(silver_rows) == len(records) == len(gates)):
        raise ValueError("candidate wrapper inputs do not have an exact row join")
    wrappers: list[dict[str, Any]] = []
    for silver, record, gate in zip(silver_rows, records, gates):
        dataset = str(record.get("dataset") or "").strip().lower()
        qid = str(record.get("qid") or "").strip()
        key = question_key(dataset, qid)
        if (
            dataset != "2wikimultihopqa"
            or str(gate.get("question_key") or "") != key
            or str(silver.get("dataset") or "").strip().lower() != dataset
            or str(silver.get("qid") or "").strip() != qid
            or str(silver.get("question") or "").strip()
            != str(record.get("question") or "").strip()
            or gate.get("m_graph") != 1
        ):
            raise ValueError(f"candidate wrapper identity/gate mismatch: {key}")
        qtype = str((silver.get("metadata") or {}).get("question_type") or "").strip()
        if qtype not in QTYPES:
            raise ValueError(f"candidate wrapper has invalid question type: {qtype!r}")
        question = str(record["question"])
        wrappers.append(
            {
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "question_key": key,
                "dataset": dataset,
                "qid": qid,
                "question": question,
                "question_sha256": question_sha256(question),
                "family_version": FAMILY_VERSION,
                "family_sha256": family_sha256(question),
                "question_type": qtype,
                "proof_passages_sha256": str(
                    (silver.get("metadata") or {}).get(
                        "retrieved_passages_sha256"
                    )
                ),
                "question_kg_record": dict(record),
                "gold_access": False,
                "evaluation_eligible": False,
            }
        )
    return wrappers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-silver", type=Path, default=Path("data/silver_data/automatic_proofkg_2wiki_train_k4_v1/silver_train.jsonl"))
    parser.add_argument("--old-records", type=Path, default=Path("data/silver_data/automatic_proofkg_2wiki_train_k4_v1/question_kg_records.jsonl"))
    parser.add_argument("--old-runtime", type=Path, default=Path("outputs/audits/automatic_proofkg_2wiki_train_k4_v1_n1500_seed42_historical_stage3_runtime/runtime_details.jsonl"))
    parser.add_argument(
        "--old-attestation-dir",
        type=Path,
        default=DEFAULT_OLD_ATTESTATION_DIR,
        help="Append-only independent edge/root attestation release.",
    )
    parser.add_argument(
        "--old-clean-runtime",
        type=Path,
        default=None,
        help=(
            "Optional newly re-executed old-runtime subset. Each row must bind the "
            "clean-reexecution contract and may replace a failed old root attestation."
        ),
    )
    parser.add_argument("--new-source", type=Path, default=Path("data/silver_data/proofkg_curriculum_mix_v1_n8000_seed42_20260829/silver_curriculum.jsonl"))
    parser.add_argument("--new-cohort", type=Path, required=True)
    parser.add_argument("--new-runtime", type=Path, required=True)
    parser.add_argument(
        "--new-raw",
        type=Path,
        default=Path("data/2wikimultihopqa/train.jsonl"),
        help="Official raw train source used only for missing outcome labels.",
    )
    parser.add_argument(
        "--new-retrieval",
        type=Path,
        default=None,
        help="Versioned canonical retrieval directory for reserve-v1 cohort rows.",
    )
    parser.add_argument(
        "--protected-ledger-dir",
        type=Path,
        default=DEFAULT_PROTECTED_LEDGER_DIR,
        help="Complete versioned identity ledger; hand-maintained exclusions are forbidden.",
    )
    parser.add_argument("--cutoff", default="2020-12-09T23:59:59Z")
    parser.add_argument("--min-total", type=int, default=800)
    parser.add_argument("--min-per-type", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite versioned output: {args.output_dir}")
    ledger_path, ledger_report_path, ledger_manifest_path, _ledger_report = (
        validate_protected_ledger_release(args.protected_ledger_dir)
    )
    protected_ledger_binding = {
        "ledger": _identity(ledger_path),
        "report": _identity(ledger_report_path),
        "manifest": _identity(ledger_manifest_path),
    }
    exclusions = [ledger_path]
    old_edge_path, old_root_path, old_attestation_metadata = (
        _validate_old_attestation_release(
            args.old_attestation_dir,
            protected_ledger_binding=protected_ledger_binding,
        )
    )
    new_cohort_rows = _read_jsonl(args.new_cohort)
    new_source_rows = _read_jsonl(args.new_source)
    cohort_index = _index(new_cohort_rows, label="new cohort preflight")
    source_index = _index(
        (
            row
            for row in new_source_rows
            if str(row.get("dataset")) == "2wikimultihopqa"
        ),
        label="new source preflight",
    )
    fallback_keys = {
        key
        for key, row in cohort_index.items()
        if _uses_retrieval_fallback(row, source_available=key in source_index)
    }
    extra_inputs: list[Path] = []
    new_raw_rows: list[dict[str, Any]] = []
    new_retrieval_rows: list[dict[str, Any]] = []
    if fallback_keys:
        if args.new_retrieval is None:
            raise ValueError(
                f"{len(fallback_keys)} new cohort rows lack source passages; "
                "--new-retrieval is required"
            )
        retrieval_path, retrieval_metadata = _validate_retrieval_release(
            args.new_retrieval
        )
        extra_inputs = [args.new_raw, retrieval_path, *retrieval_metadata]
        for path in extra_inputs:
            if not path.is_file():
                raise FileNotFoundError(path)
        fallback_qids = {key.split("::", 1)[1] for key in fallback_keys}
        new_raw_rows = _selected_raw(args.new_raw, qids=fallback_qids)
        new_retrieval_rows = _read_jsonl(retrieval_path)
    input_paths = [
        args.old_silver,
        args.old_records,
        args.old_runtime,
        old_edge_path,
        old_root_path,
        *old_attestation_metadata,
        args.new_source,
        args.new_cohort,
        args.new_runtime,
        *extra_inputs,
        *exclusions,
    ]
    if args.old_clean_runtime is not None:
        input_paths.append(args.old_clean_runtime)
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    blocked_qids, blocked_hashes, blocked_families = _blocked(exclusions)
    silver, records, gates, stats = build_supply(
        old_silver=_read_jsonl(args.old_silver),
        old_records=_read_jsonl(args.old_records),
        old_runtime=_read_jsonl(args.old_runtime),
        new_source=new_source_rows,
        new_cohort=new_cohort_rows,
        new_runtime=_read_jsonl(args.new_runtime),
        blocked_qids=blocked_qids,
        blocked_hashes=blocked_hashes,
        blocked_families=blocked_families,
        cutoff=args.cutoff,
        new_raw=new_raw_rows,
        new_retrieval=new_retrieval_rows,
        old_edge_attestations=_read_jsonl(old_edge_path),
        old_root_attestations=_read_jsonl(old_root_path),
        old_clean_runtime=(
            _read_jsonl(args.old_clean_runtime)
            if args.old_clean_runtime is not None
            else []
        ),
    )
    type_counts = Counter(row["metadata"]["question_type"] for row in silver)
    candidate_wrappers = build_candidate_wrappers(silver, records, gates)
    checks = {
        "strict_candidates_at_least_target": len(silver) >= args.min_total,
        "each_question_type_at_least_target": all(type_counts[qtype] >= args.min_per_type for qtype in QTYPES),
        "identity_join_rate_1": len(silver) == len(records) == len(gates) == len({row["question_key"] for row in records}),
        "candidate_wrapper_join_rate_1": len(candidate_wrappers) == len(silver)
        and {row["question_key"] for row in candidate_wrappers}
        == {row["question_key"] for row in records},
        "all_graph_gate_pass": all(row["m_graph"] == 1 and all(row["eligibility_checks"].values()) for row in gates),
        "all_steps_empty": all(row["steps"] == [] for row in silver),
        "all_ten_passages": all(len(row["retrieved_passages"]) == 10 for row in silver),
        "all_reused_old_records_have_fail_closed_admission": all(
            str((row.get("provenance") or {}).get("unified_source_release") or "")
            != "automatic_proofkg_train_k4_v1"
            or str(
                ((row.get("provenance") or {}).get("old_trace_admission") or {}).get("mode")
                or ""
            )
            in {
                "independent_edge_plus_root_attestation",
                "new_clean_reexecuted_runtime",
            }
            for row in records
        ),
        "blocked_qid_hash_family_overlap_zero": all(
            row["qid"] not in blocked_qids
            and question_sha256(row["question"]) not in blocked_hashes
            and family_sha256(row["question"]) not in blocked_families
            for row in silver
        ),
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "counts": dict(type_counts), "n": len(silver), "stats": stats})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "silver_train": args.output_dir / "silver_train.jsonl",
        "question_kg_records": args.output_dir / "question_kg_records.jsonl",
        "source_gate_records": args.output_dir / "source_gate_records.jsonl",
        "proof_candidates": args.output_dir / "proof_candidates.jsonl",
    }
    _write_jsonl(outputs["silver_train"], silver)
    _write_jsonl(outputs["question_kg_records"], records)
    _write_jsonl(outputs["source_gate_records"], gates)
    _write_jsonl(outputs["proof_candidates"], candidate_wrappers)
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": args.experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "counts": {"strict_candidates": len(silver), "by_question_type": dict(sorted(type_counts.items())), **stats},
        "checks": checks,
        "scientific_boundary": {
            "train_only": True,
            "graph_generation_gold_access": False,
            "source_gold_steps_or_kg_copied": False,
            "gold_answer_use": "outcome label only",
            "selection_of_final_800_performed": False,
            "old_runtime_reuse_contract": (
                "independent edge+root attestations, or newly clean re-executed runtime"
            ),
            "old_expected_qids_used_as_resolver_targets": False,
            "training_started": False,
        },
        "protected_ledger": {
            "version": COMPLETE_PROTECTED_LEDGER_VERSION,
            "complete": True,
            "current_family_recomputed": True,
            **protected_ledger_binding,
        },
        "inputs": {str(i): _identity(path) for i, path in enumerate(input_paths)},
        "outputs": {name: _identity(path) for name, path in outputs.items()},
        "training_started": False,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.output_dir, status=STATUS, extra={"phase": "unified_2wiki_proofkg_candidate_supply", "experiment_id": args.experiment_id, "report": _identity(report_path), "training_started": False})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
