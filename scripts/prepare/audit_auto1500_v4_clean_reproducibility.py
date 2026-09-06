#!/usr/bin/env python3
"""Attest whether protected-safe legacy 2Wiki ProofKG traces are reusable.

This is a read-only, CPU-only audit.  The old automatic-ProofKG runtime used a
2Wiki store which predates the current SAEG evaluation/reserve exclusions.  A
strict trajectory is therefore reusable only when two independent facts can
be established without consulting a Gold answer:

* every executed ``(head QID, PID, tail entity/literal)`` is present in the
  complete-ledger-bound clean store or in the frozen historical Wikidata
  revision cache; and
* every root surface independently resolves to the old runtime QID through
  those clean aliases or the raw labels/aliases in that historical cache.

The re-resolution worklist deliberately contains only question identity,
root *surfaces*, and hashes identifying the old plan.  Expected old root QIDs
are written to a separate ``audit_only`` file and are prohibited as resolver
inputs.  Outputs are append-only and no network, model, answer, or source Gold
trace is accessed.
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
from typing import Any, Iterable, Mapping, Protocol, Sequence

from kgproweight.kg.historical_wikidata_retriever import (
    HISTORICAL_CACHE_VERSION,
    HistoricalWikidataPropertyRetriever,
)
from kgproweight.kg.question_kg import (
    question_key,
    question_sha256,
    validate_question_kg_record,
)
from kgproweight.kg.versioned_evidence_store import (
    VersionedEvidenceStore,
    normalize_alias,
)
from kgproweight.reward.trajectory_source_gate import evaluate_graph_gate
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256


DATASET = "2wikimultihopqa"
CUTOFF = "2020-12-09T23:59:59Z"
SCHEMA_VERSION = "auto1500-v4-independent-edge-root-attestation-v1"
EDGE_SCHEMA_VERSION = "auto1500-v4-edge-attestation-v1"
QUESTION_EDGE_SCHEMA_VERSION = "auto1500-v4-question-edge-attestation-v1"
ROOT_SCHEMA_VERSION = "auto1500-v4-root-attestation-v1"
QUESTION_ROOT_SCHEMA_VERSION = "auto1500-v4-question-root-attestation-v1"
WORKLIST_SCHEMA_VERSION = "auto1500-v4-root-reresolution-worklist-v1"
EXPECTED_QID_SCHEMA_VERSION = "auto1500-v4-expected-root-qids-audit-only-v1"
STATUS = "COMPLETE_OLD_TRACE_ATTESTATION_NOT_TRAINED"
PROTECTED_LEDGER_SCHEMA_VERSION = "mixed-ppo-v4-protected-identity-ledger-v2"
PROTECTED_IDENTITY_SCHEMA_VERSION = "mixed-ppo-v4-protected-question-identity-v2"
PROTECTED_LEDGER_STATUS = "COMPLETE_FROZEN_IDENTITY_ONLY_NOT_TRAINING_DATA"

DEFAULT_OLD_SILVER = Path(
    "data/silver_data/automatic_proofkg_2wiki_train_k4_v1/silver_train.jsonl"
)
DEFAULT_OLD_RECORDS = Path(
    "data/silver_data/automatic_proofkg_2wiki_train_k4_v1/question_kg_records.jsonl"
)
DEFAULT_OLD_RUNTIME = Path(
    "outputs/audits/automatic_proofkg_2wiki_train_k4_v1_n1500_seed42_"
    "historical_stage3_runtime/runtime_details.jsonl"
)
DEFAULT_OLD_STORE = Path("indexes/versioned_2wiki_evidence_store_v2_seed20260902")
DEFAULT_CLEAN_STORE = Path(
    "indexes/versioned_2wiki_evidence_store_v6_mixed3_v4_complete_ledger_seed42"
)
DEFAULT_HISTORICAL_CACHE = Path(
    "outputs/audits/automatic_proofkg_2wiki_train_k4_v1_n1500_seed42_"
    "historical_prefetch_stage3/historical_entity_cache.jsonl"
)
DEFAULT_PROTECTED_LEDGER_DIR = Path(
    "outputs/audits/mixed_ppo_v4_protected_identity_ledger_v2"
)
DEFAULT_PEER_COHORT = Path(
    "outputs/audits/2wiki_proofkg_extension_combined_v1_n350_seed42_"
    "preregistration/cohort.question_only.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/audits/auto1500_clean_edge_root_attestation_v4_complete_ledger"
)
DEFAULT_EXPERIMENT_ID = "AUTO1500-CLEAN-EDGE-ROOT-ATTESTATION-V4-COMPLETE-LEDGER"

_QID = re.compile(r"^Q[1-9][0-9]*$")
_PID = re.compile(r"^P[1-9][0-9]*$")


class _EdgeStore(Protocol):
    def fetch_edges(self, qid: str, pids: Sequence[str]) -> list[dict[str, Any]]: ...


class _AliasStore(_EdgeStore, Protocol):
    def resolve(self, surface: str) -> Any: ...


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _resolve_project_path(raw: Any, *, label: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise ValueError(f"{label} path is empty")
    path = Path(value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def validate_protected_ledger_release(
    directory: Path,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Validate the one complete, dataset-scoped protected-identity ledger.

    The previous implementation duplicated a hand-maintained list of cohort
    paths in multiple scripts.  That list silently omitted later verifier and
    reward-confirmation cohorts.  Formal v4 consumers therefore accept only
    this versioned aggregate release and recompute every current-family hash
    from the question text before trusting it.
    """

    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError("protected ledger must be a versioned release directory")
    ledger_path = directory / "protected_identities.question_only.jsonl"
    report_path = directory / "report.json"
    manifest_path = directory / "manifest.json"
    for path in (ledger_path, report_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != PROTECTED_LEDGER_SCHEMA_VERSION
        or report.get("status") != PROTECTED_LEDGER_STATUS
        or manifest.get("status") != PROTECTED_LEDGER_STATUS
        or report.get("complete") is not True
        or report.get("identity_scope") != "dataset-scoped"
        or report.get("current_family_recomputed") is not True
    ):
        raise ValueError("protected ledger release status/schema/completeness failed")
    boundary = report.get("scientific_boundary") or {}
    if not (
        boundary.get("gold_or_outcome_values_used_for_identity_selection") is False
        and boundary.get("gold_fields_emitted") is False
        and boundary.get("data_raw_modified") is False
        and boundary.get("training_started") is False
    ):
        raise ValueError("protected ledger scientific boundary failed")

    output = report.get("output") or {}
    run = manifest.get("run") or {}
    ledger_sha = sha256_file(ledger_path)
    report_sha = sha256_file(report_path)
    if (
        str(output.get("sha256") or "") != ledger_sha
        or str(run.get("protected_identities_sha256") or "") != ledger_sha
        or str(run.get("report_sha256") or "") != report_sha
    ):
        raise ValueError("protected ledger report/manifest/output hash mismatch")

    rows = read_jsonl(ledger_path)
    qids: set[tuple[str, str]] = set()
    qhashes: set[tuple[str, str]] = set()
    families: set[tuple[str, str]] = set()
    by_dataset: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        dataset = str(row.get("dataset") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        question = str(row.get("question") or "").strip()
        qhash = question_sha256(question)
        family = family_sha256(question)
        if (
            row.get("schema_version") != PROTECTED_IDENTITY_SCHEMA_VERSION
            or dataset not in {"2wikimultihopqa", "hotpotqa", "musique"}
            or not qid
            or not question
            or row.get("gold_access") is not False
            or str(row.get("question_sha256") or "") != qhash
            or str(row.get("family_sha256") or "") != family
            or str(row.get("family_version") or "") != FAMILY_VERSION
        ):
            raise ValueError(f"malformed protected ledger row: {dataset}::{qid}")
        qids.add((dataset, qid))
        qhashes.add((dataset, qhash))
        families.add((dataset, family))
        bucket = by_dataset.setdefault(
            dataset, {"qids": set(), "question_sha256": set(), "current_families": set()}
        )
        bucket["qids"].add(qid)
        bucket["question_sha256"].add(qhash)
        bucket["current_families"].add(family)
    if len(rows) != len(qids) or len(rows) != len(qhashes):
        raise ValueError("protected ledger contains duplicate qid/question identity")
    unique = report.get("unique") or {}
    recomputed_by_dataset = {
        dataset: {name: len(values) for name, values in bucket.items()}
        for dataset, bucket in sorted(by_dataset.items())
    }
    if (
        int(output.get("rows") or -1) != len(rows)
        or int(unique.get("dataset_qids") or -1) != len(qids)
        or int(unique.get("dataset_question_sha256") or -1) != len(qhashes)
        or int(unique.get("dataset_current_families") or -1) != len(families)
        or unique.get("by_dataset") != recomputed_by_dataset
    ):
        raise ValueError("protected ledger recomputed identity counts drifted")

    inventory = report.get("source_inventory") or []
    if int(report.get("source_files") or -1) != len(inventory) or not inventory:
        raise ValueError("protected ledger source inventory is incomplete")
    for item in inventory:
        if not isinstance(item, Mapping):
            raise ValueError("protected ledger source inventory row is malformed")
        source_path = _resolve_project_path(item.get("path"), label="protected source")
        if sha256_file(source_path) != str(item.get("sha256") or ""):
            raise ValueError(f"protected ledger source hash mismatch: {source_path}")
    return ledger_path, report_path, manifest_path, report


def store_identity(path: Path) -> dict[str, Any]:
    return {
        name: file_identity(path / name)
        for name in ("store_manifest.json", "aliases.jsonl", "edges.jsonl")
    }


def validate_clean_store_protected_ledger(
    store_path: Path, *, protected_release: Mapping[str, Any]
) -> dict[str, Any]:
    """Reject a store whose exclusion scope predates the complete ledger."""

    manifest_path = store_path.resolve() / "store_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding = manifest.get("protected_ledger") or {}
    expected = {
        "schema_version": PROTECTED_LEDGER_SCHEMA_VERSION,
        "complete": True,
        "current_family_recomputed": True,
        "report_sha256": str((protected_release.get("report") or {}).get("sha256") or ""),
        "protected_identities_sha256": str(
            (protected_release.get("ledger") or {}).get("sha256") or ""
        ),
        "manifest_sha256": str(
            (protected_release.get("manifest") or {}).get("sha256") or ""
        ),
    }
    if any(binding.get(field) != value for field, value in expected.items()):
        raise ValueError(
            "clean evidence store does not bind the complete protected ledger; "
            "a newly built store is required"
        )
    boundary = manifest.get("scientific_boundary") or {}
    if not (
        boundary.get("selected_question_keys_excluded_exactly") is True
        and boundary.get("selected_current_lexical_families_recomputed_and_excluded")
        is True
    ):
        raise ValueError("clean evidence store exclusion boundary is incomplete")
    return manifest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _index(rows: Iterable[Mapping[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in rows:
        row = dict(value)
        dataset = str(row.get("dataset") or DATASET).strip().lower()
        qid = str(row.get("qid") or row.get("source_qid") or "").strip()
        key = str(row.get("question_key") or question_key(dataset, qid))
        if dataset != DATASET or not qid or key in result:
            raise ValueError(f"{label}: invalid/duplicate identity {key!r}")
        result[key] = row
    return result


def _blocked(paths: Sequence[Path]) -> tuple[set[str], set[str], set[str]]:
    qids: set[str] = set()
    hashes: set[str] = set()
    families: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            if str(row.get("dataset") or DATASET).strip().lower() != DATASET:
                continue
            qid = str(row.get("qid") or row.get("source_qid") or "").strip()
            question = str(row.get("question") or "").strip()
            if not qid or not question:
                raise ValueError(f"incomplete exclusion identity in {path}")
            qids.add(qid)
            hashes.add(question_sha256(question))
            families.add(family_sha256(question))
    return qids, hashes, families


def select_protected_safe_strict(
    *,
    old_silver: Sequence[Mapping[str, Any]],
    old_records: Sequence[Mapping[str, Any]],
    old_runtime: Sequence[Mapping[str, Any]],
    blocked_qids: set[str],
    blocked_hashes: set[str],
    blocked_families: set[str],
    cutoff: str,
) -> list[dict[str, Any]]:
    """Recompute the exact protected-safe strict old candidate population."""

    silver_by_key = _index(old_silver, label="old silver")
    records_by_key = _index(old_records, label="old question KG")
    runtime_by_key = _index(old_runtime, label="old runtime")
    if set(silver_by_key) != set(records_by_key):
        raise ValueError("old silver/question-KG identity sets differ")
    selected: list[dict[str, Any]] = []
    for key, silver in silver_by_key.items():
        base = records_by_key[key]
        runtime = runtime_by_key.get(key)
        if runtime is None:
            raise ValueError(f"old runtime join miss: {key}")
        validate_question_kg_record(base)
        question = str(silver.get("question") or "").strip()
        qid = str(silver.get("qid") or "").strip()
        qhash = question_sha256(question)
        family = family_sha256(question)
        if (
            str(runtime.get("question") or "").strip() != question
            or str(runtime.get("question_sha256") or "") != qhash
            or base.get("kg_subgraph") != runtime.get("kg_subgraph")
        ):
            raise ValueError(f"old source/runtime/question-KG drift: {key}")
        gate = evaluate_graph_gate(
            runtime,
            dataset=DATASET,
            qid=qid,
            question=question,
            historical_cutoff=cutoff,
        )
        if not gate.graph_eligible:
            raise ValueError(f"old frozen question-KG is not strict eligible: {key}")
        if qid in blocked_qids or qhash in blocked_hashes or family in blocked_families:
            continue
        selected.append(
            {
                "question_key": key,
                "dataset": DATASET,
                "qid": qid,
                "question": question,
                "question_sha256": qhash,
                "family_sha256": family,
                "question_type": str((silver.get("metadata") or {}).get("question_type") or ""),
                "runtime": runtime,
            }
        )
    return sorted(selected, key=lambda row: (row["question_type"], row["qid"]))


def _normalise_display(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    return re.sub(r"\s+", " ", text.strip().casefold())


def _literal_identity(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text.strip())


def canonical_edge_identity(edge: Mapping[str, Any]) -> tuple[str, str, str, str] | None:
    head_qid = str(edge.get("head_qid") or "").strip().upper()
    pid = str(edge.get("pid") or "").strip().upper()
    if not _QID.fullmatch(head_qid) or not _PID.fullmatch(pid):
        return None
    tail_qid = str(edge.get("tail_qid") or "").strip().upper()
    if tail_qid:
        if not _QID.fullmatch(tail_qid):
            return None
        return head_qid, pid, "entity", tail_qid
    raw = edge.get("tail_raw_value")
    if raw in (None, ""):
        raw = edge.get("tail_value")
    literal = _literal_identity(raw)
    if not literal:
        return None
    return head_qid, pid, "literal", literal


def _rendered_triple(edge: Mapping[str, Any]) -> tuple[str, str, str]:
    return tuple(
        _normalise_display(value)
        for value in (
            edge.get("head_label"),
            edge.get("relation") or edge.get("pid"),
            edge.get("tail_value") or edge.get("tail_raw_value"),
        )
    )


def _fetch_candidates(
    store: _EdgeStore, pairs: Sequence[tuple[str, str]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for qid, pid in pairs:
        for raw in store.fetch_edges(qid, [pid]):
            row = dict(raw)
            identity = canonical_edge_identity(row)
            if identity is not None and identity not in seen:
                seen.add(identity)
                rows.append(row)
    return rows


def recover_expected_edge(
    *,
    match: Sequence[Any],
    pairs: Sequence[tuple[str, str]],
    historical_edges: Sequence[Mapping[str, Any]],
    legacy_identity_edges: Sequence[Mapping[str, Any]],
    output_entities: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, str, str, str] | None, str]:
    """Recover an executed edge identity; the legacy store is identity-only."""

    if len(match) != 3:
        return None, "invalid_match_shape"
    rendered = tuple(_normalise_display(value) for value in match)
    all_candidates = [dict(row) for row in historical_edges] + [
        dict(row) for row in legacy_identity_edges
    ]

    def unique_identity(candidates: Iterable[Mapping[str, Any]]) -> tuple[str, str, str, str] | None:
        identities = {
            identity
            for candidate in candidates
            if (identity := canonical_edge_identity(candidate)) is not None
        }
        return next(iter(identities)) if len(identities) == 1 else None

    # Prefer the raw historical source.  A legacy store literal often has only
    # the presentation value (for example ``3 April 1885``), while the
    # historical edge additionally carries ``1885-04-03``.  Pooling them
    # before canonicalisation would manufacture a false ambiguity.
    historical_exact = [
        candidate for candidate in historical_edges if _rendered_triple(candidate) == rendered
    ]
    identity = unique_identity(historical_exact)
    if identity is not None:
        return identity, "historical_render_exact"

    legacy_exact = [
        candidate
        for candidate in legacy_identity_edges
        if _rendered_triple(candidate) == rendered
    ]
    identity = unique_identity(legacy_exact)
    if identity is not None:
        return identity, "legacy_identity_render_exact"

    # A handful of historical entities have two citizenship QIDs with the
    # same English demonym (for example Q145 and Q174193 both render as
    # ``British``).  The old frozen store may disambiguate which canonical
    # tail produced the retained runtime edge.  This is mapping aid only: the
    # chosen canonical tuple must still be present in an independent source
    # before the edge can be attested.
    legacy_tail = [
        candidate
        for candidate in legacy_identity_edges
        if rendered[2]
        in {
            _normalise_display(candidate.get("tail_value")),
            _normalise_display(candidate.get("tail_raw_value")),
        }
    ]
    identity = unique_identity(legacy_tail)
    if identity is not None:
        return identity, "legacy_identity_tail_disambiguation"

    # Intermediate outputs carry a QID even when display labels changed.  The
    # QID is used only to disambiguate the retrospective audit, never as input
    # to a new resolver.
    output_qids = {
        str(row.get("qid") or "").strip().upper()
        for row in output_entities
        if _QID.fullmatch(str(row.get("qid") or "").strip().upper())
        and _normalise_display(row.get("surface") or row.get("label")) == rendered[2]
    }
    if output_qids:
        identity = unique_identity(
            candidate
            for candidate in all_candidates
            if str(candidate.get("tail_qid") or "").strip().upper() in output_qids
        )
        if identity is not None:
            return identity, "trace_output_qid_disambiguation"

    tail_display = [
        candidate
        for candidate in all_candidates
        if rendered[2]
        in {
            _normalise_display(candidate.get("tail_value")),
            _normalise_display(candidate.get("tail_raw_value")),
        }
    ]
    identity = unique_identity(tail_display)
    if identity is not None:
        return identity, "tail_display_unique"

    identity = unique_identity(all_candidates)
    if identity is not None:
        return identity, "single_canonical_candidate"
    return None, "unmapped_or_ambiguous"


def _historical_entities(
    rows: Sequence[Mapping[str, Any]], *, cutoff: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        if row.get("schema_version") != HISTORICAL_CACHE_VERSION:
            continue
        if str(row.get("cutoff") or "") != cutoff:
            continue
        qid = str(row.get("qid") or "").strip().upper()
        if not _QID.fullmatch(qid):
            continue
        existing = result.get(qid)
        if existing is not None and canonical_sha256(existing) != canonical_sha256(row):
            raise ValueError(f"conflicting historical cache rows for {qid}")
        result[qid] = row
    return result


def historical_alias_match(
    *, expected_qid: str, surfaces: Sequence[str], historical_entities: Mapping[str, Mapping[str, Any]]
) -> bool:
    cached = historical_entities.get(expected_qid.upper()) or {}
    entity = cached.get("entity") or {}
    if str(entity.get("id") or expected_qid).strip().upper() != expected_qid.upper():
        return False
    aliases: set[str] = set()
    for item in (entity.get("labels") or {}).values():
        if isinstance(item, Mapping):
            aliases.add(normalize_alias(item.get("value")))
    for values in (entity.get("aliases") or {}).values():
        for item in values or []:
            if isinstance(item, Mapping):
                aliases.add(normalize_alias(item.get("value")))
    return any(normalize_alias(surface) in aliases for surface in surfaces if str(surface).strip())


def clean_alias_match(*, expected_qid: str, surfaces: Sequence[str], clean_store: _AliasStore) -> bool:
    for surface in surfaces:
        if not str(surface).strip():
            continue
        result = clean_store.resolve(str(surface))
        if not getattr(result, "abstained", True) and str(
            getattr(result, "selected_qid", "") or ""
        ).upper() == expected_qid.upper():
            return True
    return False


def audit_candidate(
    candidate: Mapping[str, Any],
    *,
    old_identity_store: _EdgeStore,
    clean_store: _AliasStore,
    historical: _EdgeStore,
    historical_entities: Mapping[str, Mapping[str, Any]],
    cutoff: str,
) -> dict[str, list[dict[str, Any]] | dict[str, Any]]:
    runtime = dict(candidate["runtime"])
    key = str(candidate["question_key"])
    qtype = str(candidate["question_type"])
    plan_sha = canonical_sha256(runtime.get("query_plan") or {})
    execution_sha = canonical_sha256(runtime.get("execution") or {})
    runtime_sha = canonical_sha256(runtime)

    edge_rows: list[dict[str, Any]] = []
    mapping_counts: Counter[str] = Counter()
    for hop in (runtime.get("execution") or {}).get("hops") or []:
        hop_index = int(hop.get("hop_index") or 0)
        qids = sorted(
            {
                str(entity.get("qid") or "").strip().upper()
                for entity in (hop.get("input_entities") or [])
                if _QID.fullmatch(str(entity.get("qid") or "").strip().upper())
            }
        )
        pids = sorted(
            {
                str(pid).strip().upper()
                for pid in (hop.get("pids") or [])
                if _PID.fullmatch(str(pid).strip().upper())
            }
        )
        pairs = [(qid, pid) for qid in qids for pid in pids]
        historical_edges = _fetch_candidates(historical, pairs)
        legacy_edges = _fetch_candidates(old_identity_store, pairs)
        clean_edges = _fetch_candidates(clean_store, pairs)
        hist_identities = {
            identity
            for row in historical_edges
            if (identity := canonical_edge_identity(row)) is not None
        }
        clean_identities = {
            identity
            for row in clean_edges
            if (identity := canonical_edge_identity(row)) is not None
        }
        for match_index, match in enumerate(hop.get("matches") or [], start=1):
            expected, method = recover_expected_edge(
                match=match,
                pairs=pairs,
                historical_edges=historical_edges,
                legacy_identity_edges=legacy_edges,
                output_entities=hop.get("output_entities") or [],
            )
            mapping_counts[method] += 1
            historical_present = expected is not None and expected in hist_identities
            clean_present = expected is not None and expected in clean_identities
            if clean_present and historical_present:
                source_class = "both"
            elif clean_present:
                source_class = "clean_v4_store_only"
            elif historical_present:
                source_class = "historical_wikidata_only"
            else:
                source_class = "neither"
            edge_rows.append(
                {
                    "schema_version": EDGE_SCHEMA_VERSION,
                    "question_key": key,
                    "dataset": DATASET,
                    "qid": str(candidate["qid"]),
                    "question_sha256": str(candidate["question_sha256"]),
                    "question_type": qtype,
                    "old_plan_sha256": plan_sha,
                    "old_execution_sha256": execution_sha,
                    "hop_index": hop_index,
                    "match_index": match_index,
                    "old_rendered_match": list(match),
                    "expected_edge_identity_audit_only": (
                        {
                            "head_qid": expected[0],
                            "pid": expected[1],
                            "tail_kind": expected[2],
                            "tail_identity": expected[3],
                        }
                        if expected is not None
                        else None
                    ),
                    "identity_recovery_method": method,
                    "clean_v4_store_present": clean_present,
                    "historical_wikidata_present": historical_present,
                    "independently_reproduced": clean_present or historical_present,
                    "reproduction_source": source_class,
                    "audit_only": True,
                    "gold_access": False,
                }
            )

    root_rows: list[dict[str, Any]] = []
    plan_anchors = [
        str(value).strip()
        for value in ((runtime.get("query_plan") or {}).get("anchors") or [])
        if str(value).strip()
    ]
    anchor_entities = (runtime.get("execution") or {}).get("anchor_entities") or {}
    if (
        not plan_anchors
        or len(plan_anchors) != len(set(plan_anchors))
        or set(plan_anchors) != set(anchor_entities)
    ):
        raise ValueError(f"planner/execution root anchor identity drift: {key}")
    for anchor_surface in sorted(plan_anchors):
        raw_entity = anchor_entities[anchor_surface]
        entity = dict(raw_entity or {})
        expected_qid = str(entity.get("qid") or "").strip().upper()
        # Only the planner's original root surface may be used.  The runtime
        # entity label/resolved_surface are outputs of the old resolver and
        # therefore cannot independently attest that resolver's QID.
        surfaces = [str(anchor_surface).strip()]
        historical_match = bool(_QID.fullmatch(expected_qid)) and historical_alias_match(
            expected_qid=expected_qid,
            surfaces=surfaces,
            historical_entities=historical_entities,
        )
        clean_match = bool(_QID.fullmatch(expected_qid)) and clean_alias_match(
            expected_qid=expected_qid, surfaces=surfaces, clean_store=clean_store
        )
        if clean_match and historical_match:
            source_class = "both"
        elif clean_match:
            source_class = "clean_v4_alias_only"
        elif historical_match:
            source_class = "historical_label_or_alias_only"
        else:
            source_class = "neither"
        root_rows.append(
            {
                "schema_version": ROOT_SCHEMA_VERSION,
                "question_key": key,
                "dataset": DATASET,
                "qid": str(candidate["qid"]),
                "question_sha256": str(candidate["question_sha256"]),
                "question_type": qtype,
                "old_plan_sha256": plan_sha,
                "old_execution_sha256": execution_sha,
                "root_anchor_surface": str(anchor_surface),
                "surface_variants": surfaces,
                "expected_old_qid_audit_only": expected_qid,
                "clean_v4_alias_match": clean_match,
                "historical_label_or_alias_match": historical_match,
                "independently_attested": clean_match or historical_match,
                "attestation_source": source_class,
                "audit_only": True,
                "must_not_be_resolver_input": True,
                "gold_access": False,
            }
        )

    edge_ok = bool(edge_rows) and all(row["independently_reproduced"] for row in edge_rows)
    root_ok = bool(root_rows) and all(row["independently_attested"] for row in root_rows)
    edge_question = {
        "schema_version": QUESTION_EDGE_SCHEMA_VERSION,
        "question_key": key,
        "dataset": DATASET,
        "qid": str(candidate["qid"]),
        "question_sha256": str(candidate["question_sha256"]),
        "family_version": FAMILY_VERSION,
        "family_sha256": str(candidate["family_sha256"]),
        "question_type": qtype,
        "old_runtime_sha256": runtime_sha,
        "old_plan_sha256": plan_sha,
        "old_execution_sha256": execution_sha,
        "kg_sha256": canonical_sha256(runtime.get("kg_subgraph") or []),
        "planned_hop_count": len((runtime.get("query_plan") or {}).get("hops") or []),
        "executed_hop_count": len((runtime.get("execution") or {}).get("hops") or []),
        "executed_edge_count": len(edge_rows),
        "independently_reproduced_edge_count": sum(
            bool(row["independently_reproduced"]) for row in edge_rows
        ),
        "all_executed_edges_independently_reproduced": edge_ok,
        "edge_details_sha256": canonical_sha256(edge_rows),
        "historical_cutoff": cutoff,
        "gold_access": False,
    }
    root_question = {
        "schema_version": QUESTION_ROOT_SCHEMA_VERSION,
        "question_key": key,
        "dataset": DATASET,
        "qid": str(candidate["qid"]),
        "question_sha256": str(candidate["question_sha256"]),
        "family_version": FAMILY_VERSION,
        "family_sha256": str(candidate["family_sha256"]),
        "question_type": qtype,
        "old_runtime_sha256": runtime_sha,
        "old_plan_sha256": plan_sha,
        "old_execution_sha256": execution_sha,
        "root_anchor_count": len(root_rows),
        "independently_attested_root_count": sum(
            bool(row["independently_attested"]) for row in root_rows
        ),
        "all_root_anchors_independently_attested": root_ok,
        "root_details_sha256": canonical_sha256(root_rows),
        "historical_cutoff": cutoff,
        "gold_access": False,
    }
    return {
        "edges": edge_rows,
        "roots": root_rows,
        "question_edge": edge_question,
        "question_root": root_question,
        "mapping_counts": dict(mapping_counts),
    }


def make_reresolution_rows(
    candidates: Sequence[Mapping[str, Any]],
    question_root_rows: Sequence[Mapping[str, Any]],
    root_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_by_key = {str(row["question_key"]): row for row in candidates}
    roots_by_key: dict[str, list[Mapping[str, Any]]] = {}
    for row in root_rows:
        roots_by_key.setdefault(str(row["question_key"]), []).append(row)
    worklist: list[dict[str, Any]] = []
    audit_only: list[dict[str, Any]] = []
    for summary in question_root_rows:
        if summary["all_root_anchors_independently_attested"]:
            continue
        key = str(summary["question_key"])
        candidate = candidate_by_key[key]
        anchors = roots_by_key[key]
        root_surfaces = sorted(
            {str(row["root_anchor_surface"]).strip() for row in anchors}
        )
        worklist.append(
            {
                "schema_version": WORKLIST_SCHEMA_VERSION,
                "question_key": key,
                "dataset": DATASET,
                "qid": str(candidate["qid"]),
                "question": str(candidate["question"]),
                "question_sha256": str(candidate["question_sha256"]),
                "family_version": FAMILY_VERSION,
                "family_sha256": str(candidate["family_sha256"]),
                "question_type": str(candidate["question_type"]),
                "old_plan_identity": {
                    "old_plan_sha256": str(summary["old_plan_sha256"]),
                    "old_execution_sha256": str(summary["old_execution_sha256"]),
                    "planner_version": str(
                        ((candidate["runtime"].get("query_plan") or {}).get("planner_version") or "")
                    ),
                    "planner_predictions_sha256": str(
                        ((candidate["runtime"].get("provenance") or {}).get("planner_predictions_sha256") or "")
                    ),
                },
                "root_anchor_surfaces": root_surfaces,
                "resolver_input_contract": {
                    "allowed": ["dataset", "qid", "question", "root_anchor_surfaces"],
                    "expected_old_qids_present": False,
                    "gold_access": False,
                },
                "gold_access": False,
            }
        )
        audit_only.append(
            {
                "schema_version": EXPECTED_QID_SCHEMA_VERSION,
                "question_key": key,
                "dataset": DATASET,
                "qid": str(candidate["qid"]),
                "question_sha256": str(candidate["question_sha256"]),
                "old_plan_sha256": str(summary["old_plan_sha256"]),
                "expected_old_root_qids": [
                    {
                        "root_anchor_surface": str(row["root_anchor_surface"]),
                        "expected_old_qid": str(row["expected_old_qid_audit_only"]),
                    }
                    for row in sorted(anchors, key=lambda item: str(item["root_anchor_surface"]))
                ],
                "audit_only": True,
                "must_not_be_model_or_resolver_input": True,
                "gold_access": False,
            }
        )
    worklist.sort(key=lambda row: (row["question_type"], row["qid"]))
    audit_only.sort(key=lambda row: row["qid"])
    if re.search(r'"Q[1-9][0-9]*"', json.dumps(worklist, ensure_ascii=False)):
        raise ValueError("expected old Wikidata QID leaked into resolver worklist")
    return worklist, audit_only


def run_audit(
    *,
    old_silver_path: Path,
    old_records_path: Path,
    old_runtime_path: Path,
    old_store_path: Path,
    clean_store_path: Path,
    historical_cache_path: Path,
    exclusion_paths: Sequence[Path],
    protected_ledger_dir: Path | None,
    peer_cohort_path: Path | None,
    output_dir: Path,
    experiment_id: str,
    cutoff: str = CUTOFF,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite versioned output: {output_dir}")
    protected_release: dict[str, Any] | None = None
    if protected_ledger_dir is not None:
        if exclusion_paths:
            raise ValueError(
                "formal complete-ledger audit cannot mix an aggregate ledger with "
                "hand-maintained --exclude paths"
            )
        ledger_path, ledger_report_path, ledger_manifest_path, ledger_report = (
            validate_protected_ledger_release(protected_ledger_dir)
        )
        exclusion_paths = [ledger_path]
        protected_release = {
            "version": PROTECTED_LEDGER_SCHEMA_VERSION,
            "complete": True,
            "current_family_recomputed": True,
            "ledger": file_identity(ledger_path),
            "report": file_identity(ledger_report_path),
            "manifest": file_identity(ledger_manifest_path),
            "unique": ledger_report["unique"],
        }
        validate_clean_store_protected_ledger(
            clean_store_path, protected_release=protected_release
        )
    for path in (
        old_silver_path,
        old_records_path,
        old_runtime_path,
        historical_cache_path,
        *exclusion_paths,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if peer_cohort_path is not None and not peer_cohort_path.is_file():
        raise FileNotFoundError(peer_cohort_path)
    old_store = VersionedEvidenceStore(old_store_path)
    clean_store = VersionedEvidenceStore(clean_store_path)
    historical = HistoricalWikidataPropertyRetriever(
        cache_path=historical_cache_path,
        cutoff=cutoff,
        offline=True,
        label_resolver=old_store,
    )
    historical_entities = _historical_entities(read_jsonl(historical_cache_path), cutoff=cutoff)
    blocked_qids, blocked_hashes, blocked_families = _blocked(exclusion_paths)
    candidates = select_protected_safe_strict(
        old_silver=read_jsonl(old_silver_path),
        old_records=read_jsonl(old_records_path),
        old_runtime=read_jsonl(old_runtime_path),
        blocked_qids=blocked_qids,
        blocked_hashes=blocked_hashes,
        blocked_families=blocked_families,
        cutoff=cutoff,
    )

    edges: list[dict[str, Any]] = []
    roots: list[dict[str, Any]] = []
    question_edges: list[dict[str, Any]] = []
    question_roots: list[dict[str, Any]] = []
    mapping_counts: Counter[str] = Counter()
    for candidate in candidates:
        audit = audit_candidate(
            candidate,
            old_identity_store=old_store,
            clean_store=clean_store,
            historical=historical,
            historical_entities=historical_entities,
            cutoff=cutoff,
        )
        edges.extend(audit["edges"])  # type: ignore[arg-type]
        roots.extend(audit["roots"])  # type: ignore[arg-type]
        question_edges.append(audit["question_edge"])  # type: ignore[arg-type]
        question_roots.append(audit["question_root"])  # type: ignore[arg-type]
        mapping_counts.update(audit["mapping_counts"])  # type: ignore[arg-type]

    edges.sort(key=lambda row: (row["question_type"], row["qid"], row["hop_index"], row["match_index"]))
    roots.sort(key=lambda row: (row["question_type"], row["qid"], row["root_anchor_surface"]))
    question_edges.sort(key=lambda row: (row["question_type"], row["qid"]))
    question_roots.sort(key=lambda row: (row["question_type"], row["qid"]))
    worklist, expected_qids_audit = make_reresolution_rows(candidates, question_roots, roots)

    peer_overlap = {"qid": 0, "question_sha256": 0, "family_sha256": 0}
    if peer_cohort_path is not None:
        peer_rows = read_jsonl(peer_cohort_path)
        peer_qids = {str(row.get("qid") or "").strip() for row in peer_rows}
        peer_hashes = {
            str(row.get("question_sha256") or question_sha256(str(row.get("question") or "")))
            for row in peer_rows
        }
        peer_families = {
            str(row.get("family_sha256") or family_sha256(str(row.get("question") or "")))
            for row in peer_rows
        }
        peer_overlap = {
            "qid": sum(str(row["qid"]) in peer_qids for row in candidates),
            "question_sha256": sum(
                str(row["question_sha256"]) in peer_hashes for row in candidates
            ),
            # Same-family train candidates are permitted by the frozen v4
            # selector; this is reported, never silently treated as identity.
            "family_sha256": sum(
                str(row["family_sha256"]) in peer_families for row in candidates
            ),
        }

    edge_source_counts = Counter(str(row["reproduction_source"]) for row in edges)
    root_source_counts = Counter(str(row["attestation_source"]) for row in roots)
    edge_complete_by_type = Counter(
        str(row["question_type"])
        for row in question_edges
        if row["all_executed_edges_independently_reproduced"]
    )
    root_complete_by_type = Counter(
        str(row["question_type"])
        for row in question_roots
        if row["all_root_anchors_independently_attested"]
    )
    worklist_by_type = Counter(str(row["question_type"]) for row in worklist)
    counts = {
        "protected_safe_strict_questions": len(candidates),
        "executed_edges": len(edges),
        "independently_reproduced_edges": sum(bool(row["independently_reproduced"]) for row in edges),
        "all_edges_attested_questions": sum(bool(row["all_executed_edges_independently_reproduced"]) for row in question_edges),
        "root_anchors": len(roots),
        "independently_attested_root_anchors": sum(bool(row["independently_attested"]) for row in roots),
        "all_roots_attested_questions": sum(bool(row["all_root_anchors_independently_attested"]) for row in question_roots),
        "reresolution_worklist_questions": len(worklist),
    }
    checks = {
        "identity_join_exact": len(candidates) == len(question_edges) == len(question_roots),
        "all_old_executed_edges_independently_reproduced": counts["executed_edges"]
        == counts["independently_reproduced_edges"],
        "worklist_equals_root_failures": len(worklist)
        == len(candidates) - counts["all_roots_attested_questions"],
        "worklist_contains_no_expected_old_wikidata_qid": not bool(
            re.search(r'"Q[1-9][0-9]*"', json.dumps(worklist, ensure_ascii=False))
        ),
        "expected_old_qids_separate_and_audit_only": len(expected_qids_audit) == len(worklist)
        and all(
            row.get("audit_only") is True
            and row.get("must_not_be_model_or_resolver_input") is True
            for row in expected_qids_audit
        ),
        "combined350_peer_qid_and_exact_question_overlap_zero": (
            peer_overlap["qid"] == 0 and peer_overlap["question_sha256"] == 0
        ),
    }
    if expected_counts is not None:
        for name, expected in expected_counts.items():
            checks[f"expected_{name}"] = counts.get(name) == expected
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "counts": counts})

    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "edge_attestations_audit_only": output_dir / "edge_attestations.audit_only.jsonl",
        "question_edge_attestations": output_dir / "question_edge_attestations.jsonl",
        "root_attestations_audit_only": output_dir / "root_attestations.audit_only.jsonl",
        "question_root_attestations": output_dir / "question_root_attestations.jsonl",
        "reresolution_worklist": output_dir / "reresolution_worklist.question_root_only.jsonl",
        "expected_old_qids_audit_only": output_dir / "reresolution_expected_old_qids.audit_only.jsonl",
    }
    rows_by_name = {
        "edge_attestations_audit_only": edges,
        "question_edge_attestations": question_edges,
        "root_attestations_audit_only": roots,
        "question_root_attestations": question_roots,
        "reresolution_worklist": worklist,
        "expected_old_qids_audit_only": expected_qids_audit,
    }
    for name, path in outputs.items():
        write_jsonl(path, rows_by_name[name])

    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "protected_ledger": protected_release
        or {
            "version": "diagnostic-hand-maintained-exclusions",
            "complete": False,
            "current_family_recomputed": True,
        },
        "historical_cutoff": cutoff,
        "counts": counts,
        "by_question_type": {
            "population": dict(sorted(Counter(str(row["question_type"]) for row in candidates).items())),
            "all_edges_attested": dict(sorted(edge_complete_by_type.items())),
            "all_roots_attested": dict(sorted(root_complete_by_type.items())),
            "requires_reresolution": dict(sorted(worklist_by_type.items())),
        },
        "edge_reproduction_sources": dict(sorted(edge_source_counts.items())),
        "root_attestation_sources": dict(sorted(root_source_counts.items())),
        "combined350_peer_overlap": peer_overlap,
        "edge_identity_recovery_methods": dict(sorted(mapping_counts.items())),
        "checks": checks,
        "contract": {
            "legacy_runtime_reuse_requires": [
                "all_executed_edges_independently_reproduced=true",
                "all_root_anchors_independently_attested=true",
                "question/runtime/plan/execution hashes exact",
            ],
            "alternative": "new clean runtime re-executed from question/root surfaces without old QID targets",
            "worklist_is_model_safe": True,
            "expected_old_qids_are_audit_only": True,
            "gold_access": False,
            "network_access": False,
            "training_started": False,
        },
        "inputs": {
            "old_silver": file_identity(old_silver_path),
            "old_question_kg": file_identity(old_records_path),
            "old_runtime": file_identity(old_runtime_path),
            "old_identity_store": store_identity(old_store_path),
            "clean_v4_store": store_identity(clean_store_path),
            "historical_cache": file_identity(historical_cache_path),
            "exclusions": [file_identity(path) for path in exclusion_paths],
            "combined350_peer_cohort": (
                file_identity(peer_cohort_path) if peer_cohort_path is not None else None
            ),
        },
        "outputs": {name: file_identity(path) for name, path in outputs.items()},
        "training_started": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(
        output_dir,
        status=STATUS,
        extra={
            "phase": "old_auto1500_independent_edge_root_attestation",
            "experiment_id": experiment_id,
            "report": file_identity(report_path),
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-silver", type=Path, default=DEFAULT_OLD_SILVER)
    parser.add_argument("--old-records", type=Path, default=DEFAULT_OLD_RECORDS)
    parser.add_argument("--old-runtime", type=Path, default=DEFAULT_OLD_RUNTIME)
    parser.add_argument("--old-store", type=Path, default=DEFAULT_OLD_STORE)
    parser.add_argument("--clean-store", type=Path, default=DEFAULT_CLEAN_STORE)
    parser.add_argument("--historical-cache", type=Path, default=DEFAULT_HISTORICAL_CACHE)
    parser.add_argument(
        "--protected-ledger-dir", type=Path, default=DEFAULT_PROTECTED_LEDGER_DIR
    )
    parser.add_argument("--peer-cohort", type=Path, default=DEFAULT_PEER_COHORT)
    parser.add_argument("--cutoff", default=CUTOFF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    args = parser.parse_args()
    report = run_audit(
        old_silver_path=args.old_silver,
        old_records_path=args.old_records,
        old_runtime_path=args.old_runtime,
        old_store_path=args.old_store,
        clean_store_path=args.clean_store,
        historical_cache_path=args.historical_cache,
        exclusion_paths=(),
        protected_ledger_dir=args.protected_ledger_dir,
        peer_cohort_path=args.peer_cohort,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
        cutoff=args.cutoff,
        expected_counts=None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
