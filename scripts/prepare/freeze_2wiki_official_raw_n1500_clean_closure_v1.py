#!/usr/bin/env python3
"""Freeze and finalise the Gold-free 2Wiki official-raw n=1500 closure.

The release has two deliberately separate locks:

``preregister``
    Freezes the method, gates, existing planner/candidate/store identities and
    the *planned* root-resolver paths.  It can run before root resolution and
    therefore never pretends that a missing resolver result already exists.

``finalize``
    Runs only after the root resolver has completed.  It validates every
    resolver artifact, independently replays the exact final-consumer lookup,
    requires projection == resolver dry-run for every root occurrence, and
    writes the immutable execution lock.  It does not execute the network
    closure.

Neither phase opens the official raw dataset, Gold answers, supporting facts,
passages, or old expected QIDs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.kg.versioned_evidence_store import normalize_alias
from kgproweight.kg.wikipedia_title_resolver import complete_question_surface_title
from kgproweight.utils.logging import dump_manifest


ROOT = Path(__file__).resolve().parents[2]
DATASET = "2wikimultihopqa"
EXPECTED_N = 1500
CUTOFF = "2020-12-09T23:59:59Z"
QTYPES = ("bridge_comparison", "comparison", "compositional", "inference")
EXPECTED_QTYPE_COUNTS = {
    "bridge_comparison": 390,
    "comparison": 390,
    "compositional": 389,
    "inference": 331,
}
QID_RE = re.compile(r"^Q[1-9][0-9]*$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

POLICY_SCHEMA = "2wiki-official-raw-n1500-clean-closure-policy-v3"
POLICY_STATUS = "FROZEN_POLICY_WAITING_FOR_ROOT_RESOLUTION"
LOCK_SCHEMA = "2wiki-official-raw-n1500-clean-closure-execution-lock-v3"
LOCK_STATUS = "FROZEN_EXECUTION_LOCK_READY_NOT_RUN_NOT_TRAINED"
EXPERIMENT_ID = (
    "2WIKI-PROOFKG-OFFICIAL-RAW-V2-CANDIDATE-POOL-N1500-"
    "CLEAN-CLOSURE-V3-SEED42"
)

DEFAULT_COHORT = ROOT / (
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "seed42_preregistration/cohort.question_only.jsonl"
)
DEFAULT_CANDIDATE_PROTOCOL = DEFAULT_COHORT.parent / "protocol.json"
DEFAULT_PLANS = ROOT / (
    "outputs/validation/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "seed42_plans_v1/predictions.question_only.jsonl"
)
DEFAULT_PLANNER_PROTOCOL = ROOT / (
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "seed42_planner_execution_v1_preregistration/protocol.json"
)
DEFAULT_PLANNER_POSTFLIGHT = ROOT / (
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "seed42_plans_v1_postflight/report.json"
)
DEFAULT_LEDGER_DIR = ROOT / "outputs/audits/mixed_ppo_v4_protected_identity_ledger_v2"
DEFAULT_STORE = ROOT / (
    "indexes/versioned_2wiki_evidence_store_v6_mixed3_v4_complete_ledger_seed42"
)
DEFAULT_ROOT_PROTOCOL = ROOT / (
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "root_resolution_v2_preregistration/protocol.json"
)
DEFAULT_ROOT_DIR = ROOT / (
    "indexes/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "root_resolution_v2"
)
DEFAULT_POLICY_DIR = ROOT / (
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "clean_closure_v3_preregistration"
)
DEFAULT_LOCK_DIR = ROOT / (
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "clean_closure_v3_execution_lock"
)
DEFAULT_RUN_DIR = ROOT / (
    "data/derived/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "clean_closure_v3"
)
DEFAULT_RESULT_DIR = ROOT / (
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "clean_closure_v3_result"
)

FORBIDDEN_KEYS = {
    "answer",
    "answers",
    "gold_answer",
    "gold_answers",
    "golden_answers",
    "supporting_facts",
    "support",
    "evidence",
    "decomposition",
    "question_decomposition",
    "passages",
    "retrieval_result",
    "target",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
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


def assert_file_identity(expected: Mapping[str, Any], path: Path, *, label: str) -> None:
    actual = file_identity(path)
    if any(actual.get(key) != expected.get(key) for key in actual):
        raise ValueError(f"{label} identity drift")


def _has_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in FORBIDDEN_KEYS or _has_forbidden_key(nested)
            for key, nested in value.items()
        )
    if isinstance(value, list):
        return any(_has_forbidden_key(item) for item in value)
    return False


def index_question_rows(
    rows: Sequence[Mapping[str, Any]], *, label: str, require_qtype: bool
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        dataset = str(row.get("dataset") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        question = str(row.get("question") or "").strip()
        if dataset != DATASET or not qid or not question:
            raise ValueError(f"{label}: invalid identity {dataset!r}::{qid!r}")
        key = question_key(dataset, qid)
        if str(row.get("question_key") or key) != key:
            raise ValueError(f"{label}: question_key mismatch for {key}")
        qhash = question_sha256(question)
        if str(row.get("question_sha256") or "") != qhash:
            raise ValueError(f"{label}: question hash mismatch for {key}")
        if row.get("gold_access") is not False:
            raise ValueError(f"{label}: gold_access must be false for {key}")
        if _has_forbidden_key(row):
            raise ValueError(f"{label}: forbidden field present for {key}")
        if require_qtype and str(row.get("question_type") or "") not in QTYPES:
            raise ValueError(f"{label}: invalid question_type for {key}")
        if key in result:
            raise ValueError(f"{label}: duplicate identity {key}")
        result[key] = row
    if len(result) != EXPECTED_N:
        raise ValueError(f"{label}: expected {EXPECTED_N} rows, got {len(result)}")
    return result


def validate_candidate_and_plans(
    *,
    cohort_path: Path,
    candidate_protocol_path: Path,
    plans_path: Path,
    planner_protocol_path: Path,
    planner_postflight_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    cohort = index_question_rows(read_jsonl(cohort_path), label="candidate", require_qtype=True)
    plans = index_question_rows(read_jsonl(plans_path), label="plans", require_qtype=False)
    if set(cohort) != set(plans):
        raise ValueError("candidate/plans identity join is not exact")
    for key in cohort:
        if any(
            str(cohort[key].get(field) or "") != str(plans[key].get(field) or "")
            for field in ("dataset", "qid", "question", "question_sha256")
        ):
            raise ValueError(f"candidate/plans identity mismatch for {key}")
    qtypes = Counter(str(row["question_type"]) for row in cohort.values())
    if dict(qtypes) != EXPECTED_QTYPE_COUNTS:
        raise ValueError(f"candidate qtype quotas drifted: {dict(qtypes)}")

    candidate_protocol = read_json(candidate_protocol_path)
    if candidate_protocol.get("status") != (
        "FROZEN_GOLD_FREE_BEFORE_PLANNER_NOT_MATERIALIZED_NOT_TRAINED"
    ):
        raise ValueError("candidate protocol status drifted")
    planner_protocol = read_json(planner_protocol_path)
    if planner_protocol.get("status") != (
        "FROZEN_GOLD_FREE_PLANNER_EXECUTION_NOT_RUN_NOT_TRAINED"
    ):
        raise ValueError("planner protocol status drifted")
    postflight = read_json(planner_postflight_path)
    if postflight.get("status") != "PASS_PLANNER_STRUCTURAL_NOT_PROOFKG_MATERIALIZED_NOT_TRAINED":
        raise ValueError("planner postflight did not pass")
    gates = postflight.get("gates") or {}
    if not gates or not all(value is True for value in gates.values()):
        raise ValueError("planner postflight gates are not all true")
    summary = postflight.get("summary") or {}
    if (
        int(summary.get("n_input", -1)) != EXPECTED_N
        or int(summary.get("n_predictions", -1)) != EXPECTED_N
        or int((summary.get("integrity") or {}).get("runtime_errors", -1)) != 0
    ):
        raise ValueError("planner postflight counts drifted")
    prediction_ref = (postflight.get("inputs") or {}).get("predictions") or {}
    if str(prediction_ref.get("md5") or "") != md5_file(plans_path):
        raise ValueError("planner postflight/predictions mismatch")
    return cohort, plans


def validate_v6_store(store_dir: Path, cohort_path: Path) -> dict[str, Any]:
    manifest_path = store_dir / "store_manifest.json"
    aliases_path = store_dir / "aliases.jsonl"
    edges_path = store_dir / "edges.jsonl"
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != "versioned-2wiki-evidence-store-1"
        or manifest.get("status") != "COMPLETE_NOT_EVALUATED"
        or manifest.get("experiment_id")
        != "VERSIONED-2WIKI-EVIDENCE-STORE-V6-MIXED3-V4-COMPLETE-LEDGER-SEED42"
    ):
        raise ValueError("wrong or incomplete v6 evidence store")
    protected = manifest.get("protected_ledger") or {}
    if (
        protected.get("complete") is not True
        or protected.get("current_family_recomputed") is not True
        or not HEX64_RE.fullmatch(str(protected.get("protected_identities_sha256") or ""))
    ):
        raise ValueError("v6 complete-ledger binding is missing")
    for name, path in (("aliases", aliases_path), ("edges", edges_path)):
        identity = (manifest.get("outputs") or {}).get(name) or {}
        if (
            str(identity.get("path") or "") != str(path.resolve())
            or str(identity.get("md5") or "") != md5_file(path)
        ):
            raise ValueError(f"v6 {name} identity mismatch")
    cohort_md5 = md5_file(cohort_path)
    excluded = (manifest.get("inputs") or {}).get("excluded_cohorts") or []
    if not any(
        str(item.get("path") or "") == str(cohort_path.resolve())
        and str(item.get("md5") or "") == cohort_md5
        for item in excluded
    ):
        raise ValueError("v6 store did not bind/exclude the official-raw n1500 cohort")
    return {
        "path": str(store_dir.resolve()),
        "store_manifest": file_identity(manifest_path),
        "aliases": file_identity(aliases_path),
        "edges": file_identity(edges_path),
        "protected_ledger": dict(protected),
    }


def validate_planned_root_protocol(
    *, root_protocol_path: Path, planned_root_dir: Path, v6_store: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the root method before its network result is observed."""

    protocol = read_json(root_protocol_path)
    if (
        protocol.get("schema_version") != "2wiki-full-root-anchor-resolution-protocol-v2"
        or protocol.get("status")
        != "FROZEN_ALL_ROOTS_BEFORE_NETWORK_NO_GOLD_NOT_TRAINED"
    ):
        raise ValueError("root resolver protocol is not the frozen v2 method")
    if (
        str((protocol.get("outputs") or {}).get("planned_materialized_output_dir") or "")
        != str(planned_root_dir.resolve())
    ):
        raise ValueError("planned root output path drift")
    worklist = (protocol.get("outputs") or {}).get("worklist") or {}
    resolver_code = (protocol.get("inputs") or {}).get("resolver_implementation") or {}
    protocol_store = (protocol.get("inputs") or {}).get("v6_store_manifest") or {}
    protocol_aliases = (protocol.get("inputs") or {}).get("v6_aliases") or {}
    assert_file_identity(
        worklist, Path(worklist.get("path") or ""), label="planned root worklist"
    )
    assert_file_identity(
        resolver_code,
        Path(resolver_code.get("path") or ""),
        label="planned root implementation",
    )
    assert_file_identity(
        protocol_store,
        Path(v6_store["store_manifest"]["path"]),
        label="planned root v6 manifest",
    )
    assert_file_identity(
        protocol_aliases,
        Path(v6_store["aliases"]["path"]),
        label="planned root v6 aliases",
    )
    counts = protocol.get("counts") or {}
    if (
        int(counts.get("questions_total", -1)) != EXPECTED_N
        or int(counts.get("questions_recognized", -1)) < int(0.97 * EXPECTED_N)
        or int(counts.get("root_anchor_occurrences", -1)) <= 0
    ):
        raise ValueError("planned root protocol count contract failed")
    return {
        "protocol": file_identity(root_protocol_path),
        "worklist": file_identity(Path(worklist["path"])),
        "resolver_implementation": file_identity(Path(resolver_code["path"])),
        "output_dir": str(planned_root_dir.resolve()),
        "required_schema": "2wiki-full-root-anchor-resolution-result-v2",
        "required_pass_status": "PASS_ROOT_ANCHOR_CONTINUE_GATE_EXACT_CONSUMER",
        "counts": {
            "questions_total": int(counts["questions_total"]),
            "questions_recognized": int(counts["questions_recognized"]),
            "root_anchor_occurrences": int(counts["root_anchor_occurrences"]),
        },
    }


def _load_exact_cache(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    sort_keys: list[tuple[str, str, str]] = []
    for row in read_jsonl(path):
        if set(row) != {"label", "qid"}:
            raise ValueError(f"non-canonical exact cache row in {path}")
        label = str(row.get("label") or "").strip()
        qid = str(row.get("qid") or "").strip()
        key = label.casefold()
        if not label or not QID_RE.fullmatch(qid):
            raise ValueError(f"invalid exact cache row in {path}")
        if key in mapping:
            raise ValueError(f"duplicate exact cache key {label!r} in {path}")
        mapping[key] = qid
        sort_keys.append((key, label, qid))
    if sort_keys != sorted(sort_keys):
        raise ValueError(f"exact cache is not deterministically sorted: {path}")
    return mapping


def _load_aliases(path: Path) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}
    for row in read_jsonl(path):
        key = str(row.get("normalized_alias") or "")
        qids = {
            str(candidate.get("qid") or "")
            for candidate in row.get("candidates") or []
            if QID_RE.fullmatch(str(candidate.get("qid") or ""))
        }
        if not key or key in aliases:
            raise ValueError(f"invalid/duplicate v6 alias key {key!r}")
        aliases[key] = qids
    return aliases


def exact_consumer_resolution(
    *,
    surface: str,
    completed_surface: str,
    title_cache: Mapping[str, str],
    v6_aliases: Mapping[str, set[str]],
    entity_cache: Mapping[str, str],
) -> tuple[str, str | None]:
    """Replay the final executor's exact root lookup and precedence."""

    qid = title_cache.get(completed_surface.strip().casefold())
    if qid:
        return "new_exact_title_cache", qid
    alias_qids = v6_aliases.get(normalize_alias(completed_surface), set())
    if len(alias_qids) == 1:
        return "clean_v6_exact_alias", next(iter(alias_qids))
    qid = entity_cache.get(surface.strip().casefold())
    if qid:
        return "new_exact_entity_cache", qid
    return "abstain", None


def _dry_qid(row: Mapping[str, Any]) -> str | None:
    value = row.get("dry_run_qid")
    if value in (None, ""):
        value = row.get("resolved_qid")
    text = str(value or "").strip()
    return text or None


def validate_root_resolution(
    *, root_protocol_path: Path, root_dir: Path, v6_store: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    report_path = root_dir / "report.json"
    manifest_path = root_dir / "manifest.json"
    results_path = root_dir / "resolution_results.jsonl"
    dry_run_path = root_dir / "consumer_dry_run.jsonl"
    title_path = root_dir / "title_cache.jsonl"
    entity_path = root_dir / "entity_cache.jsonl"
    required = (
        root_protocol_path,
        report_path,
        manifest_path,
        results_path,
        dry_run_path,
        title_path,
        entity_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol = read_json(root_protocol_path)
    report = read_json(report_path)
    manifest = read_json(manifest_path)
    if protocol.get("schema_version") != "2wiki-full-root-anchor-resolution-protocol-v2":
        raise ValueError("unexpected root-resolution protocol schema")
    if protocol.get("status") != "FROZEN_ALL_ROOTS_BEFORE_NETWORK_NO_GOLD_NOT_TRAINED":
        raise ValueError("root-resolution protocol is not the frozen pre-network version")
    if (
        str((protocol.get("outputs") or {}).get("planned_materialized_output_dir") or "")
        != str(root_dir.resolve())
    ):
        raise ValueError("root-resolution output differs from its frozen protocol")
    resolver_code = (protocol.get("inputs") or {}).get("resolver_implementation") or {}
    assert_file_identity(
        resolver_code,
        Path(resolver_code.get("path") or ""),
        label="root:resolver implementation",
    )
    protocol_store = (protocol.get("inputs") or {}).get("v6_store_manifest") or {}
    protocol_aliases = (protocol.get("inputs") or {}).get("v6_aliases") or {}
    assert_file_identity(
        protocol_store,
        Path(v6_store["store_manifest"]["path"]),
        label="root protocol:v6 store manifest",
    )
    assert_file_identity(
        protocol_aliases,
        Path(v6_store["aliases"]["path"]),
        label="root protocol:v6 aliases",
    )
    if report.get("schema_version") != "2wiki-full-root-anchor-resolution-result-v2":
        raise ValueError("unexpected root-resolution result schema")
    if report.get("status") != "PASS_ROOT_ANCHOR_CONTINUE_GATE_EXACT_CONSUMER":
        raise ValueError("root-resolution result did not pass")
    if manifest.get("status") != report.get("status"):
        raise ValueError("root-resolution manifest/report status mismatch")
    gates = report.get("gates") or {}
    required_true = (
        "question_identity_join_eq_1",
        "request_result_join_eq_1",
        "recognized_plan_rate_ge_0_97",
        "runtime_errors_zero",
        "gold_access_false",
        "v6_binding_exact",
        "worklist_all_recognized_roots_exact",
        "projection_equals_dry_run_every_occurrence",
        "all_roots_resolved_question_rate_ge_0_80",
        "anchor_occurrence_resolution_rate_ge_0_80",
        "all_pass",
    )
    if any(gates.get(name) is not True for name in required_true):
        raise ValueError("root-resolution gates are not all true")
    if gates.get("decision") not in {
        "CONTINUE_TO_CLEAN_CLOSURE",
        "CONTINUE_TO_CLEAN_CLOSURE_V1",
        "CONTINUE_TO_V6_PROPERTY_CLOSURE",
    }:
        raise ValueError("root-resolution decision is not CONTINUE")
    counts = report.get("counts") or {}
    if (
        int(counts.get("questions_total", -1)) != EXPECTED_N
        or int(counts.get("fail", -1)) != 0
        or int(counts.get("projection_dry_run_occurrence_mismatches", -1)) != 0
        or int(counts.get("projection_dry_run_occurrence_matches", -1))
        != int(counts.get("root_anchor_occurrences", -2))
    ):
        raise ValueError("root-resolution count contract failed")
    rates = report.get("rates") or {}
    if (
        float(rates.get("projection_dry_run_occurrence_match_rate", -1)) != 1.0
        or float(rates.get("dry_run_all_roots_resolved_question_rate_all_questions", -1))
        < 0.80
        or float(rates.get("dry_run_anchor_occurrence_resolution_rate", -1)) < 0.80
    ):
        raise ValueError("root-resolution rate contract failed")

    report_outputs = report.get("outputs") or {}
    for name, path in (
        ("resolution_results", results_path),
        ("consumer_dry_run", dry_run_path),
        ("title_cache", title_path),
        ("entity_cache", entity_path),
    ):
        expected = report_outputs.get(name) or {}
        assert_file_identity(expected, path, label=f"root:{name}")
    worklist_path = Path(((protocol.get("outputs") or {}).get("worklist") or {}).get("path") or "")
    worklist_rows = read_jsonl(worklist_path)
    worklist_by_id: dict[str, dict[str, Any]] = {}
    occurrence_fields = (
        "request_id",
        "question_key",
        "dataset",
        "qid",
        "question_sha256",
        "root_position",
        "root_anchor_surface",
        "completed_root_anchor_surface",
    )
    for row in worklist_rows:
        request_id = str(row.get("request_id") or "")
        if not request_id or request_id in worklist_by_id or row.get("gold_access") is not False:
            raise ValueError("invalid/duplicate frozen root worklist row")
        worklist_by_id[request_id] = row
    if len(worklist_by_id) != int(counts.get("root_anchor_occurrences", -1)):
        raise ValueError("frozen root worklist count mismatch")

    result_rows = read_jsonl(results_path)
    results_by_id: dict[str, dict[str, Any]] = {}
    for row in result_rows:
        request_id = str(row.get("request_id") or "")
        source = worklist_by_id.get(request_id)
        if source is None or request_id in results_by_id:
            raise ValueError("root result/worklist join is not exact")
        if any(str(row.get(field)) != str(source.get(field)) for field in occurrence_fields):
            raise ValueError(f"root result identity mismatch: {request_id}")
        if row.get("gold_access") is not False:
            raise ValueError("root result gold_access must be false")
        results_by_id[request_id] = row
    if set(results_by_id) != set(worklist_by_id):
        raise ValueError("root result/worklist request IDs differ")

    title_cache = _load_exact_cache(title_path)
    entity_cache = _load_exact_cache(entity_path)
    for key in set(title_cache) & set(entity_cache):
        if title_cache[key] != entity_cache[key]:
            raise ValueError(f"root title/entity cache conflict for {key!r}")
    aliases = _load_aliases(Path(v6_store["aliases"]["path"]))
    dry_rows = read_jsonl(dry_run_path)
    dry_index: dict[str, dict[str, Any]] = {}
    for row in dry_rows:
        request_id = str(row.get("request_id") or "")
        source_row = worklist_by_id.get(request_id)
        surface = str(row.get("root_anchor_surface") or "").strip()
        completed = str(
            row.get("completed_root_anchor_surface")
            or complete_question_surface_title(surface, str(row.get("question") or ""))
        ).strip()
        if source_row is None or not surface or request_id in dry_index:
            raise ValueError("invalid/duplicate root consumer dry-run row")
        if any(str(row.get(field)) != str(source_row.get(field)) for field in occurrence_fields):
            raise ValueError(f"root dry-run identity mismatch: {request_id}")
        expected_source, expected_qid = exact_consumer_resolution(
            surface=surface,
            completed_surface=completed,
            title_cache=title_cache,
            v6_aliases=aliases,
            entity_cache=entity_cache,
        )
        if _dry_qid(row) != expected_qid:
            raise ValueError(f"root dry-run does not match exact consumer: {request_id}")
        materialized = results_by_id[request_id]
        projected_qid = (
            str(materialized.get("resolved_qid") or "").strip()
            if materialized.get("outcome") == "positive"
            else ""
        )
        if str(row.get("projected_qid") or "").strip() != projected_qid:
            raise ValueError(f"root projected result/dry-run mismatch: {request_id}")
        stated_source = str(row.get("dry_run_source") or row.get("resolution_source") or "")
        if stated_source and stated_source != expected_source:
            raise ValueError(f"root dry-run source mismatch: {request_id}")
        if row.get("gold_access") is not False:
            raise ValueError("root dry-run gold_access must be false")
        dry_index[request_id] = dict(row)
    if len(dry_index) != int(counts.get("root_anchor_occurrences", -1)):
        raise ValueError("root dry-run occurrence count mismatch")

    v6_manifest_ref = (report.get("inputs") or {}).get("v6_store_manifest") or {}
    if v6_manifest_ref:
        assert_file_identity(
            v6_manifest_ref,
            Path(v6_store["store_manifest"]["path"]),
            label="root:v6 store manifest",
        )
    report_protocol_ref = (report.get("inputs") or {}).get("protocol") or {}
    report_worklist_ref = (report.get("inputs") or {}).get("worklist") or {}
    protocol_worklist_ref = (protocol.get("outputs") or {}).get("worklist") or {}
    if report_protocol_ref:
        assert_file_identity(
            report_protocol_ref, root_protocol_path, label="root report:protocol"
        )
    if report_worklist_ref:
        assert_file_identity(report_worklist_ref, worklist_path, label="root report:worklist")
    assert_file_identity(
        protocol_worklist_ref,
        Path(protocol_worklist_ref.get("path") or ""),
        label="root protocol:worklist",
    )
    locks = {
        "protocol": file_identity(root_protocol_path),
        "report": file_identity(report_path),
        "manifest": file_identity(manifest_path),
        "resolution_results": file_identity(results_path),
        "consumer_dry_run": file_identity(dry_run_path),
        "title_cache": file_identity(title_path),
        "entity_cache": file_identity(entity_path),
    }
    return report, locks, dry_index


def build_closure_command(lock: Mapping[str, Any]) -> list[str]:
    inputs = lock["inputs"]
    policy = lock["closure_policy"]
    outputs = lock["outputs"]
    root = inputs["root_resolution"]
    return [
        sys.executable,
        str(ROOT / "scripts/prepare/run_inference_proofkg_closure.py"),
        "--plans",
        inputs["plans"]["path"],
        "--protocol",
        inputs["planner_protocol"]["path"],
        "--entity_index",
        inputs["no_local_entity_index"]["path"],
        "--entity_cache",
        root["entity_cache"]["path"],
        "--title_cache",
        root["title_cache"]["path"],
        "--exact_entity_cache_only",
        "--base_historical_cache",
        inputs["historical_seed_cache"]["path"],
        "--dataset",
        DATASET,
        "--versioned_alias_store",
        inputs["v6_store"]["path"],
        "--out",
        outputs["run_dir"],
        "--experiment_id",
        lock["experiment_id"],
        "--max_rounds",
        str(policy["max_rounds"]),
        "--cutoff",
        policy["cutoff"],
        "--workers",
        str(policy["workers"]),
        "--delay",
        str(policy["delay"]),
        "--timeout",
        str(policy["timeout"]),
        "--retries",
        str(policy["retries"]),
    ]


def preregister(
    *,
    cohort_path: Path,
    candidate_protocol_path: Path,
    plans_path: Path,
    planner_protocol_path: Path,
    planner_postflight_path: Path,
    ledger_dir: Path,
    store_dir: Path,
    planned_root_protocol: Path,
    planned_root_dir: Path,
    output_dir: Path,
    planned_lock_dir: Path,
    planned_run_dir: Path,
    planned_result_dir: Path,
    experiment_id: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite preregistration: {output_dir}")
    for path in (planned_lock_dir, planned_run_dir, planned_result_dir):
        if path.exists():
            raise FileExistsError(f"planned append-only output already exists: {path}")
    cohort, plans = validate_candidate_and_plans(
        cohort_path=cohort_path,
        candidate_protocol_path=candidate_protocol_path,
        plans_path=plans_path,
        planner_protocol_path=planner_protocol_path,
        planner_postflight_path=planner_postflight_path,
    )
    store = validate_v6_store(store_dir, cohort_path)
    root_method = validate_planned_root_protocol(
        root_protocol_path=planned_root_protocol,
        planned_root_dir=planned_root_dir,
        v6_store=store,
    )
    ledger_report = ledger_dir / "report.json"
    ledger_rows = ledger_dir / "protected_identities.question_only.jsonl"
    ledger_manifest = ledger_dir / "manifest.json"
    for path in (ledger_report, ledger_rows, ledger_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    protected = store["protected_ledger"]
    if (
        sha256_file(ledger_report) != str(protected.get("report_sha256") or "")
        or sha256_file(ledger_rows)
        != str(protected.get("protected_identities_sha256") or "")
        or sha256_file(ledger_manifest) != str(protected.get("manifest_sha256") or "")
    ):
        raise ValueError("v6/protected-ledger SHA256 binding mismatch")

    output_dir.mkdir(parents=True, exist_ok=False)
    historical_seed = output_dir / "historical_property_seed.empty.jsonl"
    historical_seed.touch(exist_ok=False)
    policy = {
        "schema_version": POLICY_SCHEMA,
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": POLICY_STATUS,
        "scope": "Gold-free official-raw 2Wiki n1500 candidate closure; not training",
        "inputs": {
            "candidate_cohort": file_identity(cohort_path),
            "candidate_protocol": file_identity(candidate_protocol_path),
            "plans": file_identity(plans_path),
            "planner_protocol": file_identity(planner_protocol_path),
            "planner_postflight": file_identity(planner_postflight_path),
            "protected_ledger": {
                "report": file_identity(ledger_report),
                "identities": file_identity(ledger_rows),
                "manifest": file_identity(ledger_manifest),
            },
            "v6_store": store,
            "historical_seed_cache": file_identity(historical_seed),
        },
        "planned_root_resolution": root_method,
        "exact_root_consumer": {
            "order": [
                "new exact title cache(completed question surface)",
                "v6 exact alias(completed question surface; unique QID only)",
                "new exact entity cache(raw surface)",
                "abstain",
            ],
            "old_resolved_bit_inherited": False,
            "legacy_local_index_allowed": False,
            "projection_equals_root_dry_run_every_occurrence": True,
            "root_dry_run_equals_final_runtime_every_occurrence": True,
        },
        "closure_policy": {
            "dataset": DATASET,
            "max_rounds": 4,
            "cutoff": CUTOFF,
            "workers": 2,
            "delay": 0.4,
            "timeout": 12.0,
            "retries": 3,
            "exact_entity_cache_only": True,
            "store_first_historical_fallback": True,
            "historical_seed": "fresh empty versioned cache",
            "negative_results_not_retried": True,
            "overwrite": False,
        },
        "postflight_gates": {
            "identity_join_rate": 1.0,
            "n": EXPECTED_N,
            "runtime_errors": 0,
            "gold_access": False,
            "planner_schema_valid_rate": {"op": ">=", "value": 0.97},
            "plan_recognized_rate": {"op": ">=", "value": 0.97},
            "anchor_qid_resolved_rate": {"op": ">=", "value": 0.80},
            "proof_kg_nonempty_rate": {"op": ">=", "value": 0.80},
            "complete_plan_execution_rate": {"op": ">=", "value": 0.70},
            "strict_graph_eligible_per_question_type": {"op": ">=", "value": 200},
            "max_triples_per_question": {"op": "<=", "value": 12},
            "projection_root_dry_run_runtime_join_rate": 1.0,
            "closure_stop_reason": "no_new_requests",
            "on_failure": "retain append-only FAIL result; do not select Proof800",
        },
        "strict_eligibility_telemetry": {
            "scorer": "kgproweight.reward.trajectory_source_gate.evaluate_graph_gate",
            "selection_performed_here": False,
            "gold_access": False,
            "required_per_question_type": 200,
        },
        "planned_outputs": {
            "execution_lock_dir": str(planned_lock_dir.resolve()),
            "run_dir": str(planned_run_dir.resolve()),
            "result_dir": str(planned_result_dir.resolve()),
        },
        "counts": {
            "questions": len(cohort),
            "plans": len(plans),
            "question_types": dict(EXPECTED_QTYPE_COUNTS),
        },
        "scientific_boundary": {
            "root_resolution_complete": False,
            "property_closure_started": False,
            "proof800_selected": False,
            "training_started": False,
            "semantic_correctness": "UNKNOWN_NOT_EVALUATED",
        },
        "gold_access": False,
        "network_access": False,
        "training_started": False,
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": POLICY_SCHEMA,
        "experiment_id": experiment_id,
        "status": POLICY_STATUS,
        "counts": policy["counts"],
        "method_protocol": file_identity(protocol_path),
        "historical_seed_cache": file_identity(historical_seed),
        "next_gate": "WAIT_FOR_VERSIONED_ROOT_RESOLVER_THEN_FINALIZE_EXECUTION_LOCK",
        "network_access": False,
        "gold_access": False,
        "training_started": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        output_dir,
        status=POLICY_STATUS,
        extra={
            "phase": "freeze_2wiki_official_raw_n1500_clean_closure_policy",
            "experiment_id": experiment_id,
            "protocol": file_identity(protocol_path),
            "report": file_identity(report_path),
            "gold_access": False,
            "network_access": False,
            "training_started": False,
        },
    )
    return report


def validate_policy(policy_path: Path) -> dict[str, Any]:
    policy = read_json(policy_path)
    if policy.get("schema_version") != POLICY_SCHEMA or policy.get("status") != POLICY_STATUS:
        raise ValueError("not a frozen n1500 closure policy")
    inputs = policy.get("inputs") or {}
    for name in (
        "candidate_cohort",
        "candidate_protocol",
        "plans",
        "planner_protocol",
        "planner_postflight",
        "historical_seed_cache",
    ):
        identity = inputs.get(name) or {}
        assert_file_identity(identity, Path(identity.get("path") or ""), label=name)
    for name, identity in (inputs.get("protected_ledger") or {}).items():
        assert_file_identity(identity, Path(identity.get("path") or ""), label=f"ledger:{name}")
    actual_store = validate_v6_store(
        Path(inputs["v6_store"]["path"]), Path(inputs["candidate_cohort"]["path"])
    )
    if actual_store != inputs["v6_store"]:
        raise ValueError("method-policy v6 store identity drift")
    planned_root = policy.get("planned_root_resolution") or {}
    actual_root_method = validate_planned_root_protocol(
        root_protocol_path=Path((planned_root.get("protocol") or {}).get("path") or ""),
        planned_root_dir=Path(planned_root.get("output_dir") or ""),
        v6_store=inputs["v6_store"],
    )
    if actual_root_method != planned_root:
        raise ValueError("method-policy root-resolver identity drift")
    return policy


def finalize_execution_lock(
    *, policy_path: Path, output_dir: Path
) -> dict[str, Any]:
    policy = validate_policy(policy_path)
    planned = policy["planned_outputs"]
    if output_dir.resolve() != Path(planned["execution_lock_dir"]).resolve():
        raise ValueError("execution-lock path differs from frozen plan")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite execution lock: {output_dir}")
    for key in ("run_dir", "result_dir"):
        if Path(planned[key]).exists():
            raise FileExistsError(f"planned {key} already exists")
    root_protocol_path = Path(policy["planned_root_resolution"]["protocol"]["path"])
    root_dir = Path(policy["planned_root_resolution"]["output_dir"])
    root_report, root_locks, _ = validate_root_resolution(
        root_protocol_path=root_protocol_path,
        root_dir=root_dir,
        v6_store=policy["inputs"]["v6_store"],
    )
    no_local_index = root_dir / "NO_LOCAL_ENTITY_INDEX_ALLOWED.json"
    if no_local_index.exists():
        raise ValueError("exact final consumer forbids a local entity index")
    code_paths = (
        ROOT / "scripts/prepare/run_inference_proofkg_closure.py",
        ROOT / "scripts/pilot/build_automatic_proofkg_from_plans.py",
        ROOT / "scripts/prepare/freeze_2wiki_official_raw_n1500_clean_closure_v1.py",
        ROOT / "scripts/prepare/run_2wiki_official_raw_n1500_clean_closure_v1_locked.py",
        ROOT / "kgproweight/kg/store_first_combined_retriever.py",
        ROOT / "kgproweight/kg/historical_wikidata_retriever.py",
        ROOT / "kgproweight/kg/versioned_evidence_store.py",
        ROOT / "kgproweight/kg/wikipedia_title_resolver.py",
        ROOT / "kgproweight/reward/trajectory_source_gate.py",
    )
    lock = {
        "schema_version": LOCK_SCHEMA,
        "experiment_id": policy["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": LOCK_STATUS,
        "policy_protocol": file_identity(policy_path),
        "inputs": {
            **policy["inputs"],
            "root_resolution": root_locks,
            "no_local_entity_index": {
                "path": str(no_local_index.resolve()),
                "must_be_absent": True,
            },
            "code": {path.name: file_identity(path) for path in code_paths},
        },
        "closure_policy": policy["closure_policy"],
        "postflight_gates": policy["postflight_gates"],
        "exact_root_consumer": policy["exact_root_consumer"],
        "root_gate_snapshot": {
            "counts": root_report["counts"],
            "rates": root_report["rates"],
            "gates": root_report["gates"],
        },
        "outputs": {
            "run_dir": planned["run_dir"],
            "result_dir": planned["result_dir"],
        },
        "checks": {
            "policy_identity_exact": True,
            "root_resolution_pass": True,
            "root_projection_equals_dry_run_every_occurrence": True,
            "root_rate_gates_pass": True,
            "v6_store_complete_ledger_bound": True,
            "historical_seed_empty": Path(policy["inputs"]["historical_seed_cache"]["path"]).stat().st_size == 0,
            "exact_entity_cache_only": True,
            "local_entity_index_absent": True,
            "planned_outputs_absent": True,
            "gold_access_false": True,
            "training_not_started": True,
        },
        "scientific_boundary": {
            "strict_candidate_materialization_only": True,
            "proof800_selection_not_performed": True,
            "semantic_correctness": "UNKNOWN_NOT_EVALUATED",
            "training_started": False,
        },
        "gold_access": False,
        "network_access": False,
        "training_started": False,
    }
    if not all(lock["checks"].values()):
        raise RuntimeError(f"execution-lock checks failed: {lock['checks']}")
    lock["closure_command"] = build_closure_command(lock)
    output_dir.mkdir(parents=True, exist_ok=False)
    lock_path = output_dir / "protocol.json"
    lock_path.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": LOCK_SCHEMA,
        "experiment_id": lock["experiment_id"],
        "status": LOCK_STATUS,
        "checks": lock["checks"],
        "execution_lock": file_identity(lock_path),
        "root_resolution": root_locks,
        "closure_command": lock["closure_command"],
        "network_access": False,
        "gold_access": False,
        "training_started": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        output_dir,
        status=LOCK_STATUS,
        extra={
            "phase": "finalize_2wiki_official_raw_n1500_clean_closure_lock",
            "experiment_id": lock["experiment_id"],
            "protocol": file_identity(lock_path),
            "report": file_identity(report_path),
            "network_access": False,
            "gold_access": False,
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    pre = subparsers.add_parser("preregister")
    pre.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    pre.add_argument("--candidate-protocol", type=Path, default=DEFAULT_CANDIDATE_PROTOCOL)
    pre.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    pre.add_argument("--planner-protocol", type=Path, default=DEFAULT_PLANNER_PROTOCOL)
    pre.add_argument("--planner-postflight", type=Path, default=DEFAULT_PLANNER_POSTFLIGHT)
    pre.add_argument("--ledger-dir", type=Path, default=DEFAULT_LEDGER_DIR)
    pre.add_argument("--v6-store", type=Path, default=DEFAULT_STORE)
    pre.add_argument("--planned-root-protocol", type=Path, default=DEFAULT_ROOT_PROTOCOL)
    pre.add_argument("--planned-root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    pre.add_argument("--output-dir", type=Path, default=DEFAULT_POLICY_DIR)
    pre.add_argument("--planned-lock-dir", type=Path, default=DEFAULT_LOCK_DIR)
    pre.add_argument("--planned-run-dir", type=Path, default=DEFAULT_RUN_DIR)
    pre.add_argument("--planned-result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    pre.add_argument("--experiment-id", default=EXPERIMENT_ID)
    final = subparsers.add_parser("finalize")
    final.add_argument("--policy", type=Path, default=DEFAULT_POLICY_DIR / "protocol.json")
    final.add_argument("--output-dir", type=Path, default=DEFAULT_LOCK_DIR)
    args = parser.parse_args()
    if args.phase == "preregister":
        report = preregister(
            cohort_path=args.cohort,
            candidate_protocol_path=args.candidate_protocol,
            plans_path=args.plans,
            planner_protocol_path=args.planner_protocol,
            planner_postflight_path=args.planner_postflight,
            ledger_dir=args.ledger_dir,
            store_dir=args.v6_store,
            planned_root_protocol=args.planned_root_protocol,
            planned_root_dir=args.planned_root_dir,
            output_dir=args.output_dir,
            planned_lock_dir=args.planned_lock_dir,
            planned_run_dir=args.planned_run_dir,
            planned_result_dir=args.planned_result_dir,
            experiment_id=args.experiment_id,
        )
    else:
        report = finalize_execution_lock(policy_path=args.policy, output_dir=args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
