#!/usr/bin/env python3
"""Freeze the clean, exact-root 2Wiki n300 closure-v2 execution lock.

This is a CPU-only preregistration step.  It verifies the completed root
resolver, the isolated root caches, the clean v5 evidence store, the frozen
planner inputs, and the closure-v1 historical property cache.  It then writes
an append-only lock containing exact SHA256 identities and the only permitted
closure command.  It never executes the closure or accesses the network.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.utils.logging import dump_manifest


SCHEMA_VERSION = "2wiki-extension-clean-closure-v2-lock-1"
STATUS = "FROZEN_DIAGNOSTIC_CANDIDATE_BUILD_BEFORE_NETWORK"
DATASET = "2wikimultihopqa"
CUTOFF = "2020-12-09T23:59:59Z"
QID_RE = re.compile(r"^Q[1-9][0-9]*$")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLANS = ROOT / (
    "outputs/validation/2wiki_proofkg_extension_v1b_n300_seed42_plans/"
    "predictions.question_only.jsonl"
)
DEFAULT_PLANNER_PROTOCOL = ROOT / (
    "outputs/audits/2wiki_proofkg_extension_v1b_n300_seed42_preregistration/"
    "protocol.json"
)
DEFAULT_RESOLVER_PROTOCOL = ROOT / (
    "outputs/audits/2wiki_proofkg_extension_v1b_n300_root_anchor_resolution_v1_"
    "preregistration/protocol.json"
)
DEFAULT_RESOLVER_DIR = ROOT / (
    "indexes/2wiki_proofkg_extension_v1b_n300_root_anchor_resolution_v1"
)
DEFAULT_STORE = ROOT / "indexes/versioned_2wiki_evidence_store_v5_mixed3_v4_seed42"
DEFAULT_HISTORICAL = ROOT / (
    "data/derived/2wiki_proofkg_extension_v1b_n300_closure_v1/"
    "closure_historical_property_cache.jsonl"
)
DEFAULT_CLOSURE_V1_REPORT = ROOT / (
    "data/derived/2wiki_proofkg_extension_v1b_n300_closure_v1/closure_report.json"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/audits/2wiki_proofkg_extension_v1b_n300_clean_closure_v2_"
    "preregistration"
)
DEFAULT_RUN = ROOT / "data/derived/2wiki_proofkg_extension_v1b_n300_closure_v2"
DEFAULT_ATTESTATION = ROOT / (
    "outputs/audits/2wiki_proofkg_extension_v1b_n300_clean_closure_v2_result"
)
EXPERIMENT_ID = "2WIKI-PROOFKG-EXTENSION-V1B-N300-CLEAN-CLOSURE-V2"


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


def file_lock(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def assert_identity(identity: Mapping[str, Any], path: Path, *, label: str) -> None:
    actual = file_lock(path)
    if str(identity.get("path") or "") != actual["path"]:
        raise ValueError(f"{label} path mismatch")
    if str(identity.get("sha256") or "") != actual["sha256"]:
        raise ValueError(f"{label} SHA256 mismatch")
    expected_size = identity.get("size_bytes")
    if expected_size is None or int(expected_size) != actual["size_bytes"]:
        raise ValueError(f"{label} size mismatch")


def validate_canonical_cache(path: Path) -> dict[str, str]:
    rows = read_jsonl(path)
    mapping: dict[str, str] = {}
    sort_keys: list[tuple[str, str, str]] = []
    for row in rows:
        if set(row) != {"label", "qid"}:
            raise ValueError(f"non-canonical cache row in {path}")
        label = str(row["label"]).strip()
        qid = str(row["qid"]).strip()
        key = label.casefold()
        if not label or not QID_RE.fullmatch(qid):
            raise ValueError(f"invalid cache label/QID in {path}")
        if key in mapping and mapping[key] != qid:
            raise ValueError(f"cache label maps to multiple QIDs: {label!r}")
        if key in mapping:
            raise ValueError(f"duplicate normalized cache label: {label!r}")
        mapping[key] = qid
        sort_keys.append((key, label, qid))
    if sort_keys != sorted(sort_keys):
        raise ValueError(f"cache is not deterministically sorted: {path}")
    return mapping


def index_plan_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        dataset = str(row.get("dataset") or "")
        qid = str(row.get("qid") or "")
        question = str(row.get("question") or "").strip()
        key = question_key(dataset, qid)
        if dataset != DATASET or not qid or not question:
            raise ValueError(f"invalid planner identity: {key}")
        if str(row.get("question_key") or "") != key:
            raise ValueError(f"planner question_key mismatch: {key}")
        if str(row.get("question_sha256") or "") != question_sha256(question):
            raise ValueError(f"planner question hash mismatch: {key}")
        if row.get("gold_access") is not False:
            raise ValueError(f"planner gold_access must be false: {key}")
        if key in result:
            raise ValueError(f"duplicate planner identity: {key}")
        result[key] = row
    if len(result) != 300:
        raise ValueError(f"clean closure-v2 requires exactly 300 planner rows, got {len(result)}")
    return result


def validate_resolver_outputs(
    *, resolver_dir: Path, resolver_protocol_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = resolver_dir / "report.json"
    manifest_path = resolver_dir / "manifest.json"
    results_path = resolver_dir / "resolution_results.jsonl"
    title_path = resolver_dir / "title_cache.jsonl"
    entity_path = resolver_dir / "entity_cache.jsonl"
    report = read_json(report_path)
    manifest = read_json(manifest_path)
    resolver_protocol = read_json(resolver_protocol_path)

    if report.get("schema_version") != "2wiki-root-anchor-resolution-result-v1":
        raise ValueError("unexpected root resolver result schema")
    if report.get("status") != "PASS_ROOT_ANCHOR_CONTINUE_GATE":
        raise ValueError("root resolver did not pass its preregistered continue gate")
    gate = report.get("continue_gate_before_clean_closure_v2") or {}
    if gate.get("all_pass") is not True or gate.get("decision") != "CONTINUE_TO_CLEAN_CLOSURE_V2":
        raise ValueError("root resolver continuation decision is not PASS")
    checks = report.get("checks") or {}
    if (
        float(checks.get("request_log_join_rate", -1)) != 1.0
        or int(checks.get("runtime_errors", -1)) != 0
        or checks.get("gold_access_false") is not True
        or checks.get("old_cache_fallback") is not False
    ):
        raise ValueError("root resolver safety checks failed")
    if report.get("gold_access") is not False or report.get("training_started") is not False:
        raise ValueError("root resolver provenance is unsafe")
    if manifest.get("status") != report.get("status"):
        raise ValueError("root resolver manifest/report status mismatch")
    if resolver_protocol.get("status") != "FROZEN_BEFORE_NETWORK_NO_GOLD":
        raise ValueError("root resolver protocol was not preregistered")
    planned = ((resolver_protocol.get("outputs") or {}).get("planned_resolution_output_dir"))
    if str(planned or "") != str(resolver_dir.resolve()):
        raise ValueError("resolver output directory differs from preregistration")
    assert_identity(
        (report.get("inputs") or {}).get("protocol") or {},
        resolver_protocol_path,
        label="resolver protocol",
    )
    for name, path in (
        ("resolution_results", results_path),
        ("title_cache", title_path),
        ("entity_cache", entity_path),
    ):
        assert_identity((report.get("outputs") or {}).get(name) or {}, path, label=name)

    title_cache = validate_canonical_cache(title_path)
    entity_cache = validate_canonical_cache(entity_path)
    for label in set(title_cache) & set(entity_cache):
        if title_cache[label] != entity_cache[label]:
            raise ValueError(f"title/entity cache conflict for {label!r}")

    results = read_jsonl(results_path)
    if len(results) != int((report.get("counts") or {}).get("results", -1)):
        raise ValueError("resolver result count differs from report")
    if len({str(row.get("request_id") or "") for row in results}) != len(results):
        raise ValueError("duplicate resolver request_id")
    expected_title: dict[str, str] = {}
    expected_entity: dict[str, str] = {}
    for row in results:
        if row.get("gold_access") is not False:
            raise ValueError("gold_access drift in resolver results")
        if row.get("outcome") != "positive":
            continue
        qid = str(row.get("resolved_qid") or "")
        if not QID_RE.fullmatch(qid):
            raise ValueError("positive resolver result has invalid QID")
        method = str(row.get("resolution_method") or "")
        if method == "exact_wikipedia_title":
            label = str(row.get("completed_root_anchor_surface") or "").strip()
            target = expected_title
        elif method == "wikidata_question_context":
            label = str(row.get("root_anchor_surface") or "").strip()
            target = expected_entity
        else:
            raise ValueError(f"unsupported positive resolution method: {method!r}")
        key = label.casefold()
        if key in target and target[key] != qid:
            raise ValueError("accepted resolver results conflict on a cache key")
        target[key] = qid
    if title_cache != expected_title or entity_cache != expected_entity:
        raise ValueError("canonical resolver caches do not exactly project positive decisions")

    return report, {
        "report": file_lock(report_path),
        "manifest": file_lock(manifest_path),
        "resolution_results": file_lock(results_path),
        "title_cache": file_lock(title_path),
        "entity_cache": file_lock(entity_path),
    }


def validate_clean_store(store_dir: Path, plan_index: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    manifest_path = store_dir / "store_manifest.json"
    aliases_path = store_dir / "aliases.jsonl"
    edges_path = store_dir / "edges.jsonl"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "versioned-2wiki-evidence-store-1":
        raise ValueError("unexpected clean evidence-store schema")
    if manifest.get("status") != "COMPLETE_NOT_EVALUATED":
        raise ValueError("clean evidence store is not complete")
    if manifest.get("experiment_id") != "VERSIONED-2WIKI-EVIDENCE-STORE-V5-MIXED3-V4-SEED42":
        raise ValueError("wrong evidence store; clean v5 is required")
    for name, path in (("aliases", aliases_path), ("edges", edges_path)):
        identity = (manifest.get("outputs") or {}).get(name) or {}
        if str(identity.get("path") or "") != str(path.resolve()):
            raise ValueError(f"clean store {name} path mismatch")
        if str(identity.get("md5") or "") != md5_file(path):
            raise ValueError(f"clean store {name} MD5 mismatch")

    combined = None
    for identity in (manifest.get("inputs") or {}).get("excluded_cohorts") or []:
        if "2wiki_proofkg_extension_combined_v1_n350" in str(identity.get("path") or ""):
            combined = identity
            break
    if combined is None:
        raise ValueError("clean v5 store did not exclude the extension combined350 cohort")
    combined_path = Path(str(combined["path"])).resolve()
    if not combined_path.is_file() or md5_file(combined_path) != str(combined.get("md5") or ""):
        raise ValueError("clean-store excluded combined350 cohort identity drift")
    excluded_index = index_plan_rows_subset(read_jsonl(combined_path))
    if not set(plan_index).issubset(excluded_index):
        raise ValueError("n300 planner cohort is not contained in the clean-store exclusion")
    for key, plan in plan_index.items():
        if str(excluded_index[key].get("question_sha256") or "") != str(plan["question_sha256"]):
            raise ValueError(f"clean-store excluded cohort question drift: {key}")
    return {
        "store_manifest": file_lock(manifest_path),
        "aliases": file_lock(aliases_path),
        "edges": file_lock(edges_path),
        "excluded_combined350": file_lock(combined_path),
    }


def index_plan_rows_subset(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("dataset") or "") != DATASET:
            continue
        qid = str(row.get("qid") or "")
        question = str(row.get("question") or "").strip()
        key = question_key(DATASET, qid)
        if not qid or not question or str(row.get("question_sha256") or "") != question_sha256(question):
            raise ValueError(f"invalid excluded-cohort identity: {key}")
        if key in result:
            raise ValueError(f"duplicate excluded-cohort identity: {key}")
        result[key] = row
    return result


def validate_historical_cache(path: Path, closure_v1_report_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report = read_json(closure_v1_report_path)
    if report.get("schema_version") != "inference-proofkg-closure-v3b-1":
        raise ValueError("unexpected closure-v1 report schema")
    if report.get("stop_reason") != "no_new_requests":
        raise ValueError("closure-v1 historical cache was not converged")
    if report.get("cutoff") != CUTOFF:
        raise ValueError("closure-v1 cutoff drift")
    actual = file_lock(path)
    if actual["sha256"] != str(report.get("closure_cache_sha256") or ""):
        raise ValueError("closure-v1 historical cache SHA256 mismatch")

    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("qid") or "")
            if (
                row.get("schema_version") != "wikidata-historical-entity-revision-1"
                or row.get("cutoff") != CUTOFF
                or not QID_RE.fullmatch(qid)
                or qid in seen
            ):
                raise ValueError(f"invalid/duplicate historical row at line {line_number}")
            seen.add(qid)
    if not seen:
        raise ValueError("empty historical property cache")
    return report, {"cache": actual, "closure_v1_report": file_lock(closure_v1_report_path)}


def build_closure_command(protocol: Mapping[str, Any]) -> list[str]:
    inputs = protocol["inputs"]
    output = protocol["outputs"]
    policy = protocol["closure_policy"]
    resolver = inputs["root_resolution"]
    command = [
        sys.executable,
        str(ROOT / "scripts/prepare/run_inference_proofkg_closure.py"),
        "--plans", inputs["plans"]["path"],
        "--protocol", inputs["planner_protocol"]["path"],
        "--entity_index", inputs["no_local_entity_index"]["path"],
        "--entity_cache", resolver["entity_cache"]["path"],
        "--title_cache", resolver["title_cache"]["path"],
        "--exact_entity_cache_only",
        "--base_historical_cache", inputs["historical_property_cache"]["path"],
        "--dataset", DATASET,
        "--versioned_alias_store", inputs["clean_v5_store"]["path"],
        "--out", output["run_dir"],
        "--experiment_id", protocol["experiment_id"],
        "--max_rounds", str(policy["max_rounds"]),
        "--cutoff", policy["cutoff"],
        "--workers", str(policy["workers"]),
        "--delay", str(policy["delay"]),
        "--timeout", str(policy["timeout"]),
        "--retries", str(policy["retries"]),
    ]
    return command


def freeze(
    *,
    plans_path: Path,
    planner_protocol_path: Path,
    resolver_protocol_path: Path,
    resolver_dir: Path,
    clean_store_dir: Path,
    historical_cache_path: Path,
    closure_v1_report_path: Path,
    output_dir: Path,
    planned_run_dir: Path,
    planned_attestation_dir: Path,
    experiment_id: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite lock directory: {output_dir}")
    if planned_run_dir.exists() or planned_attestation_dir.exists():
        raise FileExistsError("planned closure-v2 run/attestation path already exists")

    plans = index_plan_rows(read_jsonl(plans_path))
    resolver_report, resolver_locks = validate_resolver_outputs(
        resolver_dir=resolver_dir, resolver_protocol_path=resolver_protocol_path
    )
    store_locks = validate_clean_store(clean_store_dir, plans)
    closure_v1_report, historical_locks = validate_historical_cache(
        historical_cache_path, closure_v1_report_path
    )
    no_local_index = resolver_dir / "NO_LOCAL_ENTITY_INDEX_ALLOWED.json"
    if no_local_index.exists():
        raise ValueError("no-local-index sentinel must remain absent")

    code_paths = [
        ROOT / "scripts/prepare/run_inference_proofkg_closure.py",
        ROOT / "scripts/pilot/build_automatic_proofkg_from_plans.py",
        ROOT / "scripts/prepare/freeze_2wiki_extension_clean_closure_v2.py",
        ROOT / "scripts/prepare/run_2wiki_extension_clean_closure_v2_locked.py",
        ROOT / "kgproweight/kg/store_first_combined_retriever.py",
        ROOT / "kgproweight/kg/historical_wikidata_retriever.py",
        ROOT / "kgproweight/kg/versioned_evidence_store.py",
    ]
    locks = {
        "plans": file_lock(plans_path),
        "planner_protocol": file_lock(planner_protocol_path),
        "resolver_protocol": file_lock(resolver_protocol_path),
        "root_resolution": resolver_locks,
        "clean_v5_store": {"path": str(clean_store_dir.resolve()), **store_locks},
        "historical_property_cache": historical_locks["cache"],
        "closure_v1_report": historical_locks["closure_v1_report"],
        "no_local_entity_index": {
            "path": str(no_local_index.resolve()),
            "must_be_absent": True,
        },
        "code": {path.name: file_lock(path) for path in code_paths},
    }
    checks = {
        "planner_rows_exactly_300": len(plans) == 300,
        "planner_gold_access_false": all(row.get("gold_access") is False for row in plans.values()),
        "resolver_continue_gate_pass": resolver_report["continue_gate_before_clean_closure_v2"]["all_pass"] is True,
        "resolver_old_cache_fallback_false": resolver_report["checks"]["old_cache_fallback"] is False,
        "root_caches_canonical_and_hash_bound": True,
        "clean_v5_excludes_combined350": True,
        "historical_cache_converged_and_hash_bound": closure_v1_report["stop_reason"] == "no_new_requests",
        "no_local_entity_index_absent": not no_local_index.exists(),
        "exact_entity_cache_only": True,
        "planned_outputs_absent": True,
        "network_access_disabled_during_freeze": True,
        "training_not_started": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"clean closure-v2 freeze checks failed: {checks}")

    protocol: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "scope": (
            "2Wiki train-only extension n300; diagnostic exact-root closure-v2; "
            "clean v5 excludes combined350 but is not complete-ledger-attested; no Gold"
        ),
        "inputs": locks,
        "closure_policy": {
            "dataset": DATASET,
            "max_rounds": 4,
            "cutoff": CUTOFF,
            "workers": 2,
            "delay": 0.4,
            "timeout": 12.0,
            "retries": 3,
            "root_resolution_order": "exact title cache -> exact clean-v5 alias -> exact isolated entity cache -> abstain",
            "exact_entity_cache_only": True,
            "store_first_historical_fallback": True,
            "seed_requests": "empty; base historical cache already contains closure-v1 state",
            "overwrite": False,
        },
        "postflight_gates": {
            "identity_join_rate": 1.0,
            "n": 300,
            "runtime_errors": 0,
            "gold_access": False,
            "plan_recognized_rate": {"op": ">=", "value": 0.80},
            "anchor_qid_resolved_rate": {"op": ">=", "value": 0.80},
            "proof_kg_nonempty_rate": {"op": ">=", "value": 0.80},
            "complete_plan_execution_rate": {"op": ">=", "value": 0.70},
            "max_triples_per_question": {"op": "<=", "value": 12},
            "on_failure": "retain append-only result as FAIL_STRUCTURAL; do not enter Proof800 supply",
        },
        "scientific_boundary": {
            "v5_not_complete_ledger_attested": True,
            "diagnostic_candidate_build_only": True,
            "final_training_eligibility": False,
            "required_successor": (
                "rebuild v6 store after the final raw candidate cohort is frozen; "
                "v6 must bind the complete protected ledger, then re-execute cleanly"
            ),
        },
        "checks": checks,
        "outputs": {
            "run_dir": str(planned_run_dir.resolve()),
            "attestation_dir": str(planned_attestation_dir.resolve()),
        },
        "network_access": False,
        "training_started": False,
        "final_training_eligibility": False,
    }
    protocol["closure_command"] = build_closure_command(protocol)

    output_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "status": STATUS,
        "checks": checks,
        "protocol": file_lock(protocol_path),
        "closure_command": protocol["closure_command"],
        "network_access": False,
        "training_started": False,
        "final_training_eligibility": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        output_dir,
        status=STATUS,
        extra={
            "phase": "freeze_2wiki_extension_clean_closure_v2",
            "experiment_id": experiment_id,
            "protocol": file_lock(protocol_path),
            "report": file_lock(report_path),
            "network_access": False,
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    parser.add_argument("--planner-protocol", type=Path, default=DEFAULT_PLANNER_PROTOCOL)
    parser.add_argument("--resolver-protocol", type=Path, default=DEFAULT_RESOLVER_PROTOCOL)
    parser.add_argument("--resolver-dir", type=Path, default=DEFAULT_RESOLVER_DIR)
    parser.add_argument("--clean-store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--historical-cache", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--closure-v1-report", type=Path, default=DEFAULT_CLOSURE_V1_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--planned-run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--planned-attestation-dir", type=Path, default=DEFAULT_ATTESTATION)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    args = parser.parse_args()
    report = freeze(
        plans_path=args.plans,
        planner_protocol_path=args.planner_protocol,
        resolver_protocol_path=args.resolver_protocol,
        resolver_dir=args.resolver_dir,
        clean_store_dir=args.clean_store,
        historical_cache_path=args.historical_cache,
        closure_v1_report_path=args.closure_v1_report,
        output_dir=args.output_dir,
        planned_run_dir=args.planned_run_dir,
        planned_attestation_dir=args.planned_attestation_dir,
        experiment_id=args.experiment_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
