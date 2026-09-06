#!/usr/bin/env python3
"""Freeze the official-raw 2Wiki canonical-retrieval policy and exact scope.

The policy is frozen before the n=1500 closure result exists.  It chooses the
answer-free predicate scope rather than all 1,500 questions: retrieve every
row that passes the already frozen strict structural/source predicate, not
only the final 800.  This saves retrieval for rows that can never enter
Proof800 while retaining every eligible replacement and family-diversity
candidate.  No answer, EM/F1, semantic score, or final selection rank is used.

After a PASS closure-v3 release exists, ``freeze-scope`` projects that fixed
predicate into an append-only identity-only request set.  Retrieval itself is
performed by ``materialize_2wiki_official_raw_canonical_retrieval_v1.py``.
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

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256


ROOT = Path(__file__).resolve().parents[2]
DATASET = "2wikimultihopqa"
SEED = 42
QTYPES = ("bridge_comparison", "comparison", "compositional", "inference")
SOURCE_COUNTS = {
    "bridge_comparison": 390,
    "comparison": 390,
    "compositional": 389,
    "inference": 331,
}
MIN_PER_TYPE = 200
RETRIEVAL_STACK = (
    "E5@100+BM25@100->RRF60@50->bge-reranker-v2-m3@10->pack3860"
)
POLICY_SCHEMA = "2wiki-official-raw-canonical-retrieval-policy-v1"
POLICY_STATUS = "FROZEN_PREDICATE_SCOPE_WAITING_CLOSURE_NOT_TRAINED"
SCOPE_SCHEMA = "2wiki-official-raw-canonical-retrieval-scope-v1"
SCOPE_STATUS = "FROZEN_ANSWER_FREE_STRICT_ELIGIBLE_SCOPE_NOT_RETRIEVED_NOT_TRAINED"
REQUEST_SCHEMA = "2wiki-official-raw-canonical-retrieval-request-v1"
POLICY_EXPERIMENT_ID = "2WIKI-OFFICIAL-RAW-N1500-CANONICAL-RETRIEVAL-SCOPE-POLICY-V1"
SCOPE_EXPERIMENT_ID = "2WIKI-OFFICIAL-RAW-N1500-CANONICAL-RETRIEVAL-SCOPE-V1"
EXPECTED_CLOSURE_POLICY_SCHEMA = "2wiki-official-raw-n1500-clean-closure-policy-v3"
EXPECTED_CLOSURE_POLICY_STATUS = "FROZEN_POLICY_WAITING_FOR_ROOT_RESOLUTION"
EXPECTED_CLOSURE_REPORT_SCHEMA = "2wiki-official-raw-clean-closure-v3"
EXPECTED_CLOSURE_PASS_STATUS = (
    "COMPLETE_DIAGNOSTIC_CLEAN_CLOSURE_NOT_SELECTED_NOT_TRAINED"
)
EXPECTED_TELEMETRY_SCHEMA = "2wiki-official-raw-strict-eligibility-telemetry-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

DEFAULT_CANDIDATE_RELEASE = ROOT / (
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "seed42_preregistration"
)
DEFAULT_CLOSURE_POLICY = ROOT / (
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "clean_closure_v3_preregistration/protocol.json"
)
DEFAULT_POLICY_DIR = ROOT / (
    "outputs/audits/2wiki_official_raw_canonical_retrieval_v1_scope_policy_"
    "preregistration"
)
DEFAULT_SCOPE_DIR = ROOT / (
    "outputs/audits/2wiki_official_raw_canonical_retrieval_v1_scope_"
    "preregistration"
)

FORBIDDEN_FIELDS = {
    "answer",
    "answers",
    "gold_answer",
    "golden_answers",
    "support",
    "supporting_facts",
    "decomposition",
    "question_decomposition",
    "evidence",
    "evidences",
    "reasoning",
    "steps",
    "target",
    "teacher_output",
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
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(row)
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


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def resolve_identity(value: Mapping[str, Any], *, label: str) -> Path:
    raw = str(value.get("path") or "").strip()
    digest = str(value.get("sha256") or "").strip()
    if not raw or not HEX64.fullmatch(digest):
        raise ValueError(f"{label}: incomplete file identity")
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"{label}: missing file or SHA256 drift")
    if value.get("size_bytes") is not None and path.stat().st_size != int(
        value["size_bytes"]
    ):
        raise ValueError(f"{label}: size drift")
    return path


def forbidden_present(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).strip().lower() in FORBIDDEN_FIELDS or forbidden_present(nested)
            for key, nested in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(forbidden_present(item) for item in value)
    return False


def index_rows(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        dataset = str(row.get("dataset") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        key = str(row.get("question_key") or question_key(dataset, qid))
        if not qid or key in output:
            raise ValueError(f"{label}: empty/duplicate identity {key!r}")
        output[key] = row
    return output


def validate_candidate_release(directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol_path = directory / "protocol.json"
    report_path = directory / "report.json"
    manifest_path = directory / "manifest.json"
    protocol = read_json(protocol_path)
    report = read_json(report_path)
    manifest = read_json(manifest_path)
    status = "FROZEN_GOLD_FREE_BEFORE_PLANNER_NOT_MATERIALIZED_NOT_TRAINED"
    if (
        protocol.get("schema_version") != "2wiki-proofkg-official-raw-protocol-v2"
        or protocol.get("status") != status
        or report.get("status") != status
        or manifest.get("status") != status
    ):
        raise ValueError("official-raw n1500 candidate release schema/status failed")
    cohort_identity = report.get("output") or (report.get("outputs") or {}).get("cohort")
    if not isinstance(cohort_identity, Mapping):
        raise ValueError("candidate release does not bind cohort")
    cohort_path = resolve_identity(cohort_identity, label="candidate cohort")
    rows = read_jsonl(cohort_path)
    counts = Counter(str(row.get("question_type") or "") for row in rows)
    seen_keys: set[str] = set()
    seen_hashes: set[str] = set()
    for row in rows:
        dataset = str(row.get("dataset") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        question = str(row.get("question") or "").strip()
        key = question_key(dataset, qid)
        qhash = question_sha256(question)
        if (
            row.get("schema_version") != "2wiki-proofkg-official-raw-question-only-v2"
            or dataset != DATASET
            or not qid
            or not question
            or str(row.get("question_sha256") or "") != qhash
            or row.get("family_version") != FAMILY_VERSION
            or str(row.get("family_sha256") or "") != family_sha256(question)
            or row.get("gold_access") is not False
            or forbidden_present(row)
            or key in seen_keys
            or qhash in seen_hashes
        ):
            raise ValueError(f"candidate identity/gold boundary failed: {key}")
        seen_keys.add(key)
        seen_hashes.add(qhash)
    if len(rows) != 1500 or counts != Counter(SOURCE_COUNTS):
        raise ValueError(f"candidate population drifted: n={len(rows)}, counts={counts}")
    return rows, {
        "cohort": file_identity(cohort_path),
        "protocol": file_identity(protocol_path),
        "report": file_identity(report_path),
        "manifest": file_identity(manifest_path),
    }


def validate_closure_policy(path: Path) -> dict[str, Any]:
    policy = read_json(path)
    if (
        policy.get("schema_version") != EXPECTED_CLOSURE_POLICY_SCHEMA
        or policy.get("status") != EXPECTED_CLOSURE_POLICY_STATUS
        or policy.get("gold_access") is not False
        or policy.get("training_started") is not False
        or (policy.get("scientific_boundary") or {}).get("proof800_selected") is not False
    ):
        raise ValueError("closure-v3 policy schema/status/boundary failed")
    if int((policy.get("counts") or {}).get("questions", -1)) != 1500:
        raise ValueError("closure-v3 policy population is not n1500")
    return policy


def freeze_policy(
    *, candidate_release: Path, closure_policy_path: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite retrieval policy: {output_dir}")
    cohort, candidate_binding = validate_candidate_release(candidate_release)
    closure_policy = validate_closure_policy(closure_policy_path)
    closure_cohort = ((closure_policy.get("inputs") or {}).get("candidate_cohort") or {})
    if closure_cohort.get("sha256") != candidate_binding["cohort"]["sha256"]:
        raise ValueError("closure-v3 policy/candidate cohort binding mismatch")
    gates = {
        "candidate_n1500_exact": len(cohort) == 1500,
        "candidate_question_type_counts_exact": Counter(
            str(row["question_type"]) for row in cohort
        )
        == Counter(SOURCE_COUNTS),
        "closure_v3_policy_bound": True,
        "scope_predicate_frozen_before_closure_result": True,
        "scope_uses_no_answer_or_correctness_signal": True,
        "training_not_started": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"official-raw retrieval policy gates failed: {gates}")
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema_version": POLICY_SCHEMA,
        "experiment_id": POLICY_EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": POLICY_STATUS,
        "candidate_population": {
            "n": 1500,
            "by_question_type": SOURCE_COUNTS,
        },
        "scope_policy": {
            "choice": "all_strict_structural_eligible_rows_not_all_n1500",
            "predicate": {
                "m_graph": 1,
                "planner_schema_valid": True,
                "all_root_anchors_resolved": True,
                "all_hops_complete": True,
                "graph_nonempty": True,
                "gold_access_false": True,
                "runtime_error_zero": True,
                "provenance_complete": True,
                "retained_edges_traceable": True,
                "no_duplicate_edges": True,
            },
            "minimum_per_question_type": MIN_PER_TYPE,
            "why_fair": (
                "the candidate universe and predicate are frozen before closure; "
                "the predicate uses only train-side structural/source telemetry and "
                "never answer correctness, EM/F1, or final selection rank"
            ),
            "why_efficient": (
                "rows failing the predicate are ineligible for Proof800 by definition; "
                "retrieving them cannot alter the selected set. Every passing row is "
                "retrieved, rather than retrieving only a preferred top-800 subset, "
                "so passage failures and family diversity retain the full reserve pool"
            ),
            "failure_policy": (
                "if closure-v3 is not PASS or any question type has fewer than 200 "
                "predicate-eligible rows, do not freeze a retrieval scope"
            ),
        },
        "retrieval": {
            "stack": RETRIEVAL_STACK,
            "top_k": 10,
            "token_budget": 3860,
            "reranker": "models/bge-reranker-v2-m3",
            "backend_fallback_allowed": False,
            "release_contract": "official-raw; reserve50 schema/status forbidden",
        },
        "inputs": {
            "candidate_release": candidate_binding,
            "closure_v3_policy": file_identity(closure_policy_path),
        },
        "code": {
            "scope_freezer": file_identity(Path(__file__)),
            "retrieval_materializer": file_identity(
                ROOT
                / "scripts/prepare/materialize_2wiki_official_raw_canonical_retrieval_v1.py"
            ),
        },
        "gates": gates,
        "scientific_boundary": {
            "answer_or_support_fields_read": False,
            "closure_result_read": False,
            "identity_selection_performed": False,
            "retrieval_started": False,
            "gpu_used": False,
            "training_started": False,
        },
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": f"{POLICY_SCHEMA}-report",
        "experiment_id": POLICY_EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": POLICY_STATUS,
        "scope_choice": protocol["scope_policy"],
        "gates": gates,
        "protocol": file_identity(protocol_path),
        "retrieval_started": False,
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
            "phase": "freeze_2wiki_official_raw_retrieval_scope_policy_v1",
            "experiment_id": POLICY_EXPERIMENT_ID,
            "protocol": file_identity(protocol_path),
            "report": file_identity(report_path),
            "retrieval_started": False,
            "training_started": False,
        },
    )
    return report


def validate_policy_release(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol_path = directory / "protocol.json"
    report_path = directory / "report.json"
    manifest_path = directory / "manifest.json"
    protocol = read_json(protocol_path)
    report = read_json(report_path)
    manifest = read_json(manifest_path)
    if (
        protocol.get("schema_version") != POLICY_SCHEMA
        or protocol.get("status") != POLICY_STATUS
        or report.get("status") != POLICY_STATUS
        or manifest.get("status") != POLICY_STATUS
        or not all(bool(v) for v in (protocol.get("gates") or {}).values())
        or (protocol.get("scientific_boundary") or {}).get("closure_result_read") is not False
    ):
        raise ValueError("official-raw retrieval policy release failed")
    if ((manifest.get("run") or {}).get("protocol") or {}).get("sha256") != sha256_file(protocol_path):
        raise ValueError("retrieval policy manifest/protocol binding drifted")
    for value in (protocol.get("code") or {}).values():
        resolve_identity(value, label="retrieval policy code")
    cohort_path = resolve_identity(
        protocol["inputs"]["candidate_release"]["cohort"], label="policy cohort"
    )
    cohort = read_jsonl(cohort_path)
    return protocol, cohort


def validate_closure_result(directory: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    report_path = directory / "report.json"
    manifest_path = directory / "manifest.json"
    report = read_json(report_path)
    manifest = read_json(manifest_path)
    if (
        report.get("schema_version") != EXPECTED_CLOSURE_REPORT_SCHEMA
        or report.get("status") != EXPECTED_CLOSURE_PASS_STATUS
        or manifest.get("status") != EXPECTED_CLOSURE_PASS_STATUS
        or report.get("all_pass") is not True
        or report.get("decision") != "CONTINUE_TO_PROOF800_SELECTION"
        or report.get("gold_access") is not False
        or report.get("training_started") is not False
        or not all(bool(v) for v in (report.get("gates") or {}).values())
    ):
        raise ValueError("closure-v3 result schema/status/gates failed")
    boundary = report.get("scientific_boundary") or {}
    if not (
        boundary.get("passages_or_answers_read") is False
        and boundary.get("proof800_selected") is False
        and boundary.get("training_started") is False
    ):
        raise ValueError("closure-v3 result scientific boundary failed")
    telemetry_identity = (report.get("outputs") or {}).get("strict_eligibility_telemetry")
    runtime_identity = (report.get("outputs") or {}).get("runtime_details")
    if not isinstance(telemetry_identity, Mapping) or not isinstance(runtime_identity, Mapping):
        raise ValueError("closure-v3 result lacks runtime/telemetry bindings")
    telemetry_path = resolve_identity(telemetry_identity, label="closure telemetry")
    runtime_path = resolve_identity(runtime_identity, label="closure runtime")
    telemetry = index_rows(read_jsonl(telemetry_path), label="closure telemetry")
    runtime = index_rows(read_jsonl(runtime_path), label="closure runtime")
    if set(telemetry) != set(runtime) or len(telemetry) != 1500:
        raise ValueError("closure-v3 telemetry/runtime join is not exact n1500")
    for key, row in telemetry.items():
        trace = runtime[key]
        if (
            row.get("schema_version") != EXPECTED_TELEMETRY_SCHEMA
            or forbidden_present(row)
            or str(row.get("runtime_record_sha256") or "") != canonical_sha256(trace)
            or str(row.get("kg_sha256") or "")
            != canonical_sha256(trace.get("kg_subgraph") or [])
            or str(row.get("execution_sha256") or "")
            != canonical_sha256(trace.get("execution") or {})
        ):
            raise ValueError(f"closure telemetry/runtime hash boundary failed: {key}")
    manifest_report = (manifest.get("run") or {}).get("report") or {}
    if (
        not isinstance(manifest_report, Mapping)
        or manifest_report.get("sha256") != sha256_file(report_path)
        or (manifest.get("run") or {}).get("training_started") is not False
    ):
        raise ValueError("closure-v3 manifest/report binding drifted")
    return telemetry, {
        "report": file_identity(report_path),
        "manifest": file_identity(manifest_path),
        "runtime_details": file_identity(runtime_path),
        "strict_eligibility_telemetry": file_identity(telemetry_path),
    }


def predicate_eligible(row: Mapping[str, Any]) -> bool:
    expected = {
        "m_graph": 1,
        "planner_schema_valid": True,
        "all_root_anchors_resolved": True,
        "all_hops_complete": True,
        "graph_nonempty": True,
        "gold_access_false": True,
        "runtime_error_zero": True,
        "provenance_complete": True,
        "retained_edges_traceable": True,
        "no_duplicate_edges": True,
    }
    return all(row.get(field) == value for field, value in expected.items())


def build_scope_requests(
    cohort_rows: Sequence[Mapping[str, Any]],
    telemetry_rows: Sequence[Mapping[str, Any]],
    *,
    min_per_type: int = MIN_PER_TYPE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cohort = index_rows(cohort_rows, label="scope cohort")
    telemetry = index_rows(telemetry_rows, label="scope telemetry")
    if set(cohort) != set(telemetry):
        raise ValueError("scope cohort/telemetry identity join is not exact")
    requests: list[dict[str, Any]] = []
    rejected = Counter()
    for key, base in cohort.items():
        trace = telemetry[key]
        for field in ("dataset", "qid", "question_sha256", "family_sha256", "question_type"):
            if str(trace.get(field) or "") != str(base.get(field) or ""):
                raise ValueError(f"scope telemetry/cohort mismatch at {field}: {key}")
        if not predicate_eligible(trace):
            rejected[str(trace.get("routing_reason") or "predicate_ineligible")] += 1
            continue
        request = {
            "schema_version": REQUEST_SCHEMA,
            "question_key": key,
            "dataset": DATASET,
            "qid": str(base["qid"]),
            "question": str(base["question"]),
            "question_sha256": str(base["question_sha256"]),
            "family_version": FAMILY_VERSION,
            "family_sha256": str(base["family_sha256"]),
            "question_type": str(base["question_type"]),
            "role": "official_raw_proofkg_rollout_retrieval",
            "closure_runtime_record_sha256": str(trace["runtime_record_sha256"]),
            "closure_kg_sha256": str(trace["kg_sha256"]),
            "closure_execution_sha256": str(trace["execution_sha256"]),
            "gold_access": False,
            "evaluation_eligible": False,
        }
        if forbidden_present(request):
            raise ValueError(f"forbidden field entered retrieval request: {key}")
        requests.append(request)
    requests.sort(key=lambda row: (QTYPES.index(str(row["question_type"])), str(row["qid"])))
    by_type = Counter(str(row["question_type"]) for row in requests)
    if any(by_type[qtype] < min_per_type for qtype in QTYPES):
        raise RuntimeError(
            "official-raw retrieval scope lacks strict capacity: "
            f"counts={dict(by_type)}, required_each={min_per_type}"
        )
    return requests, {
        "candidate_total": len(cohort),
        "strict_scope_total": len(requests),
        "strict_scope_by_question_type": {qtype: by_type[qtype] for qtype in QTYPES},
        "rejected_by_routing_reason": dict(sorted(rejected.items())),
    }


def freeze_scope(
    *, policy_dir: Path, closure_dir: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite retrieval scope: {output_dir}")
    policy, cohort = validate_policy_release(policy_dir)
    telemetry, closure_binding = validate_closure_result(closure_dir)
    requests, counts = build_scope_requests(cohort, list(telemetry.values()))
    output_dir.mkdir(parents=True, exist_ok=False)
    requests_path = output_dir / "retrieval_requests.question_only.jsonl"
    write_jsonl(requests_path, requests)
    gates = {
        "closure_v3_pass": True,
        "identity_join_rate_1": len(telemetry) == len(cohort) == 1500,
        "strict_scope_each_question_type_ge_200": all(
            counts["strict_scope_by_question_type"][qtype] >= MIN_PER_TYPE
            for qtype in QTYPES
        ),
        "scope_is_all_and_only_frozen_predicate_rows": len(requests)
        == sum(predicate_eligible(row) for row in telemetry.values()),
        "request_identity_unique": len({row["question_key"] for row in requests})
        == len(requests),
        "request_question_hash_unique": len(
            {row["question_sha256"] for row in requests}
        )
        == len(requests),
        "gold_access_false": all(row["gold_access"] is False for row in requests),
        "forbidden_fields_zero": all(not forbidden_present(row) for row in requests),
        "retrieval_not_started": True,
        "training_not_started": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"official-raw retrieval scope gates failed: {gates}")
    protocol = {
        "schema_version": SCOPE_SCHEMA,
        "experiment_id": SCOPE_EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": SCOPE_STATUS,
        "scope_policy": policy["scope_policy"],
        "population": counts,
        "retrieval": policy["retrieval"],
        "inputs": {
            "scope_policy": file_identity(policy_dir / "protocol.json"),
            "closure_v3_release": closure_binding,
        },
        "outputs": {"retrieval_requests": file_identity(requests_path)},
        "gates": gates,
        "scientific_boundary": {
            "question_only": True,
            "answer_or_support_fields_read": False,
            "semantic_correctness_used": False,
            "retrieval_started": False,
            "training_started": False,
        },
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": f"{SCOPE_SCHEMA}-report",
        "experiment_id": SCOPE_EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": SCOPE_STATUS,
        "counts": counts,
        "gates": gates,
        "inputs": protocol["inputs"],
        "outputs": {
            "retrieval_requests": file_identity(requests_path),
            "protocol": file_identity(protocol_path),
        },
        "retrieval_started": False,
        "training_started": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        output_dir,
        status=SCOPE_STATUS,
        extra={
            "phase": "freeze_2wiki_official_raw_retrieval_scope_v1",
            "experiment_id": SCOPE_EXPERIMENT_ID,
            "protocol": file_identity(protocol_path),
            "report": file_identity(report_path),
            "retrieval_started": False,
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    policy = subparsers.add_parser("freeze-policy")
    policy.add_argument("--candidate-release", type=Path, default=DEFAULT_CANDIDATE_RELEASE)
    policy.add_argument("--closure-policy", type=Path, default=DEFAULT_CLOSURE_POLICY)
    policy.add_argument("--out", type=Path, default=DEFAULT_POLICY_DIR)
    scope = subparsers.add_parser("freeze-scope")
    scope.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    scope.add_argument("--closure-dir", type=Path, required=True)
    scope.add_argument("--out", type=Path, default=DEFAULT_SCOPE_DIR)
    args = parser.parse_args()
    if args.command == "freeze-policy":
        report = freeze_policy(
            candidate_release=args.candidate_release,
            closure_policy_path=args.closure_policy,
            output_dir=args.out,
        )
    else:
        report = freeze_scope(
            policy_dir=args.policy_dir,
            closure_dir=args.closure_dir,
            output_dir=args.out,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
