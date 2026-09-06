#!/usr/bin/env python3
"""Freeze every root occurrence from the Gold-free official-raw n1500 plans.

This is the successor to the delta-only n300 root resolver.  It deliberately
does not read an earlier runtime or inherit an earlier ``resolved`` bit.  Every
root consumed by every recognized plan becomes a fresh resolution request.
The one retained invalid/unrecognized planner row remains in the denominator
and is reported, but has no invented root request.

No network access or model execution occurs in this script.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.kg.wikipedia_title_resolver import complete_question_surface_title
from kgproweight.utils.logging import dump_manifest
from scripts.pilot.build_automatic_proofkg_from_plans import convert_predicted_target


DATASET = "2wikimultihopqa"
WORKLIST_SCHEMA = "2wiki-full-root-anchor-resolution-worklist-v2"
PROTOCOL_SCHEMA = "2wiki-full-root-anchor-resolution-protocol-v2"
STATUS = "FROZEN_ALL_ROOTS_BEFORE_NETWORK_NO_GOLD_NOT_TRAINED"
EXPERIMENT_ID = (
    "2WIKI-PROOFKG-OFFICIAL-RAW-V2-N1500-FULL-ROOT-RESOLUTION-V2-PREREGISTRATION"
)

DEFAULT_COHORT = Path(
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_seed42_"
    "preregistration/cohort.question_only.jsonl"
)
DEFAULT_PLANS = Path(
    "outputs/validation/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "seed42_plans_v1/predictions.question_only.jsonl"
)
DEFAULT_POSTFLIGHT = Path(
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_seed42_"
    "plans_v1_postflight/report.json"
)
DEFAULT_ROOT_GAP_AUDIT = Path(
    "outputs/audits/2wiki_proofkg_extension_v1b_n300_root_projection_gap_audit_v1/"
    "report.json"
)
DEFAULT_V6_STORE = Path(
    "indexes/versioned_2wiki_evidence_store_v6_mixed3_v4_complete_ledger_seed42"
)
DEFAULT_OUTPUT = Path(
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "root_resolution_v1_preregistration"
)
DEFAULT_MATERIALIZED_OUTPUT = Path(
    "indexes/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_root_resolution_v1"
)

WORKLIST_FIELDS = {
    "schema_version",
    "request_id",
    "question_key",
    "dataset",
    "qid",
    "question",
    "question_sha256",
    "question_type",
    "root_position",
    "root_anchor_surface",
    "completed_root_anchor_surface",
    "gold_access",
}
FORBIDDEN_KEYS = {
    "answer",
    "answers",
    "expected_qid",
    "execution",
    "generated_text",
    "gold_answer",
    "gold_answers",
    "kg_subgraph",
    "old_qid",
    "passages",
    "predicted_target",
    "query_plan",
    "resolved_qid",
    "retrieval_result",
    "support",
    "supporting_facts",
    "target_qid",
}


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


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _has_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in FORBIDDEN_KEYS or _has_forbidden_key(nested)
            for key, nested in value.items()
        )
    if isinstance(value, list):
        return any(_has_forbidden_key(item) for item in value)
    return False


def _index(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        dataset = str(row.get("dataset") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        question = str(row.get("question") or "").strip()
        if dataset != DATASET or not qid or not question:
            raise ValueError(f"{label}: invalid identity {dataset!r}::{qid!r}")
        key = question_key(dataset, qid)
        if str(row.get("question_key") or key) != key:
            raise ValueError(f"{label}: question_key mismatch for {key}")
        qhash = question_sha256(question)
        if str(row.get("question_sha256") or qhash) != qhash:
            raise ValueError(f"{label}: question hash mismatch for {key}")
        if key in result:
            raise ValueError(f"{label}: duplicate question key {key}")
        result[key] = row
    return result


def build_all_root_worklist(
    cohort_rows: Sequence[Mapping[str, Any]],
    plan_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project all consumer-visible roots without consulting old resolution state."""

    cohort = _index(cohort_rows, label="cohort")
    plans = _index(plan_rows, label="plans")
    if not cohort or set(cohort) != set(plans):
        raise ValueError(
            f"cohort/plans identity join is not exact: cohort={len(cohort)} plans={len(plans)}"
        )
    worklist: list[dict[str, Any]] = []
    recognized = 0
    by_type: Counter[str] = Counter()
    roots_by_type: Counter[str] = Counter()
    unrecognized_by_type: Counter[str] = Counter()
    for key in sorted(cohort):
        source = cohort[key]
        plan_row = plans[key]
        question = str(source["question"]).strip()
        if str(plan_row["question"]).strip() != question:
            raise ValueError(f"planner question drift for {key}")
        if source.get("gold_access") is not False or plan_row.get("gold_access") is not False:
            raise ValueError(f"gold_access must be false for {key}")
        question_type = str(source.get("question_type") or "unknown")
        by_type[question_type] += 1
        plan, _diagnostics = convert_predicted_target(
            DATASET, question, plan_row.get("predicted_target")
        )
        if not plan.recognized:
            unrecognized_by_type[question_type] += 1
            continue
        anchors = list(plan.anchors)
        if not anchors or len(anchors) != len(set(anchors)):
            raise ValueError(f"recognized plan has invalid roots for {key}")
        recognized += 1
        roots_by_type[question_type] += len(anchors)
        for position, surface in enumerate(anchors, start=1):
            completed = complete_question_surface_title(surface, question)
            request_id = hashlib.sha256(
                "\0".join((DATASET, str(source["qid"]), str(position), surface)).encode(
                    "utf-8"
                )
            ).hexdigest()
            worklist.append(
                {
                    "schema_version": WORKLIST_SCHEMA,
                    "request_id": request_id,
                    "question_key": key,
                    "dataset": DATASET,
                    "qid": str(source["qid"]),
                    "question": question,
                    "question_sha256": question_sha256(question),
                    "question_type": question_type,
                    "root_position": position,
                    "root_anchor_surface": surface,
                    "completed_root_anchor_surface": completed,
                    "gold_access": False,
                }
            )
    worklist.sort(key=lambda row: (str(row["question_key"]), int(row["root_position"])))
    if any(set(row) != WORKLIST_FIELDS for row in worklist):
        raise RuntimeError("worklist schema drift")
    if any(_has_forbidden_key(row) for row in worklist):
        raise RuntimeError("forbidden data leaked into all-root worklist")
    if len({str(row["request_id"]) for row in worklist}) != len(worklist):
        raise RuntimeError("duplicate all-root request id")
    return worklist, {
        "questions_total": len(cohort),
        "questions_recognized": recognized,
        "questions_unrecognized": len(cohort) - recognized,
        "root_anchor_occurrences": len(worklist),
        "unique_root_surfaces": len(
            {str(row["root_anchor_surface"]).casefold() for row in worklist}
        ),
        "by_question_type": dict(sorted(by_type.items())),
        "root_occurrences_by_question_type": dict(sorted(roots_by_type.items())),
        "unrecognized_by_question_type": dict(sorted(unrecognized_by_type.items())),
    }


def freeze(
    *,
    cohort_path: Path,
    plans_path: Path,
    planner_postflight_path: Path,
    root_gap_audit_path: Path,
    v6_store_dir: Path,
    output_dir: Path,
    materialized_output_dir: Path,
    experiment_id: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite append-only protocol: {output_dir}")
    v6_manifest_path = v6_store_dir / "store_manifest.json"
    v6_aliases_path = v6_store_dir / "aliases.jsonl"
    required = (
        cohort_path,
        plans_path,
        planner_postflight_path,
        root_gap_audit_path,
        v6_manifest_path,
        v6_aliases_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    postflight = json.loads(planner_postflight_path.read_text(encoding="utf-8"))
    gap = json.loads(root_gap_audit_path.read_text(encoding="utf-8"))
    v6_manifest = json.loads(v6_manifest_path.read_text(encoding="utf-8"))
    if postflight.get("status") != "PASS_PLANNER_STRUCTURAL_NOT_PROOFKG_MATERIALIZED_NOT_TRAINED":
        raise ValueError("planner postflight is not the frozen PASS artifact")
    if gap.get("status") != "COMPLETE_CAUSE_IDENTIFIED_OLD_RESULTS_UNCHANGED":
        raise ValueError("root-gap diagnosis is not the required completed audit")
    if v6_manifest.get("status") != "COMPLETE_NOT_EVALUATED":
        raise ValueError("v6 clean store is not complete")
    if (v6_manifest.get("protected_ledger") or {}).get("complete") is not True:
        raise ValueError("v6 store is not bound to a complete protection ledger")

    worklist, counts = build_all_root_worklist(
        read_jsonl(cohort_path), read_jsonl(plans_path)
    )
    post = postflight.get("summary") or {}
    checks = {
        "question_identity_join_rate": 1.0,
        "planner_postflight_n_matches": int(post.get("n_predictions", -1))
        == counts["questions_total"],
        "planner_postflight_recognized_proxy_matches": int(post.get("schema_valid", -1))
        == counts["questions_recognized"],
        "recognized_plan_rate_ge_0_97": (
            counts["questions_recognized"] / counts["questions_total"] >= 0.97
        ),
        "all_recognized_root_occurrences_projected": len(worklist)
        == counts["root_anchor_occurrences"],
        "worklist_schema_exact": all(set(row) == WORKLIST_FIELDS for row in worklist),
        "worklist_forbidden_fields_zero": not any(_has_forbidden_key(row) for row in worklist),
        "gold_access_false": all(row["gold_access"] is False for row in worklist),
        "old_resolution_state_read": False,
        "network_access": False,
        "training_started": False,
    }
    required_true = {
        key: value
        for key, value in checks.items()
        if key not in {"old_resolution_state_read", "network_access", "training_started"}
    }
    if not all(value is True or value == 1.0 for value in required_true.values()):
        raise RuntimeError({"checks": checks})
    # Explicitly assert the intended false-valued provenance checks.
    if checks["old_resolution_state_read"] or checks["network_access"] or checks["training_started"]:
        raise RuntimeError("freeze boundary drift")

    output_dir.mkdir(parents=True, exist_ok=False)
    worklist_path = output_dir / "root_anchor_resolution_worklist.question_only.jsonl"
    write_jsonl(worklist_path, worklist)
    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "scope": "2Wiki official-raw n1500; all recognized planner roots; Gold-free root resolution",
        "scientific_correction": {
            "supersedes_delta_only_resolution_for_new_runs": True,
            "old_n300_results_unchanged": True,
            "cause": (gap.get("diagnosis") or {}).get("cause"),
            "all_roots_re_resolved": True,
            "old_resolved_bit_inherited": False,
        },
        "resolver_input_contract": {
            "allowed_information": [
                "dataset", "dataset qid", "question text", "question hash",
                "question type", "planner root surface", "root position",
                "parenthetical completion copied from the same question text",
            ],
            "forbidden_information": sorted(FORBIDDEN_KEYS),
            "gold_access": False,
            "passage_access": False,
            "old_resolution_or_expected_qid_access": False,
        },
        "resolution_policy": {
            "consumer_order": [
                "new exact Wikipedia title cache on completed root surface",
                "v6 clean exact alias on completed root surface; unique QID only",
                "new exact entity cache on raw root surface",
                "abstain",
            ],
            "network_materialization_order": [
                "exact Wikipedia title lookup on completed root surface",
                "v6 clean exact alias on completed root surface; unique QID only",
                "Wikidata candidate search on raw root surface scored only by question",
                "confidence/margin failure abstains",
            ],
            "fallback_min_score": 0.25,
            "fallback_min_margin": 0.10,
            "workers": 2,
            "request_delay_seconds": 0.4,
            "timeout_seconds": 12.0,
            "max_retries": 3,
            "new_isolated_title_and_entity_caches": True,
            "old_cache_or_wide_neighborhood_fallback": False,
            "deterministic_output_sort": ["question_key", "root_position"],
        },
        "continuation_gate": {
            "question_identity_join_rate": 1.0,
            "request_result_join_rate": 1.0,
            "recognized_plan_rate_min": 0.97,
            "runtime_errors": 0,
            "all_roots_resolved_question_rate_all_questions_min": 0.80,
            "anchor_occurrence_resolution_rate_min": 0.80,
            "projection_equals_exact_consumer_dry_run_every_occurrence": True,
            "gold_access": False,
            "on_failure": "stop; retain results; do not start property closure",
        },
        "counts": counts,
        "checks": checks,
        "inputs": {
            "candidate_cohort": file_identity(cohort_path),
            "planner_predictions": file_identity(plans_path),
            "planner_postflight": file_identity(planner_postflight_path),
            "root_gap_audit": file_identity(root_gap_audit_path),
            "v6_store_manifest": file_identity(v6_manifest_path),
            "v6_aliases": file_identity(v6_aliases_path),
            "resolver_implementation": file_identity(
                Path("scripts/prepare/materialize_2wiki_full_root_resolution_v2.py")
            ),
        },
        "outputs": {
            "worklist": file_identity(worklist_path),
            "planned_materialized_output_dir": str(materialized_output_dir.resolve()),
        },
        "proposed_command": [
            "python", "scripts/prepare/materialize_2wiki_full_root_resolution_v2.py",
            "--protocol", str((output_dir / "protocol.json").resolve()),
            "--worklist", str(worklist_path.resolve()),
            "--v6-store", str(v6_store_dir.resolve()),
            "--output-dir", str(materialized_output_dir.resolve()),
            "--workers", "2", "--delay", "0.4", "--timeout", "12", "--retries", "3",
        ],
        "network_access": False,
        "gold_access": False,
        "training_started": False,
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": PROTOCOL_SCHEMA,
        "experiment_id": experiment_id,
        "status": STATUS,
        "counts": counts,
        "checks": checks,
        "protocol": file_identity(protocol_path),
        "worklist": file_identity(worklist_path),
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
        status=STATUS,
        extra={
            "phase": "freeze_2wiki_official_raw_all_root_resolution_v2",
            "experiment_id": experiment_id,
            "protocol": file_identity(protocol_path),
            "report": file_identity(report_path),
            "worklist": file_identity(worklist_path),
            "code": {
                "path": str(Path(inspect.getsourcefile(freeze) or __file__).resolve()),
                "sha256": sha256_file(Path(inspect.getsourcefile(freeze) or __file__).resolve()),
            },
            "network_access": False,
            "gold_access": False,
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    parser.add_argument("--planner-postflight", type=Path, default=DEFAULT_POSTFLIGHT)
    parser.add_argument("--root-gap-audit", type=Path, default=DEFAULT_ROOT_GAP_AUDIT)
    parser.add_argument("--v6-store", type=Path, default=DEFAULT_V6_STORE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--materialized-output-dir", type=Path, default=DEFAULT_MATERIALIZED_OUTPUT)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    args = parser.parse_args()
    print(json.dumps(freeze(
        cohort_path=args.cohort,
        plans_path=args.plans,
        planner_postflight_path=args.planner_postflight,
        root_gap_audit_path=args.root_gap_audit,
        v6_store_dir=args.v6_store,
        output_dir=args.output_dir,
        materialized_output_dir=args.materialized_output_dir,
        experiment_id=args.experiment_id,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
