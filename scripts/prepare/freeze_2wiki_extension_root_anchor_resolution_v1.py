#!/usr/bin/env python3
"""Freeze question-only root-anchor re-resolution inputs for the 2Wiki extension.

The clean v5 evidence-store execution deliberately removed selected-question
aliases.  Its final closure therefore exposes a root-entity resolution gap.
This script turns *only* unresolved planner root-anchor surfaces plus their
question text into a versioned worklist.  It never copies a resolved/expected
Wikidata QID, Gold answer, supporting fact, passage, KG edge, or old execution
trace into the resolver input.

No network access or model execution is performed here.
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
SCHEMA_VERSION = "2wiki-root-anchor-resolution-worklist-v1"
PROTOCOL_SCHEMA_VERSION = "2wiki-root-anchor-resolution-protocol-v1"
STATUS = "FROZEN_BEFORE_NETWORK_NO_GOLD"
EXPERIMENT_ID = (
    "2WIKI-PROOFKG-EXTENSION-V1B-N300-ROOT-ANCHOR-RESOLUTION-V1-PREREGISTRATION"
)

DEFAULT_COHORT = Path(
    "outputs/audits/2wiki_proofkg_extension_v1b_n300_seed42_preregistration/"
    "cohort.question_only.jsonl"
)
DEFAULT_PLANS = Path(
    "outputs/validation/2wiki_proofkg_extension_v1b_n300_seed42_plans/"
    "predictions.question_only.jsonl"
)
DEFAULT_RUNTIME = Path(
    "data/derived/2wiki_proofkg_extension_v1b_n300_closure_v1/round_2/runtime/"
    "runtime_details.jsonl"
)
DEFAULT_RUNTIME_REPORT = Path(
    "data/derived/2wiki_proofkg_extension_v1b_n300_closure_v1/round_2/runtime/"
    "report.json"
)
DEFAULT_CLOSURE_REPORT = Path(
    "data/derived/2wiki_proofkg_extension_v1b_n300_closure_v1/closure_report.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/audits/2wiki_proofkg_extension_v1b_n300_root_anchor_resolution_v1_"
    "preregistration"
)

WORKLIST_FIELDS = {
    "schema_version",
    "request_id",
    "question_key",
    "dataset",
    "qid",
    "question",
    "question_sha256",
    "root_anchor_surface",
    "completed_root_anchor_surface",
    "gold_access",
}

# These fields are forbidden recursively in the resolver worklist.  In
# particular, Wikidata QIDs from an old resolver may appear in runtime_details,
# but no such target/reference field is ever projected into this artifact.
FORBIDDEN_KEYS = {
    "answer",
    "answers",
    "gold_answer",
    "gold_answers",
    "supporting_facts",
    "support",
    "passages",
    "retrieval_result",
    "kg_subgraph",
    "query_plan",
    "predicted_target",
    "generated_text",
    "execution",
    "resolved_qid",
    "expected_qid",
    "old_qid",
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


def _index_unique(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        dataset = str(row.get("dataset") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        question = str(row.get("question") or "").strip()
        if dataset != DATASET or not qid or not question:
            raise ValueError(f"{label}: invalid identity {dataset!r}::{qid!r}")
        key = question_key(dataset, qid)
        stored_key = str(row.get("question_key") or key)
        if stored_key != key:
            raise ValueError(f"{label}: question_key mismatch for {key}")
        actual_hash = question_sha256(question)
        if str(row.get("question_sha256") or actual_hash) != actual_hash:
            raise ValueError(f"{label}: question hash mismatch for {key}")
        if key in result:
            raise ValueError(f"{label}: duplicate question_key {key}")
        result[key] = row
    return result


def _anchor_list(plan_row: Mapping[str, Any]) -> list[str]:
    predicted = plan_row.get("predicted_target") or {}
    if not isinstance(predicted, Mapping):
        raise ValueError("planner predicted_target must be an object")
    anchors = predicted.get("anchors") or []
    if not isinstance(anchors, list):
        raise ValueError("planner anchors must be a list")
    cleaned = [" ".join(str(value).strip().split()) for value in anchors]
    if not cleaned or any(not value for value in cleaned):
        raise ValueError("planner root anchors must be non-empty")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("planner root anchors must be unique within a question")
    return cleaned


def _has_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() in FORBIDDEN_KEYS or _has_forbidden_key(nested):
                return True
    elif isinstance(value, list):
        return any(_has_forbidden_key(item) for item in value)
    return False


def build_worklist(
    *,
    cohort_rows: Sequence[Mapping[str, Any]],
    plan_rows: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return resolver-safe rows plus exact pre-resolution coverage telemetry."""

    cohort = _index_unique(cohort_rows, label="cohort")
    plans = _index_unique(plan_rows, label="plans")
    runtime = _index_unique(runtime_rows, label="runtime")
    if not cohort or set(cohort) != set(plans) or set(plans) != set(runtime):
        raise ValueError(
            "cohort/plans/runtime identity join is not exact: "
            f"cohort={len(cohort)} plans={len(plans)} runtime={len(runtime)}"
        )

    worklist: list[dict[str, Any]] = []
    total_anchor_occurrences = 0
    resolved_anchor_occurrences = 0
    all_resolved_questions = 0
    partially_resolved_questions = 0
    unresolved_questions = 0
    reasons: Counter[str] = Counter()
    by_type_population: Counter[str] = Counter()
    by_type_unresolved_questions: Counter[str] = Counter()
    unique_all: set[str] = set()
    unique_resolved: set[str] = set()
    unique_unresolved: set[str] = set()

    for key in sorted(plans):
        plan_row = plans[key]
        cohort_row = cohort[key]
        runtime_row = runtime[key]
        question = str(plan_row["question"]).strip()
        qhash = question_sha256(question)
        for other_label, other in (("cohort", cohort_row), ("runtime", runtime_row)):
            if str(other.get("question") or "").strip() != question:
                raise ValueError(f"{other_label}: question text mismatch for {key}")
            if str(other.get("question_sha256") or "") != qhash:
                raise ValueError(f"{other_label}: question hash mismatch for {key}")
        if plan_row.get("gold_access") is not False:
            raise ValueError(f"planner gold_access must be false for {key}")
        provenance = runtime_row.get("provenance") or {}
        if not isinstance(provenance, Mapping) or provenance.get("gold_access") is not False:
            raise ValueError(f"runtime gold_access must be false for {key}")
        if runtime_row.get("runtime_error"):
            raise ValueError(f"runtime error present for {key}: {runtime_row['runtime_error']}")

        # The executor applies deterministic punctuation/dependency cleanup to
        # the raw planner JSON.  Resolve the exact root surfaces the clean
        # executor will consume, while still deriving them solely from the
        # question-only planner output (never from an old QID).
        _anchor_list(plan_row)  # validate the raw planner anchor container
        converted_plan, _ = convert_predicted_target(
            DATASET, question, plan_row.get("predicted_target")
        )
        anchors = list(converted_plan.anchors)
        if not anchors or len(set(anchors)) != len(anchors):
            raise ValueError(f"converted planner roots invalid for {key}")
        runtime_plan = runtime_row.get("query_plan") or {}
        runtime_anchors = list(runtime_plan.get("anchors") or [])
        if runtime_anchors != anchors:
            raise ValueError(f"planner/runtime root-anchor mismatch for {key}")
        execution = runtime_row.get("execution") or {}
        anchor_entities = execution.get("anchor_entities") or {}
        if set(anchor_entities) != set(anchors):
            raise ValueError(f"runtime anchor diagnostics mismatch for {key}")

        question_type = str(cohort_row.get("question_type") or "unknown")
        by_type_population[question_type] += 1
        unresolved_here = 0
        resolved_here = 0
        for surface in anchors:
            total_anchor_occurrences += 1
            unique_all.add(surface)
            diagnostic = anchor_entities[surface]
            is_resolved = bool(diagnostic.get("qid")) and not bool(
                diagnostic.get("abstained")
            )
            if is_resolved:
                resolved_anchor_occurrences += 1
                resolved_here += 1
                unique_resolved.add(surface)
                continue

            unresolved_here += 1
            unique_unresolved.add(surface)
            reasons[str(diagnostic.get("abstain_reason") or "missing_qid")] += 1
            completed = complete_question_surface_title(surface, question)
            request_id = hashlib.sha256(
                "\0".join((DATASET, str(plan_row["qid"]), surface)).encode("utf-8")
            ).hexdigest()
            worklist.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "request_id": request_id,
                    "question_key": key,
                    "dataset": DATASET,
                    "qid": str(plan_row["qid"]),
                    "question": question,
                    "question_sha256": qhash,
                    "root_anchor_surface": surface,
                    "completed_root_anchor_surface": completed,
                    "gold_access": False,
                }
            )

        if unresolved_here:
            unresolved_questions += 1
            by_type_unresolved_questions[question_type] += 1
            if resolved_here:
                partially_resolved_questions += 1
        else:
            all_resolved_questions += 1

    worklist.sort(
        key=lambda row: (
            str(row["question_key"]),
            str(row["root_anchor_surface"]).casefold(),
        )
    )
    if any(set(row) != WORKLIST_FIELDS for row in worklist):
        raise RuntimeError("resolver worklist schema drift")
    if any(_has_forbidden_key(row) for row in worklist):
        raise RuntimeError("forbidden data leaked into resolver worklist")
    if len({row["request_id"] for row in worklist}) != len(worklist):
        raise RuntimeError("duplicate root-anchor resolver request_id")
    if unique_resolved & unique_unresolved:
        raise RuntimeError("same root surface has conflicting resolution status")

    stats = {
        "questions": len(plans),
        "all_roots_resolved_questions": all_resolved_questions,
        "partially_resolved_questions": partially_resolved_questions,
        "unresolved_questions": unresolved_questions,
        "anchor_occurrences": total_anchor_occurrences,
        "resolved_anchor_occurrences": resolved_anchor_occurrences,
        "unresolved_anchor_occurrences": len(worklist),
        "unique_anchor_surfaces": len(unique_all),
        "unique_resolved_anchor_surfaces": len(unique_resolved),
        "unique_unresolved_anchor_surfaces": len(unique_unresolved),
        "unresolved_reasons": dict(sorted(reasons.items())),
        "by_question_type": {
            "population": dict(sorted(by_type_population.items())),
            "unresolved_questions": dict(sorted(by_type_unresolved_questions.items())),
        },
    }
    return worklist, stats


def freeze(
    *,
    cohort_path: Path,
    plans_path: Path,
    runtime_path: Path,
    runtime_report_path: Path,
    closure_report_path: Path,
    output_dir: Path,
    experiment_id: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite versioned output: {output_dir}")
    for path in (
        cohort_path,
        plans_path,
        runtime_path,
        runtime_report_path,
        closure_report_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    runtime_report = json.loads(runtime_report_path.read_text(encoding="utf-8"))
    closure_report = json.loads(closure_report_path.read_text(encoding="utf-8"))
    worklist, counts = build_worklist(
        cohort_rows=read_jsonl(cohort_path),
        plan_rows=read_jsonl(plans_path),
        runtime_rows=read_jsonl(runtime_path),
    )
    expected_runtime_counts = runtime_report.get("counts") or {}
    checks = {
        "identity_join_rate": 1.0,
        "gold_access_false": all(row["gold_access"] is False for row in worklist),
        "runtime_errors_zero": int(expected_runtime_counts.get("runtime_errors", -1)) == 0,
        "report_question_count_matches": int(expected_runtime_counts.get("n", -1))
        == counts["questions"],
        "report_all_roots_resolved_matches": int(
            expected_runtime_counts.get("anchor_qid_resolved", -1)
        )
        == counts["all_roots_resolved_questions"],
        "worklist_equals_unresolved_anchor_occurrences": len(worklist)
        == counts["unresolved_anchor_occurrences"],
        "worklist_schema_exact": all(set(row) == WORKLIST_FIELDS for row in worklist),
        "worklist_forbidden_fields_zero": not any(
            _has_forbidden_key(row) for row in worklist
        ),
        "network_access_disabled": True,
    }
    if not all(value is True or value == 1.0 for value in checks.values()):
        raise RuntimeError({"checks": checks, "counts": counts})

    output_dir.mkdir(parents=True, exist_ok=False)
    worklist_path = output_dir / "root_anchor_resolution_worklist.question_only.jsonl"
    write_jsonl(worklist_path, worklist)

    resolver_output_dir = Path(
        "indexes/2wiki_proofkg_extension_v1b_n300_root_anchor_resolution_v1"
    )
    protocol = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "scientific_diagnosis": {
            "clean_store_closure_completed": True,
            "closure_stop_reason": closure_report.get("stop_reason"),
            "property_closure_materialized_rounds": closure_report.get(
                "last_materialized_round"
            ),
            "root_resolution_is_current_bottleneck": True,
            "current_all_roots_resolved_rate": (
                counts["all_roots_resolved_questions"] / counts["questions"]
            ),
            "current_anchor_occurrence_resolution_rate": (
                counts["resolved_anchor_occurrences"] / counts["anchor_occurrences"]
            ),
        },
        "resolver_input_contract": {
            "allowed_information": [
                "dataset",
                "dataset qid",
                "question text",
                "question hash",
                "planner root-anchor surface",
                "parenthetical completion copied from the same question text",
            ],
            "forbidden_information": sorted(FORBIDDEN_KEYS),
            "old_resolved_or_expected_wikidata_qids_as_input": False,
            "gold_access": False,
            "passage_access": False,
            "old_entity_or_title_cache_as_input": False,
        },
        "resolution_policy": {
            "order": [
                "exact Wikipedia title lookup on completed_root_anchor_surface",
                "if exact title abstains: Wikidata candidate search on root_anchor_surface, scored only with question text",
                "if confidence/margin gate fails: abstain",
            ],
            "exact_title_acceptance": (
                "one unique non-disambiguation Wikipedia page with wikibase_item"
            ),
            "fallback_min_score": 0.25,
            "fallback_min_margin": 0.10,
            "qid_pattern": "^Q[1-9][0-9]*$",
            "workers": 2,
            "request_delay_seconds": 0.4,
            "timeout_seconds": 12.0,
            "max_retries": 3,
            "new_isolated_caches_only": True,
            "candidate_and_decision_log_required": True,
            "deterministic_output_sort": ["question_key", "root_anchor_surface"],
        },
        "continue_gate_before_clean_closure_v2": {
            "request_log_join_rate": 1.0,
            "runtime_errors": 0,
            "all_roots_resolved_question_rate": {"op": ">=", "value": 0.80},
            "anchor_occurrence_resolution_rate": {"op": ">=", "value": 0.80},
            "gold_access": False,
            "old_cache_fallback": False,
            "on_failure": "stop; retain result; do not launch property closure v2",
        },
        "counts": counts,
        "checks": checks,
        "inputs": {
            "cohort": file_identity(cohort_path),
            "planner_predictions": file_identity(plans_path),
            "final_clean_runtime_details": file_identity(runtime_path),
            "final_clean_runtime_report": file_identity(runtime_report_path),
            "clean_closure_report": file_identity(closure_report_path),
        },
        "outputs": {
            "worklist": file_identity(worklist_path),
            "planned_resolution_output_dir": str(resolver_output_dir.resolve()),
        },
        "proposed_command": [
            "python",
            "scripts/prepare/materialize_2wiki_root_anchor_resolution_v1.py",
            "--protocol",
            str((output_dir / "protocol.json").resolve()),
            "--worklist",
            str(worklist_path.resolve()),
            "--output-dir",
            str(resolver_output_dir.resolve()),
            "--workers",
            "2",
            "--delay",
            "0.4",
            "--timeout",
            "12",
            "--retries",
            "3",
        ],
        "network_access": False,
        "training_started": False,
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "status": STATUS,
        "counts": counts,
        "checks": checks,
        "protocol": file_identity(protocol_path),
        "worklist": file_identity(worklist_path),
        "network_access": False,
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
            "phase": "freeze_clean_2wiki_extension_root_anchor_resolution",
            "experiment_id": experiment_id,
            "protocol": file_identity(protocol_path),
            "report": file_identity(report_path),
            "worklist": file_identity(worklist_path),
            "code": {
                "path": str(Path(inspect.getsourcefile(freeze) or __file__).resolve()),
                "sha256": sha256_file(
                    Path(inspect.getsourcefile(freeze) or __file__).resolve()
                ),
            },
            "network_access": False,
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    parser.add_argument("--runtime-details", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--runtime-report", type=Path, default=DEFAULT_RUNTIME_REPORT)
    parser.add_argument("--closure-report", type=Path, default=DEFAULT_CLOSURE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    args = parser.parse_args()
    report = freeze(
        cohort_path=args.cohort,
        plans_path=args.plans,
        runtime_path=args.runtime_details,
        runtime_report_path=args.runtime_report,
        closure_report_path=args.closure_report,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
