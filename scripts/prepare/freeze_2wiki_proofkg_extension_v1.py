#!/usr/bin/env python3
"""Freeze 300 question-only candidates for a 2Wiki ProofKG extension.

The source curriculum contains Gold-derived answers, steps, and graphs.  This
selector deliberately projects each selected row onto an answer-free identity
schema before anything is written.  It performs no planner inference, graph
construction, retrieval, network access, or model training.

Isolation has two intentionally different scopes:

* the existing automatic-ProofKG n=1500 pool is a train-side source, so new
  rows must be disjoint from it by qid and exact question hash; lexical-family
  reuse is allowed;
* all SAEG evaluation roles and the frozen never-train reserve82 are protected
  by qid, exact question hash, and the current answer-free lexical family.

This distinction is part of the frozen protocol, not an implicit relaxation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.kg.question_kg import validate_question_kg_record
from kgproweight.reward.trajectory_source_gate import evaluate_graph_gate
from kgproweight.utils.logging import dump_manifest, prepare_new_run_dir
from scripts.prepare.freeze_qpeg_v1_protocol import (
    FAMILY_VERSION,
    family_sha256,
    question_family_signature,
)


DATASET = "2wikimultihopqa"
SEED = 42
SCHEMA_VERSION = "automatic-proofkg-extension-question-only-v1"
EXPERIMENT_ID = "2WIKI-PROOFKG-EXTENSION-V1B-N300-SEED42-PREREGISTRATION"
STATUS = "FROZEN_TRAIN_ONLY_BEFORE_PLANNER_NOT_MATERIALIZED"
QUOTAS: dict[str, int] = {
    "inference": 158,
    "comparison": 72,
    "compositional": 70,
    "bridge_comparison": 0,
}
EXPECTED_N = sum(QUOTAS.values())

DEFAULT_SOURCE = Path(
    "data/silver_data/proofkg_curriculum_mix_v1_n8000_seed42_20260829/"
    "silver_curriculum.jsonl"
)
DEFAULT_ASSIGNMENTS = Path(
    "data/silver_data/query_planner_supervision_split_v1_seed20260829/"
    "assignments.jsonl"
)
DEFAULT_AUTO1500 = Path(
    "outputs/audits/automatic_proofkg_2wiki_train_k4_v1_n1500_seed42_"
    "preregistration/cohort.question_only.jsonl"
)
DEFAULT_AUTO_QUESTION_KG = Path(
    "data/silver_data/automatic_proofkg_2wiki_train_k4_v1/"
    "question_kg_records.jsonl"
)
DEFAULT_AUTO_RUNTIME = Path(
    "outputs/audits/automatic_proofkg_2wiki_train_k4_v1_n1500_seed42_"
    "historical_stage3_runtime/runtime_details.jsonl"
)
DEFAULT_AUTO_STAGE3_REPORT = Path(
    "outputs/audits/automatic_proofkg_2wiki_train_k4_v1_n1500_seed42_"
    "historical_stage3_runtime/report.json"
)
DEFAULT_EVAL_DIR = Path("outputs/audits/saeg_v1_evaluation_protocol_v1")
DEFAULT_RESERVE82 = Path(
    "outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/"
    "verifier_reserve.question_only.jsonl"
)
DEFAULT_OUT = Path(
    "outputs/audits/2wiki_proofkg_extension_v1b_n300_seed42_preregistration"
)
SUPERSEDED_V1 = Path(
    "outputs/audits/2wiki_proofkg_extension_v1_n300_seed42_preregistration"
)
EVAL_FILES = (
    "development.question_only.jsonl",
    "confirmation.question_only.jsonl",
    "canonical_reporting.question_only.jsonl",
)
FORBIDDEN_OUTPUT_FIELDS = {
    "answer",
    "answers",
    "gold_answer",
    "gold_answers",
    "steps",
    "kg_subgraph",
    "supporting_facts",
    "support",
    "decomposition",
    "evidence",
    "reasoning",
}


@dataclass(frozen=True)
class IdentityRegistry:
    qids: frozenset[str]
    question_hashes: frozenset[str]
    families: frozenset[str]
    rows: int


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            result.append(value)
    return result


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def stable_rank(seed: int, label: str, *values: str) -> str:
    payload = "\0".join((str(seed), label, *map(str, values)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise_identity(row: Mapping[str, Any], *, label: str) -> dict[str, str]:
    dataset = str(row.get("dataset") or "").strip().lower()
    qid = str(row.get("qid") or row.get("id") or "").strip()
    question = str(row.get("question") or "").strip()
    if dataset != DATASET or not qid or not question:
        raise ValueError(f"{label}: invalid 2Wiki identity {dataset!r}::{qid!r}")
    qhash = question_sha256(question)
    stored_hash = str(row.get("question_sha256") or "").strip()
    if stored_hash and stored_hash != qhash:
        raise ValueError(f"{label}: question hash mismatch for {dataset}::{qid}")
    return {
        "dataset": dataset,
        "qid": qid,
        "question": question,
        "question_sha256": qhash,
        "family_sha256": family_sha256(question),
    }


def build_identity_registry(
    rows: Iterable[Mapping[str, Any]], *, label: str
) -> IdentityRegistry:
    qids: set[str] = set()
    hashes: set[str] = set()
    families: set[str] = set()
    count = 0
    for row in rows:
        if str(row.get("dataset") or "").strip().lower() != DATASET:
            continue
        identity = _normalise_identity(row, label=label)
        qids.add(identity["qid"])
        hashes.add(identity["question_sha256"])
        families.add(identity["family_sha256"])
        count += 1
    return IdentityRegistry(
        qids=frozenset(qids),
        question_hashes=frozenset(hashes),
        families=frozenset(families),
        rows=count,
    )


def _assignment_index(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        key = str(row.get("question_key") or "").strip()
        if not key:
            raise ValueError("assignment row lacks question_key")
        if key in result:
            raise ValueError(f"duplicate assignment key: {key}")
        result[key] = row
    return result


def _by_question_key(
    rows: Iterable[Mapping[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        key = str(row.get("question_key") or "").strip()
        if not key:
            dataset = str(row.get("dataset") or "").strip().lower()
            qid = str(row.get("qid") or "").strip()
            if dataset and qid:
                key = question_key(dataset, qid)
        if not key or key in result:
            raise ValueError(f"{label}: missing/duplicate question_key {key!r}")
        result[key] = row
    return result


def audit_existing_auto_capacity(
    auto_rows: Sequence[Mapping[str, Any]],
    question_kg_rows: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    *,
    protected_registry: IdentityRegistry,
    historical_cutoff: str,
) -> dict[str, Any]:
    """Recompute the strict 1299 -> protected-safe 659 capacity ledger."""

    if not historical_cutoff.strip():
        raise ValueError("historical cutoff is empty")
    qkg_by_key = _by_question_key(question_kg_rows, label="automatic question-KG")
    runtime_by_key = _by_question_key(runtime_rows, label="automatic runtime")
    strict: list[dict[str, str]] = []
    failed_checks: Counter[str] = Counter()
    for raw in auto_rows:
        identity = _normalise_identity(raw, label="auto1500 capacity cohort")
        key = question_key(DATASET, identity["qid"])
        runtime = runtime_by_key.get(key)
        if runtime is None:
            raise ValueError(f"auto1500 runtime join miss: {key}")
        decision = evaluate_graph_gate(
            runtime,
            dataset=DATASET,
            qid=identity["qid"],
            question=identity["question"],
            historical_cutoff=historical_cutoff,
        )
        if not decision.graph_eligible:
            for check, passed in decision.checks.items():
                if not passed:
                    failed_checks[check] += 1
            continue
        qkg = qkg_by_key.get(key)
        if qkg is None:
            raise ValueError(f"strict automatic proof lacks frozen question-KG: {key}")
        validate_question_kg_record(qkg)
        if (
            str(qkg.get("question_sha256") or "") != identity["question_sha256"]
            or qkg.get("kg_subgraph") != runtime.get("kg_subgraph")
        ):
            raise ValueError(f"automatic question-KG/runtime identity drift: {key}")
        strict.append(
            {
                **identity,
                "question_type": str(raw.get("question_type") or "unknown"),
            }
        )

    if len(qkg_by_key) != len(strict) or {
        question_key(DATASET, row["qid"]) for row in strict
    } != set(qkg_by_key):
        raise ValueError("strict source-gate set and frozen question-KG set differ")

    exclusion_counts: Counter[str] = Counter()
    safe: list[dict[str, str]] = []
    for row in strict:
        if row["qid"] in protected_registry.qids:
            exclusion_counts["protected_qid"] += 1
        elif row["question_sha256"] in protected_registry.question_hashes:
            exclusion_counts["protected_question_hash"] += 1
        elif row["family_sha256"] in protected_registry.families:
            exclusion_counts["protected_current_family"] += 1
        else:
            safe.append(row)

    return {
        "auto1500_rows": len(auto_rows),
        "runtime_rows": len(runtime_by_key),
        "frozen_complete_question_kg_rows": len(qkg_by_key),
        "strict_source_gate_eligible": len(strict),
        "strict_source_gate_ineligible": len(auto_rows) - len(strict),
        "strict_ineligible_check_counts": dict(sorted(failed_checks.items())),
        "protected_exclusion_counts_first_match": dict(
            sorted(exclusion_counts.items())
        ),
        "protected_safe_strict_eligible": len(safe),
        "protected_safe_unique_qids": len({row["qid"] for row in safe}),
        "protected_safe_current_families": len(
            {row["family_sha256"] for row in safe}
        ),
        "protected_safe_by_question_type": dict(
            sorted(Counter(row["question_type"] for row in safe).items())
        ),
        "checks": {
            "auto1500_exact": len(auto_rows) == 1500,
            "strict_source_gate_exact_1299": len(strict) == 1299,
            "protected_safe_exact_659": len(safe) == 659,
            "question_kg_identity_equals_strict_gate_set": len(qkg_by_key)
            == len(strict),
        },
    }


def _choose_max_family(
    candidates: Sequence[Mapping[str, Any]],
    *,
    n: int,
    qtype: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Choose deterministically, preferring one row per current family."""

    if n == 0:
        return []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in candidates:
        row = dict(raw)
        by_family[str(row["family_sha256"])].append(row)
    for family, family_rows in by_family.items():
        family_rows.sort(
            key=lambda row: (
                stable_rank(seed, f"{qtype}-within-family", family, row["qid"]),
                row["qid"],
            )
        )
    families = sorted(
        by_family,
        key=lambda family: (stable_rank(seed, f"{qtype}-family", family), family),
    )
    selected = [by_family[family][0] for family in families[:n]]
    if len(selected) < n:
        used_qids = {row["qid"] for row in selected}
        repeats = sorted(
            (
                row
                for family_rows in by_family.values()
                for row in family_rows
                if row["qid"] not in used_qids
            ),
            key=lambda row: (
                stable_rank(seed, f"{qtype}-family-repeat", row["qid"]),
                row["qid"],
            ),
        )
        selected.extend(repeats[: n - len(selected)])
    if len(selected) != n:
        raise ValueError(
            f"{qtype}: only {len(candidates)} eligible rows/"
            f"{len(by_family)} families; need {n}"
        )
    return selected


def select_extension_candidates(
    source_rows: Iterable[Mapping[str, Any]],
    assignment_rows: Iterable[Mapping[str, Any]],
    *,
    auto_train_registry: IdentityRegistry,
    protected_registry: IdentityRegistry,
    quotas: Mapping[str, int] = QUOTAS,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the frozen question-only extension and selection telemetry."""

    assignments = _assignment_index(assignment_rows)
    candidate_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_qids: set[str] = set()
    seen_hashes: set[str] = set()
    exclusion_counts: Counter[str] = Counter()
    source_2wiki = 0
    assignments_validated = 0

    for raw in source_rows:
        if str(raw.get("dataset") or "").strip().lower() != DATASET:
            continue
        source_2wiki += 1
        identity = _normalise_identity(raw, label="source curriculum")
        qid = identity["qid"]
        qhash = identity["question_sha256"]
        if qid in seen_qids or qhash in seen_hashes:
            raise ValueError(f"source curriculum duplicate qid/hash: {qid}")
        seen_qids.add(qid)
        seen_hashes.add(qhash)

        key = question_key(DATASET, qid)
        assignment = assignments.get(key)
        if assignment is None:
            raise ValueError(f"missing planner split assignment: {key}")
        if (
            str(assignment.get("dataset") or "").strip().lower() != DATASET
            or str(assignment.get("qid") or "").strip() != qid
            or str(assignment.get("split") or "").strip() != "train"
            or not str(assignment.get("family_sha256") or "").strip()
        ):
            raise ValueError(f"invalid/non-train planner assignment: {key}")
        if (raw.get("metadata") or {}).get("train_only") is not True:
            raise ValueError(f"source row is not explicitly train_only: {key}")
        assignments_validated += 1

        # The old automatic cohort is a train-side source.  Its current-family
        # values are intentionally *not* an exclusion set.
        if qid in auto_train_registry.qids:
            exclusion_counts["auto1500_qid"] += 1
            continue
        if qhash in auto_train_registry.question_hashes:
            exclusion_counts["auto1500_question_hash"] += 1
            continue

        # Evaluation and never-train reserve identities are protected at all
        # three levels under the current FAMILY_VERSION implementation.
        if qid in protected_registry.qids:
            exclusion_counts["protected_qid"] += 1
            continue
        if qhash in protected_registry.question_hashes:
            exclusion_counts["protected_question_hash"] += 1
            continue
        if identity["family_sha256"] in protected_registry.families:
            exclusion_counts["protected_family"] += 1
            continue

        qtype = str((raw.get("metadata") or {}).get("question_type") or "")
        if qtype not in quotas:
            exclusion_counts["unsupported_question_type"] += 1
            continue
        candidate_by_type[qtype].append(
            {
                "schema_version": SCHEMA_VERSION,
                "question_key": key,
                "dataset": DATASET,
                "qid": qid,
                "question": identity["question"],
                "question_sha256": qhash,
                "target_type": "relation_graph",
                "question_type": qtype,
                "family_version": FAMILY_VERSION,
                "family_sha256": identity["family_sha256"],
                "source_assignment_family_sha256": str(
                    assignment["family_sha256"]
                ),
                "source_split": "train",
                "source_role": "automatic_proofkg_extension_candidate",
                "gold_access": False,
                "evaluation_eligible": False,
            }
        )

    selected: list[dict[str, Any]] = []
    for qtype, quota in quotas.items():
        selected.extend(
            _choose_max_family(
                candidate_by_type.get(qtype, []),
                n=int(quota),
                qtype=qtype,
                seed=seed,
            )
        )
    selected.sort(
        key=lambda row: (
            stable_rank(seed, "extension-global-order", row["qid"]),
            row["qid"],
        )
    )
    for index, row in enumerate(selected, start=1):
        row["row_id"] = f"AUTO-PROOF-EXT-V1-{index:04d}"

    selected_qids = {row["qid"] for row in selected}
    selected_hashes = {row["question_sha256"] for row in selected}
    selected_families = {row["family_sha256"] for row in selected}
    if len(selected) != len(selected_qids) or len(selected) != len(selected_hashes):
        raise ValueError("selected extension is not unique by qid and question hash")
    if any(FORBIDDEN_OUTPUT_FIELDS.intersection(row) for row in selected):
        raise ValueError("Gold/process field leaked into question-only extension")

    gates = {
        "selected_count_exact": len(selected) == sum(int(v) for v in quotas.values()),
        "quota_counts_exact": Counter(row["question_type"] for row in selected)
        == Counter({key: int(value) for key, value in quotas.items() if int(value)}),
        "unique_qid_and_question_hash": (
            len(selected) == len(selected_qids) == len(selected_hashes)
        ),
        "auto1500_qid_overlap_zero": not (
            selected_qids & set(auto_train_registry.qids)
        ),
        "auto1500_question_hash_overlap_zero": not (
            selected_hashes & set(auto_train_registry.question_hashes)
        ),
        "protected_qid_overlap_zero": not (
            selected_qids & set(protected_registry.qids)
        ),
        "protected_question_hash_overlap_zero": not (
            selected_hashes & set(protected_registry.question_hashes)
        ),
        "protected_current_family_overlap_zero": not (
            selected_families & set(protected_registry.families)
        ),
        "all_assignment_split_train": all(
            row["source_split"] == "train" for row in selected
        ),
        "gold_access_false": all(row["gold_access"] is False for row in selected),
        "forbidden_gold_process_fields_zero": all(
            not FORBIDDEN_OUTPUT_FIELDS.intersection(row) for row in selected
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"extension selection gate failed: {gates}")

    telemetry = {
        "source_2wiki_rows": source_2wiki,
        "assignments_validated": assignments_validated,
        "exclusion_counts_first_match": dict(sorted(exclusion_counts.items())),
        "eligible_before_quota_by_question_type": dict(
            sorted((key, len(value)) for key, value in candidate_by_type.items())
        ),
        "eligible_current_families_by_question_type": dict(
            sorted(
                (
                    key,
                    len({row["family_sha256"] for row in value}),
                )
                for key, value in candidate_by_type.items()
            )
        ),
        "selected_by_question_type": dict(
            sorted(Counter(row["question_type"] for row in selected).items())
        ),
        "selected_unique_families": len(selected_families),
        "selected_rows_reusing_auto1500_current_family": sum(
            row["family_sha256"] in auto_train_registry.families
            for row in selected
        ),
        "selected_auto1500_current_family_overlap_count": len(
            selected_families & set(auto_train_registry.families)
        ),
        "gates": gates,
    }
    return selected, telemetry


def _family_function_sha256() -> str:
    source = inspect.getsource(question_family_signature) + "\n" + inspect.getsource(
        family_sha256
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--auto1500", type=Path, default=DEFAULT_AUTO1500)
    parser.add_argument(
        "--auto-question-kg", type=Path, default=DEFAULT_AUTO_QUESTION_KG
    )
    parser.add_argument("--auto-runtime", type=Path, default=DEFAULT_AUTO_RUNTIME)
    parser.add_argument(
        "--auto-stage3-report", type=Path, default=DEFAULT_AUTO_STAGE3_REPORT
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--reserve82", type=Path, default=DEFAULT_RESERVE82)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    args = parser.parse_args()

    eval_paths = [args.eval_dir / name for name in EVAL_FILES]
    input_paths = [
        args.source,
        args.assignments,
        args.auto1500,
        args.auto_question_kg,
        args.auto_runtime,
        args.auto_stage3_report,
        *eval_paths,
        args.reserve82,
    ]
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    auto_rows = read_jsonl(args.auto1500)
    eval_rows = [row for path in eval_paths for row in read_jsonl(path)]
    reserve_rows = read_jsonl(args.reserve82)
    auto_registry = build_identity_registry(auto_rows, label="auto1500")
    eval_registry = build_identity_registry(eval_rows, label="saeg evaluation")
    reserve_registry = build_identity_registry(reserve_rows, label="reserve82")
    protected_registry = IdentityRegistry(
        qids=frozenset(set(eval_registry.qids) | set(reserve_registry.qids)),
        question_hashes=frozenset(
            set(eval_registry.question_hashes) | set(reserve_registry.question_hashes)
        ),
        families=frozenset(
            set(eval_registry.families) | set(reserve_registry.families)
        ),
        rows=eval_registry.rows + reserve_registry.rows,
    )
    stage3_report = json.loads(args.auto_stage3_report.read_text(encoding="utf-8"))
    historical_cutoff = str(
        (stage3_report.get("cache_policy") or {}).get("historical_cutoff") or ""
    )
    existing_capacity = audit_existing_auto_capacity(
        auto_rows,
        read_jsonl(args.auto_question_kg),
        read_jsonl(args.auto_runtime),
        protected_registry=protected_registry,
        historical_cutoff=historical_cutoff,
    )
    if not all(existing_capacity["checks"].values()):
        raise ValueError(
            f"existing automatic ProofKG capacity drift: {existing_capacity['checks']}"
        )

    selected, telemetry = select_extension_candidates(
        read_jsonl(args.source),
        read_jsonl(args.assignments),
        auto_train_registry=auto_registry,
        protected_registry=protected_registry,
        quotas=QUOTAS,
        seed=args.seed,
    )
    if len(selected) != EXPECTED_N:
        raise ValueError(f"expected {EXPECTED_N} selected rows, got {len(selected)}")

    out, experiment_id = prepare_new_run_dir(
        args.out,
        experiment_id=args.experiment_id,
        extra={"phase": "freeze_2wiki_proofkg_extension_v1"},
    )
    cohort_path = out / "cohort.question_only.jsonl"
    write_jsonl(cohort_path, selected)
    generated_at = datetime.now(timezone.utc).isoformat()

    input_refs = {
        "source_curriculum": file_ref(args.source),
        "planner_assignments": file_ref(args.assignments),
        "auto1500_train_source": file_ref(args.auto1500),
        "auto1500_complete_question_kg": file_ref(args.auto_question_kg),
        "auto1500_runtime_details": file_ref(args.auto_runtime),
        "auto1500_stage3_report": file_ref(args.auto_stage3_report),
        "saeg_evaluation": [file_ref(path) for path in eval_paths],
        "never_train_reserve82": file_ref(args.reserve82),
    }
    protocol = {
        "schema_version": "2wiki-proofkg-extension-protocol-v1",
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at,
        "status": STATUS,
        "scope": "train-only question-only candidates for automatic 2Wiki ProofKG extension",
        "supersession": {
            "supersedes": str(SUPERSEDED_V1.resolve()),
            "reason": "v1 froze the same selection but did not bind the complete 1299-to-659 capacity recomputation ledger",
            "v1_files_modified_or_deleted": False,
        },
        "selection": {
            "seed": args.seed,
            "n": EXPECTED_N,
            "quotas": QUOTAS,
            "target_type": "relation_graph",
            "ranking": "SHA256(seed,label,current-family-or-qid), family-first then deterministic repeats",
            "family_version": FAMILY_VERSION,
            "family_function_sha256": _family_function_sha256(),
        },
        "isolation_policy": {
            "auto1500": {
                "role": "existing train-side source",
                "exclude": ["dataset::qid", "exact_question_sha256"],
                "current_family_reuse_allowed": True,
                "reason": "train-side family repeats are allowed; auto1500 is not evaluation or never-train data",
            },
            "saeg_all_evaluation_roles": {
                "roles": ["development", "confirmation", "canonical_reporting"],
                "exclude": [
                    "dataset::qid",
                    "exact_question_sha256",
                    f"{FAMILY_VERSION}:family_sha256",
                ],
            },
            "reserve82": {
                "role": "single-use rankability validation; never train",
                "exclude": [
                    "dataset::qid",
                    "exact_question_sha256",
                    f"{FAMILY_VERSION}:family_sha256",
                ],
            },
        },
        "existing_capacity_recomputation": {
            "result": existing_capacity,
            "historical_cutoff": historical_cutoff,
            "strict_gate": "trajectory-source-gate-hard-mask-v1",
            "protected_union": [
                "SAEG development",
                "SAEG confirmation",
                "SAEG canonical_reporting",
                "never-train reserve82",
            ],
            "family_version": FAMILY_VERSION,
            "family_function_sha256": _family_function_sha256(),
        },
        "gold_and_runtime_boundary": {
            "source_contains_gold_derived_fields": True,
            "selection_reads_gold_for_decisions": False,
            "output_field_whitelist_projection": True,
            "answers_steps_source_kg_copied": False,
            "gold_access": False,
            "planner_started": False,
            "retrieval_or_network_started": False,
            "proofkg_materialized": False,
            "training_started": False,
        },
        "next_stage": (
            "Run the frozen question-only planner, then an independently versioned "
            "historical executor/closure and trajectory-source-gate; structural "
            "completion is UNKNOWN at this freeze."
        ),
        "inputs": input_refs,
        "output_cohort": file_ref(cohort_path),
    }
    protocol_path = out / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": "2wiki-proofkg-extension-freeze-report-v1",
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at,
        "status": STATUS,
        "supersession": {
            "supersedes": str(SUPERSEDED_V1.resolve()),
            "reason": "capacity-ledger metadata completion; selected policy unchanged",
            "v1_files_modified_or_deleted": False,
        },
        "counts": telemetry,
        "existing_auto1500_strict_capacity": existing_capacity,
        "registries": {
            "auto1500": {
                "rows": auto_registry.rows,
                "unique_qids": len(auto_registry.qids),
                "unique_question_hashes": len(auto_registry.question_hashes),
                "current_families_telemetry_only": len(auto_registry.families),
            },
            "saeg_evaluation": {
                "rows_2wiki": eval_registry.rows,
                "unique_qids": len(eval_registry.qids),
                "unique_question_hashes": len(eval_registry.question_hashes),
                "current_families": len(eval_registry.families),
            },
            "reserve82": {
                "rows_2wiki": reserve_registry.rows,
                "unique_qids": len(reserve_registry.qids),
                "unique_question_hashes": len(reserve_registry.question_hashes),
                "current_families": len(reserve_registry.families),
            },
            "protected_union": {
                "rows_not_deduplicated": protected_registry.rows,
                "unique_qids": len(protected_registry.qids),
                "unique_question_hashes": len(
                    protected_registry.question_hashes
                ),
                "current_families": len(protected_registry.families),
                "family_version": FAMILY_VERSION,
                "family_function_sha256": _family_function_sha256(),
            },
        },
        "scientific_boundary": {
            "question_only": True,
            "gold_access": False,
            "automatic_proof_success_rate": "UNKNOWN",
            "strict_source_gate_eligible_count": "UNKNOWN",
            "training_authorized": False,
        },
        "inputs": input_refs,
        "outputs": {
            "cohort": file_ref(cohort_path),
            "protocol": file_ref(protocol_path),
        },
    }
    report_path = out / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        out,
        status=STATUS,
        extra={
            "phase": "freeze_2wiki_proofkg_extension_v1",
            "experiment_id": experiment_id,
            "cohort": file_ref(cohort_path),
            "protocol": file_ref(protocol_path),
            "report": file_ref(report_path),
            "planner_started": False,
            "training_started": False,
        },
    )
    print(
        json.dumps(
            {
                "status": STATUS,
                "output_dir": str(out.resolve()),
                "selected": telemetry["selected_by_question_type"],
                "gates": telemetry["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
