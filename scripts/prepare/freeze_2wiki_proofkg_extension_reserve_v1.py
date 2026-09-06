#!/usr/bin/env python3
"""Freeze a v4-clean 2Wiki ProofKG reserve and one combined planner cohort.

The already-frozen v1b cohort contains 300 question-only identities.  This
append-only release selects 50 additional train identities (30 inference,
12 compositional, 8 comparison), then emits a deterministic 350-row union so
the planner can be loaded once.  Neither parent release is modified.

Train-side sources (the old automatic n=1500 and v1b n=300) are excluded by
qid and exact question hash.  Evaluation and never-train identities are
excluded by qid, exact question hash, and the current lexical-family hash.
No answer, source step, KG, support annotation, or decomposition is written.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_sha256
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_2wiki_proofkg_extension_v1 import (
    DATASET,
    DEFAULT_ASSIGNMENTS,
    DEFAULT_AUTO1500,
    FORBIDDEN_OUTPUT_FIELDS,
    IdentityRegistry,
    build_identity_registry,
    file_ref,
    read_jsonl,
    select_extension_candidates,
    stable_rank,
    write_jsonl,
)
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256


SEED = 42
RESERVE_QUOTAS: dict[str, int] = {
    "inference": 30,
    "compositional": 12,
    "comparison": 8,
}
RESERVE_N = sum(RESERVE_QUOTAS.values())
PARENT_N = 300
COMBINED_N = PARENT_N + RESERVE_N
RESERVE_SCHEMA = "automatic-proofkg-extension-reserve-question-only-v1"
COMBINED_SCHEMA = "automatic-proofkg-extension-combined-question-only-v1"
STATUS = "FROZEN_TRAIN_ONLY_BEFORE_PLANNER_NOT_MATERIALIZED"
RESERVE_EXPERIMENT_ID = (
    "2WIKI-PROOFKG-EXTENSION-RESERVE-V1-N50-SEED42-PREREGISTRATION"
)
COMBINED_EXPERIMENT_ID = (
    "2WIKI-PROOFKG-EXTENSION-COMBINED-V1-N350-SEED42-PREREGISTRATION"
)
DEFAULT_PARENT = Path(
    "outputs/audits/2wiki_proofkg_extension_v1b_n300_seed42_preregistration"
)
DEFAULT_SOURCE = Path("data/2wikimultihopqa/train.jsonl")
DEFAULT_RESERVE_OUT = Path(
    "outputs/audits/2wiki_proofkg_extension_reserve_v1_n50_seed42_preregistration"
)
DEFAULT_COMBINED_OUT = Path(
    "outputs/audits/2wiki_proofkg_extension_combined_v1_n350_seed42_preregistration"
)
DEFAULT_PROTECTED = (
    Path("outputs/audits/qpeg_v1_n1350_seed42_preregistration/pilot.question_only.jsonl"),
    Path("outputs/audits/qpeg_v1_n1350_seed42_preregistration/confirmation.question_only.jsonl"),
    Path("outputs/audits/qpeg_v1_n1350_seed42_preregistration/final.question_only.jsonl"),
    Path("outputs/audits/saeg_v1_evaluation_protocol_v1/development.question_only.jsonl"),
    Path("outputs/audits/saeg_v1_evaluation_protocol_v1/confirmation.question_only.jsonl"),
    Path("outputs/audits/saeg_v1_evaluation_protocol_v1/canonical_reporting.question_only.jsonl"),
    Path(
        "outputs/audits/subquestion_decomposition_v8_cohort_freeze_"
        "dev30_prospective300_seed20260904_v1/development.identity_only.jsonl"
    ),
    Path(
        "outputs/audits/subquestion_decomposition_v8_cohort_freeze_"
        "dev30_prospective300_seed20260904_v1/prospective.identity_only.jsonl"
    ),
    Path(
        "outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/"
        "verifier_reserve.question_only.jsonl"
    ),
    Path(
        "outputs/audits/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_"
        "protocol/ordinary200.question_only.jsonl"
    ),
    Path(
        "data/silver_data/sft_replay_legacy_strong_train_rendered3to5_"
        "n2000_seed42_v1c/silver_train.jsonl"
    ),
)


def _merge_registry(rows: Iterable[Mapping[str, Any]], *, label: str) -> IdentityRegistry:
    return build_identity_registry(rows, label=label)


def project_official_train_identities(
    raw_rows: Sequence[Mapping[str, Any]],
    assignment_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Project the official train split onto identity/type fields only.

    The raw file contains Gold answers and supporting annotations.  They are
    deliberately neither inspected nor copied.  Planner assignments are the
    authority for the train-only split; rows absent from that versioned split
    are recorded and skipped rather than guessed.
    """

    assignments = {
        str(row.get("question_key") or ""): row for row in assignment_rows
    }
    candidates: list[dict[str, Any]] = []
    counts = Counter()
    for raw in raw_rows:
        qid = str(raw.get("id") or raw.get("qid") or "").strip()
        question = str(raw.get("question") or "").strip()
        if not qid or not question:
            counts["invalid_identity"] += 1
            continue
        key = f"{DATASET}::{qid}"
        assignment = assignments.get(key)
        if assignment is None:
            counts["missing_assignment"] += 1
            continue
        if str(assignment.get("split") or "").strip() != "train":
            counts["non_train_assignment"] += 1
            continue
        qtype = str((raw.get("metadata") or {}).get("type") or "").strip()
        if not qtype:
            counts["missing_question_type"] += 1
            continue
        candidates.append(
            {
                "dataset": DATASET,
                "qid": qid,
                "question": question,
                "metadata": {"question_type": qtype, "train_only": True},
            }
        )
    candidates.sort(key=lambda row: (str(row["qid"]), str(row["question"])))
    output: list[dict[str, Any]] = []
    seen_qids: set[str] = set()
    seen_hashes: set[str] = set()
    for row in candidates:
        qid = str(row["qid"])
        qhash = question_sha256(str(row["question"]))
        if qid in seen_qids:
            counts["duplicate_qid"] += 1
            continue
        if qhash in seen_hashes:
            counts["duplicate_question_hash"] += 1
            continue
        seen_qids.add(qid)
        seen_hashes.add(qhash)
        output.append(row)
        counts["projected_train"] += 1
    return output, dict(sorted(counts.items()))


def _validate_question_only(rows: Sequence[Mapping[str, Any]], *, expected_n: int) -> None:
    if len(rows) != expected_n:
        raise ValueError(f"expected {expected_n} question-only rows, got {len(rows)}")
    qids: set[str] = set()
    hashes: set[str] = set()
    for row in rows:
        dataset = str(row.get("dataset") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        question = str(row.get("question") or "").strip()
        if dataset != DATASET or not qid or not question:
            raise ValueError(f"invalid combined identity: {dataset!r}::{qid!r}")
        if row.get("question_sha256") != question_sha256(question):
            raise ValueError(f"question hash mismatch: {dataset}::{qid}")
        if row.get("family_sha256") != family_sha256(question):
            raise ValueError(f"family hash mismatch: {dataset}::{qid}")
        if row.get("gold_access") is not False:
            raise ValueError(f"gold_access is not false: {dataset}::{qid}")
        if FORBIDDEN_OUTPUT_FIELDS.intersection(row):
            raise ValueError(f"forbidden field in question-only row: {dataset}::{qid}")
        if qid in qids or str(row["question_sha256"]) in hashes:
            raise ValueError(f"duplicate qid/hash: {dataset}::{qid}")
        qids.add(qid)
        hashes.add(str(row["question_sha256"]))


def build_reserve_and_combined(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    assignment_rows: Sequence[Mapping[str, Any]],
    auto1500_rows: Sequence[Mapping[str, Any]],
    parent_rows: Sequence[Mapping[str, Any]],
    protected_rows: Sequence[Mapping[str, Any]],
    quotas: Mapping[str, int] = RESERVE_QUOTAS,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select reserve rows and build a deterministic parent+reserve union."""

    _validate_question_only(parent_rows, expected_n=PARENT_N)
    train_registry = _merge_registry(
        [*auto1500_rows, *parent_rows], label="auto1500 plus extension-v1b"
    )
    protected_registry = _merge_registry(protected_rows, label="v4 protected union")
    reserve, selection = select_extension_candidates(
        source_rows,
        assignment_rows,
        auto_train_registry=train_registry,
        protected_registry=protected_registry,
        quotas=quotas,
        seed=seed,
    )
    for index, row in enumerate(reserve, start=1):
        row["schema_version"] = RESERVE_SCHEMA
        row["row_id"] = f"AUTO-PROOF-EXT-RESERVE-V1-{index:04d}"
        row["source_role"] = "automatic_proofkg_extension_reserve_candidate"
    _validate_question_only(reserve, expected_n=sum(int(value) for value in quotas.values()))

    combined = [dict(row) for row in [*parent_rows, *reserve]]
    combined.sort(
        key=lambda row: (
            stable_rank(seed, "extension-combined-planner-order", str(row["qid"])),
            str(row["qid"]),
        )
    )
    for row in combined:
        row["combined_schema_version"] = COMBINED_SCHEMA
    _validate_question_only(combined, expected_n=PARENT_N + len(reserve))

    combined_qids = {str(row["qid"]) for row in combined}
    parent_qids = {str(row["qid"]) for row in parent_rows}
    reserve_qids = {str(row["qid"]) for row in reserve}
    protected_qids = set(protected_registry.qids)
    protected_hashes = set(protected_registry.question_hashes)
    protected_families = set(protected_registry.families)
    parent_hashes = {str(row["question_sha256"]) for row in parent_rows}
    parent_families = {str(row["family_sha256"]) for row in parent_rows}
    reserve_hashes = {str(row["question_sha256"]) for row in reserve}
    reserve_families = {str(row["family_sha256"]) for row in reserve}
    combined_hashes = {str(row["question_sha256"]) for row in combined}
    combined_families = {str(row["family_sha256"]) for row in combined}
    gates = {
        "reserve_n_exact": len(reserve) == sum(int(value) for value in quotas.values()),
        "reserve_quotas_exact": Counter(row["question_type"] for row in reserve)
        == Counter({key: int(value) for key, value in quotas.items() if int(value)}),
        "parent_reserve_qid_disjoint": not (parent_qids & reserve_qids),
        "combined_exact_union": combined_qids == parent_qids | reserve_qids,
        "combined_n_exact": len(combined) == PARENT_N + len(reserve),
        "reserve_protected_qid_overlap_zero": not (reserve_qids & protected_qids),
        "reserve_protected_question_hash_overlap_zero": not (
            reserve_hashes & protected_hashes
        ),
        "reserve_protected_current_family_overlap_zero": not (
            reserve_families & protected_families
        ),
        # v1b predates the broader v4 protected union and has 19 known family
        # overlaps.  The combined file must preserve that parent exactly for a
        # one-load planner run; this gate proves the reserve adds no new leak.
        # The downstream v4 selector still removes all inherited overlaps.
        "combined_adds_no_protected_qid_overlap": (combined_qids & protected_qids)
        == (parent_qids & protected_qids),
        "combined_adds_no_protected_question_hash_overlap": (
            combined_hashes & protected_hashes
        )
        == (parent_hashes & protected_hashes),
        "combined_adds_no_protected_current_family_overlap": (
            combined_families & protected_families
        )
        == (parent_families & protected_families),
        "reserve_not_parent_family_isolation_claim": True,
        "combined_parent_preserved_for_planner_efficiency": set(parent_qids).issubset(
            combined_qids
        ),
        "gold_access_false": all(row.get("gold_access") is False for row in combined),
        "forbidden_gold_process_fields_zero": all(
            not FORBIDDEN_OUTPUT_FIELDS.intersection(row) for row in combined
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"reserve/combined gate failed: {gates}")
    return reserve, combined, {
        "selection": selection,
        "gates": gates,
        "inherited_parent_protected_overlap": {
            "qid": len(parent_qids & protected_qids),
            "question_sha256": len(parent_hashes & protected_hashes),
            "family_sha256": len(parent_families & protected_families),
        },
        "reserve_protected_overlap": {
            "qid": len(reserve_qids & protected_qids),
            "question_sha256": len(reserve_hashes & protected_hashes),
            "family_sha256": len(reserve_families & protected_families),
        },
    }


def _write_release(
    *,
    out: Path,
    experiment_id: str,
    rows: Sequence[Mapping[str, Any]],
    inputs: Mapping[str, Any],
    selection: Mapping[str, Any],
    role: str,
) -> None:
    out.mkdir(parents=True, exist_ok=False)
    cohort = out / "cohort.question_only.jsonl"
    write_jsonl(cohort, rows)
    generated_at = datetime.now(timezone.utc).isoformat()
    protocol = {
        "schema_version": f"2wiki-proofkg-{role}-protocol-v1",
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at,
        "status": STATUS,
        "scope": f"train-only question-only 2Wiki ProofKG {role}",
        "selection": dict(selection),
        "isolation_policy": {
            "old_auto1500_and_parent_v1b": ["dataset::qid", "exact_question_sha256"],
            "evaluation_and_never_train": [
                "dataset::qid",
                "exact_question_sha256",
                f"{FAMILY_VERSION}:family_sha256",
            ],
            "train_side_family_reuse_allowed": True,
            "combined_parent_inheritance": (
                "The combined planner cohort preserves v1b exactly. Any broader-v4 "
                "family overlaps inherited from v1b are reported and removed by the "
                "downstream v4 selection; the new reserve adds none."
            ),
        },
        "gold_and_runtime_boundary": {
            "output_question_only": True,
            "answers_steps_source_kg_copied": False,
            "gold_access": False,
            "planner_started": False,
            "retrieval_or_network_started": False,
            "proofkg_materialized": False,
            "training_started": False,
        },
        "inputs": dict(inputs),
        "output_cohort": file_ref(cohort),
    }
    protocol_path = out / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": f"2wiki-proofkg-{role}-freeze-report-v1",
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at,
        "status": STATUS,
        "counts": {
            "n": len(rows),
            "by_question_type": dict(
                sorted(Counter(str(row["question_type"]) for row in rows).items())
            ),
            "unique_families": len({str(row["family_sha256"]) for row in rows}),
        },
        "gates": dict(selection.get("gates") or {}),
        "inputs": dict(inputs),
        "outputs": {"cohort": file_ref(cohort), "protocol": file_ref(protocol_path)},
        "training_started": False,
    }
    report_path = out / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        out,
        status=STATUS,
        extra={
            "phase": f"freeze_2wiki_proofkg_{role}_v1",
            "experiment_id": experiment_id,
            "report": file_ref(report_path),
            "protocol": file_ref(protocol_path),
            "training_started": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--auto1500", type=Path, default=DEFAULT_AUTO1500)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--protected", action="append", type=Path, default=None)
    parser.add_argument("--reserve-out", type=Path, default=DEFAULT_RESERVE_OUT)
    parser.add_argument("--combined-out", type=Path, default=DEFAULT_COMBINED_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    protected_paths = tuple(args.protected or DEFAULT_PROTECTED)
    parent_cohort = args.parent / "cohort.question_only.jsonl"
    parent_protocol = args.parent / "protocol.json"
    paths = [
        args.source,
        args.assignments,
        args.auto1500,
        parent_cohort,
        parent_protocol,
        *protected_paths,
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.reserve_out.exists() or args.combined_out.exists():
        raise SystemExit(
            "refusing to overwrite append-only reserve/combined output: "
            f"{args.reserve_out}, {args.combined_out}"
        )

    parent_protocol_value = json.loads(parent_protocol.read_text(encoding="utf-8"))
    if (
        parent_protocol_value.get("status") != STATUS
        or (parent_protocol_value.get("output_cohort") or {}).get("sha256")
        != file_ref(parent_cohort)["sha256"]
    ):
        raise ValueError("parent v1b protocol/cohort binding is invalid")
    parent_rows = read_jsonl(parent_cohort)
    protected_rows = [row for path in protected_paths for row in read_jsonl(path)]
    assignment_rows = read_jsonl(args.assignments)
    projected_source, projection_counts = project_official_train_identities(
        read_jsonl(args.source), assignment_rows
    )
    reserve, combined, telemetry = build_reserve_and_combined(
        source_rows=projected_source,
        assignment_rows=assignment_rows,
        auto1500_rows=read_jsonl(args.auto1500),
        parent_rows=parent_rows,
        protected_rows=protected_rows,
        quotas=RESERVE_QUOTAS,
        seed=args.seed,
    )
    common_inputs = {
        "source_curriculum": file_ref(args.source),
        "planner_assignments": file_ref(args.assignments),
        "old_auto1500": file_ref(args.auto1500),
        "parent_v1b_cohort": file_ref(parent_cohort),
        "parent_v1b_protocol": file_ref(parent_protocol),
        "protected": [file_ref(path) for path in protected_paths],
        "official_train_projection_counts": projection_counts,
    }
    _write_release(
        out=args.reserve_out,
        experiment_id=RESERVE_EXPERIMENT_ID,
        rows=reserve,
        inputs=common_inputs,
        selection={
            "seed": args.seed,
            "n": RESERVE_N,
            "quotas": RESERVE_QUOTAS,
            "ranking": "family-first deterministic SHA256 rank",
            "source_projection": "official train -> dataset/qid/question/question_type only",
            **telemetry,
        },
        role="extension-reserve",
    )
    combined_inputs = {
        "parent_v1b": file_ref(parent_cohort),
        "reserve_v1": file_ref(args.reserve_out / "cohort.question_only.jsonl"),
        "reserve_protocol": file_ref(args.reserve_out / "protocol.json"),
    }
    _write_release(
        out=args.combined_out,
        experiment_id=COMBINED_EXPERIMENT_ID,
        rows=combined,
        inputs=combined_inputs,
        selection={
            "seed": args.seed,
            "n": COMBINED_N,
            "composition": {"parent_v1b": PARENT_N, "reserve_v1": RESERVE_N},
            "order": "SHA256(seed,extension-combined-planner-order,qid)",
            "gates": telemetry["gates"],
        },
        role="extension-combined",
    )
    print(
        json.dumps(
            {
                "status": STATUS,
                "reserve": file_ref(args.reserve_out / "cohort.question_only.jsonl"),
                "combined": file_ref(args.combined_out / "cohort.question_only.jsonl"),
                "reserve_by_question_type": dict(
                    sorted(Counter(row["question_type"] for row in reserve).items())
                ),
                "gates": telemetry["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
