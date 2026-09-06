#!/usr/bin/env python3
"""Freeze a Gold-free official-raw 2Wiki ProofKG candidate release.

This stage selects identities only.  It does not copy answers, supporting
facts, passages, evidence, decompositions, or graphs, and it performs no
planner inference, retrieval, network access, or training.

Isolation scopes are explicit:

* the complete protected ledger and the SFT replay are excluded by
  dataset-scoped qid, exact question hash, and the *current* lexical family;
* the old automatic n=1500 cohort and extension n=350 are train-side sources,
  so only exact qid/hash reuse is forbidden for them.

Selection is deterministic and question-type-stratified.  Within every type,
one row per current lexical family is chosen before deterministic repeats.
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

from kgproweight.kg.question_kg import question_sha256
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_qpeg_v1_protocol import (
    FAMILY_VERSION,
    family_sha256,
    question_family_signature,
)


DATASET = "2wikimultihopqa"
SEED = 42
SCHEMA_VERSION = "2wiki-proofkg-official-raw-question-only-v2"
PROTOCOL_SCHEMA = "2wiki-proofkg-official-raw-protocol-v2"
REPORT_SCHEMA = "2wiki-proofkg-official-raw-freeze-report-v2"
STATUS = "FROZEN_GOLD_FREE_BEFORE_PLANNER_NOT_MATERIALIZED_NOT_TRAINED"
EXPERIMENT_ID = (
    "2WIKI-PROOFKG-OFFICIAL-RAW-V2-CANDIDATE-POOL-N1500-SEED42-"
    "PREREGISTRATION"
)
TARGET_TYPE = "relation_graph"
QUOTAS: dict[str, int] = {
    # Inference has only 331 safe official-train identities after all frozen
    # exclusions.  Allocate every one, then distribute the remaining 1,169 as
    # evenly as integers allow across the other three types.  This is a
    # candidate pool; the final mixed-PPO schedule still binds only 800 strict
    # ProofKG rows and 200 ordinary 2Wiki rows.
    "bridge_comparison": 390,
    "comparison": 390,
    "compositional": 389,
    "inference": 331,
}
EXPECTED_N = sum(QUOTAS.values())

DEFAULT_SOURCE = Path("data/2wikimultihopqa/train.jsonl")
DEFAULT_ASSIGNMENTS = Path(
    "data/silver_data/query_planner_supervision_split_v1_seed20260829/"
    "assignments.jsonl"
)
DEFAULT_PROTECTED_LEDGER = Path(
    "outputs/audits/mixed_ppo_v4_protected_identity_ledger_v2/"
    "protected_identities.question_only.jsonl"
)
DEFAULT_PROTECTED_REPORT = Path(
    "outputs/audits/mixed_ppo_v4_protected_identity_ledger_v2/report.json"
)
DEFAULT_REPLAY = Path(
    "data/silver_data/sft_replay_legacy_strong_train_rendered3to5_"
    "n2000_seed42_v2/silver_train.jsonl"
)
DEFAULT_AUTO1500 = Path(
    "outputs/audits/automatic_proofkg_2wiki_train_k4_v1_n1500_seed42_"
    "preregistration/cohort.question_only.jsonl"
)
DEFAULT_EXTENSION350 = Path(
    "outputs/audits/2wiki_proofkg_extension_combined_v1_n350_seed42_"
    "preregistration/cohort.question_only.jsonl"
)
DEFAULT_OUT = Path(
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "seed42_preregistration"
)

OUTPUT_FIELD_WHITELIST = frozenset(
    {
        "schema_version",
        "dataset",
        "qid",
        "question",
        "question_sha256",
        "family_version",
        "family_sha256",
        "question_type",
        "target_type",
        "gold_access",
    }
)


@dataclass(frozen=True)
class IdentityRegistry:
    qids: frozenset[tuple[str, str]]
    question_hashes: frozenset[tuple[str, str]]
    families: frozenset[tuple[str, str]]

    def reasons(self, row: Mapping[str, Any], *, include_family: bool) -> tuple[str, ...]:
        dataset = str(row["dataset"])
        reasons: list[str] = []
        if (dataset, str(row["qid"])) in self.qids:
            reasons.append("qid")
        if (dataset, str(row["question_sha256"])) in self.question_hashes:
            reasons.append("question_sha256")
        if include_family and (dataset, str(row["family_sha256"])) in self.families:
            reasons.append("family_sha256")
        return tuple(reasons)


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


def file_ref(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def stable_rank(seed: int, label: str, *values: str) -> str:
    payload = "\0".join((str(seed), str(label), *map(str, values)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalise_identity(
    row: Mapping[str, Any], *, dataset_override: str | None = None, label: str
) -> dict[str, str]:
    dataset = str(dataset_override or row.get("dataset") or "").strip().lower()
    qid = str(row.get("qid") or row.get("id") or "").strip()
    question = str(row.get("question") or "").strip()
    if dataset != DATASET or not qid or not question:
        raise ValueError(f"{label}: invalid identity {dataset!r}::{qid!r}")
    qhash = question_sha256(question)
    supplied = str(row.get("question_sha256") or "").strip()
    if supplied and supplied != qhash:
        raise ValueError(f"{label}: question hash mismatch for {dataset}::{qid}")
    return {
        "dataset": dataset,
        "qid": qid,
        "question": question,
        "question_sha256": qhash,
        "family_sha256": family_sha256(question),
    }


def build_registry(
    rows: Iterable[Mapping[str, Any]], *, label: str
) -> IdentityRegistry:
    identities = []
    for row in rows:
        dataset = str(row.get("dataset") or "").strip().lower()
        if dataset != DATASET:
            continue
        identities.append(normalise_identity(row, label=label))
    return IdentityRegistry(
        qids=frozenset((row["dataset"], row["qid"]) for row in identities),
        question_hashes=frozenset(
            (row["dataset"], row["question_sha256"]) for row in identities
        ),
        families=frozenset(
            (row["dataset"], row["family_sha256"]) for row in identities
        ),
    )


def _assignment_index(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        key = str(row.get("question_key") or "").strip()
        if not key or key in result:
            raise ValueError(f"missing/duplicate assignment key {key!r}")
        result[key] = row
    return result


def _select_family_first(
    candidates: Sequence[Mapping[str, Any]], *, n: int, qtype: str, seed: int
) -> list[dict[str, Any]]:
    """Select one row per family before any repeated-family row."""

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in candidates:
        by_family[str(raw["family_sha256"])].append(dict(raw))
    for family, rows in by_family.items():
        rows.sort(
            key=lambda row: (
                stable_rank(seed, f"{qtype}-within-family", family, str(row["qid"])),
                str(row["qid"]),
            )
        )
    families = sorted(
        by_family,
        key=lambda family: (stable_rank(seed, f"{qtype}-family", family), family),
    )
    heads = [by_family[family][0] for family in families]
    selected = heads[:n]
    if len(selected) < n:
        repeats: list[dict[str, Any]] = []
        for family in families:
            repeats.extend(by_family[family][1:])
        repeats.sort(
            key=lambda row: (
                stable_rank(seed, f"{qtype}-repeat", str(row["qid"])),
                str(row["qid"]),
            )
        )
        selected.extend(repeats[: n - len(selected)])
    if len(selected) != n:
        raise ValueError(
            f"{qtype}: only {len(candidates)} eligible rows/"
            f"{len(by_family)} current families; need {n}"
        )
    return selected


def select_official_raw_candidates(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    assignment_rows: Sequence[Mapping[str, Any]],
    protected_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    old_train_rows: Sequence[Mapping[str, Any]],
    quotas: Mapping[str, int] = QUOTAS,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the deterministic, Gold-free official-raw cohort."""

    assignments = _assignment_index(assignment_rows)
    protected = build_registry(protected_rows, label="protected ledger")
    replay = build_registry(replay_rows, label="SFT replay")
    old_train = build_registry(old_train_rows, label="old automatic train sources")
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusion: Counter[str] = Counter()
    source_qids: dict[str, str] = {}
    source_hashes: set[str] = set()
    assignment_family_mismatches = 0

    # Official 2Wiki contains a small number of duplicate question surfaces.
    # Sort first so exact-surface deduplication is byte-order independent.
    ordered_source = sorted(
        source_rows,
        key=lambda row: (
            str(row.get("id") or row.get("qid") or ""),
            str(row.get("question") or ""),
        ),
    )
    for raw in ordered_source:
        identity = normalise_identity(
            raw, dataset_override=DATASET, label="official raw source"
        )
        qid = identity["qid"]
        qhash = identity["question_sha256"]
        prior_qhash = source_qids.get(qid)
        if prior_qhash is not None:
            if prior_qhash != qhash:
                raise ValueError(f"official raw qid has conflicting questions: {qid}")
            exclusion["duplicate_source_qid"] += 1
            continue
        if qhash in source_hashes:
            exclusion["duplicate_source_question_hash"] += 1
            continue
        source_qids[qid] = qhash
        source_hashes.add(qhash)

        key = f"{DATASET}::{qid}"
        assignment = assignments.get(key)
        if assignment is None:
            exclusion["missing_assignment"] += 1
            continue
        if (
            str(assignment.get("dataset") or "").strip().lower() != DATASET
            or str(assignment.get("qid") or "").strip() != qid
        ):
            raise ValueError(f"planner assignment identity mismatch: {key}")
        if str(assignment.get("split") or "").strip() != "train":
            exclusion["non_train_assignment"] += 1
            continue
        stored_assignment_family = str(assignment.get("family_sha256") or "").strip()
        if not stored_assignment_family:
            raise ValueError(f"planner assignment lacks family provenance: {key}")
        if stored_assignment_family != identity["family_sha256"]:
            assignment_family_mismatches += 1

        reasons = protected.reasons(identity, include_family=True)
        if reasons:
            exclusion[f"protected_{reasons[0]}"] += 1
            continue
        reasons = replay.reasons(identity, include_family=True)
        if reasons:
            exclusion[f"replay_{reasons[0]}"] += 1
            continue
        reasons = old_train.reasons(identity, include_family=False)
        if reasons:
            exclusion[f"old_train_exact_{reasons[0]}"] += 1
            continue

        qtype = str((raw.get("metadata") or {}).get("type") or "").strip()
        if qtype not in quotas:
            exclusion["unsupported_question_type"] += 1
            continue
        candidates[qtype].append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset": DATASET,
                "qid": qid,
                "question": identity["question"],
                "question_sha256": qhash,
                "family_version": FAMILY_VERSION,
                "family_sha256": identity["family_sha256"],
                "question_type": qtype,
                "target_type": TARGET_TYPE,
                "gold_access": False,
            }
        )

    selected: list[dict[str, Any]] = []
    for qtype, quota in quotas.items():
        selected.extend(
            _select_family_first(
                candidates.get(qtype, []), n=int(quota), qtype=qtype, seed=seed
            )
        )
    selected.sort(
        key=lambda row: (
            stable_rank(seed, "official-raw-global-order", str(row["qid"])),
            str(row["qid"]),
        )
    )

    qid_keys = {(row["dataset"], row["qid"]) for row in selected}
    hash_keys = {(row["dataset"], row["question_sha256"]) for row in selected}
    family_keys = {(row["dataset"], row["family_sha256"]) for row in selected}
    selected_by_type = Counter(str(row["question_type"]) for row in selected)
    selected_families_by_type = {
        qtype: len(
            {
                row["family_sha256"]
                for row in selected
                if row["question_type"] == qtype
            }
        )
        for qtype in quotas
    }
    gates = {
        "selected_n_exact": len(selected) == sum(map(int, quotas.values())),
        "quotas_exact": selected_by_type
        == Counter({qtype: int(n) for qtype, n in quotas.items() if int(n)}),
        "dataset_scoped_qid_unique": len(qid_keys) == len(selected),
        "dataset_scoped_question_hash_unique": len(hash_keys) == len(selected),
        "protected_qid_hash_family_overlap_zero": all(
            not protected.reasons(row, include_family=True) for row in selected
        ),
        "replay_qid_hash_family_overlap_zero": all(
            not replay.reasons(row, include_family=True) for row in selected
        ),
        "old_auto1500_extension350_exact_qid_hash_overlap_zero": all(
            not old_train.reasons(row, include_family=False) for row in selected
        ),
        "output_field_whitelist_exact": all(
            set(row) == OUTPUT_FIELD_WHITELIST for row in selected
        ),
        "gold_access_false": all(row["gold_access"] is False for row in selected),
        "target_type_relation_graph": all(
            row["target_type"] == TARGET_TYPE for row in selected
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"official-raw candidate gates failed: {gates}")

    capacity_by_type = {
        qtype: {
            "eligible_rows": len(candidates.get(qtype, [])),
            "unique_current_families": len(
                {row["family_sha256"] for row in candidates.get(qtype, [])}
            ),
            "quota": int(quota),
            "row_margin_after_quota": len(candidates.get(qtype, [])) - int(quota),
            "selected_rows": selected_by_type[qtype],
            "selected_unique_current_families": selected_families_by_type[qtype],
            "selected_repeated_family_rows": int(quota)
            - selected_families_by_type[qtype],
        }
        for qtype, quota in quotas.items()
    }
    telemetry = {
        "source_rows": len(source_rows),
        "source_unique_qids": len(source_qids),
        "source_unique_question_hashes": len(source_hashes),
        "planner_assignments": len(assignments),
        "assignment_family_hash_mismatch_rows_telemetry_only": assignment_family_mismatches,
        "exclusion_counts_first_match": dict(sorted(exclusion.items())),
        "eligible_total": sum(len(rows) for rows in candidates.values()),
        "eligible_unique_current_families": len(
            {
                (row["dataset"], row["family_sha256"])
                for rows in candidates.values()
                for row in rows
            }
        ),
        "capacity_by_question_type": capacity_by_type,
        "selected_unique_current_families": len(family_keys),
        "selected_repeated_family_rows": len(selected) - len(family_keys),
        "gates": gates,
    }
    return selected, telemetry


def validate_protected_ledger_release(
    *, ledger_path: Path, report_path: Path
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("complete") is not True:
        raise ValueError("protected ledger report is not complete")
    if report.get("current_family_recomputed") is not True:
        raise ValueError("protected ledger did not recompute current families")
    output = report.get("output") or {}
    actual_sha = sha256_file(ledger_path)
    if str(output.get("sha256") or "") != actual_sha:
        raise ValueError("protected ledger/report SHA256 mismatch")
    if int(output.get("rows") or -1) != len(read_jsonl(ledger_path)):
        raise ValueError("protected ledger/report row-count mismatch")
    return report


def family_function_sha256() -> str:
    source = inspect.getsource(question_family_signature) + "\n" + inspect.getsource(
        family_sha256
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def freeze_official_raw(
    *,
    source_path: Path,
    assignments_path: Path,
    protected_ledger_path: Path,
    protected_report_path: Path,
    replay_path: Path,
    auto1500_path: Path,
    extension350_path: Path,
    output_dir: Path,
    quotas: Mapping[str, int] = QUOTAS,
    seed: int = SEED,
    experiment_id: str = EXPERIMENT_ID,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite versioned freeze: {output_dir}")
    required = (
        source_path,
        assignments_path,
        protected_ledger_path,
        protected_report_path,
        replay_path,
        auto1500_path,
        extension350_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if not str(experiment_id).strip():
        raise ValueError("Experiment ID must be nonempty")
    ledger_report = validate_protected_ledger_release(
        ledger_path=protected_ledger_path, report_path=protected_report_path
    )
    selected, telemetry = select_official_raw_candidates(
        source_rows=read_jsonl(source_path),
        assignment_rows=read_jsonl(assignments_path),
        protected_rows=read_jsonl(protected_ledger_path),
        replay_rows=read_jsonl(replay_path),
        old_train_rows=[
            *read_jsonl(auto1500_path),
            *read_jsonl(extension350_path),
        ],
        quotas=quotas,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    cohort_path = output_dir / "cohort.question_only.jsonl"
    write_jsonl(cohort_path, selected)
    generated_at = datetime.now(timezone.utc).isoformat()
    inputs = {
        "official_raw_train": file_ref(source_path),
        "planner_split_assignments": file_ref(assignments_path),
        "complete_protected_ledger": file_ref(protected_ledger_path),
        "complete_protected_ledger_report": file_ref(protected_report_path),
        "sft_replay": file_ref(replay_path),
        "old_auto1500_exact_exclusion": file_ref(auto1500_path),
        "extension350_exact_exclusion": file_ref(extension350_path),
    }
    isolation = {
        "identity_scope": "dataset-scoped",
        "complete_protected_ledger": [
            "qid",
            "exact_question_sha256",
            f"{FAMILY_VERSION}:family_sha256",
        ],
        "sft_replay": [
            "qid",
            "exact_question_sha256",
            f"{FAMILY_VERSION}:family_sha256",
        ],
        "old_auto1500_and_extension350": ["qid", "exact_question_sha256"],
        "old_train_current_family_reuse_allowed": True,
    }
    selection = {
        "seed": seed,
        "n": sum(map(int, quotas.values())),
        "quotas": dict(quotas),
        "target_type": TARGET_TYPE,
        "algorithm": (
            "question-type-stratified; deterministic family rank; one row per "
            "current family before deterministic repeats"
        ),
        "family_version": FAMILY_VERSION,
        "family_function_sha256": family_function_sha256(),
        "capacity": telemetry["capacity_by_question_type"],
    }
    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "experiment_id": str(experiment_id).strip(),
        "generated_at_utc": generated_at,
        "status": STATUS,
        "selection": selection,
        "isolation_policy": isolation,
        "protected_ledger_binding": {
            "complete": ledger_report["complete"],
            "current_family_recomputed": ledger_report["current_family_recomputed"],
            "unique": ledger_report.get("unique"),
            "ledger_sha256": sha256_file(protected_ledger_path),
        },
        "gold_boundary": {
            "official_raw_source_contains_gold_and_evidence_fields": True,
            "full_source_json_objects_decoded": True,
            "gold_or_evidence_values_used_for_selection": False,
            "selection_fields": [
                "id",
                "question",
                "metadata.type",
                "planner assignment split",
            ],
            "output_field_whitelist": sorted(OUTPUT_FIELD_WHITELIST),
            "gold_access": False,
        },
        "execution_boundary": {
            "planner_started": False,
            "retrieval_started": False,
            "network_accessed": False,
            "gpu_used": False,
            "proofkg_materialized": False,
            "training_started": False,
        },
        "inputs": inputs,
        "output": file_ref(cohort_path),
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": str(experiment_id).strip(),
        "generated_at_utc": generated_at,
        "status": STATUS,
        "counts": telemetry,
        "scientific_boundary": {
            "candidate_identity_freeze_only": True,
            "structural_proof_yield": "UNKNOWN_NOT_RUN",
            "semantic_proof_quality": "UNKNOWN_NOT_RUN",
            "training_utility": "UNKNOWN_NOT_RUN",
            "inference_quota_uses_repeated_current_families": telemetry[
                "capacity_by_question_type"
            ]["inference"]["selected_repeated_family_rows"]
            > 0,
        },
        "inputs": inputs,
        "outputs": {
            "cohort": file_ref(cohort_path),
            "protocol": file_ref(protocol_path),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        output_dir,
        status=STATUS,
        extra={
            "phase": "freeze_2wiki_proofkg_official_raw_v2",
            "experiment_id": str(experiment_id).strip(),
            "cohort": file_ref(cohort_path),
            "protocol": file_ref(protocol_path),
            "report": file_ref(report_path),
            "planner_started": False,
            "retrieval_started": False,
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument(
        "--protected-ledger", type=Path, default=DEFAULT_PROTECTED_LEDGER
    )
    parser.add_argument(
        "--protected-report", type=Path, default=DEFAULT_PROTECTED_REPORT
    )
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--auto1500", type=Path, default=DEFAULT_AUTO1500)
    parser.add_argument("--extension350", type=Path, default=DEFAULT_EXTENSION350)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    args = parser.parse_args()
    report = freeze_official_raw(
        source_path=args.source,
        assignments_path=args.assignments,
        protected_ledger_path=args.protected_ledger,
        protected_report_path=args.protected_report,
        replay_path=args.replay,
        auto1500_path=args.auto1500,
        extension350_path=args.extension350,
        output_dir=args.out,
        seed=args.seed,
        experiment_id=args.experiment_id,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected": report["counts"]["capacity_by_question_type"],
                "gates": report["counts"]["gates"],
                "output": str(args.out.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
