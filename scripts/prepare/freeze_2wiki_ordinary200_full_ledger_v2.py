#!/usr/bin/env python3
"""Freeze a full-ledger-safe 2Wiki ordinary200 with source provenance.

The old ordinary200 is not assumed safe.  Safe parent identities are retained;
blocked rows are deterministically replaced from the train-only 2Wiki
curriculum.  Every output row points to an already materialized outcome and
passage source by versioned file, one-based line number, record hash, and
passage hash.  No answer, trajectory, passage text, KG, or evidence is copied.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.utils.logging import dump_manifest
from scripts.prepare import freeze_mixed_ppo_three_dataset_v4_proof800 as v4
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION


SCHEMA_VERSION = "2wiki-ordinary-outcome-identity-provenance-v2"
PROTOCOL_SCHEMA = "2wiki-ordinary200-full-ledger-protocol-v2"
STATUS = "FROZEN_ANSWER_FREE_OUTCOME_SOURCE_BOUND_NOT_TRAINED"
EXPERIMENT_ID = "2WIKI-ORDINARY200-FULL-LEDGER-V2-SEED42-PREREGISTRATION"
TARGET_N = 200
DEFAULT_PARENT_MATERIALIZED = Path(
    "data/silver_data/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42/"
    "silver_train.jsonl"
)
DEFAULT_CURRICULUM = Path(
    "data/silver_data/proofkg_curriculum_mix_v1_n8000_seed42_20260829/"
    "silver_curriculum.jsonl"
)
DEFAULT_PROOF_IDENTITY_POOLS = (
    Path(
        "outputs/audits/automatic_proofkg_2wiki_train_k4_v1_n1500_seed42_"
        "preregistration/cohort.question_only.jsonl"
    ),
    Path(
        "outputs/audits/2wiki_proofkg_extension_combined_v1_n350_seed42_"
        "preregistration/cohort.question_only.jsonl"
    ),
    Path(
        "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
        "seed42_preregistration/cohort.question_only.jsonl"
    ),
)
DEFAULT_OUT = Path(
    "outputs/audits/2wiki_ordinary200_full_ledger_v2_seed42_preregistration"
)
OUTPUT_FIELDS = frozenset(
    {
        "schema_version",
        "dataset",
        "qid",
        "question",
        "question_sha256",
        "family_version",
        "family_sha256",
        "question_type",
        "route",
        "source_role",
        "process_reward_eligible",
        "gold_access",
        "evaluation_eligible",
        "source_origin",
        "source_line_number",
        "source_record_sha256",
        "source_passages_sha256",
    }
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_with_lines(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append((line_number, value))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _validate_training_source_row(row: Mapping[str, Any], *, label: str) -> None:
    # Presence is checked because PPO outcome reward and the prompt require
    # these fields.  Their values never influence identity selection/ranking.
    if row.get("accepted") is not True:
        raise ValueError(f"{label}: source row is not accepted")
    if not str(row.get("answer") or "").strip():
        raise ValueError(f"{label}: source row lacks outcome")
    passages = row.get("retrieved_passages")
    if not isinstance(passages, list) or not passages:
        raise ValueError(f"{label}: source row lacks passages")


def _provenance_identity(
    row: Mapping[str, Any],
    *,
    line_number: int,
    source_origin: str,
    source_role: str,
) -> dict[str, Any]:
    identity = v4._identity(
        row,
        dataset="2wikimultihopqa",
        route="2wiki_ordinary_outcome",
        eligible=False,
        question_type=str((row.get("metadata") or {}).get("question_type") or "unknown"),
        stratum="ordinary",
        source_role=source_role,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": identity["dataset"],
        "qid": identity["qid"],
        "question": identity["question"],
        "question_sha256": identity["question_sha256"],
        "family_version": FAMILY_VERSION,
        "family_sha256": identity["family_sha256"],
        "question_type": identity["question_type"],
        "route": identity["route"],
        "source_role": source_role,
        "process_reward_eligible": False,
        "gold_access": False,
        "evaluation_eligible": False,
        "source_origin": source_origin,
        "source_line_number": int(line_number),
        "source_record_sha256": _canonical_sha256(row),
        "source_passages_sha256": _canonical_sha256(row["retrieved_passages"]),
    }


def _source_index(
    rows: Sequence[tuple[int, Mapping[str, Any]]], *, label: str
) -> dict[str, tuple[int, dict[str, Any]]]:
    result: dict[str, tuple[int, dict[str, Any]]] = {}
    for line_number, raw in rows:
        if str(raw.get("dataset") or "").strip().lower() != "2wikimultihopqa":
            continue
        row = dict(raw)
        identity = v4._identity(row, dataset="2wikimultihopqa")
        qid = str(identity["qid"])
        if qid in result:
            raise ValueError(f"{label}: duplicate qid {qid}")
        result[qid] = (line_number, row)
    return result


def _current_identity_rows(
    rows: Iterable[Mapping[str, Any]], *, label: str
) -> list[dict[str, Any]]:
    """Recompute historical family hashes instead of trusting stale fields."""

    output = []
    for raw in rows:
        if str(raw.get("dataset") or "").strip().lower() != "2wikimultihopqa":
            continue
        output.append(
            v4._identity(
                {
                    "dataset": "2wikimultihopqa",
                    "qid": str(raw.get("qid") or raw.get("id") or ""),
                    "question": str(raw.get("question") or ""),
                    "question_sha256": str(raw.get("question_sha256") or ""),
                },
                dataset="2wikimultihopqa",
                route=f"blocked_{label}",
                source_role=f"blocked_{label}",
            )
        )
    return output


def select_ordinary200(
    *,
    parent_rows: Sequence[Mapping[str, Any]],
    parent_source_rows: Sequence[tuple[int, Mapping[str, Any]]],
    replacement_source_rows: Sequence[tuple[int, Mapping[str, Any]]],
    protected_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    proof_identity_rows: Sequence[Mapping[str, Any]],
    n: int = TARGET_N,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parent_source = _source_index(parent_source_rows, label="parent materialized")
    replacement_source = _source_index(
        replacement_source_rows, label="replacement curriculum"
    )
    protected = v4.IdentityIndex()
    current_protected = _current_identity_rows(protected_rows, label="protected")
    protected.update(current_protected)
    replay = v4.IdentityIndex()
    current_replay = _current_identity_rows(replay_rows, label="replay")
    replay.update(current_replay)
    proof = v4.IdentityIndex()
    current_proof = _current_identity_rows(proof_identity_rows, label="proof_candidate")
    proof.update(current_proof)
    all_blocked = v4.IdentityIndex(
        qids=set(protected.qids | replay.qids | proof.qids),
        question_hashes=set(
            protected.question_hashes | replay.question_hashes | proof.question_hashes
        ),
        families=set(protected.families | replay.families | proof.families),
    )

    retained: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    removal_reasons: Counter[str] = Counter()
    for raw in parent_rows:
        identity = v4._identity(raw, dataset="2wikimultihopqa")
        source_entry = parent_source.get(str(identity["qid"]))
        if source_entry is None:
            raise ValueError(f"parent ordinary source join miss: {identity['qid']}")
        line_number, source_row = source_entry
        source_identity = v4._identity(source_row, dataset="2wikimultihopqa")
        if source_identity["question_sha256"] != identity["question_sha256"]:
            raise ValueError(f"parent ordinary source hash drift: {identity['qid']}")
        reasons = all_blocked.overlaps(identity)
        if reasons:
            removed.append(identity)
            removal_reasons.update(reasons)
            continue
        _validate_training_source_row(source_row, label=f"parent::{identity['qid']}")
        retained.append(
            _provenance_identity(
                source_row,
                line_number=line_number,
                source_origin="mixed_ppo_v2_materialized",
                source_role="retained_parent_ordinary",
            )
        )

    selected_index = v4.IdentityIndex()
    selected_index.update(retained)
    replacement_candidates: list[dict[str, Any]] = []
    replacement_exclusions: Counter[str] = Counter()
    for line_number, raw in replacement_source_rows:
        if str(raw.get("dataset") or "").strip().lower() != "2wikimultihopqa":
            continue
        if (
            (raw.get("metadata") or {}).get("train_only") is not True
            or str((raw.get("metadata") or {}).get("source_split") or "")
            not in {"train", "2wikimultihopqa/train"}
        ):
            replacement_exclusions["not_explicit_train_only"] += 1
            continue
        identity = v4._identity(raw, dataset="2wikimultihopqa")
        reasons = all_blocked.overlaps(identity)
        if reasons:
            replacement_exclusions[f"blocked_{reasons[0]}"] += 1
            continue
        if selected_index.overlaps(identity):
            replacement_exclusions["retained_identity_or_family"] += 1
            continue
        _validate_training_source_row(raw, label=f"replacement::{identity['qid']}")
        replacement_candidates.append(
            _provenance_identity(
                raw,
                line_number=line_number,
                source_origin="proofkg_curriculum_mix_v1",
                source_role="replacement_ordinary",
            )
        )

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in replacement_candidates:
        by_family[str(row["family_sha256"])].append(row)
    family_order = sorted(
        by_family,
        key=lambda family: (
            v4.rank("v4-ordinary200-replacement-family", "2wikimultihopqa", family),
            family,
        ),
    )
    replacements = []
    for family in family_order:
        rows = sorted(
            by_family[family],
            key=lambda row: (
                v4.rank(
                    "v4-ordinary200-replacement-within-family",
                    "2wikimultihopqa",
                    str(row["qid"]),
                ),
                str(row["qid"]),
            ),
        )
        replacements.append(rows[0])
        if len(retained) + len(replacements) == n:
            break
    if len(retained) + len(replacements) != n:
        raise ValueError(
            f"ordinary200 refill has {len(retained)} retained and "
            f"{len(replacements)} replacements; need {n}"
        )
    selected = sorted(
        [*retained, *replacements],
        key=lambda row: (
            v4.rank("v4-ordinary200-global", "2wikimultihopqa", str(row["qid"])),
            str(row["qid"]),
        ),
    )
    selected_index = v4.IdentityIndex()
    selected_index.update(selected)
    gates = {
        "ordinary_n_exact": len(selected) == n,
        "qid_unique": len(selected_index.qids) == n,
        "question_hash_unique": len(selected_index.question_hashes) == n,
        "current_family_unique": len(selected_index.families) == n,
        "protected_overlap_zero": not any(
            v4.identity_overlap_counts(selected, current_protected).values()
        ),
        "replay_overlap_zero": not any(
            v4.identity_overlap_counts(selected, current_replay).values()
        ),
        "all_proof_candidate_pools_overlap_zero": not any(
            v4.identity_overlap_counts(selected, current_proof).values()
        ),
        "output_fields_exact": all(set(row) == OUTPUT_FIELDS for row in selected),
        "all_outcome_only": all(
            row["process_reward_eligible"] is False for row in selected
        ),
        "all_gold_access_false": all(row["gold_access"] is False for row in selected),
    }
    if not all(gates.values()):
        raise RuntimeError(f"ordinary200 gates failed: {gates}")
    telemetry = {
        "parent_rows": len(parent_rows),
        "retained_parent": len(retained),
        "removed_parent": len(removed),
        "removed_parent_overlap_reasons": dict(sorted(removal_reasons.items())),
        "replacement_candidates": len(replacement_candidates),
        "replacement_candidate_current_families": len(by_family),
        "replacement_exclusion_counts_first_match": dict(
            sorted(replacement_exclusions.items())
        ),
        "replacements_selected": len(replacements),
        "selected_by_source_origin": dict(
            sorted(Counter(row["source_origin"] for row in selected).items())
        ),
        "selected_by_question_type": dict(
            sorted(Counter(row["question_type"] for row in selected).items())
        ),
        "gates": gates,
    }
    return selected, telemetry


def freeze_ordinary200(
    *,
    parent_protocol_path: Path,
    parent_materialized_path: Path,
    curriculum_path: Path,
    protected_ledger_dir: Path,
    replay_path: Path,
    proof_identity_paths: Sequence[Path],
    output_dir: Path,
    experiment_id: str = EXPERIMENT_ID,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite ordinary200 freeze: {output_dir}")
    required = (
        parent_protocol_path,
        parent_materialized_path,
        curriculum_path,
        replay_path,
        *proof_identity_paths,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    protected_path, ledger_binding = v4.validate_protected_ledger_release(
        protected_ledger_dir
    )
    parent_protocol = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    parent_rows = v4._load_protocol_output(parent_protocol, "ordinary200")
    proof_rows = [
        row for path in proof_identity_paths for row in v4.read_jsonl(path)
    ]
    selected, telemetry = select_ordinary200(
        parent_rows=parent_rows,
        parent_source_rows=_read_with_lines(parent_materialized_path),
        replacement_source_rows=_read_with_lines(curriculum_path),
        protected_rows=v4.read_jsonl(protected_path),
        replay_rows=v4.read_jsonl(replay_path),
        proof_identity_rows=proof_rows,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    cohort_path = output_dir / "ordinary200.identity_provenance.jsonl"
    _write_jsonl(cohort_path, selected)
    inputs = {
        "parent_protocol": v4.ref(parent_protocol_path),
        "parent_materialized_outcome_passages": v4.ref(parent_materialized_path),
        "replacement_curriculum_outcome_passages": v4.ref(curriculum_path),
        "complete_protected_ledger_release": ledger_binding,
        "replay": v4.ref(replay_path),
        "all_proof_candidate_identity_pools": [
            v4.ref(path) for path in proof_identity_paths
        ],
    }
    report = {
        "schema_version": PROTOCOL_SCHEMA,
        "experiment_id": str(experiment_id).strip(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "selection": {
            "policy": (
                "retain safe parent ordinary rows; replace blocked rows using "
                "deterministic one-per-current-family train-only candidates"
            ),
            "target_n": TARGET_N,
            "family_version": FAMILY_VERSION,
            "counts": telemetry,
        },
        "isolation": {
            "complete_protected_ledger": ["qid", "question_sha256", "family_sha256"],
            "replay": ["qid", "question_sha256", "family_sha256"],
            "all_proof_candidate_pools": [
                "qid",
                "question_sha256",
                "family_sha256",
            ],
        },
        "source_contract": {
            "outcome_and_passages_already_materialized": True,
            "source_file_line_and_record_hash_bound_per_row": True,
            "passages_hash_bound_per_row": True,
            "source_objects_contain_gold_outcomes": True,
            "full_source_json_objects_decoded": True,
            "outcome_value_used_for_selection_or_ranking": False,
            "outcome_and_passage_presence_validated": True,
            "answer_or_passage_text_emitted": False,
        },
        "execution_boundary": {
            "retrieval_started": False,
            "planner_started": False,
            "training_started": False,
            "old_artifacts_modified_or_deleted": False,
        },
        "inputs": inputs,
        "outputs": {"ordinary200": v4.ref(cohort_path)},
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        output_dir,
        status=STATUS,
        extra={
            "phase": "freeze_2wiki_ordinary200_full_ledger_v2",
            "experiment_id": str(experiment_id).strip(),
            "protocol_sha256": v4.sha256_file(protocol_path),
            "ordinary200": v4.ref(cohort_path),
            "retrieval_started": False,
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, default=v4.DEFAULT_PARENT_PROTOCOL)
    parser.add_argument(
        "--parent-materialized", type=Path, default=DEFAULT_PARENT_MATERIALIZED
    )
    parser.add_argument("--curriculum", type=Path, default=DEFAULT_CURRICULUM)
    parser.add_argument(
        "--protected-ledger-dir", type=Path, default=v4.DEFAULT_PROTECTED_LEDGER_DIR
    )
    parser.add_argument("--replay", type=Path, default=v4.DEFAULT_REPLAY)
    parser.add_argument("--proof-identity", action="append", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    args = parser.parse_args()
    report = freeze_ordinary200(
        parent_protocol_path=args.parent_protocol,
        parent_materialized_path=args.parent_materialized,
        curriculum_path=args.curriculum,
        protected_ledger_dir=args.protected_ledger_dir,
        replay_path=args.replay,
        proof_identity_paths=tuple(args.proof_identity or DEFAULT_PROOF_IDENTITY_POOLS),
        output_dir=args.out,
        experiment_id=args.experiment_id,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "counts": report["selection"]["counts"],
                "output": str(args.out.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
