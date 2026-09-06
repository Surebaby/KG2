#!/usr/bin/env python3
"""Freeze the complete identity ledger that mixed-PPO v4 must not train on.

This release is deliberately identity-only.  Historical artifacts were made
with more than one lexical-family implementation, so their stored family hash
is treated as provenance rather than authority.  The current family is always
recomputed from the question text before the identities are merged.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.kg.question_kg import question_sha256
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256


SCHEMA_VERSION = "mixed-ppo-v4-protected-identity-ledger-v2"
ROW_SCHEMA_VERSION = "mixed-ppo-v4-protected-question-identity-v2"
STATUS = "COMPLETE_FROZEN_IDENTITY_ONLY_NOT_TRAINING_DATA"
EXPERIMENT_ID = "MIXED-PPO-V4-PROTECTED-IDENTITY-LEDGER-V2"
DEFAULT_OUT = Path("outputs/audits/mixed_ppo_v4_protected_identity_ledger_v2")

SENSITIVE_SOURCE_FIELDS = {
    "answer",
    "answers",
    "gold_answer",
    "gold_answers",
    "golden_answers",
    "em",
    "f1",
    "supporting_facts",
    "evidence",
    "evidences",
    "decomposition",
    "question_decomposition",
}


# A path occurs once here even when several downstream artifacts reproduce the
# same cohort.  ``role`` records why the source must remain outside PPO data.
SOURCE_SPECS: tuple[tuple[str, str, str | None], ...] = (
    # QPEG/SAEG canonical evaluation and method-development identities.
    ("outputs/audits/qpeg_v1_n1350_seed42_preregistration/pilot.question_only.jsonl", "qpeg_development", None),
    ("outputs/audits/qpeg_v1_n1350_seed42_preregistration/confirmation.question_only.jsonl", "qpeg_confirmation", None),
    ("outputs/audits/qpeg_v1_n1350_seed42_preregistration/final.question_only.jsonl", "qpeg_canonical_reporting", None),
    ("outputs/audits/saeg_v1_evaluation_protocol_v1/development.question_only.jsonl", "saeg_development", None),
    ("outputs/audits/saeg_v1_evaluation_protocol_v1/confirmation.question_only.jsonl", "saeg_confirmation", None),
    ("outputs/audits/saeg_v1_evaluation_protocol_v1/canonical_reporting.question_only.jsonl", "saeg_canonical_reporting", None),
    # Subquestion/dependent-retrieval development and sealed prospective sets.
    ("outputs/audits/subquestion_decomposition_v8_cohort_freeze_dev30_prospective300_seed20260904_v1/development.identity_only.jsonl", "subquestion_development", None),
    ("outputs/audits/subquestion_decomposition_v8_cohort_freeze_dev30_prospective300_seed20260904_v1/prospective.identity_only.jsonl", "subquestion_prospective", None),
    ("outputs/audits/subquestion_dependent_retrieval_v7_development_preregistration/development.question_only.jsonl", "subquestion_consumed_development", None),
    ("outputs/audits/subquestion_decomposition_v8_consumed_smoke4x3_seed20260904_v1/smoke.identity_only.jsonl", "subquestion_consumed_smoke", None),
    ("outputs/audits/subquestion_decomposition_v9_canonical_subqa_pilot30x3_seed20260904_v1/pilot.identity_only.jsonl", "subquestion_consumed_development", None),
    # Every qid that trained, tuned, confirmed, or was reserved for the L0
    # verifier/reward-rankability line.
    ("outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_train.question_only.jsonl", "reward_verifier_train_consumed", None),
    ("outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_dev.question_only.jsonl", "reward_verifier_dev_consumed", None),
    ("outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_confirmation.question_only.jsonl", "reward_verifier_confirmation_consumed", None),
    ("outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_reserve.question_only.jsonl", "reward_verifier_future_reserve", None),
    ("outputs/audits/2wiki_train_only_rankability_n150_v1/cohort.question_only.jsonl", "reward_rankability_development_and_reserve", None),
    ("outputs/audits/2wiki_train_only_rankability_confirmation_n100_v1/cohort.question_only.jsonl", "reward_rankability_confirmation", None),
    ("outputs/audits/proofkg_dynamic_validity_confirmation_n100_seed20260902_preregistration/cohort.question_only.jsonl", "reward_dynamic_validity_confirmation", None),
    ("outputs/audits/ppo_reward_rankability_sft_hybrid_train_n100_k4_seed20260828_v1/cohort.jsonl", "historical_reward_rankability_development", "hotpotqa"),
    ("outputs/audits/proofkg_process_rankability_historical_v2_n100_k4_seed20260831_v1/candidates.jsonl", "historical_reward_rankability_development", "2wikimultihopqa"),
    ("outputs/audits/proof400_fill275_strong_sft_headroom_n100_k4_seed42_v3_preregistration/cohort.question_only.jsonl", "ppo_headroom_development_consumed", None),
    # Supply/planner confirmation sets used to change the ProofKG mechanism.
    ("outputs/audits/2wiki_confirmation270_v3/planner_inputs.confirmation.jsonl", "proofkg_supply_confirmation", None),
    ("outputs/audits/automatic_proofkg_2wiki_unseen_n100_seed20260830_preregistration/cohort.question_only.jsonl", "proofkg_supply_confirmation", None),
    ("outputs/audits/automatic_proofkg_2wiki_v2_independent_n100_seed20260830_preregistration/cohort.question_only.jsonl", "proofkg_supply_confirmation", None),
    ("outputs/audits/automatic_proofkg_2wiki_v3_independent_n100_seed20260831_preregistration/cohort.question_only.jsonl", "proofkg_supply_confirmation", None),
    ("outputs/audits/automatic_proofkg_unseen_n100_seed20260830_preregistration/cohort.question_only.jsonl", "proofkg_supply_confirmation", None),
    ("outputs/audits/versioned_2wiki_store_v1_independent_n100_seed20260901_preregistration/cohort.question_only.jsonl", "proofkg_store_confirmation", None),
    ("outputs/audits/query_aware_kg_relation_coverage_train150_seed20260828_v1/cohort.jsonl", "proofkg_consumed_development", None),
    ("outputs/audits/query_aware_proof_kg_2wiki_train150_seed20260829_confirmation_v1/cohort.jsonl", "proofkg_supply_confirmation", None),
    ("outputs/audits/query_planner_v2_a0_anchor_cohort_n50_seed20260829/cohort.jsonl", "proofkg_planner_development", "2wikimultihopqa"),
    ("outputs/audits/query_planner_v2_a1_single_review_n30_seed20260829/cohort.jsonl", "proofkg_planner_development", "2wikimultihopqa"),
    # Later controller work is paused, but its dev/confirmation identities have
    # already been consumed and therefore remain protected from this PPO line.
    ("outputs/audits/query_controller_v1_exact_text_pilot_seed42_protocol_v4_4/dev.identity_only.jsonl", "controller_development_consumed", None),
    ("outputs/audits/query_controller_v1_exact_text_pilot_seed42_protocol_v4_4/confirmation.identity_only.jsonl", "controller_confirmation", None),
    ("outputs/audits/query_controller_hotpot_silver_label_coverage_pilot30_seed20260904_v1/pilot.identity_only.jsonl", "controller_development_consumed", None),
)


CURRENT_V4_DEFAULT_PATHS = tuple(path for path, _, _ in SOURCE_SPECS[:8])


TARGETS: tuple[tuple[str, str], ...] = (
    ("mixed_v2_population", "outputs/audits/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_protocol/population.question_only.jsonl"),
    ("mixed_v2_ordinary200", "outputs/audits/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_protocol/ordinary200.question_only.jsonl"),
    ("old_auto1500_cohort", "outputs/audits/automatic_proofkg_2wiki_train_k4_v1_n1500_seed42_preregistration/cohort.question_only.jsonl"),
    ("old_auto1299_strict", "data/silver_data/automatic_proofkg_2wiki_train_k4_v1/silver_train.jsonl"),
    ("extension350", "outputs/audits/2wiki_proofkg_extension_combined_v1_n350_seed42_preregistration/cohort.question_only.jsonl"),
    ("hm_v4_interim_population", "outputs/audits/mixed_ppo_three_dataset_v4_hm_expansion_h1000_m1000_seed42_preregistration/hm_population.question_only.jsonl"),
    ("sft_replay2000", "data/silver_data/sft_replay_legacy_strong_train_rendered3to5_n2000_seed42_v1c/silver_train.jsonl"),
)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(row: Mapping[str, Any], *, dataset_override: str | None = None) -> dict[str, str]:
    dataset = str(dataset_override or row.get("dataset") or row.get("dataset_name") or "").strip().lower()
    qid = str(row.get("qid") or row.get("id") or row.get("source_id") or "").strip()
    question = str(row.get("question") or "").strip()
    if dataset not in {"hotpotqa", "2wikimultihopqa", "musique"} or not qid or not question:
        raise ValueError(f"incomplete protected identity: dataset={dataset!r}, qid={qid!r}")
    qhash = question_sha256(question)
    supplied_qhash = str(row.get("question_sha256") or "")
    if supplied_qhash and supplied_qhash != qhash:
        raise ValueError(f"protected question hash mismatch: {dataset}::{qid}")
    return {
        "dataset": dataset,
        "qid": qid,
        "question": question,
        "question_sha256": qhash,
        "family_sha256": family_sha256(question),
        "supplied_family_sha256": str(row.get("family_sha256") or ""),
    }


def _overlap(rows: Iterable[Mapping[str, Any]], protected: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    blocked = list(protected)
    qids = {(str(row["dataset"]), str(row["qid"])) for row in blocked}
    hashes = {(str(row["dataset"]), str(row["question_sha256"])) for row in blocked}
    families = {(str(row["dataset"]), str(row["family_sha256"])) for row in blocked}
    counts: Counter[str] = Counter()
    for row in rows:
        key = (str(row["dataset"]), str(row["qid"]))
        hkey = (str(row["dataset"]), str(row["question_sha256"]))
        fkey = (str(row["dataset"]), str(row["family_sha256"]))
        hits = (key in qids, hkey in hashes, fkey in families)
        counts["qid"] += hits[0]
        counts["question_sha256"] += hits[1]
        counts["family_sha256"] += hits[2]
        counts["any"] += any(hits)
    return {name: counts[name] for name in ("qid", "question_sha256", "family_sha256", "any")}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite versioned ledger: {args.out}")

    specs = [(Path(path), role, override) for path, role, override in SOURCE_SPECS]
    missing = [str(path) for path, _, _ in specs if not path.is_file()]
    missing += [path for _, path in TARGETS if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(missing)

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    source_inventory: list[dict[str, Any]] = []
    stale_family_rows = 0
    total_source_rows = 0
    sensitive_source_rows = 0
    sensitive_source_keys: Counter[str] = Counter()
    for path, role, override in specs:
        source_seen: set[tuple[str, str]] = set()
        source_stale = 0
        for raw in _read_jsonl(path):
            present_sensitive = SENSITIVE_SOURCE_FIELDS.intersection(raw)
            if present_sensitive:
                sensitive_source_rows += 1
                sensitive_source_keys.update(present_sensitive)
            item = _identity(raw, dataset_override=override)
            total_source_rows += 1
            supplied_family = item.pop("supplied_family_sha256")
            if supplied_family and supplied_family != item["family_sha256"]:
                source_stale += 1
                stale_family_rows += 1
            key = (item["dataset"], item["qid"])
            source_seen.add(key)
            prior = merged.get(key)
            if prior is None:
                merged[key] = {
                    **item,
                    "source_roles": {role},
                    "source_paths": {str(path)},
                }
            else:
                if prior["question_sha256"] != item["question_sha256"]:
                    raise ValueError(f"conflicting question for protected identity {key}")
                prior["source_roles"].add(role)
                prior["source_paths"].add(str(path))
        source_inventory.append({
            "path": str(path),
            "role": role,
            "dataset_override": override,
            "rows": sum(1 for _ in _read_jsonl(path)),
            "unique_dataset_qids": len(source_seen),
            "stored_family_hash_mismatch_rows": source_stale,
            "sha256": _sha256(path),
        })

    output_rows = []
    for key in sorted(merged):
        row = merged[key]
        output_rows.append({
            "schema_version": ROW_SCHEMA_VERSION,
            "dataset": row["dataset"],
            "qid": row["qid"],
            "question": row["question"],
            "question_sha256": row["question_sha256"],
            "family_version": FAMILY_VERSION,
            "family_sha256": row["family_sha256"],
            "source_roles": sorted(row["source_roles"]),
            "source_paths": sorted(row["source_paths"]),
            "gold_access": False,
        })

    current_rows = []
    for path in map(Path, CURRENT_V4_DEFAULT_PATHS):
        current_rows.extend(_identity(row) for row in _read_jsonl(path))

    impact: dict[str, Any] = {}
    for name, raw_path in TARGETS:
        normalised = [_identity(row) for row in _read_jsonl(Path(raw_path))]
        impact[name] = {
            "rows": len(normalised),
            "overlap_current_v4_default": _overlap(normalised, current_rows),
            "overlap_complete_ledger": _overlap(normalised, output_rows),
        }

    per_dataset = {}
    for dataset in sorted({row["dataset"] for row in output_rows}):
        selected = [row for row in output_rows if row["dataset"] == dataset]
        per_dataset[dataset] = {
            "qids": len({row["qid"] for row in selected}),
            "question_sha256": len({row["question_sha256"] for row in selected}),
            "current_families": len({row["family_sha256"] for row in selected}),
        }

    args.out.mkdir(parents=True, exist_ok=False)
    ledger_path = args.out / "protected_identities.question_only.jsonl"
    with ledger_path.open("x", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "complete": True,
        "identity_scope": "dataset-scoped",
        "current_family_recomputed": True,
        "stored_family_hash_is_non_authoritative_provenance": True,
        "source_files": len(specs),
        "source_rows": total_source_rows,
        "stored_family_hash_mismatch_rows": stale_family_rows,
        "source_rows_containing_sensitive_keys": sensitive_source_rows,
        "sensitive_source_key_counts": dict(sorted(sensitive_source_keys.items())),
        "unique": {
            "dataset_qids": len(output_rows),
            "dataset_question_sha256": len({(row["dataset"], row["question_sha256"]) for row in output_rows}),
            "dataset_current_families": len({(row["dataset"], row["family_sha256"]) for row in output_rows}),
            "by_dataset": per_dataset,
        },
        "source_inventory": source_inventory,
        "comparison": {
            "previous_v4_default_source_files": len(CURRENT_V4_DEFAULT_PATHS),
            "previous_v4_default_unique_dataset_qids": len({(row["dataset"], row["qid"]) for row in current_rows}),
            "newly_protected_dataset_qids": len({(row["dataset"], row["qid"]) for row in output_rows} - {(row["dataset"], row["qid"]) for row in current_rows}),
        },
        "target_overlap_impact": impact,
        "output": {
            "path": str(ledger_path),
            "rows": len(output_rows),
            "sha256": _sha256(ledger_path),
        },
        "scientific_boundary": {
            "source_json_objects_may_contain_gold_or_outcome_fields": True,
            "full_source_json_objects_decoded": True,
            "gold_or_outcome_values_used_for_identity_selection": False,
            "identity_fields_used": ["dataset", "qid", "question", "question_sha256"],
            "gold_fields_emitted": False,
            "data_raw_modified": False,
            "training_started": False,
            "failed_and_superseded_experiments_remain_protected": True,
        },
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(
        args.out,
        status=STATUS,
        extra={
            "phase": "mixed_ppo_v4_protected_identity_ledger_freeze",
            "experiment_id": EXPERIMENT_ID,
            "report_sha256": _sha256(report_path),
            "protected_identities_sha256": _sha256(ledger_path),
        },
    )
    print(json.dumps({
        "status": STATUS,
        "protected_identities": len(output_rows),
        "by_dataset": per_dataset,
        "target_overlap_impact": impact,
        "output": str(ledger_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
