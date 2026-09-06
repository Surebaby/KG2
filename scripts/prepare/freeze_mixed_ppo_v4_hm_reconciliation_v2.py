#!/usr/bin/env python3
"""Freeze the full-ledger H/M replacement and retrieval-reuse plan.

The original H/M expansion was retrieved before the complete protected ledger
was assembled.  Eleven MuSiQue population identities now overlap that ledger.
This append-only successor deterministically replaces those identities, binds
the 812 still-valid completed retrieval contexts, and emits only the 11 truly
new retrieval requests.  It does not execute retrieval or training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.utils.logging import dump_manifest
from scripts.prepare import freeze_mixed_ppo_three_dataset_v4_proof800 as v4


SCHEMA_VERSION = "mixed-ppo-v4-hm-full-ledger-reconciliation-v2"
STATUS = "FROZEN_HM_FULL_LEDGER_DELTA_RETRIEVAL_NOT_RUN_NOT_TRAINED"
EXPERIMENT_ID = "MIXED-PPO-V4-HM-FULL-LEDGER-RECONCILIATION-V2-SEED42"
DEFAULT_OLD_HM_PROTOCOL = Path(
    "outputs/audits/mixed_ppo_three_dataset_v4_hm_expansion_h1000_m1000_"
    "seed42_preregistration/protocol.json"
)
DEFAULT_EXISTING_RETRIEVAL_REPORT = Path(
    "outputs/audits/mixed3_v4_expansion_retrieval_h417_m401_seed42_v1/"
    "report.json"
)
DEFAULT_EXISTING_CONTEXTS = Path(
    "outputs/audits/mixed3_v4_expansion_retrieval_h417_m401_seed42_v1/"
    "retrieval_contexts.jsonl"
)
DEFAULT_OUT = Path(
    "outputs/audits/mixed_ppo_v4_hm_full_ledger_reconciliation_v2_seed42_"
    "preregistration"
)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("dataset") or "").strip().lower(), str(
        row.get("qid") or ""
    ).strip()


def _index_unique(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        key = _key(row)
        if key[0] not in {"hotpotqa", "musique"} or not key[1] or key in result:
            raise ValueError(f"{label}: invalid/duplicate identity {key}")
        result[key] = row
    return result


def reconcile_retrieval_contexts(
    required_population_rows: Sequence[Mapping[str, Any]],
    existing_context_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split requirements into reusable, new, and retired identities."""

    requirements = [v4._retrieval_request(row) for row in required_population_rows]
    required = _index_unique(requirements, label="retrieval requirements")
    contexts = _index_unique(existing_context_rows, label="existing contexts")
    reused: list[dict[str, Any]] = []
    new: list[dict[str, Any]] = []
    retired: list[dict[str, Any]] = []
    for key, request in required.items():
        context = contexts.get(key)
        if context is None:
            new.append(request)
            continue
        for field in ("question", "question_sha256", "family_sha256"):
            if str(context.get(field) or "") != str(request.get(field) or ""):
                raise ValueError(f"existing context identity drift for {key}: {field}")
        passages_sha = str(context.get("passages_sha256") or "")
        if len(passages_sha) != 64:
            raise ValueError(f"existing context lacks passages_sha256 for {key}")
        reused.append(
            {
                "schema_version": "mixed-ppo-v4-reused-context-binding-v2",
                "dataset": request["dataset"],
                "qid": request["qid"],
                "question": request["question"],
                "question_sha256": request["question_sha256"],
                "family_version": request["family_version"],
                "family_sha256": request["family_sha256"],
                "passages_sha256": passages_sha,
                "reuse_existing_context": True,
                "gold_access": False,
            }
        )
    for key, context in contexts.items():
        if key in required:
            continue
        retired.append(
            {
                "schema_version": "mixed-ppo-v4-retired-context-identity-v2",
                "dataset": key[0],
                "qid": key[1],
                "question": str(context.get("question") or ""),
                "question_sha256": str(context.get("question_sha256") or ""),
                "family_sha256": str(context.get("family_sha256") or ""),
                "reason": "removed_by_complete_protected_ledger",
                "gold_access": False,
            }
        )
    order = lambda row: (str(row["dataset"]), str(row["qid"]))
    return sorted(reused, key=order), sorted(new, key=order), sorted(retired, key=order)


def _validate_existing_retrieval_release(
    *, report_path: Path, contexts_path: Path
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "COMPLETE_ANSWER_FREE_RETRIEVAL_NOT_TRAINED":
        raise ValueError("existing retrieval release has unexpected status")
    if not all((report.get("gates") or {}).values()):
        raise ValueError("existing retrieval release gates are not all true")
    combined = (report.get("outputs") or {}).get("combined") or {}
    if v4.sha256_file(contexts_path) != str(combined.get("sha256") or ""):
        raise ValueError("existing retrieval contexts/report hash mismatch")
    return report


def freeze_hm_reconciliation(
    *,
    parent_protocol_path: Path,
    old_hm_protocol_path: Path,
    existing_retrieval_report_path: Path,
    existing_contexts_path: Path,
    hotpot_raw_path: Path,
    musique_raw_path: Path,
    replay_path: Path,
    protected_ledger_dir: Path,
    output_dir: Path,
    reserve_per_stratum: int = 25,
    experiment_id: str = EXPERIMENT_ID,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite reconciliation: {output_dir}")
    required_paths = (
        parent_protocol_path,
        old_hm_protocol_path,
        existing_retrieval_report_path,
        existing_contexts_path,
        hotpot_raw_path,
        musique_raw_path,
        replay_path,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    protected_path, ledger_binding = v4.validate_protected_ledger_release(
        protected_ledger_dir
    )

    parent_protocol = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    parent_population = v4._load_protocol_output(parent_protocol, "population")
    parent_hotpot = [row for row in parent_population if row["dataset"] == "hotpotqa"]
    parent_musique = [row for row in parent_population if row["dataset"] == "musique"]
    if len(parent_hotpot) != 600 or len(parent_musique) != 599:
        raise ValueError("parent H/M population is not the frozen 600/599")

    old_protocol = json.loads(old_hm_protocol_path.read_text(encoding="utf-8"))
    old_population = v4._load_protocol_output(old_protocol, "hm_population")
    old_requests = v4._load_protocol_output(old_protocol, "retrieval_requests")
    existing_retrieval_report = _validate_existing_retrieval_release(
        report_path=existing_retrieval_report_path,
        contexts_path=existing_contexts_path,
    )
    existing_contexts = v4.read_jsonl(existing_contexts_path)
    if set(_index_unique(old_requests, label="old requests")) != set(
        _index_unique(existing_contexts, label="existing contexts")
    ):
        raise ValueError("old request/context identity join is not exact")

    protected_rows = v4._normalise_external_identities(v4.read_jsonl(protected_path))
    replay_rows = v4._normalise_external_identities(v4.read_jsonl(replay_path))
    externally_blocked = v4.IdentityIndex()
    externally_blocked.update(protected_rows)
    externally_blocked.update(replay_rows)
    hotpot, hotpot_new, hotpot_reserve, hotpot_stats = v4.build_hotpot_population(
        parent_hotpot,
        v4.read_jsonl(hotpot_raw_path),
        externally_blocked=externally_blocked,
        reserve_per_stratum=reserve_per_stratum,
    )
    musique, musique_new, musique_reserve, musique_stats = v4.build_musique_population(
        parent_musique,
        v4.read_jsonl(musique_raw_path),
        externally_blocked=externally_blocked,
        reserve_per_stratum=reserve_per_stratum,
    )
    population = sorted(
        [*hotpot, *musique], key=lambda row: (str(row["dataset"]), str(row["qid"]))
    )
    retrieval_requirements = sorted(
        [*hotpot_new, *musique_new],
        key=lambda row: (str(row["dataset"]), str(row["qid"])),
    )
    reused, new_requests, retired_contexts = reconcile_retrieval_contexts(
        retrieval_requirements, existing_contexts
    )
    old_index = _index_unique(old_population, label="old H/M population")
    new_index = _index_unique(population, label="reconciled H/M population")
    removed_population = [old_index[key] for key in sorted(set(old_index) - set(new_index))]
    added_population = [new_index[key] for key in sorted(set(new_index) - set(old_index))]
    reserves = sorted(
        [*hotpot_reserve, *musique_reserve],
        key=lambda row: (str(row["dataset"]), str(row["stratum"]), str(row["qid"])),
    )

    population_index = v4.IdentityIndex()
    population_index.update(population)
    protected_overlap = v4.identity_overlap_counts(population, protected_rows)
    replay_overlap = v4.identity_overlap_counts(population, replay_rows)
    reserve_overlap = v4.identity_overlap_counts(reserves, population)
    gates = {
        "population_h1000_m1000": Counter(row["dataset"] for row in population)
        == Counter({"hotpotqa": 1000, "musique": 1000}),
        "population_qid_unique": len(population_index.qids) == 2000,
        "population_question_hash_unique": len(population_index.question_hashes)
        == 2000,
        "protected_overlap_zero": not any(protected_overlap.values()),
        "replay_overlap_zero": not any(replay_overlap.values()),
        "reserve_population_overlap_zero": not any(reserve_overlap.values()),
        "hotpot_historical_selection_unchanged": hotpot_stats["retained_parent"]
        == 583
        and hotpot_stats["new_retrieval"] == 417,
        "musique_five_parent_rows_replaced": musique_stats["retained_parent"]
        == 594
        and musique_stats["removed_parent"] == 5
        and musique_stats["new_retrieval"] == 406,
        "population_replaced_11_all_musique": len(removed_population) == 11
        and len(added_population) == 11
        and all(row["dataset"] == "musique" for row in removed_population + added_population),
        "retrieval_requirements_823": len(retrieval_requirements) == 823,
        "existing_contexts_reused_812": len(reused) == 812,
        "new_retrieval_requests_only_11": len(new_requests) == 11
        and all(row["dataset"] == "musique" for row in new_requests),
        "retired_contexts_6": len(retired_contexts) == 6
        and all(row["dataset"] == "musique" for row in retired_contexts),
        "reuse_plus_new_equals_requirements": len(reused) + len(new_requests)
        == len(retrieval_requirements),
        "no_retrieval_executed": True,
        "training_not_started": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"H/M reconciliation gates failed: {gates}")

    output_dir.mkdir(parents=True, exist_ok=False)
    output_paths = {
        "hm_population": output_dir / "hm_population.question_only.jsonl",
        "retrieval_requirements": output_dir
        / "retrieval_requirements.question_only.jsonl",
        "reused_context_bindings": output_dir
        / "reused_context_bindings.question_only.jsonl",
        "new_retrieval_requests": output_dir
        / "new_retrieval_requests.question_only.jsonl",
        "retired_contexts": output_dir / "retired_contexts.question_only.jsonl",
        "removed_population": output_dir
        / "removed_population.question_only.jsonl",
        "added_population": output_dir / "added_population.question_only.jsonl",
        "reserve": output_dir / "reserve.question_only.jsonl",
    }
    rows_by_name = {
        "hm_population": population,
        "retrieval_requirements": [v4._retrieval_request(row) for row in retrieval_requirements],
        "reused_context_bindings": reused,
        "new_retrieval_requests": new_requests,
        "retired_contexts": retired_contexts,
        "removed_population": removed_population,
        "added_population": added_population,
        "reserve": reserves,
    }
    for name, path in output_paths.items():
        _write_jsonl(path, rows_by_name[name])

    input_refs = {
        "parent_protocol": v4.ref(parent_protocol_path),
        "superseded_hm_protocol": v4.ref(old_hm_protocol_path),
        "completed_retrieval_report": v4.ref(existing_retrieval_report_path),
        "completed_retrieval_contexts": v4.ref(existing_contexts_path),
        "hotpot_raw": v4.ref(hotpot_raw_path),
        "musique_raw": v4.ref(musique_raw_path),
        "replay": v4.ref(replay_path),
        "protected_ledger_release": ledger_binding,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(experiment_id).strip(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "supersedes": str(old_hm_protocol_path.resolve()),
        "population": {
            "total": len(population),
            "by_dataset": dict(sorted(Counter(row["dataset"] for row in population).items())),
            "hotpot": hotpot_stats,
            "musique": musique_stats,
            "removed_from_old": len(removed_population),
            "added_to_old": len(added_population),
        },
        "retrieval_reconciliation": {
            "old_completed_contexts": len(existing_contexts),
            "required_for_reconciled_population": len(retrieval_requirements),
            "reused": len(reused),
            "new_requests": len(new_requests),
            "retired": len(retired_contexts),
            "retrieval_executed_in_this_stage": False,
        },
        "isolation": {
            "protected": protected_overlap,
            "replay": replay_overlap,
            "reserve_population": reserve_overlap,
        },
        "gates": gates,
        "scientific_boundary": {
            "answer_free_identity_reconciliation": True,
            "raw_source_objects_may_contain_gold": True,
            "gold_values_used_for_selection": False,
            "existing_passage_text_copied": False,
            "existing_passage_hashes_bound_for_reuse": True,
            "retrieval_started": False,
            "training_started": False,
            "old_artifacts_modified_or_deleted": False,
        },
        "inputs": input_refs,
        "existing_retrieval_attestation": {
            "status": existing_retrieval_report["status"],
            "gates": existing_retrieval_report["gates"],
        },
        "outputs": {name: v4.ref(path) for name, path in output_paths.items()},
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        output_dir,
        status=STATUS,
        extra={
            "phase": "mixed_ppo_v4_hm_full_ledger_reconciliation",
            "experiment_id": str(experiment_id).strip(),
            "protocol_sha256": v4.sha256_file(protocol_path),
            "new_retrieval_requests": v4.ref(output_paths["new_retrieval_requests"]),
            "retrieval_started": False,
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, default=v4.DEFAULT_PARENT_PROTOCOL)
    parser.add_argument("--old-hm-protocol", type=Path, default=DEFAULT_OLD_HM_PROTOCOL)
    parser.add_argument(
        "--existing-retrieval-report",
        type=Path,
        default=DEFAULT_EXISTING_RETRIEVAL_REPORT,
    )
    parser.add_argument(
        "--existing-contexts", type=Path, default=DEFAULT_EXISTING_CONTEXTS
    )
    parser.add_argument("--hotpot-raw", type=Path, default=Path("data/hotpotqa/train.jsonl"))
    parser.add_argument("--musique-raw", type=Path, default=Path("data/musique/train.jsonl"))
    parser.add_argument("--replay", type=Path, default=v4.DEFAULT_REPLAY)
    parser.add_argument(
        "--protected-ledger-dir", type=Path, default=v4.DEFAULT_PROTECTED_LEDGER_DIR
    )
    parser.add_argument("--reserve-per-stratum", type=int, default=25)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    args = parser.parse_args()
    report = freeze_hm_reconciliation(
        parent_protocol_path=args.parent_protocol,
        old_hm_protocol_path=args.old_hm_protocol,
        existing_retrieval_report_path=args.existing_retrieval_report,
        existing_contexts_path=args.existing_contexts,
        hotpot_raw_path=args.hotpot_raw,
        musique_raw_path=args.musique_raw,
        replay_path=args.replay,
        protected_ledger_dir=args.protected_ledger_dir,
        output_dir=args.out,
        reserve_per_stratum=args.reserve_per_stratum,
        experiment_id=args.experiment_id,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "population": report["population"],
                "retrieval_reconciliation": report["retrieval_reconciliation"],
                "gates": report["gates"],
                "output": str(args.out.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
