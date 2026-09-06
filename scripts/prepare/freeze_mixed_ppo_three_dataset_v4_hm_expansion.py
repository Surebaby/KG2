#!/usr/bin/env python3
"""Freeze the HotpotQA/MuSiQue expansion needed by mixed3-v4.

This is an answer-free, append-only preregistration stage.  It deliberately
does not require the still-unresolved 2Wiki Proof800 supply, so the 818 new
HotpotQA/MuSiQue identities can enter canonical retrieval while the 2Wiki
planner/materialisation work proceeds independently.

The selection implementation, quotas, ranking, and identity isolation are
imported from ``freeze_mixed_ppo_three_dataset_v4_proof800.py``.  Consequently
the H/M rows frozen here are exactly the H/M rows the intended final v4 freeze
will select from the same versioned inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_mixed_ppo_three_dataset_v4_proof800 import (
    DEFAULT_PARENT_PROTOCOL,
    DEFAULT_PROTECTED,
    DEFAULT_REPLAY,
    HOTPOT_TARGET_CELLS,
    MUSIQUE_TARGET_HOPS,
    IdentityIndex,
    _load_protocol_output,
    _normalise_external_identities,
    _retrieval_request,
    build_hotpot_population,
    build_musique_population,
    identity_overlap_counts,
    read_jsonl,
    ref,
    write_jsonl,
)
from scripts.prepare.freeze_mixed_ppo_three_dataset_v1 import sha256_file


SCHEMA_VERSION = "mixed-ppo-three-dataset-v4-hm-expansion-preregistration-v1"
STATUS = (
    "FROZEN_HM_IDENTITIES_RETRIEVAL_NOT_MATERIALIZED_"
    "2WIKI_UNRESOLVED_NOT_TRAINED"
)
EXPERIMENT_ID = (
    "MIXED-PPO-THREE-DATASET-V4-HM-EXPANSION-"
    "H1000-M1000-NEW818-SEED42-PREREGISTRATION"
)
EXPECTED_PARENT_SCHEMA = "mixed-ppo-three-dataset-protocol-v2-proof400"
EXPECTED_PARENT_STATUS = (
    "FROZEN_ANSWER_FREE_LEXICAL_FAMILY_V1_NOT_MATERIALIZED_NOT_TRAINED"
)
INTENDED_FINAL_SCHEMA = "mixed-ppo-three-dataset-protocol-v4-proof800"
DEFAULT_OUT = Path(
    "outputs/audits/"
    "mixed_ppo_three_dataset_v4_hm_expansion_h1000_m1000_seed42_preregistration"
)


def _assert_parent_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != EXPECTED_PARENT_SCHEMA:
        raise ValueError(
            "unexpected parent schema: "
            f"{protocol.get('schema_version')!r}"
        )
    if protocol.get("status") != EXPECTED_PARENT_STATUS:
        raise ValueError(
            "unexpected/unfrozen parent status: "
            f"{protocol.get('status')!r}"
        )


def freeze_hm_expansion(
    *,
    parent_protocol_path: Path,
    hotpot_raw_path: Path,
    musique_raw_path: Path,
    replay_path: Path,
    protected_paths: Sequence[Path],
    output_dir: Path,
    reserve_per_stratum: int = 25,
    experiment_id: str = EXPERIMENT_ID,
) -> dict[str, Any]:
    """Select and freeze the H1000/M1000 identities and new retrieval work."""

    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite H/M expansion preregistration: {output_dir}"
        )
    if reserve_per_stratum < 0:
        raise ValueError("reserve_per_stratum must be nonnegative")
    if not str(experiment_id).strip():
        raise ValueError("a nonempty Experiment ID is required")
    required = [
        parent_protocol_path,
        hotpot_raw_path,
        musique_raw_path,
        replay_path,
        *protected_paths,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    parent_protocol = json.loads(
        parent_protocol_path.read_text(encoding="utf-8")
    )
    _assert_parent_protocol(parent_protocol)
    parent_population = _load_protocol_output(parent_protocol, "population")
    parent_hotpot = [
        row for row in parent_population if row.get("dataset") == "hotpotqa"
    ]
    parent_musique = [
        row for row in parent_population if row.get("dataset") == "musique"
    ]
    if len(parent_hotpot) != 600 or len(parent_musique) != 599:
        raise ValueError(
            "parent H/M counts are not the frozen 600/599: "
            f"H={len(parent_hotpot)}, M={len(parent_musique)}"
        )

    protected_rows = _normalise_external_identities(
        row for path in protected_paths for row in read_jsonl(path)
    )
    replay_rows = _normalise_external_identities(read_jsonl(replay_path))
    externally_blocked = IdentityIndex()
    externally_blocked.update(protected_rows)
    externally_blocked.update(replay_rows)

    hotpot, hotpot_new, hotpot_reserve, hotpot_stats = build_hotpot_population(
        parent_hotpot,
        read_jsonl(hotpot_raw_path),
        externally_blocked=externally_blocked,
        reserve_per_stratum=reserve_per_stratum,
    )
    musique, musique_new, musique_reserve, musique_stats = build_musique_population(
        parent_musique,
        read_jsonl(musique_raw_path),
        externally_blocked=externally_blocked,
        reserve_per_stratum=reserve_per_stratum,
    )

    population = [*hotpot, *musique]
    population.sort(
        key=lambda row: (
            0 if row["dataset"] == "hotpotqa" else 1,
            str(row["stratum"]),
            str(row["qid"]),
        )
    )
    new_rows = [*hotpot_new, *musique_new]
    retrieval_requests = [_retrieval_request(row) for row in new_rows]
    retrieval_requests.sort(
        key=lambda row: (
            0 if row["dataset"] == "hotpotqa" else 1,
            str(row["stratum"]),
            str(row["qid"]),
        )
    )
    reserve = [*hotpot_reserve, *musique_reserve]
    reserve.sort(
        key=lambda row: (
            0 if row["dataset"] == "hotpotqa" else 1,
            str(row["stratum"]),
            str(row["qid"]),
        )
    )

    population_index = IdentityIndex()
    population_index.update(population)
    protected_overlap = identity_overlap_counts(population, protected_rows)
    replay_overlap = identity_overlap_counts(population, replay_rows)
    reserve_overlap = identity_overlap_counts(reserve, population)
    request_counts = Counter(row["dataset"] for row in retrieval_requests)
    population_counts = Counter(row["dataset"] for row in population)
    gates = {
        "hm_population_2000": len(population) == 2000,
        "hotpotqa_1000_musique_1000": population_counts
        == Counter({"hotpotqa": 1000, "musique": 1000}),
        "dataset_scoped_qid_unique": len(population_index.qids) == 2000,
        "dataset_scoped_question_hash_unique": len(
            population_index.question_hashes
        )
        == 2000,
        "protected_qid_hash_family_overlap_zero": not any(
            protected_overlap.values()
        ),
        "replay_qid_hash_family_overlap_zero": not any(replay_overlap.values()),
        "reserve_population_qid_hash_family_overlap_zero": not any(
            reserve_overlap.values()
        ),
        "hotpot_retained_583_new_417": hotpot_stats["retained_parent"] == 583
        and hotpot_stats["new_retrieval"] == 417,
        "hotpot_target_cells_exact": Counter(
            row["stratum"] for row in hotpot
        )
        == Counter(HOTPOT_TARGET_CELLS),
        "musique_retained_599_new_401": musique_stats["retained_parent"] == 599
        and musique_stats["new_retrieval"] == 401,
        "musique_target_hops_exact": Counter(
            row["stratum"] for row in musique
        )
        == Counter(MUSIQUE_TARGET_HOPS),
        "retrieval_requests_h417_m401": request_counts
        == Counter({"hotpotqa": 417, "musique": 401}),
        "request_identity_unique": len(
            {(row["dataset"], row["qid"]) for row in retrieval_requests}
        )
        == 818,
        "all_answer_free": all(
            row.get("gold_access") is False
            for row in [*population, *reserve, *retrieval_requests]
        ),
        "two_wiki_intentionally_unresolved": True,
        "retrieval_not_run": True,
        "training_not_started": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"H/M expansion preregistration gates failed: {gates}")

    output_dir.mkdir(parents=True, exist_ok=False)
    output_paths = {
        "hm_population": output_dir / "hm_population.question_only.jsonl",
        "hotpotqa_population": output_dir
        / "hotpotqa_population.question_only.jsonl",
        "musique_population": output_dir
        / "musique_population.question_only.jsonl",
        "retrieval_requests": output_dir
        / "retrieval_requests.question_only.jsonl",
        "hotpot_retrieval_requests": output_dir
        / "hotpotqa.retrieval_requests.question_only.jsonl",
        "musique_retrieval_requests": output_dir
        / "musique.retrieval_requests.question_only.jsonl",
        "reserve": output_dir / "reserve.question_only.jsonl",
    }
    rows_by_output = {
        "hm_population": population,
        "hotpotqa_population": hotpot,
        "musique_population": musique,
        "retrieval_requests": retrieval_requests,
        "hotpot_retrieval_requests": [
            row for row in retrieval_requests if row["dataset"] == "hotpotqa"
        ],
        "musique_retrieval_requests": [
            row for row in retrieval_requests if row["dataset"] == "musique"
        ],
        "reserve": reserve,
    }
    for name, path in output_paths.items():
        write_jsonl(path, rows_by_output[name])

    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(experiment_id).strip(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "scope": "hotpotqa_musique_expansion_only",
        "population": {
            "resolved_unique_total": len(population),
            "resolved_unique_by_dataset": dict(sorted(population_counts.items())),
            "retrieval_requests_by_dataset": dict(sorted(request_counts.items())),
            "hotpot": hotpot_stats,
            "musique": musique_stats,
        },
        "intended_final_v4": {
            "schema_version": INTENDED_FINAL_SCHEMA,
            "target_unique_total": 3000,
            "target_unique_by_dataset": {
                "hotpotqa": 1000,
                "2wikimultihopqa": 1000,
                "musique": 1000,
            },
            "rollouts_per_prompt": 4,
            "target_trajectories": 12000,
            "two_wiki": {
                "status": "UNRESOLVED_NOT_BOUND",
                "target_unique": 1000,
                "proofkg_process_reward_eligible_target": 800,
                "ordinary_outcome_target": 200,
            },
            "finalizer": (
                "scripts/prepare/"
                "freeze_mixed_ppo_three_dataset_v4_proof800.py"
            ),
            "finalization_ready": False,
            "training_started": False,
        },
        "isolation": {
            "scope": "dataset-scoped",
            "keys": ["qid", "question_sha256", "family_sha256"],
            "protected": protected_overlap,
            "replay": replay_overlap,
            "reserve_population": reserve_overlap,
        },
        "reserve": {
            "per_stratum": reserve_per_stratum,
            "total": len(reserve),
            "by_dataset": dict(
                sorted(Counter(row["dataset"] for row in reserve).items())
            ),
            "identity_only_not_retrieved_not_scheduled": True,
        },
        "gates": gates,
        "scientific_boundary": {
            "answer_free_freeze": True,
            "raw_source_files_may_contain_gold": True,
            "gold_fields_accessed_for_selection": False,
            "selection_metadata_accessed": {
                "hotpotqa": ["metadata.type", "metadata.level"],
                "musique": ["len(metadata.metadata.question_decomposition)"],
            },
            "supporting_sentence_content_accessed": False,
            "evaluation_outputs_read": False,
            "retrieval_run": False,
            "proofkg_materialized": False,
            "two_wiki_resolved": False,
            "schedule_frozen": False,
            "training_started": False,
            "old_assets_overwritten": False,
        },
        "inputs": {
            "parent_protocol": ref(parent_protocol_path),
            "hotpot_raw": ref(hotpot_raw_path),
            "musique_raw": ref(musique_raw_path),
            "replay": ref(replay_path),
            "protected": [ref(path) for path in protected_paths],
            "selection_implementation": ref(
                Path(
                    "scripts/prepare/"
                    "freeze_mixed_ppo_three_dataset_v4_proof800.py"
                )
            ),
        },
        "outputs": {name: ref(path) for name, path in output_paths.items()},
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    dump_manifest(
        output_dir,
        status=STATUS,
        extra={
            "phase": "mixed_ppo_v4_hm_expansion_answer_free_preregistration",
            "experiment_id": report["experiment_id"],
            "protocol_sha256": sha256_file(protocol_path),
            "retrieval_requests": report["outputs"]["retrieval_requests"],
            "two_wiki_status": "UNRESOLVED_NOT_BOUND",
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent_protocol", type=Path, default=DEFAULT_PARENT_PROTOCOL
    )
    parser.add_argument(
        "--hotpot_raw", type=Path, default=Path("data/hotpotqa/train.jsonl")
    )
    parser.add_argument(
        "--musique_raw", type=Path, default=Path("data/musique/train.jsonl")
    )
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--protected", action="append", type=Path, default=None)
    parser.add_argument("--reserve_per_stratum", type=int, default=25)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--experiment_id", default=EXPERIMENT_ID)
    args = parser.parse_args()
    report = freeze_hm_expansion(
        parent_protocol_path=args.parent_protocol,
        hotpot_raw_path=args.hotpot_raw,
        musique_raw_path=args.musique_raw,
        replay_path=args.replay,
        protected_paths=tuple(args.protected or DEFAULT_PROTECTED),
        output_dir=args.out,
        reserve_per_stratum=args.reserve_per_stratum,
        experiment_id=args.experiment_id,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "scope": report["scope"],
                "population": report["population"],
                "intended_final_v4": report["intended_final_v4"],
                "gates": report["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
