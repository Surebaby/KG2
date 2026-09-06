#!/usr/bin/env python
"""Freeze train-only QPEG schema-adaptation cohorts before data construction.

The evaluation cohorts are selected from raw dev after excluding every QID and
answer-free question family consumed by QPEG-v1/v2/v3.  Training rows come only
from raw train and are family-disjoint from both new evaluation cohorts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.kg.question_kg import question_sha256
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_qpeg_v1_protocol import (
    DATASETS,
    choose_disjoint_rows,
    family_sha256,
)


EXPERIMENT_ID = "QPEG-V4-TRAINONLY-SCHEMA-ADAPT-SEED42"
APPROVAL_ID = "USER-APPROVED-2026-09-03-QPEG-V4-SCHEMA-ADAPTATION"
TRAIN_PER_DATASET = 600
DEVELOPMENT_PER_DATASET = 50
CONFIRMATION_PER_DATASET = 100
NO_GRAPH_REPLAY_PER_DATASET = 200
FORBIDDEN_FIELDS = {
    "golden_answers", "answer", "answers", "supporting_facts", "support",
    "decomposition", "question_decomposition", "evidence", "reasoning", "sp",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _eligible_train(dataset: str, row: Mapping[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    if dataset in {"hotpotqa", "2wikimultihopqa"}:
        support = metadata.get("supporting_facts") or {}
        pairs = {
            (str(title).strip(), int(index))
            for title, index in zip(
                support.get("title") or [], support.get("sent_id") or []
            )
            if str(title).strip()
        }
        return len(pairs) >= 2
    decomposition = ((metadata.get("metadata") or {}).get("question_decomposition") or [])
    usable = [
        step for step in decomposition
        if str((step.get("support_paragraph") or {}).get("paragraph_text") or "").strip()
    ]
    return len(usable) >= 2


def _question_only(row: Mapping[str, Any], role: str) -> dict[str, Any]:
    dataset = str(row["dataset"])
    qid = str(row["qid"])
    question = str(row["question"])
    return {
        "schema_version": "qpeg-v4-question-only-v1",
        "question_key": f"{dataset}::{qid}",
        "dataset": dataset,
        "qid": qid,
        "question": question,
        "question_sha256": question_sha256(question),
        "family_sha256": family_sha256(question),
        "role": role,
        "gold_access": False,
    }


def _consumed_dev_identity(consumed_dir: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    qids: dict[str, set[str]] = defaultdict(set)
    families: dict[str, set[str]] = defaultdict(set)
    for name in ("pilot.question_only.jsonl", "confirmation.question_only.jsonl", "final.question_only.jsonl"):
        for row in _read_jsonl(consumed_dir / name):
            dataset = str(row["dataset"])
            qids[dataset].add(str(row["qid"]))
            families[dataset].add(str(row["family_sha256"]))
    return qids, families


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument(
        "--consumed",
        type=Path,
        default=Path("outputs/audits/qpeg_v1_n1350_seed42_preregistration"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/audits/qpeg_v4_schema_adaptation_protocol_v1"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite frozen protocol: {args.out}")
    args.out.mkdir(parents=True)

    consumed_qids, consumed_families = _consumed_dev_identity(args.consumed)
    train_rows_out: list[dict[str, Any]] = []
    development_out: list[dict[str, Any]] = []
    confirmation_out: list[dict[str, Any]] = []
    raw_hashes: dict[str, dict[str, str]] = {}
    dataset_report: dict[str, Any] = {}

    for dataset in DATASETS:
        dev_path = args.data_root / dataset / "dev.jsonl"
        train_path = args.data_root / dataset / "train.jsonl"
        dev_rows = _read_jsonl(dev_path)
        train_rows = _read_jsonl(train_path)
        raw_hashes[dataset] = {
            "train": _sha256(train_path),
            "dev": _sha256(dev_path),
        }

        development = choose_disjoint_rows(
            dev_rows,
            excluded_qids=set(consumed_qids[dataset]),
            excluded_families=set(consumed_families[dataset]),
            n=DEVELOPMENT_PER_DATASET,
            dataset=dataset,
            seed=84,
        )
        eval_qids = set(consumed_qids[dataset]) | {str(row["qid"]) for row in development}
        eval_families = set(consumed_families[dataset]) | {
            str(row["family_sha256"]) for row in development
        }
        confirmation = choose_disjoint_rows(
            dev_rows,
            excluded_qids=eval_qids,
            excluded_families=eval_families,
            n=CONFIRMATION_PER_DATASET,
            dataset=dataset,
            seed=85,
        )
        new_eval_families = {
            str(row["family_sha256"]) for row in development + confirmation
        }

        eligible_train = [row for row in train_rows if _eligible_train(dataset, row)]
        training = choose_disjoint_rows(
            eligible_train,
            excluded_qids=set(),
            excluded_families=new_eval_families,
            n=TRAIN_PER_DATASET,
            dataset=dataset,
            seed=86,
        )
        train_families = {str(row["family_sha256"]) for row in training}
        if train_families & new_eval_families:
            raise RuntimeError(f"{dataset}: train/evaluation family leakage")

        train_rows_out.extend(_question_only(row, "train") for row in training)
        development_out.extend(_question_only(row, "development") for row in development)
        confirmation_out.extend(_question_only(row, "confirmation") for row in confirmation)
        dataset_report[dataset] = {
            "raw_train": len(train_rows),
            "eligible_train": len(eligible_train),
            "raw_dev": len(dev_rows),
            "previously_consumed_dev_qids": len(consumed_qids[dataset]),
            "previously_consumed_dev_families": len(consumed_families[dataset]),
            "train": len(training),
            "development": len(development),
            "confirmation": len(confirmation),
            "train_new_eval_family_overlap": len(train_families & new_eval_families),
        }

    _write_jsonl(args.out / "train.question_only.jsonl", train_rows_out)
    _write_jsonl(args.out / "development.question_only.jsonl", development_out)
    _write_jsonl(args.out / "confirmation.question_only.jsonl", confirmation_out)
    _write_jsonl(
        args.out / "retrieval_requests.jsonl",
        sorted(development_out + confirmation_out, key=lambda row: (row["dataset"], row["role"], row["question_sha256"])),
    )

    all_rows = train_rows_out + development_out + confirmation_out
    forbidden = sorted({field for row in all_rows for field in FORBIDDEN_FIELDS & set(row)})
    keys = [str(row["question_key"]) for row in all_rows]
    role_families = {
        role: {str(row["family_sha256"]) for row in all_rows if row["role"] == role}
        for role in ("train", "development", "confirmation")
    }
    gates = {
        "counts_exact": len(train_rows_out) == 1800 and len(development_out) == 150 and len(confirmation_out) == 300,
        "identity_unique": len(keys) == len(set(keys)),
        "forbidden_fields_zero": not forbidden,
        "gold_access_false_in_freeze": all(row["gold_access"] is False for row in all_rows),
        "train_development_family_overlap_zero": not (role_families["train"] & role_families["development"]),
        "train_confirmation_family_overlap_zero": not (role_families["train"] & role_families["confirmation"]),
        "development_confirmation_family_overlap_zero": not (role_families["development"] & role_families["confirmation"]),
    }
    if not all(gates.values()):
        raise RuntimeError(f"protocol freeze gates failed: {gates}")

    protocol = {
        "schema_version": "qpeg-v4-schema-adaptation-protocol-v1",
        "experiment_id": EXPERIMENT_ID,
        "researcher_approval_id": APPROVAL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_TRAINING_DATA_BUILD_OR_EVALUATION_RETRIEVAL",
        "hypothesis": "Train-only supervision of the exact QPEG sentence-edge citation schema can improve graph-conditioned answers without damaging no-graph ability.",
        "consumed_final_boundary": "The previous QPEG final900 remains closed and cannot be used for tuning, checkpoint selection, or re-evaluation.",
        "cohorts": {
            "train": "600 per dataset from raw train; supporting/decomposition labels may be used only after this freeze",
            "development": "fresh 50 per dataset; checkpoint selection only",
            "confirmation": "fresh 100 per dataset; remains unopened unless the development gate passes",
        },
        "training_data": {
            "graph_rows": 1800,
            "no_graph_replay_rows": 600,
            "total_rows": 2400,
            "graph_fraction": 0.75,
            "graph_target": "train-gold supporting/decomposition sentence edges with exact provenance",
            "no_graph_replay": "same deterministic trace with Knowledge Used: []; 200 qids per dataset",
            "teacher_api_calls": 0,
        },
        "training": {
            "start_adapter": "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final",
            "base_model": "models/llama3-8b",
            "method": "continued LoRA SFT",
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "dtype": "bf16",
            "max_length": 6144,
            "learning_rate": 2e-6,
            "effective_batch": 32,
            "epochs": 1,
            "expected_updates": 75,
            "save_steps": [25, 50, 75],
            "single_primary_variable": "QPEG sentence-edge schema adaptation curriculum",
        },
        "evaluation": {
            "arms": {
                "A": "strong SFT + passages + no graph",
                "B": "strong SFT + same passages + answer-free QPEG",
                "C": "adapted SFT + same passages + answer-free QPEG",
                "D": "adapted SFT + same passages + no graph",
            },
            "primary_effect": "interaction=(C-D)-(B-A)",
            "secondary_effects": ["C-B total adapted-with-graph gain", "D-A no-graph retention"],
            "decoding": "seed42 greedy max_new_tokens512 canonical scorer",
        },
        "development_gates": {
            "macro_interaction_em_gt": 0.0,
            "macro_interaction_f1_gt": 0.0,
            "positive_interaction_datasets_ge": 2,
            "macro_C_minus_B_em_gt": 0.0,
            "macro_D_minus_A_em_ge": -0.01,
            "max_no_graph_net_loss_per_dataset": 1,
            "max_parse_rate_drop": 0.02,
        },
        "confirmation_policy": "Open once, only for the earliest checkpoint passing every development gate; no post-confirmation tuning.",
        "raw_sha256": raw_hashes,
        "dataset_report": dataset_report,
        "gates": gates,
        "scientific_boundary": "Gold support is used only on raw train. Evaluation graphs are answer-free. Passing development gates is not a paper result; confirmation is required.",
    }
    protocol_path = args.out / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": "qpeg-v4-schema-adaptation-freeze-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS",
        "counts": {"train": 1800, "development": 150, "confirmation": 300},
        "dataset_report": dataset_report,
        "forbidden_fields": forbidden,
        "gates": gates,
        "outputs": {
            name: {"path": str(args.out / name), "sha256": _sha256(args.out / name)}
            for name in (
                "train.question_only.jsonl", "development.question_only.jsonl",
                "confirmation.question_only.jsonl", "retrieval_requests.jsonl", "protocol.json",
            )
        },
    }
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra={"phase": "qpeg_v4_protocol_freeze", **report}, status="FROZEN")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
