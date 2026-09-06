#!/usr/bin/env python
"""Freeze the SAEG-v1 strong-SFT zero-training development utility run."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.data.saeg_dataset import ARMS, assert_role_allowed, iter_saeg_eval_inputs
from kgproweight.utils.logging import artifact_identity, dump_manifest


EXPERIMENT_ID = "SAEG-V1-DEVELOPMENT-STRONG-SFT-ZERO-TRAINING-NPDF-SEED42-V2"
STATUS = "FROZEN_BEFORE_DEVELOPMENT_GENERATION"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval_input",
        type=Path,
        default=Path("data/derived/saeg_v1_evaluation_inputs_seed42_v1/development.answer_free.jsonl"),
    )
    parser.add_argument(
        "--scorer_gold",
        type=Path,
        default=Path("data/derived/saeg_v1_scorer_gold_seed42_v2/development.gold.jsonl"),
    )
    parser.add_argument(
        "--evaluation_protocol",
        type=Path,
        default=Path("outputs/audits/saeg_v1_evaluation_protocol_v1/protocol.json"),
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"),
    )
    parser.add_argument("--base_model", type=Path, default=Path("models/llama3-8b"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/audits/saeg_v1_development_zero_training_protocol_v2"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite SAEG development protocol: {args.out}")
    for path in (
        args.eval_input,
        args.scorer_gold,
        args.evaluation_protocol,
        args.adapter,
        args.base_model,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    rows = list(iter_saeg_eval_inputs(args.eval_input))
    gold = read_jsonl(args.scorer_gold)
    if len(rows) != 150 or len(gold) != 150:
        raise ValueError("development must contain exactly 150 inference and scorer rows")
    for row in rows:
        assert_role_allowed(row)
    if [row["question_key"] for row in rows] != [row["question_key"] for row in gold]:
        raise ValueError("development inference/scorer identity order differs")
    if any(
        row["question_sha256"] != target["question_sha256"]
        for row, target in zip(rows, gold)
    ):
        raise ValueError("development inference/scorer question hash differs")
    if any(target.get("role") != "development" or target.get("sealed") is not False for target in gold):
        raise ValueError("scorer file is not an open development-only file")

    by_dataset = Counter(str(row["dataset"]) for row in rows)
    if by_dataset != Counter({"hotpotqa": 50, "2wikimultihopqa": 50, "musique": 50}):
        raise ValueError("development must be balanced 50/50/50")
    arm_eligible = Counter()
    for row in rows:
        for arm in ARMS:
            arm_eligible[arm] += bool((row.get("arms") or {}).get(arm, {}).get("eligible"))
    if any(row.get("wikidata_kg") for row in rows):
        raise ValueError("fresh development unexpectedly contains Wikidata evidence")
    if any(
        str((row.get("source_status") or {}).get("wikidata"))
        != "not_eligible_frozen_structural_failure"
        for row in rows
    ):
        raise ValueError("fresh development Wikidata structural status differs from the frozen failure")

    protocol = {
        "schema_version": "saeg-development-zero-training-utility-protocol-v2",
        "experiment_id": EXPERIMENT_ID,
        "researcher_approval": "2026-09-03: 可以 开始吧",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "scope": "development n=150 (50 per dataset); confirmation remains sealed",
        "arms": {
            "A_no_graph": "same frozen passages; no structured evidence",
            "B_passage": "same frozen passages + passage-derived evidence objects; empty fails closed to A",
            "C_wikidata": "Wikidata-only when structurally eligible; otherwise NOT_EVALUABLE, never treated as a zero score",
            "D_fused": "P+W when eligible; current fresh development W failed structurally, so D is expected to equal B",
        },
        "source_nonempty_counts": {
            "passage": sum(bool(row.get("passage_evidence")) for row in rows),
            "wikidata": sum(bool(row.get("wikidata_kg")) for row in rows),
        },
        "materialized_arm_eligibility_counts": dict(sorted(arm_eligible.items())),
        "evaluation_population_counts": {
            "A_no_graph": len(rows),
            "B_passage": len(rows),
            "C_wikidata": 0,
            "D_fused": len(rows),
        },
        "paired_population_rule": (
            "A/B/D contain the same 150 question keys. For the 14 rows without passage evidence, "
            "B and D fail closed to A and reuse the byte-identical prompt/generation; they are not dropped."
        ),
        "generation": {
            "seed": 42,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "max_new_tokens": 512,
            "top_k_passages": 10,
            "max_wikidata_triples": 12,
            "max_passage_evidence": 8,
        },
        "decision_gates": {
            "primary_comparison": "D_fused - A_no_graph",
            "macro_delta_em_strictly_positive": True,
            "macro_delta_f1_strictly_positive": True,
            "min_datasets_with_positive_delta_em": 2,
            "max_net_correct_loss_per_dataset": 2,
            "covered_subset_delta_em_strictly_positive": True,
            "max_parse_rate_drop_per_dataset": 0.02,
            "fallback_prediction_and_generation_exact": True,
            "c_wikidata_not_evaluable_is_not_a_failure": True,
        },
        "inputs": {
            "eval_input": {"path": str(args.eval_input), "sha256": sha256_file(args.eval_input)},
            "scorer_gold": {"path": str(args.scorer_gold), "sha256": sha256_file(args.scorer_gold)},
            "evaluation_protocol": {
                "path": str(args.evaluation_protocol),
                "sha256": sha256_file(args.evaluation_protocol),
            },
            "adapter": artifact_identity(args.adapter),
            "base_model": artifact_identity(args.base_model),
        },
        "integrity": {
            "inference_gold_identity_join_rate": 1.0,
            "inference_input_answer_free": True,
            "development_only": True,
            "confirmation_opened": False,
            "balanced_50_per_dataset": True,
        },
        "scientific_boundary": (
            "This freezes a zero-training development decision. It does not open confirmation, "
            "select an SFT checkpoint, modify evaluation, or authorize bypassing a failed utility gate."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    protocol_path = args.out / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra=protocol, status=STATUS)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
