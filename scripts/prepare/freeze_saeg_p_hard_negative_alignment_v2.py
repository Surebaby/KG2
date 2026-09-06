#!/usr/bin/env python
"""Freeze the train-only SAEG Passage-QPEG hard-negative alignment v2 protocol.

This protocol is append-only and is written before train retrieval, evidence
classification, SFT data materialisation, or any model update.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from kgproweight.utils.logging import dump_manifest


EXPERIMENT_ID = "SAEG-P-HARD-NEGATIVE-ALIGNMENT-V2-SEED42"
STATUS = "FROZEN_BEFORE_TRAIN_RETRIEVAL_DATA_BUILD_OR_MODEL_UPDATE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    out = Path("outputs/audits/saeg_p_hard_negative_alignment_v2_protocol")
    if out.exists():
        raise SystemExit(f"refusing to overwrite frozen protocol: {out}")

    cohort = Path("outputs/audits/qpeg_v4_schema_adaptation_protocol_v1/train.question_only.jsonl")
    selector = Path("outputs/training/qpeg_v3_sentence_selector_n1000x3_seed42/selector.joblib")
    selector_decision = Path(
        "outputs/audits/qpeg_v3_sentence_selector_runtime_cap_correction_v1/decision_addendum.json"
    )
    failed_utility = Path(
        "outputs/validation/saeg_v1_development_strong_sft_npdf_v2_attempt2/report.json"
    )
    raw_paths = {
        dataset: Path("data") / dataset / "train.jsonl"
        for dataset in ("hotpotqa", "2wikimultihopqa", "musique")
    }

    rows = [json.loads(line) for line in cohort.read_text(encoding="utf-8").splitlines() if line]
    per_dataset = {
        dataset: sum(str(row.get("dataset")) == dataset for row in rows)
        for dataset in raw_paths
    }
    if len(rows) != 1800 or any(value != 600 for value in per_dataset.values()):
        raise ValueError(f"unexpected frozen train cohort: total={len(rows)}, {per_dataset}")
    if any(row.get("gold_access") is not False or row.get("role") != "train" for row in rows):
        raise ValueError("train retrieval requests must be question-only and gold_access=false")

    protocol: dict[str, Any] = {
        "schema_version": "saeg-p-hard-negative-alignment-protocol-v2",
        "experiment_id": EXPERIMENT_ID,
        "researcher_approval": "USER_APPROVED_2026-09-03_P_HARD_NEGATIVE_ALIGNMENT_V2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "failed_predecessor": {
            **ref(failed_utility),
            "status": "FAIL_STOP_BEFORE_SFT",
            "observed_D_minus_A_macro_em": -0.06666666666666667,
            "interpretation": (
                "Automatic Passage-QPEG was consumed but frequently omitted required hops or selected "
                "related distractors; the predecessor train P branch used only Gold support/decomposition."
            ),
        },
        "single_primary_variable": (
            "Replace Gold-perfect-only Passage-QPEG SFT inputs with the same automatic retrieval+selector "
            "distribution used at evaluation and supervise selective citation/abstention by train-only Gold."
        ),
        "train_cohort": {
            **ref(cohort),
            "total": len(rows),
            "per_dataset": per_dataset,
            "identity_reused_from": "QPEG-V4 train-only frozen cohort",
            "evaluation_qid_or_family_overlap": 0,
        },
        "automatic_input_path": {
            "retrieval": "E5@100 + BM25@100 -> RRF(k=60)@50 -> bge-reranker-v2-m3@10 -> pack3860",
            "selector": ref(selector),
            "selector_decision": ref(selector_decision),
            "threshold": 0.77,
            "max_selected_edges": 4,
            "gold_access_during_retrieval_and_selection": False,
        },
        "train_only_gold_diagnostics": {
            "raw_train": {dataset: ref(path) for dataset, path in raw_paths.items()},
            "hotpotqa_and_2wiki_required_unit": (
                "exact normalized (supporting-fact title, supporting-fact sentence) from raw train"
            ),
            "musique_required_unit": (
                "one exact normalized support sentence per question_decomposition hop, selected from the "
                "hop support paragraph by the frozen answer-sentence rule"
            ),
            "matching": "exact normalized title and exact normalized sentence; no fuzzy or answer search",
            "gold_use_boundary": (
                "Gold is used only after answer-free retrieval/selection to classify train evidence and "
                "construct train targets. These rows are evaluation_eligible=false."
            ),
        },
        "quality_classes": {
            "complete": "all required support units are selected by automatic P",
            "partial": "at least one but not all required support units are selected",
            "misleading": "automatic P is nonempty but selects zero required support units",
            "empty": "automatic P selects zero edges",
            "unresolved_gold": "fewer than two usable required units; exclude and report",
        },
        "target_policy": {
            "complete": "cite selected P edges that exactly match required support units",
            "partial": (
                "keep the whole automatic P block visible; cite only the selected edges that match required "
                "units; never cite selected distractor edges"
            ),
            "misleading": (
                "keep the whole automatic P block visible; Passage Used must remain empty in every target step"
            ),
            "empty": "ordinary no-P replay target",
            "answer_target": "raw-train Gold answer only; no teacher API",
            "loss": "unchanged token-level SFT cross entropy; no new auxiliary loss",
        },
        "materialisation_gates": {
            "retrieval_rows_exact": 1800,
            "each_dataset_rows_exact": 600,
            "each_row_has_10_passages": True,
            "identity_and_question_hash_join_rate": 1.0,
            "selector_model_hash_match": True,
            "quality_classification_rate": 1.0,
            "minimum_complete_per_dataset": 20,
            "minimum_partial_plus_misleading_per_dataset": 100,
            "selected_edge_target_citation_errors": 0,
            "evaluation_qid_and_family_overlap": 0,
            "raw_or_gold_fields_in_retrieval_artifact": 0,
        },
        "development_evaluation": {
            "population": "reuse the already-consumed SAEG development150 only as development",
            "inputs": "bit-identical frozen v2 A_no_graph/B_passage inputs; no evaluation-data rebuild",
            "base": "strong SFT checkpoint",
            "candidate": "hard-negative-aligned continued-SFT checkpoint",
            "primary_effect": "interaction=(candidate_P-candidate_noP)-(strong_P-strong_noP)",
            "gates": {
                "macro_interaction_em_gt": 0.0,
                "macro_interaction_f1_gt": 0.0,
                "positive_interaction_datasets_ge": 2,
                "candidate_P_minus_strong_P_macro_em_gt": 0.0,
                "candidate_P_minus_candidate_noP_macro_em_ge": -0.01,
                "candidate_noP_minus_strong_noP_macro_em_ge": -0.01,
                "max_noP_net_loss_per_dataset": 1,
                "max_parse_rate_drop": 0.02,
            },
            "confirmation_policy": (
                "Do not open sealed confirmation. Submit an explicit request only after one earliest checkpoint "
                "passes every development gate."
            ),
        },
        "execution_boundary": (
            "This approval permits protocol/data implementation and train-only materialisation. It does not "
            "permit large training, opening confirmation, changing reward/loss, or starting PPO."
        ),
    }
    out.mkdir(parents=True, exist_ok=False)
    protocol_path = out / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(out, extra={"phase": "saeg_p_hard_negative_alignment_v2_protocol", **protocol}, status=STATUS)
    print(json.dumps({"status": STATUS, "out": str(out), "counts": per_dataset}, ensure_ascii=False))


if __name__ == "__main__":
    main()
