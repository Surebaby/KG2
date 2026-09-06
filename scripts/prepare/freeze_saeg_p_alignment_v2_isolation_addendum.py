#!/usr/bin/env python
"""Append-only correction for SAEG P-alignment v2 held-out family isolation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.utils.logging import dump_manifest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def main() -> None:
    out = Path("outputs/audits/saeg_p_hard_negative_alignment_v2_isolation_addendum")
    if out.exists():
        raise SystemExit(f"refusing to overwrite addendum: {out}")
    protocol_path = Path("outputs/audits/saeg_p_hard_negative_alignment_v2_protocol/protocol.json")
    train_path = Path("outputs/audits/qpeg_v4_schema_adaptation_protocol_v1/train.question_only.jsonl")
    eval_dir = Path("outputs/audits/saeg_v1_evaluation_protocol_v1")
    eval_paths = {
        role: eval_dir / f"{role}.question_only.jsonl"
        for role in ("development", "confirmation", "canonical_reporting")
    }
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_TRAIN_RETRIEVAL_DATA_BUILD_OR_MODEL_UPDATE":
        raise ValueError("unexpected parent protocol")
    train = read_jsonl(train_path)
    evaluation = [row for path in eval_paths.values() for row in read_jsonl(path)]
    heldout_families = {str(row["family_sha256"]) for row in evaluation}
    heldout_qids = {(str(row["dataset"]), str(row["qid"])) for row in evaluation}
    kept, excluded = [], []
    for row in train:
        reasons = []
        if (str(row["dataset"]), str(row["qid"])) in heldout_qids:
            reasons.append("dataset_qid_overlap")
        if str(row["family_sha256"]) in heldout_families:
            reasons.append("family_overlap")
        if reasons:
            excluded.append({
                "question_key": row["question_key"],
                "dataset": row["dataset"],
                "qid": row["qid"],
                "question_sha256": row["question_sha256"],
                "family_sha256": row["family_sha256"],
                "reasons": reasons,
            })
        else:
            kept.append(row)
    kept_qids = {(str(row["dataset"]), str(row["qid"])) for row in kept}
    kept_families = {str(row["family_sha256"]) for row in kept}
    if kept_qids & heldout_qids or kept_families & heldout_families:
        raise RuntimeError("effective cohort still overlaps held-out evaluation")

    out.mkdir(parents=True, exist_ok=False)
    effective_path = out / "effective_train.question_only.jsonl"
    excluded_path = out / "excluded.identity_only.jsonl"
    write_jsonl(effective_path, kept)
    write_jsonl(excluded_path, excluded)
    addendum = {
        "schema_version": "saeg-p-hard-negative-alignment-isolation-addendum-v1",
        "experiment_id": "SAEG-P-HARD-NEGATIVE-ALIGNMENT-V2-ISOLATION-ADDENDUM",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_CORRECTION_BEFORE_DATA_CLASSIFICATION_OR_MODEL_UPDATE",
        "parent_protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "correction": {
            "original_incorrect_field": "train_cohort.evaluation_qid_or_family_overlap=0",
            "cause": (
                "The reused QPEG-v4 train cohort was isolated from its own development/confirmation, "
                "but the later SAEG protocol also contains canonical_reporting families."
            ),
            "observed_dataset_qid_overlap": len(
                {(str(row['dataset']), str(row['qid'])) for row in train} & heldout_qids
            ),
            "observed_family_overlap": len(
                {str(row['family_sha256']) for row in train} & heldout_families
            ),
            "action": "exclude every overlapping family from all training candidates; do not replace qids",
            "parent_protocol_overwritten": False,
        },
        "counts": {
            "original": len(train),
            "kept": len(kept),
            "excluded": len(excluded),
            "kept_by_dataset": dict(sorted(Counter(str(row["dataset"]) for row in kept).items())),
            "excluded_by_dataset": dict(sorted(Counter(str(row["dataset"]) for row in excluded).items())),
        },
        "effective_gates": {
            "retrieval_artifact_may_retain_all_answer_free_1800": True,
            "training_candidates_must_equal_effective_1781": len(kept) == 1781,
            "training_evaluation_dataset_qid_overlap": 0,
            "training_evaluation_family_overlap": 0,
        },
        "inputs": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path)},
            **{role: {"path": str(path), "sha256": sha256_file(path)} for role, path in eval_paths.items()},
        },
        "outputs": {
            "effective_train": {"path": str(effective_path), "sha256": sha256_file(effective_path)},
            "excluded": {"path": str(excluded_path), "sha256": sha256_file(excluded_path)},
        },
        "scientific_boundary": (
            "This addendum corrects identity isolation only. It does not change questions, retrieval, selector, "
            "quality labels, targets, evaluation inputs, or gates."
        ),
    }
    if len(kept) != 1781 or len(excluded) != 19:
        raise RuntimeError(f"unexpected isolation result: kept={len(kept)}, excluded={len(excluded)}")
    (out / "isolation_addendum.json").write_text(
        json.dumps(addendum, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(out, extra={"phase": "saeg_p_alignment_v2_isolation", **addendum}, status=addendum["status"])
    print(json.dumps(addendum["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
