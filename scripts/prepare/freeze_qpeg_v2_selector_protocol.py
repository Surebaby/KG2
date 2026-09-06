#!/usr/bin/env python
"""Freeze QPEG-v2 selector training and confirmation policy before fitting."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from kgproweight.utils.logging import dump_manifest


OUT = Path("outputs/audits/qpeg_v2_selector_protocol_v1_run2")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite protocol: {OUT}")
    OUT.mkdir(parents=True)
    audit = Path("outputs/audits/qpeg_v2_train_supervision_n1000x3_seed42/report.json")
    sources = [
        Path("kgproweight/kg/qpeg.py"),
        Path("kgproweight/kg/qpeg_selector.py"),
        Path("scripts/diagnose/audit_qpeg_v2_train_supervision.py"),
    ]
    protocol = {
        "schema_version": "qpeg-v2-selector-protocol-v1",
        "experiment_id": "QPEG-V2-SELECTOR-TRAINONLY-N1000X3-SEED42",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_SELECTOR_DATASET_OR_FIT",
        "hypothesis": "A train-only support-relevance selector can remove distracting QPEG edges while retaining evidence-bearing edges across all three datasets.",
        "scope": "new QPEG-v2 method; QPEG-v1.1 STOP decision remains unchanged",
        "data": {
            "datasets": ["hotpotqa", "2wikimultihopqa", "musique"],
            "sample_per_dataset": 1000,
            "selection": "sha256(42::dataset::qid), first 1000",
            "split_unit": "answer-free question family",
            "split": "family_sha256 modulo 10: train=0..6, dev=7, holdout=8..9",
            "gold_policy": "train support/decomposition labels may supervise selector; no dev evaluation answer enters features or graph construction",
        },
        "labels": {
            "hotpotqa": "candidate provenance matches a train supporting_fact title/sentence",
            "2wikimultihopqa": "candidate provenance matches a train supporting_fact title/sentence",
            "musique": "candidate provenance sentence/tail contains its train decomposition-step answer",
        },
        "features": "qpeg-edge-features-v1; question/edge lexical overlap, rank, sentence index, extraction rule/type, numeric/temporal and comparison flags; no answer/support feature",
        "model": {
            "type": "DictVectorizer + LogisticRegression",
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 1000,
            "random_state": 42,
        },
        "threshold_selection": {
            "source": "train-only family-disjoint dev",
            "grid": "0.05..0.95 step 0.01",
            "constraints": {"edge_precision_min": 0.55, "qid_selected_rate_min": 0.70, "each_dataset_qid_selected_rate_min": 0.60},
            "objective": "maximum edge F1; ties choose higher precision then higher threshold",
        },
        "inference": {"max_selected_edges": 6, "below_threshold": "empty graph/no injection", "fallback_to_v1_or_legacy": False},
        "train_holdout_gates": {
            "roc_auc_min": 0.65,
            "selected_edge_precision_min": 0.55,
            "qid_selected_rate_min": 0.65,
            "each_dataset_selected_edge_precision_min": 0.45,
            "each_dataset_qid_selected_rate_min": 0.55,
        },
        "evaluation": {
            "development_pilot": "do not retune on consumed QPEG-v1 pilot; use it only as a labeled historical diagnostic after selector freeze",
            "confirmation": "existing unopened confirmation100x3, same passages/checkpoint/decoding/scorer, no-QPEG vs QPEG-v2",
            "confirmation_gate": "macro delta EM > 0 and no dataset net loss >2/100; otherwise stop before SFT/PPO",
        },
        "forbidden": ["dev/test answers as selector features", "per-qid patches", "second threshold fit on evaluation pilot", "silent legacy/Wikidata fallback"],
        "inputs": {"supervision_audit": {"path": str(audit), "sha256": _sha(audit)}},
        "code_sha256": {str(path): _sha(path) for path in sources},
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(OUT, extra={"phase": "qpeg_v2_selector_protocol", **protocol}, status=protocol["status"])
    print(json.dumps({"status": protocol["status"], "output": str(OUT / 'protocol.json')}, indent=2))


if __name__ == "__main__":
    main()
