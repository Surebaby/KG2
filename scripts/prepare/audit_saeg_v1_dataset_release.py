#!/usr/bin/env python
"""Run release gates for the complete SAEG-v1 train/evaluation datasets."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.data.prompts import build_saeg_inference_messages, build_saeg_sft_messages
from kgproweight.data.saeg_parsers import parse_saeg_steps
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256
from scripts.prepare.freeze_saeg_v1_evaluation_protocol import assert_answer_free


EXPERIMENT_ID = "SAEG-V1-DATASET-RELEASE-AUDIT"
STATUS = "PASS_DATASET_RELEASE_NOT_TRAINED_NOT_EVALUATED"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.floor((len(ordered) - 1) * fraction))]


def token_lengths(train: list[dict[str, Any]], evaluation: list[dict[str, Any]], tokenizer_path: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    train_full: list[int] = []
    train_evidence_only: list[int] = []
    for row in train:
        kwargs = dict(
            question=row["question"],
            kg_triples=row.get("kg_subgraph") or [],
            passage_evidence=row.get("passage_evidence") or [],
            answer_trace=row["teacher_output"],
        )
        full = build_saeg_sft_messages(
            retrieved_passages=(row.get("retrieved_passages") or [])[:10], **kwargs
        )
        minimum = build_saeg_sft_messages(retrieved_passages=[], **kwargs)
        train_full.append(len(tokenizer.apply_chat_template(full, tokenize=True, add_generation_prompt=False)))
        train_evidence_only.append(len(tokenizer.apply_chat_template(minimum, tokenize=True, add_generation_prompt=False)))
    eval_full: list[int] = []
    for row in evaluation:
        messages = build_saeg_inference_messages(
            question=row["question"],
            retrieved_passages=(row.get("passages") or [])[:10],
            kg_triples=row.get("wikidata_kg") or [],
            passage_evidence=row.get("passage_evidence") or [],
        )
        eval_full.append(len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)))
    def stats(values: list[int]) -> dict[str, Any]:
        return {
            "n": len(values), "min": min(values), "p50": percentile(values, .50),
            "p95": percentile(values, .95), "p99": percentile(values, .99), "max": max(values),
            "over_4096": sum(value > 4096 for value in values),
            "over_6144": sum(value > 6144 for value in values),
        }
    return {
        "tokenizer": str(tokenizer_path),
        "train_full_top10": stats(train_full),
        "train_after_all_retrieved_passages_dropped": stats(train_evidence_only),
        "evaluation_full_top10_prompt_only": stats(eval_full),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_dir", type=Path, default=Path("data/silver_data/saeg_v1_train4860_family_disjoint_seed42_v2"))
    parser.add_argument("--eval_dir", type=Path, default=Path("data/derived/saeg_v1_evaluation_inputs_seed42_v1"))
    parser.add_argument("--gold_dir", type=Path, default=Path("data/derived/saeg_v1_scorer_gold_seed42_v2"))
    parser.add_argument("--protocol_dir", type=Path, default=Path("outputs/audits/saeg_v1_evaluation_protocol_v1"))
    parser.add_argument("--fresh_proof_report", type=Path, default=Path(
        "data/derived/saeg_v1_2wiki_dev_confirmation_proofkg_v1/report.json"))
    parser.add_argument("--tokenizer", type=Path, default=Path("models/llama3-8b"))
    parser.add_argument("--out", type=Path, default=Path("outputs/audits/saeg_v1_dataset_release_audit"))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite release audit: {args.out}")
    train_path = args.train_dir / "silver_train.jsonl"
    eval_paths = {
        role: args.eval_dir / f"{role}.answer_free.jsonl"
        for role in ("development", "confirmation", "canonical_reporting")
    }
    gold_paths = {
        role: args.gold_dir / f"{role}.gold.jsonl"
        for role in ("development", "confirmation", "canonical_reporting")
    }
    inputs = {
        "train": train_path,
        **{f"eval_{key}": value for key, value in eval_paths.items()},
        **{f"gold_{key}": value for key, value in gold_paths.items()},
        "eval_protocol": args.protocol_dir / "protocol.json",
        "train_report": args.train_dir / "report.json",
        "eval_report": args.eval_dir / "report.json",
        "gold_report": args.gold_dir / "report.json",
        "fresh_proof_report": args.fresh_proof_report,
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    train = read_jsonl(train_path)
    evaluation = [row for path in eval_paths.values() for row in read_jsonl(path)]
    gold = [row for path in gold_paths.values() for row in read_jsonl(path)]
    if len(train) != 4860 or len(evaluation) != 1350 or len(gold) != 1350:
        raise ValueError("unexpected release cardinality")

    train_ids = {(row["dataset"], row["source_qid"]) for row in train}
    eval_ids = {(row["dataset"], row["qid"]) for row in evaluation}
    if train_ids & eval_ids:
        raise ValueError("train/evaluation qid overlap")
    train_families = {family_sha256(row["question"]) for row in train}
    fresh_dev = [row for row in evaluation if row["role"] == "development"]
    fresh_confirmation = [row for row in evaluation if row["role"] == "confirmation"]
    dev_families = {row["family_sha256"] for row in fresh_dev}
    confirmation_families = {row["family_sha256"] for row in fresh_confirmation}
    if train_families & dev_families or train_families & confirmation_families or dev_families & confirmation_families:
        raise ValueError("train/development/confirmation family overlap")

    train_contract_errors = 0
    p_pseudo_in_kg = 0
    for row in train:
        p_pseudo_in_kg += sum(str(triple[1]).casefold() == "evidence sentence" for triple in row.get("kg_subgraph") or [])
        parsed = parse_saeg_steps(
            row["teacher_output"],
            known_kg=row.get("kg_subgraph") or [],
            known_passage_ids=[item["passage_id"] for item in row.get("passage_evidence") or []],
        )
        train_contract_errors += sum(not step.citation_contract_valid for step in parsed)
    if train_contract_errors or p_pseudo_in_kg:
        raise ValueError("train citation/KG contract failed")
    for row in evaluation:
        assert_answer_free(row)
        p_pseudo_in_kg += sum(str(triple[1]).casefold() == "evidence sentence" for triple in row.get("wikidata_kg") or [])
    if p_pseudo_in_kg:
        raise ValueError("passage pseudo-triple leaked into a KG field")

    eval_index = {(row["role"], row["question_key"]): row for row in evaluation}
    gold_index = {(row["role"], row["question_key"]): row for row in gold}
    if set(eval_index) != set(gold_index):
        raise ValueError("answer-free/scorer-Gold identity join is not 1.0")
    if any(eval_index[key]["question_sha256"] != gold_index[key]["question_sha256"] for key in eval_index):
        raise ValueError("answer-free/scorer-Gold question hash mismatch")
    if any(not row.get("sealed") for row in gold if row["role"] == "confirmation"):
        raise ValueError("confirmation Gold not sealed")

    tokens = token_lengths(train, evaluation, args.tokenizer)
    if tokens["train_after_all_retrieved_passages_dropped"]["over_4096"]:
        raise ValueError("some train records cannot fit 4096 tokens even after retrieved-passage dropping")

    counts = Counter()
    for row in train:
        counts[f"train::{row['dataset']}::{row['evidence_mode']}"] += 1
    for row in evaluation:
        counts[f"eval::{row['role']}::{row['dataset']}"] += 1
    report = {
        "schema_version": "saeg-dataset-release-audit-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "counts": dict(sorted(counts.items())),
        "integrity": {
            "train_records": len(train),
            "evaluation_records": len(evaluation),
            "scorer_gold_records": len(gold),
            "train_eval_qid_overlap": 0,
            "train_development_family_overlap": 0,
            "train_confirmation_family_overlap": 0,
            "development_confirmation_family_overlap": 0,
            "answer_fields_in_inference_inputs": 0,
            "passage_pseudo_triples_in_kg_fields": 0,
            "invalid_train_citation_steps": 0,
            "answer_free_gold_identity_join_rate": 1.0,
            "confirmation_opened_or_scored": False,
        },
        "token_audit": tokens,
        "known_negative_results_preserved": {
            "qpeg_v4": "FAIL_STOP_DEVELOPMENT (25/50/75 all failed)",
            "fresh_2wiki_proofkg": json.loads(args.fresh_proof_report.read_text(encoding="utf-8"))["status"],
        },
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in inputs.items()},
        "scientific_boundary": (
            "Dataset release readiness only. No SAEG checkpoint has been trained or evaluated; "
            "confirmation is sealed and canonical reporting remains nonconfirmatory."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    card = f"""# SAEG-v1 Dataset Release

Status: `{STATUS}`  
Experiment ID: `{EXPERIMENT_ID}`

## What is included

- Train-only SFT release: 4,860 trajectories, with train Gold support/decomposition explicitly marked and Wikidata construction Gold-free. Two variants from the 4,862-row master were excluded for one cross-dataset family collision.
- Development: 150 answer-free inputs (50 per dataset), for method/checkpoint decisions.
- Confirmation: 300 answer-free inputs (100 per dataset), sealed and not yet model-evaluated.
- Canonical reporting: 900 answer-free inputs (300 per dataset), historical and nonconfirmatory, for baseline-compatible tables only.
- Scorer Gold: physically separate files with exact identity/hash joins.

## Citation contract

- `Knowledge Used` contains only standard Wikidata `(head, relation, tail)` triples.
- `Passage Used` contains only visible `[P<n>]` passage-evidence IDs.
- Passage sentences never appear in KG fields as `(title, evidence sentence, sentence)` pseudo-triples.

## Resource arms

- Same-resource main comparison: no graph vs Passage-QPEG, using the identical frozen Top-10 passages.
- Extra-resource arm: Wikidata/Fused on canonical 2Wiki only, where ProofKG passed its earlier structural gate.
- Fresh 2Wiki W, HotpotQA W, and MuSiQue W fail closed after their recorded structural failures.

## Scientific boundaries

- This release is data-complete, not a positive model result.
- Confirmation cannot be scored until a frozen development gate explicitly opens it once.
- Canonical reporting data cannot select checkpoints or tune the method.
- Historical failed experiments are retained and listed in `report.json`.
"""
    (args.out / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    dump_manifest(args.out, extra=report, status=STATUS)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
