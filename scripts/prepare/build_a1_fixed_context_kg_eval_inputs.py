#!/usr/bin/env python
"""Build paired legacy/proof-KG inputs for the frozen A1 engineering cohort.

Legacy KG construction is completed and hashed before the 2Wiki source rows
(which contain answers and evidence) are loaded.  The two final arms differ
only in ``kg_subgraph``; passages, questions, order, and scoring labels match.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from kgproweight.kg.kg_filter import _pid_for_triple, filter_and_rank_triples
from kgproweight.kg.question_kg import validate_question_kg_record
from kgproweight.utils.logging import dump_manifest, prepare_new_run_dir


_legacy_builder = importlib.import_module("scripts.prepare.06_build_question_kg_index")
_build_components = _legacy_builder._build_components
_resolve_one = _legacy_builder._resolve_one
Triple = Tuple[str, str, str]
BUILDER_VERSION = "a1-fixed-context-paired-inputs-1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def context_to_passages(metadata: Mapping[str, Any]) -> List[Dict[str, str]]:
    context = metadata.get("context") or {}
    titles = list(context.get("title") or [])
    contents = list(context.get("content") or [])
    if len(titles) != len(contents):
        raise ValueError("2Wiki context title/content lengths differ")
    passages: List[Dict[str, str]] = []
    for index, (title, sentences) in enumerate(zip(titles, contents), start=1):
        body = " ".join(str(value).strip() for value in (sentences or []) if str(value).strip())
        text = f"{str(title).strip()}\n{body}".strip()
        if text:
            passages.append(
                {
                    "id": f"frozen_context_{index}",
                    "contents": text,
                    "source": "2wikimultihopqa_train_context",
                }
            )
    return passages


def non_kg_projection(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if key != "kg_subgraph"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--proof_kg", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument(
        "--passages_jsonl",
        help=(
            "Optional frozen Gold-free retrieval output keyed by qid. When set, "
            "both arms use these passages instead of the dataset fixed context."
        ),
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--max_mentions", type=int, default=5)
    parser.add_argument("--max_keep", type=int, default=12)
    parser.add_argument("--min_keep", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort_path = Path(args.cohort).resolve()
    proof_path = Path(args.proof_kg).resolve()
    passages_path = Path(args.passages_jsonl).resolve() if args.passages_jsonl else None
    cohort = _read_jsonl(cohort_path)
    proof_rows = _read_jsonl(proof_path)
    if not cohort or len(cohort) != len(proof_rows):
        raise SystemExit("cohort and proof KG must be non-empty with equal row counts")
    proof_by_qid: Dict[str, Dict[str, Any]] = {}
    for row in proof_rows:
        validate_question_kg_record(row)
        qid = str(row["qid"])
        if qid in proof_by_qid:
            raise SystemExit(f"duplicate proof KG qid: {qid}")
        proof_by_qid[qid] = row
    cohort_qids = [str(row["qid"]) for row in cohort]
    if set(cohort_qids) != set(proof_by_qid):
        raise SystemExit("cohort and proof KG qids differ")
    for row in cohort:
        proof = proof_by_qid[str(row["qid"])]
        if str(row["question"]).strip() != str(proof["question"]).strip():
            raise SystemExit(f"question mismatch for qid={row['qid']}")

    run_dir, experiment_id = prepare_new_run_dir(
        args.run_dir,
        experiment_id=args.experiment_id,
        extra={
            "phase": "prepare_fixed_context_kg_paired_inputs",
            "gold_used_for_legacy_kg_build": False,
        },
    )
    legacy_runtime_path = run_dir / "legacy_kg_runtime_before_gold.jsonl"
    legacy_freeze_path = run_dir / "legacy_kg_runtime_freeze.json"
    legacy_input_path = run_dir / "arm_legacy.jsonl"
    proof_input_path = run_dir / "arm_proof.jsonl"
    report_path = run_dir / "report.json"

    try:
        # Gold-free legacy build. Source dataset rows are intentionally not
        # opened until this artifact has been written and hashed.
        linker, retriever = _build_components(offline=True)
        legacy_runtime: List[Dict[str, Any]] = []
        for index, frozen in enumerate(cohort, start=1):
            resolved = _resolve_one(
                str(frozen["question"]),
                str(frozen["dataset"]),
                str(frozen["qid"]),
                linker,
                retriever,
                args.max_mentions,
            )
            raw_triples: List[Triple] = [
                tuple(value) for value in resolved.get("triples") or [] if len(value) == 3
            ]
            pid_map = {triple: _pid_for_triple(triple) for triple in raw_triples}
            question_entities = [
                value["mention"]
                for value in resolved.get("linked_entities") or []
                if value.get("qid") and not value.get("abstained")
            ] or None
            rich = filter_and_rank_triples(
                raw_triples,
                str(frozen["question"]),
                pid_map=pid_map,
                max_keep=args.max_keep,
                min_keep=args.min_keep,
                rich=True,
                question_entities=question_entities,
            )
            triples = [[value["h"], value["r"], value["t"]] for value in rich]
            legacy_runtime.append(
                {
                    "row_id": frozen["row_id"],
                    "dataset": frozen["dataset"],
                    "qid": frozen["qid"],
                    "question": frozen["question"],
                    "kg_subgraph": triples,
                    "linked_entities": resolved.get("linked_entities") or [],
                    "raw_triple_count": len(raw_triples),
                    "builder_version": BUILDER_VERSION,
                    "gold_used_for_build": False,
                }
            )
            print(f"legacy KG {index}/{len(cohort)}", flush=True)
        _write_jsonl(legacy_runtime_path, legacy_runtime)
        legacy_freeze = {
            "status": "LEGACY_KG_FROZEN_BEFORE_SOURCE_DATA_ACCESS",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "path": str(legacy_runtime_path),
            "sha256": _sha256(legacy_runtime_path),
            "gold_used_for_build": False,
        }
        legacy_freeze_path.write_text(
            json.dumps(legacy_freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        passages_by_qid: Dict[str, Dict[str, Any]] = {}
        if passages_path is not None:
            for row in _read_jsonl(passages_path):
                qid = str(row.get("qid") or "")
                if not qid or qid in passages_by_qid:
                    raise RuntimeError(f"missing or duplicate retrieval qid: {qid!r}")
                if not isinstance(row.get("retrieved_passages"), list):
                    raise RuntimeError(f"retrieval row has no passage list: {qid}")
                passages_by_qid[qid] = row
            if set(passages_by_qid) != set(cohort_qids):
                raise RuntimeError("frozen retrieval qids differ from cohort qids")
            for frozen in cohort:
                passage_row = passages_by_qid[str(frozen["qid"])]
                if str(passage_row.get("question") or "").strip() != str(frozen["question"]).strip():
                    raise RuntimeError(f"retrieval/cohort question mismatch for {frozen['qid']}")

        # Only the final fixed-context evaluation inputs need source passages
        # and labels. They never feed back into either KG construction arm.
        selected = set(cohort_qids)
        dataset_path = Path(args.data_root).resolve() / "2wikimultihopqa" / "train.jsonl"
        source_by_qid: Dict[str, Dict[str, Any]] = {}
        with dataset_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = str(row.get("id") or "")
                if qid in selected:
                    source_by_qid[qid] = row
        if len(source_by_qid) != len(cohort):
            raise RuntimeError("some cohort qids are absent from 2Wiki train")
        legacy_by_qid = {str(row["qid"]): row for row in legacy_runtime}
        legacy_inputs: List[Dict[str, Any]] = []
        proof_inputs: List[Dict[str, Any]] = []
        for frozen in cohort:
            qid = str(frozen["qid"])
            source = source_by_qid[qid]
            question = str(source.get("question") or "").strip()
            if question != str(frozen["question"]).strip():
                raise RuntimeError(f"source/cohort question mismatch for {qid}")
            common = {
                "row_id": frozen["row_id"],
                "dataset": "2wikimultihopqa",
                "qid": qid,
                "question": question,
                "gold_answers": list(source.get("golden_answers") or []),
                "retrieved_passages": (
                    list(passages_by_qid[qid]["retrieved_passages"])
                    if passages_path is not None
                    else context_to_passages(source.get("metadata") or {})
                ),
                "scope": (
                    "family_disjoint_train_standard_retrieval_model_utility"
                    if passages_path is not None
                    else "seen_train_fixed_context_engineering_diagnostic"
                ),
            }
            legacy_inputs.append(
                {**common, "kg_subgraph": list(legacy_by_qid[qid]["kg_subgraph"])}
            )
            proof_inputs.append(
                {**common, "kg_subgraph": list(proof_by_qid[qid]["kg_subgraph"])}
            )
        if any(
            non_kg_projection(left) != non_kg_projection(right)
            for left, right in zip(legacy_inputs, proof_inputs)
        ):
            raise RuntimeError("paired arm inputs differ outside kg_subgraph")
        _write_jsonl(legacy_input_path, legacy_inputs)
        _write_jsonl(proof_input_path, proof_inputs)

        legacy_counts = [len(row["kg_subgraph"]) for row in legacy_inputs]
        proof_counts = [len(row["kg_subgraph"]) for row in proof_inputs]
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": experiment_id,
            "status": "COMPLETE_INPUTS_FROZEN_NO_MODEL_INFERENCE",
            "scope": (
                "family-disjoint 2Wiki train n100; frozen standard retrieval; "
                "zero training; no inference yet"
                if passages_path is not None
                else "seen 2Wiki train A1; fixed raw context; zero training; no inference yet"
            ),
            "protocol": {
                "paired_non_kg_fields_identical": True,
                "legacy_kg_gold_used_for_build": False,
                "proof_kg_gold_used_for_build": False,
                "passages": (
                    "frozen Gold-free standard retrieval output, identical in both arms"
                    if passages_path is not None
                    else "all 10 frozen 2Wiki context paragraphs in dataset order"
                ),
                "gold_used_only_for_future_scoring": True,
                "legacy_policy": {
                    "offline": True,
                    "max_mentions": args.max_mentions,
                    "max_keep": args.max_keep,
                    "min_keep": args.min_keep,
                },
            },
            "inputs": {
                "cohort": {"path": str(cohort_path), "sha256": _sha256(cohort_path)},
                "proof_kg": {"path": str(proof_path), "sha256": _sha256(proof_path)},
                "frozen_retrieval": (
                    {"path": str(passages_path), "sha256": _sha256(passages_path)}
                    if passages_path is not None else None
                ),
                "dataset_post_kg_build_only": {
                    "path": str(dataset_path), "sha256": _sha256(dataset_path)
                },
            },
            "kg_stats": {
                "legacy_nonempty": sum(bool(value) for value in legacy_counts),
                "legacy_mean_triples": sum(legacy_counts) / len(legacy_counts),
                "proof_nonempty": sum(bool(value) for value in proof_counts),
                "proof_mean_triples": sum(proof_counts) / len(proof_counts),
                "legacy_count_histogram": dict(Counter(legacy_counts)),
                "proof_count_histogram": dict(Counter(proof_counts)),
            },
            "outputs": {
                "legacy_kg_runtime": {
                    "path": str(legacy_runtime_path), "sha256": _sha256(legacy_runtime_path)
                },
                "legacy_kg_runtime_freeze": {
                    "path": str(legacy_freeze_path), "sha256": _sha256(legacy_freeze_path)
                },
                "arm_legacy": {
                    "path": str(legacy_input_path), "sha256": _sha256(legacy_input_path)
                },
                "arm_proof": {
                    "path": str(proof_input_path), "sha256": _sha256(proof_input_path)
                },
            },
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        dump_manifest(
            run_dir,
            extra={
                "experiment_id": experiment_id,
                "phase": "prepare_fixed_context_kg_paired_inputs",
                "scope": report["scope"],
                "inputs": report["inputs"],
                "outputs": report["outputs"],
            },
        )
        print(json.dumps({"status": report["status"], "kg_stats": report["kg_stats"]}, indent=2))
    except Exception as exc:
        dump_manifest(
            run_dir,
            extra={
                "experiment_id": experiment_id,
                "phase": "prepare_fixed_context_kg_paired_inputs",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            },
            status="FAILED_RUNTIME",
        )
        raise


if __name__ == "__main__":
    main()
