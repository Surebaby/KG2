#!/usr/bin/env python
"""Replay fixed Phase-1 passages/KG through one Teacher model.

Every submitted qid is accounted for. Source rows are never modified and API
credentials are read only from the environment.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List

from kgproweight.data.parsers import extract_final_answer
from kgproweight.data.prompts import build_teacher_messages
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path
from kgproweight.training.phase1_distill import (
    StratifiedSilverFilter,
    TeacherClient,
    _annotate_steps,
    _build_retry_messages,
    _needs_format_retry,
    _parsed_contract_errors,
    answer_match_score,
)
from kgproweight.utils.logging import dump_manifest, get_logger


logger = get_logger(__name__)


def _model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.casefold()).strip("_") or "teacher"


def _read(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _step_dict(step) -> Dict[str, Any]:
    return {
        "index": step.index,
        "text": step.text,
        "label": float(step.label),
        "cited_triples": [list(triple) for triple in step.cited_triples],
        "token_logprobs": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--teacher", default="deepseek-v4-pro")
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--expected", type=int, default=90)
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()

    source_paths = [Path(path).resolve() for path in args.source]
    rows = [row for path in source_paths for row in _read(path)]
    qids = [str(row.get("qid") or row.get("id") or "") for row in rows]
    if len(rows) != args.expected:
        raise SystemExit(f"source has {len(rows)} rows, expected {args.expected}")
    if not all(qids) or len(set(qids)) != len(qids):
        raise SystemExit("source qids are empty or duplicated")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / f"{_model_slug(args.teacher)}.candidates.jsonl"
    failure_path = output_dir / "failures.jsonl"

    teacher = TeacherClient(
        model=args.teacher,
        backend="deepseek",
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        thinking=args.thinking == "enabled",
    )
    linker = EntityLinker(cache_path=resolve_entity_cache_path(), offline=True)
    annotator = PRMAnnotator(entity_linker=linker, verbose=False)
    quality_filter = StratifiedSilverFilter()

    def run_one(source: Dict[str, Any]) -> Dict[str, Any]:
        qid = str(source.get("qid") or source.get("id") or "")
        kg = [tuple(triple) if isinstance(triple, list) else triple for triple in source.get("kg_subgraph") or []]
        passages = list(source.get("retrieved_passages") or [])
        messages = build_teacher_messages(
            question=str(source.get("question") or ""),
            retrieved_passages=passages,
            kg_triples=kg,
            top_k=10,
            max_kg_triples=12,
        )
        api_calls: List[Dict[str, Any]] = []
        try:
            raw, call_metadata = teacher.chat_with_metadata(messages)
            call_metadata["purpose"] = "primary"
            api_calls.append(call_metadata)
        except Exception as exc:  # noqa: BLE001
            return {"_status": "api_error", "qid": qid, "error": f"{type(exc).__name__}: {exc}"}
        if not raw.strip():
            return {"_status": "empty", "qid": qid, "api_calls": api_calls}

        steps = _annotate_steps(raw, kg, annotator)
        retry_attempted = _needs_format_retry(
            steps, kg, quality_filter.min_steps, raw_output=raw,
            max_steps=quality_filter.max_steps,
        )
        retry_succeeded = False
        retry_error = ""
        if retry_attempted:
            try:
                retry_raw, retry_metadata = teacher.chat_with_metadata(_build_retry_messages(messages))
                retry_metadata["purpose"] = "format_retry"
                api_calls.append(retry_metadata)
                retry_steps = _annotate_steps(retry_raw, kg, annotator) if retry_raw.strip() else []
                if retry_raw.strip() and not _needs_format_retry(
                    retry_steps, kg, quality_filter.min_steps, raw_output=retry_raw,
                    max_steps=quality_filter.max_steps,
                ):
                    raw, steps, retry_succeeded = retry_raw, retry_steps, True
            except Exception as exc:  # noqa: BLE001
                retry_error = f"{type(exc).__name__}: {exc}"

        citation_errors = _parsed_contract_errors(raw, kg)
        hard_reject = (
            "citation_contract:" + "|".join(citation_errors) if citation_errors else ""
        )
        final_answer = extract_final_answer(raw) or ""
        source_metadata = source.get("metadata") or {}
        gold = str(source_metadata.get("gold_answer") or "")
        answer_score = answer_match_score(final_answer, gold) if gold else 0.0
        decision = quality_filter.assess_quality(
            steps=steps, answer_score=answer_score, hard_reject_reason=hard_reject
        )
        triple_rate = sum(bool(step.cited_triples) for step in steps) / max(1, len(steps))
        return {
            "_status": "ok",
            "qid": qid,
            "question": source.get("question"),
            "answer": final_answer,
            "dataset": source.get("dataset"),
            "steps": [_step_dict(step) for step in steps],
            "kg_subgraph": [list(triple) if isinstance(triple, tuple) else triple for triple in kg],
            "retrieved_passages": passages,
            "teacher_output": raw,
            "teacher_model": args.teacher,
            "accepted": False,
            "metadata": {
                "gold_answer": gold,
                "answer_score": answer_score,
                "n_triples_teacher": len(kg),
                "format_retried": retry_attempted,
                "retry_succeeded": retry_succeeded,
                "retry_error": retry_error,
                "citation_contract_errors": citation_errors,
                "quality_pass": decision.accepted,
                "quality_reject_reason": "" if decision.accepted else decision.reason,
                "kg_bucket": quality_filter._bucket_for(triple_rate),
                "triple_rate": triple_rate,
                "api_calls": api_calls,
                "fixed_input_source_teacher": source.get("teacher_model"),
                "fixed_input_source_quality_pass": source_metadata.get("quality_pass"),
                "teacher_thinking": args.thinking,
                "teacher_temperature": args.temperature,
                "sample_seed": args.seed,
            },
        }

    results: List[Dict[str, Any] | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_one, row): index for index, row in enumerate(rows)}
        for completed, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:  # noqa: BLE001
                results[index] = {
                    "_status": "worker_error", "qid": qids[index],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if completed % 10 == 0:
                logger.info("Teacher swap progress: %d/%d", completed, len(rows))

    successful = [row for row in results if row and row.get("_status") == "ok"]
    failures = [row for row in results if not row or row.get("_status") != "ok"]
    with output_path.open("w", encoding="utf-8") as fh:
        for row in successful:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with failure_path.open("w", encoding="utf-8") as fh:
        for row in failures:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    dump_manifest(
        output_dir / "run",
        extra={
            "experiment": "fixed_input_teacher_swap",
            "teacher_model": args.teacher,
            "thinking": args.thinking,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "max_workers": args.max_workers,
            "timeout": args.timeout,
            "seed": args.seed,
            "source_files": [
                {"path": str(path), "records": len(_read(path)), "md5": _md5(path)}
                for path in source_paths
            ],
            "submitted": len(rows),
            "successful": len(successful),
            "failed": len(failures),
            "output": str(output_path),
            "failures": str(failure_path),
        },
    )
    print(json.dumps({
        "submitted": len(rows), "successful": len(successful), "failed": len(failures),
        "output": str(output_path), "failures": str(failure_path),
    }, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
