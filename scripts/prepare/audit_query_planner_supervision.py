#!/usr/bin/env python
"""Audit planner supervision structure and answer leakage against train sources."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from kgproweight.kg.question_kg import question_key, question_sha256
from scripts.prepare.build_query_planner_supervision import (
    SCHEMA_VERSION,
    _FORBIDDEN_KEYS,
    _contains_token_phrase,
    _norm as _unicode_norm,
    _string_values,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


def _norm(value: object) -> str:
    return _unicode_norm(value)


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _target_text(value: Any) -> str:
    parts: List[str] = []
    if isinstance(value, Mapping):
        for child in value.values():
            parts.append(_target_text(child))
    elif isinstance(value, list):
        for child in value:
            parts.append(_target_text(child))
    elif isinstance(value, str):
        parts.append(value)
    return _norm(" ".join(parts))


def validate_record(record: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    dataset, qid = str(record.get("dataset") or ""), str(record.get("qid") or "")
    question = str(record.get("question") or "")
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version")
    try:
        if record.get("question_key") != question_key(dataset, qid):
            errors.append("question_key")
    except ValueError:
        errors.append("question_key")
    if record.get("question_sha256") != question_sha256(question):
        errors.append("question_sha256")
    forbidden = _FORBIDDEN_KEYS.intersection(_walk_keys(record))
    if forbidden:
        errors.append(f"forbidden_keys:{','.join(sorted(forbidden))}")
    target, target_type = record.get("target") or {}, record.get("target_type")
    steps = list(target.get("steps") or [])
    if not steps:
        errors.append("empty_steps")
        return errors
    if target_type == "relation_graph":
        if set(target) != {"anchors", "steps"}:
            errors.append("invalid_relation_target_schema")
        if not isinstance(target.get("anchors"), list) or not all(
            isinstance(anchor, str) for anchor in target.get("anchors") or []
        ):
            errors.append("invalid_anchors")
        if (record.get("provenance") or {}).get("has_degenerate_source_anchor"):
            errors.append("invalid_anchor")
        for anchor in target.get("anchors") or []:
            if not _norm(anchor):
                errors.append("invalid_anchor")
    for index, step in enumerate(steps, start=1):
        if step.get("step") != index:
            errors.append("nonsequential_step")
        if target_type == "relation_graph":
            required = {"step", "subject", "relation_label", "pid", "output_slot", "dependencies"}
            if set(step) != required:
                errors.append("invalid_relation_step_schema")
            if not isinstance(step.get("subject"), str) or not isinstance(step.get("relation_label"), str):
                errors.append("invalid_relation_step_value")
            if step.get("output_slot") != f"hop_{index}":
                errors.append("invalid_output_slot")
            pid = step.get("pid")
            if not isinstance(pid, str) or not re.fullmatch(r"P\d+", pid):
                errors.append("unmapped_pid")
            for dependency in step.get("dependencies") or []:
                match = re.fullmatch(r"hop_(\d+)", str(dependency))
                if not match or int(match.group(1)) >= index:
                    errors.append("invalid_dependency")
        elif target_type == "subquery_graph":
            if set(target) != {"steps"}:
                errors.append("invalid_subquery_target_schema")
            required = {"step", "subquery_template", "dependencies", "output_slot"}
            if set(step) != required:
                errors.append("invalid_subquery_step_schema")
            if step.get("output_slot") != f"step_{index}" or not step.get("subquery_template"):
                errors.append("invalid_subquery_step")
            for dependency in step.get("dependencies") or []:
                match = re.fullmatch(r"step_(\d+)", str(dependency))
                if not match or int(match.group(1)) >= index:
                    errors.append("invalid_dependency")
        else:
            errors.append("unknown_target_type")
    return errors


def _source_secrets(dataset: str, row: Mapping[str, Any]) -> List[str]:
    values = list(row.get("golden_answers") or [])
    if dataset == "2wikimultihopqa":
        values.extend(((row.get("metadata") or {}).get("evidences") or {}).get("entity") or [])
    elif dataset == "musique":
        decomposition = (((row.get("metadata") or {}).get("metadata") or {}).get("question_decomposition") or [])
        values.extend(item.get("answer") for item in decomposition)
    return [str(value).strip() for value in values if str(value or "").strip()]


def audit(
    supervision_path: Path,
    source_paths: Mapping[str, Path],
    *,
    expected_keys: set[str] | None = None,
    allowed_missing: set[str] | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    allowed_missing = allowed_missing or set()
    counts = Counter()
    seen: set[str] = set()
    targets: Dict[str, Tuple[List[str], str]] = {}
    structural_examples: List[Dict[str, str]] = []
    for record in _read_jsonl(supervision_path):
        counts["records"] += 1
        counts[f"dataset::{record.get('dataset')}"] += 1
        key = str(record.get("question_key") or "")
        if key in seen:
            counts["duplicate_question_key"] += 1
        seen.add(key)
        errors = validate_record(record)
        if errors:
            counts["invalid_records"] += 1
            for error in set(errors):
                counts[f"error::{error}"] += 1
            if len(structural_examples) < 100:
                structural_examples.append({"question_key": key, "errors": ",".join(errors)})
        counts[f"target_type::{record.get('target_type')}"] += 1
        counts["steps"] += len((record.get("target") or {}).get("steps") or [])
        targets[key] = (
            list(_string_values(record.get("target") or {})),
            _norm(record.get("question") or ""),
        )

    leakage: List[Dict[str, str]] = []
    source_keys: set[str] = set()
    for dataset, path in source_paths.items():
        for row in _read_jsonl(path):
            key = question_key(dataset, str(row["id"]))
            if expected_keys is not None and key not in expected_keys:
                continue
            source_keys.add(key)
            target = targets.get(key)
            if target is None:
                if key in allowed_missing:
                    counts["source_explicitly_excluded"] += 1
                else:
                    counts["source_missing_from_supervision"] += 1
                continue
            target_values, question_text = target
            for secret in dict.fromkeys(_source_secrets(dataset, row)):
                normalized = _norm(secret)
                if len(normalized) < 3 or normalized in question_text:
                    continue
                if any(_contains_token_phrase(value, secret) for value in target_values):
                    counts["answer_or_tail_leakage"] += 1
                    if len(leakage) < 1000:
                        leakage.append(
                            {"question_key": key, "dataset": dataset, "leaked_value": secret}
                        )
    counts["supervision_missing_from_sources"] = len(seen - source_keys)
    hard_failures = (
        counts["duplicate_question_key"]
        + counts["invalid_records"]
        + counts["source_missing_from_supervision"]
        + counts["supervision_missing_from_sources"]
        + counts["answer_or_tail_leakage"]
    )
    result = {
        "status": "PASS" if hard_failures == 0 else "FAIL",
        "counts": dict(counts),
        "hard_failure_count": hard_failures,
        "structural_error_examples": structural_examples,
    }
    return result, leakage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervision", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--selection")
    parser.add_argument("--allowed_missing")
    args = parser.parse_args()
    supervision = Path(args.supervision).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing path: {output_dir}")
    output_dir.mkdir(parents=True)
    root = Path(args.data_root).resolve()
    sources = {
        dataset: root / dataset / "train.jsonl"
        for dataset in ("2wikimultihopqa", "musique")
    }
    expected_keys = None
    if args.selection:
        expected_keys = set()
        for row in _read_jsonl(Path(args.selection).resolve()):
            if str(row.get("dataset")) not in sources:
                continue
            source_id = row.get("source_id", row.get("qid"))
            if source_id is None:
                raise ValueError(f"selection row lacks source_id/qid: {row}")
            expected_keys.add(question_key(str(row["dataset"]), str(source_id)))
    allowed_missing = set()
    if args.allowed_missing:
        allowed_missing = {
            question_key(str(row["dataset"]), str(row["qid"]))
            for row in _read_jsonl(Path(args.allowed_missing).resolve())
        }
    result, leakage = audit(
        supervision, sources, expected_keys=expected_keys, allowed_missing=allowed_missing
    )
    leakage_path = output_dir / "leakage_examples.jsonl"
    with leakage_path.open("w", encoding="utf-8") as fh:
        for row in leakage:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "scope": "train-only planner supervision structural and answer/tail leakage audit",
        "inputs": {
            "supervision": {"path": str(supervision), "sha256": _sha256(supervision)},
            "sources": {
                dataset: {"path": str(path), "sha256": _sha256(path)}
                for dataset, path in sources.items()
            },
        },
        **result,
        "leakage_examples": {"path": str(leakage_path), "sha256": _sha256(leakage_path)},
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"]}, indent=2))


if __name__ == "__main__":
    main()
