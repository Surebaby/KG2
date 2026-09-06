#!/usr/bin/env python
"""Freeze family-disjoint train/dev/confirmation splits for planner supervision."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_key
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.build_query_planner_supervision import _norm


SPLIT_SCHEMA_VERSION = "query-planner-family-split-1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _replace_token_phrase(tokens: list[str], phrase: Sequence[str]) -> list[str]:
    if not phrase:
        return tokens
    result: list[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index : index + len(phrase)] == list(phrase):
            result.append("<anchor>")
            index += len(phrase)
        else:
            result.append(tokens[index])
            index += 1
    return result


def family_signature(record: Mapping[str, Any]) -> str:
    """Answer-free structural/template family used only for split grouping."""
    target = record.get("target") or {}
    if record.get("target_type") == "relation_graph":
        question_tokens = _norm(record.get("question") or "").split()
        anchors = sorted(
            (_norm(anchor).split() for anchor in target.get("anchors") or []),
            key=len,
            reverse=True,
        )
        for anchor_tokens in anchors:
            question_tokens = _replace_token_phrase(question_tokens, anchor_tokens)
        question_skeleton = " ".join(
            "<num>" if token.isdigit() else token for token in question_tokens
        )
        steps = target.get("steps") or []
        structure = [
            {
                "pid": step.get("pid"),
                "dependency": (step.get("dependencies") or ["root"])[0],
            }
            for step in steps
        ]
        payload = {
            "dataset": record.get("dataset"),
            "target_type": "relation_graph",
            "question_skeleton": question_skeleton,
            "structure": structure,
        }
    else:
        operator_skeletons = []
        for step in target.get("steps") or []:
            raw_template = str(step.get("subquery_template") or "")
            if ">>" in raw_template:
                raw_left, _, raw_right = raw_template.partition(">>")
                left_has_reference = bool(re.search(r"#\d+", raw_left))
                right = _norm(re.sub(r"#\d+", " REFERENCE ", raw_right))
                template = f"{'<ref>' if left_has_reference else '<anchor>'} >> {right.strip()}"
            else:
                template = _norm(re.sub(r"#\d+", " REFERENCE ", raw_template))
            template = re.sub(r"\b\d+\b", "<num>", template)
            operator_skeletons.append(template)
        payload = {
            "dataset": record.get("dataset"),
            "target_type": "subquery_graph",
            "operators": operator_skeletons,
            "dependencies": [step.get("dependencies") or [] for step in target.get("steps") or []],
        }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def assign_family(family: str, *, seed: int, dev_bp: int, confirmation_bp: int) -> str:
    bucket = int(hashlib.sha256(f"{seed}\0{family}".encode()).hexdigest()[:16], 16) % 10_000
    if bucket < confirmation_bp:
        return "confirmation"
    if bucket < confirmation_bp + dev_bp:
        return "dev"
    return "train"


def _seen_keys(paths: Sequence[Path], records: Sequence[Mapping[str, Any]]) -> set[str]:
    qid_lookup: Dict[str, set[str]] = defaultdict(set)
    for record in records:
        qid_lookup[str(record.get("qid"))].add(str(record.get("question_key")))
    seen: set[str] = set()
    for path in paths:
        for row in _read_jsonl(path):
            qid = str(row.get("source_id") or row.get("qid") or row.get("id") or "")
            dataset = str(row.get("dataset") or "")
            if not qid:
                raise ValueError(f"seen cohort row lacks source_id/qid/id: {row}")
            if dataset in {"2wikimultihopqa", "musique"}:
                seen.add(question_key(dataset, qid))
            else:
                seen.update(qid_lookup.get(qid, set()))
    return seen


def compute_assignments(
    records: Sequence[Mapping[str, Any]],
    *,
    seen_keys: set[str],
    seed: int,
    dev_bp: int,
    confirmation_bp: int,
) -> tuple[Dict[str, str], Dict[str, str]]:
    assignments: Dict[str, str] = {}
    family_hashes: Dict[str, str] = {}
    for record in records:
        key = str(record["question_key"])
        family = family_signature(record)
        family_hashes[key] = hashlib.sha256(family.encode()).hexdigest()
    seen_families = {
        family_hashes[key] for key in seen_keys if key in family_hashes
    }
    for record in records:
        key = str(record["question_key"])
        family = family_signature(record)
        assignments[key] = (
            "seen_diagnostics" if key in seen_keys else
            "train" if family_hashes[key] in seen_families else
            assign_family(family, seed=seed, dev_bp=dev_bp, confirmation_bp=confirmation_bp)
        )
    return assignments, family_hashes


def summarize(
    records: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
    family_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    counts, family_sets = Counter(), defaultdict(set)
    train_pids: set[str] = set()
    split_pids: Dict[str, set[str]] = defaultdict(set)
    for record in records:
        key, dataset = str(record["question_key"]), str(record["dataset"])
        split = assignments[key]
        counts[f"split::{split}"] += 1
        counts[f"split_dataset::{split}::{dataset}"] += 1
        family_sets[(split, dataset)].add(family_hashes[key])
        for step in (record.get("target") or {}).get("steps") or []:
            pid = step.get("pid")
            if pid:
                split_pids[split].add(str(pid))
                if split == "train":
                    train_pids.add(str(pid))
    for (split, dataset), families in family_sets.items():
        counts[f"families::{split}::{dataset}"] = len(families)
    leakage = {}
    for dataset in {str(record["dataset"]) for record in records}:
        for left, right in (("train", "dev"), ("train", "confirmation"), ("dev", "confirmation")):
            overlap = family_sets[(left, dataset)].intersection(family_sets[(right, dataset)])
            leakage[f"{dataset}::{left}_vs_{right}"] = len(overlap)
    return {
        "counts": dict(counts),
        "family_overlap": leakage,
        "pid_vocabulary": {
            "train": sorted(train_pids),
            "dev_oov": sorted(split_pids["dev"] - train_pids),
            "confirmation_oov": sorted(split_pids["confirmation"] - train_pids),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervision", required=True)
    parser.add_argument("--seen_cohort", action="append", default=[])
    parser.add_argument("--output_dir")
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--dev_bp", type=int, default=200)
    parser.add_argument("--confirmation_bp", type=int, default=200)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.dev_bp < 1 or args.confirmation_bp < 1 or args.dev_bp + args.confirmation_bp >= 10_000:
        raise SystemExit("invalid basis-point split rates")
    if not args.dry_run and not args.output_dir:
        raise SystemExit("--output_dir is required unless --dry_run is set")

    supervision = Path(args.supervision).resolve()
    seen_paths = [Path(path).resolve() for path in args.seen_cohort]
    records = list(_read_jsonl(supervision))
    seen = _seen_keys(seen_paths, records)
    assignments, family_hashes = compute_assignments(
        records,
        seen_keys=seen,
        seed=args.seed,
        dev_bp=args.dev_bp,
        confirmation_bp=args.confirmation_bp,
    )
    summary = summarize(records, assignments, family_hashes)
    summary["seen_keys_requested"] = len(seen)
    summary["seen_keys_present"] = sum(key in assignments for key in seen)
    seen_family_hashes = {family_hashes[key] for key in seen if key in family_hashes}
    summary["seen_family_forced_train"] = sum(
        assignments[key] == "train" and family_hashes[key] in seen_family_hashes
        for key in assignments
    )
    if any(summary["family_overlap"].values()):
        raise SystemExit(f"family overlap detected: {summary['family_overlap']}")
    if summary["pid_vocabulary"]["dev_oov"] or summary["pid_vocabulary"]["confirmation_oov"]:
        raise SystemExit(f"PID OOV detected: {summary['pid_vocabulary']}")
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing path: {output_dir}")
    output_dir.mkdir(parents=True)
    handles = {
        split: (output_dir / f"{split}.jsonl").open("x", encoding="utf-8")
        for split in ("train", "dev", "confirmation", "seen_diagnostics")
    }
    assignments_path = output_dir / "assignments.jsonl"
    with assignments_path.open("x", encoding="utf-8") as assignment_fh:
        try:
            for record in records:
                key = str(record["question_key"])
                split = assignments[key]
                handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                assignment_fh.write(json.dumps({
                    "question_key": key,
                    "dataset": record["dataset"],
                    "qid": record["qid"],
                    "split": split,
                    "family_sha256": family_hashes[key],
                }, ensure_ascii=False) + "\n")
        finally:
            for handle in handles.values():
                handle.close()

    outputs = {
        path.name: {"path": str(path), "sha256": _sha256(path)}
        for path in sorted(output_dir.glob("*.jsonl"))
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "status": "FROZEN_NOT_TRAINED_NOT_EVALUATED",
        "schema_version": SPLIT_SCHEMA_VERSION,
        "protocol": {
            "seed": args.seed,
            "dev_basis_points": args.dev_bp,
            "confirmation_basis_points": args.confirmation_bp,
            "assignment": "SHA256(seed + answer-free family signature) mod 10000",
            "confirmation_content_inspected": False,
        },
        "inputs": {
            "supervision": {"path": str(supervision), "sha256": _sha256(supervision)},
            "seen_cohorts": [
                {"path": str(path), "sha256": _sha256(path)} for path in seen_paths
            ],
        },
        **summary,
        "outputs": outputs,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(output_dir / "run", extra=report)
    print(json.dumps({"status": report["status"], **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
