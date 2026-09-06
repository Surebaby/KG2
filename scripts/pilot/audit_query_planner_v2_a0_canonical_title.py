#!/usr/bin/env python
"""Audit canonical title resolution on seen A0 diagnostics.

This retrospective engineering audit preserves both the frozen v1
confirmation and the failed alias-exact A0 run. Gold supporting titles are
used only for scoring after runtime surfaces have been produced.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.entity_linker import passage_title
from kgproweight.kg.wikipedia_title_resolver import complete_question_surface_title
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _raw_records(path: str | Path, qids: set[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        qid = str(row.get("id") or row.get("qid") or "")
        if qid in qids:
            records[qid] = row
    return records


def _first_supporting_title(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    supporting = metadata.get("supporting_facts") or {}
    titles = supporting.get("title") or []
    return str(titles[0]) if titles else ""


def _targeted_corpus_scan(
    corpus_path: str | Path,
    target_titles: Sequence[str],
) -> tuple[dict[str, list[str]], int]:
    wanted = {_norm(title) for title in target_titles if _norm(title)}
    found: dict[str, list[str]] = {key: [] for key in wanted}
    scanned = 0
    with Path(corpus_path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            scanned += 1
            document = json.loads(line)
            title = passage_title(document)
            key = _norm(title)
            if key in found and title not in found[key]:
                found[key].append(title)
    return found, scanned


def _rrf_hit(detail: Mapping[str, Any], target_title: str, source: str, topk: int) -> bool:
    retrieved = detail[f"{source}_at_{topk}"]["retrieved_titles"]
    return _norm(target_title) in {_norm(title) for title in retrieved}


def _mean(flags: Sequence[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--raw_2wiki", required=True)
    parser.add_argument("--failed_a0_details", required=True)
    parser.add_argument("--failed_a0_report", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir, experiment_id = prepare_new_run_dir(
        args.output_dir,
        extra={
            "phase": "planner_v2_a0_canonical_title_seen_engineering",
            "cohort": artifact_identity(args.cohort),
            "failed_a0_report": artifact_identity(args.failed_a0_report),
        },
    )
    cohort = list(_read_jsonl(args.cohort))
    failed_details = {
        row["question_key"]: row for row in _read_jsonl(args.failed_a0_details)
    }
    raw = _raw_records(args.raw_2wiki, {str(row["qid"]) for row in cohort})
    if len(raw) != len(cohort):
        raise SystemExit(f"raw record coverage {len(raw)}/{len(cohort)}")

    provisional: list[dict[str, Any]] = []
    targets: list[str] = []
    for row in cohort:
        qid = str(row["qid"])
        target = _first_supporting_title(raw[qid])
        if not target:
            raise SystemExit(f"missing supporting title for {qid}")
        predicted = [str(value) for value in row["predicted_anchors"]]
        if len(predicted) != 1:
            raise SystemExit(f"expected one anchor, got {len(predicted)} for {qid}")
        completed = complete_question_surface_title(predicted[0], str(row["question"]))
        targets.append(target)
        provisional.append({
            **row,
            "target_wikipedia_title": target,
            "target_source": "metadata.supporting_facts.title[0] (scoring only)",
            "predicted_surface": predicted[0],
            "completed_question_surface": completed,
            "raw_surface_title_exact": _norm(predicted[0]) == _norm(target),
            "completed_surface_title_exact": _norm(completed) == _norm(target),
        })

    found, scanned = _targeted_corpus_scan(args.corpus_path, targets)
    detail_rows: list[dict[str, Any]] = []
    for row in provisional:
        key = _norm(row["target_wikipedia_title"])
        old = failed_details[row["question_key"]]
        corpus_titles = found.get(key) or []
        detail = {
            **row,
            "target_present_in_full_wiki18": bool(corpus_titles),
            "matched_corpus_titles": corpus_titles,
            "direct_resolution_complete": bool(corpus_titles)
            and bool(row["completed_surface_title_exact"]),
        }
        for source in ("predicted_surface", "gold_alias_oracle"):
            for topk in (5, 10, 20):
                detail[f"corrected_rrf_{source}_at_{topk}"] = _rrf_hit(
                    old, row["target_wikipedia_title"], source, topk
                )
        detail_rows.append(detail)

    metrics = {
        "raw_surface_title_exact": _mean([row["raw_surface_title_exact"] for row in detail_rows]),
        "completed_surface_title_exact": _mean([
            row["completed_surface_title_exact"] for row in detail_rows
        ]),
        "target_present_in_full_wiki18": _mean([
            row["target_present_in_full_wiki18"] for row in detail_rows
        ]),
        "direct_resolution_complete": _mean([
            row["direct_resolution_complete"] for row in detail_rows
        ]),
        "corrected_rrf": {
            source: {
                str(topk): _mean([
                    row[f"corrected_rrf_{source}_at_{topk}"] for row in detail_rows
                ])
                for topk in (5, 10, 20)
            }
            for source in ("predicted_surface", "gold_alias_oracle")
        },
    }
    checks = {
        "completed_surface_title_exact_ge_0_90": metrics["completed_surface_title_exact"] >= 0.90,
        "target_present_in_full_wiki18_ge_0_95": metrics["target_present_in_full_wiki18"] >= 0.95,
        "direct_resolution_complete_ge_0_90": metrics["direct_resolution_complete"] >= 0.90,
    }
    details_path = output_dir / "details.jsonl"
    with details_path.open("x", encoding="utf-8") as fh:
        for row in detail_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "status": "PASS_SEEN_ENGINEERING_ONLY" if all(checks.values()) else "FAIL_STOP",
        "scope": "seen_confirmation_retrospective_engineering_only",
        "n": len(detail_rows),
        "scientific_boundary": {
            "replaces_v1_confirmation": False,
            "replaces_failed_alias_exact_a0": False,
            "independent_confirmation": False,
            "runtime_inputs": ["question", "predicted_anchor"],
            "gold_inputs_used_for_scoring_only": ["metadata.supporting_facts.title[0]"],
            "note": "The target correction followed detection of an alias/title construct mismatch; this is not a preregistered confirmatory result.",
        },
        "metrics": metrics,
        "engineering_checks": {"pass": all(checks.values()), "checks": checks},
        "corpus": {
            "path": str(Path(args.corpus_path).resolve()),
            "lines_scanned": scanned,
            "size_bytes": Path(args.corpus_path).stat().st_size,
        },
        "inputs": {
            "cohort": artifact_identity(args.cohort),
            "raw_2wiki": artifact_identity(args.raw_2wiki),
            "failed_a0_details": artifact_identity(args.failed_a0_details),
            "failed_a0_report": artifact_identity(args.failed_a0_report),
        },
        "details": artifact_identity(details_path),
        "selected_question_key_sha256": hashlib.sha256(
            "\n".join(row["question_key"] for row in detail_rows).encode()
        ).hexdigest(),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(output_dir, status=report["status"], extra=report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
