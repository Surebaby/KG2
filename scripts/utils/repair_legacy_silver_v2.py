#!/usr/bin/env python
"""Create a strict, derived legacy-silver repair without Teacher regeneration.

The source JSONL is immutable.  This script reparses the preserved Teacher
output against a student-visible top-K KG, reannotates with the current
conservative PRMAnnotator, hard-rejects citation-contract violations, and then
applies deterministic post-hoc quotas per dataset.  It never unions filtered
citations back into the visible KG and never overwrites the source labels.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import unicodedata

from kgproweight.data.parsers import ParsedStep, parse_steps
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.kg.kg_filter import filter_and_rank_triples
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path
from kgproweight.training.phase1_distill import (
    StratifiedSilverFilter,
    _quota_selection_counts,
    _selection_rank,
    answer_match_score,
)
from kgproweight.utils.logging import configure_logging, dump_manifest, get_logger


Triple = Tuple[str, str, str]
PROTOCOL_VERSION = "legacy_repaired_v2.strict_top12_current_prm"

configure_logging("INFO")
logger = get_logger(__name__)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _triple(value: Sequence[Any]) -> Triple:
    return tuple(str(part).strip() for part in value)  # type: ignore[return-value]


def _key(value: Sequence[Any]) -> Triple:
    return tuple(
        " ".join(unicodedata.normalize("NFC", str(part)).casefold().split())
        for part in value
    )  # type: ignore[return-value]


def _contract_errors(steps: Iterable[ParsedStep]) -> List[str]:
    errors: List[str] = []
    for step in steps:
        errors.extend(
            f"step_{step.index}:{error}" for error in step.citation_contract_errors
        )
    return list(dict.fromkeys(errors))


def _sample_selected(qid: str, modulo: int, remainder: int) -> bool:
    if modulo <= 0:
        return True
    digest = int(hashlib.sha256(qid.encode("utf-8")).hexdigest()[:16], 16)
    return digest % modulo == remainder


def _write_jsonl_row(fh, row: Dict[str, Any]) -> None:
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _finalize_per_dataset(
    candidate_path: Path,
    output_path: Path,
    accept_filter: StratifiedSilverFilter,
    seed: int,
    selection_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    eligible: Dict[str, Dict[str, List[Tuple[str, int]]]] = defaultdict(
        lambda: {"kg_rich": [], "kg_medium": [], "kg_sparse": []}
    )
    dataset_totals: Counter[str] = Counter()
    quality_rejected: Counter[str] = Counter()
    quality_reject_reasons: Counter[str] = Counter()
    with candidate_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            dataset = str(row.get("dataset") or "UNKNOWN")
            dataset_totals[dataset] += 1
            metadata = row.get("metadata") or {}
            if not metadata.get("quality_pass"):
                quality_rejected[dataset] += 1
                quality_reject_reasons[
                    str(metadata.get("quality_reject_reason") or "UNKNOWN")
                ] += 1
                continue
            bucket = str(metadata.get("kg_bucket") or "")
            if bucket not in eligible[dataset]:
                raise ValueError(f"{candidate_path}:{lineno}: invalid bucket={bucket!r}")
            eligible[dataset][bucket].append(
                (_selection_rank(row, seed, lineno), lineno)
            )

    selected_lines: set[int] = set()
    per_dataset: Dict[str, Dict[str, Any]] = {}
    for dataset in sorted(dataset_totals):
        groups = eligible[dataset]
        selected_counts = _quota_selection_counts(
            len(groups["kg_rich"]),
            len(groups["kg_medium"]),
            len(groups["kg_sparse"]),
            accept_filter.medium_quota,
            accept_filter.sparse_quota,
        )
        for bucket, rows in groups.items():
            rows.sort()
            selected_lines.update(
                lineno for _, lineno in rows[: selected_counts[bucket]]
            )
        per_dataset[dataset] = {
            "total": dataset_totals[dataset],
            "quality_rejected": quality_rejected[dataset],
            "quality_bucket_counts": {
                bucket: len(rows) for bucket, rows in groups.items()
            },
            "accepted_bucket_counts": selected_counts,
            "accepted": sum(selected_counts.values()),
        }

    source_new_acceptance: Counter[str] = Counter()
    selection_reasons: Counter[str] = Counter()
    with candidate_path.open(encoding="utf-8") as src, output_path.open(
        "x", encoding="utf-8"
    ) as dst:
        for lineno, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.setdefault("metadata", {})
            if selection_metadata:
                metadata.update(selection_metadata)
            quality_pass = bool(metadata.get("quality_pass"))
            selected = quality_pass and lineno in selected_lines
            row["accepted"] = selected
            metadata["selection_pass"] = selected
            if selected:
                metadata["selection_reason"] = "selected"
                metadata["reject_reason"] = ""
            elif quality_pass:
                reason = f"{metadata.get('kg_bucket', 'unknown')}_quota_full"
                metadata["selection_reason"] = reason
                metadata["reject_reason"] = reason
            else:
                metadata["selection_reason"] = "quality_rejected"
                metadata["reject_reason"] = (
                    metadata.get("quality_reject_reason") or "quality_rejected"
                )
            source_new_acceptance[
                f"source_{'accepted' if metadata.get('legacy_source_accepted') else 'rejected'}"
                f"->new_{'accepted' if selected else 'rejected'}"
            ] += 1
            selection_reasons[str(metadata["selection_reason"])] += 1
            _write_jsonl_row(dst, row)

    return {
        "total": sum(dataset_totals.values()),
        "accepted": len(selected_lines),
        "per_dataset": per_dataset,
        "quality_reject_reasons": dict(quality_reject_reasons),
        "selection_reasons": dict(selection_reasons),
        "source_to_new_acceptance": dict(source_new_acceptance),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--max_kg_triples", type=int, default=12)
    parser.add_argument("--min_kg_keep", type=int, default=5)
    parser.add_argument("--max_passages", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample_modulo",
        type=int,
        default=0,
        help="Hash-sample QIDs where sha256(qid) %% modulo == remainder; 0=all.",
    )
    parser.add_argument("--sample_remainder", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    candidate_path = output_path.with_name(
        f"{output_path.stem}.candidates{output_path.suffix}"
    )
    report_path = output_path.with_name(f"{output_path.stem}.report.json")
    run_dir = output_path.with_name(f"{output_path.stem}_run")
    for path in (output_path, candidate_path, report_path, run_dir):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing path: {path}")
    if args.max_kg_triples != 12 or args.min_kg_keep != 5:
        raise SystemExit("legacy_repaired_v2 protocol requires max=12 and min=5")
    if args.max_passages != 15:
        raise SystemExit("legacy_repaired_v2 protocol requires max_passages=15")
    if args.sample_modulo < 0 or not (
        args.sample_modulo == 0
        or 0 <= args.sample_remainder < args.sample_modulo
    ):
        raise SystemExit("invalid sample modulo/remainder")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_md5 = _md5(input_path)
    annotator = PRMAnnotator(
        entity_linker=EntityLinker(
            cache_path=resolve_entity_cache_path(), use_genre=False
        ),
        verbose=False,
    )
    accept_filter = StratifiedSilverFilter()

    counts: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    contract_errors: Counter[str] = Counter()
    label_transitions: Counter[str] = Counter()
    kg_counts: List[int] = []
    source_kg_counts: List[int] = []
    passages_before: List[int] = []
    passages_after: List[int] = []

    with input_path.open(encoding="utf-8") as src, candidate_path.open(
        "x", encoding="utf-8"
    ) as dst:
        for source_lineno, line in enumerate(src, 1):
            if not line.strip():
                continue
            counts["source_records_seen"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counts["source_json_errors"] += 1
                continue
            qid = str(row.get("qid") or row.get("id") or "")
            if not _sample_selected(qid, args.sample_modulo, args.sample_remainder):
                continue
            counts["selected_records"] += 1
            dataset = str(row.get("dataset") or "UNKNOWN")
            datasets[dataset] += 1
            counts[
                "source_accepted" if row.get("accepted", True) else "source_rejected"
            ] += 1
            question = str(row.get("question") or "")
            source_steps = row.get("steps") or []
            source_labels = [
                float(step.get("label") or 0.0)
                for step in source_steps
                if isinstance(step, dict)
            ]
            raw_kg = [
                _triple(value)
                for value in (row.get("kg_subgraph") or row.get("kg_triples") or [])
                if isinstance(value, (list, tuple)) and len(value) == 3
            ]
            visible_kg = list(
                filter_and_rank_triples(
                    raw_kg,
                    question=question,
                    max_keep=args.max_kg_triples,
                    min_keep=args.min_kg_keep,
                )
            )
            raw_output = str(row.get("teacher_output") or "")
            raw_parsed = parse_steps(raw_output, known_kg=raw_kg)
            visible_parsed = parse_steps(raw_output, known_kg=visible_kg)
            raw_errors = _contract_errors(raw_parsed)
            visible_errors = _contract_errors(visible_parsed)
            visible_keys = {_key(triple) for triple in visible_kg}
            out_of_view = sorted(
                {
                    triple
                    for step in raw_parsed
                    for triple in step.cited_triples
                    if _key(triple) not in visible_keys
                }
            )

            hard_reasons: List[str] = []
            if raw_errors:
                hard_reasons.append("legacy_citation_contract")
            if visible_errors:
                hard_reasons.append("student_visible_citation_contract")
            if out_of_view:
                hard_reasons.append("citation_not_in_student_visible_kg")
            if not raw_output.strip():
                hard_reasons.append("missing_teacher_output")
            hard_reasons = list(dict.fromkeys(hard_reasons))
            for reason in hard_reasons:
                contract_errors[reason] += 1

            labels = annotator.annotate_trajectory(visible_parsed, visible_kg)
            new_steps: List[Dict[str, Any]] = []
            for position, (step, label) in enumerate(zip(visible_parsed, labels)):
                legacy_label = (
                    source_labels[position] if position < len(source_labels) else None
                )
                old_bucket = (
                    "missing"
                    if legacy_label is None
                    else "neg"
                    if legacy_label < 0
                    else "zero"
                    if legacy_label == 0
                    else "frac"
                    if legacy_label < 1
                    else "one"
                )
                new_bucket = (
                    "neg" if label < 0 else "zero" if label == 0 else "frac" if label < 1 else "one"
                )
                label_transitions[f"{old_bucket}->{new_bucket}"] += 1
                new_steps.append({
                    "index": step.index,
                    "text": step.raw_text,
                    "label": float(label),
                    "legacy_label": legacy_label,
                    "cited_triples": [list(triple) for triple in step.cited_triples],
                })
            counts["source_steps"] += len(source_steps)
            counts["parsed_steps"] += len(new_steps)
            if len(new_steps) != len(source_steps):
                counts["step_count_changed"] += 1

            metadata = dict(row.get("metadata") or {})
            gold = str(metadata.get("gold_answer") or "")
            final_answer = str(row.get("answer") or "")
            answer_score = answer_match_score(final_answer, gold) if gold else 0.0

            class _Step:
                __slots__ = ("cited_triples",)

                def __init__(self, cited_triples):
                    self.cited_triples = cited_triples

            decision = accept_filter.assess_quality(
                steps=[_Step(step["cited_triples"]) for step in new_steps],
                answer_score=answer_score,
                hard_reject_reason="|".join(hard_reasons),
            )
            passages = list(row.get("retrieved_passages") or [])
            kept_passages = passages[: args.max_passages]
            source_kg_counts.append(len(raw_kg))
            kg_counts.append(len(visible_kg))
            passages_before.append(len(passages))
            passages_after.append(len(kept_passages))

            metadata.update({
                "answer_score": answer_score,
                "bucket": decision.bucket,
                "kg_bucket": accept_filter._bucket_for(decision.triple_rate),
                "triple_rate": decision.triple_rate,
                "quality_pass": decision.accepted,
                "quality_reject_reason": "" if decision.accepted else decision.reason,
                "selection_pass": False,
                "selection_reason": "pending_posthoc_selection",
                "reject_reason": "" if decision.accepted else decision.reason,
                "legacy_repair_version": PROTOCOL_VERSION,
                "legacy_source_md5": source_md5,
                "legacy_source_line": source_lineno,
                "legacy_source_accepted": bool(row.get("accepted", True)),
                "legacy_source_kg_count": len(raw_kg),
                "student_visible_kg_count": len(visible_kg),
                "passages_before": len(passages),
                "passages_after": len(kept_passages),
                "citation_contract_errors_raw_view": raw_errors,
                "citation_contract_errors_student_view": visible_errors,
                "citations_not_in_student_visible_kg": [
                    list(triple) for triple in out_of_view
                ],
                "teacher_student_kg_view_aligned": False,
                "teacher_student_kg_view_note": (
                    "Legacy Teacher saw the source KG; the repaired student view is strict top-12. "
                    "This historical mismatch cannot be removed without Teacher regeneration."
                ),
            })
            out = {
                "qid": qid,
                "question": question,
                "answer": final_answer,
                "dataset": dataset,
                "steps": new_steps,
                "kg_subgraph": [list(triple) for triple in visible_kg],
                "retrieved_passages": kept_passages,
                "accepted": False,
                "metadata": metadata,
                "teacher_output": row.get("teacher_output"),
                "teacher_model": row.get("teacher_model"),
            }
            _write_jsonl_row(dst, out)
            if counts["selected_records"] % args.log_interval == 0:
                logger.info("processed %d selected records", counts["selected_records"])

    selection = _finalize_per_dataset(
        candidate_path, output_path, accept_filter, args.seed
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "status": "DERIVED_DATA_NOT_TRAINED",
        "protocol_version": PROTOCOL_VERSION,
        "source": {
            "path": str(input_path),
            "md5": source_md5,
            "read_only": True,
        },
        "config": {
            "max_kg_triples": args.max_kg_triples,
            "min_kg_keep": args.min_kg_keep,
            "max_passages": args.max_passages,
            "seed": args.seed,
            "sample_modulo": args.sample_modulo,
            "sample_remainder": args.sample_remainder,
            "quota_scope": "per_dataset_posthoc",
            "rich_triple_rate": accept_filter.rich_triple_rate,
            "medium_triple_rate": accept_filter.medium_triple_rate,
            "medium_quota": accept_filter.medium_quota,
            "sparse_quota": accept_filter.sparse_quota,
            "continuous_rule_v1_1_used": False,
        },
        "counts": dict(counts),
        "datasets": dict(datasets),
        "contract_reject_trajectory_counts": dict(contract_errors),
        "label_transitions": dict(label_transitions),
        "kg": {
            "source_mean": sum(source_kg_counts) / max(1, len(source_kg_counts)),
            "student_visible_mean": sum(kg_counts) / max(1, len(kg_counts)),
            "student_visible_empty": sum(value == 0 for value in kg_counts),
        },
        "passages": {
            "source_mean": sum(passages_before) / max(1, len(passages_before)),
            "derived_mean": sum(passages_after) / max(1, len(passages_after)),
        },
        "selection": selection,
        "outputs": {
            "candidate": str(candidate_path),
            "selected": str(output_path),
        },
        "scientific_limit": (
            "No Teacher regeneration: trajectory content and the historical Teacher/student KG-view "
            "mismatch remain; this dataset can test downstream recovery but cannot validate improved "
            "silver generation."
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(run_dir, extra={
        "experiment_id": args.experiment_id,
        "phase": "legacy_silver_repair_v2",
        "protocol_version": PROTOCOL_VERSION,
        "source": str(input_path),
        "source_md5": source_md5,
        "output": str(output_path),
        "candidate": str(candidate_path),
        "report": str(report_path),
        "total": selection["total"],
        "accepted": selection["accepted"],
        "sample_modulo": args.sample_modulo,
        "sample_remainder": args.sample_remainder,
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
