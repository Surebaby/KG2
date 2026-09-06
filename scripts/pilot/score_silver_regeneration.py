"""Audit regenerated silver and compare matched historical questions.

The automatic verdict covers integrity only. Whether quality improved is left
to the researcher because locking scientific acceptance thresholds after seeing
the data would change the evaluation protocol post hoc.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from kgproweight.kg.kg_filter import filter_and_rank_triples


def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _norm_triple(value: Any) -> Tuple[str, str, str]:
    if isinstance(value, dict):
        parts = (value.get("head", ""), value.get("relation", ""), value.get("tail", ""))
    else:
        parts = list(value)[:3]
    if len(parts) != 3:
        return ("", "", "")
    return tuple(_norm_text(x) for x in parts)  # type: ignore[return-value]


def _read(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def _read_matching(
    path: Path, wanted: set[Tuple[str, str]]
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Stream a large historical silver file and retain paired rows only."""
    matched: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            key = (str(row.get("dataset") or ""), _norm_text(row.get("question")))
            if key in wanted:
                matched[key] = row
    return matched


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pct(a: int, b: int) -> float:
    return 100.0 * a / b if b else float("nan")


def _metrics(
    rows: Iterable[Dict[str, Any]], *, refilter_stored_kg: bool = False
) -> Dict[str, Any]:
    """Compute trajectory metrics against the KG visible to the Teacher.

    Regenerated rows already store the filtered top-12 Teacher KG, so filtering
    them again is both unnecessary and non-idempotent. Historical rows store the
    older/raw view and must be projected through the current filter for a paired
    current-policy comparison.
    """
    rows = list(rows)
    n_steps = n_citing = n_visible_citing = 0
    citations = hallucinations = 0
    n_valid = n_kg = n_all_steps_visible_cite = 0
    labels: List[float] = []
    answer_scores: List[float] = []
    for row in rows:
        raw_kg = [
            tuple(t) if isinstance(t, list) else t
            for t in (row.get("kg_subgraph") or [])
        ]
        visible_kg = raw_kg
        if refilter_stored_kg:
            visible_kg = filter_and_rank_triples(
                raw_kg, question=str(row.get("question") or ""), min_keep=5, max_keep=12
            )
        kgset = {_norm_triple(t) for t in visible_kg}
        steps = list(row.get("steps") or [])
        n_steps += len(steps)
        n_valid += int(3 <= len(steps) <= 7)
        if kgset:
            n_kg += 1
        all_visible = bool(kgset and steps)
        for step in steps:
            cited = list(step.get("cited_triples") or [])
            cited_keys = [_norm_triple(t) for t in cited]
            if cited:
                n_citing += 1
            visible = bool(cited_keys) and all(k in kgset for k in cited_keys)
            n_visible_citing += int(visible)
            all_visible = all_visible and visible
            citations += len(cited_keys)
            hallucinations += sum(k not in kgset for k in cited_keys)
            labels.append(float(step.get("label", 0.0)))
        n_all_steps_visible_cite += int(all_visible)
        answer_scores.append(float((row.get("metadata") or {}).get("answer_score", 0.0)))

    endpoints = {-1.0, 0.0, 1.0}
    reject_reasons = Counter(
        str((r.get("metadata") or {}).get("reject_reason") or "UNKNOWN")
        for r in rows
        if not r.get("accepted")
    )
    return {
        "n_trajectories": len(rows),
        "accepted_rate_pct": _pct(sum(bool(r.get("accepted")) for r in rows), len(rows)),
        "parse_valid_rate_pct": _pct(n_valid, len(rows)),
        "kg_nonempty_rate_pct": _pct(n_kg, len(rows)),
        "steps_per_trajectory": n_steps / max(1, len(rows)),
        "step_citation_rate_pct": _pct(n_citing, n_steps),
        "step_visible_citation_rate_pct": _pct(n_visible_citing, n_steps),
        "all_steps_visible_cite_given_kg_pct": _pct(n_all_steps_visible_cite, n_kg),
        # This is an exact citation/KG consistency check. Some apparent misses
        # can still be parser false positives, so it is not by itself an IHR.
        "citation_not_in_visible_kg_rate_pct": _pct(hallucinations, citations),
        "citation_count": citations,
        "citation_not_in_visible_kg_count": hallucinations,
        "answer_match_mean": sum(answer_scores) / max(1, len(answer_scores)),
        "fractional_label_rate_pct": _pct(sum(v not in endpoints for v in labels), len(labels)),
        "negative_label_rate_pct": _pct(sum(v < 0 for v in labels), len(labels)),
        "neutral_label_rate_pct": _pct(sum(v == 0 for v in labels), len(labels)),
        "positive_label_rate_pct": _pct(sum(v > 0 for v in labels), len(labels)),
        "rejection_reasons": dict(sorted(reject_reasons.items())),
    }


def _metric_bundle(
    rows: Iterable[Dict[str, Any]], *, refilter_stored_kg: bool = False
) -> Dict[str, Any]:
    rows = list(rows)
    return {
        "all": _metrics(rows, refilter_stored_kg=refilter_stored_kg),
        "accepted_only": _metrics(
            (row for row in rows if row.get("accepted")),
            refilter_stored_kg=refilter_stored_kg,
        ),
    }


def _finite_delta(new: Any, old: Any) -> Any:
    if isinstance(new, (int, float)) and isinstance(old, (int, float)):
        if math.isfinite(float(new)) and math.isfinite(float(old)):
            return float(new) - float(old)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="append", required=True)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--expected_per_dataset", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pilot_paths = [Path(p) for p in args.pilot]
    pilot_rows: List[Dict[str, Any]] = []
    integrity_errors: List[str] = []
    source_files = []
    for path in pilot_paths:
        rows = _read(path)
        pilot_rows.extend(rows)
        source_files.append({"path": str(path), "records": len(rows), "md5": _md5(path)})
        if args.expected_per_dataset is not None and len(rows) != args.expected_per_dataset:
            integrity_errors.append(
                f"{path}: {len(rows)} records, expected {args.expected_per_dataset}"
            )
        for i, row in enumerate(rows, 1):
            extra = (row.get("metadata") or {}).get("extra") or {}
            if extra.get("source_split") != "train":
                integrity_errors.append(f"{path}:{i}: source_split is not train")
            if len(row.get("kg_subgraph") or []) > 12:
                integrity_errors.append(f"{path}:{i}: stored KG exceeds 12 triples")
            md = row.get("metadata") or {}
            if md.get("n_triples_teacher") != len(row.get("kg_subgraph") or []):
                integrity_errors.append(f"{path}:{i}: stored/teacher KG count mismatch")

    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for row in pilot_rows:
        by_dataset.setdefault(str(row.get("dataset") or "UNKNOWN"), []).append(row)

    report: Dict[str, Any] = {
        "integrity_pass": not integrity_errors,
        "integrity_errors": integrity_errors,
        "source_files": source_files,
        "metric_semantics": {
            "pilot_kg_view": "stored filtered KG shown to Teacher; no second filtering",
            "historical_kg_view": "historical stored KG re-filtered with current top12/min5 policy",
            "citation_metric": "exact parsed citation membership; not an LLM-judged IHR",
        },
        "pilot": {name: _metric_bundle(rows) for name, rows in sorted(by_dataset.items())},
        "paired_baseline": {},
        "scientific_verdict": "RESEARCHER_DECISION_REQUIRED",
    }

    if args.baseline:
        baseline_path = Path(args.baseline)
        wanted = {
            (str(r.get("dataset") or ""), _norm_text(r.get("question")))
            for r in pilot_rows
        }
        baseline_index = _read_matching(baseline_path, wanted)
        for dataset, new_rows in sorted(by_dataset.items()):
            pairs = [
                (r, baseline_index.get((dataset, _norm_text(r.get("question")))))
                for r in new_rows
            ]
            pairs = [(new, old) for new, old in pairs if old is not None]
            if not pairs:
                report["paired_baseline"][dataset] = {
                    "n_pairs": 0,
                    "status": "NO_MATCHED_HISTORICAL_ARM",
                }
                continue
            new_metrics = _metrics((new for new, _ in pairs), refilter_stored_kg=False)
            old_metrics = _metrics((old for _, old in pairs), refilter_stored_kg=True)
            report["paired_baseline"][dataset] = {
                "n_pairs": len(pairs),
                "new": new_metrics,
                "historical": old_metrics,
                "delta_new_minus_historical": {
                    key: _finite_delta(new_metrics.get(key), old_metrics.get(key))
                    for key in new_metrics
                },
            }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if integrity_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
