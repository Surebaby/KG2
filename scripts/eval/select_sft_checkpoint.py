#!/usr/bin/env python
"""Apply the frozen light20 anti-forgetting gate to SFT candidate JSONLs.

This script does not recompute or alter EM/F1.  It only verifies that every
candidate used the same qid sequence and selects the earliest checkpoint that
passes the pre-registered replay thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        raise ValueError(f"empty candidate JSONL: {path}")
    qids = [str(row.get("qid") or "") for row in rows]
    if any(not value for value in qids) or len(qids) != len(set(qids)):
        raise ValueError(f"candidate has missing/duplicate qids: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(rows: list[dict]) -> dict:
    hidden = [row for row in rows if not bool(row.get("gold_in_passages"))]
    return {
        "n": len(rows),
        "parse_rate": sum(bool(row.get("well_formed")) for row in rows) / len(rows),
        "em": sum(float(row.get("em", 0.0)) for row in rows) / len(rows),
        "f1": sum(float(row.get("f1", 0.0)) for row in rows) / len(rows),
        "hidden_n": len(hidden),
        "hidden_em": (
            sum(float(row.get("em", 0.0)) for row in hidden) / len(hidden)
            if hidden else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LABEL=JSONL",
        help="Candidate in chronological order; repeat for each checkpoint.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected_n", type=int, default=200)
    parser.add_argument("--min_parse_rate", type=float, default=0.995)
    parser.add_argument("--min_em", type=float, default=0.770)
    parser.add_argument("--min_hidden_em", type=float, default=0.545)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selection report: {output}")

    parsed: list[tuple[str, Path]] = []
    for value in args.candidate:
        if "=" not in value:
            raise ValueError(f"candidate must be LABEL=JSONL: {value}")
        label, raw_path = value.split("=", 1)
        if not label.strip() or not raw_path.strip():
            raise ValueError(f"candidate must be LABEL=JSONL: {value}")
        parsed.append((label.strip(), Path(raw_path)))

    reference_qids: list[str] | None = None
    reports = []
    selected = None
    for label, path in parsed:
        rows = _read(path)
        qids = [str(row["qid"]) for row in rows]
        if len(rows) != args.expected_n:
            raise ValueError(f"{label}: expected n={args.expected_n}, got {len(rows)}")
        if reference_qids is None:
            reference_qids = qids
        elif qids != reference_qids:
            raise ValueError(f"{label}: qid order/cohort differs from first candidate")
        metrics = _metrics(rows)
        passed = (
            metrics["parse_rate"] >= args.min_parse_rate
            and metrics["em"] >= args.min_em
            and metrics["hidden_em"] is not None
            and metrics["hidden_em"] >= args.min_hidden_em
        )
        reports.append(
            {
                "label": label,
                "path": str(path),
                "sha256": _sha256(path),
                "metrics": metrics,
                "passes_gate": passed,
            }
        )
        if selected is None and passed:
            selected = label

    qid_blob = "\n".join(reference_qids or []).encode("utf-8")
    report = {
        "experiment_id": "SFT-PROOFKG-LIGHT20-V2-CHECKPOINT-SELECTION",
        "status": "PASS" if selected is not None else "FAIL_STOP",
        "selection_rule": "earliest passing candidate in declared chronological order",
        "thresholds": {
            "expected_n": args.expected_n,
            "min_parse_rate": args.min_parse_rate,
            "min_em": args.min_em,
            "min_hidden_em": args.min_hidden_em,
        },
        "qid_sha256": hashlib.sha256(qid_blob).hexdigest(),
        "selected": selected,
        "candidates": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if selected is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
