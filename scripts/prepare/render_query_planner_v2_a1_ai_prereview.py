#!/usr/bin/env python
"""Render AI pre-review decisions into the frozen A1 TSV schema."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import quote


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blank_form", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--output_tsv", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    blank_path = Path(args.blank_form)
    decisions_path = Path(args.decisions)
    output_path = Path(args.output_tsv)
    report_path = Path(args.report)
    with blank_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, dialect="excel-tab")
        fieldnames = list(reader.fieldnames or [])
        blank_rows = list(reader)
    decisions = _read_jsonl(decisions_path)
    by_id = {row["row_id"]: row for row in decisions}
    if len(by_id) != len(decisions):
        raise SystemExit("duplicate decision row_id")
    if {row["row_id"] for row in blank_rows} != set(by_id):
        raise SystemExit("blank form and decisions have different row_id sets")

    rendered: list[dict[str, str]] = []
    for blank in blank_rows:
        decision = by_id[blank["row_id"]]
        row = dict(blank)
        anchors = decision["anchors"]
        steps = decision["steps"]
        if not 1 <= len(anchors) <= 2 or not 1 <= len(steps) <= 4:
            raise SystemExit(f"invalid graph size for {blank['row_id']}")
        source_urls: list[str] = []
        for index, anchor in enumerate(anchors, start=1):
            row[f"anchor_{index}_surface"] = anchor["surface"]
            row[f"anchor_{index}_wikipedia_title"] = anchor["title"]
            row[f"anchor_{index}_wikidata_qid"] = anchor["qid"]
            source_urls.extend([
                f"https://en.wikipedia.org/wiki/{quote(anchor['title'].replace(' ', '_'), safe='()_')}",
                f"https://www.wikidata.org/wiki/{anchor['qid']}",
            ])
        row["step_count"] = str(len(steps))
        row["operation"] = decision["operation"]
        for index, step in enumerate(steps, start=1):
            row[f"step_{index}_subject_ref"] = step["subject_ref"]
            row[f"step_{index}_pid"] = step["pid"]
            row[f"step_{index}_output_slot"] = step["output_slot"]
            row[f"step_{index}_dependencies"] = ",".join(step["dependencies"])
        row["should_abstain"] = decision["should_abstain"]
        row["abstain_reason"] = decision.get("abstain_reason", "")
        row["review_confidence"] = decision["confidence"]
        row["source_urls"] = " ".join(source_urls)
        row["review_notes"] = "AI pre-review only; " + decision["notes"]
        row["review_status"] = "AI_PRE_REVIEW_COMPLETE"
        rendered.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, dialect="excel-tab")
        writer.writeheader()
        writer.writerows(rendered)

    confidence = {
        value: sum(row["review_confidence"] == value for row in rendered)
        for value in ("HIGH", "MEDIUM", "LOW")
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AI_PRE_REVIEW_COMPLETE_NOT_HUMAN_GOLD",
        "n": len(rendered),
        "confidence": confidence,
        "abstain_n": sum(row["should_abstain"] == "YES" for row in rendered),
        "human_review_completed": False,
        "scientific_boundary": {
            "counts_as_single_human_review": False,
            "counts_as_gold": False,
            "can_score_frozen_human_a1_gate": False,
            "allowed_use": "engineering triage and candidate preparation",
            "required_next_action": "human verifies at least all MEDIUM/LOW/abstain rows and records edits separately",
        },
        "inputs": {
            "blank_form": {"path": str(blank_path.resolve()), "sha256": _sha256(blank_path)},
            "decisions": {"path": str(decisions_path.resolve()), "sha256": _sha256(decisions_path)},
        },
        "output": {"path": str(output_path.resolve()), "sha256": _sha256(output_path)},
    }
    with report_path.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
