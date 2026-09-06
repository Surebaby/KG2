#!/usr/bin/env python
"""Freeze a blind, single-human A1 annotation pilot from untouched dev families."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

from kgproweight.training.query_planner import balanced_sample
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _review_columns() -> list[str]:
    columns = ["row_id", "question_key", "qid", "question"]
    for index in (1, 2):
        columns.extend([
            f"anchor_{index}_surface",
            f"anchor_{index}_wikipedia_title",
            f"anchor_{index}_wikidata_qid",
        ])
    columns.extend(["step_count", "operation"])
    for index in (1, 2, 3, 4):
        columns.extend([
            f"step_{index}_subject_ref",
            f"step_{index}_pid",
            f"step_{index}_output_slot",
            f"step_{index}_dependencies",
        ])
    columns.extend([
        "should_abstain",
        "abstain_reason",
        "review_confidence",
        "source_urls",
        "review_notes",
        "review_status",
    ])
    return columns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--old_dev_per_dataset", type=int, default=300)
    parser.add_argument("--old_dev_seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir, experiment_id = prepare_new_run_dir(
        args.output_dir,
        extra={
            "phase": "planner_v2_a1_single_human_blind_freeze",
            "dev": artifact_identity(args.dev),
            "assignments": artifact_identity(args.assignments),
            "n": args.n,
            "seed": args.seed,
        },
    )
    dev_rows = list(_read_jsonl(args.dev))
    assignments = {row["question_key"]: row for row in _read_jsonl(args.assignments)}
    old_evaluated = balanced_sample(
        Path(args.dev), per_dataset=args.old_dev_per_dataset, seed=args.old_dev_seed
    )
    old_families = {
        assignments[row["question_key"]]["family_sha256"]
        for row in old_evaluated
        if row["dataset"] == "2wikimultihopqa"
    }
    excluded_keys: set[str] = set()
    for path in args.exclude:
        excluded_keys.update(str(row["question_key"]) for row in _read_jsonl(path))

    eligible = [
        row for row in dev_rows
        if row["dataset"] == "2wikimultihopqa"
        and row["question_key"] not in excluded_keys
        and assignments[row["question_key"]]["family_sha256"] not in old_families
    ]
    # One row per untouched family prevents a repeated template family from
    # dominating this small annotatability pilot.
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        family = assignments[row["question_key"]]["family_sha256"]
        by_family.setdefault(family, []).append(row)
    if len(by_family) < args.n:
        raise SystemExit(f"only {len(by_family)} untouched families for n={args.n}")
    rng = random.Random(args.seed)
    selected_families = rng.sample(sorted(by_family), args.n)
    selected: list[dict[str, Any]] = []
    for family in selected_families:
        candidates = sorted(by_family[family], key=lambda row: row["question_key"])
        source = rng.choice(candidates)
        selected.append({
            "row_id": "",
            "question_key": source["question_key"],
            "dataset": source["dataset"],
            "qid": source["qid"],
            "question": source["question"],
            "family_sha256": family,
            "scope": "single_human_exploratory_a1",
        })
    selected.sort(key=lambda row: row["question_key"])
    for index, row in enumerate(selected, start=1):
        row["row_id"] = f"A1-{index:03d}"

    cohort_path = output_dir / "cohort.jsonl"
    with cohort_path.open("x", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    form_path = output_dir / "single_reviewer_form.tsv"
    columns = _review_columns()
    with form_path.open("x", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, dialect="excel-tab")
        writer.writeheader()
        for row in selected:
            writer.writerow({column: row.get(column, "") for column in columns})

    protocol = {
        "status": "FROZEN_BEFORE_HUMAN_REVIEW",
        "experiment_id": experiment_id,
        "scope": "single_human_exploratory_a1_not_confirmatory_gold",
        "n": len(selected),
        "seed": args.seed,
        "reviewer_count": 1,
        "blinding": {
            "reviewer_sees": ["question_key", "qid", "question"],
            "reviewer_does_not_see": [
                "model prediction", "old gold target", "answer", "supporting facts", "evidence"
            ],
        },
        "runtime_prohibited_inputs": ["answer", "supporting facts", "evidence", "gold alias"],
        "exploratory_acceptance_gates": {
            "review_completion_rate_min": 1.0,
            "non_abstained_title_qid_precision_min": 0.95,
            "resolvable_or_justified_abstain_rate_min": 0.90,
            "pid_graph_usable_rate_min": 0.85,
        },
        "unavailable_metrics": ["inter_annotator_agreement", "Cohen_kappa"],
        "boundary": "Passing permits a separately frozen model-comparison pilot, not training or formal KG replacement.",
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    guidelines_path = output_dir / "ANNOTATION_GUIDE.md"
    guidelines_path.write_text(
        "# Query Planner v2 A1 单人盲审说明\n\n"
        "本表是探索性单人标注，不是双审Gold。审阅时只使用题目和公开Wikipedia/Wikidata；"
        "禁止查看数据集答案、supporting facts、evidence、旧planner Gold或模型预测。\n\n"
        "## 填写顺序\n\n"
        "1. 抄录题目中作为推理起点的实体原文；括号消歧词属于surface的一部分。最多两个anchor。\n"
        "2. 查询对应英文Wikipedia canonical title和Wikidata QID；把核验页面URL写入source_urls。\n"
        "3. 写出完成问题需要的1–4步关系图。subject_ref只能是anchor_1/anchor_2或$hop_N；"
        "output_slot使用hop_1等；dependencies用逗号分隔。\n"
        "4. 无法唯一消歧、页面缺失或关系无法可靠判断时，should_abstain=YES并说明原因，"
        "不要猜测。\n"
        "5. review_confidence只填HIGH/MEDIUM/LOW；完成后review_status填COMPLETE。\n\n"
        "## 常用PID\n\n"
        "P22 father；P25 mother；P26 spouse；P40 child；P57 director；P175 performer；"
        "P569 date of birth；P570 date of death；P19 place of birth；P20 place of death；"
        "P27 citizenship；P17 country；P571 inception；P577 publication/release date；"
        "P69 educated at；P127 owned by。若不在表中，应通过Wikidata property页面核验。\n\n"
        "## 单人审阅限制\n\n"
        "不计算一致率或Cohen's kappa；任何通过结论只能写作single-review exploratory evidence。\n",
        encoding="utf-8",
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "status": "FROZEN_AWAITING_SINGLE_HUMAN_REVIEW",
        "eligible_rows": len(eligible),
        "eligible_families": len(by_family),
        "selected_n": len(selected),
        "selected_families": len({row["family_sha256"] for row in selected}),
        "selected_question_key_sha256": hashlib.sha256(
            "\n".join(row["question_key"] for row in selected).encode()
        ).hexdigest(),
        "old_dev_evaluated_family_overlap": len(
            {row["family_sha256"] for row in selected} & old_families
        ),
        "excluded_question_overlap": sum(
            row["question_key"] in excluded_keys for row in selected
        ),
        "artifacts": {
            "cohort": artifact_identity(cohort_path),
            "review_form": artifact_identity(form_path),
            "protocol": artifact_identity(protocol_path),
            "guide": artifact_identity(guidelines_path),
        },
        "inputs": {
            "dev": artifact_identity(args.dev),
            "assignments": artifact_identity(args.assignments),
            "exclude": [artifact_identity(path) for path in args.exclude],
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(output_dir, status=report["status"], extra=report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
