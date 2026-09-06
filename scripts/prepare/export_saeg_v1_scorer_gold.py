#!/usr/bin/env python
"""Export scorer-only Gold answers for frozen SAEG evaluation roles.

This script runs only after answer-free inference inputs are materialized.  It
never modifies them and stores development/confirmation/reporting Gold in a
separate versioned directory.  Confirmation remains sealed until its gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_saeg_v1_evaluation_protocol import assert_answer_free


EXPERIMENT_ID = "SAEG-V1-SCORER-GOLD-SEED42"
STATUS = "COMPLETE_SCORER_ONLY_CONFIRMATION_SEALED"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_inputs", type=Path, default=Path(
        "data/derived/saeg_v1_evaluation_inputs_seed42_v1"))
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path(
        "data/derived/saeg_v1_scorer_gold_seed42_v1"))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite SAEG scorer Gold: {args.out}")
    input_paths = {
        role: args.eval_inputs / f"{role}.answer_free.jsonl"
        for role in ("development", "confirmation", "canonical_reporting")
    }
    input_paths["report"] = args.eval_inputs / "report.json"
    for path in input_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    eval_report = json.loads(input_paths["report"].read_text(encoding="utf-8"))
    if eval_report.get("status") != "COMPLETE_ANSWER_FREE_CONFIRMATION_UNOPENED":
        raise ValueError("evaluation input dataset is not complete/answer-free")

    raw_paths = {
        dataset: args.data_root / dataset / "dev.jsonl"
        for dataset in ("hotpotqa", "2wikimultihopqa", "musique")
    }
    raw = {
        dataset: {str(row["id"]): row for row in read_jsonl(path)}
        for dataset, path in raw_paths.items()
    }
    args.out.mkdir(parents=True, exist_ok=False)
    output_paths = {}
    counts = Counter()
    seen_role_key = set()
    for role in ("development", "confirmation", "canonical_reporting"):
        output = []
        for item in read_jsonl(input_paths[role]):
            assert_answer_free(item)
            key = (role, str(item["question_key"]))
            if key in seen_role_key:
                raise ValueError(f"duplicate role/question: {key}")
            seen_role_key.add(key)
            source = raw[str(item["dataset"])].get(str(item["qid"]))
            if source is None:
                raise ValueError(f"missing raw dev Gold: {item['question_key']}")
            answers = [str(value).strip() for value in (source.get("golden_answers") or []) if str(value).strip()]
            if not answers:
                raise ValueError(f"empty Gold answers: {item['question_key']}")
            output.append({
                "schema_version": "saeg-scorer-gold-v1",
                "question_key": str(item["question_key"]),
                "dataset": str(item["dataset"]),
                "qid": str(item["qid"]),
                "question_sha256": str(item["question_sha256"]),
                "role": str(item["role"]),
                "partition": role,
                "golden_answers": answers,
                "scorer_only": True,
                "sealed": str(item["role"]) == "confirmation",
            })
            counts[f"role::{role}"] += 1
            counts[f"dataset::{role}::{item['dataset']}"] += 1
        path = args.out / f"{role}.gold.jsonl"
        write_jsonl(path, output)
        output_paths[role] = path
    if len(seen_role_key) != 1350:
        raise RuntimeError(f"expected 1350 scorer rows, got {len(seen_role_key)}")
    report = {
        "schema_version": "saeg-scorer-gold-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "counts": dict(sorted(counts.items())),
        "integrity": {
            "role_qualified_records": len(seen_role_key),
            "inputs_remain_answer_free": True,
            "gold_files_physically_separate": True,
            "confirmation_model_predictions_scored": False,
            "confirmation_sealed": True,
        },
        "inputs": {
            **{name: {"path": str(path), "sha256": sha256_file(path)} for name, path in input_paths.items()},
            **{f"raw_{name}_dev": {"path": str(path), "sha256": sha256_file(path)} for name, path in raw_paths.items()},
        },
        "outputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in output_paths.items()},
        "access_policy": {
            "development": "scorer may read during method development",
            "confirmation": "scorer must refuse until a frozen development gate explicitly opens one run",
            "canonical_reporting": "scorer may read for reporting; never checkpoint/model selection",
        },
    }
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(args.out, extra=report, status=STATUS)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
