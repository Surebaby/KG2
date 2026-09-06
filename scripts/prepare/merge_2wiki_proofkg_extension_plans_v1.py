#!/usr/bin/env python3
"""Merge the two frozen 2Wiki extension planner runs in combined350 order.

The v1b300 and reserve50 planner jobs were intentionally run separately.  This
CPU-only utility performs no inference and no repair: it verifies both releases
against their frozen question-only parents, then emits an append-only 350-row
planner artifact ordered exactly like the already-frozen combined cohort.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_2wiki_proofkg_extension_v1 import file_ref


DATASET = "2wikimultihopqa"
EXPECTED_COMBINED_SCHEMA = "2wiki-proofkg-extension-combined-protocol-v1"
EXPECTED_COMBINED_STATUS = "FROZEN_TRAIN_ONLY_BEFORE_PLANNER_NOT_MATERIALIZED"
EXPECTED_PLAN_STATUS = "RUNTIME_PLANS_FROZEN_NO_GOLD_AUDIT"
STATUS = "MERGED_RUNTIME_PLANS_FROZEN_NO_GOLD_AUDIT"
SCHEMA_VERSION = "2wiki-proofkg-extension-combined-plans-v1"
EXPERIMENT_ID = "2WIKI-PROOFKG-EXTENSION-COMBINED-V1-N350-SEED42-PLANS-MERGED"
DEFAULT_COMBINED = Path(
    "outputs/audits/2wiki_proofkg_extension_combined_v1_n350_seed42_"
    "preregistration/protocol.json"
)
DEFAULT_V1B = Path(
    "outputs/validation/2wiki_proofkg_extension_v1b_n300_seed42_plans"
)
DEFAULT_RESERVE = Path(
    "outputs/validation/2wiki_proofkg_extension_reserve_v1_n50_seed42_plans"
)
DEFAULT_OUT = Path(
    "outputs/validation/2wiki_proofkg_extension_combined_v1_n350_seed42_"
    "plans_merged"
)
FORBIDDEN_FIELDS = {
    "answer",
    "answers",
    "gold_answer",
    "golden_answers",
    "supporting_facts",
    "support",
    "decomposition",
    "evidence",
    "reasoning",
    "steps",
    "kg_subgraph",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_ref(identity: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(identity.get("path") or "")).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if _sha256(path) != str(identity.get("sha256") or ""):
        raise ValueError(f"{label} SHA256 mismatch: {path}")
    if identity.get("size_bytes") is not None and path.stat().st_size != int(
        identity["size_bytes"]
    ):
        raise ValueError(f"{label} size mismatch: {path}")
    return path


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # nosec: legacy report binding


def _index_and_validate(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(rows, start=1):
        row = dict(value)
        dataset = str(row.get("dataset") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        question = str(row.get("question") or "").strip()
        key = question_key(dataset, qid)
        if (
            dataset != DATASET
            or not qid
            or not question
            or row.get("question_key") != key
            or row.get("question_sha256") != question_sha256(question)
            or row.get("gold_access") is not False
        ):
            raise ValueError(f"{label}[{index}] identity/gold boundary invalid")
        present = FORBIDDEN_FIELDS & set(row)
        if present:
            raise ValueError(f"{label}/{key} contains forbidden fields: {sorted(present)}")
        if key in output:
            raise ValueError(f"duplicate {label} identity: {key}")
        output[key] = row
    return output


def _load_plan_release(
    directory: Path,
    *,
    expected_cohort: Path,
    expected_n: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    report_path = directory / "report.json"
    predictions_path = directory / "predictions.question_only.jsonl"
    manifest_path = directory / "manifest.json"
    for path in (report_path, predictions_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != EXPECTED_PLAN_STATUS:
        raise ValueError(f"planner release status invalid: {directory}")
    if int((report.get("counts") or {}).get("n", -1)) != expected_n:
        raise ValueError(f"planner release count invalid: {directory}")
    cohort_ref = (report.get("inputs") or {}).get("cohort") or {}
    if (
        Path(str(cohort_ref.get("path") or "")).resolve()
        != expected_cohort.resolve()
        or str(cohort_ref.get("md5") or "") != _md5(expected_cohort)
    ):
        raise ValueError(f"planner report/cohort binding invalid: {directory}")
    output_ref = (report.get("outputs") or {}).get("predictions") or {}
    if (
        Path(str(output_ref.get("path") or "")).resolve()
        != predictions_path.resolve()
        or str(output_ref.get("md5") or "") != _md5(predictions_path)
    ):
        raise ValueError(f"planner report/prediction binding invalid: {directory}")
    rows = _read_jsonl(predictions_path)
    if len(rows) != expected_n:
        raise ValueError(f"planner file has {len(rows)}/{expected_n} rows")
    return rows, report, {
        "report": file_ref(report_path),
        "predictions": file_ref(predictions_path),
        "manifest": file_ref(manifest_path),
    }


def merge_plan_rows(
    *,
    combined_rows: Sequence[Mapping[str, Any]],
    v1b_cohort_rows: Sequence[Mapping[str, Any]],
    reserve_cohort_rows: Sequence[Mapping[str, Any]],
    v1b_prediction_rows: Sequence[Mapping[str, Any]],
    reserve_prediction_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    combined = _index_and_validate(combined_rows, label="combined cohort")
    v1b_cohort = _index_and_validate(v1b_cohort_rows, label="v1b cohort")
    reserve_cohort = _index_and_validate(reserve_cohort_rows, label="reserve cohort")
    v1b = _index_and_validate(v1b_prediction_rows, label="v1b predictions")
    reserve = _index_and_validate(
        reserve_prediction_rows, label="reserve predictions"
    )
    if set(v1b) != set(v1b_cohort):
        raise ValueError("v1b prediction/cohort identity join is not exact")
    if set(reserve) != set(reserve_cohort):
        raise ValueError("reserve prediction/cohort identity join is not exact")
    if set(v1b) & set(reserve):
        raise ValueError("v1b/reserve prediction identities overlap")
    if set(combined) != set(v1b) | set(reserve):
        raise ValueError("combined cohort is not the exact prediction union")
    predictions = {**v1b, **reserve}
    merged: list[dict[str, Any]] = []
    for cohort in combined_rows:
        key = str(cohort["question_key"])
        prediction = dict(predictions[key])
        for field in (
            "question_key",
            "dataset",
            "qid",
            "question",
            "question_sha256",
            "gold_access",
        ):
            if prediction.get(field) != cohort.get(field):
                raise ValueError(f"combined/prediction mismatch at {field}: {key}")
        merged.append(prediction)
    gates = {
        "combined_n_350": len(merged) == 350,
        "v1b_n_300": len(v1b) == 300,
        "reserve_n_50": len(reserve) == 50,
        "exact_identity_union": len({row["question_key"] for row in merged})
        == len(merged),
        "combined_order_preserved": [row["question_key"] for row in merged]
        == [row["question_key"] for row in combined_rows],
        "gold_access_false": all(row.get("gold_access") is False for row in merged),
        "forbidden_gold_process_fields_zero": all(
            not (FORBIDDEN_FIELDS & set(row)) for row in merged
        ),
    }
    return merged, gates


def merge_release(
    *,
    combined_protocol_path: Path,
    v1b_plan_dir: Path,
    reserve_plan_dir: Path,
    output_dir: Path,
    experiment_id: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite merged plans: {output_dir}")
    protocol = json.loads(combined_protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version") != EXPECTED_COMBINED_SCHEMA
        or protocol.get("status") != EXPECTED_COMBINED_STATUS
    ):
        raise ValueError("combined350 protocol schema/status invalid")
    combined_path = _resolve_ref(protocol.get("output_cohort") or {}, label="combined cohort")
    v1b_cohort_path = _resolve_ref(
        (protocol.get("inputs") or {}).get("parent_v1b") or {}, label="v1b cohort"
    )
    reserve_cohort_path = _resolve_ref(
        (protocol.get("inputs") or {}).get("reserve_v1") or {}, label="reserve cohort"
    )
    v1b_rows, v1b_report, v1b_refs = _load_plan_release(
        v1b_plan_dir, expected_cohort=v1b_cohort_path, expected_n=300
    )
    reserve_rows, reserve_report, reserve_refs = _load_plan_release(
        reserve_plan_dir, expected_cohort=reserve_cohort_path, expected_n=50
    )
    if (v1b_report.get("inputs") or {}).get("adapter") != (
        reserve_report.get("inputs") or {}
    ).get("adapter") or (v1b_report.get("inputs") or {}).get("config") != (
        reserve_report.get("inputs") or {}
    ).get("config"):
        raise ValueError("planner adapter/config differs between v1b and reserve")
    merged, gates = merge_plan_rows(
        combined_rows=_read_jsonl(combined_path),
        v1b_cohort_rows=_read_jsonl(v1b_cohort_path),
        reserve_cohort_rows=_read_jsonl(reserve_cohort_path),
        v1b_prediction_rows=v1b_rows,
        reserve_prediction_rows=reserve_rows,
    )
    if not all(gates.values()):
        raise RuntimeError(f"combined planner merge gate failed: {gates}")
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "predictions.question_only.jsonl"
    _write_jsonl(predictions_path, merged)
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(experiment_id).strip(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "counts": {
            "n": len(merged),
            "schema_valid": sum(bool(row.get("schema_valid")) for row in merged),
            "schema_invalid": sum(not bool(row.get("schema_valid")) for row in merged),
            "by_parent": {"v1b": len(v1b_rows), "reserve": len(reserve_rows)},
            "invalid_by_parent": {
                "v1b": sum(not bool(row.get("schema_valid")) for row in v1b_rows),
                "reserve": sum(
                    not bool(row.get("schema_valid")) for row in reserve_rows
                ),
            },
        },
        "gates": gates,
        "inputs": {
            "combined_protocol": file_ref(combined_protocol_path),
            "combined_cohort": file_ref(combined_path),
            "v1b_cohort": file_ref(v1b_cohort_path),
            "reserve_cohort": file_ref(reserve_cohort_path),
            "v1b_plan_release": v1b_refs,
            "reserve_plan_release": reserve_refs,
        },
        "outputs": {"predictions": file_ref(predictions_path)},
        "scientific_boundary": {
            "planner_inference_run": False,
            "invalid_predictions_repaired_or_dropped": False,
            "gold_access": False,
            "training_started": False,
        },
        "training_started": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        output_dir,
        status=STATUS,
        extra={
            "phase": "merge_2wiki_proofkg_extension_plans_v1",
            "experiment_id": report["experiment_id"],
            "report": file_ref(report_path),
            "predictions": report["outputs"]["predictions"],
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-protocol", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--v1b-plans", type=Path, default=DEFAULT_V1B)
    parser.add_argument("--reserve-plans", type=Path, default=DEFAULT_RESERVE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    args = parser.parse_args()
    report = merge_release(
        combined_protocol_path=args.combined_protocol,
        v1b_plan_dir=args.v1b_plans,
        reserve_plan_dir=args.reserve_plans,
        output_dir=args.out,
        experiment_id=args.experiment_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
