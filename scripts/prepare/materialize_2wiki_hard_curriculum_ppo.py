#!/usr/bin/env python
"""Materialize the frozen 208-qid hard curriculum for paired PPO-O/PPO-K.

The reserve82 scorer files are deliberately not copied.  Training records are
filtered from the already frozen train-only assets and question-KG rows are
enriched with their Gold-free executor trace for process-v2.1.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.utils.logging import dump_manifest


SCHEMA_VERSION = "proofkg-hard-curriculum-ppo-materialization-1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _by_qid(rows: Iterable[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in rows:
        row = dict(value)
        qid = str(row.get("qid") or "")
        if not qid or qid in result:
            raise ValueError(f"{label} contains empty/duplicate qid: {qid!r}")
        result[qid] = row
    return result


def materialize(protocol_path: Path, reserve_result_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    reserve_result = json.loads(reserve_result_path.read_text(encoding="utf-8"))
    if reserve_result.get("status") != "PASS_READY_TO_PREPARE_PAIRED_PPO" or not reserve_result.get("all_pass"):
        raise ValueError("reserve promotion gates did not pass; paired PPO materialization is forbidden")
    if reserve_result.get("protocol", {}).get("sha256") != _sha256(protocol_path):
        raise ValueError("reserve result is not bound to the supplied frozen protocol")

    curriculum_path = Path(protocol["outputs"]["curriculum"]["path"])
    curriculum = _read_jsonl(curriculum_path)
    if _sha256(curriculum_path) != protocol["outputs"]["curriculum"]["sha256"]:
        raise ValueError("frozen curriculum hash mismatch")
    if len(curriculum) != 208:
        raise ValueError(f"expected 208 curriculum qids, got {len(curriculum)}")

    source_names = ("full_silver", "full_question_kg", "full_runtime_details")
    sources: dict[str, Path] = {}
    for name in source_names:
        path = Path(protocol["inputs"][name]["path"])
        if _sha256(path) != protocol["inputs"][name]["sha256"]:
            raise ValueError(f"frozen source hash mismatch: {name}")
        sources[name] = path
    silver = _by_qid(_read_jsonl(sources["full_silver"]), "silver")
    kg = _by_qid(_read_jsonl(sources["full_question_kg"]), "question KG")
    runtime = _by_qid(_read_jsonl(sources["full_runtime_details"]), "runtime")

    selected_silver: list[dict[str, Any]] = []
    selected_kg: list[dict[str, Any]] = []
    weights: list[dict[str, Any]] = []
    stratum_counts = {"recovery": 0, "stability": 0}
    for identity in curriculum:
        qid = str(identity["qid"])
        if qid not in silver or qid not in kg or qid not in runtime:
            raise ValueError(f"curriculum qid missing source assets: {qid}")
        source_silver, source_kg, source_runtime = silver[qid], kg[qid], runtime[qid]
        question = str(identity["question"])
        expected_hash = str(identity["question_sha256"])
        for label, row in (("silver", source_silver), ("question KG", source_kg), ("runtime", source_runtime)):
            if str(row.get("question") or "").strip() != question.strip():
                raise ValueError(f"{label} question mismatch for {qid}")
            if question_sha256(str(row["question"])) != expected_hash:
                raise ValueError(f"{label} question hash mismatch for {qid}")
        plan = source_kg.get("query_plan") or {}
        execution = source_runtime.get("execution") or {}
        if not source_kg.get("kg_subgraph") or not source_kg.get("provenance", {}).get("complete_plan_execution"):
            raise ValueError(f"curriculum qid lacks complete ProofKG: {qid}")
        if not execution.get("hops") or len(execution["hops"]) < len(plan.get("hops") or []):
            raise ValueError(f"curriculum qid lacks complete executor trace: {qid}")
        if source_runtime.get("runtime_error") is not None:
            raise ValueError(f"curriculum qid has runtime error: {qid}")

        silver_row = dict(source_silver)
        metadata = dict(silver_row.get("metadata") or {})
        metadata.update({
            "hard_curriculum_version": "2wiki-hard-contrastive-v1",
            "hard_curriculum_stratum": str(identity["stratum"]),
            "hard_curriculum_sampling_probability": float(identity["sampling_probability"]),
            "hard_curriculum_protocol_sha256": _sha256(protocol_path),
        })
        silver_row["metadata"] = metadata
        selected_silver.append(silver_row)

        kg_row = dict(source_kg)
        kg_row["execution"] = dict(execution)
        kg_row["runtime_error"] = None
        selected_kg.append(kg_row)

        stratum = str(identity["stratum"])
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
        weights.append({
            "schema_version": "proofkg-rollout-sampling-weight-1",
            "dataset": str(identity["dataset"]),
            "qid": qid,
            "question_sha256": expected_hash,
            "stratum": stratum,
            "sampling_probability": float(identity["sampling_probability"]),
        })

    if abs(sum(row["sampling_probability"] for row in weights) - 1.0) > 1e-9:
        raise ValueError("sampling probability mass is not 1.0")
    if abs(sum(row["sampling_probability"] for row in weights if row["stratum"] == "recovery") - 0.5) > 1e-9:
        raise ValueError("recovery sampling mass is not 0.5")
    if abs(sum(row["sampling_probability"] for row in weights if row["stratum"] == "stability") - 0.5) > 1e-9:
        raise ValueError("stability sampling mass is not 0.5")

    output_dir.mkdir(parents=True)
    outputs = {
        "silver_train": output_dir / "silver_train.jsonl",
        "question_kg_records": output_dir / "question_kg_records.with_execution.jsonl",
        "sampling_weights": output_dir / "sampling_weights.jsonl",
    }
    _write_jsonl(outputs["silver_train"], selected_silver)
    _write_jsonl(outputs["question_kg_records"], selected_kg)
    _write_jsonl(outputs["sampling_weights"], weights)
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "PROOFKG-2WIKI-HARD-CURRICULUM-V1-PAIRED-PPO-DATA",
        "status": "COMPLETE_NOT_TRAINED",
        "n_train_qids": len(selected_silver),
        "stratum_counts": stratum_counts,
        "sampling_mass": {
            name: sum(row["sampling_probability"] for row in weights if row["stratum"] == name)
            for name in sorted(stratum_counts)
        },
        "reserve_assets_copied": False,
        "gold_access": {
            "silver_outcome_labels": "train-only gold answer present for outcome reward",
            "question_kg_and_sampling": False,
            "reserve82_in_training": False,
        },
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "reserve_result": {"path": str(reserve_result_path), "sha256": _sha256(reserve_result_path)},
        "outputs": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in outputs.items()},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(output_dir, status=report["status"], extra={
        "experiment_id": report["experiment_id"],
        "report_sha256": _sha256(report_path),
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reserve_result", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.protocol, args.reserve_result, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
