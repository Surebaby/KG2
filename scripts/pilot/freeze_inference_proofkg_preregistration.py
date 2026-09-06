#!/usr/bin/env python
"""Freeze the inference-side ProofKG-v1 question set BEFORE any planner output.

Splits each dataset's historical n=300 dev questions into a locked
``pilot`` (30) / ``confirmation`` (270) split by stable question hash, and
freezes the retrieval context (the 10 standard-retrieval passages + legacy KG)
that every downstream arm must reuse.  The pilot may never see the confirmation
questions' model outputs, so the split is committed here, before any plan is
generated.

The frozen artifacts are question-only: no gold answers, supporting facts, or
decomposition annotations enter these files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from kgproweight.kg.question_kg import question_sha256
from kgproweight.utils.logging import dump_manifest, get_logger

logger = get_logger(__name__)

DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "sft_quota70_baseline_eval"
DEFAULT_OUT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "audits"
    / "inference_proofkg_v1_n900_seed42_preregistration"
)

SCHEMA_VERSION = "inference-proofkg-v1-preregistration-1"
PILOT_PER_DATASET = 30

# Gold-derived fields that must never enter the question-only cohort or the
# retrieval contexts. The planner / executor / ProofKG builder read with
# gold_access=false and these fields are excluded by construction.
FORBIDDEN_FIELDS = [
    "golden_answers",
    "supporting_facts",
    "answer",
    "decomposition",
    "reasoning",
    "sp",
    "evidence",
]

# Frozen structure gate (pre-generation) and SFT utility gate (post-generation).
# Values are the *pre-registered* thresholds from §15.14 / D34; they are not
# measurements.
STRUCTURE_GATES = {
    "identity_hash_join": "== 1.0",
    "planner_schema_valid": ">= 0.97",
    "runtime_error": "== 0",
    "gold_access": "false",
    "plan_recognized": ">= 0.80",
    "proofkg_nonempty": ">= 0.80",
    "complete_plan_execution": ">= 0.70",
    "max_triples_per_question": "<= 12",
}

SFT_UTILITY_GATES = {
    "per_dataset_net_degradation": "<= 1 question",
    "macro_em_gain": "> 0",
    "macro_f1_gain": "> 0",
    "parse_valid_rate": "not worse",
    "ihr": "not run at pilot scale",
}


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_inputs(root: Path) -> List[Tuple[str, Path]]:
    found: List[Tuple[str, Path]] = []
    for ds in DATASETS:
        seed_dir = root / ds / "seed_42"
        if not seed_dir.is_dir():
            raise FileNotFoundError(f"missing seed dir: {seed_dir}")
        run_dirs = [
            c for c in sorted(seed_dir.glob(f"{ds}_*_kg_proweight"))
            if (c / "intermediate_data.json").is_file()
        ]
        if len(run_dirs) != 1:
            raise ValueError(f"expected exactly one run under {seed_dir}, got {run_dirs}")
        found.append((ds, run_dirs[0]))
    return found


def _read_yaml_config(run_dir: Path) -> Dict[str, Any]:
    import yaml

    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"historical config missing: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def build_rows(run_dir: Path, ds: str) -> List[Dict[str, Any]]:
    """Return per-question rows: {dataset, qid, question, question_sha256,
    passages, passages_sha256, legacy_kg, legacy_kg_sha256}."""
    data = _read_json(run_dir / "intermediate_data.json")
    rows: List[Dict[str, Any]] = []
    for entry in data:
        qid = entry.get("id")
        question = entry.get("question") or ""
        output = entry.get("output") or {}
        passages = output.get("retrieval_result")
        legacy_kg = output.get("kg_subgraphs")
        if not isinstance(passages, list) or not passages:
            raise ValueError(f"{ds}/{qid}: empty retrieval_result")
        if not isinstance(legacy_kg, list):
            raise ValueError(f"{ds}/{qid}: kg_subgraphs not a list")
        rows.append({
            "dataset": ds,
            "qid": qid,
            "question": question,
            "question_sha256": question_sha256(question),
            "passages": passages,
            "passages_sha256": _sha256_json(passages),
            "legacy_kg": legacy_kg,
            "legacy_kg_sha256": _sha256_json(legacy_kg),
        })
    return rows


def split_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Sort by question hash ascending; first PILOT_PER_DATASET = pilot.

    Marks each row's ``_split`` in place so the downstream retrieval-context
    writer can tag the split without an O(n^2) membership test.
    """
    ordered = sorted(rows, key=lambda r: r["question_sha256"])
    pilot, confirmation = ordered[:PILOT_PER_DATASET], ordered[PILOT_PER_DATASET:]
    for r in pilot:
        r["_split"] = "pilot"
    for r in confirmation:
        r["_split"] = "confirmation"
    return pilot, confirmation


def _question_only(row: Dict[str, Any], split: str) -> Dict[str, Any]:
    return {
        "dataset": row["dataset"],
        "qid": row["qid"],
        "question": row["question"],
        "question_sha256": row["question_sha256"],
        "split": split,
    }


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out = args.out
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing path: {out}")
    out.mkdir(parents=True)

    inputs = discover_inputs(args.root)
    all_rows: List[Dict[str, Any]] = []
    pilot_rows: List[Dict[str, Any]] = []
    confirmation_rows: List[Dict[str, Any]] = []

    model_path: str | None = None
    checkpoint: str | None = None

    for ds, run_dir in inputs:
        cfg = _read_yaml_config(run_dir)
        ds_model = cfg.get("generator_model_path")
        ds_ckpt = cfg.get("generator_lora_path")
        if model_path is None:
            model_path, checkpoint = ds_model, ds_ckpt
        elif ds_model != model_path:
            raise ValueError(f"model path mismatch across datasets: {ds_model} vs {model_path}")

        rows = build_rows(run_dir, ds)
        if len(rows) != 300:
            raise ValueError(f"{ds}: expected 300 questions, got {len(rows)}")
        if len({r["qid"] for r in rows}) != 300:
            raise ValueError(f"{ds}: duplicate qids detected")
        pilot, confirmation = split_rows(rows)
        if len(pilot) != PILOT_PER_DATASET or len(confirmation) != 270:
            raise ValueError(f"{ds}: split invariant failed ({len(pilot)}/{len(confirmation)})")

        # rows are the same dict objects (split_rows sorts a list of references),
        # so _split is set on every row; keep a hash-sorted order for the freeze.
        all_rows.extend(sorted(rows, key=lambda r: r["question_sha256"]))
        pilot_rows.extend(pilot)
        confirmation_rows.extend(confirmation)
        logger.info("%s: pilot=%d confirmation=%d", ds, len(pilot), len(confirmation))

    # Question-only cohorts (no gold fields by construction).
    _write_jsonl(
        out / "pilot.question_only.jsonl",
        (_question_only(r, "pilot") for r in pilot_rows),
    )
    _write_jsonl(
        out / "confirmation.question_only.jsonl",
        (_question_only(r, "confirmation") for r in confirmation_rows),
    )

    # Retrieval context for every question (passages + legacy KG), so both arms
    # reuse identical inputs.
    _write_jsonl(
        out / "retrieval_contexts.jsonl",
        (
            {
                "dataset": r["dataset"],
                "qid": r["qid"],
                "question_sha256": r["question_sha256"],
                "split": r["_split"],
                "passages": r["passages"],
                "passages_sha256": r["passages_sha256"],
                "legacy_kg": r["legacy_kg"],
                "legacy_kg_sha256": r["legacy_kg_sha256"],
            }
            for r in all_rows
        ),
    )

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_PLANNER_OUTPUT",
        "scope": (
            "inference-side ProofKG-v1 question-set + retrieval-context freeze; "
            "no planner output, no model inference, no gold access"
        ),
        "split": {
            "method": (
                "sort each dataset's 300 dev questions by question_sha256 "
                "ascending; first 30 = pilot, remaining 270 = confirmation"
            ),
            "hash_fn": "kgproweight.kg.question_kg.question_sha256",
            "pilot_per_dataset": PILOT_PER_DATASET,
            "confirmation_per_dataset": 270,
        },
        "forbidden_fields": FORBIDDEN_FIELDS,
        "model": {"generator_model_path": model_path, "checkpoint": checkpoint},
        "planner_version": "rule-query-plan-2",
        "executor": {
            "2wikimultihopqa": "relation-graph",
            "musique": "subquery-graph",
            "hotpotqa": "none (zero-shot subquery-graph pending step 4)",
        },
        "cache_versions": {
            "property": "wikidata-property-v1",
            "property_edge": "wikidata-property-edge-v1",
            "historical": "wikidata-historical-entity-revision-1",
            "evidence_store": "versioned-2wiki-evidence-store-1",
        },
        "structure_gates": STRUCTURE_GATES,
        "sft_utility_gates": SFT_UTILITY_GATES,
        "counts": {
            "pilot": len(pilot_rows),
            "confirmation": len(confirmation_rows),
            "total": len(all_rows),
        },
        "inputs": [{"dataset": ds, "run_dir": str(run_dir)} for ds, run_dir in inputs],
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dump_manifest(
        out,
        extra={
            "experiment_id": out.name,
            "phase": "preregistration",
            "status": "FROZEN_BEFORE_PLANNER_OUTPUT",
            "counts": protocol["counts"],
            "split": protocol["split"],
        },
        status="COMPLETE",
    )

    print(json.dumps({
        "counts": protocol["counts"],
        "split_method": protocol["split"]["method"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
