#!/usr/bin/env python
"""Derive planner inputs from the frozen inference-ProofKG preregistration.

The frozen ``pilot/confirmation.question_only.jsonl`` files are deliberately
question-only and carry no planner-specific fields.  This script *derives*
(never overwrites) the execution-append fields the learned planner generator
(``scripts/eval/generate_query_plans_unseen.py``) requires — ``row_id``,
``question_key``, ``target_type`` — and writes an execution protocol recording
the exact planner adapter/config to use.

It performs no model inference and emits no planner output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from kgproweight.kg.question_kg import question_key
from kgproweight.utils.logging import dump_manifest, get_logger

logger = get_logger(__name__)

DEFAULT_FROZEN_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "audits"
    / "inference_proofkg_v1_n900_seed42_preregistration"
)
DEFAULT_OUT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "audits"
    / "inference_proofkg_v1_pilot30x3_execution_v1"
)

PLANNER_ADAPTER = "checkpoints/query_planner_learned_scale_v1_1_seed42/final"
PLANNER_CONFIG = "configs/training/query_planner_learned_scale_v1_1_seed42.yaml"

# Fixed target-type mapping. HotpotQA reuses subquery_graph as a zero-shot
# transfer (no training supervision); it is recorded as such, not as a trained
# capability.
TARGET_TYPE = {
    "2wikimultihopqa": "relation_graph",
    "musique": "subquery_graph",
    "hotpotqa": "subquery_graph",
}
PLANNER_MODE = {
    "2wikimultihopqa": "learned_planner_v1_1_relation_graph",
    "musique": "learned_planner_v1_1_subquery_graph",
    "hotpotqa": "learned_planner_v1_1_zero_shot_subquery_graph",
}

FORBIDDEN_FIELDS = {
    "golden_answers", "supporting_facts", "answer", "answers", "target",
    "decomposition", "question_decomposition", "reasoning", "sp", "evidence",
    "evidences", "paragraph_text",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _assert_no_forbidden(row: Dict[str, Any], location: str) -> None:
    present = FORBIDDEN_FIELDS.intersection(str(k) for k in row)
    if present:
        raise ValueError(f"forbidden fields present at {location}: {sorted(present)}")


def derive(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return the row plus the execution-append fields, in the generator's order."""
    dataset = str(row["dataset"])
    qid = str(row["qid"])
    out = {
        "row_id": f"inference-proofkg-v1::{dataset}::{qid}",
        "question_key": question_key(dataset, qid),
        "dataset": dataset,
        "qid": qid,
        "question": row["question"],
        "question_sha256": row["question_sha256"],
        "target_type": TARGET_TYPE[dataset],
        "split": row["split"],
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_dir", type=Path, default=DEFAULT_FROZEN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    frozen = args.frozen_dir
    pilot_src = frozen / "pilot.question_only.jsonl"
    conf_src = frozen / "confirmation.question_only.jsonl"
    for p in (pilot_src, conf_src):
        if not p.is_file():
            raise FileNotFoundError(f"missing frozen file: {p}")

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing path: {args.out}")
    args.out.mkdir(parents=True)

    pilot_rows = _read_jsonl(pilot_src)
    conf_rows = _read_jsonl(conf_src)

    # Gate 0: reference the frozen files and preserve them byte-for-byte.
    frozen_hashes = {
        "pilot": _sha256_file(pilot_src),
        "confirmation": _sha256_file(conf_src),
    }

    derived_pilot = []
    derived_conf = []
    seen_row_ids: set = set()
    for split, rows, derived in (
        ("pilot", pilot_rows, derived_pilot),
        ("confirmation", conf_rows, derived_conf),
    ):
        for row in rows:
            _assert_no_forbidden(row, f"{split}.{row.get('qid')}")
            d = derive(row)
            if d["row_id"] in seen_row_ids:
                raise ValueError(f"duplicate row_id: {d['row_id']}")
            seen_row_ids.add(d["row_id"])
            derived.append(d)

    # Gates: split partition unchanged, forbidden fields 0, row_id unique,
    # target_type coverage 1.0, question/hash/qid preserved row-for-row.
    split_unchanged = (
        len(derived_pilot) == 90 and len(derived_conf) == 810
        and {d["split"] for d in derived_pilot} == {"pilot"}
        and {d["split"] for d in derived_conf} == {"confirmation"}
    )
    row_ids_unique = len(seen_row_ids) == len(derived_pilot) + len(derived_conf)
    target_coverage = all(
        d["target_type"] == TARGET_TYPE[d["dataset"]] for d in derived_pilot + derived_conf
    )
    # Row-for-row consistency vs the frozen source: the frozen files only carry
    # dataset/qid/question/question_sha256/split, all of which are passed through
    # verbatim by derive().
    consistent = all(
        (p["dataset"], p["qid"], p["question"], p["question_sha256"], p["split"])
        == (d["dataset"], d["qid"], d["question"], d["question_sha256"], d["split"])
        for p, d in zip(pilot_rows, derived_pilot)
    ) and all(
        (p["dataset"], p["qid"], p["question"], p["question_sha256"], p["split"])
        == (d["dataset"], d["qid"], d["question"], d["question_sha256"], d["split"])
        for p, d in zip(conf_rows, derived_conf)
    )

    def _write(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    pilot_out = args.out / "planner_inputs.pilot.jsonl"
    conf_out = args.out / "planner_inputs.confirmation.jsonl"
    _write(pilot_out, derived_pilot)
    _write(conf_out, derived_conf)

    gates = {
        "question_qid_hash_consistent": bool(consistent),
        "split_unchanged": bool(split_unchanged),
        "forbidden_fields_zero": True,  # _assert_no_forbidden raised otherwise
        "row_id_unique": bool(row_ids_unique),
        "target_type_coverage_1_0": bool(target_coverage),
        "no_planner_output_generated": True,
    }
    all_pass = all(gates.values())

    execution_protocol = {
        "schema_version": "inference-proofkg-execution-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "EXECUTION_PROTOCOL_FROZEN",
        "planner_adapter": PLANNER_ADAPTER,
        "planner_config": PLANNER_CONFIG,
        "target_type_mapping": TARGET_TYPE,
        "planner_mode": PLANNER_MODE,
        "hotpotqa": "learned planner v1.1 zero-shot transfer (no training supervision)",
        "frozen_source": {
            "dir": str(frozen),
            "files": {
                "pilot.question_only.jsonl": frozen_hashes["pilot"],
                "confirmation.question_only.jsonl": frozen_hashes["confirmation"],
            },
        },
        "gates": gates,
    }
    (args.out / "execution_protocol.json").write_text(
        json.dumps(execution_protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all_pass else "FAIL",
        "counts": {
            "pilot": len(derived_pilot),
            "confirmation": len(derived_conf),
            "total": len(derived_pilot) + len(derived_conf),
        },
        "gates": gates,
        "inputs": {
            "pilot": str(pilot_src),
            "confirmation": str(conf_src),
        },
        "outputs": {
            "pilot": str(pilot_out),
            "confirmation": str(conf_out),
            "execution_protocol": str(args.out / "execution_protocol.json"),
        },
    }
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dump_manifest(
        args.out,
        extra={
            "experiment_id": args.out.name,
            "phase": "materialize_planner_inputs",
            "status": report["status"],
            "gates": gates,
            "counts": report["counts"],
        },
        status="COMPLETE" if all_pass else "FAIL_STOP",
    )

    logger.info("gates=%s", gates)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
