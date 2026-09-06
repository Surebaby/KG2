#!/usr/bin/env python
"""Build the three SFT-arm fixed-context inputs from frozen sources.

A: legacy KG (historical eval kg_subgraphs), B: ProofKG-v3 (frozen confirmation
KG), C: complete-only ProofKG (ProofKG where complete_plan_execution, else []).
Passages, questions, order and gold labels are identical across arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from kgproweight.utils.logging import dump_manifest, get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HIST = sorted((ROOT / "outputs/sft_quota70_baseline_eval/2wikimultihopqa/seed_42").glob("*_kg_proweight"))[0] / "intermediate_data.json"
DEFAULT_PROOF = ROOT / "data/derived/inference_proofkg_v1_confirmation270_2wiki_v3/question_kg_records.jsonl"
DEFAULT_OUT = ROOT / "outputs/audits/2wiki_confirmation270_sft_arms_inputs"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hist_intermediate", type=Path, default=DEFAULT_HIST)
    parser.add_argument("--proof_kg", type=Path, default=DEFAULT_PROOF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    args.out.mkdir(parents=True)

    hist = json.loads(args.hist_intermediate.read_text(encoding="utf-8"))  # JSON array
    proof_by_qid = {str(r["qid"]): r for r in _read_jsonl(args.proof_kg)}
    # confirmation270 = exactly the 270 qids in the frozen ProofKG
    rows = [r for r in hist if str(r["id"]) in proof_by_qid]
    assert len(rows) == 270, f"expected 270, got {len(rows)}"

    def arm_row(r, kg):
        return {
            "row_id": f"inference-proofkg-v1::2wikimultihopqa::{r['id']}",
            "dataset": "2wikimultihopqa",
            "qid": r["id"],
            "question": r["question"],
            "gold_answers": list(r.get("golden_answers") or []),
            "retrieved_passages": list(r["output"].get("retrieval_result") or []),
            "kg_subgraph": kg,
            # scope is a COHORT descriptor, identical across arms (the eval script
            # adds its own per-arm label); it must NOT differ between legacy/proof.
            "scope": "2wiki_confirmation270_standard_retrieval_model_utility",
        }

    arm_legacy = []
    arm_proof = []
    arm_proof_complete = []
    for r in rows:
        qid = str(r["id"])
        legacy_kg = list(r["output"].get("kg_subgraphs") or [])
        proof = proof_by_qid[qid]
        proof_kg = list(proof.get("kg_subgraph") or [])
        complete = bool(proof.get("complete_plan_execution"))
        arm_legacy.append(arm_row(r, legacy_kg))
        arm_proof.append(arm_row(r, proof_kg))
        arm_proof_complete.append(arm_row(r, proof_kg if complete else []))

    _write_jsonl(args.out / "arm_legacy.jsonl", arm_legacy)
    _write_jsonl(args.out / "arm_proof.jsonl", arm_proof)
    _write_jsonl(args.out / "arm_proof_complete_only.jsonl", arm_proof_complete)

    report = {
        "schema_version": "confirmation270-sft-arms-inputs-1",
        "n": len(rows),
        "n_complete": sum(1 for r in rows if proof_by_qid[str(r['id'])].get('complete_plan_execution')),
        "files": {
            "arm_legacy": {"path": str(args.out / "arm_legacy.jsonl"), "sha256": _sha256(args.out / "arm_legacy.jsonl")},
            "arm_proof": {"path": str(args.out / "arm_proof.jsonl"), "sha256": _sha256(args.out / "arm_proof.jsonl")},
            "arm_proof_complete_only": {"path": str(args.out / "arm_proof_complete_only.jsonl"), "sha256": _sha256(args.out / "arm_proof_complete_only.jsonl")},
        },
        "qid_order_sha256": hashlib.sha256("\n".join(str(r["id"]) for r in rows).encode()).hexdigest(),
        "passages_source": str(args.hist_intermediate),
        "proof_kg_source": str(args.proof_kg),
    }
    (args.out / "inputs_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_manifest(args.out, extra={"experiment_id": args.out.name, "phase": "build_sft_arms_inputs", "n": len(rows)}, status="COMPLETE")
    logger.info("built 3 arm inputs (n=%d, complete=%d)", len(rows), report["n_complete"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
