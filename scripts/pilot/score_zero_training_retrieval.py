#!/usr/bin/env python
"""Score the fixed-cohort E0/E1/E2 zero-training retrieval experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import comb
from pathlib import Path
from typing import Any, Dict, List

from kgproweight.eval.stats import paired_bootstrap
from kgproweight.utils.logging import dump_manifest


def _read(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    visible = [row for row in rows if row.get("gold_in_passages")]
    hidden = [row for row in rows if not row.get("gold_in_passages")]
    return {
        "n": n,
        "em": sum(float(row["em"]) for row in rows) / max(1, n),
        "f1": sum(float(row["f1"]) for row in rows) / max(1, n),
        "parse_rate": sum(bool(row.get("well_formed")) for row in rows) / max(1, n),
        "gold_visible_n": len(visible),
        "em_gold_visible": (
            sum(float(row["em"]) for row in visible) / len(visible) if visible else None
        ),
        "em_gold_hidden": (
            sum(float(row["em"]) for row in hidden) / len(hidden) if hidden else None
        ),
    }


def _mcnemar(old: List[float], new: List[float]) -> Dict[str, Any]:
    improve = sum(a == 0 and b == 1 for a, b in zip(old, new))
    degrade = sum(a == 1 and b == 0 for a, b in zip(old, new))
    discordant = improve + degrade
    if not discordant:
        p_value = 1.0
    else:
        k = min(improve, degrade)
        p_value = min(
            1.0,
            2.0 * sum(comb(discordant, i) * 0.5**discordant for i in range(k + 1)),
        )
    return {
        "improve": improve,
        "degrade": degrade,
        "net": improve - degrade,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def _paired(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_a = {row["qid"]: row for row in a}
    by_b = {row["qid"]: row for row in b}
    if list(by_a) != list(by_b):
        raise ValueError("paired qid order differs")
    em_a = [float(by_a[qid]["em"]) for qid in by_a]
    em_b = [float(by_b[qid]["em"]) for qid in by_a]
    f1_a = [float(by_a[qid]["f1"]) for qid in by_a]
    f1_b = [float(by_b[qid]["f1"]) for qid in by_a]
    improved = [qid for qid in by_a if by_a[qid]["em"] == 0 and by_b[qid]["em"] == 1]
    degraded = [qid for qid in by_a if by_a[qid]["em"] == 1 and by_b[qid]["em"] == 0]
    return {
        "em_new_minus_old": paired_bootstrap(em_b, em_a, seed=42),
        "f1_new_minus_old": paired_bootstrap(f1_b, f1_a, seed=42),
        "mcnemar": _mcnemar(em_a, em_b),
        "improved_qids": improved,
        "degraded_qids": degraded,
    }


def _visibility_transitions(
    old_inputs: List[Dict[str, Any]],
    new_inputs: List[Dict[str, Any]],
    old_scores: List[Dict[str, Any]],
    new_scores: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Stratify paired score changes by old/new passage answer visibility."""
    by_old_input = {row["qid"]: row for row in old_inputs}
    by_new_input = {row["qid"]: row for row in new_inputs}
    by_old_score = {row["qid"]: row for row in old_scores}
    by_new_score = {row["qid"]: row for row in new_scores}
    if not (
        list(by_old_input)
        == list(by_new_input)
        == list(by_old_score)
        == list(by_new_score)
    ):
        raise ValueError("visibility-transition qid order differs")

    buckets: Dict[str, List[str]] = {
        "visible_to_visible": [],
        "hidden_to_visible": [],
        "visible_to_hidden": [],
        "hidden_to_hidden": [],
    }
    for qid in by_old_input:
        old_visible = bool(by_old_input[qid].get("gold_in_passages"))
        new_visible = bool(by_new_input[qid].get("gold_in_passages"))
        old_label = "visible" if old_visible else "hidden"
        new_label = "visible" if new_visible else "hidden"
        buckets[f"{old_label}_to_{new_label}"].append(qid)

    result: Dict[str, Any] = {}
    for name, bucket_qids in buckets.items():
        old_em = [float(by_old_score[qid]["em"]) for qid in bucket_qids]
        new_em = [float(by_new_score[qid]["em"]) for qid in bucket_qids]
        old_f1 = [float(by_old_score[qid]["f1"]) for qid in bucket_qids]
        new_f1 = [float(by_new_score[qid]["f1"]) for qid in bucket_qids]
        result[name] = {
            "n": len(bucket_qids),
            "old_em": sum(old_em) / len(old_em) if old_em else None,
            "new_em": sum(new_em) / len(new_em) if new_em else None,
            "em_delta": (
                sum(b - a for a, b in zip(old_em, new_em)) / len(old_em)
                if old_em
                else None
            ),
            "old_f1": sum(old_f1) / len(old_f1) if old_f1 else None,
            "new_f1": sum(new_f1) / len(new_f1) if new_f1 else None,
            "f1_delta": (
                sum(b - a for a, b in zip(old_f1, new_f1)) / len(old_f1)
                if old_f1
                else None
            ),
            "qids": bucket_qids,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    for name in ("e0_sft", "e0_combined", "e1_sft", "e1_combined"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--e2_sft")
    parser.add_argument("--e2_combined")
    parser.add_argument("--retrieval_report", required=True)
    parser.add_argument("--kg_report")
    parser.add_argument(
        "--scope",
        default="fixed validation cohort; zero-training paired diagnostic",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    run_dir = Path(args.run_dir).resolve()
    for path in (output, run_dir):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing path: {path}")

    cohort_path = Path(args.cohort).resolve()
    cohort = _read(cohort_path)
    qids = [str(row.get("qid") or "") for row in cohort]
    if len(qids) != len(set(qids)) or any(not qid for qid in qids):
        raise SystemExit("invalid cohort qids")
    if bool(args.e2_sft) != bool(args.e2_combined):
        raise SystemExit("provide both --e2_sft and --e2_combined, or neither")
    arm_names = ["e0_sft", "e0_combined", "e1_sft", "e1_combined"]
    if args.e2_sft:
        arm_names.extend(["e2_sft", "e2_combined"])
    arms: Dict[str, List[Dict[str, Any]]] = {}
    sources: Dict[str, Any] = {}
    for name in arm_names:
        path = Path(getattr(args, name)).resolve()
        all_rows = _read(path)
        by_qid = {row["qid"]: row for row in all_rows}
        missing = [qid for qid in qids if qid not in by_qid]
        if missing:
            raise SystemExit(f"{name} missing qids: {missing}")
        arms[name] = [by_qid[qid] for qid in qids]
        sources[name] = {"path": str(path), "sha256": _sha256(path)}

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "status": "COMPLETE",
        "scope": args.scope,
        "protocol": {
            "training": "none",
            "fold": "val",
            "decode": "greedy",
            "max_new_tokens": 512,
            "prompt_passages": 15,
            "e0": "stored passages + stored KG",
            "e1": "retrieval-v2 passages + stored KG",
            "qid_sha256": hashlib.sha256("\n".join(qids).encode()).hexdigest(),
        },
        "cohort": {"path": str(cohort_path), "sha256": _sha256(cohort_path), "n": len(qids)},
        "source_files": sources,
        "retrieval_report": {
            "path": str(Path(args.retrieval_report).resolve()),
            "sha256": _sha256(Path(args.retrieval_report).resolve()),
        },
        "arms": {name: _metrics(rows) for name, rows in arms.items()},
        "paired": {
            "sft_e1_vs_e0": _paired(arms["e0_sft"], arms["e1_sft"]),
            "combined_e1_vs_e0": _paired(arms["e0_combined"], arms["e1_combined"]),
            "combined_vs_sft_under_e1": _paired(arms["e1_sft"], arms["e1_combined"]),
        },
        "scientific_verdict": "RESEARCHER_DECISION_REQUIRED",
    }
    report["visibility_transitions"] = {
        "sft_e1_vs_e0": _visibility_transitions(
            arms["e0_sft"], arms["e1_sft"], arms["e0_sft"], arms["e1_sft"]
        ),
        "combined_e1_vs_e0": _visibility_transitions(
            arms["e0_combined"],
            arms["e1_combined"],
            arms["e0_combined"],
            arms["e1_combined"],
        ),
    }
    if args.e2_sft:
        report["protocol"]["e2"] = (
            "retrieval-v2 passages + offline passage-aware KG"
        )
        if not args.kg_report:
            raise SystemExit("--kg_report is required when E2 arms are provided")
        report["kg_report"] = {
            "path": str(Path(args.kg_report).resolve()),
            "sha256": _sha256(Path(args.kg_report).resolve()),
        }
        report["paired"].update(
            {
                "sft_e2_vs_e1": _paired(arms["e1_sft"], arms["e2_sft"]),
                "combined_e2_vs_e1": _paired(
                    arms["e1_combined"], arms["e2_combined"]
                ),
                "combined_vs_sft_under_e2": _paired(
                    arms["e2_sft"], arms["e2_combined"]
                ),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(
        run_dir,
        extra={
            "experiment_id": args.experiment_id,
            "phase": "zero_training_retrieval_eval",
            "report": str(output),
            "report_sha256": _sha256(output),
            "qid_sha256": report["protocol"]["qid_sha256"],
            "n": len(qids),
        },
    )
    print(json.dumps({"arms": report["arms"], "paired": report["paired"]}, indent=2))


if __name__ == "__main__":
    main()
