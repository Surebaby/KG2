#!/usr/bin/env python
"""Apply frozen structural gates to a query-aware KG confirmation audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def adjudicate(
    manifest: Mapping[str, Any], report: Mapping[str, Any], details: Iterable[Mapping[str, Any]]
) -> Dict[str, Any]:
    gates = manifest["preregistered_gates"]
    rates = report["overall"]["rates"]
    measured = {
        "plan_recognized_rate_min": rates["plan_recognized"],
        "reference_relation_recall_min": rates["reference_relation_recall"],
        "expected_explicit_anchor_in_plan_recall_min": rates["expected_explicit_anchor_in_plan_recall"],
        "expected_explicit_anchor_linked_from_plan_recall_min": rates["expected_explicit_anchor_linked_from_plan_recall"],
        "complete_plan_execution_rate_min": rates["complete_plan_execution"],
        "full_relation_value_chain_rate_evaluable_min": rates["full_relation_value_chain_rate_evaluable"],
    }
    strata: Dict[str, list[bool]] = defaultdict(list)
    for row in details:
        strata[str(row["stratum"])].append(bool(row["query_plan"]["recognized"]))
    per_stratum = {
        stratum: sum(values) / len(values) for stratum, values in sorted(strata.items())
    }
    measured["per_stratum_plan_recognized_rate_min"] = min(per_stratum.values())
    checks = {
        name: {
            "threshold": float(gates[name]),
            "measured": float(measured[name]),
            "passed": float(measured[name]) >= float(gates[name]),
        }
        for name in gates
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "decision": "PASS_STRUCTURAL_CONFIRMATION" if passed else "FAIL_STOP_STRUCTURAL_CONFIRMATION",
        "all_gates_passed": passed,
        "checks": checks,
        "per_stratum_plan_recognized_rate": per_stratum,
        "consequence": (
            "eligible to request zero-training paired evaluation"
            if passed
            else "do not expand planner training or run zero-training/PPO from this route; confirmation set may only be reused as development diagnostics"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort_manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--details", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path, report_path, details_path = map(
        lambda value: Path(value).resolve(),
        (args.cohort_manifest, args.report, args.details),
    )
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing path: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["inputs"]["cohort"]["sha256"] != manifest["cohort"]["sha256"]:
        raise SystemExit("audit cohort hash does not match frozen manifest")
    result = adjudicate(manifest, report, _read_jsonl(details_path))
    result.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "cohort_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
                "report": {"path": str(report_path), "sha256": _sha256(report_path)},
                "details": {"path": str(details_path), "sha256": _sha256(details_path)},
            },
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
