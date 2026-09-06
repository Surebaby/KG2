"""Strict loaders and arm routing for versioned SAEG evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


EVAL_SCHEMA = "saeg-eval-input-v1"
ARMS = {"A_no_graph", "B_passage", "C_wikidata", "D_fused"}


def iter_saeg_eval_inputs(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != EVAL_SCHEMA:
                raise ValueError("unexpected SAEG evaluation schema")
            if row.get("gold_access") is not False:
                raise ValueError("SAEG inference input must be answer-free")
            if any(field in row for field in ("answer", "answers", "golden_answers")):
                raise ValueError("Gold answer field found in SAEG inference input")
            yield row


def route_eval_arm(row: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Return model-visible sources for one frozen arm.

    Empty passage graphs fail closed to no-graph and remain in the paired
    population.  The W-only arm is different: it is defined only on a dataset
    cohort whose Wikidata branch passed its frozen structural gate.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown SAEG arm: {arm}")
    passage = list(row.get("passage_evidence") or [])
    wikidata = list(row.get("wikidata_kg") or [])
    if arm == "A_no_graph":
        passage, wikidata = [], []
    elif arm == "B_passage":
        wikidata = []
    elif arm == "C_wikidata":
        if str((row.get("source_status") or {}).get("wikidata")) not in {"nonempty", "empty_fail_closed"}:
            raise ValueError("Wikidata arm is not structurally eligible for this cohort")
        passage = []
    # D_fused uses all eligible, already fail-closed sources in the master row.
    return {
        "question": str(row["question"]),
        "retrieved_passages": list(row.get("passages") or []),
        "kg_triples": wikidata,
        "passage_evidence": passage,
        "fallback_no_graph": not passage and not wikidata,
    }


def assert_role_allowed(
    row: Mapping[str, Any],
    *,
    allow_confirmation: bool = False,
    allow_reporting: bool = False,
) -> None:
    role = str(row.get("role") or "")
    if role == "confirmation" and not allow_confirmation:
        raise PermissionError("SAEG confirmation is sealed; a frozen development gate must open it once")
    if role == "reporting_only_nonconfirmatory" and not allow_reporting:
        raise PermissionError("canonical reporting cannot be used for tuning or checkpoint selection")
    if role not in {"development", "confirmation", "reporting_only_nonconfirmatory"}:
        raise ValueError(f"unknown SAEG evaluation role: {role}")
