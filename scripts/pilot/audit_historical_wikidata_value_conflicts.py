#!/usr/bin/env python
"""Audit whether a historical Wikidata revision recovers 2Wiki value conflicts.

This is a post-freeze diagnostic: it deliberately opens 2Wiki Gold and must
never be used to build runtime KG inputs for the audited questions.  Raw API
responses are retained so every classification is reproducible.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, Mapping, Sequence

import requests

from kgproweight.kg.entity_linker import WIKIDATA_USER_AGENT
from kgproweight.kg.kg_filter import _RELATION_LABEL_TO_PID
from kgproweight.kg.wikidata_property_retriever import _literal_value
from kgproweight.utils.logging import artifact_identity, dump_manifest
from scripts.pilot.audit_query_aware_kg_coverage import _value_match


API_URL = "https://www.wikidata.org/w/api.php"
AUDIT_VERSION = "historical-wikidata-conflict-audit-1"
_QID = re.compile(r"^Q[1-9][0-9]*$")


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_claim_values(entity: Mapping[str, Any], pid: str) -> list[Dict[str, str | None]]:
    values: list[Dict[str, str | None]] = []
    for claim in (entity.get("claims") or {}).get(pid) or []:
        if claim.get("rank") == "deprecated":
            continue
        snak = claim.get("mainsnak") or {}
        if snak.get("snaktype") != "value":
            continue
        datavalue = snak.get("datavalue") or {}
        raw = datavalue.get("value")
        if datavalue.get("type") == "wikibase-entityid" and isinstance(raw, Mapping):
            qid = str(raw.get("id") or "")
            if qid:
                values.append({"tail_qid": qid, "literal": None})
        else:
            literal = _literal_value(datavalue)
            if literal:
                values.append({"tail_qid": None, "literal": literal})
    return values


def reference_matches(
    values: Sequence[Mapping[str, str | None]],
    *,
    expected_id: str,
    expected_label: str,
) -> bool:
    expected_qid = expected_id if _QID.fullmatch(expected_id) else ""
    for value in values:
        if expected_qid and value.get("tail_qid") == expected_qid:
            return True
        literal = str(value.get("literal") or "")
        if literal and (
            _value_match(literal, expected_id) or _value_match(literal, expected_label)
        ):
            return True
    return False


def _fetch_revision(qid: str, cutoff: str, timeout: float) -> Dict[str, Any]:
    response = requests.get(
        API_URL,
        params={
            "action": "query",
            "prop": "revisions",
            "titles": qid,
            "rvstart": cutoff,
            "rvdir": "older",
            "rvlimit": "1",
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
            "format": "json",
            "formatversion": "2",
        },
        headers={"User-Agent": WIKIDATA_USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _revision_entity(payload: Mapping[str, Any]) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
    pages = ((payload.get("query") or {}).get("pages") or [])
    revision = (pages[0].get("revisions") or [None])[0] if pages else None
    if not revision:
        return None, {}
    slot = (revision.get("slots") or {}).get("main") or {}
    content = slot.get("content")
    if not isinstance(content, str):
        return None, {}
    try:
        entity = json.loads(content)
    except json.JSONDecodeError:
        return None, {
            "revid": revision.get("revid"),
            "parentid": revision.get("parentid"),
            "timestamp": revision.get("timestamp"),
            "content_parse_error": True,
        }
    return entity, {
        "revid": revision.get("revid"),
        "parentid": revision.get("parentid"),
        "timestamp": revision.get("timestamp"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime_details", required=True)
    parser.add_argument("--audit_details", required=True)
    parser.add_argument("--official_train", required=True)
    parser.add_argument("--cutoff", default="2020-12-09T23:59:59Z")
    parser.add_argument("--request_delay", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--seed_raw_dir",
        help="Optional prior partial raw_revisions directory; responses are copied, never modified.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    args = parser.parse_args()

    runtime_path = Path(args.runtime_details).resolve()
    audit_path = Path(args.audit_details).resolve()
    official_path = Path(args.official_train).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing path: {output_dir}")
    output_dir.mkdir(parents=True)
    raw_dir = output_dir / "raw_revisions"
    raw_dir.mkdir()

    runtime = {str(row["qid"]): row for row in _read_jsonl(runtime_path)}
    audits = {str(row["qid"]): row for row in _read_jsonl(audit_path)}
    official = {
        str(row["_id"]): row for row in json.loads(official_path.read_text(encoding="utf-8"))
        if str(row["_id"]) in audits
    }
    requests_needed: Dict[str, Dict[str, Any]] = {}
    details: list[Dict[str, Any]] = []
    for qid, audit in audits.items():
        proof = audit["proof_chain_audit"]
        run = runtime[qid]
        if not run["execution"].get("complete_plan_execution") or proof.get("all_relation_value_hit"):
            continue
        source = official[qid]
        facts = list(source.get("evidences") or [])
        ids = list(source.get("evidences_id") or [])
        if len(facts) != len(ids):
            continue
        for index, hop_audit in enumerate(proof.get("hops") or []):
            if hop_audit.get("relation_value_hit") or index >= len(ids):
                continue
            fact, id_fact = facts[index], ids[index]
            relation = str(fact[1]).strip().casefold()
            pid = _RELATION_LABEL_TO_PID.get(relation)
            if not pid:
                continue
            expected_head_qid = str(id_fact[0]).strip().upper()
            runtime_hop = (run["execution"].get("hops") or [])[index]
            runtime_input_qids = [
                str(value.get("qid") or "") for value in runtime_hop.get("input_entities") or []
            ]
            detail = {
                "qid": qid,
                "question": source.get("question"),
                "hop_index": index + 1,
                "pid": pid,
                "relation": fact[1],
                "expected_head_label": fact[0],
                "expected_head_qid": expected_head_qid,
                "expected_tail_label": str(fact[2]),
                "expected_tail_id": str(id_fact[2]).strip().upper(),
                "runtime_input_qids": runtime_input_qids,
                "runtime_matches": runtime_hop.get("matches") or [],
                "runtime_head_qid_match": expected_head_qid in runtime_input_qids,
            }
            details.append(detail)
            requests_needed.setdefault(expected_head_qid, {})

    fetched: Dict[str, tuple[Dict[str, Any] | None, Dict[str, Any], str | None]] = {}
    seed_raw_dir = Path(args.seed_raw_dir).resolve() if args.seed_raw_dir else None
    reused_raw = 0
    for index, qid in enumerate(sorted(requests_needed), start=1):
        error = None
        payload: Dict[str, Any] = {}
        seed_path = seed_raw_dir / f"{qid}.json" if seed_raw_dir else None
        if seed_path and seed_path.is_file():
            try:
                payload = json.loads(seed_path.read_text(encoding="utf-8"))
                reused_raw += 1
            except (json.JSONDecodeError, OSError) as exc:
                error = f"seed {type(exc).__name__}: {exc}"
        if not payload:
            try:
                payload = _fetch_revision(qid, args.cutoff, args.timeout)
                error = None
            except (requests.RequestException, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        raw_path = raw_dir / f"{qid}.json"
        raw_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        entity, revision = _revision_entity(payload) if payload else (None, {})
        fetched[qid] = (entity, revision, error)
        print(f"fetched {index}/{len(requests_needed)} {qid}", flush=True)
        if args.request_delay:
            time.sleep(args.request_delay)

    counts: Counter[str] = Counter()
    for detail in details:
        entity, revision, error = fetched[detail["expected_head_qid"]]
        values = extract_claim_values(entity or {}, detail["pid"])
        historical_match = reference_matches(
            values,
            expected_id=detail["expected_tail_id"],
            expected_label=detail["expected_tail_label"],
        )
        if not detail["runtime_head_qid_match"]:
            category = "runtime_head_qid_mismatch"
        elif error or entity is None:
            category = "historical_revision_unavailable"
        elif historical_match:
            category = "historical_snapshot_recovers_reference"
        elif not values:
            category = "historical_property_absent"
        else:
            category = "dataset_conflicts_with_historical_wikidata"
        detail.update({
            "historical_revision": revision,
            "historical_values": values,
            "historical_reference_match": historical_match,
            "historical_error": error,
            "category": category,
        })
        counts[category] += 1
        counts["mismatched_hops"] += 1

    details_path = output_dir / "details.jsonl"
    with details_path.open("x", encoding="utf-8") as handle:
        for detail in details:
            handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "audit_version": AUDIT_VERSION,
        "status": "COMPLETE_POSTFREEZE_DIAGNOSTIC",
        "cutoff": args.cutoff,
        "scientific_boundary": {
            "gold_access": True,
            "runtime_kg_build_allowed": False,
            "prior_gate_decision_change_allowed": False,
        },
        "counts": dict(counts),
        "unique_historical_qids_requested": len(requests_needed),
        "raw_responses_reused_from_prior_partial_run": reused_raw,
        "inputs": {
            "runtime_details": artifact_identity(runtime_path),
            "audit_details": artifact_identity(audit_path),
            "official_train": artifact_identity(official_path),
            "seed_raw_dir": str(seed_raw_dir) if seed_raw_dir else None,
        },
        "outputs": {
            "details": artifact_identity(details_path),
            "raw_revision_dir": str(raw_dir),
            "raw_revision_file_count": len(list(raw_dir.glob("*.json"))),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(output_dir, status=report["status"], extra=report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
