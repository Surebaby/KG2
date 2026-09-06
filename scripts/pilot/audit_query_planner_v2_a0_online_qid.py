#!/usr/bin/env python
"""Resolve seen A0 question surfaces to QIDs with an isolated append-only cache."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable

import requests

from kgproweight.kg.entity_linker import WIKIDATA_USER_AGENT
from kgproweight.kg.wikipedia_title_resolver import WIKIPEDIA_API_URL, title_variants
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def _follow(title: str, edges: dict[str, str]) -> str:
    seen: set[str] = set()
    while title in edges and title not in seen:
        seen.add(title)
        title = edges[title]
    return title


def _batch_resolve(surfaces: list[str]) -> dict[str, dict[str, Any]]:
    variants_by_surface = {surface: title_variants(surface) for surface in surfaces}
    variants = sorted({value for values in variants_by_surface.values() for value in values})
    edges: dict[str, str] = {}
    pages: dict[str, dict[str, Any]] = {}
    headers = {"User-Agent": WIKIDATA_USER_AGENT}
    proxy_token = os.getenv("KGPW_WIKIDATA_PROXY_TOKEN", "")
    if proxy_token:
        headers["X-Proxy-Token"] = proxy_token
    for chunk in _chunks(variants, 40):
        response = requests.get(
            WIKIPEDIA_API_URL,
            params={
                "action": "query",
                "prop": "pageprops",
                "ppprop": "wikibase_item|disambiguation",
                "titles": "|".join(chunk),
                "redirects": "1",
                "format": "json",
                "formatversion": "2",
            },
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        query = response.json().get("query") or {}
        for item in (query.get("normalized") or []) + (query.get("redirects") or []):
            edges[str(item["from"])] = str(item["to"])
        for page in query.get("pages") or []:
            pages[str(page.get("title") or "")] = page
    pages_folded = {title.casefold(): page for title, page in pages.items()}
    resolved: dict[str, dict[str, Any]] = {}
    for surface, surface_variants in variants_by_surface.items():
        candidates: dict[str, dict[str, Any]] = {}
        for variant in surface_variants:
            title = _follow(variant, edges)
            page = pages.get(title) or pages_folded.get(title.casefold()) or {}
            props = page.get("pageprops") or {}
            qid = props.get("wikibase_item")
            if page.get("missing") or "disambiguation" in props or not qid:
                continue
            candidates[str(qid)] = page
        if len(candidates) == 1:
            qid, page = next(iter(candidates.items()))
            resolved[surface] = {
                "selected_qid": qid,
                "selected_label": str(page.get("title") or surface),
                "resolved": True,
                "abstained": False,
                "abstain_reason": "",
            }
        else:
            resolved[surface] = {
                "selected_qid": None,
                "selected_label": "",
                "resolved": False,
                "abstained": True,
                "abstain_reason": (
                    "no exact non-disambiguation Wikipedia title"
                    if not candidates else "title variants resolved to different QIDs"
                ),
            }
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical_details", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir, experiment_id = prepare_new_run_dir(
        args.output_dir,
        extra={
            "phase": "planner_v2_a0_online_qid_seen_engineering",
            "canonical_details": artifact_identity(args.canonical_details),
        },
    )
    rows = list(_read_jsonl(args.canonical_details))
    surfaces = [str(row["completed_question_surface"]) for row in rows]
    resolutions = _batch_resolve(surfaces)
    details: list[dict[str, Any]] = []
    for row in rows:
        surface = str(row["completed_question_surface"])
        result = resolutions[surface]
        details.append({
            "question_key": row["question_key"],
            "qid": row["qid"],
            "runtime_surface": surface,
            **result,
        })
    cache_path = output_dir / "wikipedia_title_qid_cache.jsonl"
    with cache_path.open("x", encoding="utf-8") as fh:
        for row in details:
            if row["resolved"]:
                fh.write(json.dumps({
                    "label": row["runtime_surface"], "qid": row["selected_qid"]
                }, ensure_ascii=False) + "\n")
    resolved_rate = sum(row["resolved"] for row in details) / len(details) if details else 0.0
    details_path = output_dir / "details.jsonl"
    with details_path.open("x", encoding="utf-8") as fh:
        for row in details:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "status": "PASS_SEEN_ENGINEERING_ONLY" if resolved_rate >= 0.90 else "FAIL_STOP",
        "scope": "seen_confirmation_retrospective_engineering_only",
        "n": len(details),
        "metrics": {
            "qid_resolved_n": sum(row["resolved"] for row in details),
            "qid_resolved_rate": resolved_rate,
            "abstained_n": sum(row["abstained"] for row in details),
        },
        "engineering_gate": {"qid_resolved_rate_min": 0.90, "pass": resolved_rate >= 0.90},
        "scientific_boundary": {
            "independent_confirmation": False,
            "runtime_inputs": ["question-derived completed surface"],
            "uses_answer_or_supporting_fact_at_runtime": False,
            "network_source": "English Wikipedia API current state",
            "cache_isolated": True,
        },
        "inputs": {"canonical_details": artifact_identity(args.canonical_details)},
        "details": artifact_identity(details_path),
        "cache": artifact_identity(cache_path),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(output_dir, status=report["status"], extra=report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
