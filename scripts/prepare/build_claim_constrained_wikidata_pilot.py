#!/usr/bin/env python
"""Build answer-free, passage-constrained standard Wikidata triples.

This development-only pilot fixes the two artificial ceilings in the previous
executor: it scans real claims on an exactly resolved historical item instead
of trusting one generated PID, and it uses only the already-frozen retrieved
passages to disambiguate claim tails.  It never reads dataset source rows,
answers, supporting-fact labels, or decomposition answers.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, Iterable, Mapping

import requests

from kgproweight.kg.claim_constrained_wikidata import (
    SELECTOR_VERSION,
    select_claim_edges,
    tail_support,
)
from kgproweight.kg.entity_linker import (
    DEFAULT_PROXY_HEADERS,
    WIKIDATA_SEARCH_URL,
    WIKIDATA_USER_AGENT,
    build_passage_text,
    build_passage_titles,
)
from kgproweight.kg.historical_wikidata_retriever import (
    HistoricalWikidataPropertyRetriever,
)
from kgproweight.kg.question_kg import make_question_kg_record
from kgproweight.kg.wikidata_property_retriever import _PID_TO_RELATION
from kgproweight.kg.wikipedia_title_resolver import WikipediaTitleResolver
from kgproweight.utils.logging import dump_manifest
from scripts.pilot.build_automatic_proofkg_from_plans import (
    _clean_anchor,
    convert_predicted_target,
)


ROOT = Path(__file__).resolve().parents[2]
CUTOFF = "2020-12-09T23:59:59Z"
BUILDER_VERSION = "claim-constrained-historical-wikidata-1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"frozen input hash mismatch: {path}: {actual} != {expected}")


def _normalise_title(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _merge_seed_jsonl(target: Path, seeds: list[Path], key_fields: tuple[str, ...]) -> None:
    if target.exists():
        return
    values: dict[tuple[str, ...], dict[str, Any]] = {}
    for seed in seeds:
        if not seed.exists():
            continue
        for row in _read_jsonl(seed):
            key = tuple(str(row.get(field) or "") for field in key_fields)
            if all(key):
                values.setdefault(key, row)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(target, [values[key] for key in sorted(values)])


class _PassageLabelResolver:
    def __init__(self) -> None:
        self.qid_to_title: dict[str, str] = {}

    def add(self, qid: str, title: str) -> None:
        self.qid_to_title.setdefault(str(qid), str(title))

    def label_for_qid(self, qid: str) -> str | None:
        return self.qid_to_title.get(str(qid))


class _PropertyLabelResolver:
    """Append-only cache for display labels; factual values still come from cutoff revisions."""

    def __init__(self, path: Path, *, delay: float, timeout: float, retries: int) -> None:
        self.path = path
        self.delay = delay
        self.timeout = timeout
        self.retries = retries
        self.labels = dict(_PID_TO_RELATION)
        if path.exists():
            for row in _read_jsonl(path):
                if row.get("pid") and row.get("label"):
                    self.labels[str(row["pid"])] = str(row["label"])

    def resolve(self, pid: str) -> str:
        if pid in self.labels:
            return self.labels[pid]
        last_error = ""
        for attempt in range(1, self.retries + 1):
            try:
                response = requests.get(
                    WIKIDATA_SEARCH_URL,
                    params={
                        "action": "wbgetentities", "ids": pid, "props": "labels",
                        "languages": "en", "format": "json",
                    },
                    headers={"User-Agent": WIKIDATA_USER_AGENT, **DEFAULT_PROXY_HEADERS},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                entity = (response.json().get("entities") or {}).get(pid) or {}
                label = str((((entity.get("labels") or {}).get("en") or {}).get("value") or pid))
                self.labels[pid] = label
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "pid": pid, "label": label,
                        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                        "usage": "display_and_relation_matching_only; not an edge value",
                    }, ensure_ascii=False) + "\n")
                time.sleep(self.delay)
                return label
            except (requests.RequestException, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries:
                    time.sleep(self.delay * attempt)
        self.labels[pid] = pid
        return pid


def _passage_index(
    passages: list[Mapping[str, Any]],
    resolver: WikipediaTitleResolver,
    label_resolver: _PassageLabelResolver,
) -> tuple[dict[str, str], set[str], list[dict[str, Any]]]:
    by_surface: dict[str, str] = {}
    qids: set[str] = set()
    diagnostics: list[dict[str, Any]] = []
    for title in dict.fromkeys(build_passage_titles(passages)):
        result = resolver.resolve(title)
        diagnostics.append({
            "title": title, "qid": result.selected_qid,
            "abstained": result.abstained, "reason": result.abstain_reason,
        })
        if result.selected_qid and not result.abstained:
            by_surface[_normalise_title(title)] = result.selected_qid
            qids.add(result.selected_qid)
            label_resolver.add(result.selected_qid, title)
    return by_surface, qids, diagnostics


def _relation_texts(predicted: Mapping[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    for index, step in enumerate(predicted.get("steps") or [], start=1):
        slot = str(step.get("output_slot") or f"hop_{index}")
        relation = str(step.get("relation_label") or "").strip()
        template = str(step.get("subquery_template") or "").strip()
        if not relation and ">>" in template:
            relation = template.rpartition(">>")[2].strip()
        if not relation:
            relation = template
        values[slot] = relation
    return values


def _resolve_subject(
    surface: str,
    *,
    slots: Mapping[str, str],
    passage_by_surface: Mapping[str, str],
    resolver: WikipediaTitleResolver,
    label_resolver: _PassageLabelResolver,
) -> tuple[str | None, str]:
    if surface.startswith("$"):
        qid = slots.get(surface[1:])
        return qid, "dependency" if qid else "unresolved_dependency"
    exact = passage_by_surface.get(_normalise_title(surface))
    if exact:
        return exact, "passage_title_exact"
    result = resolver.resolve(surface)
    if result.selected_qid and not result.abstained:
        label_resolver.add(result.selected_qid, result.selected_label or surface)
        return result.selected_qid, "wikipedia_title"
    return None, result.abstain_reason or "unresolved_subject"


def _build_dataset(
    *,
    dataset: str,
    plans: list[dict[str, Any]],
    retrieval: Mapping[tuple[str, str], dict[str, Any]],
    cache_dir: Path,
    seed_historical_cache: Path,
    title_seed_caches: list[Path],
    delay: float,
    timeout: float,
    retries: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    title_cache = cache_dir / dataset / "title_cache.jsonl"
    history_cache = cache_dir / dataset / "historical_property_cache.jsonl"
    property_cache = cache_dir / dataset / "property_labels.jsonl"
    _merge_seed_jsonl(title_cache, title_seed_caches, ("label",))
    if not history_cache.exists():
        history_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(seed_historical_cache, history_cache)

    title_resolver = WikipediaTitleResolver(
        cache_path=title_cache, offline=False, timeout=timeout,
        max_retries=retries, request_delay=delay,
    )
    passage_labels = _PassageLabelResolver()
    historical = HistoricalWikidataPropertyRetriever(
        cache_path=history_cache, cutoff=CUTOFF, offline=False,
        timeout=timeout, request_delay=delay, max_retries=retries,
        label_resolver=passage_labels,
    )
    property_labels = _PropertyLabelResolver(
        property_cache, delay=delay, timeout=timeout, retries=retries,
    )

    records: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for row_index, row in enumerate(plans, start=1):
        key = (dataset, str(row["qid"]))
        context = retrieval.get(key)
        if context is None or context.get("question_sha256") != row.get("question_sha256"):
            raise ValueError(f"missing/hash-mismatched frozen retrieval context for {key}")
        passages = list(context.get("passages") or [])[:10]
        passage_blob = build_passage_text(passages, max_n=10)
        passage_by_surface, passage_qids, title_diag = _passage_index(
            passages, title_resolver, passage_labels
        )
        plan, conversion = convert_predicted_target(dataset, str(row["question"]), row.get("predicted_target"))
        relation_by_slot = _relation_texts(row.get("predicted_target") or {})
        slots: dict[str, str] = {}
        selected_all: list[dict[str, Any]] = []
        hop_details: list[dict[str, Any]] = []
        runtime_error = None
        try:
            for hop_index, hop in enumerate(plan.hops, start=1):
                subject_qid, resolution = _resolve_subject(
                    str(hop.subject), slots=slots,
                    passage_by_surface=passage_by_surface,
                    resolver=title_resolver, label_resolver=passage_labels,
                )
                if not subject_qid:
                    hop_details.append({
                        "hop_index": hop_index, "output_slot": hop.output_slot,
                        "status": "ABSTAIN", "reason": resolution,
                    })
                    continue
                claims = historical.fetch_all_edges(subject_qid)
                supported_pids = {
                    str(edge.get("pid") or "") for edge in claims
                    if tail_support(edge, passage_title_qids=passage_qids, passage_blob=passage_blob)
                }
                labels = {pid: property_labels.resolve(pid) for pid in sorted(supported_pids) if pid}
                planned_pid = str(list(hop.pids)[0]) if hop.pids else None
                selected, rejected = select_claim_edges(
                    claims, planned_pid=planned_pid,
                    planned_relation=relation_by_slot.get(hop.output_slot, ""),
                    property_labels=labels,
                    passage_title_qids=passage_qids,
                    passage_blob=passage_blob,
                    max_edges=2,
                )
                for edge in selected:
                    edge["plan_step_index"] = hop_index
                    edge["output_slot"] = hop.output_slot
                    edge["subject_resolution"] = resolution
                selected_all.extend(selected)
                tail_qids = {str(edge.get("tail_qid") or "") for edge in selected if edge.get("tail_qid")}
                if len(tail_qids) == 1:
                    slots[hop.output_slot] = next(iter(tail_qids))
                hop_details.append({
                    "hop_index": hop_index, "output_slot": hop.output_slot,
                    "subject": hop.subject, "subject_qid": subject_qid,
                    "subject_resolution": resolution, "planned_pid": planned_pid,
                    "planned_relation": relation_by_slot.get(hop.output_slot, ""),
                    "n_claims_scanned": len(claims), "selected": selected,
                    "n_rejected": len(rejected),
                    "status": "MATCH" if selected else "ABSTAIN",
                })
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"

        selected_all = selected_all[:12]
        triples = [
            [str(edge["head_label"]), str(edge["relation"]), str(edge["tail_value"])]
            for edge in selected_all
        ]
        complete = bool(plan.hops) and len(hop_details) == len(plan.hops) and all(
            detail.get("status") == "MATCH" for detail in hop_details
        )
        record = make_question_kg_record(
            dataset=dataset, qid=str(row["qid"]), question=str(row["question"]),
            triples=triples, query_plan=plan.to_dict(),
            provenance={
                "builder_version": BUILDER_VERSION,
                "selector_version": SELECTOR_VERSION,
                "gold_access": False,
                "historical_cutoff": CUTOFF,
                "complete_plan_execution": complete,
                "all_edges_historical_claim_verified": all(
                    edge.get("source_revision_id") and edge.get("source_cutoff") == CUTOFF
                    for edge in selected_all
                ),
                "all_edges_passage_tail_supported": all(edge.get("tail_support") for edge in selected_all),
            },
        )
        records.append(record)
        details.append({
            **record, "planner_schema_valid": bool(row.get("schema_valid")),
            "conversion": conversion, "passage_title_resolution": title_diag,
            "selected_edges": selected_all, "execution": {"hops": hop_details},
            "runtime_error": runtime_error,
        })
        counters["n"] += 1
        counters["plan_recognized"] += int(plan.recognized)
        counters["nonempty"] += int(bool(triples))
        counters["complete"] += int(complete)
        counters["runtime_errors"] += int(bool(runtime_error))
        counters["edges"] += len(triples)
        print(f"{dataset}: built {row_index}/{len(plans)}", flush=True)
    return records, details, dict(counters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output_dir}")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    frozen = protocol["frozen_inputs"]
    paths = {name: ROOT / value["path"] for name, value in frozen.items()}
    for name, path in paths.items():
        _assert_hash(path, frozen[name]["sha256"])

    retrieval_rows = _read_jsonl(paths["retrieval_contexts"])
    retrieval = {(str(row["dataset"]), str(row["qid"])): row for row in retrieval_rows}
    plan_paths = {
        "hotpotqa": paths["hotpot_plans"],
        "musique": paths["musique_plans"],
    }
    history_seeds = {
        "hotpotqa": paths["hotpot_seed_historical_cache"],
        "musique": paths["musique_seed_historical_cache"],
    }
    title_seeds = [
        ROOT / "indexes/inference_proofkg_hotpot_passage_v1/hotpot_title_cache.jsonl",
        ROOT / "indexes/inference_proofkg_v1_pilot30/title_cache.jsonl",
    ]

    all_records: list[dict[str, Any]] = []
    all_details: list[dict[str, Any]] = []
    by_dataset: dict[str, Any] = {}
    for dataset in ("hotpotqa", "musique"):
        records, details, counts = _build_dataset(
            dataset=dataset, plans=_read_jsonl(plan_paths[dataset]), retrieval=retrieval,
            cache_dir=args.cache_dir, seed_historical_cache=history_seeds[dataset],
            title_seed_caches=title_seeds, delay=args.delay,
            timeout=args.timeout, retries=args.retries,
        )
        all_records.extend(records)
        all_details.extend(details)
        by_dataset[dataset] = counts

    args.output_dir.mkdir(parents=True)
    _write_jsonl(args.output_dir / "question_kg_records.jsonl", all_records)
    _write_jsonl(args.output_dir / "runtime_details.jsonl", all_details)
    report = {
        "schema_version": "claim-constrained-wikidata-pilot-report-1",
        "experiment_id": protocol["experiment_id"],
        "status": "COMPLETE_DEVELOPMENT_ONLY_NOT_UTILITY_TESTED",
        "builder_version": BUILDER_VERSION,
        "selector_version": SELECTOR_VERSION,
        "historical_cutoff": CUTOFF,
        "gold_access": False,
        "by_dataset": by_dataset,
        "structural_gate": {
            "identity_hash_join": 1.0,
            "runtime_errors": sum(v["runtime_errors"] for v in by_dataset.values()),
            "historical_claim_verified_rate": (
                sum(1 for row in all_details for edge in row["selected_edges"] if edge.get("source_revision_id") and edge.get("source_cutoff") == CUTOFF)
                / max(1, sum(len(row["selected_edges"]) for row in all_details))
            ),
            "passage_tail_support_rate": (
                sum(1 for row in all_details for edge in row["selected_edges"] if edge.get("tail_support"))
                / max(1, sum(len(row["selected_edges"]) for row in all_details))
            ),
            "max_triples_per_question": max((len(row["kg_subgraph"]) for row in all_records), default=0),
        },
        "inputs": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()},
        "outputs": {
            "question_kg_records_sha256": _sha256(args.output_dir / "question_kg_records.jsonl"),
            "runtime_details_sha256": _sha256(args.output_dir / "runtime_details.jsonl"),
        },
        "next_gate": "same-passages zero-training legacy vs legacy+trusted-Wikidata",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.output_dir, status=report["status"], extra=report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
