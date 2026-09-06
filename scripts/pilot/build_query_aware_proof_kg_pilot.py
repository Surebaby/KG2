#!/usr/bin/env python
"""Build and audit a gold-free query-planned proof KG on the frozen train150 cohort.

Construction uses only question text.  It can use either the historical local
subgraph cache or an explicit, isolated targeted-property cache.  Dataset
evidence is loaded afterwards for coverage audit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.kg.kg_filter import _pid_for_triple
from kgproweight.kg.query_planner import PLANNER_VERSION, QueryPlan, plan_question
from kgproweight.kg.question_kg import make_question_kg_record
from kgproweight.kg.wikidata_retriever import _QA_RELATION_FILTER, WikidataSubgraphRetriever
from kgproweight.kg.wikidata_property_retriever import WikidataPropertyRetriever
from kgproweight.kg.wikipedia_title_resolver import (
    WikipediaTitleResolver,
    complete_question_surface_title,
)
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path, resolve_kg_cache_dir
from kgproweight.utils.logging import dump_manifest
from scripts.pilot.audit_query_aware_kg_coverage import (
    _chain_summary,
    _expected_explicit_anchors,
    _norm,
    _reference_hops,
    _surface_match,
)


Triple = Tuple[str, str, str]
BUILDER_VERSION = "query-aware-proof-kg-3-edge-aware"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _link_surface(
    linker: EntityLinker,
    surface: str,
    question: str,
    title_resolver: WikipediaTitleResolver | None = None,
) -> Dict[str, Any]:
    completed_surface = complete_question_surface_title(surface, question)
    result = title_resolver.resolve(completed_surface) if title_resolver else None
    if result is None or result.abstained:
        # Parentheticals help exact Wikipedia lookup but often reduce
        # Wikidata full-text search recall. The fallback remains question-only.
        result = linker.link_single(surface, question=question)
    return {
        "surface": surface,
        "resolved_surface": completed_surface,
        "qid": result.selected_qid,
        "label": result.selected_label,
        "score": round(float(result.score), 4),
        "margin": round(float(result.margin), 4),
        "abstained": bool(result.abstained),
        "abstain_reason": result.abstain_reason,
    }


def _head_matches(triple: Triple, entity: Mapping[str, Any]) -> bool:
    labels = [entity.get("label"), entity.get("surface")]
    return any(value and _surface_match(triple[0], value) for value in labels)


def build_local_proof_kg(
    question: str,
    plan: QueryPlan,
    linker: EntityLinker,
    retriever: Any,
    *,
    max_values_per_hop: int = 3,
    max_total: int = 12,
    title_resolver: WikipediaTitleResolver | None = None,
) -> Tuple[List[Triple], Dict[str, Any]]:
    """Execute a plan against local cached properties without gold access."""
    anchor_entities = {
        anchor: _link_surface(linker, anchor, question, title_resolver)
        for anchor in plan.anchors
    }
    slots: Dict[str, List[Dict[str, Any]]] = {}
    triples: List[Triple] = []
    seen: set[Triple] = set()
    hop_diagnostics: List[Dict[str, Any]] = []

    for hop_index, hop in enumerate(plan.hops, start=1):
        if hop.subject.startswith("$"):
            subjects = slots.get(hop.subject[1:], [])
        else:
            entity = anchor_entities.get(hop.subject)
            subjects = [entity] if entity else []
        subjects = [
            entity for entity in subjects
            if entity.get("qid") and not entity.get("abstained")
        ]
        matches: List[Triple] = []
        tail_qids: Dict[Triple, str] = {}
        triple_sources: Dict[Triple, str] = {}
        output_entities: List[Dict[str, Any]] = []
        for entity in subjects:
            edge_aware = hasattr(retriever, "fetch_edges")
            targeted = edge_aware or hasattr(retriever, "fetch_properties")
            if edge_aware:
                edges = retriever.fetch_edges(str(entity["qid"]), list(hop.pids))
                raw = [
                    (str(edge["head_label"]), str(edge["relation"]), str(edge["tail_value"]))
                    for edge in edges
                ]
                for edge, triple in zip(edges, raw):
                    if edge.get("tail_qid"):
                        tail_qids[tuple(triple)] = str(edge["tail_qid"])
                    if edge.get("source"):
                        triple_sources.setdefault(tuple(triple), str(edge["source"]))
            elif targeted:
                raw = retriever.fetch_properties(str(entity["qid"]), list(hop.pids))
            else:
                raw = retriever.fetch([str(entity["qid"])])
            for value in raw:
                triple = tuple(value)
                if _pid_for_triple(triple) not in set(hop.pids):
                    continue
                # A targeted-property response is already scoped to this QID;
                # aliases/redirect labels need not string-match the mention.
                if not targeted and not _head_matches(triple, entity):
                    continue
                if triple not in matches:
                    matches.append(triple)
                if len(matches) >= max_values_per_hop:
                    break
            if len(matches) >= max_values_per_hop:
                break
        for triple in matches:
            if triple not in seen and len(triples) < max_total:
                seen.add(triple)
                triples.append(triple)
            if hop.relation_role == "bridge":
                tail_qid = tail_qids.get(triple)
                if tail_qid:
                    output_entities.append(
                        {
                            "surface": triple[2], "qid": tail_qid, "label": triple[2],
                            "score": 1.0, "margin": 1.0, "abstained": False,
                            "abstain_reason": "",
                        }
                    )
                else:
                    output_entities.append(
                        _link_surface(linker, triple[2], question, title_resolver)
                    )
        slots[hop.output_slot] = output_entities
        hop_diagnostics.append(
            {
                "hop_index": hop_index,
                "subject": hop.subject,
                "pids": list(hop.pids),
                "input_entities": subjects,
                "matches": [list(triple) for triple in matches],
                "match_sources": [triple_sources.get(triple, "") for triple in matches],
                "output_entities": output_entities,
            }
        )
    return triples, {
        "anchor_entities": anchor_entities,
        "hops": hop_diagnostics,
        "n_triples": len(triples),
        "complete_plan_execution": bool(plan.hops) and all(row["matches"] for row in hop_diagnostics),
    }


def _relation_recall(plan: QueryPlan, references: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    plan_pids = Counter(pid for hop in plan.hops for pid in hop.pids)
    remaining = plan_pids.copy()
    evaluable = [hop for hop in references if hop["target"].get("pids")]
    hits = 0
    for hop in evaluable:
        matched = next((pid for pid in hop["target"]["pids"] if remaining[pid] > 0), None)
        if matched:
            hits += 1
            remaining[matched] -= 1
    return {
        "evaluable_reference_hops": len(evaluable),
        "hit_reference_hops": hits,
        "recall": hits / len(evaluable) if evaluable else None,
        "plan_pids": dict(plan_pids),
    }


def _aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter()
    ref_hops = ref_hits = expected_anchors = planned_expected_anchors = linked_planned_expected = 0
    for row in rows:
        counts["questions"] += 1
        counts["plan_recognized"] += int(row["query_plan"]["recognized"])
        counts["nonempty_proof_kg"] += int(bool(row["kg_subgraph"]))
        counts["complete_plan_execution"] += int(row["execution"]["complete_plan_execution"])
        counts[f"operation::{row['query_plan']['operation']}"] += 1
        reference = row["relation_plan_audit"]
        ref_hops += reference["evaluable_reference_hops"]
        ref_hits += reference["hit_reference_hops"]
        expected_anchors += len(row["expected_explicit_anchors"])
        planned_expected_anchors += len(row["planned_expected_explicit_anchors"])
        linked_planned_expected += len(row["linked_planned_expected_explicit_anchors"])
        if row["proof_chain_audit"]["evaluable"]:
            counts["proof_evaluable"] += 1
            counts["full_relation_value_chain"] += int(
                row["proof_chain_audit"]["all_relation_value_hit"] is True
            )
    n = max(1, counts["questions"])
    return {
        "counts": dict(counts),
        "rates": {
            "plan_recognized": counts["plan_recognized"] / n,
            "nonempty_proof_kg": counts["nonempty_proof_kg"] / n,
            "complete_plan_execution": counts["complete_plan_execution"] / n,
            "reference_relation_recall": ref_hits / max(1, ref_hops),
            "expected_explicit_anchor_in_plan_recall": planned_expected_anchors / max(1, expected_anchors),
            "expected_explicit_anchor_linked_from_plan_recall": linked_planned_expected / max(1, expected_anchors),
            "full_relation_value_chain_rate_evaluable": counts["full_relation_value_chain"] / max(1, counts["proof_evaluable"]),
        },
        "denominators": {
            "reference_hops": ref_hops,
            "expected_explicit_anchors": expected_anchors,
            "proof_evaluable_questions": counts["proof_evaluable"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--cohort_manifest", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--entity_index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--details", required=True)
    parser.add_argument("--question_kg", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument(
        "--retriever_backend",
        choices=("legacy_local", "targeted_properties"),
        default="legacy_local",
    )
    parser.add_argument("--property_cache")
    parser.add_argument("--entity_cache")
    parser.add_argument("--online_entity_linking", action="store_true")
    parser.add_argument("--online_properties", action="store_true")
    parser.add_argument("--title_cache")
    parser.add_argument("--online_wikipedia_titles", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort_path = Path(args.cohort).resolve()
    cohort_manifest_path = Path(args.cohort_manifest).resolve()
    entity_index_path = Path(args.entity_index).resolve()
    output_path = Path(args.output).resolve()
    details_path = Path(args.details).resolve()
    question_kg_path = Path(args.question_kg).resolve()
    run_dir = Path(args.run_dir).resolve()
    for target in (output_path, details_path, question_kg_path, run_dir):
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing path: {target}")

    cohort = _read_jsonl(cohort_path)
    cohort_manifest = json.loads(cohort_manifest_path.read_text(encoding="utf-8"))
    if _sha256(cohort_path) != cohort_manifest["cohort"]["sha256"] or len(cohort) != 150:
        raise SystemExit("cohort does not match frozen n=150 manifest")
    selected_ids: Dict[str, set[str]] = defaultdict(set)
    for row in cohort:
        selected_ids[row["dataset"]].add(row["source_id"])
    source_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    data_root = Path(args.data_root).resolve()
    source_inputs: Dict[str, Dict[str, str]] = {}
    for dataset, ids in selected_ids.items():
        path = data_root / dataset / "train.jsonl"
        for row in _read_jsonl(path):
            if str(row.get("id")) in ids:
                source_rows[(dataset, str(row["id"]))] = row
        source_inputs[dataset] = {"path": str(path), "sha256": _sha256(path)}

    if args.online_entity_linking and not args.entity_cache:
        raise SystemExit("--online_entity_linking requires an isolated --entity_cache")
    if args.online_wikipedia_titles and not args.title_cache:
        raise SystemExit("--online_wikipedia_titles requires an isolated --title_cache")
    linker = EntityLinker(
        cache_path=args.entity_cache or resolve_entity_cache_path(),
        offline=not args.online_entity_linking,
        entity_index_path=str(entity_index_path),
    )
    if args.retriever_backend == "targeted_properties":
        if not args.property_cache:
            raise SystemExit("targeted_properties requires --property_cache")
        retriever = WikidataPropertyRetriever(
            cache_path=args.property_cache,
            offline=not args.online_properties,
        )
    else:
        if args.online_properties:
            raise SystemExit("--online_properties is only valid with targeted_properties")
        retriever = WikidataSubgraphRetriever(
            max_hops=2,
            max_neighbors=30,
            relation_filter=_QA_RELATION_FILTER,
            cache_dir=resolve_kg_cache_dir(),
            offline=True,
            include_literal_values=False,
        )
    title_resolver = None
    if args.title_cache:
        title_resolver = WikipediaTitleResolver(
            cache_path=args.title_cache,
            offline=not args.online_wikipedia_titles,
        )

    records: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    for index, frozen in enumerate(cohort, start=1):
        source = source_rows[(frozen["dataset"], frozen["source_id"])]
        question = frozen["question"]

        # Gold-free boundary: plan and proof KG are complete before references.
        plan = plan_question(question)
        triples, execution = build_local_proof_kg(
            question, plan, linker, retriever, title_resolver=title_resolver
        )
        record = make_question_kg_record(
            dataset=frozen["dataset"],
            qid=frozen["source_id"],
            question=question,
            triples=triples,
            query_plan=plan.to_dict(),
            provenance={
                "builder_version": BUILDER_VERSION,
                "entity_index_sha256": _sha256(entity_index_path),
                "retriever_backend": args.retriever_backend,
                "literal_values_available": args.retriever_backend == "targeted_properties",
            },
        )
        records.append(record)

        references = _reference_hops(frozen["dataset"], source)
        expected = _expected_explicit_anchors(frozen["dataset"], source, question, references)
        planned_expected = [
            anchor for anchor in expected if any(_surface_match(anchor, value) for value in plan.anchors)
        ]
        linked_expected = [
            anchor for anchor in planned_expected
            if any(
                link.get("qid")
                and not link.get("abstained")
                and _surface_match(link.get("label"), anchor)
                for link in execution["anchor_entities"].values()
            )
        ]
        details.append(
            {
                **record,
                "gold_answers": frozen["gold_answers"],
                "stratum": frozen["stratum"],
                "execution": execution,
                "reference_hops": references,
                "relation_plan_audit": _relation_recall(plan, references),
                "expected_explicit_anchors": expected,
                "planned_expected_explicit_anchors": planned_expected,
                "linked_planned_expected_explicit_anchors": linked_expected,
                "proof_chain_audit": _chain_summary(references, triples),
            }
        )
        if index % 25 == 0:
            print(f"built {index}/150")

    overall = _aggregate(details)
    by_dataset = {
        dataset: _aggregate([row for row in details if row["dataset"] == dataset])
        for dataset in sorted(selected_ids)
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "status": "COMPLETE_GOLD_FREE_BUILD_POST_BUILD_AUDIT",
        "scope": "frozen train150; gold-free construction; post-build gold audit; no inference/training",
        "protocol": {
            "planner": PLANNER_VERSION,
            "identity": "dataset::qid plus question_sha256",
            "entity_index": "expanded local subgraph roots plus optional isolated online cache",
            "retriever_backend": args.retriever_backend,
            "online_entity_linking": bool(args.online_entity_linking),
            "online_properties": bool(args.online_properties),
            "online_wikipedia_titles": bool(args.online_wikipedia_titles),
            "tail_qid_propagation": hasattr(retriever, "fetch_edges"),
            "subgraph": (
                "exact planned QID/PID properties with literal values"
                if args.retriever_backend == "targeted_properties"
                else "legacy local cache; literal values unavailable"
            ),
            "max_values_per_hop": 3,
            "max_total_triples": 12,
            "gold_used_for_build": False,
            "gold_used_for_post_build_audit": True,
        },
        "inputs": {
            "cohort": {"path": str(cohort_path), "sha256": _sha256(cohort_path)},
            "cohort_manifest": {"path": str(cohort_manifest_path), "sha256": _sha256(cohort_manifest_path)},
            "entity_index": {"path": str(entity_index_path), "sha256": _sha256(entity_index_path)},
            "datasets": source_inputs,
        },
        "overall": overall,
        "by_dataset": by_dataset,
        "required_plan_pids": dict(
            Counter(pid for row in details for hop in row["query_plan"]["hops"] for pid in hop["pids"])
        ),
        "known_asset_blocker": (
            "planner/entity linking coverage; targeted property cache is literal-safe"
            if args.retriever_backend == "targeted_properties"
            else "legacy SPARQL/cache excludes literal tails; P569/P570/P571/P577 values unavailable"
        ),
    }
    if args.retriever_backend == "legacy_local":
        legacy_cache = Path(resolve_kg_cache_dir()) / "kg_subgraph_cache.jsonl"
        report["inputs"]["kg_subgraph_cache"] = {
            "path": str(legacy_cache), "sha256": _sha256(legacy_cache)
        }
    else:
        property_cache = Path(args.property_cache).resolve()
        report["inputs"]["property_cache"] = {
            "path": str(property_cache),
            "sha256": _sha256(property_cache) if property_cache.exists() else None,
        }
        edge_cache = Path(f"{property_cache}.edges.jsonl")
        report["inputs"]["property_edge_cache"] = {
            "path": str(edge_cache),
            "sha256": _sha256(edge_cache) if edge_cache.exists() else None,
        }
        if args.entity_cache:
            entity_cache = Path(args.entity_cache).resolve()
            report["inputs"]["entity_cache"] = {
                "path": str(entity_cache),
                "sha256": _sha256(entity_cache) if entity_cache.exists() else None,
            }
        if args.title_cache:
            title_cache = Path(args.title_cache).resolve()
            report["inputs"]["title_cache"] = {
                "path": str(title_cache),
                "sha256": _sha256(title_cache) if title_cache.exists() else None,
            }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with question_kg_path.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with details_path.open("w", encoding="utf-8") as fh:
        for row in details:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["outputs"] = {
        "question_kg": {"path": str(question_kg_path), "sha256": _sha256(question_kg_path)},
        "details": {"path": str(details_path), "sha256": _sha256(details_path)},
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(
        run_dir,
        extra={
            "experiment_id": args.experiment_id,
            "phase": "query_aware_local_proof_kg_pilot",
            "scope": report["scope"],
            "protocol": report["protocol"],
            "inputs": report["inputs"],
            "outputs": report["outputs"],
        },
    )
    print(json.dumps({"overall": overall, "by_dataset": by_dataset, "required_plan_pids": report["required_plan_pids"]}, indent=2))


if __name__ == "__main__":
    main()
