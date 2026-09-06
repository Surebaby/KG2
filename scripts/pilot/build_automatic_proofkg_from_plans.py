#!/usr/bin/env python
"""Execute frozen question-only planner outputs into per-question Proof-KG records.

This command never opens dataset source files or gold annotations.  It resolves
question-surface anchors to QIDs and fetches only the exact PIDs requested by
the frozen plan.  Ambiguous or unsupported plan steps abstain instead of being
replaced with broad, noisy KG retrieval.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from kgproweight.kg.cache import EntityCache
from kgproweight.kg.entity_linker import EntityLinker, LinkResult
from kgproweight.kg.kg_filter import _RELATION_LABEL_TO_PID
from kgproweight.kg.query_planner import QueryHop, QueryPlan
from kgproweight.kg.question_kg import make_question_kg_record
from kgproweight.kg.wikidata_property_retriever import WikidataPropertyRetriever
from kgproweight.kg.wikipedia_title_resolver import WikipediaTitleResolver
from kgproweight.kg.versioned_evidence_store import VersionedEvidenceStore
from kgproweight.kg.historical_wikidata_retriever import HistoricalWikidataPropertyRetriever
from kgproweight.kg.store_first_combined_retriever import StoreFirstCombinedRetriever
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir
from scripts.pilot.build_query_aware_proof_kg_pilot import build_local_proof_kg


BUILDER_VERSION = "automatic-proofkg-from-learned-plan-4-store-first-combined"
_PID = re.compile(r"^P[1-9][0-9]*$")


class _ResolverChain:
    def __init__(self, *resolvers: Any) -> None:
        self.resolvers = [resolver for resolver in resolvers if resolver is not None]

    def resolve(self, surface: str):
        last = None
        for resolver in self.resolvers:
            last = resolver.resolve(surface)
            if not last.abstained and last.selected_qid:
                return last
        return last


class _ExactEntityCacheLinker:
    """Question-root linker restricted to exact entries in one explicit cache.

    This is intentionally narrower than :class:`EntityLinker`: it performs no
    fuzzy matching, no local candidate-index lookup, no network access, and no
    hard-coded entity override.  Clean ProofKG materialisations use it after a
    separately audited root-resolution stage so an unresolved root fails
    closed instead of inheriting an unrelated legacy QID.
    """

    def __init__(self, cache_path: str | Path) -> None:
        self.cache = EntityCache(cache_path)

    def link_single(self, mention: str, **_kwargs: Any) -> LinkResult:
        qid = self.cache.get(mention)
        if not qid:
            return LinkResult(
                mention=mention,
                abstained=True,
                abstain_reason="exact isolated entity cache miss",
            )
        return LinkResult(
            mention=mention,
            selected_qid=qid,
            selected_label=mention,
            score=1.0,
            margin=1.0,
        )


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_anchor(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value).strip(" ,.?\"'"))
    return re.sub(r"\s*#\d+\s*", " ", text).strip(" ,")


def _relation_pid(label: object) -> str | None:
    clean = re.sub(r"\s+", " ", str(label).strip().casefold())
    direct = _RELATION_LABEL_TO_PID.get(clean)
    if direct:
        return direct
    # Prefer the longest exact relation-label phrase. This is conservative:
    # a weak keyword such as "in" must never select an unrelated PID.
    candidates = [
        (relation, pid)
        for relation, pid in _RELATION_LABEL_TO_PID.items()
        if len(relation) >= 4 and re.search(rf"(?<!\w){re.escape(relation)}(?!\w)", clean)
    ]
    return max(candidates, key=lambda value: len(value[0]))[1] if candidates else None


# Frozen deterministic NL-subquery template dictionary (MuSiQue executor/conversion v2).
# Each entry maps a conservative question pattern to an already-supported PID.
# Only semantically clear relations are mapped; comparison / count / time-limit /
# multi-input / ambiguous relations are deliberately NOT in this dictionary and
# therefore abstain. Entity group is the regex group holding the subject surface;
# a "#N" placeholder in that position is replaced by the explicit dependency.
NL_TEMPLATE_DICTIONARY = [
    (re.compile(r"^who\s+founded\s+(.+?)\s*$", re.I), "P112", 1, "nl_founded_by"),
    (re.compile(r"^who\s+(?:is|was)\s+the\s+mother\s+of\s+(.+?)\s*$", re.I), "P25", 1, "nl_mother_of"),
    (re.compile(r"^who\s+issued\s+(.+?)\s*$", re.I), "P50", 1, "nl_issued_by"),
    (re.compile(r"^who\s+played\s+(.+?)\s+in\s+(.+?)\s*$", re.I), "P161", 2, "nl_played_in"),
    (re.compile(r"^who\s+plays\s+(.+?)\s+in\s+(.+?)\s*$", re.I), "P161", 2, "nl_plays_in"),
    (re.compile(r"^what\s+series\s+is\s+(.+?)\s+a\s+part\s+of\s*\??$", re.I), "P179", 1, "nl_part_of_series"),
]


def _match_nl_template(template: str):
    for regex, pid, entity_group, rule in NL_TEMPLATE_DICTIONARY:
        match = regex.match(template.strip())
        if match:
            return pid, match.group(entity_group), rule
    return None


def _roles(steps: list[Mapping[str, Any]]) -> dict[str, str]:
    consumed = {
        str(dependency)
        for step in steps
        for dependency in (step.get("dependencies") or [])
    }
    return {
        str(step.get("output_slot") or ""): (
            "bridge" if str(step.get("output_slot") or "") in consumed else "answer_operand"
        )
        for step in steps
    }


def convert_predicted_target(
    dataset: str,
    question: str,
    predicted: Mapping[str, Any] | None,
) -> tuple[QueryPlan, list[dict[str, Any]]]:
    plan = QueryPlan(
        question_sha256=hashlib.sha256(question.strip().encode("utf-8")).hexdigest(),
        planner_version="learned-query-planner-scale-v1.1+executor-v1",
    )
    diagnostics: list[dict[str, Any]] = []
    if not isinstance(predicted, Mapping):
        plan.abstain_reason = "missing_or_invalid_predicted_target"
        return plan, diagnostics
    steps = list(predicted.get("steps") or [])
    roles = _roles(steps)
    anchors: list[str] = []
    hops: list[QueryHop] = []

    if dataset in {"2wikimultihopqa", "hotpotqa"}:
        anchors = [_clean_anchor(value) for value in predicted.get("anchors") or []]
        anchors = [value for value in anchors if value]
        for index, step in enumerate(steps, start=1):
            # Non-dependency subjects must normalise with the same rule as the
            # anchors, or a QID resolved for anchor "Who" fails to propagate to a
            # subject spelled "Who?". _clean_anchor is a no-op on "$hop_N" refs.
            subject = _clean_anchor(step.get("subject"))
            # Generic syntax normalisation (not per-question): a stray ">>"
            # subquery operator leaking into the subject (e.g. "$hop_1 >> place
            # of birth") is discarded; the subject is the part before ">>".
            if ">>" in subject:
                subject = subject.partition(">>")[0].strip()
            pid = str(step.get("pid") or "").strip().upper()
            slot = str(step.get("output_slot") or f"hop_{index}").strip()
            if not subject or not _PID.fullmatch(pid) or not slot:
                diagnostics.append({"step": index, "status": "ABSTAIN", "reason": "invalid_subject_pid_or_slot"})
                continue
            hops.append(QueryHop(subject, [pid], slot, roles.get(slot, "answer_operand")))
            diagnostics.append({"step": index, "status": "EXECUTABLE", "pid": pid})
    elif dataset == "musique":
        for index, step in enumerate(steps, start=1):
            template = str(step.get("subquery_template") or "").strip()
            dependencies = [str(value) for value in step.get("dependencies") or []]
            slot = str(step.get("output_slot") or f"step_{index}").strip()
            if len(dependencies) > 1:
                diagnostics.append({"step": index, "status": "ABSTAIN", "reason": "multi_input_aggregation"})
                continue
            left, marker, relation = template.rpartition(">>")
            if marker:
                pid = _relation_pid(relation)
                if not pid:
                    diagnostics.append({"step": index, "status": "ABSTAIN", "reason": "unknown_relation_label", "relation": relation.strip()})
                    continue
                rule, entity_surface = "canonical_entity_relation", left
            else:
                # NL subquery (non-canonical): convert only via the frozen
                # deterministic template dictionary; everything else abstains.
                nl = _match_nl_template(template)
                if nl is None:
                    diagnostics.append({"step": index, "status": "ABSTAIN", "reason": "noncanonical_natural_language_operator"})
                    continue
                pid, entity_surface, rule = nl
            if dependencies:
                subject = f"${dependencies[0]}"
            else:
                subject = _clean_anchor(entity_surface)
                if subject and subject not in anchors:
                    anchors.append(subject)
            if not subject:
                diagnostics.append({"step": index, "status": "ABSTAIN", "reason": "missing_anchor"})
                continue
            provenance = {"rule": rule, "original_template": template, "pid": pid, "direction": "forward"}
            hops.append(QueryHop(subject, [pid], slot, roles.get(slot, "answer_operand")))
            diagnostics.append({"step": index, "status": "EXECUTABLE", "pid": pid, "provenance": provenance})
    else:
        plan.abstain_reason = f"unsupported_dataset:{dataset}"
        return plan, diagnostics

    # A slot reference whose producer abstained cannot be executed safely.
    available = set(anchors)
    valid_hops: list[QueryHop] = []
    for hop in hops:
        if hop.subject.startswith("$") and hop.subject[1:] not in available:
            diagnostics.append({"output_slot": hop.output_slot, "status": "ABSTAIN", "reason": "missing_dependency"})
            continue
        valid_hops.append(hop)
        available.add(hop.output_slot)
    plan.anchors = anchors
    plan.hops = valid_hops
    plan.recognized = bool(anchors and valid_hops)
    # HotpotQA reuses the musique subquery-graph conversion but is tracked under
    # a distinct operation label (zero-shot, not a trained capability).
    if plan.recognized:
        plan.operation = (
            "execute_zero_shot_subquery_graph"
            if dataset == "hotpotqa"
            else "execute_learned_relation_graph"
        )
    else:
        plan.operation = "abstain"
    plan.confidence = "model_generated_exact_pid" if dataset == "2wikimultihopqa" else "canonical_relation_only"
    plan.abstain_reason = "" if plan.recognized else "no_safe_executable_hops"
    return plan, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--entity_index", required=True)
    parser.add_argument("--entity_cache", required=True)
    parser.add_argument("--title_cache", required=True)
    parser.add_argument("--property_cache", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument(
        "--scope",
        default="unseen-family n100; question-only automatic Proof-KG; zero training",
    )
    parser.add_argument("--online_titles", action="store_true")
    parser.add_argument("--online_entities", action="store_true")
    parser.add_argument(
        "--exact_entity_cache_only",
        action="store_true",
        help=(
            "Resolve EntityLinker fallbacks only by exact lookup in --entity_cache; "
            "disables hard-coded fixes, fuzzy cache matching, local candidates, and network."
        ),
    )
    parser.add_argument("--online_properties", action="store_true")
    parser.add_argument("--request_delay", type=float, default=0.1)
    parser.add_argument(
        "--versioned_store",
        help=(
            "Optional read-only 2Wiki training-partition evidence store. It replaces "
            "online title/property access while retaining the legacy entity linker as "
            "a question-only fallback on alias misses."
        ),
    )
    parser.add_argument(
        "--versioned_alias_store",
        help="Use a versioned 2Wiki store only as fallback after exact Wikipedia title lookup.",
    )
    parser.add_argument("--historical_property_cache")
    parser.add_argument("--historical_cutoff", default="2020-12-09T23:59:59Z")
    parser.add_argument("--online_historical_properties", action="store_true")
    parser.add_argument(
        "--status",
        default="RUNTIME_KG_FROZEN_BEFORE_GOLD_AUDIT",
        help=(
            "Run status recorded in report.json/manifest. Offline coverage "
            "diagnostics pass OFFLINE_COVERAGE_DIAGNOSTIC; the final offline "
            "rebuild keeps the default."
        ),
    )
    args = parser.parse_args()

    plan_path = Path(args.plans).resolve()
    plans = list(_read_jsonl(plan_path))
    if not plans:
        raise SystemExit("empty planner predictions")
    output_dir, experiment_id = prepare_new_run_dir(
        args.output_dir,
        experiment_id=args.experiment_id,
        extra={
            "phase": "build_question_only_automatic_proofkg",
            "plans": artifact_identity(plan_path),
            "protocol": artifact_identity(args.protocol),
        },
    )
    if args.exact_entity_cache_only:
        if args.online_entities:
            raise SystemExit("--exact_entity_cache_only cannot be mixed with --online_entities")
        linker = _ExactEntityCacheLinker(args.entity_cache)
    else:
        linker = EntityLinker(
            cache_path=args.entity_cache,
            offline=not args.online_entities,
            entity_index_path=args.entity_index,
        )
    if args.versioned_store and (
        args.versioned_alias_store or args.historical_property_cache
    ):
        raise SystemExit("legacy --versioned_store cannot be mixed with the historical hybrid backend")
    if args.versioned_store and any(
        (args.online_titles, args.online_entities, args.online_properties)
    ):
        raise SystemExit("--versioned_store cannot be mixed with online lookup flags")
    versioned_store = (
        VersionedEvidenceStore(args.versioned_store) if args.versioned_store else None
    )
    wikipedia_resolver = WikipediaTitleResolver(
        cache_path=args.title_cache,
        offline=not args.online_titles,
    )
    alias_store = (
        VersionedEvidenceStore(args.versioned_alias_store)
        if args.versioned_alias_store else None
    )
    title_resolver = versioned_store or _ResolverChain(wikipedia_resolver, alias_store)
    if args.historical_property_cache:
        if args.online_properties:
            raise SystemExit("historical properties cannot be mixed with current online properties")
        historical = HistoricalWikidataPropertyRetriever(
            cache_path=args.historical_property_cache,
            cutoff=args.historical_cutoff,
            offline=not args.online_historical_properties,
            request_delay=args.request_delay,
            label_resolver=alias_store,
        )
        # Store-first + historical-fallback (single variable vs the prior
        # replace-the-store behaviour): the versioned alias store still owns
        # property edges, and the historical cache only fills store misses.
        retriever = (
            StoreFirstCombinedRetriever(alias_store, historical)
            if alias_store is not None
            else historical
        )
    else:
        retriever = versioned_store or WikidataPropertyRetriever(
            cache_path=args.property_cache,
            offline=not args.online_properties,
            request_delay=args.request_delay,
        )

    records: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for index, row in enumerate(plans, start=1):
        question = str(row["question"]).strip()
        plan, conversion = convert_predicted_target(
            str(row["dataset"]), question, row.get("predicted_target")
        )
        runtime_error = None
        try:
            triples, execution = build_local_proof_kg(
                question,
                plan,
                linker,
                retriever,
                title_resolver=title_resolver,
            )
        except Exception as exc:  # preserve per-question failure without losing the run
            triples, execution = [], {"anchor_entities": {}, "hops": [], "n_triples": 0, "complete_plan_execution": False}
            runtime_error = f"{type(exc).__name__}: {exc}"
        provenance = {
            "builder_version": BUILDER_VERSION,
            "gold_access": False,
            # Reward routing consumes this input-side execution fact.  It is
            # computed without Gold and prevents partial paths from being
            # mistaken for complete automatic proofs during PPO.
            "complete_plan_execution": bool(
                execution.get("complete_plan_execution")
            ),
            "planner_predictions_sha256": _sha256(plan_path),
        }
        if str(row["dataset"]) == "hotpotqa":
            provenance["planner_mode"] = "zero_shot_subquery_graph"
            provenance["executor_mode"] = "precision_first"
        record = make_question_kg_record(
            dataset=str(row["dataset"]),
            qid=str(row["qid"]),
            question=question,
            triples=triples,
            query_plan=plan.to_dict(),
            provenance=provenance,
        )
        records.append(record)
        details.append(
            {
                **record,
                "row_id": row.get("row_id"),
                "planner_schema_valid": bool(row.get("schema_valid")),
                "conversion": conversion,
                "execution": execution,
                "runtime_error": runtime_error,
            }
        )
        print(f"built {index}/{len(plans)}", flush=True)

    records_path = output_dir / "runtime_question_kg.jsonl"
    details_path = output_dir / "runtime_details.jsonl"
    for path, values in ((records_path, records), (details_path, details)):
        with path.open("x", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    n = len(details)
    counts = {
        "n": n,
        "by_dataset": dict(Counter(str(row["dataset"]) for row in details)),
        "planner_schema_valid": sum(bool(row["planner_schema_valid"]) for row in details),
        "plan_recognized": sum(bool(row["query_plan"]["recognized"]) for row in details),
        "anchor_qid_resolved": sum(
            bool(row["execution"].get("anchor_entities"))
            and all(value.get("qid") and not value.get("abstained") for value in row["execution"]["anchor_entities"].values())
            for row in details
        ),
        "proof_kg_nonempty": sum(bool(row["kg_subgraph"]) for row in details),
        "complete_plan_execution": sum(bool(row["execution"].get("complete_plan_execution")) for row in details),
        "runtime_errors": sum(bool(row["runtime_error"]) for row in details),
    }
    # Retained edges are those that actually entered the ProofKG (deduplicated
    # into hop matches), distinct from the retriever's fetched-edge counts.
    retained_sources: dict[str, int] = {"store": 0, "historical_fallback": 0}
    for row in details:
        for hop in (row.get("execution") or {}).get("hops") or []:
            for src in hop.get("match_sources") or []:
                if src in retained_sources:
                    retained_sources[src] += 1
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "status": args.status,
        "scope": args.scope,
        "builder_version": BUILDER_VERSION,
        "counts": counts,
        "rates": {
            key: value / n for key, value in counts.items()
            if key not in {"n", "by_dataset", "runtime_errors"}
        },
        "inputs": {
            "plans": artifact_identity(plan_path),
            "protocol": artifact_identity(args.protocol),
            "entity_index": artifact_identity(args.entity_index),
            "entity_cache": artifact_identity(args.entity_cache),
            "title_cache": artifact_identity(args.title_cache),
            "versioned_store_manifest": (
                artifact_identity(Path(args.versioned_store or args.versioned_alias_store) / "store_manifest.json")
                if (args.versioned_store or args.versioned_alias_store) else None
            ),
        },
        "cache_policy": {
            "isolated_versioned_caches": True,
            "supply_backend": (
                "store_first_combined_retriever"
                if isinstance(retriever, StoreFirstCombinedRetriever) else
                "historical_wikidata_with_training_alias_fallback"
                if args.historical_property_cache else
                "versioned_2wiki_training_store" if versioned_store else
                "wikidata_property_cache"
            ),
            "fetched_edge_source_counts": (
                dict(retriever.source_counts)
                if isinstance(retriever, StoreFirstCombinedRetriever) else None
            ),
            "retained_match_source_counts": retained_sources,
            "historical_cutoff": args.historical_cutoff if args.historical_property_cache else None,
            "historical_property_cache": (
                artifact_identity(args.historical_property_cache)
                if args.historical_property_cache and Path(args.historical_property_cache).exists()
                else None
            ),
            "online_titles": bool(args.online_titles),
            "online_entities": bool(args.online_entities),
            "online_properties": bool(args.online_properties),
            "exact_entity_cache_only": bool(args.exact_entity_cache_only),
        },
        "outputs": {
            "runtime_question_kg": artifact_identity(records_path),
            "runtime_details": artifact_identity(details_path),
        },
        "gold_access": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(output_dir, status=report["status"], extra=report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
