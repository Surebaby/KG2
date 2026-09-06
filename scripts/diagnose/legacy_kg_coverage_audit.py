#!/usr/bin/env python3
"""Legacy KG Coverage Audit: Diagnose where information loss occurs.

Performs layered bottleneck analysis without training:
  1. Raw candidates: Does the cached subgraph contain answer-related entities/relations?
  2. QID linking: Are entities correctly linked to their Wikidata QIDs?
  3. Relation coverage: Do raw triples contain the target relations?
  4. Top-K filtering: Do useful edges survive filter_and_rank_triples?
  5. Prompt injection: Are triples preserved in the final prompt?
  6. Model utilization: Does the model cite the available KG?

Output: Per-question diagnosis pointing to the primary bottleneck.

Usage:
    python scripts/diagnose/legacy_kg_coverage_audit.py \\
      --datasets hotpotqa musique \\
      --n_samples 100 \\
      --seed 46 \\
      --split dev \\
      --kg_index indexes/kg_cache/question_kg_index_v2.json \\
      --output reports/legacy_kg_bottleneck_audit_n100_seed46.json

Requirements:
  - Gold answers/supporting_facts only used AFTER KG construction for audit
  - No modifications to existing KG indices
  - All bottleneck classifications are deterministic and reproducible
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from kgproweight.kg.cache import EntityCache, SubgraphCache
from kgproweight.kg.entity_linker import EntityLinker, extract_mentions
from kgproweight.kg.kg_filter import filter_and_rank_triples, hard_delete_triple
from kgproweight.kg.wikidata_retriever import WikidataSubgraphRetriever, _QA_RELATION_FILTER
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path, resolve_kg_cache_dir
from kgproweight.utils.logging import get_logger
from kgproweight.utils.paths import data_dir

logger = get_logger(__name__)

AUDIT_VERSION = "legacy-kg-audit-v1"


@dataclass
class LayerDiagnosis:
    """Diagnosis for each layer of the KG pipeline."""
    # Layer 1: Raw knowledge source
    raw_has_answer_mentions: bool = False
    raw_answer_entity_count: int = 0
    raw_total_entities: int = 0
    raw_total_triples: int = 0

    # Layer 2: QID linking
    mentions_extracted: int = 0
    mentions_linked: int = 0
    mentions_abstained: int = 0
    qid_linking_quality: float = 0.0  # linked / extracted

    # Layer 3: Relation/value coverage
    raw_has_target_relations: bool = False
    target_relation_types: List[str] = field(default_factory=list)
    raw_matched_relation_count: int = 0

    # Layer 4: Top-K filtering
    top12_triple_count: int = 0
    top12_has_useful_edges: bool = False
    top12_answer_mention_count: int = 0

    # Layer 5: Prompt injection (placeholder for now)
    prompt_kg_available: bool = False
    prompt_kg_truncated: bool = False

    # Layer 6: Model utilization (requires inference, marked as TODO)
    model_cited_kg: Optional[bool] = None
    model_citation_count: Optional[int] = None

    # Bottleneck classification
    bottleneck_layer: str = "UNKNOWN"
    bottleneck_reason: str = ""
    repair_strategy: str = ""


@dataclass
class QuestionAudit:
    """Complete audit record for one question."""
    dataset: str
    qid: str
    question: str
    answer: str

    # Input state
    legacy_kg_available: bool
    legacy_kg_triple_count: int

    # Layered diagnosis
    layers: LayerDiagnosis

    # Summary
    bottleneck: str
    repair_potential: str

    # Provenance
    audit_version: str = AUDIT_VERSION
    seed: int = 42


def load_dataset_questions(
    dataset: str,
    split: str = "dev",
    n_samples: Optional[int] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Load questions from dataset split with optional sampling."""
    dataset_path = data_dir() / dataset / f"{split}.jsonl"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    questions = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    logger.info(f"Loaded {len(questions)} questions from {dataset}/{split}")

    if n_samples and n_samples < len(questions):
        random.seed(seed)
        questions = random.sample(questions, n_samples)
        logger.info(f"Sampled {n_samples} questions with seed={seed}")

    return questions


def load_legacy_kg_index(index_path: str) -> Dict[str, Any]:
    """Load existing KG index (question -> triples mapping)."""
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    # Handle both list and dict formats
    if isinstance(index_data, list):
        # Convert list format to dict keyed by question
        index = {entry.get("question", ""): entry for entry in index_data if entry.get("question")}
        logger.info(f"Converted list-format KG index to dict ({len(index)} entries)")
    elif isinstance(index_data, dict):
        index = index_data
    else:
        raise ValueError(f"Unexpected KG index format: {type(index_data)}")

    logger.info(f"Loaded KG index with {len(index)} entries from {index_path}")
    return index


def infer_target_relations(question: str, answer: str, dataset: str) -> List[str]:
    """Heuristically infer what relations the question is asking about.

    This is ONLY for audit/diagnosis. Returns relation types, not specific PIDs.
    Gold answer is used AFTER KG construction to evaluate coverage.
    """
    q_lower = question.lower()
    relations = []

    # Temporal relations
    if any(word in q_lower for word in ["when", "date", "year", "born", "died", "founded", "established"]):
        relations.extend(["temporal", "date_of_birth", "date_of_death", "inception"])

    # Location relations
    if any(word in q_lower for word in ["where", "located", "place", "city", "country"]):
        relations.extend(["location", "place_of_birth", "located_in", "country"])

    # Identity/occupation relations
    if any(word in q_lower for word in ["who", "director", "writer", "actor", "founder", "president"]):
        relations.extend(["identity", "occupation", "position_held", "creator", "director"])

    # Comparison relations
    if any(word in q_lower for word in ["same", "both", "share", "common", "compare"]):
        relations.extend(["comparison", "country_of_citizenship", "occupation"])

    # Property relations
    if any(word in q_lower for word in ["what", "which", "type", "genre", "language"]):
        relations.extend(["property", "instance_of", "genre", "language"])

    return list(set(relations)) if relations else ["general"]


def check_answer_in_triples(
    triples: List[Tuple[str, str, str]],
    answer: str
) -> Tuple[bool, int]:
    """Check if answer text appears in any triple component.

    Returns (has_answer_mention, count_of_mentions).
    """
    import string
    answer_lower = answer.lower()
    # Normalize: remove punctuation for word-based matching
    answer_normalized = answer_lower.translate(str.maketrans('', '', string.punctuation))
    answer_words = set(answer_normalized.split())

    mention_count = 0
    for h, r, t in triples:
        triple_text = f"{h} {r} {t}".lower()
        # Check for exact substring match or significant word overlap
        if answer_lower in triple_text:
            mention_count += 1
        elif len(answer_words) > 1:
            # Normalize triple text the same way
            triple_normalized = triple_text.translate(str.maketrans('', '', string.punctuation))
            triple_words = set(triple_normalized.split())
            overlap = len(answer_words & triple_words)
            if overlap >= len(answer_words) * 0.6:  # 60% word overlap
                mention_count += 1

    return mention_count > 0, mention_count


def check_relation_coverage(
    triples: List[Tuple[str, str, str]],
    target_relations: List[str]
) -> Tuple[bool, int]:
    """Check if triples contain any of the target relation types.

    This is a coarse-grained check based on relation labels.
    """
    if not target_relations or not triples:
        return False, 0

    matched = 0
    for h, r, t in triples:
        r_lower = r.lower()
        for target in target_relations:
            target_lower = target.replace("_", " ").lower()
            if target_lower in r_lower or r_lower in target_lower:
                matched += 1
                break

    return matched > 0, matched


def diagnose_question(
    question_data: Dict[str, Any],
    dataset: str,
    legacy_kg_index: Dict[str, Any],
    linker: EntityLinker,
    kg_retriever: WikidataSubgraphRetriever,
    max_mentions: int = 5,
) -> QuestionAudit:
    """Perform layered diagnosis for one question."""

    qid = question_data["id"]
    question = question_data["question"]
    answer = question_data["golden_answers"][0] if question_data.get("golden_answers") else ""

    diagnosis = LayerDiagnosis()

    # Check if legacy KG is available
    legacy_entry = legacy_kg_index.get(question)
    legacy_available = legacy_entry is not None
    legacy_triples = legacy_entry.get("triples", []) if legacy_available else []
    legacy_triple_count = len(legacy_triples)

    diagnosis.prompt_kg_available = legacy_available

    # Layer 1: Raw candidates
    mentions = extract_mentions(question, max_n=max_mentions)
    diagnosis.mentions_extracted = len(mentions)

    linked_qids = []
    abstained_count = 0

    for mention in mentions:
        result = linker.link_single(mention, question=question)
        if result.selected_qid and not result.abstained:
            linked_qids.append(result.selected_qid)
        if result.abstained:
            abstained_count += 1

    diagnosis.mentions_linked = len(linked_qids)
    diagnosis.mentions_abstained = abstained_count
    diagnosis.qid_linking_quality = len(linked_qids) / max(len(mentions), 1)

    # Fetch raw triples from cache
    raw_triples = kg_retriever.fetch(linked_qids) if linked_qids else []
    diagnosis.raw_total_triples = len(raw_triples)
    diagnosis.raw_total_entities = len(linked_qids)

    # Check if answer appears in raw triples
    has_answer, answer_count = check_answer_in_triples(raw_triples, answer)
    diagnosis.raw_has_answer_mentions = has_answer
    diagnosis.raw_answer_entity_count = answer_count

    # Layer 3: Relation coverage
    target_relations = infer_target_relations(question, answer, dataset)
    diagnosis.target_relation_types = target_relations

    has_relations, relation_count = check_relation_coverage(raw_triples, target_relations)
    diagnosis.raw_has_target_relations = has_relations
    diagnosis.raw_matched_relation_count = relation_count

    # Layer 4: Top-K filtering
    if raw_triples:
        filtered_triples = filter_and_rank_triples(
            raw_triples,
            question=question,
            max_keep=12,
            min_keep=5
        )
        diagnosis.top12_triple_count = len(filtered_triples)

        # Check if useful edges survived
        has_answer_top12, answer_top12_count = check_answer_in_triples(filtered_triples, answer)
        diagnosis.top12_has_useful_edges = has_answer_top12
        diagnosis.top12_answer_mention_count = answer_top12_count

    # Bottleneck classification
    bottleneck, reason, strategy = classify_bottleneck(diagnosis)
    diagnosis.bottleneck_layer = bottleneck
    diagnosis.bottleneck_reason = reason
    diagnosis.repair_strategy = strategy

    return QuestionAudit(
        dataset=dataset,
        qid=qid,
        question=question,
        answer=answer,
        legacy_kg_available=legacy_available,
        legacy_kg_triple_count=legacy_triple_count,
        layers=diagnosis,
        bottleneck=bottleneck,
        repair_potential=strategy,
        seed=0  # Will be set by caller
    )


def classify_bottleneck(diagnosis: LayerDiagnosis) -> Tuple[str, str, str]:
    """Classify the primary bottleneck layer.

    Returns: (bottleneck_layer, reason, repair_strategy)
    """

    # Layer 1: No entities extracted or linked
    if diagnosis.mentions_extracted == 0:
        return ("L1_NO_MENTIONS",
                "No entity mentions extracted from question",
                "improve_mention_extraction")

    if diagnosis.mentions_linked == 0:
        return ("L2_LINKING_FAILURE",
                "All entity mentions failed to link to QIDs",
                "fix_entity_linker")

    if diagnosis.qid_linking_quality < 0.4:
        return ("L2_LOW_LINKING_QUALITY",
                f"Only {diagnosis.mentions_linked}/{diagnosis.mentions_extracted} mentions linked successfully",
                "improve_entity_disambiguation")

    # Layer 1: No raw triples
    if diagnosis.raw_total_triples == 0:
        return ("L1_EMPTY_CACHE",
                "No triples in cache for linked entities",
                "expand_kg_cache_coverage")

    # Layer 3: No answer-related content in raw
    if not diagnosis.raw_has_answer_mentions:
        return ("L3_ANSWER_NOT_IN_RAW",
                "Answer entity not found in raw cached triples",
                "passage_derived_required")

    # Layer 3: No target relations in raw
    if not diagnosis.raw_has_target_relations:
        return ("L3_RELATION_MISSING",
                f"Target relations {diagnosis.target_relation_types} not in raw cache",
                "passage_derived_or_expand_cache")

    # Layer 4: Filtering loss
    if diagnosis.top12_triple_count == 0:
        return ("L4_COMPLETE_FILTERING_LOSS",
                "All raw triples filtered out by filter_and_rank",
                "fix_filter_threshold")

    if diagnosis.raw_has_answer_mentions and not diagnosis.top12_has_useful_edges:
        return ("L4_FILTERING_REMOVED_USEFUL",
                "Useful edges present in raw but removed by Top-12 filtering",
                "fix_reranker")

    # Downstream: KG available but something else is wrong
    if diagnosis.top12_has_useful_edges:
        return ("L5_DOWNSTREAM",
                "Useful KG available in Top-12, bottleneck likely in prompt/model",
                "investigate_prompt_or_model")

    return ("UNKNOWN", "Could not classify bottleneck", "manual_investigation")


def generate_report(
    audits: List[QuestionAudit],
    output_path: str,
    args: argparse.Namespace
) -> None:
    """Generate JSON report and markdown summary."""

    # Aggregate statistics
    bottleneck_counts = Counter(a.bottleneck for a in audits)
    strategy_counts = Counter(a.repair_potential for a in audits)

    dataset_bottlenecks = defaultdict(lambda: Counter())
    for audit in audits:
        dataset_bottlenecks[audit.dataset][audit.bottleneck] += 1

    # Compute layer pass rates
    layer_stats = {
        "L1_mentions_extracted": sum(1 for a in audits if a.layers.mentions_extracted > 0),
        "L2_mentions_linked": sum(1 for a in audits if a.layers.mentions_linked > 0),
        "L2_linking_quality_high": sum(1 for a in audits if a.layers.qid_linking_quality >= 0.6),
        "L1_raw_triples_available": sum(1 for a in audits if a.layers.raw_total_triples > 0),
        "L3_answer_in_raw": sum(1 for a in audits if a.layers.raw_has_answer_mentions),
        "L3_relations_in_raw": sum(1 for a in audits if a.layers.raw_has_target_relations),
        "L4_top12_available": sum(1 for a in audits if a.layers.top12_triple_count > 0),
        "L4_useful_in_top12": sum(1 for a in audits if a.layers.top12_has_useful_edges),
    }

    # Generate manifest
    manifest = {
        "experiment_id": f"LEGACY_KG_AUDIT_{args.datasets}_{args.split}_N{len(audits)}_SEED{args.seed}".replace(" ", "_").upper(),
        "audit_version": AUDIT_VERSION,
        "datasets": args.datasets,
        "split": args.split,
        "total_questions": len(audits),
        "seed": args.seed,
        "kg_index_path": args.kg_index,
        "generation_date": None,  # TODO: add timestamp

        "bottleneck_distribution": dict(bottleneck_counts),
        "repair_strategy_distribution": dict(strategy_counts),
        "dataset_breakdown": {ds: dict(counts) for ds, counts in dataset_bottlenecks.items()},
        "layer_pass_rates": layer_stats,

        "key_findings": {
            "primary_bottleneck": bottleneck_counts.most_common(1)[0][0] if bottleneck_counts else "NONE",
            "repair_feasibility": {
                "legacy_rerank_fixable": strategy_counts.get("fix_reranker", 0) + strategy_counts.get("fix_filter_threshold", 0),
                "entity_linking_fixable": strategy_counts.get("fix_entity_linker", 0) + strategy_counts.get("improve_entity_disambiguation", 0),
                "passage_derived_required": strategy_counts.get("passage_derived_required", 0) + strategy_counts.get("passage_derived_or_expand_cache", 0),
            }
        }
    }

    # Write JSON output
    output = {
        "manifest": manifest,
        "audits": [asdict(a) for a in audits]
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Wrote audit results to {output_file}")

    # Write markdown summary
    md_path = output_file.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Legacy KG Coverage Audit Report\n\n")
        f.write(f"**Experiment ID**: {manifest['experiment_id']}\n\n")
        f.write(f"**Datasets**: {', '.join(args.datasets)}\n\n")
        f.write(f"**Split**: {args.split}\n\n")
        f.write(f"**Sample size**: {len(audits)} questions (seed={args.seed})\n\n")
        f.write(f"**KG Index**: `{args.kg_index}`\n\n")

        f.write(f"## Bottleneck Distribution\n\n")
        f.write(f"| Bottleneck | Count | Percentage |\n")
        f.write(f"|------------|------:|-----------:|\n")
        for bottleneck, count in bottleneck_counts.most_common():
            pct = 100.0 * count / len(audits)
            f.write(f"| {bottleneck} | {count} | {pct:.1f}% |\n")

        f.write(f"\n## Repair Strategy Distribution\n\n")
        f.write(f"| Strategy | Count | Percentage |\n")
        f.write(f"|----------|------:|-----------:|\n")
        for strategy, count in strategy_counts.most_common():
            pct = 100.0 * count / len(audits)
            f.write(f"| {strategy} | {count} | {pct:.1f}% |\n")

        f.write(f"\n## Layer Pass Rates\n\n")
        f.write(f"| Layer | Passed | Rate |\n")
        f.write(f"|-------|-------:|-----:|\n")
        for layer, passed in layer_stats.items():
            rate = 100.0 * passed / len(audits)
            f.write(f"| {layer} | {passed}/{len(audits)} | {rate:.1f}% |\n")

        f.write(f"\n## Key Findings\n\n")
        f.write(f"**Primary bottleneck**: {manifest['key_findings']['primary_bottleneck']}\n\n")

        fixability = manifest['key_findings']['repair_feasibility']
        f.write(f"**Repair feasibility**:\n")
        f.write(f"- Legacy rerank/filter fixable: {fixability['legacy_rerank_fixable']} questions\n")
        f.write(f"- Entity linking fixable: {fixability['entity_linking_fixable']} questions\n")
        f.write(f"- Passage-derived required: {fixability['passage_derived_required']} questions\n")

        f.write(f"\n## Per-Dataset Breakdown\n\n")
        for ds in args.datasets:
            if ds in dataset_bottlenecks:
                f.write(f"### {ds}\n\n")
                f.write(f"| Bottleneck | Count |\n")
                f.write(f"|------------|------:|\n")
                for bottleneck, count in dataset_bottlenecks[ds].most_common():
                    f.write(f"| {bottleneck} | {count} |\n")
                f.write(f"\n")

    logger.info(f"✓ Wrote markdown summary to {md_path}")

    # Print summary to console
    print("\n" + "="*80)
    print(f"LEGACY KG COVERAGE AUDIT SUMMARY")
    print("="*80)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Sample: {len(audits)} questions (seed={args.seed})")
    print(f"\nPrimary bottleneck: {manifest['key_findings']['primary_bottleneck']}")
    print(f"\nTop 3 bottlenecks:")
    for i, (bottleneck, count) in enumerate(bottleneck_counts.most_common(3), 1):
        pct = 100.0 * count / len(audits)
        print(f"  {i}. {bottleneck}: {count} ({pct:.1f}%)")

    print(f"\nRepair feasibility:")
    for strategy, count in fixability.items():
        pct = 100.0 * count / len(audits)
        print(f"  - {strategy}: {count} ({pct:.1f}%)")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", required=True,
                       help="Datasets to audit (e.g., hotpotqa musique)")
    parser.add_argument("--n_samples", type=int, default=None,
                       help="Sample size per dataset (None = use all)")
    parser.add_argument("--seed", type=int, default=46,
                       help="Random seed for sampling")
    parser.add_argument("--split", default="dev",
                       help="Dataset split to use")
    parser.add_argument("--kg_index", required=True,
                       help="Path to legacy KG index (e.g., indexes/kg_cache/question_kg_index_v2.json)")
    parser.add_argument("--output", required=True,
                       help="Output path for audit JSON")
    parser.add_argument("--max_mentions", type=int, default=5,
                       help="Max entity mentions to extract per question")
    parser.add_argument("--offline", action="store_true", default=True,
                       help="Cache-only mode (default, no live Wikidata calls)")
    parser.add_argument("--online", dest="offline", action="store_false",
                       help="Allow live Wikidata calls for cache misses")

    args = parser.parse_args()

    logger.info(f"Starting legacy KG coverage audit")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Sample size: {args.n_samples or 'ALL'} per dataset")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"KG index: {args.kg_index}")

    # Load components
    logger.info("Loading entity linker and KG retriever...")
    linker = EntityLinker(cache_path=resolve_entity_cache_path(), offline=args.offline)
    kg_retriever = WikidataSubgraphRetriever(
        max_hops=2,
        max_neighbors=30,
        cache_dir=resolve_kg_cache_dir(),
        offline=args.offline,
        relation_filter=_QA_RELATION_FILTER
    )

    logger.info("Loading legacy KG index...")
    legacy_kg_index = load_legacy_kg_index(args.kg_index)

    # Process each dataset
    all_audits = []
    for dataset in args.datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {dataset}")
        logger.info(f"{'='*60}")

        questions = load_dataset_questions(dataset, args.split, args.n_samples, args.seed)

        dataset_audits = []
        for i, q_data in enumerate(questions, 1):
            if i % 10 == 0:
                logger.info(f"  Processed {i}/{len(questions)} questions...")

            audit = diagnose_question(
                q_data,
                dataset,
                legacy_kg_index,
                linker,
                kg_retriever,
                max_mentions=args.max_mentions
            )
            audit.seed = args.seed
            dataset_audits.append(audit)

        all_audits.extend(dataset_audits)
        logger.info(f"✓ Completed {dataset}: {len(dataset_audits)} audits")

    # Generate report
    logger.info(f"\nGenerating audit report...")
    generate_report(all_audits, args.output, args)

    logger.info(f"\n✓ Audit complete: {len(all_audits)} total questions audited")
    logger.info(f"✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
