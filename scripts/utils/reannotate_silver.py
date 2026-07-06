#!/usr/bin/env python
"""Re-annotate an existing silver_trajectories.jsonl with the current
PRMAnnotator + KG caches.

Useful when:
  * a bug in the labeller is fixed,
  * the Wikidata cache has been refreshed,
  * the unified prompt schema is tweaked and silver step text needs to be
    re-parsed without re-running Phase 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from kgproweight.data.parsers import (
    ParsedStep,
    parse_steps,
    parsed_step_from_silver_dict,
)
from kgproweight.kg.entity_linker import EntityLinker, extract_mentions
from kgproweight.kg.wikidata_retriever import WikidataSubgraphRetriever
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path, resolve_kg_cache_dir
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.utils.logging import configure_logging, get_logger

configure_logging("INFO")
logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--kg_cache_dir",
        default=None,
        help="Directory containing kg_subgraph_cache.jsonl (defaults to $KGPW_KG_CACHE_DIR).",
    )
    p.add_argument(
        "--entity_cache_path",
        default=None,
        help="Path to entity_cache.jsonl (defaults to $KGPW_ENTITY_CACHE_PATH).",
    )
    p.add_argument("--no_refetch_kg", action="store_true", help="Skip Wikidata calls when no cached subgraph is present.")
    return p.parse_args()


def _build_parsed_steps(traj: dict) -> List[ParsedStep]:
    """Reconstruct ParsedStep objects from any of the formats we have written."""
    steps_raw = traj.get("steps", [])
    if not steps_raw:
        text = traj.get("teacher_output") or traj.get("raw_output") or ""
        return parse_steps(text)
    if isinstance(steps_raw[0], str):
        return parse_steps("\n".join(steps_raw))
    return [parsed_step_from_silver_dict(s, fallback_index=i) for i, s in enumerate(steps_raw)]


def main():
    args = parse_args()
    linker = EntityLinker(cache_path=resolve_entity_cache_path(args.entity_cache_path), use_genre=False)
    retriever = WikidataSubgraphRetriever(cache_dir=resolve_kg_cache_dir(args.kg_cache_dir))
    annotator = PRMAnnotator(entity_linker=linker)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_traj, n_labels = 0, 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            traj = json.loads(line)
            question = traj.get("question") or traj.get("query") or ""
            mentions = extract_mentions(question)
            qid_map = linker.link(mentions) if mentions else {}
            qids = [q for q in qid_map.values() if q]

            kg_triples = traj.get("kg_triples") or traj.get("kg_subgraph") or []
            kg_triples = [tuple(t) for t in kg_triples if isinstance(t, (list, tuple)) and len(t) == 3]
            if not kg_triples and qids and not args.no_refetch_kg:
                kg_triples = retriever.fetch(qids)

            steps = _build_parsed_steps(traj)
            labels = annotator.annotate_trajectory(steps, kg_triples)

            new_steps = []
            for s, label in zip(steps, labels):
                new_steps.append(
                    {
                        "index": s.index,
                        "text": s.raw_text,
                        "cited_triples": [list(t) for t in s.cited_triples],
                        "mentioned_entities": s.mentioned_entities,
                        "intermediate_conclusion": s.intermediate_conclusion,
                        "label": label,
                    }
                )
                n_labels += 1
            traj["steps"] = new_steps
            traj["kg_triples"] = [list(t) for t in kg_triples]
            fout.write(json.dumps(traj, ensure_ascii=False) + "\n")
            n_traj += 1

    logger.info("Re-annotated %d trajectories / %d steps → %s", n_traj, n_labels, out_path)


if __name__ == "__main__":
    main()
