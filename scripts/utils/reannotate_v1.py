#!/usr/bin/env python
"""Re-annotate V1 silver data with the current R9v6 pipeline.

V1 (silver_trajectories.jsonl) was generated on a machine with the full
wiki18 corpus (Answer Recall 73.6%) but with the OLD annotator:
  - KG not passed through filter_and_rank_triples (avg 105 noisy triples)
  - discrete labels only (+1/0/-1), no continuous R_KG
  - 14.15% cited-triple hallucination rate

This script keeps the Teacher output and retrieved_passages VERBATIM (they
are the expensive, irreplaceable part) and redoes only the KG-side work:

  1. filter_and_rank_triples(max_keep=12, min_keep=5)  — 3-layer KG filter
     (12/5 = the student's budget on every stage; see
     tests/test_kg_budget_alignment.py)
  2. PRMAnnotator.annotate_trajectory()    — continuous R_KG = precision x relevance
  3. StratifiedSilverFilter.decide()       — bucket quotas + answer_score gate

Writes the canonical SilverTrajectory schema so SilverDatasetReader can read it.
No network calls, no Teacher calls — pure local recompute.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kgproweight.data.parsers import parse_steps, parsed_step_from_silver_dict
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.kg.kg_filter import filter_and_rank_triples
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path
from kgproweight.training.phase1_distill import (
    StratifiedSilverFilter,
    answer_match_score,
)
from kgproweight.utils.logging import configure_logging, dump_manifest, get_logger

configure_logging("INFO")
logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/silver_data/silver_trajectories.jsonl")
    p.add_argument("--output", default="data/silver_data/silver_v1_reannotated.jsonl")
    p.add_argument("--max_kg_triples", type=int, default=12,
                   help="Must equal the student budget (12) or the reannotated "
                        "KG disagrees with what PPO/inference render.")
    p.add_argument("--max_passages", type=int, default=0,
                   help="Truncate stored passages to N (0 = keep all 50). "
                        "Downstream SFT/PPO cap at 15 anyway.")
    p.add_argument("--log_interval", type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    in_path, out_path = Path(args.input), Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    linker = EntityLinker(cache_path=resolve_entity_cache_path(), use_genre=False)
    annotator = PRMAnnotator(entity_linker=linker, verbose=False)
    accept_filter = StratifiedSilverFilter()

    n_in = n_out = n_acc = 0
    buckets: dict[str, int] = {}
    kg_before = kg_after = 0
    halluc_before = halluc_after = 0
    cited_before = cited_after = 0

    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                traj = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_in += 1

            question = traj.get("question") or ""
            raw_kg = [
                tuple(str(x) for x in t)
                for t in (traj.get("kg_subgraph") or traj.get("kg_triples") or [])
                if isinstance(t, (list, tuple)) and len(t) == 3
            ]
            kg_before += len(raw_kg)

            # --- 1. rebuild ParsedStep from stored text ----------------------
            steps_raw = traj.get("steps") or []
            if steps_raw and isinstance(steps_raw[0], dict):
                parsed = [parsed_step_from_silver_dict(s, fallback_index=i)
                          for i, s in enumerate(steps_raw)]
            else:
                parsed = parse_steps(traj.get("teacher_output") or "")

            # --- 2. 3-layer KG filter, PRESERVING Teacher-cited triples -----
            # The V1 Teacher saw the UNFILTERED subgraph and cited from it.
            # Filtering blindly would evict legitimately-cited triples and
            # inflate the apparent hallucination rate (8% -> 29% measured), and
            # would leave the PPO prompt missing triples the silver trace cites.
            # So: filtered top-K, then union back any cited triple that the
            # filter dropped but that IS genuinely in the raw subgraph.
            filtered = filter_and_rank_triples(
                raw_kg, question=question, max_keep=args.max_kg_triples, min_keep=5
            )
            raw_lookup = {
                (str(h).strip().lower(), str(r).strip().lower(), str(t).strip().lower()): (h, r, t)
                for h, r, t in raw_kg
            }
            kept = {
                (str(h).strip().lower(), str(r).strip().lower(), str(t).strip().lower())
                for h, r, t in filtered
            }
            kg = list(filtered)
            for st in parsed:
                for t in st.cited_triples:
                    if len(t) != 3:
                        continue
                    key = (str(t[0]).strip().lower(), str(t[1]).strip().lower(), str(t[2]).strip().lower())
                    if key in kept:
                        continue
                    orig = raw_lookup.get(key)
                    if orig is not None:  # genuine citation the filter dropped
                        kg.append(orig)
                        kept.add(key)
            kg_after += len(kg)

            # --- 3. re-label against the reconciled subgraph -----------------
            labels = annotator.annotate_trajectory(parsed, list(kg))

            # hallucination accounting (vs the FILTERED subgraph)
            kg_set = {(h.strip().lower(), r.strip().lower(), t.strip().lower())
                      for h, r, t in kg}
            raw_set = {(h.strip().lower(), r.strip().lower(), t.strip().lower())
                       for h, r, t in raw_kg}

            new_steps = []
            for st, label in zip(parsed, labels):
                for t in st.cited_triples:
                    if len(t) == 3:
                        k = (str(t[0]).strip().lower(), str(t[1]).strip().lower(), str(t[2]).strip().lower())
                        cited_before += 1
                        if k not in raw_set:
                            halluc_before += 1
                        cited_after += 1
                        if k not in kg_set:
                            halluc_after += 1
                new_steps.append({
                    "index": st.index,
                    "text": st.raw_text,
                    "label": float(label),
                    "cited_triples": [list(t) for t in st.cited_triples],
                })

            # --- 4. re-apply the stratified acceptance filter ----------------
            md = traj.get("metadata") or {}
            gold = str(md.get("gold_answer") or "")
            final = traj.get("answer") or ""
            answer_score = answer_match_score(final, gold) if gold else 0.0
            coverage = float(md.get("coverage", 0.0))

            class _S:  # minimal shim: filter only reads .cited_triples
                __slots__ = ("cited_triples",)

                def __init__(self, ct):
                    self.cited_triples = ct

            decision = accept_filter.decide(
                steps=[_S(s["cited_triples"]) for s in new_steps],
                coverage=coverage,
                answer_score=answer_score,
            )
            buckets[decision.bucket] = buckets.get(decision.bucket, 0) + 1
            if decision.accepted:
                n_acc += 1

            passages = traj.get("retrieved_passages") or []
            if args.max_passages > 0:
                passages = passages[: args.max_passages]

            out = {
                "qid": traj.get("qid") or traj.get("id") or "",
                "question": question,
                "answer": final,
                "dataset": traj.get("dataset") or "hotpotqa",
                "steps": new_steps,
                "kg_subgraph": [list(t) for t in kg],
                "retrieved_passages": passages,
                "accepted": decision.accepted,
                "metadata": {
                    **md,
                    "answer_score": answer_score,
                    "bucket": decision.bucket,
                    "triple_rate": decision.triple_rate,
                    "reject_reason": "" if decision.accepted else decision.reason,
                    "reannotated_from": "V1",
                    "kg_before_filter": len(raw_kg),
                    "kg_after_filter": len(kg),
                },
                "teacher_output": traj.get("teacher_output"),
                "teacher_model": traj.get("teacher_model"),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

            if n_out % args.log_interval == 0:
                logger.info("%d processed | %d accepted (%.1f%%)",
                            n_out, n_acc, 100 * n_acc / n_out)

    logger.info("=" * 60)
    logger.info("Re-annotated %d -> %d trajectories, %d accepted (%.1f%%)",
                n_in, n_out, n_acc, 100 * n_acc / max(n_out, 1))
    logger.info("Buckets: %s", buckets)
    logger.info("KG triples: %.1f -> %.1f per trajectory (3-layer filter)",
                kg_before / max(n_out, 1), kg_after / max(n_out, 1))
    logger.info("Hallucination vs RAW subgraph:      %.2f%% (%d/%d)",
                100 * halluc_before / max(cited_before, 1), halluc_before, cited_before)
    logger.info("Hallucination vs FILTERED subgraph: %.2f%% (%d/%d)",
                100 * halluc_after / max(cited_after, 1), halluc_after, cited_after)
    logger.info("Output: %s", out_path)

    dump_manifest(out_path.parent, extra={
        "phase": "reannotate_v1",
        "input": str(in_path),
        "output": str(out_path),
        "total": n_out,
        "accepted": n_acc,
        "buckets": buckets,
        "max_kg_triples": args.max_kg_triples,
        "avg_kg_before": kg_before / max(n_out, 1),
        "avg_kg_after": kg_after / max(n_out, 1),
        "halluc_rate_raw": 100 * halluc_before / max(cited_before, 1),
        "halluc_rate_filtered": 100 * halluc_after / max(cited_after, 1),
    })


if __name__ == "__main__":
    main()
