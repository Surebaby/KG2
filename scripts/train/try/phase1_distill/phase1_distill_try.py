"""Phase 1 (try variant) — Graph-Guided Trajectory Distillation, improved.

Differences vs ``kgproweight.training.phase1_distill`` (all original code is
left untouched; this is a standalone copy under ``scripts/train/try/``):

1. Lenient answer matching via ``answer_match_score`` (substring + recall +
   alias-tolerant F1) replaces the strict token-F1 ≥ 0.5 gate.
2. Stratified acceptance (``StratifiedSilverFilter``) keeps a quota of
   low-triple-rate / low-coverage trajectories instead of hard-rejecting
   them, so the α-Gate learns the α→0 fallback region.
3. Robust mention extraction (passage-title anchors + optional spaCy NER);
   ``coverage`` is recorded as a soft signal in metadata and never used to
   reject a query.
4. SPARQL failures degrade gracefully: an empty subgraph still flows through
   (the trajectory simply lands in the ``kg_sparse`` bucket) rather than
   killing the sample.
5. The one-shot format retry is preserved (it targets *format*, not yield).

This module reuses everything else from the original package.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# --- reused, unchanged, from the original package --------------------------
from kgproweight.data.parsers import extract_final_answer, parse_steps
from kgproweight.data.prompts import build_teacher_messages
from kgproweight.data.silver_dataset import (
    SilverStepRecord,
    SilverTrajectory,
)
from kgproweight.kg.coverage import coverage_score
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.kg.wikidata_retriever import WikidataSubgraphRetriever
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.retrieval.hybrid import DEFAULT_TOPK
from kgproweight.training.phase1_distill import (
    TeacherClient,
    _RetrievalAdapter,
    _build_retry_messages,
    _needs_format_retry,
    _select_relevant_triples,
)
from kgproweight.utils.logging import dump_manifest, get_logger

# --- changed logic, local to the try variant -------------------------------
from distill_helpers_try import (
    StratifiedSilverFilter,
    answer_match_score,
    extract_mentions_robust,
)
from prm_annotator_try import ImprovedPRMAnnotator

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Phase1TryConfig:
    dataset_name: str
    items: Sequence[Dict[str, Any]]
    output_path: str
    teacher_client: TeacherClient
    retriever_factory: Any
    entity_linker: EntityLinker
    kg_retriever: WikidataSubgraphRetriever
    append_output: bool = False
    prm_annotator: Optional[PRMAnnotator] = None
    top_k: int = DEFAULT_TOPK
    max_kg_triples: int = 50
    max_workers: int = 1
    accept_filter: StratifiedSilverFilter = field(default_factory=StratifiedSilverFilter)
    seed: int = 42
    teacher_temperature: float = 0.3
    extra_metadata: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Annotation (same shape as the original _annotate_steps)
# ---------------------------------------------------------------------------

def _annotate_steps(raw_output: str, kg_subgraph, annotator: PRMAnnotator) -> List[SilverStepRecord]:
    parsed = parse_steps(raw_output)
    labels = annotator.annotate_trajectory(parsed, list(kg_subgraph))
    out: List[SilverStepRecord] = []
    for step, label in zip(parsed, labels):
        out.append(
            SilverStepRecord(
                index=step.index,
                text=step.raw_text,
                label=int(label),
                cited_triples=list(step.cited_triples),
                token_logprobs=None,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Per-item processing — returns an *unfiltered* candidate; the accept/reject
# decision is taken in run_phase1 so the stratified quota can be enforced
# with a single shared counter.
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    trajectory: SilverTrajectory
    coverage: float
    answer_score: float


def _process_one(item: Dict[str, Any], cfg: Phase1TryConfig, retriever_call: _RetrievalAdapter) -> Optional[_Candidate]:
    qid = str(item.get("id") or item.get("qid") or "")
    question = str(item.get("question", "")).strip()
    gold_list = item.get("golden_answers") or ([item.get("answer", "")] if item.get("answer") else [])
    gold = str(gold_list[0]) if gold_list else ""
    if not question:
        return None

    # ---- Hybrid retrieval first: passages double as mention anchors -------
    try:
        passages = retriever_call(question)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Retrieval failed for qid=%s: %s", qid, exc)
        passages = []

    # ---- (change 3) robust mentions: NER/regex + passage titles -----------
    mentions = extract_mentions_robust(question, passages=passages, max_n=8)
    linked = cfg.entity_linker.link(mentions) if mentions else {}
    qids = [q for q in linked.values() if q]

    # ---- (change 4) SPARQL degrade: empty subgraph is allowed -------------
    triples: List = []
    if qids:
        try:
            triples = cfg.kg_retriever.fetch(qids)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SPARQL fetch failed for qid=%s (%d qids): %s", qid, len(qids), exc)
            triples = []

    # coverage is now a *soft* signal only (recorded, never rejects)
    coverage = coverage_score(mentions, linked) if mentions else 0.0

    # ---- Teacher generation ----------------------------------------------
    messages = build_teacher_messages(
        question=question,
        retrieved_passages=passages,
        kg_triples=_select_relevant_triples(
            question=question, passages=passages, triples=triples, top_n=cfg.max_kg_triples
        ),
        top_k=cfg.top_k,
        max_kg_triples=cfg.max_kg_triples,
    )
    raw_output = cfg.teacher_client.chat(messages)
    if not raw_output.strip():
        return None
    annotator = cfg.prm_annotator or ImprovedPRMAnnotator(entity_linker=cfg.entity_linker, verbose=False)
    parsed_steps = _annotate_steps(raw_output, triples, annotator)

    # ---- (change 5) one-shot corrective retry, format only ----------------
    min_steps = cfg.accept_filter.min_steps
    if _needs_format_retry(parsed_steps, triples, min_steps):
        retry_output = cfg.teacher_client.chat(_build_retry_messages(messages))
        if retry_output.strip():
            retry_steps = _annotate_steps(retry_output, triples, annotator)
            if not _needs_format_retry(retry_steps, triples, min_steps):
                raw_output = retry_output
                parsed_steps = retry_steps

    final_answer = extract_final_answer(raw_output) or ""
    # ---- (change 1) lenient answer match ----------------------------------
    answer_score = answer_match_score(final_answer, gold) if gold else 0.0

    traj = SilverTrajectory(
        qid=qid,
        question=question,
        answer=final_answer,
        dataset=cfg.dataset_name,
        steps=parsed_steps,
        kg_subgraph=triples,
        retrieved_passages=passages,
        teacher_output=raw_output,
        teacher_model=cfg.teacher_client.model,
        accepted=False,  # set in run_phase1 after the stratified decision
        metadata={
            "gold_answer": gold,
            "answer_score": answer_score,
            "coverage": coverage,
            "linked_entities": {m: q for m, q in linked.items() if q},
            "n_mentions": len(mentions),
            "kg_empty": len(triples) == 0,
            "extra": cfg.extra_metadata or {},
        },
    )
    return _Candidate(trajectory=traj, coverage=coverage, answer_score=answer_score)


# ---------------------------------------------------------------------------
# Main loop — generation can be parallel; the stratified accept decision and
# the write are serialised so the shared quota counter is race-free.
# ---------------------------------------------------------------------------

def _decide_and_write(cand: _Candidate, cfg: Phase1TryConfig, fh) -> Dict[str, Any]:
    """Apply the stratified filter to one candidate and persist it.

    Every candidate is written to disk (so nothing is silently lost and the
    rejected pool can be analysed later); ``accepted`` + bucket land in the
    record's metadata. Returns a small counter dict.
    """
    decision = cfg.accept_filter.decide(
        steps=cand.trajectory.steps,
        coverage=cand.coverage,
        answer_score=cand.answer_score,
    )
    cand.trajectory.accepted = decision.accepted
    cand.trajectory.metadata["bucket"] = decision.bucket
    cand.trajectory.metadata["triple_rate"] = decision.triple_rate
    cand.trajectory.metadata["reject_reason"] = "" if decision.accepted else decision.reason
    fh.write(json.dumps(cand.trajectory.to_dict(), ensure_ascii=False) + "\n")
    fh.flush()
    return {"accepted": int(decision.accepted), "bucket": decision.bucket}


def run_phase1(cfg: Phase1TryConfig) -> Dict[str, Any]:
    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    retriever = cfg.retriever_factory() if callable(cfg.retriever_factory) else cfg.retriever_factory
    retrieval_call = _RetrievalAdapter(retriever, top_k=cfg.top_k)

    total = 0
    written = 0
    accepted = 0
    bucket_counts: Dict[str, int] = {}

    write_mode = "a" if cfg.append_output else "w"
    with open(out_path, write_mode, encoding="utf-8") as fh:
        if cfg.max_workers <= 1:
            for item in cfg.items:
                total += 1
                cand = _process_one(item, cfg, retrieval_call)
                if cand is None:
                    continue
                res = _decide_and_write(cand, cfg, fh)
                written += 1
                accepted += res["accepted"]
                bucket_counts[res["bucket"]] = bucket_counts.get(res["bucket"], 0) + 1
        else:
            # Teacher/SPARQL calls run in parallel; decide+write stays serial.
            with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
                futures = [ex.submit(_process_one, item, cfg, retrieval_call) for item in cfg.items]
                for fut in as_completed(futures):
                    total += 1
                    try:
                        cand = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Teacher worker failed: %s", exc)
                        continue
                    if cand is None:
                        continue
                    res = _decide_and_write(cand, cfg, fh)
                    written += 1
                    accepted += res["accepted"]
                    bucket_counts[res["bucket"]] = bucket_counts.get(res["bucket"], 0) + 1

    dump_manifest(
        out_path.parent,
        extra={
            "phase": "phase1_distill_try",
            "dataset": cfg.dataset_name,
            "teacher_model": cfg.teacher_client.model,
            "teacher_temperature": cfg.teacher_temperature,
            "seed": cfg.seed,
            "output_path": str(out_path),
            "total_attempts": total,
            "written": written,
            "accepted": accepted,
            "bucket_counts": bucket_counts,
            "accepted_bucket_counts": cfg.accept_filter.stats(),
            "accept_filter": {
                "type": "stratified",
                "min_steps": cfg.accept_filter.min_steps,
                "max_steps": cfg.accept_filter.max_steps,
                "min_answer_score": cfg.accept_filter.min_answer_score,
                "rich_triple_rate": cfg.accept_filter.rich_triple_rate,
                "medium_triple_rate": cfg.accept_filter.medium_triple_rate,
                "sparse_quota": cfg.accept_filter.sparse_quota,
                "medium_quota": cfg.accept_filter.medium_quota,
            },
        },
    )
    logger.info(
        "Phase 1 (try) finished: wrote %d (accepted=%d / total=%d) buckets=%s → %s",
        written, accepted, total, bucket_counts, out_path,
    )
    return {
        "total": total,
        "written": written,
        "accepted": accepted,
        "bucket_counts": bucket_counts,
        "output": str(out_path),
    }
